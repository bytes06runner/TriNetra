"""
tests/test_lunar_reg.py — Stage O registration package (src/lunar_reg).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _import():
    """Load src/lunar_reg by file path under a private package name. The LoFTR
    helper used elsewhere in the suite rebinds `src`; touching sys.modules['src']
    here would reload other tests' classes and break their isinstance checks."""
    import importlib
    import importlib.util
    name = "lunar_reg_under_test"
    if name not in sys.modules:
        pkg_dir = REPO_ROOT / "src" / "lunar_reg"
        spec = importlib.util.spec_from_file_location(name, pkg_dir / "__init__.py",
                                                      submodule_search_locations=[str(pkg_dir)])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[name] = pkg
        spec.loader.exec_module(pkg)
    return tuple(importlib.import_module(f"{name}.{m}")
                 for m in ("dense_ncc", "models", "pitiscus_grid", "products", "validate"))


RESULTS_JSON = REPO_ROOT / "assets" / "real_cache" / "stage_o_pitiscus_registration.json"


def _texture(shape, seed=0, smooth=2.0):
    rng = np.random.default_rng(seed)
    f = np.fft.fft2(rng.standard_normal(shape))
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    f *= np.exp(-((fx ** 2 + fy ** 2) * (2 * np.pi * smooth) ** 2) / 2)
    img = np.real(np.fft.ifft2(f))
    return ((img - img.min()) / (img.max() - img.min()) * 250 + 5).astype(np.float32)


@pytest.mark.parametrize("kind", ["translation", "similarity", "affine", "poly2", "poly3"])
def test_models_recover_exact_maps(kind):
    _, models, *_ = _import()
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 1000, (200, 2))
    truth = models.fit(src, src + rng.normal(0, 3, src.shape), kind)  # an arbitrary model of this kind
    ref = truth.predict(src)
    m = models.fit(src, ref, kind)
    assert np.abs(m.predict(src) - ref).max() < 1e-6


def test_ransac_rejects_outliers():
    _, models, *_ = _import()
    rng = np.random.default_rng(2)
    src = rng.uniform(0, 1000, (300, 2))
    ref = src + np.array([12.5, -7.25]) + 1e-5 * (src[:, [0]] * src[:, [1]])
    bad = rng.choice(300, 90, replace=False)
    ref[bad] += rng.uniform(-80, 80, (90, 2))
    r = models.ransac(src, ref, "poly2", thresh=1.0)
    good = np.ones(300, bool)
    good[bad] = False
    assert r["inliers"][good].mean() > 0.99
    assert r["inliers"][bad].mean() < 0.05


def test_dense_ncc_recovers_known_shift_and_sign():
    dense_ncc, *_ = _import()
    big = _texture((700, 700), seed=3)
    src = big[100:600, 100:600]
    ref = big[100 - 11:600 - 11, 100 + 7:600 + 7]  # content at p in src sits at p + (-7, +11) in ref
    m = dense_ncc.match_nodes(src, ref, prior_offset=(-5, 9), spacing=64, half_tpl=24, radius=16)
    acc = dense_ncc.accept(m)
    d = m["ref_xy"][acc] - m["src_xy"][acc]
    assert acc.mean() > 0.9
    assert np.allclose(np.median(d, axis=0), [-7, 11], atol=0.05)


def test_heldout_validation_schemes():
    _, models, _, _, validate = _import()
    rng = np.random.default_rng(4)
    src = rng.uniform(0, 1000, (400, 2))
    ref = src + np.array([3.0, 4.0]) + rng.normal(0, 0.3, src.shape)
    for scheme in ("spatial_blocks", "random_kfold"):
        v = validate.heldout(src, ref, "translation", 1.5, scheme)
        o = v["overall"]
        assert o["n_heldout"] == 400
        assert 0.3 < o["rmse_within_trim_px"] < 0.6


def test_map_projection_roundtrip():
    _, _, pitiscus_grid, _, _ = _import()
    lon, lat = np.array([31.1, 31.4]), np.array([-51.6, -50.7])
    X, Y = pitiscus_grid.lonlat_to_map(lon, lat)
    lon2, lat2 = pitiscus_grid.map_to_lonlat(X, Y)
    assert np.allclose(lon, lon2) and np.allclose(lat, lat2)


def test_products_refuse_overwrite(tmp_path):
    *_, products, _ = _import()
    p = tmp_path / "x.csv"
    products.write_csv(p, ["h"], {"a": np.arange(3)}, {})
    with pytest.raises(FileExistsError):
        products.write_csv(p, ["h"], {"a": np.arange(3)}, {})


@pytest.mark.skipif(not RESULTS_JSON.exists(), reason="Stage O results not generated")
def test_stage_o_frozen_findings():
    d = json.loads(RESULTS_JSON.read_text())
    assert d["chosen_model"] == "poly2"
    assert d["delta_shuffle"]["pass"] is True and d["delta_shuffle"]["value_pp"] > 80
    assert all(c["n_inliers"] == 0 for c in d["controls"].values())
    assert set(d["controls"]) == {"rot90_blockwise", "rot180", "rot270_blockwise", "vflip", "hflip",
                                  "offset_nonoverlap_2000rows", "uniform_noise"}
    sb = d["model_selection"]["poly2"]["spatial_blocks"]["overall"]
    assert 0.9 < sb["rmse_within_trim_px"] < 1.1 and sb["frac_within_3px"] > 0.95
    assert all(abs(c["dx"]) < 1 and abs(c["dy"]) < 1 and c["psr"] > 100 for c in d["closure_stage_n_estimator"])


def test_pspline_fits_along_track_jitter():
    _, models, *_ = _import()
    rng = np.random.default_rng(7)
    src = np.column_stack([rng.uniform(0, 3000, 1500), rng.uniform(0, 16000, 1500)])
    jitter = np.column_stack([3 * np.sin(src[:, 1] / 900.0), 2 * np.cos(src[:, 1] / 1300.0)])
    ref = src + np.array([-600.0, -3300.0]) + jitter + rng.normal(0, 0.2, src.shape)
    r = models.pspline_trimmed(src, ref, 1.5, knot_px=1000)
    assert r["inliers"].mean() > 0.97
    assert np.sqrt((r["residuals"][r["inliers"]] ** 2).mean()) < 0.35
    poly = models.ransac(src, ref, "poly3", 1.5)
    assert r["inliers"].sum() > poly["inliers"].sum()


def test_pspline_relief_recovers_parallax_vector():
    _, models, *_ = _import()
    rng = np.random.default_rng(8)
    H, W = 4000, 800
    yy, xx = np.mgrid[0:H, 0:W]
    dtm = (40 * np.sin(xx / 97.0) * np.cos(yy / 131.0) + 20 * np.sin(yy / 53.0)).astype(np.float64)
    from scipy.ndimage import map_coordinates
    height_at = lambda xy: map_coordinates(dtm, [np.asarray(xy)[:, 1], np.asarray(xy)[:, 0]], order=1, mode="constant", cval=np.nan)
    p_true = np.array([-0.256, -0.071])  # px per metre, same order as the OHRC fit
    src = np.column_stack([rng.uniform(50, W - 50, 2000), rng.uniform(50, H - 50, 2000)])
    ref = src + np.array([3.0, -2.0])
    for _ in range(4):  # ref position depends on height at ref position
        h = height_at(ref)
        ref = src + np.array([3.0, -2.0]) + (h - np.nanmedian(h))[:, None] * p_true[None, :]
    ref = ref + rng.normal(0, 0.15, ref.shape)
    r = models.pspline_relief_trimmed(src, ref, 1.5, 500, height_at)
    assert np.allclose(r["model"].p, p_true, atol=0.01)
    assert r["inliers"].mean() > 0.97
    # prediction without observed heights (iterative lookup) stays accurate
    pred = r["model"].predict(src)
    assert np.median(np.linalg.norm(pred - ref, axis=1)) < 0.3


def test_stripes_scheme_folds_are_row_bands():
    _, models, _, _, validate = _import()
    rng = np.random.default_rng(9)
    src = np.column_stack([rng.uniform(0, 500, 600), rng.uniform(0, 5000, 600)])
    ref = src + 1.0
    v = validate.heldout(src, ref, "translation", 1.5, "stripes", stripe_px=500)
    fold = np.array(v["fold"])
    y0 = src[:, 1].min()  # stripes are anchored at the first node row
    bands = [(src[:, 1] >= y0 + b) & (src[:, 1] < y0 + b + 500) for b in range(0, 5000, 500)]
    assert all(len(set(fold[m])) == 1 for m in bands if m.any())
    assert len(set(fold)) == 5


STAGE_P_OHRC = REPO_ROOT / "assets" / "real_cache" / "stage_p_vikram_ohrc.json"
STAGE_P_TMC = REPO_ROOT / "assets" / "real_cache" / "stage_p_vikram_tmc.json"


@pytest.mark.skipif(not STAGE_P_OHRC.exists(), reason="Stage P OHRC results not generated")
def test_stage_p_ohrc_frozen_findings():
    d = json.loads(STAGE_P_OHRC.read_text())
    assert d["chosen_model"] == "pspline_relief"
    assert d["delta_shuffle"]["pass"] is True and d["delta_shuffle"]["value_pp"] > 90
    assert all(c["n_inliers"] == 0 for c in d["controls"].values()) and len(d["controls"]) == 7
    st = d["model_selection"]["pspline_relief"]["stripes"]["overall"]
    assert st["rmse_within_trim_px"] < 0.6 and st["frac_within_1px"] > 0.9
    pc = d["relief_parallax_check"]
    assert abs(pc["difference_deg"]) < 1.0  # fitted parallax agrees with the label view angle
    assert 3500 < d["isro_geometry_offset_m"]["median_norm"] < 4000
    assert max(abs(c["dx"]) + abs(c["dy"]) for c in d["product_check_vs_rasterio_reprojected_nac"]) < 1.0


@pytest.mark.skipif(not STAGE_P_TMC.exists(), reason="Stage P TMC results not generated")
def test_stage_p_tmc_frozen_findings_gated():
    d = json.loads(STAGE_P_TMC.read_text())
    assert d["delta_shuffle"]["pass"] is False  # reported negative: dense NCC fails at this site
    assert sum(s["significant"] for s in d["coarse_segments"]) >= 10  # the coarse offset itself is solid
    assert 3500 < d["isro_geometry_offset_m"]["median_norm"] < 4000
