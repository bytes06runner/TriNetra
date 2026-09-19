#!/usr/bin/env python3
"""
scripts/evaluate_stage_n.py — Stage N: independent corroboration by whole-image
Fourier phase correlation.

The estimator lives in src/corroboration/phase_correlation.py (numpy only, no
project imports). This runner does data preparation and bookkeeping only:
it loads cached image pairs, reads the primary pipeline's cached transforms,
converts them to displacements in the same frame, and runs the control battery.

Pre-declared settings (fixed before any result was inspected):
  * Hann window on both images.
  * Band limit = Nyquist of the coarser sensor expressed on the evaluation
    canvas (Hop 1: 0.5 * 240/1000 = 0.12 cyc/px; Hop 2: 0.5 * 120/800 =
    0.075 cyc/px; Pitiscus and M2 are evaluated on the 4.72 m native TMC-2
    grid, so no band limit).
  * Agreement tolerance: 2.0 px in the evaluation frame.
  * Significance: PSR of the genuine pair must exceed the maximum PSR over
    N_NULL uniform-noise references AND exceed every core control
    (Delta_PSR = PSR_genuine - max(core-control PSR) > 0).

Outputs (never overwrites; aborts if the file exists):
  assets/real_cache/stage_n_corroboration_results.json
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE = REPO_ROOT / "assets" / "real_cache"
OUT_JSON = CACHE / "stage_n_corroboration_results.json"

TOL_PX = 2.0
N_NULL = 200
SEED = 42

# Load the independent estimator by file path so the runner never resolves it
# through the `src` package (which the LoFTR checkpoint helper rebinds).
_spec = importlib.util.spec_from_file_location(
    "stage_n_phase_correlation", REPO_ROOT / "src" / "corroboration" / "phase_correlation.py")
pcm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pcm)


def fmt(r):
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()}


def core_controls(fixed, moving, offset_moving, rng, **kw):
    """Core control battery: every control replaces the reference (`moving`)."""
    ctrls = {
        "rot90": np.rot90(moving, 1),
        "rot180": np.rot90(moving, 2),
        "rot270": np.rot90(moving, 3),
        "vflip": moving[::-1, :],
        "hflip": moving[:, ::-1],
        "offset_nonoverlap": offset_moving,
        "uniform_noise": rng.uniform(0, 255, moving.shape),
    }
    return {k: fmt(pcm.estimate_shift(fixed, v, **kw)) for k, v in ctrls.items()}


def noise_null(fixed, rng, n, **kw):
    psr = [pcm.estimate_shift(fixed, rng.uniform(0, 255, fixed.shape), **kw)["psr"] for _ in range(n)]
    psr = np.array(psr)
    return {"n": int(n), "max": float(psr.max()), "p99": float(np.percentile(psr, 99)),
            "p95": float(np.percentile(psr, 95)), "mean": float(psr.mean()), "std": float(psr.std())}


def verdicts(gen, ctrls, null, primary_d, px_to_m, px_to_native):
    max_ctrl_name = max(ctrls, key=lambda k: ctrls[k]["psr"])
    max_ctrl = ctrls[max_ctrl_name]["psr"]
    delta_psr = gen["psr"] - max_ctrl
    significant = bool(gen["psr"] > null["max"] and delta_psr > 0)
    out = {
        "psr_genuine": gen["psr"],
        "max_control_psr": max_ctrl,
        "max_control_name": max_ctrl_name,
        "delta_psr": float(delta_psr),
        "noise_null_max_psr": null["max"],
        "significant": significant,
    }
    if primary_d is not None:
        ex, ey = gen["dx"] - primary_d[0], gen["dy"] - primary_d[1]
        gap = float(np.hypot(ex, ey))
        out.update({
            "primary_dx": float(primary_d[0]), "primary_dy": float(primary_d[1]),
            "gap_px": gap, "gap_native_px": gap * px_to_native, "gap_m": gap * px_to_m,
            "agree_within_tol": bool(gap <= TOL_PX),
        })
    return out


# ---------------------------------------------------------------------------
# Hops 1 and 2 (canvas pairs with a cached similarity H)
# ---------------------------------------------------------------------------
def run_hop(name, fixed, moving, H, offset_moving, cutoff, px_to_m, px_to_native, rng):
    h, w = fixed.shape
    centre = ((w - 1) / 2.0, (h - 1) / 2.0)
    d_primary = pcm.similarity_displacement(H, centre)
    A = np.asarray(H, dtype=np.float64)[:2, :2]
    res = {"name": name, "canvas_shape": [h, w], "centre_xy": centre,
           "primary_H": np.asarray(H).tolist(),
           "primary_scale": float(np.hypot(A[0, 0], A[1, 0])),
           "primary_rot_deg": float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))),
           "primary_displacement_at_centre": list(d_primary), "variants": {}}

    fixed_cond = pcm.warp_linear_about_centre(fixed, A)
    variants = {
        # declared primary variant
        "A_bandlimited": (fixed, dict(lowpass_cyc_per_px=cutoff)),
        "B_fullband": (fixed, dict(lowpass_cyc_per_px=None)),
        # conditioned on the primary pipeline's own scale+rotation: not independent,
        # asks only "given the primary's linear part, does translation agree?"
        "C_conditioned_bandlimited": (fixed_cond, dict(lowpass_cyc_per_px=cutoff)),
    }
    for vname, (fx, kw) in variants.items():
        gen = fmt(pcm.estimate_shift(fx, moving, **kw))
        ctrls = core_controls(fx, moving, offset_moving, rng, **kw)
        null = noise_null(fx, rng, N_NULL, **kw)
        res["variants"][vname] = {"settings": {k: v for k, v in kw.items()}, "genuine": gen,
                                  "controls": ctrls, "noise_null": null,
                                  "verdict": verdicts(gen, ctrls, null, d_primary, px_to_m, px_to_native)}
        v = res["variants"][vname]["verdict"]
        print(f"  [{name} {vname}] PC=({gen['dx']:+.2f},{gen['dy']:+.2f}) primary=({d_primary[0]:+.2f},{d_primary[1]:+.2f}) "
              f"gap={v['gap_px']:.2f}px ({v['gap_m']:.1f} m) PSR={gen['psr']:.2f} maxctrl={v['max_control_psr']:.2f}"
              f"({v['max_control_name']}) null_max={v['noise_null_max_psr']:.2f} dPSR={v['delta_psr']:+.2f} sig={v['significant']}")

    # third-party cross-check of this module's arithmetic (not an independent estimator claim)
    (cdx, cdy), cresp = cv2.phaseCorrelate(fixed.astype(np.float32), moving.astype(np.float32),
                                           cv2.createHanningWindow((w, h), cv2.CV_32F))
    res["cv2_phaseCorrelate_crosscheck"] = {"dx": float(cdx), "dy": float(cdy), "response": float(cresp)}
    return res


# ---------------------------------------------------------------------------
# Pitiscus common grid (4.72 m/px), rebuilt with the Stage K recipe.
# This data preparation is shared in algorithm with Stage K (common-mode risk).
# ---------------------------------------------------------------------------
def build_pitiscus_grid():
    import pandas as pd
    import rasterio
    from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator

    with rasterio.open(REPO_ROOT / "data/lroc_nac/NAC_DTM_PITISCUS_M1149280834_2M.TIF") as src:
        b = src.bounds
        o = src.read(1)
    L, Rr, B, T, g = -2841680.0, -2835360.0, -1566436.0, -1538943.33, 4.72
    mx, my = np.meshgrid(np.arange(L, Rr, g), np.arange(T, B, -g))
    ortho = cv2.remap(o.astype(np.float32), ((mx - b.left) / 2.0).astype(np.float32),
                      ((b.top - my) / 2.0).astype(np.float32), cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    df = pd.read_csv(REPO_ROOT / "data/ch2_tmc_ncn_20230130T1900132182_d_img_d32/geometry/calibrated/"
                     "20230130/ch2_tmc_ncn_20230130T1900132182_g_grd_d32.csv")
    R, phi1, lam0 = 1737400.0, np.radians(-51.0), np.radians(180.0)
    sub = df[(df["Scan"] >= 239000) & (df["Scan"] <= 247000)].copy()
    sub["X"] = R * (np.radians(sub["Longitude"]) - lam0) * np.cos(phi1)
    sub["Y"] = R * np.radians(sub["Latitude"])
    P = np.column_stack([sub["X"], sub["Y"]])
    iS, iP = LinearNDInterpolator(P, sub["Scan"]), LinearNDInterpolator(P, sub["Pixel"])
    xc, yc = np.linspace(L - 100, Rr + 100, 100), np.linspace(B - 100, T + 100, 400)
    cx, cy = np.meshgrid(xc, yc)
    ds = RegularGridInterpolator((yc, xc), iS(cx, cy))((my, mx)).astype(np.float32)
    dp = RegularGridInterpolator((yc, xc), iP(cx, cy))((my, mx)).astype(np.float32)
    s0, s1 = int(np.floor(ds.min())) - 10, int(np.ceil(ds.max())) + 10
    mm = np.memmap(REPO_ROOT / "data/ch2_tmc_ncn_20230130T1900132182_d_img_d32/data/calibrated/20230130/"
                   "ch2_tmc_ncn_20230130T1900132182_d_img_d32.img", dtype="<u2", mode="r", shape=(270516, 4000))
    tmc = cv2.remap(mm[s0:s1, :].astype(np.float32), dp, (ds - s0).astype(np.float32), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return ortho, tmc


def pitiscus_primary_models():
    """Primary-pipeline Pitiscus estimates, as maps from the unshifted TMC-2 grid
    position q to the NAC grid position. Stage K/L fit tmc_shifted -> ortho with
    tmc_shifted(q + b) = tmc(q), b = Stage J2c bulk offset."""
    b = np.array([-82.12, 224.04])
    k = json.load(open(CACHE / "stage_k_dense_results.json"))["k5_metrics"]["similarity_matrix"]
    l10 = json.load(open(CACHE / "stage_l_geometric_filter_results.json"))["l2_ransac"]["threshold_sweeps"]["10m"]["M_sim"]
    Mk, Ml = np.asarray(k, dtype=np.float64)[:2], np.asarray(l10, dtype=np.float64)[:2]

    def disp(M):
        return lambda q: tuple(np.array(pcm.similarity_displacement(M, (q[0] + b[0], q[1] + b[1]))) + b)

    return {
        "J2c_bulk_only": lambda q: (float(b[0]), float(b[1])),
        "K5_bulk_plus_324node_similarity": disp(Mk),
        "L2_bulk_plus_10m_ransac_similarity": disp(Ml),
    }, {"bulk": b.tolist(), "K5_M": Mk.tolist(), "L2_10m_M": Ml.tolist()}


def run_pitiscus(ortho, tmc, rng, tile=768):
    valid = ortho > 0
    H, W = ortho.shape
    tiles = []
    for y0 in range(0, H - tile + 1, tile):
        rows = valid[y0:y0 + tile]
        cols = np.where(rows.all(axis=0))[0]
        if len(cols) >= tile:
            x0 = int(cols[0] + (len(cols) - tile) // 2)
            if rows[:, x0:x0 + tile].all():
                tiles.append((x0, y0))
    models, model_params = pitiscus_primary_models()
    out = {"tile_size": tile, "tiles": [], "primary_model_params": model_params}
    n = len(tiles)
    for i, (x0, y0) in enumerate(tiles):
        fixed = tmc[y0:y0 + tile, x0:x0 + tile]
        moving = ortho[y0:y0 + tile, x0:x0 + tile]
        # non-overlapping TMC-2 reference terrain: the tile half the strip away
        j = (i + n // 2) % n
        ox0, oy0 = tiles[j]
        offset_moving = ortho[oy0:oy0 + tile, ox0:ox0 + tile]
        gen = fmt(pcm.estimate_shift(fixed, moving))
        ctrls = core_controls(fixed, moving, offset_moving, rng)
        null = noise_null(fixed, rng, N_NULL)
        centre = (x0 + (tile - 1) / 2.0, y0 + (tile - 1) / 2.0)
        comps = {}
        for mname, f in models.items():
            comps[mname] = verdicts(gen, ctrls, null, f(centre), 4.72, 4.72 / 2.0)
        v = comps["L2_bulk_plus_10m_ransac_similarity"]
        print(f"  [Pitiscus tile {i} at ({x0},{y0})] PC=({gen['dx']:+.2f},{gen['dy']:+.2f}) PSR={gen['psr']:.2f} "
              f"maxctrl={v['max_control_psr']:.2f}({v['max_control_name']}) null_max={v['noise_null_max_psr']:.2f} "
              f"sig={v['significant']} | gaps: " + ", ".join(f"{k.split('_')[0]}={c['gap_px']:.1f}px" for k, c in comps.items()))
        out["tiles"].append({"index": i, "x0": x0, "y0": y0, "offset_tile_index": j, "genuine": gen,
                             "controls": ctrls, "noise_null": null, "comparisons": comps})
    return out


def blockwise_rot(img, k):
    """Fabricated reference for non-square strips: rotate each square block by k*90 deg."""
    h, w = img.shape
    out = img.copy()
    y0 = 0
    while y0 + w <= h:
        out[y0:y0 + w] = np.rot90(img[y0:y0 + w], k)
        y0 += w
    if y0 < h:  # leftover rows must not keep genuine geometry
        out[y0:] = np.rot90(img[y0:], 2)
    return out


def _ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def _gradmag(a):
    gy, gx = np.gradient(a.astype(np.float64))
    return np.hypot(gx, gy)


def run_pitiscus_strip(ortho, tmc, primary_bulk, rng, cols=(345, 985), seg_h=2000):
    """POST-HOC (added after the pre-declared 768 px tiles returned no significant
    peak): whole valid strip and long segments, so that shifts of several hundred
    pixels still leave overlap. Columns 345:985 are valid in both rasters on every row."""
    c0, c1 = cols
    strip_t, strip_o = tmc[:, c0:c1], ortho[:, c0:c1]
    H = strip_o.shape[0]
    out = {"columns": [c0, c1], "segments": []}

    def battery(fixed, moving, offset_moving):
        w = moving.shape[1]
        ctrls = {
            "rot90_blockwise": blockwise_rot(moving, 1), "rot180": np.rot90(moving, 2),
            "rot270_blockwise": blockwise_rot(moving, 3), "vflip": moving[::-1, :],
            "hflip": moving[:, ::-1], "offset_nonoverlap": offset_moving,
            "uniform_noise": rng.uniform(0, 255, moving.shape),
        }
        return {k: fmt(pcm.estimate_shift(fixed, v)) for k, v in ctrls.items()}

    # whole strip; offset control = strip rolled by half its length (non-overlapping terrain)
    # Offset control: TMC-2 scans ~90 km along-track from Pitiscus (scans 150000+,
    # vs 239000-247000 at the site). Real lunar terrain, zero overlap. A roll of the
    # strip is NOT used: a circular estimator still finds the rolled content.
    mm = np.memmap(REPO_ROOT / "data/ch2_tmc_ncn_20230130T1900132182_d_img_d32/data/calibrated/20230130/"
                   "ch2_tmc_ncn_20230130T1900132182_d_img_d32.img", dtype="<u2", mode="r", shape=(270516, 4000))
    far_tmc = np.asarray(mm[150000:150000 + H, 1000:1000 + (c1 - c0)], dtype=np.float64)
    out["whole_strip_offset_control_source"] = "TMC-2 d32 scans 150000:155825, pixels 1000:1640 (~90 km from site)"
    gen = fmt(pcm.estimate_shift(strip_t, strip_o))
    ctrls = battery(strip_t, strip_o, far_tmc)
    null = noise_null(strip_t, rng, 50)
    out["whole_strip"] = {"genuine": gen, "controls": ctrls, "noise_null": null,
                          "vs_primary_bulk": verdicts(gen, ctrls, null, primary_bulk, 4.72, 4.72 / 2.0)}
    v = out["whole_strip"]["vs_primary_bulk"]
    print(f"  [Pitiscus strip POST-HOC whole] PC=({gen['dx']:+.2f},{gen['dy']:+.2f}) PSR={gen['psr']:.2f} "
          f"maxctrl={v['max_control_psr']:.2f}({v['max_control_name']}) null_max={v['noise_null_max_psr']:.2f} "
          f"sig={v['significant']} gap_vs_J2c={v['gap_px']:.1f}px ({v['gap_m']:.0f} m)")

    for y0 in (0, 1275, 2550, H - seg_h):
        f, mv = strip_t[y0:y0 + seg_h], strip_o[y0:y0 + seg_h]
        # non-overlapping reference segment: disjoint from where this segment's
        # content lies in the NAC strip under the whole-strip estimate
        lo, hi = y0 + gen["dy"], y0 + gen["dy"] + seg_h
        oy = next(o_ for o_ in range(0, H - seg_h + 1, 25) if o_ + seg_h <= lo or o_ >= hi)
        g = fmt(pcm.estimate_shift(f, mv))
        c = battery(f, mv, strip_o[oy:oy + seg_h])
        n = noise_null(f, rng, 50)
        vv = verdicts(g, c, n, primary_bulk, 4.72, 4.72 / 2.0)
        out["segments"].append({"y0": int(y0), "height": seg_h, "offset_control_y0": int(oy),
                                "genuine": g, "controls": c, "noise_null": n, "vs_primary_bulk": vv})
        print(f"  [Pitiscus strip POST-HOC rows {y0}-{y0 + seg_h}] PC=({g['dx']:+.2f},{g['dy']:+.2f}) PSR={g['psr']:.2f} "
              f"maxctrl={vv['max_control_psr']:.2f}({vv['max_control_name']}) sig={vv['significant']}")

    # Third, non-circular check (numpy NCC, no FFT): tmc tile at q vs ortho tile at q + lag,
    # for the phase-correlation lag, the lag Stage K/L applied, and 30 random lags.
    s = (int(round(gen["dx"])), int(round(gen["dy"])))
    lags = {"phase_correlation": s, "stageK_applied_bulk": (int(round(primary_bulk[0])), int(round(primary_bulk[1]))),
            "zero": (0, 0)}
    rl = np.random.default_rng(11)
    for i in range(30):
        lags[f"random_{i:02d}"] = (int(rl.integers(-150, 151)), int(rl.integers(-1200, 1201)))
    T = 512
    ncc_out = {}
    for name, (lx, ly) in lags.items():
        raw, grad = [], []
        for qy in range(300, tmc.shape[0] - T - 300, 600):
            for qx in (420, 560):
                py, px = qy + ly, qx + lx
                if py < 0 or py + T > ortho.shape[0] or px < 0 or px + T > ortho.shape[1]:
                    continue
                a, b = tmc[qy:qy + T, qx:qx + T], ortho[py:py + T, px:px + T]
                if (b == 0).any():
                    continue
                raw.append(_ncc(a, b))
                grad.append(_ncc(_gradmag(a), _gradmag(b)))
        ncc_out[name] = {"lag": [lx, ly], "n_tiles": len(raw),
                         "ncc_raw_mean": float(np.mean(raw)) if raw else None,
                         "ncc_raw_min": float(np.min(raw)) if raw else None,
                         "ncc_gradmag_mean": float(np.mean(grad)) if grad else None}
    rnd = [v["ncc_raw_mean"] for k, v in ncc_out.items() if k.startswith("random_") and v["ncc_raw_mean"] is not None]
    out["ncc_lag_check"] = {"lags": ncc_out, "random_lag_raw_mean_max": float(np.max(rnd)),
                            "random_lag_raw_mean_mean": float(np.mean(rnd)), "n_random_lags_scored": len(rnd)}
    for k in ("phase_correlation", "stageK_applied_bulk", "zero"):
        r = ncc_out[k]
        print(f"  [NCC lag check] {k:22s} lag={r['lag']} n={r['n_tiles']} raw={r['ncc_raw_mean']:+.3f} "
              f"(min {r['ncc_raw_min']:+.3f}) gradmag={r['ncc_gradmag_mean']:+.3f}")
    print(f"  [NCC lag check] random lags: max mean raw NCC={np.max(rnd):+.3f} over {len(rnd)} lags")

    # Tiles re-extracted at the phase-correlation lag (non-circular windows). Expect a
    # residual near (0, 0) with a significant peak if the strip estimate localises.
    tiles = []
    for qy in range(0, tmc.shape[0] - 768 - max(s[1], 0), 768):
        py = qy + s[1]
        if py < 0 or py + 768 > ortho.shape[0]:
            continue
        okc = np.where((ortho[py:py + 768] > 0).all(axis=0))[0]
        if len(okc) < 768:
            continue
        px = int(okc[0] + (len(okc) - 768) // 2)
        qx = px - s[0]
        if qx < 0 or qx + 768 > tmc.shape[1]:
            continue
        mv = ortho[py:py + 768, px:px + 768]
        if (mv == 0).any():
            continue
        f = tmc[qy:qy + 768, qx:qx + 768]
        g = fmt(pcm.estimate_shift(f, mv))
        oy_ = (py + 2304) % (ortho.shape[0] - 768)
        c = core_controls(f, mv, ortho[oy_:oy_ + 768, px:px + 768], rng)
        vv = verdicts(g, c, noise_null(f, rng, 50), None, 4.72, 4.72 / 2.0)
        tiles.append({"q": [qx, qy], "residual": [g["dx"], g["dy"]], "genuine": g, "verdict": vv,
                      "controls_psr": {k: v["psr"] for k, v in c.items()}})
        print(f"  [PC-lag tiles] q=({qx},{qy}) residual=({g['dx']:+.2f},{g['dy']:+.2f}) PSR={g['psr']:.2f} "
              f"maxctrl={vv['max_control_psr']:.2f} sig={vv['significant']}")
    out["pc_lag_tiles"] = tiles
    return out


# ---------------------------------------------------------------------------
# Positive controls: matched illumination across the 2.36x scale gap (Stage M)
# ---------------------------------------------------------------------------
def run_matched_illumination(rng):
    import rasterio
    with rasterio.open(REPO_ROOT / "data/lroc_nac/NAC_DTM_PITISCUS_M1149280834_2M.TIF") as src:
        o = src.read(1).astype(np.float32)
    cy, cx = int(4500 * 4.72 / 2.0), int(600 * 4.72 / 2.0)
    n472 = int(round(2048 * 2.00 / 4.72))
    crop = o[cy - 1024:cy + 1024, cx - 1024:cx + 1024]
    ref_472 = cv2.resize(cv2.GaussianBlur(crop, (7, 7), 1.2), (n472, n472), interpolation=cv2.INTER_AREA)
    src_base = cv2.resize(crop, (n472, n472), interpolation=cv2.INTER_AREA)
    far = o[cy - 1024 - 3000:cy + 1024 - 3000, cx - 1024:cx + 1024]
    offset_ref = cv2.resize(cv2.GaussianBlur(far, (7, 7), 1.2), (n472, n472), interpolation=cv2.INTER_AREA)

    # M2a: the 25 Stage M trials (circular Fourier shifts), compared with truth and
    # with the primary pipeline's cached per-trial estimate.
    trials = json.load(open(CACHE / "stage_m_synthetic_results.json"))["m2_scale_gap_registration"]["trials"]
    m2a = []
    for t in trials:
        mov = pcm.fourier_shift(src_base, t["true_dx_common"], t["true_dy_common"])
        r = pcm.estimate_shift(ref_472, mov)
        m2a.append({"trial": t["trial"], "true": [t["true_dx_common"], t["true_dy_common"]],
                    "primary": [t["rec_dx_common"], t["rec_dy_common"]], "pc": [r["dx"], r["dy"]], "psr": r["psr"],
                    "pc_vs_truth_px": float(np.hypot(r["dx"] - t["true_dx_common"], r["dy"] - t["true_dy_common"])),
                    "pc_vs_primary_px": float(np.hypot(r["dx"] - t["rec_dx_common"], r["dy"] - t["rec_dy_common"]))})
    # control battery on trial 1 geometry
    mov1 = pcm.fourier_shift(src_base, trials[0]["true_dx_common"], trials[0]["true_dy_common"])
    g1 = fmt(pcm.estimate_shift(ref_472, mov1))
    c1 = core_controls(ref_472, mov1, offset_ref, rng)
    n1 = noise_null(ref_472, rng, N_NULL)

    # M2b: non-circular known shifts. Two different windows of the native 2 m
    # orthophoto, displaced by an integer number of native pixels, resampled
    # independently to 4.72 m. True shift in the 4.72 m frame = -D * 2.00/4.72
    # (content at p in the reference window sits at p - D in the displaced window).
    rng_b = np.random.default_rng(7)
    m2b = []
    for k in range(25):
        D = rng_b.integers(-60, 61, size=2)
        win = o[cy - 1024 + D[1]:cy + 1024 + D[1], cx - 1024 + D[0]:cx + 1024 + D[0]]
        mov = cv2.resize(win, (n472, n472), interpolation=cv2.INTER_AREA)
        truth = (-D[0] * 2.0 / 4.72, -D[1] * 2.0 / 4.72)
        r = pcm.estimate_shift(ref_472, mov)
        m2b.append({"D_native_px": D.tolist(), "true": list(truth), "pc": [r["dx"], r["dy"]], "psr": r["psr"],
                    "pc_vs_truth_px": float(np.hypot(r["dx"] - truth[0], r["dy"] - truth[1]))})

    def summ(rows, key):
        e = np.array([r[key] for r in rows])
        return {"n": len(e), "rmse_px": float(np.sqrt(np.mean(e ** 2))), "max_px": float(e.max()),
                "rmse_m": float(np.sqrt(np.mean(e ** 2)) * 4.72), "all_within_tol": bool((e <= TOL_PX).all())}

    out = {
        "M2a_circular": {"trials": m2a, "pc_vs_truth": summ(m2a, "pc_vs_truth_px"),
                         "pc_vs_primary": summ(m2a, "pc_vs_primary_px"),
                         "trial1_battery": {"genuine": g1, "controls": c1, "noise_null": n1,
                                            "verdict": verdicts(g1, c1, n1, None, 4.72, 4.72 / 2.0)}},
        "M2b_noncircular": {"trials": m2b, "pc_vs_truth": summ(m2b, "pc_vs_truth_px")},
    }
    print(f"  [M2a] PC vs truth RMSE={out['M2a_circular']['pc_vs_truth']['rmse_px']:.4f}px, "
          f"PC vs primary RMSE={out['M2a_circular']['pc_vs_primary']['rmse_px']:.4f}px "
          f"(max {out['M2a_circular']['pc_vs_primary']['max_px']:.4f})")
    v = out["M2a_circular"]["trial1_battery"]["verdict"]
    print(f"  [M2a battery] PSR={v['psr_genuine']:.1f} maxctrl={v['max_control_psr']:.2f}({v['max_control_name']}) "
          f"null_max={v['noise_null_max_psr']:.2f} dPSR={v['delta_psr']:+.1f} sig={v['significant']}")
    print(f"  [M2b] PC vs truth RMSE={out['M2b_noncircular']['pc_vs_truth']['rmse_px']:.4f}px "
          f"max={out['M2b_noncircular']['pc_vs_truth']['max_px']:.4f}px")
    return out


def main():
    out_json = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_JSON
    if out_json.exists():
        sys.exit(f"Refusing to overwrite existing {out_json}")
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    results = {"settings": {"tolerance_px": TOL_PX, "noise_null_draws": N_NULL, "seed": SEED,
                            "window": "hann", "estimator": "src/corroboration/phase_correlation.py"}}

    # --- provenance --------------------------------------------------------
    h1 = np.load(CACHE / "real_flight_hop1_finetuned.npz", allow_pickle=True)
    h1_src = np.load(CACHE / "polar_flight_hop1.npz", allow_pickle=True)
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.verify_finetuned_checkpoint import assert_cache_provenance
    assert_cache_provenance(
        {k: h1_src[k] for k in h1_src.files},
        {"disp_ohrc": h1["disp_ohrc"], "disp_tmc": h1["disp_tmc"]},
        {"center_lat": float(h1["center_lat"]), "ohrc_res": float(h1["ohrc_res"]),
         "ohrc_product_id": str(h1["ohrc_product_id"])})
    h2 = np.load(CACHE / "real_flight_hop2_phase_congruency.npz", allow_pickle=True)
    h2_raw = np.load(CACHE / "real_flight_hop2.npz", allow_pickle=True)
    prov2 = {k: bool(np.array_equal(h2[k], h2_raw[k])) for k in ("disp_tmc", "disp_iirs", "raw_iirs_crop")}
    prov2["iirs_id_equal"] = str(h2["iirs_id"]) == str(h2_raw["iirs_id"])
    prov2["center_lat_equal"] = float(h2["center_lat"]) == float(h2_raw["center_lat"])
    if not all(prov2.values()):
        raise ValueError(f"Provenance violation in Hop 2 caches: {prov2}")
    results["provenance"] = {"hop1": "assert_cache_provenance(polar_flight_hop1.npz -> real_flight_hop1_finetuned.npz) passed",
                             "hop2": prov2}

    # --- Hop 1 ---------------------------------------------------------------
    print("Hop 1: OHRC -> TMC-2 (fine-tuned LoFTR cached H)")
    shiv = np.load(CACHE / "real_flight_hop1_shivshakti.npz", allow_pickle=True)
    results["hop1"] = run_hop("hop1", h1["disp_ohrc"].astype(np.float64), h1["disp_tmc"].astype(np.float64),
                              h1["H"], shiv["disp_tmc"].astype(np.float64), 0.5 * 240 / 1000, 1.020, 0.24, rng)
    results["hop1"]["offset_control_source"] = "real_flight_hop1_shivshakti.npz disp_tmc (TMC-2, Shiv Shakti, ~2000 km away)"

    # --- Hop 2 ---------------------------------------------------------------
    print("Hop 2: TMC-2 -> IIRS (phase congruency + LoFTR cached H; PC run on raw intensity canvases)")
    other = np.load(CACHE / "real_overlapping_pair.npz", allow_pickle=True)["iirs_grey"]
    off2 = cv2.resize(other, (800, 800), interpolation=cv2.INTER_CUBIC).astype(np.float64)
    results["hop2"] = run_hop("hop2", h2["disp_tmc"].astype(np.float64), h2["disp_iirs"].astype(np.float64),
                              h2["H"], off2, 0.5 * 120 / 800, 10.257, 0.15, rng)
    results["hop2"]["offset_control_source"] = "real_overlapping_pair.npz iirs_grey (IIRS, north polar site) resized to 800x800"

    # --- Pitiscus ------------------------------------------------------------
    print("Pitiscus: TMC-2 -> LROC NAC orthophoto on the 4.72 m common grid")
    ortho, tmc = build_pitiscus_grid()
    results["pitiscus"] = run_pitiscus(ortho, tmc, rng)
    print("Pitiscus POST-HOC: whole valid strip, long segments, NCC lag check")
    results["pitiscus_strip_posthoc"] = run_pitiscus_strip(ortho, tmc, (-82.12, 224.04), rng)

    # --- Matched illumination positive controls ------------------------------
    print("Matched-illumination scale-gap positive controls (Stage M geometry)")
    results["matched_illumination"] = run_matched_illumination(rng)

    results["runtime_s"] = time.time() - t0
    with open(out_json, "x") as f:
        json.dump(results, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"Saved {out_json} ({results['runtime_s']:.0f} s)")


if __name__ == "__main__":
    main()
