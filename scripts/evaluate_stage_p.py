#!/usr/bin/env python3
"""
scripts/evaluate_stage_p.py — Stage P: Chandrayaan-2 to LROC NAC orthophoto
registration at the Chandrayaan-3 (Vikram / Shiv Shakti) landing site,
NAC_DTM_VIKRAMSITE1 (LOLA-controlled, lola_rms 1.85 m).

Usage:  python scripts/evaluate_stage_p.py {tmc|ohrc} [out_json out_dir]

Design history (disclosed): a first version worked on a grid in the NAC map
CRS, as Stage O did. At this site the source swaths run diagonally across that
grid, so axis-aligned bands captured only 32-180 px of an OHRC swath and the
coarse search found nothing. This version works in the source image's own
(line, sample) frame, with the NAC resampled into it through the ISRO geometry
(src/lunar_reg/generic_grid.py: source_frame_pair). The switch was made after
seeing the first version fail; the settings below were then fixed before the
full runs.

Settings
  tmc : ch2_tmc_ncn_20230130T1900132182_d_img_d32 vs NAC_DTM_VIKRAMSITE1_M1442997156_3M.TIF
        coarse k=1 (~5 m), fine k=1, node lattice 48 px, scans 129,000-139,500
  ohrc: ch2_ohr_ncp_20211023T0027462822_d_img_d18 vs NAC_DTM_VIKRAMSITE1_M1442997156_100CM.TIF
        coarse k=15 (~4.2 m), fine k=4 (~1.1 m), node lattice 96 px, scans 11,400-78,000
  Coarse: 2000-row segments stepped by 400 rows; columns >= 98% reference-valid,
    longest contiguous run; holes filled with the segment mean. Stage N phase
    correlation per segment, 7 core controls + 50-draw noise null; a segment is
    significant if its PSR beats every control and the null maximum. Prior =
    median of significant segments; abort unless >= 2 agree within 5 px.
  Fine: dense NCC, template 64 px, radius 32 px, peak >= 0.5, second < 0.9 * peak,
    not on edge. RANSAC 1.5 px. Models translation .. poly3 (RANSAC) and an
    along-track penalised spline (knots every knot_px rows, cross-track
    quadratic, lambda by GCV on fitting data, iterative trimming at 1.5 px).
    Held-out schemes: stripes (500-row interleaved), spatial blocks (5
    contiguous), random 5-fold; all reported. Chosen by stripes held-out RMSE
    (trim 3 px), simpler within 0.05 px. Change from Stage O (which selected on
    spatial blocks), made after the polynomial models underfit the OHRC strip:
    nodes cover the whole strip, so spatial blocks measure extrapolation over
    3+ km gaps that the product never has to bridge.
  Controls on the fine run: rot90/rot270 (blockwise, square blocks of frame
    width), rot180, vflip, hflip, non-overlapping offset (reference rolled by
    half the frame height), uniform noise inside the reference-valid mask.
    Delta_shuffle gate +15 pp.

Outputs (new files only):
  assets/real_cache/stage_p_vikram_<pair>.json
  results/stage_p/<pair>/{tiepoints_<pair>_nac_vikram.csv, tiepoints_<pair>_nac_vikram_inliers.geojson,
                          registered_<pair>_vikram.tif, figures/*.png}
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.lunar_reg import dense_ncc, generic_grid as gg, models, products, validate  # noqa: E402

CACHE = REPO_ROOT / "assets" / "real_cache"
DESK = Path.home() / "Desktop" / "data"
NAC_DIR = REPO_ROOT / "data" / "lroc_nac" / "VIKRAMSITE1"
TMC_DIR = REPO_ROOT / "data" / "ch2_tmc_ncn_20230130T1900132182_d_img_d32"
LOLA_RMS_M = 1.85

PAIRS = {
    "tmc": {
        "source_id": "ch2_tmc_ncn_20230130T1900132182_d_img_d32",
        "img": TMC_DIR / "data/calibrated/20230130/ch2_tmc_ncn_20230130T1900132182_d_img_d32.img",
        "shape": (270516, 4000), "dtype": "<u2",
        "geo": TMC_DIR / "geometry/calibrated/20230130/ch2_tmc_ncn_20230130T1900132182_g_grd_d32.csv",
        "ref": NAC_DIR / "NAC_DTM_VIKRAMSITE1_M1442997156_3M.TIF",
        "scans": (129000, 139500), "k_coarse": 1, "k_fine": 1, "spacing": 48, "knot_px": 500,
        "source_sun": "label (-70.85, 32.26): az 53.02 deg, el 17.22 deg",
        "attitude": {"roll_deg": -0.021530, "pitch_deg": -0.016459, "altitude_km": 94.40},
    },
    "ohrc": {
        "source_id": "ch2_ohr_ncp_20211023T0027462822_d_img_d18",
        "img": DESK / "data/calibrated/20211023/ch2_ohr_ncp_20211023T0027462822_d_img_d18.img",
        "shape": (93693, 12000), "dtype": "u1",
        "geo": DESK / "geometry/calibrated/20211023/ch2_ohr_ncp_20211023T0027462822_g_grd_d18.csv",
        "ref": NAC_DIR / "NAC_DTM_VIKRAMSITE1_M1442997156_100CM.TIF",
        "scans": (11400, 78000), "k_coarse": 15, "k_fine": 4, "spacing": 96, "knot_px": 1000,
        "source_sun": "label: az 298.43 deg, el 9.13 deg",
        "attitude": {"roll_deg": 15.757549, "pitch_deg": 4.516213, "altitude_km": 102.02},
    },
}
REF_SUN = "NAC M1442997156 (ODE): incidence 73.84 / 73.94 deg (sun el ~16 deg), 2023-07-03"

HALF_TPL, RADIUS, PEAK_MIN, UNIQ_MAX = 32, 32, 0.5, 0.9
RANSAC_PX, SIMPLER_WITHIN_PX, GATE_PP = 1.5, 0.05, 15.0
KINDS = ["translation", "similarity", "affine", "poly2", "poly3", "pspline", "pspline_relief"]
DTM = NAC_DIR / "NAC_DTM_VIKRAMSITE1.TIF"
R_MOON_KM = 1737.4
SCHEMES = ["stripes", "spatial_blocks", "random_kfold"]
SELECT_SCHEME = "stripes"
SEG_H, SEG_STEP, COL_VALID_MIN, MIN_BAND, PRIOR_AGREE_PX = 2000, 400, 0.98, 32, 5.0

_spec = importlib.util.spec_from_file_location("stage_n_pc", REPO_ROOT / "src/corroboration/phase_correlation.py")
pcm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pcm)


def blockwise_rot(img, k):
    h, w = img.shape
    out = img.copy()
    y = 0
    while y + w <= h:
        out[y:y + w] = np.rot90(img[y:y + w], k)
        y += w
    if y < h:
        out[y:] = np.rot90(img[y:], 2)
    return out


def _fill(a, v):
    a = a.copy()
    a[~v] = a[v].mean() if v.any() else 0
    return a


def coarse_prior(src, ref, rval, rng):
    segs = []
    H = src.shape[0]
    ref_far = np.roll(ref, H // 2, axis=0)
    val_far = np.roll(rval, H // 2, axis=0)
    for y0 in range(0, H - SEG_H + 1, SEG_STEP):
        cv = rval[y0:y0 + SEG_H].mean(axis=0) >= COL_VALID_MIN
        c = np.where(cv)[0]
        if len(c) < MIN_BAND:
            continue
        run = max(np.split(c, np.where(np.diff(c) != 1)[0] + 1), key=len)
        if len(run) < MIN_BAND:
            continue
        sl = (slice(y0, y0 + SEG_H), slice(run[0], run[-1] + 1))
        v = rval[sl]
        f, mv = src[sl], _fill(ref[sl], v)
        g = pcm.estimate_shift(f, mv)
        far = _fill(ref_far[sl], val_far[sl])
        ctrl = {"rot180": np.rot90(mv, 2), "vflip": mv[::-1], "hflip": mv[:, ::-1],
                "rot90_blockwise": blockwise_rot(mv, 1), "rot270_blockwise": blockwise_rot(mv, 3),
                "offset_nonoverlap": far, "uniform_noise": rng.uniform(0, 255, mv.shape)}
        cps = {k: pcm.estimate_shift(f, x)["psr"] for k, x in ctrl.items() if np.std(x) > 0}
        null = max(pcm.estimate_shift(f, rng.uniform(0, 255, mv.shape))["psr"] for _ in range(50))
        sig = bool(g["psr"] > max(cps.values()) and g["psr"] > null)
        segs.append({"y0": y0, "cols": [int(run[0]), int(run[-1])], "dx": g["dx"], "dy": g["dy"], "psr": g["psr"],
                     "controls_psr": cps, "controls_not_applicable": sorted(set(ctrl) - set(cps)),
                     "null_max_psr": null, "significant": sig})
        print(f"  seg rows {y0}-{y0 + SEG_H} cols {run[0]}-{run[-1]}: ({g['dx']:+.2f},{g['dy']:+.2f}) "
              f"PSR {g['psr']:.1f} ctrl {max(cps.values()):.1f} null {null:.1f} {'SIG' if sig else '-'}")
    s = [x for x in segs if x["significant"]]
    if len(s) < 2:
        return None, segs
    med = np.median([[x["dx"], x["dy"]] for x in s], axis=0)
    if sum(np.hypot(x["dx"] - med[0], x["dy"] - med[1]) <= PRIOR_AGREE_PX for x in s) < 2:
        return None, segs
    return (float(med[0]), float(med[1])), segs


def match(src, ref, sval, rval, prior, spacing):
    m = dense_ncc.match_nodes(src, ref, prior, spacing, HALF_TPL, RADIUS, src_valid=sval, ref_valid=rval)
    m["accepted"] = dense_ncc.accept(m, PEAK_MIN, UNIQ_MAX)
    return m


def consensus(m, kind):
    acc, n = m["accepted"], len(m["peak"])
    if acc.sum() < 20:
        return {"n_evaluated": n, "n_accepted": int(acc.sum()), "n_inliers": 0,
                "acceptance_ratio_pct": 100.0 * acc.sum() / max(n, 1), "inlier_ratio_pct": 0.0}
    r = fit_kind(kind, m["src_xy"][acc], m["ref_xy"][acc], PAIR)
    k = int(r["inliers"].sum()) if r else 0
    return {"n_evaluated": n, "n_accepted": int(acc.sum()), "n_inliers": k,
            "acceptance_ratio_pct": 100.0 * acc.sum() / max(n, 1), "inlier_ratio_pct": 100.0 * k / max(n, 1)}


def bilinear(a, xy):
    from scipy.ndimage import map_coordinates
    return map_coordinates(np.asarray(a, dtype=np.float64), [np.asarray(xy)[:, 1], np.asarray(xy)[:, 0]],
                           order=1, mode="nearest")


def height_lookup(dtm):
    from scipy.ndimage import map_coordinates
    def f(xy):
        xy = np.asarray(xy, dtype=np.float64)
        return map_coordinates(dtm, [xy[:, 1], xy[:, 0]], order=1, mode="constant", cval=np.nan)
    return f


def fit_kind(kind, S, R, pair):
    if kind == "pspline_relief":
        return models.pspline_relief_trimmed(S, R, RANSAC_PX, pair["knot_px"], HEIGHT_AT)
    if kind == "pspline":
        return models.pspline_trimmed(S, R, RANSAC_PX, pair["knot_px"])
    return models.ransac(S, R, kind, RANSAC_PX)


def parallax_check(model, fp, pair):
    """Compare the fitted relief parallax with the view angle predicted from the label.
    The fitted vector p (frame px per metre of height) is converted to ground metres
    with the local Jacobian of the frame -> map transform (median over the frame)."""
    if getattr(model, "kind", "") != "pspline_relief":
        return None
    p = np.asarray(model.p)
    J = np.array([[np.nanmedian(np.diff(fp["map_x"], axis=1)), np.nanmedian(np.diff(fp["map_x"], axis=0))],
                  [np.nanmedian(np.diff(fp["map_y"], axis=1)), np.nanmedian(np.diff(fp["map_y"], axis=0))]])
    dm = J @ p  # metres of ground displacement per metre of height
    tan_fit = float(np.hypot(*dm))
    out = {"parallax_px_per_m": p.tolist(), "parallax_ground_m_per_m": dm.tolist(), "frame_jacobian_m_per_px": J.tolist(),
           "tan_emission_fitted": tan_fit,
           "emission_deg_fitted": float(np.degrees(np.arctan(tan_fit)))}
    att = pair.get("attitude")
    if att:
        off = np.arctan(np.hypot(np.tan(np.radians(att["roll_deg"])), np.tan(np.radians(att["pitch_deg"]))))
        e = np.arcsin(min((R_MOON_KM + att["altitude_km"]) / R_MOON_KM * np.sin(off), 1.0))
        out.update({"label_off_nadir_deg": float(np.degrees(off)), "label_emission_deg_predicted": float(np.degrees(e)),
                    "difference_deg": float(np.degrees(np.arctan(tan_fit)) - np.degrees(e)),
                    "label_source": f"roll {att['roll_deg']}, pitch {att['pitch_deg']}, altitude {att['altitude_km']} km"})
    return out


def main():
    global PAIR, HEIGHT_AT
    pair_name = sys.argv[1]
    pair = PAIRS[pair_name]
    PAIR = pair
    out_json = Path(sys.argv[2]) if len(sys.argv) == 4 else CACHE / f"stage_p_vikram_{pair_name}.json"
    out_dir = Path(sys.argv[3]) if len(sys.argv) == 4 else REPO_ROOT / "results" / "stage_p" / pair_name
    for p in (out_json, out_dir):
        if p.exists():
            sys.exit(f"Refusing to overwrite existing {p}")
    t0 = time.time()
    rng = np.random.default_rng(42)
    kc, kf = pair["k_coarse"], pair["k_fine"]

    # ---- coarse level ---------------------------------------------------------
    co = gg.source_frame_pair(pair["ref"], pair["img"], pair["shape"], pair["dtype"], pair["geo"], pair["scans"], kc)
    print(f"[{pair_name}] coarse frame {co['src'].shape} at {co['ground_m_per_px']:.2f} m/px, "
          f"ref valid {co['ref_valid'].mean() * 100:.1f}% ({time.time() - t0:.0f} s)")
    prior_c, segs = coarse_prior(co["src"], co["ref"], co["ref_valid"], rng)
    if prior_c is None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "x") as f:
            json.dump({"pair": pair_name, "status": "NO_SIGNIFICANT_COARSE_OFFSET", "coarse_segments": segs}, f, indent=2,
                      default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        sys.exit(f"[{pair_name}] no significant coarse offset; recorded")
    coarse_gpx = co["ground_m_per_px"]
    prior_m = (prior_c[0] * coarse_gpx, prior_c[1] * coarse_gpx)
    print(f"[{pair_name}] coarse prior ({prior_c[0]:+.2f}, {prior_c[1]:+.2f}) px = "
          f"({prior_c[0] * kc:+.0f} samples, {prior_c[1] * kc:+.0f} lines) ~ {np.hypot(*prior_m):.0f} m")
    del co

    # ---- fine level ------------------------------------------------------------
    fp = gg.source_frame_pair(pair["ref"], pair["img"], pair["shape"], pair["dtype"], pair["geo"], pair["scans"], kf,
                              dtm_tif=DTM)
    HEIGHT_AT = height_lookup(fp["dtm"])
    src, ref, rval = fp["src"], fp["ref"], fp["ref_valid"]
    sval = np.ones_like(rval)  # source zeros are data (shadow), not nodata
    gpx = fp["ground_m_per_px"]
    prior = (prior_c[0] * kc / kf, prior_c[1] * kc / kf)
    print(f"[{pair_name}] fine frame {src.shape} at {gpx:.2f} m/px; prior ({prior[0]:+.1f}, {prior[1]:+.1f}) px")
    gen = match(src, ref, sval, rval, prior, pair["spacing"])
    acc = gen["accepted"]
    S, R = gen["src_xy"][acc], gen["ref_xy"][acc]
    print(f"[{pair_name}] nodes {len(gen['peak'])}, accepted {acc.sum()} ({100 * acc.mean():.1f}%), mean peak {gen['peak'].mean():.3f}")
    if acc.sum() < 30:
        sys.exit(f"[{pair_name}] too few accepted nodes")

    sel = {}
    for kind in KINDS:
        sel[kind] = {sc: validate.heldout(S, R, kind, RANSAC_PX, sc, knot_px=pair["knot_px"], height_at=HEIGHT_AT)
                     for sc in SCHEMES}
        full = fit_kind(kind, S, R, pair)
        sel[kind]["insample_inliers"] = int(full["inliers"].sum()) if full else 0
        sel[kind]["insample_rmse_px"] = (float(np.sqrt((full["residuals"][full["inliers"]] ** 2).mean()))
                                         if full and full["inliers"].any() else None)
        f3 = lambda o: f"{o['rmse_within_trim_px']:.3f} w1 {o['frac_within_1px']:.2f} w3 {o['frac_within_3px']:.2f}"
        print(f"  {kind:11s} in {sel[kind]['insample_inliers']:5d} @ {sel[kind]['insample_rmse_px']:.3f} | "
              + " | ".join(f"{sc} {f3(sel[kind][sc]['overall'])}" for sc in SCHEMES))
    scores = {k: sel[k][SELECT_SCHEME]["overall"]["rmse_within_trim_px"] for k in KINDS
              if sel[k][SELECT_SCHEME]["overall"]["rmse_within_trim_px"] is not None}
    best = min(scores.values())
    chosen = next(k for k in KINDS if k in scores and scores[k] <= best + SIMPLER_WITHIN_PX)
    final = fit_kind(chosen, S, R, pair)
    model, inl, resid = final["model"], final["inliers"], final["residuals"]
    print(f"[{pair_name}] chosen {chosen}: {inl.sum()} inliers, in-sample {np.sqrt((resid[inl] ** 2).mean()):.3f} px")

    H = ref.shape[0]
    noise = np.where(rval, rng.uniform(1, 255, ref.shape), 0).astype(np.float32)
    ctrl_refs = {
        "rot90_blockwise": (blockwise_rot(ref, 1), blockwise_rot(rval.astype(np.uint8), 1).astype(bool)),
        "rot180": (np.rot90(ref, 2), np.rot90(rval, 2)),
        "rot270_blockwise": (blockwise_rot(ref, 3), blockwise_rot(rval.astype(np.uint8), 3).astype(bool)),
        "vflip": (ref[::-1], rval[::-1]),
        "hflip": (ref[:, ::-1], rval[:, ::-1]),
        "offset_nonoverlap_half_height": (np.roll(ref, H // 2, axis=0), np.roll(rval, H // 2, axis=0)),
        "uniform_noise": (noise, rval),
    }
    genuine_c = consensus(gen, chosen)
    controls = {}
    for name, (rc, vc) in ctrl_refs.items():
        controls[name] = consensus(match(src, rc, sval, vc, prior, pair["spacing"]), chosen)
        c = controls[name]
        print(f"  control {name:30s} acc {c['acceptance_ratio_pct']:5.2f}% inl {c['n_inliers']:4d} ({c['inlier_ratio_pct']:5.2f}%)")
    worst = max(controls, key=lambda k: controls[k]["inlier_ratio_pct"])
    delta = genuine_c["inlier_ratio_pct"] - controls[worst]["inlier_ratio_pct"]
    print(f"[{pair_name}] genuine {genuine_c['inlier_ratio_pct']:.2f}% vs worst {worst} "
          f"{controls[worst]['inlier_ratio_pct']:.2f}% -> Delta_shuffle {delta:+.2f} pp ({'PASS' if delta >= GATE_PP else 'GATED'})")

    # ---- tie points: source (line, sample) -> reference map (x, y) -------------
    k = fp["decimate"]
    src_line = fp["scan_range"][0] + S[:, 1] * k + (k - 1) / 2.0
    src_sample = S[:, 0] * k + (k - 1) / 2.0
    ref_x, ref_y = bilinear(fp["map_x"], R), bilinear(fp["map_y"], R)
    geo_x, geo_y = bilinear(fp["map_x"], S), bilinear(fp["map_y"], S)  # where the ISRO geometry puts the node
    gridlike = {"crs_wkt": fp["crs_wkt"], "ref_transform": fp["ref_transform"]}
    lon, lat = gg.map_to_lonlat(gridlike, ref_x, ref_y)
    rcol, rrow = gg.map_to_ref_pixel(gridlike, ref_x, ref_y)
    hold = np.array(sel[chosen][SELECT_SCHEME]["heldout_residuals"])
    geo_err = np.column_stack([ref_x - geo_x, ref_y - geo_y])
    csv_path = products.write_csv(
        out_dir / f"tiepoints_{pair_name}_nac_vikram.csv",
        [f"Stage P tie points: {pair['source_id']} (source) to {pair['ref'].name} (reference), Chandrayaan-3 landing site.",
         "src_line/src_sample: 0-based position in the full-resolution source .img.",
         "ref_x/ref_y: reference CRS (polar stereographic, lat_ts -69.3, lon_0 32.3, R 1737400 m); ref_col/ref_row: 0-based pixel centre in the reference GeoTIFF.",
         "isro_geo_dx/dy_m: reference position minus the position the ISRO geometry CSV assigns to the same source pixel.",
         f"Residuals in fine source-frame px (1 px = {gpx:.3f} m on the ground). heldout_residual_px: {SELECT_SCHEME} scheme, model never saw the node.",
         f"Reference error: lola_rms {LOLA_RMS_M} m. Residuals are relative to the reference, not ground-truth errors."],
        {"id": np.arange(len(S)), "src_line": src_line, "src_sample": src_sample, "ref_x": ref_x, "ref_y": ref_y,
         "lon": lon, "lat": lat, "ref_col": rcol, "ref_row": rrow,
         "isro_geo_dx_m": geo_err[:, 0], "isro_geo_dy_m": geo_err[:, 1],
         "ncc_peak": gen["peak"][acc], "ncc_second": gen["second"][acc], "fit_residual_px": resid,
         "is_inlier": inl.astype(int), "heldout_fold": np.array(sel[chosen][SELECT_SCHEME]["fold"]),
         "heldout_residual_px": hold},
        {"src_line": ".2f", "src_sample": ".2f", "ref_x": ".3f", "ref_y": ".3f", "lon": ".8f", "lat": ".8f",
         "ref_col": ".3f", "ref_row": ".3f", "isro_geo_dx_m": ".2f", "isro_geo_dy_m": ".2f", "ncc_peak": ".4f",
         "ncc_second": ".4f", "fit_residual_px": ".4f", "heldout_residual_px": ".4f"})
    gj_path = products.write_geojson(out_dir / f"tiepoints_{pair_name}_nac_vikram_inliers.geojson", lon[inl], lat[inl],
                                     {"id": np.arange(len(S))[inl], "src_line": np.round(src_line[inl], 2),
                                      "src_sample": np.round(src_sample[inl], 2),
                                      "fit_residual_px": np.round(resid[inl], 4), "heldout_residual_px": np.round(hold[inl], 4)},
                                     f"Stage P inliers, lon/lat on Moon sphere R=1737400 m. Reference lola_rms {LOLA_RMS_M} m; not ground truth.")

    # ---- registered product: source resampled onto a map grid in the reference CRS ----
    from scipy.interpolate import LinearNDInterpolator
    import rasterio
    from rasterio.transform import Affine
    step = 16
    ly, lx = np.mgrid[0:src.shape[0]:step, 0:src.shape[1]:step]
    lat_src = np.column_stack([lx.ravel(), ly.ravel()]).astype(np.float64)
    pos = model.predict(lat_src)  # position in the reference-in-source frame
    inside = (pos[:, 0] >= 0) & (pos[:, 0] <= src.shape[1] - 1) & (pos[:, 1] >= 0) & (pos[:, 1] <= src.shape[0] - 1)
    mxy = np.column_stack([bilinear(fp["map_x"], pos), bilinear(fp["map_y"], pos)])
    okp = inside & np.isfinite(mxy).all(1)
    inv_c = LinearNDInterpolator(mxy[okp], lat_src[okp, 0])
    inv_r = LinearNDInterpolator(mxy[okp], lat_src[okp, 1])
    out_gsd = gpx
    x0, x1 = np.floor(mxy[okp, 0].min()), np.ceil(mxy[okp, 0].max())
    y0, y1 = np.floor(mxy[okp, 1].min()), np.ceil(mxy[okp, 1].max())
    gx = x0 + out_gsd / 2 + np.arange(int((x1 - x0) / out_gsd)) * out_gsd
    gy = y1 - out_gsd / 2 - np.arange(int((y1 - y0) / out_gsd)) * out_gsd
    cst = 8
    GX, GY = np.meshgrid(gx[::cst], gy[::cst])
    cc, rr_ = inv_c(GX, GY), inv_r(GX, GY)
    FX, FY = np.meshgrid((np.arange(len(gx)) / cst).astype(np.float32), (np.arange(len(gy)) / cst).astype(np.float32))
    mc = cv2.remap(np.nan_to_num(cc, nan=-1e6).astype(np.float32), FX, FY, cv2.INTER_LINEAR)
    mr = cv2.remap(np.nan_to_num(rr_, nan=-1e6).astype(np.float32), FX, FY, cv2.INTER_LINEAR)
    registered = cv2.remap(src, mc, mr, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    del mc, mr, FX, FY
    tif_path = out_dir / f"registered_{pair_name}_vikram.tif"
    tif_path.parent.mkdir(parents=True, exist_ok=True)
    if tif_path.exists():
        sys.exit(f"Refusing to overwrite {tif_path}")
    out_tr = Affine(out_gsd, 0, x0, 0, -out_gsd, y1)
    with rasterio.open(tif_path, "w", driver="GTiff", height=registered.shape[0], width=registered.shape[1], count=1,
                       dtype="float32", crs=fp["crs_wkt"], transform=out_tr, nodata=0.0, compress="deflate", tiled=True) as dst:
        dst.write(registered.astype(np.float32), 1)
        dst.update_tags(source_product=pair["source_id"], reference_product=pair["ref"].name, model=chosen,
                        reference_lola_rms_m=str(LOLA_RMS_M), stage="P", source_decimation=str(k),
                        note="Source resampled into the reference map frame by the Stage P model. Not ground truth.")

    # ---- independent check: product vs NAC reprojected by rasterio, Stage N estimator ----
    from rasterio.warp import reproject, Resampling
    ref_on = np.zeros(registered.shape, np.float32)
    with rasterio.open(pair["ref"]) as s_:
        reproject(rasterio.band(s_, 1), ref_on, dst_transform=out_tr, dst_crs=fp["crs_wkt"],
                  resampling=Resampling.average, src_nodata=0, dst_nodata=0)
    closure = []
    T = 512
    for yy in range(0, registered.shape[0] - T, T):
        cols = np.where(((registered[yy:yy + T] > 0) & (ref_on[yy:yy + T] > 0)).all(axis=0))[0]
        if len(cols) < T:
            continue
        run = max(np.split(cols, np.where(np.diff(cols) != 1)[0] + 1), key=len)
        if len(run) < T:
            continue
        xx = int(run[0] + (len(run) - T) // 2)
        e = pcm.estimate_shift(registered[yy:yy + T, xx:xx + T], ref_on[yy:yy + T, xx:xx + T])
        closure.append({"tile_xy": [xx, yy], "dx": e["dx"], "dy": e["dy"], "psr": e["psr"]})
    if closure:
        d = np.array([[c["dx"], c["dy"]] for c in closure])
        print(f"[{pair_name}] product vs rasterio-reprojected NAC over {len(closure)} tiles: median |shift| "
              f"{np.median(np.hypot(*d.T)):.3f} px, max {np.hypot(*d.T).max():.3f} px, min PSR {min(c['psr'] for c in closure):.1f}")

    # ---- figures -------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=False)
    ok = np.isfinite(hold) & (hold < 3)
    fig, ax = plt.subplots(1, 3, figsize=(14, 10), dpi=110)
    for a_, (xy, rv, t) in zip(ax[:2], [(S[inl], resid[inl], "In-sample residuals (inliers)"),
                                         (S[ok], hold[ok], f"{SELECT_SCHEME} held-out residuals (< 3 px)")]):
        sc = a_.scatter(xy[:, 0], xy[:, 1], c=rv, s=4, cmap="viridis", vmin=0, vmax=1.5)
        a_.invert_yaxis(); a_.set_title(t, fontsize=9); a_.set_xlabel("sample (fine px)"); a_.set_ylabel("line (fine px)")
    fig.colorbar(sc, ax=ax[:2], shrink=0.6, label=f"residual (px, 1 px = {gpx:.2f} m)")
    sub = np.where(inl)[0][::max(1, inl.sum() // 400)]
    ax[2].quiver(geo_x[sub], geo_y[sub], geo_err[sub, 0], geo_err[sub, 1], np.hypot(*geo_err[sub].T), angles="xy",
                 scale_units="xy", scale=1, width=0.004)
    ax[2].set_aspect("equal"); ax[2].set_title("ISRO geometry position -> NAC position (m, to scale)", fontsize=9)
    fig.savefig(fig_dir / f"stage_p_{pair_name}_residuals_and_geometry_offset.png", bbox_inches="tight"); plt.close(fig)
    ch = [c for c in closure] or [{"tile_xy": [0, 0]}]
    xx, yy = ch[len(ch) // 2]["tile_xy"]
    def nrm(a):
        v = a[a > 0]
        lo, hi = np.percentile(v, (1, 99)) if v.size else (0, 1)
        return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    rt, gt_ = nrm(ref_on[yy:yy + T, xx:xx + T]), nrm(registered[yy:yy + T, xx:xx + T])
    cb = ((np.indices(rt.shape) // 64).sum(0) % 2).astype(bool)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.6), dpi=120)
    ax[0].imshow(np.where(cb, rt, gt_), cmap="gray"); ax[0].set_title(f"NAC / registered {pair_name.upper()} checkerboard")
    ax[1].imshow(np.dstack([rt, gt_, rt])); ax[1].set_title("magenta = NAC, green = registered source")
    for a_ in ax: a_.axis("off")
    fig.suptitle(f"Chandrayaan-3 landing site: {pair['source_id']} vs {pair['ref'].name}, {T} px at {out_gsd:.2f} m", fontsize=9)
    fig.savefig(fig_dir / f"stage_p_{pair_name}_checkerboard.png", bbox_inches="tight"); plt.close(fig)

    strip = lambda d_: {kk: vv for kk, vv in d_.items() if kk not in ("heldout_residuals", "fold")}
    ge = np.hypot(*geo_err[inl].T)
    results = {
        "pair": pair_name, "status": "COMPLETED", "source_product": pair["source_id"], "reference": pair["ref"].name,
        "source_sun": pair["source_sun"], "reference_sun": REF_SUN, "reference_lola_rms_m": LOLA_RMS_M,
        "frame": {"scan_range": list(fp["scan_range"]), "k_coarse": kc, "k_fine": kf, "fine_shape": list(src.shape),
                  "fine_ground_m_per_px": gpx, "coarse_ground_m_per_px": coarse_gpx, "ref_valid_pct": float(rval.mean() * 100)},
        "coarse_prior_px": prior_c, "coarse_prior_m": list(prior_m), "coarse_segments": segs,
        "isro_geometry_offset_m": {"median_dx": float(np.median(geo_err[inl, 0])), "median_dy": float(np.median(geo_err[inl, 1])),
                                   "median_norm": float(np.median(ge)), "p5_norm": float(np.percentile(ge, 5)),
                                   "p95_norm": float(np.percentile(ge, 95))},
        "genuine": {"n_evaluated": int(len(gen["peak"])), "n_accepted": int(acc.sum()),
                    "mean_peak": float(gen["peak"].mean()), "consensus": genuine_c},
        "model_selection": {kk: {"insample_inliers": v["insample_inliers"], "insample_rmse_px": v["insample_rmse_px"],
                                 **{sc: strip(v[sc]) for sc in SCHEMES}} for kk, v in sel.items()},
        "selection_scheme": SELECT_SCHEME,
        "chosen_model": chosen, "model_params": model.params(),
        "relief_parallax_check": parallax_check(model, fp, pair),
        "final": {"n_inliers": int(inl.sum()), "insample_rmse_px": float(np.sqrt((resid[inl] ** 2).mean()))},
        "controls": controls,
        "delta_shuffle": {"worst_control": worst, "value_pp": float(delta), "gate_pp": GATE_PP, "pass": bool(delta >= GATE_PP)},
        "product_check_vs_rasterio_reprojected_nac": closure,
        "products": {"registered_geotiff": str(tif_path), "tiepoints_csv": csv_path, "tiepoints_geojson": gj_path},
        "runtime_s": time.time() - t0,
    }
    with open(out_json, "x") as f:
        json.dump(results, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"[{pair_name}] ISRO geometry offset (inliers): median ({results['isro_geometry_offset_m']['median_dx']:+.1f}, "
          f"{results['isro_geometry_offset_m']['median_dy']:+.1f}) m, |.| {results['isro_geometry_offset_m']['median_norm']:.1f} m")
    if results["relief_parallax_check"]:
        print(f"[{pair_name}] relief parallax check: {results['relief_parallax_check']}")
    print(f"[{pair_name}] saved {out_json} ({results['runtime_s']:.0f} s)")


if __name__ == "__main__":
    main()
