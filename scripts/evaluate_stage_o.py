#!/usr/bin/env python3
"""
scripts/evaluate_stage_o.py — Stage O: Chandrayaan-2 TMC-2 to LROC NAC orthophoto
registration at Pitiscus, seeded by the Stage N independent coarse offset.

Pre-declared settings (fixed before running on the genuine pair):
  * Prior offset: Stage N whole-strip phase-correlation estimate, read from
    assets/real_cache/stage_n_corroboration_results.json (must be significant).
  * Nodes: 48 px lattice, 64 px template, search radius 32 px.
  * Node acceptance: NCC peak >= 0.5, second peak < 0.9 * peak, not on the
    search edge.
  * RANSAC inlier threshold: 1.5 common-grid px (7.08 m).
  * Candidate models: translation, similarity, affine, poly2, poly3. Chosen
    model = lowest spatial-block held-out RMSE (trim 3 px); a simpler model is
    preferred when within 0.05 px of the best.
  * Controls (all seven, each replacing the NAC reference, same prior offset):
    rot90 / rot270 (blockwise, square 1339 px blocks), rot180, vflip, hflip,
    non-overlapping offset (NAC rolled 2000 rows = 9.4 km along track), and
    uniform noise inside the NAC valid mask.
  * Delta_shuffle = genuine RANSAC inlier ratio - max(control RANSAC inlier
    ratio), ratios over nodes evaluated. Gate: >= +15 percentage points.

Outputs (all new; aborts if any exists):
  assets/real_cache/stage_o_pitiscus_registration.json
  results/stage_o/registered_tmc2_pitiscus.tif
  results/stage_o/tiepoints_tmc2_nac_pitiscus.csv
  results/stage_o/tiepoints_tmc2_nac_pitiscus_inliers.geojson
  results/stage_o/figures/*.png
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.lunar_reg import dense_ncc, models, pitiscus_grid as pg, products, validate  # noqa: E402

CACHE = REPO_ROOT / "assets" / "real_cache"
OUT_JSON = CACHE / "stage_o_pitiscus_registration.json"
OUT_DIR = REPO_ROOT / "results" / "stage_o"

SPACING, HALF_TPL, RADIUS = 48, 32, 32
PEAK_MIN, UNIQ_MAX = 0.5, 0.9
RANSAC_PX = 1.5
KINDS = ["translation", "similarity", "affine", "poly2", "poly3"]
SIMPLER_WITHIN_PX = 0.05
GATE_PP = 15.0
LOLA_RMS_M = 0.92

_spec = importlib.util.spec_from_file_location(
    "stage_n_pc", REPO_ROOT / "src" / "corroboration" / "phase_correlation.py")
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


def run_matching(src, ref, ref_valid, offset):
    m = dense_ncc.match_nodes(src, ref, offset, SPACING, HALF_TPL, RADIUS,
                              src_valid=src > 0, ref_valid=ref_valid)
    m["accepted"] = dense_ncc.accept(m, PEAK_MIN, UNIQ_MAX)
    return m


def consensus(m, kind):
    acc = m["accepted"]
    n_eval = len(m["peak"])
    if acc.sum() < 2 * models.MIN_SAMPLES[kind]:
        return {"n_evaluated": n_eval, "n_accepted": int(acc.sum()), "n_inliers": 0,
                "acceptance_ratio_pct": 100.0 * acc.sum() / max(n_eval, 1), "inlier_ratio_pct": 0.0}
    r = models.ransac(m["src_xy"][acc], m["ref_xy"][acc], kind, RANSAC_PX)
    n_inl = int(r["inliers"].sum()) if r else 0
    return {"n_evaluated": n_eval, "n_accepted": int(acc.sum()), "n_inliers": n_inl,
            "acceptance_ratio_pct": 100.0 * acc.sum() / max(n_eval, 1),
            "inlier_ratio_pct": 100.0 * n_inl / max(n_eval, 1),
            "inlier_rmse_px": float(np.sqrt((r["residuals"][r["inliers"]] ** 2).mean())) if r and n_inl else None}


def uniformity(src_xy_all, src_xy_inl, n=8):
    lo, hi = src_xy_all.min(0), src_xy_all.max(0) + 1e-9
    def cells(xy):
        c = np.floor((xy - lo) / (hi - lo) * n).astype(int).clip(0, n - 1)
        return c[:, 1] * n + c[:, 0]
    avail = np.bincount(cells(src_xy_all), minlength=n * n) > 0
    cnt = np.bincount(cells(src_xy_inl), minlength=n * n)[avail]
    occ = cnt > 0
    return {"grid": f"{n}x{n} over the node bounding box", "cells_with_nodes": int(avail.sum()),
            "cells_with_inliers": int(occ.sum()), "coverage_pct": 100.0 * occ.sum() / avail.sum(),
            "cv_inliers_per_cell": float(cnt.std() / cnt.mean()) if cnt.mean() > 0 else None}


def main():
    global OUT_JSON, OUT_DIR
    if len(sys.argv) == 3:  # development runs write elsewhere
        OUT_JSON, OUT_DIR = Path(sys.argv[1]), Path(sys.argv[2])
    outs = [OUT_JSON, OUT_DIR]
    for p in outs:
        if p.exists():
            sys.exit(f"Refusing to overwrite existing {p}")
    t0 = time.time()
    stage_n = json.load(open(CACHE / "stage_n_corroboration_results.json"))
    whole = stage_n["pitiscus_strip_posthoc"]["whole_strip"]
    assert whole["vs_primary_bulk"]["significant"], "Stage N prior offset is not significant"
    prior = (whole["genuine"]["dx"], whole["genuine"]["dy"])
    print(f"Prior offset from Stage N: ({prior[0]:+.2f}, {prior[1]:+.2f}) px, PSR {whole['genuine']['psr']:.1f}")

    g = pg.build(REPO_ROOT)
    tmc, ortho, valid = g["tmc"], g["ortho"], g["valid_nac"]
    print(f"Grid {g['shape']} built ({time.time() - t0:.0f} s)")

    # ---- genuine matching ------------------------------------------------
    gen = run_matching(tmc, ortho, valid, prior)
    acc = gen["accepted"]
    S, R = gen["src_xy"][acc], gen["ref_xy"][acc]
    print(f"Genuine: {len(gen['peak'])} nodes evaluated, {acc.sum()} accepted, mean peak {gen['peak'].mean():.3f}")

    # ---- model selection by held-out error --------------------------------
    sel = {}
    for kind in KINDS:
        cv_s = validate.heldout(S, R, kind, RANSAC_PX, "spatial_blocks")
        cv_r = validate.heldout(S, R, kind, RANSAC_PX, "random_kfold")
        full = models.ransac(S, R, kind, RANSAC_PX)
        sel[kind] = {"spatial_blocks": cv_s, "random_kfold": cv_r,
                     "insample_inliers": int(full["inliers"].sum()),
                     "insample_rmse_px": float(np.sqrt((full["residuals"][full["inliers"]] ** 2).mean()))}
        o = cv_s["overall"]
        print(f"  {kind:11s} in-sample {sel[kind]['insample_inliers']:4d} inl @ {sel[kind]['insample_rmse_px']:.3f} px | "
              f"spatial held-out RMSE {o['rmse_within_trim_px']:.3f} px, within1 {o['frac_within_1px']:.2f}, within3 {o['frac_within_3px']:.2f} | "
              f"random held-out RMSE {cv_r['overall']['rmse_within_trim_px']:.3f} px")
    scores = {k: sel[k]["spatial_blocks"]["overall"]["rmse_within_trim_px"] for k in KINDS}
    best = min(scores.values())
    chosen = next(k for k in KINDS if scores[k] <= best + SIMPLER_WITHIN_PX)
    print(f"Chosen model: {chosen}")
    final = models.ransac(S, R, chosen, RANSAC_PX)
    model, inl, res = final["model"], final["inliers"], final["residuals"]

    # ---- controls ------------------------------------------------------------
    rng = np.random.default_rng(42)
    genuine_c = consensus(gen, chosen)
    ctrl_refs = {
        "rot90_blockwise": (blockwise_rot(ortho, 1), blockwise_rot(valid.astype(np.uint8), 1).astype(bool)),
        "rot180": (np.rot90(ortho, 2), np.rot90(valid, 2)),
        "rot270_blockwise": (blockwise_rot(ortho, 3), blockwise_rot(valid.astype(np.uint8), 3).astype(bool)),
        "vflip": (ortho[::-1], valid[::-1]),
        "hflip": (ortho[:, ::-1], valid[:, ::-1]),
        "offset_nonoverlap_2000rows": (np.roll(ortho, 2000, axis=0), np.roll(valid, 2000, axis=0)),
        "uniform_noise": (np.where(valid, rng.uniform(1, 255, ortho.shape), 0).astype(np.float32), valid),
    }
    controls = {}
    for name, (ref_c, val_c) in ctrl_refs.items():
        controls[name] = consensus(run_matching(tmc, ref_c, val_c, prior), chosen)
        c = controls[name]
        print(f"  control {name:28s} accepted {c['acceptance_ratio_pct']:5.2f}%  inliers {c['n_inliers']:4d} ({c['inlier_ratio_pct']:5.2f}%)")
    worst = max(controls, key=lambda k: controls[k]["inlier_ratio_pct"])
    delta = genuine_c["inlier_ratio_pct"] - controls[worst]["inlier_ratio_pct"]
    print(f"Genuine inlier ratio {genuine_c['inlier_ratio_pct']:.2f}% vs worst control {worst} "
          f"{controls[worst]['inlier_ratio_pct']:.2f}% -> Delta_shuffle {delta:+.2f} pp "
          f"({'PASS' if delta >= GATE_PP else 'FAIL'} at +{GATE_PP})")

    # ---- products -----------------------------------------------------------
    fig_dir = OUT_DIR / "figures"
    tags = {"source_product": "ch2_tmc_ncn_20230130T1900132182_d_img_d32",
            "reference_product": "NAC_DTM_PITISCUS_M1149280834_2M.TIF",
            "reference_lola_rms_m": LOLA_RMS_M, "model": chosen, "stage": "O",
            "note": "TMC-2 resampled into the NAC orthophoto map frame by the Stage O model. Not ground truth."}
    tif = products.write_registered_geotiff(OUT_DIR / "registered_tmc2_pitiscus.tif", tmc, model,
                                            g["nac_crs_wkt"], pg.X0, pg.Y0, pg.GSD, tags)
    registered = tif.pop("registered")

    # tie points: raw TMC-2 product coordinates at each node, NAC map / native pixel coordinates
    src_i = S.round().astype(int)
    scan = g["tmc_scan_map"][src_i[:, 1], src_i[:, 0]]
    pix = g["tmc_pixel_map"][src_i[:, 1], src_i[:, 0]]
    X, Y = pg.grid_to_map(R[:, 0], R[:, 1])
    lon, lat = pg.map_to_lonlat(X, Y)
    nb, nres = g["nac_bounds"], g["nac_res"]
    nac_col = (X - nb[0]) / nres[0] - 0.5
    nac_row = (nb[3] - Y) / nres[1] - 0.5
    cv_best = sel[chosen]["spatial_blocks"]
    hold = np.array(cv_best["heldout_residuals"])
    csv_path = products.write_csv(
        OUT_DIR / "tiepoints_tmc2_nac_pitiscus.csv",
        ["Stage O tie points: Chandrayaan-2 TMC-2 (source) to LROC NAC DTM orthophoto (reference), Pitiscus.",
         "tmc_scan/tmc_pixel: 0-based line/sample in ch2_tmc_ncn_20230130T1900132182_d_img_d32.img (via ISRO geometry CSV).",
         "nac_col/nac_row: 0-based pixel-centre index in NAC_DTM_PITISCUS_M1149280834_2M.TIF.",
         "map_x/map_y: metres, equirectangular Moon sphere R=1737400, std parallel -51, CM 180 (NAC CRS). lon/lat in degrees.",
         f"Residuals in common-grid px (1 px = {pg.GSD} m). heldout_residual_px: spatial-block 5-fold, model never saw the node.",
         f"Reference error: lola_rms {LOLA_RMS_M} m. These are relative residuals, not ground-truth errors."],
        {"id": np.arange(len(S)), "tmc_scan": scan, "tmc_pixel": pix,
         "grid_src_x": S[:, 0], "grid_src_y": S[:, 1], "grid_ref_x": R[:, 0], "grid_ref_y": R[:, 1],
         "map_x": X, "map_y": Y, "lon": lon, "lat": lat, "nac_col": nac_col, "nac_row": nac_row,
         "ncc_peak": gen["peak"][acc], "ncc_second": gen["second"][acc],
         "fit_residual_px": res, "is_inlier": inl.astype(int),
         "spatial_fold": np.array(cv_best["fold"]), "heldout_residual_px": hold},
        {"tmc_scan": ".3f", "tmc_pixel": ".3f", "grid_src_x": ".0f", "grid_src_y": ".0f",
         "grid_ref_x": ".4f", "grid_ref_y": ".4f", "map_x": ".3f", "map_y": ".3f", "lon": ".8f", "lat": ".8f",
         "nac_col": ".3f", "nac_row": ".3f", "ncc_peak": ".4f", "ncc_second": ".4f",
         "fit_residual_px": ".4f", "heldout_residual_px": ".4f"})
    gj_path = products.write_geojson(
        OUT_DIR / "tiepoints_tmc2_nac_pitiscus_inliers.geojson", lon[inl], lat[inl],
        {"id": np.arange(len(S))[inl], "tmc_scan": np.round(scan[inl], 3), "tmc_pixel": np.round(pix[inl], 3),
         "fit_residual_px": np.round(res[inl], 4), "heldout_residual_px": np.round(hold[inl], 4)},
        "Stage O inlier tie points, lon/lat on the Moon sphere R=1737400 m. Not ground truth; reference lola_rms 0.92 m.")

    # ---- independent closure with the Stage N estimator -----------------------
    closure = []
    for y0 in range(0, ortho.shape[0] - 768, 768):
        cols = np.where(valid[y0:y0 + 768].all(axis=0))[0]
        if len(cols) < 768:
            continue
        x0 = int(cols[0] + (len(cols) - 768) // 2)
        a = registered[y0:y0 + 768, x0:x0 + 768]
        if (a == 0).any():
            continue
        r_ = pcm.estimate_shift(a, ortho[y0:y0 + 768, x0:x0 + 768])
        closure.append({"tile_xy": [x0, y0], "dx": r_["dx"], "dy": r_["dy"], "psr": r_["psr"]})
        print(f"  closure tile ({x0},{y0}): residual shift ({r_['dx']:+.3f}, {r_['dy']:+.3f}) px, PSR {r_['psr']:.1f}")

    # ---- figures ----------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=False)
    fig, ax = plt.subplots(1, 2, figsize=(9, 11), dpi=130)
    for a_, (xy, rr, title) in zip(ax, [(S[inl], res[inl], "In-sample residuals (inliers)"),
                                         (S[np.isfinite(hold) & (hold < 3)], hold[np.isfinite(hold) & (hold < 3)],
                                          "Spatial-block held-out residuals (< 3 px)")]):
        sc = a_.scatter(xy[:, 0], xy[:, 1], c=rr, s=6, cmap="viridis", vmin=0, vmax=1.5)
        a_.invert_yaxis(); a_.set_aspect("equal"); a_.set_title(title, fontsize=9)
        a_.set_xlabel("TMC-2 grid x (px)"); a_.set_ylabel("TMC-2 grid y (px)")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="residual (px, 1 px = 4.72 m)")
    fig.savefig(fig_dir / "stage_o_residual_maps.png", bbox_inches="tight"); plt.close(fig)

    y0, x0, T = 2600, 300, 512
    def norm(a):
        v = a[a > 0]; lo, hi = np.percentile(v, (1, 99)); return np.clip((a - lo) / (hi - lo), 0, 1)
    ref_t, reg_t = norm(ortho[y0:y0 + T, x0:x0 + T]), norm(registered[y0:y0 + T, x0:x0 + T])
    raw_t = norm(tmc[y0:y0 + T, x0:x0 + T])
    cb = ((np.indices((T, T)) // 64).sum(0) % 2).astype(bool)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.4), dpi=130)
    ax[0].imshow(np.where(cb, ref_t, raw_t), cmap="gray"); ax[0].set_title("Before: NAC / TMC-2 checkerboard (no registration)")
    ax[1].imshow(np.where(cb, ref_t, reg_t), cmap="gray"); ax[1].set_title(f"After: NAC / registered TMC-2 ({chosen})")
    ax[2].imshow(np.dstack([ref_t, reg_t, ref_t])); ax[2].set_title("After: magenta = NAC, green = TMC-2")
    for a_ in ax: a_.axis("off")
    fig.suptitle("Pitiscus, 512 x 512 px at 4.72 m. NAC sun az 342.6 deg, TMC-2 sun az 60.9 deg", fontsize=9)
    fig.savefig(fig_dir / "stage_o_checkerboard.png", bbox_inches="tight"); plt.close(fig)

    results = {
        "settings": {"spacing": SPACING, "half_tpl": HALF_TPL, "radius": RADIUS, "peak_min": PEAK_MIN,
                     "uniq_max": UNIQ_MAX, "ransac_px": RANSAC_PX, "kinds": KINDS, "gate_pp": GATE_PP,
                     "grid_gsd_m": pg.GSD, "reference_lola_rms_m": LOLA_RMS_M,
                     "prior_offset_px": prior, "prior_source": "stage_n whole-strip phase correlation"},
        "genuine": {"n_evaluated": int(len(gen["peak"])), "n_accepted": int(acc.sum()),
                    "mean_peak": float(gen["peak"].mean()), "consensus": genuine_c},
        "model_selection": {k: {kk: vv for kk, vv in v.items() if kk not in ("spatial_blocks", "random_kfold")} |
                               {"spatial_blocks": {kk: vv for kk, vv in v["spatial_blocks"].items() if kk not in ("heldout_residuals", "fold")},
                                "random_kfold": {kk: vv for kk, vv in v["random_kfold"].items() if kk not in ("heldout_residuals", "fold")}}
                            for k, v in sel.items()},
        "chosen_model": chosen, "model_params": model.params(),
        "final": {"n_inliers": int(inl.sum()), "insample_rmse_px": float(np.sqrt((res[inl] ** 2).mean())),
                  "insample_rmse_m": float(np.sqrt((res[inl] ** 2).mean()) * pg.GSD)},
        "controls": controls, "delta_shuffle": {"worst_control": worst, "value_pp": float(delta),
                                                "gate_pp": GATE_PP, "pass": bool(delta >= GATE_PP)},
        "uniformity_inliers": uniformity(gen["src_xy"], S[inl]),
        "closure_stage_n_estimator": closure,
        "products": {"registered_geotiff": tif["path"], "inverse_map_final_step_px": tif["inverse_map_final_step_px"],
                     "tiepoints_csv": csv_path, "tiepoints_geojson": gj_path,
                     "figures": [str(fig_dir / "stage_o_residual_maps.png"), str(fig_dir / "stage_o_checkerboard.png")]},
        "runtime_s": time.time() - t0,
    }
    with open(OUT_JSON, "x") as f:
        json.dump(results, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"Saved {OUT_JSON} ({results['runtime_s']:.0f} s)")


if __name__ == "__main__":
    main()
