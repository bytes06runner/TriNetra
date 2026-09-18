#!/usr/bin/env python3
"""
scripts/run_stage_d_controls.py

Executes independent negative controls on headline hops (Hop 1 and Hop 2):
  D1. Shuffle Control (90 deg, 180 deg, 270 deg, vertical flip, horizontal flip)
  D2. Offset Control (non-overlapping spatial regions from same products)
  D3. Noise Control (uniform random noise)
  D6. Diagnostic evaluation of candidate gate criteria (photometric NCC, displacement std, texture floor)
"""

import sys
import time
import copy
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.phase_congruency import compute_phase_congruency
from src.module4_registration.registration import evaluate_flight_gate

# Setup LoFTR import
ELOFTR_DIR = REPO_ROOT / "third_party" / "EfficientLoFTR"
sys.path.insert(0, str(ELOFTR_DIR))
if "src" in sys.modules and not hasattr(sys.modules["src"], "loftr"):
    del sys.modules["src"]
from src.loftr import LoFTR, full_default_cfg

DEVICE = "cpu"
RNG_SEED = 42

def load_model():
    cfg = copy.deepcopy(full_default_cfg)
    cfg['coarse']['npe'] = [256, 256, 256, 256]
    cfg['match_coarse']['train_pad_num_gt_min'] = 20
    cfg['match_coarse']['train_coarse_percent'] = 0.2
    cfg['match_fine']['local_regress_temperature'] = 10.0
    model = LoFTR(config=cfg)
    ckpt_path = REPO_ROOT / "models" / "eloftr_lunar_finetuned_epoch7.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(DEVICE)
    return model

def match_pair(model, img0: np.ndarray, img1: np.ndarray, canvas_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Runs LoFTR forward pass on a pair resized to 256x256, returns keypoints scaled to canvas_size."""
    i0 = cv2.resize(img0, (256, 256), interpolation=cv2.INTER_AREA)
    i1 = cv2.resize(img1, (256, 256), interpolation=cv2.INTER_AREA)
    t0 = torch.from_numpy(i0).float()[None, None].to(DEVICE) / 255.0
    t1 = torch.from_numpy(i1).float()[None, None].to(DEVICE) / 255.0
    with torch.no_grad():
        batch = {"image0": t0, "image1": t1}
        model(batch)
    pts0 = batch["mkpts0_f"].cpu().numpy() * (canvas_size / 256.0)
    pts1 = batch["mkpts1_f"].cpu().numpy() * (canvas_size / 256.0)
    return pts0, pts1

def evaluate_match_result(pts0: np.ndarray, pts1: np.ndarray, expected_scale: float) -> Dict[str, Any]:
    total_matches = len(pts0)
    if total_matches < 4:
        return {
            "total_matches": total_matches,
            "inliers": 0,
            "ratio_pct": 0.0,
            "scale": None,
            "rotation_deg": None,
            "cond": None,
            "gate_status": "GATED",
            "first_failing": "insufficient_matches",
            "dx_mean": None,
            "dx_std": None,
            "dy_mean": None,
            "dy_std": None,
            "H": None,
            "inlier_mask": np.zeros(total_matches, dtype=bool),
        }

    cv2.setRNGSeed(RNG_SEED)
    np.random.seed(RNG_SEED)
    M, mask = cv2.estimateAffinePartial2D(pts0, pts1, method=cv2.RANSAC, ransacReprojThreshold=15.0)
    inliers = int(np.sum(mask)) if mask is not None else 0
    ratio = (inliers / total_matches * 100.0) if total_matches > 0 else 0.0
    inlier_mask = mask.ravel().astype(bool) if mask is not None else np.zeros(total_matches, dtype=bool)

    H = np.eye(3, dtype=np.float64)
    if M is not None:
        H[:2, :] = M
        scale = float(np.sqrt(M[0, 0]**2 + M[0, 1]**2))
        rot_deg = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        cond_num = float(np.linalg.cond(H))
    else:
        scale, rot_deg, cond_num = None, None, None

    gate_eval = evaluate_flight_gate(
        inliers=inliers,
        total_matches=total_matches,
        inlier_ratio_pct=ratio,
        H=H,
        expected_scale=expected_scale,
        max_rotation_deg=30.0,
        scale_tolerance=0.25,
        cond_thresh=1e5,
    )

    # Compute displacement statistics on inliers
    if inliers > 0:
        p0_in = pts0[inlier_mask]
        p1_in = pts1[inlier_mask]
        dx = p1_in[:, 0] - p0_in[:, 0]
        dy = p1_in[:, 1] - p0_in[:, 1]
        dx_mean, dx_std = float(np.mean(dx)), float(np.std(dx))
        dy_mean, dy_std = float(np.mean(dy)), float(np.std(dy))
    else:
        dx_mean, dx_std, dy_mean, dy_std = None, None, None, None

    return {
        "total_matches": total_matches,
        "inliers": inliers,
        "ratio_pct": ratio,
        "scale": scale,
        "rotation_deg": rot_deg,
        "cond": cond_num,
        "gate_status": gate_eval["status"],
        "first_failing": gate_eval["first_failing_criterion"],
        "dx_mean": dx_mean,
        "dx_std": dx_std,
        "dy_mean": dy_mean,
        "dy_std": dy_std,
        "H": H,
        "inlier_mask": inlier_mask,
    }

def compute_patch_ncc(img0: np.ndarray, img1: np.ndarray, pts0: np.ndarray, pts1: np.ndarray, inlier_mask: np.ndarray, patch_size: int = 31) -> Tuple[float, float]:
    """Computes Normalized Cross Correlation (NCC) of intensity patches around inlier keypoints."""
    if np.sum(inlier_mask) == 0:
        return 0.0, 0.0
    p0_in = pts0[inlier_mask]
    p1_in = pts1[inlier_mask]
    half = patch_size // 2
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    ncc_vals = []
    for (x0, y0), (x1, y1) in zip(p0_in, p1_in):
        ix0, iy0 = int(round(x0)), int(round(y0))
        ix1, iy1 = int(round(x1)), int(round(y1))
        if (ix0 - half < 0 or ix0 + half >= w0 or iy0 - half < 0 or iy0 + half >= h0 or
            ix1 - half < 0 or ix1 + half >= w1 or iy1 - half < 0 or iy1 + half >= h1):
            continue
        patch0 = img0[iy0 - half : iy0 + half + 1, ix0 - half : ix0 + half + 1].astype(np.float32)
        patch1 = img1[iy1 - half : iy1 + half + 1, ix1 - half : ix1 + half + 1].astype(np.float32)
        p0_z = patch0 - np.mean(patch0)
        p1_z = patch1 - np.mean(patch1)
        denom = (np.linalg.norm(p0_z) * np.linalg.norm(p1_z))
        if denom > 1e-6:
            ncc = float(np.sum(p0_z * p1_z) / denom)
            ncc_vals.append(ncc)
    if len(ncc_vals) == 0:
        return 0.0, 0.0
    return float(np.mean(ncc_vals)), float(np.std(ncc_vals))

def main():
    print("=" * 80)
    print("STAGE D: RIGOROUS CONTROL EXPERIMENTS ON HEADLINE HOPS (HOP 1 & HOP 2)")
    print("=" * 80)

    model = load_model()

    # 1. LOAD GENUINE HEADLINE DATA
    h1_data = np.load(REPO_ROOT / "assets" / "real_cache" / "polar_flight_hop1.npz", allow_pickle=True)
    disp_ohrc = h1_data["disp_ohrc"]  # 1000x1000
    disp_tmc1 = h1_data["disp_tmc"]   # 1000x1000

    h2_data = np.load(REPO_ROOT / "assets" / "real_cache" / "real_flight_hop2_phase_congruency.npz", allow_pickle=True)
    disp_tmc2 = h2_data["disp_tmc"]   # 800x800
    disp_iirs = h2_data["disp_iirs"]  # 800x800
    pc_tmc2 = h2_data["pc_tmc"]       # 800x800
    pc_iirs_gen = h2_data["pc_iirs"]  # 800x800

    # 2. RUN GENUINE BASELINES
    print("\n[GENUINE BASELINES]")
    pts0_h1, pts1_h1 = match_pair(model, disp_ohrc, disp_tmc1, 1000)
    res_h1_gen = evaluate_match_result(pts0_h1, pts1_h1, expected_scale=0.94118)
    ncc_h1_mean, ncc_h1_std = compute_patch_ncc(disp_ohrc, disp_tmc1, pts0_h1, pts1_h1, res_h1_gen["inlier_mask"])
    print(f"Hop 1 Genuine: {res_h1_gen['inliers']}/{res_h1_gen['total_matches']} ({res_h1_gen['ratio_pct']:.2f}%) "
          f"scale={res_h1_gen['scale']:.4f} rot={res_h1_gen['rotation_deg']:+.2f}° cond={res_h1_gen['cond']:.1f} "
          f"Gate={res_h1_gen['gate_status']} NCC={ncc_h1_mean:.3f}±{ncc_h1_std:.3f}")

    pts0_h2, pts1_h2 = match_pair(model, pc_tmc2, pc_iirs_gen, 800)
    res_h2_gen = evaluate_match_result(pts0_h2, pts1_h2, expected_scale=0.9997)
    ncc_h2_mean, ncc_h2_std = compute_patch_ncc(disp_tmc2, disp_iirs, pts0_h2, pts1_h2, res_h2_gen["inlier_mask"])
    print(f"Hop 2 Genuine: {res_h2_gen['inliers']}/{res_h2_gen['total_matches']} ({res_h2_gen['ratio_pct']:.2f}%) "
          f"scale={res_h2_gen['scale']:.4f} rot={res_h2_gen['rotation_deg']:+.2f}° cond={res_h2_gen['cond']:.1f} "
          f"Gate={res_h2_gen['gate_status']} NCC={ncc_h2_mean:.3f}±{ncc_h2_std:.3f}")

    # D1. SHUFFLE CONTROLS
    print("\n" + "=" * 50)
    print("D1: SHUFFLE CONTROLS")
    print("=" * 50)

    h1_shuffle_results = {}
    transforms_h1 = {
        "rot90": np.rot90(disp_tmc1, 1),
        "rot180": np.rot90(disp_tmc1, 2),
        "rot270": np.rot90(disp_tmc1, 3),
        "vflip": np.flipud(disp_tmc1),
        "hflip": np.fliplr(disp_tmc1),
    }

    for name, ref_t in transforms_h1.items():
        p0, p1 = match_pair(model, disp_ohrc, ref_t, 1000)
        res = evaluate_match_result(p0, p1, expected_scale=0.94118)
        ncc_m, ncc_s = compute_patch_ncc(disp_ohrc, ref_t, p0, p1, res["inlier_mask"])
        h1_shuffle_results[name] = {**res, "ncc_mean": ncc_m, "ncc_std": ncc_s}
        print(f"Hop 1 Shuffle [{name}]: {res['inliers']}/{res['total_matches']} ({res['ratio_pct']:.2f}%) "
              f"scale={res['scale']} rot={res['rotation_deg']} cond={res['cond']} Gate={res['gate_status']} "
              f"failing={res['first_failing']} dx_std={res['dx_std']} dy_std={res['dy_std']} NCC={ncc_m:.3f}")

    h2_shuffle_results = {}
    transforms_h2 = {
        "rot90": np.rot90(disp_iirs, 1),
        "rot180": np.rot90(disp_iirs, 2),
        "rot270": np.rot90(disp_iirs, 3),
        "vflip": np.flipud(disp_iirs),
        "hflip": np.fliplr(disp_iirs),
    }

    for name, ref_t in transforms_h2.items():
        pc_ref, _ = compute_phase_congruency(ref_t, nscale=4, norient=6)
        p0, p1 = match_pair(model, pc_tmc2, pc_ref, 800)
        res = evaluate_match_result(p0, p1, expected_scale=0.9997)
        ncc_m, ncc_s = compute_patch_ncc(disp_tmc2, ref_t, p0, p1, res["inlier_mask"])
        h2_shuffle_results[name] = {**res, "ncc_mean": ncc_m, "ncc_std": ncc_s}
        print(f"Hop 2 Shuffle [{name}]: {res['inliers']}/{res['total_matches']} ({res['ratio_pct']:.2f}%) "
              f"scale={res['scale']} rot={res['rotation_deg']} cond={res['cond']} Gate={res['gate_status']} "
              f"failing={res['first_failing']} dx_std={res['dx_std']} dy_std={res['dy_std']} NCC={ncc_m:.3f}")

    # D2. OFFSET CONTROLS
    print("\n" + "=" * 50)
    print("D2: OFFSET CONTROLS")
    print("=" * 50)

    tmc1_img_p = Path("/Users/srijeetprasadbanerjee/Desktop/data/data/calibrated/ch2_tmc_ncn_20231205T1906512971_d_img_d18/data/calibrated/20231205/ch2_tmc_ncn_20231205T1906512971_d_img_d18.img")
    mm_tmc1 = np.memmap(str(tmc1_img_p), dtype="<u2", mode="r", shape=(210154, 4000))

    c1_raw = mm_tmc1[20000:20240, 500:740].copy()
    p1_1, p99_1 = np.percentile(c1_raw, (1.0, 99.0))
    c1_u8 = np.clip((c1_raw.astype(np.float32) - p1_1) / max(1e-6, p99_1 - p1_1) * 255.0, 0, 255).astype(np.uint8)
    disp_tmc1_off1 = cv2.resize(c1_u8, (1000, 1000), interpolation=cv2.INTER_CUBIC)

    c2_raw = mm_tmc1[120000:120240, 500:740].copy()
    p1_2, p99_2 = np.percentile(c2_raw, (1.0, 99.0))
    c2_u8 = np.clip((c2_raw.astype(np.float32) - p1_2) / max(1e-6, p99_2 - p1_2) * 255.0, 0, 255).astype(np.uint8)
    disp_tmc1_off2 = cv2.resize(c2_u8, (1000, 1000), interpolation=cv2.INTER_CUBIC)

    h1_offset_results = {}
    for name, off_img in [("offset_loc1_2000km", disp_tmc1_off1), ("offset_loc2_5000km", disp_tmc1_off2)]:
        p0, p1 = match_pair(model, disp_ohrc, off_img, 1000)
        res = evaluate_match_result(p0, p1, expected_scale=0.94118)
        ncc_m, ncc_s = compute_patch_ncc(disp_ohrc, off_img, p0, p1, res["inlier_mask"])
        h1_offset_results[name] = {**res, "ncc_mean": ncc_m, "ncc_std": ncc_s}
        print(f"Hop 1 Offset [{name}]: {res['inliers']}/{res['total_matches']} ({res['ratio_pct']:.2f}%) "
              f"scale={res['scale']} rot={res['rotation_deg']} cond={res['cond']} Gate={res['gate_status']} "
              f"failing={res['first_failing']} dx_std={res['dx_std']} dy_std={res['dy_std']} NCC={ncc_m:.3f}")

    iirs_qub_p = REPO_ROOT / "data/ch2_iir_nri_20231003T2152304115_d_img_d18/data/raw/20231003/ch2_iir_nri_20231003T2152304115_d_img_d18.qub"
    m_iir = np.memmap(str(iirs_qub_p), dtype="<u2", mode="r", shape=(256, 2264, 250))

    def prep_iirs_crop(lines_start, lines_end, samp_start, samp_end):
        sub = m_iir[30:60, lines_start:lines_end, samp_start:samp_end]
        raw = np.mean(sub, axis=0).astype(np.float32)
        col_med = np.median(raw, axis=0, keepdims=True)
        common_med = np.median(col_med)
        destriped = raw - 0.85 * (col_med - common_med)
        p2, p98 = np.percentile(destriped, (2.0, 98.0))
        u8 = np.clip((destriped - p2) / max(1e-6, p98 - p2) * 255.0, 0, 255).astype(np.uint8)
        disp = cv2.resize(u8, (800, 800), interpolation=cv2.INTER_CUBIC)
        pc, _ = compute_phase_congruency(disp, nscale=4, norient=6)
        return disp, pc

    disp_iirs_off1, pc_iirs_off1 = prep_iirs_crop(100, 220, 60, 180)
    disp_iirs_off2, pc_iirs_off2 = prep_iirs_crop(1200, 1320, 60, 180)

    h2_offset_results = {}
    for name, (disp_off, pc_off) in [("offset_loc1_27km", (disp_iirs_off1, pc_iirs_off1)),
                                     ("offset_loc2_48km", (disp_iirs_off2, pc_iirs_off2))]:
        p0, p1 = match_pair(model, pc_tmc2, pc_off, 800)
        res = evaluate_match_result(p0, p1, expected_scale=0.9997)
        ncc_m, ncc_s = compute_patch_ncc(disp_tmc2, disp_off, p0, p1, res["inlier_mask"])
        h2_offset_results[name] = {**res, "ncc_mean": ncc_m, "ncc_std": ncc_s}
        print(f"Hop 2 Offset [{name}]: {res['inliers']}/{res['total_matches']} ({res['ratio_pct']:.2f}%) "
              f"scale={res['scale']} rot={res['rotation_deg']} cond={res['cond']} Gate={res['gate_status']} "
              f"failing={res['first_failing']} dx_std={res['dx_std']} dy_std={res['dy_std']} NCC={ncc_m:.3f}")

    # D3. NOISE CONTROLS
    print("\n" + "=" * 50)
    print("D3: NOISE CONTROLS")
    print("=" * 50)

    np.random.seed(RNG_SEED)
    noise_1000 = np.random.randint(0, 256, (1000, 1000), dtype=np.uint8)
    p0_n1, p1_n1 = match_pair(model, disp_ohrc, noise_1000, 1000)
    res_h1_noise = evaluate_match_result(p0_n1, p1_n1, expected_scale=0.94118)
    ncc_n1_m, ncc_n1_s = compute_patch_ncc(disp_ohrc, noise_1000, p0_n1, p1_n1, res_h1_noise["inlier_mask"])
    print(f"Hop 1 Noise Control: {res_h1_noise['inliers']}/{res_h1_noise['total_matches']} ({res_h1_noise['ratio_pct']:.2f}%) "
          f"scale={res_h1_noise['scale']} rot={res_h1_noise['rotation_deg']} cond={res_h1_noise['cond']} "
          f"Gate={res_h1_noise['gate_status']} failing={res_h1_noise['first_failing']} NCC={ncc_n1_m:.3f}")

    noise_800 = np.random.randint(0, 256, (800, 800), dtype=np.uint8)
    pc_noise, _ = compute_phase_congruency(noise_800, nscale=4, norient=6)
    p0_n2, p1_n2 = match_pair(model, pc_tmc2, pc_noise, 800)
    res_h2_noise = evaluate_match_result(p0_n2, p1_n2, expected_scale=0.9997)
    ncc_n2_m, ncc_n2_s = compute_patch_ncc(disp_tmc2, noise_800, p0_n2, p1_n2, res_h2_noise["inlier_mask"])
    print(f"Hop 2 Noise Control: {res_h2_noise['inliers']}/{res_h2_noise['total_matches']} ({res_h2_noise['ratio_pct']:.2f}%) "
          f"scale={res_h2_noise['scale']} rot={res_h2_noise['rotation_deg']} cond={res_h2_noise['cond']} "
          f"Gate={res_h2_noise['gate_status']} failing={res_h2_noise['first_failing']} NCC={ncc_n2_m:.3f}")

    # Pure noise vs pure noise
    noise_ohrc = np.random.randint(0, 256, (1000, 1000), dtype=np.uint8)
    p0_nn, p1_nn = match_pair(model, noise_ohrc, noise_1000, 1000)
    res_nn = evaluate_match_result(p0_nn, p1_nn, expected_scale=1.0)
    print(f"Pure Noise vs Pure Noise (1000x1000): {res_nn['inliers']}/{res_nn['total_matches']} ({res_nn['ratio_pct']:.2f}%) "
          f"Gate={res_nn['gate_status']}")

    results_dict = {
        "hop1_genuine": {k: v for k, v in res_h1_gen.items() if k not in ("H", "inlier_mask")},
        "hop1_shuffle": {k: {sk: sv for sk, sv in v.items() if sk not in ("H", "inlier_mask")} for k, v in h1_shuffle_results.items()},
        "hop1_offset": {k: {sk: sv for sk, sv in v.items() if sk not in ("H", "inlier_mask")} for k, v in h1_offset_results.items()},
        "hop1_noise": {k: v for k, v in res_h1_noise.items() if k not in ("H", "inlier_mask")},
        "hop2_genuine": {k: v for k, v in res_h2_gen.items() if k not in ("H", "inlier_mask")},
        "hop2_shuffle": {k: {sk: sv for sk, sv in v.items() if sk not in ("H", "inlier_mask")} for k, v in h2_shuffle_results.items()},
        "hop2_offset": {k: {sk: sv for sk, sv in v.items() if sk not in ("H", "inlier_mask")} for k, v in h2_offset_results.items()},
        "hop2_noise": {k: v for k, v in res_h2_noise.items() if k not in ("H", "inlier_mask")},
    }
    out_json = REPO_ROOT / "outputs" / "stage_d_controls.json"
    out_json.parent.mkdir(exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nSaved Stage D results to {out_json}")

if __name__ == "__main__":
    main()
