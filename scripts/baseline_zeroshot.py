#!/usr/bin/env python3
"""baseline_zeroshot.py — Zero-shot pretrained matcher baselines on real Chandrayaan-2 flight crops.

Runs four real pretrained matchers on both Hop 1 (OHRC↔TMC-2) and Hop 2 (TMC-2↔IIRS) flight crops:
  1. SIFT + MAGSAC++      (OpenCV built-in, classical baseline)
  2. LightGlue+SuperPoint  (cvg/LightGlue, learned sparse keypoint matcher)
  3. EfficientLoFTR        (zju3dv/EfficientLoFTR, learned dense linear attention matcher)
  4. MatchAnything_ELoFTR  (zju3dv/MatchAnything, universal cross-modal learned matcher)

Outputs:
  - Prints a formatted baseline scorecard table to stdout
  - Saves all metrics to assets/baselines/zeroshot_results.json

Usage:
    python scripts/baseline_zeroshot.py
"""

import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = PROJECT_ROOT / "assets" / "real_cache"
OUTPUT_DIR = PROJECT_ROOT / "assets" / "baselines"
OUTPUT_JSON = OUTPUT_DIR / "zeroshot_results.json"
ELOFTR_DIR = PROJECT_ROOT / "third_party" / "EfficientLoFTR"

# GSD values for ground error computation
HOP1_TARGET_GSD = 4.72   # TMC-2 GSD in m/px
HOP2_TARGET_GSD = 68.38  # IIRS GSD in m/px

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

RNG_SEED = 42


def set_seed(seed=RNG_SEED):
    """Set deterministic seeds for OpenCV, NumPy, and PyTorch."""
    cv2.setRNGSeed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(RNG_SEED)


def to_gray_uint8(img):
    """Convert any image to grayscale uint8."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[2] == 3 else img[:, :, 0]
    if img.dtype != np.uint8:
        max_v = float(img.max()) if img.max() > 0 else 1.0
        if max_v > 1.0:
            img = np.clip(img / max_v * 255.0, 0, 255).astype(np.uint8)
        else:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def to_tensor_gray(img, device="cpu"):
    """Convert image to torch tensor [1, 1, H, W] float32 normalized [0, 1]."""
    gray = to_gray_uint8(img).astype(np.float32) / 255.0
    # Ensure dimensions are divisible by 32 for LoFTR backbone (8x) and aggregation (4x)
    h, w = gray.shape
    h_pad = (32 - h % 32) % 32
    w_pad = (32 - w % 32) % 32
    if h_pad > 0 or w_pad > 0:
        gray = np.pad(gray, ((0, h_pad), (0, w_pad)), mode="reflect")
    t = torch.from_numpy(gray)[None, None].to(device)
    return t


from src.module4_registration.registration import (
    evaluate_flight_gate,
    check_degeneracy,
)


def compute_reproj_rmse(pts1, pts2, H, mask):
    """Compute reprojection RMSE for inliers."""
    if H is None or len(mask) == 0 or mask.sum() == 0:
        return float('inf')
    inlier_pts1 = pts1[mask]
    inlier_pts2 = pts2[mask]
    ones = np.ones((len(inlier_pts1), 1), dtype=np.float32)
    pts1_h = np.hstack([inlier_pts1, ones])
    projected = (H @ pts1_h.T).T
    denom = projected[:, 2:3]
    denom[np.abs(denom) < 1e-8] = 1e-8
    projected = projected[:, :2] / denom
    errors = np.sqrt(np.sum((projected - inlier_pts2) ** 2, axis=1))
    return float(np.sqrt(np.mean(errors ** 2)))


# ─────────────────────────────────────────────────────────────────────────────
# Matcher 1: SIFT Canonical (4-DoF Similarity) + MAGSAC++ Option
# ─────────────────────────────────────────────────────────────────────────────
def run_sift_canonical(img1, img2, use_homography=False):
    """SIFT matching on standardized display crops with CLAHE and spatial deduplication.

    By default, estimates a 4-DoF Similarity transform (Scale, Rotation, Translation)
    via cv2.estimateAffinePartial2D, which accurately reflects the constrained geometry
    of orbital nadir pushbroom cameras and prevents 8-DoF homography overfitting.
    """
    gray1 = to_gray_uint8(img1)
    gray2 = to_gray_uint8(img2)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    g1_clahe = clahe.apply(gray1)
    g2_clahe = clahe.apply(gray2)

    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.015, edgeThreshold=10)
    kp1, des1 = sift.detectAndCompute(g1_clahe, None)
    kp2, des2 = sift.detectAndCompute(g2_clahe, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return np.empty((0, 2)), np.empty((0, 2)), np.array([], dtype=bool), None

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    # Spatial deduplication: enforce distinct physical crater features (>20 px apart)
    unique_pts1, unique_pts2 = [], []
    for m in sorted(matches, key=lambda x: x.distance):
        p1 = kp1[m.queryIdx].pt
        p2 = kp2[m.trainIdx].pt
        if all(np.linalg.norm(np.array(p1) - np.array(u1)) > 20 for u1 in unique_pts1) and \
           all(np.linalg.norm(np.array(p2) - np.array(u2)) > 20 for u2 in unique_pts2):
            unique_pts1.append(p1)
            unique_pts2.append(p2)

    pts1 = np.float32(unique_pts1)
    pts2 = np.float32(unique_pts2)
    total_candidates = len(pts1)

    if total_candidates < 4:
        return pts1, pts2, np.zeros(total_candidates, dtype=bool), None

    set_seed(RNG_SEED)
    if use_homography:
        H, mask = cv2.findHomography(pts1, pts2, cv2.USAC_MAGSAC, ransacReprojThreshold=15.0)
        inlier_mask = mask.ravel().astype(bool) if mask is not None else np.zeros(total_candidates, dtype=bool)
        return pts1, pts2, inlier_mask, H
    else:
        M, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=15.0)
        inlier_mask = mask.ravel().astype(bool) if mask is not None else np.zeros(total_candidates, dtype=bool)
        H = np.eye(3, dtype=np.float64)
        if M is not None:
            H[:2, :] = M
        return pts1, pts2, inlier_mask, H



# ─────────────────────────────────────────────────────────────────────────────
# Matcher 2: LightGlue + SuperPoint
# ─────────────────────────────────────────────────────────────────────────────
def run_lightglue_superpoint(img1, img2):
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd

    device = DEVICE
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    img1_t = to_tensor_gray(img1, device)
    img2_t = to_tensor_gray(img2, device)

    with torch.no_grad():
        feats0 = extractor.extract(img1_t)
        feats1 = extractor.extract(img2_t)
        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    m = matches01["matches"]
    if len(m) == 0:
        return np.empty((0, 2)), np.empty((0, 2)), np.array([], dtype=bool), None

    pts1 = feats0["keypoints"][m[:, 0]].cpu().numpy()
    pts2 = feats1["keypoints"][m[:, 1]].cpu().numpy()

    if len(pts1) < 4:
        return pts1, pts2, np.ones(len(pts1), dtype=bool), None

    set_seed(RNG_SEED)
    M, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=15.0)
    inlier_mask = mask.ravel().astype(bool) if mask is not None else np.zeros(len(pts1), dtype=bool)
    H = np.eye(3, dtype=np.float64)
    if M is not None:
        H[:2, :] = M

    return pts1, pts2, inlier_mask, H


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Import LoFTR without colliding with project's own src/ package
# ─────────────────────────────────────────────────────────────────────────────
def _get_loftr_class_and_config():
    if str(ELOFTR_DIR) not in sys.path:
        sys.path.insert(0, str(ELOFTR_DIR))
    if "src" in sys.modules and not hasattr(sys.modules["src"], "loftr"):
        del sys.modules["src"]
    proj_root = str(PROJECT_ROOT)
    if proj_root in sys.path:
        sys.path.remove(proj_root)

    from src.loftr import LoFTR, full_default_cfg
    if proj_root not in sys.path:
        sys.path.append(proj_root)
    return LoFTR, full_default_cfg


# ─────────────────────────────────────────────────────────────────────────────
# Matcher 3: EfficientLoFTR
# ─────────────────────────────────────────────────────────────────────────────
_ELO_MODEL = None


def get_efficient_loftr_model():
    global _ELO_MODEL
    if _ELO_MODEL is not None:
        return _ELO_MODEL

    from huggingface_hub import hf_hub_download
    LoFTR, full_default_cfg = _get_loftr_class_and_config()

    print("      Downloading/loading eloftr_outdoor.ckpt weights...")
    p = hf_hub_download("Realcat/imcui_checkpoints", "eloftr/eloftr_outdoor.ckpt")
    ckpt = torch.load(p, map_location="cpu")
    sd = {k.replace("matcher.", ""): v for k, v in ckpt["state_dict"].items()}

    model = LoFTR(config=full_default_cfg)
    model.load_state_dict(sd, strict=True)
    model = model.eval().to(DEVICE)
    _ELO_MODEL = model
    return _ELO_MODEL


def run_efficient_loftr(img1, img2):
    model = get_efficient_loftr_model()
    device = DEVICE

    img1_t = to_tensor_gray(img1, device)
    img2_t = to_tensor_gray(img2, device)

    with torch.no_grad():
        batch = {"image0": img1_t, "image1": img2_t}
        model(batch)

    pts1 = batch["mkpts0_f"].cpu().numpy()
    pts2 = batch["mkpts1_f"].cpu().numpy()

    # Filter out any keypoints falling into the padding region
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if len(pts1) > 0:
        valid = (pts1[:, 0] < w1) & (pts1[:, 1] < h1) & (pts2[:, 0] < w2) & (pts2[:, 1] < h2)
        pts1 = pts1[valid]
        pts2 = pts2[valid]

    if len(pts1) < 4:
        return pts1, pts2, np.ones(len(pts1), dtype=bool), None

    set_seed(RNG_SEED)
    M, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=15.0)
    inlier_mask = mask.ravel().astype(bool) if mask is not None else np.zeros(len(pts1), dtype=bool)
    H = np.eye(3, dtype=np.float64)
    if M is not None:
        H[:2, :] = M

    return pts1, pts2, inlier_mask, H


# ─────────────────────────────────────────────────────────────────────────────
# Matcher 4: MatchAnything (Cross-Modal Pretrained ELoFTR)
# ─────────────────────────────────────────────────────────────────────────────
_MA_MODEL = None


def get_matchanything_model():
    global _MA_MODEL
    if _MA_MODEL is not None:
        return _MA_MODEL

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    LoFTR, full_default_cfg = _get_loftr_class_and_config()

    print("      Downloading/loading MatchAnything safetensors...")
    # 1. Fetch the HF key mapping
    import requests
    url = "https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/efficientloftr/convert_efficientloftr_to_hf.py"
    r = requests.get(url, timeout=10)
    text = r.text
    start = text.find("ORIGINAL_TO_CONVERTED_KEY_MAPPING = {")
    end = text.find("def convert_old_keys_to_new_keys", start)
    namespace = {"lambda": lambda: None}
    exec(text[start:end], namespace)
    MAPPING = namespace["ORIGINAL_TO_CONVERTED_KEY_MAPPING"]

    # 2. Reference keys from eloftr_outdoor.ckpt
    p_el = hf_hub_download("Realcat/imcui_checkpoints", "eloftr/eloftr_outdoor.ckpt")
    ckpt_el = torch.load(p_el, map_location="cpu")["state_dict"]
    orig_keys = list(ckpt_el.keys())

    old_to_new = {}
    for k in orig_keys:
        new_k = k
        for pattern, replacement in MAPPING.items():
            if replacement is None:
                continue
            if callable(replacement):
                new_k = re.sub(pattern, replacement, new_k)
            else:
                new_k = re.sub(pattern, replacement, new_k)
        old_to_new[k] = new_k
    new_to_old = {v: k for k, v in old_to_new.items()}

    # 3. Load MatchAnything safetensors and invert keys
    p_ma = hf_hub_download("zju-community/matchanything_eloftr", "model.safetensors")
    sd_ma = load_file(p_ma)
    sd_loftr = {}
    for k, v in sd_ma.items():
        if k in new_to_old:
            old_k = new_to_old[k].replace("matcher.", "")
            sd_loftr[old_k] = v

    model = LoFTR(config=full_default_cfg)
    model.load_state_dict(sd_loftr, strict=True)
    model = model.eval().to(DEVICE)
    _MA_MODEL = model
    return _MA_MODEL


def run_matchanything(img1, img2):
    """MatchAnything (cross-modal pretrained ELoFTR).

    Resizes inputs to native 832x832 per preprocessor_config.json to ensure
    rotary position embeddings (RoPE) operate in-distribution, then scales
    predicted correspondences back to original display dimensions.
    """
    model = get_matchanything_model()
    device = DEVICE

    gray1 = to_gray_uint8(img1)
    gray2 = to_gray_uint8(img2)
    h1, w1 = gray1.shape[:2]
    h2, w2 = gray2.shape[:2]

    # Resize to 832x832 native preprocessor size
    r1 = cv2.resize(gray1, (832, 832), interpolation=cv2.INTER_AREA)
    r2 = cv2.resize(gray2, (832, 832), interpolation=cv2.INTER_AREA)

    t1 = torch.from_numpy(r1.astype(np.float32) / 255.0)[None, None].to(device)
    t2 = torch.from_numpy(r2.astype(np.float32) / 255.0)[None, None].to(device)

    with torch.no_grad():
        batch = {"image0": t1, "image1": t2}
        model(batch)

    pts1_832 = batch["mkpts0_f"].cpu().numpy()
    pts2_832 = batch["mkpts1_f"].cpu().numpy()

    # Scale keypoints back to original display coordinate frame
    sx1, sy1 = (w1 / 832.0), (h1 / 832.0)
    sx2, sy2 = (w2 / 832.0), (h2 / 832.0)

    if len(pts1_832) > 0:
        pts1 = pts1_832.copy()
        pts1[:, 0] *= sx1
        pts1[:, 1] *= sy1
        pts2 = pts2_832.copy()
        pts2[:, 0] *= sx2
        pts2[:, 1] *= sy2
    else:
        pts1 = np.empty((0, 2), dtype=np.float32)
        pts2 = np.empty((0, 2), dtype=np.float32)

    if len(pts1) < 4:
        return pts1, pts2, np.ones(len(pts1), dtype=bool), None

    set_seed(RNG_SEED)
    M, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=15.0)
    inlier_mask = mask.ravel().astype(bool) if mask is not None else np.zeros(len(pts1), dtype=bool)
    H = np.eye(3, dtype=np.float64)
    if M is not None:
        H[:2, :] = M

    return pts1, pts2, inlier_mask, H


# ─────────────────────────────────────────────────────────────────────────────
# Execution & Table Formatting
# ─────────────────────────────────────────────────────────────────────────────
def run_eval(img1, img2, hop_name, target_gsd):
    matchers = [
        ("SIFT Canonical (4-DoF)", run_sift_canonical, "BSD-3-Clause", "CPU"),
        ("LightGlue + SuperPoint (4-DoF)", run_lightglue_superpoint, "Apache-2.0 / Non-Commercial", DEVICE.upper()),
        ("EfficientLoFTR (4-DoF)", run_efficient_loftr, "Apache-2.0", DEVICE.upper()),
        ("MatchAnything (4-DoF)", run_matchanything, "Apache-2.0", DEVICE.upper()),
    ]

    results = []
    for name, fn, license_str, dev in matchers:
        print(f"   ▶️  Evaluating {name} on {hop_name}...")
        t0 = time.time()
        try:
            pts1, pts2, inlier_mask, H = fn(img1, img2)
            elapsed = time.time() - t0
            raw_matches = len(pts1)
            inliers = int(inlier_mask.sum()) if len(inlier_mask) > 0 else 0
            ratio = (inliers / raw_matches * 100.0) if raw_matches > 0 else 0.0
            rmse_px = compute_reproj_rmse(pts1, pts2, H, inlier_mask) if inliers >= 4 else float("inf")
            rmse_m = rmse_px * target_gsd if rmse_px != float("inf") else float("inf")

            gate_eval = evaluate_flight_gate(inliers, raw_matches, ratio)
            transform_dof = 4
            degen_eval = check_degeneracy(inliers, raw_matches, rmse_px, transform_dof=transform_dof)

            if degen_eval["is_degenerate"]:
                status = "DEGENERATE"
            elif gate_eval["is_gated"]:
                status = "GATED"
            else:
                status = "PASSED"
            err_msg = None
        except Exception as e:
            elapsed = time.time() - t0
            raw_matches, inliers, ratio = 0, 0, 0.0
            rmse_px, rmse_m = float("inf"), float("inf")
            status = "CRASH"
            err_msg = str(e)
            gate_eval = {"is_gated": True}
            degen_eval = {"is_degenerate": False, "suppress_metrics": False}
            print(f"      ❌ Crashed: {e}")

        r = {
            "matcher": name,
            "hop": hop_name,
            "raw_matches": raw_matches,
            "inliers": inliers,
            "inlier_ratio_pct": round(ratio, 2) if not degen_eval.get("suppress_metrics") else None,
            "rmse_px": round(rmse_px, 2) if (rmse_px != float("inf") and not degen_eval.get("suppress_metrics")) else None,
            "rmse_m": round(rmse_m, 1) if (rmse_m != float("inf") and not degen_eval.get("suppress_metrics")) else None,
            "unfiltered_ratio_pct": round(ratio, 2),
            "unfiltered_rmse_px": round(rmse_px, 2) if rmse_px != float("inf") else None,
            "unfiltered_rmse_m": round(rmse_m, 1) if rmse_m != float("inf") else None,
            "runtime_s": round(elapsed, 2),
            "device": dev,
            "license": license_str,
            "status": status,
            "is_gated": gate_eval.get("is_gated", False),
            "is_degenerate": degen_eval.get("is_degenerate", False),
            "degeneracy_reasons": degen_eval.get("reasons", []),
            "gate_reason": gate_eval.get("reason", ""),
            "cv2_rng_seed": RNG_SEED,
        }
        if err_msg:
            r["error"] = err_msg
        results.append(r)
        ratio_disp = f"{ratio:.1f}%" if not degen_eval.get("suppress_metrics") else "—"
        rmse_disp = f"{r['rmse_px']:.2f} px ({r['rmse_m']:.1f} m)" if r['rmse_px'] is not None else "—"
        print(f"      Inliers: {inliers}/{raw_matches} ({ratio_disp}) | RMSE: {rmse_disp} | Time: {elapsed:.2f}s | Status: {status}")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Zero-shot baseline evaluation")
    parser.add_argument("--site", choices=["shiv_shakti", "polar", "both"], default="both", help="Site to evaluate")
    args = parser.parse_args()

    print("=" * 95)
    print("🔬 TriNetra Zero-Shot Baseline Evaluation on Chandrayaan-2 Flight Data")
    print(f"   Compute Device: {DEVICE.upper()} (Apple Silicon MPS / CPU)")
    print(f"   Selected Site:  {args.site.upper()}")
    print("=" * 95)

    all_results = []

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Shiv Shakti Point (-69.58°S)
    # ─────────────────────────────────────────────────────────────────────────
    if args.site in ["shiv_shakti", "both"]:
        hop1 = np.load(CACHE_DIR / "real_flight_hop1.npz", allow_pickle=True)
        hop2 = np.load(CACHE_DIR / "real_flight_hop2.npz", allow_pickle=True)

        img1_h1 = hop1["disp_ohrc"]
        img2_h1 = hop1["disp_tmc"]
        img1_h2 = hop2["disp_tmc"]
        img2_h2 = hop2["disp_iirs"]

        print(f"\n[Shiv Shakti Point -69.58°S]")
        print(f"Hop 1 (OHRC ↔ TMC-2): OHRC {img1_h1.shape} → TMC-2 {img2_h1.shape} (GSD: {HOP1_TARGET_GSD} m/px)")
        print(f"Hop 2 (TMC-2 ↔ IIRS): TMC-2 {img1_h2.shape} → IIRS {img2_h2.shape} (GSD: {HOP2_TARGET_GSD} m/px)")

        print("\n" + "-" * 95)
        print("HOP 1 EVALUATION (Shiv Shakti Point):")
        print("-" * 95)
        res_h1 = run_eval(img1_h1, img2_h1, "Hop 1 (OHRC↔TMC-2)", HOP1_TARGET_GSD)

        print("\n" + "-" * 95)
        print("HOP 2 EVALUATION (Shiv Shakti Point):")
        print("-" * 95)
        res_h2 = run_eval(img1_h2, img2_h2, "Hop 2 (TMC-2↔IIRS)", HOP2_TARGET_GSD)
        all_results.extend(res_h1 + res_h2)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Extreme South Pole (-89.72°S, Shackleton Rim / 5m DEM Anchor)
    # ─────────────────────────────────────────────────────────────────────────
    polar_npz = CACHE_DIR / "polar_flight_hop1.npz"
    if args.site in ["polar", "both"] and polar_npz.exists():
        polar_data = np.load(polar_npz, allow_pickle=True)
        img1_polar = polar_data["disp_ohrc"]
        img2_polar = polar_data["disp_tmc"]
        polar_gsd = float(polar_data.get("tmc_gsd", 4.25))

        print(f"\n[Extreme South Pole -89.72°S (Option B Anchor with 5m LOLA DEM)]")
        print(f"Polar Hop 1: OHRC (0.24 m/px) → TMC-2 (4.25 m/px) (Scale gap: 17.71x)")

        print("\n" + "-" * 95)
        print("POLAR HOP 1 EVALUATION (Shackleton Rim -89.72°S):")
        print("-" * 95)
        res_polar = run_eval(img1_polar, img2_polar, "Polar Hop 1 (OHRC↔TMC-2)", polar_gsd)
        all_results.extend(res_polar)

        # Also save standalone polar results
        with open(OUTPUT_DIR / "polar_zeroshot_results.json", "w") as f:
            json.dump(res_polar, f, indent=2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ All baseline results saved to: {OUTPUT_JSON}")

    # Formatted terminal table
    print("\n" + "=" * 120)
    print(f"{'Matcher':<24} {'Hop / Site':<26} {'Raw':>6} {'Inliers':>8} {'Ratio':>8} {'RMSE (px)':>10} {'RMSE (m)':>10} {'Time':>7} {'Status':>10}")
    print("-" * 120)
    for r in all_results:
        rmse_px = f"{r['rmse_px']:.2f}" if r["rmse_px"] is not None else "—"
        rmse_m = f"{r['rmse_m']:.1f}" if r["rmse_m"] is not None else "—"
        ratio = f"{r['inlier_ratio_pct']:.1f}%" if r["inlier_ratio_pct"] is not None else "—"
        print(f"{r['matcher']:<24} {r['hop']:<26} {r['raw_matches']:>6} {r['inliers']:>8} {ratio:>8} {rmse_px:>10} {rmse_m:>10} {r['runtime_s']:>6.2f}s {r['status']:>10}")
    print("=" * 120)


if __name__ == "__main__":
    main()
