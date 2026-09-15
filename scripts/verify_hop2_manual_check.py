#!/usr/bin/env python3
"""
scripts/verify_hop2_manual_check.py

Independent, standalone verification of Hop 2 (TMC-2 ↔ IIRS) results:
  1. Data Provenance: Audits exact NPZ path, array keys, shapes, dtypes, flight metadata.
     Saves raw images before any processing.
  2. Independent Re-run: Evaluates 1b (Phase Congruency + Fine-Tuned LoFTR)
     with deterministic RNG seed (42).
  3. Visual Correspondence Inspection: Generates side-by-side overlays with inlier lines
     saved to outputs/qa/hop2_1b_manual_check.png.
"""

import copy
import math
import sys
import time
from pathlib import Path
from typing import Tuple, Dict, Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.phase_congruency import compute_phase_congruency

RNG_SEED = 42

def set_seed(seed=RNG_SEED):
    cv2.setRNGSeed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(RNG_SEED)

CACHE_DIR = REPO_ROOT / "assets" / "real_cache"
OUTPUTS_QA = REPO_ROOT / "outputs" / "qa"
OUTPUTS_QA.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = REPO_ROOT / "models" / "eloftr_lunar_finetuned_epoch7.pt"

HOP2_GSD = 68.38  # IIRS ground sampling distance (m/px)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Model Loader
# ─────────────────────────────────────────────────────────────────────────────
ELOFTR_DIR = REPO_ROOT / "third_party" / "EfficientLoFTR"

def load_finetuned_model():
    if str(ELOFTR_DIR) not in sys.path:
        sys.path.insert(0, str(ELOFTR_DIR))
    if "src" in sys.modules and not hasattr(sys.modules["src"], "loftr"):
        del sys.modules["src"]
    proj_root = str(REPO_ROOT)
    if proj_root in sys.path:
        sys.path.remove(proj_root)

    from src.loftr import LoFTR, full_default_cfg
    if proj_root not in sys.path:
        sys.path.append(proj_root)

    cfg = copy.deepcopy(full_default_cfg)
    cfg['coarse']['npe'] = [256, 256, 256, 256]
    cfg['match_coarse']['train_pad_num_gt_min'] = 20
    cfg['match_coarse']['train_coarse_percent'] = 0.2
    cfg['match_fine']['local_regress_temperature'] = 10.0

    model = LoFTR(config=cfg)
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint at {CHECKPOINT_PATH}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(ckpt['model_state'])
    model = model.eval().to(DEVICE)
    return model


def run_loftr_pair(model, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    i1_256 = cv2.resize(img1, (256, 256), interpolation=cv2.INTER_AREA)
    i2_256 = cv2.resize(img2, (256, 256), interpolation=cv2.INTER_AREA)

    t1 = torch.from_numpy(i1_256).float()[None, None].to(DEVICE) / 255.0
    t2 = torch.from_numpy(i2_256).float()[None, None].to(DEVICE) / 255.0

    with torch.no_grad():
        batch = {'image0': t1, 'image1': t2}
        model(batch)

    pts1 = batch['mkpts0_f'].cpu().numpy()
    pts2 = batch['mkpts1_f'].cpu().numpy()

    scale1_x, scale1_y = w1 / 256.0, h1 / 256.0
    scale2_x, scale2_y = w2 / 256.0, h2 / 256.0

    if len(pts1) > 0:
        pts1[:, 0] *= scale1_x
        pts1[:, 1] *= scale1_y
        pts2[:, 0] *= scale2_x
        pts2[:, 1] *= scale2_y

    return pts1, pts2


# ─────────────────────────────────────────────────────────────────────────────
# 2. Evaluation & Transformation Decomposition
# ─────────────────────────────────────────────────────────────────────────────
def compute_reproj_rmse(pts1, pts2, H, mask):
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


def evaluate_4dof(pts1, pts2, target_gsd=HOP2_GSD):
    set_seed(RNG_SEED)
    total_candidates = len(pts1)
    if total_candidates < 4:
        return {
            "raw_matches": total_candidates,
            "inliers": 0,
            "inlier_ratio_pct": 0.0,
            "rmse_px": None,
            "rmse_m": None,
            "status": "DEGENERATE (<4 candidates)",
            "gate_status": "GATED",
            "is_gated": True,
            "H": None,
            "mask": np.zeros(total_candidates, dtype=bool),
            "scale": None,
            "rotation_deg": None,
            "tx": None,
            "ty": None,
        }

    M, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=15.0)
    inlier_mask = mask.ravel().astype(bool) if mask is not None else np.zeros(total_candidates, dtype=bool)
    H = np.eye(3, dtype=np.float64)
    if M is not None:
        H[:2, :] = M
        s = float(np.sqrt(M[0, 0]**2 + M[0, 1]**2))
        theta_rad = float(np.arctan2(M[1, 0], M[0, 0]))
        theta_deg = float(np.degrees(theta_rad))
        tx = float(M[0, 2])
        ty = float(M[1, 2])
    else:
        s, theta_deg, tx, ty = None, None, None, None

    inliers = int(np.sum(inlier_mask))
    ratio = (inliers / total_candidates * 100.0) if total_candidates > 0 else 0.0
    rmse_px = compute_reproj_rmse(pts1, pts2, H, inlier_mask)
    rmse_m = (rmse_px * target_gsd) if rmse_px != float("inf") else None

    is_degen = inliers < 8
    is_gated = (ratio < 15.0) or (inliers < 20) or is_degen
    status = "DEGENERATE" if is_degen else ("GATED" if is_gated else "CLEARED")
    gate_status = "CLEARED" if not is_gated else "GATED"

    return {
        "raw_matches": total_candidates,
        "inliers": inliers,
        "inlier_ratio_pct": round(ratio, 2),
        "rmse_px": round(rmse_px, 2) if rmse_px != float("inf") else None,
        "rmse_m": round(rmse_m, 1) if rmse_m is not None else None,
        "status": status,
        "gate_status": gate_status,
        "is_gated": is_gated,
        "H": H,
        "mask": inlier_mask,
        "scale": round(s, 4) if s is not None else None,
        "rotation_deg": round(theta_deg, 2) if theta_deg is not None else None,
        "tx": round(tx, 2) if tx is not None else None,
        "ty": round(ty, 2) if ty is not None else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. High-Quality Overlay Renderer
# ─────────────────────────────────────────────────────────────────────────────
def render_correspondence_check(
    img_left: np.ndarray,
    img_right: np.ndarray,
    pts_left: np.ndarray,
    pts_right: np.ndarray,
    mask: np.ndarray,
    title_text: str,
    subtitle_text: str,
    out_path: Path,
):
    """
    Renders side-by-side images with lines connecting every RANSAC inlier.
    Draws inlier lines in high-contrast neon green, endpoints with orange/cyan circles,
    and displays transformation and gate metrics in a clean header.
    """
    h_l, w_l = img_left.shape[:2]
    h_r, w_r = img_right.shape[:2]
    h = max(h_l, h_r)

    # Convert to RGB
    c_left = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR) if img_left.ndim == 2 else img_left.copy()
    c_right = cv2.cvtColor(img_right, cv2.COLOR_GRAY2BGR) if img_right.ndim == 2 else img_right.copy()

    # Create canvas
    canvas_w = w_l + w_r
    header_h = 110
    canvas_h = h + header_h
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Fill header with dark slate
    canvas[:header_h, :] = (24, 28, 36)

    # Place images
    canvas[header_h:header_h + h_l, :w_l] = c_left
    canvas[header_h:header_h + h_r, w_l:w_l + w_r] = c_right

    # Draw divider line
    cv2.line(canvas, (w_l, header_h), (w_l, canvas_h), (90, 90, 90), 2)

    # Draw titles
    cv2.putText(canvas, title_text, (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle_text, (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 1, cv2.LINE_AA)

    # Labels for sensors
    cv2.putText(canvas, "TMC-2 Visible (Nadir, 4.72 m/px)", (25, header_h + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "IIRS SWIR (Hyperspectral, 68.38 m/px)", (w_l + 25, header_h + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

    inlier_indices = np.where(mask)[0]

    # Draw inlier lines and keypoints
    for idx in inlier_indices:
        p1 = (int(round(pts_left[idx][0])), int(round(pts_left[idx][1] + header_h)))
        p2 = (int(round(pts_right[idx][0] + w_l)), int(round(pts_right[idx][1] + header_h)))

        # Inlier line in lime green
        cv2.line(canvas, p1, p2, (0, 240, 70), 2, cv2.LINE_AA)

        # Endpoints
        cv2.circle(canvas, p1, 4, (0, 140, 255), -1, cv2.LINE_AA)  # Orange for TMC-2
        cv2.circle(canvas, p1, 5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, p2, 4, (255, 200, 0), -1, cv2.LINE_AA)  # Cyan/yellow for IIRS
        cv2.circle(canvas, p2, 5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), canvas)
    print(f"   Saved correspondence overlay: {out_path} ({len(inlier_indices)} inlier lines drawn)")


# ─────────────────────────────────────────────────────────────────────────────
# Main Verification Execution
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 85)
    print("🔍 TRINETRA HOP 2 (TMC-2 ↔ IIRS) INDEPENDENT VERIFICATION & AUDIT")
    print(f"   Execution Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   RNG Seed:       {RNG_SEED} (cv2.setRNGSeed / np / torch)")
    print(f"   Device:         {DEVICE.upper()}")
    print("=" * 85)

    # ─────────────────────────────────────────────────────────────────────────
    # PART 1: PROVE THE DATA IS RIGHT
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 85)
    print("📂 PART 1: DATA PROVENANCE & RAW ARRAY AUDIT")
    print("─" * 85)

    h2_path = CACHE_DIR / "real_flight_hop2.npz"
    print(f"1. Target NPZ Path: {h2_path.resolve()}")
    if not h2_path.exists():
        print(f"   ❌ ERROR: File not found at {h2_path}")
        sys.exit(1)
    print(f"   File size: {h2_path.stat().st_size:,} bytes")

    data = np.load(h2_path, allow_pickle=True)
    all_keys = list(data.files)
    print(f"2. All Keys in NPZ ({len(all_keys)} keys):")
    print(f"   {all_keys}")

    # Inspect image0 and image1
    print("\n3. Arrays used for Model Evaluation:")
    disp_tmc = data["disp_tmc"]
    disp_iirs = data["disp_iirs"]
    print(f"   image0 ('disp_tmc'):  shape={disp_tmc.shape}, dtype={disp_tmc.dtype}, min={disp_tmc.min()}, max={disp_tmc.max()}")
    print(f"   image1 ('disp_iirs'): shape={disp_iirs.shape}, dtype={disp_iirs.dtype}, min={disp_iirs.min()}, max={disp_iirs.max()}")

    # Flight Metadata
    print("\n4. Flight Scene Metadata stored in cache:")
    print(f"   TMC-2 Product ID:   {data.get('tmc_id')}")
    print(f"   IIRS Product ID:    {data.get('iirs_id')}")
    print(f"   Center Coordinates: Lat {data.get('center_lat')}°S, Lon {data.get('center_lon')}°E")
    print(f"   Spatial GSD:        TMC-2 {data.get('tmc_res')} m/px | IIRS {data.get('iir_res')} m/px")
    print(f"   Scale Gap:          {data.get('scale_gap')}× native resolution difference")
    print(f"   TMC-2 Illumination: Azimuth {data.get('tmc_sun_az')}°, Elevation {data.get('tmc_sun_el')}° (Time: {data.get('tmc_time')})")
    print(f"   IIRS Illumination:  Azimuth {data.get('iirs_sun_az')}°, Elevation {data.get('iirs_sun_el')}° (Time: {data.get('iirs_time')})")

    # Native Crops
    raw_tmc = data.get("raw_tmc_crop")
    raw_iirs = data.get("raw_iirs_crop")
    if raw_tmc is not None:
        print(f"   Native TMC-2 crop:  shape={raw_tmc.shape}, dtype={raw_tmc.dtype}")
    if raw_iirs is not None:
        print(f"   Native IIRS crop:   shape={raw_iirs.shape}, dtype={raw_iirs.dtype}")

    # Save raw images before any preprocessing
    print("\n5. Saving raw PNGs before any preprocessing:")
    raw_tmc_png = OUTPUTS_QA / "hop2_raw_tmc.png"
    raw_iirs_png = OUTPUTS_QA / "hop2_raw_iirs.png"
    cv2.imwrite(str(raw_tmc_png), disp_tmc)
    cv2.imwrite(str(raw_iirs_png), disp_iirs)
    print(f"   Saved raw TMC-2 display image: {raw_tmc_png} ({disp_tmc.shape})")
    print(f"   Saved raw IIRS display image:  {raw_iirs_png} ({disp_iirs.shape})")

    if raw_tmc is not None:
        native_tmc_png = OUTPUTS_QA / "hop2_raw_tmc_native.png"
        cv2.imwrite(str(native_tmc_png), raw_tmc)
        print(f"   Saved native TMC-2 unrescaled crop: {native_tmc_png} ({raw_tmc.shape})")
    if raw_iirs is not None:
        native_iirs_png = OUTPUTS_QA / "hop2_raw_iirs_native.png"
        cv2.imwrite(str(native_iirs_png), raw_iirs)
        print(f"   Saved native IIRS unrescaled crop:  {native_iirs_png} ({raw_iirs.shape})")

    # ─────────────────────────────────────────────────────────────────────────
    # PART 2 & 3: RE-RUN EVALUATIONS IN FRESH PROCESS & RENDER VISUALIZATIONS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 85)
    print("🤖 PART 2 & 3: INDEPENDENT EVALUATION & VISUAL CORRESPONDENCE CHECK")
    print("─" * 85)

    print("\nLoading fine-tuned EfficientLoFTR checkpoint...")
    model = load_finetuned_model()
    print("Model initialized successfully.")

    # ── EVALUATION 1B: PHASE CONGRUENCY PREPROCESSING ──
    print("\n" + "-" * 70)
    print("Evaluating 1b: Fine-Tuned LoFTR on Log-Gabor Phase Congruency Maps")
    print("-" * 70)
    t0_pc = time.time()
    pc_tmc, _ = compute_phase_congruency(disp_tmc, nscale=4, norient=6)
    pc_iirs, _ = compute_phase_congruency(disp_iirs, nscale=4, norient=6)
    pc_gen_time = time.time() - t0_pc
    print(f"   Log-Gabor phase congruency maps generated in {pc_gen_time:.2f}s")
    print(f"   pc_tmc:  shape={pc_tmc.shape}, dtype={pc_tmc.dtype}, min={pc_tmc.min()}, max={pc_tmc.max()}")
    print(f"   pc_iirs: shape={pc_iirs.shape}, dtype={pc_iirs.dtype}, min={pc_iirs.min()}, max={pc_iirs.max()}")

    # Save PC maps for reference
    cv2.imwrite(str(OUTPUTS_QA / "hop2_pc_tmc.png"), pc_tmc)
    cv2.imwrite(str(OUTPUTS_QA / "hop2_pc_iirs.png"), pc_iirs)

    t0_match = time.time()
    pts1_1b, pts2_1b = run_loftr_pair(model, pc_tmc, pc_iirs)
    res_1b = evaluate_4dof(pts1_1b, pts2_1b)
    elapsed_1b = time.time() - t0_match

    print(f"   Raw Match Candidates:   {res_1b['raw_matches']}")
    print(f"   RANSAC Inliers:         {res_1b['inliers']}")
    print(f"   Inlier Ratio:           {res_1b['inlier_ratio_pct']}%")
    print(f"   Reprojection RMSE:      {res_1b['rmse_px']} px ({res_1b['rmse_m']} m)")
    print(f"   Transformation Scale:   {res_1b['scale']}")
    print(f"   Rotation Angle:         {res_1b['rotation_deg']}°")
    print(f"   Translation (tx, ty):   ({res_1b['tx']}, {res_1b['ty']}) px")
    print(f"   Flight Gate Status:     {res_1b['gate_status']} (Elapsed: {elapsed_1b:.2f}s)")

    out_1b_png = OUTPUTS_QA / "hop2_1b_manual_check.png"
    render_correspondence_check(
        img_left=disp_tmc,
        img_right=disp_iirs,
        pts_left=pts1_1b,
        pts_right=pts2_1b,
        mask=res_1b["mask"],
        title_text=f"Hop 2 (1b: Phase Congruency) -- {res_1b['inliers']}/{res_1b['raw_matches']} Inliers ({res_1b['inlier_ratio_pct']}%) [{res_1b['gate_status']}]",
        subtitle_text=f"RMSE: {res_1b['rmse_px']} px ({res_1b['rmse_m']} m) | Scale: {res_1b['scale']} | Rot: {res_1b['rotation_deg']} deg | Trans: ({res_1b['tx']}, {res_1b['ty']}) px | seed=42",
        out_path=out_1b_png,
    )

    # Also render on top of the actual PC maps for 1b so user can see PC feature points
    out_1b_on_pc_png = OUTPUTS_QA / "hop2_1b_on_pc_maps.png"
    render_correspondence_check(
        img_left=pc_tmc,
        img_right=pc_iirs,
        pts_left=pts1_1b,
        pts_right=pts2_1b,
        mask=res_1b["mask"],
        title_text=f"Hop 2 (1b on Phase Congruency Maps) -- {res_1b['inliers']}/{res_1b['raw_matches']} Inliers ({res_1b['inlier_ratio_pct']}%)",
        subtitle_text=f"Underlying log-Gabor M_max maps | RMSE: {res_1b['rmse_px']} px ({res_1b['rmse_m']} m)",
        out_path=out_1b_on_pc_png,
    )

    print("\n" + "=" * 85)
    print("✅ INDEPENDENT VERIFICATION EXECUTION COMPLETE")
    print(f"   Hop 2 (1b Phase Congruency): {res_1b['inliers']} inliers / {res_1b['raw_matches']} raw ({res_1b['inlier_ratio_pct']}%) -- {res_1b['gate_status']}")
    print("=" * 85)


if __name__ == "__main__":
    main()
