#!/usr/bin/env python3
"""
generate_advanced_solutions_cache.py

Precomputes and validates the two advanced solutions for Hop 1:
1. DEM-Aware 3D Terrain Orthorectification (eliminating 15.8° parallax)
2. Deep Learned Feature Matching (LoFTR / Transformer Attention overcoming 114.6° solar shift)
3. Combined Solution (DEM Ortho + LoFTR Matching)

Saves compact archive to assets/real_cache/real_flight_advanced_solutions.npz.
"""

from pathlib import Path
import sys
import numpy as np
import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.module4_dem_registration.dem_ortho import LunarDEMOrthorectifier
from src.module5_learned_matcher.loftr_lunar import LunarLoFTRMatcher

CACHE_DIR = REPO_ROOT / "assets/real_cache"
HOP1_NPZ = CACHE_DIR / "real_flight_hop1.npz"
OUTPUT_NPZ = CACHE_DIR / "real_flight_advanced_solutions.npz"


def main():
    print("=" * 70)
    print("TriNetra: Generating Advanced Solutions Cache (DEM + LoFTR)")
    print("=" * 70)

    if not HOP1_NPZ.exists():
        raise FileNotFoundError(f"Hop 1 flight cache not found: {HOP1_NPZ}")

    hop1_data = np.load(str(HOP1_NPZ))
    disp_ohr = hop1_data["disp_ohrc"]
    disp_tmc = hop1_data["disp_tmc"]
    target_gsd = float(hop1_data.get("tmc_res", 4.72))

    print(f"Loaded Flight Crops: OHRC {disp_ohr.shape}, TMC-2 {disp_tmc.shape}")

    # ─────────────────────────────────────────────────────────────
    # SOLUTION 1: DEM-Aware 3D Terrain Orthorectification
    # ─────────────────────────────────────────────────────────────
    print("\n--- Running Solution 1: DEM-Aware 3D Terrain Orthorectification ---")
    dem_engine = LunarDEMOrthorectifier(
        ohrc_roll_deg=15.76,
        tmc_roll_deg=-0.02,
        ohrc_gsd_m=0.26,
        tmc_gsd_m=target_gsd,
    )

    dem_result = dem_engine.orthorectify(disp_ohr, disp_tmc)
    print(f"Elevation Range: {dem_result.elevation_min_m:.1f} m to {dem_result.elevation_max_m:.1f} m (Relief: {dem_result.relief_m:.1f} m)")
    print(f"Max Ground Parallax: {dem_result.max_parallax_m:.1f} m ({dem_result.max_parallax_m / target_gsd:.1f} TMC-2 pixels)")
    print(f"Mean Ground Parallax: {dem_result.mean_parallax_m:.1f} m")

    # SIFT on Orthorectified image vs TMC-2 (testing parallax elimination alone)
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.015)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    ortho_clahe = clahe.apply(dem_result.disp_ohrc_ortho)
    tmc_clahe = clahe.apply(disp_tmc)

    kp1, des1 = sift.detectAndCompute(ortho_clahe, None)
    kp2, des2 = sift.detectAndCompute(tmc_clahe, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches_dem = bf.match(des1, des2) if des1 is not None and des2 is not None else []

    pts1_dem = np.float32([kp1[m.queryIdx].pt for m in matches_dem]) if matches_dem else np.empty((0, 2))
    pts2_dem = np.float32([kp2[m.trainIdx].pt for m in matches_dem]) if matches_dem else np.empty((0, 2))

    if len(pts1_dem) >= 4:
        M_dem, mask_dem = cv2.estimateAffinePartial2D(pts1_dem, pts2_dem, method=cv2.RANSAC, ransacReprojThreshold=15.0)
        dem_inliers = int(np.sum(mask_dem)) if mask_dem is not None else 0
        dem_ratio = (dem_inliers / len(pts1_dem) * 100.0) if len(pts1_dem) > 0 else 0.0
        H_dem = np.eye(3, dtype=np.float64)
        if M_dem is not None:
            H_dem[:2, :] = M_dem
    else:
        dem_inliers = len(pts1_dem)
        dem_ratio = 100.0 if len(pts1_dem) > 0 else 0.0
        H_dem = np.eye(3, dtype=np.float64)

    print(f"DEM Ortho SIFT inliers: {dem_inliers} / {len(pts1_dem)} ({dem_ratio:.1f}%)")

    # ─────────────────────────────────────────────────────────────
    # SOLUTION 2: Deep Learned Feature Matching (LoFTR Attention)
    # ─────────────────────────────────────────────────────────────
    print("\n--- Running Solution 2: Deep Learned Matching (LoFTR Semantic Attention) ---")
    loftr_engine = LunarLoFTRMatcher(grid_step=16, feature_dim=64, target_gsd_m=target_gsd)
    loftr_result = loftr_engine.match(disp_ohr, disp_tmc, inlier_threshold_px=8.0)
    print(f"LoFTR Inliers: {loftr_result.inliers} / {loftr_result.total_matches} ({loftr_result.inlier_ratio:.1f}%)")
    print(f"LoFTR RMSE: {loftr_result.reproj_rmse_px:.2f} px ({loftr_result.reproj_rmse_m:.1f} m)")

    # ─────────────────────────────────────────────────────────────
    # SOLUTION 3: Combined Solution (DEM Orthorectification + LoFTR)
    # ─────────────────────────────────────────────────────────────
    print("\n--- Running Solution 3: Combined TriNetra Pipeline (DEM + LoFTR) ---")
    combined_result = loftr_engine.match(dem_result.disp_ohrc_ortho, disp_tmc, inlier_threshold_px=6.0)
    print(f"Combined Inliers: {combined_result.inliers} / {combined_result.total_matches} ({combined_result.inlier_ratio:.1f}%)")
    print(f"Combined RMSE: {combined_result.reproj_rmse_px:.2f} px ({combined_result.reproj_rmse_m:.1f} m)")

    # Save compact NPZ archive
    np.savez_compressed(
        str(OUTPUT_NPZ),
        # DEM outputs
        elevation_map_m=dem_result.elevation_map_m,
        hillshade=dem_result.hillshade,
        parallax_dx_px=dem_result.parallax_dx_px,
        parallax_dy_px=dem_result.parallax_dy_px,
        parallax_magnitude_m=dem_result.parallax_magnitude_m,
        max_parallax_m=dem_result.max_parallax_m,
        mean_parallax_m=dem_result.mean_parallax_m,
        elevation_min_m=dem_result.elevation_min_m,
        elevation_max_m=dem_result.elevation_max_m,
        relief_m=dem_result.relief_m,
        disp_ohrc_ortho=dem_result.disp_ohrc_ortho,
        H_dem=H_dem,
        dem_inliers=dem_inliers,
        dem_total_matches=len(pts1_dem),
        dem_inlier_ratio=dem_ratio,
        # LoFTR outputs
        loftr_pts1=loftr_result.pts1,
        loftr_pts2=loftr_result.pts2,
        loftr_inlier_mask=loftr_result.inlier_mask,
        loftr_confidences=loftr_result.confidences,
        loftr_inliers=loftr_result.inliers,
        loftr_total_matches=loftr_result.total_matches,
        loftr_inlier_ratio=loftr_result.inlier_ratio,
        loftr_H=loftr_result.H,
        loftr_reproj_rmse_px=loftr_result.reproj_rmse_px,
        loftr_reproj_rmse_m=loftr_result.reproj_rmse_m,
        loftr_vis_rgb=loftr_result.vis_rgb,
        # Combined outputs
        combined_pts1=combined_result.pts1,
        combined_pts2=combined_result.pts2,
        combined_inlier_mask=combined_result.inlier_mask,
        combined_inliers=combined_result.inliers,
        combined_total_matches=combined_result.total_matches,
        combined_inlier_ratio=combined_result.inlier_ratio,
        combined_H=combined_result.H,
        combined_reproj_rmse_px=combined_result.reproj_rmse_px,
        combined_reproj_rmse_m=combined_result.reproj_rmse_m,
        combined_vis_rgb=combined_result.vis_rgb,
    )
    print(f"\nSaved advanced solutions archive to: {OUTPUT_NPZ} ({OUTPUT_NPZ.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
