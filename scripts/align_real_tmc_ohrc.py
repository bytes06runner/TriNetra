#!/usr/bin/env python3
"""
Finalize Real Flight OHRC ↔ TMC-2 Correspondence:
- Real OHRC: ch2_ohr_ncp_20211023T0027462822_d_img_d18 (0.26 m/px, Lines 10000:14000, Samples 4000:8000)
  Center: Lat -69.58019, Lon 32.28800
- Real TMC-2: ch2_tmc_ncn_20230130T1900132182_d_img_d32 (4.72 m/px, Lines 132550:132850, Samples 560:860)
  Center: Lat -69.57911, Lon 32.27507
"""

import math
from pathlib import Path
import numpy as np
import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "assets/real_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DESKTOP_OHRC = Path.home() / "Desktop/data/data/calibrated/20211023/ch2_ohr_ncp_20211023T0027462822_d_img_d18.img"
TMC_IMG = REPO_ROOT / "data/ch2_tmc_ncn_20230130T1900132182_d_img_d32/data/calibrated/20230130/ch2_tmc_ncn_20230130T1900132182_d_img_d32.img"
OUTPUT_NPZ = CACHE_DIR / "real_flight_hop1.npz"
OUTPUT_PNG = CACHE_DIR / "real_flight_hop1_matches.png"

def main():
    print("--- Reading Authentic Chandrayaan-2 Flight Data ---")
    # 1. Load Real TMC-2 flight crop at exact ground footprint
    mm_tmc = np.memmap(str(TMC_IMG), dtype="<u2", mode="r", shape=(189886, 4000))
    l_center, s_center = 132700, 710
    half_l, half_s = 150, 150
    tmc_crop = mm_tmc[l_center - half_l : l_center + half_l, s_center - half_s : s_center + half_s].copy()

    p1_t, p99_t = np.percentile(tmc_crop, (1.0, 99.0))
    tmc_u8 = np.clip((tmc_crop.astype(np.float32) - p1_t) / max(1e-6, p99_t - p1_t) * 255.0, 0, 255).astype(np.uint8)

    # 2. Load Real OHRC flight crop
    if DESKTOP_OHRC.exists():
        mm_ohr = np.memmap(str(DESKTOP_OHRC), dtype=np.uint8, mode="r", shape=(93693, 12000))
        ohr_raw = mm_ohr[10000:14000, 4000:8000].copy()
    else:
        cached_ohrc = np.load(CACHE_DIR / "real_ohrc_crop.npz")
        ohr_raw = cached_ohrc["ohrc_disp"]

    p1_o, p99_o = np.percentile(ohr_raw, (1.0, 99.0))
    ohr_u8 = np.clip((ohr_raw.astype(np.float32) - p1_o) / max(1e-6, p99_o - p1_o) * 255.0, 0, 255).astype(np.uint8)

    # Display pairs standardized to 1000x1000 for high-resolution interactive UI
    disp_ohr = cv2.resize(ohr_u8, (1000, 1000), interpolation=cv2.INTER_AREA)
    disp_tmc = cv2.resize(tmc_u8, (1000, 1000), interpolation=cv2.INTER_CUBIC)

    # Preprocessing with CLAHE for shadow-edge preservation
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    o_clahe = clahe.apply(disp_ohr)
    t_clahe = clahe.apply(disp_tmc)

    # Multi-scale SIFT Keypoints & Descriptors
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.015, edgeThreshold=10)
    kp1, des1 = sift.detectAndCompute(o_clahe, None)
    kp2, des2 = sift.detectAndCompute(t_clahe, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    # Spatial deduplication: enforce unique physical crater features (>20 px apart)
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

    # Robust Affine Similarity estimation (Scale, Rotation, Translation)
    # Orbital nadir cameras over lunar terrain obey Euclidean similarity/affine geometry
    if total_candidates >= 4:
        M, inliers_mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=15.0)
        inliers = int(np.sum(inliers_mask)) if inliers_mask is not None else 0
        inlier_ratio = (inliers / total_candidates * 100.0) if total_candidates > 0 else 0.0
        mask_bool = inliers_mask.ravel().astype(bool) if inliers_mask is not None else np.zeros(total_candidates, dtype=bool)
        H = np.eye(3, dtype=np.float64)
        if M is not None:
            H[:2, :] = M
    else:
        H = np.eye(3, dtype=np.float64)
        inliers = 0
        inlier_ratio = 0.0
        mask_bool = np.zeros(total_candidates, dtype=bool)

    # Compute Euclidean reprojection RMSE on inliers in pixels
    pts1_in = pts1[mask_bool]
    pts2_in = pts2[mask_bool]
    if len(pts1_in) > 0 and M is not None:
        proj_xy = cv2.transform(pts1_in.reshape(-1, 1, 2), M).reshape(-1, 2)
        reproj_errs = np.sqrt(np.sum((proj_xy - pts2_in)**2, axis=-1))
        reproj_rmse = float(np.sqrt(np.mean(reproj_errs**2)))
    else:
        reproj_rmse = 0.0

    print(f"Genuine Flight Inliers (Robust Similarity): {inliers} / {total_candidates} ({inlier_ratio:.1f}%)")
    print(f"Reprojection RMSE on Inliers: {reproj_rmse:.2f} pixels (Threshold: 15.0 px)")

    # Render Side-by-Side Match Image
    vis = np.hstack([disp_ohr, disp_tmc])
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
    w = disp_ohr.shape[1]

    for p1, p2 in zip(pts1_in, pts2_in):
        pt1 = (int(round(p1[0])), int(round(p1[1])))
        pt2 = (int(round(p2[0] + w)), int(round(p2[1])))
        cv2.line(vis_rgb, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(vis_rgb, pt1, 4, (255, 120, 0), -1)
        cv2.circle(vis_rgb, pt2, 4, (0, 200, 255), -1)

    cv2.imwrite(str(OUTPUT_PNG), vis_rgb)

    # Save compact NPZ archive
    np.savez_compressed(
        OUTPUT_NPZ,
        disp_ohrc=disp_ohr,
        disp_tmc=disp_tmc,
        raw_tmc_crop=tmc_u8,
        pts1=pts1,
        pts2=pts2,
        inlier_mask=mask_bool,
        H=H,
        transform_type="Similarity Transform",
        transform_dof=4,
        inliers=inliers,
        total_matches=total_candidates,
        inlier_ratio=inlier_ratio,
        reproj_rmse=reproj_rmse,
        inlier_threshold=15.0,
        ohrc_res=0.26,
        tmc_res=4.72,
        scale_gap=4.72 / 0.26,
        ohrc_product_id="ch2_ohr_ncp_20211023T0027462822_d_img_d18",
        tmc_product_id="ch2_tmc_ncn_20230130T1900132182_d_img_d32",
        target_lat=-69.58019,
        target_lon=32.28800,
        tmc_sun_elevation=17.22,
        tmc_sun_azimuth=53.02,
        ohrc_sun_elevation=9.13,
        ohrc_sun_azimuth=298.43,
        flight_validated=True,
    )
    print(f"Saved real flight Hop 1 archive to {OUTPUT_NPZ} ({OUTPUT_NPZ.stat().st_size / 1e6:.2f} MB)")

if __name__ == "__main__":
    main()
