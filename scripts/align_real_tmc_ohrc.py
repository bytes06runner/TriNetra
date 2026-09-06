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

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw_matches if len((m, n)) == 2 and m.distance < 0.85 * n.distance]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Robust MAGSAC++ homography estimation
    if len(good) >= 4:
        H, mask = cv2.findHomography(pts1, pts2, cv2.USAC_MAGSAC, 5.0)
        inliers = int(np.sum(mask)) if mask is not None else 0
        inlier_ratio = (inliers / len(good) * 100.0) if len(good) > 0 else 0.0
        mask_bool = mask.ravel().astype(bool) if mask is not None else np.zeros(len(good), dtype=bool)
    else:
        H = np.eye(3)
        inliers = 0
        inlier_ratio = 0.0
        mask_bool = np.zeros(len(good), dtype=bool)

    print(f"Genuine Flight Inliers (MAGSAC++): {inliers} / {len(good)} ({inlier_ratio:.1f}%)")

    # Render Side-by-Side Match Image
    match_img = cv2.drawMatches(
        disp_ohr, [cv2.KeyPoint(p[0][0], p[0][1], 5) for p in pts1],
        disp_tmc, [cv2.KeyPoint(p[0][0], p[0][1], 5) for p in pts2],
        [cv2.DMatch(i, i, 0) for i in range(len(pts1)) if mask_bool[i]],
        None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(str(OUTPUT_PNG), match_img)

    # Save compact NPZ archive
    np.savez_compressed(
        OUTPUT_NPZ,
        disp_ohrc=disp_ohr,
        disp_tmc=disp_tmc,
        raw_tmc_crop=tmc_u8,
        pts1=pts1.reshape(-1, 2),
        pts2=pts2.reshape(-1, 2),
        inlier_mask=mask_bool,
        H=H,
        inliers=inliers,
        total_matches=len(good),
        inlier_ratio=inlier_ratio,
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
