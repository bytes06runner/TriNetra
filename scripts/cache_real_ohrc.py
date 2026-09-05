"""
Extract and cache real Chandrayaan-2 OHRC flight crop and optical TMC-2 proxy.

Reads real OHRC calibrated product:
ch2_ohr_ncp_20211023T0027462822_d_img_d18.img (0.26 m/px, 93693 x 12000 px).
Extracts optimal high-contrast crater field at Lines [10000:14000], Samples [4000:8000].
Performs 20x optical downsampling to 5.2 m/px to emulate TMC-2.
Computes multi-scale SIFT matches and USAC_MAGSAC homography.
Saves compact cache to assets/real_cache/real_ohrc_crop.npz (~3 MB).

SIH26166 — Chandrayaan-2 Multi-modal Image Correspondence Pipeline
"""

import cv2
import numpy as np
from pathlib import Path

DESKTOP_OHRC = Path.home() / "Desktop/data/data/calibrated/20211023/ch2_ohr_ncp_20211023T0027462822_d_img_d18.img"
REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "assets/real_cache"
OUTPUT_NPZ = CACHE_DIR / "real_ohrc_crop.npz"


def generate_real_ohrc_cache():
    if not DESKTOP_OHRC.exists():
        raise FileNotFoundError(f"Real OHRC file not found at {DESKTOP_OHRC}")

    print(f"Reading real OHRC flight data from {DESKTOP_OHRC}...")
    lines, samples = 93693, 12000
    mm = np.memmap(str(DESKTOP_OHRC), dtype=np.uint8, mode="r", shape=(lines, samples))

    # Optimal high-contrast crater field
    l_start, l_end = 10000, 14000
    s_start, s_end = 4000, 8000
    patch = mm[l_start:l_end, s_start:s_end].copy()

    # Percentile normalize to enhance subtle crater shadows
    p1, p99 = np.percentile(patch, (1.0, 99.0))
    patch_norm = np.clip((patch.astype(np.float32) - p1) / max(1e-6, p99 - p1) * 255.0, 0, 255).astype(np.uint8)

    # Downsample by 20x to 5.2 m/px (TMC-2 GSD emulation)
    down_w = patch.shape[1] // 20  # 200
    down_h = patch.shape[0] // 20  # 200
    tmc_proxy = cv2.resize(patch_norm, (down_w, down_h), interpolation=cv2.INTER_AREA)

    # For responsive web display and fast feature matching, create 1000x1000 display pair
    disp_ohrc = cv2.resize(patch_norm, (1000, 1000), interpolation=cv2.INTER_AREA)
    disp_tmc = cv2.resize(tmc_proxy, (1000, 1000), interpolation=cv2.INTER_CUBIC)

    # Scale-aligned SIFT matching
    print("Extracting SIFT keypoints across 20x scale gap...")
    sift = cv2.SIFT_create(nfeatures=2000)
    kp1, des1 = sift.detectAndCompute(disp_ohrc, None)
    kp2, des2 = sift.detectAndCompute(disp_tmc, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)

    H, mask = cv2.findHomography(pts1.reshape(-1, 1, 2), pts2.reshape(-1, 1, 2), cv2.USAC_MAGSAC, 3.0)
    inliers = int(np.sum(mask))
    inlier_ratio = inliers / len(good) * 100.0

    print(f"SIFT Matches: {len(good)} | Inliers: {inliers} ({inlier_ratio:.1f}%)")

    # Save to compact npz
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_NPZ,
        ohrc_disp=disp_ohrc,
        tmc_proxy=tmc_proxy,
        tmc_disp=disp_tmc,
        pts1=pts1,
        pts2=pts2,
        inlier_mask=mask.ravel().astype(bool),
        H=H,
        inliers=inliers,
        total_matches=len(good),
        inlier_ratio=inlier_ratio,
        ohrc_res=0.26,
        tmc_res=5.20,
        scale_gap=20.0,
        sun_incidence=80.87,
        sun_azimuth=298.43,
        center_lat=-69.25,
        center_lon=32.33,
    )
    print(f"Saved real OHRC cache to {OUTPUT_NPZ} ({OUTPUT_NPZ.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    generate_real_ohrc_cache()
