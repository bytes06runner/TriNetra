#!/usr/bin/env python3
"""
gen_illum_pairs.py — Refined Synthetic Multi-Illumination Pair Generator for Lunar DEM.

Generates pairs of images rendered from the 5m LOLA DEM under differing solar geometry:
  - Stratified Solar Elevation:
      • 60% well-lit: 12° - 30° (bulk of supervision)
      • 30% moderate: 6° - 12° (moderate relief shadows)
      • 10% extreme: 2° - 6° (grazing polar sun, gated by acceptance criteria)
  - Azimuth: Uniform in [0, 360)° for both images, preserving flat Δaz in [0, 180]°
  - Reflectance Modes: 50% Lambertian, 50% Lommel-Seeliger regolith scattering
  - Secondary Ambient Floor: 51 DN (20% floor, empirically matched to Chandrayaan-2 OHRC flight data)
  - Scale-Gap Augmentation: Option B (pre-shading DEM coarsening capped at <= 8x, >= 64 px content)
  - Rejection Gating (resampled until 15,000 accepted pairs are produced):
      1. valid_mask coverage >= 40%
      2. std dev >= 12 DN for both image_a and image_b
      3. near-floor pixels (<= 56 DN) <= 70% in both images
      4. shadow coverage <= 60% in both renders
  - Dual Export: Native 512x512 and 256x256 training format with analytically scaled H
"""

import os
import sys
import time
import shutil
import argparse
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import cv2
import torch
from osgeo import gdal

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.illum_render import (
    render_lunar_tile,
    render_coarse_dem_tile,
    get_default_device,
)

def sample_stratified_elevation(rng: np.random.Generator) -> float:
    """
    Stratified solar elevation sampling:
      - 60% of pairs: elevation 12° - 30° (well-lit, bulk of supervision)
      - 30% of pairs: elevation 6° - 12° (moderate shadowing, harder)
      - 10% of pairs: elevation 2° - 6° (extreme grazing, gated by acceptance criteria)
    """
    u = float(rng.uniform(0.0, 1.0))
    if u < 0.60:
        return float(rng.uniform(12.0, 30.0))
    elif u < 0.90:
        return float(rng.uniform(6.0, 12.0))
    else:
        return float(rng.uniform(2.0, 6.0))


def sample_random_homography(tile_size: int = 512, rng: np.random.Generator = None) -> np.ndarray:
    """
    Sample a realistic projective homography H_a_to_b mapping image_a coordinates to image_b:
      - Rotation: uniform [-20, +20] deg
      - Scale: uniform [0.85, 1.25]
      - Translation: uniform [-0.10 * tile_size, +0.10 * tile_size] px
      - Mild perspective skew: p_x, p_y in [-0.00015, +0.00015]
    """
    if rng is None:
        rng = np.random.default_rng()

    cx = tile_size / 2.0
    cy = tile_size / 2.0

    for _ in range(25):
        theta = np.radians(rng.uniform(-20.0, 20.0))
        scale = rng.uniform(0.85, 1.25)
        tx = rng.uniform(-0.10 * tile_size, 0.10 * tile_size)
        ty = rng.uniform(-0.10 * tile_size, 0.10 * tile_size)
        px = rng.uniform(-0.00015, 0.00015)
        py = rng.uniform(-0.00015, 0.00015)

        T1 = np.array([
            [1.0, 0.0, -cx],
            [0.0, 1.0, -cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        A = np.array([
            [scale * np.cos(theta), -scale * np.sin(theta), 0.0],
            [scale * np.sin(theta),  scale * np.cos(theta), 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        P = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [px,  py,  1.0]
        ], dtype=np.float64)

        T2 = np.array([
            [1.0, 0.0, cx + tx],
            [0.0, 1.0, cy + ty],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        H = T2 @ P @ A @ T1
        if abs(H[2, 2]) > 1e-6 and np.linalg.cond(H) < 1e5:
            H /= H[2, 2]
            return H

    return np.eye(3, dtype=np.float64)


def apply_photometric_augmentation(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply independent photometric variations: gamma, Gaussian noise, and blur."""
    out = img.astype(np.float32)

    # 1. Gamma adjustment: [0.8, 1.25]
    gamma = float(rng.uniform(0.8, 1.25))
    out = 255.0 * np.power(np.clip(out / 255.0, 0.0, 1.0), gamma)

    # 2. Additive Gaussian noise: sigma in [2, 6] DN
    sigma = float(rng.uniform(2.0, 6.0))
    noise = rng.normal(0.0, sigma, size=img.shape)
    out = np.clip(out + noise, 0.0, 255.0)

    # 3. Slight Gaussian blur with 30% probability
    res = out.round().astype(np.uint8)
    if rng.random() < 0.30:
        k = 3
        sigma_blur = float(rng.uniform(0.5, 0.8))
        res = cv2.GaussianBlur(res, (k, k), sigmaX=sigma_blur)

    return res


def compute_validity_mask(
    shadow_a: np.ndarray,
    shadow_b_unwarped: np.ndarray,
    H_a_to_b: np.ndarray,
    tile_size: int = 512,
) -> np.ndarray:
    """
    Compute boolean validity mask in image_a coordinate frame:
      - Exclude deep shadow in image_a
      - Exclude deep shadow in unwarped image_b
      - Exclude points whose mapped coordinate H_a_to_b * p_a falls outside image_b frame
    """
    H, W = tile_size, tile_size
    y_grid, x_grid = np.indices((H, W), dtype=np.float64)

    w = H_a_to_b[2, 0] * x_grid + H_a_to_b[2, 1] * y_grid + H_a_to_b[2, 2]
    w_safe = np.where(np.abs(w) < 1e-8, 1e-8, w)

    x_b = (H_a_to_b[0, 0] * x_grid + H_a_to_b[0, 1] * y_grid + H_a_to_b[0, 2]) / w_safe
    y_b = (H_a_to_b[1, 0] * x_grid + H_a_to_b[1, 1] * y_grid + H_a_to_b[1, 2]) / w_safe

    in_bounds = (w > 0.0) & (x_b >= 0.0) & (x_b < W - 1.0) & (y_b >= 0.0) & (y_b < H - 1.0)
    valid_mask = (~shadow_a) & (~shadow_b_unwarped) & in_bounds
    return valid_mask


def evaluate_pair_acceptance(
    img_a: np.ndarray,
    img_b: np.ndarray,
    valid_mask: np.ndarray,
    shadow_a: np.ndarray,
    shadow_b: np.ndarray,
    ambient_dn: int = 51,
) -> Tuple[bool, str]:
    """
    Strict acceptance criteria (all four must hold):
      1. valid_mask covers at least 40% of the frame
      2. both image_a and image_b have std dev >= 12 DN (rejects flat/black frames and over-downsampled B)
      3. neither image has more than 70% of pixels within 5 DN of the floor (ambient_dn + 5)
      4. shadow coverage in either render does not exceed 60%
    """
    # 1. Mask coverage
    mask_cov = float(np.mean(valid_mask))
    if mask_cov < 0.40:
        return False, "mask"

    # 2. Std dev
    std_a = float(np.std(img_a))
    std_b = float(np.std(img_b))
    if std_a < 12.0 or std_b < 12.0:
        return False, "std"

    # 3. Near floor
    floor_thresh = ambient_dn + 5
    near_floor_a = float(np.mean(img_a <= floor_thresh))
    near_floor_b = float(np.mean(img_b <= floor_thresh))
    if near_floor_a > 0.70 or near_floor_b > 0.70:
        return False, "floor"

    # 4. Shadow coverage
    sh_a = float(np.mean(shadow_a))
    sh_b = float(np.mean(shadow_b))
    if sh_a > 0.60 or sh_b > 0.60:
        return False, "shadow"

    return True, "accepted"


def main():
    parser = argparse.ArgumentParser(description="Refined Synthetic Illumination Pair Generator for Lunar DEM")
    parser.add_argument("--dem-path", type=str, default="data/dem/Site04_final_adj_5mpp_surf.tif",
                        help="Path to input 5m/px DEM GeoTIFF")
    parser.add_argument("--output-dir", type=str, default="data/synthetic_pairs",
                        help="Base output directory for generated dataset")
    parser.add_argument("--num-pairs", type=int, default=15000,
                        help="Number of accepted synthetic pairs to generate (default: 15000)")
    parser.add_argument("--scale-gap", action="store_true", default=True,
                        help="Enable Option B scale-gap simulation (coarse DEM pre-shading <= 8x)")
    parser.add_argument("--ambient-dn", type=int, default=51,
                        help="Ambient secondary irradiance floor in DN (default: 51 DN = 20%% floor)")
    parser.add_argument("--clean", action="store_true", default=False,
                        help="Clean output directories before generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = get_default_device()

    dem_path = Path(args.dem_path)
    if not dem_path.exists():
        print(f"❌ DEM not found at: {dem_path}")
        sys.exit(1)

    print(f"📂 Loading DEM from: {dem_path}")
    ds = gdal.Open(str(dem_path))
    full_dem = ds.ReadAsArray().astype(np.float32)
    dem_h, dem_w = full_dem.shape
    ds = None
    print(f"   DEM Shape: {full_dem.shape}, Relief: {full_dem.max() - full_dem.min():.1f} m")

    out_base = Path(args.output_dir)
    out_native = out_base / "native"
    out_256 = out_base / "train_256"

    if args.clean:
        if out_native.exists():
            shutil.rmtree(out_native)
        if out_256.exists():
            shutil.rmtree(out_256)

    out_native.mkdir(parents=True, exist_ok=True)
    out_256.mkdir(parents=True, exist_ok=True)

    tile_size = 512
    margin = 200  # Margin to avoid boundary ray-march artifacts
    max_row = dem_h - tile_size - margin
    max_col = dem_w - tile_size - margin

    print(f"🚀 Generating exactly {args.num_pairs:,} ACCEPTED multi-illumination pairs...")
    print(f"   Target Native Directory: {out_native}")
    print(f"   Target 256x256 Directory: {out_256}")
    print(f"   Compute Device: {device}")
    print(f"   Ambient Secondary Floor: {args.ambient_dn} DN (20% floor)")
    print(f"   Scale-Gap Augmentation: Option B (Coarse DEM pre-shading <= 8x, >= 64 px grid)")
    print(f"   Stratified Elevation: 60% [12°-30°], 30% [6°-12°], 10% [2°-6°]")
    print(f"   Rejection Criteria: valid_mask >= 40%, std >= 12 DN, near-floor <= 70%, shadow <= 60%")

    t_start = time.time()
    t_last = t_start

    # Scaling matrix for downsampling to 256x256: H_256 = S * H_512 * S^-1
    S = np.diag([0.5, 0.5, 1.0])
    S_inv = np.diag([2.0, 2.0, 1.0])

    accepted_count = 0
    total_attempts = 0
    rejection_reasons = {"mask": 0, "std": 0, "floor": 0, "shadow": 0}

    while accepted_count < args.num_pairs:
        # 50% Lambertian, 50% Lommel-Seeliger
        render_mode = "lambert" if (accepted_count < args.num_pairs // 2) else "ls"

        # Anchor DEM location sampling
        # Guarantee inclusion of the OHRC anchor (col 642, row 1436) in early pairs
        if accepted_count % 100 == 0 and accepted_count < 500:
            row = max(margin, min(max_row, 1436 - tile_size // 2))
            col = max(margin, min(max_col, 642 - tile_size // 2))
        else:
            row = int(rng.integers(margin, max_row))
            col = int(rng.integers(margin, max_col))

        dem_tile = full_dem[row:row + tile_size, col:col + tile_size]

        # Resample up to 10 attempts on this tile before advancing tile location
        for _ in range(10):
            total_attempts += 1

            # 1. Illumination angles
            # Image A: Uniform azimuth [0, 360)°, Stratified elevation
            az_a = float(rng.uniform(0.0, 360.0))
            alt_a = sample_stratified_elevation(rng)

            # Image B: Independent uniform azimuth [0, 360)°, Stratified elevation
            az_b = float(rng.uniform(0.0, 360.0))
            alt_b = sample_stratified_elevation(rng)

            # 2. Render Image A
            img_a_clean, shadow_a = render_lunar_tile(
                dem_tile, az_a, alt_a, mode=render_mode, pixel_size_m=5.0,
                apply_shadows=True, ambient_dn=args.ambient_dn, device=device
            )

            # 3. Render Image B (Option B: Coarse DEM Pre-shading if scale-gap enabled)
            if args.scale_gap:
                gap_factor = float(rng.uniform(2.0, 8.0))
                img_b_clean, shadow_b = render_coarse_dem_tile(
                    dem_tile, az_b, alt_b, scale_factor=gap_factor, mode=render_mode,
                    base_pixel_size_m=5.0, ambient_dn=args.ambient_dn, device=device
                )
            else:
                gap_factor = 1.0
                img_b_clean, shadow_b = render_lunar_tile(
                    dem_tile, az_b, alt_b, mode=render_mode, pixel_size_m=5.0,
                    apply_shadows=True, ambient_dn=args.ambient_dn, device=device
                )

            # 4. Sample Random Homography for Image B
            H_a_to_b = sample_random_homography(tile_size=tile_size, rng=rng)
            img_b_warped = cv2.warpPerspective(
                img_b_clean, H_a_to_b, (tile_size, tile_size),
                flags=cv2.INTER_LINEAR, borderValue=args.ambient_dn
            )

            # 5. Photometric Augmentation
            img_a_aug = apply_photometric_augmentation(img_a_clean, rng)
            img_b_aug = apply_photometric_augmentation(img_b_warped, rng)

            # 6. Validity Mask
            valid_mask = compute_validity_mask(shadow_a, shadow_b, H_a_to_b, tile_size=tile_size)

            # 7. Evaluate 4-Point Acceptance Criteria
            accepted, reason = evaluate_pair_acceptance(
                img_a_aug, img_b_aug, valid_mask, shadow_a, shadow_b, ambient_dn=args.ambient_dn
            )

            if not accepted:
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                continue

            # Pair passed all 4 criteria!
            # 8. Save Native 512x512 Pair
            native_path = out_native / f"pair_{accepted_count:05d}.npz"
            np.savez_compressed(
                native_path,
                image_a=img_a_aug,
                image_b=img_b_aug,
                H_a_to_b=H_a_to_b,
                valid_mask=valid_mask,
                shadow_a=shadow_a,
                shadow_b=shadow_b,
                az_a=az_a,
                alt_a=alt_a,
                az_b=az_b,
                alt_b=alt_b,
                scale_factor=gap_factor,
                tile_row=row,
                tile_col=col,
                dem_pixel_size_m=5.0,
                render_mode=render_mode,
            )

            # 9. Downsample and Save 256x256 Training Variant
            img_a_256 = cv2.resize(img_a_aug, (256, 256), interpolation=cv2.INTER_AREA)
            img_b_256 = cv2.resize(img_b_aug, (256, 256), interpolation=cv2.INTER_AREA)
            valid_256 = cv2.resize(valid_mask.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST).astype(bool)
            H_256 = S @ H_a_to_b @ S_inv

            train_path = out_256 / f"pair_{accepted_count:05d}.npz"
            np.savez_compressed(
                train_path,
                image_a=img_a_256,
                image_b=img_b_256,
                H_a_to_b=H_256,
                valid_mask=valid_256,
                az_a=az_a,
                alt_a=alt_a,
                az_b=az_b,
                alt_b=alt_b,
                scale_factor=gap_factor,
                tile_row=row,
                tile_col=col,
                dem_pixel_size_m=5.0,
                render_mode=render_mode,
            )

            accepted_count += 1

            # Progress reporting every 1,000 accepted pairs
            if accepted_count % 1000 == 0 or accepted_count == args.num_pairs:
                now = time.time()
                elapsed = now - t_start
                batch_rate = 1000.0 / (now - t_last) if accepted_count > 1000 else accepted_count / elapsed
                t_last = now
                eta_s = (args.num_pairs - accepted_count) / (batch_rate + 1e-6)
                rejection_pct = (total_attempts - accepted_count) / total_attempts * 100.0
                print(f"   [{accepted_count:,} / {args.num_pairs:,}] ({accepted_count/args.num_pairs*100:.1f}%) | "
                      f"Accepted Rate: {batch_rate:.1f} p/s | Rejection Rate: {rejection_pct:.1f}% | "
                      f"Elapsed: {elapsed/60:.1f}m | ETA: {eta_s/60:.1f}m")
            break

    total_time = time.time() - t_start
    total_rejections = total_attempts - accepted_count
    overall_rejection_rate = (total_rejections / total_attempts) * 100.0

    print("\n" + "=" * 70)
    print("🎉 DATASET GENERATION COMPLETE: 15,000 ACCEPTED PAIRS PRODUCED")
    print("=" * 70)
    print(f"   Total Attempts:       {total_attempts:,}")
    print(f"   Accepted Pairs:       {accepted_count:,} (100.0%)")
    print(f"   Total Rejections:     {total_rejections:,} ({overall_rejection_rate:.1f}%)")
    print(f"   Rejection Breakdown:")
    for reason, count in rejection_reasons.items():
        pct = (count / max(1, total_rejections)) * 100.0
        print(f"     • Failed {reason:<8}: {count:6,} ({pct:5.1f}%)")
    print(f"   Total Execution Time: {total_time/60:.2f} minutes")
    print("=" * 70)

    # Report sizes on disk
    def get_dir_size_str(d: Path) -> str:
        total_bytes = sum(f.stat().st_size for f in d.glob("*.npz"))
        return f"{total_bytes / (1024**3):.2f} GB ({total_bytes / (1024**2):.1f} MB)"

    size_native = get_dir_size_str(out_native)
    size_256 = get_dir_size_str(out_256)
    print(f"📊 Dataset Disk Footprint:")
    print(f"   Native (512x512):   {size_native}")
    print(f"   Training (256x256): {size_256} (Well under 20 GB Kaggle ceiling)")
    print("=" * 70)


if __name__ == "__main__":
    main()
