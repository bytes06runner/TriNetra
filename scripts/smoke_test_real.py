"""
Smoke Test for Real Chandrayaan-2 Data Loading & Common Footprint Alignment.

Loads overlapping TMC-2 (visible optical, 4.96 m/px) and IIRS (hyperspectral,
91.75 m/px) products from ~/Desktop/data/ using zero-copy memory mapping,
finds their shared geographic ground coverage, extracts matching crops,
synthesizes the sub-2000nm IIRS grey proxy, downsamples TMC-2 by 18.5×,
saves three verification PNGs, and prints image statistics.

SIH26166 — Chandrayaan-2 Multi-modal Image Correspondence Pipeline
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from src.pds_loader import load_tmc2, load_iirs, iirs_to_grey, crop
from src.geo_align import find_common_region, compute_centered_crop_slices
from src.module2_matching.scale_handler import compute_scale_ratio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("smoke_test_real")


def print_crop_stats(name: str, arr: np.ndarray) -> None:
    """Print shape, dtype, min, max, mean, and percentage of zero pixels."""
    total_px = arr.size
    zero_px = int(np.sum(arr == 0))
    pct_zero = (zero_px / total_px) * 100.0 if total_px > 0 else 0.0

    print(f"\n--- {name} ---")
    print(f"  Shape:       {arr.shape}")
    print(f"  Dtype:       {arr.dtype}")
    print(f"  Min value:   {float(np.nanmin(arr)):.4f}")
    print(f"  Max value:   {float(np.nanmax(arr)):.4f}")
    print(f"  Mean value:  {float(np.nanmean(arr)):.4f}")
    print(f"  Zero pixels: {zero_px:,} / {total_px:,} ({pct_zero:.2f}%)")


def run_smoke_test(
    data_dir: Path,
    output_dir: Path,
) -> None:
    """Run full real-data smoke test."""
    # Define exact product paths relative to data_dir
    tmc_img = data_dir / "data/calibrated/20230528/ch2_tmc_ncn_20230528T1712292966_d_img_d32.img"
    tmc_xml = data_dir / "data/calibrated/20230528/ch2_tmc_ncn_20230528T1712292966_d_img_d32.xml"
    tmc_csv = data_dir / "geometry/calibrated/20230528/ch2_tmc_ncn_20230528T1712292966_g_grd_d32.csv"

    iir_qub = data_dir / "data/calibrated/20230615/ch2_iir_nci_20230615T0132312064_d_img_n18.qub"
    iir_xml = data_dir / "data/calibrated/20230615/ch2_iir_nci_20230615T0132312064_d_img_n18.xml"
    iir_csv = data_dir / "geometry/calibrated/20230615/ch2_iir_nci_20230615T0132312064_g_grd_n18.csv"

    logger.info("Verifying file presence...")
    for p in [tmc_img, tmc_xml, tmc_csv, iir_qub, iir_xml, iir_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Required product file missing: {p}")

    # Step 1: Memory-map both products and dynamically parse XML labels
    logger.info("Memory-mapping TMC-2 product...")
    tmc_mm, tmc_meta = load_tmc2(tmc_img, tmc_xml)

    logger.info("Memory-mapping IIRS hyperspectral cube...")
    iir_mm, iir_meta = load_iirs(iir_qub, iir_xml)

    tmc_res = float(tmc_meta["pixel_resolution"])
    iir_res = float(iir_meta["pixel_resolution"])
    scale_ratio = compute_scale_ratio(tmc_res, iir_res)
    downsample_factor = 1.0 / scale_ratio

    print("\n==================================================================")
    print("CHANDRAYAAN-2 REAL PRODUCT SPECIFICATIONS")
    print("==================================================================")
    print(f"TMC-2: Shape={tmc_mm.shape}, Dtype={tmc_mm.dtype}, Resolution={tmc_res:.2f} m/px")
    print(f"       Sun Azimuth={tmc_meta['sun_azimuth']:.2f}°, Solar Incidence={tmc_meta['solar_incidence']:.2f}°")
    print(f"IIRS:  Shape={tmc_meta['elements'] if 'elements' in tmc_meta else tmc_mm.shape} -> IIRS: {iir_mm.shape}, Dtype={iir_mm.dtype}, Resolution={iir_res:.2f} m/px")
    print(f"       Sun Azimuth={iir_meta['sun_azimuth']:.2f}°, Solar Incidence={iir_meta['solar_incidence']:.2f}°")
    print(f"Computed Scale Ratio: {iir_res:.2f} m / {tmc_res:.2f} m = {downsample_factor:.2f}×")
    print("==================================================================")

    # Step 2: Use find_common_region to locate overlapping geographic ground coverage
    logger.info("Analyzing geometry grid CSVs to locate shared ground region...")
    common = find_common_region(tmc_csv, iir_csv, distance_threshold_km=15.0)

    center_lat = common["center_lat"]
    center_lon = common["center_lon"]
    tmc_center_scan = common["a"]["center_scan"]
    tmc_center_pixel = common["a"]["center_pixel"]
    iir_center_scan = common["b"]["center_scan"]
    iir_center_pixel = common["b"]["center_pixel"]

    print("\nCOMMON GEOGRAPHIC OVERLAP REGION:")
    print(f"  Center coordinates: Lat = {center_lat:.4f}°, Lon = {center_lon:.4f}° (Near Lunar North Pole)")
    print(f"  Shared Lat range:   {common['shared_lat_range'][0]:.4f}° to {common['shared_lat_range'][1]:.4f}°")
    print(f"  Closest approach:   {common['min_distance_km']*1000.0:.1f} meters on ground")
    print(f"  TMC-2 Grid Center:  Scan {tmc_center_scan}, Pixel {tmc_center_pixel}")
    print(f"  IIRS Grid Center:   Scan {iir_center_scan}, Pixel {iir_center_pixel}")

    # Step 3: Compute crop windows covering identical physical ground terrain (~19.8 km × 19.8 km)
    # TMC-2: 4000 × 4000 pixels @ 4.96 m/px = 19.84 km × 19.84 km
    tmc_crop_h, tmc_crop_w = 4000, 4000
    # IIRS: 216 × 216 pixels @ 91.75 m/px = 19.82 km × 19.82 km
    iir_crop_h, iir_crop_w = 216, 216

    # Swath midpoints: TMC-2 has 4000 samples (center 2000), IIRS has 250 samples (center 125)
    l_slice_tmc, s_slice_tmc = compute_centered_crop_slices(
        center_scan=tmc_center_scan,
        center_pixel=2000,
        crop_lines=tmc_crop_h,
        crop_samples=tmc_crop_w,
        total_lines=tmc_mm.shape[0],
        total_samples=tmc_mm.shape[1],
    )

    l_slice_iir, s_slice_iir = compute_centered_crop_slices(
        center_scan=iir_center_scan,
        center_pixel=125,
        crop_lines=iir_crop_h,
        crop_samples=iir_crop_w,
        total_lines=iir_mm.shape[1],
        total_samples=iir_mm.shape[2],
    )

    logger.info(
        "TMC-2 Crop Window: Lines [%d, %d), Samples [%d, %d) (Size: %d × %d)",
        l_slice_tmc.start, l_slice_tmc.stop, s_slice_tmc.start, s_slice_tmc.stop,
        l_slice_tmc.stop - l_slice_tmc.start, s_slice_tmc.stop - s_slice_tmc.start,
    )
    logger.info(
        "IIRS Crop Window: Lines [%d, %d), Samples [%d, %d) (Size: %d × %d)",
        l_slice_iir.start, l_slice_iir.stop, s_slice_iir.start, s_slice_iir.stop,
        l_slice_iir.stop - l_slice_iir.start, s_slice_iir.stop - s_slice_iir.start,
    )

    # Step 4: Crop TMC-2 into memory
    logger.info("Extracting in-memory TMC-2 crop...")
    tmc_crop = crop(
        tmc_mm,
        l_slice_tmc.start, l_slice_tmc.stop,
        s_slice_tmc.start, s_slice_tmc.stop,
    )

    # Step 5: Build IIRS grey proxy from bands below 2000 nm
    logger.info("Building IIRS visible proxy from sub-2000nm reflectance bands...")
    iirs_grey = iirs_to_grey(
        iir_mm,
        iir_meta["bands"],
        l_slice_iir,
        s_slice_iir,
        max_nm=2000.0,
    )

    # Step 6: Downsample TMC-2 crop by 18.5× to match IIRS ground sample distance
    logger.info("Downsampling TMC-2 crop by %.2f× using anti-aliased INTER_AREA...", downsample_factor)
    target_w = iirs_grey.shape[1]
    target_h = iirs_grey.shape[0]
    tmc_down = cv2.resize(
        tmc_crop.astype(np.float32),
        (target_w, target_h),
        interpolation=cv2.INTER_AREA,
    )

    # Step 7: Print required diagnostic statistics
    print_crop_stats("TMC-2 Full-Resolution Crop (Raw uint16 DN)", tmc_crop)
    print_crop_stats("TMC-2 Downsampled by 18.5× (float32)", tmc_down)
    print_crop_stats("IIRS Grey Proxy (sub-2000nm, uint8)", iirs_grey)

    # Step 8: Save the three PNGs
    output_dir.mkdir(parents=True, exist_ok=True)
    p_tmc_full = output_dir / "tmc2_crop_fullres.png"
    p_tmc_down = output_dir / "tmc2_crop_downsampled.png"
    p_iirs_grey = output_dir / "iirs_grey_proxy.png"

    # Normalize TMC-2 crops for 8-bit visual rendering
    # Use 2-98 percentile stretch so subtle topography is crisp
    p2_t, p98_t = np.percentile(tmc_crop, (2.0, 98.0))
    tmc_crop_vis = np.clip(
        (tmc_crop.astype(np.float32) - p2_t) / max(1e-6, p98_t - p2_t) * 255.0,
        0, 255,
    ).astype(np.uint8)

    p2_d, p98_d = np.percentile(tmc_down, (2.0, 98.0))
    tmc_down_vis = np.clip(
        (tmc_down - p2_d) / max(1e-6, p98_d - p2_d) * 255.0,
        0, 255,
    ).astype(np.uint8)

    logger.info("Saving output PNGs...")
    cv2.imwrite(str(p_tmc_full), tmc_crop_vis)
    cv2.imwrite(str(p_tmc_down), tmc_down_vis)
    cv2.imwrite(str(p_iirs_grey), iirs_grey)

    print("\n==================================================================")
    print("OUTPUT PNGs GENERATED SUCCESSFULLY:")
    print(f"  1. TMC-2 Full-Resolution Crop:  {p_tmc_full.resolve()} ({p_tmc_full.stat().st_size:,} bytes)")
    print(f"  2. TMC-2 Downsampled (18.5×):   {p_tmc_down.resolve()} ({p_tmc_down.stat().st_size:,} bytes)")
    print(f"  3. IIRS Sub-2000nm Grey Proxy:  {p_iirs_grey.resolve()} ({p_iirs_grey.stat().st_size:,} bytes)")
    print("==================================================================")
    print("\nNOTE: Smoke test completed without running feature matching, per instructions.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for real Chandrayaan-2 TMC-2 and IIRS loading.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "Desktop/data",
        help="Root directory containing real data (default: ~/Desktop/data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs",
        help="Output directory for generated verification PNGs (default: ./outputs)",
    )
    args = parser.parse_args()
    run_smoke_test(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
