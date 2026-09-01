#!/usr/bin/env python3
"""
TriNetra — Phase 1 Verification Script.

Loads a 2000×2000 centre patch from OHRC and TMC-2, extracts Band 34
(~1285 nm) from the IIRS cube, preprocesses all three, and plots them
side-by-side to verify the raw-data pipeline is functional.

Outputs:   output/phase1_verification.png
Console:   Shape, dtype, dynamic range, shadow/sat fractions.
"""
import sys
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import Chandrayaan2Loader
from src.module1_preprocessing_real import (
    OHRCRealPreprocessor,
    TMC2RealPreprocessor,
    IIRSRealPreprocessor,
)

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

# ─── File paths ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent / "data"

OHRC_IMG = ROOT / "ch2_ohr_ncp_20211023T0027462822_d_img_d18" / "data" / "calibrated" / "20211023" / "ch2_ohr_ncp_20211023T0027462822_d_img_d18.img"
OHRC_XML = OHRC_IMG.with_suffix(".xml")

TMC_IMG  = ROOT / "ch2_tmc_ncn_20221205T1633075527_d_img_d32" / "data" / "calibrated" / "20221205" / "ch2_tmc_ncn_20221205T1633075527_d_img_d32.img"
TMC_XML  = TMC_IMG.with_suffix(".xml")

IIRS_QUB = ROOT / "ch2_iir_nri_20231003T2152304115_d_img_d18" / "data" / "raw" / "20231003" / "ch2_iir_nri_20231003T2152304115_d_img_d18.qub"
IIRS_XML = IIRS_QUB.with_suffix(".xml")


def main() -> None:
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    # ── 1. Load OHRC ────────────────────────────────────────────────
    print("=" * 60)
    print("Loading OHRC (0.26 m/px, uint8, 93693 × 12000)...")
    ohrc = Chandrayaan2Loader(OHRC_IMG, OHRC_XML)
    print(f"  → {ohrc}")
    print(f"  → Metadata: sun_el={ohrc.meta.sun_elevation_deg}°, "
          f"sun_az={ohrc.meta.sun_azimuth_deg}°, area={ohrc.meta.area}")
    print(f"  → Corners: {ohrc.meta.corner_coords}")

    # Extract 2000×2000 centre patch
    ohrc_cx = ohrc.meta.lines // 2
    ohrc_cy = ohrc.meta.samples // 2
    ohrc_patch = ohrc.get_patch(ohrc_cx, ohrc_cy, size=2000)
    print(f"  → Raw patch: shape={ohrc_patch.shape}, dtype={ohrc_patch.dtype}, "
          f"min={ohrc_patch.min()}, max={ohrc_patch.max()}")

    # Preprocess
    ohrc_prep = OHRCRealPreprocessor()
    ohrc_result = ohrc_prep.preprocess(ohrc_patch)
    print(f"  → Preprocessed: range=[{ohrc_result.image.min():.3f}, {ohrc_result.image.max():.3f}], "
          f"shadow={ohrc_result.shadow_fraction:.1%}, sat={ohrc_result.saturation_fraction:.1%}")

    # ── 2. Load TMC-2 ──────────────────────────────────────────────
    print()
    print("Loading TMC-2 (5.97 m/px, uint16, 189886 × 4000)...")
    tmc = Chandrayaan2Loader(TMC_IMG, TMC_XML)
    print(f"  → {tmc}")
    print(f"  → Metadata: sun_el={tmc.meta.sun_elevation_deg}°, "
          f"sun_az={tmc.meta.sun_azimuth_deg}°, area={tmc.meta.area}")

    tmc_cx = tmc.meta.lines // 2
    tmc_cy = tmc.meta.samples // 2
    tmc_patch = tmc.get_patch(tmc_cx, tmc_cy, size=2000)
    print(f"  → Raw patch: shape={tmc_patch.shape}, dtype={tmc_patch.dtype}, "
          f"min={tmc_patch.min()}, max={tmc_patch.max()}")

    tmc_prep = TMC2RealPreprocessor()
    tmc_result = tmc_prep.preprocess(tmc_patch)
    print(f"  → Preprocessed: range=[{tmc_result.image.min():.3f}, {tmc_result.image.max():.3f}], "
          f"shadow={tmc_result.shadow_fraction:.1%}, sat={tmc_result.saturation_fraction:.1%}")

    # ── 3. Load IIRS ───────────────────────────────────────────────
    print()
    print("Loading IIRS (68.38 m/px, uint16, 256 × 2264 × 250)...")
    iirs = Chandrayaan2Loader(IIRS_QUB, IIRS_XML)
    print(f"  → {iirs}")
    print(f"  → Metadata: sun_el={iirs.meta.sun_elevation_deg}°, "
          f"sun_az={iirs.meta.sun_azimuth_deg}°, area={iirs.meta.area}")
    print(f"  → Band centers available: {len(iirs.meta.band_centers_nm) if iirs.meta.band_centers_nm else 0} bands")

    # Extract Band 34 (~1285 nm)
    band_idx, iirs_band34 = iirs.get_band_by_wavelength(1285.0)
    print(f"  → Band {band_idx + 1} (~1285 nm): shape={iirs_band34.shape}, "
          f"dtype={iirs_band34.dtype}, min={iirs_band34.min()}, max={iirs_band34.max()}")

    iirs_prep = IIRSRealPreprocessor(proxy_band_index=band_idx)
    iirs_result = iirs_prep.preprocess(band_2d=iirs_band34)
    print(f"  → Preprocessed: range=[{iirs_result.image.min():.3f}, {iirs_result.image.max():.3f}], "
          f"shadow={iirs_result.shadow_fraction:.1%}, sat={iirs_result.saturation_fraction:.1%}")

    # ── 4. Plot side-by-side ───────────────────────────────────────
    print()
    print("Generating verification plot...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    axes[0].imshow(ohrc_result.image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(
        f"OHRC (0.26 m/px)\n"
        f"Sun El: {ohrc.meta.sun_elevation_deg:.1f}°  |  "
        f"Shadow: {ohrc_result.shadow_fraction:.0%}",
        fontsize=11, fontweight="bold",
    )
    axes[0].set_xlabel(f"Patch: {ohrc_patch.shape[1]}×{ohrc_patch.shape[0]} px, "
                       f"Centre: ({ohrc_cx}, {ohrc_cy})")

    axes[1].imshow(tmc_result.image, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(
        f"TMC-2 (5.97 m/px)\n"
        f"Sun El: {tmc.meta.sun_elevation_deg:.1f}°  |  "
        f"Shadow: {tmc_result.shadow_fraction:.0%}",
        fontsize=11, fontweight="bold",
    )
    axes[1].set_xlabel(f"Patch: {tmc_patch.shape[1]}×{tmc_patch.shape[0]} px, "
                       f"Centre: ({tmc_cx}, {tmc_cy})")

    axes[2].imshow(iirs_result.image, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(
        f"IIRS Band 34 (~1285 nm, 68.38 m/px)\n"
        f"Sun El: {iirs.meta.sun_elevation_deg:.1f}°  |  "
        f"Shadow: {iirs_result.shadow_fraction:.0%}",
        fontsize=11, fontweight="bold",
    )
    axes[2].set_xlabel(f"Full band: {iirs_band34.shape[1]}×{iirs_band34.shape[0]} px")

    for ax in axes:
        ax.tick_params(labelsize=8)

    fig.suptitle(
        "TriNetra Phase 1 — Raw Data Pipeline Verification\n"
        "Chandrayaan-2: OHRC × TMC-2 × IIRS",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    save_path = out_dir / "phase1_verification.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved: {save_path}")
    plt.close(fig)

    # ── 5. Summary table ──────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"{'Instrument':<12} {'Shape':<20} {'GSD (m/px)':<12} "
          f"{'Sun El (°)':<12} {'Dyn Range':<12} {'Shadow %':<10}")
    print("-" * 72)
    for name, loader, result in [
        ("OHRC",  ohrc, ohrc_result),
        ("TMC-2", tmc,  tmc_result),
        ("IIRS",  iirs, iirs_result),
    ]:
        sh = f"{loader.shape}"
        print(f"{name:<12} {sh:<20} {loader.meta.pixel_resolution_m:<12.2f} "
              f"{loader.meta.sun_elevation_deg:<12.1f} "
              f"{result.dynamic_range:<12.1f} "
              f"{result.shadow_fraction:<10.1%}")
    print("=" * 72)
    print("\n🎯 Phase 1 data pipeline is OPERATIONAL.")


if __name__ == "__main__":
    main()
