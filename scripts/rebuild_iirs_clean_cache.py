#!/usr/bin/env python3
"""
rebuild_iirs_clean_cache.py — Regenerate IIRS proxies with defective detector repair.

Reads raw IIRS cube ch2_iir_nci_20230615T0132312064_d_img_n18.qub from Desktop
(or falls back to existing npz cache), detects bad pushbroom columns, applies
adjacent column interpolation and median destriping, and updates both
assets/real_cache/real_overlapping_pair.npz and data/real_cache/real_overlapping_pair.npz.

SIH26166 — Chandrayaan-2 Lunar Correspondence Pipeline
"""

from pathlib import Path
import sys
import numpy as np
import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.pds_loader import load_iirs, iirs_to_grey, iirs_proxy_variants, destripe_pushbroom_2d

DESKTOP_QUB = Path.home() / "Desktop/data/data/calibrated/20230615/ch2_iir_nci_20230615T0132312064_d_img_n18.qub"
DESKTOP_XML = DESKTOP_QUB.with_suffix(".xml")

TARGET_NPZ_ASSETS = REPO_ROOT / "assets/real_cache/real_overlapping_pair.npz"
TARGET_NPZ_DATA = REPO_ROOT / "data/real_cache/real_overlapping_pair.npz"
OUTPUTS_DIR = REPO_ROOT / "outputs"


def rebuild_cache():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Check if raw files exist on desktop
    if DESKTOP_QUB.exists() and DESKTOP_XML.exists():
        print(f"Loading raw IIRS cube from {DESKTOP_QUB}...")
        mmap, meta = load_iirs(DESKTOP_QUB, DESKTOP_XML)
        line_slice = slice(400 - 108, 400 + 108)
        sample_slice = slice(125 - 108, 125 + 108)

        print("Generating primary sub-2000nm band-averaged proxy with bad-column repair...")
        iirs_grey = iirs_to_grey(
            mmap,
            meta["bands"],
            line_slice,
            sample_slice,
            max_nm=2000.0,
            save_diagnostics=True,
            diag_dir=OUTPUTS_DIR,
            interleave=meta["interleave"],
        )

        print("Generating 3 proxy variants (1500nm, 3-band mean, PC1) with bad-column repair...")
        variants = iirs_proxy_variants(
            mmap,
            meta["bands"],
            line_slice,
            sample_slice,
            save_dir=OUTPUTS_DIR,
            interleave=meta["interleave"],
        )
        p_1500 = variants["single_1500nm"]
        p_3band = variants["mean_3band"]
        p_pc1 = variants["pc1"]

    else:
        print("Raw IIRS cube not found on Desktop; repairing existing cached arrays...")
        if not TARGET_NPZ_ASSETS.exists():
            raise FileNotFoundError(f"Neither raw cube nor {TARGET_NPZ_ASSETS} found.")
        old_data = np.load(TARGET_NPZ_ASSETS)

        def _clean_u8(img: np.ndarray) -> np.ndarray:
            f = destripe_pushbroom_2d(img.astype(np.float32), alpha=0.85, repair_bad_cols=True)
            p2, p98 = np.percentile(f, (2.0, 98.0))
            if p98 - p2 > 1e-6:
                return np.clip((f - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
            return np.zeros_like(img, dtype=np.uint8)

        iirs_grey = _clean_u8(old_data["iirs_grey"])
        p_1500 = _clean_u8(old_data.get("iirs_proxy_1500nm", iirs_grey))
        p_3band = _clean_u8(old_data.get("iirs_proxy_3band_mean", iirs_grey))
        p_pc1 = _clean_u8(old_data.get("iirs_proxy_pc1", iirs_grey))

    # Read base metadata and TMC crop from existing cache
    base_data = np.load(TARGET_NPZ_ASSETS)
    tmc_crop = base_data["tmc_crop"]

    save_dict = {
        "tmc_crop": tmc_crop,
        "iirs_grey": iirs_grey,
        "tmc_res": float(base_data["tmc_res"]),
        "iir_res": float(base_data["iir_res"]),
        "center_lat": float(base_data["center_lat"]),
        "center_lon": float(base_data["center_lon"]),
        "min_dist_m": float(base_data["min_dist_m"]),
        "tmc_scan": int(base_data["tmc_scan"]),
        "iir_scan": int(base_data["iir_scan"]),
        "solar_inc": float(base_data["solar_inc"]),
        "sun_az": float(base_data["sun_az"]),
        "sun_el": float(base_data["sun_el"]),
        "iirs_proxy_1500nm": p_1500,
        "iirs_proxy_3band_mean": p_3band,
        "iirs_proxy_pc1": p_pc1,
    }

    # Save to both target locations
    for target in [TARGET_NPZ_ASSETS, TARGET_NPZ_DATA]:
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **save_dict)
        print(f"Successfully saved clean cache to {target} ({target.stat().st_size / 1e6:.2f} MB)")

    # Diagnostic stats
    print("\nVerification of Cleaned Column Profiles:")
    for name, img in [
        ("band_avg", iirs_grey),
        ("1500nm", p_1500),
        ("3band", p_3band),
        ("pc1", p_pc1),
    ]:
        col_stds = np.std(np.mean(img, axis=0))
        col_max = np.max(np.mean(img, axis=0))
        print(f"  {name:10s} -> col_profile_std: {col_stds:.2f}, max_col_mean: {col_max:.1f}")


if __name__ == "__main__":
    rebuild_cache()
