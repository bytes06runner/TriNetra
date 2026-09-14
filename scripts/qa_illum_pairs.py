#!/usr/bin/env python3
"""
qa_illum_pairs.py — Visual QA Suite for Synthetic Illumination Pairs.

Produces:
1. Contact sheet of 20 randomly sampled pairs (image_a & image_b side-by-side with metadata)
2. Elevation shadow-sweep panel (2°, 5°, 10°, 20° elevation demonstrating ~28.6x shadow factor)
3. Azimuth difference distribution histogram (ASCII text bar chart and publication PNG)
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from osgeo import gdal

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.illum_render import render_lunar_tile, get_default_device


def generate_contact_sheet(pairs_dir: Path, out_path: Path, num_samples: int = 20, seed: int = 42):
    """Render a 20-pair contact sheet (4 rows x 5 pairs, image_a and image_b side-by-side)."""
    npz_files = sorted(list(pairs_dir.glob("*.npz")))
    if not npz_files:
        print(f"❌ No .npz files found in {pairs_dir}")
        return

    rng = np.random.default_rng(seed)
    # Ensure both lambert and ls are sampled
    lambert_files = [f for f in npz_files if int(f.stem.split("_")[1]) < len(npz_files) // 2]
    ls_files = [f for f in npz_files if int(f.stem.split("_")[1]) >= len(npz_files) // 2]

    sample_lambert = rng.choice(lambert_files, size=min(num_samples // 2, len(lambert_files)), replace=False)
    sample_ls = rng.choice(ls_files, size=min(num_samples - len(sample_lambert), len(ls_files)), replace=False) if ls_files else []
    selected = list(sample_lambert) + list(sample_ls)
    rng.shuffle(selected)

    print(f"📸 Generating contact sheet from {len(selected)} sampled pairs...")

    fig, axes = plt.subplots(5, 4, figsize=(20, 24), dpi=150)
    fig.suptitle("TriNetra: Synthetic Illumination Pairs Contact Sheet (20 Samples)\n[Anchor A vs Warped & Scaled Observation B]", fontsize=16, fontweight="bold", y=0.99)

    for idx, (ax, p) in enumerate(zip(axes.flatten(), selected)):
        data = np.load(p)
        img_a = data["image_a"]
        img_b = data["image_b"]
        az_a, alt_a = float(data["az_a"]), float(data["alt_a"])
        az_b, alt_b = float(data["az_b"]), float(data["alt_b"])
        mode = str(data["render_mode"]).upper()

        delta_az = min(abs(az_a - az_b), 360.0 - abs(az_a - az_b))
        delta_alt = abs(alt_a - alt_b)

        # Side-by-side composite
        h, w = img_a.shape[:2]
        composite = np.zeros((h, w * 2 + 10), dtype=np.uint8)
        composite[:, :w] = img_a
        composite[:, w:w+10] = 128  # Divider line
        composite[:, w+10:] = img_b

        ax.imshow(composite, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Pair #{p.stem.split('_')[1]} [{mode}]\nΔAz: {delta_az:.1f}° | ΔAlt: {delta_alt:.1f}°\nA: az{az_a:.0f}°/el{alt_a:.0f}° → B: az{az_b:.0f}°/el{alt_b:.1f}°",
                     fontsize=9, fontweight="semibold", pad=4)
        ax.axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved contact sheet to: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


def generate_elevation_sweep(dem_path: Path, out_path: Path):
    """
    Render a prominent relief tile across 2°, 5°, 10°, and 20° elevation
    to demonstrate the cast shadow lengthening effect (~28.6x at 2°).
    Renders two rows:
      - Row 1: Azimuth 242.98° (Real Flight Azimuth, illuminating facing slope)
      - Row 2: Azimuth 90.00° (East Azimuth, demonstrating ridge-shadow geometry)
    """
    print("🌅 Generating elevation shadow-sweep diagnostic...")
    ds = gdal.Open(str(dem_path))
    tile_size = 512
    row = 1436 - tile_size // 2
    col = 642 - tile_size // 2
    dem_tile = ds.ReadAsArray(col, row, tile_size, tile_size).astype(np.float32)
    ds = None

    elevations = [2.0, 5.0, 10.0, 20.0]
    device = get_default_device()

    fig, axes = plt.subplots(2, 4, figsize=(22, 11), dpi=150)
    fig.suptitle(f"TriNetra: Topographic Cast-Shadow Lengthening Diagnostic Sweep\n"
                 f"Anchor DEM Tile (-89.72°S, 223.13°E) | Relief: {dem_tile.max() - dem_tile.min():.1f} m",
                 fontsize=14, fontweight="bold", y=0.99)

    # Row 1: Flight Azimuth 242.98° (Facing Slope)
    for col_idx, alt in enumerate(elevations):
        ax = axes[0, col_idx]
        img, mask = render_lunar_tile(dem_tile, 242.98, alt, mode="ls", apply_shadows=True, ambient_dn=51, device=device)
        cov = mask.mean() * 100.0
        mult = 1.0 / np.tan(np.radians(alt))
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"[Flight Az: 243°] El: {alt:.1f}° (Factor {mult:.1f}×)\nShadow: {cov:.1f}% | Mean: {img.mean():.1f} DN | Std: {img.std():.1f}",
                     fontsize=10, fontweight="semibold", pad=4)
        ax.axis("off")

    # Row 2: Opposite Azimuth 90.00° (Behind Ridge)
    for col_idx, alt in enumerate(elevations):
        ax = axes[1, col_idx]
        img, mask = render_lunar_tile(dem_tile, 90.0, alt, mode="ls", apply_shadows=True, ambient_dn=51, device=device)
        cov = mask.mean() * 100.0
        mult = 1.0 / np.tan(np.radians(alt))
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"[East Az: 90°] El: {alt:.1f}° (Factor {mult:.1f}×)\nShadow: {cov:.1f}% | Mean: {img.mean():.1f} DN | Std: {img.std():.1f}",
                     fontsize=10, fontweight="semibold", pad=4)
        ax.axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved elevation sweep to: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


def plot_valid_mask_distribution(pairs_dir: Path, out_path: Path):
    """Plot and verify valid_mask coverage histogram across all accepted pairs."""
    npz_files = list(pairs_dir.glob("*.npz"))
    if not npz_files:
        print(f"❌ No .npz files found in {pairs_dir}")
        return

    print(f"📊 Analyzing valid_mask coverage across {len(npz_files):,} pairs...")
    coverages = []
    for p in npz_files:
        d = np.load(p)
        m = d["valid_mask"]
        cov = float(np.mean(m) * 100.0)
        coverages.append(cov)

    coverages = np.array(coverages)
    min_cov = float(np.min(coverages))
    mean_cov = float(np.mean(coverages))
    median_cov = float(np.median(coverages))

    print("\n" + "=" * 70)
    print("📈 Valid Mask Coverage Summary (Accepted Dataset)")
    print("=" * 70)
    print(f"  Total Pairs Analyzed:  {len(coverages):,}")
    print(f"  Minimum Coverage:      {min_cov:.2f}% (Target: >= 40.0%)")
    print(f"  Mean Coverage:         {mean_cov:.2f}%")
    print(f"  Median Coverage:       {median_cov:.2f}%")
    print(f"  All Pairs >= 40.0%:    {bool(min_cov >= 40.0)}")
    print("=" * 70 + "\n")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    n, bins, patches = ax.hist(coverages, bins=30, range=(35, 100), color="#38A169", edgecolor="#1A202C", alpha=0.85)
    ax.axvline(40.0, color="#E53E3E", linestyle="--", linewidth=2.5, label="40.0% Acceptance Threshold")
    ax.set_title(f"Valid Mask Coverage Distribution (N = {len(coverages):,})\n"
                 f"Min: {min_cov:.1f}% | Mean: {mean_cov:.1f}% | 100% Gated Above 40% Floor",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Valid Corresponding Mask Coverage (%)", fontsize=11)
    ax.set_ylabel("Number of Image Pairs", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=11)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved valid_mask distribution plot to: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


def plot_azimuth_distribution(pairs_dir: Path, out_path: Path):
    """Plot and print ASCII histogram of azimuth differences across the full dataset."""
    npz_files = list(pairs_dir.glob("*.npz"))
    if not npz_files:
        print(f"❌ No .npz files found in {pairs_dir}")
        return

    print(f"📊 Analyzing illumination distribution across {len(npz_files):,} pairs...")
    delta_azs = []
    delta_alts = []
    modes = []

    for p in npz_files:
        d = np.load(p)
        az_a = float(d["az_a"])
        az_b = float(d["az_b"])
        alt_a = float(d["alt_a"])
        alt_b = float(d["alt_b"])
        modes.append(str(d["render_mode"]))

        d_az = min(abs(az_a - az_b), 360.0 - abs(az_a - az_b))
        delta_azs.append(d_az)
        delta_alts.append(abs(alt_a - alt_b))

    delta_azs = np.array(delta_azs)
    delta_alts = np.array(delta_alts)

    # 1. ASCII Histogram
    print("\n" + "="*70)
    print("📈 ASCII Histogram: Azimuth Difference (ΔAzimuth) Distribution")
    print("="*70)
    bins = [0, 30, 60, 90, 120, 150, 180]
    counts, _ = np.histogram(delta_azs, bins=bins)
    max_count = max(counts) if len(counts) > 0 else 1
    for i in range(len(counts)):
        bar_len = int(round((counts[i] / max_count) * 40))
        bar = "█" * bar_len
        pct = (counts[i] / len(delta_azs)) * 100.0
        print(f"  {bins[i]:3d}° - {bins[i+1]:3d}° : {bar:<40} {counts[i]:5d} ({pct:5.1f}%)")
    print("="*70)
    print(f"  Total Pairs: {len(delta_azs):,} | Mean ΔAz: {delta_azs.mean():.1f}° | Std: {delta_azs.std():.1f}°")
    print(f"  Lambertian: {modes.count('lambert'):,} pairs | Lommel-Seeliger: {modes.count('ls'):,} pairs")
    print("="*70 + "\n")

    # 2. Publication Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    
    ax1.hist(delta_azs, bins=36, range=(0, 180), color="#DE7356", edgecolor="#1A202C", alpha=0.85)
    ax1.set_title(f"Azimuth Difference Distribution (N = {len(delta_azs):,})\nUniform Coverage Across 0°–180° Range", fontsize=11, fontweight="bold")
    ax1.set_xlabel("ΔAzimuth (°)", fontsize=10)
    ax1.set_ylabel("Number of Image Pairs", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.hist(delta_alts, bins=25, range=(0, 25), color="#3182CE", edgecolor="#1A202C", alpha=0.85)
    ax2.set_title(f"Elevation Difference Distribution (N = {len(delta_alts):,})\nSpanning Grazing (2°) to High Sun (30°)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("ΔElevation (°)", fontsize=10)
    ax2.set_ylabel("Number of Image Pairs", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved azimuth distribution plot to: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Visual QA Suite for Synthetic Illumination Pairs")
    parser.add_argument("--pairs-dir", type=str, default="data/synthetic_pairs/train_256",
                        help="Directory containing generated .npz pairs (default: data/synthetic_pairs/train_256)")
    parser.add_argument("--dem-path", type=str, default="data/dem/Site04_final_adj_5mpp_surf.tif",
                        help="Path to DEM file for elevation sweep")
    parser.add_argument("--out-dir", type=str, default="outputs/qa",
                        help="Output directory for QA artifacts")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs_dir = Path(args.pairs_dir)
    dem_path = Path(args.dem_path)

    # 1. Contact sheet of 20 pairs
    contact_path = out_dir / "contact_sheet_20pairs.png"
    generate_contact_sheet(pairs_dir, contact_path, num_samples=20)

    # 2. Elevation sweep on anchor tile
    sweep_path = out_dir / "elevation_sweep_shadows.png"
    generate_elevation_sweep(dem_path, sweep_path)

    # 3. Azimuth distribution histogram
    hist_path = out_dir / "azimuth_distribution.png"
    plot_azimuth_distribution(pairs_dir, hist_path)

    # 4. Valid mask coverage distribution
    mask_hist_path = out_dir / "valid_mask_distribution.png"
    plot_valid_mask_distribution(pairs_dir, mask_hist_path)

    print("\n🎉 Visual QA Suite execution complete! All deliverables saved to outputs/qa/")


if __name__ == "__main__":
    main()
