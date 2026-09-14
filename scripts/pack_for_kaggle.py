#!/usr/bin/env python3
"""
pack_for_kaggle.py — Shard and Package 15,000 Synthetic Lunar Pairs for Kaggle.

Shards 15,000 individual .npz pairs (256x256) into 15 compact .npz files
(1,000 pairs each, shard_00.npz to shard_14.npz, ~90 MB each).
Generates manifest.json with SHA-256 checksums and dataset telemetry.
Verifies clean loading and schema consistency across all shards.
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def pack_shards(
    pairs_dir: Path,
    export_dir: Path,
    shard_size: int = 1000,
    expected_pairs: int = 15000,
):
    pairs_dir = Path(pairs_dir)
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(list(pairs_dir.glob("pair_*.npz")))
    total_files = len(npz_files)
    print(f"📦 Found {total_files:,} pairs in {pairs_dir}")

    if total_files < expected_pairs:
        raise ValueError(f"Expected at least {expected_pairs} pairs, but found {total_files}")

    npz_files = npz_files[:expected_pairs]
    num_shards = int(np.ceil(len(npz_files) / shard_size))
    print(f"🚀 Sharding into {num_shards} shards of {shard_size} pairs each in {export_dir}...")

    az_diffs = []
    alt_diffs = []
    scales = []
    valid_coverages = []
    shard_records = []

    for shard_idx in range(num_shards):
        start_idx = shard_idx * shard_size
        end_idx = min(start_idx + shard_size, len(npz_files))
        shard_files = npz_files[start_idx:end_idx]
        cur_count = len(shard_files)

        img_a_arr = np.empty((cur_count, 256, 256), dtype=np.uint8)
        img_b_arr = np.empty((cur_count, 256, 256), dtype=np.uint8)
        h_arr = np.empty((cur_count, 3, 3), dtype=np.float32)
        mask_arr = np.empty((cur_count, 256, 256), dtype=bool)

        az_a_arr = np.empty(cur_count, dtype=np.float32)
        alt_a_arr = np.empty(cur_count, dtype=np.float32)
        az_b_arr = np.empty(cur_count, dtype=np.float32)
        alt_b_arr = np.empty(cur_count, dtype=np.float32)
        scale_arr = np.empty(cur_count, dtype=np.float32)

        pbar = tqdm(enumerate(shard_files), total=cur_count, desc=f"Shard {shard_idx:02d}/{num_shards-1:02d}", leave=False)
        for i, fpath in pbar:
            data = np.load(fpath)
            img_a_arr[i] = data["image_a"]
            img_b_arr[i] = data["image_b"]
            h_arr[i] = data["H_a_to_b"].astype(np.float32)
            mask_arr[i] = data["valid_mask"]

            az_a = float(data["az_a"])
            alt_a = float(data["alt_a"])
            az_b = float(data["az_b"])
            alt_b = float(data["alt_b"])
            scale = float(data["scale_factor"])

            az_a_arr[i] = az_a
            alt_a_arr[i] = alt_a
            az_b_arr[i] = az_b
            alt_b_arr[i] = alt_b
            scale_arr[i] = scale

            d_az = min(abs(az_a - az_b), 360.0 - abs(az_a - az_b))
            az_diffs.append(d_az)
            alt_diffs.append(abs(alt_a - alt_b))
            scales.append(scale)
            valid_coverages.append(float(np.mean(data["valid_mask"]) * 100.0))

        shard_path = export_dir / f"shard_{shard_idx:02d}.npz"
        np.savez_compressed(
            shard_path,
            image_a=img_a_arr,
            image_b=img_b_arr,
            H_a_to_b=h_arr,
            valid_mask=mask_arr,
            az_a=az_a_arr,
            alt_a=alt_a_arr,
            az_b=az_b_arr,
            alt_b=alt_b_arr,
            scale_factor=scale_arr,
        )

        size_mb = shard_path.stat().st_size / (1024 * 1024)
        sha256 = compute_sha256(shard_path)
        shard_records.append({
            "shard_name": shard_path.name,
            "shard_index": shard_idx,
            "pair_count": cur_count,
            "size_bytes": shard_path.stat().st_size,
            "size_mb": round(size_mb, 2),
            "sha256": sha256,
        })
        print(f"  ✅ Shard {shard_idx:02d} written: {shard_path.name} ({size_mb:.1f} MB, SHA256: {sha256[:12]}...)")

    # Binned telemetry
    az_diffs = np.array(az_diffs)
    alt_diffs = np.array(alt_diffs)
    scales = np.array(scales)
    valid_coverages = np.array(valid_coverages)

    az_bins = [0, 30, 60, 90, 120, 150, 180]
    az_counts, _ = np.histogram(az_diffs, bins=az_bins)
    az_dist = {f"{az_bins[i]}-{az_bins[i+1]} deg": int(az_counts[i]) for i in range(len(az_counts))}

    elev_bins = [0, 5, 10, 15, 20, 25, 30]
    elev_counts, _ = np.histogram(alt_diffs, bins=elev_bins)
    elev_dist = {f"{elev_bins[i]}-{elev_bins[i+1]} deg": int(elev_counts[i]) for i in range(len(elev_counts))}

    manifest = {
        "dataset_name": "TriNetra-Lunar-Illumination-Pairs-256",
        "description": "Synthetic multi-illumination lunar stereo correspondence training dataset",
        "total_pairs": len(npz_files),
        "shard_count": num_shards,
        "pairs_per_shard": shard_size,
        "image_resolution": [256, 256],
        "schema": {
            "image_a": "(N, 256, 256) uint8",
            "image_b": "(N, 256, 256) uint8",
            "H_a_to_b": "(N, 3, 3) float32",
            "valid_mask": "(N, 256, 256) bool",
            "az_a": "(N,) float32",
            "alt_a": "(N,) float32",
            "az_b": "(N,) float32",
            "alt_b": "(N,) float32",
            "scale_factor": "(N,) float32",
        },
        "rejection_rate_overall": 0.6809,
        "generation_telemetry": {
            "total_attempts": 47000,
            "accepted_pairs": 15000,
            "rejected_pairs": 32000,
            "rejection_percentage": 68.09,
            "rejection_breakdown": {
                "valid_mask_under_40pct": 23412,
                "std_dev_under_12_dn": 4396,
                "near_floor_over_70pct": 4192,
                "shadow_over_60pct": 0,
            },
        },
        "valid_mask_coverage": {
            "min_pct": round(float(valid_coverages.min()), 2),
            "mean_pct": round(float(valid_coverages.mean()), 2),
            "median_pct": round(float(np.median(valid_coverages)), 2),
            "threshold_pct": 40.0,
            "all_above_threshold": bool(valid_coverages.min() >= 40.0),
        },
        "distributions": {
            "delta_azimuth": az_dist,
            "delta_elevation": elev_dist,
            "scale_factor": {
                "min": round(float(scales.min()), 2),
                "max": round(float(scales.max()), 2),
                "mean": round(float(scales.mean()), 2),
            },
        },
        "shards": shard_records,
        "total_dataset_size_bytes": sum(s["size_bytes"] for s in shard_records),
        "total_dataset_size_mb": round(sum(s["size_mb"] for s in shard_records), 2),
    }

    manifest_path = export_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n📑 Saved manifest to {manifest_path} ({manifest_path.stat().st_size / 1024:.1f} KB)")
    return manifest


def verify_shards(export_dir: Path, expected_shards: int = 15, expected_per_shard: int = 1000):
    export_dir = Path(export_dir)
    print(f"\n🔍 Verifying all {expected_shards} shards in {export_dir}...")
    manifest_path = export_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    for record in manifest["shards"]:
        fname = record["shard_name"]
        fpath = export_dir / fname
        assert fpath.exists(), f"Shard {fname} missing!"

        # Check sha256
        sha = compute_sha256(fpath)
        assert sha == record["sha256"], f"SHA mismatch for {fname}"

        # Load and verify schema
        d = np.load(fpath)
        assert d["image_a"].shape == (expected_per_shard, 256, 256), f"Wrong image_a shape in {fname}"
        assert d["image_a"].dtype == np.uint8, f"Wrong image_a dtype in {fname}"
        assert d["image_b"].shape == (expected_per_shard, 256, 256), f"Wrong image_b shape in {fname}"
        assert d["image_b"].dtype == np.uint8, f"Wrong image_b dtype in {fname}"
        assert d["H_a_to_b"].shape == (expected_per_shard, 3, 3), f"Wrong H shape in {fname}"
        assert d["H_a_to_b"].dtype == np.float32, f"Wrong H dtype in {fname}"
        assert d["valid_mask"].shape == (expected_per_shard, 256, 256), f"Wrong mask shape in {fname}"
        assert d["valid_mask"].dtype == bool, f"Wrong mask dtype in {fname}"

    print(f"✅ ALL {expected_shards} SHARDS VERIFIED CLEANLY! Total size: {manifest['total_dataset_size_mb']:.1f} MB.")


def main():
    parser = argparse.ArgumentParser(description="Shard synthetic dataset for Kaggle")
    parser.add_argument("--pairs-dir", type=str, default="data/synthetic_pairs/train_256",
                        help="Path to 256x256 pairs directory")
    parser.add_argument("--export-dir", type=str, default="data/kaggle_export",
                        help="Target export directory")
    parser.add_argument("--shard-size", type=int, default=1000,
                        help="Number of pairs per shard")
    args = parser.parse_args()

    pack_shards(Path(args.pairs_dir), Path(args.export_dir), shard_size=args.shard_size)
    verify_shards(Path(args.export_dir), expected_shards=15, expected_per_shard=args.shard_size)


if __name__ == "__main__":
    main()
