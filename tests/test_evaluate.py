"""Unit tests for trinetra.evaluate module."""

import sys
import json
import numpy as np
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from trinetra.evaluate import (
    compute_residuals,
    decompose_similarity,
    compute_spatial_distribution,
    compute_threshold_sweep,
    evaluate_hop,
    export_match_points_csv,
    export_inliers_geojson,
    LUNAR_RADIUS_M,
    METRES_PER_DEG_LAT,
    RANSAC_THRESHOLD_PX,
)


def test_lunar_constants():
    assert LUNAR_RADIUS_M == 1737400.0
    expected_m_per_deg = np.pi * 1737400.0 / 180.0
    assert abs(METRES_PER_DEG_LAT - expected_m_per_deg) < 1e-4
    assert abs(METRES_PER_DEG_LAT - 30323.35) < 0.1


def test_compute_residuals():
    # Identity transform
    H = np.eye(3, dtype=np.float32)
    pts1 = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    pts2 = np.array([[10.0, 20.0], [33.0, 44.0]], dtype=np.float32)
    res = compute_residuals(pts1, pts2, H)
    assert len(res) == 2
    assert abs(res[0] - 0.0) < 1e-5
    assert abs(res[1] - 5.0) < 1e-5  # sqrt(3^2 + 4^2) = 5.0


def test_decompose_similarity():
    scale = 1.05
    angle_deg = 30.0
    angle_rad = np.radians(angle_deg)
    tx, ty = 12.3, -45.6
    H = np.array([
        [scale * np.cos(angle_rad), -scale * np.sin(angle_rad), tx],
        [scale * np.sin(angle_rad),  scale * np.cos(angle_rad), ty],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    decomp = decompose_similarity(H)
    assert abs(decomp["scale"] - scale) < 1e-5
    assert abs(decomp["rotation_deg"] - angle_deg) < 1e-4
    assert abs(decomp["translation_x"] - tx) < 1e-5
    assert abs(decomp["translation_y"] - ty) < 1e-5


def test_spatial_distribution_grid():
    shape = (1000, 1000)
    # Put 4 points in top-left cell, 2 points in bottom-right cell
    pts = np.array([
        [10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0],
        [900.0, 900.0], [950.0, 950.0]
    ], dtype=np.float32)

    dist = compute_spatial_distribution(pts, shape, grid_size=8)
    assert dist["grid_occupancy"] == 2
    assert dist["total_cells"] == 64
    assert abs(dist["grid_coverage_pct"] - (2 / 64 * 100)) < 1e-4
    assert dist["grid_counts_8x8"][0][0] == 4
    assert dist["grid_counts_8x8"][7][7] == 2
    assert isinstance(dist["cv_per_cell"], float)
    assert dist["cv_per_cell"] > 0


def test_threshold_sweep():
    # 5 matches with residuals [0.5, 2.5, 4.0, 12.0, 20.0]
    residuals = np.array([0.5, 2.5, 4.0, 12.0, 20.0])
    sweep = compute_threshold_sweep(residuals, ref_gsd=5.0, gates=[15.0, 5.0, 1.0])
    assert len(sweep) == 3

    # Gate 15: residuals <= 15.0 -> 4 matches (0.5, 2.5, 4.0, 12.0)
    assert sweep[0]["gate_px"] == 15.0
    assert sweep[0]["n_within_gate"] == 4
    assert sweep[0]["ratio_within_gate"] == 80.0
    expected_rmse = np.sqrt(np.mean([0.5**2, 2.5**2, 4.0**2, 12.0**2]))
    assert abs(sweep[0]["rmse_of_matches_within_gate_px"] - expected_rmse) < 1e-4

    # Gate 1: residual <= 1.0 -> 1 match (0.5)
    assert sweep[2]["gate_px"] == 1.0
    assert sweep[2]["n_within_gate"] == 1
    assert sweep[2]["rmse_of_matches_within_gate_px"] == 0.5


def test_evaluate_hop_structure():
    pts1 = np.random.uniform(100, 900, (50, 2)).astype(np.float32)
    pts2 = pts1 + np.random.normal(0, 1.0, (50, 2)).astype(np.float32)
    mask = np.ones(50, dtype=bool)
    mask[40:] = False  # 40 inliers, 10 outliers
    H = np.eye(3, dtype=np.float64)

    src_meta = {"sensor": "OHRC", "gsd": 0.26, "product_id": "test_src"}
    ref_meta = {"sensor": "TMC-2", "gsd": 4.72, "product_id": "test_ref", "anchor_lat": -70.0, "anchor_lon": 30.0, "image_shape": (1000, 1000)}

    res = evaluate_hop(
        matches={"pts1": pts1, "pts2": pts2, "inlier_mask": mask},
        transform=H,
        src_meta=src_meta,
        ref_meta=ref_meta,
        inlier_threshold_px=15.0,
    )

    assert res["inlier_count"] == 40
    assert res["total_matches"] == 50
    assert res["inlier_ratio"] == 80.0
    assert res["inlier_threshold_px"] == 15.0
    assert abs(res["inlier_threshold_metres"] - (15.0 * 4.72)) < 1e-4
    assert res["rmse_px"] > 0
    assert res["rmse_metres"] == res["rmse_px"] * 4.72
    assert "threshold_sweep" in res
    assert "spatial_distribution" in res
    assert "transform_params" in res


def test_export_matchpoints_csv_and_geojson(tmp_path):
    pts1 = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32)
    pts2 = np.array([[105.0, 202.0], [298.0, 395.0]], dtype=np.float32)
    mask = np.array([True, False], dtype=bool)
    res = np.array([5.385, 7.071], dtype=np.float64)
    src_meta = {"gsd": 0.26, "image_shape": (1000, 1000)}
    ref_meta = {"gsd": 4.72, "anchor_lat": -69.58, "anchor_lon": 32.28, "image_shape": (1000, 1000)}

    csv_p = tmp_path / "test_matches.csv"
    export_match_points_csv(csv_p, pts1, pts2, mask, res, src_meta, ref_meta)

    assert csv_p.exists()
    content = csv_p.read_text()
    # Check verbatim Lunar fallback header
    assert "Lunar radius 1737400 m" in content
    assert "Valid only within this crop. Not geodetic coordinates." in content
    lines = content.strip().split("\n")
    assert len(lines) == 4  # header comment, column names, 2 data lines

    geojson_p = tmp_path / "test_inliers.geojson"
    export_inliers_geojson(geojson_p, pts1, pts2, mask, res, src_meta, ref_meta)
    assert geojson_p.exists()
    with open(geojson_p) as f:
        data = json.load(f)
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 1  # only 1 inlier
    assert data["features"][0]["properties"]["match_id"] == 0
