"""
Geographic Footprint Alignment from Geometry Grids (geo_align).

Reads Chandrayaan-2 geometry CSV files (Longitude, Latitude, Pixel, Scan),
determines the shared geographic ground coverage across selenographic polar
coordinates using spherical 3D coordinates, and computes corresponding pixel
and scan windows for both products.

SIH26166 — Chandrayaan-2 Multi-modal Image Correspondence Pipeline
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

# Mean lunar radius in kilometers (IAU/SOFA standard)
MOON_RADIUS_KM = 1737.4


def latlon_to_xyz(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    radius_km: float = MOON_RADIUS_KM,
) -> np.ndarray:
    """Convert spherical selenographic (lat, lon) in degrees to 3D Cartesian (x, y, z).

    In polar regions (>80° latitude), lines of longitude converge at the pole.
    Spherical distance calculations via 3D Cartesian coordinates avoid polar
    singularities and wrap-around discontinuities at 0°/360° longitude.

    Args:
        lat_deg: Latitude in degrees [-90, +90].
        lon_deg: Longitude in degrees [0, 360] or [-180, +180].
        radius_km: Sphere radius in kilometers.

    Returns:
        (N, 3) ndarray of cartesian coordinates in km.
    """
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    x = radius_km * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius_km * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius_km * np.sin(lat_rad)
    return np.column_stack([x, y, z])


def find_common_region(
    csv_a: Union[str, Path],
    csv_b: Union[str, Path],
    distance_threshold_km: float = 15.0,
) -> Dict[str, Any]:
    """Find the common geographic ground region between two geometry grids.

    Reads both geometry CSV files (which contain columns Longitude, Latitude,
    Pixel, Scan), computes selenocentric 3D Euclidean distances, and finds
    the bounding scan/pixel windows as well as the center of overlap.

    Args:
        csv_a: Path to geometry grid CSV for product A (e.g. TMC-2).
        csv_b: Path to geometry grid CSV for product B (e.g. IIRS).
        distance_threshold_km: Maximum ground distance in km between grid points
                               to be considered overlapping (default 15 km,
                               matching instrument swath width).

    Returns:
        Dictionary containing:
            - 'a': Dict with scan_min, scan_max, pixel_min, pixel_max,
                   center_scan, center_pixel for product A
            - 'b': Dict with scan_min, scan_max, pixel_min, pixel_max,
                   center_scan, center_pixel for product B
            - 'shared_lat_range': (min_lat, max_lat) in degrees
            - 'shared_lon_range': (min_lon, max_lon) in degrees
            - 'center_lat': Latitude of the closest overlap point in degrees
            - 'center_lon': Longitude of the closest overlap point in degrees
            - 'min_distance_km': Minimum distance between closest grid points
            - 'num_overlap_points': Count of overlapping grid points
    """
    path_a = Path(csv_a).resolve()
    path_b = Path(csv_b).resolve()

    if not path_a.exists():
        raise FileNotFoundError(f"Geometry CSV A not found at {path_a}")
    if not path_b.exists():
        raise FileNotFoundError(f"Geometry CSV B not found at {path_b}")

    df_a = pd.read_csv(str(path_a))
    df_b = pd.read_csv(str(path_b))

    for col in ["Longitude", "Latitude", "Pixel", "Scan"]:
        if col not in df_a.columns:
            raise ValueError(f"Missing required column '{col}' in {path_a.name}")
        if col not in df_b.columns:
            raise ValueError(f"Missing required column '{col}' in {path_b.name}")

    # Convert lat/lon grid points to 3D Cartesian coordinates
    xyz_a = latlon_to_xyz(df_a["Latitude"].values, df_a["Longitude"].values)
    xyz_b = latlon_to_xyz(df_b["Latitude"].values, df_b["Longitude"].values)

    # Build KD-Tree on product A and query points in product B
    tree_a = cKDTree(xyz_a)
    dists_b_to_a, idx_a = tree_a.query(xyz_b)

    mask_b = dists_b_to_a <= distance_threshold_km
    if not np.any(mask_b):
        raise ValueError(
            f"No overlapping geographic points found between {path_a.name} and "
            f"{path_b.name} within threshold {distance_threshold_km} km."
        )

    matched_a = idx_a[mask_b]
    sub_a = df_a.iloc[matched_a]
    sub_b = df_b[mask_b]

    # Find the single closest matching pair to serve as center of interest
    best_b_idx = int(np.argmin(dists_b_to_a))
    best_a_idx = int(idx_a[best_b_idx])

    center_a = df_a.iloc[best_a_idx]
    center_b = df_b.iloc[best_b_idx]

    shared_lat_min = float(min(sub_a["Latitude"].min(), sub_b["Latitude"].min()))
    shared_lat_max = float(max(sub_a["Latitude"].max(), sub_b["Latitude"].max()))
    shared_lon_min = float(min(sub_a["Longitude"].min(), sub_b["Longitude"].min()))
    shared_lon_max = float(max(sub_a["Longitude"].max(), sub_b["Longitude"].max()))

    result: Dict[str, Any] = {
        "a": {
            "scan_min": int(sub_a["Scan"].min()),
            "scan_max": int(sub_a["Scan"].max()),
            "pixel_min": int(sub_a["Pixel"].min()),
            "pixel_max": int(sub_a["Pixel"].max()),
            "center_scan": int(center_a["Scan"]),
            "center_pixel": int(center_a["Pixel"]),
        },
        "b": {
            "scan_min": int(sub_b["Scan"].min()),
            "scan_max": int(sub_b["Scan"].max()),
            "pixel_min": int(sub_b["Pixel"].min()),
            "pixel_max": int(sub_b["Pixel"].max()),
            "center_scan": int(center_b["Scan"]),
            "center_pixel": int(center_b["Pixel"]),
        },
        "shared_lat_range": (shared_lat_min, shared_lat_max),
        "shared_lon_range": (shared_lon_min, shared_lon_max),
        "center_lat": float(center_a["Latitude"]),
        "center_lon": float(center_a["Longitude"]),
        "min_distance_km": float(dists_b_to_a[best_b_idx]),
        "num_overlap_points": int(np.sum(mask_b)),
    }

    logger.info(
        "Common region found: Center Lat=%.4f°, Lon=%.4f° | A center=(Scan %d, Pixel %d), B center=(Scan %d, Pixel %d)",
        result["center_lat"],
        result["center_lon"],
        result["a"]["center_scan"],
        result["a"]["center_pixel"],
        result["b"]["center_scan"],
        result["b"]["center_pixel"],
    )

    return result


def compute_centered_crop_slices(
    center_scan: int,
    center_pixel: int,
    crop_lines: int,
    crop_samples: int,
    total_lines: int,
    total_samples: int,
) -> Tuple[slice, slice]:
    """Compute clamped line and sample slices for a centered crop window.

    Args:
        center_scan: Desired center line index.
        center_pixel: Desired center sample index.
        crop_lines: Height of crop window in lines.
        crop_samples: Width of crop window in samples.
        total_lines: Total lines in image/raster.
        total_samples: Total samples in image/raster.

    Returns:
        (line_slice, sample_slice)
    """
    half_lines = crop_lines // 2
    half_samples = crop_samples // 2

    l0 = center_scan - half_lines
    l1 = l0 + crop_lines

    if l0 < 0:
        l0 = 0
        l1 = min(total_lines, crop_lines)
    elif l1 > total_lines:
        l1 = total_lines
        l0 = max(0, total_lines - crop_lines)

    s0 = center_pixel - half_samples
    s1 = s0 + crop_samples

    if s0 < 0:
        s0 = 0
        s1 = min(total_samples, crop_samples)
    elif s1 > total_samples:
        s1 = total_samples
        s0 = max(0, total_samples - crop_samples)

    return slice(l0, l1), slice(s0, s1)
