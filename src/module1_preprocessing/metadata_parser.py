"""
Selenographic footprint parser and search region estimator for lunar imagery.

This module provides data structures and parsing utilities for simulated SPICE-kernel
instrument footprint metadata across Chandrayaan-2 payloads (OHRC, TMC-2, IIRS).
It computes axis-aligned bounding boxes, evaluates multi-instrument spatial overlap,
constrains matching search regions, and estimates coarse pixel translation offsets
on the lunar reference sphere (Moon radius = 1737.4 km).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

# Configure module-level logger
logger = logging.getLogger(__name__)

# Mean lunar radius in kilometers and meters according to IAU/IAG standard
MOON_RADIUS_KM: float = 1737.4
MOON_RADIUS_M: float = 1737400.0

# Selenographic coordinate validity ranges (in degrees)
MIN_LAT_DEG: float = -90.0
MAX_LAT_DEG: float = 90.0
MIN_LON_DEG: float = -180.0
MAX_LON_DEG: float = 360.0


def _clip(value: float, min_val: float, max_val: float) -> float:
    """Clamp a numerical value to [min_val, max_val].

    Args:
        value: Input scalar value.
        min_val: Minimum lower bound.
        max_val: Maximum upper bound.

    Returns:
        Clamped float value.
    """
    if np is not None:
        return float(np.clip(value, min_val, max_val))
    return float(max(min_val, min(max_val, value)))


@dataclass
class InstrumentFootprint:
    """Selenographic footprint of a single instrument acquisition.

    Represents spatial, radiometric, and solar geometry parameters for a lunar
    remote sensing image acquisition (e.g., Chandrayaan-2 OHRC, TMC-2, IIRS).

    Attributes:
        instrument_name: Unique identifier of the sensor ('OHRC', 'TMC-2', 'IIRS').
        center_lat: Selenographic latitude of the scene center in degrees [-90, +90].
        center_lon: Selenographic longitude of the scene center in degrees [-180, +180].
        corner_coords: 4 corner vertices as (lat, lon) pairs in degrees.
        gsd_m: Ground Sample Distance in meters per pixel.
        sun_elevation_deg: Solar elevation angle above local horizon in degrees [-90, +90].
        sun_azimuth_deg: Solar azimuth angle in degrees [0, 360) clockwise from North.
        acquisition_time: ISO 8601 formatted timestamp string of the acquisition epoch.
        bounding_box: Computed bounding box (min_lat, max_lat, min_lon, max_lon) in degrees.
    """

    instrument_name: str
    center_lat: float
    center_lon: float
    corner_coords: List[Tuple[float, float]]
    gsd_m: float
    sun_elevation_deg: float
    sun_azimuth_deg: float
    acquisition_time: str
    bounding_box: Optional[Tuple[float, float, float, float]] = None

    def __post_init__(self) -> None:
        """Validate footprint attributes and compute bounding box if not provided."""
        # Convert any iterable corner formats into strict List[Tuple[float, float]]
        cleaned_corners: List[Tuple[float, float]] = []
        if self.corner_coords:
            for pt in self.corner_coords:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    cleaned_corners.append((float(pt[0]), float(pt[1])))
                else:
                    logger.warning(
                        "Malformed corner coordinate %s for instrument '%s'.",
                        pt,
                        self.instrument_name,
                    )
        self.corner_coords = cleaned_corners

        # Automatically compute bounding box if missing or None
        if self.bounding_box is None and self.corner_coords:
            lats = [c[0] for c in self.corner_coords]
            lons = [c[1] for c in self.corner_coords]
            self.bounding_box = (
                float(min(lats)),
                float(max(lats)),
                float(min(lons)),
                float(max(lons)),
            )

        # Validate bounding box consistency if both corners and bounding box exist
        self._validate_fields()

    def _validate_fields(self) -> None:
        """Perform boundary and sanity checks on footprint attributes."""
        if not (MIN_LAT_DEG <= self.center_lat <= MAX_LAT_DEG):
            logger.warning(
                "Instrument '%s' center_lat (%f) is outside valid range [%f, %f].",
                self.instrument_name,
                self.center_lat,
                MIN_LAT_DEG,
                MAX_LAT_DEG,
            )

        if not (MIN_LON_DEG <= self.center_lon <= MAX_LON_DEG):
            logger.warning(
                "Instrument '%s' center_lon (%f) is outside expected longitude range [%f, %f].",
                self.instrument_name,
                self.center_lon,
                MIN_LON_DEG,
                MAX_LON_DEG,
            )

        if self.gsd_m <= 0.0:
            logger.warning(
                "Instrument '%s' GSD (%f m) is non-positive; defaulting to 1.0 m.",
                self.instrument_name,
                self.gsd_m,
            )
            self.gsd_m = max(self.gsd_m, 1e-3)

        if not (-90.0 <= self.sun_elevation_deg <= 90.0):
            logger.warning(
                "Instrument '%s' sun_elevation_deg (%f) is outside [-90, +90].",
                self.instrument_name,
                self.sun_elevation_deg,
            )

        if not (0.0 <= (self.sun_azimuth_deg % 360.0) <= 360.0):
            logger.warning(
                "Instrument '%s' sun_azimuth_deg (%f) is outside standard azimuth range.",
                self.instrument_name,
                self.sun_azimuth_deg,
            )

    @property
    def lat_span_deg(self) -> float:
        """Return the latitudinal span of the bounding box in degrees."""
        if self.bounding_box is None:
            return 0.0
        return max(0.0, self.bounding_box[1] - self.bounding_box[0])

    @property
    def lon_span_deg(self) -> float:
        """Return the longitudinal span of the bounding box in degrees."""
        if self.bounding_box is None:
            return 0.0
        return max(0.0, self.bounding_box[3] - self.bounding_box[2])

    @property
    def bounding_box_area_deg2(self) -> float:
        """Return the bounding box area in square degrees."""
        return self.lat_span_deg * self.lon_span_deg

    def to_dict(self) -> Dict[str, Any]:
        """Serialize footprint to a standard dictionary format."""
        return {
            "instrument_name": self.instrument_name,
            "center_lat": self.center_lat,
            "center_lon": self.center_lon,
            "corner_coords": self.corner_coords,
            "gsd_m": self.gsd_m,
            "sun_elevation_deg": self.sun_elevation_deg,
            "sun_azimuth_deg": self.sun_azimuth_deg,
            "acquisition_time": self.acquisition_time,
            "bounding_box": self.bounding_box,
        }


class FootprintParser:
    """Parses instrument footprint metadata and computes overlap regions.

    In a real pipeline, this would parse SPICE kernel files (CK, SPK, IK, FK)
    and PDS4 labels. For this correspondence framework, it processes metadata
    dictionaries containing selenographic footprint coordinates, ground sample
    distances, and solar illumination geometry.
    """

    def __init__(self) -> None:
        """Initialize an empty FootprintParser container."""
        self.footprints: Dict[str, InstrumentFootprint] = {}

    def parse_metadata(self, metadata: Dict[str, Any]) -> Dict[str, InstrumentFootprint]:
        """Parse the metadata dict (from mock data generator or PDS label).

        The metadata dict has instrument names as keys (e.g. 'OHRC', 'TMC-2', 'IIRS'),
        each containing:
        - center_lat (float)
        - center_lon (float)
        - corner_coords (list of (lat, lon) pairs)
        - gsd_m (float)
        - sun_elevation_deg (float)
        - sun_azimuth_deg (float)
        - acquisition_time (str)
        - optional bounding_box (tuple of 4 floats)

        Computes bounding_box for each footprint if not already present.
        Stores resulting InstrumentFootprint objects in self.footprints.

        Args:
            metadata: Nested dictionary mapping instrument names to acquisition metadata.

        Returns:
            Dictionary mapping instrument names to parsed InstrumentFootprint objects.
        """
        if not isinstance(metadata, dict):
            logger.error("Metadata input must be a dictionary. Received: %s", type(metadata))
            return self.footprints

        parsed: Dict[str, InstrumentFootprint] = {}

        for instr_name, raw_data in metadata.items():
            if not isinstance(raw_data, dict):
                logger.warning(
                    "Skipping instrument '%s' as its metadata is not a dictionary.",
                    instr_name,
                )
                continue

            try:
                # Extract corner coordinates
                raw_corners = raw_data.get("corner_coords", [])
                corners: List[Tuple[float, float]] = []
                if isinstance(raw_corners, (list, tuple)):
                    for pt in raw_corners:
                        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                            corners.append((float(pt[0]), float(pt[1])))

                # Extract or infer center coordinates
                if "center_lat" in raw_data and "center_lon" in raw_data:
                    center_lat = float(raw_data["center_lat"])
                    center_lon = float(raw_data["center_lon"])
                elif corners:
                    lat_list = [pt[0] for pt in corners]
                    lon_list = [pt[1] for pt in corners]
                    center_lat = float(sum(lat_list) / len(lat_list))
                    center_lon = float(sum(lon_list) / len(lon_list))
                    logger.info(
                        "Inferred center (%f, %f) from corner coordinates for '%s'.",
                        center_lat,
                        center_lon,
                        instr_name,
                    )
                else:
                    logger.warning(
                        "Missing center coordinates and corner coordinates for '%s'. Using (0.0, 0.0).",
                        instr_name,
                    )
                    center_lat = 0.0
                    center_lon = 0.0

                # Extract or infer GSD
                gsd_m = float(raw_data.get("gsd_m", 1.0))
                if gsd_m <= 0.0:
                    logger.warning("Non-positive GSD (%f) for '%s'; fallback to 1.0.", gsd_m, instr_name)
                    gsd_m = 1.0

                # Extract solar geometry
                sun_elevation_deg = float(raw_data.get("sun_elevation_deg", 30.0))
                sun_azimuth_deg = float(raw_data.get("sun_azimuth_deg", 0.0))

                # Extract acquisition timestamp
                acquisition_time = str(raw_data.get("acquisition_time", "2023-08-23T12:00:00Z"))

                # Extract bounding box if explicitly specified
                bounding_box: Optional[Tuple[float, float, float, float]] = None
                if "bounding_box" in raw_data and raw_data["bounding_box"] is not None:
                    raw_bbox = raw_data["bounding_box"]
                    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
                        bounding_box = (
                            float(raw_bbox[0]),
                            float(raw_bbox[1]),
                            float(raw_bbox[2]),
                            float(raw_bbox[3]),
                        )

                # Fallback corner synthesis if corners were missing but center/GSD provided
                if not corners and bounding_box is None:
                    # Estimate a small 1000x1000 pixel footprint around center
                    half_span_m = 500.0 * gsd_m
                    d_lat_deg = math.degrees(half_span_m / MOON_RADIUS_M)
                    cos_lat = max(1e-6, math.cos(math.radians(center_lat)))
                    d_lon_deg = math.degrees(half_span_m / (MOON_RADIUS_M * cos_lat))

                    corners = [
                        (center_lat + d_lat_deg, center_lon - d_lon_deg),  # Top-left (NW)
                        (center_lat + d_lat_deg, center_lon + d_lon_deg),  # Top-right (NE)
                        (center_lat - d_lat_deg, center_lon + d_lon_deg),  # Bottom-right (SE)
                        (center_lat - d_lat_deg, center_lon - d_lon_deg),  # Bottom-left (SW)
                    ]
                    bounding_box = (
                        center_lat - d_lat_deg,
                        center_lat + d_lat_deg,
                        center_lon - d_lon_deg,
                        center_lon + d_lon_deg,
                    )
                    logger.info(
                        "Synthesized approximate footprint corners for '%s' around (%f, %f).",
                        instr_name,
                        center_lat,
                        center_lon,
                    )

                footprint = InstrumentFootprint(
                    instrument_name=str(instr_name),
                    center_lat=center_lat,
                    center_lon=center_lon,
                    corner_coords=corners,
                    gsd_m=gsd_m,
                    sun_elevation_deg=sun_elevation_deg,
                    sun_azimuth_deg=sun_azimuth_deg,
                    acquisition_time=acquisition_time,
                    bounding_box=bounding_box,
                )

                parsed[str(instr_name)] = footprint

            except Exception as err:
                logger.error(
                    "Failed to parse metadata for instrument '%s': %s",
                    instr_name,
                    err,
                    exc_info=True,
                )

        self.footprints.update(parsed)
        return self.footprints

    def compute_overlap(
        self,
        fp_a: InstrumentFootprint,
        fp_b: InstrumentFootprint,
    ) -> Dict[str, Any]:
        """Compute the overlap region between two footprints.

        Uses 2D axis-aligned bounding box (AABB) intersection in selenographic
        coordinates (lat, lon). Calculates relative coverage area fractions
        with respect to both footprints.

        Args:
            fp_a: First instrument footprint.
            fp_b: Second instrument footprint.

        Returns:
            Dictionary containing:
            - 'overlap_box': (min_lat, max_lat, min_lon, max_lon) or None if no overlap
            - 'overlap_area_fraction_a': fraction of fp_a covered by overlap [0.0, 1.0]
            - 'overlap_area_fraction_b': fraction of fp_b covered by overlap [0.0, 1.0]
            - 'has_overlap': boolean indicating if overlap exists
        """
        no_overlap_result: Dict[str, Any] = {
            "overlap_box": None,
            "overlap_area_fraction_a": 0.0,
            "overlap_area_fraction_b": 0.0,
            "has_overlap": False,
        }

        if fp_a is None or fp_b is None:
            logger.warning("compute_overlap called with None footprint argument.")
            return no_overlap_result

        box_a = fp_a.bounding_box
        box_b = fp_b.bounding_box

        if box_a is None or box_b is None:
            logger.warning(
                "Cannot compute overlap: missing bounding box for '%s' or '%s'.",
                fp_a.instrument_name,
                fp_b.instrument_name,
            )
            return no_overlap_result

        # Intersection bounds in latitude and longitude
        overlap_min_lat = max(box_a[0], box_b[0])
        overlap_max_lat = min(box_a[1], box_b[1])
        overlap_min_lon = max(box_a[2], box_b[2])
        overlap_max_lon = min(box_a[3], box_b[3])

        # Check for non-empty intersection
        lat_overlap_span = overlap_max_lat - overlap_min_lat
        lon_overlap_span = overlap_max_lon - overlap_min_lon

        if lat_overlap_span <= 0.0 or lon_overlap_span <= 0.0:
            return no_overlap_result

        overlap_box: Tuple[float, float, float, float] = (
            float(overlap_min_lat),
            float(overlap_max_lat),
            float(overlap_min_lon),
            float(overlap_max_lon),
        )

        overlap_area = lat_overlap_span * lon_overlap_span

        area_a = fp_a.bounding_box_area_deg2
        area_b = fp_b.bounding_box_area_deg2

        frac_a = _clip(overlap_area / area_a, 0.0, 1.0) if area_a > 1e-12 else 0.0
        frac_b = _clip(overlap_area / area_b, 0.0, 1.0) if area_b > 1e-12 else 0.0

        return {
            "overlap_box": overlap_box,
            "overlap_area_fraction_a": frac_a,
            "overlap_area_fraction_b": frac_b,
            "has_overlap": True,
        }

    def compute_all_overlaps(self) -> Dict[str, Dict[str, Any]]:
        """Compute pairwise overlaps for all loaded footprints.

        Iterates over all ordered pairs of distinct instruments in self.footprints.

        Returns:
            Dictionary keyed by 'INSTR_A-INSTR_B' with overlap information dicts.
        """
        overlaps: Dict[str, Dict[str, Any]] = {}
        instrument_keys = list(self.footprints.keys())

        for name_a in instrument_keys:
            for name_b in instrument_keys:
                if name_a == name_b:
                    continue
                pair_key = f"{name_a}-{name_b}"
                fp_a = self.footprints[name_a]
                fp_b = self.footprints[name_b]
                overlaps[pair_key] = self.compute_overlap(fp_a, fp_b)

        return overlaps

    def get_search_region(
        self,
        source_instrument: str,
        target_instrument: str,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Get the overlap bounding box to constrain the matching search space.

        Given a source and target instrument, returns the overlap region
        bounding box coordinates in selenographic degrees.

        Args:
            source_instrument: Name of the source sensor (e.g., 'OHRC').
            target_instrument: Name of the target/reference sensor (e.g., 'TMC-2').

        Returns:
            Tuple of (min_lat, max_lat, min_lon, max_lon) in degrees, or None if no overlap.
        """
        if source_instrument not in self.footprints:
            logger.warning("Source instrument '%s' not found in loaded footprints.", source_instrument)
            return None

        if target_instrument not in self.footprints:
            logger.warning("Target instrument '%s' not found in loaded footprints.", target_instrument)
            return None

        fp_src = self.footprints[source_instrument]
        fp_tgt = self.footprints[target_instrument]

        overlap_info = self.compute_overlap(fp_src, fp_tgt)
        if overlap_info.get("has_overlap", False):
            return overlap_info.get("overlap_box")

        return None

    def estimate_pixel_offset(
        self,
        fp_a: InstrumentFootprint,
        fp_b: InstrumentFootprint,
    ) -> Optional[Tuple[int, int]]:
        """Estimate rough pixel offset between two footprints based on centers and GSD.

        Calculates selenographic ground displacement using the spherical lunar model
        (Moon radius = 1737.4 km) and converts the displacement into the pixel grid
        of the target instrument (fp_b).

        Convention:
        - Target coordinate frame (fp_b):
          - Row axis (dy): increases Southward (North is negative dy, South is positive dy).
          - Column axis (dx): increases Eastward (East is positive dx, West is negative dx).
        - Offset represents the location of fp_a center relative to fp_b center in fp_b pixels:
          center_a_pixel = center_b_pixel + (dy_pixels, dx_pixels).

        Args:
            fp_a: Source instrument footprint.
            fp_b: Target instrument footprint whose pixel grid defines the coordinate frame.

        Returns:
            Tuple (dy_pixels, dx_pixels) as integer pixel shifts, or None if no spatial overlap.
        """
        if fp_a is None or fp_b is None:
            return None

        # Verify spatial overlap exists before computing initial offset
        overlap_info = self.compute_overlap(fp_a, fp_b)
        if not overlap_info.get("has_overlap", False):
            logger.debug(
                "No spatial overlap between '%s' and '%s'. Offset cannot be estimated.",
                fp_a.instrument_name,
                fp_b.instrument_name,
            )
            return None

        if fp_b.gsd_m <= 0.0:
            logger.error("Target footprint '%s' has invalid GSD: %f", fp_b.instrument_name, fp_b.gsd_m)
            return None

        # Angular differences in radians
        lat_a_rad = math.radians(fp_a.center_lat)
        lat_b_rad = math.radians(fp_b.center_lat)
        lon_a_rad = math.radians(fp_a.center_lon)
        lon_b_rad = math.radians(fp_b.center_lon)

        d_lat_rad = lat_a_rad - lat_b_rad
        d_lon_rad = lon_a_rad - lon_b_rad
        mean_lat_rad = 0.5 * (lat_a_rad + lat_b_rad)

        # Ground displacement in meters (equirectangular lunar sphere projection)
        # North displacement in meters:
        north_dist_m = MOON_RADIUS_M * d_lat_rad

        # East displacement in meters (scaled by cosine of mean latitude):
        cos_mean_lat = math.cos(mean_lat_rad)
        east_dist_m = MOON_RADIUS_M * cos_mean_lat * d_lon_rad

        # Convert ground meters to target instrument pixel coordinates
        # dy: Image row index increases downwards (South), so North is negative dy
        dy_pixels = -north_dist_m / fp_b.gsd_m
        # dx: Image col index increases rightwards (East), so East is positive dx
        dx_pixels = east_dist_m / fp_b.gsd_m

        return int(round(dy_pixels)), int(round(dx_pixels))

    def haversine_distance_m(
        self,
        lat1_deg: float,
        lon1_deg: float,
        lat2_deg: float,
        lon2_deg: float,
    ) -> float:
        """Calculate the great-circle selenographic distance between two points.

        Uses the Haversine formula on a spherical lunar surface with mean radius 1737.4 km.

        Args:
            lat1_deg: Latitude of point 1 in degrees.
            lon1_deg: Longitude of point 1 in degrees.
            lat2_deg: Latitude of point 2 in degrees.
            lon2_deg: Longitude of point 2 in degrees.

        Returns:
            Great-circle distance in meters.
        """
        phi1 = math.radians(lat1_deg)
        phi2 = math.radians(lat2_deg)
        delta_phi = math.radians(lat2_deg - lat1_deg)
        delta_lambda = math.radians(lon2_deg - lon1_deg)

        sin_dphi_half = math.sin(0.5 * delta_phi)
        sin_dlam_half = math.sin(0.5 * delta_lambda)

        a = sin_dphi_half * sin_dphi_half + math.cos(phi1) * math.cos(phi2) * sin_dlam_half * sin_dlam_half
        a = _clip(a, 0.0, 1.0)
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

        return float(MOON_RADIUS_M * c)

    def crop_bounding_box_pixels(
        self,
        instrument_name: str,
        selenographic_bbox: Tuple[float, float, float, float],
        image_shape: Tuple[int, int],
    ) -> Optional[Tuple[int, int, int, int]]:
        """Map a selenographic bounding box to pixel coordinates for a given instrument.

        Useful for cropping the coarse search window within the full-scene raster.

        Args:
            instrument_name: Target instrument name in self.footprints.
            selenographic_bbox: Tuple of (min_lat, max_lat, min_lon, max_lon) in degrees.
            image_shape: Tuple of (height, width) of the instrument image array.

        Returns:
            Tuple of (row_min, row_max, col_min, col_max) clamped to image boundaries,
            or None if the footprint is missing or has zero dimensions.
        """
        if instrument_name not in self.footprints:
            logger.warning("Instrument '%s' not found for pixel bbox conversion.", instrument_name)
            return None

        fp = self.footprints[instrument_name]
        if fp.bounding_box is None:
            return None

        fp_min_lat, fp_max_lat, fp_min_lon, fp_max_lon = fp.bounding_box
        lat_span = fp_max_lat - fp_min_lat
        lon_span = fp_max_lon - fp_min_lon

        if lat_span <= 1e-9 or lon_span <= 1e-9:
            return None

        h, w = image_shape
        sub_min_lat, sub_max_lat, sub_min_lon, sub_max_lon = selenographic_bbox

        # Row 0 is at max_lat (North), Row H is at min_lat (South)
        row_min = int(math.floor((fp_max_lat - sub_max_lat) / lat_span * h))
        row_max = int(math.ceil((fp_max_lat - sub_min_lat) / lat_span * h))

        # Col 0 is at min_lon (West), Col W is at max_lon (East)
        col_min = int(math.floor((sub_min_lon - fp_min_lon) / lon_span * w))
        col_max = int(math.ceil((sub_max_lon - fp_min_lon) / lon_span * w))

        # Clamp to image dimensions
        row_min = max(0, min(h - 1, row_min))
        row_max = max(row_min + 1, min(h, row_max))
        col_min = max(0, min(w - 1, col_min))
        col_max = max(col_min + 1, min(w, col_max))

        return (row_min, row_max, col_min, col_max)
