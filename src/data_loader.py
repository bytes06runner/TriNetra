"""
TriNetra Phase 1: Raw Binary Data Loader for Chandrayaan-2 PDS4 Archives.

Memory-mapped I/O for OHRC (.img, uint8), TMC-2 (.img, uint16 LE),
and IIRS (.qub, uint16 LE BIL/BSQ cubes).  No file is ever fully loaded
into RAM — every access goes through np.memmap + on-demand patch extraction.

Author : TriNetra Team (SIH26166)
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Metadata parsed from the PDS4 XML label
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class InstrumentMetadata:
    """Immutable metadata parsed from a PDS4 XML label."""
    instrument: str
    acquisition_time: str
    pixel_resolution_m: float
    spacecraft_altitude_km: float
    sun_azimuth_deg: float
    sun_elevation_deg: float
    solar_incidence_deg: float
    projection: str
    area: str
    corner_coords: Dict[str, Tuple[float, float]]   # {UL, UR, LL, LR} → (lat, lon)
    lines: int
    samples: int
    bands: int
    dtype: np.dtype
    band_centers_nm: Optional[List[float]] = field(default=None)


# ─────────────────────────────────────────────────────────────────────
# PDS4 XML label parser
# ─────────────────────────────────────────────────────────────────────
def _ns_strip(tag: str) -> str:
    """Remove XML namespace prefix."""
    return tag.split("}")[-1] if "}" in tag else tag


def parse_pds4_label(xml_path: Path) -> InstrumentMetadata:
    """Parse a Chandrayaan-2 PDS4 XML label and return structured metadata."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Flatten all elements for easy searching (namespace-agnostic)
    elems: Dict[str, str] = {}
    for el in root.iter():
        tag = _ns_strip(el.tag)
        if el.text and el.text.strip():
            elems.setdefault(tag, el.text.strip())

    # --- Instrument identification ---
    title = elems.get("title", "")
    if "OHRC" in title.upper() or "ohr" in str(xml_path).lower():
        instrument = "OHRC"
    elif "TMC" in title.upper() or "tmc" in str(xml_path).lower():
        instrument = "TMC-2"
    elif "IIRS" in title.upper() or "iir" in str(xml_path).lower():
        instrument = "IIRS"
    else:
        instrument = "UNKNOWN"

    # --- Acquisition time ---
    acq_time = elems.get("start_date_time", "")

    # --- Product parameters ---
    px_res = float(elems.get("pixel_resolution", "0"))
    alt    = float(elems.get("spacecraft_altitude", "0"))
    sun_az = float(elems.get("sun_azimuth", "0"))
    sun_el = float(elems.get("sun_elevation", "0"))
    sol_in = float(elems.get("solar_incidence", "0"))
    proj   = elems.get("projection", "")
    area   = elems.get("area", "")

    # --- Corner coordinates (from Refined_Corner_Coordinates) ---
    corners: Dict[str, Tuple[float, float]] = {}
    for ref_block in root.iter():
        tag = _ns_strip(ref_block.tag)
        if tag == "Refined_Corner_Coordinates":
            lat_lon = {}
            for child in ref_block:
                ctag = _ns_strip(child.tag)
                if child.text:
                    lat_lon[ctag] = float(child.text.strip())
            corners = {
                "UL": (lat_lon.get("upper_left_latitude", 0),
                       lat_lon.get("upper_left_longitude", 0)),
                "UR": (lat_lon.get("upper_right_latitude", 0),
                       lat_lon.get("upper_right_longitude", 0)),
                "LL": (lat_lon.get("lower_left_latitude", 0),
                       lat_lon.get("lower_left_longitude", 0)),
                "LR": (lat_lon.get("lower_right_latitude", 0),
                       lat_lon.get("lower_right_longitude", 0)),
            }
            break

    # If Refined_Corner_Coordinates wasn't found, try System_Level_Coordinates
    if not corners:
        for ref_block in root.iter():
            tag = _ns_strip(ref_block.tag)
            if tag == "System_Level_Coordinates":
                lat_lon = {}
                for child in ref_block:
                    ctag = _ns_strip(child.tag)
                    if child.text:
                        lat_lon[ctag] = float(child.text.strip())
                corners = {
                    "UL": (lat_lon.get("upper_left_latitude", 0),
                           lat_lon.get("upper_left_longitude", 0)),
                    "UR": (lat_lon.get("upper_right_latitude", 0),
                           lat_lon.get("upper_right_longitude", 0)),
                    "LL": (lat_lon.get("lower_left_latitude", 0),
                           lat_lon.get("lower_left_longitude", 0)),
                    "LR": (lat_lon.get("lower_right_latitude", 0),
                           lat_lon.get("lower_right_longitude", 0)),
                }
                break

    # --- Array dimensions & data type ---
    lines = 0
    samples = 0
    bands = 1
    dtype = np.uint8
    band_centers: Optional[List[float]] = None

    for arr_block in root.iter():
        arr_tag = _ns_strip(arr_block.tag)
        if arr_tag in ("Array_2D_Image", "Array_3D_Spectrum"):
            # Data type
            for el in arr_block.iter():
                if _ns_strip(el.tag) == "data_type" and el.text:
                    dt_str = el.text.strip()
                    if dt_str == "UnsignedByte":
                        dtype = np.dtype("uint8")
                    elif dt_str in ("UnsignedLSB2", "LSB_Unsigned_Integer"):
                        dtype = np.dtype("<u2")  # little-endian uint16
                    elif dt_str in ("UnsignedMSB2", "MSB_Unsigned_Integer"):
                        dtype = np.dtype(">u2")  # big-endian uint16
                    elif dt_str == "SignedLSB2":
                        dtype = np.dtype("<i2")
                    else:
                        logger.warning("Unknown PDS4 data_type: %s, defaulting to uint16 LE", dt_str)
                        dtype = np.dtype("<u2")

            # Axes
            for axis in arr_block.iter():
                if _ns_strip(axis.tag) == "Axis_Array":
                    name_el = None
                    elem_el = None
                    for child in axis:
                        ctag = _ns_strip(child.tag)
                        if ctag == "axis_name" and child.text:
                            name_el = child.text.strip().upper()
                        elif ctag == "elements" and child.text:
                            elem_el = int(child.text.strip())
                    if name_el and elem_el:
                        if name_el == "LINE":
                            lines = elem_el
                        elif name_el == "SAMPLE":
                            samples = elem_el
                        elif name_el == "BAND":
                            bands = elem_el

            # Band centers (IIRS only)
            if arr_tag == "Array_3D_Spectrum":
                band_centers = []
                for bb in arr_block.iter():
                    if _ns_strip(bb.tag) == "Band_Bin":
                        for child in bb:
                            if _ns_strip(child.tag) == "center_wavelength" and child.text:
                                band_centers.append(float(child.text.strip()))
                if not band_centers:
                    band_centers = None

            break  # We only need the first matching array block

    return InstrumentMetadata(
        instrument=instrument,
        acquisition_time=acq_time,
        pixel_resolution_m=px_res,
        spacecraft_altitude_km=alt,
        sun_azimuth_deg=sun_az,
        sun_elevation_deg=sun_el,
        solar_incidence_deg=sol_in,
        projection=proj,
        area=area,
        corner_coords=corners,
        lines=lines,
        samples=samples,
        bands=bands,
        dtype=dtype,
        band_centers_nm=band_centers,
    )


# ─────────────────────────────────────────────────────────────────────
# Memory-mapped data loader
# ─────────────────────────────────────────────────────────────────────
class Chandrayaan2Loader:
    """
    Zero-copy, memory-mapped loader for Chandrayaan-2 PDS4 binary files.

    Supports:
        - OHRC  .img  (2D, uint8,  93693 × 12000)
        - TMC-2 .img  (2D, uint16, 189886 × 4000)
        - IIRS  .qub  (3D, uint16, 256 × 2264 × 250, Band-Sequential)

    The entire file is memory-mapped (never loaded into RAM).
    Use ``get_patch()`` to extract manageable tiles for processing.
    """

    def __init__(self, data_path: Path, xml_path: Path) -> None:
        """
        Args:
            data_path: Absolute path to the .img or .qub binary file.
            xml_path:  Absolute path to the companion PDS4 .xml label.
        """
        self.data_path = Path(data_path)
        self.xml_path  = Path(xml_path)

        if not self.data_path.exists():
            raise FileNotFoundError(f"Binary file not found: {self.data_path}")
        if not self.xml_path.exists():
            raise FileNotFoundError(f"XML label not found: {self.xml_path}")

        # Parse metadata
        self.meta = parse_pds4_label(self.xml_path)

        # Compute expected shape
        if self.meta.bands > 1:
            # IIRS: BSQ layout → (bands, lines, samples)
            self._shape: Tuple[int, ...] = (
                self.meta.bands, self.meta.lines, self.meta.samples
            )
        else:
            # OHRC / TMC-2: 2D image → (lines, samples)
            self._shape = (self.meta.lines, self.meta.samples)

        # Validate file size
        expected_bytes = int(np.prod(self._shape)) * self.meta.dtype.itemsize
        actual_bytes   = self.data_path.stat().st_size
        if actual_bytes < expected_bytes:
            raise ValueError(
                f"File size mismatch for {self.data_path.name}: "
                f"expected ≥{expected_bytes:,} bytes, got {actual_bytes:,} bytes. "
                f"Shape={self._shape}, dtype={self.meta.dtype}"
            )

        # Create memory map (read-only)
        self._mmap: np.ndarray = np.memmap(
            self.data_path,
            dtype=self.meta.dtype,
            mode="r",
            shape=self._shape,
        )

        logger.info(
            "Loaded %s: shape=%s, dtype=%s, res=%.2f m/px, sun_el=%.1f°",
            self.meta.instrument, self._shape, self.meta.dtype,
            self.meta.pixel_resolution_m, self.meta.sun_elevation_deg,
        )

    # ── properties ───────────────────────────────────────────────────
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def instrument(self) -> str:
        return self.meta.instrument

    @property
    def is_hyperspectral(self) -> bool:
        return self.meta.bands > 1

    # ── 2D patch extraction (OHRC / TMC-2) ───────────────────────────
    def get_patch(
        self,
        center_line: int,
        center_sample: int,
        size: int = 2000,
    ) -> np.ndarray:
        """
        Extract a square patch centred at (center_line, center_sample).

        For 2D images (OHRC / TMC-2), returns a 2D array of shape (size, size).
        For 3D cubes (IIRS), returns ALL bands for the spatial patch:
        shape (bands, size, size).

        Out-of-bounds regions are zero-padded.

        Args:
            center_line:   Row centre (0-indexed).
            center_sample: Column centre (0-indexed).
            size:          Side length of the square patch.

        Returns:
            np.ndarray copy (not a view) of the extracted region.
        """
        half = size // 2

        if self.is_hyperspectral:
            n_bands, n_lines, n_samples = self._shape
        else:
            n_lines, n_samples = self._shape[:2]
            n_bands = 0  # sentinel

        # Compute source and destination slices with boundary clamping
        r0 = center_line  - half
        r1 = center_line  + half
        c0 = center_sample - half
        c1 = center_sample + half

        # Source clamping
        sr0 = max(r0, 0)
        sr1 = min(r1, n_lines)
        sc0 = max(c0, 0)
        sc1 = min(c1, n_samples)

        # Destination offsets (for zero-padding)
        dr0 = sr0 - r0
        dr1 = dr0 + (sr1 - sr0)
        dc0 = sc0 - c0
        dc1 = dc0 + (sc1 - sc0)

        if self.is_hyperspectral:
            patch = np.zeros((n_bands, size, size), dtype=self.meta.dtype)
            patch[:, dr0:dr1, dc0:dc1] = self._mmap[:, sr0:sr1, sc0:sc1]
        else:
            patch = np.zeros((size, size), dtype=self.meta.dtype)
            patch[dr0:dr1, dc0:dc1] = self._mmap[sr0:sr1, sc0:sc1]

        return patch

    # ── IIRS-specific: single band slice ─────────────────────────────
    def get_band(self, band_index: int) -> np.ndarray:
        """
        Extract a single spectral band from the IIRS cube (zero-indexed).

        Returns a 2D array of shape (lines, samples).
        """
        if not self.is_hyperspectral:
            raise TypeError(f"{self.instrument} is not a hyperspectral cube.")
        if band_index < 0 or band_index >= self.meta.bands:
            raise IndexError(
                f"Band index {band_index} out of range [0, {self.meta.bands})"
            )
        return np.array(self._mmap[band_index, :, :])

    def get_band_by_wavelength(self, target_nm: float) -> Tuple[int, np.ndarray]:
        """
        Extract the band closest to `target_nm` wavelength.

        Returns (band_index, 2D_array).
        """
        if self.meta.band_centers_nm is None:
            raise ValueError("No band center wavelengths available in metadata.")
        centers = np.array(self.meta.band_centers_nm)
        idx = int(np.argmin(np.abs(centers - target_nm)))
        logger.info(
            "Requested %.1f nm → Band %d (%.1f nm)", target_nm, idx, centers[idx]
        )
        return idx, self.get_band(idx)

    # ── Thumbnail (for quick preview) ────────────────────────────────
    def get_thumbnail(self, max_dim: int = 1000) -> np.ndarray:
        """
        Return a downsampled full-image thumbnail for quick visualisation.

        For IIRS, uses the first principal band (~1285 nm, band 34).
        """
        if self.is_hyperspectral:
            img = self.get_band(33)  # band 34, 0-indexed
        else:
            n_lines, n_samples = self._shape[:2]
            step = max(1, max(n_lines, n_samples) // max_dim)
            img = np.array(self._mmap[::step, ::step])
        return img

    def __repr__(self) -> str:
        return (
            f"Chandrayaan2Loader("
            f"instrument={self.instrument!r}, "
            f"shape={self.shape}, "
            f"dtype={self.meta.dtype}, "
            f"res={self.meta.pixel_resolution_m} m/px)"
        )
