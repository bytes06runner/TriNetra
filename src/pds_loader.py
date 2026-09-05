"""
Chandrayaan-2 PDS4 Zero-Copy Memory-Mapped Data Loader.

Provides dynamic XML label parsing, memory-mapped array loading for massive
TMC-2 (2D) and IIRS (3D) calibrated binaries, sub-2000nm visible proxy image
generation, and fast window cropping.

SIH26166 — Chandrayaan-2 Multi-modal Image Correspondence Pipeline
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

import numpy as np

logger = logging.getLogger(__name__)


# Mapping from PDS4 data_type string to numpy dtype string
PDS_DTYPE_MAP: Dict[str, str] = {
    "UnsignedLSB2": "<u2",         # 16-bit unsigned little-endian (TMC-2)
    "SignedLSB2": "<i2",           # 16-bit signed little-endian
    "UnsignedByte": "uint8",        # 8-bit unsigned
    "IEEE754LSBSingle": "<f4",     # 32-bit float little-endian (IIRS)
    "IEEE754MSBSingle": ">f4",     # 32-bit float big-endian
    "IEEE754LSBDouble": "<f8",     # 64-bit float little-endian
}


def _find_tag(elem: ET.Element, tag: str) -> Optional[ET.Element]:
    """Helper to find any element matching `tag` regardless of XML namespace."""
    for child in elem.iter():
        if child.tag.endswith("}" + tag) or child.tag == tag:
            return child
    return None


def _findall_tags(elem: ET.Element, tag: str) -> List[ET.Element]:
    """Helper to find all elements matching `tag` regardless of XML namespace."""
    results: List[ET.Element] = []
    for child in elem.iter():
        if child.tag.endswith("}" + tag) or child.tag == tag:
            results.append(child)
    return results


def parse_label(xml_path: Union[str, Path]) -> Dict[str, Any]:
    """Dynamically parse a Chandrayaan-2 PDS4 XML label.

    Extracts dimension element counts, data types, pixel resolutions,
    illumination geometries, corner coordinates, file sizes, and band bins.

    Args:
        xml_path: Path to the XML label file.

    Returns:
        Dictionary with extracted metadata:
            - elements: List[int] in axis order (e.g. [lines, samples] or [bands, lines, samples])
            - axis_names: List[str] in sequence order
            - data_type: str (e.g. 'UnsignedLSB2', 'IEEE754LSBSingle')
            - numpy_dtype: str (e.g. '<u2', '<f4')
            - pixel_resolution: float (meters/pixel)
            - solar_incidence: float (degrees)
            - sun_azimuth: float (degrees)
            - sun_elevation: float (degrees, if present)
            - area: str (e.g. 'North Pole')
            - refined_corners: Dict[str, float] with upper_left_latitude, etc.
            - corners: List[Tuple[str, float, float]] (corner_name, lat, lon)
            - file_size: int (file size in bytes)
            - file_name: str
            - bands: List[Tuple[int, float]] (band_number, center_wavelength_nm)
    """
    path = Path(xml_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"XML label not found at {path}")

    tree = ET.parse(str(path))
    root = tree.getroot()

    # File size and name
    file_elem = _find_tag(root, "File")
    file_size = 0
    file_name = ""
    if file_elem is not None:
        fs = _find_tag(file_elem, "file_size")
        if fs is not None and fs.text:
            file_size = int(fs.text.strip())
        fn = _find_tag(file_elem, "file_name")
        if fn is not None and fn.text:
            file_name = fn.text.strip()

    # Data type
    data_type = ""
    elem_arr = _find_tag(root, "Element_Array")
    if elem_arr is not None:
        dt = _find_tag(elem_arr, "data_type")
        if dt is not None and dt.text:
            data_type = dt.text.strip()

    # Axis array elements in sequence order
    axis_arrays = _findall_tags(root, "Axis_Array")
    axis_info = []
    for a in axis_arrays:
        seq_elem = _find_tag(a, "sequence_number")
        name_elem = _find_tag(a, "axis_name")
        elem_elem = _find_tag(a, "elements")

        seq = int(seq_elem.text.strip()) if (seq_elem is not None and seq_elem.text) else 0
        name = name_elem.text.strip() if (name_elem is not None and name_elem.text) else ""
        elem_cnt = int(elem_elem.text.strip()) if (elem_elem is not None and elem_elem.text) else 0
        axis_info.append((seq, name, elem_cnt))

    axis_info.sort(key=lambda x: x[0])
    elements = [x[2] for x in axis_info]
    axis_names = [x[1] for x in axis_info]

    # Product parameters (ISDA namespace)
    pixel_resolution = 0.0
    pr_elem = _find_tag(root, "pixel_resolution")
    if pr_elem is not None and pr_elem.text:
        pixel_resolution = float(pr_elem.text.strip())

    solar_incidence = 0.0
    si_elem = _find_tag(root, "solar_incidence")
    if si_elem is not None and si_elem.text:
        solar_incidence = float(si_elem.text.strip())

    sun_azimuth = 0.0
    sa_elem = _find_tag(root, "sun_azimuth")
    if sa_elem is not None and sa_elem.text:
        sun_azimuth = float(sa_elem.text.strip())

    sun_elevation = 0.0
    se_elem = _find_tag(root, "sun_elevation")
    if se_elem is not None and se_elem.text:
        sun_elevation = float(se_elem.text.strip())

    area = ""
    area_elem = _find_tag(root, "area")
    if area_elem is not None and area_elem.text:
        area = area_elem.text.strip()

    # Refined Corner Coordinates
    refined_corners: Dict[str, float] = {}
    corners_list: List[Tuple[str, float, float]] = []
    refined_elem = _find_tag(root, "Refined_Corner_Coordinates")
    if refined_elem is None:
        refined_elem = _find_tag(root, "System_Level_Coordinates")

    if refined_elem is not None:
        for child in refined_elem:
            tag_name = child.tag.split("}")[-1]
            if child.text:
                refined_corners[tag_name] = float(child.text.strip())

        # Extract structured 4 pairs
        corner_prefixes = ["upper_left", "upper_right", "lower_left", "lower_right"]
        for pfx in corner_prefixes:
            lat_key = f"{pfx}_latitude"
            lon_key = f"{pfx}_longitude"
            if lat_key in refined_corners and lon_key in refined_corners:
                corners_list.append((pfx, refined_corners[lat_key], refined_corners[lon_key]))

    # Band Bin Set for 3D hyperspectral products
    band_bins = _findall_tags(root, "Band_Bin")
    bands: List[Tuple[int, float]] = []
    for b in band_bins:
        num_elem = _find_tag(b, "band_number")
        wl_elem = _find_tag(b, "center_wavelength")
        if num_elem is not None and num_elem.text and wl_elem is not None and wl_elem.text:
            bands.append((int(num_elem.text.strip()), float(wl_elem.text.strip())))

    bands.sort(key=lambda x: x[0])

    meta: Dict[str, Any] = {
        "xml_path": str(path),
        "file_name": file_name,
        "file_size": file_size,
        "data_type": data_type,
        "numpy_dtype": PDS_DTYPE_MAP.get(data_type, data_type),
        "elements": elements,
        "axis_names": axis_names,
        "pixel_resolution": pixel_resolution,
        "solar_incidence": solar_incidence,
        "sun_azimuth": sun_azimuth,
        "sun_elevation": sun_elevation,
        "area": area,
        "refined_corners": refined_corners,
        "corners": corners_list,
        "bands": bands,
    }

    return meta


def load_tmc2(
    img_path: Union[str, Path],
    xml_path: Union[str, Path],
) -> Tuple[np.memmap, Dict[str, Any]]:
    """Open a Chandrayaan-2 TMC-2 image using zero-copy memory mapping.

    TMC-2 is stored as a headerless 2D binary raster with offset 0 and
    dtype '<u2' (UnsignedLSB2, 16-bit unsigned little endian).

    Args:
        img_path: Path to the .img binary file.
        xml_path: Path to the corresponding .xml PDS4 label.

    Returns:
        (memmap_array, metadata_dict) where memmap_array has shape (lines, samples).

    Raises:
        ValueError: If lines * samples * 2 != file_size from label.
        FileNotFoundError: If files do not exist.
    """
    img_p = Path(img_path).resolve()
    xml_p = Path(xml_path).resolve()

    if not img_p.exists():
        raise FileNotFoundError(f"TMC-2 image binary not found at {img_p}")

    meta = parse_label(xml_p)
    elements = meta["elements"]
    if len(elements) < 2:
        raise ValueError(f"Expected at least 2 axes in TMC-2 label, got {elements}")

    lines, samples = elements[0], elements[1]
    expected_bytes = lines * samples * 2
    actual_file_size = meta["file_size"]

    if actual_file_size > 0 and expected_bytes != actual_file_size:
        raise ValueError(
            f"TMC-2 file size mismatch: {lines} lines × {samples} samples × 2 bytes "
            f"= {expected_bytes} bytes, but XML label specifies file_size = {actual_file_size} bytes."
        )

    on_disk_size = img_p.stat().st_size
    if on_disk_size != actual_file_size:
        raise ValueError(
            f"TMC-2 on-disk file size mismatch: {on_disk_size} bytes on disk vs "
            f"{actual_file_size} bytes specified in XML label."
        )

    dtype = meta["numpy_dtype"]
    if dtype != "<u2":
        logger.warning("Expected '<u2' for TMC-2, found '%s' in label.", dtype)

    # Open with zero-copy read-only memory map
    mmap = np.memmap(str(img_p), dtype="<u2", mode="r", shape=(lines, samples))
    logger.info("Loaded TMC-2 memmap: shape=%s, dtype=%s from %s", mmap.shape, mmap.dtype, img_p.name)

    return mmap, meta


def load_iirs(
    qub_path: Union[str, Path],
    xml_path: Union[str, Path],
) -> Tuple[np.memmap, Dict[str, Any]]:
    """Open a Chandrayaan-2 IIRS hyperspectral cube using zero-copy memory mapping.

    IIRS is stored as a headerless 3D binary cube at offset 0 with
    dtype '<f4' (IEEE754LSBSingle, 32-bit float little endian).
    The storage order is Band-Sequential (BSQ): (bands, lines, samples).

    Args:
        qub_path: Path to the .qub binary file.
        xml_path: Path to the corresponding .xml PDS4 label.

    Returns:
        (memmap_array, metadata_dict) where memmap_array has shape (bands, lines, samples).

    Raises:
        ValueError: If bands * lines * samples * 4 != file_size from label.
        FileNotFoundError: If files do not exist.
    """
    qub_p = Path(qub_path).resolve()
    xml_p = Path(xml_path).resolve()

    if not qub_p.exists():
        raise FileNotFoundError(f"IIRS cube binary not found at {qub_p}")

    meta = parse_label(xml_p)
    elements = meta["elements"]
    if len(elements) < 3:
        raise ValueError(f"Expected 3 axes in IIRS label, got {elements}")

    bands, lines, samples = elements[0], elements[1], elements[2]
    expected_bytes = bands * lines * samples * 4
    actual_file_size = meta["file_size"]

    if actual_file_size > 0 and expected_bytes != actual_file_size:
        raise ValueError(
            f"IIRS file size mismatch: {bands} bands × {lines} lines × {samples} samples × 4 bytes "
            f"= {expected_bytes} bytes, but XML label specifies file_size = {actual_file_size} bytes."
        )

    on_disk_size = qub_p.stat().st_size
    if on_disk_size != actual_file_size:
        raise ValueError(
            f"IIRS on-disk file size mismatch: {on_disk_size} bytes on disk vs "
            f"{actual_file_size} bytes specified in XML label."
        )

    dtype = meta["numpy_dtype"]
    if dtype != "<f4":
        logger.warning("Expected '<f4' for IIRS, found '%s' in label.", dtype)

    # Open with zero-copy read-only memory map (Band-Sequential)
    mmap = np.memmap(str(qub_p), dtype="<f4", mode="r", shape=(bands, lines, samples))
    logger.info("Loaded IIRS memmap: shape=%s, dtype=%s from %s", mmap.shape, mmap.dtype, qub_p.name)

    return mmap, meta


def iirs_to_grey(
    cube: Union[np.ndarray, np.memmap],
    band_wavelengths: List[Tuple[int, float]],
    line_slice: slice,
    sample_slice: slice,
    max_nm: float = 2000.0,
) -> np.ndarray:
    """Generate a 2D visible-proxy grayscale image from an IIRS hyperspectral cube.

    CRITICAL PHYSICAL CONSTRAINTS:
    Bands with center wavelengths above 2000 nm are dominated by thermal emission
    from the lunar regolith rather than reflected solar radiation. Including
    thermal emission produces an inverted/distorted radiometric signal that destroys
    cross-modal structural correspondence with visible optical sensors.

    Therefore, this function:
    1. Selects only bands where center wavelength <= max_nm (bands 1-77, up to 1993.1 nm).
    2. Slices ONLY the requested line and sample window directly from the memory map.
       The full 2.4 GB cube is NEVER materialized into RAM.
    3. Averages the selected reflected-solar bands across axis 0.
    4. Applies percentile clipping at 2% and 98% and scales to uint8 [0, 255].

    Args:
        cube: 3D array or memmap of shape (bands, lines, samples).
        band_wavelengths: List of (band_number, center_wavelength_nm) pairs.
        line_slice: Slice for lines (e.g. slice(line0, line1)).
        sample_slice: Slice for samples (e.g. slice(samp0, samp1)).
        max_nm: Maximum cutoff wavelength in nanometers (default 2000.0 nm).

    Returns:
        2D uint8 numpy array of shape (line_len, sample_len).
    """
    # Identify indices of bands with wavelength <= max_nm
    # band_wavelengths is 1-indexed in band_number, but index in array is 0-indexed
    selected_indices: List[int] = []
    for idx, (band_num, wl_nm) in enumerate(band_wavelengths):
        if wl_nm <= max_nm:
            selected_indices.append(idx)

    if not selected_indices:
        # Fallback: if no wavelength info or metadata missing, take first 77 bands
        logger.warning("No bands matched <= %f nm; falling back to first 77 bands.", max_nm)
        selected_indices = list(range(min(77, cube.shape[0])))

    # Slice only the requested sub-cube from memmap
    # If selected_indices is contiguous from 0 to K, slice directly for maximum speed
    k = len(selected_indices)
    is_contiguous_prefix = (selected_indices == list(range(k)))

    if is_contiguous_prefix:
        sub_cube = cube[:k, line_slice, sample_slice]
    else:
        sub_cube = cube[selected_indices, line_slice, sample_slice]

    # Convert to float32 in memory and average across bands
    sub_cube_f32 = np.asarray(sub_cube, dtype=np.float32)
    mean_img = np.nanmean(sub_cube_f32, axis=0)

    # Robust percentile normalization to uint8 [0, 255]
    # Handle NaN or Inf if present in detector dead pixels
    mean_img = np.nan_to_num(mean_img, nan=0.0, posinf=0.0, neginf=0.0)

    p2, p98 = np.percentile(mean_img, (2.0, 98.0))
    spread = p98 - p2

    if spread > 1e-6:
        norm = np.clip((mean_img - p2) / spread * 255.0, 0, 255).astype(np.uint8)
    else:
        norm = np.zeros_like(mean_img, dtype=np.uint8)

    return norm


def crop(
    arr: Union[np.ndarray, np.memmap],
    line0: int,
    line1: int,
    samp0: int,
    samp1: int,
) -> np.ndarray:
    """Slice a window from a 2D array or memmap and return a real in-memory copy.

    Args:
        arr: 2D array or memmap of shape (lines, samples).
        line0: Start line index (inclusive).
        line1: End line index (exclusive).
        samp0: Start sample index (inclusive).
        samp1: End sample index (exclusive).

    Returns:
        In-memory numpy ndarray copy of shape (line1 - line0, samp1 - samp0).
    """
    h, w = arr.shape[:2]
    # Clamp coordinates to valid array bounds
    l0 = max(0, min(line0, h))
    l1 = max(0, min(line1, h))
    s0 = max(0, min(samp0, w))
    s1 = max(0, min(samp1, w))

    # Slice and explicitly force in-memory copy via np.array
    return np.array(arr[l0:l1, s0:s1])
