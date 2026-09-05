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



def parse_envi_header(hdr_path: Union[str, Path]) -> Dict[str, Any]:
    """Parse an ENVI header file (.hdr) and extract dimensional and layout metadata.

    Returns a dict with:
        samples (int), lines (int), bands (int), header_offset (int),
        file_type (str), data_type (int), interleave (str: 'bsq', 'bil', 'bip'),
        byte_order (int: 0 for little-endian, 1 for big-endian).
    """
    path = Path(hdr_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"ENVI header file not found at {path}")

    info: Dict[str, Any] = {
        "samples": 0,
        "lines": 0,
        "bands": 0,
        "header_offset": 0,
        "file_type": "ENVI Standard",
        "data_type": 4,
        "interleave": "bsq",
        "byte_order": 0,
    }

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, val = [part.strip() for part in line.split("=", 1)]
            key_lower = key.lower().replace(" ", "_")
            val_lower = val.lower()

            if key_lower in ("samples", "lines", "bands", "header_offset", "data_type", "byte_order"):
                try:
                    info[key_lower] = int(val)
                except ValueError:
                    pass
            elif key_lower in ("interleave", "file_type"):
                info[key_lower] = val_lower

    return info


def load_iirs(
    qub_path: Union[str, Path],
    xml_path: Union[str, Path],
    hdr_path: Optional[Union[str, Path]] = None,
) -> Tuple[np.memmap, Dict[str, Any]]:
    """Open a Chandrayaan-2 IIRS hyperspectral cube using zero-copy memory mapping.

    IIRS is stored as a 3D binary cube at offset 0 with dtype '<f4'
    (IEEE754LSBSingle, 32-bit float little endian).
    If an ENVI .hdr header is present, its interleave definition takes precedence:
      - 'bsq' -> shape (bands, lines, samples)
      - 'bil' -> shape (lines, bands, samples)

    Args:
        qub_path: Path to the .qub binary file.
        xml_path: Path to the corresponding .xml PDS4 label.
        hdr_path: Optional path to the .hdr ENVI header file. If None, auto-checks qub_path.with_suffix('.hdr').

    Returns:
        (memmap_array, metadata_dict).

    Raises:
        ValueError: If file size does not match expected byte count.
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

    # Inspect ENVI header if present
    h_p = Path(hdr_path).resolve() if hdr_path else qub_p.with_suffix(".hdr")
    interleave = "bsq"
    if h_p.exists():
        envi_info = parse_envi_header(h_p)
        meta["envi"] = envi_info
        interleave = envi_info.get("interleave", "bsq").lower()
        samples = envi_info["samples"]
        lines = envi_info["lines"]
        bands = envi_info["bands"]
        dtype_num = envi_info["data_type"]
        byte_order = envi_info["byte_order"]

        print(f"ENVI Header Analysis for {qub_p.name}:")
        print(f"  Samples:    {samples}")
        print(f"  Lines:      {lines}")
        print(f"  Bands:      {bands}")
        print(f"  Interleave: {interleave}")
        print(f"  Data Type:  {dtype_num} (float32)")
        print(f"  Byte Order: {byte_order} ({'Little-Endian' if byte_order == 0 else 'Big-Endian'})")
    else:
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

    # Determine memmap shape based on interleave
    if interleave == "bil":
        shape = (lines, bands, samples)
    else:
        # BSQ (default)
        shape = (bands, lines, samples)

    meta["interleave"] = interleave
    meta["shape"] = shape
    print(f"  Configured Memmap Layout: {interleave.upper()} with shape {shape}")

    # Open with zero-copy read-only memory map
    mmap = np.memmap(str(qub_p), dtype="<f4", mode="r", shape=shape)
    logger.info("Loaded IIRS memmap: layout=%s, shape=%s, dtype=%s from %s", interleave.upper(), mmap.shape, mmap.dtype, qub_p.name)

    return mmap, meta


def iirs_to_grey(
    cube: Union[np.ndarray, np.memmap],
    band_wavelengths: List[Tuple[int, float]],
    line_slice: slice,
    sample_slice: slice,
    max_nm: float = 2000.0,
    save_diagnostics: bool = False,
    diag_dir: Union[str, Path] = "outputs",
    interleave: str = "bsq",
) -> np.ndarray:
    """Generate a 2D visible-proxy grayscale image from an IIRS hyperspectral cube.

    PHYSICAL PIPELINE:
    1. Slice sub-window directly from memmap first.
    2. Select bands with center wavelength below max_nm (<= 2000.0 nm).
    3. Per-band normalization: divide each band by its spatial mean so each band
       contributes equally regardless of solar spectral shape or detector responsivity.
    4. Column destriping: for each detector column (sample axis), compute its median
       across lines and normalize so all columns share a common median (standard pushbroom correction).
    5. Average normalized, destriped bands.
    6. Normalize to uint8 with 2/98 percentile clipping.

    Args:
        cube: 3D array or memmap. Shape depends on interleave:
              - 'bsq': (bands, lines, samples)
              - 'bil': (lines, bands, samples)
        band_wavelengths: List of (band_number, center_wavelength_nm) pairs.
        line_slice: Slice for lines (e.g. slice(line0, line1)).
        sample_slice: Slice for samples (e.g. slice(samp0, samp1)).
        max_nm: Maximum cutoff wavelength in nanometers (default 2000.0 nm).
        save_diagnostics: If True, saves four diagnostic PNGs.
        diag_dir: Directory to save diagnostic PNGs.
        interleave: 'bsq' or 'bil'.

    Returns:
        2D uint8 numpy array of shape (line_len, sample_len).
    """
    # 1 & 2: Select bands <= max_nm
    selected_indices: List[int] = []
    for idx, (band_num, wl_nm) in enumerate(band_wavelengths):
        if wl_nm <= max_nm:
            selected_indices.append(idx)

    total_bands = cube.shape[0] if interleave == "bsq" else cube.shape[1]
    if not selected_indices:
        logger.warning("No bands matched <= %f nm; falling back to first 77 bands.", max_nm)
        selected_indices = list(range(min(77, total_bands)))

    k = len(selected_indices)

    # Slice sub-cube first from memmap
    if interleave == "bsq":
        if selected_indices == list(range(k)):
            sub_raw = cube[:k, line_slice, sample_slice]
        else:
            sub_raw = cube[selected_indices, line_slice, sample_slice]
        sub = np.asarray(sub_raw, dtype=np.float32)
    elif interleave == "bil":
        if selected_indices == list(range(k)):
            sub_raw = cube[line_slice, :k, sample_slice]
        else:
            sub_raw = cube[line_slice, selected_indices, sample_slice]
        sub = np.transpose(np.asarray(sub_raw, dtype=np.float32), (1, 0, 2))
    else:
        raise ValueError(f"Unsupported interleave: {interleave}")

    # Replace NaNs/Infs from dead detector pixels
    sub = np.nan_to_num(sub, nan=0.0, posinf=0.0, neginf=0.0)

    # Raw band-77 slice (band index 76 or the last selected sub-2000nm band)
    raw_band77 = sub[-1].copy()

    # 3: Per-band normalization: divide each band by its own spatial mean
    band_means = np.nanmean(sub, axis=(1, 2), keepdims=True)
    band_means = np.where(np.abs(band_means) > 1e-8, band_means, 1.0)
    norm_bands = sub / band_means

    norm_avg_before = np.nanmean(norm_bands, axis=0)

    # 4: Column destriping: pushbroom stripe correction
    # For each detector column (sample axis = axis 2), compute median across lines (axis 1)
    col_meds = np.nanmedian(norm_bands, axis=1, keepdims=True)  # (B, 1, W)
    common_meds = np.nanmedian(col_meds, axis=2, keepdims=True) # (B, 1, 1)
    col_safe = np.where(np.abs(col_meds) > 1e-6, col_meds, 1.0)
    destriped_bands = norm_bands * (common_meds / col_safe)

    # 5: Average normalized, destriped bands
    avg_destriped = np.nanmean(destriped_bands, axis=0)
    # Composite destripe pass to align column medians across the average
    col_med_comp = np.nanmedian(avg_destriped, axis=0, keepdims=True)
    common_med_comp = np.nanmedian(col_med_comp)
    col_safe_comp = np.where(np.abs(col_med_comp) > 1e-6, col_med_comp, 1.0)
    avg_destriped = avg_destriped * (common_med_comp / col_safe_comp)

    # Column standard deviation metrics (measuring cross-column striping noise)
    std_cols_before = float(np.std(np.nanmedian(norm_avg_before, axis=0)))
    std_cols_after = float(np.std(np.nanmedian(avg_destriped, axis=0)))
    print(f"Standard deviation along columns before destriping: {std_cols_before:.6f}")
    print(f"Standard deviation along columns after destriping:  {std_cols_after:.6f}")

    # 6: Normalize to uint8 with 2/98 percentile clipping
    p2, p98 = np.percentile(avg_destriped, (2.0, 98.0))
    if p98 - p2 > 1e-6:
        final_proxy = np.clip((avg_destriped - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
    else:
        final_proxy = np.zeros_like(avg_destriped, dtype=np.uint8)

    # Save diagnostic images if requested
    if save_diagnostics:
        import cv2
        out_p = Path(diag_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        def _to_u8(arr: np.ndarray) -> np.ndarray:
            lo, hi = np.percentile(arr, (2.0, 98.0))
            if hi - lo > 1e-6:
                return np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
            return np.zeros_like(arr, dtype=np.uint8)

        p1 = out_p / "iirs_diag_raw_band77.png"
        p2_f = out_p / "iirs_diag_perband_norm_avg.png"
        p3 = out_p / "iirs_diag_after_destriping.png"
        p4 = out_p / "iirs_diag_final_proxy.png"

        cv2.imwrite(str(p1), _to_u8(raw_band77))
        cv2.imwrite(str(p2_f), _to_u8(norm_avg_before))
        cv2.imwrite(str(p3), _to_u8(avg_destriped))
        cv2.imwrite(str(p4), final_proxy)
        print(f"Saved 4 diagnostic PNGs to {out_p.resolve()}:")
        print(f"  - {p1.name}")
        print(f"  - {p2_f.name}")
        print(f"  - {p3.name}")
        print(f"  - {p4.name}")

    return final_proxy


def iirs_proxy_variants(
    cube: Union[np.ndarray, np.memmap],
    band_wavelengths: List[Tuple[int, float]],
    line_slice: slice,
    sample_slice: slice,
    save_dir: Union[str, Path] = "outputs",
    interleave: str = "bsq",
) -> Dict[str, np.ndarray]:
    """Produce three alternative IIRS visible proxy variants from the same window, all destriped:
    1. Single band nearest 1500 nm
    2. Mean of bands nearest 950, 1500 and 1700 nm
    3. First principal component (PC1) across the sub-2000 nm bands

    Args:
        cube: 3D array or memmap.
        band_wavelengths: List of (band_number, center_wavelength_nm) pairs.
        line_slice: Slice for lines.
        sample_slice: Slice for samples.
        save_dir: Directory to save the 3 variant PNGs.
        interleave: 'bsq' or 'bil'.

    Returns:
        Dict mapping variant name to uint8 proxy image:
        {'single_1500nm': img, 'mean_3band': img, 'pc1': img}
    """
    import cv2
    out_p = Path(save_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    def _destripe_2d(arr: np.ndarray) -> np.ndarray:
        col_med = np.nanmedian(arr, axis=0, keepdims=True)
        common_med = np.nanmedian(col_med)
        col_safe = np.where(np.abs(col_med) > 1e-6, col_med, 1.0)
        return arr * (common_med / col_safe)

    def _to_u8(arr: np.ndarray) -> np.ndarray:
        p2, p98 = np.percentile(arr, (2.0, 98.0))
        if p98 - p2 > 1e-6:
            return np.clip((arr - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
        return np.zeros_like(arr, dtype=np.uint8)

    def _get_band_slice(idx: int) -> np.ndarray:
        if interleave == "bsq":
            return np.asarray(cube[idx, line_slice, sample_slice], dtype=np.float32)
        else:
            return np.asarray(cube[line_slice, idx, sample_slice], dtype=np.float32)

    # 1. Single band nearest 1500 nm
    idx_1500 = min(range(len(band_wavelengths)), key=lambda i: abs(band_wavelengths[i][1] - 1500.0))
    b1500 = np.nan_to_num(_get_band_slice(idx_1500), nan=0.0)
    b1500_destriped = _destripe_2d(b1500)
    u8_1500 = _to_u8(b1500_destriped)

    # 2. Mean of bands nearest 950, 1500, and 1700 nm
    idx_950 = min(range(len(band_wavelengths)), key=lambda i: abs(band_wavelengths[i][1] - 950.0))
    idx_1700 = min(range(len(band_wavelengths)), key=lambda i: abs(band_wavelengths[i][1] - 1700.0))

    b950 = np.nan_to_num(_get_band_slice(idx_950), nan=0.0)
    b1700 = np.nan_to_num(_get_band_slice(idx_1700), nan=0.0)

    b950_n = _destripe_2d(b950 / (np.nanmean(b950) + 1e-8))
    b1500_n = _destripe_2d(b1500 / (np.nanmean(b1500) + 1e-8))
    b1700_n = _destripe_2d(b1700 / (np.nanmean(b1700) + 1e-8))

    b3_mean = (b950_n + b1500_n + b1700_n) / 3.0
    b3_destriped = _destripe_2d(b3_mean)
    u8_3band = _to_u8(b3_destriped)

    # 3. First principal component across sub-2000 nm bands
    sub_indices = [i for i, (_, wl) in enumerate(band_wavelengths) if wl <= 2000.0]
    if not sub_indices:
        sub_indices = list(range(min(77, len(band_wavelengths))))

    k = len(sub_indices)
    if interleave == "bsq":
        sub_cube = np.asarray(cube[sub_indices, line_slice, sample_slice], dtype=np.float32)
    else:
        sub_cube = np.transpose(np.asarray(cube[line_slice, sub_indices, sample_slice], dtype=np.float32), (1, 0, 2))

    sub_cube = np.nan_to_num(sub_cube, nan=0.0)
    b_means = np.nanmean(sub_cube, axis=(1, 2), keepdims=True)
    b_means = np.where(np.abs(b_means) > 1e-8, b_means, 1.0)
    sub_norm = sub_cube / b_means

    # Destripe all bands
    H, W = sub_norm.shape[1], sub_norm.shape[2]
    norm_destriped = np.zeros_like(sub_norm)
    for i in range(k):
        norm_destriped[i] = _destripe_2d(sub_norm[i])

    # SVD PCA
    X = norm_destriped.reshape(k, H * W)
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_centered = X - X_mean
    u, s, vt = np.linalg.svd(X_centered, full_matrices=False)
    pc1 = vt[0].reshape(H, W)

    # Orient sign so it positively correlates with mean reflectance
    mean_img = np.mean(norm_destriped, axis=0)
    corr = np.corrcoef(pc1.flatten(), mean_img.flatten())[0, 1]
    if corr < 0:
        pc1 = -pc1

    pc1_destriped = _destripe_2d(pc1)
    u8_pc1 = _to_u8(pc1_destriped)

    # Save all three
    p_1500 = out_p / "iirs_proxy_1500nm.png"
    p_3band = out_p / "iirs_proxy_3band_mean.png"
    p_pc1 = out_p / "iirs_proxy_pc1.png"

    cv2.imwrite(str(p_1500), u8_1500)
    cv2.imwrite(str(p_3band), u8_3band)
    cv2.imwrite(str(p_pc1), u8_pc1)

    print(f"Saved 3 proxy variants to {out_p.resolve()}:")
    print(f"  - {p_1500.name}")
    print(f"  - {p_3band.name}")
    print(f"  - {p_pc1.name}")

    return {
        "single_1500nm": u8_1500,
        "mean_3band": u8_3band,
        "pc1": u8_pc1,
    }



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
