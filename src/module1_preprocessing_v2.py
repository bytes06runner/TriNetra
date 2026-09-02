import numpy as np
import cv2
from dataclasses import dataclass

@dataclass
class PreprocessedResult:
    """Container for preprocessed image and quality metadata."""
    image: np.ndarray            # uint8, [0, 255]
    original_dtype: np.dtype
    original_shape: tuple[int, ...]
    dynamic_range: float         # hi - lo percentile spread (pre-CLAHE)
    shadow_fraction: float       # fraction of pixels below shadow threshold
    clip_limit: float            # CLAHE clip limit used
    method: str                  # description of preprocessing applied


def percentile_stretch(image: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """
    Normalize an arbitrary-dtype image to uint8 [0, 255] using percentile stretch.
    
    1. Compute lo-th and hi-th percentiles
    2. Clip image to [p_lo, p_hi]
    3. Linearly scale to [0, 255]
    4. Cast to uint8
    """
    p_lo = np.percentile(image, lo)
    p_hi = np.percentile(image, hi)
    
    clipped = np.clip(image, p_lo, p_hi)
    
    range_val = p_hi - p_lo
    if range_val == 0:
        range_val = 1.0
        
    scaled = (clipped - p_lo) / range_val * 255.0
    return scaled.astype(np.uint8)

def shadow_aware_clahe(
    image_u8: np.ndarray,
    clip_limit: float = 3.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    shadow_threshold: int = 15,
) -> tuple[np.ndarray, float]:
    """
    Apply CLAHE to a uint8 image.
    
    1. Create CLAHE object with given clip_limit and tile_grid_size
    2. Apply to image
    3. Compute shadow_fraction = (pixels below shadow_threshold) / total_pixels
    
    Returns (clahe_image, shadow_fraction).
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_img = clahe.apply(image_u8)
    
    shadow_pixels = np.sum(clahe_img < shadow_threshold)
    total_pixels = clahe_img.size
    shadow_fraction = shadow_pixels / total_pixels
    
    return clahe_img, float(shadow_fraction)

def synthesize_tmc2(
    ohrc_patch: np.ndarray,
    ohrc_gsd: float,
    tmc2_target_gsd: float = 5.0,
) -> np.ndarray:
    """
    Synthesize a TMC-2 proxy by downsampling an OHRC patch.
    
    1. Compute scale_factor = tmc2_target_gsd / ohrc_gsd
    2. Compute new dimensions = (original / scale_factor), rounded to int
    3. Use cv2.resize with INTER_AREA (anti-aliased downsampling)
    4. Result is float32 or uint8 depending on input
    """
    scale_factor = tmc2_target_gsd / ohrc_gsd
    new_h = int(np.round(ohrc_patch.shape[0] / scale_factor))
    new_w = int(np.round(ohrc_patch.shape[1] / scale_factor))
    
    resized = cv2.resize(ohrc_patch, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized

def preprocess_ohrc(raw_patch: np.ndarray) -> PreprocessedResult:
    """
    Full OHRC preprocessing pipeline:
    1. percentile_stretch to uint8
    2. shadow_aware_clahe
    3. Return PreprocessedResult
    """
    orig_shape = raw_patch.shape
    orig_dtype = raw_patch.dtype
    
    p_lo = np.percentile(raw_patch, 1.0)
    p_hi = np.percentile(raw_patch, 99.0)
    dynamic_range = float(p_hi - p_lo)
    
    stretched = percentile_stretch(raw_patch)
    clip_limit = 3.0
    clahe_img, shadow_fraction = shadow_aware_clahe(stretched, clip_limit=clip_limit)
    
    return PreprocessedResult(
        image=clahe_img,
        original_dtype=orig_dtype,
        original_shape=orig_shape,
        dynamic_range=dynamic_range,
        shadow_fraction=shadow_fraction,
        clip_limit=clip_limit,
        method="percentile_stretch + shadow_aware_clahe"
    )

def preprocess_iirs_band(raw_band: np.ndarray) -> PreprocessedResult:
    """
    Full IIRS single-band preprocessing pipeline:
    1. percentile_stretch to uint8
    2. shadow_aware_clahe
    3. Return PreprocessedResult
    """
    orig_shape = raw_band.shape
    orig_dtype = raw_band.dtype
    
    p_lo = np.percentile(raw_band, 1.0)
    p_hi = np.percentile(raw_band, 99.0)
    dynamic_range = float(p_hi - p_lo)
    
    stretched = percentile_stretch(raw_band)
    clip_limit = 3.0
    clahe_img, shadow_fraction = shadow_aware_clahe(stretched, clip_limit=clip_limit)
    
    return PreprocessedResult(
        image=clahe_img,
        original_dtype=orig_dtype,
        original_shape=orig_shape,
        dynamic_range=dynamic_range,
        shadow_fraction=shadow_fraction,
        clip_limit=clip_limit,
        method="percentile_stretch + shadow_aware_clahe"
    )

def preprocess_synthetic_tmc2(
    ohrc_patch: np.ndarray,
    ohrc_gsd: float,
    tmc2_target_gsd: float = 5.0,
) -> PreprocessedResult:
    """
    Full synthetic TMC-2 pipeline:
    1. percentile_stretch the OHRC patch
    2. synthesize_tmc2 to downsample
    3. shadow_aware_clahe on the downsampled result
    4. Return PreprocessedResult
    """
    orig_shape = ohrc_patch.shape
    orig_dtype = ohrc_patch.dtype
    
    synth_patch = synthesize_tmc2(ohrc_patch, ohrc_gsd, tmc2_target_gsd)
    
    p_lo = np.percentile(synth_patch, 1.0)
    p_hi = np.percentile(synth_patch, 99.0)
    dynamic_range = float(p_hi - p_lo)
    
    stretched = percentile_stretch(synth_patch)
    clip_limit = 3.0
    clahe_img, shadow_fraction = shadow_aware_clahe(stretched, clip_limit=clip_limit)
    
    return PreprocessedResult(
        image=clahe_img,
        original_dtype=orig_dtype,
        original_shape=orig_shape,
        dynamic_range=dynamic_range,
        shadow_fraction=shadow_fraction,
        clip_limit=clip_limit,
        method="synthesize_tmc2 + percentile_stretch + shadow_aware_clahe"
    )
