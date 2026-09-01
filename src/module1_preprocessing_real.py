"""
TriNetra Phase 1: Multi-Modal Preprocessing for Real Chandrayaan-2 Data.

Handles:
    - OHRC (uint8 panchromatic):  Shadow-aware CLAHE
    - TMC-2 (uint16 panchromatic): Percentile stretch → uint8 → CLAHE
    - IIRS (uint16 hyperspectral):  Band-34 (~1285 nm NIR) extraction as 2D proxy

All functions accept raw numpy arrays and return float32 [0, 1] preprocessed images.

Author : TriNetra Team (SIH26166)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedPatch:
    """Container for a preprocessed image patch with quality diagnostics."""
    image: np.ndarray            # float32, [0, 1], 2D
    instrument: str
    original_dtype: str
    original_shape: tuple
    clip_limit: float
    tile_size: int
    dynamic_range: float         # max - min of the raw patch
    shadow_fraction: float       # fraction of pixels < 5th percentile
    saturation_fraction: float   # fraction of pixels > 95th percentile


# ─────────────────────────────────────────────────────────────────────
# OHRC Preprocessor (uint8, visible panchromatic)
# ─────────────────────────────────────────────────────────────────────
class OHRCRealPreprocessor:
    """
    Shadow-aware CLAHE preprocessing for OHRC 0.26 m/px images.

    The OHRC data from PRADAN is calibrated uint8.  South-polar images
    (sun_elevation ~9°) have extreme shadow coverage.  Standard histogram
    equalisation destroys shadow-edge information critical for crater
    detection.

    Strategy:
        1. Adaptive clip-limit selection based on shadow coverage.
        2. Small CLAHE tile size (16×16) to preserve local shadow structure.
        3. Bilateral denoising to suppress sensor noise while preserving edges.
        4. Final normalisation to float32 [0, 1].
    """

    def __init__(
        self,
        base_clip_limit: float = 3.0,
        tile_size: int = 16,
        denoise_d: int = 5,
        denoise_sigma_color: float = 30.0,
        denoise_sigma_space: float = 30.0,
    ) -> None:
        self.base_clip_limit = base_clip_limit
        self.tile_size = tile_size
        self.denoise_d = denoise_d
        self.denoise_sigma_color = denoise_sigma_color
        self.denoise_sigma_space = denoise_sigma_space

    def preprocess(self, patch: np.ndarray) -> PreprocessedPatch:
        """
        Preprocess an OHRC uint8 patch.

        Args:
            patch: 2D uint8 array (e.g., 2000×2000 from the loader).

        Returns:
            PreprocessedPatch with float32 [0, 1] image.
        """
        if patch.ndim != 2:
            raise ValueError(f"Expected 2D patch, got {patch.ndim}D")

        raw = patch.astype(np.uint8)
        orig_shape = raw.shape

        # Compute diagnostics on raw data
        p5  = np.percentile(raw, 5)
        p95 = np.percentile(raw, 95)
        shadow_frac = float(np.mean(raw < p5))
        sat_frac    = float(np.mean(raw > p95))
        dyn_range   = float(p95 - p5)

        # Adaptive clip limit: boost for heavily shadowed images
        if dyn_range < 30:
            clip = self.base_clip_limit * 2.0  # Very low contrast
        elif shadow_frac > 0.3:
            clip = self.base_clip_limit * 1.5  # Heavy shadows
        else:
            clip = self.base_clip_limit

        # Step 1: Bilateral denoise (edge-preserving)
        denoised = cv2.bilateralFilter(
            raw, self.denoise_d,
            self.denoise_sigma_color, self.denoise_sigma_space,
        )

        # Step 2: CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=clip,
            tileGridSize=(self.tile_size, self.tile_size),
        )
        enhanced = clahe.apply(denoised)

        # Step 3: Normalise to float32 [0, 1]
        result = enhanced.astype(np.float32) / 255.0

        logger.info(
            "OHRC preprocess: shape=%s, clip=%.1f, shadow=%.1f%%, sat=%.1f%%",
            orig_shape, clip, shadow_frac * 100, sat_frac * 100,
        )

        return PreprocessedPatch(
            image=result,
            instrument="OHRC",
            original_dtype="uint8",
            original_shape=orig_shape,
            clip_limit=clip,
            tile_size=self.tile_size,
            dynamic_range=dyn_range,
            shadow_fraction=shadow_frac,
            saturation_fraction=sat_frac,
        )


# ─────────────────────────────────────────────────────────────────────
# TMC-2 Preprocessor (uint16 LE, visible panchromatic)
# ─────────────────────────────────────────────────────────────────────
class TMC2RealPreprocessor:
    """
    Preprocessing for TMC-2 5.97 m/px images (uint16, calibrated).

    Strategy:
        1. Robust percentile-based stretch (1st–99th percentile → 0–255).
        2. Convert to uint8 for CLAHE processing.
        3. CLAHE with moderate clip limit and larger tiles (32×32).
        4. Gaussian denoising.
        5. Normalise to float32 [0, 1].
    """

    def __init__(
        self,
        clip_limit: float = 2.5,
        tile_size: int = 32,
        stretch_low: float = 1.0,
        stretch_high: float = 99.0,
        gaussian_ksize: int = 3,
    ) -> None:
        self.clip_limit = clip_limit
        self.tile_size = tile_size
        self.stretch_low = stretch_low
        self.stretch_high = stretch_high
        self.gaussian_ksize = gaussian_ksize

    def preprocess(self, patch: np.ndarray) -> PreprocessedPatch:
        """
        Preprocess a TMC-2 uint16 patch.

        Args:
            patch: 2D uint16 array.

        Returns:
            PreprocessedPatch with float32 [0, 1] image.
        """
        if patch.ndim != 2:
            raise ValueError(f"Expected 2D patch, got {patch.ndim}D")

        raw = patch.astype(np.float64)
        orig_shape = patch.shape

        # Step 1: Robust percentile stretch → [0, 255]
        lo = np.percentile(raw, self.stretch_low)
        hi = np.percentile(raw, self.stretch_high)
        dyn_range = float(hi - lo)

        if dyn_range < 1e-6:
            stretched = np.zeros_like(raw, dtype=np.uint8)
        else:
            stretched = np.clip((raw - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

        shadow_frac = float(np.mean(raw < lo))
        sat_frac    = float(np.mean(raw > hi))

        # Step 2: CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(self.tile_size, self.tile_size),
        )
        enhanced = clahe.apply(stretched)

        # Step 3: Gentle Gaussian blur
        if self.gaussian_ksize > 1:
            enhanced = cv2.GaussianBlur(
                enhanced, (self.gaussian_ksize, self.gaussian_ksize), 0
            )

        # Step 4: Normalise to float32 [0, 1]
        result = enhanced.astype(np.float32) / 255.0

        logger.info(
            "TMC-2 preprocess: shape=%s, stretch=[%.0f, %.0f], clip=%.1f",
            orig_shape, lo, hi, self.clip_limit,
        )

        return PreprocessedPatch(
            image=result,
            instrument="TMC-2",
            original_dtype="uint16",
            original_shape=orig_shape,
            clip_limit=self.clip_limit,
            tile_size=self.tile_size,
            dynamic_range=dyn_range,
            shadow_fraction=shadow_frac,
            saturation_fraction=sat_frac,
        )


# ─────────────────────────────────────────────────────────────────────
# IIRS Preprocessor (uint16 LE, 256-band hyperspectral cube)
# ─────────────────────────────────────────────────────────────────────
class IIRSRealPreprocessor:
    """
    IIRS 68.38 m/px hyperspectral cube preprocessing.

    Converts the 256-band cube into a single 2D proxy image for
    cross-modal matching against TMC-2.

    Strategies:
        1. **Single-band extraction** (default): Band 34 (~1285 nm NIR).
           This band offers the best crater-rim contrast in the near-infrared.
        2. **PCA proxy**: First principal component across selected bands.
    """

    def __init__(
        self,
        proxy_band_index: int = 33,   # 0-indexed → band 34 = 1285.3 nm
        use_pca: bool = False,
        pca_band_range: tuple = (0, 85),  # bands 1–85 (~800–2500 nm)
        clip_limit: float = 3.0,
        tile_size: int = 8,
    ) -> None:
        self.proxy_band_index = proxy_band_index
        self.use_pca = use_pca
        self.pca_band_range = pca_band_range
        self.clip_limit = clip_limit
        self.tile_size = tile_size

    def preprocess(
        self,
        cube: Optional[np.ndarray] = None,
        band_2d: Optional[np.ndarray] = None,
    ) -> PreprocessedPatch:
        """
        Generate a 2D proxy image from the IIRS cube.

        Args:
            cube:    Full 3D cube (bands, lines, samples). If provided and
                     use_pca=True, PCA is used.
            band_2d: Pre-extracted single band (lines, samples). Faster path
                     when only the proxy band is needed.

        Returns:
            PreprocessedPatch with float32 [0, 1] 2D image.
        """
        if band_2d is not None:
            raw_2d = band_2d.astype(np.float64)
            method = f"Band {self.proxy_band_index + 1}"
        elif cube is not None:
            if self.use_pca:
                raw_2d = self._pca_proxy(cube)
                method = "PCA"
            else:
                raw_2d = cube[self.proxy_band_index].astype(np.float64)
                method = f"Band {self.proxy_band_index + 1}"
        else:
            raise ValueError("Provide either `cube` or `band_2d`.")

        orig_shape = raw_2d.shape

        # Robust percentile stretch → uint8
        lo = np.percentile(raw_2d, 2)
        hi = np.percentile(raw_2d, 98)
        dyn_range = float(hi - lo)

        if dyn_range < 1e-6:
            stretched = np.zeros(raw_2d.shape, dtype=np.uint8)
        else:
            stretched = np.clip(
                (raw_2d - lo) / (hi - lo) * 255, 0, 255
            ).astype(np.uint8)

        shadow_frac = float(np.mean(raw_2d < lo))
        sat_frac    = float(np.mean(raw_2d > hi))

        # CLAHE (small tiles for the 2264 × 250 image)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(self.tile_size, self.tile_size),
        )
        enhanced = clahe.apply(stretched)

        result = enhanced.astype(np.float32) / 255.0

        logger.info(
            "IIRS preprocess (%s): shape=%s, stretch=[%.0f, %.0f]",
            method, orig_shape, lo, hi,
        )

        return PreprocessedPatch(
            image=result,
            instrument="IIRS",
            original_dtype="uint16",
            original_shape=orig_shape,
            clip_limit=self.clip_limit,
            tile_size=self.tile_size,
            dynamic_range=dyn_range,
            shadow_fraction=shadow_frac,
            saturation_fraction=sat_frac,
        )

    def _pca_proxy(self, cube: np.ndarray) -> np.ndarray:
        """
        Compute the first principal component across selected bands.

        Args:
            cube: (bands, lines, samples) uint16 array.

        Returns:
            2D float64 array (lines, samples).
        """
        b_lo, b_hi = self.pca_band_range
        subset = cube[b_lo:b_hi].astype(np.float64)  # (n_bands, lines, samples)
        n_bands, n_lines, n_samples = subset.shape

        # Reshape to (n_pixels, n_bands)
        pixels = subset.reshape(n_bands, -1).T  # (n_pixels, n_bands)

        # Centre
        mean = pixels.mean(axis=0)
        centred = pixels - mean

        # Covariance → leading eigenvector
        cov = np.cov(centred, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        pc1 = eigenvectors[:, -1]  # largest eigenvalue is last

        # Project
        projected = centred @ pc1  # (n_pixels,)
        return projected.reshape(n_lines, n_samples)
