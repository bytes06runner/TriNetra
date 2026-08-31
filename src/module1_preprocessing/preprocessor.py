"""
Main preprocessing module for Module 1 (Lunar Correspondence).
Handles per-instrument radiometric correction for OHRC and TMC-2.

The preprocessing aims to be robust to extreme lunar illumination
conditions, preparing images for downstream feature matching.
"""

import numpy as np
import cv2
import scipy.ndimage as ndimage
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any

@dataclass
class PreprocessingResult:
    """Data structure containing the result of a preprocessing operation."""
    image: np.ndarray          # Preprocessed 2D image, float32, [0, 1]
    instrument: str            # 'OHRC', 'TMC-2', 'IIRS'
    original_shape: Tuple[int, ...]
    preprocessed_shape: Tuple[int, ...]
    processing_steps: List[str]  # Log of operations applied
    quality_metrics: Dict[str, float]  # SNR, contrast, dynamic range etc.
    confidence: float          # 0.0-1.0, overall confidence the preprocessing succeeded


class BasePreprocessor(ABC):
    """Abstract base class for all instrument-specific preprocessors."""
    
    def __init__(self, target_dtype: type = np.float32):
        """
        Initialize the base preprocessor.
        
        Args:
            target_dtype: Desired data type for output array, usually np.float32.
        """
        self.target_dtype = target_dtype
        self.logger = logging.getLogger(self.__class__.__name__)
        self._processing_log: List[str] = []
    
    @abstractmethod
    def preprocess(self, image: np.ndarray, **kwargs) -> PreprocessingResult:
        """
        Execute the preprocessing pipeline.
        
        Args:
            image: Input image array.
            **kwargs: Additional optional parameters.
            
        Returns:
            PreprocessingResult object with the processed image and metadata.
        """
        pass
    
    def _validate_input(self, image: np.ndarray) -> np.ndarray:
        """
        Validate and normalize input.
        
        Accepts uint8, uint16, float32/64. Converts to float32 [0,1]. 
        Handles NaN/Inf. Logs warnings for unusual stats.
        
        Args:
            image: Input image array.
            
        Returns:
            Normalized float32 array in [0, 1].
        """
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a valid numpy array.")
            
        if image.ndim not in (2, 3):
            raise ValueError("Input image must be 2D or 3D.")
            
        if image.ndim == 3:
            if image.shape[2] == 1:
                image = image[:, :, 0]
            else:
                # Convert to grayscale if it has 3 channels
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                
        # Handle NaN and Inf
        if not np.isfinite(image).all():
            self.logger.warning("Input image contains NaN or Inf. Replacing with median.")
            median_val = np.nanmedian(image)
            image = np.nan_to_num(image, nan=median_val, posinf=median_val, neginf=median_val)
            
        # Convert to float32
        image_float = image.astype(np.float32)
        
        # Normalize to [0, 1] based on input type
        if image.dtype == np.uint8:
            image_float /= 255.0
        elif image.dtype == np.uint16:
            image_float /= 65535.0
        else:
            # For floats, rescale using min/max
            min_val = np.min(image_float)
            max_val = np.max(image_float)
            if max_val > min_val:
                image_float = (image_float - min_val) / (max_val - min_val)
            else:
                self.logger.warning("Input image has zero dynamic range.")
                image_float = np.zeros_like(image_float)
                
        return image_float
    
    def _compute_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """
        Compute image quality metrics such as SNR, contrast, dynamic range, etc.
        
        Args:
            image: 2D float32 image array in [0, 1].
            
        Returns:
            Dictionary containing computed quality metrics.
        """
        metrics = {}
        
        mean_val = np.mean(image)
        std_val = np.std(image)
        
        # SNR (Signal-to-Noise Ratio) estimated simply as mean / std
        metrics['snr'] = float(mean_val / (std_val + 1e-6))
        
        # Contrast (std/mean)
        metrics['contrast'] = float(std_val / (mean_val + 1e-6))
        
        # Dynamic range (max - min)
        metrics['dynamic_range'] = float(np.max(image) - np.min(image))
        
        # Entropy (from histogram)
        hist, _ = np.histogram(image.ravel(), bins=256, range=(0.0, 1.0))
        hist_prob = hist / hist.sum()
        hist_prob = hist_prob[hist_prob > 0]
        metrics['entropy'] = float(-np.sum(hist_prob * np.log2(hist_prob)))
        
        # Shadow fraction (fraction of pixels below 0.05)
        metrics['shadow_fraction'] = float(np.mean(image < 0.05))
        
        # Saturation fraction (fraction of pixels above 0.95)
        metrics['saturation_fraction'] = float(np.mean(image > 0.95))
        
        return metrics
    
    def _log_step(self, step_name: str):
        """
        Log a processing step and append it to the history.
        
        Args:
            step_name: Name or description of the completed step.
        """
        self._processing_log.append(step_name)
        self.logger.debug(f"Applied step: {step_name}")


class OHRCPreprocessor(BasePreprocessor):
    """
    OHRC-specific preprocessing with CLAHE tuned for shadow-edge preservation.
    
    OHRC (0.25 m/px) images contain extremely fine detail including small crater
    rims, boulder shadows, and subtle albedo variations. Standard histogram
    equalization destroys shadow edges critical for crater detection.
    
    The CLAHE implementation here uses:
    - Small tile size (16x16 or 32x32) to preserve local shadow structure
    - Moderate clip limit (2.0-3.0) to enhance without over-amplifying noise
    - Adaptive clip limit based on image statistics (higher for low-contrast/shadowed images)
    """
    
    def __init__(self, clip_limit: float = 2.5, tile_size: int = 16, adaptive_clip: bool = True):
        super().__init__()
        self.clip_limit = clip_limit
        self.tile_size = tile_size
        self.adaptive_clip = adaptive_clip
    
    def preprocess(self, image: np.ndarray, **kwargs) -> PreprocessingResult:
        """
        Full OHRC preprocessing pipeline:
        1. Validate and normalize input
        2. Apply shadow-aware denoising (bilateral filter to preserve edges)
        3. Compute adaptive CLAHE parameters based on image statistics
        4. Apply CLAHE with tuned parameters
        5. Apply gentle unsharp masking to restore edge sharpness
        6. Final normalization to [0, 1]
        7. Compute quality metrics
        8. Compute confidence score (low if >60% shadow or >20% saturated)
        """
        self._processing_log = []
        original_shape = image.shape
        
        # 1. Validate and normalize input
        img = self._validate_input(image)
        self._log_step("validation_and_normalization")
        
        # 2. Apply shadow-aware denoising
        img = self._shadow_aware_denoise(img)
        self._log_step("shadow_aware_denoise")
        
        # 3. Compute adaptive CLAHE parameters
        current_clip = self.clip_limit
        if self.adaptive_clip:
            current_clip = self._compute_adaptive_clip_limit(img)
            self._log_step(f"adaptive_clip_computation_({current_clip:.2f})")
            
        # 4. Apply CLAHE with tuned parameters
        img = self._apply_clahe(img, current_clip)
        self._log_step("clahe_application")
        
        # 5. Apply gentle unsharp masking
        img = self._unsharp_mask(img)
        self._log_step("unsharp_masking")
        
        # 6. Final normalization to [0, 1]
        img = np.clip(img, 0.0, 1.0)
        self._log_step("final_normalization")
        
        # 7. Compute quality metrics
        metrics = self._compute_quality_metrics(img)
        self._log_step("quality_metrics_computation")
        
        # 8. Compute confidence score
        confidence = 1.0
        if metrics['shadow_fraction'] > 0.6:
            confidence -= 0.5 * (metrics['shadow_fraction'] - 0.6) / 0.4
        if metrics['saturation_fraction'] > 0.2:
            confidence -= 0.5 * (metrics['saturation_fraction'] - 0.2) / 0.8
        confidence = max(0.0, min(1.0, confidence))
        
        return PreprocessingResult(
            image=img,
            instrument="OHRC",
            original_shape=original_shape,
            preprocessed_shape=img.shape,
            processing_steps=self._processing_log.copy(),
            quality_metrics=metrics,
            confidence=confidence
        )

    def _compute_adaptive_clip_limit(self, image: np.ndarray) -> float:
        """
        If adaptive_clip is True, adjust clip limit based on image contrast:
        - Low contrast (std < 0.1): increase clip to 4.0 (more enhancement needed)
        - High contrast (std > 0.25): decrease clip to 1.5 (avoid over-enhancement)
        - Otherwise: use default clip_limit
        """
        std_val = np.std(image)
        if std_val < 0.1:
            return 4.0
        elif std_val > 0.25:
            return 1.5
        else:
            return self.clip_limit

    def _apply_clahe(self, image: np.ndarray, clip_limit: float) -> np.ndarray:
        """
        Apply CLAHE using cv2.createCLAHE.
        - Convert float32 [0,1] to uint8 for cv2
        - Apply CLAHE with (tile_size, tile_size) grid
        - Convert back to float32 [0,1]
        """
        # Convert to uint8
        img_uint8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(self.tile_size, self.tile_size))
        img_clahe = clahe.apply(img_uint8)
        
        # Convert back to float32
        return img_clahe.astype(np.float32) / 255.0
    
    def _shadow_aware_denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Bilateral filter that preserves shadow edges while reducing noise.
        Use cv2.bilateralFilter with d=5, sigmaColor=0.1, sigmaSpace=5.
        Only apply in regions where local variance is low (noisy flat areas).
        """
        # OpenCV's bilateralFilter expects uint8 or float32. We have float32.
        img_filtered = cv2.bilateralFilter(image, d=5, sigmaColor=0.1, sigmaSpace=5.0)
        
        # Compute local variance using scipy
        # A simple approximation of local variance: E[X^2] - E[X]^2
        local_mean = ndimage.uniform_filter(image, size=5)
        local_sq_mean = ndimage.uniform_filter(image**2, size=5)
        local_var = np.clip(local_sq_mean - local_mean**2, 0.0, None)
        
        # Threshold variance to identify flat/noisy regions
        # Max variance of uniform [0,1] is ~0.08, flat areas will have very small variance
        flat_mask = np.clip((0.005 - local_var) / 0.005, 0.0, 1.0)
        
        # Blend original and filtered image based on the flat mask
        blended = image * (1.0 - flat_mask) + img_filtered * flat_mask
        return blended
    
    def _unsharp_mask(self, image: np.ndarray, sigma: float = 1.0, strength: float = 0.3) -> np.ndarray:
        """
        Gentle unsharp masking: I_sharp = I + strength * (I - gaussian_blur(I, sigma)).
        Clip to [0, 1].
        """
        blurred = ndimage.gaussian_filter(image, sigma=sigma)
        sharpened = image + strength * (image - blurred)
        return np.clip(sharpened, 0.0, 1.0)


class TMC2Preprocessor(BasePreprocessor):
    """
    TMC-2 standard radiometric normalization.
    
    TMC-2 (5 m/px) is the hub instrument in the matching hierarchy.
    Preprocessing focuses on consistent, stable normalization that produces
    images suitable for both matching against OHRC (same modality, 20x scale gap)
    and IIRS (cross-modal, 16x scale gap via the IIRS proxy image).
    
    Steps:
    1. Linear contrast stretch (percentile-based to handle outliers)
    2. Optional histogram equalization for uniformity
    3. Gaussian smoothing to suppress sensor noise
    """
    
    def __init__(self, percentile_low: float = 2.0, percentile_high: float = 98.0, 
                 apply_histeq: bool = False, denoise_sigma: float = 0.5):
        super().__init__()
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.apply_histeq = apply_histeq
        self.denoise_sigma = denoise_sigma
    
    def preprocess(self, image: np.ndarray, **kwargs) -> PreprocessingResult:
        """
        Full TMC-2 preprocessing:
        1. Validate and normalize input
        2. Percentile-based linear contrast stretch
        3. Optional histogram equalization
        4. Light Gaussian denoising
        5. Final [0, 1] normalization
        6. Quality metrics and confidence
        """
        self._processing_log = []
        original_shape = image.shape
        
        # 1. Validate and normalize input
        img = self._validate_input(image)
        self._log_step("validation_and_normalization")
        
        # 2. Percentile-based linear contrast stretch
        img = self._percentile_stretch(img)
        self._log_step("percentile_stretch")
        
        # 3. Optional histogram equalization
        if self.apply_histeq:
            img_uint8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            img_eq = cv2.equalizeHist(img_uint8)
            img = img_eq.astype(np.float32) / 255.0
            self._log_step("histogram_equalization")
            
        # 4. Light Gaussian denoising
        if self.denoise_sigma > 0:
            img = ndimage.gaussian_filter(img, sigma=self.denoise_sigma)
            self._log_step("gaussian_denoise")
            
        # 5. Final normalization to [0, 1]
        img = np.clip(img, 0.0, 1.0)
        self._log_step("final_normalization")
        
        # 6. Compute quality metrics and confidence
        metrics = self._compute_quality_metrics(img)
        self._log_step("quality_metrics_computation")
        
        confidence = 1.0
        if metrics['contrast'] < 0.05:
            confidence -= 0.5
        if metrics['saturation_fraction'] > 0.3:
            confidence -= 0.3
        confidence = max(0.0, min(1.0, confidence))
        
        return PreprocessingResult(
            image=img,
            instrument="TMC-2",
            original_shape=original_shape,
            preprocessed_shape=img.shape,
            processing_steps=self._processing_log.copy(),
            quality_metrics=metrics,
            confidence=confidence
        )
    
    def _percentile_stretch(self, image: np.ndarray) -> np.ndarray:
        """
        Clip to [p_low, p_high] percentiles, then rescale to [0, 1].
        More robust than min-max for images with extreme outlier pixels.
        """
        p_low_val = np.percentile(image, self.percentile_low)
        p_high_val = np.percentile(image, self.percentile_high)
        
        if p_high_val > p_low_val:
            stretched = (image - p_low_val) / (p_high_val - p_low_val)
            return np.clip(stretched, 0.0, 1.0)
        else:
            self.logger.warning("Percentile stretch failed due to zero range, returning original image.")
            return np.clip(image, 0.0, 1.0)
