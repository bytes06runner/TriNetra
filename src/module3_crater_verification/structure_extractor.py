"""
Illumination-invariant structural feature extraction.

This module extracts physical topography (like crater rims and ridges)
while ignoring transient illumination effects (shadows). It uses
mathematically rigorous, multi-scale Hessian-based ridge detection
and edge detection.

SIH26166 — Chandrayaan-2 Multi-modal Image Correspondence Pipeline
"""

import numpy as np
import cv2
import logging
from typing import Optional, Literal

# We use scikit-image for rigorous, peer-reviewed structural filters
from skimage.filters import sato, meijering, sobel, gaussian
from skimage.exposure import rescale_intensity

logger = logging.getLogger(__name__)


class StructuralExtractor:
    """Extracts illumination-invariant structural maps from lunar images.

    Lunar images suffer from extreme illumination variance. This class
    strips away raw pixel intensities and extracts structural geometry
    (crater rims, ridges, boundaries).

    Methods available:
      - 'sato': Continuous ridge detection based on the Hessian matrix.
                Excellent for crater rims.
      - 'meijering': Neuriteness filter, highly sensitive to line structures.
      - 'sobel': Standard gradient magnitude.
      - 'composite': A weighted combination of ridge and edge detectors.
    """

    def __init__(
        self,
        method: Literal["sato", "meijering", "sobel", "composite"] = "composite",
        sigmas: tuple = (1, 2, 3),
        black_ridges: bool = False,
    ):
        """
        Args:
            method: The structural extraction algorithm to use.
            sigmas: Scales (Gaussian blur standard deviations) for multi-scale
                    Hessian filters. (1, 2, 3) is good for standard GSDs.
            black_ridges: Set to True if structural features (craters) are
                          darker than the background. For lunar images, rims
                          can be bright or dark depending on the sun angle,
                          so we often check both or rely on edges.
        """
        if method not in ["sato", "meijering", "sobel", "composite"]:
            raise ValueError(f"Unknown extraction method: {method}")

        self.method = method
        self.sigmas = sigmas
        self.black_ridges = black_ridges

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract the structural map from an image.

        Args:
            image: 2-D float32 image in [0, 1].

        Returns:
            2-D float32 structural map in [0, 1]. High values indicate
            the presence of a structural feature.
        """
        img = image.astype(np.float32)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Smooth high-frequency sensor noise before structural analysis
        img_smoothed = gaussian(img, sigma=0.5, preserve_range=True)

        t0 = cv2.getTickCount()

        if self.method == "sato":
            struct_map = self._extract_sato(img_smoothed)
        elif self.method == "meijering":
            struct_map = self._extract_meijering(img_smoothed)
        elif self.method == "sobel":
            struct_map = self._extract_sobel(img_smoothed)
        elif self.method == "composite":
            struct_map = self._extract_composite(img_smoothed)
        else:
            struct_map = np.zeros_like(img_smoothed)

        t1 = cv2.getTickCount()
        time_ms = (t1 - t0) * 1000.0 / cv2.getTickFrequency()
        logger.debug(
            "Extracted structural map (%s) in %.1f ms", self.method, time_ms
        )

        return struct_map

    def _extract_sato(self, image: np.ndarray) -> np.ndarray:
        """Hessian-based multi-scale tubeness/ridge filter."""
        ridges = sato(
            image,
            sigmas=self.sigmas,
            black_ridges=self.black_ridges,
            mode='reflect'
        )
        return self._normalize(ridges)

    def _extract_meijering(self, image: np.ndarray) -> np.ndarray:
        """Meijering neuriteness filter (good for elongated crater rims)."""
        ridges = meijering(
            image,
            sigmas=self.sigmas,
            black_ridges=self.black_ridges,
            mode='reflect'
        )
        return self._normalize(ridges)

    def _extract_sobel(self, image: np.ndarray) -> np.ndarray:
        """Standard gradient magnitude."""
        edges = sobel(image)
        return self._normalize(edges)

    def _extract_composite(self, image: np.ndarray) -> np.ndarray:
        """Combines multi-scale ridges and edges for a robust map."""
        # 1. Get ridges (crater rims)
        # We compute for both bright and dark ridges since illumination
        # makes one side of the crater bright and the other dark.
        ridges_bright = sato(image, sigmas=self.sigmas, black_ridges=False, mode='reflect')
        ridges_dark = sato(image, sigmas=self.sigmas, black_ridges=True, mode='reflect')
        ridges = np.maximum(ridges_bright, ridges_dark)
        ridges = self._normalize(ridges)

        # 2. Get sharp edges
        edges = sobel(image)
        edges = self._normalize(edges)

        # 3. Combine: max operator ensures we capture both sharp boundaries
        # and continuous rim structures.
        composite = np.maximum(ridges, edges)
        
        # Soft threshold to suppress background regolith noise
        mean_val = np.mean(composite)
        std_val = np.std(composite)
        threshold = mean_val + 0.5 * std_val
        
        composite_filtered = np.where(composite < threshold, 0.0, composite - threshold)
        return self._normalize(composite_filtered)

    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        """Min-max normalize to float32 [0, 1]."""
        if image.size == 0:
            return image
        
        vmin, vmax = image.min(), image.max()
        if vmax - vmin < 1e-6:
            return np.zeros_like(image, dtype=np.float32)
            
        return ((image - vmin) / (vmax - vmin)).astype(np.float32)
