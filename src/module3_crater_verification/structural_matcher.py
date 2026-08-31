"""
Structural Fallback Matcher (XoFTR-Proxy).

If radiometric gaps are too extreme (e.g., TMC-2 to IIRS proxy fails),
this module extracts the illumination-invariant structural maps first,
then runs standard feature matchers (LightGlue/ORB) on the *structure*
rather than the pixels. This effectively mimics the behaviour of
advanced cross-modal networks like MINIMA or XoFTR without needing
specialized pre-trained weights.

SIH26166 — Chandrayaan-2 Multi-modal Image Correspondence Pipeline
"""

import numpy as np
import logging

from src.module2_matching.base_matcher import BaseMatcher, MatchResult
from .structure_extractor import StructuralExtractor

logger = logging.getLogger(__name__)


class StructuralMatcher(BaseMatcher):
    """Matches images based strictly on their structural topography.

    Wraps an existing BaseMatcher (like LightGlueMatcher), but intercepts
    the images to convert them into structural maps before matching.
    """

    def __init__(
        self,
        base_matcher: BaseMatcher,
        extractor_method: str = "composite",
        confidence_threshold: float = 0.2, # Lower threshold for structure-only
    ):
        """
        Args:
            base_matcher:     An instantiated matcher from Module 2.
            extractor_method: The structural extraction method to use.
            confidence_threshold: Threshold applied after matching.
        """
        super().__init__(confidence_threshold=confidence_threshold)
        self.matcher = base_matcher
        self.extractor = StructuralExtractor(method=extractor_method)

    def match(
        self,
        src_image: np.ndarray,
        dst_image: np.ndarray,
        scale_ratio: float = 1.0,
        src_instrument: str = "",
        dst_instrument: str = "",
        **kwargs,
    ) -> MatchResult:
        """Run structural feature matching.

        Args:
            src_image:      Source image.
            dst_image:      Destination image.
            scale_ratio:    src_gsd / dst_gsd.
            src_instrument: Source instrument name.
            dst_instrument: Destination instrument name.

        Returns:
            MatchResult based on structural correspondences.
        """
        logger.info(
            "Executing Structural Matcher (%s) for %s ↔ %s",
            self.extractor.method, src_instrument, dst_instrument
        )

        # 1. Convert to structural maps
        struct_src = self.extractor.extract(src_image)
        struct_dst = self.extractor.extract(dst_image)

        # 2. Match the structural maps directly
        # Structural maps are float32 [0,1], perfectly compatible with Module 2 matchers
        result = self.matcher.match(
            src_image=struct_src,
            dst_image=struct_dst,
            scale_ratio=scale_ratio,
            src_instrument=src_instrument,
            dst_instrument=dst_instrument,
            **kwargs
        )

        if not result.has_matches:
            return result

        # Update metadata to reflect the structural path
        result.metadata["structural_matching"] = True
        result.metadata["extractor_method"] = self.extractor.method
        
        # Filter by the structural confidence threshold
        return self._filter_by_confidence(result)
