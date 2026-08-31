"""
Base matcher data structures and abstract interface for Module 2.

Defines the canonical ``MatchResult`` data class that every matcher returns,
and the ``BaseMatcher`` ABC that all concrete matchers implement.

SIH26166 — Chandrayaan-2 Multi-modal Image Correspondence Pipeline
"""

import numpy as np
import cv2
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """Canonical output of a feature-matching operation.

    Every matcher in the pipeline — LightGlue, ORB fallback, or the
    hub-and-spoke orchestrator — returns this structure so that
    downstream modules (registration, confidence) can consume matches
    uniformly.

    Attributes:
        keypoints_src:   (N, 2) float64 array of (x, y) pixel coordinates
                         in the *source* image's original resolution.
        keypoints_dst:   (N, 2) float64 array of (x, y) pixel coordinates
                         in the *destination* image's original resolution.
        confidences:     (N,) float32 array of per-match confidence in [0, 1].
        src_instrument:  Instrument name of the source image (e.g. 'OHRC').
        dst_instrument:  Instrument name of the destination image (e.g. 'TMC-2').
        scale_gap:       The GSD ratio between source and destination
                         (e.g. 20.0 for OHRC→TMC-2).
        num_inliers:     Number of geometrically-verified inlier matches.
        match_confidence: Aggregate confidence score in [0, 1].
        metadata:        Matcher-specific auxiliary info (method name,
                         timing, pyramid level, etc.).
    """
    keypoints_src: np.ndarray
    keypoints_dst: np.ndarray
    confidences: np.ndarray
    src_instrument: str
    dst_instrument: str
    scale_gap: float
    num_inliers: int
    match_confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- Convenience properties ------------------------------------------

    @property
    def num_matches(self) -> int:
        """Total number of putative matches (before geometric verification)."""
        return len(self.keypoints_src)

    @property
    def has_matches(self) -> bool:
        """True if at least one match exists."""
        return self.num_matches > 0

    @property
    def inlier_ratio(self) -> float:
        """Fraction of matches that survived geometric verification."""
        if self.num_matches == 0:
            return 0.0
        return self.num_inliers / self.num_matches

    # -- Factory ----------------------------------------------------------

    @staticmethod
    def empty(src_instrument: str = "",
              dst_instrument: str = "",
              scale_gap: float = 1.0) -> "MatchResult":
        """Create an empty (failed / no-match) result with confidence 0."""
        return MatchResult(
            keypoints_src=np.empty((0, 2), dtype=np.float64),
            keypoints_dst=np.empty((0, 2), dtype=np.float64),
            confidences=np.empty(0, dtype=np.float32),
            src_instrument=src_instrument,
            dst_instrument=dst_instrument,
            scale_gap=scale_gap,
            num_inliers=0,
            match_confidence=0.0,
            metadata={"status": "no_matches"},
        )


# ---------------------------------------------------------------------------
# Abstract Base Matcher
# ---------------------------------------------------------------------------

class BaseMatcher(ABC):
    """Abstract base for all image matchers in the pipeline.

    Concrete subclasses must implement :meth:`match`.  The base class
    provides shared image-handling utilities used across matchers.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Args:
            confidence_threshold: Minimum per-match confidence to retain
                                  after matching (applied by
                                  :meth:`_filter_by_confidence`).
        """
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def match(
        self,
        src_image: np.ndarray,
        dst_image: np.ndarray,
        scale_ratio: float = 1.0,
        src_instrument: str = "",
        dst_instrument: str = "",
        **kwargs,
    ) -> MatchResult:
        """Run the matching pipeline on a pair of images.

        Args:
            src_image:      Source image (2-D float32 [0, 1] or uint8).
            dst_image:      Destination image (same format as source).
            scale_ratio:    GSD ratio  ``src_gsd / dst_gsd``  (< 1 means
                            source is higher-resolution).
            src_instrument: Name of the source instrument.
            dst_instrument: Name of the destination instrument.

        Returns:
            A :class:`MatchResult` with keypoints mapped to the
            **original** pixel coordinate systems of both images.
        """
        ...

    # -- Shared utilities -------------------------------------------------

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert to single-channel grayscale if necessary."""
        if image.ndim == 3:
            if image.shape[2] == 1:
                return image[:, :, 0]
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    @staticmethod
    def _normalize_to_01(image: np.ndarray) -> np.ndarray:
        """Normalise an image to float32 in [0, 1]."""
        img = image.astype(np.float32)
        if image.dtype == np.uint8:
            return img / 255.0
        if image.dtype == np.uint16:
            return img / 65535.0
        vmin, vmax = float(np.min(img)), float(np.max(img))
        if vmax > vmin:
            return (img - vmin) / (vmax - vmin)
        return np.zeros_like(img)

    @staticmethod
    def _to_uint8(image: np.ndarray) -> np.ndarray:
        """Convert a [0, 1] float image to uint8 for OpenCV routines."""
        return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)

    def _filter_by_confidence(
        self,
        result: MatchResult,
        threshold: Optional[float] = None,
    ) -> MatchResult:
        """Keep only matches whose confidence ≥ *threshold*.

        Args:
            result:    Input ``MatchResult``.
            threshold: Override for ``self.confidence_threshold``.

        Returns:
            A new ``MatchResult`` with low-confidence matches removed.
        """
        if threshold is None:
            threshold = self.confidence_threshold

        if result.num_matches == 0:
            return result

        mask = result.confidences >= threshold
        n_kept = int(np.sum(mask))

        if n_kept == 0:
            return MatchResult.empty(
                result.src_instrument,
                result.dst_instrument,
                result.scale_gap,
            )

        return MatchResult(
            keypoints_src=result.keypoints_src[mask],
            keypoints_dst=result.keypoints_dst[mask],
            confidences=result.confidences[mask],
            src_instrument=result.src_instrument,
            dst_instrument=result.dst_instrument,
            scale_gap=result.scale_gap,
            num_inliers=n_kept,
            match_confidence=float(np.mean(result.confidences[mask])),
            metadata={**result.metadata, "confidence_filter_threshold": threshold},
        )
