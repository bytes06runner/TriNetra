"""
Geometric Registration (Module 4).

Takes structurally verified matches and computes a robust geometric
transformation matrix (Homography or Affine) mapping the source image
coordinates to the destination image coordinates. Uses OpenCV's 
state-of-the-art MAGSAC++ algorithm for extreme outlier rejection.

SIH26166 — Chandrayaan-2 Multi-modal Image Correspondence Pipeline
"""

import numpy as np
import cv2
import logging
from dataclasses import dataclass
from typing import Optional

from src.module2_matching.base_matcher import MatchResult

logger = logging.getLogger(__name__)


@dataclass
class RegistrationResult:
    """Stores the final geometric transformation result."""
    transform_matrix: np.ndarray       # (3, 3) matrix
    transform_type: str                # 'homography' or 'affine'
    inliers_mask: np.ndarray           # boolean array of shape (N,)
    num_inliers: int
    rmse: float                        # Root Mean Square Error of the fit
    match_result: MatchResult          # The underlying match result
    success: bool

    @classmethod
    def empty(cls, match_result: MatchResult) -> "RegistrationResult":
        """Factory for a failed registration."""
        return cls(
            transform_matrix=np.eye(3, dtype=np.float32),
            transform_type='none',
            inliers_mask=np.zeros(match_result.num_matches, dtype=bool),
            num_inliers=0,
            rmse=float('inf'),
            match_result=match_result,
            success=False
        )


class GeometricRegistrar:
    """Computes geometric registration from matched keypoints."""

    def __init__(
        self,
        transform_type: str = "homography",
        reproj_thresh_pixels: float = 3.0,
        max_iters: int = 10000,
        confidence: float = 0.999
    ):
        """
        Args:
            transform_type: 'homography' (perspective) or 'affine' (rigid).
            reproj_thresh_pixels: Max distance (in dest pixels) for an inlier.
            max_iters: Max RANSAC/MAGSAC iterations.
            confidence: Desired probability of success.
        """
        if transform_type not in ["homography", "affine"]:
            raise ValueError(f"Unknown transform_type: {transform_type}")

        self.transform_type = transform_type
        self.reproj_thresh = reproj_thresh_pixels
        self.max_iters = max_iters
        self.confidence = confidence

    def register(self, match_result: MatchResult) -> RegistrationResult:
        """Estimate the transformation matrix.

        Args:
            match_result: A structurally verified MatchResult.

        Returns:
            RegistrationResult containing the matrix and final inlier mask.
        """
        if match_result.num_matches < 4:
            logger.warning("Not enough matches to compute geometric registration (< 4).")
            return RegistrationResult.empty(match_result)

        pts_src = match_result.keypoints_src.reshape(-1, 1, 2)
        pts_dst = match_result.keypoints_dst.reshape(-1, 1, 2)

        t0 = cv2.getTickCount()

        if self.transform_type == "homography":
            # USAC_MAGSAC is robust against severe outlier contamination
            matrix, mask = cv2.findHomography(
                pts_src,
                pts_dst,
                cv2.USAC_MAGSAC,
                self.reproj_thresh,
                maxIters=self.max_iters,
                confidence=self.confidence
            )
        else:
            matrix, mask = cv2.estimateAffinePartial2D(
                pts_src,
                pts_dst,
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=self.reproj_thresh,
                maxIters=self.max_iters,
                confidence=self.confidence
            )
            # Convert 2x3 affine to 3x3 for consistency
            if matrix is not None:
                matrix = np.vstack([matrix, [0, 0, 1]])

        t1 = cv2.getTickCount()
        time_ms = (t1 - t0) * 1000.0 / cv2.getTickFrequency()

        if matrix is None or mask is None:
            logger.error("MAGSAC++ failed to find a valid transformation.")
            return RegistrationResult.empty(match_result)

        inliers_mask = mask.ravel().astype(bool)
        num_inliers = int(np.sum(inliers_mask))

        if num_inliers < 4:
            logger.warning("MAGSAC++ yielded too few inliers (%d).", num_inliers)
            return RegistrationResult.empty(match_result)

        # Compute RMSE on the inliers
        rmse = self._compute_rmse(pts_src[inliers_mask], pts_dst[inliers_mask], matrix)

        logger.info(
            "Registration (%s) successful in %.1f ms. Inliers: %d/%d (RMSE: %.2f px)",
            self.transform_type, time_ms, num_inliers, match_result.num_matches, rmse
        )

        return RegistrationResult(
            transform_matrix=matrix.astype(np.float32),
            transform_type=self.transform_type,
            inliers_mask=inliers_mask,
            num_inliers=num_inliers,
            rmse=rmse,
            match_result=match_result,
            success=True
        )

    def _compute_rmse(self, pts_src: np.ndarray, pts_dst: np.ndarray, matrix: np.ndarray) -> float:
        """Compute the Root Mean Square Error of the transformation."""
        pts_src_transformed = cv2.perspectiveTransform(pts_src, matrix)
        errors = np.linalg.norm(pts_src_transformed - pts_dst, axis=-1)
        return float(np.sqrt(np.mean(errors ** 2)))
