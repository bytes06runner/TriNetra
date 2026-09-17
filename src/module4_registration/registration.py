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
    status: str = "PENDING"            # 'PASSED', 'GATED', 'DEGENERATE', 'FAILED'
    inlier_ratio_pct: float = 0.0

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
            success=False,
            status="FAILED",
            inlier_ratio_pct=0.0
        )


def evaluate_flight_gate(
    inliers: int,
    total_matches: int,
    inlier_ratio_pct: Optional[float] = None,
    H: Optional[np.ndarray] = None,
    expected_scale: Optional[float] = 1.0,
    max_rotation_deg: float = 30.0,
    scale_tolerance: float = 0.25,
    cond_thresh: float = 1e5,
    ignore_consensus: bool = False,
) -> dict:
    """Evaluate whether a correspondence set satisfies spaceflight safety gates.

    Expected Scale Note:
        expected_scale defaults to 1.0 because input crops are independently
        pre-scaled / decimated to a common standardized display canvas (e.g.
        1000x1000 for Hop 1, 800x800 for Hop 2). This pre-scaling absorbs the raw
        instrument GSD gaps (18.15x for Hop 1, 14.49x for Hop 2) at crop extraction
        time, so a valid geometric solution in canvas space expects a residual scale
        factor near 1.0 (within scale_tolerance = 25%).

    Flight Safety Rules (evaluated in order):
        1. Inlier Consensus Rule (unless ignore_consensus=True):
           inliers < 20  OR  inlier_ratio_pct < 15.0%.
        2. Matrix Conditioning Rule:
           cond(H) > 1e5 (ill-conditioned fit prone to numerical instability).
           Threshold justification from empirical lunar flight data:
           Passing fits exhibit cond(H) in [689, 3646], whereas degenerate fits
           exhibit cond(H) in [3.5e5, 2.4e6]. 1e5 cleanly separates both regimes.
        3. Physical Plausibility Rule:
           |rotation| > max_rotation_deg (default 30°) OR
           |scale - expected_scale| / expected_scale > scale_tolerance (default 25%).
    """
    if inlier_ratio_pct is None:
        inlier_ratio_pct = (inliers / total_matches * 100.0) if total_matches > 0 else 0.0

    reasons = []
    first_failing_criterion = None

    # Criterion 1: Inlier consensus floor (evaluated unless ablated)
    if not ignore_consensus:
        if inliers < 20 or inlier_ratio_pct < 15.0:
            reasons.append(f"Inlier count ({inliers} < 20) or ratio ({inlier_ratio_pct:.1f}% < 15.0%)")
            if first_failing_criterion is None:
                first_failing_criterion = "inlier_consensus"

    cond_val = None
    scale_val = None
    rot_val = None

    if H is not None:
        H64 = np.asarray(H, dtype=np.float64)
        cond_val = float(np.linalg.cond(H64))

        # Criterion 2: Conditioning
        if cond_val > cond_thresh:
            reasons.append(f"Ill-conditioned transform: cond(H) = {cond_val:.1e} > {cond_thresh:.1e}")
            if first_failing_criterion is None:
                first_failing_criterion = "conditioning"

        # Criterion 3: Physical Plausibility (Scale & Rotation)
        scale_val = float(np.sqrt(H64[0, 0] ** 2 + H64[1, 0] ** 2))
        rot_val = float(np.degrees(np.arctan2(H64[1, 0], H64[0, 0])))

        if abs(rot_val) > max_rotation_deg:
            reasons.append(f"Unphysical rotation: |{rot_val:.1f}°| > {max_rotation_deg:.1f}°")
            if first_failing_criterion is None:
                first_failing_criterion = "unphysical_rotation"

        if expected_scale is not None and expected_scale > 0:
            scale_err = abs(scale_val - expected_scale) / expected_scale
            if scale_err > scale_tolerance:
                reasons.append(
                    f"Unphysical scale: recovered {scale_val:.4f} diverges by {scale_err*100:.1f}% "
                    f"(> {scale_tolerance*100:.1f}%) from expected {expected_scale:.4f}"
                )
                if first_failing_criterion is None:
                    first_failing_criterion = "unphysical_scale"

    is_gated = len(reasons) > 0
    return {
        "is_gated": is_gated,
        "status": "GATED" if is_gated else "PASSED",
        "inliers": inliers,
        "total_matches": total_matches,
        "inlier_ratio_pct": float(inlier_ratio_pct),
        "condition_number": cond_val,
        "recovered_scale": scale_val,
        "recovered_rotation_deg": rot_val,
        "reasons": reasons,
        "first_failing_criterion": first_failing_criterion,
        "reason": "; ".join(reasons) if is_gated else "Meets flight safety requirements",
    }


def check_degeneracy(inliers: int, raw_matches: int, rmse: float, transform_dof: int = 4) -> dict:
    """Check whether an estimated geometric transformation is mathematically degenerate.

    Mathematical Degeneracy Suppression Rules:
        1. Degrees of Freedom Rule: inliers < 2 * transform_dof
           (e.g. requires >= 8 inliers for 4-DoF similarity, >= 16 inliers for 8-DoF homography).
        2. Residual Clamp Rule: rmse < 0.10 px and inliers < 8
           (identifies overfitted degenerate fits where residual collapses to ~0).
        3. Minimum Raw Matches: raw_matches < 8 -> suppress ratio.
    """
    reasons = []
    is_degenerate = False

    if inliers < 2 * transform_dof:
        is_degenerate = True
        reasons.append(f"Inliers ({inliers}) < 2 * DoF ({2 * transform_dof})")

    if rmse < 0.10 and inliers < 8:
        is_degenerate = True
        reasons.append(f"Overfitted residual ({rmse:.2f} px) with too few inliers ({inliers})")

    if raw_matches < 8:
        is_degenerate = True
        reasons.append(f"Raw candidate matches ({raw_matches}) < 8")

    return {
        "is_degenerate": is_degenerate,
        "status": "DEGENERATE" if is_degenerate else "VALID",
        "reasons": reasons,
        "suppress_metrics": is_degenerate
    }



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
                method=cv2.RANSAC,
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
