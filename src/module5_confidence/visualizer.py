"""
Confidence and Explainability Visualizations (Module 5).

Generates presentation-ready diagnostic plots for SIH evaluation.
Visualizes structural verification patches, matching inliers, and
final image overlays to provide transparent explainability for the pipeline.

SIH26166 — Chandrayaan-2 Multi-modal Image Correspondence Pipeline
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Optional, Tuple

from src.module2_matching.base_matcher import MatchResult
from src.module4_registration.registration import RegistrationResult

logger = logging.getLogger(__name__)


class ExplainabilityVisualizer:
    """Generates visual diagnostics for pipeline explainability."""

    @staticmethod
    def plot_matches(
        img_src: np.ndarray,
        img_dst: np.ndarray,
        match_result: MatchResult,
        inliers_mask: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        title: str = "Feature Matches"
    ) -> None:
        """Plot side-by-side matches with lines connecting keypoints."""
        if img_src.ndim == 2:
            img_src = cv2.cvtColor((img_src * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        if img_dst.ndim == 2:
            img_dst = cv2.cvtColor((img_dst * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

        # Convert to cv2 KeyPoints and DMatches format for drawMatches
        kps_src = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in match_result.keypoints_src]
        kps_dst = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in match_result.keypoints_dst]
        
        matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(kps_src))]

        # Separate inliers and outliers if mask is provided
        if inliers_mask is not None:
            # Draw outliers in red, inliers in green
            matches_mask = [int(bool_val) for bool_val in inliers_mask]
            match_color = (0, 255, 0) # Green for inliers
            single_point_color = (255, 0, 0) # Red for outliers
        else:
            matches_mask = None
            match_color = (0, 255, 0)
            single_point_color = None

        img_matches = cv2.drawMatches(
            img_src, kps_src,
            img_dst, kps_dst,
            matches, None,
            matchColor=match_color,
            singlePointColor=single_point_color,
            matchesMask=matches_mask,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        plt.figure(figsize=(15, 8))
        plt.imshow(img_matches)
        plt.title(f"{title} ({match_result.src_instrument} ↔ {match_result.dst_instrument})")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info("Saved matches plot to %s", save_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_registration_overlay(
        img_src: np.ndarray,
        img_dst: np.ndarray,
        reg_result: RegistrationResult,
        save_path: Optional[str] = None
    ) -> None:
        """Plot the source image warped over the destination image."""
        if not reg_result.success:
            logger.warning("Cannot plot overlay for failed registration.")
            return

        # Ensure uint8 [0, 255] for OpenCV warping
        if img_src.dtype == np.float32:
            img_src = (img_src * 255).astype(np.uint8)
        if img_dst.dtype == np.float32:
            img_dst = (img_dst * 255).astype(np.uint8)

        h, w = img_dst.shape[:2]
        
        # Warp the source image to the destination frame
        warped_src = cv2.warpPerspective(img_src, reg_result.transform_matrix, (w, h))

        # Create an RGB overlay (Source mapped to Red channel, Dest to Green/Blue)
        if warped_src.ndim == 3:
            warped_src = cv2.cvtColor(warped_src, cv2.COLOR_RGB2GRAY)
        if img_dst.ndim == 3:
            img_dst = cv2.cvtColor(img_dst, cv2.COLOR_RGB2GRAY)

        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[..., 0] = warped_src  # Red
        overlay[..., 1] = img_dst     # Green
        overlay[..., 2] = img_dst     # Blue

        plt.figure(figsize=(10, 10))
        plt.imshow(overlay)
        plt.title(f"Registration Overlay\nRed: Warped {reg_result.match_result.src_instrument}, Cyan: {reg_result.match_result.dst_instrument}")
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info("Saved registration overlay plot to %s", save_path)
            plt.close()
        else:
            plt.show()
