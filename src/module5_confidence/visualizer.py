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
        """Plot side-by-side matches. Scales the smaller image to match the height of the larger image
        so the visualization is readable across massive scale gaps (e.g. 320x). Custom drawing for better visuals."""
        if img_src.ndim == 2:
            img_src = cv2.cvtColor((img_src * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        if img_dst.ndim == 2:
            img_dst = cv2.cvtColor((img_dst * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

        h1, w1 = img_src.shape[:2]
        h2, w2 = img_dst.shape[:2]
        
        target_h = h1
        scale_factor = target_h / float(h2) if h2 > 0 else 1.0
        target_w = int(w2 * scale_factor)
        
        img_dst_scaled = cv2.resize(img_dst, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        
        # Create side-by-side canvas
        canvas = np.zeros((max(h1, target_h), w1 + target_w, 3), dtype=np.uint8)
        canvas[:h1, :w1] = img_src
        canvas[:target_h, w1:] = img_dst_scaled
        
        # Darken the images slightly to make lines pop out
        canvas = (canvas * 0.7).astype(np.uint8)
        
        line_thickness = max(1, h1 // 500)
        
        for i, (pt1, pt2) in enumerate(zip(match_result.keypoints_src, match_result.keypoints_dst)):
            # skip outliers
            if inliers_mask is not None and not inliers_mask[i]:
                continue
                
            x1, y1 = int(pt1[0]), int(pt1[1])
            x2, y2 = int(pt2[0] * scale_factor) + w1, int(pt2[1] * scale_factor)
            
            # Draw beautiful semi-transparent cyan/green line
            color = (0, 255, 128)
            cv2.line(canvas, (x1, y1), (x2, y2), color, line_thickness, cv2.LINE_AA)
            cv2.circle(canvas, (x1, y1), line_thickness + 2, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (x2, y2), line_thickness + 2, (0, 255, 255), -1, cv2.LINE_AA)

        plt.figure(figsize=(16, 9))
        plt.imshow(canvas)
        plt.title(f"{title} ({match_result.src_instrument} ↔ {match_result.dst_instrument} - Target scaled by {scale_factor:.1f}x)", color='#333333', fontsize=14, pad=15)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
            logger.info("Saved custom matches plot to %s", save_path)
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
