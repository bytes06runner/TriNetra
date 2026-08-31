"""
Crater-Structural Match Verification.

Acts as a verification layer ("lie detector") for Module 2 matchers.
Extracts local structural patches around matched keypoints and computes
Normalized Cross-Correlation (NCC) to verify that putative matches actually
align with physical lunar topography.

SIH26166 — Chandrayaan-2 Multi-modal Image Correspondence Pipeline
"""

import numpy as np
import logging
import cv2
from typing import Optional, Tuple

from src.module2_matching.base_matcher import MatchResult
from .structure_extractor import StructuralExtractor

logger = logging.getLogger(__name__)


class StructuralVerifier:
    """Verifies matches using illumination-invariant structural patches.

    For each matched keypoint pair, extracts a structural map patch from
    both the source and destination images, aligns their scales, and computes
    Normalized Cross-Correlation (NCC). Matches with low structural NCC are
    filtered out.
    """

    def __init__(
        self,
        patch_size_dst: int = 31,
        ncc_threshold: float = 0.3,
        extractor_method: str = "composite",
    ):
        """
        Args:
            patch_size_dst: Size of the local square patch in the lower-res
                            (destination) image coordinates. Must be odd.
            ncc_threshold:  Minimum Structural NCC to accept a match.
            extractor_method: Method used for structural extraction.
        """
        self.patch_size_dst = patch_size_dst if patch_size_dst % 2 == 1 else patch_size_dst + 1
        self.ncc_threshold = ncc_threshold
        self.extractor = StructuralExtractor(method=extractor_method)

    def verify(
        self,
        match_result: MatchResult,
        src_image: np.ndarray,
        dst_image: np.ndarray,
    ) -> MatchResult:
        """Verify putative matches structurally.

        Args:
            match_result: The putative matches from Module 2.
            src_image:    Original source image (higher-res).
            dst_image:    Original destination image (lower-res).

        Returns:
            A new :class:`MatchResult` containing only structurally verified
            inliers. The match confidences are updated to reflect the
            structural NCC scores.
        """
        if not match_result.has_matches:
            return match_result

        # 1. Extract full structural maps
        # Doing this globally once per image is more efficient than
        # extracting per-patch if the number of matches is large.
        struct_src = self.extractor.extract(src_image)
        struct_dst = self.extractor.extract(dst_image)

        scale_ratio = match_result.scale_gap  # src_gsd / dst_gsd
        # If scale_ratio is e.g. 20 (OHRC->TMC2), the source patch needs
        # to be 20x larger to cover the same physical area.
        
        # We need the relative scale. If scale_gap > 1, dst is coarser.
        if match_result.scale_gap >= 1.0:
            patch_size_src = int(self.patch_size_dst * match_result.scale_gap)
        else:
            patch_size_src = int(self.patch_size_dst / max(1e-5, match_result.scale_gap))
            
        # Ensure odd size
        if patch_size_src % 2 == 0:
            patch_size_src += 1

        half_dst = self.patch_size_dst // 2
        half_src = patch_size_src // 2

        verified_mask = np.zeros(match_result.num_matches, dtype=bool)
        structural_scores = np.zeros(match_result.num_matches, dtype=np.float32)

        for i in range(match_result.num_matches):
            pt_src = match_result.keypoints_src[i]
            pt_dst = match_result.keypoints_dst[i]

            # 2. Extract local patches
            patch_s = self._extract_patch(struct_src, pt_src, half_src)
            patch_d = self._extract_patch(struct_dst, pt_dst, half_dst)

            if patch_s is None or patch_d is None:
                continue  # Keypoint too close to edge

            # 3. Align scales (downsample high-res patch to low-res size)
            if patch_s.shape != patch_d.shape:
                patch_s_aligned = cv2.resize(
                    patch_s,
                    (self.patch_size_dst, self.patch_size_dst),
                    interpolation=cv2.INTER_AREA
                )
            else:
                patch_s_aligned = patch_s

            # 4. Compute Normalized Cross-Correlation (NCC)
            ncc = self._compute_ncc(patch_s_aligned, patch_d)
            structural_scores[i] = ncc

            if ncc >= self.ncc_threshold:
                verified_mask[i] = True

        n_kept = int(np.sum(verified_mask))
        
        logger.info(
            "Structural Verification: %d/%d passed (threshold %.2f)",
            n_kept, match_result.num_matches, self.ncc_threshold
        )

        if n_kept == 0:
            return MatchResult.empty(
                match_result.src_instrument,
                match_result.dst_instrument,
                match_result.scale_gap
            )

        # Update confidences: combine original matcher confidence with structural score
        original_confs = match_result.confidences[verified_mask]
        struct_confs = structural_scores[verified_mask]
        
        # Harmonic mean to severely penalize if either score is low
        new_confs = 2 * (original_confs * struct_confs) / np.maximum(original_confs + struct_confs, 1e-6)

        return MatchResult(
            keypoints_src=match_result.keypoints_src[verified_mask],
            keypoints_dst=match_result.keypoints_dst[verified_mask],
            confidences=new_confs,
            src_instrument=match_result.src_instrument,
            dst_instrument=match_result.dst_instrument,
            scale_gap=match_result.scale_gap,
            num_inliers=n_kept,
            match_confidence=float(np.mean(new_confs)),
            metadata={
                **match_result.metadata,
                "structural_verification": True,
                "structural_ncc_threshold": self.ncc_threshold,
                "num_pre_verification": match_result.num_matches
            }
        )

    def _extract_patch(
        self, image: np.ndarray, pt: np.ndarray, half_size: int
    ) -> Optional[np.ndarray]:
        """Extract a square patch around a keypoint."""
        x, y = int(round(pt[0])), int(round(pt[1]))
        h, w = image.shape

        if (y - half_size < 0 or y + half_size >= h or
            x - half_size < 0 or x + half_size >= w):
            return None

        return image[y - half_size : y + half_size + 1, x - half_size : x + half_size + 1]

    def _compute_ncc(self, patch1: np.ndarray, patch2: np.ndarray) -> float:
        """Compute Normalized Cross-Correlation between two patches."""
        p1 = patch1.astype(np.float32)
        p2 = patch2.astype(np.float32)

        # Zero-mean
        m1, m2 = np.mean(p1), np.mean(p2)
        p1_zm = p1 - m1
        p2_zm = p2 - m2

        num = np.sum(p1_zm * p2_zm)
        den = np.sqrt(np.sum(p1_zm ** 2) * np.sum(p2_zm ** 2))

        if den < 1e-8:
            return 0.0

        return float(num / den)
