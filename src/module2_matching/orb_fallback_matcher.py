"""
ORB-based fallback matcher using pure OpenCV.

This matcher is used when torch/kornia are not available (e.g. CPU-only
CI environments).  It provides a fully functional matching pipeline
using ORB features + brute-force Hamming distance + Lowe's ratio test.

While less accurate than LightGlue on lunar imagery, it guarantees the
pipeline can always produce *some* matches.

SIH26166 — Chandrayaan-2 Multi-modal Image Correspondence Pipeline
"""

import numpy as np
import cv2
import logging
import time
from typing import Optional

from .base_matcher import BaseMatcher, MatchResult
from .scale_handler import ScaleAligner, map_keypoints_to_original

logger = logging.getLogger(__name__)


class ORBFallbackMatcher(BaseMatcher):
    """Feature matcher using ORB + brute-force Hamming distance.

    Parameters:
        max_keypoints:        Maximum ORB features per image (default 5000).
        ratio_threshold:      Lowe's ratio test threshold (default 0.75).
        confidence_threshold: Minimum match confidence to retain.
        upsample_low_res:     Factor to upsample the lower-res image
                              (1.0 = no upsampling).
    """

    def __init__(
        self,
        max_keypoints: int = 5000,
        ratio_threshold: float = 0.75,
        confidence_threshold: float = 0.3,
        upsample_low_res: float = 1.0,
    ):
        super().__init__(confidence_threshold=confidence_threshold)
        self.max_keypoints = max_keypoints
        self.ratio_threshold = ratio_threshold
        self.upsample_low_res = upsample_low_res

        # ORB detector
        self._orb = cv2.ORB_create(nfeatures=self.max_keypoints)

        # Brute-force matcher with Hamming distance (ORB uses binary descriptors)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def match(
        self,
        src_image: np.ndarray,
        dst_image: np.ndarray,
        scale_ratio: float = 1.0,
        src_instrument: str = "",
        dst_instrument: str = "",
        **kwargs,
    ) -> MatchResult:
        """Run ORB matching between *src_image* and *dst_image*.

        Args:
            src_image:      Higher-resolution image (2-D float32 [0,1] or uint8).
            dst_image:      Lower-resolution image (same format).
            scale_ratio:    ``src_gsd / dst_gsd`` (e.g. 0.05 for OHRC/TMC-2).
            src_instrument: Source instrument name.
            dst_instrument: Destination instrument name.

        Returns:
            :class:`MatchResult` with keypoints in **original** pixel coords.
        """
        t0 = time.time()

        # 1. Ensure grayscale float32 [0,1]
        src_gray = self._to_grayscale(src_image)
        dst_gray = self._to_grayscale(dst_image)
        src_gray = self._normalize_to_01(src_gray)
        dst_gray = self._normalize_to_01(dst_gray)

        # 2. Scale alignment
        actual_scale_gap = 1.0 / scale_ratio if scale_ratio > 0 else 1.0
        aligner = ScaleAligner(upsample_factor=self.upsample_low_res)

        if scale_ratio < 1.0:
            # src is higher-res → downsample src
            aligned = aligner.align(src_gray, dst_gray, scale_ratio)
            aligned_src = aligned["aligned_src"]
            aligned_dst = aligned["aligned_dst"]
            src_scale = aligned["src_scale"]
            dst_scale = aligned["dst_scale"]
        else:
            # Images are at comparable scale — no pyramid needed
            aligned_src = src_gray
            aligned_dst = dst_gray
            src_scale = 1.0
            dst_scale = 1.0

        # 3. Convert to uint8 for ORB
        src_u8 = self._to_uint8(aligned_src)
        dst_u8 = self._to_uint8(aligned_dst)

        # 4. Detect ORB features
        kp_src, des_src = self._orb.detectAndCompute(src_u8, None)
        kp_dst, des_dst = self._orb.detectAndCompute(dst_u8, None)

        if des_src is None or des_dst is None or len(kp_src) < 2 or len(kp_dst) < 2:
            logger.warning(
                "ORB: insufficient features (src=%d, dst=%d)",
                len(kp_src) if kp_src else 0,
                len(kp_dst) if kp_dst else 0,
            )
            return MatchResult.empty(src_instrument, dst_instrument, actual_scale_gap)

        # 5. KNN match (k=2 for ratio test)
        raw_matches = self._bf.knnMatch(des_src, des_dst, k=2)

        # 6. Lowe's ratio test
        good_matches = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)

        if len(good_matches) == 0:
            logger.info("ORB: no matches passed ratio test")
            return MatchResult.empty(src_instrument, dst_instrument, actual_scale_gap)

        # 7. Extract matched keypoint coordinates (in aligned image coords)
        pts_src_aligned = np.array(
            [kp_src[m.queryIdx].pt for m in good_matches], dtype=np.float64
        )
        pts_dst_aligned = np.array(
            [kp_dst[m.trainIdx].pt for m in good_matches], dtype=np.float64
        )

        # 8. Compute per-match confidence from descriptor distance
        #    ORB Hamming distances are in [0, 256]; lower = better.
        distances = np.array([m.distance for m in good_matches], dtype=np.float32)
        max_dist = 256.0
        confidences = np.clip(1.0 - distances / max_dist, 0.0, 1.0)

        # 9. Map keypoints back to original coordinates
        pts_src_orig = map_keypoints_to_original(pts_src_aligned, src_scale)
        pts_dst_orig = map_keypoints_to_original(pts_dst_aligned, dst_scale)

        elapsed = time.time() - t0

        result = MatchResult(
            keypoints_src=pts_src_orig,
            keypoints_dst=pts_dst_orig,
            confidences=confidences,
            src_instrument=src_instrument,
            dst_instrument=dst_instrument,
            scale_gap=actual_scale_gap,
            num_inliers=len(good_matches),
            match_confidence=float(np.mean(confidences)),
            metadata={
                "method": "ORB+BFMatcher",
                "num_features_src": len(kp_src),
                "num_features_dst": len(kp_dst),
                "num_raw_matches": len(raw_matches),
                "num_ratio_passed": len(good_matches),
                "ratio_threshold": self.ratio_threshold,
                "elapsed_seconds": elapsed,
                "src_scale": src_scale,
                "dst_scale": dst_scale,
            },
        )

        # 10. Filter by confidence threshold
        result = self._filter_by_confidence(result)

        logger.info(
            "ORB match: %d features → %d ratio-passed → %d final matches "
            "(confidence %.3f) in %.2fs",
            len(kp_src),
            len(good_matches),
            result.num_matches,
            result.match_confidence,
            elapsed,
        )

        return result
