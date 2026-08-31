"""
LightGlue-based sparse matcher using kornia + PyTorch.

Primary matcher for Hop 1 (OHRC ↔ TMC-2, same-modality, 20× scale gap).
Uses SuperPoint for keypoint detection/description and LightGlue for
learned matching.  Falls back to LoFTR if LightGlue is unavailable,
and ultimately to the ORB fallback if torch/kornia are missing entirely.

SIH26166 — Chandrayaan-2 Multi-modal Image Correspondence Pipeline
"""

import numpy as np
import cv2
import logging
import time
from typing import Optional, Tuple

from .base_matcher import BaseMatcher, MatchResult
from .scale_handler import ScaleAligner, map_keypoints_to_original

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional torch / kornia imports
# ---------------------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import kornia
    from kornia.feature import LoFTR as KorniaLoFTR
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False

# Try to import LightGlue — available in kornia >= 0.7.2
LIGHTGLUE_AVAILABLE = False
if KORNIA_AVAILABLE:
    try:
        from kornia.feature import LightGlue as KorniaLightGlue
        from kornia.feature import SuperPoint as KorniaSuperPoint
        LIGHTGLUE_AVAILABLE = True
    except ImportError:
        pass


class LightGlueMatcher(BaseMatcher):
    """Sparse feature matcher using SuperPoint + LightGlue (kornia).

    Falls back through:
        1. SuperPoint + LightGlue  (best, requires kornia >= 0.7.2)
        2. LoFTR                   (good, requires kornia >= 0.6.7)
        3. ORB fallback            (basic, pure OpenCV)

    Parameters:
        max_keypoints:        Max features for SuperPoint (default 2048).
        confidence_threshold: Minimum per-match confidence to retain.
        device:               Torch device ('cpu', 'cuda', 'mps').
        upsample_low_res:     Factor to up-sample the lower-res image.
    """

    def __init__(
        self,
        max_keypoints: int = 2048,
        confidence_threshold: float = 0.5,
        device: str = "cpu",
        upsample_low_res: float = 1.0,
    ):
        super().__init__(confidence_threshold=confidence_threshold)
        self.max_keypoints = max_keypoints
        self.upsample_low_res = upsample_low_res

        # Resolve device
        if TORCH_AVAILABLE:
            self.device = torch.device(device)
        else:
            self.device = None

        # Determine which backend to use
        self._backend = self._resolve_backend()
        logger.info("LightGlueMatcher backend: %s", self._backend)

        # Lazy-initialised models (created on first use)
        self._superpoint = None
        self._lightglue = None
        self._loftr = None

    # -----------------------------------------------------------------
    # Backend resolution
    # -----------------------------------------------------------------

    def _resolve_backend(self) -> str:
        """Determine the best available matching backend."""
        if LIGHTGLUE_AVAILABLE:
            return "lightglue"
        if KORNIA_AVAILABLE:
            return "loftr"
        if TORCH_AVAILABLE:
            return "loftr"  # kornia might be partially available
        return "orb_fallback"

    # -----------------------------------------------------------------
    # Lazy model initialisation
    # -----------------------------------------------------------------

    def _init_superpoint(self):
        """Initialise SuperPoint detector (lazy)."""
        if self._superpoint is None and LIGHTGLUE_AVAILABLE:
            self._superpoint = KorniaSuperPoint(
                num_features=self.max_keypoints
            ).eval().to(self.device)

    def _init_lightglue(self):
        """Initialise LightGlue matcher (lazy)."""
        if self._lightglue is None and LIGHTGLUE_AVAILABLE:
            self._lightglue = KorniaLightGlue("superpoint").eval().to(self.device)

    def _init_loftr(self):
        """Initialise LoFTR matcher (lazy)."""
        if self._loftr is None and KORNIA_AVAILABLE:
            try:
                self._loftr = KorniaLoFTR(pretrained="outdoor").eval().to(self.device)
            except Exception as e:
                logger.warning("Failed to initialise LoFTR: %s", e)
                self._loftr = None

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
        """Run the matching pipeline.

        Dispatches to the best available backend (LightGlue → LoFTR → ORB).

        Args:
            src_image:      Source image (2-D float32 [0,1]).
            dst_image:      Destination image (same format).
            scale_ratio:    ``src_gsd / dst_gsd``.
            src_instrument: Source instrument name.
            dst_instrument: Destination instrument name.

        Returns:
            :class:`MatchResult` with keypoints in original pixel coords.
        """
        if self._backend == "lightglue":
            return self._match_lightglue(
                src_image, dst_image, scale_ratio,
                src_instrument, dst_instrument,
            )
        elif self._backend == "loftr":
            return self._match_loftr(
                src_image, dst_image, scale_ratio,
                src_instrument, dst_instrument,
            )
        else:
            return self._match_orb_fallback(
                src_image, dst_image, scale_ratio,
                src_instrument, dst_instrument,
            )

    # -----------------------------------------------------------------
    # Backend: SuperPoint + LightGlue
    # -----------------------------------------------------------------

    def _match_lightglue(
        self,
        src_image: np.ndarray,
        dst_image: np.ndarray,
        scale_ratio: float,
        src_instrument: str,
        dst_instrument: str,
    ) -> MatchResult:
        """Match using SuperPoint keypoints + LightGlue matcher."""
        t0 = time.time()
        actual_scale_gap = 1.0 / scale_ratio if scale_ratio > 0 else 1.0

        # Scale alignment
        src_gray = self._to_grayscale(src_image)
        dst_gray = self._to_grayscale(dst_image)
        src_gray = self._normalize_to_01(src_gray)
        dst_gray = self._normalize_to_01(dst_gray)

        aligner = ScaleAligner(upsample_factor=self.upsample_low_res)
        if scale_ratio < 1.0:
            aligned = aligner.align(src_gray, dst_gray, scale_ratio)
            aligned_src = aligned["aligned_src"]
            aligned_dst = aligned["aligned_dst"]
            src_scale = aligned["src_scale"]
            dst_scale = aligned["dst_scale"]
        else:
            aligned_src, aligned_dst = src_gray, dst_gray
            src_scale, dst_scale = 1.0, 1.0

        try:
            self._init_superpoint()
            self._init_lightglue()

            # Convert to torch tensors: (1, 1, H, W)
            t_src = torch.from_numpy(aligned_src).float().unsqueeze(0).unsqueeze(0).to(self.device)
            t_dst = torch.from_numpy(aligned_dst).float().unsqueeze(0).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                # Extract SuperPoint features
                feats_src = self._superpoint(t_src)
                feats_dst = self._superpoint(t_dst)

                # Match with LightGlue
                input_dict = {
                    "image0": {**feats_src, "image_size": torch.tensor(aligned_src.shape[:2]).unsqueeze(0)},
                    "image1": {**feats_dst, "image_size": torch.tensor(aligned_dst.shape[:2]).unsqueeze(0)},
                }
                matches_dict = self._lightglue(input_dict)

            # Extract matched keypoint indices and confidences
            matches = matches_dict.get("matches", matches_dict.get("matches0"))
            if matches is None or len(matches) == 0:
                return MatchResult.empty(src_instrument, dst_instrument, actual_scale_gap)

            # Convert to numpy
            if isinstance(matches, torch.Tensor):
                matches_np = matches.cpu().numpy()
            else:
                matches_np = np.array(matches)

            # Get keypoint coordinates
            kp_src_all = feats_src["keypoints"][0].cpu().numpy()  # (K, 2)
            kp_dst_all = feats_dst["keypoints"][0].cpu().numpy()

            valid = matches_np[:, 0] >= 0
            if not np.any(valid):
                return MatchResult.empty(src_instrument, dst_instrument, actual_scale_gap)

            idx_src = matches_np[valid, 0].astype(int)
            idx_dst = matches_np[valid, 1].astype(int) if matches_np.shape[1] > 1 else idx_src

            pts_src = kp_src_all[idx_src]
            pts_dst = kp_dst_all[idx_dst]

            # Confidence from matching scores
            scores = matches_dict.get("scores", matches_dict.get("matching_scores0"))
            if scores is not None and isinstance(scores, torch.Tensor):
                confs = scores.cpu().numpy().flatten()[:len(pts_src)]
            else:
                confs = np.ones(len(pts_src), dtype=np.float32) * 0.8

            # Map back to original coordinates
            pts_src_orig = map_keypoints_to_original(pts_src, src_scale)
            pts_dst_orig = map_keypoints_to_original(pts_dst, dst_scale)

            elapsed = time.time() - t0

            result = MatchResult(
                keypoints_src=pts_src_orig,
                keypoints_dst=pts_dst_orig,
                confidences=confs.astype(np.float32),
                src_instrument=src_instrument,
                dst_instrument=dst_instrument,
                scale_gap=actual_scale_gap,
                num_inliers=len(pts_src),
                match_confidence=float(np.mean(confs)),
                metadata={
                    "method": "SuperPoint+LightGlue",
                    "num_features_src": len(kp_src_all),
                    "num_features_dst": len(kp_dst_all),
                    "num_matches": len(pts_src),
                    "elapsed_seconds": elapsed,
                    "device": str(self.device),
                },
            )
            return self._filter_by_confidence(result)

        except Exception as e:
            logger.warning("LightGlue failed (%s), falling back to LoFTR", e)
            return self._match_loftr(
                src_image, dst_image, scale_ratio,
                src_instrument, dst_instrument,
            )

    # -----------------------------------------------------------------
    # Backend: LoFTR (dense matcher)
    # -----------------------------------------------------------------

    def _match_loftr(
        self,
        src_image: np.ndarray,
        dst_image: np.ndarray,
        scale_ratio: float,
        src_instrument: str,
        dst_instrument: str,
    ) -> MatchResult:
        """Match using LoFTR dense matcher."""
        t0 = time.time()
        actual_scale_gap = 1.0 / scale_ratio if scale_ratio > 0 else 1.0

        src_gray = self._normalize_to_01(self._to_grayscale(src_image))
        dst_gray = self._normalize_to_01(self._to_grayscale(dst_image))

        aligner = ScaleAligner(upsample_factor=self.upsample_low_res)
        if scale_ratio < 1.0:
            aligned = aligner.align(src_gray, dst_gray, scale_ratio)
            aligned_src = aligned["aligned_src"]
            aligned_dst = aligned["aligned_dst"]
            src_scale = aligned["src_scale"]
            dst_scale = aligned["dst_scale"]
        else:
            aligned_src, aligned_dst = src_gray, dst_gray
            src_scale, dst_scale = 1.0, 1.0

        try:
            self._init_loftr()
            if self._loftr is None:
                raise RuntimeError("LoFTR model failed to initialise")

            # LoFTR requires images with dimensions divisible by 8
            h_s, w_s = aligned_src.shape[:2]
            h_d, w_d = aligned_dst.shape[:2]
            h_s8, w_s8 = (h_s // 8) * 8, (w_s // 8) * 8
            h_d8, w_d8 = (h_d // 8) * 8, (w_d // 8) * 8

            src_crop = aligned_src[:h_s8, :w_s8]
            dst_crop = aligned_dst[:h_d8, :w_d8]

            t_src = torch.from_numpy(src_crop).float().unsqueeze(0).unsqueeze(0).to(self.device)
            t_dst = torch.from_numpy(dst_crop).float().unsqueeze(0).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                input_dict = {"image0": t_src, "image1": t_dst}
                result_dict = self._loftr(input_dict)

            kpts0 = result_dict["keypoints0"].cpu().numpy()
            kpts1 = result_dict["keypoints1"].cpu().numpy()
            confs = result_dict["confidence"].cpu().numpy()

            if len(kpts0) == 0:
                return MatchResult.empty(src_instrument, dst_instrument, actual_scale_gap)

            pts_src_orig = map_keypoints_to_original(kpts0, src_scale)
            pts_dst_orig = map_keypoints_to_original(kpts1, dst_scale)

            elapsed = time.time() - t0

            result = MatchResult(
                keypoints_src=pts_src_orig,
                keypoints_dst=pts_dst_orig,
                confidences=confs.astype(np.float32),
                src_instrument=src_instrument,
                dst_instrument=dst_instrument,
                scale_gap=actual_scale_gap,
                num_inliers=len(kpts0),
                match_confidence=float(np.mean(confs)),
                metadata={
                    "method": "LoFTR",
                    "num_matches": len(kpts0),
                    "elapsed_seconds": elapsed,
                    "device": str(self.device),
                },
            )
            return self._filter_by_confidence(result)

        except Exception as e:
            logger.warning("LoFTR failed (%s), falling back to ORB", e)
            return self._match_orb_fallback(
                src_image, dst_image, scale_ratio,
                src_instrument, dst_instrument,
            )

    # -----------------------------------------------------------------
    # Backend: ORB fallback
    # -----------------------------------------------------------------

    def _match_orb_fallback(
        self,
        src_image: np.ndarray,
        dst_image: np.ndarray,
        scale_ratio: float,
        src_instrument: str,
        dst_instrument: str,
    ) -> MatchResult:
        """Delegate to the ORBFallbackMatcher."""
        from .orb_fallback_matcher import ORBFallbackMatcher

        logger.info("Using ORB fallback matcher")
        orb = ORBFallbackMatcher(
            max_keypoints=self.max_keypoints,
            confidence_threshold=self.confidence_threshold,
            upsample_low_res=self.upsample_low_res,
        )
        return orb.match(
            src_image, dst_image, scale_ratio,
            src_instrument, dst_instrument,
        )

    # -----------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------

    @property
    def backend(self) -> str:
        """Name of the active matching backend."""
        return self._backend

    @staticmethod
    def torch_available() -> bool:
        return TORCH_AVAILABLE

    @staticmethod
    def kornia_available() -> bool:
        return KORNIA_AVAILABLE

    @staticmethod
    def lightglue_available() -> bool:
        return LIGHTGLUE_AVAILABLE
