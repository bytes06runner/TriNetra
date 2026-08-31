"""
Hub-and-spoke matching orchestrator.

Enforces the architectural constraint that **TMC-2 is the anchor**:

    OHRC ←(Hop 1, 20×)→ TMC-2 ←(Hop 2, 16×)→ IIRS

Direct OHRC ↔ IIRS matching (320×) is **never** attempted.

This module:
 1. Accepts preprocessed images from Module 1.
 2. Dispatches Hop 1 and Hop 2 to the appropriate matcher.
 3. Returns independent :class:`MatchResult` objects for each hop —
    geometric composition is handled downstream by Module 4 (registration).

SIH26166 — Chandrayaan-2 Multi-modal Image Correspondence Pipeline
"""

import numpy as np
import logging
import time
from typing import Optional, Dict, Any

from .base_matcher import BaseMatcher, MatchResult
from .lightglue_matcher import LightGlueMatcher
from .orb_fallback_matcher import ORBFallbackMatcher

logger = logging.getLogger(__name__)

# Scale ratios (src_gsd / dst_gsd) for each hop
SCALE_RATIO_HOP1 = 0.25 / 5.0    # OHRC → TMC-2:  0.05  (20× gap)
SCALE_RATIO_HOP2 = 5.0 / 80.0    # TMC-2 → IIRS:  0.0625 (16× gap)


class HubAndSpokeMatcher:
    """Orchestrates matching through TMC-2 as the hub instrument.

    Supported pairings:

    +-----------+--------+-------+--------+---------+
    | Pairing   | Hop    | Gap   | Method | Modality|
    +===========+========+=======+========+=========+
    | OHRC↔TMC  | Hop 1  | 20×   | LG/ORB | Same    |
    | TMC↔IIRS  | Hop 2  | 16×   | LG/ORB | Cross*  |
    | OHRC↔IIRS | Chain  | 320×  | H1+H2  | —       |
    +-----------+--------+-------+--------+---------+

    * Hop 2 uses the IIRS PCA proxy (2-D panchromatic-like image),
      so matching is structurally similar to same-modality.

    Parameters:
        hop1_matcher: Matcher instance for OHRC ↔ TMC-2 (default: LightGlue).
        hop2_matcher: Matcher instance for TMC-2 ↔ IIRS (default: LightGlue).
        device:       Torch device for learned matchers.
    """

    def __init__(
        self,
        hop1_matcher: Optional[BaseMatcher] = None,
        hop2_matcher: Optional[BaseMatcher] = None,
        device: str = "cpu",
    ):
        self.device = device

        # Default hop 1 matcher: LightGlue (falls back to ORB internally)
        self.hop1_matcher = hop1_matcher or LightGlueMatcher(
            max_keypoints=2048,
            confidence_threshold=0.4,
            device=device,
        )

        # Default hop 2 matcher: LightGlue with lower threshold
        # (cross-modal matching is harder, accept weaker matches)
        self.hop2_matcher = hop2_matcher or LightGlueMatcher(
            max_keypoints=2048,
            confidence_threshold=0.3,
            device=device,
        )

        self.logger = logging.getLogger(self.__class__.__name__)

    # -----------------------------------------------------------------
    # Hop 1: OHRC ↔ TMC-2
    # -----------------------------------------------------------------

    def match_hop1(
        self,
        ohrc_image: np.ndarray,
        tmc2_image: np.ndarray,
    ) -> MatchResult:
        """Match OHRC against TMC-2 (same modality, 20× scale gap).

        Args:
            ohrc_image: Preprocessed OHRC image (2-D float32 [0,1]).
            tmc2_image: Preprocessed TMC-2 image (2-D float32 [0,1]).

        Returns:
            :class:`MatchResult` with keypoints in original pixel coords
            of both instruments.
        """
        self.logger.info(
            "Hop 1: OHRC (%s) ↔ TMC-2 (%s), scale ratio %.4f",
            ohrc_image.shape, tmc2_image.shape, SCALE_RATIO_HOP1,
        )

        try:
            result = self.hop1_matcher.match(
                src_image=ohrc_image,
                dst_image=tmc2_image,
                scale_ratio=SCALE_RATIO_HOP1,
                src_instrument="OHRC",
                dst_instrument="TMC-2",
            )
        except Exception as e:
            self.logger.error("Hop 1 matching failed: %s", e)
            result = MatchResult.empty("OHRC", "TMC-2", 20.0)

        self.logger.info(
            "Hop 1 result: %d matches, confidence %.3f",
            result.num_matches, result.match_confidence,
        )
        return result

    # -----------------------------------------------------------------
    # Hop 2: TMC-2 ↔ IIRS (proxy)
    # -----------------------------------------------------------------

    def match_hop2(
        self,
        tmc2_image: np.ndarray,
        iirs_proxy: np.ndarray,
    ) -> MatchResult:
        """Match TMC-2 against the IIRS PCA proxy (16× scale gap).

        The IIRS proxy is the 2-D panchromatic-like image produced by
        Module 1's :class:`IIRSPreprocessor` (PCA / weighted / albedo).

        Args:
            tmc2_image: Preprocessed TMC-2 image (2-D float32 [0,1]).
            iirs_proxy: IIRS PCA proxy image (2-D float32 [0,1]).

        Returns:
            :class:`MatchResult` with keypoints in original pixel coords.
        """
        self.logger.info(
            "Hop 2: TMC-2 (%s) ↔ IIRS proxy (%s), scale ratio %.4f",
            tmc2_image.shape, iirs_proxy.shape, SCALE_RATIO_HOP2,
        )

        try:
            result = self.hop2_matcher.match(
                src_image=tmc2_image,
                dst_image=iirs_proxy,
                scale_ratio=SCALE_RATIO_HOP2,
                src_instrument="TMC-2",
                dst_instrument="IIRS",
            )
        except Exception as e:
            self.logger.error("Hop 2 matching failed: %s", e)
            result = MatchResult.empty("TMC-2", "IIRS", 16.0)

        self.logger.info(
            "Hop 2 result: %d matches, confidence %.3f",
            result.num_matches, result.match_confidence,
        )
        return result

    # -----------------------------------------------------------------
    # Composite: OHRC ↔ IIRS  (via TMC-2 hub)
    # -----------------------------------------------------------------

    def match_composite(
        self,
        ohrc_image: np.ndarray,
        tmc2_image: np.ndarray,
        iirs_proxy: np.ndarray,
    ) -> Dict[str, Any]:
        """Chain Hop 1 + Hop 2 through the TMC-2 hub.

        This does **not** attempt to algebraically compose point
        correspondences (that is Module 4's job).  Instead it returns
        both independent hop results for the registration module.

        Args:
            ohrc_image: Preprocessed OHRC image.
            tmc2_image: Preprocessed TMC-2 image.
            iirs_proxy: IIRS PCA proxy image.

        Returns:
            dict with keys:
                ``hop1``       — MatchResult for OHRC ↔ TMC-2
                ``hop2``       — MatchResult for TMC-2 ↔ IIRS
                ``success``    — True if both hops have ≥1 match
                ``confidence`` — geometric mean of hop confidences
        """
        t0 = time.time()

        hop1 = self.match_hop1(ohrc_image, tmc2_image)
        hop2 = self.match_hop2(tmc2_image, iirs_proxy)

        elapsed = time.time() - t0

        # Composite confidence: geometric mean of hop confidences
        if hop1.match_confidence > 0 and hop2.match_confidence > 0:
            composite_confidence = np.sqrt(
                hop1.match_confidence * hop2.match_confidence
            )
        else:
            composite_confidence = 0.0

        success = hop1.has_matches and hop2.has_matches

        self.logger.info(
            "Composite match: Hop1=%d matches (%.3f), Hop2=%d matches (%.3f), "
            "composite_confidence=%.3f, success=%s, elapsed=%.2fs",
            hop1.num_matches, hop1.match_confidence,
            hop2.num_matches, hop2.match_confidence,
            composite_confidence, success, elapsed,
        )

        return {
            "hop1": hop1,
            "hop2": hop2,
            "success": success,
            "confidence": float(composite_confidence),
            "elapsed_seconds": elapsed,
        }

    # -----------------------------------------------------------------
    # Direct pairing router
    # -----------------------------------------------------------------

    def match(
        self,
        src_image: np.ndarray,
        dst_image: np.ndarray,
        src_instrument: str,
        dst_instrument: str,
        tmc2_image: Optional[np.ndarray] = None,
    ) -> MatchResult:
        """Route a matching request to the correct hop(s).

        This convenience method determines the correct pairing and
        dispatches accordingly.  For OHRC ↔ IIRS, a TMC-2 hub image
        is required.

        Args:
            src_image:      Source preprocessed image.
            dst_image:      Destination preprocessed image.
            src_instrument: Source instrument name (case-insensitive).
            dst_instrument: Destination instrument name.
            tmc2_image:     TMC-2 hub image (required for OHRC↔IIRS).

        Returns:
            :class:`MatchResult` (for direct hops) or raises ValueError
            if attempting direct OHRC↔IIRS without the TMC-2 hub.
        """
        src_inst = src_instrument.upper().replace("-", "").replace(" ", "")
        dst_inst = dst_instrument.upper().replace("-", "").replace(" ", "")

        pair = frozenset({src_inst, dst_inst})

        if pair == frozenset({"OHRC", "TMC2"}):
            # Ensure OHRC is source (higher-res)
            if src_inst == "TMC2":
                return self.match_hop1(dst_image, src_image)
            return self.match_hop1(src_image, dst_image)

        if pair == frozenset({"TMC2", "IIRS"}):
            # Ensure TMC-2 is source (higher-res)
            if src_inst == "IIRS":
                return self.match_hop2(dst_image, src_image)
            return self.match_hop2(src_image, dst_image)

        if pair == frozenset({"OHRC", "IIRS"}):
            if tmc2_image is None:
                raise ValueError(
                    "Direct OHRC ↔ IIRS matching (320× gap) is not supported. "
                    "Provide a TMC-2 hub image via tmc2_image= to use "
                    "the composite (Hop 1 + Hop 2) path."
                )
            # Use composite — return only the hop relevant to caller
            composite = self.match_composite(src_image, tmc2_image, dst_image)
            # Return hop1 since the caller asked for OHRC-side matches
            return composite["hop1"]

        raise ValueError(
            f"Unknown instrument pair: {src_instrument} ↔ {dst_instrument}. "
            f"Supported instruments: OHRC, TMC-2, IIRS."
        )
