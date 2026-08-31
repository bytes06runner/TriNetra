"""
Module 2 — Hierarchical Cross-Instrument Matcher.

Implements the hub-and-spoke matching architecture with TMC-2 as anchor:

    OHRC ←(Hop 1, 20×)→ TMC-2 ←(Hop 2, 16×)→ IIRS

Classes:
    MatchResult          — Canonical match output (keypoints, confidences)
    BaseMatcher          — Abstract base for all matchers
    LightGlueMatcher     — SuperPoint + LightGlue (kornia/torch)
    ORBFallbackMatcher   — ORB + BFMatcher (pure OpenCV fallback)
    HubAndSpokeMatcher   — Orchestrator routing through TMC-2 hub
    GaussianPyramid      — Anti-aliased multi-scale pyramid builder
    ScaleAligner         — GSD-aware image pair alignment
"""

from .base_matcher import MatchResult, BaseMatcher
from .scale_handler import GaussianPyramid, ScaleAligner, map_keypoints_to_original
from .orb_fallback_matcher import ORBFallbackMatcher
from .lightglue_matcher import LightGlueMatcher
from .hub_matcher import HubAndSpokeMatcher

__all__ = [
    "MatchResult",
    "BaseMatcher",
    "LightGlueMatcher",
    "ORBFallbackMatcher",
    "HubAndSpokeMatcher",
    "GaussianPyramid",
    "ScaleAligner",
    "map_keypoints_to_original",
]
