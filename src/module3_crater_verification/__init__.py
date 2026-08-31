"""
Module 3: Crater-Structural Verification.

This module provides structural verification for matches obtained in Module 2.
It relies on mathematically rigorous topography extraction (crater rims, ridges)
to verify that putative matches physically align with illumination-invariant
lunar features.
"""

from .structure_extractor import StructuralExtractor
from .verifier import StructuralVerifier
from .structural_matcher import StructuralMatcher

__all__ = [
    "StructuralExtractor",
    "StructuralVerifier",
    "StructuralMatcher",
]
