"""
Module 1 — Multi-modal Preprocessing Pipeline.

Handles per-instrument radiometric correction, spectral dimensionality
reduction (IIRS), and coarse footprint-based search-space limiting.

Classes:
    OHRCPreprocessor  — CLAHE with shadow-edge preservation
    TMC2Preprocessor  — Standard radiometric normalization
    IIRSPreprocessor  — Band selection + PCA dimensionality reduction
    FootprintParser   — SPICE-kernel metadata ⇒ coarse bounding boxes
"""

from .preprocessor import (
    BasePreprocessor,
    OHRCPreprocessor,
    TMC2Preprocessor,
    PreprocessingResult,
)
from .iirs_pca import IIRSPreprocessor
from .metadata_parser import FootprintParser, InstrumentFootprint

__all__ = [
    "BasePreprocessor",
    "OHRCPreprocessor",
    "TMC2Preprocessor",
    "IIRSPreprocessor",
    "FootprintParser",
    "InstrumentFootprint",
    "PreprocessingResult",
]
