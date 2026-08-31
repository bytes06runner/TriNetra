"""
Module 4: Geometric Registration.

Uses robust estimators (MAGSAC++) to compute transformation matrices
(Homography/Affine) between matched instrument images.
"""

from .registration import GeometricRegistrar, RegistrationResult

__all__ = [
    "GeometricRegistrar",
    "RegistrationResult",
]
