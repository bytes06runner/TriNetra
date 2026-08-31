"""Instrument Specifications for Chandrayaan-2 Lunar Payloads.

SIH26166 — Multi-Modal Lunar Image Correspondence Pipeline.
Ground-truth optical, geometric, and radiometric specifications for Chandrayaan-2
payloads: Orbiter High Resolution Camera (OHRC), Terrain Mapping Camera-2 (TMC-2),
and Imaging Infra-Red Spectrometer (IIRS).

References:
    - ISRO Chandrayaan-2 Mission Specification Documents (ISSDC/ISRO).
    - Kumar et al. (2020), "Orbiter High Resolution Camera (OHRC) on Chandrayaan-2".
    - Roy et al. (2020), "Terrain Mapping Camera-2 (TMC-2) on Chandrayaan-2".
    - Chowdhury et al. (2020), "Imaging Infrared Spectrometer (IIRS) on Chandrayaan-2".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import Final, Literal, Optional, Union


@unique
class InstrumentType(str, Enum):
    """Enumeration of Chandrayaan-2 imaging payloads supported in the pipeline.

    Attributes:
        OHRC: Orbiter High Resolution Camera (nadir & oblique panchromatic visible).
        TMC2: Terrain Mapping Camera-2 (triplet stereo panchromatic visible).
        IIRS: Imaging Infra-Red Spectrometer (hyperspectral infrared).
    """

    OHRC = "OHRC"
    TMC2 = "TMC2"
    IIRS = "IIRS"

    @classmethod
    def from_str(cls, value: str) -> InstrumentType:
        """Parse an instrument name string into an InstrumentType enum member.

        Handles common aliases, punctuation variations (e.g. 'TMC-2', 'TMC_2'),
        and case variations.

        Args:
            value: Instrument name string (e.g. 'OHRC', 'tmc-2', 'iirs').

        Returns:
            InstrumentType: Matching enum member.

        Raises:
            ValueError: If the string does not match any recognized instrument.
        """
        if not isinstance(value, str):
            raise TypeError(f"Expected str, got {type(value).__name__}: {value!r}")

        normalized = value.strip().upper().replace("-", "").replace("_", "").replace(" ", "")

        aliases: dict[str, InstrumentType] = {
            "OHRC": cls.OHRC,
            "ORBITERHIGHRESOLUTIONCAMERA": cls.OHRC,
            "TMC2": cls.TMC2,
            "TMC": cls.TMC2,
            "TERRAINMAPPINGCAMERA2": cls.TMC2,
            "TERRAINMAPPINGCAMERA": cls.TMC2,
            "IIRS": cls.IIRS,
            "IMAGINGINFRAREDSPECTROMETER": cls.IIRS,
            "IMAGINGINFRAREDSPECTROMETER2": cls.IIRS,
        }

        if normalized in aliases:
            return aliases[normalized]

        valid_options = ", ".join(f"'{m.value}'" for m in cls)
        raise ValueError(
            f"Unknown instrument identifier '{value}'. Valid options are: {valid_options}."
        )


@dataclass(frozen=True)
class InstrumentSpec:
    """Immutable ground-truth optical, geometric, and radiometric specification.

    Attributes:
        name: Official acronym or display name of the instrument.
        gsd_m: Ground Sample Distance (spatial resolution) in meters per pixel at nadir.
        gsd_oblique_m: Ground Sample Distance in meters per pixel in oblique viewing mode
            (OHRC only, None for nadir-only instruments).
        swath_km: Cross-track swath width on the lunar surface in kilometers at 100 km altitude.
        spectral_type: Radiometric classification ('panchromatic_visible' | 'hyperspectral_infrared').
        spectral_range_um: Wavelength coverage interval (min_um, max_um) in micrometers (µm).
        num_bands: Total number of spectral acquisition bands.
        bit_depth: Radiometric quantization bit depth per pixel/band (e.g., 12-bit).
        detector_type: Physical detector architecture (e.g. 'TDI CCD', 'Linear APS array', 'HgCdTe MCT').
        orbit_altitude_km: Nominal circular lunar polar orbit altitude in kilometers.
    """

    name: str
    gsd_m: float
    gsd_oblique_m: Optional[float]
    swath_km: float
    spectral_type: Literal["panchromatic_visible", "hyperspectral_infrared"] | str
    spectral_range_um: tuple[float, float]
    num_bands: int
    bit_depth: int
    detector_type: str
    orbit_altitude_km: float

    def __post_init__(self) -> None:
        """Validate invariant physical constraints on payload specifications."""
        if self.gsd_m <= 0.0:
            raise ValueError(f"gsd_m must be positive, got {self.gsd_m}")
        if self.gsd_oblique_m is not None and self.gsd_oblique_m <= 0.0:
            raise ValueError(f"gsd_oblique_m must be positive, got {self.gsd_oblique_m}")
        if self.swath_km <= 0.0:
            raise ValueError(f"swath_km must be positive, got {self.swath_km}")
        if len(self.spectral_range_um) != 2 or self.spectral_range_um[0] >= self.spectral_range_um[1]:
            raise ValueError(f"Invalid spectral_range_um: {self.spectral_range_um}")
        if self.num_bands < 1:
            raise ValueError(f"num_bands must be >= 1, got {self.num_bands}")
        if self.bit_depth < 1:
            raise ValueError(f"bit_depth must be >= 1, got {self.bit_depth}")
        if self.orbit_altitude_km <= 0.0:
            raise ValueError(f"orbit_altitude_km must be positive, got {self.orbit_altitude_km}")

    @property
    def max_digital_number(self) -> int:
        """Maximum possible raw digital number (DN) for the instrument's bit depth.

        Returns:
            int: 2^bit_depth - 1 (e.g. 4095 for 12-bit data).
        """
        return (1 << self.bit_depth) - 1

    @property
    def is_hyperspectral(self) -> bool:
        """Check if instrument acquires hyperspectral data (>10 bands)."""
        return self.spectral_type == "hyperspectral_infrared" or self.num_bands > 10

    @property
    def is_panchromatic(self) -> bool:
        """Check if instrument acquires single-band panchromatic data."""
        return self.spectral_type == "panchromatic_visible" and self.num_bands == 1

    def scale_ratio_to(self, other: InstrumentSpec) -> float:
        """Compute the spatial scale factor relative to another instrument.

        Calculates `self.gsd_m / other.gsd_m`. A value > 1.0 indicates that `self`
        has a coarser resolution than `other` (e.g., TMC2 to OHRC is 20.0x).

        Args:
            other: Target instrument specification.

        Returns:
            float: Spatial scale ratio (self.gsd_m / other.gsd_m).
        """
        return float(self.gsd_m / other.gsd_m)

    def spectral_overlap_with(self, other: InstrumentSpec) -> Optional[tuple[float, float]]:
        """Compute spectral range overlap with another instrument in micrometers.

        Args:
            other: Target instrument specification.

        Returns:
            Optional[tuple[float, float]]: Overlapping (min_um, max_um) interval,
                or None if no spectral overlap exists.
        """
        overlap_min = max(self.spectral_range_um[0], other.spectral_range_um[0])
        overlap_max = min(self.spectral_range_um[1], other.spectral_range_um[1])
        if overlap_min < overlap_max:
            return (overlap_min, overlap_max)
        return None


# ---------------------------------------------------------------------------
# Ground-Truth Payload Specifications (ISRO Chandrayaan-2)
# ---------------------------------------------------------------------------

OHRC_SPEC: Final[InstrumentSpec] = InstrumentSpec(
    name="OHRC",
    gsd_m=0.25,
    gsd_oblique_m=0.32,
    swath_km=3.0,
    spectral_type="panchromatic_visible",
    spectral_range_um=(0.45, 0.70),
    num_bands=1,
    bit_depth=12,
    detector_type="TDI CCD",
    orbit_altitude_km=100.0,
)
"""Ground-truth specification for Orbiter High Resolution Camera (OHRC).

- Spatial resolution: 0.25 m/px at nadir, 0.32 m/px in oblique mode (+/-25 deg).
- Swath width: 3.0 km from 100 km orbit.
- Spectral band: Panchromatic visible (0.45 - 0.70 µm).
- Radiometry: 12-bit quantization, Time Delay Integration (TDI) CCD detector.
"""

TMC2_SPEC: Final[InstrumentSpec] = InstrumentSpec(
    name="TMC2",
    gsd_m=5.0,
    gsd_oblique_m=None,
    swath_km=20.0,
    spectral_type="panchromatic_visible",
    spectral_range_um=(0.50, 0.80),
    num_bands=1,
    bit_depth=12,
    detector_type="Linear APS array",
    orbit_altitude_km=100.0,
)
"""Ground-truth specification for Terrain Mapping Camera-2 (TMC-2).

- Spatial resolution: 5.0 m/px across Fore, Nadir, Aft triplet views.
- Swath width: 20.0 km from 100 km orbit.
- Spectral band: Panchromatic visible-NIR (0.50 - 0.80 µm).
- Radiometry: 12-bit quantization, Linear Active Pixel Sensor (APS) array.
"""

IIRS_SPEC: Final[InstrumentSpec] = InstrumentSpec(
    name="IIRS",
    gsd_m=80.0,
    gsd_oblique_m=None,
    swath_km=20.0,
    spectral_type="hyperspectral_infrared",
    spectral_range_um=(0.80, 5.00),
    num_bands=256,
    bit_depth=12,
    detector_type="HgCdTe MCT",
    orbit_altitude_km=100.0,
)
"""Ground-truth specification for Imaging Infra-Red Spectrometer (IIRS).

- Spatial resolution: 80.0 m/px at nadir.
- Swath width: 20.0 km from 100 km orbit.
- Spectral band: Hyperspectral infrared (0.80 - 5.00 µm) sampled across 256 contiguous channels (~20 nm bandpass).
- Radiometry: 12-bit quantization, Mercury Cadmium Telluride (HgCdTe / MCT) focal plane array with Stirling cooler.
"""

# ---------------------------------------------------------------------------
# Global Lookup Dictionaries
# ---------------------------------------------------------------------------

SPECS: Final[dict[InstrumentType, InstrumentSpec]] = {
    InstrumentType.OHRC: OHRC_SPEC,
    InstrumentType.TMC2: TMC2_SPEC,
    InstrumentType.IIRS: IIRS_SPEC,
}
"""Mapping from InstrumentType enum member to corresponding InstrumentSpec."""


# ---------------------------------------------------------------------------
# Computed Cross-Modal Scale Gap Constants
# ---------------------------------------------------------------------------

SCALE_GAP_TMC2_OHRC: Final[float] = float(TMC2_SPEC.gsd_m / OHRC_SPEC.gsd_m)
"""Spatial scale gap between TMC-2 (5.0 m/px) and OHRC (0.25 m/px) = 20.0x."""

SCALE_GAP_IIRS_TMC2: Final[float] = float(IIRS_SPEC.gsd_m / TMC2_SPEC.gsd_m)
"""Spatial scale gap between IIRS (80.0 m/px) and TMC-2 (5.0 m/px) = 16.0x."""

SCALE_GAP_IIRS_OHRC: Final[float] = float(IIRS_SPEC.gsd_m / OHRC_SPEC.gsd_m)
"""Spatial scale gap between IIRS (80.0 m/px) and OHRC (0.25 m/px) = 320.0x."""


# Scale gap lookup dictionary for arbitrary pairs
SCALE_GAPS: Final[dict[tuple[InstrumentType, InstrumentType], float]] = {
    (InstrumentType.TMC2, InstrumentType.OHRC): SCALE_GAP_TMC2_OHRC,
    (InstrumentType.OHRC, InstrumentType.TMC2): 1.0 / SCALE_GAP_TMC2_OHRC,
    (InstrumentType.IIRS, InstrumentType.TMC2): SCALE_GAP_IIRS_TMC2,
    (InstrumentType.TMC2, InstrumentType.IIRS): 1.0 / SCALE_GAP_IIRS_TMC2,
    (InstrumentType.IIRS, InstrumentType.OHRC): SCALE_GAP_IIRS_OHRC,
    (InstrumentType.OHRC, InstrumentType.IIRS): 1.0 / SCALE_GAP_IIRS_OHRC,
    (InstrumentType.OHRC, InstrumentType.OHRC): 1.0,
    (InstrumentType.TMC2, InstrumentType.TMC2): 1.0,
    (InstrumentType.IIRS, InstrumentType.IIRS): 1.0,
}
"""Lookup table for cross-sensor spatial resolution scale ratios (coarser / finer)."""


# ---------------------------------------------------------------------------
# Helper Lookup Functions
# ---------------------------------------------------------------------------

def get_spec(instrument: Union[InstrumentType, str]) -> InstrumentSpec:
    """Retrieve the ground-truth specification for a given Chandrayaan-2 instrument.

    Accepts either an `InstrumentType` enum or a string identifier (e.g. 'OHRC',
    'TMC-2', 'tmc2', 'IIRS', 'iirs').

    Args:
        instrument: Instrument enum member or string name.

    Returns:
        InstrumentSpec: Frozen dataclass with instrument parameters.

    Raises:
        ValueError: If instrument string or enum is unrecognized.
        TypeError: If instrument is neither InstrumentType nor str.

    Examples:
        >>> spec = get_spec("OHRC")
        >>> spec.gsd_m
        0.25
        >>> spec_tmc = get_spec(InstrumentType.TMC2)
        >>> spec_tmc.swath_km
        20.0
    """
    if isinstance(instrument, InstrumentType):
        if instrument in SPECS:
            return SPECS[instrument]
        raise ValueError(f"Unmapped InstrumentType: {instrument}")

    if isinstance(instrument, str):
        parsed_type = InstrumentType.from_str(instrument)
        return SPECS[parsed_type]

    raise TypeError(
        f"Expected InstrumentType or str, got {type(instrument).__name__}: {instrument!r}"
    )


def get_scale_gap(
    source_instrument: Union[InstrumentType, str],
    target_instrument: Union[InstrumentType, str],
) -> float:
    """Compute the spatial resolution scale ratio between two instruments.

    Returns `source_gsd_m / target_gsd_m`.

    Args:
        source_instrument: Source instrument enum or string.
        target_instrument: Target instrument enum or string.

    Returns:
        float: Scale ratio (e.g. get_scale_gap("TMC-2", "OHRC") == 20.0).

    Examples:
        >>> get_scale_gap("TMC2", "OHRC")
        20.0
        >>> get_scale_gap("IIRS", "TMC2")
        16.0
        >>> get_scale_gap("IIRS", "OHRC")
        320.0
    """
    source_spec = get_spec(source_instrument)
    target_spec = get_spec(target_instrument)
    return source_spec.scale_ratio_to(target_spec)


__all__ = [
    "InstrumentType",
    "InstrumentSpec",
    "OHRC_SPEC",
    "TMC2_SPEC",
    "IIRS_SPEC",
    "SPECS",
    "SCALE_GAP_TMC2_OHRC",
    "SCALE_GAP_IIRS_TMC2",
    "SCALE_GAP_IIRS_OHRC",
    "SCALE_GAPS",
    "get_spec",
    "get_scale_gap",
]
