"""
Scale-aware image handling for cross-instrument matching.

Provides a Gaussian pyramid builder and a scale aligner that prepare
image pairs with extreme GSD ratios (up to 20× for OHRC→TMC-2) for
downstream feature matching.

SIH26166 — Chandrayaan-2 Multi-modal Image Correspondence Pipeline
"""

import numpy as np
import cv2
import logging
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gaussian Pyramid
# ---------------------------------------------------------------------------

@dataclass
class PyramidLevel:
    """One level of a Gaussian pyramid.

    Attributes:
        image:      Downsampled image (float32, [0, 1]).
        scale:      Cumulative scale factor relative to the original
                    (e.g. 0.25 means 4× downsampled).
        level:      Zero-based pyramid level index.
        shape:      (H, W) of this level.
    """
    image: np.ndarray
    scale: float
    level: int
    shape: Tuple[int, int] = field(init=False)

    def __post_init__(self):
        self.shape = self.image.shape[:2]


def compute_scale_ratio(high_res_gsd: float, low_res_gsd: float) -> float:
    """Compute the scale ratio between high-resolution and low-resolution GSDs.

    For example, TMC-2 (4.96 m/px) and IIRS (91.75 m/px) yields:
        4.96 / 91.75 = 0.05406  (i.e. ~18.5× scale factor).

    Args:
        high_res_gsd: Ground sample distance of the finer instrument (m/px).
        low_res_gsd:  Ground sample distance of the coarser instrument (m/px).

    Returns:
        float scale ratio: high_res_gsd / low_res_gsd.
    """
    if high_res_gsd <= 0 or low_res_gsd <= 0:
        raise ValueError(f"GSDs must be positive: high={high_res_gsd}, low={low_res_gsd}")
    return float(high_res_gsd) / float(low_res_gsd)


def compute_pyramid_depth(scale_ratio: float) -> int:
    """Compute Gaussian pyramid depth dynamically from the actual scale ratio.

    Instead of assuming a hardcoded constant (like 20× or 8 levels), this
    computes ceil(log2(downsample_factor)) + 1 to ensure the pyramid reaches
    the target resolution level.

    For 18.5× downsampling (TMC-2 → IIRS):
        log2(18.5) = 4.209 → ceil is 5 → depth = 6 levels (1×, 2×, 4×, 8×, 16×, 32×).

    Args:
        scale_ratio: high_res_gsd / low_res_gsd (e.g. 0.05406 for 18.5×).

    Returns:
        Integer number of pyramid levels required.
    """
    factor = 1.0 / scale_ratio if scale_ratio < 1.0 else scale_ratio
    return max(2, int(np.ceil(np.log2(factor))) + 1)


class GaussianPyramid:
    """Build an anti-aliased Gaussian pyramid from a high-resolution image.

    Each level is produced by Gaussian-blurring then downsampling by a
    factor of 2 (standard Laplacian pyramid construction).  The pyramid
    depth can be computed dynamically from the target scale ratio.

    Example for TMC-2 (4.96 m) → IIRS (91.75 m) alignment:
        ratio = compute_scale_ratio(4.96, 91.75)  # 0.05406 (~18.5×)
        pyr = GaussianPyramid(tmc2_image, target_scale=ratio)
        level = pyr.get_level_for_scale(ratio)
    """

    def __init__(
        self,
        image: np.ndarray,
        max_levels: Optional[int] = None,
        target_scale: Optional[float] = None,
        blur_ksize: int = 5,
    ):
        """
        Args:
            image:        Source image (2-D float32 [0, 1]).
            max_levels:   Maximum number of pyramid levels to build.
                          If None, computed dynamically from target_scale (or defaults to 8).
            target_scale: Desired downsampling ratio used to compute pyramid depth.
            blur_ksize:   Gaussian blur kernel size used before each 2× down.
        """
        if image.ndim != 2:
            raise ValueError("GaussianPyramid expects a 2-D grayscale image.")

        self.original = image.astype(np.float32)
        if max_levels is not None:
            self.max_levels = max_levels
        elif target_scale is not None:
            self.max_levels = compute_pyramid_depth(target_scale)
        else:
            self.max_levels = 8
        self.blur_ksize = blur_ksize
        self._levels: List[PyramidLevel] = []
        self._built = False

    def build(self) -> "GaussianPyramid":
        """Construct all pyramid levels (idempotent)."""
        if self._built:
            return self

        self._levels = [
            PyramidLevel(image=self.original.copy(), scale=1.0, level=0)
        ]

        current = self.original.copy()
        for lvl in range(1, self.max_levels):
            h, w = current.shape[:2]
            if h < 8 or w < 8:
                break  # too small to downsample further

            # Anti-alias blur then 2× downsample
            blurred = cv2.GaussianBlur(
                current, (self.blur_ksize, self.blur_ksize), 0
            )
            current = cv2.resize(
                blurred,
                (max(1, w // 2), max(1, h // 2)),
                interpolation=cv2.INTER_AREA,
            )
            scale = current.shape[1] / self.original.shape[1]
            self._levels.append(
                PyramidLevel(image=current.copy(), scale=scale, level=lvl)
            )

        self._built = True
        logger.debug(
            "Built Gaussian pyramid: %d levels, scales %s",
            len(self._levels),
            [f"{l.scale:.4f}" for l in self._levels],
        )
        return self

    @property
    def levels(self) -> List[PyramidLevel]:
        """All computed pyramid levels (builds if needed)."""
        if not self._built:
            self.build()
        return self._levels

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    def get_level(self, index: int) -> PyramidLevel:
        """Return a specific pyramid level by index."""
        return self.levels[index]

    def get_level_for_scale(self, target_scale: float) -> PyramidLevel:
        """Return the pyramid level whose scale is closest to *target_scale*.

        Args:
            target_scale: Desired scale factor (e.g. 1/20 = 0.05 for 20×
                          downsampling).

        Returns:
            The ``PyramidLevel`` with the closest matching scale.
        """
        lvls = self.levels
        best = min(lvls, key=lambda l: abs(l.scale - target_scale))
        logger.debug(
            "Requested scale %.4f → using level %d (scale %.4f)",
            target_scale,
            best.level,
            best.scale,
        )
        return best


# ---------------------------------------------------------------------------
# Scale Aligner
# ---------------------------------------------------------------------------

class ScaleAligner:
    """Prepare an image pair with different GSDs for feature matching.

    Given two images at different ground-sample distances, the aligner:

    1.  Computes the integer scale ratio from the GSD values.
    2.  Down-samples the higher-res image (via the Gaussian pyramid) to
        approximately match the lower-res image's GSD.
    3.  Optionally up-samples the lower-res image by a small factor
        (e.g. 2×) to retain more structure for feature extraction.
    4.  Returns the aligned pair plus coordinate transforms for mapping
        matched keypoints back to original pixel systems.
    """

    def __init__(self, upsample_factor: float = 1.0):
        """
        Args:
            upsample_factor: Factor by which to upsample the lower-res
                             image (1.0 = no upsampling, 2.0 = 2× upsample).
        """
        self.upsample_factor = upsample_factor

    def align(
        self,
        high_res_image: np.ndarray,
        low_res_image: np.ndarray,
        scale_ratio: float,
    ) -> dict:
        """Align a high-res / low-res image pair for matching.

        Args:
            high_res_image: The finer-resolution image (e.g. OHRC).
            low_res_image:  The coarser-resolution image (e.g. TMC-2).
            scale_ratio:    ``high_res_gsd / low_res_gsd``  (e.g. 0.25/5 = 0.05
                            for OHRC/TMC-2, meaning OHRC is 20× finer).

        Returns:
            dict with keys:
                ``aligned_src``   – downsampled high-res image
                ``aligned_dst``   – (possibly upsampled) low-res image
                ``src_scale``     – scale factor applied to high-res
                ``dst_scale``     – scale factor applied to low-res
                ``src_offset``    – (row, col) crop offset in original coords
        """
        # -- Down-sample the high-res image --------------------------------
        needed_depth = compute_pyramid_depth(scale_ratio)
        pyr = GaussianPyramid(high_res_image, max_levels=needed_depth, target_scale=scale_ratio)
        pyr.build()

        target_scale = scale_ratio  # e.g. 0.05406 means 18.5× down
        best_level = pyr.get_level_for_scale(target_scale)
        aligned_src = best_level.image
        src_scale = best_level.scale

        # -- Optionally up-sample the low-res image ------------------------
        dst_scale = 1.0
        aligned_dst = low_res_image.astype(np.float32)

        if self.upsample_factor > 1.0:
            h, w = aligned_dst.shape[:2]
            new_h = int(h * self.upsample_factor)
            new_w = int(w * self.upsample_factor)
            aligned_dst = cv2.resize(
                aligned_dst, (new_w, new_h), interpolation=cv2.INTER_CUBIC
            )
            dst_scale = self.upsample_factor

        logger.debug(
            "ScaleAligner: src %s → %s (scale %.4f), "
            "dst %s → %s (scale %.2f)",
            high_res_image.shape,
            aligned_src.shape,
            src_scale,
            low_res_image.shape,
            aligned_dst.shape,
            dst_scale,
        )

        return {
            "aligned_src": aligned_src,
            "aligned_dst": aligned_dst,
            "src_scale": src_scale,
            "dst_scale": dst_scale,
            "src_offset": (0, 0),
        }

    def align_from_metadata(
        self,
        high_res_image: np.ndarray,
        low_res_image: np.ndarray,
        high_res_meta: dict,
        low_res_meta: dict,
    ) -> dict:
        """Align high-res and low-res images by dynamically reading GSDs from labels.

        Args:
            high_res_image: Finer resolution image (e.g. TMC-2 at 4.96 m/px).
            low_res_image:  Coarser resolution image (e.g. IIRS at 91.75 m/px).
            high_res_meta:  Metadata dict from parse_label() for high-res product.
            low_res_meta:   Metadata dict from parse_label() for low-res product.

        Returns:
            Dictionary produced by align() with computed scale ratio.
        """
        gsd_high = float(high_res_meta.get("pixel_resolution", 4.96))
        gsd_low = float(low_res_meta.get("pixel_resolution", 91.75))
        ratio = compute_scale_ratio(gsd_high, gsd_low)
        logger.info(
            "Aligning from metadata: High GSD=%.2fm, Low GSD=%.2fm -> Scale Ratio=%.5f (%.1f×)",
            gsd_high,
            gsd_low,
            ratio,
            1.0 / ratio,
        )
        return self.align(high_res_image, low_res_image, scale_ratio=ratio)


# ---------------------------------------------------------------------------
# Keypoint Coordinate Remapping
# ---------------------------------------------------------------------------

def map_keypoints_to_original(
    keypoints: np.ndarray,
    scale: float,
    offset: Tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Transform keypoints from a pyramid level back to original coords.

    Args:
        keypoints: (N, 2) array of (x, y) coordinates in the scaled /
                   cropped coordinate system.
        scale:     The scale factor that was applied (e.g. 0.25 means the
                   image was 4× downsampled, so coordinates are multiplied
                   by 1/0.25 = 4).
        offset:    (row_offset, col_offset) if a sub-region was cropped
                   before pyramid construction.

    Returns:
        (N, 2) float64 array in the original image's pixel coordinates.
    """
    if len(keypoints) == 0:
        return keypoints.copy()

    mapped = keypoints.astype(np.float64).copy()

    # Undo the scale: pixel (x, y) in scaled image → (x/scale, y/scale) in original
    if scale > 0 and scale != 1.0:
        mapped /= scale

    # Undo the crop offset (col_offset → x, row_offset → y)
    mapped[:, 0] += offset[1]  # x += col_offset
    mapped[:, 1] += offset[0]  # y += row_offset

    return mapped
