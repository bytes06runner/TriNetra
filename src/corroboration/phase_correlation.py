"""
src/corroboration/phase_correlation.py — Independent whole-image Fourier phase
correlation estimator (Stage N).

Structural independence contract
--------------------------------
This module is a second estimator whose agreement with the primary pipeline is
only evidence if it cannot share the primary pipeline's bugs. It therefore:
  * imports numpy only (no OpenCV, no scipy, no project modules);
  * does not use keypoints, template matching, RANSAC, similarity fitting,
    parabolic peak refinement, or phase congruency;
  * computes the cross-power spectrum with its own FFT calls, refines the peak
    by matrix-multiply upsampled DFT (Guizar-Sicairos et al., 2008) instead of
    a 3x3 parabola, and scores significance by peak-to-sidelobe ratio.
tests/test_corroboration.py enforces the import contract.

Shift convention
----------------
estimate_shift(fixed, moving) returns s = (dx, dy) such that
    moving(x) ~= fixed(x - s),
i.e. content located at p in `fixed` appears at p + s in `moving`.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np


def hann2d(shape: Tuple[int, int]) -> np.ndarray:
    """Separable 2D Hann window."""
    h, w = shape
    wy = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(h) / max(h - 1, 1))
    wx = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(w) / max(w - 1, 1))
    return np.outer(wy, wx)


def _prepare(img: np.ndarray, window: bool) -> np.ndarray:
    a = np.asarray(img, dtype=np.float64)
    a = a - a.mean()
    sd = a.std()
    if sd > 0:
        a = a / sd
    if window:
        a = a * hann2d(a.shape)
    return a


def _radial_lowpass(shape: Tuple[int, int], cutoff_cyc_per_px: float) -> np.ndarray:
    """Binary radial mask keeping |f| <= cutoff (cycles / pixel), FFT layout."""
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    return (np.hypot(fx, fy) <= cutoff_cyc_per_px).astype(np.float64)


def _upsampled_dft(data: np.ndarray, region: int, factor: int,
                   off_y: float, off_x: float) -> np.ndarray:
    """Inverse DFT of `data` evaluated on a (region x region) grid at 1/factor
    pixel spacing, starting at (off_y, off_x) in upsampled-pixel units."""
    ny, nx = data.shape
    fx = np.fft.ifftshift(np.arange(nx)) - np.floor(nx / 2.0)
    fy = np.fft.ifftshift(np.arange(ny)) - np.floor(ny / 2.0)
    kx = np.exp((2j * np.pi / (nx * factor)) *
                (np.arange(region)[:, None] - off_x) * fx[None, :])
    ky = np.exp((2j * np.pi / (ny * factor)) *
                (np.arange(region)[:, None] - off_y) * fy[None, :])
    return ky @ data @ kx.T


def _psr(surface: np.ndarray, py: int, px: int, exclude: int) -> Tuple[float, float]:
    """Peak-to-sidelobe ratio and peak / best-sidelobe ratio on a circular surface."""
    h, w = surface.shape
    rolled = np.roll(np.roll(surface, h // 2 - py, axis=0), w // 2 - px, axis=1)
    mask = np.ones_like(rolled, dtype=bool)
    mask[h // 2 - exclude:h // 2 + exclude + 1, w // 2 - exclude:w // 2 + exclude + 1] = False
    side = rolled[mask]
    peak = float(surface[py, px])
    sd = float(side.std())
    psr = (peak - float(side.mean())) / sd if sd > 0 else float("inf")
    best_side = float(side.max())
    ratio = peak / best_side if best_side > 0 else float("inf")
    return psr, ratio


def estimate_shift(
    fixed: np.ndarray,
    moving: np.ndarray,
    window: bool = True,
    lowpass_cyc_per_px: Optional[float] = None,
    upsample: int = 100,
    psr_exclude: int = 5,
) -> Dict[str, Any]:
    """Whole-image phase correlation. See module docstring for the sign convention.

    Returns dx, dy (pixels), integer peak, normalised peak height, PSR and
    peak-to-second-peak ratio.
    """
    a = _prepare(fixed, window)
    b = _prepare(moving, window)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")

    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    cross = fb * np.conj(fa)
    mag = np.abs(cross)
    cross = cross / np.maximum(mag, 1e-12 * mag.max() if mag.max() > 0 else 1e-12)
    if lowpass_cyc_per_px is not None:
        cross = cross * _radial_lowpass(a.shape, lowpass_cyc_per_px)

    surface = np.real(np.fft.ifft2(cross))
    surface = surface / max(float(np.count_nonzero(cross)), 1.0)
    py, px = np.unravel_index(int(np.argmax(surface)), surface.shape)
    h, w = surface.shape
    iy = py - h if py > h // 2 else py
    ix = px - w if px > w // 2 else px

    dy, dx = float(iy), float(ix)
    if upsample and upsample > 1:
        region = int(np.ceil(upsample * 1.5))
        centre = np.floor(region / 2.0)
        off_y = centre - dy * upsample
        off_x = centre - dx * upsample
        up = np.real(_upsampled_dft(cross, region, upsample, off_y, off_x))
        uy, ux = np.unravel_index(int(np.argmax(up)), up.shape)
        dy = dy + (uy - centre) / upsample
        dx = dx + (ux - centre) / upsample

    psr, peak_ratio = _psr(surface, py, px, psr_exclude)
    return {
        "dx": dx,
        "dy": dy,
        "peak": float(surface[py, px]),
        "psr": float(psr),
        "peak_to_second": float(peak_ratio),
        "int_peak_xy": (int(ix), int(iy)),
    }


def fourier_shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Exact circular translation by (dx, dy): content at p moves to p + (dx, dy)."""
    a = np.asarray(img, dtype=np.float64)
    fy = np.fft.fftfreq(a.shape[0])[:, None]
    fx = np.fft.fftfreq(a.shape[1])[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(a) * np.exp(-2j * np.pi * (fx * dx + fy * dy))))


def similarity_displacement(M: np.ndarray, point_xy: Tuple[float, float]) -> Tuple[float, float]:
    """Displacement M(p) - p of a 2x3 or 3x3 point map at point p. Pure arithmetic,
    used only to express the primary pipeline's transform in this module's units."""
    M = np.asarray(M, dtype=np.float64)
    x, y = float(point_xy[0]), float(point_xy[1])
    qx = M[0, 0] * x + M[0, 1] * y + M[0, 2]
    qy = M[1, 0] * x + M[1, 1] * y + M[1, 2]
    return qx - x, qy - y


def warp_linear_about_centre(img: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Resample `img` so that content at p moves to c + A (p - c) (bilinear,
    numpy only, zero outside). Used for the conditioned variant only."""
    a = np.asarray(img, dtype=np.float64)
    h, w = a.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    Ainv = np.linalg.inv(np.asarray(A, dtype=np.float64))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    sx = Ainv[0, 0] * (xx - cx) + Ainv[0, 1] * (yy - cy) + cx
    sy = Ainv[1, 0] * (xx - cx) + Ainv[1, 1] * (yy - cy) + cy
    x0 = np.floor(sx).astype(int)
    y0 = np.floor(sy).astype(int)
    fx = sx - x0
    fy = sy - y0
    out = np.zeros_like(a)
    valid = (x0 >= 0) & (y0 >= 0) & (x0 < w - 1) & (y0 < h - 1)
    x0v, y0v, fxv, fyv = x0[valid], y0[valid], fx[valid], fy[valid]
    out[valid] = (a[y0v, x0v] * (1 - fxv) * (1 - fyv) + a[y0v, x0v + 1] * fxv * (1 - fyv)
                  + a[y0v + 1, x0v] * (1 - fxv) * fyv + a[y0v + 1, x0v + 1] * fxv * fyv)
    return out
