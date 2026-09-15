"""
phase_congruency.py — Illumination & Contrast-Invariant Structural Feature Extraction.

Implements Phase Congruency via a multi-scale, multi-orientation 2D Log-Gabor filter bank.
Based on the Kovesi (1999, 2000) formulation and the RIFT/RIFT2 cross-modal registration
framework (Li et al., IEEE TGRS 2020).

Phase congruency identifies feature points (edges, ridges, crater rims) where local Fourier
frequency components are in phase, independent of overall image intensity, contrast inversion,
or monotonic sensor gain disparities. This provides an ideal representation for matching
visible-spectrum reflectance (TMC-2) against shortwave-infrared radiance (IIRS).
"""

from typing import Tuple, Optional
import numpy as np
import cv2


def compute_phase_congruency(
    img: np.ndarray,
    nscale: int = 4,
    norient: int = 6,
    min_wavelength: float = 3.0,
    mult: float = 2.0,
    sigma_on_f: float = 0.55,
    d_theta_on_sigma: float = 1.2,
    k: float = 2.0,
    epsilon: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute phase congruency and maximum moment map for a grayscale image.

    Args:
        img: 2D numpy array (grayscale uint8 or float32).
        nscale: Number of wavelet scales (default: 4).
        norient: Number of filter orientations (default: 6).
        min_wavelength: Wavelength of smallest scale filter in pixels (default: 3.0).
        mult: Scaling factor between successive filters (default: 2.0).
        sigma_on_f: Ratio of log-Gabor filter std dev to center frequency (default: 0.55).
        d_theta_on_sigma: Ratio of angular interval to angular Gaussian std dev (default: 1.2).
        k: Multiplier of noise estimate for thresholding (default: 2.0).
        epsilon: Small constant to avoid division by zero.

    Returns:
        pc_uint8: Maximum moment of phase congruency normalized to [0, 255] uint8.
        pc_float: Raw phase congruency energy map (float32, [0, 1]).
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[2] == 3 else img[:, :, 0]

    img_f = img.astype(np.float32)
    rows, cols = img_f.shape

    # Pre-compute frequency grids
    # Standard 2D Fourier coordinate meshes normalized to [-0.5, 0.5]
    u = np.fft.fftfreq(cols).astype(np.float32)
    v = np.fft.fftfreq(rows).astype(np.float32)
    u_grid, v_grid = np.meshgrid(u, v)

    # Normalized radius grid (radius from origin in frequency domain)
    radius = np.sqrt(u_grid**2 + v_grid**2)
    radius[0, 0] = 1.0  # Prevent log(0) at DC component

    # Polar angle grid [-pi, pi]
    theta = np.arctan2(-v_grid, u_grid)
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    # Angular filter bandwidth
    theta_sigma = (np.pi / norient) / d_theta_on_sigma

    # Construct log-Gabor radial filter components
    log_gabor_radial = []
    for s in range(nscale):
        wavelength = min_wavelength * (mult**s)
        fo = 1.0 / wavelength  # Center frequency
        # Log-Gabor radial transfer function
        r_filt = np.exp(-((np.log(radius / fo)) ** 2) / (2.0 * (np.log(sigma_on_f)) ** 2))
        r_filt[0, 0] = 0.0  # Zero DC component
        log_gabor_radial.append(r_filt)

    # Compute 2D FFT of input image
    image_fft = np.fft.fft2(img_f)

    # Initialize accumulators for moment analysis
    total_energy = np.zeros((rows, cols), dtype=np.float32)
    total_amplitude = np.zeros((rows, cols), dtype=np.float32)

    # Moments for orientation analysis
    cov_x2 = np.zeros((rows, cols), dtype=np.float32)
    cov_y2 = np.zeros((rows, cols), dtype=np.float32)
    cov_xy = np.zeros((rows, cols), dtype=np.float32)

    for o in range(norient):
        angle = o * np.pi / norient
        # Angular filter: difference angle wrapped to [-pi/2, pi/2]
        ds = sintheta * np.cos(angle) - costheta * np.sin(angle)
        dc = costheta * np.cos(angle) + sintheta * np.sin(angle)
        dtheta = np.abs(np.arctan2(ds, dc))
        spread = np.exp(-(dtheta**2) / (2.0 * theta_sigma**2))

        # Accumulators across scales for this orientation
        sum_e = np.zeros((rows, cols), dtype=np.float32)  # Even (real) response
        sum_o = np.zeros((rows, cols), dtype=np.float32)  # Odd (imaginary) response
        sum_an = np.zeros((rows, cols), dtype=np.float32) # Amplitude sum

        for s in range(nscale):
            filt = log_gabor_radial[s] * spread
            # Filter image in frequency domain
            response = np.fft.ifft2(image_fft * filt)
            e = np.real(response).astype(np.float32)
            o_resp = np.imag(response).astype(np.float32)
            an = np.sqrt(e**2 + o_resp**2)

            sum_e += e
            sum_o += o_resp
            sum_an += an

        # Local energy at orientation o
        energy_o = np.sqrt(sum_e**2 + sum_o**2)

        # Noise threshold estimation: Rayleigh distribution of response amplitude
        # Noise mean estimated from smallest scale response median
        med_val = np.median(sum_an)
        tau = med_val / np.sqrt(np.log(4.0)) if med_val > 0 else 0.0
        t_noise = tau * np.sqrt(np.pi / 2.0) * k

        # Apply soft noise thresholding
        energy_thresh = np.maximum(energy_o - t_noise, 0.0)

        # Directional moment projections
        cos_ang = np.cos(angle)
        sin_ang = np.sin(angle)
        cov_x2 += (energy_thresh * cos_ang) ** 2
        cov_y2 += (energy_thresh * sin_ang) ** 2
        cov_xy += (energy_thresh * cos_ang) * (energy_thresh * sin_ang)

        total_energy += energy_thresh
        total_amplitude += sum_an

    # Phase congruency: normalized energy
    pc = total_energy / (total_amplitude + epsilon)
    pc = np.clip(pc, 0.0, 1.0)

    # Maximum moment M_max of phase congruency (principal curvature / corner energy)
    # Trace and determinant of the covariance tensor
    cov_diff = cov_x2 - cov_y2
    cov_rad = np.sqrt(cov_diff**2 + 4.0 * cov_xy**2)
    max_moment = 0.5 * (cov_x2 + cov_y2 + cov_rad)

    # Normalize max_moment to [0, 255] uint8 using robust percentile stretching
    p2, p98 = np.percentile(max_moment, (2.0, 98.0))
    if p98 - p2 > 1e-6:
        pc_uint8 = np.clip((max_moment - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
    else:
        max_v = float(np.max(max_moment)) if np.max(max_moment) > 0 else 1.0
        pc_uint8 = np.clip(max_moment / max_v * 255.0, 0, 255).astype(np.uint8)

    return pc_uint8, pc.astype(np.float32)


def preprocess_pair_phase_congruency(
    img1: np.ndarray,
    img2: np.ndarray,
    nscale: int = 4,
    norient: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute phase congruency for a pair of cross-modal images.

    Args:
        img1: Source image (e.g. TMC-2 crop).
        img2: Target image (e.g. IIRS proxy).

    Returns:
        pc1_u8, pc2_u8: Normalized phase congruency uint8 maps.
    """
    pc1_u8, _ = compute_phase_congruency(img1, nscale=nscale, norient=norient)
    pc2_u8, _ = compute_phase_congruency(img2, nscale=nscale, norient=norient)
    return pc1_u8, pc2_u8
