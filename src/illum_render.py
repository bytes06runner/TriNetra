"""
illum_render.py — Physically Grounded Lunar Illumination Renderers & Cast Shadow Engine.

Implements:
1. Lambertian Hillshading (Horn/GDAL equivalent and GDAL CLI wrapper)
2. Lommel-Seeliger Regolith Radiance Model (non-Lambertian lunar backscattering)
3. GPU/MPS Vectorized 200-Step Ray-March Cast Shadow Masking
"""

import os
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Union, Tuple


def get_default_device() -> torch.device:
    """Return Apple Silicon MPS if available, otherwise CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_surface_normals(dem_array: np.ndarray, pixel_size_m: float = 5.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-pixel unit surface normal vectors (nx, ny, nz) in (East, North, Up) coordinates.
    
    np.gradient on row/col:
      row index increases downward (South, -Y)
      col index increases rightward (East, +X)
    Therefore:
      dZ/dx = dZ/dcol
      dZ/dy = -dZ/drow
      n_unnormalized = (-dZ/dx, -dZ/dy, 1) = (-dZ/dcol, dZ/drow, 1)
    """
    dZ_drow, dZ_dcol = np.gradient(dem_array.astype(np.float32), pixel_size_m)
    dz_dx = dZ_dcol
    dz_dy = -dZ_drow

    nx_un = -dz_dx
    ny_un = -dz_dy
    nz_un = np.ones_like(dem_array, dtype=np.float32)

    norm = np.sqrt(nx_un**2 + ny_un**2 + nz_un**2)
    norm[norm < 1e-8] = 1e-8

    nx = nx_un / norm
    ny = ny_un / norm
    nz = nz_un / norm
    return nx, ny, nz


def compute_sun_vector(az_deg: float, alt_deg: float) -> Tuple[float, float, float]:
    """
    Compute unit vector pointing TOWARD the sun in (East, North, Up) coordinates.
    
    az_deg: Sun azimuth in degrees clockwise from North (0° = North, 90° = East, 180° = South, 270° = West).
    alt_deg: Sun altitude / elevation above the horizon in degrees.
    """
    phi = np.radians(az_deg)
    theta = np.radians(alt_deg)

    sx = np.sin(phi) * np.cos(theta)  # East
    sy = np.cos(phi) * np.cos(theta)  # North
    sz = np.sin(theta)                 # Up
    return float(sx), float(sy), float(sz)


def render_hillshade(
    dem_source: Union[np.ndarray, str],
    pixel_size_m: float = 5.0,
    azimuth_deg: float = 45.0,
    altitude_deg: float = 10.0,
    out_path: Optional[str] = None,
) -> np.ndarray:
    """
    Render Lambertian hillshade.
    
    If out_path is specified and dem_source is a file path, wraps:
      gdaldem hillshade -az {az} -alt {alt} -z 1.0 -compute_edges {dem_source} {out_path}
    Otherwise, computes in-memory Lambertian hillshade via analytical surface normals.
    """
    if out_path is not None and isinstance(dem_source, str) and os.path.exists(dem_source):
        cmd = [
            "gdaldem", "hillshade",
            dem_source, out_path,
            "-az", str(azimuth_deg),
            "-alt", str(altitude_deg),
            "-z", "1.0",
            "-compute_edges",
            "-q"
        ]
        subprocess.run(cmd, check=True)
        from osgeo import gdal
        ds = gdal.Open(out_path)
        arr = ds.ReadAsArray()
        ds = None
        return arr

    if isinstance(dem_source, str):
        from osgeo import gdal
        ds = gdal.Open(dem_source)
        dem_array = ds.ReadAsArray().astype(np.float32)
        ds = None
    else:
        dem_array = dem_source.astype(np.float32)

    nx, ny, nz = compute_surface_normals(dem_array, pixel_size_m)
    sx, sy, sz = compute_sun_vector(azimuth_deg, altitude_deg)

    # Cosine of solar incidence angle mu0 = n . s
    mu0 = nx * sx + ny * sy + nz * sz
    mu0_clipped = np.clip(mu0, 0.0, 1.0)

    # Scale to uint8 [0, 255]
    hillshade = (mu0_clipped * 255.0).round().astype(np.uint8)
    return hillshade


def render_lommel_seeliger(
    dem_array: np.ndarray,
    pixel_size_m: float = 5.0,
    az_deg: float = 45.0,
    alt_deg: float = 10.0,
    albedo: float = 1.0,
) -> np.ndarray:
    """
    Render Lunar Lommel-Seeliger regolith scattering.
    
    Lommel-Seeliger law for lunar particulate media:
      r = albedo * mu0 / (4 * (mu0 + mu))
    where:
      mu0 = cos(incidence) = n . s
      mu  = cos(emission) = 1.0 (assuming nadir-viewing geometry)
    Self-shadowed facets (mu0 <= 0) receive r = 0.
    Normalized such that normal incidence (mu0 = 1.0) maps to 255.
    """
    dem_float = dem_array.astype(np.float32)
    nx, ny, nz = compute_surface_normals(dem_float, pixel_size_m)
    sx, sy, sz = compute_sun_vector(az_deg, alt_deg)

    mu0 = nx * sx + ny * sy + nz * sz

    # Non-Lambertian Lommel-Seeliger response:
    # At normal incidence (mu0=1, mu=1), r = albedo / 8.
    # We map r / (albedo/8) to 255: 255 * (2 * mu0 / (mu0 + 1))
    valid = mu0 > 0.0
    r_norm = np.zeros_like(dem_float, dtype=np.float32)
    r_norm[valid] = (2.0 * mu0[valid]) / (mu0[valid] + 1.0)

    ls_img = (np.clip(r_norm, 0.0, 1.0) * 255.0 * albedo).round().astype(np.uint8)
    return ls_img


def compute_cast_shadow_mask(
    dem_array: np.ndarray,
    az_deg: float,
    alt_deg: float,
    pixel_size_m: float = 5.0,
    max_steps: int = 200,
    return_sky_view: bool = False,
    device: Optional[torch.device] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute cast shadow mask via stepped ray-marching along the horizontal projection
    of the sun vector toward the sun position.
    
    For each pixel (r, c) at elevation z(r, c):
      step k moves distance d_k = k * pixel_size_m toward azimuth az_deg.
      If any terrain height along the ray rises above:
        z_ray = z(r, c) + d_k * tan(alt_deg)
      then the pixel is shadowed.
    
    If return_sky_view=True, also returns the directional sky-view factor S_v in [0.35, 1.0],
    derived from the maximum horizon obstacle elevation angle:
      theta_obs = arctan(max_k (z_k - z_0) / d_k)
      S_v = clip(1.0 - theta_obs / 45.0, 0.35, 1.0)
    
    Vectorized using PyTorch grid_sample on MPS/CUDA/CPU.
    """
    if device is None:
        device = get_default_device()

    dem_f = dem_array.astype(np.float32)
    H, W = dem_f.shape

    rad_az = az_deg * np.pi / 180.0
    rad_alt = alt_deg * np.pi / 180.0

    # Marching TOWARD the sun:
    # Col step is toward East (+X) = sin(az)
    # Row step is toward North (+Y, decreasing row) = -cos(az)
    dc = float(np.sin(rad_az))
    dr = float(-np.cos(rad_az))
    tan_alt = float(np.tan(rad_alt))

    # Normalized step offsets for grid_sample coordinate frame [-1, 1]
    norm_dc = dc * (2.0 / (W - 1))
    norm_dr = dr * (2.0 / (H - 1))

    dem_t = torch.from_numpy(dem_f).to(device)
    min_elev = dem_t.min().item() - 1000.0
    # Shift DEM so all valid values are positive; grid_sample zeros padding maps below min_elev
    dem_shifted = (dem_t - min_elev)[None, None]  # (1, 1, H, W)

    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij"
    )
    base_grid = torch.stack([x_coords, y_coords], dim=-1)[None]  # (1, H, W, 2)
    step_offset = torch.tensor([norm_dc, norm_dr], device=device, dtype=torch.float32)[None, None, None, :]

    max_tan_obs = torch.full((H, W), -1e9, device=device, dtype=torch.float32)

    for k in range(1, max_steps + 1):
        step_grid = base_grid + k * step_offset
        sampled = F.grid_sample(dem_shifted, step_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        sampled_elev = sampled[0, 0] + min_elev

        dist_m = k * pixel_size_m
        diff = sampled_elev - dem_t
        tan_k = diff / dist_m
        max_tan_obs = torch.maximum(max_tan_obs, tan_k)

    shadow_mask = (max_tan_obs > tan_alt).cpu().numpy()

    if return_sky_view:
        obs_elev_deg = np.rad2deg(np.arctan(np.clip(max_tan_obs.cpu().numpy(), a_min=0.0, a_max=10.0)))
        # Sky view factor diminishes as blocking obstacle rises above horizon
        sky_view = np.clip(1.0 - (obs_elev_deg / 45.0), 0.35, 1.0).astype(np.float32)
        return shadow_mask, sky_view

    return shadow_mask


def apply_cast_shadows(
    rendered_img: np.ndarray,
    shadow_mask: np.ndarray,
    ambient_dn: Union[int, float] = 51,
    sky_view: Optional[np.ndarray] = None,
    stochastic: bool = True,
    noise_std: float = 3.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply cast shadows to a rendered image.
    
    Shadowed pixels receive a floor value representing scattered secondary illumination
    from surrounding illuminated lunar topography (default 51 DN = 20% floor, matching
    empirical Chandrayaan-2 OHRC polar flight statistics).
    
    If stochastic=True:
      - Scales ambient floor by local sky-view factor S_v from horizon ray-marching
      - Adds subtle regolith micro-texture / sensor shot noise ~ N(0, noise_std)
      This eliminates discrete delta-spikes in the shadow histogram.
    """
    out = rendered_img.copy()
    if not np.any(shadow_mask):
        return out

    if stochastic:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, noise_std, size=rendered_img.shape).astype(np.float32)
        if sky_view is not None:
            ambient_dist = float(ambient_dn) * sky_view + noise
        else:
            ambient_dist = float(ambient_dn) + noise
        shadow_vals = np.clip(np.round(ambient_dist), 10.0, 85.0).astype(np.uint8)
        out[shadow_mask] = shadow_vals[shadow_mask]
    else:
        out[shadow_mask] = np.uint8(np.clip(ambient_dn, 0, 255))

    return out


def render_lunar_tile(
    dem_array: np.ndarray,
    az_deg: float,
    alt_deg: float,
    mode: str = "lambert",
    pixel_size_m: float = 5.0,
    apply_shadows: bool = True,
    max_steps: int = 200,
    ambient_dn: int = 51,
    stochastic_floor: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render a lunar DEM tile with specified illumination and reflectance mode.
    
    Returns:
      rendered_image: uint8 (H, W)
      shadow_mask: bool (H, W) indicating cast shadows
    """
    if mode.lower() in ("lambert", "lambertian", "hillshade"):
        base_img = render_hillshade(dem_array, pixel_size_m, az_deg, alt_deg)
    elif mode.lower() in ("ls", "lommel_seeliger", "lommel-seeliger"):
        base_img = render_lommel_seeliger(dem_array, pixel_size_m, az_deg, alt_deg)
    else:
        raise ValueError(f"Unknown render mode: '{mode}'. Choose 'lambert' or 'ls'.")

    if apply_shadows:
        shadow_mask, sky_view = compute_cast_shadow_mask(
            dem_array, az_deg, alt_deg, pixel_size_m=pixel_size_m, max_steps=max_steps,
            return_sky_view=True, device=device
        )
        final_img = apply_cast_shadows(
            base_img, shadow_mask, ambient_dn=ambient_dn, sky_view=sky_view,
            stochastic=stochastic_floor
        )
    else:
        shadow_mask = np.zeros(dem_array.shape, dtype=bool)
        final_img = base_img

    return final_img, shadow_mask


def render_coarse_dem_tile(
    dem_array: np.ndarray,
    az_deg: float,
    alt_deg: float,
    scale_factor: float,
    mode: str = "ls",
    base_pixel_size_m: float = 5.0,
    ambient_dn: int = 51,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Option B Scale-Gap Simulation:
    Render image directly from the DEM at a coarser effective GSD by averaging DEM cells
    before shading, then upsampling back to the native raster grid.
    
    This simulates a coarser sensor (e.g. TMC-2 @ 5-10m vs OHRC @ 0.26m) more faithfully
    because surface normals and shading are computed at the sensor's actual scale.
    """
    import cv2
    H, W = dem_array.shape
    coarse_w = max(64, int(round(W / scale_factor)))
    coarse_h = max(64, int(round(H / scale_factor)))
    
    dem_coarse = cv2.resize(dem_array, (coarse_w, coarse_h), interpolation=cv2.INTER_AREA)
    coarse_pixel_size = float(base_pixel_size_m * (W / coarse_w))
    
    img_coarse, mask_coarse = render_lunar_tile(
        dem_coarse, az_deg, alt_deg, mode=mode,
        pixel_size_m=coarse_pixel_size, apply_shadows=True,
        ambient_dn=ambient_dn, device=device
    )
    img_upsampled = cv2.resize(img_coarse, (W, H), interpolation=cv2.INTER_CUBIC)
    mask_upsampled = cv2.resize(mask_coarse.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    return img_upsampled, mask_upsampled

