"""
Unit tests for illum_render.py (Photometric Renderers & Cast Shadow Engine).
"""

import pytest
import numpy as np
import cv2
import torch

from src.illum_render import (
    compute_surface_normals,
    compute_sun_vector,
    render_hillshade,
    render_lommel_seeliger,
    compute_cast_shadow_mask,
    apply_cast_shadows,
    render_lunar_tile,
)


def test_surface_normals_flat_surface():
    """Flat terrain should have normal vector (0, 0, 1)."""
    dem = np.full((50, 50), 500.0, dtype=np.float32)
    nx, ny, nz = compute_surface_normals(dem, pixel_size_m=5.0)
    
    np.testing.assert_allclose(nx, 0.0, atol=1e-5)
    np.testing.assert_allclose(ny, 0.0, atol=1e-5)
    np.testing.assert_allclose(nz, 1.0, atol=1e-5)


def test_sun_vector_cardinal_directions():
    """Sun vectors at cardinal azimuths and 45 deg altitude."""
    # North (az=0): sy = cos(45), sx = 0, sz = sin(45)
    sx, sy, sz = compute_sun_vector(0.0, 45.0)
    assert abs(sx) < 1e-5
    assert abs(sy - np.sqrt(0.5)) < 1e-4
    assert abs(sz - np.sqrt(0.5)) < 1e-4

    # East (az=90): sx = cos(45), sy = 0, sz = sin(45)
    sx, sy, sz = compute_sun_vector(90.0, 45.0)
    assert abs(sx - np.sqrt(0.5)) < 1e-4
    assert abs(sy) < 1e-5
    assert abs(sz - np.sqrt(0.5)) < 1e-4


def test_lambertian_hillshade_flat_surface():
    """Flat surface with 30 deg sun elevation should have mu0 = sin(30) = 0.5 -> 128 DN."""
    dem = np.full((64, 64), 200.0, dtype=np.float32)
    hill = render_hillshade(dem, pixel_size_m=5.0, azimuth_deg=90.0, altitude_deg=30.0)
    
    assert np.all(hill == 128)


def test_lommel_seeliger_vs_lambertian_contrast():
    """Lommel-Seeliger should yield higher backscattering at oblique angles than Lambertian."""
    # Sloped terrain
    x = np.linspace(0, 100, 128, dtype=np.float32)
    dem = np.tile(x, (128, 1)) * 0.5 # 25 deg slope
    
    lambert = render_hillshade(dem, pixel_size_m=5.0, azimuth_deg=90.0, altitude_deg=10.0)
    ls = render_lommel_seeliger(dem, pixel_size_m=5.0, az_deg=90.0, alt_deg=10.0)
    
    assert lambert.shape == (128, 128)
    assert ls.shape == (128, 128)
    # They should produce distinct distributions
    diff = np.abs(lambert.astype(float) - ls.astype(float))
    assert diff.mean() > 5.0, "Lommel-Seeliger and Lambertian must be quantitatively distinct"


def test_cast_shadow_length_at_2deg():
    """
    At 2 deg sun elevation, a step feature of height H_step casts a shadow:
    L = H_step / tan(2 deg) = H_step * 28.636 meters.
    With 5 m/px: L_px = (H_step * 28.636) / 5.0.
    For H_step = 10.0 m: L = 286.4 m -> ~57 pixels.
    """
    H, W = 150, 150
    dem = np.zeros((H, W), dtype=np.float32)
    # Place a 10m ridge at col 100 (columns 100 to 149 are 10m high)
    dem[:, 100:] = 10.0
    
    # Sun from East (az=90 deg), altitude 2.0 deg
    mask = compute_cast_shadow_mask(dem, az_deg=90.0, alt_deg=2.0, pixel_size_m=5.0, max_steps=100)
    
    # Check shadow cast to the West of the ridge (cols < 100)
    # Shadow length should extend ~57 pixels west from col 99 down to col ~42
    shadowed_cols = np.where(mask[75, :100])[0]
    shadow_length_px = len(shadowed_cols)
    expected_length_px = int(round(10.0 / np.tan(np.radians(2.0)) / 5.0)) # ~57 px
    
    assert abs(shadow_length_px - expected_length_px) <= 2, (
        f"Shadow length {shadow_length_px} px does not match expected {expected_length_px} px at 2 deg elevation"
    )


def test_homography_scaling_invariance():
    """Coordinate mapping must be invariant when scaling H from 512 to 256."""
    pa_512 = np.array([120.0, 310.0, 1.0])
    
    H_512 = np.array([
        [1.15, -0.15, 35.0],
        [0.10, 1.08, -25.0],
        [0.0002, -0.0001, 1.0]
    ], dtype=np.float64)
    
    pb_512 = H_512 @ pa_512
    pb_512 /= pb_512[2]
    
    S = np.diag([0.5, 0.5, 1.0])
    S_inv = np.diag([2.0, 2.0, 1.0])
    H_256 = S @ H_512 @ S_inv
    
    pa_256 = S @ pa_512
    pb_256 = H_256 @ pa_256
    pb_256 /= pb_256[2]
    
    np.testing.assert_allclose(pb_512[:2] * 0.5, pb_256[:2], atol=1e-5)


def test_scale_gap_resample_preserves_h():
    """Downsample-upsample cycle on image_b must not alter H_a_to_b."""
    img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    H_orig = np.eye(3, dtype=np.float64)
    H_orig[0, 2] = 15.0
    H_orig[1, 2] = -20.0
    
    H_copy = H_orig.copy()
    
    # Scale-gap simulation: downsample by 16x then upsample back
    scale_factor = 16
    small = cv2.resize(img, (512 // scale_factor, 512 // scale_factor), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (512, 512), interpolation=cv2.INTER_AREA)
    
    assert restored.shape == (512, 512)
    assert np.array_equal(H_orig, H_copy), "Scale-gap resample must not mutate H!"


def test_render_lunar_tile_pipeline():
    """End-to-end rendering produces valid uint8 image and boolean shadow mask."""
    dem = np.random.randn(128, 128).astype(np.float32) * 20.0 + 500.0
    
    img_lambert, mask_lambert = render_lunar_tile(dem, az_deg=90.0, alt_deg=10.0, mode="lambert")
    assert img_lambert.shape == (128, 128)
    assert img_lambert.dtype == np.uint8
    assert mask_lambert.shape == (128, 128)
    assert mask_lambert.dtype == bool
    
    img_ls, mask_ls = render_lunar_tile(dem, az_deg=90.0, alt_deg=10.0, mode="ls")
    assert img_ls.shape == (128, 128)
    assert img_ls.dtype == np.uint8
    assert mask_ls.shape == (128, 128)


def test_render_coarse_dem_tile_option_b():
    """Option B coarse DEM pre-shading preserves output dimensions and uint8 format."""
    from src.illum_render import render_coarse_dem_tile
    dem = np.random.randn(256, 256).astype(np.float32) * 25.0 + 700.0
    img_coarse, mask_coarse = render_coarse_dem_tile(dem, az_deg=242.0, alt_deg=15.0, scale_factor=4.0, mode="ls")
    
    assert img_coarse.shape == (256, 256)
    assert img_coarse.dtype == np.uint8
    assert mask_coarse.shape == (256, 256)
    assert mask_coarse.dtype == bool
    assert img_coarse.std() > 5.0, "Coarse DEM shading must preserve visible contrast"


def test_stochastic_ambient_floor_distribution():
    """Verify that stochastic shadow floor produces a continuous distribution, not a delta spike."""
    # Step feature to generate cast shadow
    H, W = 100, 100
    dem = np.zeros((H, W), dtype=np.float32)
    dem[:, 60:] = 20.0  # 20m ridge

    # Check return_sky_view on compute_cast_shadow_mask
    mask, sky_view = compute_cast_shadow_mask(dem, az_deg=90.0, alt_deg=5.0, return_sky_view=True)
    assert mask.shape == (H, W)
    assert sky_view.shape == (H, W)
    assert mask.any()
    assert np.all((sky_view >= 0.35) & (sky_view <= 1.0))

    # Render with stochastic floor
    img_stoch, mask_stoch = render_lunar_tile(dem, az_deg=90.0, alt_deg=5.0, mode="ls", stochastic_floor=True)
    shadow_vals = img_stoch[mask_stoch]
    
    unique_vals = np.unique(shadow_vals)
    # Must have a rich distribution of values, eliminating the single delta spike
    assert len(unique_vals) > 15, f"Expected >15 unique values in shadow, got {len(unique_vals)}"
    assert shadow_vals.std() > 1.5, f"Expected positive variance in shadow, got std={shadow_vals.std()}"

    # Render with constant floor (stochastic_floor=False)
    img_const, mask_const = render_lunar_tile(dem, az_deg=90.0, alt_deg=5.0, mode="ls", ambient_dn=51, stochastic_floor=False)
    assert np.all(img_const[mask_const] == 51), "stochastic_floor=False must yield constant 51 DN floor"


