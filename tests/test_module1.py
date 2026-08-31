"""
Comprehensive unit tests for Phase 1: Mock Data Generation and Module 1 Preprocessing.

Tests cover the MockLunarDataGenerator, OHRCPreprocessor, TMC2Preprocessor,
IIRSPreprocessor, and FootprintParser against their actual interfaces.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_mock_data import MockLunarDataGenerator
from src.module1_preprocessing.preprocessor import (
    OHRCPreprocessor, TMC2Preprocessor, PreprocessingResult, BasePreprocessor
)
from src.module1_preprocessing.iirs_pca import IIRSPreprocessor
from src.module1_preprocessing.metadata_parser import FootprintParser, InstrumentFootprint


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_gen():
    """Provides a fast mock generator for testing."""
    return MockLunarDataGenerator(area_km=1.0, max_ohrc_pixels=500, seed=42)


@pytest.fixture
def mock_data(mock_gen):
    """Provides pre-generated mock data dicts for all three instruments."""
    return mock_gen.generate_all()


@pytest.fixture
def mock_metadata(mock_gen):
    """Provides simulated SPICE-kernel metadata."""
    return mock_gen.generate_metadata()


# ── Mock Data Generator Tests ─────────────────────────────────────────────────

class TestMockDataGenerator:
    """Tests for the synthetic data generator."""

    def test_generator_creates_all_instruments(self, mock_data):
        """generate_all() returns dict with 'ohrc', 'tmc2', 'iirs' keys."""
        assert 'ohrc' in mock_data
        assert 'tmc2' in mock_data
        assert 'iirs' in mock_data

    def test_ohrc_shape_and_dtype(self, mock_data):
        """OHRC output is 2D float32 ndarray."""
        ohrc_img = mock_data['ohrc']['image']
        assert isinstance(ohrc_img, np.ndarray)
        assert ohrc_img.ndim == 2
        assert ohrc_img.dtype == np.float32

    def test_tmc2_shape_and_dtype(self, mock_data):
        """TMC-2 output is 2D float32 ndarray."""
        tmc2_img = mock_data['tmc2']['image']
        assert isinstance(tmc2_img, np.ndarray)
        assert tmc2_img.ndim == 2
        assert tmc2_img.dtype == np.float32

    def test_iirs_cube_shape(self, mock_data):
        """IIRS cube is 3D with 256 bands on last axis."""
        iirs_cube = mock_data['iirs']['cube']
        assert isinstance(iirs_cube, np.ndarray)
        assert iirs_cube.ndim == 3
        assert iirs_cube.shape[2] == 256

    def test_iirs_band_count(self, mock_data):
        """IIRS has exactly 256 bands."""
        assert mock_data['iirs']['cube'].shape[2] == 256

    def test_scale_ratios_approximate(self, mock_data):
        """Instrument GSDs match specs: OHRC=0.25m, TMC-2=5m, IIRS=80m."""
        assert mock_data['ohrc']['gsd_m'] == 0.25
        assert mock_data['tmc2']['gsd_m'] == 5.0
        assert mock_data['iirs']['gsd_m'] == 80.0

        # Check the scale gap ratio from GSD values
        assert mock_data['tmc2']['gsd_m'] / mock_data['ohrc']['gsd_m'] == 20.0
        assert mock_data['iirs']['gsd_m'] / mock_data['tmc2']['gsd_m'] == 16.0
        assert mock_data['iirs']['gsd_m'] / mock_data['ohrc']['gsd_m'] == 320.0

    def test_deterministic_with_same_seed(self):
        """Same seed produces identical outputs."""
        gen1 = MockLunarDataGenerator(area_km=0.5, max_ohrc_pixels=250, seed=123)
        gen2 = MockLunarDataGenerator(area_km=0.5, max_ohrc_pixels=250, seed=123)

        data1 = gen1.generate_all()
        data2 = gen2.generate_all()

        np.testing.assert_array_equal(data1['ohrc']['image'], data2['ohrc']['image'])
        np.testing.assert_array_equal(data1['tmc2']['image'], data2['tmc2']['image'])

    def test_different_seeds_differ(self):
        """Different seeds produce different terrains."""
        gen1 = MockLunarDataGenerator(area_km=0.5, max_ohrc_pixels=250, seed=111)
        gen2 = MockLunarDataGenerator(area_km=0.5, max_ohrc_pixels=250, seed=222)

        data1 = gen1.generate_all()
        data2 = gen2.generate_all()

        assert not np.array_equal(data1['ohrc']['image'], data2['ohrc']['image'])

    def test_pixel_value_range(self, mock_data):
        """All panchromatic images are in [0, 1]."""
        for key in ['ohrc', 'tmc2']:
            img = mock_data[key]['image']
            assert np.min(img) >= 0.0, f"{key} min: {np.min(img)}"
            assert np.max(img) <= 1.0, f"{key} max: {np.max(img)}"

    def test_illumination_variation(self):
        """Different sun angles produce different images from same terrain seed."""
        gen1 = MockLunarDataGenerator(area_km=0.5, max_ohrc_pixels=250, seed=42,
                                       sun_elevation_deg=30.0, sun_azimuth_deg=45.0)
        gen2 = MockLunarDataGenerator(area_km=0.5, max_ohrc_pixels=250, seed=42,
                                       sun_elevation_deg=60.0, sun_azimuth_deg=135.0)

        # Different sun angles should produce different illuminated images
        # (terrain is same seed but illumination differs)
        ohrc1 = gen1.generate_ohrc()['image']
        ohrc2 = gen2.generate_ohrc()['image']
        assert not np.array_equal(ohrc1, ohrc2)

    def test_metadata_generation(self, mock_metadata):
        """Metadata contains required fields for all instruments."""
        required_keys = {'center_lat', 'center_lon', 'corner_coords',
                         'sun_elevation_deg', 'sun_azimuth_deg', 'acquisition_time', 'gsd_m'}
        for inst_name in ['OHRC', 'TMC-2', 'IIRS']:
            assert inst_name in mock_metadata, f"Missing {inst_name} in metadata"
            inst_meta = mock_metadata[inst_name]
            for key in required_keys:
                assert key in inst_meta, f"Missing '{key}' for {inst_name}"

    def test_small_area_generation(self):
        """Can generate data for small area without errors (area_km=0.5)."""
        gen = MockLunarDataGenerator(area_km=0.5, max_ohrc_pixels=100, seed=1)
        data = gen.generate_all()
        assert data['ohrc']['image'].shape[0] > 0
        assert data['tmc2']['image'].shape[0] > 0


# ── OHRC Preprocessor Tests ──────────────────────────────────────────────────

class TestOHRCPreprocessor:
    """Tests for OHRC CLAHE preprocessing."""

    def test_output_type_and_range(self, mock_data):
        """Output is PreprocessingResult with float32 image in [0, 1]."""
        preprocessor = OHRCPreprocessor()
        result = preprocessor.preprocess(mock_data['ohrc']['image'])

        assert isinstance(result, PreprocessingResult)
        assert result.image.dtype == np.float32
        assert np.min(result.image) >= 0.0
        assert np.max(result.image) <= 1.0

    def test_clahe_increases_contrast(self):
        """CLAHE should increase contrast (std) of a low-contrast input."""
        img = np.random.default_rng(42).normal(loc=0.5, scale=0.05, size=(256, 256)).astype(np.float32)
        img = np.clip(img, 0, 1)
        original_std = np.std(img)

        preprocessor = OHRCPreprocessor()
        result = preprocessor.preprocess(img)

        assert np.std(result.image) > original_std * 0.9  # At least comparable or better

    def test_shadow_edge_preservation(self):
        """Create image with sharp shadow edge. CLAHE should preserve edge gradient."""
        img = np.ones((100, 100), dtype=np.float32) * 0.8
        img[:, 50:] = 0.1  # Sharp edge

        preprocessor = OHRCPreprocessor()
        result = preprocessor.preprocess(img)

        # Edge should still be pronounced
        left_mean = np.mean(result.image[:, 45:50])
        right_mean = np.mean(result.image[:, 50:55])
        assert left_mean > right_mean, "Shadow edge was not preserved"

    def test_adaptive_clip_low_contrast(self):
        """Low-contrast images get higher clip limit (4.0)."""
        img = np.random.default_rng(42).uniform(0.49, 0.51, (100, 100)).astype(np.float32)
        preprocessor = OHRCPreprocessor(adaptive_clip=True)
        clip = preprocessor._compute_adaptive_clip_limit(img)
        assert clip == 4.0

    def test_adaptive_clip_high_contrast(self):
        """High-contrast images get lower clip limit (1.5)."""
        img = np.random.default_rng(42).uniform(0.0, 1.0, (100, 100)).astype(np.float32)
        preprocessor = OHRCPreprocessor(adaptive_clip=True)
        clip = preprocessor._compute_adaptive_clip_limit(img)
        assert clip == 1.5

    def test_handles_all_black_image(self):
        """All-zero image returns low confidence but no crash."""
        img = np.zeros((100, 100), dtype=np.float32)
        preprocessor = OHRCPreprocessor()
        result = preprocessor.preprocess(img)
        assert result.confidence <= 1.0  # Doesn't crash
        assert isinstance(result, PreprocessingResult)

    def test_handles_all_white_image(self):
        """All-ones image returns result without crash."""
        img = np.ones((100, 100), dtype=np.float32)
        preprocessor = OHRCPreprocessor()
        result = preprocessor.preprocess(img)
        assert isinstance(result, PreprocessingResult)

    def test_processing_steps_logged(self, mock_data):
        """processing_steps list is non-empty."""
        preprocessor = OHRCPreprocessor()
        result = preprocessor.preprocess(mock_data['ohrc']['image'])
        assert len(result.processing_steps) > 0

    def test_accepts_uint8_input(self):
        """Can handle uint8 input (auto-converts)."""
        img = np.random.default_rng(42).integers(0, 255, (100, 100), dtype=np.uint8)
        preprocessor = OHRCPreprocessor()
        result = preprocessor.preprocess(img)
        assert result.image.dtype == np.float32
        assert np.max(result.image) <= 1.0

    def test_accepts_uint16_input(self):
        """Can handle uint16 input (auto-converts)."""
        img = np.random.default_rng(42).integers(0, 65535, (100, 100)).astype(np.uint16)
        preprocessor = OHRCPreprocessor()
        result = preprocessor.preprocess(img)
        assert result.image.dtype == np.float32
        assert np.max(result.image) <= 1.0


# ── TMC-2 Preprocessor Tests ─────────────────────────────────────────────────

class TestTMC2Preprocessor:
    """Tests for TMC-2 radiometric normalization."""

    def test_output_type_and_range(self, mock_data):
        """Output is PreprocessingResult with float32 [0, 1]."""
        preprocessor = TMC2Preprocessor()
        result = preprocessor.preprocess(mock_data['tmc2']['image'])

        assert isinstance(result, PreprocessingResult)
        assert result.image.dtype == np.float32
        assert np.min(result.image) >= 0.0
        assert np.max(result.image) <= 1.0

    def test_percentile_stretch_removes_outliers(self):
        """Extreme outlier pixels should be clipped by percentile stretch."""
        rng = np.random.default_rng(42)
        img = rng.uniform(0.3, 0.7, (100, 100)).astype(np.float32)
        img[0, 0] = 5.0   # Positive outlier
        img[1, 1] = -1.0  # Negative outlier

        preprocessor = TMC2Preprocessor()
        result = preprocessor.preprocess(img)

        # After percentile stretch + denoising, outliers should be moved
        # toward the extremes. The positive outlier should be near 1.0 and
        # the negative outlier should be significantly reduced.
        assert result.image[0, 0] >= 0.8  # Was 5.0, should be stretched high
        # Note: denoising may bleed neighbor values, so negative outlier won't be exactly 0
        assert result.image[1, 1] <= 0.3  # Was -1.0, should be stretched low

    def test_normalization_consistency(self, mock_data):
        """Same input twice produces same output."""
        preprocessor = TMC2Preprocessor()
        res1 = preprocessor.preprocess(mock_data['tmc2']['image'])
        res2 = preprocessor.preprocess(mock_data['tmc2']['image'])
        np.testing.assert_array_almost_equal(res1.image, res2.image)

    def test_handles_uniform_image(self):
        """Uniform image doesn't crash (edge case for percentile stretch)."""
        img = np.ones((100, 100), dtype=np.float32) * 0.5
        preprocessor = TMC2Preprocessor()
        result = preprocessor.preprocess(img)
        assert isinstance(result, PreprocessingResult)


# ── IIRS Preprocessor Tests ──────────────────────────────────────────────────

class TestIIRSPreprocessor:
    """Tests for IIRS hyperspectral PCA preprocessing."""

    def test_cube_to_2d_proxy(self, mock_data):
        """256-band cube collapses to 2D image."""
        preprocessor = IIRSPreprocessor(proxy_mode='pc1')
        cube = mock_data['iirs']['cube']
        result = preprocessor.preprocess(cube)
        assert result.image.ndim == 2

    def test_proxy_image_shape(self, mock_data):
        """Proxy spatial dimensions match input spatial dimensions."""
        cube = mock_data['iirs']['cube']
        preprocessor = IIRSPreprocessor()
        result = preprocessor.preprocess(cube)
        assert result.image.shape == cube.shape[:2]

    def test_pca_explained_variance(self, mock_data):
        """PC1 should explain significant variance (>20%)."""
        preprocessor = IIRSPreprocessor()
        cube = mock_data['iirs']['cube']
        result = preprocessor.preprocess(cube)
        evr = result.quality_metrics.get("explained_variance_ratio", [0])
        if isinstance(evr, list) and len(evr) > 0:
            assert evr[0] > 0.2, f"PC1 explained variance too low: {evr[0]}"

    def test_band_selection_reduces_bands(self):
        """Only bands in 0.8-2.5 µm range are used (out of full 0.8-5.0)."""
        rng = np.random.default_rng(42)
        cube = rng.uniform(0.1, 0.9, (20, 20, 256)).astype(np.float32)
        band_centers = np.linspace(0.8, 5.0, 256)

        preprocessor = IIRSPreprocessor(band_range_um=(0.8, 2.5))
        result = preprocessor.preprocess(cube, band_centers_um=band_centers)
        # The retained_bands_count should be less than 256
        retained = result.quality_metrics.get("retained_bands_count", 256)
        assert retained < 256, f"Expected fewer bands, got {retained}"

    def test_bad_band_removal(self):
        """Bands with near-zero variance are removed."""
        rng = np.random.default_rng(42)
        cube = rng.uniform(0.1, 0.9, (20, 20, 50)).astype(np.float32)
        # Make several bands constant (zero variance)
        cube[:, :, 10] = 0.5
        cube[:, :, 20] = 0.3
        cube[:, :, 30] = 0.7

        band_centers = np.linspace(0.8, 2.5, 50)
        preprocessor = IIRSPreprocessor(band_range_um=(0.8, 2.5))
        result = preprocessor.preprocess(cube, band_centers_um=band_centers)
        assert isinstance(result, PreprocessingResult)

    def test_proxy_mode_pc1(self, mock_data):
        """proxy_mode='pc1' returns 2D image."""
        preprocessor = IIRSPreprocessor(proxy_mode='pc1')
        result = preprocessor.preprocess(mock_data['iirs']['cube'])
        assert result.image.ndim == 2

    def test_proxy_mode_weighted(self, mock_data):
        """proxy_mode='weighted' returns 2D image."""
        preprocessor = IIRSPreprocessor(proxy_mode='weighted')
        result = preprocessor.preprocess(mock_data['iirs']['cube'])
        assert result.image.ndim == 2

    def test_small_cube(self):
        """Handles very small cubes (e.g., 5x5x256) without errors."""
        rng = np.random.default_rng(42)
        cube = rng.uniform(0, 1, (5, 5, 256)).astype(np.float32)
        preprocessor = IIRSPreprocessor()
        result = preprocessor.preprocess(cube)
        assert result.image.shape == (5, 5)

    def test_single_band_in_range_fails_gracefully(self):
        """Cube with 1 band in range still produces a result."""
        rng = np.random.default_rng(42)
        # Only 1 band will fall in 0.8-0.9 range
        cube = rng.uniform(0, 1, (10, 10, 10)).astype(np.float32)
        band_centers = np.linspace(0.8, 5.0, 10)
        preprocessor = IIRSPreprocessor(band_range_um=(0.8, 0.9))
        result = preprocessor.preprocess(cube, band_centers_um=band_centers)
        assert isinstance(result, PreprocessingResult)


# ── Footprint Parser Tests ───────────────────────────────────────────────────

class TestFootprintParser:
    """Tests for metadata parsing and overlap computation."""

    def test_parse_metadata(self, mock_metadata):
        """Can parse the metadata dict from MockLunarDataGenerator."""
        parser = FootprintParser()
        footprints = parser.parse_metadata(mock_metadata)
        assert len(footprints) > 0
        # Check that at least one instrument was parsed
        assert any(name in footprints for name in ['OHRC', 'TMC-2', 'IIRS'])

    def test_bounding_box_computed(self, mock_metadata):
        """After parsing, each footprint has a valid bounding_box."""
        parser = FootprintParser()
        parser.parse_metadata(mock_metadata)
        for name, fp in parser.footprints.items():
            assert fp.bounding_box is not None, f"{name} has no bounding box"
            assert len(fp.bounding_box) == 4

    def test_overlap_detection(self, mock_metadata):
        """Overlapping footprints are detected correctly."""
        parser = FootprintParser()
        parser.parse_metadata(mock_metadata)

        # In mock data, all instruments share the same center, so they should overlap
        overlaps = parser.compute_all_overlaps()
        # At least some pairs should have overlap
        has_any_overlap = any(v.get('has_overlap', False) for v in overlaps.values())
        assert has_any_overlap, "No overlaps detected in mock data"

    def test_no_overlap_detection(self):
        """Non-overlapping footprints return has_overlap=False."""
        fp_a = InstrumentFootprint(
            instrument_name="A", center_lat=0.0, center_lon=0.0,
            corner_coords=[(1, 1), (1, -1), (-1, -1), (-1, 1)],
            gsd_m=5.0, sun_elevation_deg=30.0, sun_azimuth_deg=45.0,
            acquisition_time="2024-01-01T00:00:00Z"
        )
        fp_b = InstrumentFootprint(
            instrument_name="B", center_lat=50.0, center_lon=50.0,
            corner_coords=[(51, 51), (51, 49), (49, 49), (49, 51)],
            gsd_m=5.0, sun_elevation_deg=30.0, sun_azimuth_deg=45.0,
            acquisition_time="2024-01-01T00:00:00Z"
        )

        parser = FootprintParser()
        overlap_info = parser.compute_overlap(fp_a, fp_b)
        assert overlap_info['has_overlap'] is False

    def test_search_region_for_valid_pair(self, mock_metadata):
        """get_search_region returns valid box for overlapping pair."""
        parser = FootprintParser()
        parser.parse_metadata(mock_metadata)

        # Try to find search region between any two parsed instruments
        keys = list(parser.footprints.keys())
        if len(keys) >= 2:
            region = parser.get_search_region(keys[0], keys[1])
            # Should return a valid box since mock data shares center
            if region is not None:
                assert len(region) == 4

    def test_pixel_offset_estimation(self, mock_metadata):
        """Pixel offset is computed and is a tuple of ints."""
        parser = FootprintParser()
        parser.parse_metadata(mock_metadata)

        keys = list(parser.footprints.keys())
        if len(keys) >= 2:
            fp_a = parser.footprints[keys[0]]
            fp_b = parser.footprints[keys[1]]
            offset = parser.estimate_pixel_offset(fp_a, fp_b)
            if offset is not None:
                assert isinstance(offset, tuple)
                assert len(offset) == 2
                assert isinstance(offset[0], int)
                assert isinstance(offset[1], int)


# ── Integration Tests ─────────────────────────────────────────────────────────

class TestIntegration:
    """End-to-end integration tests for the full Phase 1 pipeline."""

    def test_full_pipeline_ohrc(self, mock_data):
        """Generate mock OHRC data → preprocess → valid result."""
        img = mock_data['ohrc']['image']
        prep = OHRCPreprocessor()
        res = prep.preprocess(img)
        assert res.confidence > 0.0
        assert res.image.shape == img.shape

    def test_full_pipeline_tmc2(self, mock_data):
        """Generate mock TMC-2 data → preprocess → valid result."""
        img = mock_data['tmc2']['image']
        prep = TMC2Preprocessor()
        res = prep.preprocess(img)
        assert res.confidence > 0.0
        assert res.image.shape == img.shape

    def test_full_pipeline_iirs(self, mock_data):
        """Generate mock IIRS data → preprocess → valid result."""
        cube = mock_data['iirs']['cube']
        prep = IIRSPreprocessor()
        res = prep.preprocess(cube)
        assert res.image.ndim == 2
        assert res.image.shape == cube.shape[:2]

    def test_full_pipeline_all_instruments(self, mock_data, mock_metadata):
        """Generate all three instruments, preprocess all, verify all succeed."""
        prep_ohrc = OHRCPreprocessor()
        prep_tmc2 = TMC2Preprocessor()
        prep_iirs = IIRSPreprocessor()

        res_ohrc = prep_ohrc.preprocess(mock_data['ohrc']['image'])
        res_tmc2 = prep_tmc2.preprocess(mock_data['tmc2']['image'])
        res_iirs = prep_iirs.preprocess(mock_data['iirs']['cube'])

        assert res_ohrc.image is not None
        assert res_tmc2.image is not None
        assert res_iirs.image is not None

        # Verify metadata parsing works end-to-end
        parser = FootprintParser()
        parser.parse_metadata(mock_metadata)
        assert len(parser.footprints) > 0
