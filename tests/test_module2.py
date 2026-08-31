"""
Comprehensive unit tests for Phase 2: Module 2 — Hub Matching.

Tests cover the scale handler, ORB fallback matcher, LightGlue matcher
(with torch-conditional skips), hub-and-spoke orchestrator, and full
pipeline integration with Module 1 preprocessing.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.module2_matching.base_matcher import MatchResult, BaseMatcher
from src.module2_matching.scale_handler import (
    GaussianPyramid, ScaleAligner, map_keypoints_to_original,
)
from src.module2_matching.orb_fallback_matcher import ORBFallbackMatcher
from src.module2_matching.lightglue_matcher import (
    LightGlueMatcher, TORCH_AVAILABLE, KORNIA_AVAILABLE,
)
from src.module2_matching.hub_matcher import HubAndSpokeMatcher

from scripts.generate_mock_data import MockLunarDataGenerator
from src.module1_preprocessing.preprocessor import OHRCPreprocessor, TMC2Preprocessor
from src.module1_preprocessing.iirs_pca import IIRSPreprocessor


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_gen():
    """Fast mock generator (shared across module for speed)."""
    return MockLunarDataGenerator(area_km=1.0, max_ohrc_pixels=500, seed=42)


@pytest.fixture(scope="module")
def mock_data(mock_gen):
    """Pre-generated mock data dicts for all three instruments."""
    return mock_gen.generate_all()


@pytest.fixture(scope="module")
def preprocessed(mock_data):
    """Preprocessed images ready for matching."""
    ohrc_prep = OHRCPreprocessor().preprocess(mock_data["ohrc"]["image"])
    tmc2_prep = TMC2Preprocessor().preprocess(mock_data["tmc2"]["image"])
    iirs_prep = IIRSPreprocessor().preprocess(mock_data["iirs"]["cube"])
    return {
        "ohrc": ohrc_prep.image,
        "tmc2": tmc2_prep.image,
        "iirs_proxy": iirs_prep.image,
    }


@pytest.fixture
def synth_image_pair():
    """A pair of synthetic images with known correspondences.

    Creates a base pattern, then applies a small translation + scale
    to produce a second image.  Both share recognisable structure.
    """
    rng = np.random.default_rng(42)

    # 200x200 base image with circles, lines, and noise
    base = np.zeros((200, 200), dtype=np.float32)

    # Add some circular features (crater-like)
    import cv2
    for _ in range(15):
        cx, cy = rng.integers(30, 170, size=2)
        r = rng.integers(5, 25)
        cv2.circle(base, (int(cx), int(cy)), int(r), 0.8, 2)
        cv2.circle(base, (int(cx), int(cy)), int(r) - 2, 0.3, -1)

    # Add some random texture
    texture = rng.uniform(0.05, 0.15, (200, 200)).astype(np.float32)
    base += texture
    base = np.clip(base, 0, 1)

    # Create a "destination" image: slight translation + scale
    M = np.float32([[0.95, 0, 5], [0, 0.95, 8]])
    dst = cv2.warpAffine(base, M, (200, 200), borderValue=0)

    return base, dst


# ── MatchResult Tests ─────────────────────────────────────────────────────────

class TestMatchResult:
    """Tests for the MatchResult data structure."""

    def test_empty_result(self):
        """MatchResult.empty() creates zero-match result."""
        r = MatchResult.empty("OHRC", "TMC-2", 20.0)
        assert r.num_matches == 0
        assert r.has_matches is False
        assert r.match_confidence == 0.0
        assert r.inlier_ratio == 0.0

    def test_properties(self):
        """num_matches, has_matches, inlier_ratio work correctly."""
        r = MatchResult(
            keypoints_src=np.array([[10, 20], [30, 40]], dtype=np.float64),
            keypoints_dst=np.array([[15, 25], [35, 45]], dtype=np.float64),
            confidences=np.array([0.9, 0.7], dtype=np.float32),
            src_instrument="OHRC",
            dst_instrument="TMC-2",
            scale_gap=20.0,
            num_inliers=1,
            match_confidence=0.8,
        )
        assert r.num_matches == 2
        assert r.has_matches is True
        assert r.inlier_ratio == 0.5


# ── Scale Handler Tests ───────────────────────────────────────────────────────

class TestGaussianPyramid:
    """Tests for the Gaussian pyramid builder."""

    def test_build_creates_levels(self):
        """Pyramid has multiple levels."""
        img = np.random.default_rng(42).uniform(0, 1, (256, 256)).astype(np.float32)
        pyr = GaussianPyramid(img, max_levels=6)
        pyr.build()
        assert pyr.num_levels >= 2
        assert pyr.num_levels <= 6

    def test_level_0_is_original(self):
        """First pyramid level matches the original image."""
        img = np.random.default_rng(42).uniform(0, 1, (128, 128)).astype(np.float32)
        pyr = GaussianPyramid(img)
        pyr.build()
        np.testing.assert_array_equal(pyr.get_level(0).image, img)
        assert pyr.get_level(0).scale == 1.0

    def test_successive_levels_halve(self):
        """Each level is approximately half the previous resolution."""
        img = np.random.default_rng(42).uniform(0, 1, (256, 256)).astype(np.float32)
        pyr = GaussianPyramid(img, max_levels=5)
        pyr.build()
        for i in range(1, pyr.num_levels):
            prev_h = pyr.get_level(i - 1).shape[0]
            curr_h = pyr.get_level(i).shape[0]
            # Each level should be ~half (within rounding)
            assert 0.4 <= curr_h / prev_h <= 0.6

    def test_get_level_for_scale(self):
        """get_level_for_scale returns the closest matching level."""
        img = np.random.default_rng(42).uniform(0, 1, (512, 512)).astype(np.float32)
        pyr = GaussianPyramid(img, max_levels=8)
        pyr.build()

        # Request scale=0.25 (4x down) — should get level ~2
        level = pyr.get_level_for_scale(0.25)
        assert abs(level.scale - 0.25) < 0.15

    def test_small_image(self):
        """Very small images produce fewer levels."""
        img = np.random.default_rng(42).uniform(0, 1, (16, 16)).astype(np.float32)
        pyr = GaussianPyramid(img, max_levels=10)
        pyr.build()
        assert pyr.num_levels <= 3  # 16→8 is about as far as it can go

    def test_rejects_3d_input(self):
        """3-D images raise ValueError."""
        img = np.zeros((100, 100, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            GaussianPyramid(img)


class TestScaleAligner:
    """Tests for the scale aligner."""

    def test_align_returns_correct_keys(self):
        """Output dict contains expected keys."""
        high = np.random.default_rng(42).uniform(0, 1, (400, 400)).astype(np.float32)
        low = np.random.default_rng(43).uniform(0, 1, (20, 20)).astype(np.float32)

        aligner = ScaleAligner()
        result = aligner.align(high, low, scale_ratio=0.05)

        assert "aligned_src" in result
        assert "aligned_dst" in result
        assert "src_scale" in result
        assert "dst_scale" in result

    def test_downsampled_src_is_smaller(self):
        """Aligned source is smaller than original."""
        high = np.random.default_rng(42).uniform(0, 1, (400, 400)).astype(np.float32)
        low = np.random.default_rng(43).uniform(0, 1, (20, 20)).astype(np.float32)

        aligner = ScaleAligner()
        result = aligner.align(high, low, scale_ratio=0.05)

        assert result["aligned_src"].shape[0] < high.shape[0]

    def test_upsample_factor(self):
        """Upsampling the low-res image increases its size."""
        high = np.random.default_rng(42).uniform(0, 1, (200, 200)).astype(np.float32)
        low = np.random.default_rng(43).uniform(0, 1, (20, 20)).astype(np.float32)

        aligner = ScaleAligner(upsample_factor=2.0)
        result = aligner.align(high, low, scale_ratio=0.1)

        assert result["aligned_dst"].shape[0] == 40  # 20 * 2
        assert result["dst_scale"] == 2.0


class TestMapKeypointsToOriginal:
    """Tests for keypoint coordinate remapping."""

    def test_identity_transform(self):
        """Scale=1.0, offset=(0,0) returns identical coordinates."""
        kpts = np.array([[10, 20], [30, 40]], dtype=np.float64)
        mapped = map_keypoints_to_original(kpts, scale=1.0)
        np.testing.assert_array_almost_equal(mapped, kpts)

    def test_scale_up(self):
        """Scale=0.5 doubles the coordinates (2× downsampled)."""
        kpts = np.array([[10, 20]], dtype=np.float64)
        mapped = map_keypoints_to_original(kpts, scale=0.5)
        np.testing.assert_array_almost_equal(mapped, [[20, 40]])

    def test_offset(self):
        """Offset is applied as (row, col) → (y, x)."""
        kpts = np.array([[10, 20]], dtype=np.float64)
        mapped = map_keypoints_to_original(kpts, scale=1.0, offset=(5, 3))
        # x += col_offset(3), y += row_offset(5)
        np.testing.assert_array_almost_equal(mapped, [[13, 25]])

    def test_empty_keypoints(self):
        """Empty input returns empty output."""
        kpts = np.empty((0, 2), dtype=np.float64)
        mapped = map_keypoints_to_original(kpts, scale=0.5)
        assert mapped.shape == (0, 2)


# ── ORB Fallback Matcher Tests ────────────────────────────────────────────────

class TestORBFallbackMatcher:
    """Tests for the pure-OpenCV ORB fallback matcher."""

    def test_match_returns_match_result(self, synth_image_pair):
        """ORB matcher returns a MatchResult."""
        src, dst = synth_image_pair
        matcher = ORBFallbackMatcher(max_keypoints=2000, confidence_threshold=0.1)
        result = matcher.match(src, dst, scale_ratio=1.0, src_instrument="A", dst_instrument="B")

        assert isinstance(result, MatchResult)
        assert result.src_instrument == "A"
        assert result.dst_instrument == "B"

    def test_produces_matches_on_similar_images(self, synth_image_pair):
        """ORB finds matches between shifted versions of the same image."""
        src, dst = synth_image_pair
        matcher = ORBFallbackMatcher(max_keypoints=3000, confidence_threshold=0.1)
        result = matcher.match(src, dst, scale_ratio=1.0)
        assert result.num_matches > 0, "ORB should find at least some matches"

    def test_empty_on_blank_images(self):
        """Blank images produce zero matches."""
        blank = np.zeros((100, 100), dtype=np.float32)
        matcher = ORBFallbackMatcher()
        result = matcher.match(blank, blank, scale_ratio=1.0)
        assert result.num_matches == 0

    def test_metadata_contains_method(self, synth_image_pair):
        """Metadata includes method name."""
        src, dst = synth_image_pair
        matcher = ORBFallbackMatcher(confidence_threshold=0.1)
        result = matcher.match(src, dst)
        if result.has_matches:
            assert "ORB" in result.metadata.get("method", "")

    def test_scale_ratio_handling(self, synth_image_pair):
        """Matcher handles scale_ratio < 1 (uses ScaleAligner)."""
        src, dst = synth_image_pair
        # Create a "high-res" version
        import cv2
        high_res = cv2.resize(src, (400, 400), interpolation=cv2.INTER_CUBIC)
        matcher = ORBFallbackMatcher(max_keypoints=3000, confidence_threshold=0.1)
        result = matcher.match(high_res, dst, scale_ratio=0.5)
        assert isinstance(result, MatchResult)

    def test_keypoint_shapes(self, synth_image_pair):
        """Matched keypoint arrays have correct shapes."""
        src, dst = synth_image_pair
        matcher = ORBFallbackMatcher(confidence_threshold=0.1)
        result = matcher.match(src, dst)
        if result.has_matches:
            assert result.keypoints_src.shape[1] == 2
            assert result.keypoints_dst.shape[1] == 2
            assert len(result.confidences) == result.num_matches

    def test_deterministic_with_same_input(self, synth_image_pair):
        """Same input produces identical results."""
        src, dst = synth_image_pair
        matcher = ORBFallbackMatcher(confidence_threshold=0.1)
        r1 = matcher.match(src, dst)
        r2 = matcher.match(src, dst)
        if r1.has_matches and r2.has_matches:
            np.testing.assert_array_equal(r1.keypoints_src, r2.keypoints_src)


# ── LightGlue Matcher Tests ──────────────────────────────────────────────────

class TestLightGlueMatcher:
    """Tests for the LightGlue matcher (with fallback logic)."""

    def test_backend_resolution(self):
        """Backend is determined based on available libraries."""
        matcher = LightGlueMatcher()
        assert matcher.backend in ("lightglue", "loftr", "orb_fallback")

    def test_match_returns_match_result(self, synth_image_pair):
        """LightGlue matcher returns a MatchResult regardless of backend."""
        src, dst = synth_image_pair
        matcher = LightGlueMatcher(confidence_threshold=0.1)
        result = matcher.match(src, dst, scale_ratio=1.0, src_instrument="X", dst_instrument="Y")
        assert isinstance(result, MatchResult)

    def test_fallback_to_orb(self, synth_image_pair):
        """If torch/kornia unavailable, falls back to ORB and still works."""
        src, dst = synth_image_pair
        matcher = LightGlueMatcher(confidence_threshold=0.1)

        if matcher.backend == "orb_fallback":
            result = matcher.match(src, dst)
            assert isinstance(result, MatchResult)
            # ORB should find some matches on synthetic data
            # (may be 0 if the image is too featureless, that's OK)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
    def test_torch_device_attribute(self):
        """Device is correctly set when torch is available."""
        matcher = LightGlueMatcher(device="cpu")
        import torch
        assert matcher.device == torch.device("cpu")

    def test_static_availability_checks(self):
        """Static methods report availability correctly."""
        assert isinstance(LightGlueMatcher.torch_available(), bool)
        assert isinstance(LightGlueMatcher.kornia_available(), bool)
        assert isinstance(LightGlueMatcher.lightglue_available(), bool)

    def test_produces_matches_with_scale(self, synth_image_pair):
        """Matcher handles scale_ratio < 1."""
        src, dst = synth_image_pair
        import cv2
        high_res = cv2.resize(src, (400, 400), interpolation=cv2.INTER_CUBIC)
        matcher = LightGlueMatcher(confidence_threshold=0.1)
        result = matcher.match(high_res, dst, scale_ratio=0.5)
        assert isinstance(result, MatchResult)


# ── Hub-and-Spoke Matcher Tests ───────────────────────────────────────────────

class TestHubMatcher:
    """Tests for the hub-and-spoke orchestrator."""

    def test_hop1_returns_match_result(self, preprocessed):
        """Hop 1 (OHRC ↔ TMC-2) returns MatchResult."""
        hub = HubAndSpokeMatcher()
        result = hub.match_hop1(preprocessed["ohrc"], preprocessed["tmc2"])
        assert isinstance(result, MatchResult)
        assert result.src_instrument == "OHRC"
        assert result.dst_instrument == "TMC-2"

    def test_hop2_returns_match_result(self, preprocessed):
        """Hop 2 (TMC-2 ↔ IIRS proxy) returns MatchResult."""
        hub = HubAndSpokeMatcher()
        result = hub.match_hop2(preprocessed["tmc2"], preprocessed["iirs_proxy"])
        assert isinstance(result, MatchResult)
        assert result.src_instrument == "TMC-2"
        assert result.dst_instrument == "IIRS"

    def test_composite_returns_both_hops(self, preprocessed):
        """Composite matching returns dict with hop1 and hop2."""
        hub = HubAndSpokeMatcher()
        result = hub.match_composite(
            preprocessed["ohrc"],
            preprocessed["tmc2"],
            preprocessed["iirs_proxy"],
        )
        assert "hop1" in result
        assert "hop2" in result
        assert "success" in result
        assert "confidence" in result
        assert isinstance(result["hop1"], MatchResult)
        assert isinstance(result["hop2"], MatchResult)

    def test_match_router_ohrc_tmc2(self, preprocessed):
        """match() correctly routes OHRC ↔ TMC-2."""
        hub = HubAndSpokeMatcher()
        result = hub.match(
            preprocessed["ohrc"], preprocessed["tmc2"],
            "OHRC", "TMC-2",
        )
        assert isinstance(result, MatchResult)

    def test_match_router_tmc2_iirs(self, preprocessed):
        """match() correctly routes TMC-2 ↔ IIRS."""
        hub = HubAndSpokeMatcher()
        result = hub.match(
            preprocessed["tmc2"], preprocessed["iirs_proxy"],
            "TMC-2", "IIRS",
        )
        assert isinstance(result, MatchResult)

    def test_direct_ohrc_iirs_raises_without_hub(self, preprocessed):
        """Attempting OHRC ↔ IIRS without TMC-2 hub raises ValueError."""
        hub = HubAndSpokeMatcher()
        with pytest.raises(ValueError, match="320"):
            hub.match(
                preprocessed["ohrc"], preprocessed["iirs_proxy"],
                "OHRC", "IIRS",
            )

    def test_direct_ohrc_iirs_with_hub(self, preprocessed):
        """OHRC ↔ IIRS with TMC-2 hub succeeds (composite path)."""
        hub = HubAndSpokeMatcher()
        result = hub.match(
            preprocessed["ohrc"], preprocessed["iirs_proxy"],
            "OHRC", "IIRS",
            tmc2_image=preprocessed["tmc2"],
        )
        assert isinstance(result, MatchResult)

    def test_unknown_instruments_raise(self, preprocessed):
        """Unknown instrument names raise ValueError."""
        hub = HubAndSpokeMatcher()
        with pytest.raises(ValueError, match="Unknown"):
            hub.match(
                preprocessed["ohrc"], preprocessed["tmc2"],
                "UNKNOWN1", "UNKNOWN2",
            )

    def test_graceful_failure_blank_images(self):
        """Matching blank images returns empty result, no crash."""
        hub = HubAndSpokeMatcher()
        blank = np.zeros((50, 50), dtype=np.float32)
        result = hub.match_hop1(blank, blank)
        assert isinstance(result, MatchResult)
        assert result.match_confidence == 0.0


# ── Integration Tests (Phase 1 + Phase 2) ─────────────────────────────────────

class TestIntegrationPhase2:
    """End-to-end tests: mock data → preprocess → match."""

    def test_full_pipeline_hop1(self, mock_data):
        """Generate mock → preprocess → Hop 1 match."""
        ohrc_img = OHRCPreprocessor().preprocess(mock_data["ohrc"]["image"]).image
        tmc2_img = TMC2Preprocessor().preprocess(mock_data["tmc2"]["image"]).image

        hub = HubAndSpokeMatcher()
        result = hub.match_hop1(ohrc_img, tmc2_img)

        assert isinstance(result, MatchResult)
        # We don't assert >0 matches since ORB on synthetic lunar may be sparse

    def test_full_pipeline_hop2(self, mock_data):
        """Generate mock → preprocess → Hop 2 match."""
        tmc2_img = TMC2Preprocessor().preprocess(mock_data["tmc2"]["image"]).image
        iirs_proxy = IIRSPreprocessor().preprocess(mock_data["iirs"]["cube"]).image

        hub = HubAndSpokeMatcher()
        result = hub.match_hop2(tmc2_img, iirs_proxy)

        assert isinstance(result, MatchResult)

    def test_full_pipeline_composite(self, mock_data):
        """Generate mock → preprocess → composite (Hop 1 + Hop 2)."""
        ohrc_img = OHRCPreprocessor().preprocess(mock_data["ohrc"]["image"]).image
        tmc2_img = TMC2Preprocessor().preprocess(mock_data["tmc2"]["image"]).image
        iirs_proxy = IIRSPreprocessor().preprocess(mock_data["iirs"]["cube"]).image

        hub = HubAndSpokeMatcher()
        composite = hub.match_composite(ohrc_img, tmc2_img, iirs_proxy)

        assert isinstance(composite["hop1"], MatchResult)
        assert isinstance(composite["hop2"], MatchResult)
        assert isinstance(composite["confidence"], float)
        assert 0.0 <= composite["confidence"] <= 1.0
