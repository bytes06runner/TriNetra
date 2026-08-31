"""
Tests for Module 3: Crater-Structural Verification.
"""

import pytest
import numpy as np

from src.module3_crater_verification import (
    StructuralExtractor,
    StructuralVerifier,
    StructuralMatcher
)
from src.module2_matching.base_matcher import MatchResult

class DummyMatcher:
    """Mock matcher for testing StructuralMatcher."""
    def match(self, src_image, dst_image, **kwargs):
        # Return a fake result
        return MatchResult(
            keypoints_src=np.array([[10, 10]]),
            keypoints_dst=np.array([[10, 10]]),
            confidences=np.array([0.9]),
            src_instrument="TEST",
            dst_instrument="TEST",
            scale_gap=1.0,
            num_inliers=1,
            match_confidence=0.9,
            metadata={"dummy": True}
        )

@pytest.fixture
def mock_image():
    """Generates a synthetic crater-like image."""
    img = np.zeros((100, 100), dtype=np.float32)
    # Draw a mock crater rim (circle)
    for y in range(100):
        for x in range(100):
            r = np.sqrt((x - 50)**2 + (y - 50)**2)
            if 20 < r < 25:
                img[y, x] = 1.0 # Bright rim
            elif 25 <= r < 30:
                img[y, x] = -0.5 # Dark outer shadow
    # Normalize to [0, 1]
    img = (img - img.min()) / (img.max() - img.min())
    return img

class TestStructuralExtractor:
    
    @pytest.mark.parametrize("method", ["sato", "meijering", "sobel", "composite"])
    def test_extractor_methods(self, mock_image, method):
        extractor = StructuralExtractor(method=method, sigmas=(1, 2))
        struct = extractor.extract(mock_image)
        
        assert struct.shape == mock_image.shape
        assert struct.dtype == np.float32
        assert np.min(struct) >= 0.0
        assert np.max(struct) <= 1.0
        
        # The center should be relatively flat (0)
        assert struct[50, 50] < 0.2
        # The rim area should contain high structural values
        assert np.max(struct[50, 15:35]) > 0.3

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            StructuralExtractor(method="invalid_method")

class TestStructuralVerifier:
    
    def test_verify_rejects_bad_matches(self):
        verifier = StructuralVerifier(patch_size_dst=15, ncc_threshold=0.8, extractor_method="sobel")
        
        # Create identical images, but shift the matches so they are completely wrong
        img1 = np.random.rand(100, 100).astype(np.float32)
        img2 = img1.copy()
        
        bad_match = MatchResult(
            keypoints_src=np.array([[50, 50]]),
            keypoints_dst=np.array([[10, 10]]), # Shifted far away
            confidences=np.array([0.9]),
            src_instrument="A",
            dst_instrument="B",
            scale_gap=1.0,
            num_inliers=1,
            match_confidence=0.9,
            metadata={}
        )
        
        verified = verifier.verify(bad_match, img1, img2)
        
        # Should reject because patches around (50,50) and (10,10) are different random noise
        assert not verified.has_matches
        assert verified.num_matches == 0

    def test_verify_accepts_good_matches(self):
        verifier = StructuralVerifier(patch_size_dst=15, ncc_threshold=0.8, extractor_method="sobel")
        
        # Use a deterministic pattern to ensure high NCC
        img = np.zeros((100, 100), dtype=np.float32)
        img[40:60, 40:60] = 1.0 # White square
        
        good_match = MatchResult(
            keypoints_src=np.array([[40, 40]]), # Corner of the square
            keypoints_dst=np.array([[40, 40]]),
            confidences=np.array([0.9]),
            src_instrument="A",
            dst_instrument="B",
            scale_gap=1.0,
            num_inliers=1,
            match_confidence=0.9,
            metadata={}
        )
        
        verified = verifier.verify(good_match, img, img)
        
        # Should accept because patches are identical
        assert verified.has_matches
        assert verified.num_matches == 1
        # Confidences should be updated
        assert verified.confidences[0] > 0.0

    def test_scale_alignment_during_verification(self):
        verifier = StructuralVerifier(patch_size_dst=15, ncc_threshold=0.5, extractor_method="sobel")
        
        # High res image
        img_src = np.zeros((200, 200), dtype=np.float32)
        img_src[80:120, 80:120] = 1.0
        
        # Low res image (2x smaller)
        img_dst = np.zeros((100, 100), dtype=np.float32)
        img_dst[40:60, 40:60] = 1.0
        
        scaled_match = MatchResult(
            keypoints_src=np.array([[80, 80]]), # Corner
            keypoints_dst=np.array([[40, 40]]), # Corresponding corner
            confidences=np.array([0.9]),
            src_instrument="HR",
            dst_instrument="LR",
            scale_gap=2.0, # Source is 2x resolution
            num_inliers=1,
            match_confidence=0.9,
            metadata={}
        )
        
        verified = verifier.verify(scaled_match, img_src, img_dst)
        assert verified.has_matches

class TestStructuralMatcher:
    
    def test_structural_matcher_delegates(self, mock_image):
        base_matcher = DummyMatcher()
        matcher = StructuralMatcher(base_matcher, extractor_method="sobel")
        
        result = matcher.match(
            src_image=mock_image,
            dst_image=mock_image,
            scale_ratio=1.0,
            src_instrument="A",
            dst_instrument="B"
        )
        
        assert result.has_matches
        assert result.metadata["structural_matching"] is True
        assert result.metadata["extractor_method"] == "sobel"
