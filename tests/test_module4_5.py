"""
Tests for Module 4 (Registration) and Module 5 (Confidence Visualizations).
"""

import pytest
import numpy as np
import os
import cv2

from src.module2_matching.base_matcher import MatchResult
from src.module4_registration.registration import GeometricRegistrar, RegistrationResult
from src.module5_confidence.visualizer import ExplainabilityVisualizer


@pytest.fixture
def perfect_square_match():
    """Generates a MatchResult describing a perfect translation."""
    # A simple square translating by (10, 20), with 10 points
    src_pts = np.array([
        [0, 0], [100, 0], [100, 100], [0, 100],
        [50, 0], [100, 50], [50, 100], [0, 50],
        [25, 25], [75, 75]
    ], dtype=np.float32)
    
    dst_pts = src_pts + np.array([10, 20], dtype=np.float32)
    
    return MatchResult(
        keypoints_src=src_pts,
        keypoints_dst=dst_pts,
        confidences=np.ones(10, dtype=np.float32),
        src_instrument="SRC",
        dst_instrument="DST",
        scale_gap=1.0,
        num_inliers=10,
        match_confidence=1.0,
        metadata={}
    )

@pytest.fixture
def match_with_outliers(perfect_square_match):
    """Adds outliers to the perfect match."""
    src_pts = np.vstack([perfect_square_match.keypoints_src, [[150, 150], [160, 160]]])
    # Outliers point to wrong locations
    dst_pts = np.vstack([perfect_square_match.keypoints_dst, [[0, 0], [300, 300]]])
    
    return MatchResult(
        keypoints_src=src_pts,
        keypoints_dst=dst_pts,
        confidences=np.ones(12, dtype=np.float32),
        src_instrument="SRC",
        dst_instrument="DST",
        scale_gap=1.0,
        num_inliers=12,
        match_confidence=1.0,
        metadata={}
    )


class TestGeometricRegistrar:
    
    def test_insufficient_matches(self):
        registrar = GeometricRegistrar()
        bad_match = MatchResult(
            keypoints_src=np.array([[0, 0]]),
            keypoints_dst=np.array([[1, 1]]),
            confidences=np.array([1.0]),
            src_instrument="A", dst_instrument="B",
            scale_gap=1.0, num_inliers=1, match_confidence=1.0, metadata={}
        )
        
        result = registrar.register(bad_match)
        assert not result.success
        assert result.num_inliers == 0
        assert np.array_equal(result.transform_matrix, np.eye(3))

    def test_perfect_homography(self, perfect_square_match):
        registrar = GeometricRegistrar(transform_type="homography")
        result = registrar.register(perfect_square_match)
        
        assert result.success
        assert result.num_inliers == 10
        assert result.rmse < 1e-4
        
        # Matrix should be a pure translation: tx=10, ty=20
        # [[1, 0, 10],
        #  [0, 1, 20],
        #  [0, 0,  1]]
        np.testing.assert_allclose(result.transform_matrix[0, 2], 10.0, atol=1e-3)
        np.testing.assert_allclose(result.transform_matrix[1, 2], 20.0, atol=1e-3)
        
    def test_magsac_rejects_outliers(self, match_with_outliers):
        registrar = GeometricRegistrar(transform_type="homography", reproj_thresh_pixels=3.0)
        result = registrar.register(match_with_outliers)
        
        assert result.success
        assert result.num_inliers == 10  # Should reject the 2 outliers
        assert np.sum(result.inliers_mask) == 10
        
        # Last two points were outliers
        assert not result.inliers_mask[10]
        assert not result.inliers_mask[11]


class TestExplainabilityVisualizer:
    
    def test_plot_matches_saves_file(self, tmp_path, match_with_outliers):
        img_src = np.zeros((150, 150), dtype=np.float32)
        img_dst = np.zeros((150, 150), dtype=np.float32)
        save_path = tmp_path / "matches.png"
        
        inliers_mask = np.array([True]*10 + [False, False])
        
        ExplainabilityVisualizer.plot_matches(
            img_src, img_dst, match_with_outliers, inliers_mask, str(save_path)
        )
        
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_plot_overlay_saves_file(self, tmp_path, perfect_square_match):
        img_src = np.zeros((150, 150), dtype=np.float32)
        img_src[50:100, 50:100] = 1.0 # White square
        
        img_dst = np.zeros((150, 150), dtype=np.float32)
        
        registrar = GeometricRegistrar()
        reg_result = registrar.register(perfect_square_match)
        
        save_path = tmp_path / "overlay.png"
        
        ExplainabilityVisualizer.plot_registration_overlay(
            img_src, img_dst, reg_result, str(save_path)
        )
        
        assert save_path.exists()
        assert save_path.stat().st_size > 0
