"""
tests/test_subpixel_refiner.py — Unit tests for sub-pixel refiner and spatial distribution bucketer.
"""

import numpy as np
import cv2
import pytest
from src.module2_matching.subpixel_refiner import refine_correspondences, bucket_matches_grid


def test_refine_correspondences_phase_correlate():
    img_src = np.zeros((100, 100), dtype=np.float32)
    cv2.circle(img_src, (50, 50), 10, 200.0, -1)
    cv2.circle(img_src, (50, 50), 6, 50.0, -1)
    img_src = cv2.GaussianBlur(img_src, (3, 3), 1.0)
    
    # Subpixel shift: dx=0.8, dy=-0.6
    M = np.float32([[1, 0, 0.8], [0, 1, -0.6]])
    img_ref = cv2.warpAffine(img_src, M, (100, 100))
    
    pts_src = np.array([[50.0, 50.0]], dtype=np.float32)
    pts_ref_init = np.array([[50.0, 50.0]], dtype=np.float32)
    
    refined_pts, n_ref, n_rej, mean_s, shifts = refine_correspondences(
        img_src, img_ref, pts_src, pts_ref_init, patch_size=21, conf_floor=0.1, method='phase_correlate'
    )
    assert n_ref == 1
    assert n_rej == 0
    assert mean_s > 0.0
    dx = float(refined_pts[0, 0] - 50.0)
    dy = float(refined_pts[0, 1] - 50.0)
    assert abs(dx - 0.8) < 0.2
    assert abs(dy - (-0.6)) < 0.2


def test_refine_correspondences_ncc():
    img_src = np.zeros((100, 100), dtype=np.float32)
    cv2.circle(img_src, (50, 50), 10, 200.0, -1)
    cv2.circle(img_src, (50, 50), 6, 50.0, -1)
    img_src = cv2.GaussianBlur(img_src, (3, 3), 1.0)
    
    M = np.float32([[1, 0, 0.5], [0, 1, -0.4]])
    img_ref = cv2.warpAffine(img_src, M, (100, 100))
    
    pts_src = np.array([[50.0, 50.0]], dtype=np.float32)
    pts_ref_init = np.array([[50.0, 50.0]], dtype=np.float32)
    
    refined_pts, n_ref, n_rej, mean_s, shifts = refine_correspondences(
        img_src, img_ref, pts_src, pts_ref_init, patch_size=15, conf_floor=0.3, method='ncc'
    )
    assert n_ref == 1
    assert n_rej == 0
    dx = float(refined_pts[0, 0] - 50.0)
    dy = float(refined_pts[0, 1] - 50.0)
    assert abs(dx - 0.5) < 0.3
    assert abs(dy - (-0.4)) < 0.3


def test_refine_correspondences_rejection_on_noise():
    img_src = np.zeros((100, 100), dtype=np.float32)
    img_ref = np.zeros((100, 100), dtype=np.float32)
    
    pts_src = np.array([[50.0, 50.0]], dtype=np.float32)
    pts_ref = np.array([[50.0, 50.0]], dtype=np.float32)
    
    refined_pts, n_ref, n_rej, mean_s, shifts = refine_correspondences(
        img_src, img_ref, pts_src, pts_ref, patch_size=15, conf_floor=0.3
    )
    assert n_ref == 0
    assert n_rej == 1
    assert np.allclose(refined_pts, pts_ref)


def test_bucket_matches_grid():
    pts_src = np.zeros((100, 2), dtype=np.float32)
    pts_ref = np.random.uniform(0, 10, (100, 2)).astype(np.float32)
    
    b_src, b_ref, indices, cap, stats = bucket_matches_grid(
        pts_src, pts_ref, ref_shape=(1000, 1000), grid_size=8, cap=5
    )
    assert cap == 5
    assert len(b_src) == 5
    assert len(b_ref) == 5
    assert len(indices) == 5
    assert stats['original_matches'] == 100
    assert stats['bucketed_matches'] == 5
