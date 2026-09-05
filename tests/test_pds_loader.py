"""Unit tests for src/pds_loader.py, src/geo_align.py, and updated scale_handler.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import unittest
from pathlib import Path
import tempfile
import numpy as np

from src.pds_loader import parse_label, crop, iirs_to_grey, PDS_DTYPE_MAP
from src.geo_align import latlon_to_xyz, compute_centered_crop_slices
from src.module2_matching.scale_handler import compute_scale_ratio, compute_pyramid_depth, GaussianPyramid, ScaleAligner


class TestPDSLoaderAndScale(unittest.TestCase):

    def test_scale_ratio_and_depth(self):
        # TMC-2 (4.96) to IIRS (91.75)
        ratio = compute_scale_ratio(4.96, 91.75)
        self.assertAlmostEqual(ratio, 0.05406, places=4)
        depth = compute_pyramid_depth(ratio)
        self.assertEqual(depth, 6)

    def test_gaussian_pyramid_dynamic(self):
        img = np.random.rand(256, 256).astype(np.float32)
        pyr = GaussianPyramid(img, target_scale=0.05406)
        pyr.build()
        self.assertEqual(pyr.num_levels, 6)
        level = pyr.get_level_for_scale(0.05406)
        self.assertIsNotNone(level)

    def test_scale_aligner_from_meta(self):
        high_img = np.random.rand(400, 400).astype(np.float32)
        low_img = np.random.rand(50, 50).astype(np.float32)
        aligner = ScaleAligner()
        res = aligner.align_from_metadata(
            high_img,
            low_img,
            {"pixel_resolution": 4.96},
            {"pixel_resolution": 91.75},
        )
        self.assertIn("aligned_src", res)
        self.assertIn("aligned_dst", res)
        self.assertEqual(res["aligned_dst"].shape, (50, 50))

    def test_crop_helper(self):
        arr = np.arange(100).reshape((10, 10))
        c = crop(arr, 2, 5, 3, 7)
        self.assertEqual(c.shape, (3, 4))
        self.assertTrue(isinstance(c, np.ndarray))

    def test_latlon_to_xyz(self):
        # Test North Pole (lat=90)
        xyz = latlon_to_xyz(np.array([90.0]), np.array([0.0]))
        self.assertAlmostEqual(xyz[0, 0], 0.0, places=3)
        self.assertAlmostEqual(xyz[0, 1], 0.0, places=3)
        self.assertAlmostEqual(xyz[0, 2], 1737.4, places=3)

    def test_compute_centered_crop_slices(self):
        l_slice, s_slice = compute_centered_crop_slices(500, 100, 200, 50, 1000, 200)
        self.assertEqual(l_slice.stop - l_slice.start, 200)
        self.assertEqual(s_slice.stop - s_slice.start, 50)
        self.assertEqual(l_slice.start, 400)
        self.assertEqual(s_slice.start, 75)

    def test_iirs_to_grey(self):
        cube = np.ones((80, 50, 50), dtype=np.float32)
        band_wls = [(i + 1, 700.0 + i * 20.0) for i in range(80)]
        grey = iirs_to_grey(cube, band_wls, slice(10, 30), slice(10, 30), max_nm=2000.0)
        self.assertEqual(grey.shape, (20, 20))
        self.assertEqual(grey.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
