from __future__ import annotations
"""
Data loader module for memory-mapped access to massive Chandrayaan-2 PDS4 binary files.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.pds_parser import PDS4Parser, PDS4Label, BandInfo


class MemmapLoader:
    """Zero-copy memory-mapped loader for PDS4 binary data."""
    
    def __init__(self, data_path: str | Path, xml_path: str | Path):
        """Parse XML label and create memmap."""
        parser = PDS4Parser(xml_path)
        self.label: PDS4Label = parser.parse()
        self._mmap = np.memmap(
            str(data_path),
            dtype=self.label.array.dtype,
            mode='r',
            offset=self.label.array.offset_bytes,
            shape=self.label.array.shape
        )
    
    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the data array."""
        return self._mmap.shape
    
    @property
    def ndim(self) -> int:
        """Get the number of dimensions of the data array."""
        return self._mmap.ndim
    
    def extract_patch(
        self,
        center_line: int | None = None,
        center_sample: int | None = None,
        size: int = 4000,
    ) -> np.ndarray:
        """
        Extract a square patch from a 2D image.
        If center is None, use image center.
        Returns a copy (not a view) of shape (size, size).
        Handles edge clamping.
        """
        if self.ndim != 2:
            raise ValueError("extract_patch only supports 2D images.")
        
        lines, samples = self.shape
        
        if center_line is None:
            center_line = lines // 2
        if center_sample is None:
            center_sample = samples // 2
            
        half_size = size // 2
        start_line = center_line - half_size
        end_line = start_line + size
        start_sample = center_sample - half_size
        end_sample = start_sample + size
        
        # Clamped indices for the source array
        src_start_line = max(0, start_line)
        src_end_line = min(lines, end_line)
        src_start_sample = max(0, start_sample)
        src_end_sample = min(samples, end_sample)
        
        sub_img = self._mmap[src_start_line:src_end_line, src_start_sample:src_end_sample]
        
        # Calculate padding needed
        pad_top = src_start_line - start_line
        pad_bottom = end_line - src_end_line
        pad_left = src_start_sample - start_sample
        pad_right = end_sample - src_end_sample
        
        # Use edge clamping for out of bound areas
        return np.pad(sub_img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')
    
    def extract_band(
        self,
        band_index: int,
    ) -> np.ndarray:
        """
        Extract a single band from a 3D spectral cube.
        For BSQ layout (Band, Line, Sample), returns shape (lines, samples).
        Returns a copy.
        """
        if self.ndim != 3:
            raise ValueError("extract_band only supports 3D images.")
        
        return self._mmap[band_index, :, :].copy()
    
    def extract_band_by_wavelength(
        self,
        target_nm: float,
    ) -> tuple[BandInfo, np.ndarray]:
        """
        Find the band closest to target_nm and extract it.
        Returns (band_info, 2d_array).
        """
        if not hasattr(self.label, 'bands') or not self.label.bands:
            raise ValueError("No band information available in the label.")
            
        best_band = min(
            self.label.bands,
            key=lambda b: abs(b.center_wavelength_nm - target_nm)
        )
        
        # band_number is 1-indexed
        band_index = best_band.band_number - 1
        
        return best_band, self.extract_band(band_index)
    
    def get_thumbnail(
        self,
        max_dim: int = 1000,
    ) -> np.ndarray:
        """
        Create a downsampled thumbnail of the full image.
        For 3D cubes, uses the median of bands 30-40 (NIR window).
        Uses strided slicing, not cv2.resize, for memory efficiency.
        """
        if self.ndim == 2:
            lines, samples = self.shape
            img = self._mmap
        elif self.ndim == 3:
            # Median of bands 30-40
            img = np.median(self._mmap[30:41, :, :], axis=0)
            lines, samples = img.shape
        else:
            raise ValueError("Unsupported number of dimensions.")
            
        stride_y = max(1, lines // max_dim)
        stride_x = max(1, samples // max_dim)
        stride = max(stride_y, stride_x)
        
        return img[::stride, ::stride].copy()
