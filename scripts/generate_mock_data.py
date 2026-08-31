"""
Synthetic lunar data generator that creates geometrically-consistent mock data for OHRC, TMC-2, and IIRS.
This module is designed for the SIH26166 Chandrayaan-2 lunar image correspondence pipeline.
"""

import numpy as np
import scipy.ndimage
import cv2
import json
import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

class MockLunarDataGenerator:
    """
    Generates synthetic but geometrically and radiometrically plausible mock data
    for Chandrayaan-2 instruments: OHRC, TMC-2, and IIRS.
    All instruments view the same underlying terrain model.
    """

    # Instrument specs
    OHRC_GSD = 0.25
    TMC2_GSD = 5.0
    IIRS_GSD = 80.0
    IIRS_BANDS = 256
    
    def __init__(
        self,
        area_km: float = 4.0,
        sun_elevation_deg: float = 30.0,
        sun_azimuth_deg: float = 45.0,
        seed: int = 42,
        max_ohrc_pixels: int = 4000
    ) -> None:
        """
        Initialize the generator.

        Args:
            area_km: The physical area simulated in km.
            sun_elevation_deg: Sun elevation angle in degrees.
            sun_azimuth_deg: Sun azimuth angle in degrees.
            seed: Random seed for reproducibility.
            max_ohrc_pixels: Caps the OHRC dimension.
        """
        self.area_km = float(area_km)
        self.sun_elevation_deg = float(sun_elevation_deg)
        self.sun_azimuth_deg = float(sun_azimuth_deg)
        self.seed = int(seed)
        self.max_ohrc_pixels = int(max_ohrc_pixels)
        
        self.rng = np.random.default_rng(self.seed)
        
        # Base grid size calculation: area_km * 1000 / 2000 -> base_gsd
        # To get ~2000x2000 grid for area_km
        self.base_size = 2000
        self.base_gsd = (self.area_km * 1000.0) / self.base_size
        
        # Cache for lazy evaluation
        self._heightmap: Optional[np.ndarray] = None
        self._albedo_map: Optional[np.ndarray] = None
        self._mineral_map: Optional[np.ndarray] = None
        
        # Shared random center lat/lon
        self.center_lat = self.rng.uniform(-70.0, 70.0)
        self.center_lon = self.rng.uniform(-180.0, 180.0)

    @property
    def heightmap(self) -> np.ndarray:
        """
        Base terrain at base_gsd resolution.
        """
        if self._heightmap is not None:
            return self._heightmap

        try:
            # Generate multiple octaves of noise
            noise = np.zeros((self.base_size, self.base_size), dtype=np.float64)
            scales = [100.0, 50.0, 20.0, 5.0]
            weights = [1.0, 0.5, 0.25, 0.1]
            
            for scale, weight in zip(scales, weights):
                raw_noise = self.rng.normal(0, 1, (self.base_size, self.base_size))
                smoothed = scipy.ndimage.gaussian_filter(raw_noise, sigma=scale)
                noise += smoothed * weight

            noise -= np.min(noise)
            if np.max(noise) > 0:
                noise /= np.max(noise)
            else:
                noise = np.zeros_like(noise)

            height = noise * 0.2  # Base undulation
            
            # Place craters
            num_craters = self.rng.integers(30, 80)
            Y, X = np.ogrid[:self.base_size, :self.base_size]
            
            for _ in range(num_craters):
                cx = self.rng.integers(0, self.base_size)
                cy = self.rng.integers(0, self.base_size)
                r = self.rng.integers(3, 200)
                
                dist_sq = (X - cx)**2 + (Y - cy)**2
                mask_in = dist_sq < r**2
                mask_rim = (dist_sq >= r**2) & (dist_sq < (r*1.5)**2)
                
                # Inside crater (parabolic)
                if np.any(mask_in):
                    dist_norm = np.sqrt(dist_sq[mask_in]) / r
                    depth = 0.1 * r / 200.0
                    height[mask_in] -= depth * (1.0 - dist_norm**2)
                    
                # Rim (exponential decay)
                if np.any(mask_rim):
                    dist_norm = np.sqrt(dist_sq[mask_rim]) / r
                    rim_height = 0.05 * r / 200.0
                    height[mask_rim] += rim_height * np.exp(-(dist_norm - 1.0) * 4.0)

            # Normalize to [0, 1]
            height -= np.min(height)
            max_h = np.max(height)
            if max_h > 0:
                height /= max_h
                
            self._heightmap = height
            
        except Exception:
            self._heightmap = np.zeros((self.base_size, self.base_size), dtype=np.float64)

        return self._heightmap

    @property
    def albedo_map(self) -> np.ndarray:
        """
        Surface reflectance at base resolution.
        """
        if self._albedo_map is not None:
            return self._albedo_map

        try:
            # Base albedo variation
            raw_noise = self.rng.normal(0.15, 0.02, (self.base_size, self.base_size))
            smoothed = scipy.ndimage.gaussian_filter(raw_noise, sigma=30.0)
            albedo = np.clip(smoothed, 0.1, 0.25).astype(np.float32)
            
            self._albedo_map = albedo
        except Exception:
            self._albedo_map = np.full((self.base_size, self.base_size), 0.15, dtype=np.float32)

        return self._albedo_map

    @property
    def mineral_map(self) -> np.ndarray:
        """
        Categorical map at base resolution. 
        4 mineral classes: 0=regolith, 1=pyroxene, 2=plagioclase, 3=olivine.
        """
        if self._mineral_map is not None:
            return self._mineral_map

        try:
            num_seeds = 15
            seeds = self.rng.integers(0, self.base_size, size=(num_seeds, 2))
            
            # Bias toward regolith (0)
            classes = self.rng.choice([0, 0, 0, 1, 2, 3], size=num_seeds)
            
            # Nearest neighbor classification
            Y, X = np.indices((self.base_size, self.base_size))
            
            mineral = np.zeros((self.base_size, self.base_size), dtype=int)
            min_dist = np.full((self.base_size, self.base_size), np.inf)
            
            for i in range(num_seeds):
                sy, sx = seeds[i]
                c = classes[i]
                dist_sq = (X - sx)**2 + (Y - sy)**2
                mask = dist_sq < min_dist
                mineral[mask] = c
                min_dist[mask] = dist_sq[mask]
                
            # Add some noise
            noise_mask = self.rng.random((self.base_size, self.base_size)) < 0.05
            mineral[noise_mask] = 0
            
            self._mineral_map = scipy.ndimage.median_filter(mineral, size=5)
        except Exception:
            self._mineral_map = np.zeros((self.base_size, self.base_size), dtype=int)

        return self._mineral_map

    def _apply_illumination(self, heightmap: np.ndarray, albedo: np.ndarray) -> np.ndarray:
        """
        Applies Lambertian illumination model.
        """
        try:
            gy, gx = np.gradient(heightmap, self.base_gsd, self.base_gsd)
            normal = np.dstack((-gx, -gy, np.ones_like(heightmap)))
            norm = np.linalg.norm(normal, axis=2, keepdims=True)
            norm[norm == 0] = 1.0
            normal /= norm
            
            el_rad = np.radians(self.sun_elevation_deg)
            az_rad = np.radians(self.sun_azimuth_deg)
            
            sx = np.cos(el_rad) * np.sin(az_rad)
            sy = np.cos(el_rad) * np.cos(az_rad)
            sz = np.sin(el_rad)
            sun_dir = np.array([sx, sy, sz])
            
            dot_prod = np.sum(normal * sun_dir, axis=2)
            
            ambient = 0.02
            # Simple shadow estimation
            shadow_mask = dot_prod < 0
            
            illumination = np.maximum(0, dot_prod)
            
            image = albedo * illumination + ambient
            image[shadow_mask] = ambient * albedo[shadow_mask]
            
            return np.clip(image, 0, 1).astype(np.float32)
        except Exception:
            return np.zeros_like(heightmap, dtype=np.float32)

    def generate_ohrc(self) -> Dict[str, Any]:
        """
        Generates simulated OHRC data.
        """
        try:
            hmap = self.heightmap
            alb = self.albedo_map
            illuminated = self._apply_illumination(hmap, alb)
            
            target_size = int(self.base_size * (self.base_gsd / self.OHRC_GSD))
            
            if target_size > self.max_ohrc_pixels:
                factor = target_size / self.base_size
                crop_size_base = int(self.max_ohrc_pixels / factor)
                start = (self.base_size - crop_size_base) // 2
                end = start + crop_size_base
                
                illuminated_crop = illuminated[start:end, start:end]
                final_size = self.max_ohrc_pixels
            else:
                illuminated_crop = illuminated
                final_size = target_size
                
            if final_size > 0:
                ohrc_img = cv2.resize(illuminated_crop, (final_size, final_size), interpolation=cv2.INTER_CUBIC)
            else:
                ohrc_img = np.zeros((1, 1), dtype=np.float32)
                
            noise = self.rng.normal(0, 0.005, ohrc_img.shape)
            ohrc_img = np.clip(ohrc_img + noise, 0, 1).astype(np.float32)
            
            extent_m = self.OHRC_GSD * ohrc_img.shape[0]
            half_ext = extent_m / 2.0
            
            return {
                'image': ohrc_img,
                'gsd_m': self.OHRC_GSD,
                'instrument': 'OHRC',
                'shape': ohrc_img.shape,
                'extent_m': (-half_ext, half_ext, -half_ext, half_ext)
            }
        except Exception:
            return {
                'image': np.zeros((1, 1), dtype=np.float32),
                'gsd_m': self.OHRC_GSD,
                'instrument': 'OHRC',
                'shape': (1, 1),
                'extent_m': (0, 0, 0, 0)
            }

    def generate_tmc2(self) -> Dict[str, Any]:
        """
        Generates simulated TMC-2 data.
        """
        try:
            hmap = self.heightmap
            alb = self.albedo_map
            illuminated = self._apply_illumination(hmap, alb)
            
            target_size = int(self.base_size * (self.base_gsd / self.TMC2_GSD))
            
            if target_size > 0:
                tmc_img = cv2.resize(illuminated, (target_size, target_size), interpolation=cv2.INTER_AREA)
            else:
                tmc_img = np.zeros((1, 1), dtype=np.float32)
                
            noise = self.rng.normal(0, 0.01, tmc_img.shape)
            tmc_img = np.clip(tmc_img + noise, 0, 1).astype(np.float32)
            
            extent_m = self.TMC2_GSD * tmc_img.shape[0]
            half_ext = extent_m / 2.0
            
            return {
                'image': tmc_img,
                'gsd_m': self.TMC2_GSD,
                'instrument': 'TMC-2',
                'shape': tmc_img.shape,
                'extent_m': (-half_ext, half_ext, -half_ext, half_ext)
            }
        except Exception:
            return {
                'image': np.zeros((1, 1), dtype=np.float32),
                'gsd_m': self.TMC2_GSD,
                'instrument': 'TMC-2',
                'shape': (1, 1),
                'extent_m': (0, 0, 0, 0)
            }

    def generate_iirs(self) -> Dict[str, Any]:
        """
        Generates simulated IIRS spectral cube.
        """
        try:
            hmap = self.heightmap
            alb = self.albedo_map
            illuminated = self._apply_illumination(hmap, alb)
            mineral = self.mineral_map
            
            target_size = int(self.base_size * (self.base_gsd / self.IIRS_GSD))
            target_size = max(target_size, 1)
            
            ill_small = cv2.resize(illuminated, (target_size, target_size), interpolation=cv2.INTER_AREA)
            min_small = cv2.resize(mineral.astype(np.uint8), (target_size, target_size), interpolation=cv2.INTER_NEAREST)
            
            bands = np.linspace(0.8, 5.0, self.IIRS_BANDS)
            cube = np.zeros((target_size, target_size, self.IIRS_BANDS), dtype=np.float32)
            
            # Compute spectral profiles
            for b_idx, wl in enumerate(bands):
                refl = np.ones((target_size, target_size), dtype=np.float32)
                
                # Regolith (0)
                refl[min_small == 0] *= (1.0 - 0.05 * (wl - 0.8) / 4.2)
                
                # Pyroxene (1): absorptions at 1.0 and 2.0
                p_absorp = 0.2 * np.exp(-((wl - 1.0)/0.1)**2) + 0.3 * np.exp(-((wl - 2.0)/0.2)**2)
                refl[min_small == 1] *= (1.0 - p_absorp)
                
                # Plagioclase (2): absorption at 1.25
                pl_absorp = 0.15 * np.exp(-((wl - 1.25)/0.15)**2)
                refl[min_small == 2] *= (1.0 - pl_absorp)
                
                # Olivine (3): broad at 1.05
                ol_absorp = 0.25 * np.exp(-((wl - 1.05)/0.25)**2)
                refl[min_small == 3] *= (1.0 - ol_absorp)
                
                noise = self.rng.normal(0, 0.008, ill_small.shape)
                band_img = ill_small * refl + noise
                cube[:, :, b_idx] = np.clip(band_img, 0, 1)
                
            extent_m = self.IIRS_GSD * target_size
            half_ext = extent_m / 2.0
            
            return {
                'cube': cube.astype(np.float32),
                'band_centers_um': bands.tolist(),
                'gsd_m': self.IIRS_GSD,
                'instrument': 'IIRS',
                'shape': cube.shape,
                'extent_m': (-half_ext, half_ext, -half_ext, half_ext)
            }
        except Exception:
            return {
                'cube': np.zeros((1, 1, self.IIRS_BANDS), dtype=np.float32),
                'band_centers_um': np.linspace(0.8, 5.0, self.IIRS_BANDS).tolist(),
                'gsd_m': self.IIRS_GSD,
                'instrument': 'IIRS',
                'shape': (1, 1, self.IIRS_BANDS),
                'extent_m': (0, 0, 0, 0)
            }

    def generate_metadata(self) -> Dict[str, Any]:
        """
        Generates simulated SPICE-kernel-style metadata.
        """
        try:
            now = datetime.datetime.now(datetime.timezone.utc)
            
            def make_meta(inst, gsd):
                return {
                    'center_lat': self.center_lat,
                    'center_lon': self.center_lon,
                    'corner_coords': [
                        (self.center_lat - 0.1, self.center_lon - 0.1),
                        (self.center_lat - 0.1, self.center_lon + 0.1),
                        (self.center_lat + 0.1, self.center_lon + 0.1),
                        (self.center_lat + 0.1, self.center_lon - 0.1),
                    ],
                    'sun_elevation_deg': self.sun_elevation_deg,
                    'sun_azimuth_deg': self.sun_azimuth_deg,
                    'acquisition_time': now.isoformat(),
                    'instrument_name': inst,
                    'gsd_m': gsd
                }

            return {
                'OHRC': make_meta('OHRC', self.OHRC_GSD),
                'TMC-2': make_meta('TMC-2', self.TMC2_GSD),
                'IIRS': make_meta('IIRS', self.IIRS_GSD),
            }
        except Exception:
            return {}
            
    def generate_all(self) -> Dict[str, Any]:
        """
        Generates data for all three instruments.
        """
        return {
            'ohrc': self.generate_ohrc(),
            'tmc2': self.generate_tmc2(),
            'iirs': self.generate_iirs()
        }


if __name__ == '__main__':
    print("Generating mock lunar data...")
    generator = MockLunarDataGenerator()
    
    data = generator.generate_all()
    metadata = generator.generate_metadata()
    
    out_dir = Path("data/mock")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ohrc = data['ohrc']
    print(f"OHRC generated: {ohrc['shape']}, type {ohrc['image'].dtype}")
    np.save(out_dir / "ohrc_mock.npy", ohrc['image'])
    
    tmc2 = data['tmc2']
    print(f"TMC-2 generated: {tmc2['shape']}, type {tmc2['image'].dtype}")
    np.save(out_dir / "tmc2_mock.npy", tmc2['image'])
    
    iirs = data['iirs']
    print(f"IIRS generated: {iirs['shape']}, type {iirs['cube'].dtype}")
    np.save(out_dir / "iirs_cube_mock.npy", iirs['cube'])
    
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Mock data saved to {out_dir.resolve()}")
