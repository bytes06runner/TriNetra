import numpy as np
import scipy.ndimage as ndimage
import logging
from typing import Optional, Tuple, Dict, Any

from .preprocessor import BasePreprocessor, PreprocessingResult

logger = logging.getLogger(__name__)

class IIRSPreprocessor(BasePreprocessor):
    """IIRS hyperspectral preprocessing: 256-band cube -> matchable 2D proxy image.
    
    Strategy:
    1. Bad-band removal: Drop bands with excessive noise or known atmospheric 
       absorption features (even on the Moon, detector response varies).
    2. Spectral subsetting: Select bands in the 0.8-2.5 µm range that are most 
       correlated with visible-range albedo (longer wavelength thermal bands 
       carry mineralogical info but hurt visible-to-IR matching).
    3. PCA: Reduce selected bands to top-N principal components.
    4. Proxy image construction: Use PC1 (highest variance) as the primary 
       2D proxy, optionally weighted with PC2/PC3 for a false-color composite.
    
    The proxy image is what gets passed to Module 2 for TMC-2 <-> IIRS matching.
    """
    
    def __init__(self, n_components: int = 3, 
                 band_range_um: Tuple[float, float] = (0.8, 2.5),
                 noise_threshold: float = 0.05,
                 proxy_mode: str = 'pc1'):
        """
        Args:
            n_components: Number of PCA components to extract.
            band_range_um: Wavelength range to use for PCA (0.8-2.5 µm default,
                          excluding thermal bands >2.5 µm that hurt cross-modal matching).
            noise_threshold: Bands with variance below this fraction of median 
                            band variance are considered noisy and dropped.
            proxy_mode: How to construct the 2D proxy image:
                - 'pc1': Use only the first principal component (best for matching)
                - 'weighted': Weighted combination of top-3 PCs by explained variance
                - 'albedo_corr': Select bands most correlated with mean reflectance
        """
        super().__init__()
        self.n_components = n_components
        self.band_range_um = band_range_um
        self.noise_threshold = noise_threshold
        self.proxy_mode = proxy_mode
        self._loadings: Optional[np.ndarray] = None

    def preprocess(self, cube: np.ndarray, band_centers_um: Optional[np.ndarray] = None, 
                   **kwargs) -> PreprocessingResult:
        """
        Full IIRS preprocessing pipeline.
        
        Args:
            cube: 3D array [H, W, num_bands] — the raw hyperspectral cube.
            band_centers_um: 1D array of band center wavelengths in µm.
                If None, assume linear spacing 0.8-5.0 µm across the band axis.
        
        Returns:
            PreprocessingResult with the 2D proxy image.
        """
        # 1. Validate cube dimensions
        if not isinstance(cube, np.ndarray) or cube.ndim != 3:
            logger.error("Input cube must be a 3D numpy array.")
            return PreprocessingResult(
                image=np.zeros((1, 1), dtype=np.float32),
                instrument="IIRS",
                original_shape=cube.shape if isinstance(cube, np.ndarray) else (0,),
                preprocessed_shape=(1, 1),
                processing_steps=["error: invalid input shape"],
                quality_metrics={"error": 1.0},
                confidence=0.0,
            )
        
        H, W, num_bands = cube.shape
        
        # 2. Generate band_centers if not provided
        if band_centers_um is None:
            band_centers_um = np.linspace(0.8, 5.0, num_bands)
        
        if len(band_centers_um) != num_bands:
            logger.error("Band centers length must match the number of bands.")
            return PreprocessingResult(
                image=np.zeros((H, W), dtype=np.float32),
                instrument="IIRS", original_shape=cube.shape,
                preprocessed_shape=(H, W), processing_steps=["error: band mismatch"],
                quality_metrics={"error": 1.0}, confidence=0.0,
            )

        # 3. Spectral subsetting
        cube_sub, bands_sub = self._select_band_range(cube, band_centers_um)
        
        if cube_sub.shape[-1] == 0:
            logger.error("No bands left after spectral subsetting.")
            return PreprocessingResult(
                image=np.zeros((H, W), dtype=np.float32),
                instrument="IIRS", original_shape=cube.shape,
                preprocessed_shape=(H, W), processing_steps=["error: no bands in range"],
                quality_metrics={"error": 1.0}, confidence=0.0,
            )

        # 4. Bad-band detection
        cube_clean, bands_clean = self._remove_bad_bands(cube_sub, bands_sub)
        
        if cube_clean.shape[-1] == 0:
            logger.error("No bands left after bad band removal.")
            return PreprocessingResult(
                image=np.zeros((H, W), dtype=np.float32),
                instrument="IIRS", original_shape=cube.shape,
                preprocessed_shape=(H, W), processing_steps=["error: no bands after clean"],
                quality_metrics={"error": 1.0}, confidence=0.0,
            )

        # 5. Spatial-spectral denoising (light Gaussian smoothing per band)
        cube_denoised = np.empty_like(cube_clean)
        for b in range(cube_clean.shape[-1]):
            cube_denoised[..., b] = ndimage.gaussian_filter(cube_clean[..., b], sigma=0.5)

        # 6. Reshape cube to 2D matrix [n_pixels, n_bands]
        n_pixels = H * W
        n_bands = cube_denoised.shape[-1]
        data_2d = cube_denoised.reshape((n_pixels, n_bands))

        # 7. Standardize (zero mean, unit variance per band)
        means = np.mean(data_2d, axis=0)
        stds = np.std(data_2d, axis=0)
        stds[stds == 0] = 1.0 # avoid division by zero
        data_std = (data_2d - means) / stds

        # 8. PCA via manual eigendecomposition
        components, eigenvalues, eigenvectors = self._compute_pca(data_std)
        self._loadings = eigenvectors

        # Quality metrics
        total_variance = np.sum(eigenvalues) if np.sum(eigenvalues) > 0 else 1.0
        explained_variance_ratio = eigenvalues[:self.n_components] / total_variance

        # 9 & 10. Construct proxy image and normalize
        proxy_image = self._construct_proxy(components, eigenvalues, (H, W))
        
        # Normalize proxy to [0, 1]
        p_min, p_max = np.min(proxy_image), np.max(proxy_image)
        if p_max > p_min:
            proxy_image = (proxy_image - p_min) / (p_max - p_min)
        else:
            proxy_image = np.zeros_like(proxy_image)

        # 11. Metrics
        metrics = {
            "explained_variance_ratio": explained_variance_ratio.tolist(),
            "total_explained_variance": float(np.sum(explained_variance_ratio)),
            "retained_bands_count": n_bands
        }

        # The confidence can be tied to the explained variance of the first PC
        confidence = float(explained_variance_ratio[0]) if len(explained_variance_ratio) > 0 else 0.0

        return PreprocessingResult(
            image=proxy_image.astype(np.float32),
            instrument="IIRS",
            original_shape=cube.shape,
            preprocessed_shape=proxy_image.shape,
            processing_steps=["spectral_subsetting", "bad_band_removal", "denoising", "pca", "proxy_construction"],
            quality_metrics=metrics,
            confidence=confidence,
        )

    def _select_band_range(self, cube: np.ndarray, band_centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Keep only bands within the configured wavelength range."""
        valid_idx = np.where((band_centers >= self.band_range_um[0]) & (band_centers <= self.band_range_um[1]))[0]
        return cube[..., valid_idx], band_centers[valid_idx]

    def _remove_bad_bands(self, cube: np.ndarray, band_centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove bands with variance below threshold."""
        # Calculate variance per band
        variances = np.var(cube, axis=(0, 1))
        median_var = np.median(variances)
        threshold = self.noise_threshold * median_var
        
        valid_idx = np.where(variances >= threshold)[0]
        return cube[..., valid_idx], band_centers[valid_idx]

    def _compute_pca(self, data_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute PCA using eigendecomposition of covariance matrix.
        Use numpy.linalg.eigh for efficiency (covariance is symmetric).
        Return: (components [n_pixels, n_components], 
                 eigenvalues [n_components], 
                 eigenvectors [n_bands, n_components])
        Handle edge case: if n_pixels < n_bands, use the dual formulation.
        """
        n_pixels, n_bands = data_2d.shape
        
        if n_pixels >= n_bands:
            # Standard covariance matrix [n_bands, n_bands]
            cov = np.dot(data_2d.T, data_2d) / (n_pixels - 1)
            evals, evecs = np.linalg.eigh(cov)
        else:
            # Dual formulation [n_pixels, n_pixels]
            cov = np.dot(data_2d, data_2d.T) / (n_pixels - 1)
            evals, evecs_dual = np.linalg.eigh(cov)
            # Recover standard eigenvectors
            evecs = np.dot(data_2d.T, evecs_dual)
            # Normalize eigenvectors
            evecs = evecs / np.linalg.norm(evecs, axis=0)

        # eigh returns in ascending order, we want descending
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]
        
        # Keep only top n_components (or max available)
        n_comp = min(self.n_components, len(evals))
        evals = evals[:n_comp]
        evecs = evecs[:, :n_comp]
        
        # Project data onto principal components
        components = np.dot(data_2d, evecs)
        
        return components, evals, evecs

    def _construct_proxy(self, components: np.ndarray, eigenvalues: np.ndarray,
                          spatial_shape: Tuple[int, int]) -> np.ndarray:
        """Construct the 2D proxy image from PCA components.
        - pc1: reshape first component to spatial_shape, normalize to [0,1]
        - weighted: sum of top-3 PCs weighted by explained variance ratio
        - albedo_corr: use the component most correlated with mean band reflectance 
                       (Approximated here by PC1 if not explicitly computed)
        """
        H, W = spatial_shape
        n_comp = components.shape[1]
        
        if self.proxy_mode == 'weighted' and n_comp > 0:
            weight_limit = min(3, n_comp)
            weights = eigenvalues[:weight_limit] / (np.sum(eigenvalues[:weight_limit]) + 1e-8)
            proxy = np.zeros(components.shape[0])
            for i in range(weight_limit):
                proxy += components[:, i] * weights[i]
            return proxy.reshape((H, W))
            
        elif self.proxy_mode == 'albedo_corr' and n_comp > 0:
            # In a full implementation, we'd check correlation of each PC with mean albedo.
            # Here we default to the first component as a heuristic.
            return components[:, 0].reshape((H, W))
            
        else:
            # Default to pc1
            if n_comp == 0:
                return np.zeros((H, W))
            return components[:, 0].reshape((H, W))

    def get_band_importance(self) -> Optional[np.ndarray]:
        """Return the PCA loading weights for each band, useful for 
        interpretability (which spectral bands drive the proxy image)."""
        return self._loadings
