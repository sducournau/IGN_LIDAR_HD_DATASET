"""
GPU-based feature computation strategy (single batch).

This strategy uses CuPy and cuML for GPU-accelerated feature computation.
Best for medium datasets (1-10M points) that fit in GPU memory.

For larger datasets (> 10M points), use GPUChunkedStrategy instead.

Author: IGN LiDAR HD Development Team
Date: October 21, 2025
Version: 3.1.0-dev (Week 2 refactoring)
"""

from typing import Dict, Optional
import numpy as np
import logging

from .strategies import BaseFeatureStrategy

logger = logging.getLogger(__name__)

# Try to import GPU dependencies
try:
    import cupy as cp
    from .features_gpu import GPUFeatureComputer
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None
    GPUFeatureComputer = None


class GPUStrategy(BaseFeatureStrategy):
    """
    GPU-based feature computation for medium datasets (single batch).
    
    This strategy is optimal for:
    - Medium datasets (1-10M points)
    - Systems with GPU (CuPy + cuML)
    - When data fits in GPU memory
    
    For larger datasets (> 10M points), use GPUChunkedStrategy instead.
    
    Performance:
    - Medium (1-5M points): 5-15 seconds (10-30× faster than CPU)
    - Large (5-10M points): 15-45 seconds
    
    Requirements:
    - CuPy (CUDA arrays)
    - cuML (GPU algorithms) - optional but recommended
    
    Attributes:
        k_neighbors (int): Number of neighbors for geometric features
        batch_size (int): Maximum batch size for GPU processing
    """
    
    def __init__(
        self,
        k_neighbors: int = 20,
        radius: float = 1.0,
        batch_size: int = 8_000_000,
        verbose: bool = False
    ):
        """
        Initialize GPU strategy.
        
        Args:
            k_neighbors: Number of neighbors for local features
            radius: Search radius in meters (not used in single-batch mode)
            batch_size: Maximum points to process in one GPU batch
            verbose: Enable detailed logging
            
        Raises:
            RuntimeError: If GPU is not available
        """
        super().__init__(k_neighbors=k_neighbors, radius=radius, verbose=verbose)
        
        if not GPU_AVAILABLE or GPUFeatureComputer is None:
            raise RuntimeError(
                "GPU strategy requires CuPy. Install with: pip install cupy-cuda11x\n"
                "For CPU, use CPUStrategy instead."
            )
        
        self.batch_size = batch_size
        self.gpu_computer = GPUFeatureComputer(use_gpu=True, batch_size=batch_size)
        
        if verbose:
            logger.info(f"Initialized GPU strategy: k={k_neighbors}, batch_size={batch_size:,}")
    
    def compute(
        self,
        points: np.ndarray,
        intensities: Optional[np.ndarray] = None,
        rgb: Optional[np.ndarray] = None,
        nir: Optional[np.ndarray] = None,
        classification: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Compute features using GPU (single batch).
        
        Args:
            points: (N, 3) array of XYZ coordinates
            intensities: (N,) array of intensity values (optional)
            rgb: (N, 3) array of RGB values (optional)
            nir: (N,) array of near-infrared values (optional)
            classification: (N,) array of ASPRS classification codes (optional)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with feature arrays (same keys as CPUStrategy)
        """
        n_points = len(points)
        
        if self.verbose:
            logger.info(f"Computing features for {n_points:,} points using GPU strategy")
        
        if n_points > 10_000_000:
            logger.warning(
                f"Dataset has {n_points:,} points (> 10M). "
                "Consider using GPUChunkedStrategy for better memory efficiency."
            )
        
        # Create dummy classification if not provided
        if classification is None:
            classification = np.zeros(n_points, dtype=np.uint8)
        
        # Use existing GPU feature computer
        # This wraps the existing features_gpu.py implementation
        normals, curvature, height, geo_features = self.gpu_computer.compute_all_features(
            points=points,
            classification=classification,
            k=self.k_neighbors,
            include_building_features=False,
            mode='lod2'  # Standard geometric features
        )
        
        # Build result dictionary
        result = {
            'normals': normals.astype(np.float32),
            'curvature': curvature.astype(np.float32),
            'height': height.astype(np.float32),
        }
        
        # Add geometric features
        for key, value in geo_features.items():
            result[key] = value.astype(np.float32)
        
        # Compute RGB features if provided
        if rgb is not None:
            rgb_features = self._compute_rgb_features_gpu(rgb)
            result.update(rgb_features)
        
        # Compute NDVI if NIR and RGB provided
        if nir is not None and rgb is not None:
            red = rgb[:, 0]
            ndvi = self.compute_ndvi(nir, red)
            result['ndvi'] = ndvi.astype(np.float32)
        
        if self.verbose:
            logger.info(f"GPU computation complete: {len(result)} feature types")
        
        return result
    
    def compute_features(
        self,
        points: np.ndarray,
        classification: Optional[np.ndarray] = None,
        auto_k: bool = False,
        include_extra: bool = False,
        patch_center: Optional[np.ndarray] = None,
        mode: str = 'lod2',
        radius: Optional[float] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Adapter method for compatibility with FeatureOrchestrator.
        
        This method provides the old interface expected by the orchestrator,
        while internally calling compute_all_features with the correct mode.
        
        Args:
            points: (N, 3) array of XYZ coordinates
            classification: (N,) array of ASPRS classification codes (optional)
            auto_k: Whether to auto-estimate k (ignored for GPU)
            include_extra: Whether to include extra features
            patch_center: Center point of the patch (for distance features)
            mode: Feature mode ('lod2', 'lod3', 'asprs_classes', etc.)
            radius: Search radius (ignored, uses k-NN)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with feature arrays
        """
        n_points = len(points)
        
        # Create dummy classification if not provided
        if classification is None:
            classification = np.zeros(n_points, dtype=np.uint8)
        
        # Extract RGB and NIR from kwargs if present
        rgb = kwargs.get('rgb', None)
        nir = kwargs.get('nir', None)
        
        # Call compute_all_features with the correct mode
        normals, curvature, height, geo_features = self.gpu_computer.compute_all_features(
            points=points,
            classification=classification,
            k=self.k_neighbors,
            include_building_features=include_extra,
            mode=mode  # Pass the mode from the orchestrator
        )
        
        # Build result dictionary
        result = {
            'normals': normals.astype(np.float32),
            'curvature': curvature.astype(np.float32),
            'height': height.astype(np.float32),
        }
        
        # Add geometric features
        for key, value in geo_features.items():
            result[key] = value.astype(np.float32)
        
        # Compute RGB features if provided and not already computed
        if rgb is not None and 'rgb_mean' not in result:
            rgb_features = self._compute_rgb_features_gpu(rgb)
            result.update(rgb_features)
        
        # Compute NDVI if NIR and RGB provided and not already computed
        if nir is not None and rgb is not None and 'ndvi' not in result:
            red = rgb[:, 0]
            ndvi = self.compute_ndvi(nir, red)
            result['ndvi'] = ndvi.astype(np.float32)
        
        # Add distance_to_center if patch_center provided and include_extra is True
        if include_extra and patch_center is not None:
            distances = np.linalg.norm(points - patch_center, axis=1)
            result['distance_to_center'] = distances.astype(np.float32)
        
        return result
    
    def _compute_rgb_features_gpu(self, rgb: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute RGB-based features using GPU.
        
        Args:
            rgb: (N, 3) array of RGB values [0-255]
            
        Returns:
            Dictionary with RGB features
        """
        # Transfer to GPU
        rgb_gpu = cp.asarray(rgb, dtype=cp.float32) / 255.0
        
        # Basic RGB statistics
        rgb_mean = cp.mean(rgb_gpu, axis=1)
        rgb_std = cp.std(rgb_gpu, axis=1)
        rgb_range = cp.max(rgb_gpu, axis=1) - cp.min(rgb_gpu, axis=1)
        
        # Color indices
        r, g, b = rgb_gpu[:, 0], rgb_gpu[:, 1], rgb_gpu[:, 2]
        
        # Excess Green Index
        exg = 2 * g - r - b
        
        # Vegetation index
        vegetation_index = (g - r) / (g + r + 1e-8)
        
        # Transfer back to CPU
        return {
            'rgb_mean': cp.asnumpy(rgb_mean).astype(np.float32),
            'rgb_std': cp.asnumpy(rgb_std).astype(np.float32),
            'rgb_range': cp.asnumpy(rgb_range).astype(np.float32),
            'excess_green': cp.asnumpy(exg).astype(np.float32),
            'vegetation_index': cp.asnumpy(vegetation_index).astype(np.float32),
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return f"GPUStrategy(k_neighbors={self.k_neighbors}, batch_size={self.batch_size:,})"


# Export
__all__ = ['GPUStrategy']
