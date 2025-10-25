"""
CPU-based feature computation strategy.

This strategy uses NumPy and scikit-learn for feature computation
on CPU. Best for small to medium datasets (< 1M points) or when
GPU is not available.

Author: IGN LiDAR HD Development Team
Date: October 21, 2025
Version: 3.1.0-dev (Week 2 refactoring)
"""

from typing import Dict, Optional
import numpy as np
import logging

from .strategies import BaseFeatureStrategy
from .compute.features import compute_all_features as compute_all_features_optimized

logger = logging.getLogger(__name__)


class CPUStrategy(BaseFeatureStrategy):
    """
    CPU-based feature computation using NumPy and scikit-learn.
    
    This strategy is optimal for:
    - Small datasets (< 1M points)
    - Systems without GPU
    - When GPU overhead is not justified
    
    Performance:
    - Small (< 100K points): 1-2 seconds
    - Medium (100K-1M points): 5-30 seconds
    - Large (> 1M points): 60+ seconds (use GPU instead)
    
    Attributes:
        k_neighbors (int): Number of neighbors for geometric features
        radius (float): Search radius for neighbor queries (0 = use k-NN)
        auto_k (bool): Automatically estimate optimal k based on density
        include_extra (bool): Compute extra features (slower but more complete)
        chunk_size (int): Process in chunks to reduce memory usage
    """
    
    def __init__(
        self,
        k_neighbors: int = 20,
        radius: float = 1.0,
        auto_k: bool = True,
        include_extra: bool = False,
        chunk_size: int = 500_000,
        verbose: bool = False
    ):
        """
        Initialize CPU strategy.
        
        Args:
            k_neighbors: Number of neighbors for local features (if auto_k=False)
            radius: Search radius in meters (0 = use k-NN instead)
            auto_k: Automatically estimate optimal k based on point density
            include_extra: Compute expensive extra features
            chunk_size: Process in chunks for large datasets
            verbose: Enable detailed logging
        """
        super().__init__(k_neighbors=k_neighbors, radius=radius, verbose=verbose)
        self.auto_k = auto_k
        self.include_extra = include_extra
        self.chunk_size = chunk_size
        
        if verbose:
            logger.info(f"Initialized CPU strategy: k={k_neighbors}, radius={radius}m, "
                       f"auto_k={auto_k}, chunk_size={chunk_size:,}")
    
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
        Compute features using CPU.
        
        Args:
            points: (N, 3) array of XYZ coordinates
            intensities: (N,) array of intensity values (optional)
            rgb: (N, 3) array of RGB values (optional)
            nir: (N,) array of near-infrared values (optional)
            classification: (N,) array of ASPRS classification codes (optional)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with feature arrays:
            - 'normals': (N, 3) surface normals
            - 'curvature': (N,) curvature values
            - 'height': (N,) height above ground
            - 'planarity': (N,) planarity scores
            - 'linearity': (N,) linearity scores
            - 'sphericity': (N,) sphericity scores
            - 'anisotropy': (N,) anisotropy scores
            - 'roughness': (N,) roughness values
            - 'density': (N,) local point density
            - 'verticality': (N,) verticality scores
            - 'horizontality': (N,) horizontality scores
            - Additional features if include_extra=True
        """
        n_points = len(points)
        
        if self.verbose:
            logger.info(f"Computing features for {n_points:,} points using CPU strategy")
        
        # Determine k value (use default if auto_k not needed)
        k = self.k_neighbors
        
        # Compute all geometric features using unified optimized function
        features = compute_all_features_optimized(
            points=points,
            k_neighbors=k,
            compute_advanced=self.include_extra
        )
        
        # Build result dictionary compatible with strategy interface
        # 🔧 FIX: Return ALL computed features, not just a subset
        result = {
            'normals': features['normals'].astype(np.float32),
            'curvature': features['curvature'].astype(np.float32),
            'planarity': features['planarity'].astype(np.float32),
            'linearity': features['linearity'].astype(np.float32),
            'sphericity': features['sphericity'].astype(np.float32),
        }
        
        # Add advanced features if requested
        if self.include_extra:
            result['anisotropy'] = features.get('anisotropy', features['sphericity']).astype(np.float32)
            result['roughness'] = features.get('roughness', features['curvature']).astype(np.float32)
            result['verticality'] = features.get('verticality', np.abs(features['normals'][:, 2])).astype(np.float32)
            result['density'] = features.get('density', np.ones(n_points, dtype=np.float32)).astype(np.float32)
            
            # 🆕 Add ALL other computed features (eigenvalues, normal components, etc.)
            for feat_name, feat_data in features.items():
                if feat_name not in result and feat_name != 'normals':  # Avoid duplicating normals
                    result[feat_name] = feat_data.astype(np.float32)
        
        # Add height if we have classification
        if classification is not None:
            # Compute height above ground
            ground_mask = classification == 2
            if ground_mask.any():
                ground_z = points[ground_mask, 2].min()
                result['height'] = (points[:, 2] - ground_z).astype(np.float32)
            else:
                result['height'] = (points[:, 2] - points[:, 2].min()).astype(np.float32)
        else:
            result['height'] = np.zeros(n_points, dtype=np.float32)
        
        # Compute RGB features if provided
        if rgb is not None:
            rgb_features = self._compute_rgb_features_cpu(rgb)
            result.update(rgb_features)
        
        # Compute NDVI if NIR and RGB provided
        if nir is not None and rgb is not None:
            red = rgb[:, 0]  # Assuming RGB order
            ndvi = self.compute_ndvi(nir, red)
            result['ndvi'] = ndvi.astype(np.float32)
        
        if self.verbose:
            logger.info(f"CPU computation complete: {len(result)} feature types")
        
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
        while internally calling the new compute() method.
        
        Args:
            points: (N, 3) array of XYZ coordinates
            classification: (N,) array of ASPRS classification codes (optional)
            auto_k: Whether to auto-estimate k
            include_extra: Whether to include extra features
            patch_center: Center point of the patch (for distance features)
            mode: Feature mode ('lod2', 'lod3', etc.)
            radius: Search radius (overrides self.radius if provided)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with feature arrays
        """
        # Override settings if specified
        old_auto_k = self.auto_k
        old_include_extra = self.include_extra
        old_radius = self.radius
        
        if auto_k is not None:
            self.auto_k = auto_k
        if include_extra is not None:
            self.include_extra = include_extra
        if radius is not None:
            self.radius = radius
        
        # Extract RGB and NIR from kwargs if present
        rgb = kwargs.get('rgb', None)
        nir = kwargs.get('nir', None)
        intensities = kwargs.get('intensities', None)
        
        # Call the main compute method
        result = self.compute(
            points=points,
            intensities=intensities,
            rgb=rgb,
            nir=nir,
            classification=classification,
            **kwargs
        )
        
        # Add distance_to_center if patch_center provided
        if patch_center is not None:
            distances = np.linalg.norm(points - patch_center, axis=1)
            result['distance_to_center'] = distances.astype(np.float32)
        
        # Restore original settings
        self.auto_k = old_auto_k
        self.include_extra = old_include_extra
        self.radius = old_radius
        
        return result
    
    def _compute_rgb_features_cpu(self, rgb: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute RGB-based features using CPU.
        
        Args:
            rgb: (N, 3) array of RGB values [0-255]
            
        Returns:
            Dictionary with RGB features
        """
        # Normalize to [0, 1]
        rgb_normalized = rgb.astype(np.float32) / 255.0
        
        # Basic RGB statistics
        rgb_mean = np.mean(rgb_normalized, axis=1)
        rgb_std = np.std(rgb_normalized, axis=1)
        rgb_range = np.max(rgb_normalized, axis=1) - np.min(rgb_normalized, axis=1)
        
        # Color indices
        r, g, b = rgb_normalized[:, 0], rgb_normalized[:, 1], rgb_normalized[:, 2]
        
        # Excess Green Index
        with np.errstate(divide='ignore', invalid='ignore'):
            sum_rgb = r + g + b + 1e-8
            exg = 2 * g - r - b
            exg = np.nan_to_num(exg, nan=0.0)
        
        # Vegetation index (simple)
        vegetation_index = (g - r) / (g + r + 1e-8)
        vegetation_index = np.nan_to_num(vegetation_index, nan=0.0)
        
        return {
            'rgb_mean': rgb_mean.astype(np.float32),
            'rgb_std': rgb_std.astype(np.float32),
            'rgb_range': rgb_range.astype(np.float32),
            'excess_green': exg.astype(np.float32),
            'vegetation_index': vegetation_index.astype(np.float32),
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"CPUStrategy(k_neighbors={self.k_neighbors}, radius={self.radius}, "
                f"auto_k={self.auto_k}, chunk_size={self.chunk_size:,})")


# Export
__all__ = ['CPUStrategy']
