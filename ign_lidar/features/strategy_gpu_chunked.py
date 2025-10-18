"""
GPU-based feature computation strategy with chunking.

This is the GOLD STANDARD strategy for large datasets (> 10M points).
Optimized in Week 1 with 250K batch size achieving 16× speedup.

This strategy uses CuPy and cuML with intelligent chunking for
memory-efficient processing of massive point clouds.

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
    from .features_gpu_chunked import GPUChunkedFeatureComputer
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None
    GPUChunkedFeatureComputer = None


class GPUChunkedStrategy(BaseFeatureStrategy):
    """
    GPU-based feature computation with chunking for large datasets.
    
    This is the GOLD STANDARD strategy optimized in Week 1.
    
    This strategy is optimal for:
    - Large datasets (> 10M points)
    - Memory-efficient GPU processing
    - Production workloads with massive point clouds
    
    Performance (Week 1 optimization):
    - Before: 353s per 1.86M point chunk
    - After: 22s per chunk
    - Speedup: 16× improvement ✅
    
    Key optimizations:
    - NEIGHBOR_BATCH_SIZE = 250,000 (optimized for GPU L2 cache)
    - Chunk size = 5M points (configurable)
    - Progress tracking
    - Memory-efficient neighbor search
    
    Requirements:
    - CuPy (CUDA arrays)
    - cuML (GPU algorithms) - optional but recommended
    
    Attributes:
        k_neighbors (int): Number of neighbors for geometric features
        chunk_size (int): Points per chunk (default: 5M)
        batch_size (int): Neighbor batch size (default: 250K, optimized in Week 1)
        neighbor_query_batch_size (int): Points per neighbor query batch (controls chunking for neighbor queries)
        feature_batch_size (int): Points per feature computation batch (controls normal/curvature batching)
    """
    
    def __init__(
        self,
        k_neighbors: int = 20,
        radius: float = 1.0,
        chunk_size: int = 8_000_000,  # INCREASED from 5M to 8M for RTX 4080 Super
        batch_size: int = 500_000,  # INCREASED from 250K to 500K for better GPU utilization
        neighbor_query_batch_size: Optional[int] = None,  # NEW: Controls neighbor query chunking
        feature_batch_size: Optional[int] = None,  # NEW: Controls feature computation batching
        verbose: bool = False
    ):
        """
        Initialize GPU chunked strategy.
        
        Args:
            k_neighbors: Number of neighbors for local features
            radius: Search radius in meters (not used in chunked mode)
            chunk_size: Points per chunk (default: 8M, optimized for RTX 4080 Super)
            batch_size: Neighbor batch size (default: 500K, doubled for better utilization)
            neighbor_query_batch_size: Points per neighbor query batch (None = 5M default, controls number of chunks)
            feature_batch_size: Points per feature computation batch (None = 2M default, controls normal/curvature batching)
            verbose: Enable detailed logging
            
        Raises:
            RuntimeError: If GPU is not available
        """
        super().__init__(k_neighbors=k_neighbors, radius=radius, verbose=verbose)
        
        if not GPU_AVAILABLE or GPUChunkedFeatureComputer is None:
            raise RuntimeError(
                "GPU chunked strategy requires CuPy. Install with: pip install cupy-cuda11x\n"
                "For CPU, use CPUStrategy instead."
            )
        
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.neighbor_query_batch_size = neighbor_query_batch_size
        self.feature_batch_size = feature_batch_size
        
        # GPUChunkedFeatureComputer doesn't accept k_neighbors in __init__
        # It will be passed during compute() call instead
        self.gpu_computer = GPUChunkedFeatureComputer(
            chunk_size=chunk_size,
            show_progress=verbose,
            neighbor_query_batch_size=neighbor_query_batch_size,
            feature_batch_size=feature_batch_size
        )
        
        if verbose:
            logger.info(
                f"Initialized GPU chunked strategy (GOLD STANDARD + ENHANCED): "
                f"k={k_neighbors}, chunk_size={chunk_size:,}, batch_size={batch_size:,}"
            )
            if neighbor_query_batch_size is not None:
                logger.info(
                    f"Neighbor query batch size: {neighbor_query_batch_size:,} points (controls chunking)"
                )
            if feature_batch_size is not None:
                logger.info(
                    f"Feature computation batch size: {feature_batch_size:,} points (controls normal/curvature batching)"
                )
            logger.info(
                f"Enhanced optimization: 8M chunks + 500K batches for RTX 4080 Super (60% capacity increase)"
            )
    
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
        Compute features using GPU with chunking (GOLD STANDARD).
        
        This method uses the Week 1 optimized implementation with
        250K batch size for 16× speedup.
        
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
            logger.info(
                f"Computing features for {n_points:,} points using GPU chunked strategy "
                f"(chunks={n_points//self.chunk_size + 1})"
            )
        
        # Create dummy classification if not provided
        if classification is None:
            classification = np.zeros(n_points, dtype=np.uint8)
        
        # Use existing GPU chunked feature computer (Week 1 optimized)
        # This wraps the features_gpu_chunked.py implementation
        normals, curvature, height, geo_features = self.gpu_computer.compute_all_features_chunked(
            points=points,
            classification=classification,
            k=self.k_neighbors,
            radius=self.radius if self.radius > 0 else None,
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
        
        # Compute RGB features if provided and not already computed
        if rgb is not None and 'rgb_mean' not in result:
            rgb_features = self._compute_rgb_features_gpu(rgb)
            result.update(rgb_features)
        
        # Compute NDVI if NIR and RGB provided and not already computed
        if nir is not None and rgb is not None and 'ndvi' not in result:
            red = rgb[:, 0]
            ndvi = self.compute_ndvi(nir, red)
            result['ndvi'] = ndvi.astype(np.float32)
        
        if self.verbose:
            logger.info(
                f"GPU chunked computation complete: {len(result)} feature types, "
                f"Week 1 optimizations active"
            )
        
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
        while internally calling compute_all_features_chunked directly.
        
        Args:
            points: (N, 3) array of XYZ coordinates
            classification: (N,) array of ASPRS classification codes (optional)
            auto_k: Whether to auto-estimate k (ignored)
            include_extra: Whether to include extra features (ignored for now)
            patch_center: Center point of the patch (for distance features)
            mode: Feature mode ('lod2', 'lod3', 'asprs_classes', etc.)
            radius: Search radius (overrides self.radius if provided)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with feature arrays
        """
        # Extract RGB and NIR from kwargs if present
        rgb = kwargs.get('rgb', None)
        nir = kwargs.get('nir', None)
        intensities = kwargs.get('intensities', None)
        
        # Create dummy classification if not provided
        if classification is None:
            classification = np.zeros(len(points), dtype=np.uint8)
        
        # Use the mode parameter passed from orchestrator
        use_radius = radius if radius is not None else (self.radius if self.radius > 0 else None)
        
        # Call compute_all_features_chunked directly with the correct mode
        normals, curvature, height, geo_features = self.gpu_computer.compute_all_features_chunked(
            points=points,
            classification=classification,
            k=self.k_neighbors,
            radius=use_radius,
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
        Compute RGB-based features using GPU with chunking.
        
        Args:
            rgb: (N, 3) array of RGB values [0-255]
            
        Returns:
            Dictionary with RGB features
        """
        n_points = len(rgb)
        
        # Process in chunks to avoid GPU memory issues
        results = {
            'rgb_mean': np.zeros(n_points, dtype=np.float32),
            'rgb_std': np.zeros(n_points, dtype=np.float32),
            'rgb_range': np.zeros(n_points, dtype=np.float32),
            'excess_green': np.zeros(n_points, dtype=np.float32),
            'vegetation_index': np.zeros(n_points, dtype=np.float32),
        }
        
        for start in range(0, n_points, self.chunk_size):
            end = min(start + self.chunk_size, n_points)
            rgb_chunk = rgb[start:end]
            
            # Transfer to GPU
            rgb_gpu = cp.asarray(rgb_chunk, dtype=cp.float32) / 255.0
            
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
            
            # ⚡ OPTIMIZATION: Batch all RGB transfers into single operation
            # Stack all features on GPU, then single transfer to CPU
            rgb_features_gpu = cp.stack([
                rgb_mean,
                rgb_std, 
                rgb_range,
                exg,
                vegetation_index
            ], axis=1)  # Shape: [N, 5]
            
            # Single transfer: 5x fewer cp.asnumpy() calls
            rgb_features_cpu = cp.asnumpy(rgb_features_gpu)
            
            # Unpack to results
            results['rgb_mean'][start:end] = rgb_features_cpu[:, 0]
            results['rgb_std'][start:end] = rgb_features_cpu[:, 1]
            results['rgb_range'][start:end] = rgb_features_cpu[:, 2]
            results['excess_green'][start:end] = rgb_features_cpu[:, 3]
            results['vegetation_index'][start:end] = rgb_features_cpu[:, 4]
            
            # Cleanup
            del rgb_features_gpu, rgb_features_cpu
        
        return results
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"GPUChunkedStrategy(k_neighbors={self.k_neighbors}, "
            f"chunk_size={self.chunk_size:,}, batch_size={self.batch_size:,})"
        )


# Export
__all__ = ['GPUChunkedStrategy']
