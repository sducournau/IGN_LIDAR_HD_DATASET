"""
GPU-based feature computation strategy with chunking.

Now uses the unified GPUProcessor which automatically handles chunking
for large datasets (> 10M points).

This strategy uses the unified GPUProcessor with intelligent auto-chunking for
memory-efficient processing of massive point clouds.

Author: IGN LiDAR HD Development Team
Date: October 19, 2025
Version: 3.1.0-dev (Phase 2A.4 - GPU consolidation)
"""

from typing import Dict, Optional
import numpy as np
import logging

from .strategies import BaseFeatureStrategy

logger = logging.getLogger(__name__)

# Try to import GPU dependencies
try:
    import cupy as cp
    from .gpu_processor import GPUProcessor
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None
    GPUProcessor = None


class GPUChunkedStrategy(BaseFeatureStrategy):
    """
    GPU-based feature computation with auto-chunking for large datasets.
    
    Now uses the unified GPUProcessor which automatically handles chunking
    and selects the optimal strategy (batch vs chunked) based on dataset size.
    
    This strategy is optimal for:
    - Large datasets (> 10M points) - automatically uses chunked processing
    - Medium datasets (1-10M points) - automatically uses batch processing
    - Memory-efficient GPU processing
    - Production workloads with massive point clouds
    
    Performance:
    - Batch mode (<10M): 0.5-5 seconds (10-30× faster than CPU)
    - Chunked mode (>10M): Auto-selected, FAISS acceleration available
    - FAISS speedup: 50-100× for k-NN queries on massive datasets
    
    Requirements:
    - CuPy (CUDA arrays)
    - cuML (GPU algorithms) - optional but recommended
    - FAISS (optional, for 50-100× k-NN speedup)
    
    Attributes:
        k_neighbors (int): Number of neighbors for geometric features
        chunk_size (int): Points per chunk (default: 1M, handled by GPUProcessor)
        gpu_processor (GPUProcessor): Unified GPU processor with auto-chunking
    """
    
    def __init__(
        self,
        k_neighbors: int = 20,
        radius: float = 1.0,
        chunk_size: int = 1_000_000,  # Chunk size for GPUProcessor
        batch_size: int = 8_000_000,  # Batch size before auto-chunking
        neighbor_query_batch_size: Optional[int] = None,  # Deprecated, kept for compatibility
        feature_batch_size: Optional[int] = None,  # Deprecated, kept for compatibility
        verbose: bool = False
    ):
        """
        Initialize GPU chunked strategy with unified processor.
        
        Args:
            k_neighbors: Number of neighbors for local features
            radius: Search radius in meters (not used, k-NN preferred)
            chunk_size: Points per chunk for GPUProcessor (default: 1M)
            batch_size: Max batch size before auto-chunking (default: 8M)
            neighbor_query_batch_size: Deprecated (kept for backward compatibility)
            feature_batch_size: Deprecated (kept for backward compatibility)
            verbose: Enable detailed logging
            
        Raises:
            RuntimeError: If GPU is not available
        """
        super().__init__(k_neighbors=k_neighbors, radius=radius, verbose=verbose)
        
        if not GPU_AVAILABLE or GPUProcessor is None:
            raise RuntimeError(
                "GPU chunked strategy requires CuPy and GPUProcessor. Install with: pip install cupy-cuda11x\n"
                "For CPU, use CPUStrategy instead."
            )
        
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        
        # Initialize unified GPU processor with chunking parameters
        # Auto-chunks for datasets > 10M points
        self.gpu_processor = GPUProcessor(
            batch_size=batch_size,
            chunk_size=chunk_size,
            show_progress=verbose
        )
        
        if verbose:
            logger.info(
                f"Initialized GPU chunked strategy (unified processor): "
                f"k={k_neighbors}, chunk_size={chunk_size:,}, batch_size={batch_size:,}"
            )
            logger.info(f"  GPU available: {self.gpu_processor.use_gpu}")
            logger.info(f"  Auto-chunking: enabled")
            logger.info(f"  Batch strategy: used for <10M points")
            logger.info(f"  Chunked strategy: used for >10M points (auto-selected)")
    
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
        Compute features using unified GPUProcessor (auto batch/chunked).
        
        Automatically selects optimal strategy:
        - Batch mode: <10M points (fast, single GPU batch)
        - Chunked mode: >10M points (memory-efficient, FAISS acceleration)
        
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
            strategy_name = "chunked" if n_points > 10_000_000 else "batch"
            logger.info(
                f"Computing features for {n_points:,} points using GPU strategy "
                f"({strategy_name} mode auto-selected)"
            )
        
        # Compute geometric features with unified processor
        # Automatically uses batch (<10M) or chunked (>10M) strategy
        normals = self.gpu_processor.compute_normals(points, k=self.k_neighbors, show_progress=self.verbose)
        curvature = self.gpu_processor.compute_curvature(points, normals, k=self.k_neighbors, show_progress=self.verbose)
        
        # Compute height relative to minimum Z
        z_min = points[:, 2].min()
        height = (points[:, 2] - z_min).astype(np.float32)
        
        # Build result dictionary
        result = {
            'normals': normals.astype(np.float32),
            'curvature': curvature.astype(np.float32),
            'height': height.astype(np.float32),
        }
        
        # Compute additional geometric features
        # Verticality: dot product of normal with vertical axis
        verticality = np.abs(normals[:, 2]).astype(np.float32)
        result['verticality'] = verticality
        
        # Planarity: 1 - curvature (normalized)
        planarity = (1.0 - np.minimum(curvature, 1.0)).astype(np.float32)
        result['planarity'] = planarity
        
        # Sphericity: curvature normalized
        sphericity = np.minimum(curvature, 1.0).astype(np.float32)
        result['sphericity'] = sphericity
        
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
        
        Uses the unified GPUProcessor for feature computation.
        
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
        # Extract RGB and NIR from kwargs if present
        rgb = kwargs.get('rgb', None)
        nir = kwargs.get('nir', None)
        
        # Use the compute method which handles everything
        result = self.compute(
            points=points,
            intensities=kwargs.get('intensities', None),
            rgb=rgb,
            nir=nir,
            classification=classification
        )
        
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
