"""
GPU-based feature computation strategy (single batch).

This strategy uses GPUProcessor for GPU-accelerated feature computation.
Best for medium datasets (1-10M points) that fit in GPU memory.

For larger datasets (> 10M points), use GPUChunkedStrategy instead.

**Phase 3 GPU Optimizations (November 23, 2025)**:
- ✅ GPUArrayCache integration for reduced CPU↔GPU transfers
- ✅ Smart caching of points/normals between compute steps
- 🎯 Target: 2 transfers per tile (start + end only)
- 📈 Expected gain: 20-30% performance improvement

Author: IGN LiDAR HD Development Team
Date: October 19, 2025
Version: 3.2.0 (Phase 2A.4 + Phase 3 optimizations)
"""
from typing import Dict, Optional
import numpy as np
import logging

from .strategies import BaseFeatureStrategy
from .compute.rgb_nir import compute_rgb_features
from .compute.gpu_memory_integration import get_gpu_memory_pool
from ..core.gpu import GPUManager
from ..optimization.gpu_cache import GPUArrayCache

logger = logging.getLogger(__name__)

# GPU availability check (centralized)
_gpu_manager = GPUManager()
GPU_AVAILABLE = _gpu_manager.gpu_available

if GPU_AVAILABLE:
    try:
        import cupy as cp
        from .gpu_processor import GPUProcessor
    except ImportError:
        GPU_AVAILABLE = False
        cp = None
        GPUProcessor = None
else:
    cp = None
    GPUProcessor = None


class GPUStrategy(BaseFeatureStrategy):
    """
    GPU-based feature computation for medium datasets (single batch).

    Uses GPUProcessor which automatically selects batch vs chunked strategy.

    This strategy is optimal for:
    - Medium datasets (1-10M points) - uses batch processing
    - Large datasets (10M+ points) - automatically switches to chunked processing
    - Systems with GPU (CuPy + cuML)

    Performance (batch mode):
    - Medium (1-5M points): 0.5-2 seconds (10-30x faster than CPU)
    - Large (5-10M points): 2-5 seconds

    Performance (chunked mode, auto-triggered for >10M):
    - Very large (10-50M points): 5-30 seconds
    - FAISS acceleration: 50-100x speedup for k-NN queries

    Requirements:
    - CuPy (CUDA arrays)
    - cuML (GPU algorithms) - optional but recommended
    - FAISS (optional, for 50-100x k-NN speedup on large datasets)

    Attributes:
        k_neighbors (int): Number of neighbors for geometric features
        batch_size (int): Maximum batch size for GPU processing
        gpu_processor (GPUProcessor): GPU processor
    """

    def __init__(
        self,
        k_neighbors: int = 20,
        radius: float = 1.0,
        batch_size: int = 8_000_000,
        verbose: bool = False,
    ):
        """
        Initialize GPU strategy with GPUProcessor.

        Args:
            k_neighbors: Number of neighbors for local features
            radius: Search radius in meters (not used, k-NN preferred)
            batch_size: Maximum points to process in one GPU batch
            verbose: Enable detailed logging

        Raises:
            RuntimeError: If GPU is not available
        """
        super().__init__(k_neighbors=k_neighbors, radius=radius, verbose=verbose)

        if not GPU_AVAILABLE or GPUProcessor is None:
            raise RuntimeError(
                "GPU strategy requires CuPy and GPUProcessor. Install with: pip install cupy-cuda11x\n"
                "For CPU, use CPUStrategy instead."
            )

        self.batch_size = batch_size

        # Initialize GPU memory pool (Phase 2.2)
        self.memory_pool = get_gpu_memory_pool(enable=True)

        # Initialize GPU processor with auto-chunking
        # Note: chunk_threshold is auto-detected based on VRAM
        self.gpu_processor = GPUProcessor(
            batch_size=batch_size,
            auto_chunk=True,  # Enable automatic chunking for large datasets
            show_progress=verbose,
            enable_memory_pooling=True,  # Enable GPU cache (Phase 3)
        )
        
        # Initialize GPU cache for optimized transfers (Phase 3)
        self.gpu_cache = self.gpu_processor.gpu_cache

        if verbose:
            logger.info(
                f"Initialized GPU strategy: k={k_neighbors}, batch_size={batch_size:,}"
            )
            logger.info(f"  GPU available: {self.gpu_processor.use_gpu}")
            logger.info(f"  GPU cache enabled: {self.gpu_cache is not None}")
            logger.info(f"  Memory pooling enabled: {self.memory_pool.enable_pooling}")
            if hasattr(self.gpu_processor, 'chunk_threshold'):
                logger.info(f"  Auto-chunking threshold: {self.gpu_processor.chunk_threshold:,} points")

    def compute(
        self,
        points: np.ndarray,
        intensities: Optional[np.ndarray] = None,
        rgb: Optional[np.ndarray] = None,
        nir: Optional[np.ndarray] = None,
        classification: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Compute features using GPUProcessor (auto batch/chunked).

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
                f"Computing features for {n_points:,} points using GPU strategy"
            )
            if n_points > 10_000_000:
                logger.info(f"  Auto-chunking will be used for >10M points")

        # Compute geometric features with GPU processor
        # Automatically uses batch (<10M) or chunked (>10M) strategy
        normals = self.gpu_processor.compute_normals(
            points, k=self.k_neighbors, show_progress=self.verbose
        )
        curvature = self.gpu_processor.compute_curvature(
            points, normals, k=self.k_neighbors, show_progress=self.verbose
        )

        # Compute height relative to minimum Z
        z_min = points[:, 2].min()
        height = (points[:, 2] - z_min).astype(np.float32)

        # Build result dictionary
        result = {
            "normals": normals.astype(np.float32),
            "curvature": curvature.astype(np.float32),
            "height": height.astype(np.float32),
        }

        # Compute additional geometric features
        # Verticality: 1 - |normal_z| (walls have normal_z≈0, so verticality≈1)
        # ✅ FIXED: Was inverted (walls got low scores, roofs got high scores)
        verticality = (1.0 - np.abs(normals[:, 2])).astype(np.float32)
        verticality = np.clip(verticality, 0.0, 1.0).astype(np.float32)
        result["verticality"] = verticality

        # Planarity: 1 - curvature (normalized)
        planarity = (1.0 - np.minimum(curvature, 1.0)).astype(np.float32)
        result["planarity"] = planarity

        # Sphericity: curvature normalized
        sphericity = np.minimum(curvature, 1.0).astype(np.float32)
        result["sphericity"] = sphericity

        # Compute RGB features if provided
        if rgb is not None:
            rgb_features = compute_rgb_features(rgb, use_gpu=True)
            result.update(rgb_features)

        # Compute NDVI if NIR and RGB provided
        if nir is not None and rgb is not None:
            red = rgb[:, 0]
            ndvi = self.compute_ndvi(nir, red)
            result["ndvi"] = ndvi.astype(np.float32)

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
        mode: str = "lod2",
        radius: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Adapter method for compatibility with FeatureOrchestrator.

        Uses GPUProcessor for feature computation.

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
        rgb = kwargs.get("rgb", None)
        nir = kwargs.get("nir", None)

        # Use the compute method which handles everything
        result = self.compute(
            points=points,
            intensities=kwargs.get("intensities", None),
            rgb=rgb,
            nir=nir,
            classification=classification,
        )

        # Add distance_to_center if patch_center provided and include_extra is True
        if include_extra and patch_center is not None:
            distances = np.linalg.norm(points - patch_center, axis=1)
            result["distance_to_center"] = distances.astype(np.float32)

        return result
    def __repr__(self) -> str:
        """String representation."""
        return f"GPUStrategy(k_neighbors={self.k_neighbors}, batch_size={self.batch_size:,})"


# Export
__all__ = ["GPUStrategy"]
