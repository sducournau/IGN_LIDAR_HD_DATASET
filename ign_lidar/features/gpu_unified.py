"""
Unified GPU Feature Computation Module

This module consolidates GPU feature computation functionality that was previously
scattered across features_gpu.py and features_gpu_chunked.py, removing duplication
and providing a cleaner, more maintainable interface.

Key improvements:
- Single GPU feature computer class with unified chunked processing
- Simplified mode selection (auto, chunked, non-chunked)
- Consistent parameter naming
- Better memory management
- Removed "enhanced" terminology in favor of descriptive method names
"""

from typing import Dict, Tuple, Optional, Union
import numpy as np
import logging
import gc
from tqdm import tqdm

logger = logging.getLogger(__name__)

# GPU imports with fallback
GPU_AVAILABLE = False
CUML_AVAILABLE = False

try:
    import cupy as cp
    from cupyx.scipy.spatial import distance as cp_distance
    GPU_AVAILABLE = True
    logger.info("✓ CuPy available - GPU enabled")
except ImportError:
    logger.warning("⚠ CuPy not available - GPU disabled")
    cp = None

try:
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    from cuml.decomposition import PCA as cuPCA
    CUML_AVAILABLE = True
    logger.info("✓ RAPIDS cuML available - GPU algorithms enabled")
except ImportError:
    logger.warning("⚠ RAPIDS cuML not available - using sklearn fallback")
    cuNearestNeighbors = None
    cuPCA = None

# CPU fallback imports
from sklearn.neighbors import KDTree

# Import core feature implementations
from ..features.core import (
    compute_normals as core_compute_normals,
    compute_curvature as core_compute_curvature,
    compute_eigenvalue_features as core_compute_eigenvalue_features,
    compute_density_features as core_compute_density_features,
    compute_verticality as core_compute_verticality,
)


class GPUFeatureComputer:
    """
    Unified GPU feature computer with automatic chunking and fallback support.
    
    This class consolidates all GPU feature computation functionality, automatically
    choosing between chunked and non-chunked processing based on data size and
    available GPU memory.
    
    Features:
    - Automatic chunking for large datasets
    - GPU memory management and monitoring
    - Fallback to CPU when GPU unavailable
    - Unified interface for all GPU feature computation
    
    Args:
        use_gpu: Enable GPU acceleration if available
        chunk_size: Target chunk size for large datasets (auto if None)
        vram_limit_gb: VRAM limit in GB for auto chunk sizing
        show_progress: Show progress bars during computation
        auto_optimize: Enable automatic optimization and memory management
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        chunk_size: Optional[int] = None,
        vram_limit_gb: float = 8.0,
        show_progress: bool = True,
        auto_optimize: bool = True
    ):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.use_cuml = self.use_gpu and CUML_AVAILABLE
        self.show_progress = show_progress
        self.auto_optimize = auto_optimize
        self.vram_limit_gb = vram_limit_gb
        
        # Auto-determine chunk size based on VRAM if not specified
        if chunk_size is None:
            self.chunk_size = self._estimate_optimal_chunk_size()
        else:
            self.chunk_size = chunk_size
        
        # Initialize CUDA context safely
        if self.use_gpu:
            self._initialize_cuda_context()
        
        self._log_initialization()
    
    def _initialize_cuda_context(self) -> bool:
        """Initialize CUDA context safely for multiprocessing."""
        if not self.use_gpu or cp is None:
            return False
            
        try:
            # Force CUDA context initialization
            _ = cp.cuda.Device()
            # Test basic operation
            test_array = cp.array([1.0], dtype=cp.float32)
            _ = cp.asnumpy(test_array)
            return True
        except Exception as e:
            logger.warning(f"CUDA initialization failed: {e}")
            logger.warning("Falling back to CPU mode")
            self.use_gpu = False
            self.use_cuml = False
            return False
    
    def _estimate_optimal_chunk_size(self) -> int:
        """Estimate optimal chunk size based on available GPU memory."""
        if not self.use_gpu:
            return 1_000_000  # Default for CPU fallback
        
        try:
            # Get GPU memory info
            mempool = cp.get_default_memory_pool()
            free_bytes = mempool.free_bytes()
            
            # Conservative estimate: use 60% of free memory
            usable_bytes = int(free_bytes * 0.6)
            
            # Estimate memory per point (coordinates + features + neighbors)
            # Roughly 200 bytes per point for all computations
            bytes_per_point = 200
            
            chunk_size = max(100_000, min(5_000_000, usable_bytes // bytes_per_point))
            
            logger.info(f"Auto-estimated chunk size: {chunk_size:,} points "
                       f"(based on {usable_bytes/1024**3:.1f}GB available GPU memory)")
            
            return chunk_size
            
        except Exception as e:
            logger.warning(f"Could not estimate GPU memory, using default chunk size: {e}")
            return 2_500_000  # Conservative default
    
    def _log_initialization(self):
        """Log initialization details."""
        if self.use_gpu:
            mode = "GPU with cuML" if self.use_cuml else "GPU with sklearn fallback"
            logger.info(f"🚀 {mode} (chunk_size={self.chunk_size:,})")
        else:
            logger.info("💻 CPU mode (install CuPy for GPU acceleration)")
    
    def _to_gpu(self, array: np.ndarray) -> Union[np.ndarray, 'cp.ndarray']:
        """Transfer array to GPU memory."""
        if self.use_gpu and cp is not None:
            return cp.asarray(array, dtype=cp.float32)
        return array
    
    def _to_cpu(self, array) -> np.ndarray:
        """Transfer array to CPU memory."""
        if self.use_gpu and cp is not None and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return np.asarray(array)
    
    def _free_gpu_memory(self):
        """Explicitly free GPU memory."""
        if self.use_gpu and cp is not None:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            gc.collect()
    
    def _should_use_chunking(self, num_points: int) -> bool:
        """Determine if chunking should be used based on data size."""
        # Always use chunking for very large datasets
        if num_points > 10_000_000:
            return True
        
        # Use chunking if estimated memory usage exceeds VRAM limit
        if self.use_gpu:
            estimated_memory_gb = (num_points * 200) / (1024**3)  # Rough estimate
            return estimated_memory_gb > self.vram_limit_gb * 0.8
        
        # For CPU, use chunking for large datasets to manage memory
        return num_points > 5_000_000
    
    def compute_features(
        self,
        points: np.ndarray,
        classification: np.ndarray,
        k: int = 10,
        radius: Optional[float] = None,
        mode: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute all features using the most appropriate method.
        
        Automatically chooses between chunked and non-chunked processing
        based on data size and available resources.
        
        Args:
            points: [N, 3] point coordinates
            classification: [N] ASPRS classification codes
            k: number of neighbors for KNN
            radius: search radius for geometric features (optional)
            mode: processing mode ('chunked', 'non-chunked', or None for auto)
            
        Returns:
            normals: [N, 3] surface normals
            curvature: [N] curvature values  
            height: [N] height above ground
            geo_features: dict with geometric features
        """
        N = len(points)
        
        # Determine processing mode
        if mode is None:
            use_chunking = self._should_use_chunking(N)
        elif mode == 'chunked':
            use_chunking = True
        elif mode == 'non-chunked':
            use_chunking = False
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'chunked', 'non-chunked', or None")
        
        logger.info(f"Processing {N:,} points using "
                   f"{'chunked' if use_chunking else 'non-chunked'} mode")
        
        if use_chunking:
            return self._compute_features_chunked(points, classification, k, radius)
        else:
            return self._compute_features_batch(points, classification, k, radius)
    
    def _compute_features_chunked(
        self,
        points: np.ndarray,
        classification: np.ndarray, 
        k: int,
        radius: Optional[float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Compute features using chunked processing for memory efficiency."""
        # Import the specific chunked implementation
        from .features_gpu_chunked import GPUChunkedFeatureComputer
        
        chunked_computer = GPUChunkedFeatureComputer(
            chunk_size=self.chunk_size,
            use_gpu=self.use_gpu,
            show_progress=self.show_progress,
            auto_optimize=self.auto_optimize
        )
        
        return chunked_computer.compute_all_features_chunked(
            points, classification, k=k, radius=radius
        )
    
    def _compute_features_batch(
        self,
        points: np.ndarray,
        classification: np.ndarray,
        k: int, 
        radius: Optional[float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Compute features using batch processing (non-chunked)."""
        # Import the specific non-chunked implementation
        from .features_gpu import GPUFeatureComputer as OriginalGPUComputer
        
        gpu_computer = OriginalGPUComputer(
            use_gpu=self.use_gpu,
            batch_size=min(self.chunk_size, len(points))
        )
        
        # The original compute_all_features doesn't support radius parameter
        # For now, we ignore the radius parameter for non-chunked processing
        # This maintains backward compatibility
        return gpu_computer.compute_all_features(
            points, classification, k=k
        )


def create_gpu_computer(
    use_gpu: bool = True,
    chunk_size: Optional[int] = None,
    **kwargs
) -> GPUFeatureComputer:
    """
    Factory function to create a GPU feature computer.
    
    This replaces the complex factory pattern with a simple function that
    returns a unified GPU computer.
    """
    return GPUFeatureComputer(
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        **kwargs
    )


# Convenience functions for backward compatibility
def compute_features_gpu(
    points: np.ndarray,
    classification: np.ndarray,
    k: int = 10,
    use_chunking: Optional[bool] = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Convenience function for GPU feature computation.
    
    Args:
        points: Point cloud coordinates
        classification: Point classifications
        k: Number of neighbors
        use_chunking: Force chunking mode (None for auto)
        **kwargs: Additional parameters
    """
    computer = GPUFeatureComputer(**kwargs)
    
    mode = None
    if use_chunking is True:
        mode = 'chunked'
    elif use_chunking is False:
        mode = 'non-chunked'
    
    return computer.compute_features(points, classification, k=k, mode=mode)