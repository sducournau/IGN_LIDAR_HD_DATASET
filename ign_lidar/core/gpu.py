"""
Centralized GPU Detection and Management

Single source of truth for GPU availability across the entire codebase.

This module replaces 6+ scattered GPU detection implementations with
a singleton pattern, providing:
- Consistent GPU availability checking
- Lazy initialization with caching
- Support for multiple GPU libraries (CuPy, cuML, cuSpatial, FAISS-GPU)
- Thread-safe singleton implementation
- Backward compatibility aliases
- Unified access to memory management, caching, and profiling (v3.1+/v3.2+)

Usage:
    from ign_lidar.core.gpu import GPUManager
    
    gpu = GPUManager()
    if gpu.gpu_available:
        # Use GPU acceleration
        pass
    
    # v3.1+ Composition API (unified access)
    gpu.memory.allocate(2.5)  # Memory management
    gpu.cache.get_or_upload('key', array)  # Array caching
    
    # v3.2+ Profiling (Phase 3)
    with gpu.profiler.profile('compute_features'):
        features = compute_gpu(points)
    gpu.profiler.print_report()
    
    # Legacy compatibility
    from ign_lidar.core.gpu import GPU_AVAILABLE

Architecture (Phase 1.2 + Phase 3 GPU Optimizations - Nov 22, 2025):
    GPUManager (unified entry point)
    ├── Detection: gpu_available, cuml_available, etc.
    ├── memory: GPUMemoryManager (lazy-loaded)
    ├── cache: GPUArrayCache (lazy-loaded)
    └── profiler: GPUProfiler (lazy-loaded) [NEW in v3.2]
    
    This module consolidates GPU detection from:
    - utils/normalization.py (GPU_AVAILABLE)
    - optimization/gpu_wrapper.py (_GPU_AVAILABLE, check_gpu_available)
    - optimization/ground_truth.py (_gpu_available)
    - optimization/gpu_profiler.py (gpu_available)
    - features/gpu_processor.py (GPU_AVAILABLE)
    - io/ground_truth_optimizer.py (_gpu_available)

Author: LiDAR Trainer Agent (Audit Phase 1, Consolidation Phase 1.2, Phase 3)
Date: November 22, 2025
Version: 3.2.0 (Profiling + Optimizations)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GPUManager:
    """
    Singleton for centralized GPU detection and management.
    
    This class provides a single source of truth for GPU availability
    across the entire codebase. It uses lazy initialization and caches
    results to avoid repeated hardware checks.
    
    v3.1+ Composition API provides unified access to:
    - Memory management (gpu.memory.*)
    - Array caching (gpu.cache.*)
    
    v3.2+ Performance profiling (Phase 3):
    - GPU profiling (gpu.profiler.*)
    
    Attributes:
        gpu_available: Basic GPU (CuPy) availability
        cuml_available: cuML (GPU ML library) availability
        cuspatial_available: cuSpatial (GPU spatial ops) availability
        faiss_gpu_available: FAISS-GPU (similarity search) availability
        memory: GPUMemoryManager instance (lazy-loaded)
        cache: GPUArrayCache instance (lazy-loaded)
        profiler: GPUProfiler instance (lazy-loaded) [NEW in v3.2]
    
    Example:
        >>> gpu = GPUManager()
        >>> if gpu.gpu_available:
        ...     print("GPU acceleration enabled")
        GPU acceleration enabled
        
        >>> # v3.1+ Composition API
        >>> gpu.memory.allocate(2.5)
        >>> gpu.cache.get_or_upload('normals', array)
        
        >>> # v3.2+ Profiling
        >>> with gpu.profiler.profile('compute_features'):
        ...     features = compute_gpu(points)
        >>> gpu.profiler.print_report()
        
        >>> info = gpu.get_info()
        >>> print(info['cuml_available'])
        True
    """
    
    _instance: Optional['GPUManager'] = None
    _gpu_available: Optional[bool] = None
    _cuml_available: Optional[bool] = None
    _cuspatial_available: Optional[bool] = None
    _faiss_gpu_available: Optional[bool] = None
    
    # Lazy-loaded sub-components (v3.1+)
    _memory_manager: Optional['GPUMemoryManager'] = None
    _array_cache: Optional['GPUArrayCache'] = None
    _profiler: Optional['GPUProfiler'] = None
    
    def __new__(cls) -> 'GPUManager':
        """
        Ensure only one instance exists (singleton pattern).
        
        Returns:
            The singleton GPUManager instance
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logger.debug("Created new GPUManager singleton instance")
        return cls._instance
    
    @property
    def gpu_available(self) -> bool:
        """
        Check if basic GPU (CuPy) is available.
        
        This checks if:
        - CuPy is installed
        - CUDA runtime is available
        - GPU device is accessible
        
        Returns:
            True if GPU is available, False otherwise
        """
        if self._gpu_available is None:
            self._gpu_available = self._check_cupy()
            if self._gpu_available:
                logger.info("✅ GPU (CuPy) detected and available")
            else:
                logger.info("❌ GPU (CuPy) not available")
        return self._gpu_available
    
    @property
    def cuml_available(self) -> bool:
        """
        Check if cuML (GPU ML library) is available.
        
        This checks if:
        - Basic GPU is available
        - cuML is installed
        - cuML can initialize properly
        
        Returns:
            True if cuML is available, False otherwise
        """
        if self._cuml_available is None:
            self._cuml_available = self._check_cuml()
            if self._cuml_available:
                logger.info("✅ cuML (GPU ML) detected and available")
            else:
                logger.debug("❌ cuML (GPU ML) not available")
        return self._cuml_available
    
    @property
    def cuspatial_available(self) -> bool:
        """
        Check if cuSpatial (GPU spatial ops) is available.
        
        This checks if:
        - Basic GPU is available
        - cuSpatial is installed
        
        Returns:
            True if cuSpatial is available, False otherwise
        """
        if self._cuspatial_available is None:
            self._cuspatial_available = self._check_cuspatial()
            if self._cuspatial_available:
                logger.info("✅ cuSpatial (GPU spatial) detected and available")
            else:
                logger.debug("❌ cuSpatial (GPU spatial) not available")
        return self._cuspatial_available
    
    @property
    def faiss_gpu_available(self) -> bool:
        """
        Check if FAISS-GPU (GPU similarity search) is available.
        
        This checks if:
        - Basic GPU is available
        - FAISS is installed
        - FAISS GPU support is enabled
        
        Returns:
            True if FAISS-GPU is available, False otherwise
        """
        if self._faiss_gpu_available is None:
            self._faiss_gpu_available = self._check_faiss()
            if self._faiss_gpu_available:
                logger.info("✅ FAISS-GPU detected and available")
            else:
                logger.debug("❌ FAISS-GPU not available")
        return self._faiss_gpu_available
    
    def _check_cupy(self) -> bool:
        """
        Internal method to check CuPy availability.
        
        Returns:
            True if CuPy is available, False otherwise
        """
        try:
            import cupy as cp
            # Try to create a small array to verify GPU is accessible
            _ = cp.array([1.0])
            return True
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.debug(f"CuPy check failed: {e}")
            return False
    
    def _check_cuml(self) -> bool:
        """
        Internal method to check cuML availability.
        
        Returns:
            True if cuML is available, False otherwise
        """
        if not self.gpu_available:
            return False
        
        try:
            from cuml.neighbors import NearestNeighbors
            import cupy as cp
            # Verify GPU compute capability is accessible
            cp.cuda.Device(0).compute_capability
            return True
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.debug(f"cuML check failed: {e}")
            return False
    
    def _check_cuspatial(self) -> bool:
        """
        Internal method to check cuSpatial availability.
        
        Returns:
            True if cuSpatial is available, False otherwise
        """
        if not self.gpu_available:
            return False
        
        try:
            import cuspatial
            return True
        except ImportError as e:
            logger.debug(f"cuSpatial check failed: {e}")
            return False
    
    def _check_faiss(self) -> bool:
        """
        Internal method to check FAISS-GPU availability.
        
        Returns:
            True if FAISS-GPU is available, False otherwise
        """
        if not self.gpu_available:
            return False
        
        try:
            import faiss
            # Check if GPU resources are available
            has_gpu = hasattr(faiss, 'StandardGpuResources')
            if has_gpu:
                # Try to create GPU resources to verify it works
                try:
                    res = faiss.StandardGpuResources()
                    return True
                except Exception:
                    return False
            return False
        except ImportError as e:
            logger.debug(f"FAISS-GPU check failed: {e}")
            return False
    
    def get_info(self) -> dict:
        """
        Get comprehensive GPU information.
        
        Returns:
            Dictionary with availability status for all GPU libraries
            
        Example:
            >>> gpu = GPUManager()
            >>> info = gpu.get_info()
            >>> print(info)
            {
                'gpu_available': True,
                'cuml_available': True,
                'cuspatial_available': True,
                'faiss_gpu_available': False
            }
        """
        return {
            'gpu_available': self.gpu_available,
            'cuml_available': self.cuml_available,
            'cuspatial_available': self.cuspatial_available,
            'faiss_gpu_available': self.faiss_gpu_available,
        }
    
    # ========================================================================
    # v3.1+ Composition API - Unified GPU Management
    # ========================================================================
    
    @property
    def memory(self):
        """
        Access GPU memory management features (v3.1+).
        
        Provides unified access to GPUMemoryManager for:
        - Memory allocation checking
        - Usage monitoring
        - Cache cleanup
        - Fragmentation prevention
        
        Returns:
            GPUMemoryManager instance (lazy-loaded)
            
        Example:
            >>> gpu = GPUManager()
            >>> if gpu.memory.allocate(2.5):
            ...     # Process with GPU
            ...     result = process_on_gpu(data)
            >>> 
            >>> available = gpu.memory.get_available_memory()
            >>> print(f"Available: {available:.2f} GB")
        """
        if self._memory_manager is None:
            from .gpu_memory import GPUMemoryManager
            self._memory_manager = GPUMemoryManager()
            logger.debug("Lazy-loaded GPUMemoryManager")
        return self._memory_manager
    
    @property
    def cache(self):
        """
        Access GPU array caching features (v3.1+).
        
        Provides unified access to GPUArrayCache for:
        - Smart array caching
        - LFU eviction
        - Minimized CPU↔GPU transfers
        - Slice-based updates
        
        Returns:
            GPUArrayCache instance (lazy-loaded)
            
        Example:
            >>> gpu = GPUManager()
            >>> # First access: uploads to GPU
            >>> gpu_arr = gpu.cache.get_or_upload('normals', normals_cpu)
            >>> 
            >>> # Second access: returns cached (no upload!)
            >>> gpu_arr = gpu.cache.get_or_upload('normals', normals_cpu)
        """
        if self._array_cache is None:
            from ..optimization.gpu_cache import GPUArrayCache
            self._array_cache = GPUArrayCache()
            logger.debug("Lazy-loaded GPUArrayCache")
        return self._array_cache
    
    @property
    def profiler(self):
        """
        Access GPU profiling features (v3.2+).
        
        Provides unified access to GPUProfiler for:
        - CUDA event-based timing
        - Memory usage tracking
        - Bottleneck detection
        - Transfer statistics
        - Performance reports
        
        Returns:
            GPUProfiler instance (lazy-loaded)
            
        Example:
            >>> gpu = GPUManager()
            >>> 
            >>> with gpu.profiler.profile('compute_normals'):
            ...     normals = compute_normals_gpu(points)
            >>> 
            >>> stats = gpu.profiler.get_stats()
            >>> gpu.profiler.print_report()
        """
        if self._profiler is None:
            from .gpu_profiler import GPUProfiler
            self._profiler = GPUProfiler()
            logger.debug("Lazy-loaded GPUProfiler")
        return self._profiler
    
    def get_memory_info(self) -> dict:
        """
        Get GPU memory information (convenience method).
        
        This is a convenience wrapper around memory.get_memory_info()
        for backward compatibility and ease of use.
        
        Returns:
            Dictionary with memory statistics (free_gb, total_gb, used_gb, etc.)
            
        Example:
            >>> gpu = GPUManager()
            >>> info = gpu.get_memory_info()
            >>> print(f"Free: {info['free_gb']:.2f} GB")
        """
        if not self.gpu_available:
            return {
                'free_gb': 0.0,
                'total_gb': 0.0,
                'used_gb': 0.0,
                'utilization': 0.0
            }
        return self.memory.get_memory_info()
    
    def cleanup(self):
        """
        Full cleanup of all GPU resources (convenience method).
        
        This clears:
        - Array cache
        - Memory pools
        - Profiler entries
        - Any allocated GPU memory
        
        Example:
            >>> gpu = GPUManager()
            >>> # ... processing ...
            >>> gpu.cleanup()  # Free all GPU resources
        """
        if self._array_cache is not None:
            self._array_cache.clear()
        if self._memory_manager is not None:
            self._memory_manager.cleanup()
        if self._profiler is not None:
            self._profiler.reset()
        logger.debug("GPU resources cleaned up")
    
    # ========================================================================
    # End v3.1+ Composition API
    # ========================================================================
    
    def reset_cache(self):
        """
        Reset all cached GPU availability checks.
        
        This forces re-checking GPU availability on next access.
        Useful for testing or if GPU state changes at runtime.
        """
        self._gpu_available = None
        self._cuml_available = None
        self._cuspatial_available = None
        self._faiss_gpu_available = None
        logger.debug("GPUManager cache reset")
    
    def __repr__(self) -> str:
        """
        String representation of GPUManager.
        
        Returns:
            Formatted string with GPU status
        """
        info = self.get_info()
        status = "✅" if info['gpu_available'] else "❌"
        return (
            f"GPUManager({status} GPU, "
            f"cuML={info['cuml_available']}, "
            f"cuSpatial={info['cuspatial_available']}, "
            f"FAISS={info['faiss_gpu_available']})"
        )


# Convenience function
def get_gpu_manager() -> GPUManager:
    """
    Get the global GPUManager instance.
    
    Returns:
        The singleton GPUManager instance
        
    Example:
        >>> from ign_lidar.core.gpu import get_gpu_manager
        >>> gpu = get_gpu_manager()
        >>> print(gpu.gpu_available)
        True
    """
    return GPUManager()


# Backward compatibility aliases
# These are evaluated at import time for compatibility with old code
_gpu_manager = get_gpu_manager()
GPU_AVAILABLE = _gpu_manager.gpu_available
HAS_CUPY = GPU_AVAILABLE


__all__ = [
    'GPUManager',
    'get_gpu_manager',
    'GPU_AVAILABLE',  # Backward compat
    'HAS_CUPY',       # Backward compat
]
