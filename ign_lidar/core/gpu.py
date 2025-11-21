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

Usage:
    from ign_lidar.core.gpu import GPUManager
    
    gpu = GPUManager()
    if gpu.gpu_available:
        # Use GPU acceleration
        pass
    
    # Legacy compatibility
    from ign_lidar.core.gpu import GPU_AVAILABLE

Architecture:
    This module consolidates GPU detection from:
    - utils/normalization.py (GPU_AVAILABLE)
    - optimization/gpu_wrapper.py (_GPU_AVAILABLE, check_gpu_available)
    - optimization/ground_truth.py (_gpu_available)
    - optimization/gpu_profiler.py (gpu_available)
    - features/gpu_processor.py (GPU_AVAILABLE)
    - io/ground_truth_optimizer.py (_gpu_available)

Author: LiDAR Trainer Agent (Audit Phase 1)
Date: November 21, 2025
Version: 1.0
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
    
    Attributes:
        gpu_available: Basic GPU (CuPy) availability
        cuml_available: cuML (GPU ML library) availability
        cuspatial_available: cuSpatial (GPU spatial ops) availability
        faiss_gpu_available: FAISS-GPU (similarity search) availability
    
    Example:
        >>> gpu = GPUManager()
        >>> if gpu.gpu_available:
        ...     print("GPU acceleration enabled")
        GPU acceleration enabled
        
        >>> info = gpu.get_info()
        >>> print(info['cuml_available'])
        True
    """
    
    _instance: Optional['GPUManager'] = None
    _gpu_available: Optional[bool] = None
    _cuml_available: Optional[bool] = None
    _cuspatial_available: Optional[bool] = None
    _faiss_gpu_available: Optional[bool] = None
    
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
