"""
Unified GPU Acceleration Wrapper

Provides a consistent decorator pattern for adding GPU acceleration to functions
with automatic fallback to CPU implementations.

This module implements the architectural recommendations from the GPU optimization audit
to standardize GPU/CPU switching across the codebase.

Example:
    >>> from ign_lidar.optimization.gpu_wrapper import gpu_accelerated
    >>> 
    >>> @gpu_accelerated(cpu_fallback=True)
    >>> def process_points(points, k=12):
    >>>     # CPU implementation
    >>>     from sklearn.neighbors import NearestNeighbors
    >>>     nbrs = NearestNeighbors(n_neighbors=k)
    >>>     return nbrs.fit(points)
    >>> 
    >>> def process_points_gpu(points, k=12):
    >>>     # GPU implementation
    >>>     import cupy as cp
    >>>     from cuml.neighbors import NearestNeighbors
    >>>     points_gpu = cp.asarray(points)
    >>>     nbrs = NearestNeighbors(n_neighbors=k)
    >>>     return nbrs.fit(points_gpu)
    >>> 
    >>> # Automatic GPU/CPU selection
    >>> result = process_points(points, k=12, use_gpu=True)
"""

from functools import wraps
from typing import Callable, Any, Optional
import logging

from ..core.gpu import GPUManager

logger = logging.getLogger(__name__)

# GPU availability check (centralized)
_gpu_manager = GPUManager()


def check_gpu_available() -> bool:
    """
    Check if GPU acceleration is available.
    
    DEPRECATED: Use GPUManager directly instead.
    This function is kept for backward compatibility.
    
    Returns:
        True if CuPy and cuML are available, False otherwise
    """
    return _gpu_manager.gpu_available and _gpu_manager.cuml_available


def gpu_accelerated(
    cpu_fallback: bool = True,
    log_performance: bool = False,
    gpu_func_suffix: str = "_gpu"
):
    """
    Decorator to automatically add GPU acceleration with CPU fallback.
    
    This decorator looks for a GPU version of the function with the specified suffix
    (default: "_gpu"). If use_gpu=True is passed and GPU is available, it calls the
    GPU version. Otherwise, it calls the CPU version.
    
    Args:
        cpu_fallback: If True, fall back to CPU on GPU failure (default True)
        log_performance: If True, log timing information (default False)
        gpu_func_suffix: Suffix for GPU function name (default "_gpu")
    
    Example:
        >>> @gpu_accelerated(cpu_fallback=True, log_performance=True)
        >>> def compute_distances(points, k=10):
        >>>     # CPU implementation
        >>>     return sklearn_knn(points, k)
        >>> 
        >>> def compute_distances_gpu(points, k=10):
        >>>     # GPU implementation
        >>>     return cuml_knn(points, k)
        >>> 
        >>> # Automatically uses GPU if available
        >>> result = compute_distances(points, k=10, use_gpu=True)
    """
    def decorator(cpu_func: Callable) -> Callable:
        @wraps(cpu_func)
        def wrapper(*args, **kwargs) -> Any:
            # Extract use_gpu flag
            use_gpu = kwargs.pop('use_gpu', False)
            
            # If GPU not requested, use CPU directly
            if not use_gpu:
                return cpu_func(*args, **kwargs)
            
            # Check if GPU is available
            if not check_gpu_available():
                if cpu_fallback:
                    logger.debug(f"{cpu_func.__name__}: GPU not available, using CPU")
                    return cpu_func(*args, **kwargs)
                else:
                    raise RuntimeError(
                        f"{cpu_func.__name__}: GPU requested but not available. "
                        "Install cupy-cuda11x or cupy-cuda12x and cuml."
                    )
            
            # Look for GPU version
            gpu_func_name = f"{cpu_func.__name__}{gpu_func_suffix}"
            
            # Try to find GPU function in the same module/namespace
            gpu_func = None
            
            # Check in function's globals (same module)
            if hasattr(cpu_func, '__globals__'):
                gpu_func = cpu_func.__globals__.get(gpu_func_name)
            
            # Check in function's class (for methods)
            if gpu_func is None and len(args) > 0:
                obj = args[0]
                if hasattr(obj, gpu_func_name):
                    gpu_func = getattr(obj, gpu_func_name)
            
            if gpu_func is None:
                if cpu_fallback:
                    logger.warning(
                        f"{cpu_func.__name__}: No GPU version found "
                        f"(looking for '{gpu_func_name}'), using CPU"
                    )
                    return cpu_func(*args, **kwargs)
                else:
                    raise RuntimeError(
                        f"{cpu_func.__name__}: GPU requested but no GPU "
                        f"implementation found (expected '{gpu_func_name}')"
                    )
            
            # Try GPU execution
            try:
                if log_performance:
                    import time
                    start = time.time()
                    result = gpu_func(*args, **kwargs)
                    elapsed = time.time() - start
                    logger.info(f"{cpu_func.__name__} (GPU): {elapsed:.3f}s")
                    return result
                else:
                    return gpu_func(*args, **kwargs)
                    
            except Exception as e:
                if cpu_fallback:
                    logger.warning(
                        f"{cpu_func.__name__}: GPU execution failed ({e}), "
                        f"falling back to CPU"
                    )
                    if log_performance:
                        import time
                        start = time.time()
                        result = cpu_func(*args, **kwargs)
                        elapsed = time.time() - start
                        logger.info(f"{cpu_func.__name__} (CPU fallback): {elapsed:.3f}s")
                        return result
                    else:
                        return cpu_func(*args, **kwargs)
                else:
                    raise
        
        return wrapper
    return decorator


class GPUContext:
    """
    Context manager for GPU operations with automatic cleanup.
    
    Example:
        >>> with GPUContext() as gpu:
        >>>     if gpu.available:
        >>>         points_gpu = gpu.to_gpu(points)
        >>>         # ... GPU operations ...
        >>>         result = gpu.to_cpu(result_gpu)
    """
    
    def __init__(self, cleanup_on_exit: bool = True):
        """
        Initialize GPU context.
        
        Args:
            cleanup_on_exit: Free GPU memory on context exit (default True)
        """
        self.cleanup_on_exit = cleanup_on_exit
        self.available = check_gpu_available()
        self.cp = None
        
        if self.available:
            import cupy as cp
            self.cp = cp
    
    def __enter__(self):
        """Enter context."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and cleanup."""
        if self.cleanup_on_exit and self.available:
            self.free_memory()
        return False
    
    def to_gpu(self, array, dtype=None):
        """
        Transfer array to GPU.
        
        Args:
            array: NumPy array
            dtype: Target dtype (default: preserve)
        
        Returns:
            CuPy array
        """
        if not self.available:
            return array
        
        if dtype is not None:
            return self.cp.asarray(array, dtype=dtype)
        else:
            return self.cp.asarray(array)
    
    def to_cpu(self, array):
        """
        Transfer array to CPU.
        
        Args:
            array: CuPy array
        
        Returns:
            NumPy array
        """
        if not self.available:
            return array
        
        if hasattr(array, 'get'):
            return array.get()
        elif hasattr(self.cp, 'asnumpy'):
            return self.cp.asnumpy(array)
        else:
            return array
    
    def free_memory(self):
        """Free GPU memory."""
        if self.available and self.cp is not None:
            try:
                mempool = self.cp.get_default_memory_pool()
                mempool.free_all_blocks()
                pinned_mempool = self.cp.get_default_pinned_memory_pool()
                pinned_mempool.free_all_blocks()
            except Exception as e:
                logger.debug(f"Could not free GPU memory: {e}")


def require_gpu(func: Callable) -> Callable:
    """
    Decorator to require GPU acceleration (no CPU fallback).
    
    Raises RuntimeError if GPU is not available.
    
    Example:
        >>> @require_gpu
        >>> def gpu_only_function(points):
        >>>     import cupy as cp
        >>>     return cp.asarray(points)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not check_gpu_available():
            raise RuntimeError(
                f"{func.__name__} requires GPU acceleration. "
                "Install cupy-cuda11x or cupy-cuda12x and cuml."
            )
        return func(*args, **kwargs)
    return wrapper


__all__ = [
    'gpu_accelerated',
    'GPUContext',
    'require_gpu',
    'check_gpu_available',
]
