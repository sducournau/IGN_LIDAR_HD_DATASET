"""
Memory management utilities for LiDAR processing.

This module provides memory cleanup and management functions to prevent
out-of-memory errors during large-scale LiDAR processing.
"""

import gc
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def aggressive_memory_cleanup() -> None:
    """
    Aggressive memory cleanup to prevent OOM.
    
    Clears all caches and forces garbage collection across different
    computation backends (CPU, CUDA, CuPy).
    
    This function:
    1. Forces Python garbage collection
    2. Clears PyTorch CUDA cache if available
    3. Clears CuPy memory pools if available
    4. Performs final garbage collection
    
    Safe to call even if GPU libraries are not installed.
    """
    gc.collect()
    
    # Clear CUDA cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("Cleared CUDA cache")
    except (ImportError, RuntimeError):
        pass
    
    # Clear CuPy cache if available
    try:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        logger.debug("Cleared CuPy memory pools")
    except (ImportError, AttributeError):
        pass
    
    gc.collect()


def clear_gpu_cache() -> bool:
    """
    Clear GPU memory cache for PyTorch and CuPy.
    
    Returns:
        bool: True if any GPU cache was cleared, False otherwise
    """
    cleared = False
    
    # Clear PyTorch CUDA cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            cleared = True
            logger.debug("Cleared PyTorch CUDA cache")
    except (ImportError, RuntimeError):
        pass
    
    # Clear CuPy cache
    try:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        cleared = True
        logger.debug("Cleared CuPy memory pools")
    except (ImportError, AttributeError):
        pass
    
    return cleared


def estimate_memory_usage(num_points: int, 
                         num_features: int = 10,
                         include_rgb: bool = False,
                         dtype_size: int = 4) -> float:
    """
    Estimate memory usage for a point cloud in MB.
    
    Args:
        num_points: Number of points in the cloud
        num_features: Number of features per point (default: 10)
        include_rgb: Whether RGB data is included
        dtype_size: Size of data type in bytes (4 for float32, 8 for float64)
        
    Returns:
        float: Estimated memory usage in megabytes
    """
    # Base features (xyz, classification, features)
    base_mem = num_points * (3 + 1 + num_features) * dtype_size
    
    # RGB channels
    rgb_mem = num_points * 3 * dtype_size if include_rgb else 0
    
    # Total in MB
    total_mb = (base_mem + rgb_mem) / (1024 * 1024)
    
    return total_mb


def check_available_memory() -> Optional[float]:
    """
    Check available system memory in GB.
    
    Returns:
        float: Available memory in GB, or None if cannot determine
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        return available_gb
    except ImportError:
        logger.warning("psutil not available, cannot check memory")
        return None


def check_gpu_memory() -> Optional[float]:
    """
    Check available GPU memory in GB.
    
    Returns:
        float: Available GPU memory in GB, or None if no GPU or cannot determine
    """
    # Try PyTorch first
    try:
        import torch
        if torch.cuda.is_available():
            free_mem, total_mem = torch.cuda.mem_get_info()
            return free_mem / (1024 ** 3)
    except (ImportError, RuntimeError):
        pass
    
    # Try CuPy
    try:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        # CuPy doesn't directly give free memory, but we can get used
        used_bytes = mempool.used_bytes()
        total_bytes = mempool.total_bytes()
        free_bytes = total_bytes - used_bytes
        return free_bytes / (1024 ** 3)
    except (ImportError, AttributeError):
        pass
    
    return None
