"""
GPU Memory Optimization Utilities

This module provides utilities for optimizing GPU memory usage during
point cloud processing, including smart caching and transfer minimization.

Version: 1.0.0
"""

import logging
from typing import Optional, Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None


class GPUArrayCache:
    """
    Smart cache for GPU arrays to minimize redundant transfers.
    
    Keeps frequently accessed arrays on GPU and tracks access patterns
    to optimize memory usage.
    
    Example:
        >>> cache = GPUArrayCache(max_size_gb=4.0)
        >>> 
        >>> # First access: uploads to GPU
        >>> gpu_arr = cache.get_or_upload('normals', normals_cpu)
        >>> 
        >>> # Second access: returns cached GPU array (no upload!)
        >>> gpu_arr = cache.get_or_upload('normals', normals_cpu)
    """
    
    def __init__(self, max_size_gb: float = 8.0):  # INCREASED from 4GB to 8GB for better caching
        self.max_size_gb = max_size_gb
        self.cache: Dict[str, cp.ndarray] = {}
        self.cache_sizes: Dict[str, int] = {}
        self.access_count: Dict[str, int] = {}
        self.enabled = HAS_CUPY
        
    def get_or_upload(
        self, 
        key: str, 
        cpu_array: np.ndarray,
        force_update: bool = False
    ) -> 'cp.ndarray':
        """
        Get cached GPU array or upload if not cached.
        
        Args:
            key: Cache key for the array
            cpu_array: CPU array to upload if not cached
            force_update: Force re-upload even if cached
            
        Returns:
            GPU array (cached or freshly uploaded)
        """
        if not self.enabled:
            return cpu_array
        
        # Update existing cache entry
        if force_update and key in self.cache:
            self.cache[key] = cp.asarray(cpu_array, dtype=cp.float32)
            self.access_count[key] += 1
            return self.cache[key]
        
        # Return cached entry
        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]
        
        # Check if we have space
        array_size_bytes = cpu_array.nbytes
        current_size = sum(self.cache_sizes.values())
        
        if current_size + array_size_bytes > self.max_size_gb * (1024**3):
            # Evict least frequently used item
            self._evict_lfu()
        
        # Upload and cache
        gpu_array = cp.asarray(cpu_array, dtype=cp.float32)
        self.cache[key] = gpu_array
        self.cache_sizes[key] = array_size_bytes
        self.access_count[key] = 1
        
        logger.debug(
            f"Cached '{key}' on GPU: {array_size_bytes/(1024**2):.1f}MB "
            f"(total: {sum(self.cache_sizes.values())/(1024**2):.1f}MB)"
        )
        
        return gpu_array
    
    def update_slice(
        self, 
        key: str, 
        start_idx: int, 
        end_idx: int,
        data: 'cp.ndarray'
    ) -> None:
        """
        Update a slice of cached GPU array in-place.
        
        This avoids re-uploading the entire array when only a portion changes.
        
        Args:
            key: Cache key for the array
            start_idx: Start index of slice to update
            end_idx: End index of slice to update
            data: GPU array with new data for the slice
        """
        if not self.enabled or key not in self.cache:
            return
        
        self.cache[key][start_idx:end_idx] = data
        self.access_count[key] += 1
    
    def get(self, key: str) -> Optional['cp.ndarray']:
        """Get cached GPU array without uploading."""
        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]
        return None
    
    def invalidate(self, key: str) -> None:
        """Remove array from cache."""
        if key in self.cache:
            del self.cache[key]
            del self.cache_sizes[key]
            del self.access_count[key]
    
    def _evict_lfu(self) -> None:
        """Evict least frequently used item from cache."""
        if not self.cache:
            return
        
        # Find item with lowest access count
        lfu_key = min(self.access_count, key=self.access_count.get)
        
        logger.debug(
            f"Evicting '{lfu_key}' from GPU cache "
            f"({self.cache_sizes[lfu_key]/(1024**2):.1f}MB, "
            f"{self.access_count[lfu_key]} accesses)"
        )
        
        self.invalidate(lfu_key)
    
    def clear(self) -> None:
        """Clear all cached arrays."""
        self.cache.clear()
        self.cache_sizes.clear()
        self.access_count.clear()
        
        if HAS_CUPY:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
    
    def get_stats(self) -> Dict[str, float]:
        """Get cache statistics."""
        total_size_gb = sum(self.cache_sizes.values()) / (1024**3)
        total_accesses = sum(self.access_count.values())
        
        return {
            'num_cached': len(self.cache),
            'total_size_gb': total_size_gb,
            'utilization_pct': (total_size_gb / self.max_size_gb) * 100,
            'total_accesses': total_accesses,
            'cache_keys': list(self.cache.keys())
        }


class TransferOptimizer:
    """
    Analyzes and optimizes GPU-CPU memory transfers.
    
    Tracks transfer patterns and provides recommendations for
    reducing unnecessary data movement.
    """
    
    def __init__(self):
        self.upload_count = 0
        self.download_count = 0
        self.upload_bytes = 0
        self.download_bytes = 0
        self.transfer_log = []
        
    def log_upload(self, array_size_bytes: int, description: str = "") -> None:
        """Log a CPU->GPU transfer."""
        self.upload_count += 1
        self.upload_bytes += array_size_bytes
        self.transfer_log.append({
            'direction': 'upload',
            'size_mb': array_size_bytes / (1024**2),
            'description': description
        })
    
    def log_download(self, array_size_bytes: int, description: str = "") -> None:
        """Log a GPU->CPU transfer."""
        self.download_count += 1
        self.download_bytes += array_size_bytes
        self.transfer_log.append({
            'direction': 'download',
            'size_mb': array_size_bytes / (1024**2),
            'description': description
        })
    
    def get_stats(self) -> Dict:
        """Get transfer statistics."""
        total_bytes = self.upload_bytes + self.download_bytes
        
        return {
            'upload_count': self.upload_count,
            'download_count': self.download_count,
            'upload_gb': self.upload_bytes / (1024**3),
            'download_gb': self.download_bytes / (1024**3),
            'total_gb': total_bytes / (1024**3),
            'avg_upload_mb': (self.upload_bytes / self.upload_count / (1024**2)) 
                            if self.upload_count > 0 else 0,
            'avg_download_mb': (self.download_bytes / self.download_count / (1024**2))
                              if self.download_count > 0 else 0,
        }
    
    def print_summary(self) -> None:
        """Print transfer summary."""
        stats = self.get_stats()
        
        logger.info("📊 GPU Transfer Statistics:")
        logger.info(f"   Uploads: {stats['upload_count']} ({stats['upload_gb']:.2f}GB)")
        logger.info(f"   Downloads: {stats['download_count']} ({stats['download_gb']:.2f}GB)")
        logger.info(f"   Total: {stats['total_gb']:.2f}GB")
        logger.info(f"   Avg Upload: {stats['avg_upload_mb']:.1f}MB")
        logger.info(f"   Avg Download: {stats['avg_download_mb']:.1f}MB")
    
    def reset(self) -> None:
        """Reset all counters."""
        self.upload_count = 0
        self.download_count = 0
        self.upload_bytes = 0
        self.download_bytes = 0
        self.transfer_log.clear()


def estimate_gpu_memory_for_features(
    num_points: int,
    feature_set: str = 'minimal'
) -> Tuple[float, Dict[str, float]]:
    """
    Estimate GPU memory required for feature computation.
    
    Args:
        num_points: Number of points to process
        feature_set: Feature set to compute ('minimal', 'standard', 'full')
        
    Returns:
        Tuple of (total_gb, breakdown_dict)
    """
    # Base arrays (always needed)
    points_mem = num_points * 3 * 4  # float32
    normals_mem = num_points * 3 * 4
    indices_mem = num_points * 12 * 4  # k=12 neighbors
    
    # Feature-specific memory
    feature_mem = {
        'minimal': num_points * 8 * 4,   # 8 features
        'standard': num_points * 15 * 4,  # 15 features
        'full': num_points * 30 * 4      # 30 features
    }
    
    # Intermediate computations (covariance matrices, etc.)
    intermediate_mem = num_points * 20 * 4  # Conservative estimate
    
    # Total
    total_bytes = (
        points_mem + 
        normals_mem + 
        indices_mem + 
        feature_mem.get(feature_set, feature_mem['standard']) +
        intermediate_mem
    )
    
    # Add 20% overhead for CuPy/CUDA runtime
    total_bytes *= 1.2
    
    breakdown = {
        'points_gb': points_mem / (1024**3),
        'normals_gb': normals_mem / (1024**3),
        'indices_gb': indices_mem / (1024**3),
        'features_gb': feature_mem.get(feature_set, feature_mem['standard']) / (1024**3),
        'intermediate_gb': intermediate_mem / (1024**3),
        'overhead_gb': total_bytes * 0.2 / (1024**3),
    }
    
    return total_bytes / (1024**3), breakdown


def optimize_chunk_size_for_vram(
    num_points: int,
    available_vram_gb: float,
    feature_set: str = 'minimal',
    safety_factor: float = 0.75
) -> int:
    """
    Calculate optimal chunk size given VRAM constraints.
    
    Args:
        num_points: Total number of points
        available_vram_gb: Available GPU VRAM in GB
        feature_set: Feature set to compute
        safety_factor: Safety factor (0.75 = use 75% of available VRAM)
        
    Returns:
        Optimal chunk size in number of points
    """
    usable_vram = available_vram_gb * safety_factor
    
    # Binary search for optimal chunk size
    min_chunk = 100_000
    max_chunk = min(num_points, 20_000_000)
    
    optimal_chunk = min_chunk
    
    while min_chunk <= max_chunk:
        mid_chunk = (min_chunk + max_chunk) // 2
        required_mem, _ = estimate_gpu_memory_for_features(mid_chunk, feature_set)
        
        if required_mem <= usable_vram:
            optimal_chunk = mid_chunk
            min_chunk = mid_chunk + 100_000
        else:
            max_chunk = mid_chunk - 100_000
    
    logger.info(
        f"Optimized chunk size: {optimal_chunk:,} points "
        f"({estimate_gpu_memory_for_features(optimal_chunk, feature_set)[0]:.2f}GB / "
        f"{usable_vram:.2f}GB available)"
    )
    
    return optimal_chunk
