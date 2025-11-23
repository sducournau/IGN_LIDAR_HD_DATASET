"""
GPU Array Caching

Smart cache for GPU arrays to minimize redundant CPU-GPU transfers.
Tracks access patterns and evicts least frequently used items.

Version: 1.0.0
"""

import logging
from typing import Optional, Dict
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
    
    Features:
    - LFU (Least Frequently Used) eviction policy
    - Cache hit/miss statistics
    - Automatic size management
    - In-place slice updates
    
    Example:
        >>> cache = GPUArrayCache(max_size_gb=4.0)
        >>> 
        >>> # First access: uploads to GPU (cache miss)
        >>> gpu_arr = cache.get_or_upload('normals', normals_cpu)
        >>> 
        >>> # Second access: returns cached GPU array (cache hit!)
        >>> gpu_arr = cache.get_or_upload('normals', normals_cpu)
        >>> 
        >>> # Check performance
        >>> stats = cache.get_stats()
        >>> print(f"Cache hit rate: {stats['hit_rate']:.1%}")
    """
    
    def __init__(self, max_size_gb: float = 8.0):  # INCREASED from 4GB to 8GB for better caching
        self.max_size_gb = max_size_gb
        self.cache: Dict[str, 'cp.ndarray'] = {}
        self.cache_sizes: Dict[str, int] = {}
        self.access_count: Dict[str, int] = {}
        self.enabled = HAS_CUPY
        
        # Cache performance metrics
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._total_uploads_mb: float = 0.0
        self._eviction_count: int = 0
        
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
            self._cache_hits += 1  # Still a hit (avoid re-upload)
            return self.cache[key]
        
        # Return cached entry (CACHE HIT)
        if key in self.cache:
            self.access_count[key] += 1
            self._cache_hits += 1
            return self.cache[key]
        
        # CACHE MISS - need to upload
        self._cache_misses += 1
        
        # Check if we have space
        array_size_bytes = cpu_array.nbytes
        array_size_mb = array_size_bytes / (1024**2)
        current_size = sum(self.cache_sizes.values())
        
        if current_size + array_size_bytes > self.max_size_gb * (1024**3):
            # Evict least frequently used item
            self._evict_lfu()
        
        # Upload and cache
        gpu_array = cp.asarray(cpu_array, dtype=cp.float32)
        self.cache[key] = gpu_array
        self.cache_sizes[key] = array_size_bytes
        self.access_count[key] = 1
        
        # Track upload metrics
        self._total_uploads_mb += array_size_mb
        
        logger.debug(
            f"🔼 Cache MISS: uploaded '{key}' to GPU ({array_size_mb:.1f}MB) "
            f"| Total cached: {sum(self.cache_sizes.values())/(1024**2):.1f}MB "
            f"| Hit rate: {self.get_hit_rate():.1%}"
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
        
        self._eviction_count += 1
        
        logger.debug(
            f"⚠️  Evicting '{lfu_key}' from GPU cache "
            f"({self.cache_sizes[lfu_key]/(1024**2):.1f}MB, "
            f"{self.access_count[lfu_key]} accesses) "
            f"| Total evictions: {self._eviction_count}"
        )
        
        self.invalidate(lfu_key)
    
    def clear(self) -> None:
        """Clear all cached arrays and reset metrics."""
        self.cache.clear()
        self.cache_sizes.clear()
        self.access_count.clear()
        
        # Reset metrics
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_uploads_mb = 0.0
        self._eviction_count = 0
        
        if HAS_CUPY:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
    
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_accesses = self._cache_hits + self._cache_misses
        if total_accesses == 0:
            return 0.0
        return self._cache_hits / total_accesses
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get comprehensive cache statistics.
        
        Returns:
            Dictionary with cache performance metrics including:
            - num_cached: Number of arrays in cache
            - total_size_gb: Total cache size
            - utilization_pct: Cache utilization percentage
            - total_accesses: Total access count
            - cache_hits: Number of cache hits
            - cache_misses: Number of cache misses
            - hit_rate: Cache hit rate (0.0-1.0)
            - total_uploads_mb: Total data uploaded
            - eviction_count: Number of evictions
            - saved_transfers_mb: Estimated data transfer savings
        """
        total_size_gb = sum(self.cache_sizes.values()) / (1024**3)
        total_accesses = sum(self.access_count.values())
        
        # Calculate saved transfers (hits * average array size)
        avg_array_size_mb = (self._total_uploads_mb / max(self._cache_misses, 1))
        saved_transfers_mb = self._cache_hits * avg_array_size_mb
        
        return {
            'num_cached': len(self.cache),
            'total_size_gb': total_size_gb,
            'utilization_pct': (total_size_gb / self.max_size_gb) * 100,
            'total_accesses': total_accesses,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': self.get_hit_rate(),
            'total_uploads_mb': self._total_uploads_mb,
            'eviction_count': self._eviction_count,
            'saved_transfers_mb': saved_transfers_mb,
            'cache_keys': list(self.cache.keys())
        }
    
    def print_stats(self) -> None:
        """Print formatted cache statistics."""
        stats = self.get_stats()
        
        logger.info("=" * 60)
        logger.info("GPU Array Cache Statistics")
        logger.info("=" * 60)
        logger.info(f"  Cached arrays: {stats['num_cached']}")
        logger.info(f"  Cache size: {stats['total_size_gb']:.2f} GB / {self.max_size_gb:.2f} GB")
        logger.info(f"  Utilization: {stats['utilization_pct']:.1f}%")
        logger.info(f"  Cache hits: {stats['cache_hits']}")
        logger.info(f"  Cache misses: {stats['cache_misses']}")
        logger.info(f"  Hit rate: {stats['hit_rate']:.1%}")
        logger.info(f"  Total uploads: {stats['total_uploads_mb']:.1f} MB")
        logger.info(f"  Saved transfers: {stats['saved_transfers_mb']:.1f} MB")
        logger.info(f"  Evictions: {stats['eviction_count']}")
        logger.info("=" * 60)
