"""
Classification Caching Optimization - Phase 3

Caches classification results to avoid redundant computation when processing
multiple tiles with similar characteristics.

This optimization is useful for:
- Batch processing of similar-sized tiles
- Pipeline processing with consistent parameters
- Real-time classification with recurring patterns
"""

import logging
from typing import Dict, Optional, Tuple
import numpy as np
from functools import lru_cache
import hashlib

logger = logging.getLogger(__name__)


class ClassificationCache:
    """
    Cache for classification results with hash-based lookup.
    
    Caches classification results indexed by:
    - Points hash (first 1000 points)
    - Feature mode
    - K-neighbors parameter
    
    This avoids redundant computation for duplicate or similar patches.
    """
    
    def __init__(self, max_size: int = 1000, ttl: Optional[float] = None):
        """
        Initialize classification cache.
        
        Args:
            max_size: Maximum number of cached results (LRU)
            ttl: Time-to-live for cached results in seconds (None = never expire)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self.hit_count = 0
        self.miss_count = 0
        
        logger.info(f"ClassificationCache initialized: max_size={max_size}, ttl={ttl}s")
    
    def _make_key(
        self,
        points: np.ndarray,
        mode: str = "lod2",
        k_neighbors: int = 10,
    ) -> str:
        """
        Create cache key from points and parameters.
        
        Args:
            points: Point cloud array
            mode: Feature mode
            k_neighbors: K-neighbors parameter
            
        Returns:
            Cache key string
        """
        # Hash first 1000 points to identify patch
        sample_size = min(1000, len(points))
        points_sample = points[:sample_size].tobytes()
        points_hash = hashlib.md5(points_sample).hexdigest()
        
        # Create composite key
        key = f"{points_hash}_{mode}_{k_neighbors}"
        return key
    
    def get(
        self,
        points: np.ndarray,
        mode: str = "lod2",
        k_neighbors: int = 10,
    ) -> Optional[Dict]:
        """
        Get cached classification result.
        
        Args:
            points: Point cloud array
            mode: Feature mode
            k_neighbors: K-neighbors parameter
            
        Returns:
            Cached result dict or None if not cached
        """
        import time
        
        key = self._make_key(points, mode, k_neighbors)
        
        if key in self.cache:
            result, timestamp = self.cache[key]
            
            # Check TTL if set
            if self.ttl is not None:
                age = time.time() - timestamp
                if age > self.ttl:
                    del self.cache[key]
                    self.miss_count += 1
                    return None
            
            self.hit_count += 1
            return result
        
        self.miss_count += 1
        return None
    
    def put(
        self,
        points: np.ndarray,
        result: Dict,
        mode: str = "lod2",
        k_neighbors: int = 10,
    ) -> None:
        """
        Cache classification result.
        
        Args:
            points: Point cloud array
            result: Classification result
            mode: Feature mode
            k_neighbors: K-neighbors parameter
        """
        import time
        
        # Don't cache if we're at max size
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (FIFO, simple approach)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        key = self._make_key(points, mode, k_neighbors)
        self.cache[key] = (result, time.time())
    
    def clear(self) -> None:
        """Clear all cached results."""
        self.cache.clear()
        logger.info("Classification cache cleared")
    
    def stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hit_count,
            "misses": self.miss_count,
            "total": total,
            "hit_rate": hit_rate,
        }


# Global singleton cache
_classification_cache: Optional[ClassificationCache] = None


def get_classification_cache(max_size: int = 1000) -> ClassificationCache:
    """
    Get singleton classification cache.
    
    Args:
        max_size: Maximum cache size
        
    Returns:
        Global classification cache instance
    """
    global _classification_cache
    
    if _classification_cache is None:
        _classification_cache = ClassificationCache(max_size=max_size)
    
    return _classification_cache


class CachedClassificationStrategy:
    """
    Classification strategy with caching support.
    
    Wraps any classification strategy and adds caching to avoid
    redundant computation for duplicate patches.
    
    Usage:
        >>> from ign_lidar.optimization.classification_cache import CachedClassificationStrategy
        >>> from ign_lidar.features import CPUStrategy
        >>> 
        >>> # Wrap strategy with caching
        >>> base_strategy = CPUStrategy(k_neighbors=10)
        >>> cached_strategy = CachedClassificationStrategy(base_strategy)
        >>> 
        >>> # First call computes
        >>> result1 = cached_strategy.compute_features(points)
        >>> 
        >>> # Second call with same points uses cache
        >>> result2 = cached_strategy.compute_features(points)  # From cache!
    """
    
    def __init__(self, base_strategy, use_cache: bool = True, cache_size: int = 1000):
        """
        Initialize cached classification strategy.
        
        Args:
            base_strategy: Base classification strategy to wrap
            use_cache: Whether to use caching
            cache_size: Maximum cache size
        """
        self.base_strategy = base_strategy
        self.use_cache = use_cache
        self.cache = get_classification_cache(cache_size) if use_cache else None
        self.cache_hits = 0
        self.cache_misses = 0
    
    def compute_features(self, points: np.ndarray, **kwargs) -> Dict:
        """
        Compute features with optional caching.
        
        Args:
            points: Point cloud array
            **kwargs: Additional arguments passed to base strategy
            
        Returns:
            Features dictionary
        """
        mode = kwargs.get("mode", "lod2")
        k = kwargs.get("k_neighbors", 10)
        
        # Try cache first
        if self.use_cache and self.cache is not None:
            cached = self.cache.get(points, mode, k)
            if cached is not None:
                self.cache_hits += 1
                logger.debug(f"Cache hit: mode={mode}, k={k}")
                return cached
            
            self.cache_misses += 1
        
        # Compute using base strategy
        result = self.base_strategy.compute_features(points, **kwargs)
        
        # Cache result
        if self.use_cache and self.cache is not None:
            self.cache.put(points, result, mode, k)
        
        return result
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        if self.cache is None:
            return {"enabled": False}
        
        stats = self.cache.stats()
        stats["enabled"] = True
        stats["wrapper_hits"] = self.cache_hits
        stats["wrapper_misses"] = self.cache_misses
        return stats
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CachedClassificationStrategy("
            f"base={self.base_strategy}, cache={'enabled' if self.use_cache else 'disabled'})"
        )


if __name__ == "__main__":
    # Simple test
    cache = get_classification_cache(max_size=10)
    
    # Create test data
    points = np.random.rand(1000, 3).astype(np.float32)
    result = {"features": {"test": np.array([1, 2, 3])}}
    
    # Test caching
    cache.put(points, result)
    retrieved = cache.get(points)
    assert retrieved is not None
    
    stats = cache.stats()
    print(f"✓ Cache test passed: {stats}")
