"""
Tile Data Cache Module

Eliminates redundant tile loading by caching recently loaded tiles.
This addresses the triple-loading issue where the same tile is loaded:
1. During prefetching (for ground truth bbox)
2. During skip checking (for validation)
3. During processing (for feature computation)

Author: GitHub Copilot
Date: October 16, 2025
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from collections import OrderedDict
import gc

logger = logging.getLogger(__name__)


class TileDataCache:
    """
    LRU cache for loaded tile data to avoid redundant I/O.
    
    Typical tile size: 15-20M points ≈ 300-500 MB in memory
    With max_tiles=2, memory usage ≈ 600-1000 MB (acceptable overhead)
    """
    
    def __init__(self, max_tiles: int = 2, enable_stats: bool = True):
        """
        Initialize tile data cache.
        
        Args:
            max_tiles: Maximum number of tiles to keep in cache (LRU eviction)
            enable_stats: Track cache hit/miss statistics
        """
        self._cache: OrderedDict[Path, Dict[str, Any]] = OrderedDict()
        self._max_tiles = max_tiles
        self._enable_stats = enable_stats
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
    def get_or_load(
        self, 
        tile_path: Path, 
        loader_fn: Callable[[Path], Optional[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        """
        Get tile data from cache or load it using the provided loader function.
        
        Args:
            tile_path: Path to tile file (used as cache key)
            loader_fn: Function to load tile if not in cache
                      Should return dict with keys: points, classification, etc.
                      
        Returns:
            Tile data dictionary, or None if loading failed
        """
        # Check cache
        if tile_path in self._cache:
            # Move to end (mark as recently used)
            self._cache.move_to_end(tile_path)
            self._hits += 1
            
            if self._enable_stats:
                hit_rate = self._hits / (self._hits + self._misses) * 100
                logger.info(
                    f"  ♻️  Using cached tile data "
                    f"(cache hit rate: {hit_rate:.1f}%)"
                )
            
            return self._cache[tile_path]
        
        # Cache miss - load tile
        self._misses += 1
        logger.debug(f"  📥 Cache miss, loading tile: {tile_path.name}")
        
        tile_data = loader_fn(tile_path)
        
        if tile_data is None:
            logger.warning(f"  ⚠️  Failed to load tile: {tile_path.name}")
            return None
        
        # Add to cache
        self._cache[tile_path] = tile_data
        
        # LRU eviction if cache is full
        if len(self._cache) > self._max_tiles:
            # Remove oldest (first) item
            oldest_path, oldest_data = self._cache.popitem(last=False)
            self._evictions += 1
            
            logger.debug(
                f"  🗑️  Evicted {oldest_path.name} from cache "
                f"(size: {len(self._cache)}/{self._max_tiles})"
            )
            
            # Clean up memory
            del oldest_data
            gc.collect()
        
        return tile_data
    
    def invalidate(self, tile_path: Path) -> None:
        """
        Remove a specific tile from cache.
        
        Args:
            tile_path: Path to tile to invalidate
        """
        if tile_path in self._cache:
            del self._cache[tile_path]
            logger.debug(f"  🔄 Invalidated cache for {tile_path.name}")
    
    def clear(self) -> None:
        """Clear all cached data and reset statistics."""
        self._cache.clear()
        gc.collect()
        
        if self._enable_stats:
            logger.info(
                f"  🧹 Cache cleared. Final stats: "
                f"{self._hits} hits, {self._misses} misses, "
                f"{self._evictions} evictions"
            )
        
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with hits, misses, evictions, and hit_rate
        """
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hits': self._hits,
            'misses': self._misses,
            'evictions': self._evictions,
            'size': len(self._cache),
            'max_size': self._max_tiles,
            'hit_rate_percent': hit_rate
        }
    
    def log_stats(self) -> None:
        """Log current cache statistics."""
        stats = self.get_stats()
        logger.info(
            f"📊 Tile Cache Stats: "
            f"{stats['hits']} hits ({stats['hit_rate_percent']:.1f}%), "
            f"{stats['misses']} misses, "
            f"{stats['evictions']} evictions, "
            f"size: {stats['size']}/{stats['max_size']}"
        )
    
    def __len__(self) -> int:
        """Return current cache size."""
        return len(self._cache)
    
    def __contains__(self, tile_path: Path) -> bool:
        """Check if tile is in cache."""
        return tile_path in self._cache
    
    def __repr__(self) -> str:
        """String representation of cache state."""
        stats = self.get_stats()
        return (
            f"TileDataCache("
            f"size={stats['size']}/{stats['max_size']}, "
            f"hit_rate={stats['hit_rate_percent']:.1f}%"
            f")"
        )


# Global cache instance (optional - for simple usage)
_global_cache: Optional[TileDataCache] = None


def get_global_cache(max_tiles: int = 2) -> TileDataCache:
    """
    Get or create global tile cache instance.
    
    Args:
        max_tiles: Maximum tiles to cache (only used on first call)
        
    Returns:
        Global TileDataCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = TileDataCache(max_tiles=max_tiles)
    return _global_cache


def clear_global_cache() -> None:
    """Clear and reset global cache."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()
        _global_cache = None
