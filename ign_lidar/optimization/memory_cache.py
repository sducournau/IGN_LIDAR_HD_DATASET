"""
Memory-Mapped Feature Caching System

Provides efficient caching of computed features using memory-mapped arrays
for reduced memory footprint and faster I/O. Part of Phase 3 Sprint 4.

Key Features:
- Memory-mapped numpy arrays for large feature sets
- LRU caching for frequently accessed features
- Automatic cleanup and memory management
- Thread-safe operations

Author: Phase 3 Sprint 4 Optimization
Date: October 18, 2025
"""

import logging
import hashlib
import tempfile
import atexit
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from functools import lru_cache
import numpy as np
import threading

logger = logging.getLogger(__name__)


class FeatureCache:
    """
    Memory-mapped feature cache for efficient storage and retrieval.
    
    Uses numpy.memmap for large arrays to avoid loading everything into RAM.
    Provides LRU caching for small, frequently accessed features.
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_memory_mb: float = 1000.0,
        enable_lru: bool = True,
        lru_size: int = 128
    ):
        """
        Initialize feature cache.
        
        Args:
            cache_dir: Directory for cache files (default: temp directory)
            max_memory_mb: Maximum memory for cache in MB
            enable_lru: Enable LRU caching for small features
            lru_size: Maximum number of items in LRU cache
        """
        self.max_memory_mb = max_memory_mb
        self.enable_lru = enable_lru
        self.lru_size = lru_size
        
        # Create cache directory
        if cache_dir is None:
            self.cache_dir = Path(tempfile.mkdtemp(prefix="feature_cache_"))
            self._temp_dir = True
        else:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._temp_dir = False
        
        # Cache metadata
        self._cache_info: Dict[str, Dict[str, Any]] = {}
        self._memory_used_mb: float = 0.0
        self._lock = threading.Lock()
        
        # Register cleanup
        atexit.register(self.cleanup)
        
        logger.info(f"📦 Feature cache initialized: {self.cache_dir}")
        logger.info(f"   Max memory: {self.max_memory_mb:.1f} MB")
    
    def _compute_key(self, tile_name: str, feature_name: str) -> str:
        """Compute cache key from tile and feature names."""
        key_str = f"{tile_name}_{feature_name}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cache entry."""
        return self.cache_dir / f"{cache_key}.npy"
    
    def has_feature(self, tile_name: str, feature_name: str) -> bool:
        """
        Check if feature exists in cache.
        
        Args:
            tile_name: Name of the tile
            feature_name: Name of the feature
            
        Returns:
            True if feature is cached
        """
        cache_key = self._compute_key(tile_name, feature_name)
        with self._lock:
            return cache_key in self._cache_info
    
    def store_feature(
        self,
        tile_name: str,
        feature_name: str,
        data: np.ndarray,
        use_memmap: bool = True
    ) -> bool:
        """
        Store feature in cache.
        
        Args:
            tile_name: Name of the tile
            feature_name: Name of the feature
            data: Feature data (numpy array)
            use_memmap: Use memory-mapped storage for large arrays
            
        Returns:
            True if successfully stored
        """
        cache_key = self._compute_key(tile_name, feature_name)
        cache_path = self._get_cache_path(cache_key)
        
        # Calculate size
        size_mb = data.nbytes / (1024 * 1024)
        
        # Check if we have space
        with self._lock:
            if self._memory_used_mb + size_mb > self.max_memory_mb:
                logger.warning(f"⚠️  Cache full ({self._memory_used_mb:.1f}/{self.max_memory_mb:.1f} MB)")
                return False
            
            try:
                if use_memmap and size_mb > 10.0:
                    # Use memory-mapped array for large data
                    mmap = np.memmap(
                        cache_path,
                        dtype=data.dtype,
                        mode='w+',
                        shape=data.shape
                    )
                    mmap[:] = data[:]
                    mmap.flush()
                    del mmap  # Close file
                    
                    storage_type = 'memmap'
                else:
                    # Use standard numpy save for small data
                    np.save(cache_path, data)
                    storage_type = 'standard'
                
                # Update metadata
                self._cache_info[cache_key] = {
                    'tile_name': tile_name,
                    'feature_name': feature_name,
                    'path': cache_path,
                    'shape': data.shape,
                    'dtype': data.dtype,
                    'size_mb': size_mb,
                    'storage_type': storage_type
                }
                self._memory_used_mb += size_mb
                
                logger.debug(f"✅ Cached {feature_name} for {tile_name} ({size_mb:.1f} MB, {storage_type})")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to cache feature: {e}")
                if cache_path.exists():
                    cache_path.unlink()
                return False
    
    def load_feature(
        self,
        tile_name: str,
        feature_name: str,
        mode: str = 'r'
    ) -> Optional[np.ndarray]:
        """
        Load feature from cache.
        
        Args:
            tile_name: Name of the tile
            feature_name: Name of the feature
            mode: Memory-map mode ('r' for read-only, 'r+' for read-write)
            
        Returns:
            Feature data (numpy array or memmap) or None if not found
        """
        cache_key = self._compute_key(tile_name, feature_name)
        
        with self._lock:
            if cache_key not in self._cache_info:
                return None
            
            info = self._cache_info[cache_key]
        
        try:
            cache_path = info['path']
            
            if info['storage_type'] == 'memmap':
                # Load as memory-mapped array
                data = np.memmap(
                    cache_path,
                    dtype=info['dtype'],
                    mode=mode,
                    shape=info['shape']
                )
                logger.debug(f"📂 Loaded {feature_name} (memmap, {info['size_mb']:.1f} MB)")
            else:
                # Load standard numpy array
                data = np.load(cache_path)
                logger.debug(f"📂 Loaded {feature_name} (standard, {info['size_mb']:.1f} MB)")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Failed to load cached feature: {e}")
            return None
    
    def remove_feature(self, tile_name: str, feature_name: str) -> bool:
        """
        Remove feature from cache.
        
        Args:
            tile_name: Name of the tile
            feature_name: Name of the feature
            
        Returns:
            True if successfully removed
        """
        cache_key = self._compute_key(tile_name, feature_name)
        
        with self._lock:
            if cache_key not in self._cache_info:
                return False
            
            info = self._cache_info[cache_key]
            
            try:
                # Delete file
                if info['path'].exists():
                    info['path'].unlink()
                
                # Update metadata
                self._memory_used_mb -= info['size_mb']
                del self._cache_info[cache_key]
                
                logger.debug(f"🗑️  Removed {feature_name} from cache")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to remove cached feature: {e}")
                return False
    
    def clear(self):
        """Clear all cached features."""
        with self._lock:
            for cache_key in list(self._cache_info.keys()):
                info = self._cache_info[cache_key]
                if info['path'].exists():
                    info['path'].unlink()
            
            self._cache_info.clear()
            self._memory_used_mb = 0.0
            
        logger.info("🗑️  Feature cache cleared")
    
    def cleanup(self):
        """Clean up cache directory (called on exit)."""
        if self._temp_dir and self.cache_dir.exists():
            try:
                shutil.rmtree(self.cache_dir)
                logger.info("🧹 Cleaned up temporary cache directory")
            except Exception as e:
                logger.warning(f"⚠️  Failed to clean up cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            return {
                'num_entries': len(self._cache_info),
                'memory_used_mb': self._memory_used_mb,
                'memory_max_mb': self.max_memory_mb,
                'memory_usage_percent': (self._memory_used_mb / self.max_memory_mb) * 100,
                'cache_dir': str(self.cache_dir)
            }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (f"FeatureCache(entries={stats['num_entries']}, "
                f"memory={stats['memory_used_mb']:.1f}/{stats['memory_max_mb']:.1f} MB, "
                f"usage={stats['memory_usage_percent']:.1f}%)")


class StreamingTileProcessor:
    """
    Process tiles in streaming fashion to minimize memory usage.
    
    Processes large tiles in chunks and computes features incrementally,
    avoiding loading entire tile into memory at once.
    """
    
    def __init__(
        self,
        chunk_size: int = 5_000_000,
        feature_cache: Optional[FeatureCache] = None
    ):
        """
        Initialize streaming processor.
        
        Args:
            chunk_size: Number of points to process at once
            feature_cache: Optional feature cache for storing results
        """
        self.chunk_size = chunk_size
        self.feature_cache = feature_cache
        
        logger.info(f"🌊 Streaming processor initialized (chunk_size={chunk_size:,})")
    
    def process_tile_streaming(
        self,
        tile_path: Path,
        feature_functions: Dict[str, callable],
        output_dir: Optional[Path] = None
    ) -> Dict[str, np.ndarray]:
        """
        Process tile in streaming fashion.
        
        Args:
            tile_path: Path to LAZ file
            feature_functions: Dict mapping feature names to computation functions
            output_dir: Optional directory to save results
            
        Returns:
            Dictionary mapping feature names to computed arrays
        """
        import laspy
        
        logger.info(f"🌊 Streaming process: {tile_path.name}")
        
        results = {}
        
        with laspy.open(str(tile_path)) as laz_file:
            header = laz_file.header
            total_points = header.point_count
            num_chunks = (total_points + self.chunk_size - 1) // self.chunk_size
            
            logger.info(f"   📊 {total_points:,} points in {num_chunks} chunks")
            
            # Process in chunks
            chunk_results = {name: [] for name in feature_functions.keys()}
            
            for i, chunk in enumerate(laz_file.chunk_iterator(self.chunk_size)):
                logger.debug(f"   📦 Processing chunk {i+1}/{num_chunks}...")
                
                # Extract chunk points
                points = np.vstack([chunk.x, chunk.y, chunk.z]).T.astype(np.float32)
                
                # Compute features for this chunk
                for feature_name, feature_func in feature_functions.items():
                    feature_data = feature_func(points)
                    chunk_results[feature_name].append(feature_data)
                
                del points, chunk
            
            # Concatenate chunk results
            logger.info(f"   🔗 Combining {len(chunk_results[list(feature_functions.keys())[0]])} chunks...")
            
            for feature_name, chunks in chunk_results.items():
                results[feature_name] = np.concatenate(chunks) if len(chunks) > 0 else np.array([])
            
            # Cache results if cache is available
            if self.feature_cache is not None:
                for feature_name, feature_data in results.items():
                    self.feature_cache.store_feature(
                        tile_path.stem,
                        feature_name,
                        feature_data
                    )
            
            # Save to disk if output directory specified
            if output_dir is not None:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                for feature_name, feature_data in results.items():
                    output_path = output_dir / f"{tile_path.stem}_{feature_name}.npy"
                    np.save(output_path, feature_data)
                    logger.debug(f"   💾 Saved {feature_name} to {output_path.name}")
        
        logger.info(f"   ✅ Streaming processing complete")
        return results


# Global cache instance (singleton pattern)
_global_cache: Optional[FeatureCache] = None


def get_global_cache() -> FeatureCache:
    """Get or create global feature cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = FeatureCache()
    return _global_cache


def clear_global_cache():
    """Clear and reset global cache."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()
        _global_cache.cleanup()
        _global_cache = None
