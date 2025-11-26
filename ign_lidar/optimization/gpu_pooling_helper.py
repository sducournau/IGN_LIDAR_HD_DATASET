#!/usr/bin/env python3
"""
GPU Memory Pooling Helper Classes

Phase 2 Implementation: Centralized pooling management for consistent reuse
across all GPU compute strategies.

This module provides utilities to ensure systematic pooling of GPU memory
across compute operations.
"""

from contextlib import contextmanager
from typing import Dict, Any, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


class GPUPoolingContext:
    """
    Context manager for GPU memory pooling.
    
    Ensures buffers are properly allocated from pool at start,
    used during computation, and returned to pool at end.
    
    Example:
        with GPUPoolingContext(gpu_pool, num_features=5) as ctx:
            buffer = ctx.get_buffer('feature_normals', shape=(1000,))
            # Use buffer...
            # Automatically returned to pool on exit
    """
    
    def __init__(self, gpu_pool, num_features: int, max_feature_size_mb: int = 100):
        """
        Initialize pooling context.
        
        Args:
            gpu_pool: GPUMemoryPool instance
            num_features: Expected number of features to allocate
            max_feature_size_mb: Maximum size per feature buffer
        """
        self.gpu_pool = gpu_pool
        self.num_features = num_features
        self.max_feature_size_mb = max_feature_size_mb
        self.buffers: Dict[str, Any] = {}
        self.stats = {
            'allocations': 0,
            'reuses': 0,
            'total_size_mb': 0.0,
            'peak_size_mb': 0.0,
        }
    
    def get_buffer(self, name: str, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        """
        Get buffer from pool or allocate if needed.
        
        Args:
            name: Unique buffer name
            shape: Buffer shape
            dtype: NumPy dtype
            
        Returns:
            GPU array buffer
        """
        if name in self.buffers:
            self.stats['reuses'] += 1
            logger.debug(f"Reusing buffer {name}")
            return self.buffers[name]
        
        # Allocate new buffer from pool
        try:
            if self.gpu_pool is not None:
                buffer = self.gpu_pool.get_array(shape=shape, dtype=dtype, name=name)
            else:
                # Fallback if no pool (CPU mode)
                buffer = np.zeros(shape, dtype=dtype)
            
            self.buffers[name] = buffer
            self.stats['allocations'] += 1
            
            # Track memory usage
            size_mb = np.prod(shape) * np.dtype(dtype).itemsize / (1024 * 1024)
            self.stats['total_size_mb'] += size_mb
            self.stats['peak_size_mb'] = max(self.stats['peak_size_mb'], self.stats['total_size_mb'])
            
            logger.debug(f"Allocated buffer {name}: {size_mb:.1f}MB")
            return buffer
            
        except Exception as e:
            logger.error(f"Failed to allocate buffer {name}: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pooling statistics."""
        reuse_rate = (
            self.stats['reuses'] / max(1, self.stats['reuses'] + self.stats['allocations'])
        )
        return {
            **self.stats,
            'reuse_rate': reuse_rate,
            'num_buffers': len(self.buffers),
        }
    
    def __enter__(self):
        """Enter context."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and return buffers to pool."""
        for name, buffer in self.buffers.items():
            try:
                if self.gpu_pool is not None:
                    self.gpu_pool.return_array(buffer)
                logger.debug(f"Returned buffer {name} to pool")
            except Exception as e:
                logger.warning(f"Failed to return buffer {name}: {e}")
        
        self.buffers.clear()


@contextmanager
def pooled_features(gpu_pool, feature_names: list, n_points: int, dtype=np.float32):
    """
    Context manager for computing features with pooled GPU memory.
    
    Handles allocation and cleanup of feature buffers.
    
    Example:
        with pooled_features(pool, ['normals', 'curvature', 'height'], 1000) as buffers:
            buffers['normals'] = compute_normals(points)
            buffers['curvature'] = compute_curvature(points)
            # ... work with buffers
    
    Args:
        gpu_pool: GPUMemoryPool instance
        feature_names: List of feature names to allocate
        n_points: Number of points
        dtype: Data type for buffers
        
    Yields:
        Dictionary mapping feature names to GPU arrays
    """
    buffers = {}
    try:
        # Allocate all buffers upfront
        for name in feature_names:
            if gpu_pool is not None:
                buffers[name] = gpu_pool.get_array(
                    shape=(n_points,),
                    dtype=dtype,
                    name=name
                )
            else:
                buffers[name] = np.zeros(n_points, dtype=dtype)
        
        logger.info(f"Allocated {len(buffers)} pooled buffers for {n_points:,} points")
        yield buffers
        
    finally:
        # Return all buffers to pool
        for name, buffer in buffers.items():
            try:
                if gpu_pool is not None:
                    gpu_pool.return_array(buffer)
            except Exception as e:
                logger.warning(f"Failed to return {name}: {e}")


class PoolingStatistics:
    """
    Track GPU pooling statistics for performance monitoring.
    
    Measures:
    - Allocation efficiency (reuse rate)
    - Memory fragmentation
    - Peak usage
    - Performance impact
    """
    
    def __init__(self):
        self.total_allocations = 0
        self.total_reuses = 0
        self.peak_memory_mb = 0.0
        self.total_memory_mb = 0.0
        self.features_computed = 0
    
    def record_allocation(self, size_mb: float):
        """Record a new allocation."""
        self.total_allocations += 1
        self.total_memory_mb += size_mb
        self.peak_memory_mb = max(self.peak_memory_mb, self.total_memory_mb)
    
    def record_reuse(self):
        """Record buffer reuse."""
        self.total_reuses += 1
    
    def record_feature(self):
        """Record computed feature."""
        self.features_computed += 1
    
    @property
    def reuse_rate(self) -> float:
        """Calculate reuse rate (0-1)."""
        total = self.total_allocations + self.total_reuses
        if total == 0:
            return 0.0
        return self.total_reuses / total
    
    @property
    def efficiency(self) -> float:
        """
        Calculate pooling efficiency (0-1).
        1.0 = perfect reuse, 0.0 = no reuse.
        """
        return self.reuse_rate
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'total_allocations': self.total_allocations,
            'total_reuses': self.total_reuses,
            'reuse_rate': f"{self.reuse_rate:.1%}",
            'peak_memory_mb': f"{self.peak_memory_mb:.1f}",
            'avg_feature_size_mb': (
                f"{self.total_memory_mb / max(1, self.features_computed):.1f}"
            ),
            'features_computed': self.features_computed,
        }
    
    def print_summary(self):
        """Print formatted summary."""
        summary = self.get_summary()
        print("\n" + "=" * 60)
        print("GPU MEMORY POOLING STATISTICS")
        print("=" * 60)
        for key, value in summary.items():
            print(f"  {key:.<40} {value}")
        print("=" * 60 + "\n")


# Example usage for Phase 2 implementation
if __name__ == '__main__':
    print("GPU Memory Pooling Helper Classes (Phase 2)")
    print("\nExample patterns for Phase 2 implementation:")
    
    example_code = """
    # Pattern 1: Using GPUPoolingContext
    from ign_lidar.optimization.gpu_cache import GPUMemoryPool
    
    pool = GPUMemoryPool(max_arrays=50, max_size_gb=12.0)
    
    with GPUPoolingContext(pool, num_features=5) as ctx:
        normals = ctx.get_buffer('normals', shape=(1000,))
        curvature = ctx.get_buffer('curvature', shape=(1000,))
        height = ctx.get_buffer('height', shape=(1000,))
        
        # Compute features (buffers reused from pool)
        compute_normals_into_buffer(points, normals)
        compute_curvature_into_buffer(points, curvature)
        
        stats = ctx.get_stats()
        print(f"Reuse rate: {stats['reuse_rate']:.1%}")
    
    # Pattern 2: Using pooled_features context
    with pooled_features(pool, ['normals', 'curvature', 'height'], 1000) as buffers:
        buffers['normals'][:] = compute_normals(points)
        buffers['curvature'][:] = compute_curvature(points)
        
        # All buffers returned automatically
    
    # Pattern 3: Manual pooling with statistics
    stats = PoolingStatistics()
    
    for feature in features:
        buffer = pool.get_array(shape=(1000,), dtype=np.float32)
        stats.record_allocation(buffer.nbytes / (1024*1024))
        
        result = compute_feature(buffer)
        results[feature] = result
        
        pool.return_array(buffer)
        stats.record_reuse()
    
    stats.print_summary()
    """
    
    print(example_code)
    print("\nUsage:")
    print("  1. Import from this module in strategy_gpu.py")
    print("  2. Replace ad-hoc allocation with GPUPoolingContext")
    print("  3. Add PoolingStatistics for monitoring")
    print("  4. Test with large datasets (50M+ points)")
    print("  5. Validate >90% reuse rate achieved")
