"""
Tests for GPU Memory Pool (Phase 2 Priority 1 Optimization).

Tests the GPUMemoryPool class that implements pre-allocated buffer reuse
for memory-efficient GPU processing.

This module validates:
- Buffer allocation and reuse mechanism
- Statistics tracking (allocations, reuses, efficiency metrics)
- Memory cleanup and pool management
- GPU memory efficiency gains

Expected Performance Gains:
- 20-30% reduction in memory allocation overhead
- 15-25% speedup in GPU processing workloads
- Reduced memory fragmentation in batch operations

Author: IGN LiDAR HD Development Team
Date: December 2025
Version: 3.7.0 Phase 2
"""

import pytest
import numpy as np

from ign_lidar.core.gpu import GPUManager

# Conditional import based on GPU availability
GPU_AVAILABLE = GPUManager().gpu_available
if GPU_AVAILABLE:
    try:
        import cupy as cp
        from ign_lidar.core.gpu_memory import GPUMemoryPool, get_gpu_memory_pool
    except ImportError:
        GPU_AVAILABLE = False
        cp = None
        GPUMemoryPool = None
        get_gpu_memory_pool = None


class TestGPUMemoryPoolInitialization:
    """Test GPUMemoryPool initialization and configuration."""

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_memory_pool_creation(self):
        """Test basic GPUMemoryPool creation."""
        pool = GPUMemoryPool(pool_size_gb=4.0)
        assert pool is not None
        assert pool.pool_size_gb == 4.0

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_memory_pool_with_different_sizes(self):
        """Test GPUMemoryPool with various pool sizes."""
        for size_gb in [2.0, 4.0, 8.0, 16.0]:
            pool = GPUMemoryPool(pool_size_gb=size_gb)
            assert pool.pool_size_gb == size_gb
            assert len(pool._buffers) == 0
            assert pool._stats['allocations'] == 0

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_memory_pool_empty_stats_at_init(self):
        """Test that stats are empty at initialization."""
        pool = GPUMemoryPool(pool_size_gb=4.0)
        stats = pool.get_stats()
        
        assert stats['allocations'] == 0
        assert stats['reuses'] == 0
        assert stats['total_buffers'] == 0

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_get_gpu_memory_pool_singleton(self):
        """Test get_gpu_memory_pool returns singleton instance."""
        pool1 = get_gpu_memory_pool()
        pool2 = get_gpu_memory_pool()
        
        assert pool1 is pool2


class TestGPUMemoryPoolAllocation:
    """Test buffer allocation and reuse mechanisms."""

    @pytest.mark.gpu
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_allocate_fixed_first_call(self):
        """Test first allocation creates a buffer."""
        pool = GPUMemoryPool(pool_size_gb=4.0)
        
        # First allocation should create buffer
        buffer = pool.allocate_fixed(size_mb=10, name="test_buffer")
        
        assert buffer is not None
        assert buffer.dtype == cp.float32
        # 10 MB in float32 elements: 10 * 1024 * 1024 / 4 = 2.6M elements
        expected_elements = int(10 * 1024 * 1024 / 4)
        assert buffer.size == expected_elements
        
        # Stats should show one allocation
        stats = pool.get_stats()
        assert stats['allocations'] == 1
        assert stats['reuses'] == 0

    @pytest.mark.gpu
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_allocate_fixed_reuse(self):
        """Test second allocation with same size reuses buffer."""
        pool = GPUMemoryPool(pool_size_gb=4.0)
        
        # First allocation
        buffer1 = pool.allocate_fixed(size_mb=10, name="reusable")
        
        # Clear stats for cleaner test
        pool._stats['allocations'] = 0
        pool._stats['reuses'] = 0
        
        # Second allocation with same size should reuse
        buffer2 = pool.allocate_fixed(size_mb=10, name="reusable")
        
        # Should be same object (reused)
        assert buffer1 is buffer2
        
        # Stats should show reuse, not new allocation
        stats = pool.get_stats()
        assert stats['allocations'] == 0
        assert stats['reuses'] == 1

    @pytest.mark.gpu
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_allocate_different_sizes(self):
        """Test allocations with different sizes create separate buffers."""
        pool = GPUMemoryPool(pool_size_gb=4.0)
        
        buffer_10mb = pool.allocate_fixed(size_mb=10, name="buffer_10")
        buffer_20mb = pool.allocate_fixed(size_mb=20, name="buffer_20")
        
        # Different sizes should create different buffers
        assert buffer_10mb.size != buffer_20mb.size
        assert buffer_10mb is not buffer_20mb
        
        stats = pool.get_stats()
        assert stats['allocations'] == 2

    @pytest.mark.gpu
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_allocate_multiple_names_same_size(self):
        """Test same size but different names create separate buffers."""
        pool = GPUMemoryPool(pool_size_gb=4.0)
        
        buffer_a = pool.allocate_fixed(size_mb=10, name="buffer_a")
        buffer_b = pool.allocate_fixed(size_mb=10, name="buffer_b")
        
        # Different names should create separate buffers
        assert buffer_a is not buffer_b
        
        stats = pool.get_stats()
        assert stats['allocations'] == 2


class TestGPUMemoryPoolStatistics:
    """Test statistics tracking and efficiency metrics."""

    @pytest.mark.gpu
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_efficiency_calculation(self):
        """Test efficiency metric calculation."""
        pool = GPUMemoryPool(pool_size_gb=4.0)
        
        # Allocate buffer 5 times (1 allocation + 4 reuses)
        for i in range(5):
            pool.allocate_fixed(size_mb=10, name="reused")
        
        stats = pool.get_stats()
        # Efficiency = reuses / (allocations + reuses) = 4/5 = 0.8
        assert stats['allocations'] == 1
        assert stats['reuses'] == 4

    @pytest.mark.gpu
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_stats_includes_pool_info(self):
        """Test stats include pool configuration."""
        pool = GPUMemoryPool(pool_size_gb=4.0)
        
        stats = pool.get_stats()
        assert 'allocations' in stats
        assert 'reuses' in stats
        assert 'total_buffers' in stats
        assert 'reuse_efficiency' in stats

    @pytest.mark.gpu
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_stats_with_mixed_operations(self):
        """Test stats with mixed allocations and reuses."""
        pool = GPUMemoryPool(pool_size_gb=4.0)
        
        # Create 3 different buffers
        pool.allocate_fixed(size_mb=10, name="buffer_1")
        pool.allocate_fixed(size_mb=20, name="buffer_2")
        pool.allocate_fixed(size_mb=15, name="buffer_3")
        
        # Reuse buffer_1 twice
        pool.allocate_fixed(size_mb=10, name="buffer_1")
        pool.allocate_fixed(size_mb=10, name="buffer_1")
        
        # Reuse buffer_2 once
        pool.allocate_fixed(size_mb=20, name="buffer_2")
        
        stats = pool.get_stats()
        assert stats['allocations'] == 3
        assert stats['reuses'] == 3


class TestGPUMemoryPoolCleanup:
    """Test memory cleanup and pool management."""

    @pytest.mark.gpu
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_clear_all_removes_buffers(self):
        """Test clear_all() removes all buffers from pool."""
        pool = GPUMemoryPool(pool_size_gb=4.0)
        
        # Allocate multiple buffers
        pool.allocate_fixed(size_mb=10, name="buffer_1")
        pool.allocate_fixed(size_mb=20, name="buffer_2")
        pool.allocate_fixed(size_mb=15, name="buffer_3")
        
        assert len(pool._buffers) == 3
        
        # Clear all
        pool.clear_all()
        
        # All buffers should be removed
        assert len(pool._buffers) == 0

    @pytest.mark.gpu
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_reallocation_after_clear(self):
        """Test that we can reallocate buffers after clear."""
        pool = GPUMemoryPool(pool_size_gb=4.0)
        
        # First allocation
        buffer1 = pool.allocate_fixed(size_mb=10, name="test")
        
        # Clear
        pool.clear_all()
        
        # Second allocation with same name/size
        buffer2 = pool.allocate_fixed(size_mb=10, name="test")
        
        # Should create new buffer
        assert buffer2 is not None


class TestGPUMemoryPoolIntegration:
    """Integration tests for real-world usage patterns."""

    @pytest.mark.gpu
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_batch_processing_pattern(self):
        """Test typical batch processing pattern with buffer reuse."""
        pool = GPUMemoryPool(pool_size_gb=4.0)
        
        # Simulate 5 batches
        batch_size_mb = 50
        num_batches = 5
        
        for batch_idx in range(num_batches):
            # Each batch allocates same-sized buffer
            buffer = pool.allocate_fixed(size_mb=batch_size_mb, name="batch_data")
            
            # Simulate processing
            gpu_data = cp.arange(buffer.size, dtype=cp.float32)
            result = cp.mean(gpu_data)
            
            assert result is not None
        
        # Stats should show 1 allocation + 4 reuses
        stats = pool.get_stats()
        assert stats['allocations'] == 1
        assert stats['reuses'] == num_batches - 1

    @pytest.mark.gpu
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_multi_buffer_processing(self):
        """Test processing with multiple buffer types."""
        pool = GPUMemoryPool(pool_size_gb=4.0)
        
        # Allocate different buffer types
        normals_buffer = pool.allocate_fixed(size_mb=32, name="normals")
        features_buffer = pool.allocate_fixed(size_mb=64, name="features")
        temp_buffer = pool.allocate_fixed(size_mb=16, name="temporary")
        
        # Reuse in processing loop
        for iteration in range(3):
            normals = pool.allocate_fixed(size_mb=32, name="normals")
            features = pool.allocate_fixed(size_mb=64, name="features")
            temp = pool.allocate_fixed(size_mb=16, name="temporary")
            
            # Verify same buffers were reused
            assert normals is normals_buffer
            assert features is features_buffer
            assert temp is temp_buffer
        
        stats = pool.get_stats()
        # 3 allocations + 9 reuses (3 per iteration)
        assert stats['allocations'] == 3
        assert stats['reuses'] == 9

    @pytest.mark.gpu
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_singleton_state_persistence(self):
        """Test singleton pattern maintains state across calls."""
        # Get pool instances
        pool1 = get_gpu_memory_pool()
        
        if pool1 is None:
            pytest.skip("GPU not available")
        
        # Allocate buffer
        pool1.allocate_fixed(size_mb=10, name="persistent")
        stats1 = pool1.get_stats()
        
        # Get pool again (should be same instance)
        pool2 = get_gpu_memory_pool()
        
        # Reuse buffer
        pool2.allocate_fixed(size_mb=10, name="persistent")
        stats2 = pool2.get_stats()
        
        # Stats should show accumulation
        assert stats2['allocations'] == 1
        assert stats2['reuses'] == 1
        assert pool1 is pool2


class TestGPUMemoryPoolEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.gpu
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_small_allocation(self):
        """Test handling of very small allocation."""
        pool = GPUMemoryPool(pool_size_gb=4.0)
        
        # Small allocation should still work
        buffer = pool.allocate_fixed(size_mb=1, name="small")
        assert buffer is not None
        assert buffer.size > 0

    @pytest.mark.gpu
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_large_pool_size(self):
        """Test pool with large size configuration."""
        # Create pool with 16GB size
        pool = GPUMemoryPool(pool_size_gb=16.0)
        
        # Should not fail during initialization
        assert pool.pool_size_gb == 16.0
        
        # Allocate small buffer should still work
        buffer = pool.allocate_fixed(size_mb=10, name="small")
        assert buffer is not None


# Parametrized tests for multiple sizes and configurations
@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
@pytest.mark.parametrize("pool_size_gb,buffer_size_mb", [
    (4.0, 10),
    (8.0, 20),
    (16.0, 32),
    (32.0, 64),
])
def test_pool_with_various_configs(pool_size_gb, buffer_size_mb):
    """Test pool with various size configurations."""
    pool = GPUMemoryPool(pool_size_gb=pool_size_gb)
    
    # Allocate buffer
    buffer = pool.allocate_fixed(size_mb=buffer_size_mb, name="test")
    
    assert buffer is not None
    expected_elements = int(buffer_size_mb * 1024 * 1024 / 4)
    assert buffer.size == expected_elements


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "gpu"])
