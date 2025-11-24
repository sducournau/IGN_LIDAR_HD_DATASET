"""
Tests for the unified UnifiedGPUManager.

Tests the new unified GPU management interface that consolidates
GPUManager, GPUMemoryManager, and GPUProfiler.
"""

import pytest
import numpy as np
from ign_lidar.core.gpu_unified import UnifiedGPUManager, get_gpu_manager


class TestUnifiedGPUManager:
    """Tests for UnifiedGPUManager."""

    def test_singleton_pattern(self):
        """Test that UnifiedGPUManager is a singleton."""
        manager1 = UnifiedGPUManager.get_instance()
        manager2 = UnifiedGPUManager.get_instance()
        assert manager1 is manager2

    def test_initialization(self):
        """Test manager initialization."""
        manager = UnifiedGPUManager.get_instance()
        assert manager is not None

    def test_get_instance_via_function(self):
        """Test getting instance via convenience function."""
        manager = get_gpu_manager()
        assert manager is not None
        assert isinstance(manager, UnifiedGPUManager)

    def test_is_available_property(self):
        """Test checking GPU availability."""
        manager = UnifiedGPUManager.get_instance()
        available = manager.is_available
        assert isinstance(available, bool)

    def test_available_memory_gb_property(self):
        """Test getting available memory."""
        manager = UnifiedGPUManager.get_instance()
        memory_gb = manager.available_memory_gb
        assert isinstance(memory_gb, float)
        assert memory_gb >= 0.0

    def test_memory_usage_percent_property(self):
        """Test getting memory usage percentage."""
        manager = UnifiedGPUManager.get_instance()
        usage_percent = manager.memory_usage_percent
        assert isinstance(usage_percent, float)
        assert 0.0 <= usage_percent <= 100.0

    def test_empty_batch_transfer(self):
        """Test batch transfer with empty list."""
        manager = UnifiedGPUManager.get_instance()
        result = manager.transfer_batch([])
        assert result == []

    def test_batch_transfer_none_arrays(self):
        """Test batch transfer with None arrays."""
        manager = UnifiedGPUManager.get_instance()
        arrays = [None, None]
        result = manager.transfer_batch(arrays)
        assert len(result) == 2
        assert result[0] is None
        assert result[1] is None

    def test_cache_array(self):
        """Test caching arrays."""
        if not UnifiedGPUManager.get_instance().is_available:
            pytest.skip("GPU not available")

        manager = UnifiedGPUManager.get_instance()

        # Create dummy GPU array (mock)
        array = np.random.rand(10, 10).astype(np.float32)

        # Should not raise error
        manager.cache_array(array, "test_key")

        # Verify it's retrievable
        cached = manager.get_cached_array("test_key")
        assert cached is not None

    def test_get_cached_array_not_found(self):
        """Test retrieving non-existent cached array."""
        manager = UnifiedGPUManager.get_instance()
        result = manager.get_cached_array("nonexistent_key")
        assert result is None

    def test_clear_cache(self):
        """Test clearing cache."""
        manager = UnifiedGPUManager.get_instance()
        manager.clear_cache()
        # Should not raise error

    def test_cleanup(self):
        """Test GPU cleanup."""
        manager = UnifiedGPUManager.get_instance()
        manager.cleanup()
        # Should not raise error

    def test_get_memory_stats(self):
        """Test getting memory statistics."""
        manager = UnifiedGPUManager.get_instance()
        stats = manager.get_memory_stats()

        assert isinstance(stats, dict)
        assert "available" in stats
        assert "available_memory_gb" in stats
        assert "memory_usage_percent" in stats
        assert "cache_size_mb" in stats
        assert "cache_entries" in stats

    def test_memory_stats_values(self):
        """Test that memory stats have valid values."""
        manager = UnifiedGPUManager.get_instance()
        stats = manager.get_memory_stats()

        assert isinstance(stats["available"], bool)
        assert isinstance(stats["available_memory_gb"], (int, float))
        assert isinstance(stats["memory_usage_percent"], (int, float))
        assert isinstance(stats["cache_size_mb"], (int, float))
        assert isinstance(stats["cache_entries"], int)

    def test_repr(self):
        """Test string representation."""
        manager = UnifiedGPUManager.get_instance()
        repr_str = repr(manager)
        assert "UnifiedGPUManager" in repr_str

    def test_start_stop_profiling(self):
        """Test profiling start/stop."""
        manager = UnifiedGPUManager.get_instance()
        # Should not raise errors
        manager.start_profiling("test_phase")
        manager.end_profiling("test_phase")

    def test_get_profiling_stats(self):
        """Test getting profiling statistics."""
        manager = UnifiedGPUManager.get_instance()
        stats = manager.get_profiling_stats()
        # Should return dict (even if empty)
        assert isinstance(stats, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
