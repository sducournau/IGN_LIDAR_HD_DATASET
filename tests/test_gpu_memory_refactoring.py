"""
Unit tests for centralized GPU memory management and FAISS utilities.

Tests the Phase 1 refactoring:
- GPUMemoryManager singleton
- Memory allocation checking
- Cache cleanup
- FAISS temp memory calculation
- FAISS index creation

Author: LiDAR Trainer Agent (Phase 1: GPU Bottlenecks)
Date: November 21, 2025
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestGPUMemoryManager:
    """Tests for centralized GPU memory management."""
    
    def test_singleton_pattern(self):
        """Test that GPUMemoryManager is a singleton."""
        from ign_lidar.core.gpu_memory import GPUMemoryManager, get_gpu_memory_manager
        
        instance1 = GPUMemoryManager()
        instance2 = GPUMemoryManager()
        instance3 = get_gpu_memory_manager()
        
        assert instance1 is instance2
        assert instance2 is instance3
        assert id(instance1) == id(instance2) == id(instance3)
    
    def test_gpu_not_available(self):
        """Test behavior when GPU is not available."""
        from ign_lidar.core.gpu_memory import GPUMemoryManager
        
        # Create fresh instance
        manager = GPUMemoryManager()
        
        if not manager.gpu_available:
            # When no GPU, methods should return safe defaults
            assert manager.get_available_memory() == 0.0
            assert manager.get_total_memory() == 0.0
            assert manager.get_used_memory() == 0.0
            assert manager.allocate(1.0) is False
            
            # Cleanup should not crash
            manager.free_cache()  # Should not raise
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not pytest.importorskip("cupy", reason="CuPy not available"), reason="GPU not available")
    def test_memory_info(self):
        """Test memory info retrieval (GPU required)."""
        from ign_lidar.core.gpu_memory import get_gpu_memory_manager
        
        gpu_mem = get_gpu_memory_manager()
        
        if not gpu_mem.gpu_available:
            pytest.skip("GPU not available")
        
        # Get memory info
        used, available, total = gpu_mem.get_memory_info()
        
        # Sanity checks
        assert used >= 0.0
        assert available >= 0.0
        assert total > 0.0
        assert used <= total
        assert available <= total
        assert abs((used + available) - total) < 0.1  # Allow small discrepancy
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not pytest.importorskip("cupy", reason="CuPy not available"), reason="GPU not available")
    def test_allocation_check(self):
        """Test allocation checking (GPU required)."""
        from ign_lidar.core.gpu_memory import get_gpu_memory_manager
        
        gpu_mem = get_gpu_memory_manager()
        
        if not gpu_mem.gpu_available:
            pytest.skip("GPU not available")
        
        # Small allocation should succeed
        assert gpu_mem.allocate(0.1) is True  # 100 MB
        
        # Huge allocation should fail
        assert gpu_mem.allocate(999999) is False  # 999 TB
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not pytest.importorskip("cupy", reason="CuPy not available"), reason="GPU not available")
    def test_cache_cleanup(self):
        """Test GPU cache cleanup (GPU required)."""
        from ign_lidar.core.gpu_memory import get_gpu_memory_manager
        import cupy as cp
        
        gpu_mem = get_gpu_memory_manager()
        
        if not gpu_mem.gpu_available:
            pytest.skip("GPU not available")
        
        # Allocate some memory
        arr = cp.zeros((1000, 1000), dtype=cp.float32)
        used_before = gpu_mem.get_used_memory()
        
        # Free the array
        del arr
        
        # Cleanup cache
        gpu_mem.free_cache()
        
        used_after = gpu_mem.get_used_memory()
        
        # Memory usage should decrease (or at least not increase)
        assert used_after <= used_before
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not pytest.importorskip("cupy", reason="CuPy not available"), reason="GPU not available")
    def test_memory_limit(self):
        """Test setting memory limit (GPU required)."""
        from ign_lidar.core.gpu_memory import get_gpu_memory_manager
        
        gpu_mem = get_gpu_memory_manager()
        
        if not gpu_mem.gpu_available:
            pytest.skip("GPU not available")
        
        # Set limit (should not crash)
        gpu_mem.set_memory_limit(4.0)  # 4 GB
        
        # Remove limit (should not crash)
        gpu_mem.set_memory_limit(None)
    
    def test_usage_percentage(self):
        """Test usage percentage calculation."""
        from ign_lidar.core.gpu_memory import get_gpu_memory_manager
        
        gpu_mem = get_gpu_memory_manager()
        
        usage = gpu_mem.get_usage_percentage()
        
        if gpu_mem.gpu_available:
            assert 0.0 <= usage <= 100.0
        else:
            assert usage == 0.0
    
    def test_repr(self):
        """Test string representation."""
        from ign_lidar.core.gpu_memory import get_gpu_memory_manager
        
        gpu_mem = get_gpu_memory_manager()
        repr_str = repr(gpu_mem)
        
        assert "GPUMemoryManager" in repr_str
        
        if gpu_mem.gpu_available:
            assert "GB" in repr_str
            assert "usage=" in repr_str


class TestFAISSUtils:
    """Tests for FAISS utilities."""
    
    def test_faiss_imports(self):
        """Test FAISS import availability."""
        from ign_lidar.optimization.faiss_utils import HAS_FAISS, HAS_FAISS_GPU
        
        # Should be boolean
        assert isinstance(HAS_FAISS, bool)
        assert isinstance(HAS_FAISS_GPU, bool)
        
        # GPU requires CPU
        if HAS_FAISS_GPU:
            assert HAS_FAISS is True
    
    def test_calculate_temp_memory_basic(self):
        """Test basic temp memory calculation."""
        from ign_lidar.optimization.faiss_utils import calculate_faiss_temp_memory
        
        # Small dataset
        temp_bytes = calculate_faiss_temp_memory(
            n_points=10_000,
            k=30
        )
        
        assert temp_bytes > 0
        assert temp_bytes >= 128 * 1024**2  # At least 128 MB
        assert temp_bytes <= 1024**3  # At most 1 GB
    
    def test_calculate_temp_memory_large(self):
        """Test temp memory for large dataset."""
        from ign_lidar.optimization.faiss_utils import calculate_faiss_temp_memory
        
        # Large dataset
        temp_bytes = calculate_faiss_temp_memory(
            n_points=10_000_000,
            k=30
        )
        
        temp_gb = temp_bytes / (1024**3)
        
        assert temp_bytes > 0
        assert temp_gb <= 1.0  # Capped at 1 GB
    
    def test_calculate_temp_memory_with_queries(self):
        """Test temp memory with separate query set."""
        from ign_lidar.optimization.faiss_utils import calculate_faiss_temp_memory
        
        # Small query set
        temp_bytes = calculate_faiss_temp_memory(
            n_points=1_000_000,
            k=30,
            n_queries=1000  # Only 1000 queries
        )
        
        temp_gb = temp_bytes / (1024**3)
        
        assert temp_bytes > 0
        # Should be smaller than self-query case
        assert temp_gb < 1.0
    
    def test_calculate_temp_memory_safety_factor(self):
        """Test safety factor in temp memory calculation."""
        from ign_lidar.optimization.faiss_utils import calculate_faiss_temp_memory
        
        n_points = 100_000
        k = 30
        
        # Conservative safety factor
        temp_conservative = calculate_faiss_temp_memory(
            n_points, k, safety_factor=0.1
        )
        
        # Aggressive safety factor
        temp_aggressive = calculate_faiss_temp_memory(
            n_points, k, safety_factor=0.5
        )
        
        # More aggressive should request more memory
        assert temp_aggressive >= temp_conservative
    
    def test_select_index_type_small(self):
        """Test index type selection for small dataset."""
        from ign_lidar.optimization.faiss_utils import select_faiss_index_type
        
        index_type = select_faiss_index_type(
            n_points=50_000,
            n_dims=3
        )
        
        assert index_type == 'flat'  # Small dataset → exact search
    
    def test_select_index_type_large(self):
        """Test index type selection for large dataset."""
        from ign_lidar.optimization.faiss_utils import select_faiss_index_type
        
        index_type = select_faiss_index_type(
            n_points=2_000_000,
            n_dims=3
        )
        
        assert index_type == 'ivf'  # Large dataset → approximate search
    
    def test_calculate_ivf_nlist(self):
        """Test IVF nlist calculation."""
        from ign_lidar.optimization.faiss_utils import calculate_ivf_nlist
        
        # Small dataset
        nlist_small = calculate_ivf_nlist(n_points=100_000)
        assert 16 <= nlist_small <= 65536
        
        # Large dataset
        nlist_large = calculate_ivf_nlist(n_points=10_000_000)
        assert 16 <= nlist_large <= 65536
        
        # Larger dataset should have more clusters
        assert nlist_large >= nlist_small
    
    @pytest.mark.skipif(not pytest.importorskip("faiss", reason="FAISS not available"), reason="FAISS not available")
    def test_create_faiss_index_cpu(self):
        """Test FAISS CPU index creation."""
        from ign_lidar.optimization.faiss_utils import create_faiss_index
        
        # Create small CPU index
        index, res = create_faiss_index(
            n_dims=3,
            n_points=1000,
            use_gpu=False,
            approximate=False
        )
        
        assert index is not None
        assert res is None  # No GPU resources for CPU index
        assert index.d == 3  # 3 dimensions
    
    @pytest.mark.gpu
    @pytest.mark.skipif(
        not pytest.importorskip("faiss", reason="FAISS not available") or 
        not hasattr(__import__("faiss"), "StandardGpuResources"),
        reason="FAISS-GPU not available"
    )
    def test_create_faiss_index_gpu(self):
        """Test FAISS GPU index creation (GPU required)."""
        from ign_lidar.optimization.faiss_utils import create_faiss_index, HAS_FAISS_GPU
        
        if not HAS_FAISS_GPU:
            pytest.skip("FAISS-GPU not available")
        
        # Create small GPU index
        index, res = create_faiss_index(
            n_dims=3,
            n_points=1000,
            use_gpu=True,
            approximate=False
        )
        
        assert index is not None
        assert res is not None  # GPU resources should be provided
        assert index.d == 3
    
    @pytest.mark.skipif(not pytest.importorskip("faiss", reason="FAISS not available"), reason="FAISS not available")
    def test_create_faiss_gpu_resources(self):
        """Test FAISS GPU resources creation."""
        from ign_lidar.optimization.faiss_utils import create_faiss_gpu_resources, HAS_FAISS_GPU
        
        if not HAS_FAISS_GPU:
            pytest.skip("FAISS-GPU not available")
        
        # Auto-calculate temp memory
        res = create_faiss_gpu_resources(n_points=100_000, k=30)
        assert res is not None
        
        # Manual temp memory
        res = create_faiss_gpu_resources(temp_memory_bytes=256 * 1024**2)
        assert res is not None


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_cleanup_gpu_memory(self):
        """Test cleanup_gpu_memory convenience function."""
        from ign_lidar.core.gpu_memory import cleanup_gpu_memory
        
        # Should not crash
        cleanup_gpu_memory()
    
    def test_check_gpu_memory(self):
        """Test check_gpu_memory convenience function."""
        from ign_lidar.core.gpu_memory import check_gpu_memory
        
        # Small allocation
        result = check_gpu_memory(0.1)
        assert isinstance(result, bool)
        
        # Huge allocation
        result = check_gpu_memory(999999)
        assert result is False


class TestBackwardCompatibility:
    """Tests for backward compatibility."""
    
    def test_gpu_memory_import(self):
        """Test GPUMemoryManager can be imported from core."""
        from ign_lidar.core import GPUMemoryManager, get_gpu_memory_manager
        
        assert GPUMemoryManager is not None
        assert get_gpu_memory_manager is not None
        
        # Should return singleton
        gpu_mem = get_gpu_memory_manager()
        assert isinstance(gpu_mem, GPUMemoryManager)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
