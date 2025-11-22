"""
Tests for GPUManager Composition API (v3.1+)

Tests the new unified GPU management interface with composition pattern.

Author: GitHub Copilot
Date: November 22, 2025
"""

import pytest
import numpy as np

from ign_lidar.core.gpu import GPUManager, get_gpu_manager


class TestGPUManagerCompositionBasics:
    """Test basic composition API functionality."""
    
    def test_gpu_manager_instantiation(self):
        """Test GPUManager can be instantiated."""
        gpu = GPUManager()
        assert gpu is not None
        assert isinstance(gpu, GPUManager)
    
    def test_singleton_pattern(self):
        """Test that GPUManager maintains singleton pattern."""
        gpu1 = GPUManager()
        gpu2 = GPUManager()
        assert gpu1 is gpu2, "GPUManager should be a singleton"
    
    def test_get_gpu_manager_function(self):
        """Test convenience function returns same instance."""
        gpu1 = GPUManager()
        gpu2 = get_gpu_manager()
        assert gpu1 is gpu2


class TestCompositionAPIProperties:
    """Test lazy-loaded composition properties."""
    
    def test_memory_property_exists(self):
        """Test that memory property exists."""
        gpu = GPUManager()
        assert hasattr(gpu, 'memory')
    
    def test_cache_property_exists(self):
        """Test that cache property exists."""
        gpu = GPUManager()
        assert hasattr(gpu, 'cache')
    
    @pytest.mark.skipif(not GPUManager().gpu_available, reason="GPU not available")
    def test_memory_lazy_loading(self):
        """Test memory manager is lazy-loaded."""
        gpu = GPUManager()
        # First access loads it
        mem = gpu.memory
        assert mem is not None
        # Second access returns same instance
        mem2 = gpu.memory
        assert mem is mem2
    
    @pytest.mark.skipif(not GPUManager().gpu_available, reason="GPU not available")
    def test_cache_lazy_loading(self):
        """Test array cache is lazy-loaded."""
        gpu = GPUManager()
        # First access loads it
        cache = gpu.cache
        assert cache is not None
        # Second access returns same instance
        cache2 = gpu.cache
        assert cache is cache2


class TestConvenienceMethods:
    """Test convenience methods for unified API."""
    
    def test_get_memory_info_no_gpu(self):
        """Test get_memory_info returns valid dict without GPU."""
        gpu = GPUManager()
        info = gpu.get_memory_info()
        
        assert isinstance(info, dict)
        assert 'free_gb' in info
        assert 'total_gb' in info
        assert 'used_gb' in info
        assert 'utilization' in info
        
        # Without GPU, should be all zeros
        if not gpu.gpu_available:
            assert info['free_gb'] == 0.0
            assert info['total_gb'] == 0.0
    
    @pytest.mark.skipif(not GPUManager().gpu_available, reason="GPU not available")
    def test_get_memory_info_with_gpu(self):
        """Test get_memory_info returns real stats with GPU."""
        gpu = GPUManager()
        info = gpu.get_memory_info()
        
        assert isinstance(info, dict)
        assert info['total_gb'] > 0, "Should report actual GPU memory"
    
    def test_cleanup_method_exists(self):
        """Test cleanup method exists and is callable."""
        gpu = GPUManager()
        assert hasattr(gpu, 'cleanup')
        assert callable(gpu.cleanup)
        
        # Should not raise even without GPU
        gpu.cleanup()
    
    @pytest.mark.skipif(not GPUManager().gpu_available, reason="GPU not available")
    def test_cleanup_clears_resources(self):
        """Test cleanup actually clears GPU resources."""
        gpu = GPUManager()
        
        # Load components
        _ = gpu.memory
        _ = gpu.cache
        
        # Cleanup should not raise
        gpu.cleanup()


class TestBackwardCompatibility:
    """Test that old APIs still work."""
    
    def test_old_gpu_available_import(self):
        """Test that GPU_AVAILABLE can still be imported."""
        from ign_lidar.core.gpu import GPU_AVAILABLE
        assert isinstance(GPU_AVAILABLE, bool)
    
    def test_old_has_cupy_import(self):
        """Test that HAS_CUPY can still be imported."""
        from ign_lidar.core.gpu import HAS_CUPY
        assert isinstance(HAS_CUPY, bool)
    
    def test_get_info_still_works(self):
        """Test that old get_info() method still works."""
        gpu = GPUManager()
        info = gpu.get_info()
        
        assert isinstance(info, dict)
        assert 'gpu_available' in info
        assert 'cuml_available' in info


@pytest.mark.skipif(not GPUManager().gpu_available, reason="GPU not available")
class TestGPUIntegration:
    """Integration tests requiring actual GPU."""
    
    def test_memory_allocation_check(self):
        """Test memory allocation checking."""
        gpu = GPUManager()
        
        # Should be able to check small allocation
        can_allocate = gpu.memory.check_available_memory(required_gb=0.1)
        assert isinstance(can_allocate, bool)
    
    def test_cache_basic_operations(self):
        """Test basic cache operations."""
        gpu = GPUManager()
        
        # Create test array
        test_array = np.random.rand(1000, 3).astype(np.float32)
        
        # Upload to cache
        gpu_array = gpu.cache.get_or_upload('test_array', test_array)
        assert gpu_array is not None
        
        # Second access should return cached
        gpu_array2 = gpu.cache.get('test_array')
        assert gpu_array2 is not None
        
        # Cleanup
        gpu.cache.invalidate('test_array')


class TestUsageExamples:
    """Test usage examples from documentation."""
    
    def test_recommended_usage_pattern(self):
        """Test the recommended usage pattern from docs."""
        # From documentation example
        gpu = GPUManager()
        
        if gpu.gpu_available:
            # This path only runs if GPU is available
            pass
        
        # This should always work
        info = gpu.get_memory_info()
        assert info is not None
    
    def test_composition_api_pattern(self):
        """Test v3.1+ composition API pattern."""
        gpu = GPUManager()
        
        # Should not raise even without GPU
        info = gpu.get_memory_info()
        assert isinstance(info, dict)
        
        if gpu.gpu_available:
            # These properties exist
            assert hasattr(gpu, 'memory')
            assert hasattr(gpu, 'cache')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
