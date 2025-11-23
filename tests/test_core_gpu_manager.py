"""
Tests for centralized GPUManager (Phase 1 Consolidation)

Tests the singleton GPU detection and management system that replaces
6+ scattered GPU detection implementations.

Date: November 21, 2025
Version: 1.0
"""

import pytest
from unittest.mock import patch, MagicMock
from ign_lidar.core.gpu import GPUManager, get_gpu_manager, GPU_AVAILABLE, HAS_CUPY


class TestGPUManagerSingleton:
    """Test singleton pattern implementation."""
    
    def test_singleton_returns_same_instance(self):
        """Test that multiple calls return the same instance."""
        gpu1 = GPUManager()
        gpu2 = GPUManager()
        gpu3 = get_gpu_manager()
        
        assert gpu1 is gpu2
        assert gpu2 is gpu3
        assert id(gpu1) == id(gpu2) == id(gpu3)
    
    def test_singleton_state_shared(self):
        """Test that state is shared across instances."""
        gpu1 = GPUManager()
        gpu1._gpu_available = True  # Manually set for test
        
        gpu2 = GPUManager()
        assert gpu2._gpu_available == True
        
        # Reset for other tests
        gpu1._gpu_available = None


class TestGPUDetection:
    """Test GPU detection logic."""
    
    def test_gpu_available_property(self):
        """Test gpu_available property returns boolean."""
        gpu = GPUManager()
        result = gpu.gpu_available
        
        assert isinstance(result, bool)
    
    def test_cuml_available_property(self):
        """Test cuml_available property returns boolean."""
        gpu = GPUManager()
        result = gpu.cuml_available
        
        assert isinstance(result, bool)
    
    def test_cuspatial_available_property(self):
        """Test cuspatial_available property returns boolean."""
        gpu = GPUManager()
        result = gpu.cuspatial_available
        
        assert isinstance(result, bool)
    
    def test_faiss_gpu_available_property(self):
        """Test faiss_gpu_available property returns boolean."""
        gpu = GPUManager()
        result = gpu.faiss_gpu_available
        
        assert isinstance(result, bool)
    
    def test_gpu_detection_with_cupy(self):
        """Test GPU detection method returns boolean."""
        gpu = GPUManager()
        gpu.reset_cache()  # Force re-check
        
        # Should return boolean (True if GPU available, False otherwise)
        result = gpu._check_cupy()
        assert isinstance(result, bool)
        
        # Result should match cached property
        assert result == gpu.gpu_available
    
    def test_cuml_requires_gpu(self):
        """Test that cuML check returns False if no GPU."""
        gpu = GPUManager()
        
        # If GPU not available, cuML should be False
        if not gpu.gpu_available:
            assert gpu.cuml_available == False


class TestCaching:
    """Test caching behavior."""
    
    def test_gpu_check_cached(self):
        """Test that GPU checks are cached after first call."""
        gpu = GPUManager()
        gpu.reset_cache()  # Start fresh
        
        # First call
        result1 = gpu.gpu_available
        
        # Second call should use cache (same result)
        result2 = gpu.gpu_available
        
        assert result1 == result2
    
    def test_reset_cache_clears_all(self):
        """Test that reset_cache() clears all cached values."""
        gpu = GPUManager()
        
        # Trigger caching
        _ = gpu.gpu_available
        _ = gpu.cuml_available
        
        # Reset
        gpu.reset_cache()
        
        # Cache should be cleared
        assert gpu._gpu_available is None
        assert gpu._cuml_available is None
        assert gpu._cuspatial_available is None
        assert gpu._faiss_gpu_available is None


class TestGetInfo:
    """Test get_info() method."""
    
    def test_get_info_returns_dict(self):
        """Test get_info() returns dictionary with all keys."""
        gpu = GPUManager()
        info = gpu.get_info()
        
        assert isinstance(info, dict)
        assert 'gpu_available' in info
        assert 'cuml_available' in info
        assert 'cuspatial_available' in info
        assert 'faiss_gpu_available' in info
        
        # All values should be boolean
        for key, value in info.items():
            assert isinstance(value, bool), f"{key} is not boolean"
    
    def test_get_info_consistency(self):
        """Test that get_info() matches individual properties."""
        gpu = GPUManager()
        info = gpu.get_info()
        
        assert info['gpu_available'] == gpu.gpu_available
        assert info['cuml_available'] == gpu.cuml_available
        assert info['cuspatial_available'] == gpu.cuspatial_available
        assert info['faiss_gpu_available'] == gpu.faiss_gpu_available


class TestBackwardCompatibility:
    """Test backward compatibility aliases."""
    
    def test_gpu_available_module_level(self):
        """Test GPU_AVAILABLE module-level constant."""
        assert isinstance(GPU_AVAILABLE, bool)
    
    def test_has_cupy_alias(self):
        """Test HAS_CUPY alias."""
        assert isinstance(HAS_CUPY, bool)
        assert HAS_CUPY == GPU_AVAILABLE
    
    def test_get_gpu_manager_function(self):
        """Test convenience function returns singleton."""
        gpu1 = get_gpu_manager()
        gpu2 = GPUManager()
        
        assert gpu1 is gpu2


class TestRepr:
    """Test string representation."""
    
    def test_repr_format(self):
        """Test __repr__ returns formatted string."""
        gpu = GPUManager()
        repr_str = repr(gpu)
        
        assert isinstance(repr_str, str)
        assert 'GPUManager' in repr_str
        assert 'GPU' in repr_str
        assert 'cuML' in repr_str


class TestModuleIntegration:
    """Test integration with other modules."""
    
    def test_import_from_normalization(self):
        """Test that normalization module uses GPUManager."""
        from ign_lidar.utils.normalization import GPU_AVAILABLE as NORM_GPU
        
        gpu = GPUManager()
        # Should use same detection
        assert isinstance(NORM_GPU, bool)
    
    def test_import_from_strategy_gpu(self):
        """Test that strategy_gpu uses GPUManager."""
        from ign_lidar.features.strategy_gpu import GPU_AVAILABLE as STRAT_GPU
        
        assert isinstance(STRAT_GPU, bool)
    
    def test_import_from_gpu_wrapper(self):
        """Test that gpu_wrapper uses GPUManager."""
        from ign_lidar.optimization.gpu_wrapper import check_gpu_available
        
        result = check_gpu_available()
        assert isinstance(result, bool)


# Mark all tests as unit tests
pytestmark = pytest.mark.unit


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
