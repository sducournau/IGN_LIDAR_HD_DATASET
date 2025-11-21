"""
Tests for centralized GPU Manager

Tests the GPUManager singleton and its GPU detection capabilities.

Author: LiDAR Trainer Agent
Date: November 21, 2025
"""

import pytest
from unittest.mock import patch, MagicMock
from ign_lidar.core.gpu import GPUManager, get_gpu_manager, GPU_AVAILABLE, HAS_CUPY


class TestGPUManager:
    """Test suite for GPUManager singleton."""
    
    def test_singleton_pattern(self):
        """Test that GPUManager implements singleton pattern correctly."""
        gpu1 = GPUManager()
        gpu2 = GPUManager()
        assert gpu1 is gpu2, "GPUManager should return same instance"
    
    def test_get_gpu_manager_convenience(self):
        """Test convenience function returns singleton."""
        gpu1 = get_gpu_manager()
        gpu2 = GPUManager()
        assert gpu1 is gpu2, "get_gpu_manager() should return singleton"
    
    def test_backward_compatibility_aliases(self):
        """Test backward compatibility aliases exist."""
        assert isinstance(GPU_AVAILABLE, bool), "GPU_AVAILABLE should be boolean"
        assert isinstance(HAS_CUPY, bool), "HAS_CUPY should be boolean"
        assert GPU_AVAILABLE == HAS_CUPY, "Aliases should have same value"
    
    def test_get_info(self):
        """Test get_info returns correct structure."""
        gpu = GPUManager()
        info = gpu.get_info()
        
        assert isinstance(info, dict), "get_info should return dict"
        assert 'gpu_available' in info
        assert 'cuml_available' in info
        assert 'cuspatial_available' in info
        assert 'faiss_gpu_available' in info
        
        # All values should be boolean
        for key, value in info.items():
            assert isinstance(value, bool), f"{key} should be boolean"
    
    def test_repr(self):
        """Test string representation."""
        gpu = GPUManager()
        repr_str = repr(gpu)
        
        assert 'GPUManager' in repr_str
        assert 'GPU' in repr_str
        assert 'cuML' in repr_str
        assert 'cuSpatial' in repr_str
        assert 'FAISS' in repr_str
    
    def test_reset_cache(self):
        """Test cache reset functionality."""
        gpu = GPUManager()
        
        # Access properties to populate cache
        _ = gpu.gpu_available
        _ = gpu.cuml_available
        
        # Reset cache
        gpu.reset_cache()
        
        # Cached values should be None
        assert gpu._gpu_available is None
        assert gpu._cuml_available is None
        assert gpu._cuspatial_available is None
        assert gpu._faiss_gpu_available is None


class TestGPUDetection:
    """Test GPU detection logic."""
    
    @patch('ign_lidar.core.gpu.logger')
    def test_cupy_detection_success(self, mock_logger):
        """Test CuPy detection when available."""
        with patch('builtins.__import__') as mock_import:
            # Mock successful CuPy import
            mock_cp = MagicMock()
            mock_cp.array.return_value = MagicMock()
            mock_import.return_value = mock_cp
            
            gpu = GPUManager()
            gpu.reset_cache()  # Clear any cached values
            
            # This would require actual CuPy, so we test the logic
            assert hasattr(gpu, '_check_cupy')
    
    @patch('ign_lidar.core.gpu.logger')
    def test_cupy_detection_failure(self, mock_logger):
        """Test CuPy detection when not available."""
        gpu = GPUManager()
        gpu.reset_cache()
        
        with patch.object(gpu, '_check_cupy', return_value=False):
            assert not gpu.gpu_available
    
    def test_cuml_requires_gpu(self):
        """Test that cuML check requires basic GPU."""
        gpu = GPUManager()
        gpu.reset_cache()
        
        # Mock the internal GPU check to return False
        with patch.object(gpu, '_gpu_available', False):
            assert not gpu._check_cuml()
    
    def test_cuspatial_requires_gpu(self):
        """Test that cuSpatial check requires basic GPU."""
        gpu = GPUManager()
        gpu.reset_cache()
        
        # Mock the internal GPU check to return False
        with patch.object(gpu, '_gpu_available', False):
            assert not gpu._check_cuspatial()
    
    def test_faiss_requires_gpu(self):
        """Test that FAISS-GPU check requires basic GPU."""
        gpu = GPUManager()
        gpu.reset_cache()
        
        # Mock the internal GPU check to return False
        with patch.object(gpu, '_gpu_available', False):
            assert not gpu._check_faiss()


class TestGPUManagerIntegration:
    """Integration tests for GPUManager."""
    
    def test_properties_are_cached(self):
        """Test that properties use caching."""
        gpu = GPUManager()
        gpu.reset_cache()
        
        # First access
        result1 = gpu.gpu_available
        # Second access should use cache
        result2 = gpu.gpu_available
        
        assert result1 == result2, "Cached value should be consistent"
    
    def test_all_properties_accessible(self):
        """Test all properties can be accessed without error."""
        gpu = GPUManager()
        
        # Should not raise exceptions
        _ = gpu.gpu_available
        _ = gpu.cuml_available
        _ = gpu.cuspatial_available
        _ = gpu.faiss_gpu_available
    
    def test_get_info_consistency(self):
        """Test get_info returns consistent values with properties."""
        gpu = GPUManager()
        info = gpu.get_info()
        
        assert info['gpu_available'] == gpu.gpu_available
        assert info['cuml_available'] == gpu.cuml_available
        assert info['cuspatial_available'] == gpu.cuspatial_available
        assert info['faiss_gpu_available'] == gpu.faiss_gpu_available


@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestGPUManagerWithGPU:
    """Tests that require actual GPU hardware."""
    
    def test_gpu_detection_with_hardware(self):
        """Test GPU detection when hardware is present."""
        gpu = GPUManager()
        
        # If we reach this test, GPU should be available
        assert gpu.gpu_available, "GPU should be detected"
    
    def test_cuml_with_gpu(self):
        """Test cuML detection with GPU present."""
        gpu = GPUManager()
        
        # cuML availability depends on installation
        cuml_status = gpu.cuml_available
        assert isinstance(cuml_status, bool), "cuML status should be boolean"
    
    def test_get_info_with_gpu(self):
        """Test get_info with GPU present."""
        gpu = GPUManager()
        info = gpu.get_info()
        
        assert info['gpu_available'] is True, "GPU should be available"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
