"""
Unit tests for automatic mode selection

Tests the ModeSelector class to ensure correct mode selection
based on point cloud size, GPU availability, and memory constraints.

Author: Simon Ducournau / GitHub Copilot
Date: October 18, 2025
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from ign_lidar.features.mode_selector import (
    ModeSelector,
    ComputationMode,
    get_mode_selector
)
from ign_lidar.core.gpu import GPU_AVAILABLE


class TestModeSelector:
    """Test suite for ModeSelector class."""
    
    @pytest.fixture
    def mock_gpu_available(self):
        """Mock GPU availability."""
        with patch('ign_lidar.features.mode_selector.ModeSelector._check_gpu_availability') as mock:
            mock.return_value = True
            yield mock
    
    @pytest.fixture
    def mock_gpu_unavailable(self):
        """Mock GPU unavailability."""
        with patch('ign_lidar.features.mode_selector.ModeSelector._check_gpu_availability') as mock:
            mock.return_value = False
            yield mock
    
    @pytest.fixture
    def selector_with_gpu(self, mock_gpu_available):
        """Create selector with GPU available."""
        return ModeSelector(gpu_memory_gb=16.0, cpu_memory_gb=32.0, prefer_gpu=True)
    
    @pytest.fixture
    def selector_without_gpu(self, mock_gpu_unavailable):
        """Create selector without GPU."""
        return ModeSelector(gpu_memory_gb=0.0, cpu_memory_gb=32.0, prefer_gpu=True)
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available in test environment")
    def test_initialization_with_gpu(self, selector_with_gpu):
        """Test selector initialization with GPU available."""
        assert selector_with_gpu.gpu_available is True
        assert selector_with_gpu.gpu_memory_gb == 16.0
        assert selector_with_gpu.cpu_memory_gb == 32.0
        assert selector_with_gpu.prefer_gpu is True
    
    @pytest.mark.skipif(GPU_AVAILABLE, reason="Test requires no GPU environment")
    def test_initialization_without_gpu(self, selector_without_gpu):
        """Test selector initialization without GPU."""
        assert selector_without_gpu.gpu_available is False
        assert selector_without_gpu.gpu_memory_gb == 0.0
        assert selector_without_gpu.cpu_memory_gb == 32.0
    
    # Test small clouds (< 500K points)
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available in test environment")
    def test_small_cloud_with_gpu(self, selector_with_gpu):
        """Small cloud with GPU available should use GPU."""
        mode = selector_with_gpu.select_mode(num_points=100_000)
        assert mode == ComputationMode.GPU
    
    @pytest.mark.skipif(GPU_AVAILABLE, reason="Test requires no GPU environment")
    def test_small_cloud_without_gpu(self, selector_without_gpu):
        """Small cloud without GPU should use CPU."""
        mode = selector_without_gpu.select_mode(num_points=100_000)
        assert mode == ComputationMode.CPU
    
    def test_small_cloud_prefer_cpu(self, mock_gpu_available):
        """Small cloud with prefer_gpu=False should use CPU."""
        selector = ModeSelector(
            gpu_memory_gb=16.0,
            cpu_memory_gb=32.0,
            prefer_gpu=False
        )
        mode = selector.select_mode(num_points=100_000)
        assert mode == ComputationMode.CPU
    
    # Test medium clouds (500K - 5M points)
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available in test environment")
    def test_medium_cloud_with_gpu(self, selector_with_gpu):
        """Medium cloud with GPU available should use GPU."""
        mode = selector_with_gpu.select_mode(num_points=2_000_000)
        assert mode == ComputationMode.GPU
    
    @pytest.mark.skipif(GPU_AVAILABLE, reason="Test requires no GPU environment")
    def test_medium_cloud_without_gpu(self, selector_without_gpu):
        """Medium cloud without GPU should use CPU."""
        mode = selector_without_gpu.select_mode(num_points=2_000_000)
        assert mode == ComputationMode.CPU
    
    # Test large clouds (5M - 10M points)
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available in test environment")
    def test_large_cloud_with_gpu(self, selector_with_gpu):
        """Large cloud with GPU available should use GPU or GPU_CHUNKED."""
        mode = selector_with_gpu.select_mode(num_points=7_000_000)
        assert mode in (ComputationMode.GPU, ComputationMode.GPU_CHUNKED)
    
    @pytest.mark.skipif(GPU_AVAILABLE, reason="Test requires no GPU environment")
    def test_large_cloud_without_gpu(self, selector_without_gpu):
        """Large cloud without GPU should use CPU if memory allows."""
        mode = selector_without_gpu.select_mode(num_points=7_000_000)
        assert mode == ComputationMode.CPU
    
    # Test very large clouds (> 10M points)
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available in test environment")
    def test_very_large_cloud_with_gpu(self, selector_with_gpu):
        """Very large cloud should use GPU_CHUNKED."""
        mode = selector_with_gpu.select_mode(num_points=15_000_000)
        assert mode == ComputationMode.GPU_CHUNKED
    
    @pytest.mark.skipif(GPU_AVAILABLE, reason="Test requires no GPU environment")
    def test_very_large_cloud_without_gpu(self, selector_without_gpu):
        """Very large cloud without GPU should use CPU if memory allows."""
        mode = selector_without_gpu.select_mode(num_points=15_000_000)
        assert mode == ComputationMode.CPU
    
    # Test force flags
    
    def test_force_cpu(self, selector_with_gpu):
        """Force CPU mode should work even with GPU available."""
        mode = selector_with_gpu.select_mode(num_points=1_000_000, force_cpu=True)
        assert mode == ComputationMode.CPU
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available in test environment")
    def test_force_gpu(self, selector_with_gpu):
        """Force GPU mode should select GPU or GPU_CHUNKED."""
        mode = selector_with_gpu.select_mode(num_points=1_000_000, force_gpu=True)
        assert mode in (ComputationMode.GPU, ComputationMode.GPU_CHUNKED)
    
    @pytest.mark.skipif(GPU_AVAILABLE, reason="Test requires no GPU environment")
    def test_force_gpu_without_gpu_raises(self, selector_without_gpu):
        """Force GPU without GPU available should raise ValueError."""
        with pytest.raises(ValueError, match="GPU mode forced but GPU not available"):
            selector_without_gpu.select_mode(num_points=1_000_000, force_gpu=True)
    
    def test_force_cpu_insufficient_memory_raises(self, mock_gpu_available):
        """Force CPU with insufficient memory should raise ValueError."""
        selector = ModeSelector(gpu_memory_gb=16.0, cpu_memory_gb=0.1, prefer_gpu=True)
        with pytest.raises(ValueError, match="insufficient memory"):
            selector.select_mode(num_points=10_000_000, force_cpu=True)
    
    # Test user override
    
    def test_user_mode_override(self, selector_with_gpu):
        """User-specified mode should override automatic selection."""
        mode = selector_with_gpu.select_mode(
            num_points=1_000_000,
            user_mode=ComputationMode.CPU
        )
        assert mode == ComputationMode.CPU
    
    # Test boundary mode
    
    def test_boundary_mode(self, selector_with_gpu):
        """Boundary mode should be selected when boundary_mode=True."""
        mode = selector_with_gpu.select_mode(
            num_points=100_000,
            boundary_mode=True
        )
        assert mode == ComputationMode.BOUNDARY
    
    def test_boundary_mode_small_cloud(self, selector_without_gpu):
        """Boundary mode works without GPU for small clouds."""
        mode = selector_without_gpu.select_mode(
            num_points=5_000,
            boundary_mode=True
        )
        assert mode == ComputationMode.BOUNDARY
    
    # Test memory estimation
    
    def test_estimate_memory_cpu(self, selector_with_gpu):
        """Test memory estimation for CPU mode."""
        estimated, available = selector_with_gpu.estimate_memory_usage(
            num_points=1_000_000,
            mode=ComputationMode.CPU
        )
        assert estimated > 0
        assert available == 32.0
        assert estimated < available  # Should fit in memory
    
    def test_estimate_memory_gpu(self, selector_with_gpu):
        """Test memory estimation for GPU mode."""
        estimated, available = selector_with_gpu.estimate_memory_usage(
            num_points=1_000_000,
            mode=ComputationMode.GPU
        )
        assert estimated > 0
        assert available == 16.0
    
    def test_estimate_memory_gpu_chunked(self, selector_with_gpu):
        """Test memory estimation for GPU_CHUNKED mode."""
        estimated, available = selector_with_gpu.estimate_memory_usage(
            num_points=10_000_000,
            mode=ComputationMode.GPU_CHUNKED
        )
        assert estimated > 0
        # Chunked should use less memory per point
        estimated_regular, _ = selector_with_gpu.estimate_memory_usage(
            num_points=10_000_000,
            mode=ComputationMode.GPU
        )
        assert estimated < estimated_regular
    
    # Test recommendations
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available in test environment")
    def test_get_recommendations(self, selector_with_gpu):
        """Test getting recommendations for a point cloud."""
        recommendations = selector_with_gpu.get_recommendations(num_points=1_000_000)
        
        assert "recommended_mode" in recommendations
        assert "num_points" in recommendations
        assert "estimated_memory_gb" in recommendations
        assert "available_memory_gb" in recommendations
        assert "memory_utilization_pct" in recommendations
        assert "estimated_time_seconds" in recommendations
        assert "gpu_available" in recommendations
        assert "alternative_modes" in recommendations
        
        assert recommendations["num_points"] == 1_000_000
        assert recommendations["gpu_available"] is True
        assert isinstance(recommendations["alternative_modes"], list)
    
    def test_recommendations_alternative_modes(self, selector_with_gpu):
        """Test that alternative modes are listed."""
        recommendations = selector_with_gpu.get_recommendations(num_points=1_000_000)
        alternatives = recommendations["alternative_modes"]
        
        assert len(alternatives) > 0
        for alt in alternatives:
            assert "mode" in alt
            assert "viable" in alt
            assert "reason" in alt
            assert "estimated_memory_gb" in alt
    
    # Test edge cases
    
    def test_zero_points(self, selector_with_gpu):
        """Test with zero points (edge case)."""
        mode = selector_with_gpu.select_mode(num_points=0)
        # Should handle gracefully (likely CPU mode)
        assert mode in ComputationMode
    
    def test_very_small_cloud(self, selector_with_gpu):
        """Test with very small cloud (1000 points)."""
        mode = selector_with_gpu.select_mode(num_points=1_000)
        assert mode in (ComputationMode.CPU, ComputationMode.GPU)
    
    def test_threshold_boundary_values(self, selector_with_gpu):
        """Test values at threshold boundaries."""
        # Just below small threshold
        mode1 = selector_with_gpu.select_mode(num_points=499_999)
        # Just above small threshold
        mode2 = selector_with_gpu.select_mode(num_points=500_001)
        
        # Both should work, may select different modes
        assert mode1 in ComputationMode
        assert mode2 in ComputationMode
    
    # Test factory function
    
    def test_get_mode_selector_factory(self):
        """Test factory function creates valid selector."""
        with patch('ign_lidar.features.mode_selector.ModeSelector._check_gpu_availability') as mock:
            mock.return_value = True
            selector = get_mode_selector(gpu_memory_gb=8.0, cpu_memory_gb=16.0)
            
            assert isinstance(selector, ModeSelector)
            assert selector.gpu_memory_gb == 8.0
            assert selector.cpu_memory_gb == 16.0


class TestModeSelectionIntegration:
    """Integration tests for mode selection with realistic scenarios."""
    
    @pytest.fixture
    def workstation_selector(self):
        """Simulate high-end workstation (GPU + lots of RAM)."""
        with patch('ign_lidar.features.mode_selector.ModeSelector._check_gpu_availability') as mock:
            mock.return_value = True
            yield ModeSelector(gpu_memory_gb=24.0, cpu_memory_gb=64.0, prefer_gpu=True)
    
    @pytest.fixture
    def laptop_selector(self):
        """Simulate laptop (small GPU or no GPU)."""
        with patch('ign_lidar.features.mode_selector.ModeSelector._check_gpu_availability') as mock:
            mock.return_value = False
            yield ModeSelector(gpu_memory_gb=0.0, cpu_memory_gb=16.0, prefer_gpu=True)
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available in test environment")
    def test_workstation_small_dataset(self, workstation_selector):
        """Workstation processing small dataset."""
        mode = workstation_selector.select_mode(num_points=250_000)
        assert mode == ComputationMode.GPU
        
        recommendations = workstation_selector.get_recommendations(num_points=250_000)
        assert recommendations["memory_utilization_pct"] < 50  # Should be low
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available in test environment")
    def test_workstation_large_dataset(self, workstation_selector):
        """Workstation processing large dataset."""
        mode = workstation_selector.select_mode(num_points=20_000_000)
        assert mode == ComputationMode.GPU_CHUNKED
        
        recommendations = workstation_selector.get_recommendations(num_points=20_000_000)
        assert recommendations["estimated_time_seconds"] < 10  # Should be fast
    
    @pytest.mark.skipif(GPU_AVAILABLE, reason="Test requires no GPU environment")
    def test_laptop_small_dataset(self, laptop_selector):
        """Laptop processing small dataset."""
        mode = laptop_selector.select_mode(num_points=250_000)
        assert mode == ComputationMode.CPU
        
        recommendations = laptop_selector.get_recommendations(num_points=250_000)
        assert recommendations["recommended_mode"] == "cpu"
    
    @pytest.mark.skipif(GPU_AVAILABLE, reason="Test requires no GPU environment")
    def test_laptop_medium_dataset(self, laptop_selector):
        """Laptop processing medium dataset."""
        mode = laptop_selector.select_mode(num_points=2_000_000)
        assert mode == ComputationMode.CPU
        
        recommendations = laptop_selector.get_recommendations(num_points=2_000_000)
        # Should work but take longer
        assert recommendations["estimated_time_seconds"] > 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
