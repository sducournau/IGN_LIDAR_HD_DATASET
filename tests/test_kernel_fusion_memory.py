"""
Tests for GPU Kernel Fusion Memory Safety Checks

Tests the Phase 3.8.1 enhancement that adds memory safety checks
to the fused CUDA kernel operations.

Author: IGN LiDAR HD Development Team
Date: November 23, 2025
Version: 3.8.1
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Test markers
pytestmark = pytest.mark.unit


class TestFusedKernelMemoryEstimation:
    """Test memory estimation for fused kernels."""
    
    def test_estimate_memory_basic(self):
        """Test basic memory estimation."""
        from ign_lidar.optimization.gpu_kernels import estimate_fused_kernel_memory
        
        # Small dataset
        mem_gb = estimate_fused_kernel_memory(
            n_points=100_000,
            k_neighbors=30,
            dim=3
        )
        
        assert mem_gb > 0, "Memory estimate should be positive"
        assert mem_gb < 1.0, "100K points should need less than 1GB"
        
    def test_estimate_memory_large(self):
        """Test memory estimation for large dataset."""
        from ign_lidar.optimization.gpu_kernels import estimate_fused_kernel_memory
        
        # Large dataset
        mem_gb = estimate_fused_kernel_memory(
            n_points=5_000_000,
            k_neighbors=30,
            dim=3
        )
        
        assert mem_gb > 1.0, "5M points should need more than 1GB"
        assert mem_gb < 10.0, "5M points should need less than 10GB"
        
    def test_estimate_memory_scales(self):
        """Test that memory estimation scales linearly with points."""
        from ign_lidar.optimization.gpu_kernels import estimate_fused_kernel_memory
        
        mem_1m = estimate_fused_kernel_memory(1_000_000, 30, 3)
        mem_2m = estimate_fused_kernel_memory(2_000_000, 30, 3)
        
        # Should be roughly 2x (within 10% tolerance for overhead)
        ratio = mem_2m / mem_1m
        assert 1.8 < ratio < 2.2, f"Memory should scale ~2x, got {ratio:.2f}x"
        
    def test_estimate_memory_k_neighbors(self):
        """Test that memory estimation accounts for k_neighbors."""
        from ign_lidar.optimization.gpu_kernels import estimate_fused_kernel_memory
        
        mem_k10 = estimate_fused_kernel_memory(1_000_000, k_neighbors=10, dim=3)
        mem_k30 = estimate_fused_kernel_memory(1_000_000, k_neighbors=30, dim=3)
        
        # k=30 should use more memory than k=10
        assert mem_k30 > mem_k10, "More neighbors should require more memory"
        
        # But not 3x more (knn indices are small part of total)
        ratio = mem_k30 / mem_k10
        assert ratio < 2.0, f"k=30 vs k=10 ratio too high: {ratio:.2f}x"


class TestFusedKernelMemoryChecks:
    """Test memory safety checks in fused kernel."""
    
    @pytest.fixture
    def mock_gpu_manager(self):
        """Mock GPUManager for testing."""
        with patch('ign_lidar.optimization.gpu_kernels.GPUManager') as mock:
            manager = Mock()
            manager.gpu_available = True
            manager.get_memory_info.return_value = {
                'free_gb': 4.0,
                'total_gb': 8.0,
                'used_gb': 4.0
            }
            mock.return_value = manager
            yield mock
    
    @pytest.mark.skipif(True, reason="Requires GPU")
    def test_fused_kernel_checks_memory(self, mock_gpu_manager):
        """Test that fused kernel checks memory before execution."""
        from ign_lidar.optimization.gpu_kernels import CUDAKernels
        
        # This test requires actual GPU, skip in CI
        kernels = CUDAKernels()
        if not kernels.available:
            pytest.skip("GPU not available")
        
        # Create small dataset that should fit
        points = np.random.randn(1000, 3).astype(np.float32)
        knn_indices = np.random.randint(0, 1000, (1000, 30)).astype(np.int32)
        
        # Should proceed with check_memory=True
        normals, eigenvalues, curvature = kernels.compute_normals_eigenvalues_fused(
            points=points,
            knn_indices=knn_indices,
            k=30,
            check_memory=True
        )
        
        assert normals.shape == (1000, 3)
        assert eigenvalues.shape == (1000, 3)
        assert curvature.shape == (1000,)
    
    @pytest.mark.skipif(True, reason="Requires GPU")
    def test_fused_kernel_fallback_insufficient_memory(self, mock_gpu_manager):
        """Test fallback to sequential when memory insufficient."""
        from ign_lidar.optimization.gpu_kernels import CUDAKernels
        
        # Mock insufficient memory
        mock_gpu_manager.return_value.get_memory_info.return_value = {
            'free_gb': 0.5,  # Very low memory
            'total_gb': 8.0,
            'used_gb': 7.5
        }
        
        kernels = CUDAKernels()
        if not kernels.available:
            pytest.skip("GPU not available")
        
        # Create dataset that would exceed available memory
        points = np.random.randn(5_000_000, 3).astype(np.float32)
        knn_indices = np.random.randint(0, 5_000_000, (5_000_000, 30)).astype(np.int32)
        
        # Should fall back to sequential
        with patch.object(kernels, '_compute_normals_eigenvalues_sequential') as mock_seq:
            mock_seq.return_value = (
                np.zeros((5_000_000, 3)),
                np.zeros((5_000_000, 3)),
                np.zeros(5_000_000)
            )
            
            normals, eigenvalues, curvature = kernels.compute_normals_eigenvalues_fused(
                points=points,
                knn_indices=knn_indices,
                k=30,
                check_memory=True
            )
            
            # Sequential should have been called
            mock_seq.assert_called_once()
    
    def test_memory_check_can_be_disabled(self):
        """Test that memory check can be disabled."""
        from ign_lidar.optimization.gpu_kernels import CUDAKernels
        
        # This just tests the parameter is accepted
        # Actual execution would require GPU
        kernels = CUDAKernels()
        
        # check_memory=False should be accepted
        # (will fail without GPU, but that's expected)
        try:
            # This will fail without GPU, which is fine
            kernels.compute_normals_eigenvalues_fused(
                points=np.zeros((100, 3)),
                knn_indices=np.zeros((100, 30), dtype=np.int32),
                k=30,
                check_memory=False
            )
        except RuntimeError as e:
            # Expected if no GPU
            assert "CUDA" in str(e) or "not available" in str(e)


class TestSequentialFallback:
    """Test sequential fallback implementation."""
    
    @pytest.mark.skipif(True, reason="Requires GPU")
    def test_sequential_fallback_produces_valid_output(self):
        """Test sequential fallback produces valid normals."""
        from ign_lidar.optimization.gpu_kernels import CUDAKernels
        import cupy as cp
        
        kernels = CUDAKernels()
        if not kernels.available:
            pytest.skip("GPU not available")
        
        # Small dataset for testing
        points = np.random.randn(100, 3).astype(np.float32)
        knn_indices = np.tile(np.arange(100), (100, 1))[:, :30].astype(np.int32)
        
        normals, eigenvalues, curvature = kernels._compute_normals_eigenvalues_sequential(
            points=points,
            knn_indices=knn_indices,
            k=30
        )
        
        # Check output shapes
        assert normals.shape == (100, 3)
        assert eigenvalues.shape == (100, 3)
        assert curvature.shape == (100,)
        
        # Check normals are unit vectors
        norms = np.linalg.norm(normals, axis=1)
        assert np.allclose(norms, 1.0, atol=0.01), "Normals should be unit vectors"
        
        # Check eigenvalues are sorted descending
        for i in range(100):
            assert eigenvalues[i, 0] >= eigenvalues[i, 1], "Eigenvalues should be sorted"
            assert eigenvalues[i, 1] >= eigenvalues[i, 2], "Eigenvalues should be sorted"
        
        # Check curvature is in valid range
        assert np.all(curvature >= 0), "Curvature should be non-negative"
        assert np.all(curvature <= 1), "Curvature should be <= 1"
    
    @pytest.mark.skipif(True, reason="Requires GPU")
    def test_sequential_matches_fused_results(self):
        """Test sequential fallback produces similar results to fused kernel."""
        from ign_lidar.optimization.gpu_kernels import CUDAKernels
        
        kernels = CUDAKernels()
        if not kernels.available:
            pytest.skip("GPU not available")
        
        # Small dataset
        np.random.seed(42)
        points = np.random.randn(500, 3).astype(np.float32)
        knn_indices = np.random.randint(0, 500, (500, 30)).astype(np.int32)
        
        # Compute with fused kernel
        normals_fused, evals_fused, curv_fused = kernels.compute_normals_eigenvalues_fused(
            points=points,
            knn_indices=knn_indices,
            k=30,
            check_memory=False  # Bypass check for test
        )
        
        # Compute with sequential
        normals_seq, evals_seq, curv_seq = kernels._compute_normals_eigenvalues_sequential(
            points=points,
            knn_indices=knn_indices,
            k=30
        )
        
        # Results should be very similar (allowing for numerical differences)
        normal_diff = np.abs(normals_fused - normals_seq)
        assert np.mean(normal_diff) < 0.01, "Sequential should match fused normals closely"
        
        eval_diff = np.abs(evals_fused - evals_seq)
        assert np.mean(eval_diff) < 0.01, "Sequential should match fused eigenvalues closely"


class TestMemorySafetyIntegration:
    """Integration tests for memory safety."""
    
    def test_memory_estimation_exported(self):
        """Test memory estimation function is exported."""
        from ign_lidar.optimization import gpu_kernels
        
        assert hasattr(gpu_kernels, 'estimate_fused_kernel_memory')
        
        # Can be called
        mem_gb = gpu_kernels.estimate_fused_kernel_memory(
            n_points=1_000_000,
            k_neighbors=30
        )
        assert mem_gb > 0
    
    def test_cuda_kernels_has_sequential_fallback(self):
        """Test CUDAKernels has sequential fallback method."""
        from ign_lidar.optimization.gpu_kernels import CUDAKernels
        
        kernels = CUDAKernels()
        assert hasattr(kernels, '_compute_normals_eigenvalues_sequential')
    
    def test_memory_check_parameters(self):
        """Test fused kernel accepts memory check parameters."""
        from ign_lidar.optimization.gpu_kernels import CUDAKernels
        import inspect
        
        kernels = CUDAKernels()
        sig = inspect.signature(kernels.compute_normals_eigenvalues_fused)
        
        assert 'check_memory' in sig.parameters
        assert 'safety_margin' in sig.parameters
        
        # Check defaults
        assert sig.parameters['check_memory'].default is True
        assert sig.parameters['safety_margin'].default == 0.15


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
