"""
Unit tests for GPU-Core Bridge Module.

Tests the bridge pattern between GPU-accelerated eigenvalue computation
and canonical core feature implementations.

Author: IGN LiDAR HD Dataset Team
Date: October 2025
"""

import pytest
import numpy as np
from typing import Dict

# Import module under test
from ign_lidar.features.core.gpu_bridge import (
    GPUCoreBridge,
    compute_eigenvalues_gpu,
    compute_eigenvalue_features_gpu,
    CUPY_AVAILABLE,
)

# Import core module for comparison
from ign_lidar.features.core import compute_eigenvalue_features


# Test fixtures
@pytest.fixture
def small_point_cloud():
    """Small point cloud for basic tests (1000 points)."""
    np.random.seed(42)
    return np.random.rand(1000, 3).astype(np.float32)


@pytest.fixture
def medium_point_cloud():
    """Medium point cloud (100,000 points)."""
    np.random.seed(42)
    return np.random.rand(100_000, 3).astype(np.float32)


@pytest.fixture
def large_point_cloud():
    """Large point cloud for batching tests (1,000,000 points)."""
    np.random.seed(42)
    return np.random.rand(1_000_000, 3).astype(np.float32)


@pytest.fixture
def neighbors_small():
    """Neighbor indices for small point cloud (k=20)."""
    np.random.seed(42)
    return np.random.randint(0, 1000, size=(1000, 20), dtype=np.int32)


@pytest.fixture
def neighbors_medium():
    """Neighbor indices for medium point cloud (k=20)."""
    np.random.seed(42)
    return np.random.randint(0, 100_000, size=(100_000, 20), dtype=np.int32)


@pytest.fixture
def neighbors_large():
    """Neighbor indices for large point cloud (k=20)."""
    np.random.seed(42)
    return np.random.randint(0, 1_000_000, size=(1_000_000, 20), dtype=np.int32)


@pytest.fixture
def planar_point_cloud():
    """Point cloud representing a planar surface."""
    np.random.seed(42)
    # Create points on XY plane with small noise
    x = np.random.rand(1000) * 10
    y = np.random.rand(1000) * 10
    z = np.random.rand(1000) * 0.1  # Small variation in Z
    return np.column_stack([x, y, z]).astype(np.float32)


@pytest.fixture
def linear_point_cloud():
    """Point cloud representing a linear feature."""
    np.random.seed(42)
    # Create points along X axis with small noise
    t = np.linspace(0, 10, 1000)
    x = t + np.random.rand(1000) * 0.1
    y = np.random.rand(1000) * 0.1
    z = np.random.rand(1000) * 0.1
    return np.column_stack([x, y, z]).astype(np.float32)


# Test class for GPUCoreBridge
class TestGPUCoreBridge:
    """Test suite for GPUCoreBridge class."""
    
    def test_initialization_cpu(self):
        """Test CPU-only initialization."""
        bridge = GPUCoreBridge(use_gpu=False)
        assert bridge.use_gpu is False
        assert bridge.batch_size == 500_000
        assert bridge.epsilon == 1e-10
    
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_initialization_gpu(self):
        """Test GPU initialization when CuPy is available."""
        bridge = GPUCoreBridge(use_gpu=True)
        assert bridge.use_gpu is True
        assert bridge.batch_size == 500_000
    
    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        bridge = GPUCoreBridge(
            use_gpu=False,
            batch_size=100_000,
            epsilon=1e-8
        )
        assert bridge.batch_size == 100_000
        assert bridge.epsilon == 1e-8
    
    def test_compute_eigenvalues_cpu(self, small_point_cloud, neighbors_small):
        """Test CPU eigenvalue computation."""
        bridge = GPUCoreBridge(use_gpu=False)
        eigenvalues = bridge.compute_eigenvalues_gpu(
            small_point_cloud, neighbors_small
        )
        
        # Check output shape
        assert eigenvalues.shape == (1000, 3)
        
        # Check eigenvalues are sorted descending
        assert np.all(eigenvalues[:, 0] >= eigenvalues[:, 1])
        assert np.all(eigenvalues[:, 1] >= eigenvalues[:, 2])
        
        # Check eigenvalues are positive
        assert np.all(eigenvalues >= 0)
    
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_compute_eigenvalues_gpu(self, small_point_cloud, neighbors_small):
        """Test GPU eigenvalue computation."""
        bridge = GPUCoreBridge(use_gpu=True)
        eigenvalues = bridge.compute_eigenvalues_gpu(
            small_point_cloud, neighbors_small
        )
        
        # Check output shape
        assert eigenvalues.shape == (1000, 3)
        
        # Check eigenvalues are sorted descending
        assert np.all(eigenvalues[:, 0] >= eigenvalues[:, 1])
        assert np.all(eigenvalues[:, 1] >= eigenvalues[:, 2])
        
        # Check eigenvalues are positive
        assert np.all(eigenvalues >= 0)
    
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_gpu_cpu_consistency(self, small_point_cloud, neighbors_small):
        """Test that GPU and CPU produce consistent results."""
        bridge_gpu = GPUCoreBridge(use_gpu=True)
        bridge_cpu = GPUCoreBridge(use_gpu=False)
        
        eigenvalues_gpu = bridge_gpu.compute_eigenvalues_gpu(
            small_point_cloud, neighbors_small
        )
        eigenvalues_cpu = bridge_cpu.compute_eigenvalues_gpu(
            small_point_cloud, neighbors_small
        )
        
        # Should be very close (within floating-point tolerance)
        np.testing.assert_allclose(
            eigenvalues_gpu,
            eigenvalues_cpu,
            rtol=1e-5,
            atol=1e-7
        )
    
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_batched_computation(self, large_point_cloud, neighbors_large):
        """Test batched GPU computation for large datasets."""
        bridge = GPUCoreBridge(use_gpu=True, batch_size=500_000)
        eigenvalues = bridge.compute_eigenvalues_gpu(
            large_point_cloud, neighbors_large
        )
        
        # Check output shape
        assert eigenvalues.shape == (1_000_000, 3)
        
        # Check eigenvalues are valid
        assert np.all(eigenvalues[:, 0] >= eigenvalues[:, 1])
        assert np.all(eigenvalues[:, 1] >= eigenvalues[:, 2])
        assert np.all(eigenvalues >= 0)
    
    def test_eigenvalue_features_integration(self, small_point_cloud, neighbors_small):
        """Test integration with core eigenvalue features."""
        bridge = GPUCoreBridge(use_gpu=False)
        features = bridge.compute_eigenvalue_features_gpu(
            small_point_cloud, neighbors_small
        )
        
        # Check that all expected features are present
        expected_features = [
            'linearity', 'planarity', 'sphericity', 'anisotropy',
            'eigenentropy', 'omnivariance', 'sum_eigenvalues'
        ]
        for feat in expected_features:
            assert feat in features
            assert features[feat].shape == (1000,)
    
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_feature_consistency_gpu_vs_cpu(self, small_point_cloud, neighbors_small):
        """Test that GPU bridge produces same features as direct core computation."""
        # Compute using GPU bridge
        bridge = GPUCoreBridge(use_gpu=True)
        features_gpu = bridge.compute_eigenvalue_features_gpu(
            small_point_cloud, neighbors_small
        )
        
        # Compute using CPU bridge
        bridge_cpu = GPUCoreBridge(use_gpu=False)
        features_cpu = bridge_cpu.compute_eigenvalue_features_gpu(
            small_point_cloud, neighbors_small
        )
        
        # Compare all features
        for key in features_gpu.keys():
            np.testing.assert_allclose(
                features_gpu[key],
                features_cpu[key],
                rtol=1e-5,
                atol=1e-7,
                err_msg=f"Feature '{key}' differs between GPU and CPU"
            )
    
    def test_planar_features(self, planar_point_cloud, neighbors_small):
        """Test that planar surfaces have high planarity."""
        bridge = GPUCoreBridge(use_gpu=False)
        features = bridge.compute_eigenvalue_features_gpu(
            planar_point_cloud, neighbors_small
        )
        
        # Planar surfaces should have high planarity
        assert np.mean(features['planarity']) > 0.5
        # And low sphericity
        assert np.mean(features['sphericity']) < 0.3
    
    def test_linear_features(self, linear_point_cloud, neighbors_small):
        """Test that linear features have high linearity."""
        bridge = GPUCoreBridge(use_gpu=False)
        features = bridge.compute_eigenvalue_features_gpu(
            linear_point_cloud, neighbors_small
        )
        
        # Linear features should have high linearity
        assert np.mean(features['linearity']) > 0.5
        # And low sphericity
        assert np.mean(features['sphericity']) < 0.3
    
    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        bridge = GPUCoreBridge(use_gpu=False)
        
        # Invalid point cloud shape
        with pytest.raises(ValueError):
            bridge.compute_eigenvalues_gpu(
                np.random.rand(100, 2),  # Wrong: should be (N, 3)
                np.random.randint(0, 100, size=(100, 20))
            )
        
        # Invalid neighbors shape
        with pytest.raises(ValueError):
            bridge.compute_eigenvalues_gpu(
                np.random.rand(100, 3),
                np.random.randint(0, 100, size=(50, 20))  # Wrong: should match N
            )
    
    def test_eigenvectors_option(self, small_point_cloud, neighbors_small):
        """Test computation with eigenvectors."""
        bridge = GPUCoreBridge(use_gpu=False)
        eigenvalues, eigenvectors = bridge.compute_eigenvalues_gpu(
            small_point_cloud, neighbors_small,
            return_eigenvectors=True
        )
        
        # Check shapes
        assert eigenvalues.shape == (1000, 3)
        assert eigenvectors.shape == (1000, 3, 3)
        
        # Check that eigenvectors are orthonormal
        for i in range(min(10, len(eigenvectors))):  # Check first 10
            v = eigenvectors[i]
            # Check orthogonality
            gram = v.T @ v
            np.testing.assert_allclose(
                gram,
                np.eye(3),
                rtol=1e-5,
                atol=1e-7
            )


# Test convenience functions
class TestConvenienceFunctions:
    """Test suite for convenience functions."""
    
    def test_compute_eigenvalues_gpu_function(self, small_point_cloud, neighbors_small):
        """Test convenience function for eigenvalues."""
        eigenvalues = compute_eigenvalues_gpu(
            small_point_cloud,
            neighbors_small,
            use_gpu=False
        )
        
        assert eigenvalues.shape == (1000, 3)
        assert np.all(eigenvalues[:, 0] >= eigenvalues[:, 1])
    
    def test_compute_eigenvalue_features_gpu_function(self, small_point_cloud, neighbors_small):
        """Test convenience function for features."""
        features = compute_eigenvalue_features_gpu(
            small_point_cloud,
            neighbors_small,
            use_gpu=False
        )
        
        assert 'linearity' in features
        assert 'planarity' in features
        assert features['linearity'].shape == (1000,)
    
    def test_convenience_vs_class_consistency(self, small_point_cloud, neighbors_small):
        """Test that convenience functions match class methods."""
        # Using convenience function
        features_func = compute_eigenvalue_features_gpu(
            small_point_cloud,
            neighbors_small,
            use_gpu=False
        )
        
        # Using class
        bridge = GPUCoreBridge(use_gpu=False)
        features_class = bridge.compute_eigenvalue_features_gpu(
            small_point_cloud,
            neighbors_small
        )
        
        # Should be identical
        for key in features_func.keys():
            np.testing.assert_array_equal(
                features_func[key],
                features_class[key]
            )


# Performance benchmarks (optional, marked for manual execution)
class TestPerformance:
    """Performance tests (run manually with --benchmark flag)."""
    
    @pytest.mark.benchmark
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_gpu_speedup(self, medium_point_cloud, neighbors_medium):
        """Benchmark GPU vs CPU speedup."""
        import time
        
        # CPU timing
        bridge_cpu = GPUCoreBridge(use_gpu=False)
        start = time.time()
        eigenvalues_cpu = bridge_cpu.compute_eigenvalues_gpu(
            medium_point_cloud, neighbors_medium
        )
        cpu_time = time.time() - start
        
        # GPU timing
        bridge_gpu = GPUCoreBridge(use_gpu=True)
        start = time.time()
        eigenvalues_gpu = bridge_gpu.compute_eigenvalues_gpu(
            medium_point_cloud, neighbors_medium
        )
        gpu_time = time.time() - start
        
        speedup = cpu_time / gpu_time
        print(f"\nDataset: {len(medium_point_cloud)} points")
        print(f"CPU time: {cpu_time:.3f}s")
        print(f"GPU time: {gpu_time:.3f}s")
        print(f"Speedup: {speedup:.1f}×")
        
        # Expected speedup should be at least 5× for medium datasets
        assert speedup > 5.0, f"GPU speedup {speedup:.1f}× is below target (5×)"
    
    @pytest.mark.benchmark
    def test_feature_computation_overhead(self, medium_point_cloud, neighbors_medium):
        """Test overhead of feature computation vs eigenvalue computation."""
        import time
        
        bridge = GPUCoreBridge(use_gpu=False)
        
        # Time eigenvalue computation only
        start = time.time()
        eigenvalues = bridge.compute_eigenvalues_gpu(
            medium_point_cloud, neighbors_medium
        )
        eigenvalue_time = time.time() - start
        
        # Time full feature computation
        start = time.time()
        features = bridge.compute_eigenvalue_features_gpu(
            medium_point_cloud, neighbors_medium
        )
        total_time = time.time() - start
        
        feature_overhead = total_time - eigenvalue_time
        overhead_percent = (feature_overhead / total_time) * 100
        
        print(f"\nEigenvalue computation: {eigenvalue_time:.3f}s")
        print(f"Feature computation overhead: {feature_overhead:.3f}s ({overhead_percent:.1f}%)")
        
        # Feature computation should be fast compared to eigenvalues
        assert overhead_percent < 50, "Feature computation overhead too high"


# Edge case tests
class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_neighbors(self):
        """Test handling of zero neighbors."""
        bridge = GPUCoreBridge(use_gpu=False)
        points = np.random.rand(10, 3).astype(np.float32)
        neighbors = np.zeros((10, 0), dtype=np.int32)
        
        # Should handle gracefully or raise clear error
        with pytest.raises((ValueError, IndexError)):
            bridge.compute_eigenvalues_gpu(points, neighbors)
    
    def test_single_point(self):
        """Test with single point."""
        bridge = GPUCoreBridge(use_gpu=False)
        points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        neighbors = np.array([[0, 0, 0]], dtype=np.int32)
        
        eigenvalues = bridge.compute_eigenvalues_gpu(points, neighbors)
        assert eigenvalues.shape == (1, 3)
    
    def test_identical_points(self):
        """Test with identical neighbor points."""
        bridge = GPUCoreBridge(use_gpu=False)
        points = np.zeros((100, 3), dtype=np.float32)
        neighbors = np.zeros((100, 20), dtype=np.int32)
        
        eigenvalues = bridge.compute_eigenvalues_gpu(points, neighbors)
        
        # All eigenvalues should be near zero for identical points
        assert np.all(eigenvalues < 1e-6)
    
    def test_numerical_stability(self):
        """Test numerical stability with very small/large values."""
        bridge = GPUCoreBridge(use_gpu=False)
        
        # Very small values
        points_small = np.random.rand(100, 3).astype(np.float32) * 1e-6
        neighbors = np.random.randint(0, 100, size=(100, 20), dtype=np.int32)
        eigenvalues_small = bridge.compute_eigenvalues_gpu(points_small, neighbors)
        assert not np.any(np.isnan(eigenvalues_small))
        assert not np.any(np.isinf(eigenvalues_small))
        
        # Very large values
        points_large = np.random.rand(100, 3).astype(np.float32) * 1e6
        eigenvalues_large = bridge.compute_eigenvalues_gpu(points_large, neighbors)
        assert not np.any(np.isnan(eigenvalues_large))
        assert not np.any(np.isinf(eigenvalues_large))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
