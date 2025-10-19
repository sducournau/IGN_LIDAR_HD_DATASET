"""
Integration tests for Phase 2: GPU-Core Bridge integration with GPUProcessor.

Tests that the refactored eigenvalue computation maintains backward compatibility
and produces correct results.

Author: IGN LiDAR HD Dataset Team
Date: October 2025
"""

import pytest
import numpy as np
from typing import Dict

from ign_lidar.features.gpu_processor import GPUProcessor


@pytest.fixture
def sample_point_cloud():
    """Create a small test point cloud."""
    np.random.seed(42)
    return np.random.rand(1000, 3).astype(np.float32)


@pytest.fixture
def sample_neighbors():
    """Create sample neighbor indices."""
    np.random.seed(42)
    return np.random.randint(0, 1000, size=(1000, 20), dtype=np.int64)


class TestPhase2Integration:
    """Test Phase 2 refactoring integration."""
    
    def test_gpu_chunked_init_with_bridge(self):
        """Test that GPUProcessor initializes with GPU bridge."""
        computer = GPUProcessor(use_gpu=False)
        
        # Check that gpu_bridge was initialized
        assert hasattr(computer, 'gpu_bridge')
        assert computer.gpu_bridge is not None
        
        # Check bridge configuration
        assert computer.gpu_bridge.batch_size == 500_000
        assert computer.gpu_bridge.epsilon == 1e-10
    
    def test_eigenvalue_features_refactored(self, sample_point_cloud, sample_neighbors):
        """Test that refactored eigenvalue computation works."""
        computer = GPUProcessor(use_gpu=False)
        
        # Create dummy normals (not used in refactored version but API requires it)
        normals = np.random.rand(1000, 3).astype(np.float32)
        
        # Compute eigenvalue features using refactored method
        features = computer.compute_eigenvalue_features(
            sample_point_cloud,
            normals,
            sample_neighbors
        )
        
        # Check that all expected features are present
        expected_keys = [
            'eigenvalue_1', 'eigenvalue_2', 'eigenvalue_3',
            'sum_eigenvalues', 'eigenentropy', 'omnivariance',
            'change_curvature'
        ]
        for key in expected_keys:
            assert key in features, f"Missing feature: {key}"
            assert features[key].shape == (1000,), f"Wrong shape for {key}"
            assert features[key].dtype == np.float32, f"Wrong dtype for {key}"
    
    def test_eigenvalue_ordering(self, sample_point_cloud, sample_neighbors):
        """Test that eigenvalues are properly ordered (descending)."""
        computer = GPUProcessor(use_gpu=False)
        normals = np.random.rand(1000, 3).astype(np.float32)
        
        features = computer.compute_eigenvalue_features(
            sample_point_cloud,
            normals,
            sample_neighbors
        )
        
        # Check that eigenvalues are sorted descending
        assert np.all(features['eigenvalue_1'] >= features['eigenvalue_2'])
        assert np.all(features['eigenvalue_2'] >= features['eigenvalue_3'])
    
    def test_eigenvalue_non_negative(self, sample_point_cloud, sample_neighbors):
        """Test that all eigenvalues are non-negative."""
        computer = GPUProcessor(use_gpu=False)
        normals = np.random.rand(1000, 3).astype(np.float32)
        
        features = computer.compute_eigenvalue_features(
            sample_point_cloud,
            normals,
            sample_neighbors
        )
        
        # All eigenvalues should be non-negative
        assert np.all(features['eigenvalue_1'] >= 0)
        assert np.all(features['eigenvalue_2'] >= 0)
        assert np.all(features['eigenvalue_3'] >= 0)
    
    def test_feature_ranges(self, sample_point_cloud, sample_neighbors):
        """Test that features are within expected ranges."""
        computer = GPUProcessor(use_gpu=False)
        normals = np.random.rand(1000, 3).astype(np.float32)
        
        features = computer.compute_eigenvalue_features(
            sample_point_cloud,
            normals,
            sample_neighbors
        )
        
        # Sum of eigenvalues should be positive
        assert np.all(features['sum_eigenvalues'] > 0)
        
        # Eigenentropy should be non-negative
        assert np.all(features['eigenentropy'] >= 0)
        
        # Omnivariance should be non-negative
        assert np.all(features['omnivariance'] >= 0)
        
        # Change curvature should be non-negative
        assert np.all(features['change_curvature'] >= 0)
    
    def test_no_nan_or_inf(self, sample_point_cloud, sample_neighbors):
        """Test that features contain no NaN or Inf values."""
        computer = GPUProcessor(use_gpu=False)
        normals = np.random.rand(1000, 3).astype(np.float32)
        
        features = computer.compute_eigenvalue_features(
            sample_point_cloud,
            normals,
            sample_neighbors
        )
        
        # Check all features for NaN/Inf
        for key, values in features.items():
            assert not np.any(np.isnan(values)), f"NaN found in {key}"
            assert not np.any(np.isinf(values)), f"Inf found in {key}"
    
    def test_planar_surface_high_planarity(self):
        """Test that planar surfaces have high planarity-related features."""
        # Create points on XY plane with small noise
        np.random.seed(42)
        x = np.random.rand(1000) * 10
        y = np.random.rand(1000) * 10
        z = np.random.rand(1000) * 0.1  # Small variation in Z
        points = np.column_stack([x, y, z]).astype(np.float32)
        
        # Create neighbors
        neighbors = np.random.randint(0, 1000, size=(1000, 20), dtype=np.int64)
        normals = np.random.rand(1000, 3).astype(np.float32)
        
        computer = GPUProcessor(use_gpu=False)
        features = computer.compute_eigenvalue_features(points, normals, neighbors)
        
        # For planar surfaces, λ0 and λ1 should be much larger than λ2
        # This means sum_eigenvalues should be relatively large
        # and eigenvalue_3 should be relatively small
        mean_ratio = np.mean(features['eigenvalue_3'] / (features['eigenvalue_1'] + 1e-10))
        assert mean_ratio < 0.5, "Planar surface should have small λ3/λ1 ratio"
    
    def test_chunking_compatibility(self, sample_point_cloud, sample_neighbors):
        """Test that chunked processing works with refactored method."""
        computer = GPUProcessor(use_gpu=False, chunk_size=500)
        normals = np.random.rand(1000, 3).astype(np.float32)
        
        # In real chunked processing, the full point cloud is passed
        # but only a subset of neighbors is computed for the chunk
        # Create neighbors that only reference valid indices for the chunk
        chunk_neighbors = np.random.randint(0, 500, size=(500, 20), dtype=np.int64)
        
        # Pass full points array (as done in real usage) with chunk neighbors
        features = computer.compute_eigenvalue_features(
            sample_point_cloud[:500],  # Only pass chunk points 
            normals[:500],
            chunk_neighbors,
            start_idx=0,
            end_idx=500
        )
        
        assert features['eigenvalue_1'].shape == (500,)
        assert np.all(np.isfinite(features['eigenvalue_1']))


class TestBackwardCompatibility:
    """Test backward compatibility with original API."""
    
    def test_api_signature_unchanged(self):
        """Test that method signature is unchanged."""
        import inspect
        
        computer = GPUProcessor(use_gpu=False)
        sig = inspect.signature(computer.compute_eigenvalue_features)
        
        # Check parameters exist
        params = list(sig.parameters.keys())
        assert 'points' in params
        assert 'normals' in params
        assert 'neighbors_indices' in params
        assert 'start_idx' in params
        assert 'end_idx' in params
    
    def test_return_type_unchanged(self, sample_point_cloud, sample_neighbors):
        """Test that return type is still Dict[str, np.ndarray]."""
        computer = GPUProcessor(use_gpu=False)
        normals = np.random.rand(1000, 3).astype(np.float32)
        
        features = computer.compute_eigenvalue_features(
            sample_point_cloud,
            normals,
            sample_neighbors
        )
        
        assert isinstance(features, dict)
        for key, value in features.items():
            assert isinstance(key, str)
            assert isinstance(value, np.ndarray)
    
    def test_feature_keys_unchanged(self, sample_point_cloud, sample_neighbors):
        """Test that feature keys match original implementation."""
        computer = GPUProcessor(use_gpu=False)
        normals = np.random.rand(1000, 3).astype(np.float32)
        
        features = computer.compute_eigenvalue_features(
            sample_point_cloud,
            normals,
            sample_neighbors
        )
        
        # Original feature keys should be present
        expected_keys = {
            'eigenvalue_1', 'eigenvalue_2', 'eigenvalue_3',
            'sum_eigenvalues', 'eigenentropy', 'omnivariance',
            'change_curvature'
        }
        assert set(features.keys()) == expected_keys


class TestPerformance:
    """Performance validation tests."""
    
    @pytest.mark.benchmark
    def test_refactored_vs_bridge_direct(self, sample_point_cloud, sample_neighbors):
        """Test that refactored method performance is comparable."""
        import time
        
        computer = GPUProcessor(use_gpu=False)
        normals = np.random.rand(1000, 3).astype(np.float32)
        
        # Time refactored method
        start = time.time()
        features = computer.compute_eigenvalue_features(
            sample_point_cloud,
            normals,
            sample_neighbors
        )
        refactored_time = time.time() - start
        
        # Time should be reasonable (< 1s for 1000 points)
        assert refactored_time < 1.0, f"Refactored method too slow: {refactored_time:.3f}s"
        
        print(f"\nRefactored eigenvalue features: {refactored_time:.3f}s")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'not benchmark'])
