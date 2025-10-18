"""
Tests for features/core/curvature.py module.
"""

import numpy as np
import pytest

from ign_lidar.features.core.curvature import (
    compute_curvature,
    compute_mean_curvature,
    compute_shape_index,
    compute_curvedness,
    compute_all_curvature_features,
    compute_curvature_from_normals,
    compute_curvature_from_normals_batched,
)


class TestCurvatureFeatures:
    """Test suite for curvature feature computation."""
    
    def test_compute_curvature_standard(self):
        """Test standard curvature computation."""
        # Create test eigenvalues (λ1 >= λ2 >= λ3)
        eigenvalues = np.array([
            [1.0, 0.5, 0.1],
            [1.0, 0.8, 0.01],
            [1.0, 0.1, 0.05]
        ])
        
        curvature = compute_curvature(eigenvalues, method='standard')
        
        # Check shape
        assert curvature.shape == (3,)
        
        # Check range [0, 1]
        assert np.all(curvature >= 0)
        assert np.all(curvature <= 1)
        
        # Verify calculation: λ3 / (λ1 + λ2 + λ3)
        expected_0 = 0.1 / (1.0 + 0.5 + 0.1)
        np.testing.assert_allclose(curvature[0], expected_0, rtol=1e-5)
    
    def test_compute_curvature_normalized(self):
        """Test normalized curvature computation."""
        eigenvalues = np.array([[1.0, 0.5, 0.1], [1.0, 0.8, 0.01]])
        
        curvature = compute_curvature(eigenvalues, method='normalized')
        
        # Check shape
        assert curvature.shape == (2,)
        
        # Verify: λ3 / λ1
        expected = np.array([0.1 / 1.0, 0.01 / 1.0])
        np.testing.assert_allclose(curvature, expected, rtol=1e-5)
    
    def test_compute_curvature_gaussian(self):
        """Test Gaussian curvature computation."""
        eigenvalues = np.array([[1.0, 0.5, 0.1]])
        
        curvature = compute_curvature(eigenvalues, method='gaussian')
        
        # Verify: (λ2 * λ3) / (λ1^2)
        expected = (0.5 * 0.1) / (1.0 * 1.0)
        np.testing.assert_allclose(curvature[0], expected, rtol=1e-5)
    
    def test_curvature_input_validation(self):
        """Test input validation."""
        # Invalid input type
        with pytest.raises(ValueError, match="eigenvalues must be a numpy array"):
            compute_curvature([[1, 2, 3]])
        
        # Invalid shape
        with pytest.raises(ValueError, match="eigenvalues must have shape"):
            compute_curvature(np.random.rand(10, 2))
        
        # Unknown method
        eigenvalues = np.random.rand(10, 3)
        with pytest.raises(ValueError, match="Unknown curvature method"):
            compute_curvature(eigenvalues, method='unknown')
    
    def test_compute_mean_curvature(self):
        """Test mean curvature computation."""
        eigenvalues = np.array([[1.0, 0.5, 0.1], [1.0, 0.8, 0.01]])
        
        mean_curv = compute_mean_curvature(eigenvalues)
        
        # Check shape
        assert mean_curv.shape == (2,)
        
        # Verify: (λ2 + λ3) / λ1
        expected = np.array([(0.5 + 0.1) / 1.0, (0.8 + 0.01) / 1.0])
        np.testing.assert_allclose(mean_curv, expected, rtol=1e-5)
    
    def test_compute_shape_index(self):
        """Test shape index computation."""
        eigenvalues = np.array([[1.0, 0.5, 0.1], [1.0, 0.8, 0.01]])
        
        shape_idx = compute_shape_index(eigenvalues)
        
        # Check shape
        assert shape_idx.shape == (2,)
        
        # Check range [-1, 1]
        assert np.all(shape_idx >= -1)
        assert np.all(shape_idx <= 1)
    
    def test_compute_curvedness(self):
        """Test curvedness computation."""
        eigenvalues = np.array([[1.0, 0.5, 0.1], [1.0, 0.8, 0.01]])
        
        curvedness = compute_curvedness(eigenvalues)
        
        # Check shape
        assert curvedness.shape == (2,)
        
        # Curvedness should be positive
        assert np.all(curvedness >= 0)
    
    def test_compute_all_curvature_features(self):
        """Test computation of all curvature features."""
        eigenvalues = np.array([[1.0, 0.5, 0.1], [1.0, 0.8, 0.01]])
        
        features = compute_all_curvature_features(eigenvalues)
        
        # Check that all expected features are present
        assert isinstance(features, dict)
        assert 'curvature' in features
        assert 'mean_curvature' in features
        assert 'shape_index' in features
        assert 'curvedness' in features
        
        # Check shapes
        for key, value in features.items():
            assert value.shape == (2,)
    
    def test_epsilon_handling(self):
        """Test that epsilon prevents division by zero."""
        # Create eigenvalues with very small values
        eigenvalues = np.array([[1e-15, 1e-16, 1e-17]])
        
        # Should not raise error or produce inf/nan
        curvature = compute_curvature(eigenvalues)
        assert np.all(np.isfinite(curvature))
        
        mean_curv = compute_mean_curvature(eigenvalues)
        assert np.all(np.isfinite(mean_curv))
    
    def test_flat_surface(self):
        """Test curvature for flat surface (high λ2, low λ3)."""
        # Flat surface: λ1 ≈ λ2 >> λ3
        eigenvalues = np.array([[1.0, 0.95, 0.01]])
        
        curvature = compute_curvature(eigenvalues)
        
        # Should have low curvature (flat)
        assert curvature[0] < 0.1
    
    def test_sharp_edge(self):
        """Test curvature for sharp edge (low λ2, low λ3)."""
        # Sharp edge: λ1 >> λ2 ≈ λ3
        eigenvalues = np.array([[1.0, 0.1, 0.05]])
        
        curvature = compute_curvature(eigenvalues)
        
        # Should have moderate curvature
        assert 0.01 < curvature[0] < 0.2


class TestNormalBasedCurvature:
    """Test suite for normal-based curvature computation."""
    
    def test_compute_curvature_from_normals_planar(self):
        """Test curvature from normals on a planar surface."""
        np.random.seed(42)
        
        # Create planar surface (all normals point up)
        N = 100
        points = np.random.rand(N, 3).astype(np.float32)
        normals = np.tile([0, 0, 1], (N, 1)).astype(np.float32)
        
        # Create fake neighbor indices
        k = 10
        indices = np.random.randint(0, N, (N, k))
        
        curvature = compute_curvature_from_normals(points, normals, indices)
        
        # Check shape and dtype
        assert curvature.shape == (N,)
        assert curvature.dtype == np.float32
        
        # Planar surface should have very low curvature
        assert np.mean(curvature) < 0.1
        assert np.all(curvature >= 0)
    
    def test_compute_curvature_from_normals_curved(self):
        """Test curvature from normals on a curved surface."""
        np.random.seed(42)
        
        # Create curved surface with varying normals
        N = 100
        points = np.random.rand(N, 3).astype(np.float32)
        normals = np.random.rand(N, 3).astype(np.float32)
        # Normalize
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        
        # Create fake neighbor indices
        k = 10
        indices = np.random.randint(0, N, (N, k))
        
        curvature = compute_curvature_from_normals(points, normals, indices)
        
        # Check shape
        assert curvature.shape == (N,)
        
        # Random normals should have higher curvature than planar
        assert np.mean(curvature) > 0.3
        assert np.all(curvature >= 0)
    
    def test_compute_curvature_from_normals_batched_cpu(self):
        """Test batched curvature computation on CPU."""
        np.random.seed(42)
        
        # Create test data
        N = 1000
        points = np.random.rand(N, 3).astype(np.float32)
        normals = np.random.rand(N, 3).astype(np.float32)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        
        # Compute curvature with batching
        curvature = compute_curvature_from_normals_batched(
            points, normals, k=10, batch_size=500, use_gpu=False
        )
        
        # Check shape and dtype
        assert curvature.shape == (N,)
        assert curvature.dtype == np.float32
        assert np.all(np.isfinite(curvature))
        assert np.all(curvature >= 0)
    
    @pytest.mark.skipif(
        not hasattr(np, '__version__'),  # Placeholder condition
        reason="GPU test requires cuML"
    )
    def test_compute_curvature_from_normals_gpu(self):
        """Test curvature from normals on GPU (if available)."""
        try:
            import cupy as cp
            
            np.random.seed(42)
            
            # Create test data
            N = 100
            points_cpu = np.random.rand(N, 3).astype(np.float32)
            normals_cpu = np.tile([0, 0, 1], (N, 1)).astype(np.float32)
            
            # Transfer to GPU
            points_gpu = cp.asarray(points_cpu)
            normals_gpu = cp.asarray(normals_cpu)
            
            # Create fake neighbor indices
            k = 10
            indices_gpu = cp.random.randint(0, N, (N, k))
            
            curvature_gpu = compute_curvature_from_normals(
                points_gpu, normals_gpu, indices_gpu
            )
            
            # Check result type
            assert isinstance(curvature_gpu, cp.ndarray)
            assert curvature_gpu.shape == (N,)
            assert curvature_gpu.dtype == cp.float32
            
            # Check values
            curvature_cpu = cp.asnumpy(curvature_gpu)
            assert np.mean(curvature_cpu) < 0.1  # Planar surface
            
        except ImportError:
            pytest.skip("CuPy not available")
    
    def test_compute_curvature_from_normals_consistency(self):
        """Test that CPU and batched methods give consistent results."""
        np.random.seed(42)
        
        # Create test data
        N = 500
        points = np.random.rand(N, 3).astype(np.float32)
        normals = np.random.rand(N, 3).astype(np.float32)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        
        # Method 1: Manual KNN + compute_curvature_from_normals
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=10)
        knn.fit(points)
        _, indices = knn.kneighbors(points)
        curvature1 = compute_curvature_from_normals(points, normals, indices)
        
        # Method 2: Batched convenience function
        curvature2 = compute_curvature_from_normals_batched(
            points, normals, k=10, use_gpu=False
        )
        
        # Results should be very close
        np.testing.assert_allclose(curvature1, curvature2, rtol=1e-5, atol=1e-6)
    
    def test_compute_curvature_from_normals_sphere(self):
        """Test curvature on a spherical surface."""
        # Create points on a sphere
        N = 100
        theta = np.linspace(0, np.pi, int(np.sqrt(N)))
        phi = np.linspace(0, 2*np.pi, int(np.sqrt(N)))
        theta, phi = np.meshgrid(theta, phi)
        
        points = np.stack([
            np.sin(theta).flatten(),
            np.cos(theta).flatten() * np.sin(phi).flatten(),
            np.cos(theta).flatten() * np.cos(phi).flatten()
        ], axis=1).astype(np.float32)
        
        # Normals point outward from sphere origin
        normals = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Compute curvature
        curvature = compute_curvature_from_normals_batched(
            points, normals, k=10, use_gpu=False
        )
        
        # Sphere should have relatively uniform, moderate curvature
        assert curvature.shape == (points.shape[0],)
        assert np.all(curvature >= 0)
        assert np.std(curvature) < 0.5  # Relatively uniform


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
