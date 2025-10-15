"""
Tests for features/core/normals.py module.
"""

import numpy as np
import pytest

from ign_lidar.features.core.normals import (
    compute_normals,
    compute_normals_fast,
    compute_normals_accurate,
    CUPY_AVAILABLE,
)


class TestComputeNormals:
    """Test suite for normal computation."""
    
    def test_basic_normals_computation(self):
        """Test basic normal computation on a simple point cloud."""
        # Create a simple planar point cloud (XY plane)
        n_points = 100
        points = np.random.rand(n_points, 2)
        points = np.column_stack([points, np.zeros(n_points)])  # Z = 0
        
        normals, eigenvalues = compute_normals(points, k_neighbors=10, use_gpu=False)
        
        # Check shapes
        assert normals.shape == (n_points, 3)
        assert eigenvalues.shape == (n_points, 3)
        
        # Check that normals are unit vectors
        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)
        
        # For planar points, normals should mostly point up (0, 0, 1)
        # Allow some variation due to noise
        assert np.mean(np.abs(normals[:, 2])) > 0.8
    
    def test_normals_on_sphere(self):
        """Test normal computation on a sphere."""
        # Create points on a unit sphere
        n_points = 200
        theta = np.random.rand(n_points) * 2 * np.pi
        phi = np.random.rand(n_points) * np.pi
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        points = np.column_stack([x, y, z])
        
        normals, eigenvalues = compute_normals(points, k_neighbors=15)
        
        # Check shapes
        assert normals.shape == (n_points, 3)
        assert eigenvalues.shape == (n_points, 3)
        
        # Normals should be unit vectors
        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)
        
        # For sphere, normals should point radially outward
        # (approximately parallel to position vectors)
        dots = np.abs(np.sum(normals * points, axis=1))
        assert np.mean(dots) > 0.7  # Most should be fairly aligned
    
    def test_input_validation(self):
        """Test input validation."""
        # Invalid input type
        with pytest.raises(ValueError, match="points must be a numpy array"):
            compute_normals([[1, 2, 3]], k_neighbors=5)
        
        # Invalid shape
        with pytest.raises(ValueError, match="points must have shape"):
            compute_normals(np.random.rand(10, 2), k_neighbors=5)
        
        # Not enough points
        with pytest.raises(ValueError, match="Not enough points"):
            compute_normals(np.random.rand(5, 3), k_neighbors=10)
        
        # k_neighbors too small
        with pytest.raises(ValueError, match="k_neighbors must be >= 3"):
            compute_normals(np.random.rand(100, 3), k_neighbors=2)
    
    def test_eigenvalues_sorted(self):
        """Test that eigenvalues are sorted in descending order."""
        points = np.random.rand(100, 3)
        normals, eigenvalues = compute_normals(points, k_neighbors=10)
        
        # Check that eigenvalues are sorted descending
        for i in range(eigenvalues.shape[0]):
            assert eigenvalues[i, 0] >= eigenvalues[i, 1]
            assert eigenvalues[i, 1] >= eigenvalues[i, 2]
    
    def test_normals_fast(self):
        """Test fast normal computation convenience function."""
        points = np.random.rand(100, 3)
        normals = compute_normals_fast(points)
        
        assert normals.shape == (100, 3)
        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)
    
    def test_normals_accurate(self):
        """Test accurate normal computation with more neighbors."""
        points = np.random.rand(100, 3)
        normals, eigenvalues = compute_normals_accurate(points, k=50)
        
        assert normals.shape == (100, 3)
        assert eigenvalues.shape == (100, 3)
        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_gpu_computation(self):
        """Test GPU normal computation."""
        points = np.random.rand(100, 3)
        
        # Compute on CPU and GPU
        normals_cpu, eigvals_cpu = compute_normals(points, k_neighbors=10, use_gpu=False)
        normals_gpu, eigvals_gpu = compute_normals(points, k_neighbors=10, use_gpu=True)
        
        # Results should be similar (not exact due to numerical differences)
        np.testing.assert_allclose(normals_cpu, normals_gpu, rtol=1e-2, atol=1e-3)
        np.testing.assert_allclose(eigvals_cpu, eigvals_gpu, rtol=1e-2, atol=1e-3)
    
    def test_gpu_unavailable_error(self):
        """Test error when GPU requested but unavailable."""
        if not CUPY_AVAILABLE:
            points = np.random.rand(100, 3)
            with pytest.raises(RuntimeError, match="GPU computation requested but CuPy"):
                compute_normals(points, k_neighbors=10, use_gpu=True)
    
    def test_deterministic_output(self):
        """Test that same input produces same output."""
        points = np.random.rand(50, 3)
        
        normals1, eigvals1 = compute_normals(points, k_neighbors=10)
        normals2, eigvals2 = compute_normals(points, k_neighbors=10)
        
        np.testing.assert_array_equal(normals1, normals2)
        np.testing.assert_array_equal(eigvals1, eigvals2)
    
    def test_large_point_cloud(self):
        """Test normal computation on larger point cloud."""
        # This tests performance and memory handling
        points = np.random.rand(5000, 3)
        normals, eigenvalues = compute_normals(points, k_neighbors=20)
        
        assert normals.shape == (5000, 3)
        assert eigenvalues.shape == (5000, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
