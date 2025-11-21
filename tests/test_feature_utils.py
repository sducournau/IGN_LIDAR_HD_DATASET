"""
Tests for Feature Computation Utilities

Tests the shared utilities in ign_lidar/features/utils.py
"""

import pytest
import numpy as np

# Import both possible KDTree types for compatibility
try:
    from ign_lidar.optimization import KDTree as GPUKDTree
    HAS_GPU_KDTREE = True
except ImportError:
    HAS_GPU_KDTREE = False
    GPUKDTree = None

from sklearn.neighbors import KDTree as CPUKDTree

from ign_lidar.features.utils import (
    build_kdtree,
    compute_local_eigenvalues,
    validate_point_cloud,
    validate_normals,
    validate_k_neighbors,
    get_optimal_leaf_size,
    quick_kdtree,
)


class TestBuildKDTree:
    """Test KDTree building functionality."""
    
    def test_build_kdtree_default(self):
        """Test KDTree with default parameters."""
        points = np.random.rand(100, 3)
        tree = build_kdtree(points)
        
        # Accept both GPU and CPU KDTree types
        if HAS_GPU_KDTREE:
            assert isinstance(tree, (CPUKDTree, GPUKDTree))
        else:
            assert isinstance(tree, CPUKDTree)
        assert tree.data.shape == (100, 3) or tree.n == 100  # GPU uses .n, CPU uses .data
    
    def test_build_kdtree_custom_leaf_size(self):
        """Test KDTree with custom leaf size."""
        points = np.random.rand(100, 3)
        tree = build_kdtree(points, leaf_size=40)
        
        # Accept both GPU and CPU KDTree types
        if HAS_GPU_KDTREE:
            assert isinstance(tree, (CPUKDTree, GPUKDTree))
        else:
            assert isinstance(tree, CPUKDTree)
        # Note: GPU KDTree (FAISS) doesn't expose leaf_size, CPU doesn't expose it either
        # We just verify it builds successfully
    
    def test_build_kdtree_different_metrics(self):
        """Test KDTree with different distance metrics."""
        points = np.random.rand(100, 3)
        
        for metric in ['euclidean', 'manhattan', 'chebyshev']:
            tree = build_kdtree(points, metric=metric)
            # Accept both GPU and CPU KDTree types
            # Note: GPU only supports euclidean, but build_kdtree handles fallback
            if HAS_GPU_KDTREE:
                assert isinstance(tree, (CPUKDTree, GPUKDTree))
            else:
                assert isinstance(tree, CPUKDTree)
    
    def test_quick_kdtree(self):
        """Test quick_kdtree convenience function."""
        points = np.random.rand(100, 3)
        tree = quick_kdtree(points)
        
        # Accept both GPU and CPU KDTree types
        if HAS_GPU_KDTREE:
            assert isinstance(tree, (CPUKDTree, GPUKDTree))
        else:
            assert isinstance(tree, CPUKDTree)


class TestComputeLocalEigenvalues:
    """Test eigenvalue computation functionality."""
    
    def test_compute_eigenvalues_basic(self):
        """Test basic eigenvalue computation."""
        # Create simple planar surface
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 10, 50)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(2500)])
        
        eigenvalues = compute_local_eigenvalues(points, k=10)
        
        # Check shape
        assert eigenvalues.shape == (2500, 3)
        
        # Check eigenvalues are sorted (λ3 ≤ λ2 ≤ λ1)
        assert np.all(eigenvalues[:, 0] <= eigenvalues[:, 1])
        assert np.all(eigenvalues[:, 1] <= eigenvalues[:, 2])
        
        # For planar surface, smallest eigenvalue should be near zero
        assert np.mean(eigenvalues[:, 0]) < 0.1
    
    def test_compute_eigenvalues_with_tree(self):
        """Test eigenvalue computation with pre-built tree."""
        points = np.random.rand(100, 3)
        tree = build_kdtree(points)
        
        eigenvalues = compute_local_eigenvalues(points, tree=tree, k=10)
        
        assert eigenvalues.shape == (100, 3)
    
    def test_compute_eigenvalues_return_tree(self):
        """Test eigenvalue computation with tree return."""
        points = np.random.rand(100, 3)
        
        eigenvalues, tree = compute_local_eigenvalues(
            points, k=10, return_tree=True
        )
        
        assert eigenvalues.shape == (100, 3)
        # Accept both GPU and CPU KDTree types
        if HAS_GPU_KDTREE:
            assert isinstance(tree, (CPUKDTree, GPUKDTree))
        else:
            assert isinstance(tree, CPUKDTree)
    
    def test_compute_eigenvalues_different_k(self):
        """Test eigenvalue computation with different k values."""
        points = np.random.rand(100, 3)
        
        for k in [5, 10, 20, 30]:
            eigenvalues = compute_local_eigenvalues(points, k=k)
            assert eigenvalues.shape == (100, 3)
            assert np.all(np.isfinite(eigenvalues))
    
    def test_compute_eigenvalues_sorted(self):
        """Test that eigenvalues are properly sorted."""
        points = np.random.rand(100, 3)
        eigenvalues = compute_local_eigenvalues(points, k=10)
        
        # λ3 ≤ λ2 ≤ λ1
        assert np.all(eigenvalues[:, 0] <= eigenvalues[:, 1])
        assert np.all(eigenvalues[:, 1] <= eigenvalues[:, 2])


class TestValidatePointCloud:
    """Test point cloud validation."""
    
    def test_validate_valid_points(self):
        """Test validation passes for valid points."""
        points = np.random.rand(100, 3)
        validate_point_cloud(points)  # Should not raise
    
    def test_validate_wrong_type(self):
        """Test validation fails for wrong type."""
        points = [[1, 2, 3], [4, 5, 6]]  # List, not ndarray
        
        with pytest.raises(TypeError, match="must be numpy array"):
            validate_point_cloud(points)
    
    def test_validate_wrong_shape(self):
        """Test validation fails for wrong shape."""
        # Wrong dimensionality
        points = np.random.rand(100, 4)
        with pytest.raises(ValueError, match="must be \\[N, 3\\]"):
            validate_point_cloud(points)
        
        # 1D array
        points = np.random.rand(100)
        with pytest.raises(ValueError, match="must be \\[N, 3\\]"):
            validate_point_cloud(points)
    
    def test_validate_too_few_points(self):
        """Test validation fails for too few points."""
        points = np.random.rand(5, 3)
        
        with pytest.raises(ValueError, match="at least 10 points"):
            validate_point_cloud(points, min_points=10)
    
    def test_validate_nan_values(self):
        """Test validation fails for NaN values."""
        points = np.random.rand(100, 3)
        points[10, 1] = np.nan
        
        with pytest.raises(ValueError, match="contains.*NaN"):
            validate_point_cloud(points, check_finite=True)
    
    def test_validate_inf_values(self):
        """Test validation fails for Inf values."""
        points = np.random.rand(100, 3)
        points[10, 1] = np.inf
        
        with pytest.raises(ValueError, match="contains.*Inf"):
            validate_point_cloud(points, check_finite=True)
    
    def test_validate_skip_finite_check(self):
        """Test validation skips finite check when disabled."""
        points = np.random.rand(100, 3)
        points[10, 1] = np.nan
        
        # Should not raise when check_finite=False
        validate_point_cloud(points, check_finite=False)
    
    def test_validate_custom_param_name(self):
        """Test custom parameter name in error messages."""
        points = [[1, 2, 3]]  # Wrong type
        
        with pytest.raises(TypeError, match="my_points"):
            validate_point_cloud(points, param_name="my_points")


class TestValidateNormals:
    """Test normal vector validation."""
    
    def test_validate_valid_normals(self):
        """Test validation passes for valid normals."""
        normals = np.random.rand(100, 3)
        validate_normals(normals, num_points=100)  # Should not raise
    
    def test_validate_wrong_type(self):
        """Test validation fails for wrong type."""
        normals = [[1, 0, 0], [0, 1, 0]]  # List, not ndarray
        
        with pytest.raises(TypeError, match="must be numpy array"):
            validate_normals(normals, num_points=2)
    
    def test_validate_wrong_shape(self):
        """Test validation fails for wrong shape."""
        normals = np.random.rand(100, 3)
        
        # Wrong number of points
        with pytest.raises(ValueError, match="must be"):
            validate_normals(normals, num_points=50)
        
        # Wrong dimensionality
        normals = np.random.rand(100, 4)
        with pytest.raises(ValueError, match="must be"):
            validate_normals(normals, num_points=100)
    
    def test_validate_nan_values(self):
        """Test validation fails for NaN values."""
        normals = np.random.rand(100, 3)
        normals[10, 1] = np.nan
        
        with pytest.raises(ValueError, match="contains.*NaN"):
            validate_normals(normals, num_points=100, check_finite=True)
    
    def test_validate_skip_finite_check(self):
        """Test validation skips finite check when disabled."""
        normals = np.random.rand(100, 3)
        normals[10, 1] = np.nan
        
        # Should not raise when check_finite=False
        validate_normals(normals, num_points=100, check_finite=False)
    
    def test_validate_custom_param_name(self):
        """Test custom parameter name in error messages."""
        normals = [[1, 0, 0]]  # Wrong type
        
        with pytest.raises(TypeError, match="my_normals"):
            validate_normals(normals, num_points=1, param_name="my_normals")


class TestValidateKNeighbors:
    """Test k-neighbors validation."""
    
    def test_validate_valid_k(self):
        """Test validation passes for valid k."""
        validate_k_neighbors(k=10, num_points=100)  # Should not raise
        validate_k_neighbors(k=1, num_points=100)   # Should not raise
        validate_k_neighbors(k=100, num_points=100) # Should not raise
    
    def test_validate_wrong_type(self):
        """Test validation fails for wrong type."""
        with pytest.raises(TypeError, match="must be integer"):
            validate_k_neighbors(k=10.5, num_points=100)
        
        with pytest.raises(TypeError, match="must be integer"):
            validate_k_neighbors(k="10", num_points=100)
    
    def test_validate_negative_k(self):
        """Test validation fails for negative k."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_k_neighbors(k=-10, num_points=100)
    
    def test_validate_zero_k(self):
        """Test validation fails for zero k."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_k_neighbors(k=0, num_points=100)
    
    def test_validate_k_exceeds_points(self):
        """Test validation fails when k > num_points."""
        with pytest.raises(ValueError, match="exceeds number of points"):
            validate_k_neighbors(k=200, num_points=100)
    
    def test_validate_custom_param_name(self):
        """Test custom parameter name in error messages."""
        with pytest.raises(TypeError, match="num_neighbors"):
            validate_k_neighbors(k=10.5, num_points=100, param_name="num_neighbors")


class TestGetOptimalLeafSize:
    """Test optimal leaf size calculation."""
    
    def test_small_dataset(self):
        """Test leaf size for small dataset."""
        leaf_size = get_optimal_leaf_size(num_points=5000)
        assert leaf_size == 20
    
    def test_medium_dataset(self):
        """Test leaf size for medium dataset."""
        leaf_size = get_optimal_leaf_size(num_points=500_000)
        assert leaf_size == 30
    
    def test_large_dataset(self):
        """Test leaf size for large dataset."""
        leaf_size = get_optimal_leaf_size(num_points=5_000_000)
        assert leaf_size == 40
    
    def test_gpu_fallback(self):
        """Test leaf size for GPU fallback."""
        # GPU fallback always uses 40
        leaf_size = get_optimal_leaf_size(num_points=1000, use_gpu_fallback=True)
        assert leaf_size == 40
        
        leaf_size = get_optimal_leaf_size(num_points=5_000_000, use_gpu_fallback=True)
        assert leaf_size == 40


class TestIntegration:
    """Integration tests combining multiple utilities."""
    
    def test_full_workflow(self):
        """Test complete workflow using all utilities."""
        # Create test data
        points = np.random.rand(1000, 3)
        
        # Validate input
        validate_point_cloud(points, min_points=100)
        validate_k_neighbors(k=20, num_points=len(points))
        
        # Build KDTree
        tree = build_kdtree(points)
        
        # Compute eigenvalues
        eigenvalues = compute_local_eigenvalues(points, tree=tree, k=20)
        
        # Validate results
        assert eigenvalues.shape == (1000, 3)
        assert np.all(np.isfinite(eigenvalues))
        assert np.all(eigenvalues[:, 0] <= eigenvalues[:, 1])
        assert np.all(eigenvalues[:, 1] <= eigenvalues[:, 2])
    
    def test_error_propagation(self):
        """Test that errors propagate correctly through workflow."""
        # Invalid points
        points = np.random.rand(100, 4)  # Wrong shape
        
        with pytest.raises(ValueError):
            validate_point_cloud(points)
        
        # Invalid k
        points = np.random.rand(100, 3)
        with pytest.raises(ValueError):
            validate_k_neighbors(k=200, num_points=len(points))
    
    def test_realistic_point_cloud(self):
        """Test with realistic point cloud (planar building facade)."""
        # Create vertical planar surface
        x = np.linspace(0, 10, 100)
        z = np.linspace(0, 5, 50)
        xx, zz = np.meshgrid(x, z)
        points = np.column_stack([xx.ravel(), np.zeros(5000), zz.ravel()])
        
        # Add small noise
        points += np.random.randn(5000, 3) * 0.01
        
        # Compute eigenvalues
        eigenvalues = compute_local_eigenvalues(points, k=10)
        
        # For planar surface, λ3 should be very small
        # and λ1, λ2 should be similar (isotropic in plane)
        # Note: With small noise and k=10, eigenvalues will be small
        assert np.mean(eigenvalues[:, 0]) < 0.01  # λ3 ≈ 0 (perpendicular to plane)
        assert np.mean(eigenvalues[:, 1]) > 0.001 # λ2 > 0 (in-plane variation)
        assert np.mean(eigenvalues[:, 2]) > 0.001 # λ1 > 0 (in-plane variation)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
