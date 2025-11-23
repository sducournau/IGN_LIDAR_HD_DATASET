"""
Tests for KNN Engine Radius Search

Validates the new radius_search functionality added in v3.6.0.

Author: Phase 1+ Consolidation
Date: November 23, 2025
"""

import pytest
import numpy as np
from ign_lidar.optimization import KNNEngine, radius_search


class TestKNNEngineRadiusSearch:
    """Test radius search functionality in KNNEngine."""
    
    @pytest.fixture
    def sample_points(self):
        """Create sample 3D point cloud."""
        np.random.seed(42)
        # Create clustered points for predictable radius search results
        points = []
        
        # Cluster 1: around origin
        points.append(np.random.randn(50, 3) * 0.5)
        
        # Cluster 2: around [10, 10, 10]
        points.append(np.random.randn(50, 3) * 0.5 + np.array([10, 10, 10]))
        
        # Cluster 3: around [20, 0, 0]
        points.append(np.random.randn(50, 3) * 0.5 + np.array([20, 0, 0]))
        
        return np.vstack(points).astype(np.float32)
    
    def test_radius_search_sklearn(self, sample_points):
        """Test radius search with sklearn backend."""
        engine = KNNEngine(backend='sklearn')
        
        # Search within 2.0 units
        distances, indices = engine.radius_search(sample_points, radius=2.0)
        
        # Check output structure
        assert len(distances) == len(sample_points)
        assert len(indices) == len(sample_points)
        
        # Check that results are lists (variable length)
        assert isinstance(distances, (list, np.ndarray))
        assert isinstance(indices, (list, np.ndarray))
        
        # Verify at least some points found neighbors
        neighbor_counts = [len(idx) for idx in indices]
        assert min(neighbor_counts) >= 1  # At least self
        assert max(neighbor_counts) > 1   # Some points have multiple neighbors
        
        # Verify distances are within radius
        for i, dists in enumerate(distances):
            assert np.all(dists <= 2.0), f"Point {i} has distances > radius"
            assert np.all(dists >= 0), f"Point {i} has negative distances"
    
    def test_radius_search_with_max_neighbors(self, sample_points):
        """Test radius search with max_neighbors limit."""
        engine = KNNEngine(backend='sklearn')
        
        # Large radius to get many neighbors, but limit to 10
        distances, indices = engine.radius_search(
            sample_points,
            radius=5.0,
            max_neighbors=10
        )
        
        # Check that no query has more than 10 neighbors
        for i, idx in enumerate(indices):
            assert len(idx) <= 10, f"Point {i} has {len(idx)} neighbors > 10"
    
    def test_radius_search_separate_query(self, sample_points):
        """Test radius search with separate query points."""
        engine = KNNEngine(backend='sklearn')
        
        # Use first 10 points as reference
        ref_points = sample_points[:10]
        
        # Use points 10-20 as queries
        query_points = sample_points[10:20]
        
        distances, indices = engine.radius_search(
            ref_points,
            radius=3.0,
            query_points=query_points
        )
        
        # Check output size matches query size
        assert len(distances) == len(query_points)
        assert len(indices) == len(query_points)
        
        # Verify all indices point to reference points
        for idx_list in indices:
            assert np.all(idx_list < len(ref_points))
            assert np.all(idx_list >= 0)
    
    def test_radius_search_empty_results(self):
        """Test radius search with no neighbors (small radius)."""
        # Create widely spaced points
        points = np.array([
            [0, 0, 0],
            [100, 0, 0],
            [0, 100, 0],
            [0, 0, 100],
        ], dtype=np.float32)
        
        engine = KNNEngine(backend='sklearn')
        
        # Very small radius - only self should be found
        distances, indices = engine.radius_search(points, radius=0.1)
        
        # Each point should only find itself
        for i, idx_list in enumerate(indices):
            assert len(idx_list) == 1, f"Point {i} should only find itself"
            assert idx_list[0] == i, f"Point {i} self-reference incorrect"
    
    def test_radius_search_consistency_with_knn(self, sample_points):
        """Test that radius search is consistent with k-NN for small radius."""
        engine = KNNEngine(backend='sklearn')
        
        # K-NN search
        knn_distances, knn_indices = engine.search(sample_points, k=5)
        
        # Radius search with radius = max distance from k-NN
        max_knn_dist = np.max(knn_distances[:, -1])
        radius_distances, radius_indices = engine.radius_search(
            sample_points,
            radius=max_knn_dist * 1.1  # Slightly larger to ensure all k neighbors
        )
        
        # Check that k-NN neighbors are subset of radius neighbors
        for i in range(len(sample_points)):
            knn_set = set(knn_indices[i])
            radius_set = set(radius_indices[i])
            
            # All k-NN neighbors should be in radius results
            assert knn_set.issubset(radius_set), f"Point {i}: k-NN not subset of radius"


class TestRadiusSearchConvenienceFunction:
    """Test convenience function for radius search."""
    
    @pytest.fixture
    def points(self):
        """Simple test points."""
        np.random.seed(42)
        return np.random.randn(100, 3).astype(np.float32)
    
    def test_radius_search_function(self, points):
        """Test radius_search convenience function."""
        distances, indices = radius_search(points, radius=2.0)
        
        assert len(distances) == len(points)
        assert len(indices) == len(points)
        
        # Verify distances within radius
        for dists in distances:
            if len(dists) > 0:
                assert np.all(dists <= 2.0)
    
    def test_radius_search_with_backend_specification(self, points):
        """Test specifying backend explicitly."""
        distances, indices = radius_search(
            points,
            radius=2.0,
            backend='sklearn'
        )
        
        assert len(distances) == len(points)
    
    def test_radius_search_with_max_neighbors(self, points):
        """Test max_neighbors parameter."""
        distances, indices = radius_search(
            points,
            radius=5.0,
            max_neighbors=15
        )
        
        for idx_list in indices:
            assert len(idx_list) <= 15


class TestRadiusSearchIntegration:
    """Integration tests with normals computation."""
    
    def test_normals_with_radius_search(self):
        """Test that normals computation works with radius search."""
        from ign_lidar.features.compute import compute_normals
        
        # Create simple plane
        np.random.seed(42)
        x = np.random.rand(100) * 10
        y = np.random.rand(100) * 10
        z = np.zeros(100)  # Flat plane
        points = np.column_stack([x, y, z]).astype(np.float32)
        
        # Compute normals with radius search
        normals, eigenvalues = compute_normals(
            points,
            search_radius=2.0  # Use radius instead of k_neighbors
        )
        
        # Check output shape
        assert normals.shape == (100, 3)
        assert eigenvalues.shape == (100, 3)
        
        # For a flat plane, normals should point up (0, 0, ±1)
        # Allow some noise from random positions
        z_component = np.abs(normals[:, 2])
        assert np.mean(z_component) > 0.8, "Normals should be mostly vertical"
    
    def test_normals_radius_vs_knn(self):
        """Compare normals from radius vs k-NN search."""
        from ign_lidar.features.compute import compute_normals
        
        # Create test data
        np.random.seed(42)
        points = np.random.randn(50, 3).astype(np.float32) * 0.5
        
        # Compute with k-NN
        normals_knn, _ = compute_normals(points, k_neighbors=10)
        
        # Compute with radius (chosen to give ~10 neighbors on average)
        normals_radius, _ = compute_normals(points, search_radius=1.5)
        
        # Results should be similar (not identical due to different neighbor sets)
        # Check that most normals point in similar directions
        dot_products = np.sum(normals_knn * normals_radius, axis=1)
        similarity = np.mean(np.abs(dot_products))
        
        # Lower threshold since different neighbor selection can give different results
        assert similarity > 0.5, f"Normals similarity too low: {similarity:.2f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
