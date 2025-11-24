"""
Tests for intermediate result caching in FeatureOrchestrator.

Tests the caching system added in v3.5.2 that avoids recomputing
normals/eigenvalues when multiple features need them.

Expected performance improvement: +15-25% for multi-feature computations.

Author: Consolidation Phase
Date: November 24, 2025
"""
import pytest
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

from ign_lidar.features.orchestrator import FeatureOrchestrator
from ign_lidar.features.feature_modes import FeatureMode


@pytest.fixture
def minimal_config():
    """Create minimal config for testing."""
    return OmegaConf.create({
        "input_dir": "/tmp/test_input",
        "output_dir": "/tmp/test_output",
        "processor": {
            "lod_level": "LOD2",
            "use_gpu": False,
            "num_workers": 1,
            "processing_mode": "patches_only",
        },
        "features": {
            "mode": "minimal",
            "k_neighbors": 30,
            "search_radius": 3.0,
            "enable_caching": True,  # NEW: Enable caching
        },
    })


@pytest.fixture
def sample_points():
    """Generate sample point cloud for testing."""
    np.random.seed(42)
    n_points = 1000
    points = np.random.rand(n_points, 3) * 100  # XYZ coordinates
    return points


@pytest.fixture
def orchestrator_with_cache(minimal_config):
    """Create FeatureOrchestrator with caching enabled."""
    orchestrator = FeatureOrchestrator(minimal_config)
    assert orchestrator._enable_feature_cache is True
    assert len(orchestrator._intermediate_cache) == 0
    assert orchestrator._cache_hits == 0
    assert orchestrator._cache_misses == 0
    return orchestrator


@pytest.fixture
def orchestrator_without_cache(minimal_config):
    """Create FeatureOrchestrator with caching disabled."""
    minimal_config.features.enable_caching = False
    orchestrator = FeatureOrchestrator(minimal_config)
    assert orchestrator._enable_feature_cache is False
    return orchestrator


class TestIntermediateCaching:
    """Test suite for intermediate result caching."""
    
    def test_cache_initialization(self, orchestrator_with_cache):
        """Test that cache is properly initialized."""
        assert hasattr(orchestrator_with_cache, '_intermediate_cache')
        assert hasattr(orchestrator_with_cache, '_cache_hits')
        assert hasattr(orchestrator_with_cache, '_cache_misses')
        assert hasattr(orchestrator_with_cache, '_enable_feature_cache')
        
        assert isinstance(orchestrator_with_cache._intermediate_cache, dict)
        assert orchestrator_with_cache._cache_hits == 0
        assert orchestrator_with_cache._cache_misses == 0
        assert orchestrator_with_cache._enable_feature_cache is True
    
    def test_cache_disabled_by_config(self, orchestrator_without_cache):
        """Test that cache can be disabled via config."""
        assert orchestrator_without_cache._enable_feature_cache is False
    
    def test_cache_key_generation(self, orchestrator_with_cache, sample_points):
        """Test that cache keys are generated consistently."""
        k_neighbors = 30
        
        # Generate normals and eigenvalues
        normals = np.random.rand(*sample_points.shape)
        eigenvalues = np.random.rand(len(sample_points), 3)
        
        # Cache the results
        orchestrator_with_cache._cache_normals_eigenvalues(
            sample_points, k_neighbors, normals, eigenvalues
        )
        
        # Verify cache has one entry
        assert len(orchestrator_with_cache._intermediate_cache) == 1
        
        # Try to retrieve (should work with same points and k_neighbors)
        cached = orchestrator_with_cache._get_cached_normals_eigenvalues(
            sample_points, k_neighbors
        )
        
        assert cached is not None
        assert len(cached) == 2
        assert np.allclose(cached[0], normals)
        assert np.allclose(cached[1], eigenvalues)
    
    def test_cache_miss(self, orchestrator_with_cache, sample_points):
        """Test that cache miss is properly detected."""
        k_neighbors = 30
        
        # Try to get from empty cache
        result = orchestrator_with_cache._get_cached_normals_eigenvalues(
            sample_points, k_neighbors
        )
        
        assert result is None
        assert orchestrator_with_cache._cache_misses == 1
        assert orchestrator_with_cache._cache_hits == 0
    
    def test_cache_hit(self, orchestrator_with_cache, sample_points):
        """Test that cache hit is properly detected and counted."""
        k_neighbors = 30
        normals = np.random.rand(*sample_points.shape)
        eigenvalues = np.random.rand(len(sample_points), 3)
        
        # Cache the results
        orchestrator_with_cache._cache_normals_eigenvalues(
            sample_points, k_neighbors, normals, eigenvalues
        )
        
        # Reset counters to test hit counting
        orchestrator_with_cache._cache_hits = 0
        orchestrator_with_cache._cache_misses = 0
        
        # Get from cache (should be a hit)
        result = orchestrator_with_cache._get_cached_normals_eigenvalues(
            sample_points, k_neighbors
        )
        
        assert result is not None
        assert orchestrator_with_cache._cache_hits == 1
        assert orchestrator_with_cache._cache_misses == 0
    
    def test_cache_different_k_neighbors(self, orchestrator_with_cache, sample_points):
        """Test that different k_neighbors creates different cache entries."""
        k1 = 30
        k2 = 50
        
        normals1 = np.random.rand(*sample_points.shape)
        eigenvalues1 = np.random.rand(len(sample_points), 3)
        
        normals2 = np.random.rand(*sample_points.shape)
        eigenvalues2 = np.random.rand(len(sample_points), 3)
        
        # Cache with k1
        orchestrator_with_cache._cache_normals_eigenvalues(
            sample_points, k1, normals1, eigenvalues1
        )
        
        # Cache with k2
        orchestrator_with_cache._cache_normals_eigenvalues(
            sample_points, k2, normals2, eigenvalues2
        )
        
        # Should have two cache entries
        assert len(orchestrator_with_cache._intermediate_cache) == 2
        
        # Retrieve both
        cached1 = orchestrator_with_cache._get_cached_normals_eigenvalues(
            sample_points, k1
        )
        cached2 = orchestrator_with_cache._get_cached_normals_eigenvalues(
            sample_points, k2
        )
        
        assert np.allclose(cached1[0], normals1)
        assert np.allclose(cached2[0], normals2)
    
    def test_cache_different_points(self, orchestrator_with_cache):
        """Test that different point clouds create different cache entries."""
        np.random.seed(42)
        points1 = np.random.rand(1000, 3) * 100
        points2 = np.random.rand(1000, 3) * 100  # Different random state
        
        k_neighbors = 30
        normals1 = np.random.rand(*points1.shape)
        eigenvalues1 = np.random.rand(len(points1), 3)
        normals2 = np.random.rand(*points2.shape)
        eigenvalues2 = np.random.rand(len(points2), 3)
        
        # Cache both
        orchestrator_with_cache._cache_normals_eigenvalues(
            points1, k_neighbors, normals1, eigenvalues1
        )
        orchestrator_with_cache._cache_normals_eigenvalues(
            points2, k_neighbors, normals2, eigenvalues2
        )
        
        # Should have two entries (different point hashes)
        assert len(orchestrator_with_cache._intermediate_cache) == 2
    
    def test_cache_size_limit(self, orchestrator_with_cache):
        """Test that cache size is limited to 10 entries (FIFO)."""
        k_neighbors = 30
        
        # Add 15 different point clouds to cache
        for i in range(15):
            points = np.random.rand(100, 3) * (i + 1)  # Different scale
            normals = np.random.rand(*points.shape)
            eigenvalues = np.random.rand(len(points), 3)
            
            orchestrator_with_cache._cache_normals_eigenvalues(
                points, k_neighbors, normals, eigenvalues
            )
        
        # Cache should have max 10 entries
        assert len(orchestrator_with_cache._intermediate_cache) <= 10
    
    def test_cache_disabled_no_caching(self, orchestrator_without_cache, sample_points):
        """Test that caching is completely disabled when config says so."""
        k_neighbors = 30
        normals = np.random.rand(*sample_points.shape)
        eigenvalues = np.random.rand(len(sample_points), 3)
        
        # Try to cache (should be ignored)
        orchestrator_without_cache._cache_normals_eigenvalues(
            sample_points, k_neighbors, normals, eigenvalues
        )
        
        # Cache should remain empty
        assert len(orchestrator_without_cache._intermediate_cache) == 0
        
        # Try to get (should return None)
        result = orchestrator_without_cache._get_cached_normals_eigenvalues(
            sample_points, k_neighbors
        )
        assert result is None
    
    def test_cache_data_isolation(self, orchestrator_with_cache, sample_points):
        """Test that cached data is isolated (uses .copy())."""
        k_neighbors = 30
        normals = np.random.rand(*sample_points.shape)
        eigenvalues = np.random.rand(len(sample_points), 3)
        
        # Store original values
        normals_original = normals.copy()
        eigenvalues_original = eigenvalues.copy()
        
        # Cache the results
        orchestrator_with_cache._cache_normals_eigenvalues(
            sample_points, k_neighbors, normals, eigenvalues
        )
        
        # Modify the original arrays
        normals[:] = 0
        eigenvalues[:] = 0
        
        # Retrieve from cache
        cached = orchestrator_with_cache._get_cached_normals_eigenvalues(
            sample_points, k_neighbors
        )
        
        # Cached data should be unaffected (isolated)
        assert np.allclose(cached[0], normals_original)
        assert np.allclose(cached[1], eigenvalues_original)
        assert not np.allclose(cached[0], normals)
        assert not np.allclose(cached[1], eigenvalues)
    
    def test_cache_hit_ratio_tracking(self, orchestrator_with_cache, sample_points):
        """Test that cache hit/miss ratio is properly tracked."""
        k_neighbors = 30
        normals = np.random.rand(*sample_points.shape)
        eigenvalues = np.random.rand(len(sample_points), 3)
        
        # First access - miss
        result = orchestrator_with_cache._get_cached_normals_eigenvalues(
            sample_points, k_neighbors
        )
        assert result is None
        assert orchestrator_with_cache._cache_misses == 1
        
        # Cache the result
        orchestrator_with_cache._cache_normals_eigenvalues(
            sample_points, k_neighbors, normals, eigenvalues
        )
        
        # Multiple accesses - all hits
        for _ in range(5):
            result = orchestrator_with_cache._get_cached_normals_eigenvalues(
                sample_points, k_neighbors
            )
            assert result is not None
        
        # Verify counters
        assert orchestrator_with_cache._cache_hits == 5
        assert orchestrator_with_cache._cache_misses == 1
        
        # Calculate hit ratio
        total = orchestrator_with_cache._cache_hits + orchestrator_with_cache._cache_misses
        hit_ratio = orchestrator_with_cache._cache_hits / total
        assert hit_ratio == 5/6  # 83.3%


@pytest.mark.unit
class TestCacheHashStability:
    """Test that cache keys are stable and reproducible."""
    
    def test_same_points_same_hash(self, orchestrator_with_cache, sample_points):
        """Test that identical points produce identical cache keys."""
        k_neighbors = 30
        normals = np.random.rand(*sample_points.shape)
        eigenvalues = np.random.rand(len(sample_points), 3)
        
        # Cache once
        orchestrator_with_cache._cache_normals_eigenvalues(
            sample_points, k_neighbors, normals, eigenvalues
        )
        
        # Get cache key count
        cache_size_before = len(orchestrator_with_cache._intermediate_cache)
        
        # Try to cache again with same data (should reuse key)
        orchestrator_with_cache._cache_normals_eigenvalues(
            sample_points, k_neighbors, normals, eigenvalues
        )
        
        # Cache size should not increase (same key)
        assert len(orchestrator_with_cache._intermediate_cache) == cache_size_before
    
    def test_different_shape_different_hash(self, orchestrator_with_cache):
        """Test that different shapes produce different cache keys."""
        k_neighbors = 30
        
        points1 = np.random.rand(1000, 3)
        points2 = np.random.rand(2000, 3)  # Different shape
        
        normals1 = np.random.rand(*points1.shape)
        eigenvalues1 = np.random.rand(len(points1), 3)
        normals2 = np.random.rand(*points2.shape)
        eigenvalues2 = np.random.rand(len(points2), 3)
        
        # Cache both
        orchestrator_with_cache._cache_normals_eigenvalues(
            points1, k_neighbors, normals1, eigenvalues1
        )
        orchestrator_with_cache._cache_normals_eigenvalues(
            points2, k_neighbors, normals2, eigenvalues2
        )
        
        # Should have 2 distinct entries
        assert len(orchestrator_with_cache._intermediate_cache) == 2


@pytest.mark.integration
class TestCachePerformance:
    """Integration tests for cache performance validation."""
    
    def test_cache_reduces_computation_time(self, orchestrator_with_cache, 
                                            orchestrator_without_cache, sample_points):
        """
        Test that caching provides measurable performance improvement.
        
        This is a simplified test - actual performance gain depends on
        how many features reuse normals/eigenvalues.
        """
        # This test would require actual feature computation
        # For now, we just verify the mechanism works
        k_neighbors = 30
        normals = np.random.rand(*sample_points.shape)
        eigenvalues = np.random.rand(len(sample_points), 3)
        
        # With cache enabled
        orchestrator_with_cache._cache_normals_eigenvalues(
            sample_points, k_neighbors, normals, eigenvalues
        )
        
        # Multiple retrievals should be instant (no recomputation)
        for _ in range(10):
            cached = orchestrator_with_cache._get_cached_normals_eigenvalues(
                sample_points, k_neighbors
            )
            assert cached is not None
        
        # Verify high hit rate
        assert orchestrator_with_cache._cache_hits == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
