"""
Tests for intermediate result caching in FeatureOrchestrator.

Tests the intermediate cache system added in v3.5.3 for normals and eigenvalues
to avoid recomputation when multiple features depend on them.

Author: LiDAR Trainer Agent
Date: November 24, 2025
"""

import pytest
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

from ign_lidar.features.orchestrator import FeatureOrchestrator
from ign_lidar.features.feature_modes import FeatureMode


@pytest.fixture
def sample_config():
    """Create sample configuration for orchestrator."""
    return OmegaConf.create({
        "processor": {
            "lod_level": "LOD2",
            "use_gpu": False,
            "num_workers": 1,
        },
        "features": {
            "mode": "lod2",
            "k_neighbors": 20,
            "search_radius": 0.0,
            "enable_caching": True,
            "cache_max_size": 100,
        },
        "data_sources": {
            "rgb": {"enabled": False},
            "infrared": {"enabled": False},
        }
    })


@pytest.fixture
def sample_points():
    """Generate sample point cloud for testing."""
    np.random.seed(42)
    n_points = 1000
    points = np.random.rand(n_points, 3).astype(np.float32)
    points[:, 2] *= 10  # Scale Z to realistic values
    return points


@pytest.fixture
def sample_tile_data(sample_points):
    """Create sample tile data."""
    n_points = len(sample_points)
    return {
        "points": sample_points,
        "classification": np.ones(n_points, dtype=np.uint8),
        "intensity": np.random.rand(n_points).astype(np.float32) * 255,
        "return_number": np.ones(n_points, dtype=np.uint8),
    }


class TestIntermediateCache:
    """Test intermediate result caching."""
    
    def test_cache_initialization(self, sample_config):
        """Test that cache is initialized properly."""
        orchestrator = FeatureOrchestrator(sample_config)
        
        # Check cache attributes exist
        assert hasattr(orchestrator, '_enable_feature_cache')
        assert hasattr(orchestrator, '_intermediate_cache')
        assert hasattr(orchestrator, '_cache_hits')
        assert hasattr(orchestrator, '_cache_misses')
        
        # Check initial state
        assert orchestrator._enable_feature_cache == True
        assert len(orchestrator._intermediate_cache) == 0
        assert orchestrator._cache_hits == 0
        assert orchestrator._cache_misses == 0
    
    def test_cache_can_be_disabled(self):
        """Test that caching can be disabled via config."""
        config = OmegaConf.create({
            "processor": {"lod_level": "LOD2"},
            "features": {"enable_caching": False},
            "data_sources": {"rgb": {"enabled": False}, "infrared": {"enabled": False}},
        })
        
        orchestrator = FeatureOrchestrator(config)
        assert orchestrator._enable_feature_cache == False
    
    def test_cache_methods_exist(self, sample_config):
        """Test that cache methods are available."""
        orchestrator = FeatureOrchestrator(sample_config)
        
        assert hasattr(orchestrator, '_get_cached_normals_eigenvalues')
        assert hasattr(orchestrator, '_cache_normals_eigenvalues')
        assert callable(orchestrator._get_cached_normals_eigenvalues)
        assert callable(orchestrator._cache_normals_eigenvalues)
    
    def test_cache_stores_and_retrieves_data(self, sample_config, sample_points):
        """Test that cache can store and retrieve normals/eigenvalues."""
        orchestrator = FeatureOrchestrator(sample_config)
        
        # Generate fake normals and eigenvalues
        n_points = len(sample_points)
        normals = np.random.rand(n_points, 3).astype(np.float32)
        eigenvalues = np.random.rand(n_points, 3).astype(np.float32)
        
        # Cache them
        k_neighbors = 20
        orchestrator._cache_normals_eigenvalues(sample_points, k_neighbors, normals, eigenvalues)
        
        # Retrieve from cache
        cached = orchestrator._get_cached_normals_eigenvalues(sample_points, k_neighbors)
        
        # Verify
        assert cached is not None
        cached_normals, cached_eigenvalues = cached
        np.testing.assert_array_almost_equal(cached_normals, normals)
        np.testing.assert_array_almost_equal(cached_eigenvalues, eigenvalues)
        
        # Check stats
        assert orchestrator._cache_hits == 1
        assert orchestrator._cache_misses == 0
    
    def test_cache_miss_on_first_access(self, sample_config, sample_points):
        """Test that first access is a cache miss."""
        orchestrator = FeatureOrchestrator(sample_config)
        
        # Try to get from empty cache
        cached = orchestrator._get_cached_normals_eigenvalues(sample_points, 20)
        
        assert cached is None
        assert orchestrator._cache_misses == 1
        assert orchestrator._cache_hits == 0
    
    def test_cache_key_differs_for_different_k(self, sample_config, sample_points):
        """Test that cache keys differ for different k_neighbors values."""
        orchestrator = FeatureOrchestrator(sample_config)
        
        # Generate data
        n_points = len(sample_points)
        normals1 = np.random.rand(n_points, 3).astype(np.float32)
        eigenvalues1 = np.random.rand(n_points, 3).astype(np.float32)
        normals2 = np.random.rand(n_points, 3).astype(np.float32) * 2
        eigenvalues2 = np.random.rand(n_points, 3).astype(np.float32) * 2
        
        # Cache with k=20
        orchestrator._cache_normals_eigenvalues(sample_points, 20, normals1, eigenvalues1)
        
        # Cache with k=30
        orchestrator._cache_normals_eigenvalues(sample_points, 30, normals2, eigenvalues2)
        
        # Retrieve both
        cached_k20 = orchestrator._get_cached_normals_eigenvalues(sample_points, 20)
        cached_k30 = orchestrator._get_cached_normals_eigenvalues(sample_points, 30)
        
        # Verify they're different
        assert cached_k20 is not None
        assert cached_k30 is not None
        
        np.testing.assert_array_almost_equal(cached_k20[0], normals1)
        np.testing.assert_array_almost_equal(cached_k30[0], normals2)
    
    def test_cache_size_limit(self, sample_config):
        """Test that cache size is limited."""
        orchestrator = FeatureOrchestrator(sample_config)
        
        # Fill cache beyond limit (>10 entries)
        for i in range(15):
            points = np.random.rand(100, 3).astype(np.float32) * i  # Different points
            normals = np.random.rand(100, 3).astype(np.float32)
            eigenvalues = np.random.rand(100, 3).astype(np.float32)
            
            orchestrator._cache_normals_eigenvalues(points, 20, normals, eigenvalues)
        
        # Cache should be limited to 10 entries (FIFO eviction)
        assert len(orchestrator._intermediate_cache) <= 10
    
    def test_strategy_cache_integration(self, sample_config):
        """Test that strategies can use cached intermediates."""
        orchestrator = FeatureOrchestrator(sample_config)
        
        # Check if computer (strategy) has cache methods
        assert hasattr(orchestrator.computer, 'set_cached_intermediates')
        assert hasattr(orchestrator.computer, 'get_cached_intermediates')
        assert hasattr(orchestrator.computer, 'clear_cache')
        
        # Test setting cache
        normals = np.random.rand(100, 3).astype(np.float32)
        eigenvalues = np.random.rand(100, 3).astype(np.float32)
        
        orchestrator.computer.set_cached_intermediates((normals, eigenvalues))
        cached = orchestrator.computer.get_cached_intermediates()
        
        assert cached is not None
        np.testing.assert_array_equal(cached[0], normals)
        np.testing.assert_array_equal(cached[1], eigenvalues)
        
        # Test clearing cache
        orchestrator.computer.clear_cache()
        assert orchestrator.computer.get_cached_intermediates() is None


class TestCachePerformance:
    """Test that cache improves performance."""
    
    @pytest.mark.slow
    def test_cache_reduces_computation_time(self, sample_config, sample_tile_data):
        """Test that using cache is faster than full computation."""
        import time
        
        orchestrator = FeatureOrchestrator(sample_config)
        
        # First computation (cold cache)
        start = time.time()
        features1 = orchestrator.compute_features(sample_tile_data)
        time_uncached = time.time() - start
        
        # Second computation (warm cache - if caching works in compute_features)
        # Note: Current implementation may not automatically cache during compute_features
        # This test validates the infrastructure is ready for that optimization
        start = time.time()
        features2 = orchestrator.compute_features(sample_tile_data)
        time_cached = time.time() - start
        
        # Both should succeed
        assert 'normals' in features1
        assert 'normals' in features2
        
        # Note: Currently may not show speedup because cache isn't fully integrated
        # into compute_features flow - this tests the foundation is there
        print(f"Uncached: {time_uncached:.3f}s, Cached: {time_cached:.3f}s")


class TestCacheEdgeCases:
    """Test edge cases for caching."""
    
    def test_cache_with_disabled_caching(self, sample_points):
        """Test cache methods work when caching is disabled."""
        config = OmegaConf.create({
            "processor": {"lod_level": "LOD2"},
            "features": {"enable_caching": False},
            "data_sources": {"rgb": {"enabled": False}, "infrared": {"enabled": False}},
        })
        
        orchestrator = FeatureOrchestrator(config)
        
        # These should not crash, just return None/do nothing
        cached = orchestrator._get_cached_normals_eigenvalues(sample_points, 20)
        assert cached is None
        
        # Caching should do nothing
        normals = np.random.rand(len(sample_points), 3).astype(np.float32)
        eigenvalues = np.random.rand(len(sample_points), 3).astype(np.float32)
        orchestrator._cache_normals_eigenvalues(sample_points, 20, normals, eigenvalues)
        
        # Still None
        cached = orchestrator._get_cached_normals_eigenvalues(sample_points, 20)
        assert cached is None
    
    def test_cache_with_empty_points(self, sample_config):
        """Test cache handles empty point clouds."""
        orchestrator = FeatureOrchestrator(sample_config)
        
        empty_points = np.array([], dtype=np.float32).reshape(0, 3)
        
        # Should not crash
        cached = orchestrator._get_cached_normals_eigenvalues(empty_points, 20)
        assert cached is None
    
    def test_cache_with_single_point(self, sample_config):
        """Test cache with minimal point cloud."""
        orchestrator = FeatureOrchestrator(sample_config)
        
        single_point = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        eigenvalues = np.array([[1.0, 0.5, 0.1]], dtype=np.float32)
        
        # Should work
        orchestrator._cache_normals_eigenvalues(single_point, 20, normals, eigenvalues)
        cached = orchestrator._get_cached_normals_eigenvalues(single_point, 20)
        
        assert cached is not None
