"""
Test suite for Ground Truth Optimizer V2 caching system.

Tests:
- Spatial hash-based cache key generation
- LRU cache eviction
- Memory and disk caching
- Batch processing
- Cache statistics

Author: Task 12 - Ground Truth Optimizer V2
Date: 2025-11-21
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

from ign_lidar.io.ground_truth_optimizer import GroundTruthOptimizer


@pytest.fixture
def sample_points():
    """Generate sample point cloud."""
    np.random.seed(42)
    return np.random.rand(1000, 3) * 100  # 1000 points in 100x100x100 space


@pytest.fixture
def sample_ground_truth():
    """Generate sample ground truth features."""
    pytest.importorskip("geopandas")
    pytest.importorskip("shapely")

    import geopandas as gpd
    from shapely.geometry import Polygon

    # Create simple building polygon
    buildings = gpd.GeoDataFrame(
        {"geometry": [Polygon([(0, 0), (50, 0), (50, 50), (0, 50)])]},
        crs="EPSG:2154"
    )

    return {"buildings": buildings}


class TestCacheKeyGeneration:
    """Test cache key generation."""

    def test_identical_tiles_same_key(self, sample_points, sample_ground_truth):
        """Test that identical tiles generate the same cache key."""
        optimizer = GroundTruthOptimizer(enable_cache=True)

        key1 = optimizer._generate_cache_key(
            sample_points, sample_ground_truth, use_ndvi_refinement=True
        )
        key2 = optimizer._generate_cache_key(
            sample_points, sample_ground_truth, use_ndvi_refinement=True
        )

        assert key1 == key2
        assert len(key1) == 32  # MD5 hex digest length

    def test_different_points_different_key(self, sample_ground_truth):
        """Test that different point clouds generate different keys."""
        optimizer = GroundTruthOptimizer(enable_cache=True)

        points1 = np.random.rand(100, 3)
        points2 = np.random.rand(100, 3) + 100  # Different location

        key1 = optimizer._generate_cache_key(
            points1, sample_ground_truth, use_ndvi_refinement=True
        )
        key2 = optimizer._generate_cache_key(
            points2, sample_ground_truth, use_ndvi_refinement=True
        )

        assert key1 != key2

    def test_different_ndvi_setting_different_key(self, sample_points, sample_ground_truth):
        """Test that NDVI setting affects cache key."""
        optimizer = GroundTruthOptimizer(enable_cache=True)

        key1 = optimizer._generate_cache_key(
            sample_points, sample_ground_truth, use_ndvi_refinement=True
        )
        key2 = optimizer._generate_cache_key(
            sample_points, sample_ground_truth, use_ndvi_refinement=False
        )

        assert key1 != key2


class TestMemoryCache:
    """Test memory caching functionality."""

    def test_cache_hit(self, sample_points, sample_ground_truth):
        """Test cache hit on second access."""
        pytest.importorskip("geopandas")

        optimizer = GroundTruthOptimizer(
            enable_cache=True,
            verbose=False,
            force_method="strtree"
        )

        # First call - should be cache miss
        labels1 = optimizer.label_points(sample_points, sample_ground_truth)
        assert optimizer._cache_misses == 1
        assert optimizer._cache_hits == 0

        # Second call - should be cache hit
        labels2 = optimizer.label_points(sample_points, sample_ground_truth)
        assert optimizer._cache_misses == 1
        assert optimizer._cache_hits == 1

        # Labels should be identical
        np.testing.assert_array_equal(labels1, labels2)

    def test_cache_disabled(self, sample_points, sample_ground_truth):
        """Test that caching can be disabled."""
        pytest.importorskip("geopandas")

        optimizer = GroundTruthOptimizer(
            enable_cache=False,
            verbose=False,
            force_method="strtree"
        )

        # Both calls should compute (no caching)
        labels1 = optimizer.label_points(sample_points, sample_ground_truth)
        labels2 = optimizer.label_points(sample_points, sample_ground_truth)

        # No cache tracking when disabled
        assert optimizer._cache_hits == 0
        assert optimizer._cache_misses == 0

    def test_cache_clear(self, sample_points, sample_ground_truth):
        """Test cache clearing."""
        pytest.importorskip("geopandas")

        optimizer = GroundTruthOptimizer(
            enable_cache=True,
            verbose=False,
            force_method="strtree"
        )

        # Add to cache
        optimizer.label_points(sample_points, sample_ground_truth)
        assert len(optimizer._cache) > 0

        # Clear cache
        optimizer.clear_cache()
        assert len(optimizer._cache) == 0
        assert optimizer._current_cache_size_mb == 0.0


class TestLRUEviction:
    """Test LRU cache eviction."""

    def test_max_entries_eviction(self):
        """Test eviction when max entries exceeded."""
        pytest.importorskip("geopandas")
        from shapely.geometry import Polygon
        import geopandas as gpd

        optimizer = GroundTruthOptimizer(
            enable_cache=True,
            max_cache_entries=3,  # Small cache
            verbose=False,
            force_method="strtree"
        )

        ground_truth = {
            "buildings": gpd.GeoDataFrame(
                {"geometry": [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])]},
                crs="EPSG:2154"
            )
        }

        # Add 4 different tiles (should evict oldest)
        for i in range(4):
            points = np.random.rand(100, 3) * 100 + i * 100  # Different locations
            optimizer.label_points(points, ground_truth)

        # Cache should have at most 3 entries
        assert len(optimizer._cache) <= 3

    def test_max_size_eviction(self):
        """Test eviction when max size exceeded."""
        pytest.importorskip("geopandas")
        from shapely.geometry import Polygon
        import geopandas as gpd

        optimizer = GroundTruthOptimizer(
            enable_cache=True,
            max_cache_size_mb=0.1,  # Tiny cache (100KB)
            verbose=False,
            force_method="strtree"
        )

        ground_truth = {
            "buildings": gpd.GeoDataFrame(
                {"geometry": [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])]},
                crs="EPSG:2154"
            )
        }

        # Add multiple tiles
        for i in range(10):
            points = np.random.rand(10000, 3) + i * 100  # Large tiles, different locations
            optimizer.label_points(points, ground_truth)

        # Cache size should not exceed limit
        assert optimizer._current_cache_size_mb <= optimizer.max_cache_size_mb * 1.1  # 10% tolerance


class TestDiskCache:
    """Test disk caching functionality."""

    def test_disk_cache_persistence(self, sample_points, sample_ground_truth):
        """Test that disk cache persists across instances."""
        pytest.importorskip("geopandas")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # First optimizer - create cache
            optimizer1 = GroundTruthOptimizer(
                enable_cache=True,
                cache_dir=cache_dir,
                verbose=False,
                force_method="strtree"
            )

            labels1 = optimizer1.label_points(sample_points, sample_ground_truth)

            # Check cache file exists
            cache_files = list(cache_dir.glob("*.pkl"))
            assert len(cache_files) > 0

            # Second optimizer - should load from disk
            optimizer2 = GroundTruthOptimizer(
                enable_cache=True,
                cache_dir=cache_dir,
                verbose=False,
                force_method="strtree"
            )

            labels2 = optimizer2.label_points(sample_points, sample_ground_truth)

            # Should be cache hit from disk
            assert optimizer2._cache_hits == 1
            np.testing.assert_array_equal(labels1, labels2)

    def test_disk_cache_clear(self):
        """Test disk cache clearing."""
        pytest.importorskip("geopandas")
        from shapely.geometry import Polygon
        import geopandas as gpd

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            optimizer = GroundTruthOptimizer(
                enable_cache=True,
                cache_dir=cache_dir,
                verbose=False,
                force_method="strtree"
            )

            ground_truth = {
                "buildings": gpd.GeoDataFrame(
                    {"geometry": [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])]},
                    crs="EPSG:2154"
                )
            }

            points = np.random.rand(100, 3)
            optimizer.label_points(points, ground_truth)

            # Check cache file exists
            cache_files = list(cache_dir.glob("*.pkl"))
            assert len(cache_files) > 0

            # Clear cache
            optimizer.clear_cache()

            # Cache files should be deleted
            cache_files = list(cache_dir.glob("*.pkl"))
            assert len(cache_files) == 0


class TestCacheStatistics:
    """Test cache statistics."""

    def test_cache_stats(self, sample_points, sample_ground_truth):
        """Test cache statistics reporting."""
        pytest.importorskip("geopandas")

        optimizer = GroundTruthOptimizer(
            enable_cache=True,
            verbose=False,
            force_method="strtree"
        )

        # Initial stats
        stats = optimizer.get_cache_stats()
        assert stats['enabled'] is True
        assert stats['entries'] == 0
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['hit_ratio'] == 0.0

        # After one computation
        optimizer.label_points(sample_points, sample_ground_truth)
        stats = optimizer.get_cache_stats()
        assert stats['entries'] == 1
        assert stats['misses'] == 1

        # After cache hit
        optimizer.label_points(sample_points, sample_ground_truth)
        stats = optimizer.get_cache_stats()
        assert stats['hits'] == 1
        assert stats['hit_ratio'] == 0.5  # 1 hit, 1 miss


class TestBatchProcessing:
    """Test batch processing functionality."""

    def test_batch_processing(self):
        """Test batch processing multiple tiles."""
        pytest.importorskip("geopandas")
        from shapely.geometry import Polygon
        import geopandas as gpd

        optimizer = GroundTruthOptimizer(
            enable_cache=True,
            verbose=False,
            force_method="strtree"
        )

        ground_truth = {
            "buildings": gpd.GeoDataFrame(
                {"geometry": [Polygon([(0, 0), (50, 0), (50, 50), (0, 50)])]},
                crs="EPSG:2154"
            )
        }

        # Create batch of tiles
        tiles = []
        for i in range(5):
            points = np.random.rand(100, 3) * 100 + i * 100
            tiles.append({"points": points})

        # Batch process
        results = optimizer.label_points_batch(tiles, ground_truth)

        assert len(results) == 5
        for labels in results:
            assert len(labels) == 100

    def test_batch_with_cache_reuse(self):
        """Test that batch processing reuses cache for duplicate tiles."""
        pytest.importorskip("geopandas")
        from shapely.geometry import Polygon
        import geopandas as gpd

        optimizer = GroundTruthOptimizer(
            enable_cache=True,
            verbose=False,
            force_method="strtree"
        )

        ground_truth = {
            "buildings": gpd.GeoDataFrame(
                {"geometry": [Polygon([(0, 0), (50, 0), (50, 50), (0, 50)])]},
                crs="EPSG:2154"
            )
        }

        points = np.random.rand(100, 3) * 100

        # Create batch with duplicate tiles
        tiles = [{"points": points}, {"points": points}, {"points": points}]

        # Batch process
        results = optimizer.label_points_batch(tiles, ground_truth)

        # Should have cache hits
        stats = optimizer.get_cache_stats()
        assert stats['hits'] == 2  # First is miss, next 2 are hits
        assert stats['hit_ratio'] == 2/3

        # All results should be identical
        np.testing.assert_array_equal(results[0], results[1])
        np.testing.assert_array_equal(results[1], results[2])


@pytest.mark.benchmark
class TestCachePerformance:
    """Performance benchmarks for caching."""

    def test_cache_speedup(self, sample_points, sample_ground_truth):
        """Benchmark cache speedup."""
        pytest.importorskip("geopandas")
        import time

        optimizer = GroundTruthOptimizer(
            enable_cache=True,
            verbose=False,
            force_method="strtree"
        )

        # First call (cache miss)
        start = time.time()
        labels1 = optimizer.label_points(sample_points, sample_ground_truth)
        time_uncached = time.time() - start

        # Second call (cache hit)
        start = time.time()
        labels2 = optimizer.label_points(sample_points, sample_ground_truth)
        time_cached = time.time() - start

        # Cache should be significantly faster
        speedup = time_uncached / time_cached if time_cached > 0 else float('inf')
        print(f"\nCache speedup: {speedup:.1f}x (uncached: {time_uncached:.4f}s, cached: {time_cached:.4f}s)")

        # Cache should be at least 10x faster
        assert speedup > 10, f"Expected >10x speedup, got {speedup:.1f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
