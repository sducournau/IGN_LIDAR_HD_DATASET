"""
Tests for GroundTruthOptimizer V2 Features (Task #12)

Tests cache functionality, batch processing, and backward compatibility.
"""

import hashlib
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ign_lidar.optimization.ground_truth import GroundTruthOptimizer

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_points():
    """Create sample point cloud."""
    return np.random.rand(1000, 3) * 100


@pytest.fixture
def sample_ground_truth():
    """Create sample ground truth features (mock GeoDataFrames)."""
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import Polygon

    # Create simple polygon
    polygon = Polygon([(0, 0), (50, 0), (50, 50), (0, 50)])
    gdf = geopandas.GeoDataFrame(
        {"geometry": [polygon], "feature_type": ["building"]}, crs="EPSG:2154"
    )

    return {"buildings": gdf}


# ============================================================================
# V2 Cache Tests
# ============================================================================


class TestCacheV2:
    """Test V2 caching features."""

    def test_cache_enabled_by_default(self):
        """Cache should be enabled by default."""
        opt = GroundTruthOptimizer()
        assert opt.enable_cache is True
        assert opt._cache is not None
        assert isinstance(opt._cache, dict)

    def test_cache_disabled(self):
        """Can disable cache."""
        opt = GroundTruthOptimizer(enable_cache=False)
        assert opt.enable_cache is False

    def test_cache_key_generation(self, sample_points, sample_ground_truth):
        """Cache key should be deterministic and spatial."""
        opt = GroundTruthOptimizer(enable_cache=True)

        # Generate key twice for same input
        key1 = opt._generate_cache_key(sample_points, sample_ground_truth, True)
        key2 = opt._generate_cache_key(sample_points, sample_ground_truth, True)

        assert key1 == key2
        assert len(key1) == 32  # MD5 hex digest
        assert isinstance(key1, str)

    def test_cache_key_different_for_different_inputs(
        self, sample_points, sample_ground_truth
    ):
        """Different inputs should produce different cache keys."""
        opt = GroundTruthOptimizer(enable_cache=True)

        key1 = opt._generate_cache_key(sample_points, sample_ground_truth, True)

        # Different points
        different_points = sample_points + 100
        key2 = opt._generate_cache_key(different_points, sample_ground_truth, True)

        # Different NDVI setting
        key3 = opt._generate_cache_key(sample_points, sample_ground_truth, False)

        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    def test_cache_add_and_retrieve(self):
        """Can add to cache and retrieve."""
        opt = GroundTruthOptimizer(enable_cache=True)

        labels = np.array([0, 1, 2, 3, 4])
        cache_key = "test_key_123"
        size_mb = labels.nbytes / (1024**2)

        # Add to cache
        opt._add_to_cache(cache_key, labels, size_mb)

        # Retrieve from cache
        retrieved = opt._get_from_cache(cache_key)

        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, labels)
        assert opt._cache_hits == 1
        assert opt._cache_misses == 0

    def test_cache_miss(self):
        """Cache miss increments counter."""
        opt = GroundTruthOptimizer(enable_cache=True)

        retrieved = opt._get_from_cache("nonexistent_key")

        assert retrieved is None
        assert opt._cache_hits == 0
        assert opt._cache_misses == 1

    def test_cache_lru_eviction(self):
        """LRU eviction when cache exceeds max entries."""
        opt = GroundTruthOptimizer(
            enable_cache=True, max_cache_entries=3, max_cache_size_mb=1000
        )

        # Add 4 entries (should evict first one)
        for i in range(4):
            labels = np.array([i] * 100)
            cache_key = f"key_{i}"
            size_mb = labels.nbytes / (1024**2)
            opt._add_to_cache(cache_key, labels, size_mb)

        # First entry should be evicted
        assert "key_0" not in opt._cache
        assert "key_1" in opt._cache
        assert "key_2" in opt._cache
        assert "key_3" in opt._cache
        assert len(opt._cache) == 3

    def test_cache_size_eviction(self):
        """Eviction when cache exceeds max size."""
        opt = GroundTruthOptimizer(
            enable_cache=True, max_cache_entries=100, max_cache_size_mb=0.001  # 1KB
        )

        # Add entries until size limit exceeded
        large_labels = np.ones(1000, dtype=np.int32)  # ~4KB
        size_mb = large_labels.nbytes / (1024**2)

        opt._add_to_cache("key_1", large_labels, size_mb)
        opt._add_to_cache("key_2", large_labels, size_mb)  # Should evict key_1

        assert "key_1" not in opt._cache
        assert "key_2" in opt._cache

    def test_cache_clear(self):
        """Can clear cache."""
        opt = GroundTruthOptimizer(enable_cache=True)

        # Add entries
        for i in range(3):
            labels = np.array([i] * 10)
            opt._add_to_cache(f"key_{i}", labels, 0.001)

        assert len(opt._cache) == 3

        # Clear cache
        opt.clear_cache()

        assert len(opt._cache) == 0
        assert opt._current_cache_size_mb == 0.0
        assert opt._cache_hits == 0
        assert opt._cache_misses == 0

    def test_cache_stats(self):
        """Cache stats return correct information."""
        opt = GroundTruthOptimizer(
            enable_cache=True, max_cache_entries=10, max_cache_size_mb=100
        )

        # Add some entries and make hits/misses
        opt._add_to_cache("key_1", np.array([1, 2, 3]), 0.01)
        opt._get_from_cache("key_1")  # Hit
        opt._get_from_cache("key_2")  # Miss

        stats = opt.get_cache_stats()

        assert stats["enabled"] is True
        assert stats["entries"] == 1
        assert stats["max_entries"] == 10
        assert stats["max_size_mb"] == 100
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_ratio"] == 0.5
        assert stats["disk_cache_enabled"] is False

    def test_disk_cache(self):
        """Disk cache saves and loads correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            opt = GroundTruthOptimizer(enable_cache=True, cache_dir=cache_dir)

            labels = np.array([10, 20, 30, 40, 50])
            cache_key = "disk_test_key"
            size_mb = labels.nbytes / (1024**2)

            # Add to cache (should save to disk)
            opt._add_to_cache(cache_key, labels, size_mb)

            # Verify disk file exists
            cache_file = cache_dir / f"{cache_key}.pkl"
            assert cache_file.exists()

            # Clear memory cache but keep disk
            opt._cache.clear()
            opt._current_cache_size_mb = 0.0

            # Retrieve from disk
            retrieved = opt._get_from_cache(cache_key)
            assert retrieved is not None
            np.testing.assert_array_equal(retrieved, labels)


# ============================================================================
# V2 Batch Processing Tests
# ============================================================================


class TestBatchProcessing:
    """Test batch processing features."""

    def test_label_points_batch_empty(self):
        """Empty batch returns empty list."""
        opt = GroundTruthOptimizer()
        result = opt.label_points_batch([], {})
        assert result == []

    @patch.object(GroundTruthOptimizer, "label_points")
    def test_label_points_batch_calls_individual(
        self, mock_label_points, sample_ground_truth
    ):
        """Batch processing calls label_points for each tile."""
        opt = GroundTruthOptimizer(enable_cache=False)

        # Mock return values
        mock_label_points.side_effect = [
            np.array([0, 1, 2]),
            np.array([3, 4, 5]),
        ]

        tiles = [
            {"points": np.random.rand(100, 3)},
            {"points": np.random.rand(200, 3)},
        ]

        results = opt.label_points_batch(tiles, sample_ground_truth)

        assert len(results) == 2
        assert mock_label_points.call_count == 2
        np.testing.assert_array_equal(results[0], [0, 1, 2])
        np.testing.assert_array_equal(results[1], [3, 4, 5])

    @patch.object(GroundTruthOptimizer, "label_points")
    def test_label_points_batch_with_ndvi(self, mock_label_points, sample_ground_truth):
        """Batch processing handles NDVI correctly."""
        opt = GroundTruthOptimizer(enable_cache=False)
        mock_label_points.return_value = np.array([0, 1, 2])

        tiles = [
            {"points": np.random.rand(100, 3), "ndvi": np.random.rand(100)},
            {"points": np.random.rand(200, 3)},  # No NDVI
        ]

        results = opt.label_points_batch(tiles, sample_ground_truth)

        assert len(results) == 2
        # Check NDVI was passed for first tile
        first_call = mock_label_points.call_args_list[0]
        assert first_call[1]["ndvi"] is not None
        # Check NDVI was None for second tile
        second_call = mock_label_points.call_args_list[1]
        assert second_call[1]["ndvi"] is None


# ============================================================================
# Backward Compatibility Tests
# ============================================================================


class TestBackwardCompatibility:
    """Test backward compatibility with old API."""

    def test_old_import_shows_deprecation(self):
        """Importing from old location shows deprecation warning."""
        import warnings
        import importlib
        import sys
        
        # Reset warnings and force reimport
        warnings.simplefilter("always", DeprecationWarning)
        if "ign_lidar.io.ground_truth_optimizer" in sys.modules:
            del sys.modules["ign_lidar.io.ground_truth_optimizer"]
            
        with pytest.warns(DeprecationWarning, match="deprecated"):
            from ign_lidar.io.ground_truth_optimizer import GroundTruthOptimizer

    def test_old_import_works(self):
        """Old import still works functionally."""
        import warnings
        
        # Reset warnings to ensure we catch it
        warnings.simplefilter("always", DeprecationWarning)
        
        with pytest.warns(DeprecationWarning):
            # Force reimport to trigger warning
            import importlib
            import sys
            if "ign_lidar.io.ground_truth_optimizer" in sys.modules:
                del sys.modules["ign_lidar.io.ground_truth_optimizer"]
            from ign_lidar.io.ground_truth_optimizer import GroundTruthOptimizer

        opt = GroundTruthOptimizer(enable_cache=False, verbose=False)
        assert opt is not None
        assert hasattr(opt, "label_points")

    def test_old_init_signature_compatible(self):
        """Old initialization parameters still work."""
        import warnings
        import importlib
        import sys
        
        # Reset warnings and reimport
        warnings.simplefilter("always", DeprecationWarning)
        if "ign_lidar.io.ground_truth_optimizer" in sys.modules:
            del sys.modules["ign_lidar.io.ground_truth_optimizer"]
        
        with pytest.warns(DeprecationWarning):
            from ign_lidar.io.ground_truth_optimizer import GroundTruthOptimizer

        # Test old-style initialization
        opt = GroundTruthOptimizer(
            force_method="strtree", gpu_chunk_size=1_000_000, verbose=False
        )

        assert opt.force_method == "strtree"
        assert opt.gpu_chunk_size == 1_000_000
        assert opt.verbose is False


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
class TestV2Integration:
    """Integration tests for V2 features."""

    def test_cache_speedup_on_repeated_tiles(self, sample_points, sample_ground_truth):
        """Cache provides speedup on repeated tiles."""
        opt = GroundTruthOptimizer(enable_cache=True, verbose=False)

        # Mock the actual labeling method to return consistent results
        with patch.object(
            opt, "_label_strtree", return_value=np.zeros(len(sample_points))
        ):
            # First call (cache miss)
            labels1 = opt.label_points(sample_points, sample_ground_truth)

            # Second call with same inputs (cache hit)
            labels2 = opt.label_points(sample_points, sample_ground_truth)

        stats = opt.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_ratio"] == 0.5
        np.testing.assert_array_equal(labels1, labels2)

    def test_batch_processing_with_cache(self, sample_ground_truth):
        """Batch processing leverages cache for repeated tiles."""
        opt = GroundTruthOptimizer(enable_cache=True, verbose=False)

        # Create tiles (some repeated)
        points1 = np.random.rand(100, 3) * 100
        points2 = np.random.rand(100, 3) * 100 + 50  # Different location
        points3 = points1.copy()  # Repeat first tile

        tiles = [
            {"points": points1},
            {"points": points2},
            {"points": points3},  # Should hit cache
        ]

        with patch.object(opt, "_label_strtree", return_value=np.zeros(100)):
            results = opt.label_points_batch(tiles, sample_ground_truth)

        stats = opt.get_cache_stats()
        assert len(results) == 3
        assert stats["hits"] >= 1  # At least one cache hit for repeated tile


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
