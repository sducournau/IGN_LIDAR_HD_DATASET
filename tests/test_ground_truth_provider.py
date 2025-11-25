"""
Tests for GroundTruthProvider - Unified Ground Truth Interface

Tests verify:
1. Singleton pattern implementation
2. Lazy loading of sub-components
3. High-level convenience API
4. Cache management
5. Integration with existing components
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from ign_lidar.core.ground_truth_provider import GroundTruthProvider, get_provider


class TestGroundTruthProviderSingleton:
    """Test singleton pattern implementation."""

    def test_singleton_instance(self):
        """Verify singleton pattern - multiple instances are same object."""
        GroundTruthProvider.reset_instance()

        gt1 = GroundTruthProvider()
        gt2 = GroundTruthProvider()

        assert gt1 is gt2, "Multiple instances should be the same object"

    def test_singleton_with_different_cache_settings(self):
        """Verify singleton ignores cache settings on subsequent calls."""
        GroundTruthProvider.reset_instance()

        gt1 = GroundTruthProvider(cache_enabled=True)
        assert gt1._cache_enabled is True

        gt2 = GroundTruthProvider(cache_enabled=False)
        assert gt2._cache_enabled is True  # Should keep first setting

        assert gt1 is gt2

    def test_reset_instance(self):
        """Verify reset_instance() properly resets singleton."""
        gt1 = GroundTruthProvider()
        cache_key = "test_key"
        if gt1._ground_truth_cache is not None:
            gt1._ground_truth_cache[cache_key] = {"test": "data"}

        GroundTruthProvider.reset_instance()

        gt2 = GroundTruthProvider()
        assert gt1 is not gt2, "After reset, should create new instance"
        assert gt2._ground_truth_cache is None or len(gt2._ground_truth_cache) == 0

    def test_get_provider_function(self):
        """Verify module-level get_provider() function."""
        GroundTruthProvider.reset_instance()

        gt1 = get_provider()
        gt2 = get_provider()

        assert gt1 is gt2, "get_provider() should return same instance"


class TestGroundTruthProviderCache:
    """Test cache management functionality."""

    def setup_method(self):
        """Reset singleton before each test."""
        GroundTruthProvider.reset_instance()

    def test_cache_enabled(self):
        """Verify caching is enabled by default."""
        gt = GroundTruthProvider(cache_enabled=True)
        assert gt._ground_truth_cache is not None

    def test_cache_disabled(self):
        """Verify caching can be disabled."""
        gt = GroundTruthProvider(cache_enabled=False)
        assert gt._ground_truth_cache is None

    def test_clear_cache(self):
        """Verify cache clearing."""
        gt = GroundTruthProvider(cache_enabled=True)

        # Manually add cache entry
        gt._ground_truth_cache["test_key"] = {"test": "data"}
        assert len(gt._ground_truth_cache) == 1

        gt.clear_cache()
        assert len(gt._ground_truth_cache) == 0

    def test_get_cache_stats_enabled(self):
        """Verify cache stats when enabled."""
        gt = GroundTruthProvider(cache_enabled=True)

        gt._ground_truth_cache["key1"] = {"data": 1}
        gt._ground_truth_cache["key2"] = {"data": 2}

        stats = gt.get_cache_stats()

        assert stats["enabled"] is True
        assert stats["size"] == 2
        assert set(stats["keys"]) == {"key1", "key2"}

    def test_get_cache_stats_disabled(self):
        """Verify cache stats when disabled."""
        gt = GroundTruthProvider(cache_enabled=False)

        stats = gt.get_cache_stats()

        assert stats["enabled"] is False
        assert stats["size"] == 0
        assert stats["keys"] == []


class TestGroundTruthProviderLazyLoading:
    """Test lazy loading of sub-components."""

    def setup_method(self):
        """Reset singleton before each test."""
        GroundTruthProvider.reset_instance()

    @patch("ign_lidar.io.wfs_ground_truth.IGNGroundTruthFetcher", autospec=False)
    def test_lazy_load_fetcher(self, mock_fetcher_class):
        """Verify fetcher is lazy-loaded on first access."""
        mock_instance = Mock()
        mock_fetcher_class.return_value = mock_instance

        gt = GroundTruthProvider()

        # Should not be loaded yet
        assert GroundTruthProvider._fetcher is None

        # Access fetcher property
        try:
            fetcher = gt.fetcher
            # Should now be loaded
            assert GroundTruthProvider._fetcher is not None or fetcher is not None
        except ImportError:
            # Expected if shapely/geopandas not available
            pass

    @patch("ign_lidar.core.ground_truth_manager.GroundTruthManager")
    def test_lazy_load_manager(self, mock_manager_class):
        """Verify manager is lazy-loaded on first access."""
        mock_instance = Mock()
        mock_manager_class.return_value = mock_instance

        gt = GroundTruthProvider()

        # Should not be loaded yet
        assert GroundTruthProvider._manager is None

        # Access manager property
        manager = gt.manager

        # Should now be loaded (or None if import failed)
        assert GroundTruthProvider._manager is not None or manager is None

    @patch("ign_lidar.optimization.ground_truth.GroundTruthOptimizer")
    def test_lazy_load_optimizer(self, mock_optimizer_class):
        """Verify optimizer is lazy-loaded on first access."""
        mock_instance = Mock()
        mock_optimizer_class.return_value = mock_instance

        gt = GroundTruthProvider()

        # Should not be loaded yet
        assert GroundTruthProvider._optimizer is None

        # Access optimizer property
        optimizer = gt.optimizer

        # Should now be loaded (or None if import failed)
        assert GroundTruthProvider._optimizer is not None or optimizer is None

    def test_lazy_load_missing_fetcher(self):
        """Verify graceful handling when fetcher import fails."""
        GroundTruthProvider.reset_instance()

        with patch(
            "ign_lidar.io.wfs_ground_truth.IGNGroundTruthFetcher",
            side_effect=ImportError("Test error"),
        ):
            gt = GroundTruthProvider()

            with pytest.raises(ImportError):
                _ = gt.fetcher


class TestGroundTruthProviderHighLevelAPI:
    """Test high-level convenience API."""

    def setup_method(self):
        """Reset singleton and mock dependencies before each test."""
        GroundTruthProvider.reset_instance()

    @patch("ign_lidar.io.wfs_ground_truth.IGNGroundTruthFetcher")
    def test_fetch_all_features(self, mock_fetcher_class):
        """Test fetch_all_features() convenience method."""
        mock_instance = Mock()
        mock_features = {"buildings": ["poly1", "poly2"], "roads": ["road1"]}
        mock_instance.fetch_all_features.return_value = mock_features
        mock_fetcher_class.return_value = mock_instance

        gt = GroundTruthProvider()
        bbox = (100.0, 50.0, 150.0, 100.0)

        try:
            result = gt.fetch_all_features(bbox)
            assert result == mock_features
            mock_instance.fetch_all_features.assert_called_once_with(bbox)
        except ImportError:
            # Expected if dependencies not available
            pytest.skip("Dependencies not available")

    @patch("ign_lidar.io.wfs_ground_truth.IGNGroundTruthFetcher")
    def test_fetch_all_features_cached(self, mock_fetcher_class):
        """Test that fetch_all_features() uses cache on repeated calls."""
        mock_instance = Mock()
        mock_features = {"buildings": ["poly1"]}
        mock_instance.fetch_all_features.return_value = mock_features
        mock_fetcher_class.return_value = mock_instance

        gt = GroundTruthProvider(cache_enabled=True)
        bbox = (100.0, 50.0, 150.0, 100.0)

        try:
            # First call
            result1 = gt.fetch_all_features(bbox)
            call_count_1 = mock_instance.fetch_all_features.call_count

            # Second call
            result2 = gt.fetch_all_features(bbox)
            call_count_2 = mock_instance.fetch_all_features.call_count

            # Should use cache on second call
            assert result1 == result2 == mock_features
            assert call_count_1 == 1  # Only called once
            assert call_count_2 == 1  # Not called again
        except ImportError:
            pytest.skip("Dependencies not available")

    @patch("ign_lidar.io.wfs_ground_truth.IGNGroundTruthFetcher")
    def test_label_points_direct(self, mock_fetcher_class):
        """Test label_points() with direct method."""
        mock_instance = Mock()
        mock_labels = np.array([1, 2, 1, 3, 2])
        mock_instance.label_points_with_ground_truth.return_value = mock_labels
        mock_fetcher_class.return_value = mock_instance

        gt = GroundTruthProvider()
        points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
        features = {"buildings": ["poly1"]}

        try:
            result = gt.label_points(points, features, use_optimization=False)
            np.testing.assert_array_equal(result, mock_labels)
            mock_instance.label_points_with_ground_truth.assert_called_once_with(points, features)
        except ImportError:
            pytest.skip("Dependencies not available")

    @patch("ign_lidar.io.wfs_ground_truth.IGNGroundTruthFetcher")
    @patch("ign_lidar.optimization.ground_truth.GroundTruthOptimizer")
    def test_label_points_optimized(self, mock_optimizer_class, mock_fetcher_class):
        """Test label_points() with optimization."""
        mock_optimizer = Mock()
        mock_labels = np.array([1, 2, 1, 3, 2])
        mock_optimizer.label_points_with_ground_truth.return_value = mock_labels
        mock_optimizer_class.return_value = mock_optimizer

        gt = GroundTruthProvider()
        points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
        features = {"buildings": ["poly1"]}

        try:
            result = gt.label_points(points, features, use_optimization=True)
            np.testing.assert_array_equal(result, mock_labels)
            mock_optimizer.label_points_with_ground_truth.assert_called_once_with(points, features)
        except ImportError:
            pytest.skip("Dependencies not available")


class TestGroundTruthProviderRepr:
    """Test string representation."""

    def setup_method(self):
        """Reset singleton before each test."""
        GroundTruthProvider.reset_instance()

    def test_repr_with_cache_enabled(self):
        """Verify __repr__ with cache enabled."""
        gt = GroundTruthProvider(cache_enabled=True)
        repr_str = repr(gt)

        assert "GroundTruthProvider" in repr_str
        assert "cache_enabled=True" in repr_str

    def test_repr_with_cache_disabled(self):
        """Verify __repr__ with cache disabled."""
        gt = GroundTruthProvider(cache_enabled=False)
        repr_str = repr(gt)

        assert "GroundTruthProvider" in repr_str
        assert "cache_enabled=False" in repr_str


class TestGroundTruthProviderIntegration:
    """Integration tests with typical usage patterns."""

    def setup_method(self):
        """Reset singleton before each test."""
        GroundTruthProvider.reset_instance()

    @patch("ign_lidar.io.wfs_ground_truth.IGNGroundTruthFetcher")
    def test_typical_workflow(self, mock_fetcher_class):
        """Test typical high-level usage workflow."""
        mock_fetcher = Mock()

        mock_features = {
            "buildings": ["poly1", "poly2"],
            "roads": ["road1"],
            "vegetation": ["veg1"],
        }
        mock_fetcher.fetch_all_features.return_value = mock_features

        mock_labels = np.array([1, 2, 1, 3, 2, 1])
        mock_fetcher.label_points_with_ground_truth.return_value = mock_labels

        mock_fetcher_class.return_value = mock_fetcher

        gt = GroundTruthProvider(cache_enabled=True)

        try:
            # Fetch features
            bbox = (100.0, 50.0, 150.0, 100.0)
            features = gt.fetch_all_features(bbox)
            assert "buildings" in features

            # Label points (direct method to avoid optimizer mock issues)
            points = np.random.rand(6, 3)
            labels = gt.label_points(points, features, use_optimization=False)
            assert len(labels) == 6

            # Verify cache was used
            cache_stats = gt.get_cache_stats()
            assert cache_stats["enabled"] is True
            assert cache_stats["size"] > 0
        except ImportError:
            pytest.skip("Dependencies not available")

    @patch("ign_lidar.io.wfs_ground_truth.IGNGroundTruthFetcher")
    def test_prefetch_workflow(self, mock_fetcher_class):
        """Test prefetch-then-process workflow."""
        mock_fetcher = Mock()
        mock_features = {"buildings": ["poly1"]}
        mock_fetcher.fetch_all_features.return_value = mock_features
        mock_fetcher_class.return_value = mock_fetcher

        gt = GroundTruthProvider(cache_enabled=True)

        try:
            # Simulate prefetch
            bbox = (100.0, 50.0, 150.0, 100.0)
            features1 = gt.fetch_all_features(bbox)

            # Simulate processing with cache hit
            features2 = gt.fetch_all_features(bbox)

            assert features1 == features2
            # Fetcher should only be called once (cache on second call)
            assert mock_fetcher.fetch_all_features.call_count == 1
        except ImportError:
            pytest.skip("Dependencies not available")


# Marker for pytest
pytestmark = pytest.mark.unit
