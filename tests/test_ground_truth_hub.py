"""
Tests for GroundTruthHub v2.0 - Unified ground truth API.

This test suite verifies:
- Singleton pattern behavior
- Lazy loading of sub-components
- Property caching
- Convenience methods
- Backward compatibility
- Cache consolidation
- Error handling

Test Structure:
    - TestGroundTruthHubSingleton: Singleton and lazy loading
    - TestGroundTruthHubProperties: Property access and caching
    - TestGroundTruthHubConvenience: High-level convenience methods
    - TestGroundTruthHubBackwardCompatibility: Legacy API support
    - TestGroundTruthHubIntegration: Full pipeline integration

Author: GitHub Copilot
Date: November 22, 2025
Version: 2.0.0
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, Any

from ign_lidar.core import ground_truth, GroundTruthHub


class TestGroundTruthHubSingleton:
    """Test singleton pattern and instance management."""
    
    def test_singleton_instance(self):
        """GroundTruthHub should always return the same instance."""
        hub1 = GroundTruthHub()
        hub2 = GroundTruthHub()
        
        assert hub1 is hub2, "GroundTruthHub should be a singleton"
    
    def test_module_level_instance(self):
        """Module-level 'ground_truth' should be the singleton instance."""
        hub = GroundTruthHub()
        
        assert ground_truth is hub, "Module instance should be singleton"
    
    def test_initial_state_no_components_loaded(self):
        """Initially, no components should be loaded."""
        hub = GroundTruthHub()
        
        assert hub._fetcher is None, "Fetcher should not be loaded initially"
        assert hub._optimizer is None, "Optimizer should not be loaded initially"
        assert hub._manager is None, "Manager should not be loaded initially"
        assert hub._refiner is None, "Refiner should not be loaded initially"
    
    def test_repr_no_components(self):
        """Repr should show no components when none are loaded."""
        hub = GroundTruthHub()
        # Reset all components to None
        hub._fetcher = None
        hub._optimizer = None
        hub._manager = None
        hub._refiner = None
        
        repr_str = repr(hub)
        assert "no components loaded" in repr_str.lower()


class TestGroundTruthHubLazyLoading:
    """Test lazy loading behavior of sub-components."""
    
    def test_fetcher_lazy_loading(self):
        """Fetcher should only be created on first access."""
        hub = GroundTruthHub()
        hub._fetcher = None  # Reset
        
        assert hub._fetcher is None, "Fetcher should not exist before access"
        
        # Access property
        fetcher = hub.fetcher
        
        assert hub._fetcher is not None, "Fetcher should be created after access"
        assert fetcher is hub._fetcher, "Should return cached instance"
    
    def test_optimizer_lazy_loading(self):
        """Optimizer should only be created on first access."""
        hub = GroundTruthHub()
        hub._optimizer = None  # Reset
        
        assert hub._optimizer is None, "Optimizer should not exist before access"
        
        # Access property
        optimizer = hub.optimizer
        
        assert hub._optimizer is not None, "Optimizer should be created after access"
        assert optimizer is hub._optimizer, "Should return cached instance"
    
    def test_manager_lazy_loading(self):
        """Manager should only be created on first access."""
        hub = GroundTruthHub()
        hub._manager = None  # Reset
        
        assert hub._manager is None, "Manager should not exist before access"
        
        # Access property
        manager = hub.manager
        
        assert hub._manager is not None, "Manager should be created after access"
        assert manager is hub._manager, "Should return cached instance"
    
    def test_refiner_lazy_loading(self):
        """Refiner should only be created on first access."""
        hub = GroundTruthHub()
        hub._refiner = None  # Reset
        
        assert hub._refiner is None, "Refiner should not exist before access"
        
        # Access property
        refiner = hub.refiner
        
        assert hub._refiner is not None, "Refiner should be created after access"
        assert refiner is hub._refiner, "Should return cached instance"
    
    def test_property_caching(self):
        """Properties should return cached instances, not recreate."""
        hub = GroundTruthHub()
        hub._fetcher = None
        
        # First access
        fetcher1 = hub.fetcher
        # Second access
        fetcher2 = hub.fetcher
        
        assert fetcher1 is fetcher2, "Should return same cached instance"


class TestGroundTruthHubConvenienceMethods:
    """Test high-level convenience methods."""
    
    @pytest.fixture
    def mock_hub(self):
        """Create hub with mocked sub-components."""
        hub = GroundTruthHub()
        
        # Mock manager
        hub._manager = Mock()
        hub._manager.prefetch_ground_truth_for_tile.return_value = {
            'buildings': Mock(),
            'roads': Mock()
        }
        
        # Mock optimizer
        hub._optimizer = Mock()
        hub._optimizer.label_points.return_value = np.array([1, 2, 0, 1, 2])
        hub._optimizer.get_cache_stats.return_value = {
            'hits': 10,
            'misses': 5,
            'hit_rate': 0.67
        }
        
        # Mock refiner
        hub._refiner = Mock()
        hub._refiner.refine_all.return_value = (
            np.array([1, 2, 0, 1, 2]),  # labels
            {'n_refined': 3}  # metadata
        )
        
        return hub
    
    def test_fetch_and_label_success(self, mock_hub):
        """fetch_and_label should coordinate manager and optimizer."""
        points = np.random.rand(100, 3)
        
        labels, metadata = mock_hub.fetch_and_label(
            tile_path="test.laz",
            points=points
        )
        
        # Verify manager was called
        mock_hub._manager.prefetch_ground_truth_for_tile.assert_called_once()
        
        # Verify optimizer was called
        mock_hub._optimizer.label_points.assert_called_once()
        
        # Verify metadata
        assert 'n_labeled' in metadata
        assert 'n_total' in metadata
        assert metadata['n_total'] == 100
    
    def test_fetch_and_label_no_data(self, mock_hub):
        """fetch_and_label should handle no ground truth data."""
        points = np.random.rand(100, 3)
        
        # Mock returns no data
        mock_hub._manager.prefetch_ground_truth_for_tile.return_value = {}
        
        labels, metadata = mock_hub.fetch_and_label(
            tile_path="test.laz",
            points=points
        )
        
        assert len(labels) == 100
        assert np.all(labels == 0), "Should return zero labels"
        assert metadata['n_labeled'] == 0
    
    def test_prefetch_batch(self, mock_hub):
        """prefetch_batch should delegate to manager."""
        tile_paths = ["tile1.laz", "tile2.laz", "tile3.laz"]
        
        mock_hub._manager.prefetch_ground_truth_batch.return_value = {
            'n_tiles': 3,
            'n_success': 3,
            'n_failed': 0
        }
        
        result = mock_hub.prefetch_batch(tile_paths)
        
        mock_hub._manager.prefetch_ground_truth_batch.assert_called_once()
        assert result['n_tiles'] == 3
        assert result['n_success'] == 3
    
    def test_process_tile_complete_with_refinement(self, mock_hub):
        """process_tile_complete should run full pipeline with refinement."""
        points = np.random.rand(100, 3)
        features = {
            'normals': np.random.rand(100, 3),
            'curvature': np.random.rand(100)
        }
        
        labels, stats = mock_hub.process_tile_complete(
            tile_path="test.laz",
            points=points,
            features=features,
            refine=True
        )
        
        # Verify refiner was called
        mock_hub._refiner.refine_all.assert_called_once()
        
        # Verify statistics
        assert 'duration' in stats
        assert stats['refined'] is True
        assert 'refine_metadata' in stats
    
    def test_process_tile_complete_without_refinement(self, mock_hub):
        """process_tile_complete should skip refinement if disabled."""
        points = np.random.rand(100, 3)
        
        labels, stats = mock_hub.process_tile_complete(
            tile_path="test.laz",
            points=points,
            features=None,
            refine=False
        )
        
        # Verify refiner was NOT called
        mock_hub._refiner.refine_all.assert_not_called()
        
        # Verify statistics
        assert stats['refined'] is False
        assert stats['refine_reason'] == 'disabled'
    
    def test_clear_all_caches(self, mock_hub):
        """clear_all_caches should clear all component caches."""
        # Mock fetcher
        mock_hub._fetcher = Mock()
        mock_hub._fetcher._cache = {'key1': 'value1', 'key2': 'value2'}
        
        # Optimizer already mocked with cache
        mock_hub._optimizer._cache = {'key3': 'value3'}
        
        # Manager cache
        mock_hub._manager._ground_truth_cache = {'key4': 'value4'}
        
        cleared = mock_hub.clear_all_caches()
        
        # Verify all caches cleared
        assert cleared['fetcher'] == 2
        assert cleared['optimizer'] == 1
        assert cleared['manager'] == 1
        assert sum(cleared.values()) == 4
    
    def test_get_statistics(self, mock_hub):
        """get_statistics should collect stats from all components."""
        # Add fetcher mock
        mock_hub._fetcher = Mock()
        mock_hub._fetcher._cache = {'key1': 'value1'}
        
        # Fix manager mock to support len()
        mock_hub._manager._ground_truth_cache = {'key4': 'value4'}
        
        stats = mock_hub.get_statistics()
        
        # Verify components listed
        assert 'fetcher' in stats['components_loaded']
        assert 'optimizer' in stats['components_loaded']
        assert 'manager' in stats['components_loaded']
        assert 'refiner' in stats['components_loaded']
        
        # Verify stats structure
        assert 'optimizer' in stats
        assert stats['optimizer']['hit_rate'] == 0.67


class TestGroundTruthHubBackwardCompatibility:
    """Test backward compatibility with existing imports."""
    
    def test_old_imports_still_work(self):
        """Old import paths should still work."""
        # These should not raise ImportError
        from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
        from ign_lidar.optimization.ground_truth import GroundTruthOptimizer
        from ign_lidar.core.ground_truth_manager import GroundTruthManager
        from ign_lidar.core.classification.ground_truth_refinement import GroundTruthRefiner
        
        # Should be able to instantiate
        assert IGNGroundTruthFetcher is not None
        assert GroundTruthOptimizer is not None
        assert GroundTruthManager is not None
        assert GroundTruthRefiner is not None
    
    def test_hub_provides_same_classes(self):
        """Hub properties should return the same classes as direct imports."""
        from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
        from ign_lidar.optimization.ground_truth import GroundTruthOptimizer
        from ign_lidar.core.ground_truth_manager import GroundTruthManager
        from ign_lidar.core.classification.ground_truth_refinement import GroundTruthRefiner
        
        # Create fresh hub instance and reset cached components
        hub = GroundTruthHub()
        hub._fetcher = None
        hub._optimizer = None
        hub._manager = None
        hub._refiner = None
        
        # Check types match
        assert isinstance(hub.fetcher, IGNGroundTruthFetcher)
        assert isinstance(hub.optimizer, GroundTruthOptimizer)
        assert isinstance(hub.manager, GroundTruthManager)
        assert isinstance(hub.refiner, GroundTruthRefiner)


class TestGroundTruthHubIntegration:
    """Integration tests with real (non-mocked) components."""
    
    def test_can_access_all_properties(self):
        """Should be able to access all properties without errors."""
        hub = GroundTruthHub()
        
        # These should not raise errors
        fetcher = hub.fetcher
        optimizer = hub.optimizer
        manager = hub.manager
        refiner = hub.refiner
        
        assert fetcher is not None
        assert optimizer is not None
        assert manager is not None
        assert refiner is not None
    
    def test_repr_with_loaded_components(self):
        """Repr should show loaded components."""
        hub = GroundTruthHub()
        
        # Load some components
        _ = hub.fetcher
        _ = hub.optimizer
        
        repr_str = repr(hub)
        assert "fetcher" in repr_str
        assert "optimizer" in repr_str
    
    def test_statistics_with_no_components_loaded(self):
        """get_statistics should work with no components loaded."""
        hub = GroundTruthHub()
        # Reset all
        hub._fetcher = None
        hub._optimizer = None
        hub._manager = None
        hub._refiner = None
        
        stats = hub.get_statistics()
        
        assert stats['components_loaded'] == []
    
    def test_clear_caches_with_no_components(self):
        """clear_all_caches should work with no components loaded."""
        hub = GroundTruthHub()
        # Reset all
        hub._fetcher = None
        hub._optimizer = None
        hub._manager = None
        hub._refiner = None
        
        cleared = hub.clear_all_caches()
        
        assert cleared == {}


class TestGroundTruthHubErrorHandling:
    """Test error handling and edge cases."""
    
    def test_fetch_and_label_empty_points(self):
        """fetch_and_label should handle empty point arrays."""
        hub = GroundTruthHub()
        hub._manager = Mock()
        hub._manager.prefetch_ground_truth_for_tile.return_value = {'buildings': Mock()}
        hub._optimizer = Mock()
        hub._optimizer.label_points.return_value = np.array([])
        
        points = np.empty((0, 3))
        labels, metadata = hub.fetch_and_label("test.laz", points)
        
        assert len(labels) == 0
        assert metadata['n_total'] == 0
    
    def test_process_tile_complete_no_features_no_refinement(self):
        """process_tile_complete should skip refinement if no features."""
        hub = GroundTruthHub()
        hub._manager = Mock()
        hub._manager.prefetch_ground_truth_for_tile.return_value = {'buildings': Mock()}
        hub._optimizer = Mock()
        hub._optimizer.label_points.return_value = np.array([1, 2, 3])
        hub._refiner = Mock()
        
        points = np.random.rand(3, 3)
        labels, stats = hub.process_tile_complete(
            tile_path="test.laz",
            points=points,
            features=None,  # No features
            refine=True  # Want refinement but can't
        )
        
        # Should not call refiner
        hub._refiner.refine_all.assert_not_called()
        assert stats['refined'] is False
        assert stats['refine_reason'] == 'no_features'


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_points():
    """Generate sample point cloud."""
    np.random.seed(42)
    return np.random.rand(1000, 3) * 100  # 1000 points in 100x100x100 space


@pytest.fixture
def sample_features():
    """Generate sample features."""
    np.random.seed(42)
    return {
        'normals': np.random.rand(1000, 3),
        'curvature': np.random.rand(1000),
        'planarity': np.random.rand(1000),
        'verticality': np.random.rand(1000)
    }


# ============================================================================
# Parametrized Tests
# ============================================================================

@pytest.mark.parametrize("component_name,property_name", [
    ("_fetcher", "fetcher"),
    ("_optimizer", "optimizer"),
    ("_manager", "manager"),
    ("_refiner", "refiner"),
])
def test_all_components_lazy_load(component_name, property_name):
    """All components should lazy load consistently."""
    hub = GroundTruthHub()
    
    # Reset component
    setattr(hub, component_name, None)
    
    # Verify not loaded
    assert getattr(hub, component_name) is None
    
    # Access property
    component = getattr(hub, property_name)
    
    # Verify loaded
    assert getattr(hub, component_name) is not None
    assert component is getattr(hub, component_name)


@pytest.mark.parametrize("feature_types", [
    None,
    ['buildings'],
    ['buildings', 'roads'],
    ['buildings', 'roads', 'vegetation'],
])
def test_fetch_and_label_with_different_features(feature_types):
    """fetch_and_label should work with different feature type combinations."""
    hub = GroundTruthHub()
    hub._manager = Mock()
    hub._manager.prefetch_ground_truth_for_tile.return_value = {
        ft: Mock() for ft in (feature_types or ['buildings'])
    }
    hub._optimizer = Mock()
    hub._optimizer.label_points.return_value = np.array([1, 2, 3])
    
    points = np.random.rand(3, 3)
    labels, metadata = hub.fetch_and_label(
        tile_path="test.laz",
        points=points,
        feature_types=feature_types
    )
    
    assert len(labels) == 3
    assert 'feature_types' in metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
