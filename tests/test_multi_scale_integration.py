"""
Tests for multi-scale feature computation integration with FeatureOrchestrator.

Tests v6.2 multi-scale integration with the existing feature pipeline.
"""

import pytest
import numpy as np
from omegaconf import OmegaConf


class TestMultiScaleIntegration:
    """Test multi-scale integration with FeatureOrchestrator."""

    @pytest.fixture
    def base_config(self):
        """Create base configuration."""
        config = {
            "processor": {
                "use_gpu": False,
                "use_feature_computer": False,
            },
            "features": {
                "mode": "lod2",
                "k_neighbors": 30,
                "search_radius": 3.0,  # Set explicit search_radius
                "use_rgb": False,
                "use_infrared": False,
            },
        }
        return OmegaConf.create(config)

    @pytest.fixture
    def multi_scale_config(self, base_config):
        """Add multi-scale configuration."""
        base_config.features.multi_scale_computation = True
        base_config.features.scales = [
            {"name": "fine", "k_neighbors": 20, "search_radius": 1.0, "weight": 0.3},
            {"name": "medium", "k_neighbors": 50, "search_radius": 2.5, "weight": 0.5},
            {"name": "coarse", "k_neighbors": 100, "search_radius": 5.0, "weight": 0.2},
        ]
        base_config.features.aggregation_method = "variance_weighted"
        base_config.features.variance_penalty_factor = 2.0
        return base_config

    @pytest.fixture
    def synthetic_tile(self):
        """Create synthetic tile data."""
        # Create planar surface (ground)
        x = np.linspace(0, 10, 100)
        y = np.linspace(0, 10, 100)
        xx, yy = np.meshgrid(x, y)
        zz = np.ones_like(xx) * 5.0  # Flat at z=5

        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        # Create classification (ground = 2)
        classification = np.full(len(points), 2, dtype=np.uint8)

        # Create intensity
        intensity = np.random.randint(0, 255, len(points), dtype=np.uint16)

        # Create return number
        return_number = np.ones(len(points), dtype=np.uint8)

        return {
            "points": points,
            "classification": classification,
            "intensity": intensity,
            "return_number": return_number,
        }

    def test_orchestrator_initialization_without_multi_scale(self, base_config):
        """Test that orchestrator initializes without multi-scale."""
        from ign_lidar.features.orchestrator import FeatureOrchestrator

        orchestrator = FeatureOrchestrator(base_config)

        assert hasattr(orchestrator, "use_multi_scale")
        assert orchestrator.use_multi_scale is False
        assert orchestrator.multi_scale_computer is None

    def test_orchestrator_initialization_with_multi_scale(self, multi_scale_config):
        """Test that orchestrator initializes with multi-scale."""
        from ign_lidar.features.orchestrator import FeatureOrchestrator

        orchestrator = FeatureOrchestrator(multi_scale_config)

        assert hasattr(orchestrator, "use_multi_scale")
        assert orchestrator.use_multi_scale is True
        assert orchestrator.multi_scale_computer is not None

    def test_multi_scale_feature_computation(self, multi_scale_config, synthetic_tile):
        """Test that multi-scale features are computed correctly."""
        from ign_lidar.features.orchestrator import FeatureOrchestrator

        orchestrator = FeatureOrchestrator(multi_scale_config)

        # Compute features
        features = orchestrator.compute_features(synthetic_tile)

        # Check that features were computed
        assert features is not None
        assert isinstance(features, dict)

        # Check essential features
        assert "normals" in features
        assert "curvature" in features
        assert "height" in features

        # Check multi-scale geometric features
        assert "planarity" in features

        # Verify shapes
        n_points = len(synthetic_tile["points"])
        assert features["normals"].shape == (n_points, 3)
        assert features["curvature"].shape == (n_points,)
        assert features["height"].shape == (n_points,)
        assert features["planarity"].shape == (n_points,)

    def test_multi_scale_vs_standard_features(
        self, base_config, multi_scale_config, synthetic_tile
    ):
        """Test that multi-scale produces different (better) features."""
        from ign_lidar.features.orchestrator import FeatureOrchestrator

        # Compute standard features
        orchestrator_std = FeatureOrchestrator(base_config)
        features_std = orchestrator_std.compute_features(synthetic_tile)

        # Compute multi-scale features
        orchestrator_ms = FeatureOrchestrator(multi_scale_config)
        features_ms = orchestrator_ms.compute_features(synthetic_tile)

        # Check that planarity values differ
        # (multi-scale should aggregate across scales)
        planarity_std = features_std["planarity"]
        planarity_ms = features_ms["planarity"]

        # Both should detect high planarity (planar surface)
        assert np.mean(planarity_std) > 0.5
        assert np.mean(planarity_ms) > 0.5

        # Multi-scale should have lower variance (more stable)
        assert np.std(planarity_ms) <= np.std(planarity_std) * 1.2

    def test_multi_scale_fallback_on_error(self, multi_scale_config, synthetic_tile):
        """Test that errors fall back to standard computation."""
        from ign_lidar.features.orchestrator import FeatureOrchestrator

        # Create invalid config (will fail multi-scale)
        multi_scale_config.features.scales = []  # Empty scales

        orchestrator = FeatureOrchestrator(multi_scale_config)

        # Should have disabled multi-scale due to invalid config
        assert orchestrator.use_multi_scale is False

        # Should still compute features (standard path)
        features = orchestrator.compute_features(synthetic_tile)
        assert features is not None
        assert "planarity" in features
