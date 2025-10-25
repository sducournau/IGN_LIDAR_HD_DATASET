"""
Tests for multi-scale feature computation configuration.

Tests the configuration schema extension for multi-scale features (v6.2).
"""

import pytest
from omegaconf import OmegaConf
from ign_lidar.config.schema import FeaturesConfig


class TestMultiScaleConfig:
    """Test multi-scale configuration schema."""

    def test_default_config(self):
        """Test that default config has multi-scale disabled."""
        config = FeaturesConfig()
        assert config.multi_scale_computation is False
        assert config.scales is None
        assert config.aggregation_method == "variance_weighted"
        assert config.variance_penalty_factor == 2.0

    def test_multi_scale_validation_requires_scales(self):
        """Test that multi-scale requires at least 2 scales."""
        with pytest.raises(ValueError, match="at least 2 scales"):
            FeaturesConfig(multi_scale_computation=True, scales=None)

        with pytest.raises(ValueError, match="at least 2 scales"):
            FeaturesConfig(multi_scale_computation=True, scales=[])

        with pytest.raises(ValueError, match="at least 2 scales"):
            FeaturesConfig(
                multi_scale_computation=True,
                scales=[
                    {
                        "name": "fine",
                        "k_neighbors": 30,
                        "search_radius": 1.0,
                        "weight": 0.5,
                    }
                ],
            )

    def test_multi_scale_validation_requires_fields(self):
        """Test that each scale requires all fields."""
        with pytest.raises(ValueError, match="missing required fields"):
            FeaturesConfig(
                multi_scale_computation=True,
                scales=[
                    {"name": "fine", "k_neighbors": 30},  # Missing fields
                    {
                        "name": "coarse",
                        "k_neighbors": 100,
                        "search_radius": 3.0,
                        "weight": 0.5,
                    },
                ],
            )

    def test_multi_scale_validation_positive_values(self):
        """Test that numeric fields must be positive."""
        # Invalid k_neighbors
        with pytest.raises(ValueError, match="k_neighbors must be > 0"):
            FeaturesConfig(
                multi_scale_computation=True,
                scales=[
                    {
                        "name": "fine",
                        "k_neighbors": 0,
                        "search_radius": 1.0,
                        "weight": 0.5,
                    },
                    {
                        "name": "coarse",
                        "k_neighbors": 100,
                        "search_radius": 3.0,
                        "weight": 0.5,
                    },
                ],
            )

        # Invalid search_radius
        with pytest.raises(ValueError, match="search_radius must be > 0"):
            FeaturesConfig(
                multi_scale_computation=True,
                scales=[
                    {
                        "name": "fine",
                        "k_neighbors": 30,
                        "search_radius": 0,
                        "weight": 0.5,
                    },
                    {
                        "name": "coarse",
                        "k_neighbors": 100,
                        "search_radius": 3.0,
                        "weight": 0.5,
                    },
                ],
            )

        # Invalid weight (negative)
        with pytest.raises(ValueError, match="weight must be >= 0"):
            FeaturesConfig(
                multi_scale_computation=True,
                scales=[
                    {
                        "name": "fine",
                        "k_neighbors": 30,
                        "search_radius": 1.0,
                        "weight": -0.5,
                    },
                    {
                        "name": "coarse",
                        "k_neighbors": 100,
                        "search_radius": 3.0,
                        "weight": 0.5,
                    },
                ],
            )

    def test_valid_multi_scale_config(self):
        """Test that valid multi-scale config is accepted."""
        config = FeaturesConfig(
            multi_scale_computation=True,
            scales=[
                {
                    "name": "fine",
                    "k_neighbors": 30,
                    "search_radius": 1.0,
                    "weight": 0.3,
                },
                {
                    "name": "medium",
                    "k_neighbors": 80,
                    "search_radius": 2.5,
                    "weight": 0.5,
                },
                {
                    "name": "coarse",
                    "k_neighbors": 150,
                    "search_radius": 5.0,
                    "weight": 0.2,
                },
            ],
            aggregation_method="variance_weighted",
            variance_penalty_factor=2.0,
        )

        assert config.multi_scale_computation is True
        assert len(config.scales) == 3
        assert config.scales[0]["name"] == "fine"
        assert config.scales[1]["k_neighbors"] == 80
        assert config.scales[2]["search_radius"] == 5.0

    def test_adaptive_aggregation_requires_adaptive_selection(self):
        """Test that adaptive aggregation requires adaptive_scale_selection."""
        with pytest.raises(ValueError, match="requires adaptive_scale_selection"):
            FeaturesConfig(
                multi_scale_computation=True,
                scales=[
                    {
                        "name": "fine",
                        "k_neighbors": 30,
                        "search_radius": 1.0,
                        "weight": 0.5,
                    },
                    {
                        "name": "coarse",
                        "k_neighbors": 100,
                        "search_radius": 3.0,
                        "weight": 0.5,
                    },
                ],
                aggregation_method="adaptive",
                adaptive_scale_selection=False,
            )

    def test_performance_warning_many_scales(self):
        """Test warning for parallel computation with many scales."""
        with pytest.warns(UserWarning, match="significant memory"):
            FeaturesConfig(
                multi_scale_computation=True,
                scales=[
                    {
                        "name": f"scale_{i}",
                        "k_neighbors": 30 + i * 20,
                        "search_radius": 1.0 + i * 0.5,
                        "weight": 1.0 / 5,
                    }
                    for i in range(5)
                ],
                parallel_scale_computation=True,
            )

    def test_omegaconf_integration(self):
        """Test that config works with OmegaConf (Hydra)."""
        # Create config dict (as Hydra would load from YAML)
        cfg_dict = {
            "multi_scale_computation": True,
            "scales": [
                {
                    "name": "fine",
                    "k_neighbors": 30,
                    "search_radius": 1.0,
                    "weight": 0.5,
                },
                {
                    "name": "coarse",
                    "k_neighbors": 100,
                    "search_radius": 3.0,
                    "weight": 0.5,
                },
            ],
            "aggregation_method": "variance_weighted",
            "variance_penalty_factor": 2.0,
        }

        # Convert to OmegaConf (simulating Hydra)
        omega_cfg = OmegaConf.create(cfg_dict)

        # Should be able to instantiate FeaturesConfig
        # Note: In real use, OmegaConf.structured() would handle this
        config = FeaturesConfig(
            multi_scale_computation=omega_cfg.multi_scale_computation,
            scales=OmegaConf.to_container(omega_cfg.scales),
            aggregation_method=omega_cfg.aggregation_method,
            variance_penalty_factor=omega_cfg.variance_penalty_factor,
        )

        assert config.multi_scale_computation is True
        assert len(config.scales) == 2


class TestMultiScaleArtifactDetectionConfig:
    """Test artifact detection configuration."""

    def test_artifact_detection_defaults(self):
        """Test default artifact detection settings."""
        config = FeaturesConfig()
        assert config.artifact_detection is False
        assert config.artifact_variance_threshold == 0.15
        assert config.artifact_gradient_threshold == 0.10
        assert config.auto_suppress_artifacts is True

    def test_artifact_detection_enabled(self):
        """Test enabling artifact detection."""
        config = FeaturesConfig(
            multi_scale_computation=True,
            scales=[
                {
                    "name": "fine",
                    "k_neighbors": 30,
                    "search_radius": 1.0,
                    "weight": 0.5,
                },
                {
                    "name": "coarse",
                    "k_neighbors": 100,
                    "search_radius": 3.0,
                    "weight": 0.5,
                },
            ],
            artifact_detection=True,
            artifact_variance_threshold=0.20,
            artifact_gradient_threshold=0.15,
        )

        assert config.artifact_detection is True
        assert config.artifact_variance_threshold == 0.20
        assert config.artifact_gradient_threshold == 0.15


class TestMultiScaleAdaptiveSelectionConfig:
    """Test adaptive scale selection configuration."""

    def test_adaptive_selection_defaults(self):
        """Test default adaptive selection settings."""
        config = FeaturesConfig()
        assert config.adaptive_scale_selection is False
        assert config.complexity_threshold == 0.5
        assert config.homogeneity_threshold == 0.8

    def test_adaptive_selection_enabled(self):
        """Test enabling adaptive scale selection."""
        config = FeaturesConfig(
            multi_scale_computation=True,
            scales=[
                {
                    "name": "fine",
                    "k_neighbors": 30,
                    "search_radius": 1.0,
                    "weight": 0.5,
                },
                {
                    "name": "coarse",
                    "k_neighbors": 100,
                    "search_radius": 3.0,
                    "weight": 0.5,
                },
            ],
            adaptive_scale_selection=True,
            complexity_threshold=0.6,
            homogeneity_threshold=0.7,
            aggregation_method="adaptive",
        )

        assert config.adaptive_scale_selection is True
        assert config.complexity_threshold == 0.6
        assert config.homogeneity_threshold == 0.7


class TestMultiScaleOutputConfig:
    """Test multi-scale output configuration."""

    def test_output_defaults(self):
        """Test default output settings."""
        config = FeaturesConfig()
        assert config.save_scale_quality_metrics is False
        assert config.save_selected_scale is False

    def test_output_enabled(self):
        """Test enabling output options."""
        config = FeaturesConfig(
            multi_scale_computation=True,
            scales=[
                {
                    "name": "fine",
                    "k_neighbors": 30,
                    "search_radius": 1.0,
                    "weight": 0.5,
                },
                {
                    "name": "coarse",
                    "k_neighbors": 100,
                    "search_radius": 3.0,
                    "weight": 0.5,
                },
            ],
            save_scale_quality_metrics=True,
            save_selected_scale=True,
        )

        assert config.save_scale_quality_metrics is True
        assert config.save_selected_scale is True


class TestMultiScalePerformanceConfig:
    """Test multi-scale performance configuration."""

    def test_performance_defaults(self):
        """Test default performance settings."""
        config = FeaturesConfig()
        assert config.reuse_kdtrees_across_scales is True
        assert config.parallel_scale_computation is False
        assert config.cache_scale_results is True

    def test_performance_settings(self):
        """Test custom performance settings."""
        config = FeaturesConfig(
            multi_scale_computation=True,
            scales=[
                {
                    "name": "fine",
                    "k_neighbors": 30,
                    "search_radius": 1.0,
                    "weight": 0.5,
                },
                {
                    "name": "coarse",
                    "k_neighbors": 100,
                    "search_radius": 3.0,
                    "weight": 0.5,
                },
            ],
            reuse_kdtrees_across_scales=False,
            parallel_scale_computation=True,
            cache_scale_results=False,
        )

        assert config.reuse_kdtrees_across_scales is False
        assert config.parallel_scale_computation is True
        assert config.cache_scale_results is False
