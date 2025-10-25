"""
Tests for the unified Config system (v3.2+).

Tests the new simplified configuration system including:
- Config class initialization
- Preset system
- Auto-configuration from environment
- Migration from old format
- Validation

Author: IGN LiDAR HD Team
Date: October 25, 2025
"""

from pathlib import Path

import numpy as np
import pytest

from ign_lidar.config import AdvancedConfig, Config, FeatureConfig


class TestConfigBasics:
    """Test basic Config functionality."""

    def test_config_creation_minimal(self, tmp_path):
        """Test creating a minimal config."""
        config = Config(
            input_dir=str(tmp_path / "input"), output_dir=str(tmp_path / "output")
        )

        assert config.input_dir == str(tmp_path / "input")
        assert config.output_dir == str(tmp_path / "output")
        assert config.mode == "lod2"  # Default
        assert config.use_gpu is False  # Default

    def test_config_with_custom_values(self, tmp_path):
        """Test creating config with custom values."""
        config = Config(
            input_dir=str(tmp_path / "input"),
            output_dir=str(tmp_path / "output"),
            mode="lod3",
            use_gpu=True,
            num_workers=8,
        )

        assert config.mode == "lod3"
        assert config.use_gpu is True
        assert config.num_workers == 8

    def test_feature_config(self, tmp_path):
        """Test FeatureConfig."""
        features = FeatureConfig(feature_set="full", k_neighbors=40, use_rgb=True)

        assert features.feature_set == "full"
        assert features.k_neighbors == 40
        assert features.use_rgb is True


class TestConfigPresets:
    """Test preset system."""

    def test_asprs_production_preset(self):
        """Test ASPRS production preset."""
        config = Config.preset("asprs_production")

        assert config.mode == "asprs"
        assert config.processing_mode == "both"
        assert config.features.feature_set == "standard"

    def test_lod2_buildings_preset(self):
        """Test LOD2 buildings preset."""
        config = Config.preset("lod2_buildings")

        assert config.mode == "lod2"
        assert config.processing_mode == "patches_only"

    def test_gpu_optimized_preset(self):
        """Test GPU optimized preset."""
        config = Config.preset("gpu_optimized")

        assert config.use_gpu is True
        assert config.num_workers == 1

    def test_preset_with_overrides(self, tmp_path):
        """Test preset with overrides."""
        config = Config.preset(
            "asprs_production",
            input_dir=str(tmp_path / "input"),
            output_dir=str(tmp_path / "output"),
            num_workers=16,
        )

        # Preset values
        assert config.mode == "asprs"

        # Overridden values
        assert config.num_workers == 16
        assert config.input_dir == str(tmp_path / "input")

    def test_invalid_preset_name(self):
        """Test error with invalid preset name."""
        with pytest.raises(ValueError, match="Unknown preset"):
            Config.preset("nonexistent_preset")


class TestConfigFromEnvironment:
    """Test auto-configuration from environment."""

    def test_from_environment_basic(self, tmp_path):
        """Test basic auto-configuration."""
        config = Config.from_environment(
            input_dir=str(tmp_path / "input"), output_dir=str(tmp_path / "output")
        )

        assert config.input_dir == str(tmp_path / "input")
        assert config.output_dir == str(tmp_path / "output")

        # Should auto-detect CPU count
        assert config.num_workers >= 1
        assert config.num_workers <= 16

    def test_from_environment_with_overrides(self, tmp_path):
        """Test auto-config with overrides."""
        config = Config.from_environment(
            input_dir=str(tmp_path / "input"),
            output_dir=str(tmp_path / "output"),
            mode="lod3",
            num_workers=4,
        )

        assert config.mode == "lod3"
        assert config.num_workers == 4


class TestConfigMigration:
    """Test migration from old config format."""

    def test_migrate_v31_to_v32(self):
        """Test migrating v3.1 config to v3.2."""
        old_config = {
            "input_dir": "/data/tiles",
            "output_dir": "/data/output",
            "processor": {"lod_level": "LOD2", "use_gpu": True, "num_workers": 8},
            "features": {"mode": "lod2", "k_neighbors": 30, "use_rgb": True},
        }

        config = Config.from_dict(old_config)

        assert config.mode == "lod2"
        assert config.use_gpu is True
        assert config.num_workers == 8
        assert config.features.k_neighbors == 30
        assert config.features.use_rgb is True

    def test_migrate_feature_mode_mapping(self):
        """Test mapping of old feature modes."""
        old_configs = [
            {"processor": {}, "features": {"mode": "minimal"}},
            {"processor": {}, "features": {"mode": "lod2"}},
            {"processor": {}, "features": {"mode": "lod3"}},
            {"processor": {}, "features": {"mode": "full"}},
        ]

        expected_feature_sets = ["minimal", "standard", "full", "full"]

        for old_cfg, expected in zip(old_configs, expected_feature_sets):
            old_cfg["input_dir"] = "/data"
            old_cfg["output_dir"] = "/out"
            config = Config.from_dict(old_cfg)
            assert config.features.feature_set == expected


class TestConfigValidation:
    """Test config validation."""

    def test_validation_missing_input_dir(self, tmp_path):
        """Test validation fails with missing input_dir."""
        config = Config(input_dir="", output_dir=str(tmp_path / "output"))  # Empty

        with pytest.raises(ValueError, match="input_dir is required"):
            config.validate()

    def test_validation_invalid_patch_size(self, tmp_path):
        """Test validation fails with invalid patch_size."""
        config = Config(
            input_dir=str(tmp_path), output_dir=str(tmp_path / "output"), patch_size=-10
        )

        with pytest.raises(ValueError, match="patch_size must be > 0"):
            config.validate()

    def test_validation_invalid_num_workers(self, tmp_path):
        """Test validation fails with invalid num_workers."""
        config = Config(
            input_dir=str(tmp_path), output_dir=str(tmp_path / "output"), num_workers=0
        )

        with pytest.raises(ValueError, match="num_workers must be >= 1"):
            config.validate()


class TestFeatureConfig:
    """Test FeatureConfig specifics."""

    def test_ndvi_requires_rgb_and_nir(self):
        """Test that NDVI requires both RGB and NIR."""
        # This should warn and disable NDVI
        with pytest.warns(UserWarning, match="compute_ndvi=True requires"):
            features = FeatureConfig(compute_ndvi=True, use_rgb=True, use_nir=False)

        # NDVI should be disabled
        assert features.compute_ndvi is False

    def test_multi_scale_default_scales(self):
        """Test multi-scale gets default scales if not specified."""
        features = FeatureConfig(multi_scale=True)

        assert features.scales is not None
        assert len(features.scales) == 3
        assert "fine" in features.scales
        assert "medium" in features.scales
        assert "coarse" in features.scales


class TestConfigSerialization:
    """Test config serialization."""

    def test_to_dict(self, tmp_path):
        """Test converting Config to dict."""
        config = Config(
            input_dir=str(tmp_path / "input"),
            output_dir=str(tmp_path / "output"),
            mode="lod2",
            use_gpu=True,
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["mode"] == "lod2"
        assert config_dict["use_gpu"] is True


@pytest.fixture
def sample_config(tmp_path):
    """Fixture providing a sample config."""
    return Config(
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "output"),
        mode="lod2",
        use_gpu=False,
        num_workers=4,
    )


def test_config_repr(sample_config):
    """Test config string representation."""
    repr_str = repr(sample_config)
    assert "Config" in repr_str
