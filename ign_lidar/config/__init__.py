"""
Configuration module for IGN LiDAR HD.

v3.2+ Changes:
    - New Config class (replaces ProcessorConfig + FeaturesConfig)
    - Smart presets for common use cases
    - Auto-configuration from environment
    - Simplified parameter structure (15 vs 118 params)

Quick Start (v3.2+):
    >>> from ign_lidar.config import Config
    >>> config = Config.preset('asprs_production')
    >>> config.input_dir = '/data/tiles'
    >>> config.output_dir = '/data/output'

Legacy (v3.1, deprecated):
    >>> from ign_lidar.config import ProcessorConfig, FeaturesConfig
    # Still works but will be removed in v4.0
"""

# ============================================================================
# New Configuration (v3.2+) - RECOMMENDED
# ============================================================================

import warnings

from .config import AdvancedConfig, Config, FeatureConfig
from .building_config import (
    BuildingConfig,
    EnhancedBuildingConfig,
)  # EnhancedBuildingConfig is deprecated alias

# Week 3: Modern preset-based configuration loader
from .preset_loader import (
    ConfigLoader,
    ConfigLoaderError,
    PresetConfigLoader,
    load_config_with_preset,
)
from .schema import (
    BBoxConfig,
    FeaturesConfig,
    IGNLiDARConfig,
    OutputConfig,
    PreprocessConfig,
    ProcessorConfig,
    StitchingConfig,
)

# ============================================================================
# Legacy Configuration (v3.1 and earlier) - DEPRECATED
# ============================================================================


__all__ = [
    # NEW API (v3.2+) - Use this!
    "Config",
    "FeatureConfig",
    "AdvancedConfig",
    "BuildingConfig",
    "EnhancedBuildingConfig",  # Deprecated alias, use BuildingConfig
    # Legacy Hydra schemas (deprecated, will be removed in v4.0)
    "ProcessorConfig",
    "FeaturesConfig",
    "PreprocessConfig",
    "StitchingConfig",
    "OutputConfig",
    "BBoxConfig",
    "IGNLiDARConfig",
    "register_configs",
    # Legacy preset loaders (deprecated)
    "PresetConfigLoader",
    "ConfigLoader",
    "load_config_with_preset",
    "ConfigLoaderError",
]


def register_configs() -> None:
    """
    Register all structured configs with Hydra ConfigStore.

    This enables:
    - Type-safe configuration with IDE autocomplete
    - Better validation at configuration time
    - Cleaner config composition
    - Type hints in CLI

    Note: ConfigStore registration is deferred until needed by Hydra
    to avoid issues with complex type annotations (Literal, Union, etc).
    This function should be called explicitly when initializing Hydra.
    """
    try:
        from hydra.core.config_store import ConfigStore

        cs = ConfigStore.instance()

        # Note: Some complex types (Literal, Union) may not be fully supported
        # by OmegaConf's structured config. We register what we can.

        # Register config groups with base schemas
        # These use simpler types and work well with ConfigStore
        cs.store(group="features", name="base_schema", node=FeaturesConfig)
        cs.store(group="preprocess", name="base_schema", node=PreprocessConfig)
        cs.store(group="stitching", name="base_schema", node=StitchingConfig)
        cs.store(group="output", name="base_schema", node=OutputConfig)
        cs.store(group="bbox", name="base_schema", node=BBoxConfig)

        # Note: ProcessorConfig and IGNLiDARConfig are not registered due to
        # Literal types which OmegaConf doesn't fully support yet.
        # These configs still work via YAML files.

    except Exception as e:
        # ConfigStore registration is optional - configs will still work via YAML
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(f"ConfigStore registration skipped: {e}")


# Note: Auto-registration is commented out to avoid import-time errors
# Call register_configs() explicitly when needed, or use YAML configs directly
# register_configs()
