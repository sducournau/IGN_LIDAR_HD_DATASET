"""
Configuration module for IGN LiDAR HD v2.0.

Provides Hydra-based structured configuration with:
- Type-safe configuration schemas
- Hierarchical composition
- Command-line overrides
- Validation
- ConfigStore registration for better type safety

Week 3 Addition:
- Preset-based configuration system
- Modern PresetConfigLoader with inheritance
"""

from .schema import (
    ProcessorConfig,
    FeaturesConfig,
    PreprocessConfig,
    StitchingConfig,
    OutputConfig,
    BBoxConfig,
    IGNLiDARConfig,
)

# Week 3: Modern preset-based configuration loader
from .preset_loader import (
    PresetConfigLoader,
    ConfigLoader,
    load_config_with_preset,
    ConfigLoaderError,
)

__all__ = [
    # Hydra schemas
    "ProcessorConfig",
    "FeaturesConfig",
    "PreprocessConfig",
    "StitchingConfig",
    "OutputConfig",
    "BBoxConfig",
    "IGNLiDARConfig",
    "register_configs",
    # Week 3: Preset loaders
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
