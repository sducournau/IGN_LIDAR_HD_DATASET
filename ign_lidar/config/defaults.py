"""
Default configuration values for IGN LiDAR HD.

Provides factory functions and constants for default configurations.
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


# Default configuration instance
DEFAULT_CONFIG = IGNLiDARConfig()

# Common configuration presets
PRESET_CONFIGS = {
    "buildings_lod2": {
        "description": "Optimized for building LOD2 classification",
        "processor": ProcessorConfig(
            lod_level="LOD2",
            num_points=8192,
        ),
        "features": FeaturesConfig(
            mode="minimal",
            include_extra=True,  # Building-specific features
            use_rgb=False,
        ),
        "preprocess": PreprocessConfig(
            enabled=True,
        ),
    },
    "buildings_lod3": {
        "description": "Optimized for building LOD3 classification",
        "processor": ProcessorConfig(
            lod_level="LOD3",
            num_points=16384,
        ),
        "features": FeaturesConfig(
            mode="full",
            include_extra=True,
            use_rgb=True,
        ),
        "preprocess": PreprocessConfig(
            enabled=True,
        ),
    },
    "vegetation": {
        "description": "Optimized for vegetation segmentation",
        "processor": ProcessorConfig(
            num_points=16384,
        ),
        "features": FeaturesConfig(
            mode="full",
            use_rgb=True,
            use_infrared=True,
            compute_ndvi=True,
        ),
    },
    "pointnet_training": {
        "description": "Optimized for PointNet++ training",
        "processor": ProcessorConfig(
            num_points=16384,
            augment=True,
            num_augmentations=5,
            use_gpu=True,
        ),
        "features": FeaturesConfig(
            mode="full",
            k_neighbors=10,
            sampling_method="fps",  # Farthest Point Sampling
            normalize_xyz=True,
            normalize_features=True,
            use_rgb=True,
        ),
        "stitching": StitchingConfig(
            enabled=True,
            buffer_size=10.0,
        ),
        "output": OutputConfig(
            format="torch",
        ),
    },
    "semantic_sota": {
        "description": "State-of-the-art semantic segmentation (full features)",
        "processor": ProcessorConfig(
            num_points=32768,
            use_gpu=True,
        ),
        "features": FeaturesConfig(
            mode="full",
            k_neighbors=20,
            include_extra=True,
            use_rgb=True,
            use_infrared=True,
            compute_ndvi=True,
            sampling_method="fps",
            normalize_xyz=True,
            normalize_features=True,
        ),
        "stitching": StitchingConfig(
            enabled=True,
            buffer_size=15.0,
        ),
        "preprocess": PreprocessConfig(
            enabled=True,
        ),
    },
    "fast": {
        "description": "Fast processing with minimal features",
        "processor": ProcessorConfig(
            num_points=4096,
            num_workers=-1,  # Use all CPUs
        ),
        "features": FeaturesConfig(
            mode="minimal",
            k_neighbors=10,
            include_extra=False,
            use_rgb=False,
        ),
        "preprocess": PreprocessConfig(
            enabled=False,
        ),
        "stitching": StitchingConfig(
            enabled=False,
        ),
    },
}


def get_preset_config(preset_name: str) -> dict:
    """
    Get a preset configuration by name.
    
    Args:
        preset_name: Name of the preset configuration
        
    Returns:
        Dictionary with preset configuration values
        
    Raises:
        KeyError: If preset_name doesn't exist
    """
    if preset_name not in PRESET_CONFIGS:
        available = ", ".join(PRESET_CONFIGS.keys())
        raise KeyError(
            f"Unknown preset '{preset_name}'. "
            f"Available presets: {available}"
        )
    
    return PRESET_CONFIGS[preset_name]


def list_presets() -> dict:
    """
    List all available preset configurations.
    
    Returns:
        Dictionary mapping preset names to their descriptions
    """
    return {
        name: cfg["description"]
        for name, cfg in PRESET_CONFIGS.items()
    }
