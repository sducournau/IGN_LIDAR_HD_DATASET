"""
Configuration System for IGN LiDAR HD v3.2+

This module replaces both schema.py and schema_simplified.py with a single,
intuitive configuration system that dramatically simplifies user experience.

Key Features:
- Single Config class with 15 top-level parameters (vs 118 previously)
- Smart presets for common use cases
- Auto-configuration from environment
- Nested AdvancedConfig for expert users
- Backward compatibility with v3.1 configs

Quick Start:
    >>> from ign_lidar.config import Config
    >>> config = Config.preset('asprs_production')
    >>> config.input_dir = '/data/tiles'
    >>> config.output_dir = '/data/output'

Author: IGN LiDAR HD Team
Date: October 25, 2025
Version: 3.2.0
"""

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

try:
    from omegaconf import MISSING, OmegaConf

    HAS_OMEGACONF = True
except ImportError:
    MISSING = "???"
    HAS_OMEGACONF = False


# ============================================================================
# Main Configuration Classes
# ============================================================================


@dataclass
class FeatureConfig:
    """
    Feature computation configuration.

    Simplified from the old FeaturesConfig with 80+ parameters.
    Now uses 'feature_set' to select a predefined group of features.

    Attributes:
        feature_set: Predefined feature group
            - 'minimal': Ultra-fast, ~8 features (height, planarity, verticality)
            - 'standard': Recommended, ~20 features (geometric + basic spectral)
            - 'full': All features, ~45 features (includes multi-scale, NDVI, etc.)
        k_neighbors: Number of neighbors for local features
        search_radius: Search radius in meters (None = auto-calculate from k_neighbors)
        use_rgb: Include RGB features from orthophotos
        use_nir: Include near-infrared features
        compute_ndvi: Compute NDVI vegetation index (requires RGB + NIR)
        multi_scale: Enable multi-scale feature computation
        scales: Scale names for multi-scale (e.g., ['fine', 'medium', 'coarse'])
    """

    feature_set: str = "standard"  # 'minimal', 'standard', or 'full'
    k_neighbors: int = 30
    search_radius: Optional[float] = None

    # Spectral features
    use_rgb: bool = False
    use_nir: bool = False
    compute_ndvi: bool = False

    # Multi-scale (simplified from 20+ params to 2)
    multi_scale: bool = False
    scales: Optional[List[str]] = None

    def __post_init__(self):
        """Validate feature configuration."""
        if self.compute_ndvi and not (self.use_rgb and self.use_nir):
            warnings.warn(
                "compute_ndvi=True requires both use_rgb=True and use_nir=True. "
                "Setting compute_ndvi=False.",
                UserWarning,
            )
            self.compute_ndvi = False

        if self.multi_scale and not self.scales:
            # Default scales if multi_scale enabled but scales not specified
            self.scales = ["fine", "medium", "coarse"]

    @property
    def feature_list(self) -> List[str]:
        """Get actual feature names based on feature_set."""
        from ign_lidar.features import get_feature_list_for_mode

        # Map feature_set to old mode names for compatibility
        mode_map = {
            "minimal": "minimal",
            "standard": "lod2",  # Standard = LOD2 feature set
            "full": "full",
        }
        mode = mode_map[self.feature_set]
        return get_feature_list_for_mode(mode)


@dataclass
class AdvancedConfig:
    """
    Advanced configuration for expert users.

    Most users should not need to modify these parameters.
    Provides fine-grained control over preprocessing, ground truth,
    classification, and performance tuning.

    These are nested to keep the main Config simple while still
    allowing experts full control when needed.
    """

    # Preprocessing options (outlier removal, etc.)
    preprocessing: Optional[Dict[str, Any]] = None

    # Ground truth configuration (BD TOPO, cadastre)
    ground_truth: Optional[Dict[str, Any]] = None

    # Classification fine-tuning
    classification: Optional[Dict[str, Any]] = None

    # Performance tuning (batch sizes, memory limits, etc.)
    performance: Optional[Dict[str, Any]] = None

    # Reclassification options
    reclassification: Optional[Dict[str, Any]] = None

    # Multi-scale advanced options (for experts)
    multi_scale_advanced: Optional[Dict[str, Any]] = None


@dataclass
class Config:
    """
    Configuration for IGN LiDAR HD processing.

    This is the main configuration class that replaces both schema.py
    and schema_simplified.py. It provides a clean, intuitive interface
    with only 15 top-level parameters that cover 95% of use cases.

    Quick Start Examples:
        # Use a preset
        >>> config = Config.preset('asprs_production')
        >>> config.input_dir = '/data/tiles'
        >>> config.output_dir = '/data/output'

        # Auto-configure from environment
        >>> config = Config.from_environment(
        ...     input_dir='/data/tiles',
        ...     output_dir='/data/output'
        ... )

        # Manual configuration
        >>> config = Config(
        ...     input_dir='/data/tiles',
        ...     output_dir='/data/output',
        ...     mode='lod2',
        ...     use_gpu=True
        ... )

    Attributes:
        input_dir: Input tile directory (required)
        output_dir: Output directory (required)
        mode: Classification mode ('asprs', 'lod2', or 'lod3')
        processing_mode: Output type ('patches_only', 'both', 'enriched_only')
        use_gpu: Enable GPU acceleration
        num_workers: Number of parallel workers
        patch_size: Patch size in meters
        num_points: Target points per patch
        features: Feature computation configuration
        advanced: Advanced options (nested, for experts only)
    """

    # =======================================================================
    # REQUIRED PARAMETERS (must be set by user)
    # =======================================================================

    input_dir: str = MISSING
    output_dir: str = MISSING

    # =======================================================================
    # ESSENTIAL PARAMETERS (commonly modified, sensible defaults)
    # =======================================================================

    # Classification mode: 'asprs', 'lod2', or 'lod3'
    mode: str = "lod2"

    # Processing mode: 'patches_only', 'both', 'enriched_only', 'reclassify_only'
    processing_mode: str = "patches_only"

    # Hardware
    use_gpu: bool = False
    num_workers: int = 4

    # Patch configuration
    patch_size: float = 150.0
    num_points: int = 16384
    patch_overlap: float = 0.1

    # Architecture (for ML datasets)
    architecture: Literal[
        "pointnet++", "hybrid", "octree", "transformer", "sparse_conv", "multi"
    ] = "pointnet++"

    # =======================================================================
    # FEATURE CONFIGURATION (nested)
    # =======================================================================

    features: FeatureConfig = field(default_factory=FeatureConfig)

    # =======================================================================
    # OPTIONAL ADVANCED CONFIGURATION (for experts)
    # =======================================================================

    advanced: Optional[AdvancedConfig] = None

    # =======================================================================
    # PRESET SYSTEM
    # =======================================================================

    @classmethod
    def preset(cls, name: str, **overrides) -> "Config":
        """
        Load a preset configuration.

        Available presets:
            - 'asprs_production': ASPRS classification for production use
            - 'lod2_buildings': LOD2 building detection and classification
            - 'lod3_detailed': LOD3 detailed architectural classification
            - 'gpu_optimized': GPU-accelerated processing for large datasets
            - 'minimal_fast': Minimal features for quick testing

        Args:
            name: Preset name
            **overrides: Override any preset parameters

        Returns:
            Config instance with preset values

        Example:
            >>> config = Config.preset('asprs_production',
            ...                        num_workers=8,
            ...                        use_gpu=True)
            >>> config.input_dir = '/data/tiles'

        Raises:
            ValueError: If preset name is not recognized
        """
        presets = _get_presets()

        if name not in presets:
            available = ", ".join(f"'{p}'" for p in presets.keys())
            raise ValueError(
                f"Unknown preset '{name}'. Available presets: {available}\n\n"
                f"See documentation for preset descriptions: "
                f"https://sducournau.github.io/IGN_LIDAR_HD_DATASET/guides/configuration/"
            )

        # Get preset config
        preset_config = presets[name].copy()

        # Apply overrides
        preset_config.update(overrides)

        # Create Config instance
        return cls(**preset_config)

    @classmethod
    def from_environment(cls, input_dir: str, output_dir: str, **overrides) -> "Config":
        """
        Auto-configure based on system environment.

        Automatically detects:
            - GPU availability (CUDA/CuPy)
            - CPU count (sets num_workers)
            - Available memory (for batch sizing)
            - Input data characteristics (for feature parameters)

        Args:
            input_dir: Input tile directory (required)
            output_dir: Output directory (required)
            **overrides: Override any auto-detected parameters

        Returns:
            Config instance with auto-detected values

        Example:
            >>> config = Config.from_environment(
            ...     input_dir='/data/tiles',
            ...     output_dir='/data/output',
            ...     mode='lod2'  # Override auto-detected mode
            ... )
        """
        # Detect GPU availability
        try:
            from ign_lidar.core.gpu_context import GPU_AVAILABLE

            use_gpu = GPU_AVAILABLE
        except ImportError:
            use_gpu = False

        # Detect CPU count
        cpu_count = os.cpu_count() or 4
        # Limit to 16 workers max (diminishing returns beyond that)
        num_workers = min(cpu_count, 16)

        # If GPU is available, use fewer workers (GPU handles parallelism)
        if use_gpu:
            num_workers = min(num_workers, 2)

        # Build auto-config
        auto_config = {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "use_gpu": use_gpu,
            "num_workers": num_workers,
        }

        # Apply overrides
        auto_config.update(overrides)

        return cls(**auto_config)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """
        Create Config from dictionary (e.g., from YAML file).

        Supports both old (v3.1) and new (v3.2) config formats.
        Old format is automatically migrated with a warning.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Config instance

        Example:
            >>> import yaml
            >>> with open('config.yaml') as f:
            ...     config_dict = yaml.safe_load(f)
            >>> config = Config.from_dict(config_dict)
        """
        # Check if this is an old-format config (has 'processor' or 'features' keys)
        if "processor" in config_dict or (
            "features" in config_dict and isinstance(config_dict.get("features"), dict)
        ):
            warnings.warn(
                "Detected old configuration format (v3.1 or earlier). "
                "Automatically migrating to v3.2 format. "
                "Please update your config file using: ign-lidar migrate-config config.yaml",
                DeprecationWarning,
                stacklevel=2,
            )
            config_dict = _migrate_config(config_dict)

        if HAS_OMEGACONF:
            # Use OmegaConf for validation and type checking
            # Create structured config, then merge with dict
            cfg = OmegaConf.structured(cls)
            OmegaConf.merge(cfg, config_dict)
            return OmegaConf.to_object(cfg)
        else:
            # Fallback to plain dataclass
            return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Config to dictionary.

        Useful for serialization to YAML/JSON.

        Returns:
            Configuration as dictionary
        """
        if HAS_OMEGACONF and OmegaConf.is_config(self):
            result = OmegaConf.to_container(self, resolve=True)
            # Ensure we return a dict
            return result if isinstance(result, dict) else {}
        else:
            # Manual conversion for plain dataclass
            from dataclasses import asdict

            return asdict(self)

    def validate(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        # Check required fields
        if self.input_dir == MISSING or not self.input_dir:
            raise ValueError("input_dir is required")
        if self.output_dir == MISSING or not self.output_dir:
            raise ValueError("output_dir is required")

        # Check paths exist
        input_path = Path(self.input_dir)
        if not input_path.exists():
            raise ValueError(f"input_dir does not exist: {self.input_dir}")

        # Check numeric parameters
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be > 0, got {self.patch_size}")
        if self.num_points <= 0:
            raise ValueError(f"num_points must be > 0, got {self.num_points}")
        if self.num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {self.num_workers}")
        if not 0 <= self.patch_overlap < 1:
            raise ValueError(
                f"patch_overlap must be in [0, 1), got {self.patch_overlap}"
            )

        # Validate feature config
        if self.features.k_neighbors < 1:
            raise ValueError(
                f"k_neighbors must be >= 1, got {self.features.k_neighbors}"
            )


# ============================================================================
# Preset Definitions
# ============================================================================


def _get_presets() -> Dict[str, Dict[str, Any]]:
    """
    Get predefined configuration presets.

    Returns:
        Dictionary mapping preset name to configuration
    """
    return {
        "asprs_production": {
            "mode": "asprs",
            "processing_mode": "both",
            "features": FeatureConfig(
                feature_set="standard",
                k_neighbors=30,
            ),
        },
        "lod2_buildings": {
            "mode": "lod2",
            "processing_mode": "patches_only",
            "features": FeatureConfig(
                feature_set="standard",
                k_neighbors=30,
            ),
        },
        "lod3_detailed": {
            "mode": "lod3",
            "processing_mode": "both",
            "features": FeatureConfig(
                feature_set="full",
                k_neighbors=40,
            ),
        },
        "gpu_optimized": {
            "use_gpu": True,
            "num_workers": 1,  # GPU handles parallelism
            "features": FeatureConfig(
                feature_set="standard",
                k_neighbors=30,
            ),
        },
        "minimal_fast": {
            "mode": "asprs",
            "processing_mode": "patches_only",
            "num_workers": 8,
            "features": FeatureConfig(
                feature_set="minimal",
                k_neighbors=20,
            ),
        },
    }


# ============================================================================
# Migration Utilities
# ============================================================================


def _migrate_config(old_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate old v3.1 config format to v3.2 format.

    Args:
        old_config: Old configuration dictionary

    Returns:
        New configuration dictionary
    """
    new_config = {}

    # Top-level fields
    new_config["input_dir"] = old_config.get("input_dir", MISSING)
    new_config["output_dir"] = old_config.get("output_dir", MISSING)

    # Processor section → top-level
    if "processor" in old_config:
        proc = old_config["processor"]
        new_config["mode"] = proc.get("lod_level", "lod2").lower()
        new_config["use_gpu"] = proc.get("use_gpu", False)
        new_config["num_workers"] = proc.get("num_workers", 4)
        new_config["patch_size"] = proc.get("patch_size", 150.0)
        new_config["num_points"] = proc.get("num_points", 16384)
        new_config["patch_overlap"] = proc.get("patch_overlap", 0.1)
        new_config["architecture"] = proc.get("architecture", "pointnet++")
        new_config["processing_mode"] = proc.get("processing_mode", "patches_only")

    # Features section → features
    if "features" in old_config and isinstance(old_config["features"], dict):
        old_features = old_config["features"]

        # Map old 'mode' to new 'feature_set'
        old_mode = old_features.get("mode", "full")
        feature_set_map = {
            "minimal": "minimal",
            "lod2": "standard",
            "lod3": "full",
            "asprs_classes": "standard",
            "full": "full",
            "custom": "standard",
        }
        feature_set = feature_set_map.get(old_mode, "standard")

        new_config["features"] = {
            "feature_set": feature_set,
            "k_neighbors": old_features.get("k_neighbors", 30),
            "search_radius": old_features.get("search_radius"),
            "use_rgb": old_features.get("use_rgb", False),
            "use_nir": old_features.get("use_infrared", False),  # Note: renamed
            "compute_ndvi": old_features.get("compute_ndvi", False),
            "multi_scale": old_features.get("multi_scale_computation", False),
        }

    # Advanced options → advanced
    advanced = {}

    if "preprocessing" in old_config:
        advanced["preprocessing"] = old_config["preprocessing"]

    if "data_sources" in old_config:
        advanced["ground_truth"] = old_config["data_sources"]

    if "processor" in old_config and "reclassification" in old_config["processor"]:
        advanced["reclassification"] = old_config["processor"]["reclassification"]

    if advanced:
        new_config["advanced"] = advanced

    return new_config


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "Config",
    "FeatureConfig",
    "AdvancedConfig",
]
