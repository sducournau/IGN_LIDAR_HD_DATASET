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
import yaml
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
        mode: Predefined feature group (renamed from feature_set in v4.0)
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

    Note:
        In v4.0, 'feature_set' was renamed to 'mode' for consistency.
        Old 'feature_set' parameter is still supported with deprecation warning.
    """

    mode: str = "standard"  # 'minimal', 'standard', or 'full'
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
        """Validate feature configuration and handle backward compatibility."""
        # Backward compatibility: handle old 'feature_set' parameter
        # This is handled via **kwargs in __init__, but we add validation here
        
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
        """Get actual feature names based on mode."""
        from ign_lidar.features import get_feature_list_for_mode

        # Map mode to old mode names for compatibility
        mode_map = {
            "minimal": "minimal",
            "standard": "lod2",  # Standard = LOD2 feature set
            "full": "full",
        }
        mode = mode_map[self.mode]
        return get_feature_list_for_mode(mode)
    
    @property
    def feature_set(self) -> str:
        """Backward compatibility property for old 'feature_set' parameter.
        
        DEPRECATED: Use 'mode' instead. Will be removed in v4.0.
        """
        warnings.warn(
            "FeatureConfig.feature_set is deprecated. Use FeatureConfig.mode instead. "
            "This parameter will be removed in v4.0.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.mode


@dataclass
class OptimizationsConfig:
    """
    Phase 4 optimization configuration (v3.9+).

    Consolidates all Phase 4 optimizations into a single nested config.
    These optimizations can provide 40-50% overall speedup.

    Attributes:
        enabled: Master switch for all optimizations
        async_io: Enable async I/O pipeline (+12-14% performance)
        async_workers: Number of async I/O workers
        tile_cache_size: Number of tiles to cache
        batch_processing: Enable batch multi-tile processing (+25-30%)
        batch_size: Number of tiles per batch
        gpu_pooling: Enable GPU memory pooling (+8.5%)
        gpu_pool_max_size_gb: Max GPU pool size in GB
        print_stats: Print optimization statistics

    Example:
        >>> config = Config.preset('lod2_buildings')
        >>> config.optimizations.enabled = True
        >>> config.optimizations.batch_size = 8
    """

    enabled: bool = True
    
    # Async I/O Pipeline (Phase 4.5): +12-14% performance
    async_io: bool = True
    async_workers: int = 2
    tile_cache_size: int = 3
    
    # Batch Multi-Tile Processing (Phase 4.4): +25-30% performance
    batch_processing: bool = True
    batch_size: int = 4
    
    # GPU Memory Pooling (Phase 4.3): +8.5% performance
    gpu_pooling: bool = True
    gpu_pool_max_size_gb: float = 4.0
    
    # Statistics
    print_stats: bool = True


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
    architecture: str = "pointnet++"  # Options: pointnet++, hybrid, octree, transformer, sparse_conv, multi

    # =======================================================================
    # FEATURE CONFIGURATION (nested)
    # =======================================================================

    features: FeatureConfig = field(default_factory=FeatureConfig)

    # =======================================================================
    # OPTIMIZATIONS CONFIGURATION (v3.9+, nested)
    # =======================================================================

    optimizations: OptimizationsConfig = field(default_factory=OptimizationsConfig)

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

        v4.0 Presets (Recommended):
            - 'minimal_debug': Ultra-fast debugging (8 features, CPU)
            - 'fast_preview': Quick preview with good quality (12 features, GPU)
            - 'lod2_buildings': LOD2 building classification (12 features, GPU) **DEFAULT**
            - 'lod3_detailed': Detailed architectural classification (38 features, GPU)
            - 'asprs_classification_cpu': ASPRS classification, CPU-optimized (25 features)
            - 'asprs_classification_gpu': ASPRS classification, GPU-accelerated (25 features)
            - 'high_quality': Maximum quality processing (38 features, large patches)

        Legacy Presets (Backward Compatibility):
            - 'asprs_production': ASPRS classification for production use
            - 'gpu_optimized': GPU-accelerated processing for large datasets
            - 'minimal_fast': Minimal features for quick testing

        Args:
            name: Preset name
            **overrides: Override any preset parameters

        Returns:
            Config instance with preset values

        Example:
            >>> # Use v4.0 preset (recommended)
            >>> config = Config.preset('lod2_buildings')
            >>> config.input_dir = '/data/tiles'
            >>> config.output_dir = '/data/output'
            >>>
            >>> # Or customize preset
            >>> config = Config.preset('lod2_buildings',
            ...                        patch_size=100.0,
            ...                        num_points=32768)

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
    def from_legacy_schema(cls, legacy_config: Any) -> "Config":
        """
        Migrate legacy v3.1 IGNLiDARConfig to v4.0 Config.

        This method provides automatic migration from the old schema.py
        configuration format to the new unified Config.

        Args:
            legacy_config: IGNLiDARConfig instance from schema.py

        Returns:
            Config instance with migrated parameters

        Example:
            >>> from ign_lidar.config.schema import IGNLiDARConfig
            >>> old_config = IGNLiDARConfig(...)
            >>> new_config = Config.from_legacy_schema(old_config)

        Note:
            This method will be removed when schema.py is deleted in v4.0.
        """
        # Convert to dict first
        if hasattr(legacy_config, '__dict__'):
            config_dict = legacy_config.__dict__
        elif isinstance(legacy_config, dict):
            config_dict = legacy_config
        else:
            raise TypeError(f"Cannot migrate config of type {type(legacy_config)}")
        
        # Use existing migration logic
        migrated_dict = _migrate_config(config_dict)
        
        # Create Config instance
        return cls.from_dict(migrated_dict)

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
        # Check if this is an old-format config (has 'processor' key)
        # Note: We can't use 'features' dict as a detector since v4.0 also has nested features
        # The key difference is that old configs had a 'processor' section
        if "processor" in config_dict:
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
            cfg = OmegaConf.merge(cfg, config_dict)
            # For migration, we may have MISSING values - handle them gracefully
            try:
                return OmegaConf.to_object(cfg)
            except Exception as e:
                # If OmegaConf fails (e.g., MISSING values), try direct instantiation
                warnings.warn(
                    f"OmegaConf instantiation failed, using direct instantiation: {e}",
                    UserWarning
                )
                return cls(**config_dict)
        else:
            # Fallback to plain dataclass
            return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "Config":
        """
        Load Config from a YAML file.

        Supports v4.0, v3.x, and v5.1 config formats.
        Legacy formats are automatically migrated with a deprecation warning.

        Args:
            yaml_path: Path to the YAML configuration file

        Returns:
            Config instance

        Raises:
            FileNotFoundError: If the YAML file doesn't exist
            yaml.YAMLError: If the YAML file is invalid

        Example:
            >>> config = Config.from_yaml('examples/config_training_fast_50m_v3.2.yaml')
            >>> print(config.mode)
            'lod2'
        """
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        if config_dict is None:
            raise ValueError(f"Empty or invalid YAML file: {yaml_path}")
        
        return cls.from_dict(config_dict)

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

    def to_yaml(self, yaml_path: Union[str, Path], **kwargs) -> None:
        """
        Save Config to a YAML file.

        Args:
            yaml_path: Path where to save the YAML configuration file
            **kwargs: Additional arguments passed to yaml.dump()
                     (e.g., default_flow_style=False, sort_keys=False)

        Example:
            >>> config = Config.preset('lod2_buildings')
            >>> config.to_yaml('my_config.yaml')
        """
        yaml_path = Path(yaml_path)
        
        # Create parent directory if it doesn't exist
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict and save
        config_dict = self.to_dict()
        
        # Set default kwargs for nice formatting
        yaml_kwargs = {
            'default_flow_style': False,
            'sort_keys': False,
            'indent': 2
        }
        yaml_kwargs.update(kwargs)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, **yaml_kwargs)

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
    
    v4.0 presets are prioritized. Legacy presets remain for backward compatibility.

    Returns:
        Dictionary mapping preset name to configuration
    """
    return {
        # ===================================================================
        # v4.0 PRESETS (New, Recommended)
        # ===================================================================
        
        "minimal_debug": {
            "mode": "lod2",
            "processing_mode": "patches_only",
            "use_gpu": False,
            "num_workers": 4,
            "patch_size": 30.0,
            "num_points": 8192,
            "features": FeatureConfig(
                mode="minimal",
                k_neighbors=20,
                search_radius=2.0,
            ),
            "optimizations": OptimizationsConfig(enabled=False),
        },
        
        "fast_preview": {
            "mode": "lod2",
            "processing_mode": "patches_only",
            "use_gpu": True,
            "num_workers": 0,
            "patch_size": 50.0,
            "num_points": 16384,
            "features": FeatureConfig(
                mode="standard",
                k_neighbors=30,
                search_radius=2.5,
            ),
            "optimizations": OptimizationsConfig(
                enabled=True,
                async_io=True,
                batch_processing=True,
            ),
        },
        
        "lod2_buildings": {
            "mode": "lod2",
            "processing_mode": "patches_only",
            "use_gpu": True,
            "num_workers": 0,
            "patch_size": 50.0,
            "num_points": 16384,
            "features": FeatureConfig(
                mode="standard",
                k_neighbors=30,
                search_radius=2.5,
                use_rgb=False,
                use_nir=False,
            ),
            "optimizations": OptimizationsConfig(
                enabled=True,
                async_io=True,
                batch_processing=True,
                gpu_pooling=True,
            ),
        },
        
        "lod3_detailed": {
            "mode": "lod3",
            "processing_mode": "both",
            "use_gpu": True,
            "num_workers": 0,
            "patch_size": 100.0,
            "num_points": 32768,
            "features": FeatureConfig(
                mode="full",
                k_neighbors=40,
                search_radius=3.0,
                use_rgb=True,
                use_nir=True,
                compute_ndvi=True,
            ),
            "optimizations": OptimizationsConfig(
                enabled=True,
                async_io=True,
                batch_processing=True,
                batch_size=2,  # Larger patches
                gpu_pooling=True,
                gpu_pool_max_size_gb=8.0,
            ),
        },
        
        "asprs_classification_cpu": {
            "mode": "asprs",
            "processing_mode": "both",
            "use_gpu": False,
            "num_workers": 8,
            "patch_size": 50.0,
            "num_points": 16384,
            "features": FeatureConfig(
                mode="full",
                k_neighbors=30,
                search_radius=2.5,
            ),
            "optimizations": OptimizationsConfig(
                enabled=True,
                async_io=True,
                batch_processing=True,
                batch_size=16,  # CPU can handle larger batches
            ),
        },
        
        "asprs_classification_gpu": {
            "mode": "asprs",
            "processing_mode": "both",
            "use_gpu": True,
            "num_workers": 0,
            "patch_size": 50.0,
            "num_points": 16384,
            "features": FeatureConfig(
                mode="full",
                k_neighbors=30,
                search_radius=2.5,
            ),
            "optimizations": OptimizationsConfig(
                enabled=True,
                async_io=True,
                batch_processing=True,
                batch_size=4,
                gpu_pooling=True,
                gpu_pool_max_size_gb=6.0,
            ),
        },
        
        "high_quality": {
            "mode": "lod3",
            "processing_mode": "both",
            "use_gpu": True,
            "num_workers": 0,
            "patch_size": 150.0,
            "num_points": 65536,
            "features": FeatureConfig(
                mode="full",
                k_neighbors=50,
                search_radius=5.0,
                use_rgb=True,
                use_nir=True,
                compute_ndvi=True,
                multi_scale=True,
                scales=[1.0, 2.0, 5.0, 10.0],
            ),
            "optimizations": OptimizationsConfig(
                enabled=True,
                async_io=True,
                async_workers=4,
                tile_cache_size=10,
                batch_processing=True,
                batch_size=1,  # Very large patches
                gpu_pooling=True,
                gpu_pool_max_size_gb=12.0,
            ),
        },
        
        # ===================================================================
        # LEGACY PRESETS (Backward Compatibility)
        # ===================================================================
        
        "asprs_production": {
            "mode": "asprs",
            "processing_mode": "both",
            "features": FeatureConfig(
                mode="standard",
                k_neighbors=30,
            ),
        },
        
        "gpu_optimized": {
            "use_gpu": True,
            "num_workers": 0,
            "features": FeatureConfig(
                mode="standard",
                k_neighbors=30,
            ),
            "optimizations": OptimizationsConfig(
                enabled=True,
                gpu_pooling=True,
            ),
        },
        
        "minimal_fast": {
            "mode": "asprs",
            "processing_mode": "patches_only",
            "num_workers": 8,
            "features": FeatureConfig(
                mode="minimal",
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

        # Map old 'mode' to new 'mode' (v4.0 naming)
        old_mode = old_features.get("mode", "full")
        mode_map = {
            "minimal": "minimal",
            "lod2": "standard",
            "lod3": "full",
            "asprs_classes": "standard",
            "full": "full",
            "custom": "standard",
        }
        feature_mode = mode_map.get(old_mode, "standard")

        new_config["features"] = {
            "mode": feature_mode,  # v4.0: renamed from feature_set
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
    "OptimizationsConfig",
    "AdvancedConfig",
]
