"""
Main LiDAR Processing Class
"""

import gc
import logging
import multiprocessing as mp
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import laspy
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..classification_schema import ASPRS_TO_LOD2, ASPRS_TO_LOD3

# Dataset manager for ML dataset creation with train/val/test splits
from ..datasets.dataset_manager import DatasetConfig, DatasetManager
from ..features.architectural_styles import get_architectural_style_id

# Phase 4.3: New unified orchestrator V5 (consolidated)
from ..features.orchestrator import FeatureOrchestrator
from ..io.metadata import MetadataManager

# Classification module (unified in v3.1.0)
from .classification import UnifiedClassifier, refine_classification_unified

# Import refactored modules from classification package
# Note: FeatureManager has been replaced by FeatureOrchestrator in Phase 4.3
from .classification.config_validator import ConfigValidator

# Phase 3.4: Tile processing modules
from .classification.io import (
    TileLoader,
    save_patch_hdf5,
    save_patch_laz,
    save_patch_multi_format,
    save_patch_npz,
    save_patch_torch,
)
from .classification.patch_extractor import (
    AugmentationConfig,
    PatchConfig,
    extract_and_augment_patches,
    format_patch_for_architecture,
)

# Reclassification module (optimized)
from .classification.reclassifier import (
    OptimizedReclassifier,
    reclassify_tile_optimized,
)
from .gpu_context import disable_gpu_for_multiprocessing

# Optimization factory for intelligent strategy selection
from .optimization_factory import auto_optimize_config, optimization_factory
from .processing_metadata import ProcessingMetadata
from .skip_checker import PatchSkipChecker

# Note: FeatureComputer has been replaced by FeatureOrchestrator in Phase 4.3


# Import from modules (refactored in Phase 3.2)


# Configure logging
logger = logging.getLogger(__name__)

# Processing mode type definition
ProcessingMode = Literal["patches_only", "both", "enriched_only", "reclassify_only"]


class LiDARProcessor:
    """
    Main class for processing IGN LiDAR HD data into ML-ready datasets.
    """

    def __init__(self, config: Union[DictConfig, Dict] = None, **kwargs):
        """
        Initialize processor with config object or individual parameters (backward compatible).

        Args:
            config: Configuration object (DictConfig or dict) containing all settings.
                   If None, will build config from kwargs for backward compatibility.
            **kwargs: Individual parameters (deprecated, use config instead).
                     Supported for backward compatibility with existing code.

        Config Structure (when using config object):
            processor:
                lod_level: 'LOD2' or 'LOD3'
                processing_mode: 'patches_only', 'both', or 'enriched_only'
                augment: Enable data augmentation
                num_augmentations: Number of augmentations per patch
                bbox: Bounding box (xmin, ymin, xmax, ymax)
                patch_size: Patch size in meters (default: 150.0)
                patch_overlap: Overlap ratio (default: 0.1)
                num_points: Target points per patch (default: 16384)
                use_gpu: GPU acceleration for features (default: False)
                use_gpu_chunked: Chunked GPU processing (default: True)
                gpu_batch_size: GPU batch size (default: 1,000,000)
                preprocess: Apply preprocessing (default: False)
                use_stitching: Enable tile stitching (default: False)
                buffer_size: Buffer zone size in meters (default: 10.0)
                architecture: Target DL architecture (default: 'pointnet++')
                output_format: Output format(s), e.g., 'npz', 'hdf5,laz'

            features:
                include_extra_features: Compute extra building features
                feature_mode: Feature mode ('minimal', 'lod2', 'lod3', 'full')
                k_neighbors: Neighbors for feature computation (None = auto)
                include_architectural_style: Include architectural style
                style_encoding: Style encoding ('constant' or 'multihot')
                use_rgb: Add RGB from orthophotos
                rgb_cache_dir: Cache directory for RGB tiles
                use_infrared: Add NIR from LAZ files
                compute_ndvi: Compute NDVI from RGB+NIR

        Legacy kwargs (deprecated):
            All parameters from previous signature supported for backward compatibility.
            See migration guide for transitioning to config-based approach.
        """
        # Handle config initialization
        if config is None:
            # Build config from kwargs for backward compatibility
            config = self._build_config_from_kwargs(kwargs)
            logger.debug(
                "Built config from legacy kwargs (consider migrating to config-based initialization)"
            )
        elif not isinstance(config, (DictConfig, dict)):
            raise TypeError(
                f"config must be DictConfig or dict, got {type(config).__name__}. "
                f"Pass None and use kwargs for legacy parameter-based initialization."
            )

        # Convert dict to DictConfig if needed
        if isinstance(config, dict):
            config = OmegaConf.create(config)

        # Validate configuration
        self._validate_config(config)

        # Apply auto-optimization if enabled
        if OmegaConf.select(config, "processing.auto_optimize", default=False):
            logger.info(
                "🧠 Auto-optimization enabled - analyzing system capabilities..."
            )

            # Get optimization recommendations
            recommendations = optimization_factory.recommend_optimization(
                OmegaConf.to_object(config)
            )

            if recommendations.get("config_updates"):
                logger.info(
                    f"📈 Applying optimization strategy: {recommendations['strategy'].value}"
                )

                # Apply recommended updates
                for key, value in recommendations["config_updates"].items():
                    if key == "architecture":
                        OmegaConf.update(config, "processing.architecture", value)
                    else:
                        OmegaConf.update(config, f"processor.{key}", value)

                # Log performance expectations
                if recommendations.get("estimated_performance"):
                    logger.info(
                        f"⚡ Expected improvement: {recommendations['estimated_performance']}"
                    )

                # Log any warnings
                for warning in recommendations.get("warnings", []):
                    logger.warning(f"⚠️  {warning}")

        # Store config
        self.config = config

        # Extract commonly used values for convenient access
        self.lod_level = config.processor.lod_level
        self.processing_mode = config.processor.processing_mode
        self.patch_size = OmegaConf.select(
            config, "processor.patch_size", default=150.0
        )
        self.num_points = OmegaConf.select(
            config, "processor.num_points", default=16384
        )
        self.architecture = OmegaConf.select(
            config, "processor.architecture", default="pointnet++"
        )
        self.output_format = config.processor.output_format
        self.save_patches = OmegaConf.select(
            config, "output.save_patches", default=False
        )

        # Derive save/only flags from processing mode for internal use
        self.save_enriched_laz = self.processing_mode in ["both", "enriched_only"]
        self.only_enriched_laz = self.processing_mode == "enriched_only"

        logger.info(f"✨ Processing mode: {self.processing_mode}")
        if not self.save_patches:
            logger.info(f"   📦 Patch generation: DISABLED (save_patches=false)")
        logger.info(f"Initialized LiDARProcessor with {self.lod_level}")

        # Validate output format using ConfigValidator
        validated_formats = ConfigValidator.validate_output_format(self.output_format)
        logger.debug(f"Validated output formats: {validated_formats}")

        # Validate processing mode
        ConfigValidator.validate_processing_mode(self.processing_mode)

        # Phase 4.3: Initialize unified feature orchestrator V5 (consolidated)
        # All optimizations are now built into the main FeatureOrchestrator
        logger.info("🚀 Using FeatureOrchestrator V5 with integrated optimizations")
        self.feature_orchestrator = FeatureOrchestrator(config)

        # Keep backward-compatible references for legacy code
        self.feature_manager = self.feature_orchestrator  # Backward compatibility alias

        # Setup stitching configuration and initialize stitcher if needed
        # Handle both old (processor.use_stitching) and new (stitching.enabled) config structures
        if hasattr(config, "stitching") and hasattr(config.stitching, "enabled"):
            use_stitching = config.stitching.enabled
            buffer_size = getattr(config.stitching, "buffer_size", 10.0)
        else:
            use_stitching = config.processor.get("use_stitching", False)
            buffer_size = config.processor.get("buffer_size", 10.0)

        stitching_config = ConfigValidator.setup_stitching_config(
            use_stitching, buffer_size, config.processor.get("stitching_config", None)
        )
        self.stitcher = ConfigValidator.init_stitcher(stitching_config)

        # Set class mapping based on LOD level
        if self.lod_level == "ASPRS":
            # ASPRS mode: No class mapping - use ASPRS codes directly
            self.class_mapping = None
            self.default_class = 1  # ASPRS unclassified
        elif self.lod_level == "LOD2":
            self.class_mapping = ASPRS_TO_LOD2
            self.default_class = 14
        else:  # LOD3
            self.class_mapping = ASPRS_TO_LOD3
            self.default_class = 29

        # Initialize intelligent skip checker
        self.skip_checker = PatchSkipChecker(
            output_format=self.output_format,
            architecture=self.architecture,
            num_augmentations=config.processor.get("num_augmentations", 3),
            augment=config.processor.get("augment", False),
            validate_content=True,  # Enable content validation
            min_file_size=1024,  # 1KB minimum
            only_enriched_laz=self.only_enriched_laz,
        )

        # Initialize dataset manager for ML dataset creation (with train/val/test splits)
        self.dataset_manager = None
        if config.get("dataset", {}).get("enabled", False):
            dataset_config = DatasetConfig(
                train_ratio=config.dataset.get("train_ratio", 0.7),
                val_ratio=config.dataset.get("val_ratio", 0.15),
                test_ratio=config.dataset.get("test_ratio", 0.15),
                random_seed=config.dataset.get("random_seed", 42),
                split_by_tile=config.dataset.get("split_by_tile", True),
                create_split_dirs=config.dataset.get("create_split_dirs", True),
                patch_sizes=config.dataset.get("patch_sizes", [int(self.patch_size)]),
                balance_across_sizes=config.dataset.get("balance_across_sizes", False),
            )
            # Dataset manager will be initialized in process_directory with output_dir
            self._dataset_config = dataset_config
            logger.info("📊 Dataset mode enabled - will create train/val/test splits")
        else:
            self._dataset_config = None

        # Phase 4.3: Initialize tile processing modules
        self.tile_loader = TileLoader(self.config)

        # Initialize data fetcher for ground truth (BD TOPO, BD Forêt, RPG, Cadastre)
        self.data_fetcher = None
        logger.info(f"🔍 Checking data sources configuration...")
        logger.info(f"🔍 Available config keys: {list(config.keys())}")

        # Debug: Check if data_sources exists
        logger.debug(f"🔍 DEBUG: 'data_sources' in config = {'data_sources' in config}")
        if "data_sources" in config:
            logger.info(f"🔍 data_sources keys: {list(config.data_sources.keys())}")
            logger.debug(
                f"🔍 DEBUG: config.data_sources keys = {list(config.data_sources.keys())}"
            )
            if "bd_topo" in config.data_sources:
                logger.info(
                    f"🔍 bd_topo.enabled = {config.data_sources.bd_topo.enabled}"
                )
                logger.debug(
                    f"🔍 DEBUG: config.data_sources.bd_topo.enabled = {config.data_sources.bd_topo.enabled}"
                )
        else:
            logger.warning(
                f"⚠️  'data_sources' not found in config! This means ground truth data won't be loaded."
            )

        # Use OmegaConf.select() for compatibility with Hydra configs
        # Support both flat (v4.0) and nested (v5.0) structure
        # Check nested structure FIRST (V5 structure), then fall back to flat structure (V4)
        bd_topo_enabled = OmegaConf.select(
            config, "data_sources.bd_topo.enabled", default=False
        )

        if bd_topo_enabled:
            # Extract from nested structure (V5)
            bd_topo_buildings = OmegaConf.select(
                config, "data_sources.bd_topo.features.buildings", default=False
            )
            bd_topo_roads = OmegaConf.select(
                config, "data_sources.bd_topo.features.roads", default=False
            )
            bd_topo_water = OmegaConf.select(
                config, "data_sources.bd_topo.features.water", default=False
            )
            bd_topo_vegetation = OmegaConf.select(
                config, "data_sources.bd_topo.features.vegetation", default=False
            )
            # Bridges and power_lines might be in nested OR flat structure
            bd_topo_bridges = OmegaConf.select(
                config,
                "data_sources.bd_topo.features.bridges",
                default=OmegaConf.select(
                    config, "data_sources.bd_topo_bridges", default=False
                ),
            )
            bd_topo_power_lines = OmegaConf.select(
                config,
                "data_sources.bd_topo.features.power_lines",
                default=OmegaConf.select(
                    config, "data_sources.bd_topo_power_lines", default=False
                ),
            )

            logger.info(f"🔍 DEBUG: Extracted from NESTED structure (V5):")
            logger.info(
                f"   buildings={bd_topo_buildings}, roads={bd_topo_roads}, water={bd_topo_water}"
            )
            logger.info(
                f"   vegetation={bd_topo_vegetation}, bridges={bd_topo_bridges}, power_lines={bd_topo_power_lines}"
            )
        else:
            # Fall back to flat structure (V4)
            bd_topo_buildings = OmegaConf.select(
                config, "data_sources.bd_topo_buildings", default=False
            )
            bd_topo_roads = OmegaConf.select(
                config, "data_sources.bd_topo_roads", default=False
            )
            bd_topo_water = OmegaConf.select(
                config, "data_sources.bd_topo_water", default=False
            )
            bd_topo_vegetation = OmegaConf.select(
                config, "data_sources.bd_topo_vegetation", default=False
            )
            bd_topo_bridges = OmegaConf.select(
                config, "data_sources.bd_topo_bridges", default=False
            )
            bd_topo_power_lines = OmegaConf.select(
                config, "data_sources.bd_topo_power_lines", default=False
            )

            # Check if ANY feature is enabled
            bd_topo_enabled = any(
                [
                    bd_topo_buildings,
                    bd_topo_roads,
                    bd_topo_water,
                    bd_topo_vegetation,
                    bd_topo_bridges,
                    bd_topo_power_lines,
                ]
            )

            if bd_topo_enabled:
                logger.info(f"🔍 DEBUG: Extracted from FLAT structure (V4):")
                logger.info(
                    f"   buildings={bd_topo_buildings}, roads={bd_topo_roads}, water={bd_topo_water}"
                )
                logger.info(
                    f"   vegetation={bd_topo_vegetation}, bridges={bd_topo_bridges}, power_lines={bd_topo_power_lines}"
                )

        bd_foret_enabled = OmegaConf.select(
            config, "data_sources.bd_foret_enabled", default=False
        )
        rpg_enabled = OmegaConf.select(
            config, "data_sources.rpg_enabled", default=False
        )
        cadastre_enabled = OmegaConf.select(
            config, "data_sources.cadastre_enabled", default=False
        )

        logger.debug(
            f"🔍 DEBUG: bd_topo_enabled = {bd_topo_enabled} (type: {type(bd_topo_enabled)})"
        )
        logger.debug(
            f"🔍 DEBUG: bd_foret_enabled = {bd_foret_enabled} (type: {type(bd_foret_enabled)})"
        )
        logger.debug(
            f"🔍 DEBUG: rpg_enabled = {rpg_enabled} (type: {type(rpg_enabled)})"
        )
        logger.debug(
            f"🔍 DEBUG: cadastre_enabled = {cadastre_enabled} (type: {type(cadastre_enabled)})"
        )

        logger.info(
            f"   - BD TOPO®: {'✅ Enabled' if bd_topo_enabled else '❌ Disabled'}"
        )
        logger.info(
            f"   - BD Forêt®: {'✅ Enabled' if bd_foret_enabled else '❌ Disabled'}"
        )
        logger.info(f"   - RPG: {'✅ Enabled' if rpg_enabled else '❌ Disabled'}")
        logger.info(
            f"   - Cadastre: {'✅ Enabled' if cadastre_enabled else '❌ Disabled'}"
        )

        if bd_topo_enabled or bd_foret_enabled or rpg_enabled or cadastre_enabled:
            try:
                from ..io.data_fetcher import DataFetchConfig, DataFetcher

                # Build data source configuration
                cache_dir = Path(
                    OmegaConf.select(config, "cache_dir", default="data/cache")
                )

                # Extract BD TOPO features using the flat configuration structure (v4.0)
                # Create configuration with ALL BD TOPO features and parameters
                fetch_config = DataFetchConfig(
                    # BD TOPO® features - use flat structure
                    include_buildings=bd_topo_buildings,
                    include_roads=bd_topo_roads,
                    include_railways=OmegaConf.select(
                        config, "data_sources.bd_topo_railways", default=False
                    ),
                    include_water=bd_topo_water,
                    include_vegetation=bd_topo_vegetation,
                    include_bridges=bd_topo_bridges,
                    include_parking=OmegaConf.select(
                        config, "data_sources.bd_topo_parking", default=False
                    ),
                    include_cemeteries=OmegaConf.select(
                        config, "data_sources.bd_topo_cemeteries", default=False
                    ),
                    include_power_lines=bd_topo_power_lines,
                    include_sports=OmegaConf.select(
                        config, "data_sources.bd_topo_sports", default=False
                    ),
                    # Other data sources
                    include_forest=bd_foret_enabled,
                    include_agriculture=rpg_enabled,
                    include_cadastre=cadastre_enabled,
                    group_by_parcel=OmegaConf.select(
                        config, "data_sources.cadastre_group_by_parcel", default=False
                    ),
                    # Buffer parameters - check both flat and nested structures
                    road_width_fallback=OmegaConf.select(
                        config,
                        "data_sources.bd_topo_road_width_fallback",
                        default=OmegaConf.select(
                            config,
                            "data_sources.bd_topo.parameters.road_width_fallback",
                            default=4.0,
                        ),
                    ),
                    railway_width_fallback=OmegaConf.select(
                        config,
                        "data_sources.bd_topo_railway_width_fallback",
                        default=OmegaConf.select(
                            config,
                            "data_sources.bd_topo.parameters.railway_width_fallback",
                            default=3.5,
                        ),
                    ),
                    power_line_buffer=OmegaConf.select(
                        config,
                        "data_sources.bd_topo_power_line_buffer",
                        default=OmegaConf.select(
                            config,
                            "data_sources.bd_topo.parameters.power_line_buffer",
                            default=2.0,
                        ),
                    ),
                    # RPG year
                    rpg_year=OmegaConf.select(
                        config,
                        "data_sources.rpg_year",
                        default=OmegaConf.select(
                            config, "data_sources.rpg.year", default=2024
                        ),
                    ),
                )

                # Initialize data fetcher
                self.data_fetcher = DataFetcher(
                    cache_dir=cache_dir, config=fetch_config
                )
                logger.info("✅ Data fetcher initialized successfully")
                logger.info(f"   Cache directory: {cache_dir}")
                logger.info("")

                # Log enabled data sources and features
                enabled_sources = []
                if bd_topo_enabled:
                    enabled_features = []
                    if bd_topo_roads:
                        enabled_features.append("roads")
                    if OmegaConf.select(
                        config, "data_sources.bd_topo_railways", default=False
                    ):
                        enabled_features.append("railways")
                    if bd_topo_buildings:
                        enabled_features.append("buildings")
                    if OmegaConf.select(
                        config, "data_sources.bd_topo_cemeteries", default=False
                    ):
                        enabled_features.append("cemeteries")
                    if bd_topo_power_lines:
                        enabled_features.append("power_lines")
                    if OmegaConf.select(
                        config, "data_sources.bd_topo_sports", default=False
                    ):
                        enabled_features.append("sports")
                    if OmegaConf.select(
                        config, "data_sources.bd_topo_parking", default=False
                    ):
                        enabled_features.append("parking")
                    if bd_topo_bridges:
                        enabled_features.append("bridges")
                    if bd_topo_water:
                        enabled_features.append("water")
                    if bd_topo_vegetation:
                        enabled_features.append("vegetation")
                    if enabled_features:
                        enabled_sources.append(
                            f"BD TOPO ({', '.join(enabled_features)})"
                        )
                if bd_foret_enabled:
                    enabled_sources.append("BD Forêt")
                if rpg_enabled:
                    enabled_sources.append("RPG")
                if cadastre_enabled:
                    enabled_sources.append("Cadastre")

                if enabled_sources:
                    logger.info(
                        f"   📦 Enabled data sources: {', '.join(enabled_sources)}"
                    )

            except ImportError as e:
                logger.error(f"❌ Could not initialize data fetcher: {e}")
                logger.error(f"   Ground truth classification will NOT be applied")
                logger.error(
                    f"   Install required packages: pip install geopandas shapely"
                )
                self.data_fetcher = None
        else:
            logger.info(
                f"ℹ️  No data sources enabled - ground truth classification disabled"
            )

    def _validate_config(self, config: DictConfig) -> None:
        """Validate configuration object has required fields."""
        required_sections = ["processor", "features"]
        for section in required_sections:
            if section not in config:
                raise ValueError(
                    f"Config missing required section: '{section}'. "
                    f"Available sections: {list(config.keys())}"
                )

        required_processor_fields = ["lod_level", "processing_mode", "output_format"]
        for field in required_processor_fields:
            if field not in config.processor:
                raise ValueError(
                    f"Config.processor missing required field: '{field}'. "
                    f"Available fields: {list(config.processor.keys())}"
                )

    def _build_config_from_kwargs(self, kwargs: Dict[str, Any]) -> DictConfig:
        """
        Build a config object from legacy kwargs for backward compatibility.

        Args:
            kwargs: Dictionary of legacy parameter names and values

        Returns:
            DictConfig object with processor and features sections
        """
        # Determine processing_mode - old flags take precedence for backward compatibility
        save_enriched = kwargs.get("save_enriched_laz")
        only_enriched = kwargs.get("only_enriched_laz")

        # If old flags are explicitly provided, they override processing_mode
        if save_enriched is not None or only_enriched is not None:
            # Infer from legacy flags (backward compatibility priority)
            save_enriched = save_enriched if save_enriched is not None else False
            only_enriched = only_enriched if only_enriched is not None else False

            if only_enriched:
                processing_mode = "enriched_only"
            elif save_enriched:
                processing_mode = "both"
            else:
                processing_mode = "patches_only"
        else:
            # No old flags provided, use explicit processing_mode or default
            processing_mode = kwargs.get("processing_mode", "patches_only")

        # Create config structure with defaults
        config_dict = {
            "processor": {
                "lod_level": kwargs.get("lod_level", "LOD2"),
                "processing_mode": processing_mode,
                "augment": kwargs.get("augment", False),
                "num_augmentations": kwargs.get("num_augmentations", 3),
                "bbox": kwargs.get("bbox", None),
                "patch_size": kwargs.get("patch_size", 150.0),
                "patch_overlap": kwargs.get("patch_overlap", 0.1),
                "num_points": kwargs.get("num_points", 16384),
                "use_gpu": kwargs.get("use_gpu", False),
                "use_gpu_chunked": kwargs.get("use_gpu_chunked", True),
                "gpu_batch_size": kwargs.get("gpu_batch_size", 1_000_000),
                "preprocess": kwargs.get("preprocess", False),
                "preprocess_config": kwargs.get("preprocess_config", None),
                "use_stitching": kwargs.get("use_stitching", False),
                "buffer_size": kwargs.get("buffer_size", 10.0),
                "stitching_config": kwargs.get("stitching_config", None),
                "architecture": kwargs.get("architecture", "pointnet++"),
                "output_format": kwargs.get("output_format", "npz"),
            },
            "features": {
                "include_extra_features": kwargs.get("include_extra_features", False),
                "feature_mode": kwargs.get("feature_mode", None),
                "k_neighbors": kwargs.get("k_neighbors", None),
                "include_architectural_style": kwargs.get(
                    "include_architectural_style", False
                ),
                "style_encoding": kwargs.get("style_encoding", "constant"),
                "use_rgb": kwargs.get("include_rgb", False),
                "rgb_cache_dir": kwargs.get("rgb_cache_dir", None),
                "use_infrared": kwargs.get("include_infrared", False),
                "compute_ndvi": kwargs.get("compute_ndvi", False),
            },
        }

        return OmegaConf.create(config_dict)

    # Backward compatibility properties
    @property
    def rgb_fetcher(self):
        """Access RGB fetcher (backward compatibility)."""
        return self.feature_orchestrator.rgb_fetcher

    @property
    def infrared_fetcher(self):
        """Access infrared fetcher (backward compatibility)."""
        return self.feature_orchestrator.infrared_fetcher

    @property
    def use_gpu(self):
        """Check if GPU is enabled (backward compatibility)."""
        return self.feature_orchestrator.use_gpu

    @property
    def include_rgb(self):
        """Check if RGB is enabled (backward compatibility)."""
        return self.config.features.use_rgb

    @property
    def include_infrared(self):
        """Check if infrared is enabled (backward compatibility)."""
        return self.config.features.use_infrared

    @property
    def compute_ndvi(self):
        """Check if NDVI computation is enabled (backward compatibility)."""
        return self.config.features.compute_ndvi

    @property
    def include_extra_features(self):
        """Check if extra features are enabled (backward compatibility)."""
        # Support both include_extra and include_extra_features for backward compatibility
        if hasattr(self.config.features, "include_extra"):
            return self.config.features.include_extra
        return self.config.features.get("include_extra_features", False)

    @property
    def k_neighbors(self):
        """Get k neighbors value (backward compatibility)."""
        return self.config.features.k_neighbors

    @property
    def feature_mode(self):
        """Get feature mode (backward compatibility)."""
        return self.config.features.mode

    @property
    def include_architectural_style(self):
        """Check if architectural style is enabled (backward compatibility)."""
        return self.config.features.get("include_architectural_style", False)

    @property
    def style_encoding(self):
        """Get style encoding method (backward compatibility)."""
        return self.config.features.style_encoding

    @property
    def augment(self):
        """Check if augmentation is enabled (backward compatibility)."""
        return self.config.processor.get("augment", False)

    @property
    def num_augmentations(self):
        """Get number of augmentations (backward compatibility)."""
        return self.config.processor.get("num_augmentations", 0)

    @property
    def bbox(self):
        """Get bounding box (backward compatibility)."""
        # bbox is at root level, not in processor section
        return self.config.get("bbox")

    @property
    def patch_overlap(self):
        """Get patch overlap (backward compatibility)."""
        return self.config.processor.get("patch_overlap", 0.0)

    @property
    def use_gpu_chunked(self):
        """Check if chunked GPU processing is enabled (backward compatibility)."""
        return self.config.processor.use_gpu_chunked

    @property
    def gpu_batch_size(self):
        """Get GPU batch size (backward compatibility)."""
        return self.config.processor.gpu_batch_size

    @property
    def preprocess(self):
        """Check if preprocessing is enabled (backward compatibility)."""
        return self.config.processor.preprocess

    @property
    def preprocess_config(self):
        """Get preprocessing config (backward compatibility)."""
        return self.config.processor.preprocess_config

    @property
    def use_stitching(self):
        """Check if stitching is enabled (backward compatibility)."""
        # Handle both old (processor.use_stitching) and new (stitching.enabled) config structures
        if hasattr(self.config, "stitching") and hasattr(
            self.config.stitching, "enabled"
        ):
            return self.config.stitching.enabled
        else:
            return getattr(self.config.processor, "use_stitching", False)

    @property
    def buffer_size(self):
        """Get buffer size (backward compatibility)."""
        # Handle both old (processor.buffer_size) and new (stitching.buffer_size) config structures
        if hasattr(self.config, "stitching") and hasattr(
            self.config.stitching, "buffer_size"
        ):
            return self.config.stitching.buffer_size
        else:
            return getattr(self.config.processor, "buffer_size", 10.0)

    @property
    def rgb_cache_dir(self):
        """Get RGB cache directory (backward compatibility)."""
        return self.config.features.rgb_cache_dir

    def _save_patch_as_laz(
        self,
        save_path: Path,
        arch_data: Dict[str, np.ndarray],
        original_patch: Dict[str, np.ndarray],
    ) -> None:
        """
        Save a patch as a LAZ point cloud file with all computed features.

        Args:
            save_path: Output LAZ file path
            arch_data: Formatted patch data (architecture-specific)
            original_patch: Original patch data with metadata
        """
        # Extract coordinates from arch_data
        # Most architectures have 'points' or 'coords' key
        if "points" in arch_data:
            coords = arch_data["points"][:, :3].copy()  # XYZ
        elif "coords" in arch_data:
            coords = arch_data["coords"][:, :3].copy()
        else:
            logger.warning(
                f"Cannot save LAZ patch: no coordinates found in {list(arch_data.keys())}"
            )
            return

        # Restore LAMB93 coordinates if metadata available
        # Extract tile coordinates from filename if possible
        try:
            filename = save_path.stem
            parts = filename.split("_")
            # Look for tile coordinates in filename (e.g., LHD_FXX_0649_6863_...)
            if len(parts) >= 4 and parts[2].isdigit() and parts[3].isdigit():
                tile_x = int(parts[2])
                tile_y = int(parts[3])
                # LAMB93 tiles are 1km x 1km, tile center at (tile_x * 1000 + 500, tile_y * 1000 + 500)
                tile_center_x = tile_x * 1000 + 500
                tile_center_y = tile_y * 1000 + 500

                # If metadata has centroid, restore original coordinates
                if "metadata" in original_patch and isinstance(
                    original_patch["metadata"], dict
                ):
                    metadata = original_patch["metadata"]
                    if "centroid" in metadata:
                        centroid = np.array(metadata["centroid"])
                        coords[:, 0] = coords[:, 0] + centroid[0] + tile_center_x
                        coords[:, 1] = coords[:, 1] + centroid[1] + tile_center_y
                        coords[:, 2] = coords[:, 2] + centroid[2]
                    else:
                        # No centroid, just apply tile offset
                        coords[:, 0] = coords[:, 0] + tile_center_x
                        coords[:, 1] = coords[:, 1] + tile_center_y
        except Exception as e:
            logger.debug(f"Could not restore LAMB93 coordinates: {e}")
            # Continue with normalized coordinates

        # Determine point format based on available data
        # Format 3: supports RGB (LAS 1.4) - most compatible
        # Format 8: supports RGB + NIR (LAS 1.4)
        has_rgb = "rgb" in arch_data
        has_nir = "nir" in original_patch and original_patch["nir"] is not None
        point_format = 8 if (has_rgb and has_nir) else (3 if has_rgb else 6)

        # Create LAZ file
        header = laspy.LasHeader(version="1.4", point_format=point_format)
        header.offsets = [
            np.floor(coords[:, 0].min()),
            np.floor(coords[:, 1].min()),
            np.floor(coords[:, 2].min()),
        ]
        header.scales = [0.001, 0.001, 0.001]

        las = laspy.LasData(header)
        las.x = coords[:, 0]
        las.y = coords[:, 1]
        las.z = coords[:, 2]

        # Add classification if available
        if "labels" in arch_data:
            las.classification = arch_data["labels"].astype(np.uint8)
        elif "classification" in original_patch:
            las.classification = original_patch["classification"].astype(np.uint8)

        # Add intensity if available
        if "intensity" in original_patch:
            las.intensity = (original_patch["intensity"] * 65535).astype(np.uint16)

        # Add RGB if available
        if "rgb" in arch_data:
            rgb = (arch_data["rgb"] * 65535).astype(np.uint16)
            las.red = rgb[:, 0]
            las.green = rgb[:, 1]
            las.blue = rgb[:, 2]

        # Add NIR if available and point format supports it (format 8)
        if (
            point_format == 8
            and "nir" in original_patch
            and original_patch["nir"] is not None
        ):
            nir = (original_patch["nir"] * 65535).astype(np.uint16)
            las.nir = nir

        # ===== Add computed features as extra dimensions =====
        # Add ALL geometric/geometric features dynamically (not just a hardcoded subset)
        # This ensures all computed features (including advanced ones like eigenvalues,
        # architectural features, density features) are saved to LAZ

        # Track added dimensions to avoid duplicates
        added_dimensions = set()

        # Define keys to skip (standard fields, special handling, or already processed)
        skip_keys = {
            "points",
            "coords",
            "labels",
            "classification",
            "intensity",
            "rgb",
            "nir",
            "ndvi",
            "normals",
            "return_number",
            "metadata",
            "features",
            "knn_graph",
            "voxel_coords",
            "voxel_features",
            "voxel_labels",
            "_patch_center",
            "_patch_bounds",
            "_version",
            "_patch_idx",
            "_spatial_idx",
        }

        # Iterate through all features in original_patch and add as extra dimensions
        for feat_name, feat_data in original_patch.items():
            # Skip if in skip list or not a numpy array
            if feat_name in skip_keys or not isinstance(feat_data, np.ndarray):
                continue

            # Skip if already added
            if feat_name in added_dimensions:
                continue

            # Skip if already a standard LAS field
            if hasattr(las, feat_name) and feat_name not in ["height", "curvature"]:
                continue

            # Skip multi-dimensional arrays (except for specific known ones handled separately)
            if feat_data.ndim > 1:
                continue

            try:
                # Determine appropriate data type
                # Keep cluster IDs as integers for semantic meaning
                if feat_name in [
                    "cluster_id",
                    "building_cluster_id",
                    "parcel_cluster_id",
                ]:
                    dtype = np.int32
                    feat_data_typed = feat_data.astype(np.int32)
                elif feat_data.dtype in [np.float32, np.float64]:
                    dtype = np.float32
                    feat_data_typed = feat_data.astype(np.float32)
                elif feat_data.dtype in [np.int32, np.uint32]:
                    dtype = np.int32
                    feat_data_typed = feat_data.astype(np.int32)
                elif feat_data.dtype == np.uint8:
                    dtype = np.uint8
                    feat_data_typed = feat_data.astype(np.uint8)
                else:
                    dtype = np.float32
                    feat_data_typed = feat_data.astype(np.float32)

                # Add extra dimension (description max 31 chars)
                desc = f"Feature: {feat_name}"[:31]
                las.add_extra_dim(
                    laspy.ExtraBytesParams(name=feat_name, type=dtype, description=desc)
                )
                setattr(las, feat_name, feat_data_typed)
                added_dimensions.add(feat_name)
            except Exception as e:
                logger.debug(
                    f"  ⚠️  Could not add feature " f"'{feat_name}' to LAZ: {e}"
                )

        # Normals (normal_x, normal_y, normal_z for consistency)
        if "normals" in original_patch:
            try:
                normals = original_patch["normals"]
                for i, comp in enumerate(["normal_x", "normal_y", "normal_z"]):
                    if comp not in added_dimensions:
                        las.add_extra_dim(
                            laspy.ExtraBytesParams(
                                name=comp,
                                type=np.float32,
                                description=f"Normal {comp[-1]}",
                            )
                        )
                        setattr(las, comp, normals[:, i].astype(np.float32))
                        added_dimensions.add(comp)
            except Exception as e:
                logger.warning(f"  ⚠️  Could not add normals to LAZ: {e}")

        # Height features
        height_features = ["height", "z_normalized", "z_from_ground", "z_from_median"]
        for feat_name in height_features:
            if feat_name in original_patch and feat_name not in added_dimensions:
                try:
                    # Truncate description to 31 chars max (LAZ limit - needs null terminator)
                    desc = f"Height: {feat_name}"[:31]
                    las.add_extra_dim(
                        laspy.ExtraBytesParams(
                            name=feat_name, type=np.float32, description=desc
                        )
                    )
                    setattr(
                        las, feat_name, original_patch[feat_name].astype(np.float32)
                    )
                    added_dimensions.add(feat_name)
                except Exception as e:
                    logger.warning(
                        f"  ⚠️  Could not add feature '{feat_name}' to LAZ: {e}"
                    )

        # Radiometric features (NIR, NDVI, etc.)
        # Add NIR as extra dimension if not already included in point format
        if (
            point_format != 8
            and "nir" in original_patch
            and original_patch["nir"] is not None
        ):
            if "nir" not in added_dimensions:
                try:
                    las.add_extra_dim(
                        laspy.ExtraBytesParams(
                            name="nir",
                            type=np.float32,
                            description="NIR reflectance (norm 0-1)",
                        )
                    )
                    las.nir = original_patch["nir"].astype(np.float32)
                    added_dimensions.add("nir")
                except Exception as e:
                    logger.warning(f"  ⚠️  Could not add NIR to LAZ: {e}")

        if "ndvi" in original_patch and original_patch["ndvi"] is not None:
            if "ndvi" not in added_dimensions:
                try:
                    las.add_extra_dim(
                        laspy.ExtraBytesParams(
                            name="ndvi",
                            type=np.float32,
                            description="NDVI (vegetation index)",
                        )
                    )
                    las.ndvi = original_patch["ndvi"].astype(np.float32)
                    added_dimensions.add("ndvi")
                except Exception as e:
                    logger.warning(f"  ⚠️  Could not add NDVI to LAZ: {e}")

        # Return number (if not already in standard fields)
        if (
            "return_number" in original_patch
            and "return_number" not in added_dimensions
        ):
            try:
                # Return number might already be a standard field
                if not hasattr(las, "return_number"):
                    las.add_extra_dim(
                        laspy.ExtraBytesParams(
                            name="return_number",
                            type=np.uint8,
                            description="Return number",
                        )
                    )
                    las.return_number = original_patch["return_number"].astype(np.uint8)
                    added_dimensions.add("return_number")
            except Exception as e:
                logger.warning(f"  ⚠️  Could not add return_number to LAZ: {e}")

        # Write LAZ file
        las.write(str(save_path))

    def _augment_ground_with_dtm(
        self,
        points: np.ndarray,
        classification: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment ground points using RGE ALTI DTM (UPGRADED V3.1).

        This function uses the new comprehensive DTM augmentation module to add
        synthetic ground points intelligently in areas where they are most needed:
        - Under vegetation (CRITICAL for accurate height computation)
        - Under buildings (ground-level reference)
        - Coverage gaps (improve terrain coverage)

        The new system provides:
        - Intelligent area prioritization (vegetation > buildings > gaps)
        - Better validation (height consistency, spatial filtering)
        - Detailed statistics and logging
        - Building polygon integration for targeted augmentation

        Args:
            points: Original point cloud [N, 3] (X, Y, Z)
            classification: Point classifications [N]
            bbox: Bounding box (minx, miny, maxx, maxy)

        Returns:
            Tuple of (augmented_points, augmented_classification)
        """
        try:
            from ..io.rge_alti_fetcher import RGEALTIFetcher
            from .classification.dtm_augmentation import (
                AugmentationStrategy,
                DTMAugmentationConfig,
                DTMAugmenter,
            )

            # Get DTM configuration
            dtm_config = OmegaConf.select(
                self.config, "data_sources.rge_alti", default={}
            )
            augment_config = OmegaConf.select(
                self.config, "ground_truth.rge_alti", default={}
            )

            # Initialize RGE ALTI fetcher
            cache_dir = self.config.get("cache_dir")
            if cache_dir:
                cache_dir = Path(cache_dir) / "rge_alti"

            fetcher = RGEALTIFetcher(
                cache_dir=str(cache_dir) if cache_dir else None,
                resolution=dtm_config.get("resolution", 1.0),
                use_wms=dtm_config.get("use_wcs", True),  # Map use_wcs to use_wms
                api_key=dtm_config.get("api_key", "pratique"),
                prefer_lidar_hd=dtm_config.get(
                    "prefer_lidar_hd", True
                ),  # Use LiDAR HD MNT by default
            )

            # Map strategy string to enum
            strategy_map = {
                "full": AugmentationStrategy.FULL,
                "gaps": AugmentationStrategy.GAPS,
                "intelligent": AugmentationStrategy.INTELLIGENT,
            }
            strategy_name = augment_config.get("augmentation_strategy", "intelligent")
            strategy = strategy_map.get(strategy_name, AugmentationStrategy.INTELLIGENT)

            # Get priority areas configuration
            priority_config = augment_config.get("augmentation_priority", {})

            # Build augmentation configuration from config file
            aug_config = DTMAugmentationConfig(
                enabled=True,
                strategy=strategy,
                spacing=augment_config.get("augmentation_spacing", 2.0),
                min_spacing_to_existing=augment_config.get(
                    "min_spacing_to_existing", 1.5
                ),
                augment_vegetation=priority_config.get("vegetation", True),
                augment_buildings=priority_config.get("buildings", True),
                augment_water=priority_config.get("water", False),
                augment_roads=priority_config.get("roads", False),
                augment_gaps=priority_config.get("gaps", True),
                max_height_difference=augment_config.get("max_height_difference", 5.0),
                validate_against_neighbors=augment_config.get(
                    "validate_against_neighbors", True
                ),
                min_neighbors_for_validation=augment_config.get(
                    "min_neighbors_for_validation", 3
                ),
                neighbor_search_radius=augment_config.get(
                    "neighbor_search_radius", 10.0
                ),
                synthetic_ground_class=augment_config.get("synthetic_ground_class", 2),
                mark_as_synthetic=augment_config.get("mark_as_synthetic", True),
                verbose=True,
            )

            # Get building polygons if available (for targeted augmentation under buildings)
            building_polygons = None
            if self.data_fetcher is not None:
                try:
                    # Try to fetch building polygons for this bbox
                    minx, miny, maxx, maxy = bbox
                    building_gdf = self.data_fetcher._fetch_bd_topo_buildings(
                        minx, miny, maxx, maxy
                    )
                    if building_gdf is not None and len(building_gdf) > 0:
                        building_polygons = building_gdf
                        logger.debug(
                            f"      Using {len(building_polygons)} building polygons for targeted augmentation"
                        )
                except Exception as e:
                    logger.debug(f"      Could not fetch building polygons: {e}")

            # Create augmenter and run augmentation
            augmenter = DTMAugmenter(config=aug_config)

            augmented_points, augmented_labels, augmentation_attrs = (
                augmenter.augment_point_cloud(
                    points=points,
                    labels=classification,
                    dtm_fetcher=fetcher,
                    bbox=bbox,
                    building_polygons=building_polygons,
                    crs="EPSG:2154",
                )
            )

            # Check if any points were added
            n_added = len(augmented_points) - len(points)
            if n_added > 0:
                # Store augmentation attributes for output if requested
                if augment_config.get("save_augmentation_report", True):
                    self._store_augmentation_stats(augmentation_attrs, n_added)

                return augmented_points, augmented_labels
            else:
                logger.info(
                    f"      ℹ️  No ground points added (sufficient existing coverage)"
                )
                return points, classification

        except ImportError as e:
            logger.error(f"      ❌ DTM augmentation module not available: {e}")
            logger.error(
                f"         Install required packages: pip install rasterio requests scipy geopandas"
            )
            return points, classification
        except Exception as e:
            logger.error(f"      ❌ Ground augmentation failed: {e}")
            logger.debug(f"      Exception details:", exc_info=True)
            return points, classification

    def _store_augmentation_stats(self, augmentation_attrs: dict, n_added: int):
        """Store DTM augmentation statistics for later reporting."""
        if not hasattr(self, "_augmentation_stats"):
            self._augmentation_stats = []

        # Extract area distribution
        area_labels = augmentation_attrs.get("augmentation_area", [])
        if len(area_labels) > 0:
            from .classification.dtm_augmentation import AugmentationArea

            # Count points per area
            area_counts = {}
            for area_val in area_labels:
                area_name = (
                    area_val
                    if isinstance(area_val, str)
                    else AugmentationArea(area_val).name
                )
                area_counts[area_name] = area_counts.get(area_name, 0) + 1

            self._augmentation_stats.append(
                {"total_added": n_added, "area_distribution": area_counts}
            )

    def _redownload_tile(self, laz_file: Path) -> bool:
        """
        Attempt to re-download a corrupted tile from IGN WFS.

        Args:
            laz_file: Path to the corrupted LAZ file

        Returns:
            True if re-download succeeded, False otherwise
        """
        try:
            import shutil

            from ..downloader import IGNLiDARDownloader

            # Get filename
            filename = laz_file.name

            logger.info(f"  🌐 Re-downloading {filename} from IGN WFS...")

            # Backup corrupted file
            backup_path = laz_file.with_suffix(".laz.corrupted")
            if laz_file.exists():
                shutil.move(str(laz_file), str(backup_path))
                logger.debug(f"  Backed up corrupted file to {backup_path.name}")

            # Initialize downloader with output directory
            downloader = IGNLiDARDownloader(output_dir=laz_file.parent)

            # Download tile (force re-download, don't skip)
            success, was_skipped = downloader.download_tile(
                filename=filename, force=True, skip_existing=False
            )

            if success and laz_file.exists():
                # Verify the download
                try:
                    import laspy

                    test_las = laspy.read(str(laz_file))
                    if len(test_las.points) > 0:
                        logger.info(
                            f"  ✓ Re-downloaded tile verified "
                            f"({len(test_las.points):,} points)"
                        )
                        # Remove backup if successful
                        if backup_path.exists():
                            backup_path.unlink()
                        return True
                    else:
                        logger.error(f"  ✗ Re-downloaded tile has no points")
                        # Restore backup
                        if backup_path.exists():
                            if laz_file.exists():
                                laz_file.unlink()
                            shutil.move(str(backup_path), str(laz_file))
                        return False
                except Exception as verify_error:
                    logger.error(
                        f"  ✗ Re-downloaded tile is also corrupted: {verify_error}"
                    )
                    # Restore backup
                    if backup_path.exists():
                        if laz_file.exists():
                            laz_file.unlink()
                        shutil.move(str(backup_path), str(laz_file))
                    return False
            else:
                logger.error(f"  ✗ Download failed or file not created")
                # Restore backup
                if backup_path.exists():
                    if not laz_file.exists():
                        shutil.move(str(backup_path), str(laz_file))
                return False

        except ImportError as ie:
            logger.warning(
                f"  ⚠️  IGNLidarDownloader not available for auto-recovery: {ie}"
            )
            return False
        except Exception as e:
            logger.error(f"  ✗ Re-download failed: {e}")
            return False

    def _prefetch_ground_truth_for_tile(self, laz_file: Path) -> Optional[dict]:
        """
        Pre-fetch ground truth data for a single tile.
        This method replaces bulk prefetching with per-tile prefetching.

        Args:
            laz_file: Path to LAZ file to prefetch data for

        Returns:
            Dictionary containing fetched ground truth data, or None if failed
        """
        try:
            # Quick bbox estimation from LAZ header (fast read)
            import laspy

            las = laspy.read(str(laz_file))
            bbox = (
                float(las.x.min()),
                float(las.y.min()),
                float(las.x.max()),
                float(las.y.max()),
            )

            # Fetch data for this specific tile
            use_cache = (
                self.config.get("data_sources", {})
                .get("bd_topo", {})
                .get("cache_enabled", True)
            )
            fetched_data = self.data_fetcher.fetch_all(bbox=bbox, use_cache=use_cache)

            return {"bbox": bbox, "data": fetched_data}
        except Exception as e:
            logger.debug(f"Ground truth prefetch failed for {laz_file.name}: {e}")
            return None

    def _prefetch_ground_truth(self, laz_files: List[Path]):
        """
        DEPRECATED: This method is no longer used.
        Use _prefetch_ground_truth_for_tile() for per-tile prefetching instead.

        Pre-fetch ground truth data for all tiles to warm up the cache.
        This eliminates network I/O delays during processing.

        OPTIMIZATION: Phase 1 - Quick Win
        - Fetches all WFS data in parallel before processing
        - Warms up cache with 8 concurrent workers
        - Reduces per-tile fetch time from 2-3s to <0.1s
        - Expected speedup: +40-60% on data fetching phase

        Args:
            laz_files: List of LAZ files to prefetch data for
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        import laspy
        from tqdm import tqdm

        logger.info(f"📍 Pre-fetching ground truth for {len(laz_files)} tiles...")

        def fetch_for_tile(laz_file):
            """Fetch ground truth for a single tile."""
            try:
                # Quick bbox estimation from LAZ header (fast read)
                las = laspy.read(str(laz_file))
                bbox = (
                    float(las.x.min()),
                    float(las.y.min()),
                    float(las.x.max()),
                    float(las.y.max()),
                )

                # Fetch data (fills cache)
                use_cache = (
                    self.config.get("data_sources", {})
                    .get("bd_topo", {})
                    .get("cache_enabled", True)
                )
                self.data_fetcher.fetch_all(bbox=bbox, use_cache=use_cache)
                return True
            except Exception as e:
                logger.debug(f"Prefetch failed for {laz_file.name}: {e}")
                return False

        # Parallel fetch with progress bar (8 threads for I/O-bound task)
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(fetch_for_tile, f): f for f in laz_files}

            success_count = 0
            for future in tqdm(
                as_completed(futures),
                total=len(laz_files),
                desc="Prefetching",
                unit="tile",
            ):
                if future.result():
                    success_count += 1

        logger.info(
            f"✅ Ground truth pre-fetched ({success_count}/{len(laz_files)} tiles cached)"
        )
        if success_count < len(laz_files):
            logger.warning(
                f"⚠️  {len(laz_files) - success_count} tiles failed to prefetch (will retry during processing)"
            )

    def _process_tile_with_data(
        self,
        laz_file: Path,
        output_dir: Path,
        tile_data: dict,
        tile_idx: int = 0,
        total_tiles: int = 0,
        skip_existing: bool = True,
        prefetched_ground_truth: dict = None,
    ) -> int:
        """
        Process a single LAZ tile with pre-loaded data and optionally pre-fetched ground truth.

        OPTIMIZATION: Phase 2 - Pipeline Optimization
        This method accepts pre-loaded tile data and optionally pre-fetched ground truth,
        allowing both I/O and ground truth fetching to be pipelined for better GPU utilization.

        Args:
            laz_file: Path to LAZ file
            output_dir: Output directory
            tile_data: Pre-loaded tile data from tile_loader
            tile_idx: Current tile index (for progress display)
            total_tiles: Total number of tiles (for progress display)
            skip_existing: Skip processing if patches already exist
            prefetched_ground_truth: Pre-fetched ground truth data (optional)

        Returns:
            Number of patches created (0 if skipped)
        """
        progress_prefix = f"[{tile_idx}/{total_tiles}]" if total_tiles > 0 else ""

        # If tile data loading failed, log and skip
        if tile_data is None:
            logger.error(f"{progress_prefix} ✗ Failed to load tile: {laz_file.name}")
            return 0

        # Check if we should skip (but skip the tile loading since it's already done)
        if skip_existing:
            # Quick check for existing outputs
            should_skip, skip_info = self.skip_checker.should_skip_tile(
                tile_path=laz_file,
                output_dir=output_dir,
                expected_patches=None,
                save_enriched=self.save_enriched_laz,
                include_rgb=self.config.features.use_rgb,
                include_infrared=self.config.features.use_infrared,
                compute_ndvi=self.config.features.compute_ndvi,
                include_extra_features=self.include_extra_features,
                include_classification=OmegaConf.select(
                    self.config, "data_sources.bd_topo.enabled", default=False
                ),
                include_forest=OmegaConf.select(
                    self.config, "data_sources.bd_foret.enabled", default=False
                ),
                include_agriculture=OmegaConf.select(
                    self.config, "data_sources.rpg.enabled", default=False
                ),
                include_cadastre=OmegaConf.select(
                    self.config, "data_sources.cadastre.enabled", default=False
                ),
            )

            if should_skip:
                skip_msg = self.skip_checker.format_skip_message(laz_file, skip_info)
                logger.info(f"{progress_prefix} {skip_msg}")
                return 0

        # Continue with normal tile processing (tile_data already loaded)
        logger.info(f"{progress_prefix} Processing: {laz_file.name}")

        # If we have prefetched ground truth, the cache should be warm
        # This eliminates the network I/O delay during the actual processing
        if prefetched_ground_truth is not None:
            logger.debug(
                f"{progress_prefix} ✅ Ground truth cache warmed via prefetching"
            )

        # ⚡ OPTIMIZATION: Reuse pre-loaded tile_data instead of calling process_tile
        # This eliminates redundant LAZ file loading (saves ~2-3 seconds per tile)
        # Pass tile_data directly to _process_tile_core
        return self._process_tile_core(
            laz_file, output_dir, tile_data, tile_idx, total_tiles, skip_existing
        )

    def process_tile(
        self,
        laz_file: Path,
        output_dir: Path,
        tile_idx: int = 0,
        total_tiles: int = 0,
        skip_existing: bool = True,
    ) -> int:
        """
        Process a single LAZ tile.

        Args:
            laz_file: Path to LAZ file
            output_dir: Output directory
            tile_idx: Current tile index (for progress display)
            total_tiles: Total number of tiles (for progress display)
            skip_existing: Skip processing if patches already exist

        Returns:
            Number of patches created (0 if skipped)
        """
        progress_prefix = f"[{tile_idx}/{total_tiles}]" if total_tiles > 0 else ""

        # Use intelligent skip checker to validate existing outputs
        if skip_existing:
            # First check metadata-based skip (config changes)
            metadata_mgr = ProcessingMetadata(output_dir)
            should_reprocess, reprocess_reason = metadata_mgr.should_reprocess(
                laz_file.stem, self.config
            )

            if should_reprocess:
                if reprocess_reason == "config_changed":
                    logger.info(
                        f"{progress_prefix} Reprocessing {laz_file.name}: "
                        f"Configuration changed since last processing"
                    )
                elif reprocess_reason == "no_metadata_found":
                    logger.debug(
                        f"{progress_prefix} No metadata found for {laz_file.name}, "
                        f"will check for existing outputs"
                    )
                elif reprocess_reason and "output_file_missing" in reprocess_reason:
                    logger.info(
                        f"{progress_prefix} Reprocessing {laz_file.name}: "
                        f"Output file missing"
                    )
            else:
                # Metadata indicates we can skip (config unchanged and outputs exist)
                logger.info(
                    f"{progress_prefix} ⏭️  {laz_file.name}: "
                    f"Already processed with same config, skipping"
                )
                return 0

            # If no metadata or reprocessing needed, use output-based skip checker
            if reprocess_reason == "no_metadata_found":
                should_skip, skip_info = self.skip_checker.should_skip_tile(
                    tile_path=laz_file,
                    output_dir=output_dir,
                    expected_patches=None,  # We don't know expected count beforehand
                    save_enriched=self.save_enriched_laz,
                    include_rgb=self.config.features.use_rgb,
                    include_infrared=self.config.features.use_infrared,
                    compute_ndvi=self.config.features.compute_ndvi,
                    include_extra_features=self.include_extra_features,
                    include_classification=OmegaConf.select(
                        self.config, "data_sources.bd_topo.enabled", default=False
                    )
                    and OmegaConf.select(
                        self.config,
                        "data_sources.bd_topo.features.buildings",
                        default=False,
                    ),
                    include_forest=OmegaConf.select(
                        self.config, "data_sources.bd_foret.enabled", default=False
                    ),
                    include_agriculture=OmegaConf.select(
                        self.config, "data_sources.rpg.enabled", default=False
                    ),
                    include_cadastre=OmegaConf.select(
                        self.config, "data_sources.cadastre.enabled", default=False
                    ),
                )

                if should_skip:
                    # Format detailed skip message
                    skip_msg = self.skip_checker.format_skip_message(
                        laz_file, skip_info
                    )
                    logger.info(f"{progress_prefix} {skip_msg}")
                    return 0
                else:
                    # Log reason for processing (helpful for debugging)
                    reason = skip_info.get("reason", "unknown")
                    if reason == "no_patches_found":
                        logger.debug(
                            f"{progress_prefix} Processing: No existing outputs found"
                        )
                    elif reason == "enriched_laz_invalid":
                        missing_features = skip_info.get("missing_features", [])
                        logger.info(
                            f"{progress_prefix} Reprocessing: Enriched LAZ missing features "
                            f"({', '.join(missing_features[:3])}...)"
                        )
                    elif reason == "corrupted_patches_found":
                        num_corrupted = skip_info.get("corrupted_count", 0)
                        logger.info(
                            f"{progress_prefix} Reprocessing: Found {num_corrupted} "
                            f"corrupted patches"
                        )

        logger.info(f"{progress_prefix} Processing: {laz_file.name}")

        # 1. Load tile data using TileLoader module (Phase 3.4)
        tile_data = self.tile_loader.load_tile(laz_file, max_retries=2)

        if tile_data is None:
            logger.error(f"  ✗ Failed to load tile: {laz_file.name}")
            return 0

        # Validate tile has sufficient points
        if not self.tile_loader.validate_tile(tile_data):
            logger.warning(f"  ⚠️  Insufficient points in tile: {laz_file.name}")
            return 0

        # Delegate to core processing method
        return self._process_tile_core(
            laz_file, output_dir, tile_data, tile_idx, total_tiles, skip_existing
        )

    def _process_tile_core(
        self,
        laz_file: Path,
        output_dir: Path,
        tile_data: dict,
        tile_idx: int = 0,
        total_tiles: int = 0,
        skip_existing: bool = True,
    ) -> int:
        """
        Core tile processing logic that works with pre-loaded tile data.

        ⚡ OPTIMIZATION: This method accepts pre-loaded tile_data to avoid redundant I/O.
        Called by both process_tile() and _process_tile_with_data().

        Args:
            laz_file: Path to LAZ file (for output naming)
            output_dir: Output directory
            tile_data: Pre-loaded tile data from TileLoader
            tile_idx: Current tile index (for progress display)
            total_tiles: Total number of tiles (for progress display)
            skip_existing: Skip processing if patches already exist

        Returns:
            Number of patches created (0 if skipped)
        """
        progress_prefix = f"[{tile_idx}/{total_tiles}]" if total_tiles > 0 else ""
        tile_start = time.time()

        # Load tile metadata if architectural style is requested
        architectural_style_id = 0  # Default: unknown
        multi_styles = None  # For multi-label encoding

        if self.include_architectural_style:
            metadata_mgr = MetadataManager(laz_file.parent)
            tile_metadata = metadata_mgr.load_tile_metadata(laz_file)

            if tile_metadata:
                # Check for new multi-label styles
                if "architectural_styles" in tile_metadata:
                    multi_styles = tile_metadata["architectural_styles"]
                    style_names = [s.get("style_name", "?") for s in multi_styles]
                    logger.info(f"  🏛️  Multi-style: {', '.join(style_names)}")
                else:
                    # Fall back to single style (legacy)
                    characteristics = tile_metadata.get("characteristics", [])
                    category = tile_metadata.get("location", {}).get("category")
                    architectural_style_id = get_architectural_style_id(
                        characteristics=characteristics, category=category
                    )
                    loc_name = tile_metadata.get("location", {}).get("name", "?")
                    logger.info(f"  🏛️  Style: {architectural_style_id} ({loc_name})")
            else:
                logger.debug(f"  No metadata for {laz_file.name}, style=0")
        else:
            tile_metadata = None

        # Extract data from TileLoader (includes loading, extraction, bbox filtering, preprocessing)
        points = tile_data["points"]
        intensity = tile_data["intensity"]
        return_number = tile_data["return_number"]
        classification = tile_data["classification"]
        input_rgb = tile_data.get("input_rgb")
        input_nir = tile_data.get("input_nir")
        input_ndvi = tile_data.get("input_ndvi")
        enriched_features = tile_data.get("enriched_features", {})

        # Store original data - used for ALL versions (already filtered/preprocessed by TileLoader)
        original_data = {
            "points": points,
            "intensity": intensity,
            "return_number": return_number,
            "classification": classification,
            "input_rgb": input_rgb,  # Preserve input RGB if present
            "input_nir": input_nir,  # Preserve input NIR if present
            "input_ndvi": input_ndvi,  # Preserve input NDVI if present
            "enriched_features": enriched_features,  # Preserve enriched features if present
            "las": tile_data.get("las"),  # Preserve LAS object for header info
            "header": tile_data.get("header"),  # Preserve header for chunked loading
        }

        # Data is already preprocessed by TileLoader if enabled
        # Set up variables for feature computation (TileLoader has already handled loading/preprocessing)
        points_v = points
        intensity_v = intensity
        return_number_v = return_number
        classification_v = classification
        input_rgb_v = input_rgb
        input_nir_v = input_nir
        input_ndvi_v = input_ndvi
        enriched_features_v = enriched_features

        # 1a. Augment ground points with RGE ALTI DTM (BEFORE feature computation)
        # This step adds synthetic ground points from DTM to fill gaps under vegetation, buildings, etc.
        # ⚡ CRITICAL: Must be done BEFORE feature computation so features are computed on all points
        rge_alti_enabled = OmegaConf.select(
            self.config, "data_sources.rge_alti.enabled", default=False
        )
        augment_ground = OmegaConf.select(
            self.config, "ground_truth.rge_alti.augment_ground", default=False
        )

        if rge_alti_enabled and augment_ground:
            logger.info(f"  🌍 Augmenting ground points with RGE ALTI DTM...")
            try:
                # Get tile bounding box
                bbox = (
                    float(points_v[:, 0].min()),
                    float(points_v[:, 1].min()),
                    float(points_v[:, 0].max()),
                    float(points_v[:, 1].max()),
                )

                # Augment ground points using DTM
                points_augmented, classification_augmented = (
                    self._augment_ground_with_dtm(
                        points=points_v, classification=classification_v, bbox=bbox
                    )
                )

                # Update point cloud with augmented data
                n_added = len(points_augmented) - len(points_v)
                if n_added > 0:
                    logger.info(
                        f"  ✅ Added {n_added:,} synthetic ground points from DTM"
                    )

                    # Replace original data with augmented data
                    points_v = points_augmented
                    classification_v = classification_augmented

                    # Extend other arrays with null/default values for synthetic points
                    # Synthetic points don't have intensity, return_number, RGB, NIR data
                    if intensity_v is not None:
                        intensity_v = np.concatenate(
                            [intensity_v, np.zeros(n_added, dtype=intensity_v.dtype)]
                        )
                    if return_number_v is not None:
                        return_number_v = np.concatenate(
                            [
                                return_number_v,
                                np.ones(
                                    n_added, dtype=return_number_v.dtype
                                ),  # Set to 1 (single return)
                            ]
                        )
                    if input_rgb_v is not None:
                        input_rgb_v = np.concatenate(
                            [
                                input_rgb_v,
                                np.zeros((n_added, 3), dtype=input_rgb_v.dtype),
                            ]
                        )
                    if input_nir_v is not None:
                        input_nir_v = np.concatenate(
                            [input_nir_v, np.zeros(n_added, dtype=input_nir_v.dtype)]
                        )
                    if input_ndvi_v is not None:
                        input_ndvi_v = np.concatenate(
                            [input_ndvi_v, np.zeros(n_added, dtype=input_ndvi_v.dtype)]
                        )

                    # Update original_data to include augmented points
                    original_data["points"] = points_v
                    original_data["classification"] = classification_v
                    original_data["intensity"] = intensity_v
                    original_data["return_number"] = return_number_v
                    original_data["input_rgb"] = input_rgb_v
                    original_data["input_nir"] = input_nir_v
                    original_data["input_ndvi"] = input_ndvi_v

                    # CRITICAL FIX: Update tile_data as well so feature computation uses augmented points
                    # This ensures all feature arrays match the augmented point count
                    tile_data["points"] = points_v
                    tile_data["classification"] = classification_v
                    tile_data["intensity"] = intensity_v
                    tile_data["return_number"] = return_number_v
                    tile_data["input_rgb"] = input_rgb_v
                    tile_data["input_nir"] = input_nir_v
                    tile_data["input_ndvi"] = input_ndvi_v

                    # If enriched_features exist from input, they need to be extended too
                    # Otherwise feature orchestrator might use stale features with wrong size
                    if enriched_features_v:
                        logger.debug(
                            f"      Extending {len(enriched_features_v)} enriched features with default values"
                        )
                        extended_enriched = {}
                        for feat_name, feat_data in enriched_features_v.items():
                            if (
                                isinstance(feat_data, np.ndarray)
                                and len(feat_data.shape) > 0
                            ):
                                if (
                                    len(feat_data) == len(points) - n_added
                                ):  # Original point count
                                    # Extend with zeros/defaults
                                    if len(feat_data.shape) == 1:
                                        extended_enriched[feat_name] = np.concatenate(
                                            [
                                                feat_data,
                                                np.zeros(
                                                    n_added, dtype=feat_data.dtype
                                                ),
                                            ]
                                        )
                                    else:
                                        # Multi-dimensional feature
                                        extended_enriched[feat_name] = np.concatenate(
                                            [
                                                feat_data,
                                                np.zeros(
                                                    (n_added, *feat_data.shape[1:]),
                                                    dtype=feat_data.dtype,
                                                ),
                                            ]
                                        )
                                else:
                                    # Feature size doesn't match - keep as is but warn
                                    logger.warning(
                                        f"      Enriched feature '{feat_name}' size mismatch: {len(feat_data)} vs {len(points) - n_added}"
                                    )
                                    extended_enriched[feat_name] = feat_data
                            else:
                                # Not an array or scalar - keep as is
                                extended_enriched[feat_name] = feat_data

                        enriched_features_v = extended_enriched
                        tile_data["enriched_features"] = enriched_features_v
                        original_data["enriched_features"] = enriched_features_v
                else:
                    logger.info(
                        f"  ℹ️  No ground points added (sufficient existing coverage)"
                    )

            except Exception as e:
                logger.error(f"  ❌ Failed to augment ground points with DTM: {e}")
                logger.debug(f"  Exception details:", exc_info=True)

        # 1b. Pre-fetch ground truth data for cluster ID computation
        # This must happen BEFORE feature computation so building/parcel
        # geometries are available when computing cluster IDs
        gt_data = None
        if self.data_fetcher is not None:
            try:
                # Get tile bounding box
                bbox = (
                    float(points_v[:, 0].min()),
                    float(points_v[:, 1].min()),
                    float(points_v[:, 0].max()),
                    float(points_v[:, 1].max()),
                )

                logger.info("  📍 Pre-fetching ground truth data for cluster IDs...")

                # Fetch all ground truth data for this tile
                use_cache = (
                    self.config.get("data_sources", {})
                    .get("bd_topo", {})
                    .get("cache_enabled", True)
                )
                gt_data = self.data_fetcher.fetch_all(bbox=bbox, use_cache=use_cache)

                # Add ground truth geometries to tile_data for cluster IDs
                if gt_data is not None and "ground_truth" in gt_data:
                    ground_truth_features = gt_data["ground_truth"]

                    # Extract building and parcel geometries for cluster IDs
                    tile_data["ground_truth_features"] = {}

                    if (
                        "buildings" in ground_truth_features
                        and ground_truth_features["buildings"] is not None
                        and len(ground_truth_features["buildings"]) > 0
                    ):
                        buildings = ground_truth_features["buildings"]
                        tile_data["ground_truth_features"]["buildings"] = buildings
                        n_bldgs = len(buildings)
                        logger.debug(
                            f"  ✓ {n_bldgs} building geometries "
                            "available for cluster IDs"
                        )

                    # Check for parcel data in cadastre
                    # Note: gt_data["cadastre"] is directly a GeoDataFrame, not a dict with "parcels" key
                    if (
                        "cadastre" in gt_data
                        and gt_data["cadastre"] is not None
                        and len(gt_data["cadastre"]) > 0
                    ):
                        parcels = gt_data["cadastre"]
                        tile_data["ground_truth_features"]["parcels"] = parcels
                        logger.debug(
                            f"  ✓ {len(parcels)} parcel geometries "
                            "available for cluster IDs"
                        )

            except Exception as e:
                logger.warning(f"  ⚠️  Failed to pre-fetch ground truth data: {e}")
                logger.debug("  Exception details:", exc_info=True)

        # 2. Compute all features using FeatureOrchestrator (Phase 4.3)
        # Store tile metadata in tile_data for orchestrator to use
        if tile_metadata:
            tile_data["tile_metadata"] = tile_metadata

        all_features = self.feature_orchestrator.compute_features(tile_data=tile_data)

        # Extract feature arrays for patch creation
        normals = all_features.get("normals")
        curvature = all_features.get("curvature")
        height = all_features.get("height")

        # For ground truth classification, use height_above_ground if available (critical for road/rail filtering!)
        height_above_ground = all_features.get(
            "height_above_ground", height
        )  # Fallback to height if not available

        # CRITICAL CHECK: Verify feature arrays match point count after DTM augmentation
        n_points_current = len(points_v)
        if (
            height_above_ground is not None
            and len(height_above_ground) != n_points_current
        ):
            logger.error(f"  ❌ CRITICAL: height_above_ground size mismatch!")
            logger.error(f"     Expected: {n_points_current:,} points")
            logger.error(f"     Got: {len(height_above_ground):,} points")
            logger.error(
                f"     This indicates features were computed before DTM augmentation"
            )
            raise ValueError(
                f"Feature array size mismatch: {len(height_above_ground)} != {n_points_current}"
            )

        # Extract geometric features (excluding main features and spectral/style features)
        excluded_features = {
            "normals",
            "curvature",
            "height",
            "rgb",
            "nir",
            "ndvi",
            "architectural_style",
        }
        geo_features = {
            k: v for k, v in all_features.items() if k not in excluded_features
        }

        # 3. Remap labels (ASPRS → LOD) - skip for ASPRS mode
        if self.class_mapping is not None:
            # LOD2 or LOD3: Apply class mapping
            labels_v = np.array(
                [
                    self.class_mapping.get(c, self.default_class)
                    for c in classification_v
                ],
                dtype=np.uint8,
            )
        else:
            # ASPRS mode: Use classification codes directly
            labels_v = classification_v.astype(np.uint8)

        # 3a. Fetch and apply ground truth classification (BD TOPO, BD Forêt, RPG, etc.)
        # This is the PRIMARY ground truth classification step (step 3a)
        if self.data_fetcher is not None:
            try:
                # Get tile bounding box
                bbox = (
                    float(points_v[:, 0].min()),
                    float(points_v[:, 1].min()),
                    float(points_v[:, 0].max()),
                    float(points_v[:, 1].max()),
                )

                logger.info(f"  📍 Fetching ground truth for tile bbox: {bbox}")

                # Fetch all ground truth data for this tile
                use_cache = (
                    self.config.get("data_sources", {})
                    .get("bd_topo", {})
                    .get("cache_enabled", True)
                )
                gt_data = self.data_fetcher.fetch_all(bbox=bbox, use_cache=use_cache)

                # Apply ground truth classification if BD TOPO data is available
                if gt_data is not None and "ground_truth" in gt_data:
                    # Check if we should use the new optimized ground truth classifier
                    use_optimized_gt = self.config.processor.get(
                        "use_optimized_ground_truth", True
                    )

                    if use_optimized_gt:
                        # NEW: Use GroundTruthOptimizer (10-100× faster, GPU-accelerated)
                        from ..io.ground_truth_optimizer import GroundTruthOptimizer

                        # Prepare ground truth features dictionary
                        ground_truth_features = gt_data["ground_truth"]

                        # Log what ground truth features are available
                        available_features = [
                            k
                            for k, v in ground_truth_features.items()
                            if v is not None and len(v) > 0
                        ]
                        if available_features:
                            logger.info(
                                f"  🗺️  Applying ground truth from BD TOPO® (optimized): {', '.join(available_features)}"
                            )
                            for feat_type in available_features:
                                logger.info(
                                    f"      - {feat_type}: {len(ground_truth_features[feat_type])} features"
                                )
                        else:
                            logger.warning(
                                f"  ⚠️  No ground truth features found for this tile!"
                            )

                        # Store original classification for comparison
                        labels_before = labels_v.copy()

                        # Create optimizer with auto-selection of best method
                        # GPU chunked for large datasets (>10M pts), GPU for medium, CPU STRtree for small
                        optimizer = GroundTruthOptimizer(
                            force_method="auto",  # Auto-select best method
                            gpu_chunk_size=5_000_000,
                            verbose=True,
                        )

                        # Apply ground truth classification
                        logger.info(
                            f"  🔄 Classifying {len(points_v):,} points with ground truth..."
                        )
                        labels_v = optimizer.label_points(
                            points=points_v,
                            ground_truth_features=ground_truth_features,
                            label_priority=None,  # Use default priority
                            ndvi=all_features.get("ndvi"),
                            use_ndvi_refinement=self.config.features.get(
                                "compute_ndvi", False
                            ),
                            ndvi_vegetation_threshold=0.3,
                            ndvi_building_threshold=0.15,
                        )

                        # Log classification changes
                        n_changed = np.sum(labels_v != labels_before)
                        pct_changed = (n_changed / len(labels_v)) * 100
                        logger.info(
                            f"  ✅ Ground truth applied: {n_changed:,} points changed ({pct_changed:.2f}%)"
                        )

                        # Log new classes found
                        new_classes = np.unique(labels_v)
                        has_roads = 11 in new_classes
                        has_rails = 10 in new_classes
                        has_parking = 40 in new_classes
                        has_sports = 41 in new_classes
                        has_cemetery = 42 in new_classes
                        has_power = 43 in new_classes

                        if has_roads or has_rails:
                            logger.info(
                                f"      🛣️  Roads (11): {'✅' if has_roads else '❌'}"
                            )
                            logger.info(
                                f"      🚂 Railways (10): {'✅' if has_rails else '❌'}"
                            )
                        if has_parking or has_sports or has_cemetery or has_power:
                            logger.info(
                                f"      🅿️  Parking (40): {'✅' if has_parking else '❌'}"
                            )
                            logger.info(
                                f"      ⚽ Sports (41): {'✅' if has_sports else '❌'}"
                            )
                            logger.info(
                                f"      🪦 Cemetery (42): {'✅' if has_cemetery else '❌'}"
                            )
                            logger.info(
                                f"      ⚡ Power Lines (43): {'✅' if has_power else '❌'}"
                            )

                    else:
                        # Use UnifiedClassifier with comprehensive strategy
                        logger.info(
                            "  Using UnifiedClassifier with comprehensive strategy"
                        )
                        from .classification import (
                            ClassificationStrategy,
                            UnifiedClassifier,
                        )

                        # Get building and transport detection modes from config
                        building_mode = self.config.processor.get(
                            "building_detection_mode", "asprs"
                        )
                        transport_mode = self.config.processor.get(
                            "transport_detection_mode", "asprs_extended"
                        )

                        # Get BD TOPO parameters from config for ground truth classification
                        bd_topo_params = (
                            self.config.get("data_sources", {})
                            .get("bd_topo", {})
                            .get("parameters", {})
                        )

                        logger.info(f"  🔧 Building detection mode: {building_mode}")
                        logger.info(f"  🔧 Transport detection mode: {transport_mode}")

                        # Create unified classifier with comprehensive strategy and appropriate config
                        classifier = UnifiedClassifier(
                            strategy=ClassificationStrategy.COMPREHENSIVE,
                            use_ground_truth=True,
                            use_ndvi=self.config.features.get("compute_ndvi", False),
                            use_geometric=True,
                            building_detection_mode=building_mode,
                            transport_detection_mode=transport_mode,
                            # Pass BD TOPO parameters for road/railway filtering
                            road_buffer_tolerance=bd_topo_params.get(
                                "road_buffer_tolerance", 0.5
                            ),
                        )

                        # Prepare ground truth features dictionary
                        ground_truth_features = gt_data["ground_truth"]

                        # Log what ground truth features are available
                        available_features = [
                            k
                            for k, v in ground_truth_features.items()
                            if v is not None and len(v) > 0
                        ]
                        if available_features:
                            logger.info(
                                f"  🗺️  Applying ground truth from BD TOPO®: {', '.join(available_features)}"
                            )
                            for feat_type in available_features:
                                logger.info(
                                    f"      - {feat_type}: {len(ground_truth_features[feat_type])} features"
                                )
                        else:
                            logger.warning(
                                f"  ⚠️  No ground truth features found for this tile!"
                            )

                        # Store original classification for comparison
                        labels_before = labels_v.copy()

                        # Apply ground truth classification using the advanced classifier
                        logger.info(f"  🔄 Classifying points with ground truth...")
                        labels_v = classifier._classify_by_ground_truth(
                            labels=labels_v,
                            points=points_v,
                            ground_truth_features=ground_truth_features,
                            ndvi=all_features.get("ndvi"),
                            height=height_above_ground,  # Use height_above_ground for proper road/rail filtering!
                            planarity=geo_features.get("planarity"),
                            intensity=intensity_v,
                        )

                        # Log classification changes
                        n_changed = np.sum(labels_v != labels_before)
                        pct_changed = (n_changed / len(labels_v)) * 100
                        logger.info(
                            f"  ✅ Ground truth applied: {n_changed:,} points changed ({pct_changed:.2f}%)"
                        )

                        # Log new classes found
                        new_classes = np.unique(labels_v)
                        has_roads = 11 in new_classes
                        has_rails = 10 in new_classes
                        has_parking = 40 in new_classes
                        has_sports = 41 in new_classes
                        has_cemetery = 42 in new_classes
                        has_power = 43 in new_classes

                        if has_roads or has_rails:
                            logger.info(
                                f"      🛣️  Roads (11): {'✅' if has_roads else '❌'}"
                            )
                            logger.info(
                                f"      🚂 Railways (10): {'✅' if has_rails else '❌'}"
                            )
                        if has_parking or has_sports or has_cemetery or has_power:
                            logger.info(
                                f"      🅿️  Parking (40): {'✅' if has_parking else '❌'}"
                            )
                            logger.info(
                                f"      ⚽ Sports (41): {'✅' if has_sports else '❌'}"
                            )
                            logger.info(
                                f"      🪦 Cemetery (42): {'✅' if has_cemetery else '❌'}"
                            )
                            logger.info(
                                f"      ⚡ Power Lines (43): {'✅' if has_power else '❌'}"
                            )

                        # CRITICAL FIX: Recompute classification-dependent features
                        # After ground truth classification changes labels_v, we need to recompute features
                        # that depend on classification labels (primarily height_above_ground)
                        if n_changed > 0:
                            logger.info(
                                f"  🔄 Recomputing classification-dependent features after ground truth update..."
                            )

                            # Update tile_data with new classification
                            tile_data["classification"] = labels_v
                            classification_v = labels_v

                            # CRITICAL: Recompute height_above_ground with updated ground classification
                            # The height feature uses ground points (class == 2) to determine ground level
                            # If ground truth changed ground classification, height must be recomputed
                            ground_mask = labels_v == 2
                            if ground_mask.any():
                                ground_z = points_v[ground_mask, 2].min()
                                height_updated = (points_v[:, 2] - ground_z).astype(
                                    np.float32
                                )
                                n_ground = np.sum(ground_mask)
                                logger.info(
                                    f"      ✅ Height recomputed using {n_ground:,} ground points (ground_z={ground_z:.2f}m)"
                                )
                            else:
                                height_updated = (
                                    points_v[:, 2] - points_v[:, 2].min()
                                ).astype(np.float32)
                                logger.info(
                                    f"      ⚠️  No ground points found after GT update, using min Z"
                                )

                            # Update height in all feature dicts
                            all_features["height"] = height_updated
                            all_features["height_above_ground"] = height_updated
                            height = height_updated
                            height_above_ground = height_updated
                            if "height" in geo_features:
                                geo_features["height"] = height_updated
                            if "height_above_ground" in geo_features:
                                geo_features["height_above_ground"] = height_updated

                            # Update original_data to ensure save operations use correct classification
                            original_data["classification"] = labels_v

                            logger.info(
                                f"  ✅ Classification and height features updated"
                            )

                else:
                    logger.warning(
                        f"  ⚠️  No ground truth data returned from data fetcher!"
                    )

            except Exception as e:
                logger.error(f"  ❌ Failed to apply ground truth classification: {e}")
                logger.debug(f"  Exception details:", exc_info=True)
        else:
            logger.warning(
                f"  ⚠️  DataFetcher is None - ground truth classification skipped!"
            )
            logger.warning(
                f"      Check if data_sources.bd_topo.enabled = true in config"
            )

        # 3aa. Optional: Apply optimized reclassification (if enabled in config)
        # This is an ADDITIONAL refinement pass using the optimized reclassifier (step 3aa)
        # Note: Step 3a above already applies basic ground truth classification
        reclassification_config = self.config.processor.get("reclassification", {})

        # Check if reclassification is enabled
        reclass_enabled = reclassification_config.get("enabled", False)
        logger.debug(f"  Reclassification config enabled: {reclass_enabled}")
        logger.debug(f"  DataFetcher available: {self.data_fetcher is not None}")

        if reclass_enabled and self.data_fetcher is not None:
            try:
                logger.info(f"  🔄 Applying optimized reclassification (step 3aa)...")

                # Get tile bounding box (reuse from step 3a if available)
                bbox = (
                    float(points_v[:, 0].min()),
                    float(points_v[:, 1].min()),
                    float(points_v[:, 0].max()),
                    float(points_v[:, 1].max()),
                )

                # Fetch ground truth features for reclassification (may use cache from step 3a)
                use_cache = (
                    self.config.get("data_sources", {})
                    .get("bd_topo", {})
                    .get("cache_enabled", True)
                )
                logger.debug(f"  Fetching ground truth (use_cache={use_cache})...")
                gt_data = self.data_fetcher.fetch_all(bbox=bbox, use_cache=use_cache)

                if gt_data is not None and "ground_truth" in gt_data:
                    logger.info(
                        f"  ✅ Ground truth data available for reclassification"
                    )

                    # Store labels before reclassification for comparison
                    labels_before_reclass = labels_v.copy()

                    # Initialize optimized reclassifier
                    chunk_size = reclassification_config.get("chunk_size", 150000)
                    acceleration_mode = reclassification_config.get(
                        "acceleration_mode", "auto"
                    )
                    show_progress = reclassification_config.get("show_progress", False)
                    use_geometric_rules = reclassification_config.get(
                        "use_geometric_rules", True
                    )

                    logger.info(f"      Acceleration mode: {acceleration_mode}")
                    logger.info(f"      Chunk size: {chunk_size:,}")
                    logger.info(f"      Geometric rules: {use_geometric_rules}")

                    reclassifier = OptimizedReclassifier(
                        chunk_size=chunk_size,
                        show_progress=show_progress,
                        acceleration_mode=acceleration_mode,
                        use_geometric_rules=use_geometric_rules,
                    )

                    # Reclassify points in-memory
                    logger.info(f"  🔄 Reclassifying {len(points_v):,} points...")
                    labels_v, reclass_stats = reclassifier.reclassify(
                        points=points_v,
                        current_labels=labels_v,
                        ground_truth_features=gt_data["ground_truth"],
                    )

                    # Log reclassification results
                    if reclass_stats:
                        for feature_type, count in reclass_stats.items():
                            if count > 0:
                                logger.info(
                                    f"      {feature_type}: {count:,} points reclassified"
                                )

                    n_reclass_changed = np.sum(labels_v != labels_before_reclass)
                    pct_reclass_changed = (n_reclass_changed / len(labels_v)) * 100
                    logger.info(
                        f"  ✅ Reclassification completed: {n_reclass_changed:,} points changed ({pct_reclass_changed:.2f}%)"
                    )

                    # 🆕 V5.2: Reclassify vegetation above BD TOPO surfaces
                    # This step uses height_above_ground from RGE ALTI DTM to identify
                    # vegetation (trees, bushes) that are above roads, sports facilities, etc.
                    reclassify_vegetation = reclassification_config.get(
                        "reclassify_vegetation_above_surfaces", True
                    )

                    if reclassify_vegetation and height_above_ground is not None:
                        logger.info(
                            f"\n  🌳 Reclassifying vegetation above BD TOPO surfaces..."
                        )

                        # Get thresholds from config
                        veg_height_threshold = reclassification_config.get(
                            "vegetation_height_threshold", 2.0
                        )
                        veg_ndvi_threshold = reclassification_config.get(
                            "vegetation_ndvi_threshold", 0.3
                        )

                        # Get NDVI if available
                        ndvi_for_veg = all_features.get("ndvi")

                        # Reclassify vegetation above surfaces
                        labels_before_veg = labels_v.copy()
                        labels_v, veg_stats = (
                            reclassifier.reclassify_vegetation_above_surfaces(
                                points=points_v,
                                labels=labels_v,
                                ground_truth_features=gt_data["ground_truth"],
                                height_above_ground=height_above_ground,
                                ndvi=ndvi_for_veg,
                                height_threshold=veg_height_threshold,
                                ndvi_threshold=veg_ndvi_threshold,
                            )
                        )

                        # Log vegetation reclassification results
                        n_veg_changed = np.sum(labels_v != labels_before_veg)
                        if n_veg_changed > 0:
                            logger.info(
                                f"  ✅ Vegetation reclassification: {n_veg_changed:,} points reclassified"
                            )

                            # Log details by surface type
                            for surface_type in [
                                "roads",
                                "sports",
                                "cemeteries",
                                "parking",
                            ]:
                                key = f"{surface_type}_vegetation"
                                if key in veg_stats and veg_stats[key] > 0:
                                    logger.info(
                                        f"      {surface_type}: {veg_stats[key]:,} vegetation points"
                                    )
                        else:
                            logger.info(
                                f"  ℹ️  No vegetation found above BD TOPO surfaces"
                            )
                    elif not reclassify_vegetation:
                        logger.debug(
                            f"  ℹ️  Vegetation reclassification disabled in config"
                        )
                    elif height_above_ground is None:
                        logger.warning(
                            f"  ⚠️  height_above_ground not available - skipping vegetation reclassification"
                        )
                        logger.warning(
                            f"      → Enable RGE ALTI: data_sources.rge_alti.enabled = true"
                        )

                    # Show final class distribution
                    unique_final, counts_final = np.unique(labels_v, return_counts=True)
                    logger.info(f"\n  📊 Final classification distribution:")
                    for cls_code in [
                        11,
                        10,
                        40,
                        41,
                        42,
                        43,
                        3,
                        4,
                        5,
                    ]:  # BD TOPO + vegetation classes
                        if cls_code in unique_final:
                            count = counts_final[unique_final == cls_code][0]
                            pct = (count / len(labels_v)) * 100
                            cls_names = {
                                11: "Roads",
                                10: "Railways",
                                40: "Parking",
                                41: "Sports",
                                42: "Cemetery",
                                43: "Power Lines",
                                3: "Low Vegetation",
                                4: "Medium Vegetation",
                                5: "High Vegetation",
                            }
                            logger.info(
                                f"      Class {cls_code} ({cls_names.get(cls_code, 'Unknown')}): {count:,} ({pct:.2f}%)"
                            )
                else:
                    logger.warning(
                        f"  ⚠️  No ground truth data available for reclassification"
                    )

            except Exception as e:
                logger.error(f"  ❌ Reclassification failed: {e}")
                logger.debug(f"  Exception details:", exc_info=True)
        elif reclass_enabled and self.data_fetcher is None:
            logger.error(f"  ❌ Reclassification is enabled but DataFetcher is None!")
            logger.error(
                f"      → Check if data_sources.bd_topo.enabled = true in config"
            )
        else:
            logger.debug(f"  ℹ️  Reclassification disabled (enabled={reclass_enabled})")

        # 3ab. 🆕 V5.2.1: Filter variable objects using DTM heights
        # This step removes temporary/mobile objects like vehicles, urban furniture, etc.
        # Uses height_above_ground computed from RGE ALTI DTM
        try:
            from ..core.classification.variable_object_filter import (
                apply_variable_object_filtering,
            )

            # Apply filtering if enabled and height_above_ground is available
            if height_above_ground is not None:
                labels_before_filter = labels_v.copy()
                labels_v, filter_stats = apply_variable_object_filtering(
                    points=points_v,
                    classification=labels_v,
                    height_above_ground=height_above_ground,
                    config=self.config,
                    features=geo_features,  # Pass geometric features for wall detection
                )

                # Log filtering results
                if filter_stats.get("total_filtered", 0) > 0:
                    n_filtered = filter_stats["total_filtered"]
                    pct_filtered = (n_filtered / len(labels_v)) * 100
                    logger.info(
                        f"  ✅ Variable object filtering: {n_filtered:,} points ({pct_filtered:.2f}%)"
                    )

                    # Log breakdown by type
                    if filter_stats.get("vehicles_filtered", 0) > 0:
                        logger.info(
                            f"      🚗 Vehicles: {filter_stats['vehicles_filtered']:,} points"
                        )
                    if filter_stats.get("furniture_filtered", 0) > 0:
                        logger.info(
                            f"      🪑 Urban furniture: {filter_stats['furniture_filtered']:,} points"
                        )
                    if filter_stats.get("walls_filtered", 0) > 0:
                        logger.info(
                            f"      🧱 Walls/fences: {filter_stats['walls_filtered']:,} points"
                        )
            else:
                logger.debug(
                    "  ℹ️  height_above_ground not available - skipping variable object filtering"
                )

        except ImportError:
            logger.debug("  ℹ️  Variable object filter module not available")
        except Exception as e:
            logger.warning(f"  ⚠️  Variable object filtering failed: {e}")
            logger.debug("  Exception details:", exc_info=True)

        # 3b. Refine classification using NDVI, ground truth, and geometric features
        if self.lod_level == "LOD2":
            # Prepare features for refinement
            refinement_features = {
                "points": points_v  # Always include point coordinates
            }
            if "ndvi" in all_features:
                refinement_features["ndvi"] = all_features["ndvi"]
            if height is not None:
                refinement_features["height"] = height
            if "planarity" in geo_features:
                refinement_features["planarity"] = geo_features["planarity"]
            if "verticality" in geo_features:
                refinement_features["verticality"] = geo_features["verticality"]
            if "density" in geo_features:
                refinement_features["density"] = geo_features["density"]
            if "roughness" in geo_features:
                refinement_features["roughness"] = geo_features["roughness"]
            if intensity_v is not None:
                refinement_features["intensity"] = intensity_v

            # Prepare ground truth data if available
            ground_truth_data = None
            if (
                "ground_truth_building_mask" in tile_data
                or "ground_truth_road_mask" in tile_data
            ):
                ground_truth_data = {}
                if "ground_truth_building_mask" in tile_data:
                    ground_truth_data["building_mask"] = tile_data[
                        "ground_truth_building_mask"
                    ]
                if "ground_truth_road_mask" in tile_data:
                    ground_truth_data["road_mask"] = tile_data["ground_truth_road_mask"]

            # Apply refinement using unified classifier
            if refinement_features:
                labels_v, refinement_stats = refine_classification_unified(
                    labels=labels_v,
                    features=refinement_features,
                    ground_truth_data=ground_truth_data,
                    lod_level=self.lod_level,
                )

        # 4. Combine features (FeatureComputer has already computed everything)
        # 🔍 DEBUG: Log what's in all_features before combining
        logger.info(
            f"  🔍 DEBUG: all_features keys from orchestrator: {sorted(all_features.keys())}"
        )
        logger.info(
            f"  🔍 DEBUG: geo_features keys: {sorted(geo_features.keys()) if isinstance(geo_features, dict) else 'Not a dict'}"
        )

        all_features_v = {
            "normals": normals,
            "curvature": curvature,
            "intensity": intensity_v,
            "return_number": return_number_v,
            "height": height,
            **(geo_features if isinstance(geo_features, dict) else {}),
        }

        # Add RGB, NIR, NDVI, architectural_style, and cluster IDs
        # if computed by FeatureComputer
        for feat_name in [
            "rgb",
            "nir",
            "ndvi",
            "architectural_style",
            "cluster_id",
            "building_cluster_id",
            "parcel_cluster_id",
        ]:
            if feat_name in all_features:
                all_features_v[feat_name] = all_features[feat_name]

        # 🔍 DEBUG: Log what's in all_features_v after combining
        logger.info(
            f"  🔍 DEBUG: all_features_v keys after combining: {sorted(all_features_v.keys())}"
        )
        for k, v in list(all_features_v.items())[:20]:  # First 20 for brevity
            if hasattr(v, "shape"):
                logger.info(f"      - {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                logger.info(f"      - {k}: type={type(v)}")

        # Preserve input spectral data if present
        if "input_rgb" in all_features:
            all_features_v["input_rgb"] = all_features["input_rgb"]
        if "input_nir" in all_features:
            all_features_v["input_nir"] = all_features["input_nir"]

        # Add enriched features from input if present
        if enriched_features_v:
            for feat_name, feat_data in enriched_features_v.items():
                enriched_key = (
                    f"enriched_{feat_name}"
                    if feat_name in all_features_v
                    else feat_name
                )
                all_features_v[enriched_key] = feat_data

        # 5. Check if we should skip patch extraction
        # Skip if: enriched_only mode OR save_patches is explicitly False
        if self.only_enriched_laz or not self.save_patches:
            # Skip patch extraction - we're only saving the updated enriched LAZ tile
            logger.info(f"  💾 Saving updated enriched tile (no patch extraction)")

            # Import the new function
            from .classification.io import save_enriched_tile_laz

            # Prepare output path - in enriched_only mode, save directly to output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{laz_file.stem}_enriched.laz"

            try:
                # Determine RGB/NIR to save (prefer fetched/computed over input)
                save_rgb = (
                    all_features_v.get("rgb")
                    if all_features_v.get("rgb") is not None
                    else original_data.get("input_rgb")
                )
                save_nir = (
                    all_features_v.get("nir")
                    if all_features_v.get("nir") is not None
                    else original_data.get("input_nir")
                )

                # Remove RGB/NIR from features dict to avoid duplication
                features_to_save = {
                    k: v
                    for k, v in all_features_v.items()
                    if k not in ["rgb", "nir", "input_rgb", "input_nir"]
                }

                # Save the enriched tile with all computed features
                save_enriched_tile_laz(
                    save_path=output_path,
                    points=original_data["points"],
                    classification=labels_v,
                    intensity=original_data["intensity"],
                    return_number=original_data["return_number"],
                    features=features_to_save,
                    original_las=original_data.get("las"),
                    header=original_data.get("header"),
                    input_rgb=save_rgb,
                    input_nir=save_nir,
                )

                tile_time = time.time() - tile_start
                pts_processed = len(original_data["points"])
                logger.info(
                    f"  ✅ Completed: tile processed in {tile_time:.1f}s "
                    f"({pts_processed:,} points)"
                )
                return 1  # Return 1 to count as successfully processed

            except Exception as e:
                logger.error(f"  ✗ Failed to save enriched tile: {e}")
                import traceback

                logger.debug(traceback.format_exc())
                return 0

        # 5. Extract patches and create augmented versions using patch_extractor module
        patch_config = PatchConfig(
            patch_size=self.patch_size,
            overlap=self.patch_overlap,
            min_points=10000,
            target_num_points=self.num_points,
            augment=self.augment,
            num_augmentations=self.num_augmentations,
        )

        aug_config = AugmentationConfig() if self.augment else None

        # Extract all patches (base + augmented versions)
        # RGB, NIR, and NDVI are now included in all_features_v and will be
        # extracted along with other features, ensuring spatial correspondence
        all_patches_collected = extract_and_augment_patches(
            points=points_v,
            features=all_features_v,
            labels=labels_v,
            patch_config=patch_config,
            augment_config=aug_config,
            architecture=self.architecture,
            logger_instance=logger,
        )

        # 7. Save all collected patches
        output_dir.mkdir(parents=True, exist_ok=True)
        num_saved = 0

        rgb_suffix = " + RGB" if self.include_rgb else ""
        total_patches = len(all_patches_collected)
        num_versions = 1 + (self.num_augmentations if self.augment else 0)
        logger.info(
            f"  💾 Saving {total_patches} patches{rgb_suffix} "
            f"({num_versions} versions)"
        )

        # Determine split if using dataset manager
        tile_split = None
        if self.dataset_manager is not None:
            tile_split = self.dataset_manager.get_tile_split(laz_file.stem)
            logger.info(f"  📂 Tile assigned to {tile_split} split")

        # Save patches with proper naming
        # Each patch has _version and _patch_idx metadata
        for patch in all_patches_collected:
            version = patch.pop("_version", "original")
            base_idx = patch.pop("_patch_idx", 0)

            # Use dataset manager for path determination if enabled
            if self.dataset_manager is not None:
                # Get format extension
                formats_list = [fmt.strip() for fmt in self.output_format.split(",")]
                ext = formats_list[0] if len(formats_list) == 1 else "npz"
                if ext in ["pt", "pth", "pytorch", "torch"]:
                    ext = "pt"
                elif ext == "hdf5":
                    ext = "h5"

                base_path = self.dataset_manager.get_patch_path(
                    tile_name=laz_file.stem,
                    patch_idx=base_idx,
                    architecture=self.architecture,
                    version=version,
                    split=tile_split,
                    extension=ext,
                ).with_suffix(
                    ""
                )  # Remove extension, will be added by save functions
            else:
                # Traditional naming without dataset splits
                if version == "original":
                    patch_name = (
                        f"{laz_file.stem}_{self.architecture}_patch_{base_idx:04d}"
                    )
                else:
                    patch_name = (
                        f"{laz_file.stem}_{self.architecture}_patch_{base_idx:04d}_"
                        f"{version}"
                    )
                base_path = output_dir / patch_name

            # Check if multiple formats are requested
            formats_list = [fmt.strip() for fmt in self.output_format.split(",")]

            if len(formats_list) > 1:
                # Multi-format: use save_patch_multi_format
                # Get the architecture-specific formatted data for LAZ
                arch_formatted = format_patch_for_architecture(
                    patch,
                    self.architecture,
                    num_points=None,  # Keep original num_points
                )
                num_saved += save_patch_multi_format(
                    base_path,
                    arch_formatted,
                    formats_list,
                    original_patch=patch,
                    lod_level=self.lod_level,
                )
            else:
                # Single format: use format-specific save function
                fmt = formats_list[0]
                if fmt == "npz":
                    save_path = base_path.with_suffix(".npz")
                    save_patch_npz(save_path, patch, lod_level=self.lod_level)
                elif fmt == "hdf5":
                    save_path = base_path.with_suffix(".h5")
                    save_patch_hdf5(save_path, patch)
                elif fmt in ["pt", "pth", "pytorch", "torch"]:
                    save_path = base_path.with_suffix(".pt")
                    save_patch_torch(save_path, patch)
                elif fmt == "laz":
                    save_path = base_path.with_suffix(".laz")
                    arch_formatted = format_patch_for_architecture(
                        patch,
                        self.architecture,
                        num_points=None,  # Keep original num_points
                    )
                    save_patch_laz(save_path, arch_formatted, patch)
                else:
                    # Fallback
                    save_path = base_path.with_suffix(".npz")
                    save_patch_npz(save_path, patch, lod_level=self.lod_level)
                num_saved += 1

            # Record patch in dataset manager for statistics
            if self.dataset_manager is not None:
                self.dataset_manager.record_patch_saved(
                    tile_name=laz_file.stem,
                    split=tile_split,
                    patch_size=int(self.patch_size),
                )

        # 🆕 Save enriched LAZ tile in "both" mode (after patches are saved)
        if self.save_enriched_laz and not self.only_enriched_laz:
            from .classification.io import save_enriched_tile_laz

            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{laz_file.stem}_enriched.laz"

            try:
                logger.info(f"  💾 Saving enriched LAZ tile...")

                # Determine RGB/NIR to save (prefer fetched/computed over input)
                save_rgb = (
                    all_features_v.get("rgb")
                    if all_features_v.get("rgb") is not None
                    else original_data.get("input_rgb")
                )
                save_nir = (
                    all_features_v.get("nir")
                    if all_features_v.get("nir") is not None
                    else original_data.get("input_nir")
                )

                # Remove RGB/NIR from features dict to avoid duplication
                features_to_save = {
                    k: v
                    for k, v in all_features_v.items()
                    if k not in ["rgb", "nir", "input_rgb", "input_nir", "points"]
                }

                # 🔍 DEBUG: Log features being sent to save function
                logger.info(
                    f"  🔍 DEBUG: Features in all_features_v: {sorted(all_features_v.keys())}"
                )
                logger.info(
                    f"  🔍 DEBUG: Features to save (after RGB/NIR removal): {sorted(features_to_save.keys())}"
                )
                for k, v in features_to_save.items():
                    if hasattr(v, "shape"):
                        logger.info(
                            f"      - {k}: shape={v.shape}, dtype={v.dtype}, ndim={v.ndim}"
                        )
                    else:
                        logger.info(f"      - {k}: type={type(v)}")

                # Save the enriched tile with all computed features
                save_enriched_tile_laz(
                    save_path=output_path,
                    points=original_data["points"],
                    classification=labels_v,
                    intensity=original_data["intensity"],
                    return_number=original_data["return_number"],
                    features=features_to_save,
                    original_las=original_data.get("las"),
                    header=original_data.get("header"),
                    input_rgb=save_rgb,
                    input_nir=save_nir,
                )

            except Exception as e:
                logger.error(f"  ✗ Failed to save enriched LAZ tile: {e}")
                import traceback

                logger.debug(traceback.format_exc())

        tile_time = time.time() - tile_start
        pts_processed = len(original_data["points"])

        # Save processing metadata for intelligent skip detection
        try:
            metadata_mgr = ProcessingMetadata(output_dir)
            output_files = {}

            # Record enriched LAZ if saved
            if self.save_enriched_laz:
                enriched_path = output_dir / f"{laz_file.stem}_enriched.laz"
                if enriched_path.exists():
                    output_files["enriched_laz"] = {
                        "path": str(enriched_path),
                        "size_bytes": enriched_path.stat().st_size,
                    }

            # Record patch information
            if num_saved > 0:
                output_files["patches"] = {
                    "count": num_saved,
                    "format": self.output_format,
                }

            metadata_mgr.save_metadata(
                tile_name=laz_file.stem,
                config=self.config,
                processing_time=tile_time,
                num_points=pts_processed,
                output_files=output_files,
            )
        except Exception as e:
            logger.warning(f"  ⚠️  Failed to save processing metadata: {e}")

        logger.info(
            f"  ✅ Completed: {num_saved} patches in {tile_time:.1f}s "
            f"(from {pts_processed:,} original points)"
        )
        return num_saved

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        num_workers: int = 1,
        save_metadata: bool = True,
        skip_existing: bool = True,
    ) -> int:
        """
        Process directory of LAZ files.

        Args:
            input_dir: Directory containing LAZ files
            output_dir: Output directory
            num_workers: Number of parallel workers
            save_metadata: Whether to save stats.json
            skip_existing: Skip tiles that already have patches in output

        Returns:
            Total number of patches created
        """
        start_time = time.time()

        # Initialize dataset manager if enabled
        if self._dataset_config is not None:
            self.dataset_manager = DatasetManager(
                output_dir=output_dir,
                config=self._dataset_config,
                patch_size=int(self.patch_size),
            )
            logger.info(f"📊 Dataset manager initialized for {output_dir}")

        # Check system memory and adjust workers if needed
        try:
            import psutil

            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()

            available_gb = mem.available / (1024**3)
            swap_percent = swap.percent

            logger.info(f"System Memory: {available_gb:.1f}GB available")

            # If swap is heavily used, reduce workers automatically
            if swap_percent > 50:
                logger.warning(f"⚠️  High swap usage detected ({swap_percent:.0f}%)")
                logger.warning("⚠️  Memory pressure detected - reducing workers to 1")
                num_workers = 1

            # Processing needs ~2-3GB per worker
            min_gb_per_worker = 2.5
            max_safe_workers = int(available_gb / min_gb_per_worker)

            if num_workers > max_safe_workers:
                logger.warning(f"⚠️  Limited RAM ({available_gb:.1f}GB available)")
                logger.warning(
                    f"⚠️  Reducing workers from {num_workers} "
                    f"to {max(1, max_safe_workers)}"
                )
                num_workers = max(1, max_safe_workers)

        except ImportError:
            logger.debug("psutil not available - skipping memory checks")

        # Find LAZ files (recursively) - exclude enriched files to avoid reprocessing
        laz_files = list(input_dir.rglob("*.laz")) + list(input_dir.rglob("*.LAZ"))

        # Filter out enriched files (those ending with _enriched.laz)
        laz_files = [f for f in laz_files if not f.stem.endswith("_enriched")]

        if not laz_files:
            logger.error(f"No LAZ files found in {input_dir}")
            return 0

        total_tiles = len(laz_files)
        logger.info(f"Found {total_tiles} LAZ files")

        # OPTIMIZATION: Use tile-by-tile prefetching instead of bulk prefetching
        # This reduces memory usage and allows processing to start immediately
        k_display = self.k_neighbors or "auto"
        logger.info(
            f"Configuration: LOD={self.lod_level} | k={k_display} | "
            f"patch_size={self.patch_size}m | augment={self.augment}"
        )
        logger.info("")

        # Initialize metadata manager
        metadata_mgr = MetadataManager(output_dir) if save_metadata else None

        # Copy directory structure from source
        if metadata_mgr:
            logger.info("Copying directory structure from source...")
            metadata_mgr.copy_directory_structure(input_dir)

        # Process files
        tiles_processed = 0
        tiles_skipped = 0

        if num_workers > 1:
            # Check for GPU + multiprocessing conflict
            if self.feature_orchestrator.use_gpu:
                logger.warning(
                    "⚠️  GPU acceleration is not compatible with multiprocessing due to CUDA context limitations"
                )
                logger.warning(
                    "🔧 Automatically disabling GPU for multiprocessing mode"
                )
                logger.warning(
                    "💡 To use GPU: set num_workers=1, or disable GPU: use_gpu=false"
                )

                # Completely disable GPU for multiprocessing to prevent CUDA context issues
                disable_gpu_for_multiprocessing()

                # Disable GPU in feature orchestrator configuration
                self.feature_orchestrator.use_gpu = False
                if hasattr(self.feature_orchestrator, "computer"):
                    self.feature_orchestrator.computer.use_gpu = False
                if hasattr(self.feature_orchestrator, "config"):
                    if hasattr(self.feature_orchestrator.config, "processor"):
                        self.feature_orchestrator.config.processor.use_gpu = False
                    if hasattr(self.feature_orchestrator.config, "features"):
                        if hasattr(
                            self.feature_orchestrator.config.features,
                            "gpu_optimization",
                        ):
                            self.feature_orchestrator.config.features.gpu_optimization.enabled = (
                                False
                            )

            logger.info(f"🚀 Processing with {num_workers} parallel workers")
            logger.info("=" * 70)

            # For parallel processing, we can't easily pass tile index
            process_func = partial(
                self.process_tile,
                output_dir=output_dir,
                total_tiles=total_tiles,
                skip_existing=skip_existing,
            )

            with mp.Pool(num_workers) as pool:
                results = list(
                    tqdm(
                        pool.imap(process_func, laz_files),
                        total=total_tiles,
                        desc="Processing tiles",
                        unit="tile",
                    )
                )

            # Handle both dict and int return types for backwards compatibility
            num_patches_list = []
            for r in results:
                if isinstance(r, dict):
                    num_patches_list.append(r.get("num_patches", 0))
                else:
                    num_patches_list.append(r)

            total_patches = sum(num_patches_list)
            tiles_skipped = sum(1 for n in num_patches_list if n == 0)
            tiles_processed = total_tiles - tiles_skipped
        else:
            # ✅ OPTIMIZATION: Sequential processing with tile-by-tile prefetching (Phase 2)
            # Double-buffering: Load tile N+1 and prefetch ground truth N+1 while GPU processes tile N
            # Expected speedup: +40-80% (eliminates both I/O stalls and ground truth fetch delays)
            logger.info("🔄 Processing sequentially with tile-by-tile prefetching")
            logger.info("=" * 70)

            total_patches = 0

            # Use ThreadPoolExecutor for async I/O and ground truth prefetching
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(
                max_workers=3
            ) as io_pool:  # 3 workers: tile loading + ground truth fetching + spare
                # Prefetch first tile and its ground truth
                if laz_files:
                    next_tile_future = io_pool.submit(
                        self.tile_loader.load_tile, laz_files[0], max_retries=2
                    )
                    # Prefetch ground truth for first tile if data fetcher is available
                    if self.data_fetcher is not None:
                        next_ground_truth_future = io_pool.submit(
                            self._prefetch_ground_truth_for_tile, laz_files[0]
                        )
                    else:
                        next_ground_truth_future = None
                else:
                    next_tile_future = None
                    next_ground_truth_future = None

                for idx, laz_file in enumerate(laz_files, 1):
                    # Wait for prefetched tile data and ground truth
                    tile_data = (
                        next_tile_future.result()
                        if next_tile_future is not None
                        else None
                    )
                    prefetched_ground_truth = (
                        next_ground_truth_future.result()
                        if next_ground_truth_future is not None
                        else None
                    )

                    # Start prefetching NEXT tile and ground truth (async I/O, parallel with GPU processing)
                    if idx < len(laz_files):
                        next_tile_future = io_pool.submit(
                            self.tile_loader.load_tile, laz_files[idx], max_retries=2
                        )
                        # Prefetch ground truth for next tile if data fetcher is available
                        if self.data_fetcher is not None:
                            next_ground_truth_future = io_pool.submit(
                                self._prefetch_ground_truth_for_tile, laz_files[idx]
                            )
                        else:
                            next_ground_truth_future = None
                    else:
                        next_tile_future = None
                        next_ground_truth_future = None

                    # Process current tile (GPU busy, I/O and ground truth fetching run in parallel)
                    result = self._process_tile_with_data(
                        laz_file,
                        output_dir,
                        tile_data,
                        tile_idx=idx,
                        total_tiles=total_tiles,
                        skip_existing=skip_existing,
                        prefetched_ground_truth=prefetched_ground_truth,
                    )

                    # Handle both dict and int return types for backwards compatibility
                    if isinstance(result, dict):
                        num_patches = result.get("num_patches", 0)
                    else:
                        num_patches = result

                    total_patches += num_patches

                    if num_patches == 0:
                        tiles_skipped += 1
                    else:
                        tiles_processed += 1

                # ✅ OPTIMIZATION: Smart garbage collection (Phase 1 - Quick Win)
                # Fast cleanup after each tile (generation 0 only - ~5ms)
                del result
                if "tile_data" in locals():
                    del tile_data
                if "all_features" in locals():
                    del all_features
                gc.collect(generation=0)

                # Deep cleanup every 10 tiles (GPU + full GC)
                if idx % 10 == 0:
                    # GPU memory cleanup (if using CuPy)
                    try:
                        import cupy as cp

                        mempool = cp.get_default_memory_pool()
                        pinned_mempool = cp.get_default_pinned_memory_pool()
                        mempool.free_all_blocks()
                        pinned_mempool.free_all_blocks()
                        logger.debug("  🎮 GPU memory freed")
                    except:
                        pass

                    # Full CPU garbage collection
                    gc.collect()

                    # Log memory status
                    try:
                        import psutil

                        mem = psutil.virtual_memory()
                        logger.debug(
                            f"  💾 System Memory: {mem.available/(1024**3):.1f}GB available"
                        )
                        try:
                            import cupy as cp

                            free_mem, total_mem = cp.cuda.Device().mem_info
                            logger.debug(
                                f"  🎮 GPU Memory: {free_mem/(1024**3):.1f}GB / {total_mem/(1024**3):.1f}GB free"
                            )
                        except:
                            pass
                    except ImportError:
                        pass

        logger.info("")
        logger.info("=" * 70)
        logger.info("📊 Processing Summary:")
        logger.info(f"  Total tiles: {total_tiles}")
        logger.info(f"  ✅ Processed: {tiles_processed}")
        logger.info(f"  ⏭️  Skipped: {tiles_skipped}")
        logger.info(f"  📦 Total patches created: {total_patches}")
        logger.info("=" * 70)

        # Save metadata
        if metadata_mgr:
            processing_time = time.time() - start_time
            stats = metadata_mgr.create_processing_stats(
                input_dir=input_dir,
                num_tiles=len(laz_files),
                num_patches=total_patches,
                lod_level=self.lod_level,
                k_neighbors=self.k_neighbors,
                patch_size=self.patch_size,
                augmentation=self.augment,
                num_augmentations=self.num_augmentations,
            )
            stats["processing_time_seconds"] = round(processing_time, 2)
            metadata_mgr.save_stats(stats)

        # Save dataset metadata if dataset manager is enabled
        if self.dataset_manager is not None:
            processing_time = time.time() - start_time
            additional_info = {
                "lod_level": self.lod_level,
                "architecture": self.architecture,
                "patch_size_meters": self.patch_size,
                "num_points": self.num_points,
                "augmentation_enabled": self.config.processor.get("augment", False),
                "num_augmentations": self.config.processor.get("num_augmentations", 0),
                "processing_time_seconds": round(processing_time, 2),
                "tiles_processed": tiles_processed,
                "tiles_skipped": tiles_skipped,
            }
            self.dataset_manager.save_metadata(additional_info=additional_info)

        return total_patches

    def reclassify_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        cache_dir: Optional[Path] = None,
        chunk_size: int = 100000,
        show_progress: bool = True,
        skip_existing: bool = True,
        acceleration_mode: str = "auto",
    ) -> int:
        """
        Reclassify all LAZ files in a directory with BD TOPO® ground truth.

        This is an optimized mode that only updates classification codes using
        spatial indexing for fast processing. Use this when you already have
        enriched tiles but need to apply/update ground truth classification.

        Acceleration modes:
        - 'cpu': Use CPU with STRtree spatial indexing (~5-10 min for 18M points)
        - 'gpu': Use RAPIDS cuSpatial GPU acceleration (~1-2 min for 18M points)
        - 'gpu+cuml': Use full RAPIDS stack (~30-60 sec for 18M points)
        - 'auto': Automatically select best available backend (recommended)

        Args:
            input_dir: Directory containing enriched LAZ files
            output_dir: Output directory for reclassified files
            cache_dir: Cache directory for ground truth data (default: data/cache)
            chunk_size: Points per processing chunk (default: 100,000)
            show_progress: Show progress bars (default: True)
            skip_existing: Skip already reclassified files (default: True)
            acceleration_mode: Acceleration backend ('cpu', 'gpu', 'gpu+cuml', 'auto')

        Returns:
            Number of files processed
        """
        from ..io.data_fetcher import DataFetchConfig, DataFetcher

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if cache_dir is None:
            cache_dir = Path("data/cache")

        logger.info("=" * 80)
        logger.info("🔄 RECLASSIFICATION MODE - Optimized Ground Truth Application")
        logger.info("=" * 80)
        logger.info(f"Input:  {input_dir}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Cache:  {cache_dir}")
        logger.info(f"Chunk size: {chunk_size:,} points")
        logger.info("=" * 80)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all LAZ files - exclude enriched files to avoid reprocessing
        laz_files = sorted(input_dir.glob("*.laz"))
        laz_files = [f for f in laz_files if not f.stem.endswith("_enriched")]

        if not laz_files:
            logger.warning(f"No LAZ files found in {input_dir}")
            return 0

        logger.info(f"Found {len(laz_files)} LAZ files to process")

        # Initialize data fetcher with all BD TOPO features
        logger.info("\n📍 Initializing BD TOPO® data fetcher...")

        config = DataFetchConfig(
            include_buildings=True,
            include_roads=True,
            include_railways=True,
            include_water=True,
            include_vegetation=True,
            include_bridges=True,
            include_parking=True,
            include_cemeteries=True,
            include_power_lines=True,
            include_sports=True,
            include_forest=False,
            include_agriculture=False,
            include_cadastre=False,
            road_width_fallback=4.0,
            railway_width_fallback=3.5,
            power_line_buffer=2.0,
        )

        data_fetcher = DataFetcher(cache_dir=cache_dir, config=config)

        # Initialize reclassifier with acceleration mode
        reclassifier = OptimizedReclassifier(
            chunk_size=chunk_size,
            show_progress=show_progress,
            acceleration_mode=acceleration_mode,
        )

        # Process each file
        tiles_processed = 0
        tiles_skipped = 0
        start_time = time.time()

        for idx, laz_file in enumerate(laz_files, 1):
            logger.info(f"\n[{idx}/{len(laz_files)}] Processing: {laz_file.name}")

            # Output file path
            output_file = output_dir / f"{laz_file.stem}_reclassified.laz"

            # Skip if already exists
            if skip_existing and output_file.exists():
                logger.info(f"  ⏭️  Skipped (already exists): {output_file.name}")
                tiles_skipped += 1
                continue

            try:
                # 1. Load file to get bbox
                las = laspy.read(str(laz_file))
                points = np.vstack([las.x, las.y, las.z]).T

                bbox = (
                    float(points[:, 0].min()),
                    float(points[:, 1].min()),
                    float(points[:, 0].max()),
                    float(points[:, 1].max()),
                )

                logger.info(f"  📦 Points: {len(points):,}")
                logger.info(f"  📏 Bbox: {bbox}")

                # 2. Fetch ground truth for this tile's bbox
                logger.info(f"  🗺️  Fetching ground truth...")
                gt_data = data_fetcher.fetch_all(bbox=bbox, use_cache=True)

                if not gt_data or "ground_truth" not in gt_data:
                    logger.warning(f"  ⚠️  No ground truth data available, skipping")
                    tiles_skipped += 1
                    continue

                ground_truth_features = gt_data["ground_truth"]

                # Log what was fetched
                n_features = sum(
                    1
                    for gdf in ground_truth_features.values()
                    if gdf is not None and len(gdf) > 0
                )
                logger.info(f"  ✓ Fetched {n_features} feature types")

                # 3. Reclassify
                logger.info(f"  🎯 Reclassifying...")
                stats = reclassifier.reclassify_file(
                    input_laz=laz_file,
                    output_laz=output_file,
                    ground_truth_features=ground_truth_features,
                )

                tiles_processed += 1
                logger.info(f"  ✅ Completed: {output_file.name}")

            except Exception as e:
                logger.error(f"  ❌ Failed: {e}", exc_info=True)
                tiles_skipped += 1
                continue

            # Garbage collection every 5 tiles
            if idx % 5 == 0:
                gc.collect()

        # Summary
        processing_time = time.time() - start_time

        logger.info("\n" + "=" * 80)
        logger.info("📊 Reclassification Summary:")
        logger.info(f"  Total files: {len(laz_files)}")
        logger.info(f"  ✅ Processed: {tiles_processed}")
        logger.info(f"  ⏭️  Skipped: {tiles_skipped}")
        logger.info(f"  ⏱️  Time: {processing_time:.1f}s")
        logger.info("=" * 80)

        return tiles_processed
