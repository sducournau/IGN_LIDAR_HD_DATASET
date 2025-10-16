"""
IGN LiDAR HD Dataset Processing Library

A Python library for processing IGN LiDAR HD data into machine learning-ready datasets
with building LOD (Level of Detail) classification support.

Version 2.5.3 - Critical Fix Release:
- Fixed ground truth classification from BD TOPO® (roads, cemeteries, power lines, sports)
- Fixed ASPRS mode classification (was incorrectly using LOD3 mapping)
- Added proper DataFetcher integration with all BD TOPO features
- New data_sources configuration directory for multi-source integration
- Enhanced configuration files with consistent parameter structure

All ground truth classifications now work correctly across ASPRS, LOD2, and LOD3 modes.
"""

__version__ = "2.5.3"
__author__ = "imagodata"
__email__ = "simon.ducournau@google.com"

# ============================================================================
# NEW v2.0 IMPORTS (Recommended)
# ============================================================================

# Core processing modules
from .core.processor import LiDARProcessor

# Feature extraction
from .features import (
    compute_normals,
    compute_curvature,
    extract_geometric_features,
    compute_all_features_optimized,
    compute_all_features_with_gpu,
)

# Preprocessing
from .preprocessing import (
    statistical_outlier_removal,
    radius_outlier_removal,
    voxel_downsample,
    preprocess_point_cloud,
    add_rgb_to_patch,
    augment_tile_with_rgb,
    add_infrared_to_patch,
    augment_tile_with_infrared,
)

# Ground Truth (WFS) - Import on demand to avoid dependency issues
try:
    from .io.wfs_ground_truth import (
        IGNWFSConfig,
        IGNGroundTruthFetcher,
        fetch_ground_truth_for_tile,
        generate_patches_with_ground_truth,
    )
except ImportError:
    # shapely/geopandas not installed
    IGNWFSConfig = None
    IGNGroundTruthFetcher = None
    fetch_ground_truth_for_tile = None
    generate_patches_with_ground_truth = None

# Configuration (Hydra) - Import on demand to avoid dependency issues
try:
    from .config.schema import (
        IGNLiDARConfig,
        ProcessorConfig,
        FeaturesConfig,
        PreprocessConfig,
        StitchingConfig,
        OutputConfig,
    )
except ImportError:
    # Hydra dependencies not installed
    IGNLiDARConfig = None
    ProcessorConfig = None
    FeaturesConfig = None
    PreprocessConfig = None
    StitchingConfig = None
    OutputConfig = None

# ============================================================================
# BACKWARD COMPATIBILITY IMPORTS (Legacy - Still Supported)
# ============================================================================

# Root level modules (unchanged location)
from .downloader import IGNLiDARDownloader
from .classes import LOD2_CLASSES, LOD3_CLASSES
from .asprs_classes import (
    ASPRSClass,
    ASPRS_CLASS_NAMES,
    ClassificationMode,
    get_classification_for_building,
    get_classification_for_road,
    get_classification_for_vegetation,
    get_classification_for_water,
    get_classification_for_railway,
    get_classification_for_sports,
    get_classification_for_cemetery,
    get_classification_for_power_line,
    get_classification_for_parking,
    get_classification_for_bridge,
    get_class_name,
    get_class_color,
    RAILWAY_NATURE_TO_ASPRS,
    SPORTS_NATURE_TO_ASPRS,
    CEMETERY_NATURE_TO_ASPRS,
    POWER_LINE_NATURE_TO_ASPRS,
    PARKING_NATURE_TO_ASPRS,
    BRIDGE_NATURE_TO_ASPRS,
)

# Reorganized modules - backward compatibility imports
# Core utilities (moved to core/)
from .core import AdaptiveMemoryManager, MemoryConfig, PerformanceMonitor
from .core import ProcessingError, GPUMemoryError, MemoryPressureError

# Feature utilities (moved to features/)
from .features import (
    ARCHITECTURAL_STYLES,
    STYLE_NAME_TO_ID,
    get_tile_architectural_style,
    get_patch_architectural_style,
    compute_architectural_style_features,
)

# Dataset utilities (moved to datasets/)
from .datasets import STRATEGIC_LOCATIONS, WORKING_TILES
from .datasets import get_tiles_by_environment, get_tiles_by_priority, get_tiles_by_region

# IO utilities (moved to io/)
from .io import MetadataManager, simplify_for_qgis

# Preprocessing utilities (moved to preprocessing/)
from .preprocessing import augment_raw_points, extract_patches, analyze_tile

# Legacy imports for backward compatibility
# These point to the new locations
try:
    from .features.features import (
        compute_normals,
        compute_curvature,
        extract_geometric_features
    )
except ImportError:
    # Fallback to old location if new structure not complete
    from .features import (
        compute_normals,
        compute_curvature,
        extract_geometric_features
    )

# Backward compatibility for moved modules - these imports should be available 
# at the root level for legacy code
try:
    # Import core modules at root level for backward compatibility
    from .core.processor import LiDARProcessor as processor_LiDARProcessor
    from .core.tile_stitcher import TileStitcher
    
    # Make them available as if imported from root
    import sys
    import types
    
    # Create processor module for backward compatibility
    processor_module = types.ModuleType('ign_lidar.processor')
    processor_module.LiDARProcessor = processor_LiDARProcessor
    sys.modules['ign_lidar.processor'] = processor_module
    
    # Create tile_stitcher module for backward compatibility  
    tile_stitcher_module = types.ModuleType('ign_lidar.tile_stitcher')
    tile_stitcher_module.TileStitcher = TileStitcher
    sys.modules['ign_lidar.tile_stitcher'] = tile_stitcher_module
    
except ImportError:
    pass

__all__ = [
    # ========== Core v2.0 ==========
    # Processor
    "LiDARProcessor",
    
    # Features
    "compute_normals",
    "compute_curvature",
    "extract_geometric_features",
    "compute_all_features_optimized",
    "compute_all_features_with_gpu",
    
    # Preprocessing
    "statistical_outlier_removal",
    "radius_outlier_removal",
    "voxel_downsample",
    "preprocess_point_cloud",
    "add_rgb_to_patch",
    "augment_tile_with_rgb",
    "add_infrared_to_patch",
    "augment_tile_with_infrared",
    
    # Ground Truth (WFS)
    "IGNWFSConfig",
    "IGNGroundTruthFetcher",
    "fetch_ground_truth_for_tile",
    "generate_patches_with_ground_truth",
    
    # Configuration
    "IGNLiDARConfig",
    "ProcessorConfig",
    "FeaturesConfig",
    "PreprocessConfig",
    "StitchingConfig",
    "OutputConfig",
    
    # ========== Root Level (Core Package) ==========
    # Downloader
    "IGNLiDARDownloader",
    
    # Classification
    "LOD2_CLASSES",
    "LOD3_CLASSES",
    "ASPRSClass",
    "ASPRS_CLASS_NAMES",
    "ClassificationMode",
    "get_classification_for_building",
    "get_classification_for_road",
    "get_classification_for_vegetation",
    "get_classification_for_water",
    "get_classification_for_railway",
    "get_classification_for_sports",
    "get_classification_for_cemetery",
    "get_classification_for_power_line",
    "get_classification_for_parking",
    "get_classification_for_bridge",
    "get_class_name",
    "get_class_color",
    "RAILWAY_NATURE_TO_ASPRS",
    "SPORTS_NATURE_TO_ASPRS",
    "CEMETERY_NATURE_TO_ASPRS",
    "POWER_LINE_NATURE_TO_ASPRS",
    "PARKING_NATURE_TO_ASPRS",
    "BRIDGE_NATURE_TO_ASPRS",
    
    # ========== Reorganized Modules (Backward Compatibility) ==========
    # Core utilities
    "AdaptiveMemoryManager",
    "MemoryConfig", 
    "PerformanceMonitor",
    "ProcessingError",
    "GPUMemoryError",
    "MemoryPressureError",
    "TileStitcher",  # Backward compatibility for tile stitching
    
    # Feature utilities
    "ARCHITECTURAL_STYLES",
    "STYLE_NAME_TO_ID",
    "get_tile_architectural_style",
    "get_patch_architectural_style",
    "compute_architectural_style_features",
    
    # Dataset utilities
    "STRATEGIC_LOCATIONS",
    "WORKING_TILES",
    "get_tiles_by_environment",
    "get_tiles_by_priority",
    "get_tiles_by_region",
    
    # IO utilities
    "MetadataManager",
    "simplify_for_qgis",
    
    # Preprocessing utilities
    "augment_raw_points",
    "extract_patches",
    "analyze_tile",
]