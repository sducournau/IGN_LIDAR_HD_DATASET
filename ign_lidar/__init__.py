"""
IGN LiDAR HD Dataset Processing Library

A Python library for processing IGN LiDAR HD data into machine learning-ready datasets
with building LOD (Level of Detail) classification support.

Version 3.3.3 - Gap Detection Enhancement:
- Automatic gap/void detection in building perimeters
- Angular sector analysis (36 sectors = 10° each)
- Gap-based adaptive buffer adjustment
- Quality metrics for building detection
- Directional gap identification (N, NE, E, SE, S, SW, W, NW)
- Export of problematic buildings for manual review

Version 3.0.0 - Major Release:
- Complete configuration system overhaul with unified v4.0 schema
- Enhanced GPU optimization with significantly improved utilization
- Streamlined presets for common processing scenarios  
- Better hardware-specific configurations and performance tuning
- Improved documentation and migration tools from legacy versions

All features now work with the new unified configuration architecture.
"""

__version__ = "3.3.3"
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

# NEW v3.1: Unified classification schema (consolidates asprs_classes.py + classes.py)
from .classification_schema import (
    ASPRSClass,
    LOD2Class,
    LOD3Class,
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
    LOD2_CLASSES,  # Backward compatibility
    LOD3_CLASSES,  # Backward compatibility
    ASPRS_TO_LOD2,
    ASPRS_TO_LOD3,
)

# Backward compatibility: Keep old imports working
# DEPRECATED: These will be removed in v4.0
try:
    # Create backward compatibility modules with deprecation warnings
    import sys
    import types
    import warnings
    
    # Helper class to add deprecation warnings to module access
    class _DeprecatedModule(types.ModuleType):
        """Module wrapper that issues deprecation warnings on first access."""
        
        def __init__(self, name, new_location, removal_version='4.0'):
            super().__init__(name)
            self._warned = False
            self._new_location = new_location
            self._removal_version = removal_version
            self._attributes = {}
        
        def __setattr__(self, name, value):
            if name.startswith('_'):
                # Internal attributes
                super().__setattr__(name, value)
            else:
                # Public attributes - store in dict
                if not hasattr(self, '_attributes'):
                    super().__setattr__('_attributes', {})
                self._attributes[name] = value
        
        def __getattr__(self, name):
            if not self._warned and not name.startswith('_'):
                warnings.warn(
                    f"\n{'='*70}\n"
                    f"DEPRECATION WARNING\n"
                    f"{'='*70}\n"
                    f"Importing from '{self.__name__}' is deprecated.\n"
                    f"Use '{self._new_location}' instead.\n"
                    f"This compatibility layer will be removed in version {self._removal_version}.\n\n"
                    f"Migration:\n"
                    f"  OLD: from {self.__name__} import {name}\n"
                    f"  NEW: from {self._new_location} import {name}\n"
                    f"{'='*70}",
                    DeprecationWarning,
                    stacklevel=2
                )
                self._warned = True
            
            # Return from attributes dict
            if name in self._attributes:
                return self._attributes[name]
            raise AttributeError(f"module '{self.__name__}' has no attribute '{name}'")
    
    # classes.py compatibility (DEPRECATED)
    classes_module = _DeprecatedModule(
        'ign_lidar.classes',
        'ign_lidar.classification_schema',
        removal_version='4.0'
    )
    classes_module.LOD2_CLASSES = LOD2_CLASSES
    classes_module.LOD3_CLASSES = LOD3_CLASSES
    classes_module.ASPRS_TO_LOD2 = ASPRS_TO_LOD2
    classes_module.ASPRS_TO_LOD3 = ASPRS_TO_LOD3
    sys.modules['ign_lidar.classes'] = classes_module
    
    # asprs_classes.py compatibility (DEPRECATED)
    asprs_classes_module = _DeprecatedModule(
        'ign_lidar.asprs_classes',
        'ign_lidar.classification_schema',
        removal_version='4.0'
    )
    asprs_classes_module.ASPRSClass = ASPRSClass
    asprs_classes_module.ASPRS_CLASS_NAMES = ASPRS_CLASS_NAMES
    asprs_classes_module.ClassificationMode = ClassificationMode
    asprs_classes_module.get_classification_for_building = get_classification_for_building
    asprs_classes_module.get_classification_for_road = get_classification_for_road
    asprs_classes_module.get_classification_for_vegetation = get_classification_for_vegetation
    asprs_classes_module.get_classification_for_water = get_classification_for_water
    asprs_classes_module.get_classification_for_railway = get_classification_for_railway
    asprs_classes_module.get_classification_for_sports = get_classification_for_sports
    asprs_classes_module.get_classification_for_cemetery = get_classification_for_cemetery
    asprs_classes_module.get_classification_for_power_line = get_classification_for_power_line
    asprs_classes_module.get_classification_for_parking = get_classification_for_parking
    asprs_classes_module.get_classification_for_bridge = get_classification_for_bridge
    asprs_classes_module.get_class_name = get_class_name
    asprs_classes_module.get_class_color = get_class_color
    asprs_classes_module.RAILWAY_NATURE_TO_ASPRS = RAILWAY_NATURE_TO_ASPRS
    asprs_classes_module.SPORTS_NATURE_TO_ASPRS = SPORTS_NATURE_TO_ASPRS
    asprs_classes_module.CEMETERY_NATURE_TO_ASPRS = CEMETERY_NATURE_TO_ASPRS
    asprs_classes_module.POWER_LINE_NATURE_TO_ASPRS = POWER_LINE_NATURE_TO_ASPRS
    asprs_classes_module.PARKING_NATURE_TO_ASPRS = PARKING_NATURE_TO_ASPRS
    asprs_classes_module.BRIDGE_NATURE_TO_ASPRS = BRIDGE_NATURE_TO_ASPRS
    sys.modules['ign_lidar.asprs_classes'] = asprs_classes_module
    
except Exception:
    # If backward compatibility setup fails, continue anyway
    pass

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

# Feature extraction - use modern core modules directly
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