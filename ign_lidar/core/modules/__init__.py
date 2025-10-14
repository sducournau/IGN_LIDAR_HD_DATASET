"""
Core processing modules for LiDAR data processing.

This package contains modular components extracted from the monolithic processor
to improve maintainability, testability, and code organization.

Modules:
    memory: Memory management and cleanup utilities
    serialization: Save/export functionality for patches and enriched data
    loader: Data loading and validation
    enrichment: Feature computation and enrichment
    patch_extractor: Patch extraction and augmentation
    stitching: Tile stitching and boundary processing
    config_validator: Configuration validation and normalization
    tile_loader: Tile loading and I/O operations (Phase 3.4)
    
Note: FeatureManager and FeatureComputer have been consolidated into 
      FeatureOrchestrator in ign_lidar.features.orchestrator (Phase 4.3)
"""

from .memory import aggressive_memory_cleanup, clear_gpu_cache
# Note: FeatureManager and FeatureComputer have been consolidated into FeatureOrchestrator
# in ign_lidar.features.orchestrator (Phase 4.3)
from .config_validator import ConfigValidator, ProcessingMode
from .tile_loader import TileLoader
from .serialization import (
    save_patch_npz,
    save_patch_hdf5,
    save_patch_torch,
    save_patch_laz,
    save_patch_multi_format,
    validate_format_support
)
from .loader import (
    LiDARData,
    LiDARLoadError,
    LiDARCorruptionError,
    load_laz_file,
    validate_lidar_data,
    map_classification,
    get_tile_info
)
from .enrichment import (
    EnrichmentConfig,
    EnrichmentResult,
    fetch_rgb_colors,
    fetch_infrared,
    compute_ndvi,
    compute_geometric_features_standard,
    compute_geometric_features_boundary_aware,
    enrich_point_cloud,
)
from .patch_extractor import (
    PatchConfig,
    AugmentationConfig,
    extract_patches,
    resample_patch,
    augment_raw_points,
    augment_patch,
    create_patch_versions,
    format_patch_for_architecture,
    extract_and_augment_patches,
)
from .stitching import (
    TileStitcher,
    StitchingConfig,
    create_stitcher,
    check_neighbors_available,
    compute_boundary_aware_features,
    extract_and_normalize_features,
    should_use_stitching,
    get_stitching_stats,
)

__all__ = [
    # Memory management
    'aggressive_memory_cleanup',
    'clear_gpu_cache',
    # Configuration and management (Phase 3.3)
    'ConfigValidator',
    'ProcessingMode',
    # Tile loading (Phase 3.4)
    'TileLoader',
    # Serialization
    'save_patch_npz',
    'save_patch_hdf5',
    'save_patch_torch',
    'save_patch_laz',
    'save_patch_multi_format',
    'validate_format_support',
    # Loader
    'LiDARData',
    'LiDARLoadError',
    'LiDARCorruptionError',
    'load_laz_file',
    'validate_lidar_data',
    'map_classification',
    'get_tile_info',
    # Enrichment
    'EnrichmentConfig',
    'EnrichmentResult',
    'fetch_rgb_colors',
    'fetch_infrared',
    'compute_ndvi',
    'compute_geometric_features_standard',
    'compute_geometric_features_boundary_aware',
    'enrich_point_cloud',
    # Patch extraction
    'PatchConfig',
    'AugmentationConfig',
    'extract_patches',
    'resample_patch',
    'augment_raw_points',
    'augment_patch',
    'create_patch_versions',
    'format_patch_for_architecture',
    'extract_and_augment_patches',
    # Stitching
    'TileStitcher',
    'StitchingConfig',
    'create_stitcher',
    'check_neighbors_available',
    'compute_boundary_aware_features',
    'extract_and_normalize_features',
    'should_use_stitching',
    'get_stitching_stats',
]
