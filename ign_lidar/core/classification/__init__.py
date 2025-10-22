"""
Classification modules for LiDAR point cloud classification.

This package contains classification components for ASPRS/BD TOPO point cloud labeling,
ground truth refinement, and advanced classification strategies.

📍 **Note**: Relocated from `core.modules` to `core.classification` in v3.1.0 for better
semantic clarity. The old import path is deprecated but still works in v3.x.

Key Modules:
    Classification & Rules:
        - advanced_classification: Advanced classifier with ASPRS/BD TOPO integration
        - classification_thresholds: Classification threshold management
        - classification_refinement: Post-classification refinement
        - adaptive_classifier: Adaptive classification with ground truth
        - hierarchical_classifier: Hierarchical classification strategies
        - reclassifier: Optimized reclassification with GPU acceleration
        
    Ground Truth:
        - ground_truth_refinement: BD TOPO/cadastre ground truth integration
        - ground_truth_artifact_checker: Artifact detection and validation
        - parcel_classifier: Cadastral parcel-based classification
        
    Rule Engines:
        - geometric_rules: Geometric rules engine for classification
        - spectral_rules: Spectral rules engine (NIR, NDVI, etc.)
        - grammar_3d: 3D grammar rules for urban structures
        
    Support:
        - feature_validator: Feature validation and quality checks
        - memory: Memory management and cleanup utilities
        - serialization: Save/export functionality for patches and enriched data
        - loader: Data loading and validation
        - config_validator: Configuration validation and normalization
        - tile_loader: Tile loading and I/O operations

Migration:
    # Old (deprecated in v3.1.0)
    from ign_lidar.core.modules.classification_thresholds import ClassificationThresholds
    
    # New (recommended)
    from ign_lidar.core.classification.classification_thresholds import ClassificationThresholds

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

# Reclassification modules (optional - may not be available)
try:
    from .reclassifier import OptimizedReclassifier, reclassify_tile_optimized
    from .geometric_rules import GeometricRulesEngine
    _HAS_RECLASSIFIER = True
except ImportError:
    _HAS_RECLASSIFIER = False
    OptimizedReclassifier = None
    reclassify_tile_optimized = None
    GeometricRulesEngine = None

# Ground truth refinement module (new in v5.2)
try:
    from .ground_truth_refinement import GroundTruthRefiner, GroundTruthRefinementConfig
    _HAS_GT_REFINEMENT = True
except ImportError:
    _HAS_GT_REFINEMENT = False
    GroundTruthRefiner = None
    GroundTruthRefinementConfig = None

# Ground truth artifact detection module (new in v5.0)
try:
    from .ground_truth_artifact_checker import (
        GroundTruthArtifactChecker,
        validate_features_before_classification,
        get_artifact_free_features,
        ArtifactReport
    )
    _HAS_ARTIFACT_CHECKER = True
except ImportError:
    _HAS_ARTIFACT_CHECKER = False
    GroundTruthArtifactChecker = None
    validate_features_before_classification = None
    get_artifact_free_features = None
    ArtifactReport = None

# Unified classifier module (new in v3.1.0 - consolidation)
try:
    from .unified_classifier import (
        UnifiedClassifier,
        ClassificationStrategy,
        ClassificationRule,
        FeatureImportance,
        UnifiedClassifierConfig,
        classify_points_unified,
        refine_classification_unified
    )
    _HAS_UNIFIED_CLASSIFIER = True
except ImportError:
    _HAS_UNIFIED_CLASSIFIER = False
    UnifiedClassifier = None
    ClassificationStrategy = None
    ClassificationRule = None
    FeatureImportance = None
    UnifiedClassifierConfig = None
    classify_points_unified = None
    refine_classification_unified = None

# Backward compatibility removed in v3.1.0
# Use UnifiedClassifier instead:
#   - AdvancedClassifier → UnifiedClassifier(strategy='comprehensive')
#   - AdaptiveClassifier → UnifiedClassifier(strategy='adaptive')
#   - refine_classification() → UnifiedClassifier().refine_classification()

_HAS_ADAPTIVE_CLASSIFIER = _HAS_UNIFIED_CLASSIFIER

# Adaptive building classifier module (new in v5.2.2 - Enhanced building classification)
try:
    from .adaptive_building_classifier import (
        AdaptiveBuildingClassifier,
        BuildingFeatureSignature,
        PointBuildingScore,
        ClassificationConfidence
    )
    _HAS_ADAPTIVE_BUILDING = True
except ImportError:
    _HAS_ADAPTIVE_BUILDING = False
    AdaptiveBuildingClassifier = None
    BuildingFeatureSignature = None
    PointBuildingScore = None
    ClassificationConfidence = None

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
    # Reclassification (optional)
    'OptimizedReclassifier',
    'reclassify_tile_optimized',
    'GeometricRulesEngine',
    # Ground truth refinement (optional, v5.2)
    'GroundTruthRefiner',
    'GroundTruthRefinementConfig',
    # Ground truth artifact detection (optional, v5.0)
    'GroundTruthArtifactChecker',
    'validate_features_before_classification',
    'get_artifact_free_features',
    'ArtifactReport',
    # Unified classifier (v3.1.0 - new)
    'UnifiedClassifier',
    'ClassificationStrategy',
    'UnifiedClassifierConfig',
    'classify_points_unified',
    'refine_classification_unified',
    'ClassificationRule',
    'FeatureImportance',
    # Adaptive building classifier (optional, v5.2.2)
    'AdaptiveBuildingClassifier',
    'BuildingFeatureSignature',
    'PointBuildingScore',
    'ClassificationConfidence',
]
