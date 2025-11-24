"""
Classification modules for LiDAR point cloud classification.

This package contains classification components for ASPRS/BD TOPO point cloud labeling,
ground truth refinement, and advanced classification strategies.

v3.2+ Changes:
    - BaseClassifier interface for all classifiers
    - ClassificationResult standardized return type
    - Single Classifier facade for easy access

Quick Start (v3.2+):
    >>> from ign_lidar.core.classification import Classifier
    >>> classifier = Classifier(mode='lod2', strategy='adaptive')
    >>> result = classifier.classify(points, features)

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

from .base import BaseClassifier, ClassificationResult

# Note: FeatureManager and FeatureComputer have been consolidated into FeatureOrchestrator
# in ign_lidar.features.orchestrator (Phase 4.3)
from .config_validator import ConfigValidator, ProcessingMode
from .enrichment import (
    EnrichmentConfig,
    EnrichmentResult,
    compute_geometric_features_boundary_aware,
    compute_geometric_features_standard,
    compute_ndvi,
    enrich_point_cloud,
    fetch_infrared,
    fetch_rgb_colors,
)
from .io import (
    LiDARCorruptionError,
    LiDARData,
    LiDARLoadError,
    TileLoader,
    get_tile_info,
    load_laz_file,
    map_classification,
    save_patch_hdf5,
    save_patch_laz,
    save_patch_multi_format,
    save_patch_npz,
    save_patch_torch,
    validate_format_support,
    validate_lidar_data,
)
from .memory import aggressive_memory_cleanup, clear_gpu_cache
from .patch_extractor import (
    AugmentationConfig,
    PatchConfig,
    augment_patch,
    augment_raw_points,
    create_patch_versions,
    extract_and_augment_patches,
    extract_patches,
    format_patch_for_architecture,
    resample_patch,
)
from .stitching import (
    StitchingConfig,
    TileStitcher,
    check_neighbors_available,
    compute_boundary_aware_features,
    create_stitcher,
    extract_and_normalize_features,
    get_stitching_stats,
    should_use_stitching,
)

# ============================================================================
# Classification Interface (v3.2+)
# ============================================================================

# Unified Classification Engine (NEW in v3.6.0)
try:
    from .engine import (
        ClassificationEngine,
        ClassificationMode,
        ClassificationStrategy,
        SpectralClassificationStrategy,
        GeometricClassificationStrategy,
        ASPRSClassificationStrategy,
    )

    _HAS_CLASSIFICATION_ENGINE = True
except ImportError:
    _HAS_CLASSIFICATION_ENGINE = False
    ClassificationEngine = None
    ClassificationMode = None

# Reclassification modules (optional - may not be available)
try:
    from .geometric_rules import GeometricRulesEngine
    from .reclassifier import Reclassifier, reclassify_tile

    _HAS_RECLASSIFIER = True
except ImportError:
    _HAS_RECLASSIFIER = False
    Reclassifier = None
    reclassify_tile = None
    GeometricRulesEngine = None

# Ground truth refinement module (new in v5.2)
try:
    from .ground_truth_refinement import GroundTruthRefinementConfig, GroundTruthRefiner

    _HAS_GT_REFINEMENT = True
except ImportError:
    _HAS_GT_REFINEMENT = False
    GroundTruthRefiner = None
    GroundTruthRefinementConfig = None

# Ground truth artifact detection module (new in v5.0)
try:
    from .ground_truth_artifact_checker import (
        ArtifactReport,
        GroundTruthArtifactChecker,
        get_artifact_free_features,
        validate_features_before_classification,
    )

    _HAS_ARTIFACT_CHECKER = True
except ImportError:
    _HAS_ARTIFACT_CHECKER = False
    GroundTruthArtifactChecker = None
    validate_features_before_classification = None
    get_artifact_free_features = None
    ArtifactReport = None

# Classifier module (v3.1.0 consolidation, renamed in v3.3.0)
try:
    from .classifier import (
        Classifier,
        ClassifierConfig,
        ClassificationRule,
        ClassificationStrategy,
        FeatureImportance,
        classify_points,
        refine_classification,
    )

    _HAS_CLASSIFIER = True
except ImportError:
    _HAS_CLASSIFIER = False
    Classifier = None
    ClassifierConfig = None
    ClassificationStrategy = None
    ClassificationRule = None
    FeatureImportance = None
    classify_points = None
    refine_classification = None

# Backward compatibility removed in v3.1.0
# Use Classifier instead:
#   - AdvancedClassifier → Classifier(strategy='comprehensive')
#   - AdaptiveClassifier → Classifier(strategy='adaptive')
#   - refine_classification() → Classifier().refine_classification()

_HAS_ADAPTIVE_CLASSIFIER = _HAS_CLASSIFIER

# Convenience function for creating classifier with common settings
if _HAS_CLASSIFIER:

    def create_classifier(strategy="comprehensive", use_gpu=False, **kwargs):
        """
        Convenience function to create a classifier with common settings.

        Args:
            strategy: 'basic', 'adaptive', or 'comprehensive'
            use_gpu: Enable GPU acceleration (requires CuPy)
            **kwargs: Additional Classifier parameters

        Returns:
            Classifier instance

        Example:
            >>> from ign_lidar.core.classification import create_classifier
            >>> classifier = create_classifier('adaptive', use_gpu=True)
            >>> result = classifier.classify(points, features)
        """
        return Classifier(strategy=strategy, use_gpu=use_gpu, **kwargs)


# Adaptive building classifier module (new in v5.2.2 - building classification)
try:
    from .adaptive_building_classifier import (
        AdaptiveBuildingClassifier,
        BuildingFeatureSignature,
        ClassificationConfidence,
        PointBuildingScore,
    )

    _HAS_ADAPTIVE_BUILDING = True
except ImportError:
    _HAS_ADAPTIVE_BUILDING = False
    AdaptiveBuildingClassifier = None
    BuildingFeatureSignature = None
    PointBuildingScore = None
    ClassificationConfidence = None

__all__ = [
    # ========================================================================
    # Classification Interface (v3.2+) - Use these!
    # ========================================================================
    "Classifier",  # ← Main entry point
    "BaseClassifier",
    "ClassificationResult",
    "create_classifier",  # ← Convenience function
    # ========================================================================
    # Memory & Core Utilities
    # ========================================================================
    # Memory management
    "aggressive_memory_cleanup",
    "clear_gpu_cache",
    # Configuration and management (Phase 3.3)
    "ConfigValidator",
    "ProcessingMode",
    # Tile loading (Phase 3.4)
    "TileLoader",
    # Serialization
    "save_patch_npz",
    "save_patch_hdf5",
    "save_patch_torch",
    "save_patch_laz",
    "save_patch_multi_format",
    "validate_format_support",
    # Loader
    "LiDARData",
    "LiDARLoadError",
    "LiDARCorruptionError",
    "load_laz_file",
    "validate_lidar_data",
    "map_classification",
    "get_tile_info",
    # Enrichment
    "EnrichmentConfig",
    "EnrichmentResult",
    "fetch_rgb_colors",
    "fetch_infrared",
    "compute_ndvi",
    "compute_geometric_features_standard",
    "compute_geometric_features_boundary_aware",
    "enrich_point_cloud",
    # Patch extraction
    "PatchConfig",
    "AugmentationConfig",
    "extract_patches",
    "resample_patch",
    "augment_raw_points",
    "augment_patch",
    "create_patch_versions",
    "format_patch_for_architecture",
    "extract_and_augment_patches",
    # Stitching
    "TileStitcher",
    "StitchingConfig",
    "create_stitcher",
    "check_neighbors_available",
    "compute_boundary_aware_features",
    "extract_and_normalize_features",
    "should_use_stitching",
    "get_stitching_stats",
    # Classification Engine (new in v3.6.0)
    "ClassificationEngine",
    "ClassificationMode",
    "ClassificationStrategy",
    "SpectralClassificationStrategy",
    "GeometricClassificationStrategy",
    "ASPRSClassificationStrategy",
    # Reclassification (optional)
    "Reclassifier",
    "reclassify_tile",
    "GeometricRulesEngine",
    # Ground truth refinement (optional, v5.2)
    "GroundTruthRefiner",
    "GroundTruthRefinementConfig",
    # Ground truth artifact detection (optional, v5.0)
    "GroundTruthArtifactChecker",
    "validate_features_before_classification",
    "get_artifact_free_features",
    "ArtifactReport",
    # Classifier (v3.1.0, renamed in v3.3.0)
    "Classifier",
    "ClassifierConfig",
    "ClassificationStrategy",
    "classify_points",
    "refine_classification",
    "ClassificationRule",
    "FeatureImportance",
    "create_classifier",
    # Adaptive building classifier (optional, v5.2.2)
    "AdaptiveBuildingClassifier",
    "BuildingFeatureSignature",
    "PointBuildingScore",
    "ClassificationConfidence",
]

# DTM Augmentation module (new in v3.1.0 - MNT integration)
try:
    from .dtm_augmentation import (
        AugmentationArea,
        AugmentationStrategy,
        DTMAugmentationConfig,
        DTMAugmentationStats,
        DTMAugmenter,
        augment_with_dtm,
    )

    _HAS_DTM_AUGMENTATION = True
except ImportError:
    _HAS_DTM_AUGMENTATION = False
    DTMAugmenter = None
    DTMAugmentationConfig = None
    DTMAugmentationStats = None
    AugmentationStrategy = None
    AugmentationArea = None
    augment_with_dtm = None

# Export DTM augmentation
__all__ += [
    "DTMAugmenter",
    "DTMAugmentationConfig",
    "DTMAugmentationStats",
    "AugmentationStrategy",
    "AugmentationArea",
    "augment_with_dtm",
]
