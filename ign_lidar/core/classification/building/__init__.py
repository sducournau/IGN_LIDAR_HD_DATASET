"""
Building Classification Module

This module provides comprehensive building classification functionality including:
- Adaptive building classification with ground truth guidance
- Multi-mode building detection (ASPRS, LOD2, LOD3)
- Building point clustering by footprint
- Multi-source building polygon fusion

The module is organized into:
- base: Abstract base classes and common interfaces
- utils: Shared utility functions
- adaptive: AdaptiveBuildingClassifier
- detection: BuildingDetector
- clustering: BuildingClusterer
- fusion: BuildingFusion

Usage:
    from ign_lidar.core.classification.building import AdaptiveBuildingClassifier

    classifier = AdaptiveBuildingClassifier(mode='asprs')
    result = classifier.classify(points)

Author: Phase 2 - Building Module Restructuring
Date: October 22, 2025
"""

# Base classes and enumerations
from .base import (
    BuildingMode,
    BuildingSource,
    ClassificationConfidence,
    BuildingConfigBase,
    BuildingClassificationResult,
    BuildingClassifierBase,
    BuildingDetectorBase,
    BuildingClustererBase,
    BuildingFusionBase,
)

# Utility functions
from . import utils

# Concrete implementations
try:
    from .adaptive import (
        AdaptiveBuildingClassifier,
        BuildingFeatureSignature,
        PointBuildingScore,
    )

    _HAS_ADAPTIVE = True
except ImportError as e:
    import logging

    logging.getLogger(__name__).warning(f"Failed to import adaptive module: {e}")
    AdaptiveBuildingClassifier = None
    BuildingFeatureSignature = None
    PointBuildingScore = None
    _HAS_ADAPTIVE = False

try:
    from .detection import (
        BuildingDetector,
        BuildingDetectionMode,
        BuildingDetectionConfig,
    )

    _HAS_DETECTION = True
except ImportError:
    BuildingDetector = None
    BuildingDetectionMode = None
    BuildingDetectionConfig = None
    _HAS_DETECTION = False

try:
    from .clustering import BuildingClusterer, BuildingCluster

    _HAS_CLUSTERING = True
except ImportError:
    BuildingClusterer = None
    BuildingCluster = None
    _HAS_CLUSTERING = False

try:
    from .fusion import BuildingFusion, PolygonQuality

    _HAS_FUSION = True
except ImportError:
    BuildingFusion = None
    PolygonQuality = None
    _HAS_FUSION = False

try:
    from .extrusion_3d import (
        Building3DExtruder,
        BoundingBox3D,
        FloorSegment,
        create_3d_bboxes_from_ground_truth,
    )

    _HAS_EXTRUSION_3D = True
except ImportError as e:
    import logging

    logging.getLogger(__name__).warning(f"Failed to import 3D extrusion module: {e}")
    Building3DExtruder = None
    BoundingBox3D = None
    FloorSegment = None
    create_3d_bboxes_from_ground_truth = None
    _HAS_EXTRUSION_3D = False

# Phase 2 LOD3 Detectors (v3.1-3.3)
try:
    from .roof_classifier import (
        RoofTypeClassifier,
        RoofType,
        RoofSegment,
        RoofClassificationResult,
    )

    _HAS_ROOF_CLASSIFIER = True
except ImportError as e:
    import logging

    logging.getLogger(__name__).warning(f"Failed to import roof classifier: {e}")
    RoofTypeClassifier = None
    RoofType = None
    RoofSegment = None
    RoofClassificationResult = None
    _HAS_ROOF_CLASSIFIER = False

try:
    from .chimney_detector import (
        ChimneyDetector,
        SuperstructureType,
        SuperstructureSegment,
        ChimneyDetectionResult,
    )

    _HAS_CHIMNEY_DETECTOR = True
except ImportError as e:
    import logging

    logging.getLogger(__name__).warning(f"Failed to import chimney detector: {e}")
    ChimneyDetector = None
    SuperstructureType = None
    SuperstructureSegment = None
    ChimneyDetectionResult = None
    _HAS_CHIMNEY_DETECTOR = False

try:
    from .balcony_detector import (
        BalconyDetector,
        ProtrusionType,
        ProtrusionSegment,
        BalconyDetectionResult,
    )

    _HAS_BALCONY_DETECTOR = True
except ImportError as e:
    import logging

    logging.getLogger(__name__).warning(f"Failed to import balcony detector: {e}")
    BalconyDetector = None
    ProtrusionType = None
    ProtrusionSegment = None
    BalconyDetectionResult = None
    _HAS_BALCONY_DETECTOR = False

# Phase 2.4: Building Classifier (v3.4.0)
try:
    from .building_classifier import (
        BuildingClassifier,
        BuildingClassifierConfig,
        BuildingClassificationResult,
        classify_building,
    )

    _HAS_BUILDING_CLASSIFIER = True
except ImportError as e:
    import logging

    logging.getLogger(__name__).warning(f"Failed to import building classifier: {e}")
    BuildingClassifier = None
    BuildingClassifierConfig = None
    BuildingClassificationResult = None
    classify_building = None
    _HAS_BUILDING_CLASSIFIER = False

# Adaptive polygon buffering and integration modules are planned for future implementation
# These features are currently integrated directly into the adaptive classifier
AdaptivePolygonBuffer = None
AdaptiveBufferConfig = None
BuildingBoundaryAnalysis = None
AdaptiveGroundTruthProcessor = None
integrate_adaptive_buffering_with_wfs = None
_HAS_ADAPTIVE_BUFFERING = False

__all__ = [
    # Enumerations
    "BuildingMode",
    "BuildingSource",
    "ClassificationConfidence",
    # Base classes
    "BuildingConfigBase",
    "BuildingClassificationResult",
    "BuildingClassifierBase",
    "BuildingDetectorBase",
    "BuildingClustererBase",
    "BuildingFusionBase",
    # Utilities module (consolidated spatial operations, bbox utilities)
    "utils",
    # Adaptive building classifier
    "AdaptiveBuildingClassifier",
    "BuildingFeatureSignature",
    "PointBuildingScore",
    # Building detection
    "BuildingDetector",
    "BuildingDetectionMode",
    "BuildingDetectionConfig",
    # Building clustering
    "BuildingClusterer",
    "BuildingCluster",
    # Building fusion
    "BuildingFusion",
    "PolygonQuality",
    # 3D Extrusion (uses consolidated bbox utilities from utils)
    "Building3DExtruder",
    "BoundingBox3D",
    "FloorSegment",
    "create_3d_bboxes_from_ground_truth",
    # Phase 2 LOD3 Detectors (v3.1-3.3)
    "RoofTypeClassifier",
    "RoofType",
    "RoofSegment",
    "RoofClassificationResult",
    "ChimneyDetector",
    "SuperstructureType",
    "SuperstructureSegment",
    "ChimneyDetectionResult",
    "BalconyDetector",
    "ProtrusionType",
    "ProtrusionSegment",
    "BalconyDetectionResult",
    # Phase 2.4: Building Classifier (v3.4.0)
    "BuildingClassifier",
    "BuildingClassifierConfig",
    "BuildingClassificationResult",
    "classify_building",
    # Adaptive Polygon Buffering (v3.3.0)
    "AdaptivePolygonBuffer",
    "AdaptiveBufferConfig",
    "BuildingBoundaryAnalysis",
    "AdaptiveGroundTruthProcessor",
    "integrate_adaptive_buffering_with_wfs",
]

__version__ = "3.4.0"  # Bumped for enhanced building classifier integration
