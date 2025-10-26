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
    # Adaptive Polygon Buffering (v3.3.0)
    "AdaptivePolygonBuffer",
    "AdaptiveBufferConfig",
    "BuildingBoundaryAnalysis",
    "AdaptiveGroundTruthProcessor",
    "integrate_adaptive_buffering_with_wfs",
]

__version__ = "3.3.1"  # Bumped for module consolidation
