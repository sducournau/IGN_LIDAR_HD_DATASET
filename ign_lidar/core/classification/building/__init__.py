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
    BuildingFusionBase
)

# Utility functions
from . import utils

# Concrete implementations
try:
    from .adaptive import (
        AdaptiveBuildingClassifier,
        BuildingFeatureSignature,
        PointBuildingScore
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
    from .detection import BuildingDetector, BuildingDetectionMode, BuildingDetectionConfig
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

__all__ = [
    # Enumerations
    'BuildingMode',
    'BuildingSource',
    'ClassificationConfidence',
    
    # Base classes
    'BuildingConfigBase',
    'BuildingClassificationResult',
    'BuildingClassifierBase',
    'BuildingDetectorBase',
    'BuildingClustererBase',
    'BuildingFusionBase',
    
    # Utilities module
    'utils',
    
    # Adaptive building classifier
    'AdaptiveBuildingClassifier',
    'BuildingFeatureSignature',
    'PointBuildingScore',
    
    # Building detection
    'BuildingDetector',
    'BuildingDetectionMode',
    'BuildingDetectionConfig',
    
    # Building clustering
    'BuildingClusterer',
    'BuildingCluster',
    
    # Building fusion
    'BuildingFusion',
    'PolygonQuality',
]

__version__ = '3.1.0'
