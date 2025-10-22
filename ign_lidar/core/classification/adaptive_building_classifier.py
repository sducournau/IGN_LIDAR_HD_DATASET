"""
DEPRECATED: Backward Compatibility Wrapper for Adaptive Building Classifier

This module is DEPRECATED and maintained only for backward compatibility.
The functionality has been moved to:
    ign_lidar.core.classification.building.adaptive

Please update your imports:
    OLD: from ign_lidar.core.classification.adaptive_building_classifier import AdaptiveBuildingClassifier
    NEW: from ign_lidar.core.classification.building import AdaptiveBuildingClassifier

This wrapper will be removed in version 4.0.0 (est. mid-2026).

Migration Guide: See docs/BUILDING_MODULE_MIGRATION_GUIDE.md

Author: Phase 2 - Building Module Restructuring
Date: October 22, 2025
"""

import warnings

# Emit deprecation warning
warnings.warn(
    "Module 'ign_lidar.core.classification.adaptive_building_classifier' is deprecated. "
    "Use 'ign_lidar.core.classification.building.adaptive' instead. "
    "This wrapper will be removed in v4.0.0. "
    "See docs/BUILDING_MODULE_MIGRATION_GUIDE.md for migration instructions.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location
from .building.adaptive import (
    ClassificationConfidence,
    BuildingFeatureSignature,
    PointBuildingScore,
    AdaptiveBuildingClassifier
)

__all__ = [
    'ClassificationConfidence',
    'BuildingFeatureSignature',
    'PointBuildingScore',
    'AdaptiveBuildingClassifier',
]
