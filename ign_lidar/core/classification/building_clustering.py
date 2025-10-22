"""
DEPRECATED: Backward Compatibility Wrapper for Building Clustering

This module is DEPRECATED and maintained only for backward compatibility.
The functionality has been moved to:
    ign_lidar.core.classification.building.clustering

Please update your imports:
    OLD: from ign_lidar.core.classification.building_clustering import BuildingClusterer
    NEW: from ign_lidar.core.classification.building import BuildingClusterer

This wrapper will be removed in version 4.0.0 (est. mid-2026).

Migration Guide: See docs/BUILDING_MODULE_MIGRATION_GUIDE.md

Author: Phase 2 - Building Module Restructuring
Date: October 22, 2025
"""

import warnings

warnings.warn(
    "Module 'ign_lidar.core.classification.building_clustering' is deprecated. "
    "Use 'ign_lidar.core.classification.building.clustering' instead. "
    "This wrapper will be removed in v4.0.0. "
    "See docs/BUILDING_MODULE_MIGRATION_GUIDE.md for migration instructions.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location
from .building.clustering import *

__all__ = [
    'BuildingCluster',
    'BuildingClusterer',
]
