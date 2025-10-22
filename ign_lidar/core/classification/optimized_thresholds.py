"""
Optimized Classification Thresholds and Rules

⚠️ DEPRECATED - Use ign_lidar.core.classification.thresholds instead

This module is deprecated as of v3.1.0 and will be removed in v4.0.0.
All functionality has been consolidated into the unified thresholds module.

Migration:
    # Old (deprecated)
    from ign_lidar.core.classification.optimized_thresholds import (
        NDVIThresholds, GeometricThresholds, ClassificationThresholds
    )
    
    # New (recommended)
    from ign_lidar.core.classification.thresholds import (
        ThresholdConfig,
        NDVIThresholds,
        GeometricThresholds,
        get_thresholds
    )

For full migration guide, see: docs/THRESHOLD_MIGRATION_GUIDE.md

Author: IGN LiDAR HD Dataset Team
Date: October 15, 2025
Updated: October 22, 2025 - Deprecated in favor of thresholds.py
"""

import warnings

# Import from the new unified module for backward compatibility
from .thresholds import (
    NDVIThresholds,
    GeometricThresholds,
    HeightThresholds,
    ThresholdConfig as ClassificationThresholds,
)

# Issue deprecation warning when module is imported
warnings.warn(
    "optimized_thresholds.py is deprecated as of v3.1.0 and will be removed in v4.0.0. "
    "All functionality has been moved to 'ign_lidar.core.classification.thresholds'. "
    "Please update your imports. See docs/THRESHOLD_MIGRATION_GUIDE.md for details.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export for backward compatibility
__all__ = [
    'NDVIThresholds',
    'GeometricThresholds',
    'HeightThresholds',
    'ClassificationThresholds',
]
