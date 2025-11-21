"""
DEPRECATED: This module has been merged into optimization/ground_truth.py

This file provides backward compatibility aliases for code that imports
from the old location. All functionality (including V2 cache features)
is now available in the ign_lidar.optimization.ground_truth module.

Migration Guide:
    # OLD (deprecated):
    from ign_lidar.io.ground_truth_optimizer import GroundTruthOptimizer
    
    # NEW (recommended):
    from ign_lidar.optimization.ground_truth import GroundTruthOptimizer

The new module includes all V2 features:
- Intelligent caching with spatial hashing
- LRU eviction policy
- Batch processing (label_points_batch)
- Cache statistics (get_cache_stats, clear_cache)

This deprecation alias will be removed in v4.0.0.

Date: November 21, 2025
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing GroundTruthOptimizer from ign_lidar.io.ground_truth_optimizer is deprecated. "
    "Please use: from ign_lidar.optimization.ground_truth import GroundTruthOptimizer instead. "
    "This alias will be removed in v4.0.0.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location
from ign_lidar.optimization.ground_truth import GroundTruthOptimizer

__all__ = ['GroundTruthOptimizer']
