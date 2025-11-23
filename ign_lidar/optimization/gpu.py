"""
GPU-Accelerated Ground Truth Classification - DEPRECATED

DEPRECATED: This module is deprecated as of v3.3.0 (Nov 2025).
File renamed to ground_truth_classifier.py for clarity.

Reason for rename:
- Generic name "gpu.py" was confusing (what GPU functionality?)
- New name clearly indicates purpose: ground truth classification
- Avoids confusion with core/gpu.py (GPUManager)

Migration:
    # Old (v3.0-v3.2)
    from ign_lidar.optimization.gpu import GPUGroundTruthClassifier
    
    # New (v3.3+)
    from ign_lidar.optimization.ground_truth_classifier import GPUGroundTruthClassifier
    
    # Or use the helper function (recommended)
    from ign_lidar.optimization.ground_truth import label_with_ground_truth_gpu

This stub will be removed in v4.0.

Author: IGN LiDAR HD Development Team
Date: November 23, 2025 (Consolidation Phase 1)
Version: 3.3.0 (Deprecated)
"""

import warnings
import logging

logger = logging.getLogger(__name__)

# Deprecation warning issued at import time
warnings.warn(
    "ign_lidar.optimization.gpu is deprecated and will be removed in v4.0. "
    "The module has been renamed to ground_truth_classifier.py for clarity. "
    "Use: from ign_lidar.optimization.ground_truth_classifier import GPUGroundTruthClassifier",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location for backward compatibility
from .ground_truth_classifier import (
    GPUGroundTruthClassifier,
    HAS_CUPY,
    HAS_CUSPATIAL,
    HAS_SPATIAL,
)

__all__ = [
    'GPUGroundTruthClassifier',
    'HAS_CUPY',
    'HAS_CUSPATIAL',
    'HAS_SPATIAL',
]
