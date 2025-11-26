"""
DEPRECATED: GPU Memory management moved to gpu_cache package

This module is maintained for backward compatibility only.
All new code should import from ign_lidar.optimization.gpu_cache instead.

Migration path:
  OLD: from ign_lidar.optimization.gpu_memory import GPUMemoryPool
  NEW: from ign_lidar.optimization.gpu_cache import GPUMemoryPool

Removal timeline: v4.0 (planned for 2026)
"""

import warnings
import logging

logger = logging.getLogger(__name__)

# Issue deprecation warning on import
warnings.warn(
    "ign_lidar.optimization.gpu_memory is deprecated. "
    "Use ign_lidar.optimization.gpu_cache instead. "
    "This module will be removed in v4.0.",
    DeprecationWarning,
    stacklevel=2
)

# Import all public symbols from gpu_cache
from ign_lidar.optimization.gpu_cache import (
    GPUArrayCache,
    GPUMemoryPool,
    TransferOptimizer,
    estimate_gpu_memory_for_features,
    optimize_chunk_size_for_vram,
)

__all__ = [
    'GPUArrayCache',
    'GPUMemoryPool',
    'TransferOptimizer',
    'estimate_gpu_memory_for_features',
    'optimize_chunk_size_for_vram',
]
