"""
GPU Memory Optimization (DEPRECATED)

⚠️ DEPRECATION WARNING ⚠️

This module has been reorganized in v3.3.0 for better clarity:
- GPUArrayCache → optimization.gpu_cache.arrays
- TransferOptimizer → optimization.gpu_cache.transfer  
- GPUMemoryPool → optimization.gpu_cache.transfer

This stub provides backward compatibility and will be removed in v4.0.

Migration Guide:
    # OLD (v3.0-v3.2)
    from ign_lidar.optimization.gpu_memory import GPUArrayCache, GPUMemoryPool
    
    # NEW (v3.3+)
    from ign_lidar.optimization.gpu_cache import GPUArrayCache, GPUMemoryPool
    from ign_lidar.optimization.gpu_cache import TransferOptimizer

Reason for Change:
- Confusion with core.gpu_memory (system-level GPUMemoryManager)
- Better organization: arrays vs transfer optimization
- Clearer module hierarchy

Version: Deprecated in 3.3.0, will be removed in 4.0.0
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "ign_lidar.optimization.gpu_memory is deprecated and will be removed in v4.0. "
    "Use 'from ign_lidar.optimization.gpu_cache import GPUArrayCache, GPUMemoryPool' instead. "
    "See docstring for full migration guide.",
    DeprecationWarning,
    stacklevel=2
)

# Backward compatibility imports
from .gpu_cache.arrays import GPUArrayCache
from .gpu_cache.transfer import (
    TransferOptimizer,
    GPUMemoryPool,
    estimate_gpu_memory_for_features,
    optimize_chunk_size_for_vram
)

__all__ = [
    'GPUArrayCache',
    'TransferOptimizer',
    'GPUMemoryPool',
    'estimate_gpu_memory_for_features',
    'optimize_chunk_size_for_vram',
]
