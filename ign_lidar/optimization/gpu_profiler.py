"""
DEPRECATED: GPU Profiler moved to ign_lidar.core.gpu_profiler

This module is maintained for backward compatibility only.
All new code should import from ign_lidar.core.gpu_profiler instead.

Migration path:
  OLD: from ign_lidar.optimization.gpu_profiler import GPUProfiler
  NEW: from ign_lidar.core.gpu_profiler import GPUProfiler

Removal timeline: v4.0 (planned for 2026)
"""

import warnings
import logging

logger = logging.getLogger(__name__)

# Issue deprecation warning on import
warnings.warn(
    "ign_lidar.optimization.gpu_profiler is deprecated. "
    "Use ign_lidar.core.gpu_profiler instead. "
    "This module will be removed in v4.0.",
    DeprecationWarning,
    stacklevel=2
)

# Import all public symbols from core module
from ign_lidar.core.gpu_profiler import (
    GPUProfiler,
    ProfileEntry,
    ProfilingStats,
    create_profiler,
    HAS_CUPY,
)

# Aliases for backward compatibility
GPUOperationMetrics = ProfileEntry
ProfilerSession = GPUProfiler


def get_profiler(enable: bool = False, **kwargs):
    """
    DEPRECATED: Get profiler instance.
    
    Maintained for backward compatibility.
    Use create_profiler() instead.
    
    Args:
        enable: Whether to enable profiling
        **kwargs: Additional arguments passed to create_profiler
        
    Returns:
        GPUProfiler instance
    """
    warnings.warn(
        "get_profiler() is deprecated. Use create_profiler() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return create_profiler(enabled=enable, **kwargs)


__all__ = [
    'GPUProfiler',
    'ProfileEntry',
    'GPUOperationMetrics',  # Alias
    'ProfilerSession',  # Alias
    'ProfilingStats',
    'create_profiler',
    'get_profiler',  # Deprecated
    'HAS_CUPY',
]
