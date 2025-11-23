"""
GPU Performance Profiling and Monitoring Utilities - DEPRECATED

DEPRECATED: This module is deprecated as of v3.3.0 (Nov 2025).
Use ign_lidar.core.gpu_profiler instead, which provides:
- CUDA event-based timing (more accurate)
- Bottleneck analysis (transfer vs compute)
- Memory tracking
- Unified API with core.GPUManager

Migration Guide:
    # Old (v3.0-v3.2)
    from ign_lidar.optimization.gpu_profiler import GPUProfiler, GPUOperationMetrics
    profiler = GPUProfiler(enable=True, session_name="eval")
    
    # New (v3.3+)
    from ign_lidar.core.gpu_profiler import GPUProfiler, ProfileEntry
    profiler = GPUProfiler(enabled=True, use_cuda_events=True)

Changes:
- GPUOperationMetrics → ProfileEntry
- ProfilerSession → ProfilingStats
- enable → enabled
- Session-based API → Direct profiling with get_stats()

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
    "ign_lidar.optimization.gpu_profiler is deprecated and will be removed in v4.0. "
    "Use ign_lidar.core.gpu_profiler instead for unified GPU profiling with "
    "CUDA events and bottleneck analysis. See migration guide in docstring.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location for backward compatibility
from ..core.gpu_profiler import (
    GPUProfiler,
    ProfileEntry,
    ProfilingStats,
    create_profiler,
    HAS_CUPY
)

# Legacy aliases for backward compatibility
GPUOperationMetrics = ProfileEntry  # Old name (v3.0-v3.2)
ProfilerSession = ProfilingStats     # Old name (v3.0-v3.2)

def get_profiler(enable: bool = True, session_name: str = "default") -> GPUProfiler:
    """
    Get or create global GPU profiler instance (legacy API).
    
    DEPRECATED: Use create_profiler() from core.gpu_profiler instead.
    
    Args:
        enable: Enable profiling
        session_name: Session name (ignored in new API)
        
    Returns:
        GPUProfiler instance
    """
    warnings.warn(
        "get_profiler() is deprecated. Use create_profiler() from "
        "ign_lidar.core.gpu_profiler instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return create_profiler(enabled=enable)

__all__ = [
    'GPUProfiler',
    'GPUOperationMetrics',  # Alias
    'ProfileEntry', 
    'ProfilerSession',       # Alias
    'ProfilingStats',
    'create_profiler',
    'get_profiler',          # Legacy
    'HAS_CUPY'
]
