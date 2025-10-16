"""
Ground Truth Classification Optimization Module

This module provides multiple optimization strategies for ground truth classification:
1. GPU acceleration (CuPy/cuSpatial) - 100-1000× speedup
2. Vectorized (GeoPandas spatial joins) - 30-100× speedup
3. STRtree spatial indexing - 10-30× speedup
4. Pre-filtering - 2-5× speedup

The module automatically selects the best available optimization at runtime.

Usage:
    from ign_lidar.optimization import auto_optimize
    auto_optimize()  # Automatically applies best optimization
    
    # Or use specific optimizers directly:
    from ign_lidar.optimization.vectorized import VectorizedGroundTruthClassifier
    from ign_lidar.optimization.strtree import STRtreeGroundTruthClassifier
    from ign_lidar.optimization.gpu import GPUGroundTruthClassifier

Author: IGN LiDAR HD Team
Date: October 16, 2025
"""

from .auto_select import (
    auto_optimize,
    OptimizationLevel,
    check_gpu_available,
    check_geopandas_available,
    check_strtree_available,
)


def apply_strtree_optimization():
    """Apply STRtree spatial indexing optimization (10-30× speedup)."""
    from .strtree import patch_advanced_classifier
    patch_advanced_classifier()


def apply_vectorized_optimization():
    """Apply GeoPandas vectorized optimization (30-100× speedup)."""
    from .vectorized import patch_advanced_classifier
    patch_advanced_classifier()


def apply_gpu_optimization():
    """Apply GPU-accelerated optimization (100-1000× speedup)."""
    from .gpu import patch_advanced_classifier
    patch_advanced_classifier()


def apply_prefilter_optimization():
    """Apply pre-filtering optimization (2-5× speedup)."""
    from .prefilter import patch_classifier
    patch_classifier()


__all__ = [
    'auto_optimize',
    'apply_strtree_optimization',
    'apply_vectorized_optimization',
    'apply_gpu_optimization',
    'apply_prefilter_optimization',
    'OptimizationLevel',
    'check_gpu_available',
    'check_geopandas_available',
    'check_strtree_available',
]

# Version info
__version__ = '1.0.0'
