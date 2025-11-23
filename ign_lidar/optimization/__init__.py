"""
Ground Truth Classification Optimization Module

Week 2 Consolidation: Ground truth classification with automatic optimization.

This module provides the GroundTruthOptimizer class that automatically selects
the best method based on dataset size and available hardware:

1. GPU Chunked (100-1000x speedup) - Large datasets (>10M points) with GPU
2. GPU Basic (100-500x speedup) - Medium datasets (1-10M points) with GPU
3. CPU STRtree (10-30x speedup) - Works everywhere, spatial indexing
4. CPU Vectorized (5-10x speedup) - GeoPandas fallback

Usage:
    from ign_lidar.optimization import GroundTruthOptimizer
    
    optimizer = GroundTruthOptimizer(verbose=True)
    labels = optimizer.label_points(points, ground_truth_features)
    
    # Legacy auto_optimize still available for backward compatibility
    from ign_lidar.optimization import auto_optimize
    auto_optimize()

Author: IGN LiDAR HD Team
Date: October 21, 2025 (Week 2 Consolidation)
Version: 2.0
"""

from .auto_select import (
    auto_optimize,
    OptimizationLevel,
    check_gpu_available,
    check_geopandas_available,
    check_strtree_available,
)

# Week 2: Ground truth optimizer (replaces 7 implementations)
from .ground_truth import GroundTruthOptimizer

# Phase 1: GPU Accelerated Operations (November 2025)
from .gpu_accelerated_ops import (
    eigh,
    eigvalsh,
    knn,
    cdist,
    svd,
    set_force_cpu,
    get_gpu_info,
    HAS_CUPY,
    HAS_FAISS,
    HAS_CUML,
)
from .gpu_kdtree import (
    GPUKDTree,
    create_kdtree,
    cKDTree,  # Drop-in replacement for scipy.spatial.cKDTree
    KDTree,   # Drop-in replacement for sklearn.neighbors.KDTree
)

# Phase 2: KNN Engine (November 2025)
from .knn_engine import (
    KNNEngine,
    KNNBackend,
    knn_search,
    radius_search,
    build_knn_graph,
    HAS_FAISS_GPU,
)

# Phase 2: GPU Transfer Profiler (November 2025)
from .gpu_transfer_profiler import (
    GPUTransferProfiler,
    TransferEvent,
    get_global_profiler,
    enable_automatic_tracking,
)

# Phase 3: Adaptive Chunking (November 2025 Audit)
from .adaptive_chunking import (
    auto_chunk_size,
    estimate_gpu_memory_required,
    get_recommended_strategy,
    calculate_optimal_chunk_count,
)

# Phase 2: CUDA Stream Manager (November 2025)
from .cuda_streams import (
    CUDAStreamManager,
    StreamConfig,
)

# Backward compatibility: gpu_dataframe_ops moved to io/ in v3.1.0
# Maintain import for v3.x compatibility
try:
    from ..io.gpu_dataframe import GPUDataFrameOps
    # Create alias for old import path
    import sys
    sys.modules['ign_lidar.optimization.gpu_dataframe_ops'] = sys.modules['ign_lidar.io.gpu_dataframe']
except ImportError:
    GPUDataFrameOps = None


def apply_strtree_optimization():
    """Apply STRtree spatial indexing optimization (10-30x speedup)."""
    from .strtree import patch_advanced_classifier
    patch_advanced_classifier()


def apply_vectorized_optimization():
    """Apply GeoPandas vectorized optimization (30-100x speedup)."""
    from .vectorized import patch_advanced_classifier
    patch_advanced_classifier()


def apply_gpu_optimization():
    """Apply GPU-accelerated optimization (100-1000x speedup)."""
    from .gpu import patch_advanced_classifier
    patch_advanced_classifier()


def apply_prefilter_optimization():
    """Apply pre-filtering optimization (2-5x speedup)."""
    from .prefilter import patch_classifier
    patch_classifier()


__all__ = [
    # Week 2: Primary interface
    'GroundTruthOptimizer',
    # Phase 1: GPU Accelerated Operations (Nov 2025)
    'eigh', 'eigvalsh', 'knn', 'cdist', 'svd',
    'set_force_cpu', 'get_gpu_info',
    'GPUKDTree', 'create_kdtree', 'cKDTree', 'KDTree',
    'HAS_CUPY', 'HAS_FAISS', 'HAS_CUML',
    # Phase 2: Unified KNN Engine (Nov 2025)
    'KNNEngine', 'KNNBackend', 'knn_search', 'radius_search', 'build_knn_graph', 'HAS_FAISS_GPU',
    # Phase 2: GPU Transfer Profiler (Nov 2025)
    'GPUTransferProfiler', 'TransferEvent', 'get_global_profiler', 'enable_automatic_tracking',
    # Phase 3: Adaptive Chunking (Nov 2025 Audit)
    'auto_chunk_size', 'estimate_gpu_memory_required', 'get_recommended_strategy',
    'calculate_optimal_chunk_count',
    # Phase 2: CUDA Stream Manager (Nov 2025)
    'CUDAStreamManager', 'StreamConfig',
    # GPU dataframe operations (relocated to io/ in v3.1.0)
    'GPUDataFrameOps',  # Backward compatibility alias
    # Legacy interfaces (backward compatibility)
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
__version__ = '2.0.0'  # Week 2: Ground truth optimizer
