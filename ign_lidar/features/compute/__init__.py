"""
Feature computation module - canonical implementations of geometric features.

This module provides unified, well-tested implementations of all geometric features
with clean, consistent naming conventions. Optimized with JIT compilation where available.

📍 **Note**: Relocated from `features.core` to `features.compute` in v3.1.0 for better
semantic clarity and to avoid confusion with `core` package. The old import path is
deprecated but still works in v3.x.

Usage:
    # New (recommended)
    from ign_lidar.features.compute import compute_normals, compute_curvature

    # Old (deprecated but works)
    from ign_lidar.features.core import compute_normals, compute_curvature

    normals, eigenvalues = compute_normals(points, k_neighbors=20)
    curvature = compute_curvature(eigenvalues)

Modules:
    features: Optimized feature computation (JIT-compiled, recommended)
    normals: Standard normal computation (fallback)
    curvature: Curvature-based features
    eigenvalues: Eigenvalue-based geometric features
    height: Height-based features (height above ground, etc.)
    density: Density-based features
    architectural: Architectural element features
    geometric: General geometric computations
    gpu_bridge: GPU-accelerated feature computation bridge
    utils: Shared utility functions

Migration:
    # Old (deprecated in v3.1.0)
    from ign_lidar.features.core.eigenvalues import compute_eigenvalues

    # New (recommended)
    from ign_lidar.features.compute.eigenvalues import compute_eigenvalues
"""

# Optimized feature computation (JIT-compiled, preferred)
try:
    from .features import compute_all_features_optimized, compute_normals

    OPTIMIZED_AVAILABLE = True
except ImportError:
    # Fallback to standard implementation
    from .normals import compute_normals

    OPTIMIZED_AVAILABLE = False

# Architectural features
from .architectural import (
    compute_architectural_features,
    compute_building_regularity,
    compute_corner_likelihood,
    compute_facade_score,
    compute_horizontality,
    compute_roof_likelihood,
)
from .architectural import compute_verticality as compute_normal_verticality
from .architectural import compute_wall_likelihood

# Curvature features
from .curvature import (
    compute_all_curvature_features,
    compute_curvature,
    compute_curvature_from_normals,
    compute_curvature_from_normals_batched,
    compute_curvedness,
    compute_mean_curvature,
    compute_shape_index,
)

# Density features
from .density import (
    compute_density_features,
    compute_density_variance,
    compute_local_spacing,
    compute_neighborhood_size,
    compute_point_density,
    compute_relative_height_density,
)

# Eigenvalue features
from .eigenvalues import (
    compute_anisotropy,
    compute_eigenentropy,
    compute_eigenvalue_features,
    compute_linearity,
    compute_omnivariance,
    compute_planarity,
    compute_sphericity,
    compute_verticality,
)

# Feature filtering (artifact reduction) - Unified module v3.1.0
from .feature_filter import (
    smooth_feature_spatial,
    validate_feature,
    smooth_planarity_spatial,
    smooth_linearity_spatial,
    smooth_horizontality_spatial,
    validate_planarity,
    validate_linearity,
    validate_horizontality,
)

# Legacy imports for backward compatibility (deprecated)
try:
    from .planarity_filter import (
        smooth_planarity_spatial as _legacy_smooth_planarity,
        validate_planarity as _legacy_validate_planarity,
    )
except ImportError:
    pass  # planarity_filter.py may be removed in future versions

# Geometric features (consolidated)
from .geometric import extract_geometric_features

# GPU-Core Bridge (Phase 1: GPU refactoring)
from .gpu_bridge import (
    CUPY_AVAILABLE,
    GPUCoreBridge,
    compute_eigenvalue_features_gpu,
    compute_eigenvalues_gpu,
)

# Height features
from .height import (
    compute_height_above_ground,
    compute_height_bins,
    compute_height_percentile,
    compute_normalized_height,
    compute_relative_height,
)

# Ground classification features (with DTM augmentation support)
from .is_ground import (
    compute_ground_density,
    compute_is_ground,
    compute_is_ground_with_stats,
    identify_ground_gaps,
)

# Additional normal computation utilities
from .normals import compute_normals_accurate, compute_normals_fast

# Unified API dispatcher - this is the main public API for compute_all_features
from .unified import compute_all_features  # Main public API
from .unified import ComputeMode

# Utilities
from .utils import (
    batched_inverse_3x3,
    clip_features,
    compute_angle_between_vectors,
    compute_covariance_matrix,
    compute_covariances_from_neighbors,
    compute_eigenvalue_features_from_covariances,
    compute_local_frame,
    get_array_module,
    handle_nan_inf,
    inverse_power_iteration,
    normalize_features,
    normalize_vectors,
    safe_divide,
    sort_eigenvalues,
    standardize_features,
    validate_eigenvalues,
    validate_normals,
    validate_points,
)

# Consolidated __all__ export list
__all__ = [
    # Normal computation
    "compute_normals",
    "compute_normals_fast",
    "compute_normals_accurate",
    # Curvature features
    "compute_curvature",
    "compute_mean_curvature",
    "compute_shape_index",
    "compute_curvedness",
    "compute_all_curvature_features",
    # Eigenvalue features
    "compute_eigenvalue_features",
    "compute_linearity",
    "compute_planarity",
    "compute_sphericity",
    # Feature filtering (artifact reduction) - v3.1.0 unified
    "smooth_feature_spatial",
    "validate_feature",
    "smooth_planarity_spatial",
    "smooth_linearity_spatial",
    "smooth_horizontality_spatial",
    "validate_planarity",
    "validate_linearity",
    "validate_horizontality",
    "compute_anisotropy",
    "compute_omnivariance",
    "compute_eigenentropy",
    "compute_verticality",
    # Density features
    "compute_density_features",
    "compute_point_density",
    "compute_local_spacing",
    "compute_density_variance",
    "compute_neighborhood_size",
    "compute_relative_height_density",
    # Height features
    "compute_height_above_ground",
    "compute_relative_height",
    "compute_normalized_height",
    "compute_height_percentile",
    "compute_height_bins",
    # Ground classification features
    "compute_is_ground",
    "compute_is_ground_with_stats",
    "compute_ground_density",
    "identify_ground_gaps",
    # Architectural features
    "compute_architectural_features",
    "compute_normal_verticality",
    "compute_horizontality",
    "compute_wall_likelihood",
    "compute_roof_likelihood",
    "compute_facade_score",
    "compute_building_regularity",
    "compute_corner_likelihood",
    # Utilities
    "validate_points",
    "validate_eigenvalues",
    "validate_normals",
    "normalize_vectors",
    "safe_divide",
    "compute_covariance_matrix",
    "sort_eigenvalues",
    "clip_features",
    "compute_angle_between_vectors",
    "standardize_features",
    "normalize_features",
    "handle_nan_inf",
    "compute_local_frame",
    "get_array_module",
    "batched_inverse_3x3",
    "inverse_power_iteration",
    "compute_eigenvalue_features_from_covariances",
    "compute_covariances_from_neighbors",
    # Geometric features (consolidated)
    "extract_geometric_features",
    # Unified API (main public API for feature computation)
    "compute_all_features",  # High-level API with mode selection
    "compute_all_features_optimized",  # Low-level CPU-optimized implementation
    "ComputeMode",
    # GPU-Core Bridge (Phase 1 refactoring)
    "GPUCoreBridge",
    "compute_eigenvalues_gpu",
    "compute_eigenvalue_features_gpu",
    "CUPY_AVAILABLE",
]

__version__ = "1.0.0"
__author__ = "IGN LiDAR HD Dataset Team"
