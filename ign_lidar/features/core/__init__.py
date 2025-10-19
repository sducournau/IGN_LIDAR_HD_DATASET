"""
Core feature computation module - canonical implementations.

This module provides unified, well-tested implementations of all
geometric features with clean, consistent naming conventions.

Usage:
    from ign_lidar.features.core import compute_normals, compute_curvature
    
    normals, eigenvalues = compute_normals(points, k_neighbors=20)
    curvature = compute_curvature(eigenvalues)

Modules:
    features: Optimized feature computation (JIT-compiled, recommended)
    normals: Standard normal computation (fallback)
    curvature: Curvature-based features
    eigenvalues: Eigenvalue-based geometric features
    density: Density-based features
    architectural: Architectural element features
    utils: Shared utility functions
"""

# Optimized feature computation (JIT-compiled, preferred)
try:
    from .features import (
        compute_normals,
        compute_all_features,
    )
    OPTIMIZED_AVAILABLE = True
except ImportError:
    # Fallback to standard implementation
    from .normals import (
        compute_normals,
    )
    OPTIMIZED_AVAILABLE = False

# Additional normal computation utilities
from .normals import (
    compute_normals_fast,
    compute_normals_accurate,
)

# Curvature features
from .curvature import (
    compute_curvature,
    compute_mean_curvature,
    compute_shape_index,
    compute_curvedness,
    compute_all_curvature_features,
    compute_curvature_from_normals,
    compute_curvature_from_normals_batched,
)

# Eigenvalue features
from .eigenvalues import (
    compute_eigenvalue_features,
    compute_linearity,
    compute_planarity,
    compute_sphericity,
    compute_anisotropy,
    compute_omnivariance,
    compute_eigenentropy,
    compute_verticality,
)

# Density features
from .density import (
    compute_density_features,
    compute_point_density,
    compute_local_spacing,
    compute_density_variance,
    compute_neighborhood_size,
    compute_relative_height_density,
)

# Height features
from .height import (
    compute_height_above_ground,
    compute_relative_height,
    compute_normalized_height,
    compute_height_percentile,
    compute_height_bins,
)

# Architectural features
from .architectural import (
    compute_architectural_features,
    compute_verticality as compute_normal_verticality,
    compute_horizontality,
    compute_wall_likelihood,
    compute_roof_likelihood,
    compute_facade_score,
    compute_building_regularity,
    compute_corner_likelihood,
)

# Utilities
from .utils import (
    validate_points,
    validate_eigenvalues,
    validate_normals,
    normalize_vectors,
    safe_divide,
    compute_covariance_matrix,
    sort_eigenvalues,
    clip_features,
    compute_angle_between_vectors,
    standardize_features,
    normalize_features,
    handle_nan_inf,
    compute_local_frame,
    get_array_module,
    batched_inverse_3x3,
    inverse_power_iteration,
    compute_eigenvalue_features_from_covariances,
    compute_covariances_from_neighbors,
)

# Geometric features (consolidated)
from .geometric import (
    extract_geometric_features,
)

# Unified API dispatcher (replaces all compute_all_features variants)
from .unified import (
    compute_all_features as compute_all_features_dispatcher,
    ComputeMode,
)

# GPU-Core Bridge (Phase 1: GPU refactoring)
from .gpu_bridge import (
    GPUCoreBridge,
    compute_eigenvalues_gpu,
    compute_eigenvalue_features_gpu,
    CUPY_AVAILABLE,
)

__all__ = [
    # Optimized feature computation (main API)
    'compute_normals',
    'compute_all_features',
    
    # Normal utilities
    'compute_normals_fast',
    'compute_normals_accurate',
    
    # Curvature features
    'compute_curvature',
    'compute_mean_curvature',
    'compute_shape_index',
    'compute_curvedness',
    'compute_all_curvature_features',
    'compute_curvature_from_normals',
    'compute_curvature_from_normals_batched',
    
    # Eigenvalue features
    'compute_eigenvalue_features',
    'compute_linearity',
    'compute_planarity',
    'compute_sphericity',
    'compute_anisotropy',
    'compute_omnivariance',
    'compute_eigenentropy',
    'compute_verticality',
    
    # Density features
    'compute_density_features',
    'compute_point_density',
    'compute_local_spacing',
    'compute_density_variance',
    'compute_neighborhood_size',
    'compute_relative_height_density',
    
    # Architectural features
    'compute_architectural_features',
    'compute_normal_verticality',
    'compute_horizontality',
    'compute_wall_likelihood',
    'compute_roof_likelihood',
    'compute_facade_score',
    'compute_building_regularity',
    'compute_corner_likelihood',
    
    # Geometric features
    'extract_geometric_features',
    
    # Unified API
    'compute_all_features_dispatcher',
    'ComputeMode',
    
    # Utilities
    'validate_points',
    'validate_eigenvalues',
    'validate_normals',
    'normalize_vectors',
    'safe_divide',
    'compute_covariance_matrix',
    'sort_eigenvalues',
    'clip_features',
    'compute_angle_between_vectors',
    'standardize_features',
    'normalize_features',
    'handle_nan_inf',
    'compute_local_frame',
    
    # Flags
    'OPTIMIZED_AVAILABLE',
]

__all__ = [
    # Normal computation
    'compute_normals',
    'compute_normals_fast',
    'compute_normals_accurate',
    
    # Curvature features
    'compute_curvature',
    'compute_mean_curvature',
    'compute_shape_index',
    'compute_curvedness',
    'compute_all_curvature_features',
    
    # Eigenvalue features
    'compute_eigenvalue_features',
    'compute_linearity',
    'compute_planarity',
    'compute_sphericity',
    'compute_anisotropy',
    'compute_omnivariance',
    'compute_eigenentropy',
    'compute_verticality',
    
    # Density features
    'compute_density_features',
    'compute_point_density',
    'compute_local_spacing',
    'compute_density_variance',
    'compute_neighborhood_size',
    'compute_relative_height_density',
    
    # Height features
    'compute_height_above_ground',
    'compute_relative_height',
    'compute_normalized_height',
    'compute_height_percentile',
    'compute_height_bins',
    
    # Architectural features
    'compute_architectural_features',
    'compute_normal_verticality',
    'compute_horizontality',
    'compute_wall_likelihood',
    'compute_roof_likelihood',
    'compute_facade_score',
    'compute_building_regularity',
    'compute_corner_likelihood',
    
    # Utilities
    'validate_points',
    'validate_eigenvalues',
    'validate_normals',
    'normalize_vectors',
    'safe_divide',
    'compute_covariance_matrix',
    'sort_eigenvalues',
    'clip_features',
    'compute_angle_between_vectors',
    'standardize_features',
    'normalize_features',
    'handle_nan_inf',
    'compute_local_frame',
    'get_array_module',
    'batched_inverse_3x3',
    'inverse_power_iteration',
    'compute_eigenvalue_features_from_covariances',
    'compute_covariances_from_neighbors',
    
    # Geometric features (consolidated)
    'extract_geometric_features',
    
    # Unified API (replaces all compute_all_features variants)
    'compute_all_features',
    'ComputeMode',
    
    # GPU-Core Bridge (Phase 1 refactoring)
    'GPUCoreBridge',
    'compute_eigenvalues_gpu',
    'compute_eigenvalue_features_gpu',
    'CUPY_AVAILABLE',
]

__version__ = '1.0.0'
__author__ = 'IGN LiDAR HD Dataset Team'
