"""
Core feature computation module - canonical implementations.

This module provides unified, well-tested implementations of all
geometric features, replacing the duplicated code found across
features.py, features_gpu.py, features_gpu_chunked.py, and features_boundary.py.

Usage:
    from ign_lidar.features.core import compute_normals, compute_curvature
    
    normals, eigenvalues = compute_normals(points, k_neighbors=20)
    curvature = compute_curvature(eigenvalues)

Modules:
    normals: Normal vector computation
    curvature: Curvature-based features
    eigenvalues: Eigenvalue-based geometric features
    density: Density-based features
    architectural: Architectural element features
    utils: Shared utility functions
"""

# Normal computation
from .normals import (
    compute_normals,
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
)

# Geometric features (consolidated)
from .geometric import (
    extract_geometric_features,
)

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
    
    # Geometric features (consolidated)
    'extract_geometric_features',
]

__version__ = '1.0.0'
__author__ = 'IGN LiDAR HD Dataset Team'
