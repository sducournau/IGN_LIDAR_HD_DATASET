"""
Canonical implementation of geometric feature extraction.

This module provides the single source of truth for computing
comprehensive geometric features from LiDAR point clouds.
"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging

from ign_lidar.optimization import KDTree  # GPU-accelerated drop-in replacement
from .eigenvalues import compute_eigenvalue_features
from .density import compute_density_features
from .utils import validate_points, handle_nan_inf, compute_covariance_matrix
from ..utils import build_kdtree  # Use unified build_kdtree with GPU auto-selection

logger = logging.getLogger(__name__)


def _compute_eigenvalues_from_neighbors(
    points: np.ndarray,
    neighbors_indices: list
) -> np.ndarray:
    """
    Compute eigenvalues from neighbor indices.
    
    Args:
        points: [N, 3] point coordinates
        neighbors_indices: list of neighbor indices for each point
        
    Returns:
        eigenvalues: [N, 3] eigenvalues in descending order
    """
    n_points = len(points)
    eigenvalues = np.zeros((n_points, 3), dtype=np.float32)
    
    for i, neighbors_idx in enumerate(neighbors_indices):
        if len(neighbors_idx) < 3:
            # Not enough neighbors for meaningful eigenvalues
            eigenvalues[i] = [1.0, 0.0, 0.0]  # Default: linear
            continue
            
        # Get neighbor points
        neighbors = points[neighbors_idx]
        
        # Compute covariance matrix
        cov_matrix = compute_covariance_matrix(neighbors)
        
        # Compute eigenvalues
        eigvals = np.linalg.eigvals(cov_matrix)
        
        # Sort in descending order
        eigvals = np.sort(eigvals)[::-1]
        
        # Ensure non-negative (numerical precision issues)
        eigvals = np.maximum(eigvals, 0.0)
        
        eigenvalues[i] = eigvals
    
    return eigenvalues


def extract_geometric_features(
    points: np.ndarray,
    normals: np.ndarray,
    k: int = 10,
    radius: Optional[float] = None
) -> Dict[str, np.ndarray]:
    """
    Extract comprehensive geometric features for each point.
    
    This is the canonical implementation that replaces the duplicated
    versions in features.py and features_gpu.py.
    
    Features computed (eigenvalue-based):
    - Linearity: (λ0-λ1)/λ0 - 1D structures (edges, cables) [0,1]
    - Planarity: (λ1-λ2)/λ0 - 2D structures (roofs, walls) [0,1]
    - Sphericity: λ2/λ0 - 3D structures (vegetation, noise) [0,1]
    - Anisotropy: (λ0-λ2)/λ0 - general directionality [0,1]
    - Roughness: λ2/Σλ - surface roughness [0,1]
    - Density: 1/mean_distance - local point density
    
    Args:
        points: [N, 3] point coordinates
        normals: [N, 3] normal vectors (not used directly, kept for compatibility)
        k: number of neighbors (used if radius=None)
        radius: search radius in meters (recommended to avoid scan artifacts)
        
    Returns:
        features: dictionary of geometric features
        
    Raises:
        ValueError: If points array is invalid
    """
    # Validate inputs
    validate_points(points)
    
    n_points = len(points)
    logger.debug(f"Computing geometric features for {n_points:,} points")
    
    # Build spatial index using unified build_kdtree (with GPU auto-selection)
    tree = build_kdtree(points, metric='euclidean', leaf_size=30)
    
    # Find neighbors
    if radius is not None:
        # Radius-based search (recommended)
        neighbors_indices = tree.query_radius(points, r=radius)
        neighbors_indices = [list(indices) for indices in neighbors_indices]
    else:
        # K-nearest neighbors
        k_effective = min(k, n_points - 1)
        _, neighbors_indices_array = tree.query(points, k=k_effective)
        neighbors_indices = [list(indices) for indices in neighbors_indices_array]
    
    # Compute eigenvalues from neighbors
    eigenvalues = _compute_eigenvalues_from_neighbors(points, neighbors_indices)
    
    # Compute eigenvalue-based features
    eigenvalue_features = compute_eigenvalue_features(
        eigenvalues, epsilon=1e-10, include_all=True
    )
    
    # Compute density features
    density_features = compute_density_features(
        points, k_neighbors=k, search_radius=radius
    )
    
    # Compute extended density features
    from .density import compute_extended_density_features
    extended_density_features = compute_extended_density_features(
        points, k_neighbors=k
    )
    
    # Combine all features
    features = {}
    features.update(eigenvalue_features)
    features.update(density_features)
    features.update(extended_density_features)
    
    # Handle any NaN/inf values
    for key, values in features.items():
        features[key] = handle_nan_inf(values)
    
    logger.debug(f"Computed {len(features)} geometric features")
    return features