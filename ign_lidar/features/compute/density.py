"""
Canonical implementation of density feature computation.

This module provides unified density-based features.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.neighbors import NearestNeighbors
import logging
import multiprocessing

logger = logging.getLogger(__name__)


def _get_safe_n_jobs() -> int:
    """Get safe n_jobs for sklearn avoiding multiprocessing conflicts."""
    if multiprocessing.current_process().name != 'MainProcess':
        return 1  # Disable sklearn parallelism in workers
    return -1  # Use all CPUs in main process


def compute_density_features(
    points: np.ndarray,
    k_neighbors: int = 20,
    search_radius: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute comprehensive density-based features.
    
    Density features describe the local point distribution and spacing,
    useful for identifying vegetation, sparse/dense areas, etc.
    
    Parameters
    ----------
    points : np.ndarray
        Point cloud array of shape (N, 3) with XYZ coordinates
    k_neighbors : int, optional
        Number of nearest neighbors for density estimation (default: 20)
    search_radius : float, optional
        Fixed radius for density computation. If None, uses k-nearest neighbors.
        
    Returns
    -------
    features : dict
        Dictionary containing:
        - 'point_density': Number of neighbors per unit volume
        - 'mean_distance': Average distance to k nearest neighbors
        - 'std_distance': Standard deviation of distances
        - 'local_density_ratio': Ratio of local to global density
        
    Examples
    --------
    >>> points = np.random.rand(1000, 3)
    >>> features = compute_density_features(points, k_neighbors=20)
    >>> print(features.keys())
    dict_keys(['point_density', 'mean_distance', 'std_distance', 'local_density_ratio'])
    
    Notes
    -----
    Density interpretation:
    - High density: Dense vegetation, building interiors
    - Low density: Sparse areas, isolated objects, noise
    """
    # Input validation
    if not isinstance(points, np.ndarray):
        raise ValueError("points must be a numpy array")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if points.shape[0] < k_neighbors:
        raise ValueError(f"Not enough points ({points.shape[0]}) for k_neighbors={k_neighbors}")
    
    n_points = points.shape[0]
    
    # Build KD-tree for neighbor search
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='kd_tree')
    nbrs.fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    # Mean distance to k-nearest neighbors (excluding self at distance 0)
    mean_distance = np.mean(distances[:, 1:], axis=1).astype(np.float32)
    
    # Standard deviation of distances
    std_distance = np.std(distances[:, 1:], axis=1).astype(np.float32)
    
    # Point density: number of neighbors / volume of sphere
    # Volume of sphere: (4/3) * π * r³, where r is the distance to k-th neighbor
    max_distance = distances[:, -1]
    volume = (4.0 / 3.0) * np.pi * (max_distance ** 3)
    point_density = (k_neighbors / volume).astype(np.float32)
    
    # Handle edge case where distance is zero
    point_density[max_distance == 0] = 0.0
    
    # Local density ratio: local density / global density
    global_density = n_points / compute_bounding_box_volume(points)
    local_density_ratio = (point_density / global_density).astype(np.float32)
    
    features = {
        'point_density': point_density,
        'mean_distance': mean_distance,
        'std_distance': std_distance,
        'local_density_ratio': local_density_ratio,
        'density': point_density,  # Alias for compatibility
    }
    
    return features


def compute_extended_density_features(
    points: np.ndarray,
    k_neighbors: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute extended density features including neighborhood characteristics.
    
    Parameters
    ----------
    points : np.ndarray
        Point cloud array of shape (N, 3) with XYZ coordinates
    k_neighbors : int, optional
        Number of nearest neighbors (default: 20)
        
    Returns
    -------
    features : dict
        Dictionary containing extended features:
        - 'num_points_2m': Number of points within 2m radius
        - 'neighborhood_extent': Size of k-neighbor bounding box
        - 'height_extent_ratio': Height range / horizontal range ratio
        - 'vertical_std': Standard deviation of heights in neighborhood
    """
    n_points = points.shape[0]
    
    # Build KD-tree for neighbor search
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='kd_tree')
    nbrs.fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    # Also build radius neighbors for 2m count
    nbrs_radius = NearestNeighbors(radius=2.0, algorithm='kd_tree')
    nbrs_radius.fit(points)
    
    # Initialize arrays
    num_points_2m = np.zeros(n_points, dtype=np.int32)
    neighborhood_extent = np.zeros(n_points, dtype=np.float32)
    height_extent_ratio = np.zeros(n_points, dtype=np.float32)
    vertical_std = np.zeros(n_points, dtype=np.float32)
    
    for i in range(n_points):
        # Number of points within 2m radius
        radius_distances, radius_indices = nbrs_radius.radius_neighbors(points[i:i+1])
        num_points_2m[i] = len(radius_indices[0]) - 1  # Exclude self
        
        # Get k-neighbors
        neighbor_indices = indices[i]
        neighbor_points = points[neighbor_indices]
        
        # Neighborhood extent (bounding box diagonal)
        min_coords = np.min(neighbor_points, axis=0)
        max_coords = np.max(neighbor_points, axis=0)
        extent_3d = np.linalg.norm(max_coords - min_coords)
        neighborhood_extent[i] = extent_3d
        
        # Height extent ratio
        height_range = max_coords[2] - min_coords[2]
        horizontal_range = np.sqrt((max_coords[0] - min_coords[0])**2 + 
                                 (max_coords[1] - min_coords[1])**2)
        if horizontal_range > 1e-10:
            height_extent_ratio[i] = height_range / horizontal_range
        else:
            height_extent_ratio[i] = 0.0
            
        # Vertical standard deviation
        vertical_std[i] = np.std(neighbor_points[:, 2])
    
    features = {
        'num_points_2m': num_points_2m,
        'neighborhood_extent': neighborhood_extent,
        'height_extent_ratio': height_extent_ratio,
        'vertical_std': vertical_std,
    }
    
    return features


def compute_point_density(
    points: np.ndarray,
    k_neighbors: int = 20
) -> np.ndarray:
    """
    Compute local point density.
    
    Parameters
    ----------
    points : np.ndarray
        Point cloud array of shape (N, 3)
    k_neighbors : int, optional
        Number of neighbors for density estimation
        
    Returns
    -------
    density : np.ndarray
        Point density values of shape (N,)
    """
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='kd_tree')
    nbrs.fit(points)
    distances, _ = nbrs.kneighbors(points)
    
    max_distance = distances[:, -1]
    volume = (4.0 / 3.0) * np.pi * (max_distance ** 3)
    density = (k_neighbors / volume).astype(np.float32)
    
    # Handle edge case
    density[max_distance == 0] = 0.0
    
    return density


def compute_local_spacing(
    points: np.ndarray,
    k_neighbors: int = 8
) -> np.ndarray:
    """
    Compute local point spacing (average distance to neighbors).
    
    Useful for identifying point cloud resolution and uniformity.
    
    Parameters
    ----------
    points : np.ndarray
        Point cloud array of shape (N, 3)
    k_neighbors : int, optional
        Number of neighbors to consider (default: 8)
        
    Returns
    -------
    spacing : np.ndarray
        Local spacing values of shape (N,)
    """
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='kd_tree')
    nbrs.fit(points)
    distances, _ = nbrs.kneighbors(points)
    
    # Mean distance excluding self (first column is distance to self = 0)
    spacing = np.mean(distances[:, 1:], axis=1).astype(np.float32)
    
    return spacing


def compute_density_variance(
    points: np.ndarray,
    k_neighbors: int = 20
) -> np.ndarray:
    """
    Compute variance of local point density.
    
    High variance indicates non-uniform point distribution.
    
    Parameters
    ----------
    points : np.ndarray
        Point cloud array of shape (N, 3)
    k_neighbors : int, optional
        Number of neighbors
        
    Returns
    -------
    variance : np.ndarray
        Density variance values of shape (N,)
    """
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='kd_tree')
    nbrs.fit(points)
    distances, _ = nbrs.kneighbors(points)
    
    # Variance of distances
    variance = np.var(distances[:, 1:], axis=1).astype(np.float32)
    
    return variance


def compute_neighborhood_size(
    points: np.ndarray,
    search_radius: float
) -> np.ndarray:
    """
    Compute number of neighbors within a fixed radius.
    
    Parameters
    ----------
    points : np.ndarray
        Point cloud array of shape (N, 3)
    search_radius : float
        Search radius for counting neighbors
        
    Returns
    -------
    neighbor_count : np.ndarray
        Number of neighbors within radius for each point, shape (N,)
    """
    nbrs = NearestNeighbors(
        radius=search_radius, 
        algorithm='kd_tree',
        n_jobs=_get_safe_n_jobs()
    )
    nbrs.fit(points)
    
    neighbor_count = np.zeros(points.shape[0], dtype=np.int32)
    
    for i in range(points.shape[0]):
        distances, indices = nbrs.radius_neighbors(points[i:i+1])
        neighbor_count[i] = len(indices[0]) - 1  # Exclude self
    
    return neighbor_count


def compute_bounding_box_volume(points: np.ndarray) -> float:
    """
    Compute volume of axis-aligned bounding box.
    
    Parameters
    ----------
    points : np.ndarray
        Point cloud array of shape (N, 3)
        
    Returns
    -------
    volume : float
        Volume of bounding box
    """
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    dimensions = max_coords - min_coords
    volume = np.prod(dimensions)
    
    return float(volume)


def compute_relative_height_density(
    points: np.ndarray,
    k_neighbors: int = 20
) -> np.ndarray:
    """
    Compute density weighted by relative height.
    
    Useful for distinguishing ground vegetation from tall vegetation.
    
    Parameters
    ----------
    points : np.ndarray
        Point cloud array of shape (N, 3)
    k_neighbors : int, optional
        Number of neighbors
        
    Returns
    -------
    height_density : np.ndarray
        Height-weighted density values of shape (N,)
    """
    # Compute basic density
    density = compute_point_density(points, k_neighbors)
    
    # Normalize height to [0, 1]
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])
    relative_height = (points[:, 2] - z_min) / (z_max - z_min + 1e-10)
    
    # Weight density by relative height
    height_density = (density * relative_height).astype(np.float32)
    
    return height_density
