"""
Numba-accelerated computational kernels for feature computation.

ARCHITECTURE NOTE - LOW-LEVEL HELPERS:
This module provides JIT-compiled helper functions for performance-critical operations.
These are LOW-LEVEL utilities called by canonical implementations, NOT standalone features.

Call Hierarchy:
  1. FeatureOrchestrator.compute_features() - Entry point
     ↓
  2. compute.normals.compute_normals() - Canonical CPU implementation
     ↓
  3. THIS FILE - Low-level Numba-accelerated helpers:
     - compute_covariance_matrices_numba()
     - compute_local_point_density_numba()

Note (v4.0): Removed deprecated normal extraction functions:
  - compute_normals_from_eigenvectors_numba() (removed)
  - compute_normals_from_eigenvectors_numpy() (removed)
  - compute_normals_from_eigenvectors() (removed)
  Use compute.normals.compute_normals() instead.

Usage Guidelines:
  - DO NOT call these functions directly in application code
  - Use compute.normals.compute_normals() or FeatureOrchestrator instead
  - These provide JIT-compiled speedups for CPU fallback paths
  - Gracefully degrade to NumPy when Numba unavailable

Implementation Details:
  - All functions have @jit decorators with nopython=True for performance
  - Parallel execution with prange where applicable
  - Pure NumPy fallbacks when Numba not available

Author: IGN LiDAR HD Team
Date: 2025-11-22 (Architecture doc update)
"""

import numpy as np
import warnings
from typing import Tuple, Optional

# Try to import Numba
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Define no-op decorator when Numba unavailable
    def jit(*args, **kwargs):
        """No-op decorator when Numba is not available."""
        def decorator(func):
            return func
        return decorator
    
    # Define prange as regular range when Numba unavailable
    prange = range


def is_numba_available() -> bool:
    """
    Check if Numba is available for JIT compilation.
    
    Returns:
        bool: True if Numba is available, False otherwise.
    """
    return NUMBA_AVAILABLE


@jit(nopython=True, parallel=True, cache=True)
def compute_covariance_matrices_numba(
    points: np.ndarray,
    indices: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Compute covariance matrices for each point using its k-nearest neighbors.
    
    This is a Numba-accelerated version of the covariance computation that
    provides significant speedup over pure NumPy when processing large point clouds.
    
    Args:
        points: Point cloud array [N, 3] with XYZ coordinates
        indices: KNN indices array [N, k] from neighbor search
        k: Number of neighbors
        
    Returns:
        Covariance matrices [N, 3, 3] for each point
        
    Note:
        This function uses Numba JIT compilation with parallel processing.
        If Numba is unavailable, it falls back to the pure NumPy implementation.
    """
    N = points.shape[0]
    cov_matrices = np.zeros((N, 3, 3), dtype=np.float32)
    
    for i in prange(N):
        # Get neighbor points
        neighbor_indices = indices[i]
        neighbors = points[neighbor_indices]
        
        # Compute centroid
        centroid = np.zeros(3, dtype=np.float32)
        for j in range(k):
            centroid[0] += neighbors[j, 0]
            centroid[1] += neighbors[j, 1]
            centroid[2] += neighbors[j, 2]
        centroid /= k
        
        # Center the neighbors
        centered = np.zeros((k, 3), dtype=np.float32)
        for j in range(k):
            centered[j, 0] = neighbors[j, 0] - centroid[0]
            centered[j, 1] = neighbors[j, 1] - centroid[1]
            centered[j, 2] = neighbors[j, 2] - centroid[2]
        
        # Compute covariance matrix: C = (1/(k-1)) * centered^T * centered
        for row in range(3):
            for col in range(3):
                cov_sum = 0.0
                for j in range(k):
                    cov_sum += centered[j, row] * centered[j, col]
                cov_matrices[i, row, col] = cov_sum / (k - 1)
    
    return cov_matrices


def compute_covariance_matrices_numpy(
    points: np.ndarray,
    indices: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Compute covariance matrices using pure NumPy (fallback).
    
    This is the fallback implementation when Numba is not available.
    It uses vectorized NumPy operations for reasonable performance.
    
    Args:
        points: Point cloud array [N, 3] with XYZ coordinates
        indices: KNN indices array [N, k] from neighbor search
        k: Number of neighbors
        
    Returns:
        Covariance matrices [N, 3, 3] for each point
    """
    # Vectorized approach using einsum
    neighbor_points = points[indices]
    centroids = np.mean(neighbor_points, axis=1, keepdims=True)
    centered = neighbor_points - centroids
    cov_matrices = np.einsum("mki,mkj->mij", centered, centered) / (k - 1)
    
    return cov_matrices.astype(np.float32)


def compute_covariance_matrices(
    points: np.ndarray,
    indices: np.ndarray,
    k: int,
    use_numba: Optional[bool] = None
) -> np.ndarray:
    """
    Compute covariance matrices with automatic Numba/NumPy selection.
    
    This function automatically selects the best implementation based on
    Numba availability and user preference.
    
    Args:
        points: Point cloud array [N, 3] with XYZ coordinates
        indices: KNN indices array [N, k] from neighbor search
        k: Number of neighbors
        use_numba: Force Numba usage (True), force NumPy (False), or auto (None)
        
    Returns:
        Covariance matrices [N, 3, 3] for each point
        
    Example:
        >>> points = np.random.rand(1000, 3).astype(np.float32)
        >>> indices = np.random.randint(0, 1000, (1000, 30))
        >>> cov = compute_covariance_matrices(points, indices, k=30)
        >>> cov.shape
        (1000, 3, 3)
    """
    # Determine which implementation to use
    if use_numba is None:
        use_numba = NUMBA_AVAILABLE
    elif use_numba and not NUMBA_AVAILABLE:
        warnings.warn(
            "Numba requested but not available. Install with: pip install numba",
            UserWarning
        )
        use_numba = False
    
    if use_numba:
        return compute_covariance_matrices_numba(points, indices, k)
    else:
        return compute_covariance_matrices_numpy(points, indices, k)



@jit(nopython=True, parallel=True, cache=True)
def compute_local_point_density_numba(
    points: np.ndarray,
    indices: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Compute local point density using k-nearest neighbors (Numba-accelerated).
    
    Density is computed as k / volume_of_sphere, where the sphere is defined
    by the distance to the k-th nearest neighbor.
    
    Args:
        points: Point cloud array [N, 3]
        indices: KNN indices array [N, k]
        k: Number of neighbors
        
    Returns:
        Local density values [N]
    """
    N = points.shape[0]
    densities = np.zeros(N, dtype=np.float32)
    
    for i in prange(N):
        # Get k-th nearest neighbor
        kth_neighbor_idx = indices[i, k-1]
        kth_neighbor = points[kth_neighbor_idx]
        
        # Compute distance to k-th neighbor
        dx = points[i, 0] - kth_neighbor[0]
        dy = points[i, 1] - kth_neighbor[1]
        dz = points[i, 2] - kth_neighbor[2]
        radius = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        # Compute density: k / volume_of_sphere
        # Volume = (4/3) * pi * r^3
        if radius > 1e-6:
            volume = (4.0 / 3.0) * 3.14159265359 * radius * radius * radius
            densities[i] = k / volume
        else:
            densities[i] = 0.0
    
    return densities


def compute_local_point_density_numpy(
    points: np.ndarray,
    indices: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Compute local point density using NumPy (fallback).
    
    Args:
        points: Point cloud array [N, 3]
        indices: KNN indices array [N, k]
        k: Number of neighbors
        
    Returns:
        Local density values [N]
    """
    # Get k-th nearest neighbors (vectorized)
    kth_neighbors = points[indices[:, k-1]]
    
    # Compute distances
    distances = np.linalg.norm(points - kth_neighbors, axis=1)
    
    # Compute volumes and densities
    volumes = (4.0 / 3.0) * np.pi * distances**3
    
    # Avoid division by zero
    densities = np.zeros(len(points), dtype=np.float32)
    valid_mask = volumes > 1e-6
    densities[valid_mask] = k / volumes[valid_mask]
    
    return densities


def compute_local_point_density(
    points: np.ndarray,
    indices: np.ndarray,
    k: int,
    use_numba: Optional[bool] = None
) -> np.ndarray:
    """
    Compute local point density with automatic Numba/NumPy selection.
    
    Args:
        points: Point cloud array [N, 3]
        indices: KNN indices array [N, k]
        k: Number of neighbors
        use_numba: Force Numba usage (True), force NumPy (False), or auto (None)
        
    Returns:
        Local density values [N]
        
    Example:
        >>> points = np.random.rand(1000, 3).astype(np.float32)
        >>> indices = np.random.randint(0, 1000, (1000, 30))
        >>> density = compute_local_point_density(points, indices, k=30)
        >>> density.shape
        (1000,)
    """
    if use_numba is None:
        use_numba = NUMBA_AVAILABLE
    elif use_numba and not NUMBA_AVAILABLE:
        use_numba = False
    
    if use_numba:
        return compute_local_point_density_numba(points, indices, k)
    else:
        return compute_local_point_density_numpy(points, indices, k)


def get_numba_info() -> dict:
    """
    Get information about Numba availability and configuration.
    
    Returns:
        Dictionary with Numba status information
        
    Example:
        >>> info = get_numba_info()
        >>> print(f"Numba available: {info['available']}")
        >>> if info['available']:
        ...     print(f"Version: {info['version']}")
    """
    info = {
        'available': NUMBA_AVAILABLE,
        'version': None,
        'threading_layer': None,
        'num_threads': None,
    }
    
    if NUMBA_AVAILABLE:
        try:
            import numba
            info['version'] = numba.__version__
            
            # Get threading layer info
            try:
                from numba import config
                info['threading_layer'] = config.THREADING_LAYER
                info['num_threads'] = config.NUMBA_NUM_THREADS
            except (ImportError, AttributeError):
                pass
        except Exception:
            pass
    
    return info
