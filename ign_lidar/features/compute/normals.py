"""
Standard normal vector computation (fallback implementation).

This module provides the standard CPU implementation without JIT compilation.
Use this when Numba is not available or for small datasets.

For optimized computation, use the features module instead.
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.neighbors import NearestNeighbors
import logging

logger = logging.getLogger(__name__)


def compute_normals(
    points: np.ndarray,
    k_neighbors: int = 20,
    search_radius: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normal vectors and eigenvalues using standard CPU implementation.
    
    This is the fallback implementation without JIT compilation.
    For optimized computation, use features.compute_normals() instead.
    
    Parameters
    ----------
    points : np.ndarray
        Point cloud array of shape (N, 3) with XYZ coordinates
    k_neighbors : int, optional
        Number of nearest neighbors to use for normal estimation (default: 20)
    search_radius : float, optional
        Search radius for neighborhood. If None, uses k-nearest neighbors.
        
    Returns
    -------
    normals : np.ndarray
        Normal vectors of shape (N, 3), unit length
    eigenvalues : np.ndarray
        Eigenvalues of shape (N, 3), sorted in descending order
        
    Examples
    --------
    >>> points = np.random.rand(1000, 3)
    >>> normals, eigenvalues = compute_normals(points, k_neighbors=20)
    >>> assert normals.shape == (1000, 3)
    >>> assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    """
    # Input validation
    if not isinstance(points, np.ndarray):
        raise ValueError("points must be a numpy array")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if points.shape[0] < k_neighbors:
        raise ValueError(f"Not enough points ({points.shape[0]}) for k_neighbors={k_neighbors}")
    if k_neighbors < 3:
        raise ValueError(f"k_neighbors must be >= 3, got {k_neighbors}")
    
    # CPU computation
    return _compute_normals_cpu(points, k_neighbors, search_radius)


def _compute_normals_cpu(
    points: np.ndarray,
    k_neighbors: int,
    search_radius: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """CPU implementation of normal computation using scikit-learn."""
    n_points = points.shape[0]
    normals = np.zeros((n_points, 3), dtype=np.float32)
    eigenvalues = np.zeros((n_points, 3), dtype=np.float32)
    
    # Build KD-tree for neighbor search
    if search_radius is not None:
        # Radius-based search
        nbrs = NearestNeighbors(radius=search_radius, algorithm='kd_tree')
        nbrs.fit(points)
        
        for i in range(n_points):
            distances, indices = nbrs.radius_neighbors(points[i:i+1])
            neighbors = points[indices[0]]
            
            if len(neighbors) < 3:
                # Not enough neighbors, use default normal
                normals[i] = [0, 0, 1]
                eigenvalues[i] = [1, 0, 0]
                continue
                
            # Compute covariance matrix
            centroid = np.mean(neighbors, axis=0)
            centered = neighbors - centroid
            cov_matrix = np.dot(centered.T, centered) / len(neighbors)
            
            # Eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(cov_matrix)
            
            # Sort eigenvalues and eigenvectors in descending order
            idx = eigvals.argsort()[::-1]
            eigenvalues[i] = eigvals[idx].astype(np.float32)
            
            # Normal is eigenvector corresponding to smallest eigenvalue
            normal = eigvecs[:, idx[2]]
            normals[i] = normal.astype(np.float32)
    else:
        # K-nearest neighbors search
        nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='kd_tree')
        nbrs.fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        for i in range(n_points):
            neighbors = points[indices[i]]
            
            # Compute covariance matrix
            centroid = np.mean(neighbors, axis=0)
            centered = neighbors - centroid
            cov_matrix = np.dot(centered.T, centered) / k_neighbors
            
            # Eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(cov_matrix)
            
            # Sort eigenvalues and eigenvectors in descending order
            idx = eigvals.argsort()[::-1]
            eigenvalues[i] = eigvals[idx].astype(np.float32)
            
            # Normal is eigenvector corresponding to smallest eigenvalue
            normal = eigvecs[:, idx[2]]
            normals[i] = normal.astype(np.float32)
    
    # Ensure normals are unit length
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # Avoid division by zero
    normals = normals / norms
    
    return normals, eigenvalues


# Convenience functions for common use cases
def compute_normals_fast(points: np.ndarray) -> np.ndarray:
    """
    Fast normal computation with default parameters (returns only normals).
    
    Parameters
    ----------
    points : np.ndarray
        Point cloud array of shape (N, 3)
        
    Returns
    -------
    normals : np.ndarray
        Normal vectors of shape (N, 3)
    """
    normals, _ = compute_normals(points, k_neighbors=10)
    return normals


def compute_normals_accurate(
    points: np.ndarray, 
    k: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accurate normal computation with more neighbors.
    
    Parameters
    ----------
    points : np.ndarray
        Point cloud array of shape (N, 3)
    k : int, optional
        Number of neighbors (default: 50)
        
    Returns
    -------
    normals : np.ndarray
        Normal vectors of shape (N, 3)
    eigenvalues : np.ndarray
        Eigenvalues of shape (N, 3)
    """
    return compute_normals(points, k_neighbors=k)
