"""
Canonical implementation of normal vector computation.

This module provides the unified implementation that replaces
the 4 duplicate implementations found in:
- features.py (CPU)
- features_gpu.py (GPU)
- features_gpu_chunked.py (GPU chunked)
- features_boundary.py (boundary-aware)
"""

import numpy as np
from typing import Optional, Tuple, Union
from sklearn.neighbors import NearestNeighbors
import logging

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logger.debug("CuPy not available, GPU computation disabled")


def compute_normals(
    points: np.ndarray,
    k_neighbors: int = 20,
    search_radius: Optional[float] = None,
    use_gpu: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normal vectors and eigenvalues for point cloud.
    
    This is the canonical implementation that unifies all normal computation
    variants across the codebase. It automatically handles CPU/GPU dispatch
    and provides consistent output format.
    
    Parameters
    ----------
    points : np.ndarray
        Point cloud array of shape (N, 3) with XYZ coordinates
    k_neighbors : int, optional
        Number of nearest neighbors to use for normal estimation (default: 20)
    search_radius : float, optional
        Search radius for neighborhood. If None, uses k-nearest neighbors.
        If provided, uses radius search instead.
    use_gpu : bool, optional
        Whether to use GPU acceleration via CuPy (default: False)
        
    Returns
    -------
    normals : np.ndarray
        Normal vectors of shape (N, 3), unit length
    eigenvalues : np.ndarray
        Eigenvalues of shape (N, 3), sorted in descending order
        
    Raises
    ------
    ValueError
        If points array is invalid or k_neighbors is too small
    RuntimeError
        If GPU requested but CuPy not available
        
    Examples
    --------
    >>> points = np.random.rand(1000, 3)
    >>> normals, eigenvalues = compute_normals(points, k_neighbors=20)
    >>> assert normals.shape == (1000, 3)
    >>> assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    
    >>> # GPU computation
    >>> normals_gpu, eigvals_gpu = compute_normals(points, use_gpu=True)
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
    
    # GPU dispatch
    if use_gpu:
        if not CUPY_AVAILABLE:
            raise RuntimeError("GPU computation requested but CuPy is not available")
        return _compute_normals_gpu(points, k_neighbors, search_radius)
    
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


def _compute_normals_gpu(
    points: np.ndarray,
    k_neighbors: int,
    search_radius: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """GPU implementation using CuPy."""
    # Transfer data to GPU
    points_gpu = cp.asarray(points, dtype=cp.float32)
    n_points = points_gpu.shape[0]
    
    normals_gpu = cp.zeros((n_points, 3), dtype=cp.float32)
    eigenvalues_gpu = cp.zeros((n_points, 3), dtype=cp.float32)
    
    # For each point, compute normals using GPU
    # Note: This is a simplified GPU implementation
    # For production, consider using cuSpatial or custom CUDA kernels
    for i in range(n_points):
        point = points_gpu[i]
        
        # Compute distances to all other points
        diff = points_gpu - point
        distances = cp.linalg.norm(diff, axis=1)
        
        # Get k nearest neighbors
        indices = cp.argpartition(distances, k_neighbors)[:k_neighbors]
        neighbors = points_gpu[indices]
        
        # Compute covariance matrix
        centroid = cp.mean(neighbors, axis=0)
        centered = neighbors - centroid
        cov_matrix = cp.dot(centered.T, centered) / k_neighbors
        
        # Eigendecomposition
        eigvals, eigvecs = cp.linalg.eigh(cov_matrix)
        
        # Sort in descending order
        idx = cp.argsort(eigvals)[::-1]
        eigenvalues_gpu[i] = eigvals[idx]
        
        # Normal is eigenvector of smallest eigenvalue
        normal = eigvecs[:, idx[2]]
        normals_gpu[i] = normal
    
    # Normalize
    norms = cp.linalg.norm(normals_gpu, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals_gpu = normals_gpu / norms
    
    # Transfer back to CPU
    normals = cp.asnumpy(normals_gpu)
    eigenvalues = cp.asnumpy(eigenvalues_gpu)
    
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
    normals, _ = compute_normals(points, k_neighbors=10, use_gpu=False)
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
    return compute_normals(points, k_neighbors=k, use_gpu=False)
