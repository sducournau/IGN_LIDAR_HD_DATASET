"""
Normal Vector Computation - CPU Canonical Implementation

ARCHITECTURE NOTE:
This is the CANONICAL CPU implementation for normal computation.

Call Hierarchy:
  1. FeatureOrchestrator.compute_features() - RECOMMENDED ENTRY POINT
     ↓ (routes automatically based on use_gpu flag)
  2. compute_normals() - THIS FILE (CPU canonical)
  3. GPUProcessor.compute_normals() - GPU optimized (features/gpu_processor.py)
  4. numba_accelerated.* - Low-level helpers (features/numba_accelerated.py)
  5. gpu_kernels.compute_normals_and_eigenvalues() - Fused GPU kernel

Usage Guidelines:
  - For new code: Use FeatureOrchestrator (handles CPU/GPU routing)
  - For CPU-only: Import from this file directly
  - For GPU: Use features.gpu_processor.GPUProcessor
  - DO NOT duplicate this implementation - extend or optimize it

Implementation Details:
  - Standard CPU implementation without JIT compilation
  - Fallback when Numba is not available
  - Variants: compute_normals(), compute_normals_fast(), compute_normals_accurate()
"""

import numpy as np
from typing import Optional, Tuple
import logging

from ign_lidar.optimization.gpu_accelerated_ops import eigh
from ign_lidar.optimization import knn_search  # Phase 2: Unified KNN engine

logger = logging.getLogger(__name__)


def compute_normals(
    points: np.ndarray,
    k_neighbors: int = 20,
    search_radius: Optional[float] = None,
    method: str = 'standard',
    return_eigenvalues: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Compute normal vectors and eigenvalues using standard CPU implementation.
    
    ✅ **CANONICAL CPU IMPLEMENTATION (v3.5.2+)**
    This is the recommended public API for computing normals on CPU.
    
    For JIT-optimized computation, this automatically dispatches to:
    - `features.compute.features.compute_all_features_optimized()` (Numba JIT)
    
    For GPU computation, use:
    - `optimization.gpu_kernels.compute_normals_eigenvalues_fused()` (CUDA)
    
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
    method : str, optional
        Computation method: 'fast' (k=10), 'accurate' (k=50), or 'standard' (use k_neighbors)
    return_eigenvalues : bool, optional
        Whether to return eigenvalues (default: True). Set False for faster computation.
        
    Returns
    -------
    normals : np.ndarray
        Normal vectors of shape (N, 3), unit length
    eigenvalues : np.ndarray or None
        Eigenvalues of shape (N, 3), sorted in descending order.
        None if return_eigenvalues=False.
        
    Examples
    --------
    >>> points = np.random.rand(1000, 3)
    >>> # Standard computation
    >>> normals, eigenvalues = compute_normals(points, k_neighbors=20)
    >>> # Fast computation (fewer neighbors, normals only)
    >>> normals, _ = compute_normals(points, method='fast', return_eigenvalues=False)
    >>> # Accurate computation (more neighbors)
    >>> normals, eigenvalues = compute_normals(points, method='accurate')
    """
    # Adjust k_neighbors based on method
    if method == 'fast':
        k_neighbors = 10
    elif method == 'accurate':
        k_neighbors = 50
    elif method != 'standard':
        raise ValueError(f"Invalid method '{method}'. Use 'fast', 'accurate', or 'standard'")
    
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
    normals, eigenvalues = _compute_normals_cpu(points, k_neighbors, search_radius)
    
    if not return_eigenvalues:
        return normals, None
    return normals, eigenvalues


def _compute_normals_cpu(
    points: np.ndarray,
    k_neighbors: int,
    search_radius: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CPU implementation of normal computation using unified KNN engine (Phase 2).
    
    Now uses the unified knn_search() from Phase 2 for automatic backend selection
    (FAISS-GPU, FAISS-CPU, cuML, or sklearn based on data size and hardware).
    """
    n_points = points.shape[0]
    normals = np.zeros((n_points, 3), dtype=np.float32)
    eigenvalues = np.zeros((n_points, 3), dtype=np.float32)
    
    if search_radius is not None:
        # Radius-based search using unified KNN engine (v3.6.0+)
        from ign_lidar.optimization import KNNEngine
        
        engine = KNNEngine(backend='auto')
        distances_list, indices_list = engine.radius_search(
            points,
            radius=search_radius
        )
        
        for i in range(n_points):
            neighbors = points[indices_list[i]]
            
            if len(neighbors) < 3:
                # Not enough neighbors, use default normal
                normals[i] = [0, 0, 1]
                eigenvalues[i] = [1, 0, 0]
                continue
                
            # Compute covariance matrix
            centroid = np.mean(neighbors, axis=0)
            centered = neighbors - centroid
            cov_matrix = np.dot(centered.T, centered) / len(neighbors)
            
            # Eigendecomposition (GPU-accelerated with CPU fallback)
            eigvals, eigvecs = eigh(cov_matrix)
            
            # Sort eigenvalues and eigenvectors in descending order
            idx = eigvals.argsort()[::-1]
            eigenvalues[i] = eigvals[idx].astype(np.float32)
            
            # Normal is eigenvector corresponding to smallest eigenvalue
            normal = eigvecs[:, idx[2]]
            normals[i] = normal.astype(np.float32)
    else:
        # Phase 2: Use unified KNN engine (automatic backend selection)
        # Replaces manual sklearn implementation with optimized multi-backend support
        distances, indices = knn_search(points, k=k_neighbors, backend='auto')
        
        for i in range(n_points):
            neighbors = points[indices[i]]
            
            # Compute covariance matrix
            centroid = np.mean(neighbors, axis=0)
            centered = neighbors - centroid
            cov_matrix = np.dot(centered.T, centered) / k_neighbors
            
            # Eigendecomposition (GPU-accelerated with CPU fallback)
            eigvals, eigvecs = eigh(cov_matrix)
            
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

