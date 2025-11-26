"""
Canonical implementation of curvature feature computation.

This module provides curvature computation replacing duplicates in:
- features.py (CPU)
- features_gpu.py (GPU)
- features_gpu_chunked.py (GPU chunked)
- features_boundary.py (boundary-aware)

Two complementary methods are provided:
1. Eigenvalue-based: Fast, computed from covariance matrix eigenvalues
2. Normal-based: More accurate, computed from normal differences in k-neighbors
"""

import numpy as np
from typing import Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)

# ✅ NEW (v3.5.2): Centralized GPU imports via GPUManager
from ign_lidar.core.gpu import GPUManager

gpu = GPUManager()
CUPY_AVAILABLE = gpu.gpu_available

if CUPY_AVAILABLE:
    cp = gpu.get_cupy()
else:
    cp = None


def compute_curvature(
    eigenvalues: np.ndarray,
    epsilon: float = 1e-10,
    method: str = 'standard'
) -> np.ndarray:
    """
    Compute surface curvature from eigenvalues.
    
    Curvature measures the change in surface normal direction, 
    indicating how curved the local surface is around each point.
    
    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues of shape (N, 3), sorted in descending order (λ1 >= λ2 >= λ3)
    epsilon : float, optional
        Small value to prevent division by zero (default: 1e-10)
    method : str, optional
        Curvature computation method:
        - 'standard': λ3 / (λ1 + λ2 + λ3) 
        - 'normalized': λ3 / λ1
        - 'gaussian': λ2 * λ3 / (λ1^2) (Gaussian curvature approximation)
        (default: 'standard')
        
    Returns
    -------
    curvature : np.ndarray
        Curvature values of shape (N,), range [0, 1] for standard method
        
    Raises
    ------
    ValueError
        If eigenvalues array is invalid or method is unknown
        
    Examples
    --------
    >>> eigenvalues = np.array([[1.0, 0.5, 0.1], [1.0, 0.8, 0.01]])
    >>> curvature = compute_curvature(eigenvalues)
    >>> assert curvature.shape == (2,)
    >>> assert np.all(curvature >= 0) and np.all(curvature <= 1)
    
    Notes
    -----
    Curvature interpretation:
    - High values (close to 1): Sharp edges, corners
    - Medium values (~0.5): Curved surfaces
    - Low values (close to 0): Flat surfaces
    """
    # Input validation
    if not isinstance(eigenvalues, np.ndarray):
        raise ValueError("eigenvalues must be a numpy array")
    if eigenvalues.ndim != 2 or eigenvalues.shape[1] != 3:
        raise ValueError(f"eigenvalues must have shape (N, 3), got {eigenvalues.shape}")
    
    lambda1 = eigenvalues[:, 0]
    lambda2 = eigenvalues[:, 1]
    lambda3 = eigenvalues[:, 2]
    
    if method == 'standard':
        # Standard curvature: λ3 / (λ1 + λ2 + λ3)
        sum_eigenvalues = lambda1 + lambda2 + lambda3
        curvature = lambda3 / (sum_eigenvalues + epsilon)
        
    elif method == 'normalized':
        # Normalized curvature: λ3 / λ1
        curvature = lambda3 / (lambda1 + epsilon)
        
    elif method == 'gaussian':
        # Gaussian curvature approximation
        curvature = (lambda2 * lambda3) / (lambda1 * lambda1 + epsilon)
        
    else:
        raise ValueError(f"Unknown curvature method: {method}. "
                       f"Choose from: 'standard', 'normalized', 'gaussian'")
    
    return curvature.astype(np.float32)


def compute_mean_curvature(eigenvalues: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Compute mean curvature from eigenvalues.
    
    Mean curvature is the average of the principal curvatures.
    
    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues of shape (N, 3), sorted in descending order
    epsilon : float, optional
        Small value to prevent division by zero
        
    Returns
    -------
    mean_curvature : np.ndarray
        Mean curvature values of shape (N,)
    """
    lambda1 = eigenvalues[:, 0]
    lambda2 = eigenvalues[:, 1]
    lambda3 = eigenvalues[:, 2]
    
    # Mean curvature approximation
    mean_curvature = (lambda2 + lambda3) / (lambda1 + epsilon)
    
    return mean_curvature.astype(np.float32)


def compute_shape_index(eigenvalues: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Compute shape index from eigenvalues.
    
    Shape index classifies surface type (e.g., ridge, valley, saddle).
    Range: [-1, 1]
    
    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues of shape (N, 3), sorted in descending order
    epsilon : float, optional
        Small value to prevent division by zero
        
    Returns
    -------
    shape_index : np.ndarray
        Shape index values of shape (N,), range [-1, 1]
        
    Notes
    -----
    Shape index interpretation:
    - ~1.0: Spherical cap (peak)
    - ~0.5: Ridge
    - ~0.0: Saddle
    - ~-0.5: Valley
    - ~-1.0: Spherical cup (pit)
    """
    lambda1 = eigenvalues[:, 0]
    lambda2 = eigenvalues[:, 1]
    lambda3 = eigenvalues[:, 2]
    
    # Principal curvatures approximation
    k1 = lambda2 / (lambda1 + epsilon)
    k2 = lambda3 / (lambda1 + epsilon)
    
    # Shape index: (2/π) * arctan((k1 + k2) / (k1 - k2))
    shape_index = (2.0 / np.pi) * np.arctan2(k1 + k2, k1 - k2 + epsilon)
    
    return shape_index.astype(np.float32)


def compute_curvedness(eigenvalues: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Compute curvedness (magnitude of curvature) from eigenvalues.
    
    Curvedness measures how strongly curved a surface is, 
    independent of the type of curvature.
    
    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues of shape (N, 3), sorted in descending order
    epsilon : float, optional
        Small value to prevent division by zero
        
    Returns
    -------
    curvedness : np.ndarray
        Curvedness values of shape (N,), always positive
        
    Notes
    -----
    Higher values indicate stronger curvature (more curved surface).
    """
    lambda1 = eigenvalues[:, 0]
    lambda2 = eigenvalues[:, 1]
    lambda3 = eigenvalues[:, 2]
    
    # Principal curvatures approximation
    k1 = lambda2 / (lambda1 + epsilon)
    k2 = lambda3 / (lambda1 + epsilon)
    
    # Curvedness: sqrt((k1^2 + k2^2) / 2)
    curvedness = np.sqrt((k1**2 + k2**2) / 2.0)
    
    return curvedness.astype(np.float32)


def compute_all_curvature_features(
    eigenvalues: np.ndarray,
    epsilon: float = 1e-10
) -> dict:
    """
    Compute all curvature-based features.
    
    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues of shape (N, 3), sorted in descending order
    epsilon : float, optional
        Small value to prevent division by zero
        
    Returns
    -------
    features : dict
        Dictionary containing:
        - 'curvature': Standard curvature
        - 'mean_curvature': Mean curvature
        - 'shape_index': Shape index
        - 'curvedness': Curvedness magnitude
        
    Examples
    --------
    >>> eigenvalues = np.random.rand(1000, 3)
    >>> eigenvalues = np.sort(eigenvalues, axis=1)[:, ::-1]  # Sort descending
    >>> features = compute_all_curvature_features(eigenvalues)
    >>> print(features.keys())
    dict_keys(['curvature', 'mean_curvature', 'shape_index', 'curvedness'])
    """
    return {
        'curvature': compute_curvature(eigenvalues, epsilon=epsilon),
        'mean_curvature': compute_mean_curvature(eigenvalues, epsilon=epsilon),
        'shape_index': compute_shape_index(eigenvalues, epsilon=epsilon),
        'curvedness': compute_curvedness(eigenvalues, epsilon=epsilon),
    }


def compute_curvature_from_normals(
    points: np.ndarray,
    normals: np.ndarray,
    neighbor_indices: np.ndarray
) -> np.ndarray:
    """
    Compute curvature from normal vector differences in local neighborhoods.
    
    This method measures curvature as the mean change in normal orientation
    among k-nearest neighbors. It's more accurate than eigenvalue-based methods
    for capturing actual surface curvature, especially for complex geometries.
    
    This function eliminates duplicated code from:
    - features_gpu.py::compute_curvature() (lines 574-705)
    - features_gpu_chunked.py::compute_curvature()
    
    Parameters
    ----------
    points : np.ndarray or cp.ndarray
        Point coordinates, shape (N, 3)
    normals : np.ndarray or cp.ndarray
        Normal vectors, shape (N, 3)
    neighbor_indices : np.ndarray or cp.ndarray
        K-nearest neighbor indices, shape (N, k)
        
    Returns
    -------
    curvature : np.ndarray or cp.ndarray
        Curvature values, shape (N,), dtype float32
        
    Examples
    --------
    >>> import numpy as np
    >>> # Simulate a planar surface
    >>> points = np.random.rand(100, 3).astype(np.float32)
    >>> normals = np.tile([0, 0, 1], (100, 1)).astype(np.float32)
    >>> 
    >>> # Build KNN (using sklearn or cuML)
    >>> from sklearn.neighbors import NearestNeighbors
    >>> knn = NearestNeighbors(n_neighbors=10)
    >>> knn.fit(points)
    >>> _, indices = knn.kneighbors(points)
    >>> 
    >>> # Compute curvature
    >>> curvature = compute_curvature_from_normals(points, normals, indices)
    >>> assert curvature.shape == (100,)
    >>> # Planar surface should have low curvature
    >>> assert np.mean(curvature) < 0.1
    
    Notes
    -----
    Works with both NumPy (CPU) and CuPy (GPU) arrays seamlessly.
    
    Algorithm:
    1. For each point, get k nearest neighbors
    2. Compute normal differences: Δn = n_neighbor - n_query
    3. Compute L2 norms: ||Δn||
    4. Average: curvature = mean(||Δn||)
    
    Higher values indicate sharper surface changes (edges, corners).
    Lower values indicate flatter surfaces.
    """
    # Get array module (numpy or cupy)
    if CUPY_AVAILABLE and isinstance(points, cp.ndarray):
        xp = cp
    else:
        xp = np
    
    N, k = neighbor_indices.shape
    
    # Get neighbor normals: [N, k, 3]
    neighbor_normals = normals[neighbor_indices]
    
    # Expand query normals for broadcasting: [N, 1, 3]
    query_normals_expanded = normals[:, xp.newaxis, :]
    
    # Compute normal differences: [N, k, 3]
    normal_diff = neighbor_normals - query_normals_expanded
    
    # Compute L2 norms: [N, k]
    norm_diff = xp.linalg.norm(normal_diff, axis=2)
    
    # Average across neighbors: [N]
    curvature = xp.mean(norm_diff, axis=1)
    
    return curvature.astype(xp.float32)


def compute_curvature_from_normals_batched(
    points: np.ndarray,
    normals: np.ndarray,
    k: int = 10,
    batch_size: int = 50000,
    use_gpu: bool = False
) -> np.ndarray:
    """
    Compute curvature from normals with automatic KNN and batching.
    
    This is a convenience wrapper that handles KNN computation and batching
    automatically. For more control, use compute_curvature_from_normals().
    
    Parameters
    ----------
    points : np.ndarray
        Point coordinates, shape (N, 3)
    normals : np.ndarray
        Normal vectors, shape (N, 3)
    k : int, optional
        Number of nearest neighbors (default: 10)
    batch_size : int, optional
        Batch size for GPU processing (default: 50000)
    use_gpu : bool, optional
        Whether to use GPU acceleration if available (default: False)
        
    Returns
    -------
    curvature : np.ndarray
        Curvature values, shape (N,), dtype float32
        
    Examples
    --------
    >>> import numpy as np
    >>> points = np.random.rand(1000, 3).astype(np.float32)
    >>> normals = np.random.rand(1000, 3).astype(np.float32)
    >>> normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    >>> 
    >>> curvature = compute_curvature_from_normals_batched(points, normals, k=10)
    >>> assert curvature.shape == (1000,)
    >>> assert curvature.dtype == np.float32
    
    Notes
    -----
    Automatically selects between CPU (sklearn) and GPU (cuML) KNN.
    GPU path requires cuML to be installed.
    """
    N = len(points)
    
    # GPU path
    if use_gpu and CUPY_AVAILABLE:
        try:
            from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
            
            # Transfer to GPU
            points_gpu = cp.asarray(points, dtype=cp.float32)
            normals_gpu = cp.asarray(normals, dtype=cp.float32)
            curvature_gpu = cp.zeros(N, dtype=cp.float32)
            
            # Build GPU KNN
            knn = cuNearestNeighbors(n_neighbors=k, metric='euclidean')
            knn.fit(points_gpu)
            
            # Process in batches
            num_batches = (N + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, N)
                
                batch_points = points_gpu[start_idx:end_idx]
                _, indices = knn.kneighbors(batch_points)
                
                # Compute curvature for batch
                batch_normals = normals_gpu[start_idx:end_idx]
                batch_curvature = compute_curvature_from_normals(
                    batch_points, batch_normals, indices
                )
                
                curvature_gpu[start_idx:end_idx] = batch_curvature
            
            # Transfer back to CPU
            return cp.asnumpy(curvature_gpu)
            
        except Exception as e:
            logger.warning(f"GPU curvature failed: {e}, falling back to CPU")
            # Fall through to CPU path
    
    # CPU path - Use KNNEngine for GPU-accelerated KNN when available
    from ign_lidar.optimization import KNNEngine
    
    # Build KNN index using KNNEngine (auto GPU/CPU selection)
    engine = KNNEngine()
    distances, indices = engine.search(points, k=k)
    
    # Compute curvature
    curvature = compute_curvature_from_normals(points, normals, indices)
    
    return curvature

