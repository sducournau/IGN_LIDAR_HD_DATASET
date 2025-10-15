"""
Canonical implementation of curvature feature computation.

This module provides unified curvature computation replacing duplicates in:
- features.py (CPU)
- features_gpu.py (GPU)
- features_gpu_chunked.py (GPU chunked)
- features_boundary.py (boundary-aware)
"""

import numpy as np
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


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
