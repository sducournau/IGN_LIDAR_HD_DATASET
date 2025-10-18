"""
Shared utility functions for feature computation.

This module provides common helper functions used across
the core feature modules.
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def validate_points(points: np.ndarray, min_points: int = 3) -> None:
    """
    Validate point cloud array.
    
    Parameters
    ----------
    points : np.ndarray
        Point cloud array
    min_points : int, optional
        Minimum required number of points (default: 3)
        
    Raises
    ------
    ValueError
        If validation fails
    """
    if not isinstance(points, np.ndarray):
        raise ValueError("points must be a numpy array")
    if points.ndim != 2:
        raise ValueError(f"points must be 2D array, got {points.ndim}D")
    if points.shape[1] != 3:
        raise ValueError(f"points must have 3 columns (X, Y, Z), got {points.shape[1]}")
    if points.shape[0] < min_points:
        raise ValueError(f"points must have at least {min_points} rows, got {points.shape[0]}")


def validate_eigenvalues(eigenvalues: np.ndarray) -> None:
    """
    Validate eigenvalues array.
    
    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues array
        
    Raises
    ------
    ValueError
        If validation fails
    """
    if not isinstance(eigenvalues, np.ndarray):
        raise ValueError("eigenvalues must be a numpy array")
    if eigenvalues.ndim != 2 or eigenvalues.shape[1] != 3:
        raise ValueError(f"eigenvalues must have shape (N, 3), got {eigenvalues.shape}")


def validate_normals(normals: np.ndarray) -> None:
    """
    Validate normal vectors array.
    
    Parameters
    ----------
    normals : np.ndarray
        Normal vectors array
        
    Raises
    ------
    ValueError
        If validation fails
    """
    if not isinstance(normals, np.ndarray):
        raise ValueError("normals must be a numpy array")
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError(f"normals must have shape (N, 3), got {normals.shape}")


def normalize_vectors(vectors: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Normalize vectors to unit length.
    
    Parameters
    ----------
    vectors : np.ndarray
        Array of vectors of shape (N, 3)
    epsilon : float, optional
        Small value to prevent division by zero
        
    Returns
    -------
    normalized : np.ndarray
        Normalized vectors of shape (N, 3)
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # Avoid division by zero
    normalized = vectors / (norms + epsilon)
    return normalized.astype(np.float32)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Safely divide arrays, preventing division by zero.
    
    Parameters
    ----------
    numerator : np.ndarray
        Numerator array
    denominator : np.ndarray
        Denominator array
    epsilon : float, optional
        Small value added to denominator to prevent division by zero
        
    Returns
    -------
    result : np.ndarray
        Division result
    """
    return numerator / (denominator + epsilon)


def compute_covariance_matrix(points: np.ndarray) -> np.ndarray:
    """
    Compute covariance matrix for a set of points.
    
    Parameters
    ----------
    points : np.ndarray
        Point cloud array of shape (N, 3)
        
    Returns
    -------
    cov_matrix : np.ndarray
        Covariance matrix of shape (3, 3)
    """
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    cov_matrix = np.dot(centered.T, centered) / points.shape[0]
    return cov_matrix.astype(np.float32)


def sort_eigenvalues(eigenvalues: np.ndarray, eigenvectors: Optional[np.ndarray] = None) -> Tuple:
    """
    Sort eigenvalues in descending order.
    
    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues array
    eigenvectors : np.ndarray, optional
        Corresponding eigenvectors
        
    Returns
    -------
    sorted_eigenvalues : np.ndarray
        Eigenvalues sorted in descending order
    sorted_eigenvectors : np.ndarray or None
        Eigenvectors sorted accordingly (if provided)
    """
    idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    
    if eigenvectors is not None:
        sorted_eigenvectors = eigenvectors[:, idx]
        return sorted_eigenvalues, sorted_eigenvectors
    
    return sorted_eigenvalues, None


def clip_features(features: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    """
    Clip feature values to specified range.
    
    Parameters
    ----------
    features : np.ndarray
        Feature array
    min_val : float, optional
        Minimum value (default: 0.0)
    max_val : float, optional
        Maximum value (default: 1.0)
        
    Returns
    -------
    clipped : np.ndarray
        Clipped features
    """
    return np.clip(features, min_val, max_val).astype(np.float32)


def compute_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Compute angle (in radians) between vectors.
    
    Parameters
    ----------
    v1 : np.ndarray
        First vector or array of vectors of shape (N, 3)
    v2 : np.ndarray
        Second vector or array of vectors of shape (N, 3) or (3,)
        
    Returns
    -------
    angles : np.ndarray
        Angles in radians, shape (N,)
    """
    # Normalize vectors
    v1_norm = normalize_vectors(v1)
    
    if v2.ndim == 1:
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
        dot_product = np.dot(v1_norm, v2_norm)
    else:
        v2_norm = normalize_vectors(v2)
        dot_product = np.sum(v1_norm * v2_norm, axis=1)
    
    # Clip to valid range for arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angles = np.arccos(dot_product)
    
    return angles.astype(np.float32)


def standardize_features(features: np.ndarray) -> np.ndarray:
    """
    Standardize features to zero mean and unit variance.
    
    Parameters
    ----------
    features : np.ndarray
        Feature array of shape (N, F)
        
    Returns
    -------
    standardized : np.ndarray
        Standardized features
    """
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std == 0] = 1.0  # Avoid division by zero
    
    standardized = (features - mean) / std
    return standardized.astype(np.float32)


def normalize_features(features: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize features to [0, 1] range.
    
    Parameters
    ----------
    features : np.ndarray
        Feature array of shape (N, F)
    method : str, optional
        Normalization method: 'minmax' or 'l2'
        
    Returns
    -------
    normalized : np.ndarray
        Normalized features
    """
    if method == 'minmax':
        min_val = np.min(features, axis=0)
        max_val = np.max(features, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0  # Avoid division by zero
        normalized = (features - min_val) / range_val
        
    elif method == 'l2':
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = features / norms
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized.astype(np.float32)


def handle_nan_inf(features: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    Replace NaN and Inf values with a fill value.
    
    Parameters
    ----------
    features : np.ndarray
        Feature array
    fill_value : float, optional
        Value to replace NaN/Inf with (default: 0.0)
        
    Returns
    -------
    cleaned : np.ndarray
        Cleaned feature array
    """
    features = features.copy()
    features[np.isnan(features)] = fill_value
    features[np.isinf(features)] = fill_value
    return features.astype(np.float32)


def compute_local_frame(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute local coordinate frame from points.
    
    Parameters
    ----------
    points : np.ndarray
        Point cloud array of shape (N, 3)
        
    Returns
    -------
    origin : np.ndarray
        Origin of local frame (centroid), shape (3,)
    axes : np.ndarray
        Local axes (eigenvectors), shape (3, 3)
    """
    centroid = np.mean(points, axis=0)
    cov_matrix = compute_covariance_matrix(points)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort in descending order
    idx = eigenvalues.argsort()[::-1]
    axes = eigenvectors[:, idx]
    
    return centroid.astype(np.float32), axes.astype(np.float32)


# ============================================================================
# GPU-Compatible Utilities (NumPy/CuPy agnostic)
# ============================================================================

def get_array_module(array):
    """
    Get numpy or cupy module for array.
    
    This allows writing code that works with both CPU (NumPy) 
    and GPU (CuPy) arrays.
    
    Parameters
    ----------
    array : np.ndarray or cp.ndarray
        Input array
        
    Returns
    -------
    module : module
        numpy or cupy module
        
    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([1, 2, 3])
    >>> xp = get_array_module(arr)
    >>> assert xp is np
    >>> result = xp.sum(arr)  # Works with both np and cp
    
    Notes
    -----
    This function checks for the CUDA array interface to detect CuPy arrays.
    If CuPy is not installed or array is not on GPU, returns numpy module.
    """
    if hasattr(array, '__cuda_array_interface__'):
        try:
            import cupy as cp
            return cp
        except ImportError:
            logger.warning("CuPy array detected but CuPy not installed, falling back to NumPy")
            return np
    return np


def batched_inverse_3x3(
    matrices: np.ndarray,
    epsilon: float = 1e-12
) -> np.ndarray:
    """
    Compute inverse of many 3x3 matrices using analytic adjugate formula.
    
    This is much faster than np.linalg.inv() for batched small matrices.
    Works with both NumPy and CuPy arrays (GPU-compatible).
    
    This function eliminates duplicated code from:
    - features_gpu.py::_batched_inverse_3x3()
    - features_gpu_chunked.py::_batched_inverse_3x3_gpu()
    
    Parameters
    ----------
    matrices : np.ndarray or cp.ndarray
        Batch of 3x3 matrices, shape (M, 3, 3)
    epsilon : float, optional
        Small value to stabilize near-singular matrices (default: 1e-12)
        
    Returns
    -------
    inv_matrices : np.ndarray or cp.ndarray
        Inverse matrices, shape (M, 3, 3)
        
    Examples
    --------
    >>> import numpy as np
    >>> mats = np.random.rand(1000, 3, 3)
    >>> # Make symmetric positive definite
    >>> mats = np.einsum('mij,mkj->mik', mats, mats)
    >>> inv_mats = batched_inverse_3x3(mats)
    >>> identity = np.einsum('mij,mjk->mik', mats, inv_mats)
    >>> assert np.allclose(identity, np.eye(3), atol=1e-5)
    
    Notes
    -----
    For near-singular matrices (det < epsilon), returns identity matrix.
    Uses analytic cofactor expansion for speed (no LAPACK calls).
    
    Algorithm:
    1. Compute cofactors (adjugate matrix)
    2. Compute determinant
    3. Inverse = adjugate / determinant
    """
    # Get array module (numpy or cupy)
    xp = get_array_module(matrices)
    
    # Extract matrix elements
    a11 = matrices[:, 0, 0]
    a12 = matrices[:, 0, 1]
    a13 = matrices[:, 0, 2]
    a21 = matrices[:, 1, 0]
    a22 = matrices[:, 1, 1]
    a23 = matrices[:, 1, 2]
    a31 = matrices[:, 2, 0]
    a32 = matrices[:, 2, 1]
    a33 = matrices[:, 2, 2]
    
    # Cofactors (adjugate matrix elements)
    # Cofactor C_ij = (-1)^(i+j) * det(M_ij) where M_ij is minor
    c11 = a22 * a33 - a23 * a32
    c12 = -(a21 * a33 - a23 * a31)
    c13 = a21 * a32 - a22 * a31
    c21 = -(a12 * a33 - a13 * a32)
    c22 = a11 * a33 - a13 * a31
    c23 = -(a11 * a32 - a12 * a31)
    c31 = a12 * a23 - a13 * a22
    c32 = -(a11 * a23 - a13 * a21)
    c33 = a11 * a22 - a12 * a21
    
    # Determinant using first row expansion
    det = a11 * c11 + a12 * c12 + a13 * c13
    
    # Stabilize near-singular matrices
    small = xp.abs(det) < epsilon
    det_safe = det + small.astype(det.dtype) * epsilon
    inv_det = 1.0 / det_safe
    
    # Compute inverse: A^-1 = adjugate(A) / det(A)
    # Note: adjugate = transpose of cofactor matrix
    inv = xp.empty_like(matrices)
    inv[:, 0, 0] = c11 * inv_det
    inv[:, 0, 1] = c21 * inv_det  # Transpose: C^T[0,1] = C[1,0]
    inv[:, 0, 2] = c31 * inv_det  # Transpose: C^T[0,2] = C[2,0]
    inv[:, 1, 0] = c12 * inv_det  # Transpose: C^T[1,0] = C[0,1]
    inv[:, 1, 1] = c22 * inv_det
    inv[:, 1, 2] = c32 * inv_det  # Transpose: C^T[1,2] = C[2,1]
    inv[:, 2, 0] = c13 * inv_det  # Transpose: C^T[2,0] = C[0,2]
    inv[:, 2, 1] = c23 * inv_det  # Transpose: C^T[2,1] = C[1,2]
    inv[:, 2, 2] = c33 * inv_det
    
    # For near-singular matrices, use identity
    eye_3x3 = xp.eye(3, dtype=inv.dtype)
    inv = xp.where(small[:, None, None], eye_3x3, inv)
    
    return inv


def inverse_power_iteration(
    matrices: np.ndarray,
    num_iterations: int = 8,
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Compute eigenvector for smallest eigenvalue using inverse power iteration.
    
    For symmetric 3x3 covariance matrices, this is 10-50× faster than
    full eigendecomposition (np.linalg.eigh or cupy.linalg.eigh).
    Works with both NumPy and CuPy arrays (GPU-compatible).
    
    This function eliminates duplicated code from:
    - features_gpu.py::_smallest_eigenvector_from_covariances()
    - features_gpu_chunked.py::_smallest_eigenvector_from_covariances_gpu()
    
    Parameters
    ----------
    matrices : np.ndarray or cp.ndarray
        Symmetric 3x3 covariance matrices, shape (M, 3, 3)
    num_iterations : int, optional
        Number of power iterations (default: 8, sufficient for convergence)
    epsilon : float, optional
        Regularization to avoid singularities (default: 1e-6)
        
    Returns
    -------
    eigenvectors : np.ndarray or cp.ndarray
        Normalized eigenvectors for smallest eigenvalue, shape (M, 3)
        Oriented upward (positive Z component)
        
    Examples
    --------
    >>> import numpy as np
    >>> # Create symmetric covariance matrices
    >>> points = np.random.rand(100, 10, 3)  # 100 neighborhoods of 10 points
    >>> centered = points - points.mean(axis=1, keepdims=True)
    >>> cov = np.einsum('mki,mkj->mij', centered, centered) / 10
    >>> eigenvec = inverse_power_iteration(cov, num_iterations=8)
    >>> assert eigenvec.shape == (100, 3)
    >>> assert np.allclose(np.linalg.norm(eigenvec, axis=1), 1.0)
    
    Notes
    -----
    Algorithm (Inverse Power Method):
    1. Regularize matrices: C' = C + ε*I
    2. Compute inverse: C'^-1
    3. Power iteration: v = C'^-1 @ v, normalize
    4. Orient upward: flip if v[2] < 0
    
    This method is ideal for computing surface normals from covariances,
    as the smallest eigenvector corresponds to the surface normal direction.
    
    Convergence is typically achieved in 6-10 iterations for most cases.
    """
    xp = get_array_module(matrices)
    M = matrices.shape[0]
    
    # Regularize to avoid singularities
    reg_matrices = matrices + epsilon * xp.eye(3, dtype=matrices.dtype)[None, ...]
    
    # Compute batched inverse using analytic formula
    inv_matrices = batched_inverse_3x3(reg_matrices, epsilon=epsilon * 10)
    
    # Initialize random vectors
    v = xp.ones((M, 3), dtype=matrices.dtype)
    v = v / xp.linalg.norm(v, axis=1, keepdims=True)
    
    # Power iteration
    for _ in range(num_iterations):
        # v = inv_matrices @ v
        v = xp.einsum('mij,mj->mi', inv_matrices, v)
        # Normalize
        norms = xp.linalg.norm(v, axis=1, keepdims=True)
        norms = xp.maximum(norms, epsilon)
        v = v / norms
    
    # Orient upward (positive Z)
    flip_mask = v[:, 2] < 0
    v[flip_mask] *= -1
    
    # Handle invalid results (NaN/Inf)
    invalid = ~xp.isfinite(v).all(axis=1)
    if xp.any(invalid):
        default = xp.array([0.0, 0.0, 1.0], dtype=v.dtype)
        v = xp.where(invalid[:, None], default, v)
        logger.warning(f"Found {xp.sum(invalid)} invalid eigenvectors, replaced with [0,0,1]")
    
    return v
