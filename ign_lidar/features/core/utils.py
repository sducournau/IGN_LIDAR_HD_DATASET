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
