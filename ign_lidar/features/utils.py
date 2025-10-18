"""
Shared Utilities for Feature Computation

This module provides common utilities used across CPU, GPU, and boundary feature
computation modules to reduce code duplication and ensure consistency.

Functions:
    - build_kdtree: Build KDTree with optimal default parameters
    - compute_local_eigenvalues: Compute eigenvalues from k-nearest neighbors
    - validate_point_cloud: Validate point cloud array
    - validate_normals: Validate normal vectors
    - validate_k_neighbors: Validate k parameter

Author: IGN LIDAR HD Dataset Project
Date: January 2025
"""

from typing import Optional, Tuple, Union
import numpy as np
from sklearn.neighbors import KDTree
import logging

logger = logging.getLogger(__name__)


def build_kdtree(
    points: np.ndarray,
    metric: str = 'euclidean',
    leaf_size: int = 30
) -> KDTree:
    """
    Build KDTree with optimal default parameters.
    
    This function provides a consistent interface for KDTree construction
    across all feature computation modules.
    
    Args:
        points: [N, 3] point cloud coordinates
        metric: Distance metric (default: 'euclidean')
               Other options: 'manhattan', 'chebyshev', 'minkowski'
        leaf_size: Leaf size for tree construction (default: 30)
                   - 30 for CPU operations (balanced speed/memory)
                   - 40 for GPU fallbacks (larger batches)
                   - 20 for small datasets (faster queries)
    
    Returns:
        KDTree instance ready for queries
    
    Example:
        >>> tree = build_kdtree(points)
        >>> distances, indices = tree.query(points, k=10)
    
    Note:
        The default leaf_size=30 is optimal for most CPU-based operations.
        For GPU fallback code, consider leaf_size=40 for better batching.
    """
    return KDTree(points, metric=metric, leaf_size=leaf_size)


def compute_local_eigenvalues(
    points: np.ndarray,
    tree: Optional[KDTree] = None,
    k: int = 20,
    return_tree: bool = False,
    leaf_size: int = 30
) -> Union[np.ndarray, Tuple[np.ndarray, KDTree]]:
    """
    Compute local eigenvalues from k-nearest neighbors using PCA.
    
    This is the standard eigenvalue computation used across all feature
    modules. It builds covariance matrices from local neighborhoods and
    computes their eigenvalues.
    
    Args:
        points: [N, 3] point cloud coordinates
        tree: Pre-built KDTree (optional, will build if None)
        k: Number of neighbors for local neighborhood
        return_tree: If True, return (eigenvalues, tree) tuple
        leaf_size: Leaf size for KDTree if building new tree
    
    Returns:
        eigenvalues: [N, 3] sorted eigenvalues (λ3 ≤ λ2 ≤ λ1)
                     λ1 = largest eigenvalue (main direction)
                     λ2 = medium eigenvalue
                     λ3 = smallest eigenvalue (normal direction)
        tree: KDTree (only if return_tree=True)
    
    Example:
        >>> eigenvalues = compute_local_eigenvalues(points, k=20)
        >>> planarity = (eigenvalues[:, 1] - eigenvalues[:, 0]) / eigenvalues[:, 2]
    
    Note:
        This function uses vectorized NumPy operations (einsum) for
        maximum performance. The covariance matrices are computed as:
        C = (1/(k-1)) * Σ(p_i - μ)(p_i - μ)^T
    
    Algorithm:
        1. Find k-nearest neighbors for each point
        2. Compute local centroids (means)
        3. Center points around centroids
        4. Build covariance matrices: C = X^T X / (k-1)
        5. Compute eigenvalues via np.linalg.eigvalsh
    """
    # Build KDTree if not provided
    if tree is None:
        tree = build_kdtree(points, leaf_size=leaf_size)
    
    # Find k-nearest neighbors
    _, indices = tree.query(points, k=k)
    
    # Get neighbor coordinates [N, k, 3]
    neighbors_all = points[indices]
    
    # Compute local centroids [N, 1, 3]
    centroids = neighbors_all.mean(axis=1, keepdims=True)
    
    # Center points around centroids [N, k, 3]
    centered = neighbors_all - centroids
    
    # Compute covariance matrices [N, 3, 3]
    # Using einsum for efficient batch matrix multiplication
    cov_matrices = np.einsum('nki,nkj->nij', centered, centered) / (k - 1)
    
    # Compute eigenvalues [N, 3]
    # eigvalsh returns sorted eigenvalues (smallest to largest)
    eigenvalues = np.linalg.eigvalsh(cov_matrices)
    
    if return_tree:
        return eigenvalues, tree
    return eigenvalues


def validate_point_cloud(
    points: np.ndarray,
    min_points: int = 1,
    check_finite: bool = True,
    param_name: str = "points"
) -> None:
    """
    Validate point cloud array.
    
    Performs comprehensive validation of point cloud inputs to ensure
    they meet requirements for feature computation.
    
    Args:
        points: Point cloud to validate
        min_points: Minimum required points (default: 1)
        check_finite: If True, check for NaN/Inf (default: True)
        param_name: Parameter name for error messages (default: "points")
    
    Raises:
        TypeError: If points is not a numpy array
        ValueError: If validation fails (wrong shape, too few points, NaN/Inf)
    
    Example:
        >>> validate_point_cloud(points, min_points=10)
        >>> # Continues if valid, raises ValueError if invalid
    
    Validation Checks:
        1. Type check: Must be numpy.ndarray
        2. Shape check: Must be [N, 3]
        3. Size check: Must have at least min_points
        4. Finite check: No NaN or Inf values (if check_finite=True)
    """
    # Type check
    if not isinstance(points, np.ndarray):
        raise TypeError(
            f"{param_name} must be numpy array, got {type(points).__name__}"
        )
    
    # Shape check
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"{param_name} must be [N, 3], got shape {points.shape}"
        )
    
    # Size check
    if len(points) < min_points:
        raise ValueError(
            f"{param_name} must have at least {min_points} points, "
            f"got {len(points)}"
        )
    
    # Finite check
    if check_finite and not np.isfinite(points).all():
        n_nan = np.isnan(points).sum()
        n_inf = np.isinf(points).sum()
        raise ValueError(
            f"{param_name} contains {n_nan} NaN and {n_inf} Inf values"
        )


def validate_normals(
    normals: np.ndarray,
    num_points: int,
    check_finite: bool = True,
    param_name: str = "normals"
) -> None:
    """
    Validate normal vectors.
    
    Ensures normal vectors have correct shape and valid values.
    
    Args:
        normals: Normal vectors to validate
        num_points: Expected number of points
        check_finite: If True, check for NaN/Inf (default: True)
        param_name: Parameter name for error messages (default: "normals")
    
    Raises:
        TypeError: If normals is not a numpy array
        ValueError: If validation fails (wrong shape, NaN/Inf)
    
    Example:
        >>> validate_normals(normals, num_points=len(points))
        >>> # Continues if valid, raises ValueError if invalid
    
    Validation Checks:
        1. Type check: Must be numpy.ndarray
        2. Shape check: Must be [num_points, 3]
        3. Finite check: No NaN or Inf values (if check_finite=True)
    
    Note:
        This function does NOT check if normals are unit vectors.
        Some algorithms work with non-normalized normals.
    """
    # Type check
    if not isinstance(normals, np.ndarray):
        raise TypeError(
            f"{param_name} must be numpy array, got {type(normals).__name__}"
        )
    
    # Shape check
    expected_shape = (num_points, 3)
    if normals.shape != expected_shape:
        raise ValueError(
            f"{param_name} must be {expected_shape}, got {normals.shape}"
        )
    
    # Finite check
    if check_finite and not np.isfinite(normals).all():
        n_nan = np.isnan(normals).sum()
        n_inf = np.isinf(normals).sum()
        raise ValueError(
            f"{param_name} contains {n_nan} NaN and {n_inf} Inf values"
        )


def validate_k_neighbors(
    k: int,
    num_points: int,
    param_name: str = "k"
) -> None:
    """
    Validate k-neighbors parameter.
    
    Ensures k is valid for the given point cloud size.
    
    Args:
        k: Number of neighbors
        num_points: Number of points in dataset
        param_name: Parameter name for error messages (default: "k")
    
    Raises:
        TypeError: If k is not an integer
        ValueError: If k is invalid (<=0 or >num_points)
    
    Example:
        >>> validate_k_neighbors(k=10, num_points=len(points))
        >>> # Continues if valid, raises ValueError if invalid
    
    Validation Checks:
        1. Type check: Must be int
        2. Range check: Must be > 0
        3. Size check: Must be <= num_points
    
    Note:
        For most algorithms, k should be >= 3 for meaningful results.
        This function only checks k > 0 to be permissive.
    """
    # Type check
    if not isinstance(k, (int, np.integer)):
        raise TypeError(
            f"{param_name} must be integer, got {type(k).__name__}"
        )
    
    # Range check
    if k <= 0:
        raise ValueError(
            f"{param_name} must be positive, got {k}"
        )
    
    # Size check
    if k > num_points:
        raise ValueError(
            f"{param_name}={k} exceeds number of points ({num_points})"
        )


def get_optimal_leaf_size(
    num_points: int,
    use_gpu_fallback: bool = False
) -> int:
    """
    Get optimal KDTree leaf size based on dataset size and usage.
    
    Args:
        num_points: Number of points in dataset
        use_gpu_fallback: If True, optimize for GPU fallback code
    
    Returns:
        Optimal leaf_size value
    
    Rules:
        - Small datasets (<10k points): leaf_size=20
        - Medium datasets (10k-1M points): leaf_size=30
        - Large datasets (>1M points): leaf_size=40
        - GPU fallback: leaf_size=40 (optimized for batching)
    
    Example:
        >>> leaf_size = get_optimal_leaf_size(len(points))
        >>> tree = build_kdtree(points, leaf_size=leaf_size)
    """
    if use_gpu_fallback:
        return 40
    
    if num_points < 10_000:
        return 20
    elif num_points < 1_000_000:
        return 30
    else:
        return 40


# Module-level convenience functions for backward compatibility
def quick_kdtree(points: np.ndarray) -> KDTree:
    """
    Quick KDTree build with automatic parameter selection.
    
    This is a convenience wrapper that automatically selects
    optimal parameters based on dataset size.
    
    Args:
        points: [N, 3] point cloud
    
    Returns:
        KDTree instance
    
    Example:
        >>> tree = quick_kdtree(points)
    """
    leaf_size = get_optimal_leaf_size(len(points))
    return build_kdtree(points, leaf_size=leaf_size)
