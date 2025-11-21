"""
Planarity Filtering Module - Artifact Reduction

This module provides spatial filtering to reduce line/dash artifacts
in planarity features. These artifacts typically occur at object
boundaries where neighborhoods cross edges.

Author: Simon Ducournau
Date: October 30, 2025
Version: 3.0.6
"""

import logging
from typing import Tuple

import numpy as np

from ign_lidar.optimization import cKDTree  # GPU-accelerated drop-in replacement
from ign_lidar.optimization.gpu_accelerated_ops import knn

logger = logging.getLogger(__name__)


def smooth_planarity_spatial(
    planarity: np.ndarray,
    points: np.ndarray,
    k_neighbors: int = 15,
    std_threshold: float = 0.3,
    min_valid_neighbors: int = 5,
) -> Tuple[np.ndarray, dict]:
    """
    Apply spatial smoothing to planarity to reduce line/dash artifacts.

    Artifacts typically occur at object boundaries where the neighborhood
    spans multiple surfaces. This function detects and corrects these cases
    by using the median of spatially coherent neighbors.

    Algorithm:
    1. For each point, find k nearest neighbors
    2. Compute std dev of neighbor planarity values
    3. If std > threshold (indicates boundary crossing), replace with median
    4. Handle NaN/Inf by interpolating from valid neighbors

    Parameters
    ----------
    planarity : np.ndarray
        Planarity feature values, shape (N,)
    points : np.ndarray
        Point cloud coordinates, shape (N, 3)
    k_neighbors : int, optional
        Number of neighbors for spatial filtering (default: 15)
    std_threshold : float, optional
        Standard deviation threshold for artifact detection (default: 0.3)
        Higher values = less aggressive filtering
    min_valid_neighbors : int, optional
        Minimum number of valid neighbors required for interpolation
        (default: 5)

    Returns
    -------
    smoothed : np.ndarray
        Smoothed planarity values, shape (N,)
    stats : dict
        Statistics about the filtering:
        - n_artifacts_fixed: Number of boundary artifacts corrected
        - n_nan_fixed: Number of NaN/Inf values interpolated
        - n_unchanged: Number of values left unchanged

    Examples
    --------
    >>> planarity = np.array([0.8, np.nan, 0.9, 0.1, 0.85])  # Has NaN artifact
    >>> points = np.random.rand(5, 3)
    >>> smoothed, stats = smooth_planarity_spatial(planarity, points)
    >>> assert np.isfinite(smoothed).all()
    >>> print(f"Fixed {stats['n_nan_fixed']} NaN values")

    Notes
    -----
    - This filter is designed to be conservative: it only modifies values
      that show clear signs of artifacts
    - The median is used (not mean) to be robust to outliers
    - Original planarity computation is unchanged, only post-processing

    See Also
    --------
    compute_planarity : Original planarity computation
    compute_covariances_from_neighbors : Covariance computation
    """
    if len(planarity) == 0:
        return planarity.copy(), {
            "n_artifacts_fixed": 0,
            "n_nan_fixed": 0,
            "n_unchanged": 0,
        }

    smoothed = planarity.copy()
    n_artifacts_fixed = 0
    n_nan_fixed = 0

    # 🔥 GPU-accelerated KNN for spatial smoothing
    k_query = min(k_neighbors + 1, len(points))
    
    distances, neighbor_indices = knn(
        points,
        points,
        k=k_query
    )
    # Remove self (first neighbor)
    neighbor_indices = neighbor_indices[:, 1:]

    for i in range(len(points)):
        current_value = planarity[i]
        neighbors_idx = neighbor_indices[i]

        # Get neighbor planarity values
        neighbor_values = planarity[neighbors_idx]

        # Case 1: Current value is NaN/Inf
        if not np.isfinite(current_value):
            # Interpolate from valid neighbors
            valid_neighbors = neighbor_values[np.isfinite(neighbor_values)]

            if len(valid_neighbors) >= min_valid_neighbors:
                smoothed[i] = np.median(valid_neighbors)
                n_nan_fixed += 1
            else:
                # Not enough valid neighbors, use global median as fallback
                finite_mask = np.isfinite(planarity)
                global_median = np.median(planarity[finite_mask])
                fallback = global_median if np.isfinite(global_median) else 0.5
                smoothed[i] = fallback
                n_nan_fixed += 1

            continue

        # Case 2: Check for boundary artifact (high variance in neighborhood)
        valid_neighbors = neighbor_values[np.isfinite(neighbor_values)]

        if len(valid_neighbors) >= min_valid_neighbors:
            neighbor_std = np.std(valid_neighbors)

            # High variance indicates boundary crossing
            if neighbor_std > std_threshold:
                # Replace with median of neighbors
                smoothed[i] = np.median(valid_neighbors)
                n_artifacts_fixed += 1

    n_unchanged = len(planarity) - n_artifacts_fixed - n_nan_fixed

    stats = {
        "n_artifacts_fixed": n_artifacts_fixed,
        "n_nan_fixed": n_nan_fixed,
        "n_unchanged": n_unchanged,
    }

    if n_artifacts_fixed > 0 or n_nan_fixed > 0:
        logger.info(
            f"Planarity filtering: {n_artifacts_fixed} boundary artifacts, "
            f"{n_nan_fixed} NaN/Inf fixed"
        )

    return smoothed, stats


def validate_planarity(
    planarity: np.ndarray,
    clip_outliers: bool = True,
    sigma: float = 3.0,
) -> Tuple[np.ndarray, dict]:
    """
    Validate and sanitize planarity values.

    Performs basic validation:
    - Detects NaN/Inf values
    - Clips outliers beyond sigma standard deviations
    - Ensures values are in [0, 1] range

    Parameters
    ----------
    planarity : np.ndarray
        Planarity values to validate
    clip_outliers : bool, optional
        Whether to clip outliers (default: True)
    sigma : float, optional
        Number of standard deviations for outlier clipping (default: 3.0)

    Returns
    -------
    validated : np.ndarray
        Validated planarity values
    stats : dict
        Validation statistics

    Examples
    --------
    >>> planarity = np.array([0.5, 0.9, np.inf, -0.1, 1.5])
    >>> validated, stats = validate_planarity(planarity)
    >>> assert validated.min() >= 0.0 and validated.max() <= 1.0
    """
    validated = planarity.copy()

    # Count invalid values
    n_nan = np.sum(np.isnan(validated))
    n_inf = np.sum(np.isinf(validated))
    n_out_of_range = np.sum((validated < 0) | (validated > 1))

    # Replace NaN/Inf with median
    if n_nan > 0 or n_inf > 0:
        valid_mask = np.isfinite(validated)
        if np.any(valid_mask):
            median_value = np.median(validated[valid_mask])
        else:
            median_value = 0.5  # Fallback

        validated[~np.isfinite(validated)] = median_value

    # Clip to [0, 1] range
    validated = np.clip(validated, 0.0, 1.0)

    # Optional: Clip outliers
    if clip_outliers and len(validated) > 0:
        mean = np.mean(validated)
        std = np.std(validated)
        lower_bound = max(0.0, mean - sigma * std)
        upper_bound = min(1.0, mean + sigma * std)
        validated = np.clip(validated, lower_bound, upper_bound)

    stats = {
        "n_nan": n_nan,
        "n_inf": n_inf,
        "n_out_of_range": n_out_of_range,
        "valid_range": (validated.min(), validated.max()),
    }

    if n_nan > 0 or n_inf > 0 or n_out_of_range > 0:
        logger.warning(
            f"Planarity validation: {n_nan} NaN, {n_inf} Inf, "
            f"{n_out_of_range} out of range"
        )

    return validated, stats
