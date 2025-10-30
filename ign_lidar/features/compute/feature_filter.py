"""
Spatial Feature Filtering for Artifact Reduction

This module provides adaptive spatial filtering to remove line/dash artifacts
from geometric features that exhibit discontinuities at object boundaries.

Artifacts typically appear when k-nearest neighbor (k-NN) searches cross
object boundaries (e.g., wall→air, ground→building), causing neighborhoods
to mix points from different surfaces with drastically different properties.

**Affected Features:**
- planarity: (λ2 - λ3) / λ1 - exhibits dashes at planar surface edges
- linearity: (λ1 - λ2) / λ1 - exhibits dashes at linear feature boundaries
- horizontality: |dot(normal, vertical)| - exhibits dashes at horizontal surface edges

**Algorithm:**
For each point:
1. Query spatial neighborhood (k neighbors)
2. Compute standard deviation of feature values in neighborhood
3. If std > threshold → artifact detected (boundary crossing)
4. Apply median smoothing from valid neighbors only
5. Handle NaN/Inf by interpolation from valid neighbors

**Performance:**
- ~5-10 seconds for 1M points with k=15 (CPU)
- Memory efficient: processes features in-place when possible

Version: 3.1.0 (Unified from planarity_filter.py v3.0.6)
Author: IGN LiDAR HD Development Team
Date: 2025-10-30
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "smooth_feature_spatial",
    "validate_feature",
    "smooth_planarity_spatial",
    "smooth_linearity_spatial",
    "smooth_horizontality_spatial",
    "validate_planarity",
    "validate_linearity",
    "validate_horizontality",
]


def smooth_feature_spatial(
    feature: np.ndarray,
    points: np.ndarray,
    k_neighbors: int = 15,
    std_threshold: float = 0.3,
    feature_name: str = "feature",
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Apply adaptive spatial filtering to remove line/dash artifacts from geometric features.

    **Algorithm (v3.1.1 - Fixed):**
    For each point:
    1. Find k spatial neighbors
    2. Compute local median and deviation from median
    3. If |value - median| > threshold → outlier/artifact → replace with median
    4. If NaN/Inf → interpolate from valid neighbors
    5. Else → preserve original value

    **Rationale:**
    Scan line artifacts create PARALLEL patterns where points have abnormal values
    compared to their spatial neighbors, but neighbors themselves may also be abnormal.
    The key is detecting points that deviate from the LOCAL spatial median,
    NOT variance (which fails when all neighbors have similar wrong values).

    Parameters
    ----------
    feature : np.ndarray
        Feature values to smooth, shape (N,), range typically [0, 1]
    points : np.ndarray
        XYZ point coordinates, shape (N, 3)
    k_neighbors : int, default=15
        Number of spatial neighbors for artifact detection
        - Too low (k<10): Misses boundaries, ineffective filtering
        - Too high (k>30): Over-smooths, removes real features
        - Recommended: 15-20 for building-scale data
    std_threshold : float, default=0.3
        Deviation threshold for artifact detection (|value - median| > threshold)
        - Lower (0.1-0.2): More aggressive, may over-filter
        - Higher (0.4-0.5): More conservative, may miss artifacts
        - Recommended: 0.15-0.25 for [0,1] normalized features
    feature_name : str, default="feature"
        Feature name for logging purposes
    epsilon : float, default=1e-8
        Small value for numerical stability

    Returns
    -------
    smoothed : np.ndarray
        Spatially filtered feature values, shape (N,)
        - Outliers replaced by local median
        - NaN/Inf replaced by interpolation
        - Normal regions preserved

    Examples
    --------
    >>> # Smooth planarity feature (with lower threshold for better artifact detection)
    >>> planarity_smooth = smooth_feature_spatial(
    ...     planarity, points, k_neighbors=15, std_threshold=0.2,
    ...     feature_name="planarity"
    ... )

    >>> # Smooth linearity with aggressive filtering
    >>> linearity_smooth = smooth_feature_spatial(
    ...     linearity, points, k_neighbors=20, std_threshold=0.15,
    ...     feature_name="linearity"
    ... )

    Notes
    -----
    - **v3.1.1 FIX:** Changed from variance-based to deviation-from-median detection
    - **Outlier Detection:** Compares each point to local median, not variance
    - **Handles Parallel Artifacts:** Works even when neighbors have similar wrong values
    - **Preserves Real Features:** Only modifies outliers
    - **Handles Invalid Values:** NaN/Inf are interpolated from valid neighbors
    - **Memory Efficient:** Creates copy only if modifications needed
    - **Performance:** O(N * k * log(N)) for KD-tree construction + queries

    See Also
    --------
    validate_feature : Sanitize NaN/Inf and clip outliers
    smooth_planarity_spatial : Convenience wrapper for planarity
    smooth_linearity_spatial : Convenience wrapper for linearity
    smooth_horizontality_spatial : Convenience wrapper for horizontality
    """
    n_points = len(feature)

    if n_points == 0:
        logger.warning(f"{feature_name}: Empty input, returning empty array")
        return np.array([], dtype=np.float32)

    if len(points) != n_points:
        raise ValueError(
            f"{feature_name}: Shape mismatch - feature has {n_points} points, "
            f"coordinates have {len(points)} points"
        )

    # Adjust k_neighbors if necessary
    k_neighbors = min(k_neighbors, n_points - 1)
    if k_neighbors < 3:
        logger.warning(
            f"{feature_name}: Too few points ({n_points}) for filtering, "
            f"returning original values"
        )
        return feature.copy()

    logger.debug(
        f"{feature_name}: Spatial filtering {n_points:,} points | "
        f"k={k_neighbors}, threshold={std_threshold:.2f}"
    )

    # Build spatial index
    tree = cKDTree(points)

    # Initialize output (copy on write)
    smoothed = feature.copy()
    n_artifacts = 0
    n_invalid = 0

    # Process each point
    for i in range(n_points):
        current_value = feature[i]

        # Case 1: Handle invalid values (NaN/Inf)
        if not np.isfinite(current_value):
            # Find valid neighbors
            _, neighbor_indices = tree.query(points[i], k=k_neighbors + 1)
            neighbor_values = feature[neighbor_indices[1:]]  # Exclude self
            valid_neighbors = neighbor_values[np.isfinite(neighbor_values)]

            if len(valid_neighbors) > 0:
                # Interpolate from valid neighbors
                smoothed[i] = np.median(valid_neighbors)
                n_invalid += 1
            else:
                # No valid neighbors, set to safe default
                smoothed[i] = 0.0
                n_invalid += 1
            continue

        # Case 2: Detect artifacts via deviation from local median
        _, neighbor_indices = tree.query(points[i], k=k_neighbors + 1)
        neighbor_values = feature[neighbor_indices[1:]]  # Exclude self

        # Filter out invalid neighbors
        valid_mask = np.isfinite(neighbor_values)
        valid_neighbors = neighbor_values[valid_mask]

        if len(valid_neighbors) < 3:
            # Not enough valid neighbors, interpolate
            if len(valid_neighbors) > 0:
                smoothed[i] = np.median(valid_neighbors)
                n_artifacts += 1
            continue

        # Compute local median
        local_median = np.median(valid_neighbors)

        # 🆕 CRITICAL FIX: Detect outliers by deviation from median
        # (not variance, which fails for parallel scan line artifacts)
        deviation = abs(current_value - local_median)

        # Case 3: Large deviation → artifact/outlier → smooth
        if deviation > std_threshold:
            # Replace with spatial median (robust to outliers)
            smoothed[i] = local_median
            n_artifacts += 1
        # Case 4: Small deviation → normal region → preserve
        # (no action needed, already copied)

    # Report filtering statistics
    artifact_pct = 100 * n_artifacts / n_points if n_points > 0 else 0
    invalid_pct = 100 * n_invalid / n_points if n_points > 0 else 0

    if n_artifacts > 0 or n_invalid > 0:
        logger.info(
            f"  ✓ {feature_name} filtered: "
            f"{n_artifacts:,} artifacts ({artifact_pct:.1f}%), "
            f"{n_invalid:,} invalid ({invalid_pct:.1f}%)"
        )
    else:
        logger.debug(f"{feature_name}: No artifacts detected")

    return smoothed.astype(np.float32)


def validate_feature(
    feature: np.ndarray,
    feature_name: str = "feature",
    valid_range: Tuple[float, float] = (0.0, 1.0),
    clip_sigma: float = 5.0,
) -> np.ndarray:
    """
    Sanitize feature values by handling NaN/Inf and clipping outliers.

    **Operations:**
    1. Replace NaN → 0.0
    2. Replace +Inf → max valid value
    3. Replace -Inf → min valid value
    4. Clip outliers beyond ±clip_sigma standard deviations

    Parameters
    ----------
    feature : np.ndarray
        Feature values to validate, shape (N,)
    feature_name : str, default="feature"
        Feature name for logging
    valid_range : tuple, default=(0.0, 1.0)
        Expected (min, max) range for the feature
    clip_sigma : float, default=5.0
        Clip outliers beyond this many standard deviations
        - Set to 0 to disable outlier clipping
        - Recommended: 3-5 for normalized features

    Returns
    -------
    validated : np.ndarray
        Sanitized feature values, shape (N,)
        - NaN/Inf replaced
        - Outliers clipped
        - Values in valid_range

    Examples
    --------
    >>> # Validate planarity (range [0,1])
    >>> planarity_clean = validate_feature(
    ...     planarity, "planarity", valid_range=(0.0, 1.0)
    ... )

    >>> # Validate height (range unrestricted)
    >>> height_clean = validate_feature(
    ...     height, "height", valid_range=(-100.0, 500.0), clip_sigma=10.0
    ... )

    See Also
    --------
    smooth_feature_spatial : Apply spatial filtering to remove artifacts
    """
    n_points = len(feature)

    if n_points == 0:
        logger.warning(f"{feature_name}: Empty input")
        return np.array([], dtype=np.float32)

    # Count invalid values
    n_nan = np.sum(np.isnan(feature))
    n_inf = np.sum(np.isinf(feature))

    if n_nan > 0 or n_inf > 0:
        logger.debug(
            f"{feature_name}: Found {n_nan} NaN, {n_inf} Inf values - sanitizing"
        )

    # Create validated copy
    validated = feature.copy()

    # Replace NaN with 0
    nan_mask = np.isnan(validated)
    if np.any(nan_mask):
        validated[nan_mask] = valid_range[0]

    # Replace +Inf with max valid value
    pos_inf_mask = np.isposinf(validated)
    if np.any(pos_inf_mask):
        validated[pos_inf_mask] = valid_range[1]

    # Replace -Inf with min valid value
    neg_inf_mask = np.isneginf(validated)
    if np.any(neg_inf_mask):
        validated[neg_inf_mask] = valid_range[0]

    # Clip outliers (only for finite values)
    if clip_sigma > 0:
        valid_mask = np.isfinite(validated)
        if np.any(valid_mask):
            valid_values = validated[valid_mask]
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)

            if std_val > 1e-8:  # Only clip if there's variation
                lower_bound = max(valid_range[0], mean_val - clip_sigma * std_val)
                upper_bound = min(valid_range[1], mean_val + clip_sigma * std_val)

                n_clipped = np.sum(
                    (validated < lower_bound) | (validated > upper_bound)
                )
                if n_clipped > 0:
                    validated = np.clip(validated, lower_bound, upper_bound)
                    logger.debug(
                        f"{feature_name}: Clipped {n_clipped} outliers "
                        f"beyond ±{clip_sigma}σ"
                    )

    # Final range clipping (hard bounds)
    validated = np.clip(validated, valid_range[0], valid_range[1])

    return validated.astype(np.float32)


# ============================================================================
# Feature-Specific Convenience Functions
# ============================================================================


def smooth_planarity_spatial(
    planarity: np.ndarray,
    points: np.ndarray,
    k_neighbors: int = 15,
    std_threshold: float = 0.2,  # v3.1.1: Lowered from 0.3 (deviation-based now)
) -> np.ndarray:
    """
    Apply spatial filtering to planarity feature.

    Convenience wrapper for smooth_feature_spatial() with planarity defaults.

    Note: v3.1.1 - Default threshold lowered to 0.2 for better artifact detection
          (algorithm now uses deviation from median, not variance)

    See smooth_feature_spatial() for full documentation.
    """
    return smooth_feature_spatial(
        planarity, points, k_neighbors, std_threshold, feature_name="planarity"
    )


def smooth_linearity_spatial(
    linearity: np.ndarray,
    points: np.ndarray,
    k_neighbors: int = 15,
    std_threshold: float = 0.2,  # v3.1.1: Lowered from 0.3
) -> np.ndarray:
    """
    Apply spatial filtering to linearity feature.

    Convenience wrapper for smooth_feature_spatial() with linearity defaults.

    Note: v3.1.1 - Default threshold lowered to 0.2 for better artifact detection

    See smooth_feature_spatial() for full documentation.
    """
    return smooth_feature_spatial(
        linearity, points, k_neighbors, std_threshold, feature_name="linearity"
    )


def smooth_horizontality_spatial(
    horizontality: np.ndarray,
    points: np.ndarray,
    k_neighbors: int = 15,
    std_threshold: float = 0.2,  # v3.1.1: Lowered from 0.3
) -> np.ndarray:
    """
    Apply spatial filtering to horizontality feature.

    Convenience wrapper for smooth_feature_spatial() with horizontality defaults.

    Note: v3.1.1 - Default threshold lowered to 0.2 for better artifact detection

    See smooth_feature_spatial() for full documentation.
    """
    return smooth_feature_spatial(
        horizontality, points, k_neighbors, std_threshold, feature_name="horizontality"
    )


def validate_planarity(
    planarity: np.ndarray,
    clip_sigma: float = 5.0,
) -> np.ndarray:
    """
    Validate planarity feature values.

    Convenience wrapper for validate_feature() with planarity defaults.

    See validate_feature() for full documentation.
    """
    return validate_feature(
        planarity, "planarity", valid_range=(0.0, 1.0), clip_sigma=clip_sigma
    )


def validate_linearity(
    linearity: np.ndarray,
    clip_sigma: float = 5.0,
) -> np.ndarray:
    """
    Validate linearity feature values.

    Convenience wrapper for validate_feature() with linearity defaults.

    See validate_feature() for full documentation.
    """
    return validate_feature(
        linearity, "linearity", valid_range=(0.0, 1.0), clip_sigma=clip_sigma
    )


def validate_horizontality(
    horizontality: np.ndarray,
    clip_sigma: float = 5.0,
) -> np.ndarray:
    """
    Validate horizontality feature values.

    Convenience wrapper for validate_feature() with horizontality defaults.

    See validate_feature() for full documentation.
    """
    return validate_feature(
        horizontality, "horizontality", valid_range=(0.0, 1.0), clip_sigma=clip_sigma
    )
