"""
Is Ground Feature Computation

This module provides the "is_ground" binary feature with DTM augmentation support.
The feature indicates whether each point is classified as ground (ASPRS class 2)
or has been augmented from DTM data.

Key Features:
- Binary indicator: 1 for ground points, 0 for non-ground
- DTM-aware: Detects synthetic ground points from DTM augmentation
- Multi-source: Supports original LiDAR ground + DTM-augmented points
- Efficient: Simple boolean operation on classification array

Use Cases:
- Height computation: Identify ground reference points
- Classification validation: Check ground/non-ground separation
- DTM quality: Assess impact of DTM augmentation
- Feature engineering: Binary ground indicator for ML models

Author: IGN LiDAR HD Development Team
Date: October 25, 2025
Version: 3.1.0
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_is_ground(
    classification: np.ndarray,
    synthetic_flags: Optional[np.ndarray] = None,
    ground_class: int = 2,
    include_synthetic: bool = True,
) -> np.ndarray:
    """
    Compute binary is_ground feature for point cloud.

    Returns 1 for ground points (ASPRS class 2) and 0 for non-ground points.
    Optionally includes synthetic ground points added via DTM augmentation.

    Parameters
    ----------
    classification : np.ndarray
        ASPRS classification codes of shape (N,)
    synthetic_flags : np.ndarray, optional
        Boolean array indicating synthetic points from DTM augmentation
        Shape (N,). If None, all ground-classified points are natural LiDAR.
    ground_class : int, optional
        ASPRS classification code for ground points (default: 2)
    include_synthetic : bool, optional
        Whether to include DTM-augmented synthetic ground points (default: True)

    Returns
    -------
    is_ground : np.ndarray
        Binary ground indicator of shape (N,)
        - 1: Ground point (natural LiDAR or DTM-augmented)
        - 0: Non-ground point

    Raises
    ------
    ValueError
        If input arrays have incompatible shapes

    Examples
    --------
    >>> # Basic usage with LiDAR classification
    >>> classification = np.array([2, 2, 6, 3, 2, 5])
    >>> is_ground = compute_is_ground(classification)
    >>> print(is_ground)
    [1 1 0 0 1 0]

    >>> # With DTM augmentation (synthetic flags)
    >>> synthetic_flags = np.array([False, False, False, False, True, False])
    >>> is_ground = compute_is_ground(classification, synthetic_flags)
    >>> print(is_ground)  # Point 4 is synthetic ground
    [1 1 0 0 1 0]

    >>> # Exclude synthetic points
    >>> is_ground = compute_is_ground(classification, synthetic_flags, include_synthetic=False)
    >>> print(is_ground)  # Point 4 excluded (synthetic)
    [1 1 0 0 0 0]

    Notes
    -----
    Ground Classification:
    - ASPRS class 2: Ground (standard)
    - DTM augmentation: Synthetic points added from RGE ALTI DTM
    - Validation: Synthetic points are validated before inclusion

    DTM Augmentation Workflow:
    1. LiDAR processing identifies ground points (class 2)
    2. DTM augmentation adds synthetic ground points in gaps
    3. Synthetic points marked with flag in point attributes
    4. This function combines both sources into single ground indicator

    Performance:
    - O(N) time complexity (simple boolean operation)
    - Minimal memory overhead (boolean array)
    """
    # Input validation
    if not isinstance(classification, np.ndarray) or classification.ndim != 1:
        raise ValueError(
            f"classification must be 1D array, got shape {classification.shape}"
        )

    if synthetic_flags is not None:
        if not isinstance(synthetic_flags, np.ndarray) or synthetic_flags.ndim != 1:
            raise ValueError(
                f"synthetic_flags must be 1D array, got shape {synthetic_flags.shape}"
            )
        if len(classification) != len(synthetic_flags):
            raise ValueError(
                f"classification and synthetic_flags must have same length: "
                f"{len(classification)} != {len(synthetic_flags)}"
            )

    # Compute base ground mask from classification
    is_ground = classification == ground_class

    # Handle synthetic points
    if synthetic_flags is not None and not include_synthetic:
        # Exclude synthetic ground points
        is_ground = is_ground & (~synthetic_flags)
        n_synthetic = np.sum(synthetic_flags & (classification == ground_class))
        if n_synthetic > 0:
            logger.debug(
                f"Excluded {n_synthetic:,} synthetic ground points "
                f"(include_synthetic=False)"
            )

    # Convert to int8 for efficient storage
    return is_ground.astype(np.int8)


def compute_is_ground_with_stats(
    classification: np.ndarray,
    synthetic_flags: Optional[np.ndarray] = None,
    ground_class: int = 2,
    include_synthetic: bool = True,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Compute is_ground feature with detailed statistics.

    Returns both the feature array and statistics about ground point distribution.
    Useful for quality assessment and reporting.

    Parameters
    ----------
    classification : np.ndarray
        ASPRS classification codes
    synthetic_flags : np.ndarray, optional
        Synthetic point flags from DTM augmentation
    ground_class : int, optional
        Ground classification code
    include_synthetic : bool, optional
        Include synthetic ground points
    verbose : bool, optional
        Log statistics (default: True)

    Returns
    -------
    is_ground : np.ndarray
        Binary ground indicator
    stats : dict
        Statistics dictionary with keys:
        - 'total_points': Total number of points
        - 'natural_ground': Natural LiDAR ground points
        - 'synthetic_ground': DTM-augmented ground points
        - 'total_ground': Total ground points (natural + synthetic)
        - 'non_ground': Non-ground points
        - 'ground_percentage': Percentage of points that are ground
        - 'synthetic_percentage': Percentage of ground that is synthetic

    Examples
    --------
    >>> classification = np.array([2, 2, 6, 3, 2, 5])
    >>> synthetic_flags = np.array([False, False, False, False, True, False])
    >>> is_ground, stats = compute_is_ground_with_stats(classification, synthetic_flags)
    >>> print(f"Ground coverage: {stats['ground_percentage']:.1f}%")
    Ground coverage: 50.0%
    >>> print(f"Synthetic contribution: {stats['synthetic_percentage']:.1f}%")
    Synthetic contribution: 33.3%
    """
    # Compute is_ground feature
    is_ground = compute_is_ground(
        classification=classification,
        synthetic_flags=synthetic_flags,
        ground_class=ground_class,
        include_synthetic=include_synthetic,
    )

    # Compute statistics
    total_points = len(classification)

    # Compute natural ground points (excluding synthetic if flags provided)
    if synthetic_flags is None:
        # No synthetic flags: all ground-classified points are natural
        natural_ground = np.sum(classification == ground_class)
    else:
        # Has synthetic flags: exclude synthetic points from natural count
        natural_ground = np.sum((classification == ground_class) & (~synthetic_flags))

    synthetic_ground = 0
    if synthetic_flags is not None and include_synthetic:
        synthetic_ground = np.sum((classification == ground_class) & synthetic_flags)

    total_ground = np.sum(is_ground)
    non_ground = total_points - total_ground

    ground_percentage = (total_ground / total_points * 100) if total_points > 0 else 0
    synthetic_percentage = (
        (synthetic_ground / total_ground * 100) if total_ground > 0 else 0
    )

    stats = {
        "total_points": total_points,
        "natural_ground": int(natural_ground),
        "synthetic_ground": int(synthetic_ground),
        "total_ground": int(total_ground),
        "non_ground": int(non_ground),
        "ground_percentage": float(ground_percentage),
        "synthetic_percentage": float(synthetic_percentage),
    }

    # Log statistics if verbose
    if verbose:
        logger.info("=== Ground Point Statistics ===")
        logger.info("  Total points: {stats['total_points']:,}")
        logger.info(f"  Natural ground: {stats['natural_ground']:,}")
        if stats["synthetic_ground"] > 0:
            logger.info(f"  Synthetic ground (DTM): {stats['synthetic_ground']:,}")
        logger.info(
            f"  Total ground: {stats['total_ground']:,} "
            f"({stats['ground_percentage']:.1f}%)"
        )
        logger.info(f"  Non-ground: {stats['non_ground']:,}")
        if stats["synthetic_ground"] > 0:
            logger.info(
                f"  DTM contribution: {stats['synthetic_percentage']:.1f}% "
                f"of ground"
            )

    return is_ground, stats


def compute_ground_density(
    points: np.ndarray, is_ground: np.ndarray, grid_size: float = 10.0
) -> tuple[np.ndarray, float]:
    """
    Compute spatial density of ground points on a grid.

    Useful for identifying gaps in ground coverage that may benefit from
    DTM augmentation.

    Parameters
    ----------
    points : np.ndarray
        Point cloud array of shape (N, 3) with XYZ coordinates
    is_ground : np.ndarray
        Binary ground indicator from compute_is_ground()
    grid_size : float, optional
        Grid cell size in meters (default: 10.0)

    Returns
    -------
    density_map : np.ndarray
        2D grid showing ground point density (points per m²)
    mean_density : float
        Mean ground point density across entire area

    Examples
    --------
    >>> points = np.random.rand(10000, 3) * 100
    >>> classification = np.random.choice([2, 6], 10000)
    >>> is_ground = compute_is_ground(classification)
    >>> density_map, mean_density = compute_ground_density(points, is_ground)
    >>> print(f"Average ground density: {mean_density:.1f} points/m²")
    """
    # Extract ground points
    ground_points = points[is_ground.astype(bool)]

    if len(ground_points) == 0:
        logger.warning("No ground points found for density computation")
        return np.array([]), 0.0

    # Compute bounding box
    min_xy = np.min(ground_points[:, :2], axis=0)
    max_xy = np.max(ground_points[:, :2], axis=0)

    # Create grid
    nx = int(np.ceil((max_xy[0] - min_xy[0]) / grid_size))
    ny = int(np.ceil((max_xy[1] - min_xy[1]) / grid_size))

    if nx == 0 or ny == 0:
        single_cell_density = len(ground_points) / (grid_size**2)
        return np.array([[len(ground_points)]]), single_cell_density

    # Compute grid indices
    grid_x = ((ground_points[:, 0] - min_xy[0]) / grid_size).astype(int)
    grid_y = ((ground_points[:, 1] - min_xy[1]) / grid_size).astype(int)

    # Clip to valid range
    grid_x = np.clip(grid_x, 0, nx - 1)
    grid_y = np.clip(grid_y, 0, ny - 1)

    # Count points per cell
    density_map = np.zeros((ny, nx), dtype=np.int32)
    for i in range(len(ground_points)):
        density_map[grid_y[i], grid_x[i]] += 1

    # Convert to density (points per m²)
    cell_area = grid_size**2
    density_map = density_map.astype(np.float32) / cell_area

    # Compute mean density
    total_area = nx * ny * cell_area
    mean_density = len(ground_points) / total_area

    return density_map, mean_density


def identify_ground_gaps(
    points: np.ndarray,
    is_ground: np.ndarray,
    grid_size: float = 10.0,
    min_density_threshold: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """
    Identify spatial gaps in ground coverage.

    Returns a mask of points in areas with insufficient ground coverage,
    which are candidates for DTM augmentation.

    Parameters
    ----------
    points : np.ndarray
        Point cloud array of shape (N, 3)
    is_ground : np.ndarray
        Binary ground indicator
    grid_size : float, optional
        Grid cell size for analysis (default: 10.0m)
    min_density_threshold : float, optional
        Minimum ground point density (points/m²) to consider adequate
        (default: 0.5)

    Returns
    -------
    gap_mask : np.ndarray
        Boolean mask indicating points in ground-sparse areas
    gap_stats : dict
        Statistics about identified gaps

    Examples
    --------
    >>> gap_mask, stats = identify_ground_gaps(points, is_ground)
    >>> print(f"Found {stats['n_gap_cells']} cells needing augmentation")
    >>> print(f"{stats['pct_gap']:.1f}% of area has sparse ground coverage")
    """
    # Compute density map
    density_map, mean_density = compute_ground_density(points, is_ground, grid_size)

    if density_map.size == 0:
        empty_gap_stats = {"n_gap_cells": 0, "pct_gap": 0.0}
        return np.zeros(len(points), dtype=bool), empty_gap_stats

    # Identify gap cells
    gap_cells = density_map < min_density_threshold
    n_gap_cells = np.sum(gap_cells)
    pct_gap = (n_gap_cells / gap_cells.size * 100) if gap_cells.size > 0 else 0

    # Map points to grid cells
    min_xy = np.min(points[:, :2], axis=0)
    grid_x = ((points[:, 0] - min_xy[0]) / grid_size).astype(int)
    grid_y = ((points[:, 1] - min_xy[1]) / grid_size).astype(int)

    # Clip to valid range
    ny, nx = density_map.shape
    grid_x = np.clip(grid_x, 0, nx - 1)
    grid_y = np.clip(grid_y, 0, ny - 1)

    # Create mask for points in gap cells
    gap_mask = gap_cells[grid_y, grid_x]

    gap_stats = {
        "n_gap_cells": int(n_gap_cells),
        "pct_gap": float(pct_gap),
        "mean_density": float(mean_density),
        "min_density_threshold": float(min_density_threshold),
        "n_points_in_gaps": int(np.sum(gap_mask)),
    }

    logger.info(
        f"  Ground gap analysis: {n_gap_cells}/{gap_cells.size} cells "
        f"({pct_gap:.1f}%) below density threshold"
    )

    return gap_mask, gap_stats


__all__ = [
    "compute_is_ground",
    "compute_is_ground_with_stats",
    "compute_ground_density",
    "identify_ground_gaps",
]
