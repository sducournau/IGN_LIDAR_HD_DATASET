"""
Point Cloud Preprocessing Module

This module provides robust preprocessing functions to reduce artifacts
in geometric feature computation for IGN LiDAR HD data.

Preprocessing techniques implemented:
- Statistical Outlier Removal (SOR): Remove points based on distance statistics
- Radius Outlier Removal (ROR): Remove isolated points
- Voxel Downsampling: Homogenize point density

These techniques help reduce artifacts such as:
- Scan line patterns (dashes) in planarity/curvature maps
- Noisy normals from outlier contamination
- Degenerate features from sparse neighborhoods
"""

from typing import Dict, Tuple, Any, Optional
import numpy as np
import logging
import multiprocessing

logger = logging.getLogger(__name__)


def _get_safe_n_jobs() -> int:
    """
    Get safe n_jobs value for sklearn in multiprocessing context.
    
    sklearn's parallel loops can't be nested in multiprocessing workers,
    so we return 1 when inside a worker process.
    
    Returns:
        1 if in multiprocessing context, -1 otherwise
    """
    try:
        # Check if we're in a multiprocessing worker
        current_process = multiprocessing.current_process()
        if hasattr(current_process, '_identity') and current_process._identity:
            return 1  # We're in a worker process
        return -1  # Main process, use all cores
    except:
        return 1  # Safe fallback


def statistical_outlier_removal(
    points: np.ndarray,
    k: int = 12,
    std_multiplier: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove statistical outliers based on mean distance to k-nearest neighbors.
    
    For each point, computes the mean distance to its k nearest neighbors.
    Points whose mean distance exceeds (global_mean + std_multiplier * global_std)
    are considered outliers and removed.
    
    Args:
        points: [N, 3] point coordinates
        k: number of neighbors for statistics (default 12)
        std_multiplier: threshold in standard deviations (default 2.0)
                       Lower values = stricter filtering
        
    Returns:
        filtered_points: [M, 3] cleaned points (M <= N)
        inlier_mask: [N] boolean mask of kept points (True = kept)
        
    Example:
        >>> points_clean, mask = statistical_outlier_removal(points, k=12, std_multiplier=2.0)
        >>> print(f"Removed {np.sum(~mask)} outliers ({np.sum(~mask)/len(points)*100:.1f}%)")
    """
    from sklearn.neighbors import NearestNeighbors
    
    N = len(points)
    if N < k + 1:
        logger.warning(f"Too few points ({N}) for SOR with k={k}, returning all points")
        return points, np.ones(N, dtype=bool)
    
    # Build kNN tree
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree', n_jobs=_get_safe_n_jobs())
    nbrs.fit(points)
    distances, _ = nbrs.kneighbors(points)
    
    # Compute mean distance to neighbors (excluding self at index 0)
    mean_distances = distances[:, 1:].mean(axis=1)
    
    # Compute global statistics
    global_mean = mean_distances.mean()
    global_std = mean_distances.std()
    
    # Threshold: points with mean distance > threshold are outliers
    threshold = global_mean + std_multiplier * global_std
    inlier_mask = mean_distances < threshold
    
    filtered_points = points[inlier_mask]
    
    removed_count = np.sum(~inlier_mask)
    removed_pct = removed_count / N * 100
    
    logger.debug(f"SOR: Removed {removed_count:,} outliers ({removed_pct:.2f}%)")
    logger.debug(f"  Mean dist: {global_mean:.4f} ± {global_std:.4f}, threshold: {threshold:.4f}")
    
    return filtered_points, inlier_mask


def radius_outlier_removal(
    points: np.ndarray,
    radius: float = 1.0,
    min_neighbors: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove points with too few neighbors within a given radius.
    
    Effective for removing isolated points that survived SOR or are artifacts
    from object edges, scan boundaries, or measurement errors.
    
    Args:
        points: [N, 3] point coordinates
        radius: search radius in meters (default 1.0m)
        min_neighbors: minimum required neighbors (excluding self) (default 4)
        
    Returns:
        filtered_points: [M, 3] cleaned points (M <= N)
        inlier_mask: [N] boolean mask of kept points
        
    Example:
        >>> # Remove points with fewer than 4 neighbors within 1.0m
        >>> points_clean, mask = radius_outlier_removal(points, radius=1.0, min_neighbors=4)
    """
    from sklearn.neighbors import NearestNeighbors
    
    N = len(points)
    if N == 0:
        return points, np.ones(N, dtype=bool)
    
    # Build ball tree for radius queries
    nbrs = NearestNeighbors(radius=radius, algorithm='kd_tree', n_jobs=_get_safe_n_jobs())
    nbrs.fit(points)
    
    # Count neighbors for each point (including self)
    neighbors = nbrs.radius_neighbors(points, return_distance=False)
    neighbor_counts = np.array([len(n) for n in neighbors])
    
    # Keep points with enough neighbors (excluding self, so > min_neighbors)
    inlier_mask = neighbor_counts > min_neighbors
    filtered_points = points[inlier_mask]
    
    removed_count = np.sum(~inlier_mask)
    removed_pct = removed_count / N * 100
    
    logger.debug(f"ROR: Removed {removed_count:,} isolated points ({removed_pct:.2f}%)")
    logger.debug(f"  Radius: {radius:.2f}m, min_neighbors: {min_neighbors}")
    
    return filtered_points, inlier_mask


def voxel_downsample(
    points: np.ndarray,
    voxel_size: float = 0.5,
    method: str = 'centroid'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample point cloud using voxel grid to homogenize density.
    
    Divides space into voxels of given size and replaces all points in each
    voxel with either their centroid or a random point.
    
    Benefits:
    - Homogenizes density (reduces scan line artifacts)
    - Speeds up feature computation
    - Reduces memory usage
    
    Args:
        points: [N, 3] point coordinates
        voxel_size: size of voxel in meters (default 0.5m)
        method: 'centroid' (average) or 'random' (random point from voxel)
        
    Returns:
        downsampled_points: [M, 3] voxelized points (M <= N)
        voxel_indices: [N] voxel index for each original point
        
    Example:
        >>> # Downsample to 0.5m voxels
        >>> points_ds, voxel_idx = voxel_downsample(points, voxel_size=0.5)
        >>> print(f"Reduced from {len(points)} to {len(points_ds)} points")
    """
    N = len(points)
    if N == 0:
        return points, np.array([], dtype=np.int32)
    
    # Compute voxel indices for each point
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    
    # Convert 3D indices to unique keys
    # Use large multipliers to avoid collisions
    voxel_keys = (
        voxel_indices[:, 0].astype(np.int64) * 1_000_000_000 +
        voxel_indices[:, 1].astype(np.int64) * 1_000_000 +
        voxel_indices[:, 2].astype(np.int64)
    )
    
    unique_voxels, inverse_indices = np.unique(
        voxel_keys, return_inverse=True
    )
    
    n_voxels = len(unique_voxels)
    
    if method == 'centroid':
        # Average points in each voxel
        downsampled = np.zeros((n_voxels, 3), dtype=np.float32)
        for i in range(n_voxels):
            mask = inverse_indices == i
            downsampled[i] = points[mask].mean(axis=0)
    elif method == 'random':
        # Random point from each voxel
        downsampled = []
        for i in range(n_voxels):
            mask = inverse_indices == i
            idx = np.random.choice(np.where(mask)[0])
            downsampled.append(points[idx])
        downsampled = np.array(downsampled, dtype=np.float32)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'centroid' or 'random'")
    
    reduction_pct = (1 - n_voxels / N) * 100
    
    logger.debug(f"Voxel: Reduced from {N:,} to {n_voxels:,} points ({reduction_pct:.1f}% reduction)")
    logger.debug(f"  Voxel size: {voxel_size:.3f}m, method: {method}")
    
    return downsampled, inverse_indices


def preprocess_point_cloud(
    points: np.ndarray,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply full preprocessing pipeline to point cloud.
    
    Applies a sequence of filters to clean point cloud before feature computation:
    1. Statistical Outlier Removal (SOR) - remove measurement noise
    2. Radius Outlier Removal (ROR) - remove isolated points
    3. Voxel Downsampling (optional) - homogenize density
    
    Default configuration:
    {
        'sor': {'enable': True, 'k': 12, 'std_multiplier': 2.0},
        'ror': {'enable': True, 'radius': 1.0, 'min_neighbors': 4},
        'voxel': {'enable': False, 'voxel_size': 0.5, 'method': 'centroid'}
    }
    
    Args:
        points: [N, 3] raw point coordinates
        config: optional preprocessing parameters (see above for defaults)
        
    Returns:
        processed_points: [M, 3] cleaned points (M <= N)
        stats: preprocessing statistics dictionary with keys:
               - original_points: int
               - sor_removed: int (if SOR enabled)
               - ror_removed: int (if ROR enabled)
               - voxel_reduced: int (if voxel enabled)
               - final_points: int
               - reduction_ratio: float (0-1)
               - processing_time_ms: float
               
    Example:
        >>> # Use default preprocessing
        >>> points_clean, stats = preprocess_point_cloud(points)
        >>> print(f"Reduced from {stats['original_points']:,} to {stats['final_points']:,}")
        >>> print(f"Quality improvement: {stats['reduction_ratio']*100:.1f}% points removed")
        
        >>> # Custom configuration
        >>> config = {
        ...     'sor': {'enable': True, 'k': 8, 'std_multiplier': 1.5},
        ...     'ror': {'enable': True, 'radius': 1.5, 'min_neighbors': 6},
        ...     'voxel': {'enable': True, 'voxel_size': 0.3, 'method': 'centroid'}
        ... }
        >>> points_clean, stats = preprocess_point_cloud(points, config)
    """
    import time
    start_time = time.time()
    
    # Default configuration
    if config is None:
        config = {
            'sor': {'enable': True, 'k': 12, 'std_multiplier': 2.0},
            'ror': {'enable': True, 'radius': 1.0, 'min_neighbors': 4},
            'voxel': {'enable': False, 'voxel_size': 0.5, 'method': 'centroid'}
        }
    
    stats = {'original_points': len(points)}
    processed = points.copy()
    
    logger.info(f"Preprocessing point cloud: {len(points):,} points")
    
    # Step 1: Statistical Outlier Removal
    if config.get('sor', {}).get('enable', False):
        sor_config = config['sor']
        processed, mask = statistical_outlier_removal(
            processed,
            k=sor_config.get('k', 12),
            std_multiplier=sor_config.get('std_multiplier', 2.0)
        )
        stats['sor_removed'] = np.sum(~mask)
        logger.info(f"  SOR: {stats['sor_removed']:,} outliers removed")
    
    # Step 2: Radius Outlier Removal
    if config.get('ror', {}).get('enable', False):
        ror_config = config['ror']
        processed, mask = radius_outlier_removal(
            processed,
            radius=ror_config.get('radius', 1.0),
            min_neighbors=ror_config.get('min_neighbors', 4)
        )
        stats['ror_removed'] = np.sum(~mask)
        logger.info(f"  ROR: {stats['ror_removed']:,} isolated points removed")
    
    # Step 3: Voxel Downsampling (optional)
    if config.get('voxel', {}).get('enable', False):
        voxel_config = config['voxel']
        original_size = len(processed)
        processed, _ = voxel_downsample(
            processed,
            voxel_size=voxel_config.get('voxel_size', 0.5),
            method=voxel_config.get('method', 'centroid')
        )
        stats['voxel_reduced'] = original_size - len(processed)
        logger.info(f"  Voxel: {stats['voxel_reduced']:,} points downsampled")
    
    stats['final_points'] = len(processed)
    stats['reduction_ratio'] = 1.0 - (stats['final_points'] / stats['original_points'])
    stats['processing_time_ms'] = (time.time() - start_time) * 1000
    
    logger.info(f"  Final: {stats['final_points']:,} points ({stats['reduction_ratio']*100:.1f}% reduction)")
    logger.info(f"  Time: {stats['processing_time_ms']:.1f}ms")
    
    return processed, stats


def estimate_preprocessing_impact(
    points: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
    sample_size: int = 10000
) -> Dict[str, float]:
    """
    Estimate preprocessing impact without processing full cloud.
    
    Useful for large clouds to preview preprocessing effects.
    
    Args:
        points: [N, 3] point coordinates
        config: preprocessing configuration
        sample_size: number of points to sample for estimation
        
    Returns:
        estimates: dictionary with estimated metrics:
                  - estimated_sor_removal_pct: float
                  - estimated_ror_removal_pct: float
                  - estimated_total_removal_pct: float
    """
    if len(points) <= sample_size:
        # Process full cloud if small enough
        _, stats = preprocess_point_cloud(points, config)
        return {
            'estimated_sor_removal_pct': stats.get('sor_removed', 0) / stats['original_points'] * 100,
            'estimated_ror_removal_pct': stats.get('ror_removed', 0) / stats['original_points'] * 100,
            'estimated_total_removal_pct': stats['reduction_ratio'] * 100
        }
    
    # Sample points
    sample_indices = np.random.choice(len(points), sample_size, replace=False)
    sample_points = points[sample_indices]
    
    # Process sample
    _, stats = preprocess_point_cloud(sample_points, config)
    
    return {
        'estimated_sor_removal_pct': stats.get('sor_removed', 0) / stats['original_points'] * 100,
        'estimated_ror_removal_pct': stats.get('ror_removed', 0) / stats['original_points'] * 100,
        'estimated_total_removal_pct': stats['reduction_ratio'] * 100
    }


# Convenience functions for common use cases

def preprocess_for_features(points: np.ndarray, mode: str = 'standard') -> np.ndarray:
    """
    Preprocess point cloud with predefined settings for feature computation.
    
    Modes:
    - 'light': Minimal preprocessing (SOR only, k=8, std=2.5)
    - 'standard': Balanced preprocessing (SOR + ROR, default params)
    - 'aggressive': Strong preprocessing (stricter SOR, ROR, optional voxel)
    
    Args:
        points: [N, 3] point coordinates
        mode: preprocessing mode ('light', 'standard', 'aggressive')
        
    Returns:
        processed_points: [M, 3] cleaned points
    """
    if mode == 'light':
        config = {
            'sor': {'enable': True, 'k': 8, 'std_multiplier': 2.5},
            'ror': {'enable': False},
            'voxel': {'enable': False}
        }
    elif mode == 'standard':
        config = {
            'sor': {'enable': True, 'k': 12, 'std_multiplier': 2.0},
            'ror': {'enable': True, 'radius': 1.0, 'min_neighbors': 4},
            'voxel': {'enable': False}
        }
    elif mode == 'aggressive':
        config = {
            'sor': {'enable': True, 'k': 12, 'std_multiplier': 1.5},
            'ror': {'enable': True, 'radius': 1.5, 'min_neighbors': 6},
            'voxel': {'enable': True, 'voxel_size': 0.3, 'method': 'centroid'}
        }
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'light', 'standard', or 'aggressive'")
    
    processed, _ = preprocess_point_cloud(points, config)
    return processed


def preprocess_for_urban(points: np.ndarray) -> np.ndarray:
    """
    Preprocess point cloud optimized for urban/building environments.
    
    Uses stricter filtering to handle complex geometries and high point density.
    """
    config = {
        'sor': {'enable': True, 'k': 15, 'std_multiplier': 1.8},
        'ror': {'enable': True, 'radius': 0.8, 'min_neighbors': 5},
        'voxel': {'enable': False}  # Preserve building details
    }
    processed, _ = preprocess_point_cloud(points, config)
    return processed


def preprocess_for_natural(points: np.ndarray) -> np.ndarray:
    """
    Preprocess point cloud optimized for natural/vegetation environments.
    
    Uses gentler filtering to preserve organic structures.
    """
    config = {
        'sor': {'enable': True, 'k': 10, 'std_multiplier': 2.5},
        'ror': {'enable': True, 'radius': 1.5, 'min_neighbors': 3},
        'voxel': {'enable': False}
    }
    processed, _ = preprocess_point_cloud(points, config)
    return processed
