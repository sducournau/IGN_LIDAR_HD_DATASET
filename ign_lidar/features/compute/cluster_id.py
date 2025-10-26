"""
Cluster ID computation for point clouds.

This module provides functions to compute cluster IDs:
- General spatial clustering (DBSCAN)
- Building-specific cluster IDs (based on BD TOPO® building footprints)
- Parcel-specific cluster IDs (based on cadastral parcels)

These cluster IDs are used for:
1. Object segmentation (grouping points belonging to same object)
2. Training labels for instance segmentation models
3. Feature engineering (object-level features)
"""

import logging
import multiprocessing
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def _get_safe_n_jobs() -> int:
    """
    Get safe n_jobs parameter for sklearn that avoids conflicts with multiprocessing.

    When running inside a multiprocessing worker, sklearn's own parallelism
    (using loky/joblib) conflicts and causes warnings. In this case, return 1
    to disable sklearn parallelism and let the outer multiprocessing handle it.

    Returns:
        1 if in multiprocessing context, -1 otherwise (use all cores)
    """
    try:
        # Check if we're in a multiprocessing worker
        current_process = multiprocessing.current_process()
        # If process name contains 'ForkPoolWorker' or similar, we're in a worker
        if current_process.name != "MainProcess":
            return 1  # Disable sklearn parallelism in workers
    except Exception:
        pass

    return -1  # Use all cores if not in worker


logger = logging.getLogger(__name__)

try:
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False
    logger.warning("Spatial libraries not available for cluster ID features")

try:
    from sklearn.cluster import DBSCAN

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not available for spatial clustering")


def compute_spatial_cluster_ids(
    points: np.ndarray, eps: float = 0.5, min_samples: int = 10, use_z: bool = True
) -> np.ndarray:
    """
    Assign unique cluster IDs to points using spatial proximity clustering (DBSCAN).

    This implements general spatial clustering independent of ground truth geometries.
    Points that are spatially close together are assigned the same cluster ID.

    Algorithm:
    1. Apply DBSCAN clustering to find spatially connected components
    2. Assign cluster IDs (1-based for clusters, 0 for noise)
    3. Convert to consecutive IDs starting from 1

    Parameters
    ----------
    points : np.ndarray
        Point cloud (N, 3) with XYZ coordinates
    eps : float, optional
        Maximum distance between two points to be in the same cluster (meters).
        Default: 0.5m
    min_samples : int, optional
        Minimum number of points to form a cluster.
        Default: 10 points
    use_z : bool, optional
        Whether to use Z coordinate in clustering (3D) or just XY (2D).
        Default: True (3D clustering)

    Returns
    -------
    cluster_ids : np.ndarray
        Array of shape (N,) with cluster IDs:
        - 0: Noise points (not in any cluster)
        - 1, 2, 3, ...: Cluster IDs for each connected component

    Notes
    -----
    - DBSCAN is used because it can find clusters of arbitrary shape
    - Points labeled as noise (-1 by DBSCAN) are assigned ID 0
    - All other points get consecutive cluster IDs starting from 1
    - Useful for instance segmentation without ground truth

    Examples
    --------
    >>> points = np.random.rand(1000, 3) * 100
    >>> cluster_ids = compute_spatial_cluster_ids(points, eps=1.0, min_samples=5)
    >>> n_clusters = len(np.unique(cluster_ids)) - 1  # Subtract 1 for noise
    """
    if not HAS_SKLEARN:
        logger.warning(
            "  ⚠️  scikit-learn not available for spatial clustering. "
            "Returning zeros."
        )
        return np.zeros(len(points), dtype=np.int32)

    if len(points) == 0:
        return np.array([], dtype=np.int32)

    # Select coordinates for clustering
    if use_z:
        coords = points[:, :3]  # XYZ
    else:
        coords = points[:, :2]  # XY only (2D clustering)

    # Apply DBSCAN clustering
    logger.info(
        f"  🔍 Running DBSCAN clustering (eps={eps}m, min_samples={min_samples}, "
        f"{'3D' if use_z else '2D'})..."
    )

    try:
        from sklearn.cluster import DBSCAN

        # ✅ FIXED: Use safe n_jobs to avoid conflicts with multiprocessing
        n_jobs = _get_safe_n_jobs()
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs)
        labels = clusterer.fit_predict(coords)

        # Convert DBSCAN labels to our convention:
        # DBSCAN: -1 = noise, 0, 1, 2, ... = cluster IDs
        # Ours:    0 = noise, 1, 2, 3, ... = cluster IDs
        cluster_ids = labels + 1  # Shift by 1: -1 -> 0, 0 -> 1, 1 -> 2, ...
        cluster_ids = cluster_ids.astype(np.int32)

        # Log statistics
        n_noise = np.sum(cluster_ids == 0)
        n_clusters = len(np.unique(cluster_ids)) - (1 if n_noise > 0 else 0)
        n_clustered = len(points) - n_noise

        logger.info(
            f"  ✓ Spatial clustering: {n_clusters} clusters, "
            f"{n_clustered:,} points assigned, {n_noise:,} noise points"
        )

        return cluster_ids

    except Exception as e:
        logger.error(f"  ✗ Failed to compute spatial cluster IDs: {e}")
        return np.zeros(len(points), dtype=np.int32)


def compute_building_cluster_ids(
    points: np.ndarray, building_geometries: Optional[gpd.GeoDataFrame] = None
) -> np.ndarray:
    """
    Assign unique cluster IDs to points based on building polygons.

    Each point inside a building polygon gets the building's cluster ID.
    Points outside all buildings get cluster ID 0 (background).

    Algorithm:
    1. Build spatial index (STRtree) for efficient polygon queries
    2. For each point, find containing building polygon
    3. Assign cluster ID based on building index (1-based)

    Parameters
    ----------
    points : np.ndarray
        Point cloud XYZ coordinates, shape (N, 3)
    building_geometries : gpd.GeoDataFrame, optional
        Building polygons from BD TOPO or other source

    Returns
    -------
    cluster_ids : np.ndarray
        Building cluster IDs, shape (N,), dtype=int32
        - 0: Background (not in any building)
        - 1 to M: Building cluster ID (1-indexed)

    Notes
    -----
    - Cluster IDs are assigned sequentially based on building index
    - If a point is in multiple buildings (overlapping polygons),
      the first matching building is used
    - Computation time: O(N log M) where N=points, M=buildings

    Example
    -------
    >>> points = np.array([[x1, y1, z1], [x2, y2, z2], ...])
    >>> buildings = gpd.read_file("buildings.shp")
    >>> cluster_ids = compute_building_cluster_ids(points, buildings)
    >>> print(f"Found {np.max(cluster_ids)} buildings")
    >>> print(f"{np.sum(cluster_ids == 0)} points in background")
    """
    if not HAS_SPATIAL:
        logger.warning("Spatial libraries not available - returning zeros")
        return np.zeros(len(points), dtype=np.int32)

    if building_geometries is None or len(building_geometries) == 0:
        logger.debug("No building geometries provided - returning zeros")
        return np.zeros(len(points), dtype=np.int32)

    n_points = len(points)
    cluster_ids = np.zeros(n_points, dtype=np.int32)

    logger.info(
        f"Computing building cluster IDs for {n_points:,} points and {len(building_geometries)} buildings..."
    )

    # Build spatial index for buildings
    building_list = building_geometries.geometry.tolist()
    tree = STRtree(building_list)

    # Process points in batches for better performance
    batch_size = 10000
    n_batches = (n_points + batch_size - 1) // batch_size

    points_assigned = 0

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_points)
        batch_points = points[start_idx:end_idx]

        # Query each point
        for local_idx, pt in enumerate(batch_points):
            global_idx = start_idx + local_idx
            pt_geom = Point(pt[0], pt[1])  # XY only

            # Find candidate buildings (bounding box intersection)
            candidate_indices = tree.query(pt_geom)

            # Check actual containment
            for building_idx in candidate_indices:
                if building_list[building_idx].contains(pt_geom):
                    # Assign cluster ID (1-indexed)
                    cluster_ids[global_idx] = building_idx + 1
                    points_assigned += 1
                    break  # Use first matching building

        # Progress logging
        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            progress = (end_idx / n_points) * 100
            logger.debug(
                f"  Progress: {progress:.1f}% ({end_idx:,}/{n_points:,} points)"
            )

    n_background = np.sum(cluster_ids == 0)
    n_buildings = np.max(cluster_ids)

    logger.info(f"  ✓ Assigned {points_assigned:,} points to {n_buildings} buildings")
    logger.info(
        f"  ✓ Background points: {n_background:,} ({n_background/n_points*100:.1f}%)"
    )

    return cluster_ids


def compute_parcel_cluster_ids(
    points: np.ndarray, parcel_geometries: Optional[gpd.GeoDataFrame] = None
) -> np.ndarray:
    """
    Assign unique cluster IDs to points based on cadastral parcels.

    Each point inside a parcel polygon gets the parcel's cluster ID.
    Points outside all parcels get cluster ID 0 (background).

    Algorithm:
    1. Build spatial index (STRtree) for efficient polygon queries
    2. For each point, find containing parcel polygon
    3. Assign cluster ID based on parcel index (1-based)

    Parameters
    ----------
    points : np.ndarray
        Point cloud XYZ coordinates, shape (N, 3)
    parcel_geometries : gpd.GeoDataFrame, optional
        Cadastral parcel polygons from cadastre or other source

    Returns
    -------
    cluster_ids : np.ndarray
        Parcel cluster IDs, shape (N,), dtype=int32
        - 0: Background (not in any parcel)
        - 1 to M: Parcel cluster ID (1-indexed)

    Notes
    -----
    - Cluster IDs are assigned sequentially based on parcel index
    - If a point is in multiple parcels (overlapping polygons),
      the first matching parcel is used
    - Computation time: O(N log M) where N=points, M=parcels
    - Parcels typically cover larger areas than buildings

    Example
    -------
    >>> points = np.array([[x1, y1, z1], [x2, y2, z2], ...])
    >>> parcels = gpd.read_file("parcelles.shp")
    >>> cluster_ids = compute_parcel_cluster_ids(points, parcels)
    >>> print(f"Found {np.max(cluster_ids)} parcels")
    >>> print(f"Mean points per parcel: {len(points) / np.max(cluster_ids):.0f}")
    """
    if not HAS_SPATIAL:
        logger.warning("Spatial libraries not available - returning zeros")
        return np.zeros(len(points), dtype=np.int32)

    if parcel_geometries is None or len(parcel_geometries) == 0:
        logger.debug("No parcel geometries provided - returning zeros")
        return np.zeros(len(points), dtype=np.int32)

    n_points = len(points)
    cluster_ids = np.zeros(n_points, dtype=np.int32)

    logger.info(
        f"Computing parcel cluster IDs for {n_points:,} points and {len(parcel_geometries)} parcels..."
    )

    # Build spatial index for parcels
    parcel_list = parcel_geometries.geometry.tolist()
    tree = STRtree(parcel_list)

    # Process points in batches for better performance
    batch_size = 10000
    n_batches = (n_points + batch_size - 1) // batch_size

    points_assigned = 0

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_points)
        batch_points = points[start_idx:end_idx]

        # Query each point
        for local_idx, pt in enumerate(batch_points):
            global_idx = start_idx + local_idx
            pt_geom = Point(pt[0], pt[1])  # XY only

            # Find candidate parcels (bounding box intersection)
            candidate_indices = tree.query(pt_geom)

            # Check actual containment
            for parcel_idx in candidate_indices:
                if parcel_list[parcel_idx].contains(pt_geom):
                    # Assign cluster ID (1-indexed)
                    cluster_ids[global_idx] = parcel_idx + 1
                    points_assigned += 1
                    break  # Use first matching parcel

        # Progress logging
        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            progress = (end_idx / n_points) * 100
            logger.debug(
                f"  Progress: {progress:.1f}% ({end_idx:,}/{n_points:,} points)"
            )

    n_background = np.sum(cluster_ids == 0)
    n_parcels = np.max(cluster_ids)

    logger.info(f"  ✓ Assigned {points_assigned:,} points to {n_parcels} parcels")
    logger.info(
        f"  ✓ Background points: {n_background:,} ({n_background/n_points*100:.1f}%)"
    )

    if n_parcels > 0:
        avg_points_per_parcel = points_assigned / n_parcels
        logger.info(f"  ✓ Average points per parcel: {avg_points_per_parcel:.0f}")

    return cluster_ids


def compute_cluster_statistics(
    cluster_ids: np.ndarray, features: Optional[Dict[str, np.ndarray]] = None
) -> Dict[int, Dict[str, float]]:
    """
    Compute statistics per cluster (building or parcel).

    For each cluster, computes:
    - n_points: Number of points
    - mean_height: Mean height (if available)
    - std_height: Height standard deviation
    - mean_ndvi: Mean NDVI (if available)
    - coverage_3d: 3D bounding box volume

    Parameters
    ----------
    cluster_ids : np.ndarray
        Cluster IDs, shape (N,)
    features : dict, optional
        Dictionary of feature arrays (height, ndvi, etc.)

    Returns
    -------
    stats : dict
        Dictionary mapping cluster_id -> statistics dict

    Example
    -------
    >>> cluster_ids = compute_building_cluster_ids(points, buildings)
    >>> features = {'height': height_array, 'ndvi': ndvi_array}
    >>> stats = compute_cluster_statistics(cluster_ids, features)
    >>> print(f"Building 5 has {stats[5]['n_points']} points")
    """
    unique_ids = np.unique(cluster_ids)
    unique_ids = unique_ids[unique_ids > 0]  # Exclude background (0)

    stats = {}

    for cluster_id in unique_ids:
        mask = cluster_ids == cluster_id
        n_points = np.sum(mask)

        cluster_stats = {"n_points": int(n_points)}

        # Compute feature statistics if available
        if features is not None:
            if "height" in features:
                heights = features["height"][mask]
                cluster_stats["mean_height"] = float(np.mean(heights))
                cluster_stats["std_height"] = float(np.std(heights))
                cluster_stats["min_height"] = float(np.min(heights))
                cluster_stats["max_height"] = float(np.max(heights))

            if "ndvi" in features:
                ndvi_vals = features["ndvi"][mask]
                cluster_stats["mean_ndvi"] = float(np.mean(ndvi_vals))
                cluster_stats["std_ndvi"] = float(np.std(ndvi_vals))

        stats[int(cluster_id)] = cluster_stats

    return stats


def validate_cluster_ids(cluster_ids: np.ndarray) -> Dict[str, any]:
    """
    Validate cluster ID array and return diagnostic information.

    Parameters
    ----------
    cluster_ids : np.ndarray
        Cluster IDs to validate

    Returns
    -------
    validation : dict
        Validation results with keys:
        - is_valid: bool
        - n_clusters: int (excluding background)
        - n_background: int
        - min_id: int
        - max_id: int
        - cluster_sizes: dict (id -> count)
    """
    unique_ids, counts = np.unique(cluster_ids, return_counts=True)

    n_background = counts[unique_ids == 0][0] if 0 in unique_ids else 0
    cluster_ids_nonzero = unique_ids[unique_ids > 0]

    cluster_sizes = {
        int(cid): int(count) for cid, count in zip(unique_ids, counts) if cid > 0
    }

    validation = {
        "is_valid": True,
        "n_clusters": len(cluster_ids_nonzero),
        "n_background": int(n_background),
        "min_id": (
            int(np.min(cluster_ids_nonzero)) if len(cluster_ids_nonzero) > 0 else 0
        ),
        "max_id": (
            int(np.max(cluster_ids_nonzero)) if len(cluster_ids_nonzero) > 0 else 0
        ),
        "cluster_sizes": cluster_sizes,
    }

    # Check for issues
    if len(cluster_ids_nonzero) == 0:
        validation["is_valid"] = False
        validation["error"] = "No clusters found (all points are background)"

    return validation
