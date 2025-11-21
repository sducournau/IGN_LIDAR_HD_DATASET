"""
Building Module Utilities - Shared Helper Functions

This module contains shared utility functions used across building
classification modules to reduce code duplication.

Functions include:
- Spatial operations (point-in-polygon, buffering)
- Height filtering
- Geometric calculations (centroids, areas)
- Feature computations (planarity, verticality)

Author: Phase 2 - Building Module Restructuring
Date: October 22, 2025
"""

import logging
from typing import Optional, List, Tuple, TYPE_CHECKING
import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from shapely.geometry import Polygon, MultiPolygon, Point
    import geopandas as gpd

try:
    from shapely.geometry import Polygon, MultiPolygon, Point, box
    from shapely.strtree import STRtree
    from shapely.ops import unary_union
    from shapely.affinity import translate, scale, rotate
    import geopandas as gpd
    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False
    STRtree = None
    logger.warning("Shapely not available. Spatial operations will be disabled.")


# ============================================================================
# Spatial Utilities
# ============================================================================

def check_spatial_dependencies() -> bool:
    """
    Check if spatial dependencies are available.
    
    Returns:
        True if shapely and geopandas are available, False otherwise
    """
    return HAS_SPATIAL


def create_spatial_index(polygons: List['Polygon']) -> Optional['STRtree']:
    """
    Create spatial index (STRtree) from list of polygons.
    
    Args:
        polygons: List of shapely Polygon objects
        
    Returns:
        STRtree spatial index, or None if spatial dependencies unavailable
    """
    if not HAS_SPATIAL or not polygons:
        return None
    
    try:
        return STRtree(polygons)
    except Exception as e:
        logger.warning(f"Failed to create spatial index: {e}")
        return None


def points_in_polygon(
    points: np.ndarray,
    polygon: 'Polygon',
    return_mask: bool = True
) -> np.ndarray:
    """
    Find points within a polygon.
    
    Args:
        points: Array of shape (N, 2) or (N, 3) with XY(Z) coordinates
        polygon: Shapely Polygon
        return_mask: If True, return boolean mask; if False, return indices
        
    Returns:
        Boolean mask (N,) or integer indices of points inside polygon
    """
    if not HAS_SPATIAL:
        raise ImportError("Shapely required for point-in-polygon tests")
    
    # Extract XY coordinates
    xy = points[:, :2] if points.shape[1] >= 2 else points
    
    # Vectorized point-in-polygon test
    from shapely.vectorized import contains
    mask = contains(polygon, xy[:, 0], xy[:, 1])
    
    if return_mask:
        return mask
    else:
        return np.where(mask)[0]


def buffer_polygon(
    polygon: 'Polygon',
    distance: float,
    resolution: int = 16
) -> 'Polygon':
    """
    Create buffer around polygon.
    
    Args:
        polygon: Input polygon
        distance: Buffer distance (meters, negative for erosion)
        resolution: Number of segments per quadrant
        
    Returns:
        Buffered polygon
    """
    if not HAS_SPATIAL:
        raise ImportError("Shapely required for buffering")
    
    return polygon.buffer(distance, resolution=resolution)


def compute_polygon_centroid(polygon: 'Polygon') -> np.ndarray:
    """
    Compute centroid of polygon.
    
    Args:
        polygon: Input polygon
        
    Returns:
        Array [x, y] with centroid coordinates
    """
    if not HAS_SPATIAL:
        raise ImportError("Shapely required for centroid computation")
    
    centroid = polygon.centroid
    return np.array([centroid.x, centroid.y])


def compute_convex_hull_polygon(points_2d: np.ndarray) -> Optional['Polygon']:
    """
    Compute 2D convex hull of points and return as Shapely polygon.
    
    Args:
        points_2d: Point coordinates [N, 2] (XY)
        
    Returns:
        Shapely Polygon representing convex hull, or None if computation fails
    """
    if not HAS_SPATIAL:
        raise ImportError("Shapely required for convex hull")
    
    if len(points_2d) < 3:
        logger.warning(f"Need at least 3 points for convex hull, got {len(points_2d)}")
        return None
    
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points_2d)
        hull_points = points_2d[hull.vertices]
        return Polygon(hull_points)
    except Exception as e:
        logger.warning(f"Failed to compute convex hull: {e}")
        return None


def query_points_in_polygons(
    points: np.ndarray,
    polygons: List['Polygon'],
    use_spatial_index: bool = True
) -> np.ndarray:
    """
    Efficiently query which polygon (if any) contains each point.
    
    Args:
        points: Point coordinates [N, 2+] (XY...)
        polygons: List of Shapely polygons
        use_spatial_index: Use STRtree spatial index for faster queries
        
    Returns:
        Array [N] with polygon index (0 to len(polygons)-1), or -1 if no match
    """
    if not HAS_SPATIAL:
        raise ImportError("Shapely required for point queries")
    
    if len(polygons) == 0:
        return np.full(len(points), -1, dtype=np.int32)
    
    xy = points[:, :2]
    result = np.full(len(points), -1, dtype=np.int32)
    
    if use_spatial_index and len(polygons) > 10:
        # Use spatial index for large polygon sets
        tree = STRtree(polygons)
        
        for i, pt_coords in enumerate(xy):
            pt = Point(pt_coords[0], pt_coords[1])
            potential_matches_idx = tree.query(pt, predicate='intersects')
            
            # Check actual containment
            for idx in potential_matches_idx:
                if polygons[idx].contains(pt):
                    result[i] = idx
                    break
    else:
        # Brute force for small polygon sets
        for i, pt_coords in enumerate(xy):
            pt = Point(pt_coords[0], pt_coords[1])
            for j, poly in enumerate(polygons):
                if poly.contains(pt):
                    result[i] = j
                    break
    
    return result


def create_building_mask_from_polygons(
    points: np.ndarray,
    polygons: List['Polygon'],
    use_spatial_index: bool = True
) -> np.ndarray:
    """
    Create boolean mask indicating which points are inside any polygon.
    
    This is a common operation across building classification modules.
    
    Args:
        points: Point coordinates [N, 2+] (XY...)
        polygons: List of Shapely polygons (building footprints)
        use_spatial_index: Use STRtree spatial index for faster queries
        
    Returns:
        Boolean mask [N] where True = point is inside a polygon
    """
    polygon_indices = query_points_in_polygons(points, polygons, use_spatial_index)
    return polygon_indices >= 0


# ============================================================================
# Height Filtering
# ============================================================================

def filter_by_height(
    points: np.ndarray,
    heights: np.ndarray,
    min_height: Optional[float] = None,
    max_height: Optional[float] = None,
    return_mask: bool = False
) -> np.ndarray:
    """
    Filter points by height range.
    
    Args:
        points: Point cloud array (N, 3+)
        heights: Height values above ground (N,)
        min_height: Minimum height threshold (meters), None for no minimum
        max_height: Maximum height threshold (meters), None for no maximum
        return_mask: If True, return boolean mask; if False, return filtered points
        
    Returns:
        Filtered points array or boolean mask
    """
    mask = np.ones(len(points), dtype=bool)
    
    if min_height is not None:
        mask &= (heights >= min_height)
    
    if max_height is not None:
        mask &= (heights <= max_height)
    
    if return_mask:
        return mask
    else:
        return points[mask]


def compute_height_statistics(heights: np.ndarray) -> dict:
    """
    Compute height statistics for a set of points.
    
    Args:
        heights: Height values above ground (N,)
        
    Returns:
        Dictionary with statistics: min, max, mean, median, std, percentiles
    """
    if len(heights) == 0:
        return {
            'min': 0.0,
            'max': 0.0,
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'p25': 0.0,
            'p75': 0.0,
            'p90': 0.0
        }
    
    return {
        'min': float(np.min(heights)),
        'max': float(np.max(heights)),
        'mean': float(np.mean(heights)),
        'median': float(np.median(heights)),
        'std': float(np.std(heights)),
        'p25': float(np.percentile(heights, 25)),
        'p75': float(np.percentile(heights, 75)),
        'p90': float(np.percentile(heights, 90))
    }


# ============================================================================
# Geometric Calculations
# ============================================================================

def compute_centroid_3d(points: np.ndarray) -> np.ndarray:
    """
    Compute 3D centroid of point cloud.
    
    Args:
        points: Point cloud array (N, 3+)
        
    Returns:
        Centroid coordinates [x, y, z]
    """
    if len(points) == 0:
        return np.array([0.0, 0.0, 0.0])
    
    return np.mean(points[:, :3], axis=0)


def compute_point_cloud_area(points: np.ndarray) -> float:
    """
    Estimate area covered by point cloud (2D convex hull).
    
    Args:
        points: Point cloud array (N, 2+)
        
    Returns:
        Area in square meters
    """
    if len(points) < 3:
        return 0.0
    
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points[:, :2])
        return hull.volume  # In 2D, volume is area
    except Exception as e:
        logger.warning(f"Failed to compute convex hull area: {e}")
        # Fallback: bounding box area
        xy = points[:, :2]
        ranges = np.ptp(xy, axis=0)
        return float(ranges[0] * ranges[1])


def compute_principal_axes(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute principal axes of point cloud using PCA.
    
    Args:
        points: Point cloud array (N, 2+)
        
    Returns:
        Tuple of (eigenvalues, eigenvectors) sorted by eigenvalue magnitude
    """
    if len(points) < 2:
        return np.array([1.0, 1.0]), np.eye(2)
    
    # Center points
    xy = points[:, :2]
    centered = xy - np.mean(xy, axis=0)
    
    # Compute covariance matrix
    cov = np.cov(centered.T)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by eigenvalue (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues, eigenvectors


# ============================================================================
# Feature Computations
# ============================================================================

def compute_verticality(normals: np.ndarray) -> np.ndarray:
    """
    Compute verticality from surface normals.
    
    Verticality is the absolute value of the Z component of the normal.
    High values (close to 1) indicate vertical surfaces (walls).
    
    Args:
        normals: Surface normals array (N, 3)
        
    Returns:
        Verticality values (N,), range [0, 1]
    """
    if len(normals) == 0:
        return np.array([])
    
    return np.abs(normals[:, 2])


def compute_horizontality(normals: np.ndarray) -> np.ndarray:
    """
    Compute horizontality from surface normals.
    
    Horizontality measures how horizontal the surface is.
    High values (close to 1) indicate horizontal surfaces (roofs, ground).
    
    Args:
        normals: Surface normals array (N, 3)
        
    Returns:
        Horizontality values (N,), range [0, 1]
    """
    if len(normals) == 0:
        return np.array([])
    
    # Horizontality is abs(nz) when normal points up/down
    # For surfaces, we want 1 - abs(nx, ny) contribution
    return np.abs(normals[:, 2])


def compute_planarity(eigenvalues: np.ndarray) -> np.ndarray:
    """
    Compute planarity from eigenvalues of local covariance.
    
    Planarity measures how flat a surface is locally.
    Formula: (lambda2 - lambda1) / lambda3
    High values (close to 1) indicate planar surfaces.
    
    Args:
        eigenvalues: Eigenvalues array (N, 3), sorted descending
        
    Returns:
        Planarity values (N,), range [0, 1]
    """
    if len(eigenvalues) == 0:
        return np.array([])
    
    # Avoid division by zero
    lambda1 = eigenvalues[:, 2]  # Smallest
    lambda2 = eigenvalues[:, 1]  # Middle
    lambda3 = eigenvalues[:, 0]  # Largest
    
    eps = 1e-8
    planarity = (lambda2 - lambda1) / (lambda3 + eps)
    
    return np.clip(planarity, 0.0, 1.0)


# ============================================================================
# Distance Computations
# ============================================================================

def compute_distances_to_centroids(
    points: np.ndarray,
    centroids: np.ndarray
) -> np.ndarray:
    """
    Compute distances from points to nearest centroid.
    
    Args:
        points: Point coordinates (N, 2+)
        centroids: Centroid coordinates (M, 2+)
        
    Returns:
        Array (N,) with distance to nearest centroid
    """
    if len(points) == 0 or len(centroids) == 0:
        return np.array([])
    
    # Extract XY coordinates
    pts_xy = points[:, :2]
    cent_xy = centroids[:, :2]
    
    # Compute pairwise distances
    from scipy.spatial.distance import cdist
    distances = cdist(pts_xy, cent_xy, metric='euclidean')
    
    # Return minimum distance for each point
    return np.min(distances, axis=1)


def compute_nearest_centroid_indices(
    points: np.ndarray,
    centroids: np.ndarray
) -> np.ndarray:
    """
    Find index of nearest centroid for each point.
    
    Args:
        points: Point coordinates (N, 2+)
        centroids: Centroid coordinates (M, 2+)
        
    Returns:
        Array (N,) with index of nearest centroid (0 to M-1)
    """
    if len(points) == 0 or len(centroids) == 0:
        return np.array([])
    
    # Extract XY coordinates
    pts_xy = points[:, :2]
    cent_xy = centroids[:, :2]
    
    # Compute pairwise distances
    from scipy.spatial.distance import cdist
    distances = cdist(pts_xy, cent_xy, metric='euclidean')
    
    # Return index of minimum distance for each point
    return np.argmin(distances, axis=1)


# ============================================================================
# Bounding Box Utilities (for 3D Extrusion)
# ============================================================================

def compute_bounding_box_volume(points: np.ndarray) -> float:
    """
    Compute volume of axis-aligned 3D bounding box.
    
    This is the canonical implementation used by extrusion_3d.py and other
    building classification modules. Consolidates logic previously in
    features/compute/density.py.
    
    Args:
        points: Point cloud array [N, 3] (XYZ coordinates)
        
    Returns:
        Volume of bounding box in cubic units (typically cubic meters)
        
    Example:
        >>> points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        >>> volume = compute_bounding_box_volume(points)
        >>> print(f"Volume: {volume:.2f} m³")
    """
    if len(points) == 0:
        return 0.0
    
    # Compute min/max along each axis
    min_coords = np.min(points[:, :3], axis=0)
    max_coords = np.max(points[:, :3], axis=0)
    
    # Compute dimensions (length, width, height)
    dimensions = max_coords - min_coords
    
    # Volume = length x width x height
    volume = np.prod(dimensions)
    
    return float(volume)


def compute_bounding_box_2d(points: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute 2D axis-aligned bounding box (AABB) for point cloud.
    
    Args:
        points: Point cloud array [N, 2+] (at least XY coordinates)
        
    Returns:
        Tuple (xmin, ymin, xmax, ymax) in meters
        
    Example:
        >>> points = np.array([[0, 0, 5], [10, 10, 5], [5, 5, 5]])
        >>> bbox = compute_bounding_box_2d(points)
        >>> print(f"BBox: {bbox}")  # (0.0, 0.0, 10.0, 10.0)
    """
    if len(points) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    
    xy = points[:, :2]
    
    xmin = float(np.min(xy[:, 0]))
    ymin = float(np.min(xy[:, 1]))
    xmax = float(np.max(xy[:, 0]))
    ymax = float(np.max(xy[:, 1]))
    
    return (xmin, ymin, xmax, ymax)


def create_bbox_polygon(bbox: Tuple[float, float, float, float]) -> Optional['Polygon']:
    """
    Create Shapely Polygon from 2D bounding box.
    
    Args:
        bbox: Tuple (xmin, ymin, xmax, ymax)
        
    Returns:
        Shapely Polygon representing bounding box
    """
    if not HAS_SPATIAL:
        raise ImportError("Shapely required for bbox polygon creation")
    
    from shapely.geometry import box as shapely_box
    
    xmin, ymin, xmax, ymax = bbox
    return shapely_box(xmin, ymin, xmax, ymax)


# ============================================================================
# Validation
# ============================================================================

def validate_point_cloud(points: np.ndarray, min_points: int = 1) -> bool:
    """
    Validate point cloud array.
    
    Args:
        points: Point cloud array
        min_points: Minimum number of points required
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(points, np.ndarray):
        logger.warning("Points must be numpy array")
        return False
    
    if len(points.shape) != 2:
        logger.warning(f"Points must be 2D array, got shape {points.shape}")
        return False
    
    if points.shape[1] < 2:
        logger.warning(f"Points must have at least 2 columns (XY), got {points.shape[1]}")
        return False
    
    if len(points) < min_points:
        logger.warning(f"Points must have at least {min_points} points, got {len(points)}")
        return False
    
    return True
