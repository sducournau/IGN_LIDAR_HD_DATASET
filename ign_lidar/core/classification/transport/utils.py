"""
Transport Module Shared Utilities

This module provides shared utility functions for transport classification:
- Geometric validation functions (height, planarity, roughness)
- Intensity filtering and refinement
- Curvature calculation for adaptive buffering
- Type-specific tolerance determination
- Intersection detection

Author: Transport Module Consolidation (Phase 3)
Date: October 22, 2025
Version: 3.1.0
"""

import logging
from typing import Optional, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Optional dependencies with graceful fallback
try:
    from shapely.geometry import Point, LineString
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    logger.debug("shapely not available - geometric operations limited")

try:
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.debug("scipy not available - curvature calculation uses fallback")


# ============================================================================
# Validation Functions
# ============================================================================

def validate_transport_height(
    height: np.ndarray,
    height_min: float,
    height_max: float,
    transport_type: str = "road"
) -> np.ndarray:
    """
    Validate height values for transport features.
    
    Args:
        height: Height above ground [N] in meters
        height_min: Minimum valid height (m)
        height_max: Maximum valid height (m)
        transport_type: Type of transport ("road" or "railway")
        
    Returns:
        Boolean mask [N] of valid height values
    """
    valid_mask = (height >= height_min) & (height <= height_max)
    
    n_valid = valid_mask.sum()
    n_total = len(height)
    logger.debug(
        f"{transport_type.capitalize()} height validation: "
        f"{n_valid:,}/{n_total:,} points in range [{height_min:.1f}, {height_max:.1f}]m"
    )
    
    return valid_mask


def check_transport_planarity(
    planarity: np.ndarray,
    planarity_min: float,
    transport_type: str = "road"
) -> np.ndarray:
    """
    Check planarity values for transport features.
    
    Transport surfaces (roads/railways) should have high planarity.
    
    Args:
        planarity: Planarity values [N], range [0, 1]
        planarity_min: Minimum planarity threshold
        transport_type: Type of transport ("road" or "railway")
        
    Returns:
        Boolean mask [N] of planar surface points
    """
    planar_mask = planarity >= planarity_min
    
    n_planar = planar_mask.sum()
    n_total = len(planarity)
    logger.debug(
        f"{transport_type.capitalize()} planarity check: "
        f"{n_planar:,}/{n_total:,} points with planarity >= {planarity_min:.2f}"
    )
    
    return planar_mask


def filter_by_roughness(
    roughness: np.ndarray,
    roughness_max: float,
    transport_type: str = "road"
) -> np.ndarray:
    """
    Filter points by surface roughness.
    
    Transport surfaces should have low roughness (smooth).
    
    Args:
        roughness: Surface roughness [N]
        roughness_max: Maximum roughness threshold
        transport_type: Type of transport ("road" or "railway")
        
    Returns:
        Boolean mask [N] of smooth surface points
    """
    smooth_mask = roughness <= roughness_max
    
    n_smooth = smooth_mask.sum()
    n_total = len(roughness)
    logger.debug(
        f"{transport_type.capitalize()} roughness filter: "
        f"{n_smooth:,}/{n_total:,} points with roughness <= {roughness_max:.3f}"
    )
    
    return smooth_mask


def filter_by_intensity(
    intensity: np.ndarray,
    intensity_min: float,
    intensity_max: float,
    material: str = "asphalt"
) -> np.ndarray:
    """
    Filter points by LiDAR intensity for material detection.
    
    Different road/rail materials have characteristic intensity ranges:
    - Asphalt: 0.2-0.6 (dark, low reflectance)
    - Concrete: 0.4-0.8 (lighter, higher reflectance)
    - Gravel: 0.3-0.7 (variable reflectance)
    
    Args:
        intensity: LiDAR intensity [N], normalized [0, 1]
        intensity_min: Minimum intensity threshold
        intensity_max: Maximum intensity threshold
        material: Material type for logging
        
    Returns:
        Boolean mask [N] of points matching intensity range
    """
    intensity_mask = (intensity >= intensity_min) & (intensity <= intensity_max)
    
    n_match = intensity_mask.sum()
    n_total = len(intensity)
    logger.debug(
        f"{material.capitalize()} intensity filter: "
        f"{n_match:,}/{n_total:,} points in range [{intensity_min:.2f}, {intensity_max:.2f}]"
    )
    
    return intensity_mask


def check_horizontality(
    normals: np.ndarray,
    horizontality_min: float = 0.9
) -> np.ndarray:
    """
    Check surface horizontality from normal vectors.
    
    Transport surfaces should be mostly horizontal (normal pointing up).
    
    Args:
        normals: Surface normals [N, 3] (nx, ny, nz)
        horizontality_min: Minimum z-component of normal
        
    Returns:
        Boolean mask [N] of horizontal surface points
    """
    if normals.shape[1] != 3:
        raise ValueError(f"Expected normals shape (N, 3), got {normals.shape}")
    
    # Horizontal surfaces have normal vector pointing up (high z-component)
    horizontality = np.abs(normals[:, 2])
    horizontal_mask = horizontality >= horizontality_min
    
    n_horizontal = horizontal_mask.sum()
    n_total = len(normals)
    logger.debug(
        f"Horizontality check: {n_horizontal:,}/{n_total:,} points "
        f"with |nz| >= {horizontality_min:.2f}"
    )
    
    return horizontal_mask


# ============================================================================
# Curvature Calculation
# ============================================================================

def calculate_curvature(
    coords: np.ndarray,
    smooth_sigma: float = 1.0
) -> np.ndarray:
    """
    Calculate curvature at each point along a line.
    
    Uses second derivative of coordinates to compute local curvature.
    Higher curvature indicates sharper turns.
    
    Args:
        coords: Array of coordinates [N, 2] or [N, 3]
        smooth_sigma: Gaussian smoothing sigma for noise reduction
        
    Returns:
        Array of curvature values [N] (normalized 0-1, where 1 is highest curvature)
    """
    if len(coords) < 3:
        # Too short for curvature calculation
        return np.zeros(len(coords))
    
    # Extract XY coordinates
    xy = coords[:, :2].astype(float)
    
    # Smooth coordinates to reduce noise
    if smooth_sigma > 0 and HAS_SCIPY:
        xy[:, 0] = gaussian_filter1d(xy[:, 0], smooth_sigma)
        xy[:, 1] = gaussian_filter1d(xy[:, 1], smooth_sigma)
    
    # Calculate first and second derivatives
    dx = np.gradient(xy[:, 0])
    dy = np.gradient(xy[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Curvature formula: κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = np.power(dx**2 + dy**2, 1.5)
    
    # Avoid division by zero
    denominator = np.where(denominator < 1e-10, 1e-10, denominator)
    
    curvature = numerator / denominator
    
    # Normalize to 0-1 range
    if curvature.max() > 0:
        curvature = curvature / curvature.max()
    
    return curvature


def compute_adaptive_width(
    curvature: np.ndarray,
    base_width: float,
    curvature_factor: float = 0.25
) -> np.ndarray:
    """
    Compute adaptive buffer width based on curvature.
    
    Wider buffers at curves to better capture road/rail geometry.
    
    Args:
        curvature: Curvature values [N] (0-1)
        base_width: Base width from attributes (m)
        curvature_factor: Width increase factor (0.0-1.0)
        
    Returns:
        Array of adaptive widths [N] in meters
    """
    # Width increases with curvature
    width_adjustment = 1.0 + (curvature * curvature_factor)
    adaptive_widths = base_width * width_adjustment
    
    logger.debug(
        f"Adaptive width: {base_width:.1f}m base → "
        f"[{adaptive_widths.min():.1f}, {adaptive_widths.max():.1f}]m range"
    )
    
    return adaptive_widths


# ============================================================================
# Type-Specific Tolerance
# ============================================================================

def get_road_type_tolerance(road_type: str, config) -> float:
    """
    Get buffer tolerance based on road type.
    
    Different road types have different width characteristics and
    need different buffer tolerances for accurate classification.
    
    Args:
        road_type: Road type from BD TOPO® (nature attribute)
        config: BufferingConfig with type-specific tolerances
        
    Returns:
        Buffer tolerance in meters
    """
    if not hasattr(config, 'type_specific_tolerance') or not config.type_specific_tolerance:
        return 0.5  # Default
    
    # Map BD TOPO® road types to tolerances
    type_map = {
        'Autoroute': config.tolerance_motorway,
        'Route à 2 chaussées': config.tolerance_motorway,
        'Route principale': config.tolerance_primary,
        'Route secondaire': config.tolerance_secondary,
        'Route tertiaire': config.tolerance_secondary,
        'Rue résidentielle': config.tolerance_residential,
        'Route de service': config.tolerance_service,
        'Voie piétonne': config.tolerance_service,
        'Piste cyclable': config.tolerance_service,
    }
    
    tolerance = type_map.get(road_type, 0.4)  # Default for unknown types
    
    logger.debug(f"Road type '{road_type}' → tolerance {tolerance:.2f}m")
    
    return tolerance


def get_railway_type_tolerance(railway_type: str, config) -> float:
    """
    Get buffer tolerance based on railway type.
    
    Args:
        railway_type: Railway type/nature from BD TOPO®
        config: BufferingConfig with type-specific tolerances
        
    Returns:
        Buffer tolerance in meters
    """
    if not hasattr(config, 'type_specific_tolerance') or not config.type_specific_tolerance:
        return 0.5  # Default
    
    # Check for tram/light rail
    railway_lower = str(railway_type).lower()
    if 'tramway' in railway_lower or 'tram' in railway_lower:
        tolerance = config.tolerance_railway_tram
    else:
        tolerance = config.tolerance_railway_main
    
    logger.debug(f"Railway type '{railway_type}' → tolerance {tolerance:.2f}m")
    
    return tolerance


# ============================================================================
# Intersection Detection
# ============================================================================

def detect_intersections(
    geometries: List,
    threshold: float = 1.0
) -> List:
    """
    Detect intersections between road/rail centerlines.
    
    Useful for buffer enhancement at junctions where multiple
    transport features meet.
    
    Args:
        geometries: List of LineString geometries (centerlines)
        threshold: Maximum distance to consider intersection (m)
        
    Returns:
        List of Point geometries representing intersections
    """
    if not HAS_SHAPELY:
        logger.warning("shapely not available - intersection detection disabled")
        return []
    
    intersections = []
    
    for i, geom1 in enumerate(geometries):
        if not isinstance(geom1, LineString):
            continue
            
        for geom2 in geometries[i+1:]:
            if not isinstance(geom2, LineString):
                continue
            
            try:
                # Check if geometries are close enough
                if geom1.distance(geom2) < threshold:
                    intersection = geom1.intersection(geom2)
                    
                    if intersection and not intersection.is_empty:
                        if isinstance(intersection, Point):
                            intersections.append(intersection)
                        elif hasattr(intersection, 'geoms'):
                            # MultiPoint or GeometryCollection
                            for geom in intersection.geoms:
                                if isinstance(geom, Point):
                                    intersections.append(geom)
            except Exception as e:
                logger.debug(f"Intersection detection failed: {e}")
                continue
    
    logger.debug(f"Detected {len(intersections)} intersections from {len(geometries)} geometries")
    
    return intersections


# ============================================================================
# Geometric Helpers
# ============================================================================

def create_adaptive_buffer(
    geometry: 'LineString',
    base_width: float,
    curvature_aware: bool = True,
    curvature_factor: float = 0.25
) -> 'Polygon':
    """
    Create buffer with adaptive width based on geometry curvature.
    
    Args:
        geometry: Road/rail centerline (LineString)
        base_width: Base width from attributes (m)
        curvature_aware: Enable curvature-based width adjustment
        curvature_factor: Width increase factor at curves
        
    Returns:
        Buffered polygon with variable width
    """
    if not HAS_SHAPELY:
        raise ImportError("shapely required for adaptive buffering")
    
    if not curvature_aware or len(geometry.coords) < 3:
        # Simple fixed-width buffer
        return geometry.buffer(base_width / 2.0, cap_style=2)
    
    # Extract coordinates
    coords = np.array(geometry.coords)
    
    # Calculate curvature
    curvatures = calculate_curvature(coords)
    
    # Pad curvature to match coords length
    curvatures_padded = np.zeros(len(coords))
    if len(curvatures) > 0:
        curvatures_padded[1:-1] = curvatures[:len(coords)-2]
        curvatures_padded[0] = curvatures[0]
        curvatures_padded[-1] = curvatures[-1] if len(curvatures) > 1 else curvatures[0]
    
    # Create segments with adaptive widths
    segments = []
    for i in range(len(coords) - 1):
        # Calculate width adjustment
        curve_adj = curvatures_padded[i] * curvature_factor
        segment_width = base_width * (1.0 + curve_adj)
        
        # Create buffered segment
        segment = LineString([coords[i], coords[i+1]])
        buffered = segment.buffer(segment_width / 2.0, cap_style=1)
        segments.append(buffered)
    
    # Union all segments
    if len(segments) == 0:
        return geometry.buffer(base_width / 2.0, cap_style=2)
    
    return unary_union(segments)


def calculate_distance_to_centerline(
    points: np.ndarray,
    centerline: 'LineString'
) -> np.ndarray:
    """
    Calculate distance from points to centerline.
    
    Useful for confidence scoring - points closer to centerline
    are more likely to be correctly classified.
    
    Args:
        points: Point coordinates [N, 2] or [N, 3]
        centerline: Road/rail centerline (LineString)
        
    Returns:
        Array of distances [N] in meters
    """
    if not HAS_SHAPELY:
        raise ImportError("shapely required for distance calculation")
    
    xy = points[:, :2]
    distances = np.array([centerline.distance(Point(pt)) for pt in xy])
    
    return distances


# ============================================================================
# Export
# ============================================================================

__all__ = [
    # Validation functions
    'validate_transport_height',
    'check_transport_planarity',
    'filter_by_roughness',
    'filter_by_intensity',
    'check_horizontality',
    
    # Curvature functions
    'calculate_curvature',
    'compute_adaptive_width',
    
    # Type-specific functions
    'get_road_type_tolerance',
    'get_railway_type_tolerance',
    
    # Intersection detection
    'detect_intersections',
    
    # Geometric helpers
    'create_adaptive_buffer',
    'calculate_distance_to_centerline',
    
    # Availability flags
    'HAS_SHAPELY',
    'HAS_SCIPY',
]
