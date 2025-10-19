"""
Plane Detection Module - Horizontal, Vertical, and Inclined Planes

This module provides comprehensive plane detection for building classification:
- Horizontal planes (toits plats, terrasses, dalles)
- Vertical planes (murs, façades, pignons)
- Inclined planes (toits en pente, versants)
- Complex architectural elements (lucarnes, cheminées, balcons)

Each detection function uses geometric features (normals, planarity, verticality)
and spatial relationships to classify building elements.

Author: Plane Detection Enhancement
Date: October 19, 2025
"""

import logging
from typing import Optional, Dict, List, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PlaneType(str, Enum):
    """Types of architectural planes."""
    HORIZONTAL = "horizontal"          # Toits plats, dalles
    VERTICAL = "vertical"              # Murs, façades
    INCLINED = "inclined"              # Toits en pente
    NEAR_HORIZONTAL = "near_horizontal"  # Légèrement incliné (<15°)
    NEAR_VERTICAL = "near_vertical"    # Presque vertical (>75°)


@dataclass
class PlaneSegment:
    """Represents a detected plane segment."""
    plane_type: PlaneType
    point_indices: np.ndarray
    normal: np.ndarray  # [3] normal vector
    centroid: np.ndarray  # [3] XYZ centroid
    planarity: float  # 0-1, higher = more planar
    area: float  # m²
    orientation_angle: float  # degrees from horizontal
    
    # Additional attributes
    height_mean: float = 0.0
    height_std: float = 0.0
    n_points: int = 0


class PlaneDetector:
    """
    Comprehensive plane detector for architectural elements.
    
    Detects and classifies:
    1. Horizontal planes: Flat roofs, terraces, floors
    2. Vertical planes: Walls, facades
    3. Inclined planes: Sloped roofs, pitched surfaces
    4. Architectural details: Dormers, chimneys, balconies
    """
    
    def __init__(
        self,
        # Horizontal plane thresholds
        horizontal_angle_max: float = 10.0,  # degrees from horizontal
        horizontal_planarity_min: float = 0.75,
        
        # Vertical plane thresholds
        vertical_angle_min: float = 75.0,  # degrees from horizontal
        vertical_planarity_min: float = 0.65,
        
        # Inclined plane thresholds
        inclined_angle_min: float = 15.0,  # degrees from horizontal
        inclined_angle_max: float = 70.0,
        inclined_planarity_min: float = 0.70,
        
        # General thresholds
        min_points_per_plane: int = 50,
        max_plane_distance: float = 0.15,  # meters
        use_spatial_coherence: bool = True
    ):
        """
        Initialize plane detector.
        
        Args:
            horizontal_angle_max: Maximum angle from horizontal for flat surfaces (degrees)
            horizontal_planarity_min: Minimum planarity for horizontal planes
            vertical_angle_min: Minimum angle from horizontal for vertical surfaces (degrees)
            vertical_planarity_min: Minimum planarity for vertical planes
            inclined_angle_min: Minimum angle for inclined surfaces (degrees)
            inclined_angle_max: Maximum angle for inclined surfaces (degrees)
            inclined_planarity_min: Minimum planarity for inclined planes
            min_points_per_plane: Minimum points to form valid plane
            max_plane_distance: Maximum distance from plane for inliers (meters)
            use_spatial_coherence: Use spatial clustering for plane segmentation
        """
        self.horizontal_angle_max = horizontal_angle_max
        self.horizontal_planarity_min = horizontal_planarity_min
        self.vertical_angle_min = vertical_angle_min
        self.vertical_planarity_min = vertical_planarity_min
        self.inclined_angle_min = inclined_angle_min
        self.inclined_angle_max = inclined_angle_max
        self.inclined_planarity_min = inclined_planarity_min
        self.min_points_per_plane = min_points_per_plane
        self.max_plane_distance = max_plane_distance
        self.use_spatial_coherence = use_spatial_coherence
        
        logger.info("Plane Detector initialized")
        logger.info(f"  Horizontal: angle ≤{horizontal_angle_max}°, planarity ≥{horizontal_planarity_min}")
        logger.info(f"  Vertical: angle ≥{vertical_angle_min}°, planarity ≥{vertical_planarity_min}")
        logger.info(f"  Inclined: {inclined_angle_min}° ≤ angle ≤ {inclined_angle_max}°")
    
    def detect_horizontal_planes(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        planarity: np.ndarray,
        height: Optional[np.ndarray] = None,
        min_height: float = 2.0
    ) -> List[PlaneSegment]:
        """
        Detect horizontal planes (flat roofs, terraces, floors).
        
        Horizontal planes characteristics:
        - Normal vector nearly vertical (nz ≈ 1 or nz ≈ -1)
        - High planarity (smooth, flat surface)
        - Spatially coherent (points form continuous surface)
        - Typically at building height (>2m above ground)
        
        Args:
            points: Point coordinates [N, 3]
            normals: Surface normals [N, 3]
            planarity: Planarity values [N] in [0, 1]
            height: Height above ground [N] (optional)
            min_height: Minimum height for roof detection (meters)
            
        Returns:
            List of PlaneSegment objects for horizontal planes
        """
        # Compute angle from horizontal (angle with Z axis)
        # For horizontal planes, normal should point up/down (nz ≈ ±1)
        nz_abs = np.abs(normals[:, 2])
        angle_from_horizontal = np.degrees(np.arccos(np.clip(nz_abs, 0, 1)))
        
        # Select horizontal plane candidates
        horizontal_mask = (
            (angle_from_horizontal <= self.horizontal_angle_max) &
            (planarity >= self.horizontal_planarity_min)
        )
        
        # Filter by height if available (roofs are elevated)
        if height is not None:
            horizontal_mask = horizontal_mask & (height >= min_height)
        
        n_horizontal = horizontal_mask.sum()
        logger.info(f"Detected {n_horizontal:,} horizontal plane points")
        
        if n_horizontal < self.min_points_per_plane:
            return []
        
        # Segment into individual planes
        horizontal_indices = np.where(horizontal_mask)[0]
        planes = self._segment_planes(
            points[horizontal_indices],
            normals[horizontal_indices],
            planarity[horizontal_indices],
            horizontal_indices,
            PlaneType.HORIZONTAL
        )
        
        logger.info(f"  Segmented into {len(planes)} horizontal plane(s)")
        return planes
    
    def detect_vertical_planes(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        planarity: np.ndarray,
        height: Optional[np.ndarray] = None,
        min_height: float = 0.5
    ) -> List[PlaneSegment]:
        """
        Detect vertical planes (walls, facades).
        
        Vertical planes characteristics:
        - Normal vector nearly horizontal (nz ≈ 0)
        - Good planarity (flat wall surface)
        - Vertical extent (height variation)
        - Spatially coherent
        
        Args:
            points: Point coordinates [N, 3]
            normals: Surface normals [N, 3]
            planarity: Planarity values [N] in [0, 1]
            height: Height above ground [N] (optional)
            min_height: Minimum height for wall detection (meters)
            
        Returns:
            List of PlaneSegment objects for vertical planes
        """
        # Compute angle from horizontal
        # For vertical planes, normal should be horizontal (nz ≈ 0)
        nz_abs = np.abs(normals[:, 2])
        angle_from_horizontal = np.degrees(np.arccos(np.clip(nz_abs, 0, 1)))
        
        # Select vertical plane candidates
        vertical_mask = (
            (angle_from_horizontal >= self.vertical_angle_min) &
            (planarity >= self.vertical_planarity_min)
        )
        
        # Filter by height if available (walls should have some elevation)
        if height is not None:
            vertical_mask = vertical_mask & (height >= min_height)
        
        n_vertical = vertical_mask.sum()
        logger.info(f"Detected {n_vertical:,} vertical plane points")
        
        if n_vertical < self.min_points_per_plane:
            return []
        
        # Segment into individual planes (walls)
        vertical_indices = np.where(vertical_mask)[0]
        planes = self._segment_planes(
            points[vertical_indices],
            normals[vertical_indices],
            planarity[vertical_indices],
            vertical_indices,
            PlaneType.VERTICAL
        )
        
        logger.info(f"  Segmented into {len(planes)} vertical plane(s) (walls)")
        return planes
    
    def detect_inclined_planes(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        planarity: np.ndarray,
        height: Optional[np.ndarray] = None,
        min_height: float = 2.0
    ) -> List[PlaneSegment]:
        """
        Detect inclined planes (sloped roofs, pitched surfaces).
        
        Inclined planes characteristics:
        - Normal vector at intermediate angle (15° < angle < 70°)
        - Good planarity (smooth sloped surface)
        - Typically at roof height
        - Common angles: 30-45° for pitched roofs
        
        Args:
            points: Point coordinates [N, 3]
            normals: Surface normals [N, 3]
            planarity: Planarity values [N] in [0, 1]
            height: Height above ground [N] (optional)
            min_height: Minimum height for roof detection (meters)
            
        Returns:
            List of PlaneSegment objects for inclined planes
        """
        # Compute angle from horizontal
        nz_abs = np.abs(normals[:, 2])
        angle_from_horizontal = np.degrees(np.arccos(np.clip(nz_abs, 0, 1)))
        
        # Select inclined plane candidates
        inclined_mask = (
            (angle_from_horizontal >= self.inclined_angle_min) &
            (angle_from_horizontal <= self.inclined_angle_max) &
            (planarity >= self.inclined_planarity_min)
        )
        
        # Filter by height if available (roofs are elevated)
        if height is not None:
            inclined_mask = inclined_mask & (height >= min_height)
        
        n_inclined = inclined_mask.sum()
        logger.info(f"Detected {n_inclined:,} inclined plane points")
        
        if n_inclined < self.min_points_per_plane:
            return []
        
        # Segment into individual planes (roof facets)
        inclined_indices = np.where(inclined_mask)[0]
        planes = self._segment_planes(
            points[inclined_indices],
            normals[inclined_indices],
            planarity[inclined_indices],
            inclined_indices,
            PlaneType.INCLINED
        )
        
        logger.info(f"  Segmented into {len(planes)} inclined plane(s) (roof facets)")
        return planes
    
    def detect_all_planes(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        planarity: np.ndarray,
        height: Optional[np.ndarray] = None
    ) -> Dict[PlaneType, List[PlaneSegment]]:
        """
        Detect all types of planes in one pass.
        
        Args:
            points: Point coordinates [N, 3]
            normals: Surface normals [N, 3]
            planarity: Planarity values [N] in [0, 1]
            height: Height above ground [N] (optional)
            
        Returns:
            Dictionary mapping PlaneType to list of PlaneSegment objects
        """
        logger.info("Detecting all plane types...")
        
        results = {
            PlaneType.HORIZONTAL: self.detect_horizontal_planes(
                points, normals, planarity, height
            ),
            PlaneType.VERTICAL: self.detect_vertical_planes(
                points, normals, planarity, height
            ),
            PlaneType.INCLINED: self.detect_inclined_planes(
                points, normals, planarity, height
            )
        }
        
        # Summary statistics
        total_planes = sum(len(planes) for planes in results.values())
        total_points = sum(
            sum(p.n_points for p in planes)
            for planes in results.values()
        )
        
        logger.info(f"Plane detection complete:")
        logger.info(f"  Total planes: {total_planes}")
        logger.info(f"  Total points: {total_points:,}")
        logger.info(f"  Horizontal: {len(results[PlaneType.HORIZONTAL])} planes")
        logger.info(f"  Vertical: {len(results[PlaneType.VERTICAL])} planes")
        logger.info(f"  Inclined: {len(results[PlaneType.INCLINED])} planes")
        
        return results
    
    def classify_roof_types(
        self,
        horizontal_planes: List[PlaneSegment],
        inclined_planes: List[PlaneSegment]
    ) -> Dict[str, List[PlaneSegment]]:
        """
        Classify roof types based on detected planes.
        
        Roof types:
        - Flat roof: Only horizontal planes
        - Gable roof: Two inclined planes (pitched roof)
        - Hip roof: Four or more inclined planes
        - Complex roof: Mixed horizontal and inclined planes
        
        Args:
            horizontal_planes: List of detected horizontal planes
            inclined_planes: List of detected inclined planes
            
        Returns:
            Dictionary with roof type classifications
        """
        roof_classification = {
            'flat': [],
            'gable': [],
            'hip': [],
            'complex': []
        }
        
        n_horizontal = len(horizontal_planes)
        n_inclined = len(inclined_planes)
        
        if n_horizontal > 0 and n_inclined == 0:
            # Pure flat roof
            roof_classification['flat'] = horizontal_planes
            logger.info("Classified as FLAT ROOF")
            
        elif n_inclined == 2:
            # Likely gable roof (two pitched sides)
            roof_classification['gable'] = inclined_planes
            logger.info("Classified as GABLE ROOF (2 inclined planes)")
            
        elif n_inclined >= 4:
            # Hip roof (four or more sides)
            roof_classification['hip'] = inclined_planes
            logger.info(f"Classified as HIP ROOF ({n_inclined} inclined planes)")
            
        elif n_horizontal > 0 and n_inclined > 0:
            # Complex roof with mixed elements
            roof_classification['complex'] = horizontal_planes + inclined_planes
            logger.info(f"Classified as COMPLEX ROOF ({n_horizontal} horizontal, {n_inclined} inclined)")
        
        return roof_classification
    
    def _segment_planes(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        planarity: np.ndarray,
        original_indices: np.ndarray,
        plane_type: PlaneType
    ) -> List[PlaneSegment]:
        """
        Segment points into individual planar regions.
        
        Uses spatial clustering and normal similarity to group points
        into coherent plane segments.
        
        Args:
            points: Filtered point coordinates [M, 3]
            normals: Filtered normals [M, 3]
            planarity: Filtered planarity [M]
            original_indices: Indices into original point cloud [M]
            plane_type: Type of plane being segmented
            
        Returns:
            List of PlaneSegment objects
        """
        if len(points) < self.min_points_per_plane:
            return []
        
        # Simple segmentation: treat all points as one plane
        # TODO: Implement proper region growing or clustering
        
        # Compute plane properties
        centroid = points.mean(axis=0)
        mean_normal = normals.mean(axis=0)
        mean_normal = mean_normal / np.linalg.norm(mean_normal)
        mean_planarity = planarity.mean()
        
        # Compute orientation angle
        nz_abs = abs(mean_normal[2])
        orientation_angle = np.degrees(np.arccos(np.clip(nz_abs, 0, 1)))
        
        # Estimate area (rough approximation)
        xy_extent = points[:, :2].max(axis=0) - points[:, :2].min(axis=0)
        area_estimate = xy_extent[0] * xy_extent[1]
        
        # Compute height statistics
        heights = points[:, 2]
        height_mean = heights.mean()
        height_std = heights.std()
        
        plane = PlaneSegment(
            plane_type=plane_type,
            point_indices=original_indices,
            normal=mean_normal,
            centroid=centroid,
            planarity=mean_planarity,
            area=area_estimate,
            orientation_angle=orientation_angle,
            height_mean=height_mean,
            height_std=height_std,
            n_points=len(points)
        )
        
        return [plane]


def detect_architectural_elements(
    points: np.ndarray,
    normals: np.ndarray,
    planarity: np.ndarray,
    height: np.ndarray,
    planes: Dict[PlaneType, List[PlaneSegment]]
) -> Dict[str, List[np.ndarray]]:
    """
    Detect specific architectural elements using plane analysis.
    
    Elements detected:
    - Balconies: Small horizontal planes projecting from walls
    - Chimneys: Small vertical structures on roofs
    - Dormers: Vertical planes protruding from inclined roofs
    - Parapets: Low walls on flat roofs
    
    Args:
        points: Point coordinates [N, 3]
        normals: Surface normals [N, 3]
        planarity: Planarity values [N]
        height: Height above ground [N]
        planes: Detected planes by type
        
    Returns:
        Dictionary mapping element type to point indices
    """
    elements = {
        'balconies': [],
        'chimneys': [],
        'dormers': [],
        'parapets': []
    }
    
    # Detect balconies: small horizontal planes at intermediate heights
    horizontal_planes = planes.get(PlaneType.HORIZONTAL, [])
    for plane in horizontal_planes:
        if 2.0 < plane.height_mean < 15.0 and plane.area < 20.0:  # Small area
            if plane.n_points < 500:  # Not a full roof
                elements['balconies'].append(plane.point_indices)
                logger.debug(f"Detected balcony: {plane.n_points} points at {plane.height_mean:.1f}m")
    
    # Detect chimneys: small vertical structures on roofs
    vertical_planes = planes.get(PlaneType.VERTICAL, [])
    for plane in vertical_planes:
        if plane.height_mean > 8.0 and plane.area < 10.0:  # High, small area
            if plane.n_points < 300:  # Small structure
                elements['chimneys'].append(plane.point_indices)
                logger.debug(f"Detected chimney: {plane.n_points} points at {plane.height_mean:.1f}m")
    
    # Detect dormers: vertical planes above inclined roofs
    inclined_planes = planes.get(PlaneType.INCLINED, [])
    if inclined_planes and vertical_planes:
        for v_plane in vertical_planes:
            for i_plane in inclined_planes:
                # Check if vertical plane is above inclined plane
                if v_plane.height_mean > i_plane.height_mean + 1.0:
                    elements['dormers'].append(v_plane.point_indices)
                    logger.debug(f"Detected dormer: {v_plane.n_points} points")
                    break
    
    # Detect parapets: low vertical structures on flat roofs
    if horizontal_planes and vertical_planes:
        for h_plane in horizontal_planes:
            for v_plane in vertical_planes:
                # Check if vertical plane is just above horizontal (parapet)
                height_diff = abs(v_plane.height_mean - h_plane.height_mean)
                if 0.5 < height_diff < 2.0 and v_plane.n_points < 200:
                    elements['parapets'].append(v_plane.point_indices)
                    logger.debug(f"Detected parapet: {v_plane.n_points} points")
    
    # Log summary
    for elem_type, elem_list in elements.items():
        if elem_list:
            total_points = sum(len(indices) for indices in elem_list)
            logger.info(f"Detected {len(elem_list)} {elem_type} ({total_points:,} points)")
    
    return elements


__all__ = [
    'PlaneType',
    'PlaneSegment',
    'PlaneDetector',
    'detect_architectural_elements'
]
