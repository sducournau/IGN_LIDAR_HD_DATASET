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
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PlaneType(str, Enum):
    """Types of architectural planes."""

    HORIZONTAL = "horizontal"  # Toits plats, dalles
    VERTICAL = "vertical"  # Murs, façades
    INCLINED = "inclined"  # Toits en pente
    NEAR_HORIZONTAL = "near_horizontal"  # Légèrement incliné (<15°)
    NEAR_VERTICAL = "near_vertical"  # Presque vertical (>75°)


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
    id: int = -1  # Plane ID for feature extraction


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
        use_spatial_coherence: bool = True,
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
        logger.info(
            f"  Horizontal: angle ≤{horizontal_angle_max}°, planarity ≥{horizontal_planarity_min}"
        )
        logger.info(
            f"  Vertical: angle ≥{vertical_angle_min}°, planarity ≥{vertical_planarity_min}"
        )
        logger.info(
            f"  Inclined: {inclined_angle_min}° ≤ angle ≤ {inclined_angle_max}°"
        )

    def detect_horizontal_planes(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        planarity: np.ndarray,
        height: Optional[np.ndarray] = None,
        min_height: float = 2.0,
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
        horizontal_mask = (angle_from_horizontal <= self.horizontal_angle_max) & (
            planarity >= self.horizontal_planarity_min
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
            PlaneType.HORIZONTAL,
        )

        logger.info(f"  Segmented into {len(planes)} horizontal plane(s)")
        return planes

    def detect_vertical_planes(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        planarity: np.ndarray,
        height: Optional[np.ndarray] = None,
        min_height: float = 0.5,
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
        vertical_mask = (angle_from_horizontal >= self.vertical_angle_min) & (
            planarity >= self.vertical_planarity_min
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
            PlaneType.VERTICAL,
        )

        logger.info(f"  Segmented into {len(planes)} vertical plane(s) (walls)")
        return planes

    def detect_inclined_planes(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        planarity: np.ndarray,
        height: Optional[np.ndarray] = None,
        min_height: float = 2.0,
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
            (angle_from_horizontal >= self.inclined_angle_min)
            & (angle_from_horizontal <= self.inclined_angle_max)
            & (planarity >= self.inclined_planarity_min)
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
            PlaneType.INCLINED,
        )

        logger.info(f"  Segmented into {len(planes)} inclined plane(s) (roof facets)")
        return planes

    def detect_all_planes(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        planarity: np.ndarray,
        height: Optional[np.ndarray] = None,
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
            ),
        }

        # Summary statistics
        total_planes = sum(len(planes) for planes in results.values())
        total_points = sum(
            sum(p.n_points for p in planes) for planes in results.values()
        )

        logger.info(f"Plane detection complete:")
        logger.info(f"  Total planes: {total_planes}")
        logger.info(f"  Total points: {total_points:,}")
        logger.info(f"  Horizontal: {len(results[PlaneType.HORIZONTAL])} planes")
        logger.info(f"  Vertical: {len(results[PlaneType.VERTICAL])} planes")
        logger.info(f"  Inclined: {len(results[PlaneType.INCLINED])} planes")

        return results

    def classify_roof_types(
        self, horizontal_planes: List[PlaneSegment], inclined_planes: List[PlaneSegment]
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
        roof_classification = {"flat": [], "gable": [], "hip": [], "complex": []}

        n_horizontal = len(horizontal_planes)
        n_inclined = len(inclined_planes)

        if n_horizontal > 0 and n_inclined == 0:
            # Pure flat roof
            roof_classification["flat"] = horizontal_planes
            logger.info("Classified as FLAT ROOF")

        elif n_inclined == 2:
            # Likely gable roof (two pitched sides)
            roof_classification["gable"] = inclined_planes
            logger.info("Classified as GABLE ROOF (2 inclined planes)")

        elif n_inclined >= 4:
            # Hip roof (four or more sides)
            roof_classification["hip"] = inclined_planes
            logger.info(f"Classified as HIP ROOF ({n_inclined} inclined planes)")

        elif n_horizontal > 0 and n_inclined > 0:
            # Complex roof with mixed elements
            roof_classification["complex"] = horizontal_planes + inclined_planes
            logger.info(
                f"Classified as COMPLEX ROOF ({n_horizontal} horizontal, {n_inclined} inclined)"
            )

        return roof_classification

    def _segment_planes(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        planarity: np.ndarray,
        original_indices: np.ndarray,
        plane_type: PlaneType,
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

        # Use spatial coherence if enabled
        if self.use_spatial_coherence:
            return self._segment_with_region_growing(
                points, normals, planarity, original_indices, plane_type
            )

        # Fallback: treat all points as one plane
        return self._create_single_plane_segment(
            points, normals, planarity, original_indices, plane_type
        )

    def _segment_with_region_growing(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        planarity: np.ndarray,
        original_indices: np.ndarray,
        plane_type: PlaneType,
    ) -> List[PlaneSegment]:
        """
        Segment planes using DBSCAN spatial clustering and normal
        similarity.

        Args:
            points: Point coordinates [M, 3]
            normals: Normal vectors [M, 3]
            planarity: Planarity values [M]
            original_indices: Original indices [M]
            plane_type: Type of plane being segmented

        Returns:
            List of PlaneSegment objects
        """
        from sklearn.cluster import DBSCAN

        # Step 1: Spatial clustering with DBSCAN
        # Adjust eps based on plane type
        if plane_type == PlaneType.HORIZONTAL:
            eps = 1.0  # Horizontal planes can be larger
        elif plane_type == PlaneType.VERTICAL:
            eps = 0.8  # Vertical planes (walls) more compact
        else:
            eps = 1.2  # Inclined roofs can be large

        clustering = DBSCAN(
            eps=eps, min_samples=max(10, self.min_points_per_plane // 5)
        ).fit(points)

        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters == 0:
            logger.debug(f"No {plane_type} clusters found")
            return []

        logger.debug(f"Found {n_clusters} {plane_type} clusters")

        # Step 2: Refine clusters using normal similarity
        plane_segments = []

        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_points = points[cluster_mask]
            cluster_normals = normals[cluster_mask]
            cluster_planarity = planarity[cluster_mask]
            cluster_original_indices = original_indices[cluster_mask]

            if len(cluster_points) < self.min_points_per_plane:
                continue

            # Check normal coherence within cluster
            mean_normal = cluster_normals.mean(axis=0)
            mean_normal = mean_normal / (np.linalg.norm(mean_normal) + 1e-8)

            # Compute dot products (normal similarity)
            dot_products = np.abs(np.dot(cluster_normals, mean_normal))

            # Filter by normal similarity (>0.9 means <~25° deviation)
            normal_threshold = (
                0.9
                if plane_type in [PlaneType.HORIZONTAL, PlaneType.VERTICAL]
                else 0.85
            )

            coherent_mask = dot_products >= normal_threshold
            coherent_points = cluster_points[coherent_mask]
            coherent_normals = cluster_normals[coherent_mask]
            coherent_planarity = cluster_planarity[coherent_mask]
            coherent_indices = cluster_original_indices[coherent_mask]

            if len(coherent_points) < self.min_points_per_plane:
                continue

            # Create plane segment
            plane_segment = self._create_plane_segment(
                coherent_points,
                coherent_normals,
                coherent_planarity,
                coherent_indices,
                plane_type,
                segment_id=cluster_id,
            )

            plane_segments.append(plane_segment)

        logger.info(
            f"Segmented {len(plane_segments)} {plane_type} planes "
            f"from {len(points)} points"
        )

        return plane_segments

    def _create_single_plane_segment(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        planarity: np.ndarray,
        original_indices: np.ndarray,
        plane_type: PlaneType,
    ) -> List[PlaneSegment]:
        """
        Create a single plane segment from all points (fallback).

        Args:
            points: Point coordinates [M, 3]
            normals: Normal vectors [M, 3]
            planarity: Planarity values [M]
            original_indices: Original indices [M]
            plane_type: Type of plane

        Returns:
            List with single PlaneSegment
        """
        plane = self._create_plane_segment(
            points, normals, planarity, original_indices, plane_type, segment_id=0
        )
        return [plane]

    def _create_plane_segment(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        planarity: np.ndarray,
        original_indices: np.ndarray,
        plane_type: PlaneType,
        segment_id: int = -1,
    ) -> PlaneSegment:
        """
        Create a PlaneSegment object from point data.

        Args:
            points: Point coordinates [M, 3]
            normals: Normal vectors [M, 3]
            planarity: Planarity values [M]
            original_indices: Original indices [M]
            plane_type: Type of plane
            segment_id: Segment ID for tracking

        Returns:
            PlaneSegment object
        """
        # Compute plane properties
        centroid = points.mean(axis=0)
        mean_normal = normals.mean(axis=0)
        mean_normal = mean_normal / (np.linalg.norm(mean_normal) + 1e-8)
        mean_planarity = planarity.mean()

        # Compute orientation angle
        nz_abs = abs(mean_normal[2])
        orientation_angle = np.degrees(np.arccos(np.clip(nz_abs, 0, 1)))

        # Estimate area (bounding box approximation)
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
            n_points=len(points),
            id=segment_id,
        )

        return plane


def detect_architectural_elements(
    points: np.ndarray,
    normals: np.ndarray,
    planarity: np.ndarray,
    height: np.ndarray,
    planes: Dict[PlaneType, List[PlaneSegment]],
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
    elements = {"balconies": [], "chimneys": [], "dormers": [], "parapets": []}

    # Detect balconies: small horizontal planes at intermediate heights
    horizontal_planes = planes.get(PlaneType.HORIZONTAL, [])
    for plane in horizontal_planes:
        if 2.0 < plane.height_mean < 15.0 and plane.area < 20.0:  # Small area
            if plane.n_points < 500:  # Not a full roof
                elements["balconies"].append(plane.point_indices)
                logger.debug(
                    f"Detected balcony: {plane.n_points} points at {plane.height_mean:.1f}m"
                )

    # Detect chimneys: small vertical structures on roofs
    vertical_planes = planes.get(PlaneType.VERTICAL, [])
    for plane in vertical_planes:
        if plane.height_mean > 8.0 and plane.area < 10.0:  # High, small area
            if plane.n_points < 300:  # Small structure
                elements["chimneys"].append(plane.point_indices)
                logger.debug(
                    f"Detected chimney: {plane.n_points} points at {plane.height_mean:.1f}m"
                )

    # Detect dormers: vertical planes above inclined roofs
    inclined_planes = planes.get(PlaneType.INCLINED, [])
    if inclined_planes and vertical_planes:
        for v_plane in vertical_planes:
            for i_plane in inclined_planes:
                # Check if vertical plane is above inclined plane
                if v_plane.height_mean > i_plane.height_mean + 1.0:
                    elements["dormers"].append(v_plane.point_indices)
                    logger.debug(f"Detected dormer: {v_plane.n_points} points")
                    break

    # Detect parapets: low vertical structures on flat roofs
    if horizontal_planes and vertical_planes:
        for h_plane in horizontal_planes:
            for v_plane in vertical_planes:
                # Check if vertical plane is just above horizontal (parapet)
                height_diff = abs(v_plane.height_mean - h_plane.height_mean)
                if 0.5 < height_diff < 2.0 and v_plane.n_points < 200:
                    elements["parapets"].append(v_plane.point_indices)
                    logger.debug(f"Detected parapet: {v_plane.n_points} points")

    # Log summary
    for elem_type, elem_list in elements.items():
        if elem_list:
            total_points = sum(len(indices) for indices in elem_list)
            logger.info(
                f"Detected {len(elem_list)} {elem_type} ({total_points:,} points)"
            )

    return elements


class PlaneFeatureExtractor:
    """
    Extract plane-based features for each point in the point cloud.

    This class assigns plane-based features to individual points, enabling
    ML models to learn from plane geometry and spatial relationships.

    Features extracted:
    - plane_id: ID of nearest plane (-1 if no plane assigned)
    - plane_type: Type of plane (horizontal=0, vertical=1, inclined=2, none=-1)
    - distance_to_plane: Perpendicular distance to plane surface (meters)
    - plane_area: Area of containing plane (m²)
    - plane_orientation: Angle of plane normal from horizontal (degrees)
    - plane_planarity: Planarity score of plane [0,1]
    - position_on_plane_u: Normalized U coordinate on plane [0,1]
    - position_on_plane_v: Normalized V coordinate on plane [0,1]

    Usage:
        >>> detector = PlaneDetector()
        >>> extractor = PlaneFeatureExtractor(detector)
        >>> features = extractor.detect_and_assign_planes(
        ...     points, normals, planarity, height
        ... )
        >>> print(features['plane_id'])  # Array of plane IDs per point
    """

    def __init__(self, plane_detector: PlaneDetector):
        """
        Initialize plane feature extractor.

        Args:
            plane_detector: Configured PlaneDetector instance
        """
        self.plane_detector = plane_detector
        self.planes = []
        self.plane_type_map = {
            PlaneType.HORIZONTAL: 0,
            PlaneType.NEAR_HORIZONTAL: 0,
            PlaneType.VERTICAL: 1,
            PlaneType.NEAR_VERTICAL: 1,
            PlaneType.INCLINED: 2,
        }

    def detect_and_assign_planes(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        planarity: np.ndarray,
        height: Optional[np.ndarray] = None,
        max_assignment_distance: float = 0.5,
    ) -> Dict[str, np.ndarray]:
        """
        Detect planes and assign plane-based features to each point.

        Args:
            points: Point coordinates [N, 3] (X, Y, Z)
            normals: Surface normals [N, 3]
            planarity: Planarity values [N] in [0, 1]
            height: Height above ground [N] (optional)
            max_assignment_distance: Maximum distance to assign point to plane (meters)

        Returns:
            Dictionary with plane-based feature arrays:
            - plane_id [N]: Plane ID (-1 if not assigned)
            - plane_type [N]: Plane type (0=horizontal, 1=vertical, 2=inclined, -1=none)
            - distance_to_plane [N]: Distance to plane (meters, inf if not assigned)
            - plane_area [N]: Area of plane (m², 0 if not assigned)
            - plane_orientation [N]: Plane angle from horizontal (degrees, 0 if not assigned)
            - plane_planarity [N]: Planarity of plane [0,1] (0 if not assigned)
            - position_on_plane_u [N]: U coordinate on plane [0,1]
            - position_on_plane_v [N]: V coordinate on plane [0,1]
        """
        logger.info("🔷 Extracting plane-based features...")

        # 1. Detect all planes
        planes_dict = self.plane_detector.detect_all_planes(
            points, normals, planarity, height
        )

        # Flatten planes into single list with IDs
        all_planes = []
        plane_id = 0
        for plane_type, plane_list in planes_dict.items():
            for plane in plane_list:
                # Add ID to plane segment
                plane.id = plane_id
                all_planes.append(plane)
                plane_id += 1

        self.planes = all_planes
        n_points = len(points)

        logger.info(f"   Detected {len(all_planes)} planes total")

        # Initialize feature arrays
        features = {
            "plane_id": np.full(n_points, -1, dtype=np.int32),
            "plane_type": np.full(n_points, -1, dtype=np.int8),
            "distance_to_plane": np.full(n_points, np.inf, dtype=np.float32),
            "plane_area": np.zeros(n_points, dtype=np.float32),
            "plane_orientation": np.zeros(n_points, dtype=np.float32),
            "plane_planarity": np.zeros(n_points, dtype=np.float32),
            "position_on_plane_u": np.zeros(n_points, dtype=np.float32),
            "position_on_plane_v": np.zeros(n_points, dtype=np.float32),
        }

        if len(all_planes) == 0:
            logger.warning("   No planes detected - all features set to default values")
            return features

        # 2. For each plane, compute distance of all points to plane
        for plane in all_planes:
            # Plane equation: ax + by + cz + d = 0
            # where (a, b, c) = normal vector, d = -normal · centroid
            a, b, c = plane.normal
            d = -np.dot(plane.normal, plane.centroid)

            # Signed distance from each point to plane
            # Distance = |ax + by + cz + d| / sqrt(a² + b² + c²)
            # Since normal is unit vector, denominator = 1
            signed_distances = (
                a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
            )
            distances = np.abs(signed_distances)

            # Find points closer to this plane than previously assigned planes
            closer_mask = distances < features["distance_to_plane"]

            # Also apply maximum distance threshold
            within_threshold = distances <= max_assignment_distance
            update_mask = closer_mask & within_threshold

            if not np.any(update_mask):
                continue

            # Update features for points closer to this plane
            features["plane_id"][update_mask] = plane.id
            features["plane_type"][update_mask] = self.plane_type_map.get(
                plane.plane_type, -1
            )
            features["distance_to_plane"][update_mask] = distances[update_mask]
            features["plane_area"][update_mask] = plane.area
            features["plane_orientation"][update_mask] = plane.orientation_angle
            features["plane_planarity"][update_mask] = plane.planarity

        # 3. Compute normalized position on plane (UV coordinates)
        for plane in all_planes:
            plane_mask = features["plane_id"] == plane.id
            if not np.any(plane_mask):
                continue

            plane_points = points[plane_mask]

            # Project points onto plane coordinate system
            u_coords, v_coords = self._project_to_plane_coords(
                plane_points, plane.centroid, plane.normal
            )

            features["position_on_plane_u"][plane_mask] = u_coords
            features["position_on_plane_v"][plane_mask] = v_coords

        # Statistics
        n_assigned = (features["plane_id"] >= 0).sum()
        pct_assigned = 100.0 * n_assigned / n_points if n_points > 0 else 0.0

        logger.info(
            f"   Assigned {n_assigned:,} / {n_points:,} points to planes ({pct_assigned:.1f}%)"
        )

        # Per-type statistics
        for plane_type_name, plane_type_val in [
            ("horizontal", 0),
            ("vertical", 1),
            ("inclined", 2),
        ]:
            n_type = (features["plane_type"] == plane_type_val).sum()
            if n_type > 0:
                logger.info(f"      {plane_type_name}: {n_type:,} points")

        return features

    def _project_to_plane_coords(
        self, points: np.ndarray, plane_centroid: np.ndarray, plane_normal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project points onto plane local coordinate system (U, V).

        Creates a 2D coordinate system on the plane where:
        - U axis: perpendicular to normal in XY plane (or X axis if normal is vertical)
        - V axis: perpendicular to both normal and U

        Coordinates are normalized to [0, 1] within plane bounds.

        Args:
            points: Point coordinates [M, 3]
            plane_centroid: Plane center [3]
            plane_normal: Plane normal vector [3]

        Returns:
            Tuple of (u_coords, v_coords), each [M] in range [0, 1]
        """
        # Translate points to plane centroid
        centered = points - plane_centroid

        # Create plane coordinate system (U, V axes)
        # V axis: perpendicular to normal in XY plane
        if abs(plane_normal[2]) > 0.9:  # Near-horizontal plane
            # Use X axis as reference
            v_axis = np.array([1.0, 0.0, 0.0])
        else:
            # Cross product with Z axis gives horizontal direction
            v_axis = np.array([-plane_normal[1], plane_normal[0], 0.0])
            v_norm = np.linalg.norm(v_axis)
            if v_norm > 1e-6:
                v_axis = v_axis / v_norm
            else:
                v_axis = np.array([1.0, 0.0, 0.0])

        # U axis: perpendicular to both normal and V
        u_axis = np.cross(plane_normal, v_axis)
        u_norm = np.linalg.norm(u_axis)
        if u_norm > 1e-6:
            u_axis = u_axis / u_norm
        else:
            u_axis = np.array([0.0, 1.0, 0.0])

        # Project onto axes
        u_coords = centered @ u_axis
        v_coords = centered @ v_axis

        # Normalize to [0, 1]
        u_min, u_max = u_coords.min(), u_coords.max()
        v_min, v_max = v_coords.min(), v_coords.max()

        if u_max > u_min:
            u_coords = (u_coords - u_min) / (u_max - u_min)
        else:
            u_coords = np.full_like(u_coords, 0.5)

        if v_max > v_min:
            v_coords = (v_coords - v_min) / (v_max - v_min)
        else:
            v_coords = np.full_like(v_coords, 0.5)

        return u_coords, v_coords

    def get_plane_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about detected planes.

        Returns:
            Dictionary with plane statistics:
            - n_planes: Total number of planes
            - n_horizontal: Number of horizontal planes
            - n_vertical: Number of vertical planes
            - n_inclined: Number of inclined planes
            - total_area: Total area of all planes (m²)
            - avg_planarity: Average planarity of planes
        """
        if not self.planes:
            return {
                "n_planes": 0,
                "n_horizontal": 0,
                "n_vertical": 0,
                "n_inclined": 0,
                "total_area": 0.0,
                "avg_planarity": 0.0,
            }

        n_horizontal = sum(
            1
            for p in self.planes
            if p.plane_type in [PlaneType.HORIZONTAL, PlaneType.NEAR_HORIZONTAL]
        )
        n_vertical = sum(
            1
            for p in self.planes
            if p.plane_type in [PlaneType.VERTICAL, PlaneType.NEAR_VERTICAL]
        )
        n_inclined = sum(1 for p in self.planes if p.plane_type == PlaneType.INCLINED)

        total_area = sum(p.area for p in self.planes)
        avg_planarity = np.mean([p.planarity for p in self.planes])

        return {
            "n_planes": len(self.planes),
            "n_horizontal": n_horizontal,
            "n_vertical": n_vertical,
            "n_inclined": n_inclined,
            "total_area": total_area,
            "avg_planarity": avg_planarity,
        }


__all__ = [
    "PlaneType",
    "PlaneSegment",
    "PlaneDetector",
    "PlaneFeatureExtractor",
    "detect_architectural_elements",
]
