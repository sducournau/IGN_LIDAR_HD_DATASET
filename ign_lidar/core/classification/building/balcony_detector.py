"""
Balcony and Overhang Detection for LOD3 Building Classification.

This module detects and classifies horizontal protrusions from building facades,
including balconies, overhangs, and roof extensions. It uses geometric analysis
to identify structures extending beyond the main building envelope.

The detection pipeline:
1. Identify facade planes and building envelope
2. Calculate horizontal distance from facade for all points
3. Detect horizontal clusters protruding from facade
4. Classify protrusion types based on geometry and position
5. Assign ASPRS classification codes

Author: IGN LiDAR HD Processing Library
Version: 3.3.0 (Phase 2.3)
Date: January 2025
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


class ProtrusionType(Enum):
    """Types of horizontal protrusions that can be detected."""

    BALCONY = "balcony"
    OVERHANG = "overhang"
    CANOPY = "canopy"
    UNKNOWN = "unknown"


@dataclass
class ProtrusionSegment:
    """
    Represents a detected horizontal protrusion from a building facade.

    Attributes:
        type: Classified protrusion type
        points_mask: Boolean mask of points belonging to this protrusion
        centroid: 3D centroid of the protrusion [x, y, z]
        distance_from_facade: Average horizontal distance from facade (m)
        max_distance_from_facade: Maximum distance from facade (m)
        height_above_ground: Average height above ground (meters)
        width: Width along facade (meters)
        depth: Depth perpendicular to facade (meters)
        area: Horizontal area (square meters)
        verticality: Average verticality score (0-1)
        point_count: Number of points in this protrusion
        facade_side: Which facade side (0-3) this protrudes from
        confidence: Detection confidence score (0-1)
    """

    type: ProtrusionType
    points_mask: np.ndarray
    centroid: np.ndarray
    distance_from_facade: float
    max_distance_from_facade: float
    height_above_ground: float
    width: float
    depth: float
    area: float
    verticality: float
    point_count: int
    facade_side: int
    confidence: float


@dataclass
class BalconyDetectionResult:
    """
    Complete result of balcony and overhang detection.

    Attributes:
        protrusions: List of detected protrusion segments
        balcony_indices: Indices of points classified as balconies
        overhang_indices: Indices of points classified as overhangs
        canopy_indices: Indices of points classified as canopies
        facade_lines: List of facade line segments used for detection
        detection_success: Whether detection completed successfully
        num_balconies: Number of balconies detected
        num_overhangs: Number of overhangs detected
        num_canopies: Number of canopies detected
    """

    protrusions: List[ProtrusionSegment]
    balcony_indices: np.ndarray
    overhang_indices: np.ndarray
    canopy_indices: np.ndarray
    facade_lines: List[LineString]
    detection_success: bool
    num_balconies: int
    num_overhangs: int
    num_canopies: int


class BalconyDetector:
    """
    Detector for balconies and horizontal protrusions from building facades.

    This class identifies horizontal structures extending beyond the main
    building envelope using geometric analysis of point clouds relative to
    facade planes.

    The detection algorithm:
    1. Extract building envelope (polygon or OBB)
    2. Identify facade line segments
    3. Compute horizontal distance from facade for each point
    4. Detect horizontal clusters beyond threshold distance
    5. Classify protrusion types based on geometry and position
    6. Filter false positives using confidence scoring

    Key Parameters:
        min_distance_from_facade: Min distance beyond facade (meters)
        min_balcony_points: Minimum points required for valid balcony
        max_balcony_depth: Maximum depth from facade (meters)
        min_height_above_ground: Minimum height for balconies (meters)
        dbscan_eps: Clustering distance threshold (meters)
        dbscan_min_samples: Minimum samples per cluster

    Usage:
        >>> detector = BalconyDetector(
        ...     min_distance_from_facade=0.5,
        ...     min_balcony_points=25
        ... )
        >>> result = detector.detect_protrusions(
        ...     points=building_points,
        ...     features=computed_features,
        ...     building_polygon=polygon,
        ...     ground_elevation=10.0
        ... )
        >>> print(f"Detected {result.num_balconies} balconies")
    """

    def __init__(
        self,
        min_distance_from_facade: float = 0.5,
        min_balcony_points: int = 25,
        max_balcony_depth: float = 3.0,
        min_height_above_ground: float = 2.0,
        max_height_from_roof: float = 2.0,
        dbscan_eps: float = 0.5,
        dbscan_min_samples: int = 15,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize balcony detector with detection parameters.

        Args:
            min_distance_from_facade: Min distance beyond facade (m),
                default 0.5m
            min_balcony_points: Min points for valid balcony, default 25
            max_balcony_depth: Max depth from facade (m), default 3.0m
            min_height_above_ground: Min height for balconies (m),
                default 2.0m
            max_height_from_roof: Max distance below roof (m), default 2.0m
            dbscan_eps: Clustering distance threshold (m), default 0.5m
            dbscan_min_samples: Min samples per cluster, default 15
            confidence_threshold: Min confidence for detection, default 0.5
        """
        self.min_distance_from_facade = min_distance_from_facade
        self.min_balcony_points = min_balcony_points
        self.max_balcony_depth = max_balcony_depth
        self.min_height_above_ground = min_height_above_ground
        self.max_height_from_roof = max_height_from_roof
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.confidence_threshold = confidence_threshold

        logger.debug(
            f"BalconyDetector initialized: "
            f"min_distance={min_distance_from_facade}m, "
            f"min_points={min_balcony_points}, "
            f"max_depth={max_balcony_depth}m"
        )

    def detect_protrusions(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        building_polygon: Polygon,
        ground_elevation: float,
        roof_elevation: Optional[float] = None,
    ) -> BalconyDetectionResult:
        """
        Detect balconies and horizontal protrusions from building facades.

        Args:
            points: Building point cloud [N, 3] with XYZ coordinates
            features: Dictionary of computed features (verticality, etc.)
            building_polygon: Building footprint polygon
            ground_elevation: Ground level elevation (meters)
            roof_elevation: Optional roof level elevation (meters)

        Returns:
            BalconyDetectionResult with detected protrusions

        Raises:
            ValueError: If input validation fails
        """
        # Validate inputs
        if points.shape[0] == 0:
            logger.warning("Empty point cloud provided")
            return self._empty_result()

        if "verticality" not in features:
            logger.warning("Verticality feature missing")
            return self._empty_result()

        try:
            # Step 1: Extract facade line segments from building polygon
            facade_lines = self._extract_facade_lines(building_polygon)

            if len(facade_lines) == 0:
                logger.warning("No facade lines extracted from polygon")
                return self._empty_result()

            # Step 2: Compute distance from facade for each point
            (
                distances_from_facade,
                closest_facade_indices,
            ) = self._compute_distance_from_facades(points, facade_lines)

            # Step 3: Compute height above ground
            heights_above_ground = points[:, 2] - ground_elevation

            # Step 4: Detect candidate protrusion points
            candidate_mask = self._detect_candidates(
                points,
                features,
                distances_from_facade,
                heights_above_ground,
                roof_elevation,
            )

            if np.sum(candidate_mask) < self.min_balcony_points:
                logger.debug("No significant horizontal protrusions detected")
                return BalconyDetectionResult(
                    protrusions=[],
                    balcony_indices=np.array([], dtype=int),
                    overhang_indices=np.array([], dtype=int),
                    canopy_indices=np.array([], dtype=int),
                    facade_lines=facade_lines,
                    detection_success=True,
                    num_balconies=0,
                    num_overhangs=0,
                    num_canopies=0,
                )

            # Step 5: Cluster protrusions
            protrusions = self._cluster_and_classify_protrusions(
                points,
                features,
                candidate_mask,
                distances_from_facade,
                heights_above_ground,
                closest_facade_indices,
            )

            # Step 6: Separate by type
            balcony_indices = []
            overhang_indices = []
            canopy_indices = []

            for prot in protrusions:
                indices = np.where(prot.points_mask)[0]
                if prot.type == ProtrusionType.BALCONY:
                    balcony_indices.extend(indices)
                elif prot.type == ProtrusionType.OVERHANG:
                    overhang_indices.extend(indices)
                elif prot.type == ProtrusionType.CANOPY:
                    canopy_indices.extend(indices)

            return BalconyDetectionResult(
                protrusions=protrusions,
                balcony_indices=np.array(balcony_indices, dtype=int),
                overhang_indices=np.array(overhang_indices, dtype=int),
                canopy_indices=np.array(canopy_indices, dtype=int),
                facade_lines=facade_lines,
                detection_success=True,
                num_balconies=sum(
                    1 for p in protrusions if p.type == ProtrusionType.BALCONY
                ),
                num_overhangs=sum(
                    1 for p in protrusions if p.type == ProtrusionType.OVERHANG
                ),
                num_canopies=sum(
                    1 for p in protrusions if p.type == ProtrusionType.CANOPY
                ),
            )

        except Exception as e:
            logger.error(f"Balcony detection failed: {e}", exc_info=True)
            return self._empty_result()

    def _extract_facade_lines(self, building_polygon: Polygon) -> List[LineString]:
        """
        Extract facade line segments from building polygon.

        Args:
            building_polygon: Building footprint polygon

        Returns:
            List of LineString objects representing facade segments
        """
        facade_lines = []

        # Get exterior coordinates
        coords = list(building_polygon.exterior.coords)

        # Create line segments for each edge
        for i in range(len(coords) - 1):
            line = LineString([coords[i], coords[i + 1]])
            facade_lines.append(line)

        return facade_lines

    def _compute_distance_from_facades(
        self, points: np.ndarray, facade_lines: List[LineString]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute horizontal distance from each point to nearest facade.

        Args:
            points: Point cloud [N, 3]
            facade_lines: List of facade line segments

        Returns:
            Tuple of (distances[N], closest_facade_index[N])
        """
        n_points = points.shape[0]
        distances = np.full(n_points, np.inf)
        closest_facade_indices = np.zeros(n_points, dtype=int)

        # For each point, find distance to closest facade
        for point_idx in range(n_points):
            point_xy = Point(points[point_idx, 0], points[point_idx, 1])

            for facade_idx, facade_line in enumerate(facade_lines):
                dist = facade_line.distance(point_xy)

                if dist < distances[point_idx]:
                    distances[point_idx] = dist
                    closest_facade_indices[point_idx] = facade_idx

        return distances, closest_facade_indices

    def _detect_candidates(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        distances_from_facade: np.ndarray,
        heights_above_ground: np.ndarray,
        roof_elevation: Optional[float],
    ) -> np.ndarray:
        """
        Detect candidate protrusion points using multiple criteria.

        Args:
            points: Point cloud [N, 3]
            features: Feature dictionary
            distances_from_facade: Distance to nearest facade [N]
            heights_above_ground: Height above ground [N]
            roof_elevation: Optional roof elevation

        Returns:
            Boolean mask of candidate points
        """
        verticality = features["verticality"]

        # Ensure all arrays have same length
        n_points = len(points)
        if len(verticality) != n_points:
            logger.error(
                f"Verticality length ({len(verticality)}) != points length ({n_points})"
            )
            return np.zeros(n_points, dtype=bool)
        if len(distances_from_facade) != n_points:
            logger.error(
                f"Distances length ({len(distances_from_facade)}) != points length ({n_points})"
            )
            return np.zeros(n_points, dtype=bool)
        if len(heights_above_ground) != n_points:
            logger.error(
                f"Heights length ({len(heights_above_ground)}) != points length ({n_points})"
            )
            return np.zeros(n_points, dtype=bool)

        # Criteria for horizontal protrusions:
        # 1. Beyond facade by minimum distance
        beyond_facade = distances_from_facade > self.min_distance_from_facade

        # 2. Within reasonable depth from facade
        reasonable_depth = distances_from_facade < self.max_balcony_depth

        # 3. Above ground level (not ground)
        above_ground = heights_above_ground > self.min_height_above_ground

        # 4. Low verticality (horizontal surfaces) or moderate (railings)
        horizontal_or_railing = verticality < 0.7

        # 5. Below roof level (if known)
        if roof_elevation is not None:
            below_roof = points[:, 2] < roof_elevation - self.max_height_from_roof
        else:
            below_roof = np.ones(len(points), dtype=bool)

        candidate_mask = (
            beyond_facade
            & reasonable_depth
            & above_ground
            & horizontal_or_railing
            & below_roof
        )

        return candidate_mask

    def _cluster_and_classify_protrusions(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        candidate_mask: np.ndarray,
        distances_from_facade: np.ndarray,
        heights_above_ground: np.ndarray,
        closest_facade_indices: np.ndarray,
    ) -> List[ProtrusionSegment]:
        """
        Cluster candidate points and classify protrusion types.

        Args:
            points: Point cloud [N, 3]
            features: Feature dictionary
            candidate_mask: Boolean mask of candidate points
            distances_from_facade: Distance to facade [N]
            heights_above_ground: Height above ground [N]
            closest_facade_indices: Index of closest facade [N]

        Returns:
            List of detected and classified protrusions
        """
        candidate_indices = np.where(candidate_mask)[0]
        candidate_points = points[candidate_indices]

        if len(candidate_points) < self.min_balcony_points:
            return []

        # Cluster using DBSCAN in 3D space
        clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        labels = clustering.fit_predict(candidate_points)

        protrusions = []

        # Process each cluster
        for cluster_id in range(labels.max() + 1):
            cluster_mask_local = labels == cluster_id
            cluster_indices = candidate_indices[cluster_mask_local]

            if len(cluster_indices) < self.min_balcony_points:
                continue

            cluster_points = points[cluster_indices]

            # Classify this cluster
            prot = self._classify_protrusion_cluster(
                cluster_indices,
                cluster_points,
                features,
                distances_from_facade,
                heights_above_ground,
                closest_facade_indices,
            )

            if prot and prot.confidence >= self.confidence_threshold:
                protrusions.append(prot)

        logger.debug(f"Detected {len(protrusions)} protrusions from clusters")
        return protrusions

    def _classify_protrusion_cluster(
        self,
        cluster_indices: np.ndarray,
        cluster_points: np.ndarray,
        features: Dict[str, np.ndarray],
        distances_from_facade: np.ndarray,
        heights_above_ground: np.ndarray,
        closest_facade_indices: np.ndarray,
    ) -> Optional[ProtrusionSegment]:
        """
        Classify a cluster as balcony, overhang, or canopy.

        Classification logic:
        - Balcony: Mid-height, moderate depth, mixed verticality (floor+rail)
        - Overhang: High elevation, shallow depth, low verticality
        - Canopy: Low-mid height, shallow depth, low verticality

        Args:
            cluster_indices: Indices of points in cluster
            cluster_points: Points in cluster [M, 3]
            features: Feature dictionary
            distances_from_facade: Distance to facade [N]
            heights_above_ground: Height above ground [N]
            closest_facade_indices: Closest facade index [N]

        Returns:
            ProtrusionSegment or None if classification fails
        """
        # Compute geometric properties
        centroid = np.mean(cluster_points, axis=0)
        point_count = len(cluster_indices)

        # Distance and height statistics
        cluster_distances = distances_from_facade[cluster_indices]
        avg_distance = float(np.mean(cluster_distances))
        max_distance = float(np.max(cluster_distances))

        cluster_heights = heights_above_ground[cluster_indices]
        avg_height = float(np.mean(cluster_heights))

        # Dimensions
        xy_coords = cluster_points[:, :2]
        xy_min = np.min(xy_coords, axis=0)
        xy_max = np.max(xy_coords, axis=0)
        dimensions = xy_max - xy_min
        width = float(np.max(dimensions))
        depth = float(max_distance - np.min(cluster_distances))
        area = float(np.prod(dimensions))

        # Verticality
        cluster_verticality = features["verticality"][cluster_indices]
        avg_verticality = float(np.mean(cluster_verticality))

        # Most common facade side
        facade_sides = closest_facade_indices[cluster_indices]
        facade_side = int(np.bincount(facade_sides).argmax())

        # Classification rules
        confidence = 0.0
        prot_type = ProtrusionType.UNKNOWN

        # Balcony: 2-15m height, 0.8-3m depth, area 2-20m², mixed vertical
        if (
            2.0 <= avg_height <= 15.0
            and 0.8 <= depth <= 3.0
            and 2.0 <= area <= 20.0
            and 0.2 <= avg_verticality <= 0.6
        ):
            prot_type = ProtrusionType.BALCONY
            confidence = min(
                1.0,
                0.5
                + 0.2 * (depth / 2.0)
                + 0.2 * (area / 10.0)
                + 0.1 * (point_count / 50.0),
            )

        # Overhang/Eave: High (>8m), shallow (<1.5m), horizontal
        elif avg_height > 8.0 and depth < 1.5 and avg_verticality < 0.3 and area > 1.0:
            prot_type = ProtrusionType.OVERHANG
            confidence = min(
                1.0, 0.6 + 0.2 * (1.0 - avg_verticality) + 0.2 * (area / 5.0)
            )

        # Canopy: Low-mid (2-8m), shallow (<2m), horizontal, larger area
        elif (
            2.0 <= avg_height <= 8.0
            and depth < 2.0
            and avg_verticality < 0.3
            and area > 3.0
        ):
            prot_type = ProtrusionType.CANOPY
            confidence = min(
                1.0, 0.5 + 0.3 * (area / 10.0) + 0.2 * (1.0 - avg_verticality)
            )

        else:
            # Unknown type, low confidence
            confidence = 0.3

        # Create full points mask
        full_mask = np.zeros(len(distances_from_facade), dtype=bool)
        full_mask[cluster_indices] = True

        return ProtrusionSegment(
            type=prot_type,
            points_mask=full_mask,
            centroid=centroid,
            distance_from_facade=avg_distance,
            max_distance_from_facade=max_distance,
            height_above_ground=avg_height,
            width=width,
            depth=depth,
            area=area,
            verticality=avg_verticality,
            point_count=int(point_count),
            facade_side=facade_side,
            confidence=float(confidence),
        )

    def _empty_result(self) -> BalconyDetectionResult:
        """
        Create an empty result for cases where detection fails or nothing.

        Returns:
            Empty BalconyDetectionResult
        """
        return BalconyDetectionResult(
            protrusions=[],
            balcony_indices=np.array([], dtype=int),
            overhang_indices=np.array([], dtype=int),
            canopy_indices=np.array([], dtype=int),
            facade_lines=[],
            detection_success=False,
            num_balconies=0,
            num_overhangs=0,
            num_canopies=0,
        )
