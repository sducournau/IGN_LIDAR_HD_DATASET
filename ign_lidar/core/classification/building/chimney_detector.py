"""
Chimney and Superstructure Detection for LOD3 Building Classification.

This module detects and classifies vertical superstructures on building roofs,
including chimneys, ventilation structures, and antennas. It uses geometric
analysis of point clouds to identify protrusions above the main roof surface.

The detection pipeline:
1. Identify roof surface and compute plane equations
2. Calculate height above roof for all building points
3. Detect vertical clusters protruding above roof
4. Classify superstructure types based on geometry
5. Assign ASPRS classification codes

Author: IGN LiDAR HD Processing Library
Version: 3.2.0 (Phase 2.2)
Date: January 2025
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


class SuperstructureType(Enum):
    """Types of roof superstructures that can be detected."""

    CHIMNEY = "chimney"
    ANTENNA = "antenna"
    VENTILATION = "ventilation"
    UNKNOWN = "unknown"


@dataclass
class SuperstructureSegment:
    """
    Represents a detected superstructure on a building roof.

    Attributes:
        type: Classified superstructure type
        points_mask: Boolean mask of points belonging to this superstructure
        centroid: 3D centroid of the superstructure [x, y, z]
        height_above_roof: Average height above roof surface (meters)
        max_height_above_roof: Maximum height above roof surface (meters)
        volume: Approximate volume (cubic meters)
        base_area: Base area in horizontal plane (square meters)
        aspect_ratio: Height to base diameter ratio
        verticality: Average verticality score (0-1, 1=perfectly vertical)
        point_count: Number of points in this superstructure
        confidence: Detection confidence score (0-1)
    """

    type: SuperstructureType
    points_mask: np.ndarray
    centroid: np.ndarray
    height_above_roof: float
    max_height_above_roof: float
    volume: float
    base_area: float
    aspect_ratio: float
    verticality: float
    point_count: int
    confidence: float


@dataclass
class ChimneyDetectionResult:
    """
    Complete result of chimney and superstructure detection.

    Attributes:
        superstructures: List of detected superstructure segments
        chimney_indices: Indices of points classified as chimneys
        antenna_indices: Indices of points classified as antennas
        ventilation_indices: Indices of points classified as ventilation
        roof_plane_normal: Normal vector of main roof plane [nx, ny, nz]
        roof_plane_d: Plane equation parameter (distance from origin)
        detection_success: Whether detection completed successfully
        num_chimneys: Number of chimneys detected
        num_antennas: Number of antennas detected
        num_ventilations: Number of ventilation structures detected
    """

    superstructures: List[SuperstructureSegment]
    chimney_indices: np.ndarray
    antenna_indices: np.ndarray
    ventilation_indices: np.ndarray
    roof_plane_normal: Optional[np.ndarray]
    roof_plane_d: Optional[float]
    detection_success: bool
    num_chimneys: int
    num_antennas: int
    num_ventilations: int


class ChimneyDetector:
    """
    Detector for chimneys and other vertical superstructures on building roofs.

    This class identifies vertical protrusions above roof surfaces using geometric
    analysis. It computes height-above-roof features and clusters vertical elements.

    The detection algorithm:
    1. Identify roof points using verticality and elevation
    2. Fit robust plane to roof surface (RANSAC)
    3. Compute height above roof for all building points
    4. Detect vertical clusters above threshold height
    5. Classify superstructure types based on geometry
    6. Filter false positives using confidence scoring

    Key Parameters:
        min_height_above_roof: Minimum height above roof to be considered (meters)
        min_chimney_points: Minimum points required for valid chimney
        max_chimney_diameter: Maximum horizontal diameter for chimney (meters)
        verticality_threshold: Minimum verticality for superstructure points
        dbscan_eps: Clustering distance threshold (meters)
        dbscan_min_samples: Minimum samples per cluster

    Usage:
        >>> detector = ChimneyDetector(
        ...     min_height_above_roof=1.0,
        ...     min_chimney_points=20
        ... )
        >>> result = detector.detect_superstructures(
        ...     points=building_points,
        ...     features=computed_features,
        ...     roof_indices=known_roof_points
        ... )
        >>> print(f"Detected {result.num_chimneys} chimneys")
    """

    def __init__(
        self,
        min_height_above_roof: float = 1.0,
        min_chimney_points: int = 20,
        max_chimney_diameter: float = 3.0,
        verticality_threshold: float = 0.6,
        dbscan_eps: float = 0.5,
        dbscan_min_samples: int = 10,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize chimney detector with detection parameters.

        Args:
            min_height_above_roof: Minimum height above roof (meters), default 1.0m
            min_chimney_points: Minimum points for valid chimney, default 20
            max_chimney_diameter: Maximum chimney diameter (meters), default 3.0m
            verticality_threshold: Minimum verticality score, default 0.6
            dbscan_eps: Clustering distance threshold (meters), default 0.5m
            dbscan_min_samples: Minimum samples per cluster, default 10
            confidence_threshold: Minimum confidence for detection, default 0.5
        """
        self.min_height_above_roof = min_height_above_roof
        self.min_chimney_points = min_chimney_points
        self.max_chimney_diameter = max_chimney_diameter
        self.verticality_threshold = verticality_threshold
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.confidence_threshold = confidence_threshold

        logger.debug(
            f"ChimneyDetector initialized: "
            f"min_height={min_height_above_roof}m, "
            f"min_points={min_chimney_points}, "
            f"max_diameter={max_chimney_diameter}m"
        )

    def detect_superstructures(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        roof_indices: Optional[np.ndarray] = None,
    ) -> ChimneyDetectionResult:
        """
        Detect chimneys and superstructures on a building roof.

        Args:
            points: Building point cloud [N, 3] with XYZ coordinates
            features: Dictionary of computed features (normals, verticality, etc.)
            roof_indices: Optional pre-identified roof point indices

        Returns:
            ChimneyDetectionResult with detected superstructures

        Raises:
            ValueError: If input validation fails
        """
        # Validate inputs
        if points.shape[0] == 0:
            logger.warning("Empty point cloud provided, returning empty result")
            return self._empty_result()

        if "verticality" not in features:
            logger.warning("Verticality feature missing, cannot detect chimneys")
            return self._empty_result()

        try:
            # Step 1: Identify roof surface
            if roof_indices is None:
                roof_indices = self._identify_roof_points(points, features)

            if len(roof_indices) < 50:
                logger.warning(
                    f"Insufficient roof points ({len(roof_indices)}), "
                    "cannot fit plane"
                )
                return self._empty_result()

            # Step 2: Fit robust plane to roof
            roof_normal, roof_d = self._fit_roof_plane(points[roof_indices])

            if roof_normal is None:
                logger.warning("Failed to fit roof plane")
                return self._empty_result()

            # Step 3: Compute height above roof for all points
            if roof_normal is not None and roof_d is not None:
                height_above_roof = self._compute_height_above_roof(
                    points, roof_normal, roof_d
                )
            else:
                return self._empty_result()

            # Step 4: Detect vertical protrusions
            candidate_mask = self._detect_protrusions(
                points, features, height_above_roof
            )

            if np.sum(candidate_mask) < self.min_chimney_points:
                logger.debug("No significant vertical protrusions detected")
                return ChimneyDetectionResult(
                    superstructures=[],
                    chimney_indices=np.array([], dtype=int),
                    antenna_indices=np.array([], dtype=int),
                    ventilation_indices=np.array([], dtype=int),
                    roof_plane_normal=roof_normal,
                    roof_plane_d=roof_d,
                    detection_success=True,
                    num_chimneys=0,
                    num_antennas=0,
                    num_ventilations=0,
                )

            # Step 5: Cluster protrusions into distinct superstructures
            superstructures = self._cluster_and_classify_protrusions(
                points, features, candidate_mask, height_above_roof
            )

            # Step 6: Separate by type
            chimney_indices = []
            antenna_indices = []
            ventilation_indices = []

            for ss in superstructures:
                indices = np.where(ss.points_mask)[0]
                if ss.type == SuperstructureType.CHIMNEY:
                    chimney_indices.extend(indices)
                elif ss.type == SuperstructureType.ANTENNA:
                    antenna_indices.extend(indices)
                elif ss.type == SuperstructureType.VENTILATION:
                    ventilation_indices.extend(indices)

            return ChimneyDetectionResult(
                superstructures=superstructures,
                chimney_indices=np.array(chimney_indices, dtype=int),
                antenna_indices=np.array(antenna_indices, dtype=int),
                ventilation_indices=np.array(ventilation_indices, dtype=int),
                roof_plane_normal=roof_normal,
                roof_plane_d=roof_d,
                detection_success=True,
                num_chimneys=sum(
                    1 for ss in superstructures if ss.type == SuperstructureType.CHIMNEY
                ),
                num_antennas=sum(
                    1 for ss in superstructures if ss.type == SuperstructureType.ANTENNA
                ),
                num_ventilations=sum(
                    1
                    for ss in superstructures
                    if ss.type == SuperstructureType.VENTILATION
                ),
            )

        except Exception as e:
            logger.error(f"Chimney detection failed: {e}", exc_info=True)
            return self._empty_result()

    def _identify_roof_points(
        self, points: np.ndarray, features: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Identify roof points using verticality and elevation.

        Args:
            points: Point cloud [N, 3]
            features: Feature dictionary

        Returns:
            Indices of roof points
        """
        verticality = features["verticality"]

        # Roof points have low verticality (horizontal surfaces)
        horizontal_mask = verticality < 0.3

        # Use upper 75% of points by height
        z_values = points[:, 2]
        height_threshold = np.percentile(z_values[horizontal_mask], 25)
        upper_points_mask = z_values > height_threshold

        roof_mask = horizontal_mask & upper_points_mask
        return np.where(roof_mask)[0]

    def _fit_roof_plane(
        self, roof_points: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Fit a robust plane to roof points using RANSAC-like approach.

        Args:
            roof_points: Roof point cloud [N, 3]

        Returns:
            Tuple of (normal_vector, d_parameter) or (None, None) if fails
        """
        if roof_points.shape[0] < 3:
            return None, None

        try:
            # Simple plane fit using SVD on centered points
            centroid = np.mean(roof_points, axis=0)
            centered = roof_points - centroid

            # SVD: last component is normal to best-fit plane
            _, _, vh = np.linalg.svd(centered)
            normal = vh[2, :]

            # Ensure normal points upward
            if normal[2] < 0:
                normal = -normal

            # Compute d parameter: ax + by + cz = d
            d = np.dot(normal, centroid)

            return normal, d

        except Exception as e:
            logger.warning(f"Plane fitting failed: {e}")
            return None, None

    def _compute_height_above_roof(
        self, points: np.ndarray, roof_normal: np.ndarray, roof_d: float
    ) -> np.ndarray:
        """
        Compute signed distance from each point to roof plane.

        Args:
            points: Point cloud [N, 3]
            roof_normal: Plane normal vector [3]
            roof_d: Plane d parameter

        Returns:
            Height above roof for each point [N]
        """
        # Distance from point to plane: |ax + by + cz - d| / sqrt(a² + b² + c²)
        # Since normal is normalized, denominator is 1
        distances = np.dot(points, roof_normal) - roof_d

        # Positive = above roof, negative = below roof
        return distances

    def _detect_protrusions(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        height_above_roof: np.ndarray,
    ) -> np.ndarray:
        """
        Detect vertical protrusions above roof surface.

        Args:
            points: Point cloud [N, 3]
            features: Feature dictionary
            height_above_roof: Height above roof for each point [N]

        Returns:
            Boolean mask of candidate protrusion points
        """
        verticality = features["verticality"]

        # Criteria for protrusion:
        # 1. Significantly above roof surface
        # 2. High verticality (vertical surfaces)
        # 3. Not too far above roof (likely noise or trees)
        above_roof_mask = height_above_roof > self.min_height_above_roof
        vertical_mask = verticality > self.verticality_threshold
        reasonable_height_mask = height_above_roof < 10.0  # Max 10m above roof

        candidate_mask = above_roof_mask & vertical_mask & reasonable_height_mask
        return candidate_mask

    def _cluster_and_classify_protrusions(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        candidate_mask: np.ndarray,
        height_above_roof: np.ndarray,
    ) -> List[SuperstructureSegment]:
        """
        Cluster protrusion points and classify superstructure types.

        Args:
            points: Point cloud [N, 3]
            features: Feature dictionary
            candidate_mask: Boolean mask of candidate points
            height_above_roof: Height above roof [N]

        Returns:
            List of detected and classified superstructures
        """
        candidate_indices = np.where(candidate_mask)[0]
        candidate_points = points[candidate_indices]

        if len(candidate_points) < self.min_chimney_points:
            return []

        # Cluster using DBSCAN in 3D space
        clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        labels = clustering.fit_predict(candidate_points)

        superstructures = []

        # Process each cluster
        for cluster_id in range(labels.max() + 1):
            cluster_mask_local = labels == cluster_id
            cluster_indices = candidate_indices[cluster_mask_local]

            if len(cluster_indices) < self.min_chimney_points:
                continue

            cluster_points = points[cluster_indices]

            # Compute cluster geometry
            ss = self._classify_superstructure_cluster(
                cluster_indices, cluster_points, features, height_above_roof
            )

            if ss and ss.confidence >= self.confidence_threshold:
                superstructures.append(ss)

        logger.debug(f"Detected {len(superstructures)} superstructures from clusters")
        return superstructures

    def _classify_superstructure_cluster(
        self,
        cluster_indices: np.ndarray,
        cluster_points: np.ndarray,
        features: Dict[str, np.ndarray],
        height_above_roof: np.ndarray,
    ) -> Optional[SuperstructureSegment]:
        """
        Classify a cluster of points as a specific superstructure type.

        Classification logic:
        - Chimney: Moderate height, square/circular base, solid structure
        - Antenna: Very tall, thin, high aspect ratio
        - Ventilation: Low height, moderate base area

        Args:
            cluster_indices: Indices of points in this cluster
            cluster_points: Points in this cluster [M, 3]
            features: Feature dictionary
            height_above_roof: Height above roof [N]

        Returns:
            SuperstructureSegment or None if classification fails
        """
        # Compute geometric properties
        centroid = np.mean(cluster_points, axis=0)
        point_count = len(cluster_indices)

        # Height statistics
        cluster_heights = height_above_roof[cluster_indices]
        avg_height = np.mean(cluster_heights)
        max_height = np.max(cluster_heights)

        # Base area (horizontal extent)
        xy_coords = cluster_points[:, :2]
        xy_min = np.min(xy_coords, axis=0)
        xy_max = np.max(xy_coords, axis=0)
        base_dimensions = xy_max - xy_min
        base_area = np.prod(base_dimensions)
        base_diameter = np.mean(base_dimensions)

        # Aspect ratio
        aspect_ratio = max_height / max(base_diameter, 0.1)

        # Average verticality
        cluster_verticality = features["verticality"][cluster_indices]
        avg_verticality = np.mean(cluster_verticality)

        # Approximate volume (cylinder approximation)
        volume = base_area * max_height

        # Classification rules
        confidence = 0.0
        ss_type = SuperstructureType.UNKNOWN

        # Chimney: 1-5m tall, 0.5-2m diameter, aspect ratio 1.2-5
        if (
            1.0 <= max_height <= 5.0
            and 0.3 <= base_diameter <= 3.0
            and 1.2 <= aspect_ratio <= 7.0
            and avg_verticality > 0.5
        ):
            ss_type = SuperstructureType.CHIMNEY
            confidence = min(
                1.0, 0.6 + 0.2 * (avg_verticality - 0.5) + 0.2 * (point_count / 100.0)
            )

        # Antenna: >3m tall, thin (<1.0m), high aspect ratio (>6)
        elif (
            max_height > 3.0
            and base_diameter < 1.0
            and aspect_ratio > 6.0
            and avg_verticality > 0.7
        ):
            ss_type = SuperstructureType.ANTENNA
            confidence = min(1.0, 0.5 + 0.3 * (aspect_ratio / 15.0) + 0.2 * avg_verticality)

        # Ventilation: 0.5-2m tall, moderate base (0.5-1.5m), low aspect
        elif (
            0.5 <= max_height <= 2.5
            and 0.3 <= base_diameter <= 2.0
            and 0.3 <= aspect_ratio <= 2.5
        ):
            ss_type = SuperstructureType.VENTILATION
            confidence = min(
                1.0, 0.4 + 0.3 * (avg_verticality - 0.4) + 0.3 * (point_count / 50.0)
            )

        else:
            # Unknown type, low confidence
            confidence = 0.3

        # Create full points mask
        full_mask = np.zeros(len(height_above_roof), dtype=bool)
        full_mask[cluster_indices] = True

        return SuperstructureSegment(
            type=ss_type,
            points_mask=full_mask,
            centroid=centroid,
            height_above_roof=float(avg_height),
            max_height_above_roof=float(max_height),
            volume=float(volume),
            base_area=float(base_area),
            aspect_ratio=float(aspect_ratio),
            verticality=float(avg_verticality),
            point_count=int(point_count),
            confidence=float(confidence),
        )

    def _empty_result(self) -> ChimneyDetectionResult:
        """
        Create an empty result for cases where detection fails or finds nothing.

        Returns:
            Empty ChimneyDetectionResult
        """
        return ChimneyDetectionResult(
            superstructures=[],
            chimney_indices=np.array([], dtype=int),
            antenna_indices=np.array([], dtype=int),
            ventilation_indices=np.array([], dtype=int),
            roof_plane_normal=None,
            roof_plane_d=None,
            detection_success=False,
            num_chimneys=0,
            num_antennas=0,
            num_ventilations=0,
        )
