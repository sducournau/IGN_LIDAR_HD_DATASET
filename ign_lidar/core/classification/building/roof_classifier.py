"""
Roof Type Classifier for LOD3 Building Classification.

This module provides advanced roof type detection and classification:
- Flat vs pitched roof detection
- Roof type classification (gabled, hipped, complex)
- Ridge line detection
- Roof edge detection
- Dormer detection

Part of Phase 2 (v3.1) LOD3 enhancements.

Author: IGN LiDAR HD Development Team
Date: October 26, 2025
Version: 3.1.0
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import logging

import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

from ign_lidar.classification_schema import ASPRSClass

logger = logging.getLogger(__name__)


class RoofType(Enum):
    """Roof type enumeration."""

    FLAT = "flat"
    GABLED = "gabled"  # 2 slopes (pignon)
    HIPPED = "hipped"  # 4 slopes
    COMPLEX = "complex"  # Mansard, gambrel, etc.
    UNKNOWN = "unknown"


@dataclass
class RoofSegment:
    """
    Represents a detected roof segment/plane.

    Attributes:
        points: Point indices belonging to this segment
        normal: Normal vector of the plane [nx, ny, nz]
        centroid: Center point of the segment [x, y, z]
        area: Estimated area (m²)
        slope_angle: Slope angle in degrees (0° = flat, 90° = vertical)
        planarity: How planar the segment is (0-1, higher is more planar)
        roof_type: Detected roof type for this segment
    """

    points: np.ndarray
    normal: np.ndarray
    centroid: np.ndarray
    area: float
    slope_angle: float
    planarity: float
    roof_type: RoofType = RoofType.UNKNOWN


@dataclass
class RoofClassificationResult:
    """
    Result of roof classification for a building.

    Attributes:
        roof_type: Overall roof type for the building
        segments: List of detected roof segments/planes
        ridge_lines: Detected ridge line points (indices)
        edge_points: Detected roof edge points (indices)
        dormer_points: Detected dormer points (indices)
        confidence: Confidence score (0-1)
        stats: Additional statistics
    """

    roof_type: RoofType
    segments: List[RoofSegment]
    ridge_lines: np.ndarray
    edge_points: np.ndarray
    dormer_points: np.ndarray
    confidence: float
    stats: Dict[str, Any]


class RoofTypeClassifier:
    """
    Classifier for detecting and classifying roof types in point clouds.

    Uses geometric features (normals, planarity, verticality) to:
    1. Identify roof points
    2. Segment roof into planes
    3. Classify roof type (flat, gabled, hipped, complex)
    4. Detect architectural details (ridges, edges, dormers)

    Example:
        >>> classifier = RoofTypeClassifier(
        ...     flat_threshold=15.0,
        ...     min_plane_points=100
        ... )
        >>> result = classifier.classify_roof(
        ...     points=building_points,
        ...     features=computed_features
        ... )
        >>> print(f"Roof type: {result.roof_type}, "
        ...       f"Confidence: {result.confidence:.2f}")
    """

    def __init__(
        self,
        flat_threshold: float = 15.0,
        pitched_threshold: float = 20.0,
        min_plane_points: int = 100,
        planarity_threshold: float = 0.85,
        verticality_threshold: float = 0.3,
        ridge_curvature_threshold: float = 0.1,
        edge_detection_enabled: bool = True,
        dormer_detection_enabled: bool = True,
    ):
        """
        Initialize roof type classifier.

        Args:
            flat_threshold: Max slope (degrees) for flat roof (default: 15°)
            pitched_threshold: Min slope (degrees) for pitched (default: 20°)
            min_plane_points: Minimum points to form a valid roof plane
            planarity_threshold: Min planarity for valid plane (0-1)
            verticality_threshold: Maximum verticality for roof (vs walls)
            ridge_curvature_threshold: Curvature threshold for ridges
            edge_detection_enabled: Enable roof edge detection
            dormer_detection_enabled: Enable dormer detection
        """
        self.flat_threshold = flat_threshold
        self.pitched_threshold = pitched_threshold
        self.min_plane_points = min_plane_points
        self.planarity_threshold = planarity_threshold
        self.verticality_threshold = verticality_threshold
        self.ridge_curvature_threshold = ridge_curvature_threshold
        self.edge_detection_enabled = edge_detection_enabled
        self.dormer_detection_enabled = dormer_detection_enabled

        logger.info(
            f"RoofTypeClassifier initialized: "
            f"flat<{flat_threshold}°, pitched>{pitched_threshold}°, "
            f"planarity>{planarity_threshold}"
        )

    def classify_roof(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        labels: Optional[np.ndarray] = None,
    ) -> RoofClassificationResult:
        """
        Classify roof type and detect architectural details.

        Args:
            points: Point cloud [N, 3] with XYZ coordinates
            features: Dict of computed features (normals, planarity, etc.)
            labels: Optional existing classification labels

        Returns:
            RoofClassificationResult with detected roof type and details

        Raises:
            ValueError: If required features are missing
        """
        # Validate inputs
        if points.shape[0] == 0:
            return self._empty_result()

        required_features = ["normals", "planarity", "verticality"]
        missing = [f for f in required_features if f not in features]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        # Step 1: Identify roof points
        roof_mask = self._identify_roof_points(points, features, labels)
        if roof_mask.sum() < self.min_plane_points:
            logger.debug(f"Insufficient roof points: {roof_mask.sum()}")
            return self._empty_result()

        roof_points = points[roof_mask]
        roof_normals = features["normals"][roof_mask]
        roof_planarity = features["planarity"][roof_mask]

        # Step 2: Segment roof into planes
        segments = self._segment_roof_planes(roof_points, roof_normals, roof_planarity)

        if not segments:
            logger.debug("No valid roof segments detected")
            return self._empty_result()

        # Step 3: Classify overall roof type
        roof_type, confidence = self._classify_roof_type(segments)

        # Step 4: Detect architectural details
        ridge_indices = (
            self._detect_ridge_lines(roof_points, roof_normals, segments)
            if len(segments) > 1
            else np.array([], dtype=int)
        )

        edge_indices = (
            self._detect_roof_edges(roof_points, features.get("curvature", None))
            if self.edge_detection_enabled
            else np.array([], dtype=int)
        )

        dormer_indices = (
            self._detect_dormers(roof_points, roof_normals, segments)
            if self.dormer_detection_enabled
            else np.array([], dtype=int)
        )

        # Convert local indices back to original point cloud indices
        roof_indices = np.where(roof_mask)[0]
        ridge_lines = (
            roof_indices[ridge_indices]
            if len(ridge_indices) > 0
            else np.array([], dtype=int)
        )
        edge_points = (
            roof_indices[edge_indices]
            if len(edge_indices) > 0
            else np.array([], dtype=int)
        )
        dormer_points = (
            roof_indices[dormer_indices]
            if len(dormer_indices) > 0
            else np.array([], dtype=int)
        )

        # Compile statistics
        stats = {
            "total_roof_points": roof_mask.sum(),
            "num_segments": len(segments),
            "avg_slope": np.mean([s.slope_angle for s in segments]),
            "avg_planarity": np.mean([s.planarity for s in segments]),
            "ridge_points": len(ridge_lines),
            "edge_points": len(edge_points),
            "dormer_points": len(dormer_points),
        }

        return RoofClassificationResult(
            roof_type=roof_type,
            segments=segments,
            ridge_lines=ridge_lines,
            edge_points=edge_points,
            dormer_points=dormer_points,
            confidence=confidence,
            stats=stats,
        )

    def _identify_roof_points(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Identify points belonging to roof surfaces.

        Uses verticality and height to separate roofs from walls/ground.

        Args:
            points: Point cloud [N, 3]
            features: Feature dictionary
            labels: Optional existing labels

        Returns:
            Boolean mask [N] where True indicates roof points
        """
        verticality = features["verticality"]

        # Roof points have low verticality (horizontal surfaces)
        roof_mask = verticality < self.verticality_threshold

        # If labels provided, filter to building-classified points
        if labels is not None:
            building_mask = labels == ASPRSClass.BUILDING
            roof_mask = roof_mask & building_mask

        # Use height to prioritize upper points (roofs vs ground)
        if roof_mask.sum() > 0:
            roof_heights = points[roof_mask, 2]
            height_percentile_75 = np.percentile(points[:, 2], 75)
            upper_mask = np.zeros(len(points), dtype=bool)
            upper_mask[roof_mask] = roof_heights > height_percentile_75
            roof_mask = roof_mask & upper_mask

        return roof_mask

    def _segment_roof_planes(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        planarity: np.ndarray,
    ) -> List[RoofSegment]:
        """
        Segment roof into planar regions using normal-based clustering.

        Args:
            points: Roof points [N, 3]
            normals: Normal vectors [N, 3]
            planarity: Planarity values [N]

        Returns:
            List of detected roof segments
        """
        if len(points) < self.min_plane_points:
            return []

        # Filter to planar points
        planar_mask = planarity > self.planarity_threshold
        if planar_mask.sum() < self.min_plane_points:
            return []

        planar_points = points[planar_mask]
        planar_normals = normals[planar_mask]

        # Cluster by normal direction using DBSCAN
        # Similar normals => same plane
        try:
            clustering = DBSCAN(eps=0.15, min_samples=self.min_plane_points // 2)
            cluster_labels = clustering.fit_predict(planar_normals)
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            return []

        # Extract segments
        segments = []
        for cluster_id in np.unique(cluster_labels):
            if cluster_id == -1:  # Noise
                continue

            cluster_mask = cluster_labels == cluster_id
            if cluster_mask.sum() < self.min_plane_points:
                continue

            segment_points = planar_points[cluster_mask]
            segment_normals = planar_normals[cluster_mask]

            # Compute segment properties
            normal = np.mean(segment_normals, axis=0)
            normal = normal / (np.linalg.norm(normal) + 1e-10)

            centroid = np.mean(segment_points, axis=0)

            # Estimate area (convex hull approximation)
            try:
                from scipy.spatial import ConvexHull

                if len(segment_points) >= 4:  # Minimum for 3D hull
                    hull = ConvexHull(segment_points)
                    area = hull.volume  # In 3D, volume is actually surface area
                else:
                    area = 0.0
            except:
                area = 0.0

            # Compute slope angle (0° = horizontal, 90° = vertical)
            slope_angle = np.degrees(np.arccos(np.abs(normal[2])))

            # Compute segment planarity
            segment_planarity = float(np.mean(planarity[planar_mask][cluster_mask]))

            # Create segment
            segment = RoofSegment(
                points=np.where(planar_mask)[0][cluster_mask],
                normal=normal,
                centroid=centroid,
                area=area,
                slope_angle=slope_angle,
                planarity=segment_planarity,
                roof_type=self._classify_segment_type(slope_angle),
            )

            segments.append(segment)

        return segments

    def _classify_segment_type(self, slope_angle: float) -> RoofType:
        """
        Classify a single segment based on slope angle.

        Args:
            slope_angle: Slope angle in degrees

        Returns:
            RoofType for this segment
        """
        if slope_angle < self.flat_threshold:
            return RoofType.FLAT
        elif slope_angle > self.pitched_threshold:
            return RoofType.GABLED  # Preliminary, refined in overall classification
        else:
            return RoofType.UNKNOWN

    def _classify_roof_type(
        self, segments: List[RoofSegment]
    ) -> Tuple[RoofType, float]:
        """
        Classify overall roof type based on detected segments.

        Logic:
        - 1 flat segment => FLAT
        - 2 pitched segments with opposite slopes => GABLED
        - 3-4 pitched segments => HIPPED
        - >4 segments or mixed => COMPLEX

        Args:
            segments: List of roof segments

        Returns:
            (roof_type, confidence_score)
        """
        if not segments:
            return RoofType.UNKNOWN, 0.0

        # Count segment types
        flat_count = sum(1 for s in segments if s.roof_type == RoofType.FLAT)
        pitched_count = len(segments) - flat_count

        # Calculate avg confidence from planarity
        avg_planarity = np.mean([s.planarity for s in segments])
        confidence = float(avg_planarity)

        # Classification logic
        if len(segments) == 1:
            if segments[0].roof_type == RoofType.FLAT:
                return RoofType.FLAT, confidence
            else:
                return (
                    RoofType.GABLED,
                    confidence * 0.8,
                )  # Lower confidence for single pitched

        elif len(segments) == 2:
            # Check if normals are opposite (gabled roof)
            n1, n2 = segments[0].normal, segments[1].normal
            # Project to horizontal plane and check angle
            n1_h = np.array([n1[0], n1[1], 0])
            n2_h = np.array([n2[0], n2[1], 0])
            n1_h = n1_h / (np.linalg.norm(n1_h) + 1e-10)
            n2_h = n2_h / (np.linalg.norm(n2_h) + 1e-10)
            dot = np.dot(n1_h, n2_h)

            if dot < -0.7:  # Opposite directions (>135°)
                return RoofType.GABLED, confidence
            else:
                return RoofType.COMPLEX, confidence * 0.7

        elif 3 <= len(segments) <= 4 and pitched_count >= 3:
            return RoofType.HIPPED, confidence

        elif len(segments) > 4:
            return RoofType.COMPLEX, confidence

        else:
            return RoofType.UNKNOWN, confidence * 0.5

    def _detect_ridge_lines(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        segments: List[RoofSegment],
    ) -> np.ndarray:
        """
        Detect ridge lines (intersections between roof planes).

        Ridge points are characterized by:
        - High curvature
        - Located between segments with different normals
        - Upper part of roof

        Args:
            points: Roof points [N, 3]
            normals: Normal vectors [N, 3]
            segments: Detected roof segments

        Returns:
            Indices of ridge line points
        """
        if len(segments) < 2:
            return np.array([], dtype=int)

        # Find points with high curvature (sharp transitions)
        # Approximate curvature from normal variation in neighborhood
        try:
            tree = KDTree(points)
            curvatures = np.zeros(len(points))

            for i in range(len(points)):
                _, neighbors = tree.query(points[i], k=min(20, len(points)))
                neighbor_normals = normals[neighbors]
                # Curvature ≈ std of normals
                curvatures[i] = np.std(neighbor_normals)

            # Ridge points have high curvature
            ridge_mask = curvatures > self.ridge_curvature_threshold

            # Ridge points are typically at top of building
            if ridge_mask.sum() > 0:
                z_percentile_90 = np.percentile(points[:, 2], 90)
                high_mask = points[:, 2] > z_percentile_90
                ridge_mask = ridge_mask & high_mask

            ridge_indices = np.where(ridge_mask)[0]
            return ridge_indices

        except Exception as e:
            logger.warning(f"Ridge detection failed: {e}")
            return np.array([], dtype=int)

    def _detect_roof_edges(
        self,
        points: np.ndarray,
        curvature: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Detect roof edge/border points.

        Edge points are at the perimeter of the roof.

        Args:
            points: Roof points [N, 3]
            curvature: Optional precomputed curvature

        Returns:
            Indices of roof edge points
        """
        if len(points) < 10:
            return np.array([], dtype=int)

        try:
            # Project to 2D (XY plane)
            points_2d = points[:, :2]

            # Find convex hull (perimeter)
            from scipy.spatial import ConvexHull

            hull = ConvexHull(points_2d)

            # Points on hull are edge points
            edge_indices = np.unique(hull.simplices.flatten())

            return edge_indices

        except Exception as e:
            logger.warning(f"Edge detection failed: {e}")
            return np.array([], dtype=int)

    def _detect_dormers(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        segments: List[RoofSegment],
    ) -> np.ndarray:
        """
        Detect dormer structures (vertical protrusions from roof).

        Dormers are characterized by:
        - Higher verticality than main roof
        - Located above main roof plane
        - Relatively small extent

        Args:
            points: Roof points [N, 3]
            normals: Normal vectors [N, 3]
            segments: Main roof segments

        Returns:
            Indices of dormer points
        """
        if len(points) < 50:  # Too small for dormers
            return np.array([], dtype=int)

        try:
            # Compute verticality
            verticality = np.abs(normals[:, 2])

            # Dormer points are more vertical than roof
            dormer_mask = verticality > 0.5

            if dormer_mask.sum() < 10:
                return np.array([], dtype=int)

            # Cluster vertical points
            dormer_points = points[dormer_mask]
            clustering = DBSCAN(eps=1.0, min_samples=10)
            cluster_labels = clustering.fit_predict(dormer_points)

            # Keep only small clusters (dormers are small)
            valid_dormers = np.zeros(len(points), dtype=bool)
            for cluster_id in np.unique(cluster_labels):
                if cluster_id == -1:
                    continue
                cluster_mask = cluster_labels == cluster_id
                cluster_size = cluster_mask.sum()

                # Dormers are typically 10-100 points
                if 10 <= cluster_size <= 100:
                    dormer_indices = np.where(dormer_mask)[0][cluster_mask]
                    valid_dormers[dormer_indices] = True

            return np.where(valid_dormers)[0]

        except Exception as e:
            logger.warning(f"Dormer detection failed: {e}")
            return np.array([], dtype=int)

    def _empty_result(self) -> RoofClassificationResult:
        """Create empty result for cases with no roof detection."""
        return RoofClassificationResult(
            roof_type=RoofType.UNKNOWN,
            segments=[],
            ridge_lines=np.array([], dtype=int),
            edge_points=np.array([], dtype=int),
            dormer_points=np.array([], dtype=int),
            confidence=0.0,
            stats={"total_roof_points": 0, "num_segments": 0},
        )
