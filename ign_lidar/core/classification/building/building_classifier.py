"""
Building Classifier with LOD3 Architectural Details

This module provides a comprehensive building classifier that integrates:
- Roof type detection (Phase 2.1)
- Chimney & superstructure detection (Phase 2.2)
- Balcony & horizontal protrusion detection (Phase 2.3)

The BuildingClassifier coordinates multiple specialized detectors
to provide complete LOD3 building classification with architectural details.

Author: IGN LiDAR HD Processing Library
Version: 3.4.0 (Phase 2.4 Integration)
Date: October 2025
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon

from .balcony_detector import BalconyDetectionResult, BalconyDetector
from .chimney_detector import ChimneyDetectionResult, ChimneyDetector
from .roof_classifier import RoofClassificationResult, RoofTypeClassifier

logger = logging.getLogger(__name__)


@dataclass
class BuildingClassificationResult:
    """
    Result from building classification with LOD3 details.

    Attributes:
        roof_result: Roof type classification result
        chimney_result: Chimney detection result
        balcony_result: Balcony detection result
        point_labels: Per-point classification labels [N]
        building_stats: Dictionary of building-level statistics
        success: Whether classification was successful
    """

    roof_result: Optional[RoofClassificationResult] = None
    chimney_result: Optional[ChimneyDetectionResult] = None
    balcony_result: Optional[BalconyDetectionResult] = None
    point_labels: Optional[np.ndarray] = None
    building_stats: Dict = field(default_factory=dict)
    success: bool = False


@dataclass
class BuildingClassifierConfig:
    """
    Configuration for building classifier.

    Attributes:
        enable_roof_detection: Enable roof type detection
        enable_chimney_detection: Enable chimney detection
        enable_balcony_detection: Enable balcony detection

        # Roof detection parameters
        roof_flat_threshold: Max angle for flat roof (degrees)
        roof_dbscan_eps: DBSCAN epsilon for roof segmentation
        roof_dbscan_min_samples: DBSCAN min samples

        # Chimney detection parameters
        chimney_min_height_above_roof: Min height above roof (m)
        chimney_min_points: Min points for chimney cluster
        chimney_dbscan_eps: DBSCAN epsilon for clustering

        # Balcony detection parameters
        balcony_min_distance_from_facade: Min distance from facade (m)
        balcony_min_points: Min points for balcony cluster
        balcony_max_depth: Max protrusion depth (m)
        balcony_dbscan_eps: DBSCAN epsilon for clustering
    """

    # Feature toggles
    enable_roof_detection: bool = True
    enable_chimney_detection: bool = True
    enable_balcony_detection: bool = True

    # Roof detection
    roof_flat_threshold: float = 15.0
    roof_dbscan_eps: float = 0.3
    roof_dbscan_min_samples: int = 30

    # Chimney detection
    chimney_min_height_above_roof: float = 0.5
    chimney_min_points: int = 20
    chimney_dbscan_eps: float = 0.5

    # Balcony detection
    balcony_min_distance_from_facade: float = 0.5
    balcony_min_points: int = 25
    balcony_max_depth: float = 3.0
    balcony_dbscan_eps: float = 0.5


class BuildingClassifier:
    """
    Comprehensive building classifier with LOD3 architectural details.

    Integrates multiple specialized detectors:
    - RoofTypeClassifier: Detect roof types and elements
    - ChimneyDetector: Detect vertical superstructures
    - BalconyDetector: Detect horizontal protrusions

    Example:
        >>> classifier = BuildingClassifier()
        >>> result = classifier.classify_building(
        ...     points, features, building_polygon, ground_elevation
        ... )
        >>> print(f"Roof type: {result.roof_result.roof_type.name}")
        >>> print(f"Chimneys: {result.chimney_result.num_chimneys}")
        >>> print(f"Balconies: {result.balcony_result.num_balconies}")
    """

    def __init__(self, config: Optional[BuildingClassifierConfig] = None):
        """
        Initialize building classifier.

        Args:
            config: Configuration object. If None, uses defaults.
        """
        self.config = config or BuildingClassifierConfig()

        # Initialize detectors based on config
        if self.config.enable_roof_detection:
            self.roof_classifier = RoofTypeClassifier(
                flat_threshold=self.config.roof_flat_threshold,
                min_plane_points=self.config.roof_dbscan_min_samples,
            )
        else:
            self.roof_classifier = None

        if self.config.enable_chimney_detection:
            self.chimney_detector = ChimneyDetector(
                min_height_above_roof=self.config.chimney_min_height_above_roof,
                min_chimney_points=self.config.chimney_min_points,
                dbscan_eps=self.config.chimney_dbscan_eps,
            )
        else:
            self.chimney_detector = None

        if self.config.enable_balcony_detection:
            self.balcony_detector = BalconyDetector(
                min_distance_from_facade=self.config.balcony_min_distance_from_facade,
                min_balcony_points=self.config.balcony_min_points,
                max_balcony_depth=self.config.balcony_max_depth,
                dbscan_eps=self.config.balcony_dbscan_eps,
            )
        else:
            self.balcony_detector = None

        logger.info(
            f"BuildingClassifier initialized with: "
            f"roof={self.config.enable_roof_detection}, "
            f"chimney={self.config.enable_chimney_detection}, "
            f"balcony={self.config.enable_balcony_detection}"
        )

    def classify_building(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        building_polygon: Polygon,
        ground_elevation: float,
        roof_elevation: Optional[float] = None,
    ) -> BuildingClassificationResult:
        """
        Perform comprehensive LOD3 classification on building points.

        Args:
            points: Building point cloud [N, 3] with XYZ coordinates
            features: Dictionary of computed features:
                - 'verticality': [N] Verticality measure (0=horizontal, 1=vertical)
                - 'normals': [N, 3] Surface normals (required for roof)
                - 'planarity': [N] Planarity measure (required for roof)
                - Additional features as needed
            building_polygon: Building footprint polygon
            ground_elevation: Ground elevation (z value)
            roof_elevation: Optional roof elevation. If None, estimated automatically.

        Returns:
            BuildingClassificationResult with all detection results

        Raises:
            ValueError: If required features are missing
        """
        logger.info(
            f"Classifying building with {len(points)} points, "
            f"ground elevation {ground_elevation:.2f}m"
        )

        result = BuildingClassificationResult()

        try:
            # Validate inputs
            self._validate_inputs(points, features, building_polygon)

            # Initialize point labels (0 = unclassified)
            point_labels = np.zeros(len(points), dtype=int)

            # Estimate roof elevation if not provided
            if roof_elevation is None:
                roof_elevation = self._estimate_roof_elevation(points)
                logger.debug(f"Estimated roof elevation: {roof_elevation:.2f}m")

            # 1. Roof Type Detection
            if self.roof_classifier is not None:
                logger.debug("Running roof type classification...")
                result.roof_result = self.roof_classifier.classify_roof(
                    points, features
                )
                logger.info(
                    f"Roof classification: {result.roof_result.roof_type.name}, "
                    f"{len(result.roof_result.segments)} segments"
                )

            # 2. Chimney Detection (requires roof result for roof indices)
            if self.chimney_detector is not None and result.roof_result is not None:
                logger.debug("Running chimney detection...")
                # Extract roof indices from roof segments
                roof_indices = self._extract_roof_indices(result.roof_result)
                result.chimney_result = self.chimney_detector.detect_superstructures(
                    points, features, roof_indices
                )
                logger.info(
                    f"Detected {result.chimney_result.num_chimneys} chimneys, "
                    f"{result.chimney_result.num_antennas} antennas"
                )

            # 3. Balcony Detection
            if self.balcony_detector is not None:
                logger.debug("Running balcony detection...")
                result.balcony_result = self.balcony_detector.detect_protrusions(
                    points, features, building_polygon, ground_elevation
                )
                logger.info(
                    f"Detected {result.balcony_result.num_balconies} balconies, "
                    f"{result.balcony_result.num_overhangs} overhangs"
                )

            # 4. Merge Classifications into Point Labels
            point_labels = self._merge_classifications(
                points,
                result.roof_result,
                result.chimney_result,
                result.balcony_result,
                ground_elevation,
                roof_elevation,
            )

            result.point_labels = point_labels

            # 5. Compute Building Statistics
            result.building_stats = self._compute_building_stats(
                points, point_labels, result
            )

            result.success = True
            logger.info("Enhanced classification completed successfully")

        except Exception as e:
            logger.error(f"Enhanced classification failed: {e}", exc_info=True)
            result.success = False

        return result

    def _validate_inputs(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        building_polygon: Polygon,
    ) -> None:
        """Validate input parameters."""
        if len(points) == 0:
            raise ValueError("Empty point cloud")

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected [N, 3] points, got shape {points.shape}")

        # Check required features
        required_features = ["verticality"]
        if self.roof_classifier is not None:
            required_features.extend(["normals", "planarity"])

        for feat in required_features:
            if feat not in features:
                raise ValueError(f"Missing required feature: {feat}")

        if not isinstance(building_polygon, Polygon):
            raise ValueError("building_polygon must be a Shapely Polygon")

    def _estimate_roof_elevation(self, points: np.ndarray) -> float:
        """Estimate roof elevation from point cloud."""
        # Use 95th percentile of z values as roof estimate
        roof_z = np.percentile(points[:, 2], 95)
        return float(roof_z)

    def _extract_roof_indices(
        self, roof_result: RoofClassificationResult
    ) -> np.ndarray:
        """Extract roof point indices from roof classification result."""
        # Combine all roof segment indices
        roof_indices = []
        for segment in roof_result.segments:
            roof_indices.extend(segment.points.tolist())

        return np.array(roof_indices, dtype=int)

    def _merge_classifications(
        self,
        points: np.ndarray,
        roof_result: Optional[RoofClassificationResult],
        chimney_result: Optional[ChimneyDetectionResult],
        balcony_result: Optional[BalconyDetectionResult],
        ground_elevation: float,
        roof_elevation: float,
    ) -> np.ndarray:
        """
        Merge all classification results into per-point labels.

        Priority order (highest to lowest):
        1. Chimneys/superstructures (most specific)
        2. Balconies/protrusions (facade-level)
        3. Roof elements (building-level)
        4. Default labels based on height

        Args:
            points: Point cloud [N, 3]
            roof_result: Roof classification result
            chimney_result: Chimney detection result
            balcony_result: Balcony detection result
            ground_elevation: Ground elevation
            roof_elevation: Roof elevation

        Returns:
            Point labels [N] with integer classification codes
        """
        n_points = len(points)
        labels = np.zeros(n_points, dtype=int)

        # Default classification by height
        # (This is a simplified approach - could be more sophisticated)
        heights = points[:, 2]
        mid_height = (ground_elevation + roof_elevation) / 2

        # Default: below mid-height = facade, above = roof
        labels[heights < mid_height] = 6  # Building (generic)
        labels[heights >= mid_height] = 63  # Roof (generic)

        # Apply roof classification (if available)
        if roof_result is not None and len(roof_result.segments) > 0:
            # Mark all roof segment points as roof
            for segment in roof_result.segments:
                labels[segment.points] = 63  # Roof (generic)

            # Mark ridge lines if detected
            if len(roof_result.ridge_lines) > 0:
                labels[roof_result.ridge_lines] = 64  # Building.RoofRidge

            # Mark edges if detected
            if len(roof_result.edge_points) > 0:
                labels[roof_result.edge_points] = 65  # Building.RoofEdge

            # Mark dormers if detected
            if len(roof_result.dormer_points) > 0:
                labels[roof_result.dormer_points] = 66  # Building.RoofDormer

        # Apply balcony classification (higher priority)
        if balcony_result is not None and len(balcony_result.balcony_indices) > 0:
            # Balconies override default facade labels
            labels[balcony_result.balcony_indices] = 70  # Building.Balcony
            labels[balcony_result.overhang_indices] = 71  # Building.Overhang
            labels[balcony_result.canopy_indices] = 72  # Building.Canopy

        # Apply chimney classification (highest priority)
        if chimney_result is not None and len(chimney_result.chimney_indices) > 0:
            # Chimneys override roof labels
            labels[chimney_result.chimney_indices] = 68  # Building.Chimney
            labels[chimney_result.antenna_indices] = 69  # Building.Antenna

        return labels

    def _compute_building_stats(
        self,
        points: np.ndarray,
        point_labels: np.ndarray,
        result: BuildingClassificationResult,
    ) -> Dict:
        """Compute building-level statistics."""
        stats = {
            "total_points": len(points),
            "height_range": float(np.ptp(points[:, 2])),
            "footprint_area": None,
        }

        # Roof statistics
        if result.roof_result is not None:
            stats["roof_type"] = result.roof_result.roof_type.name
            stats["num_roof_segments"] = len(result.roof_result.segments)
            stats["roof_confidence"] = result.roof_result.confidence

        # Chimney statistics
        if result.chimney_result is not None:
            stats["num_chimneys"] = result.chimney_result.num_chimneys
            stats["num_antennas"] = result.chimney_result.num_antennas
            stats["num_ventilations"] = result.chimney_result.num_ventilations

        # Balcony statistics
        if result.balcony_result is not None:
            stats["num_balconies"] = result.balcony_result.num_balconies
            stats["num_overhangs"] = result.balcony_result.num_overhangs
            stats["num_canopies"] = result.balcony_result.num_canopies

        # Label distribution
        unique_labels, counts = np.unique(point_labels, return_counts=True)
        stats["label_distribution"] = dict(zip(unique_labels.tolist(), counts.tolist()))

        return stats

    def get_detector_stats(self) -> Dict[str, bool]:
        """
        Get status of enabled detectors.

        Returns:
            Dictionary with detector enable status
        """
        return {
            "roof_classifier": self.roof_classifier is not None,
            "chimney_detector": self.chimney_detector is not None,
            "balcony_detector": self.balcony_detector is not None,
        }


# Convenience function for simple classification
def classify_building(
    points: np.ndarray,
    features: Dict[str, np.ndarray],
    building_polygon: Polygon,
    ground_elevation: float,
    config: Optional[BuildingClassifierConfig] = None,
) -> BuildingClassificationResult:
    """
    Convenience function for building classification.

    Args:
        points: Building point cloud [N, 3]
        features: Feature dictionary
        building_polygon: Building footprint
        ground_elevation: Ground elevation
        config: Optional configuration

    Returns:
        BuildingClassificationResult with all detections
    """
    classifier = BuildingClassifier(config)
    return classifier.classify_building(
        points, features, building_polygon, ground_elevation
    )


# ============================================================================
# DEPRECATION ALIASES (Remove in v4.0)
# ============================================================================


# Deprecated aliases removed in v4.0
# Use BuildingClassifier, BuildingClassifierConfig, and BuildingClassificationResult directly
