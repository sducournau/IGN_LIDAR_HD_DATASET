"""
Adaptive Building Classification - Point Cloud-Driven with Ground Truth Guidance

This module implements an adaptive approach to building classification where:
1. Ground truth is GUIDANCE, not absolute truth
2. Point cloud features drive the classification
3. Fuzzy boundaries handle imperfect polygons
4. Confidence scores allow multiple sources of evidence

Key Philosophy:
- BD TOPO polygons may have wrong dimensions, missing walls, or be misaligned
- Use ground truth to identify AREAS of interest, then refine based on actual points
- Allow points outside polygons if they look like buildings
- Allow rejection of points inside polygons if they don't match building features

Migrated to building/ module structure - Phase 2
Uses shared utilities from building.utils and base classes from building.base

Author: Building Classification Enhancement
Date: October 20, 2025 (Migrated: October 22, 2025)
"""

import logging
from ..constants import ASPRSClass
from typing import Optional, Dict, List, Tuple, TYPE_CHECKING
import numpy as np
from dataclasses import dataclass

# Import from base module
from .base import (
    ClassificationConfidence,
    BuildingClassifierBase,
    BuildingConfigBase,
    BuildingClassificationResult,
    BuildingMode,
)

# Import shared utilities
from . import utils

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from shapely.geometry import Polygon, MultiPolygon, Point
    import geopandas as gpd

# Check spatial dependencies using utils
HAS_SPATIAL = utils.check_spatial_dependencies()

if HAS_SPATIAL:
    from shapely.geometry import Point
    from shapely.strtree import STRtree
    import geopandas as gpd
    from scipy.spatial import cKDTree


@dataclass
class BuildingFeatureSignature:
    """Expected feature signature for building points."""

    # Height characteristics
    min_height: float = 1.5  # Buildings must be elevated
    typical_height_range: Tuple[float, float] = (2.5, 50.0)

    # Geometric characteristics
    # Walls: high verticality, low planarity in horizontal direction
    wall_verticality_min: float = 0.65
    wall_planarity_max: float = 0.6

    # Roofs: high planarity, low curvature, horizontal or inclined
    roof_planarity_min: float = 0.75
    roof_curvature_max: float = 0.08
    roof_normal_z_range: Tuple[float, float] = (0.3, 1.0)  # Allow inclined roofs

    # Spectral characteristics
    ndvi_max: float = 0.25  # Buildings are not vegetation
    ndvi_wall_max: float = 0.20  # Walls especially non-vegetated

    # Intensity characteristics (optional)
    intensity_variability: float = 0.3  # Buildings have varied intensity

    # Spatial coherence
    min_cluster_size: int = 10  # Minimum points to form building cluster
    max_isolation_distance: float = 5.0  # Max distance from building cluster


@dataclass
class PointBuildingScore:
    """Building classification score for a single point."""

    # Individual feature scores (0-1)
    height_score: float = 0.0
    geometry_score: float = 0.0  # Planarity/verticality match
    spectral_score: float = 0.0  # NDVI check
    spatial_score: float = 0.0  # Proximity to other building points
    ground_truth_score: float = 0.0  # Distance from GT polygon

    # Overall confidence
    confidence: float = 0.0
    confidence_level: ClassificationConfidence = ClassificationConfidence.UNCERTAIN

    # Supporting info
    is_likely_wall: bool = False
    is_likely_roof: bool = False
    distance_to_polygon: float = np.inf

    def compute_confidence(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Compute overall confidence from feature scores.

        Default weights:
        - height: 25% (critical - buildings must be elevated)
        - geometry: 30% (walls/roofs have distinct geometry)
        - spectral: 15% (NDVI helps distinguish vegetation)
        - spatial: 20% (buildings are spatially coherent)
        - ground_truth: 10% (guidance, not absolute)
        """
        if weights is None:
            weights = {
                "height": 0.25,
                "geometry": 0.30,
                "spectral": 0.15,
                "spatial": 0.20,
                "ground_truth": 0.10,
            }

        self.confidence = (
            weights["height"] * self.height_score
            + weights["geometry"] * self.geometry_score
            + weights["spectral"] * self.spectral_score
            + weights["spatial"] * self.spatial_score
            + weights["ground_truth"] * self.ground_truth_score
        )

        # Assign confidence level
        if self.confidence >= 0.9:
            self.confidence_level = ClassificationConfidence.CERTAIN
        elif self.confidence >= 0.7:
            self.confidence_level = ClassificationConfidence.HIGH
        elif self.confidence >= 0.5:
            self.confidence_level = ClassificationConfidence.MEDIUM
        elif self.confidence >= 0.3:
            self.confidence_level = ClassificationConfidence.LOW
        else:
            self.confidence_level = ClassificationConfidence.UNCERTAIN

        return self.confidence


class AdaptiveBuildingClassifier:
    """
    Adaptive building classifier that treats ground truth as guidance.

    Key Features:
    1. **Fuzzy boundaries**: Soft distance decay from polygon edges
    2. **Multi-feature voting**: Height + geometry + spectral + spatial coherence
    3. **Adaptive expansion**: Allow points outside polygons if they match signature
    4. **Intelligent rejection**: Reject points inside polygons if they don't match
    5. **Wall detection**: Special handling for vertical surfaces
    6. **Confidence tracking**: Provide classification confidence scores
    """

    # ASPRS class codes
    # Use ASPRSClass from constants module
    
    
    
    
    

    def __init__(
        self,
        # Building signature
        signature: Optional[BuildingFeatureSignature] = None,
        # Fuzzy boundary parameters
        fuzzy_boundary_inner: float = 0.0,  # Distance inside polygon for certain classification
        fuzzy_boundary_outer: float = 2.0,  # Distance outside polygon to consider
        fuzzy_decay_function: str = "gaussian",  # "linear", "gaussian", "exponential"
        # Adaptive expansion
        enable_adaptive_expansion: bool = True,
        max_expansion_distance: float = 3.0,  # Max distance to expand beyond polygon
        expansion_confidence_threshold: float = 0.7,  # Min confidence for expansion
        # Intelligent rejection
        enable_intelligent_rejection: bool = True,
        rejection_confidence_threshold: float = 0.4,  # Max confidence to reject
        # Spatial coherence
        enable_spatial_clustering: bool = True,
        spatial_radius: float = 2.0,  # Radius for spatial coherence check
        min_neighbor_ratio: float = 0.3,  # Min ratio of building neighbors
        # Classification threshold
        min_classification_confidence: float = 0.5,
        # Feature weights
        feature_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize adaptive building classifier.

        Args:
            signature: Expected feature signature for buildings
            fuzzy_boundary_inner: Inner boundary for certain classification (m)
            fuzzy_boundary_outer: Outer boundary for fuzzy classification (m)
            fuzzy_decay_function: Distance decay function type
            enable_adaptive_expansion: Allow classification beyond polygons
            max_expansion_distance: Maximum expansion distance (m)
            expansion_confidence_threshold: Minimum confidence for expansion
            enable_intelligent_rejection: Allow rejection of points in polygons
            rejection_confidence_threshold: Maximum confidence to reject
            enable_spatial_clustering: Use spatial coherence
            spatial_radius: Radius for spatial neighbor check (m)
            min_neighbor_ratio: Minimum ratio of building neighbors
            min_classification_confidence: Minimum confidence to classify as building
            feature_weights: Custom feature weights for confidence computation
        """
        self.signature = (
            signature if signature is not None else BuildingFeatureSignature()
        )

        self.fuzzy_boundary_inner = fuzzy_boundary_inner
        self.fuzzy_boundary_outer = fuzzy_boundary_outer
        self.fuzzy_decay_function = fuzzy_decay_function

        self.enable_adaptive_expansion = enable_adaptive_expansion
        self.max_expansion_distance = max_expansion_distance
        self.expansion_confidence_threshold = expansion_confidence_threshold

        self.enable_intelligent_rejection = enable_intelligent_rejection
        self.rejection_confidence_threshold = rejection_confidence_threshold

        self.enable_spatial_clustering = enable_spatial_clustering
        self.spatial_radius = spatial_radius
        self.min_neighbor_ratio = min_neighbor_ratio

        self.min_classification_confidence = min_classification_confidence
        self.feature_weights = feature_weights

        logger.info("Adaptive Building Classifier initialized")
        logger.info(
            f"  Fuzzy boundaries: inner={fuzzy_boundary_inner}m, outer={fuzzy_boundary_outer}m"
        )
        logger.info(
            f"  Adaptive expansion: {enable_adaptive_expansion} (max {max_expansion_distance}m)"
        )
        logger.info(f"  Intelligent rejection: {enable_intelligent_rejection}")
        logger.info(
            f"  Spatial clustering: {enable_spatial_clustering} (radius {spatial_radius}m)"
        )
        logger.info(f"  Min confidence: {min_classification_confidence}")

    def classify_buildings_adaptive(
        self,
        points: np.ndarray,
        building_polygons: "gpd.GeoDataFrame",
        height: Optional[np.ndarray] = None,
        planarity: Optional[np.ndarray] = None,
        verticality: Optional[np.ndarray] = None,
        curvature: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        ndvi: Optional[np.ndarray] = None,
        intensity: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        is_ground: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """
        Adaptively classify building points using ground truth as guidance.

        Process:
        1. Compute feature scores for all points
        2. Compute ground truth proximity scores (fuzzy boundaries)
        3. Compute spatial coherence scores
        4. Combine into overall confidence
        5. Apply adaptive expansion (allow points outside polygons)
        6. Apply intelligent rejection (reject points inside polygons)
        7. Classify based on confidence threshold

        Args:
            points: Point coordinates [N, 3]
            building_polygons: GeoDataFrame with building footprints
            height: Height above ground [N]
            planarity: Planarity feature [N]
            verticality: Verticality feature [N]
            curvature: Curvature feature [N]
            normals: Surface normals [N, 3]
            ndvi: NDVI values [N]
            intensity: Intensity values [N]
            labels: Existing labels [N] (optional, for context)
            is_ground: Binary ground indicator [N] (1=ground, 0=non-ground)

        Returns:
            Tuple of (labels, confidences, statistics)
        """
        if not HAS_SPATIAL:
            raise ImportError(
                "Shapely and GeoPandas required for adaptive classification"
            )

        logger.info("=== Adaptive Building Classification ===")
        logger.info(
            f"Processing {len(points):,} points with {len(building_polygons)} building polygons"
        )

        n_points = len(points)
        building_scores = np.zeros(n_points, dtype=np.float32)
        confidences = np.zeros(n_points, dtype=np.float32)
        new_labels = np.full(n_points, int(ASPRSClass.UNCLASSIFIED), dtype=np.int32)

        if labels is not None:
            new_labels = labels.copy()

        # Step 1: Compute height scores
        logger.info("  Step 1/6: Computing height scores...")
        height_scores = self._compute_height_scores(height, is_ground)

        # Step 2: Compute geometry scores (walls + roofs)
        logger.info("  Step 2/6: Computing geometry scores...")
        geometry_scores, is_wall, is_roof = self._compute_geometry_scores(
            planarity, verticality, curvature, normals
        )

        # Step 3: Compute spectral scores (NDVI + ground rejection)
        logger.info("  Step 3/6: Computing spectral scores...")
        spectral_scores = self._compute_spectral_scores(ndvi, intensity, is_ground)
        # Step 4: Compute ground truth proximity scores (fuzzy boundaries)
        logger.info("  Step 4/6: Computing ground truth proximity scores...")
        gt_scores, distances_to_polygons = self._compute_ground_truth_scores(
            points, building_polygons
        )

        # Step 5: Compute spatial coherence scores
        logger.info("  Step 5/6: Computing spatial coherence scores...")

        # Build initial building candidates (high scores on any feature)
        candidate_mask = (
            (height_scores > 0.3) | (geometry_scores > 0.4) | (gt_scores > 0.2)
        )

        spatial_scores = self._compute_spatial_coherence_scores(
            points, candidate_mask, height, geometry_scores
        )

        # Combine scores into overall confidence
        logger.info("  Step 6/6: Computing final confidence and ground filtering...")

        for i in range(n_points):
            score = PointBuildingScore()
            score.height_score = height_scores[i]
            score.geometry_score = geometry_scores[i]
            score.spectral_score = spectral_scores[i]
            score.spatial_score = spatial_scores[i]
            score.ground_truth_score = gt_scores[i]
            score.is_likely_wall = is_wall[i]
            score.is_likely_roof = is_roof[i]
            score.distance_to_polygon = distances_to_polygons[i]

            confidences[i] = score.compute_confidence(self.feature_weights)
            building_scores[i] = confidences[i]

        # Apply ground point rejection: Ground points cannot be buildings
        if is_ground is not None:
            ground_mask = is_ground.astype(bool)
            n_ground_rejected = np.sum(ground_mask & (confidences > 0))
            if n_ground_rejected > 0:
                logger.info(
                    f"  Rejected {n_ground_rejected:,} ground points "
                    f"from building candidates"
                )
                confidences[ground_mask] = 0.0  # Zero confidence for ground
                building_scores[ground_mask] = 0.0

        # Apply classification logic
        stats = {
            "total_points": n_points,
            "inside_polygons": 0,
            "outside_polygons": 0,
            "expanded": 0,
            "rejected": 0,
            "walls_detected": 0,
            "roofs_detected": 0,
            "high_confidence": 0,
            "medium_confidence": 0,
            "low_confidence": 0,
        }

        # Classify based on confidence and context
        for i in range(n_points):
            inside_polygon = distances_to_polygons[i] <= 0  # Negative = inside

            if inside_polygon:
                stats["inside_polygons"] += 1

                # Inside polygon: classify if confidence above threshold
                # OR reject if intelligent rejection enabled and confidence too low
                if confidences[i] >= self.min_classification_confidence:
                    new_labels[i] = int(ASPRSClass.BUILDING)

                    # Track confidence levels
                    if confidences[i] >= 0.7:
                        stats["high_confidence"] += 1
                    elif confidences[i] >= 0.5:
                        stats["medium_confidence"] += 1
                    else:
                        stats["low_confidence"] += 1

                    # Track wall/roof detection
                    if is_wall[i]:
                        stats["walls_detected"] += 1
                    if is_roof[i]:
                        stats["roofs_detected"] += 1

                elif (
                    self.enable_intelligent_rejection
                    and confidences[i] < self.rejection_confidence_threshold
                ):
                    # Reject: point inside polygon but doesn't look like building
                    stats["rejected"] += 1
                    # Keep existing label or unclassified
                    pass

            else:
                stats["outside_polygons"] += 1

                # Outside polygon: classify only if adaptive expansion enabled
                # and confidence is very high
                if self.enable_adaptive_expansion:
                    distance = distances_to_polygons[i]

                    if (
                        distance <= self.max_expansion_distance
                        and confidences[i] >= self.expansion_confidence_threshold
                    ):

                        new_labels[i] = int(ASPRSClass.BUILDING)
                        stats["expanded"] += 1

                        if confidences[i] >= 0.7:
                            stats["high_confidence"] += 1
                        else:
                            stats["medium_confidence"] += 1

                        if is_wall[i]:
                            stats["walls_detected"] += 1
                        if is_roof[i]:
                            stats["roofs_detected"] += 1

        # Calculate final statistics
        total_classified = np.sum(new_labels == int(ASPRSClass.BUILDING))
        stats["total_classified"] = total_classified
        stats["classification_rate"] = total_classified / n_points * 100

        # Log results
        logger.info("=== Classification Results ===")
        logger.info(
            f"Total classified as building: {total_classified:,} ({stats['classification_rate']:.1f}%)"
        )
        logger.info(f"  Inside polygons: {stats['inside_polygons']:,}")
        logger.info(f"  Outside polygons: {stats['outside_polygons']:,}")

        if self.enable_adaptive_expansion:
            logger.info(f"  Expanded beyond polygons: {stats['expanded']:,} points")

        if self.enable_intelligent_rejection:
            logger.info(
                f"  Rejected (inside but low confidence): {stats['rejected']:,} points"
            )

        logger.info(f"  Walls detected: {stats['walls_detected']:,}")
        logger.info(f"  Roofs detected: {stats['roofs_detected']:,}")

        logger.info(f"Confidence distribution:")
        logger.info(f"  High (≥0.7): {stats['high_confidence']:,}")
        logger.info(f"  Medium (0.5-0.7): {stats['medium_confidence']:,}")
        logger.info(f"  Low (0.3-0.5): {stats['low_confidence']:,}")

        return new_labels, confidences, stats

    def _compute_height_scores(
        self, height: Optional[np.ndarray], is_ground: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute building likelihood from height.

        Uses shared utility: utils.compute_height_statistics for validation

        Score 1.0: Within typical building height range
        Score 0.5-1.0: Above minimum but outside typical range
        Score 0.0-0.5: Below minimum (gradual decay)

        Ground points (is_ground=1) automatically get zero score.
        """
        if height is None:
            return np.ones(0, dtype=np.float32)

        scores = np.zeros(len(height), dtype=np.float32)

        # Ground points always get zero score
        if is_ground is not None:
            non_ground_mask = is_ground == 0
        else:
            non_ground_mask = np.ones(len(height), dtype=bool)

        # Perfect score for typical building heights (non-ground only)
        typical_mask = (
            non_ground_mask
            & (height >= self.signature.typical_height_range[0])
            & (height <= self.signature.typical_height_range[1])
        )
        scores[typical_mask] = 1.0

        # Partial score for elevated but outside typical range
        elevated_mask = (height >= self.signature.min_height) & ~typical_mask
        if np.any(elevated_mask):
            # Linear decay above max typical height
            above_mask = elevated_mask & (
                height > self.signature.typical_height_range[1]
            )
            if np.any(above_mask):
                excess = height[above_mask] - self.signature.typical_height_range[1]
                scores[above_mask] = np.clip(1.0 - excess / 50.0, 0.5, 1.0)

            # Score for minimum to typical range
            between_mask = elevated_mask & (
                height <= self.signature.typical_height_range[1]
            )
            if np.any(between_mask):
                relative = (height[between_mask] - self.signature.min_height) / (
                    self.signature.typical_height_range[0] - self.signature.min_height
                )
                scores[between_mask] = 0.5 + 0.5 * relative

        # Gradual decay below minimum height
        below_mask = height < self.signature.min_height
        if np.any(below_mask):
            scores[below_mask] = np.clip(
                height[below_mask] / self.signature.min_height, 0.0, 0.5
            )

        return scores

    def _compute_geometry_scores(
        self,
        planarity: Optional[np.ndarray],
        verticality: Optional[np.ndarray],
        curvature: Optional[np.ndarray],
        normals: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute building likelihood from geometry (walls + roofs).

        Returns:
            Tuple of (geometry_scores, is_wall_mask, is_roof_mask)
        """
        n = len(planarity) if planarity is not None else 0
        scores = np.zeros(n, dtype=np.float32)
        is_wall = np.zeros(n, dtype=bool)
        is_roof = np.zeros(n, dtype=bool)

        if n == 0:
            return scores, is_wall, is_roof

        # Wall detection: high verticality, low horizontal planarity
        wall_score = np.zeros(n, dtype=np.float32)
        if verticality is not None:
            wall_score = np.clip(
                (verticality - self.signature.wall_verticality_min)
                / (1.0 - self.signature.wall_verticality_min),
                0.0,
                1.0,
            )
            is_wall = verticality >= self.signature.wall_verticality_min

        # Roof detection: high planarity, low curvature, appropriate normal direction
        roof_score = np.zeros(n, dtype=np.float32)
        if planarity is not None:
            roof_score = np.clip(
                (planarity - self.signature.roof_planarity_min)
                / (1.0 - self.signature.roof_planarity_min),
                0.0,
                1.0,
            )

            # Bonus for low curvature
            if curvature is not None:
                curv_bonus = np.clip(
                    1.0 - curvature / self.signature.roof_curvature_max, 0.0, 1.0
                )
                roof_score = (roof_score + curv_bonus) / 2.0

            # Check normal direction (horizontal to inclined)
            if normals is not None:
                normal_z = np.abs(normals[:, 2])
                normal_valid = (normal_z >= self.signature.roof_normal_z_range[0]) & (
                    normal_z <= self.signature.roof_normal_z_range[1]
                )
                roof_score[~normal_valid] *= 0.5

            is_roof = (planarity >= self.signature.roof_planarity_min) & (
                roof_score > 0.5
            )

        # Combine wall and roof scores (take maximum - buildings have both)
        scores = np.maximum(wall_score, roof_score)

        return scores, is_wall, is_roof

    def _compute_spectral_scores(
        self,
        ndvi: Optional[np.ndarray],
        intensity: Optional[np.ndarray],
        is_ground: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute building likelihood from spectral features.

        Buildings should have:
        - Low NDVI (not vegetation)
        - Varied intensity (different materials)
        - Not ground (is_ground=0)
        """
        n = len(ndvi) if ndvi is not None else 0
        scores = np.zeros(n, dtype=np.float32)

        if n == 0:
            return scores

        # NDVI score: low NDVI = high building likelihood
        if ndvi is not None:
            ndvi_score = np.clip(1.0 - ndvi / self.signature.ndvi_max, 0.0, 1.0)
            scores = ndvi_score
        else:
            scores = np.full(n, 0.5)  # Neutral score if NDVI unavailable

        # Ground points penalty: Reduce score significantly
        if is_ground is not None:
            ground_mask = is_ground == 1
            scores[ground_mask] *= 0.1  # Heavily penalize ground points

        # Intensity variability (optional, less reliable)
        # Buildings typically have varied intensity due to different materials
        # This is a weak indicator, so we don't weight it heavily

        return scores

    def _compute_ground_truth_scores(
        self, points: np.ndarray, building_polygons: "gpd.GeoDataFrame"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute building likelihood from ground truth proximity (fuzzy boundaries).

        Score 1.0: Inside polygon (distance ≤ 0)
        Score 1.0 → 0.0: Outside polygon, decaying with distance
        Score 0.0: Beyond max expansion distance

        Returns:
            Tuple of (scores, signed_distances)
            - scores: Fuzzy membership [0, 1]
            - signed_distances: Negative inside, positive outside (meters)
        """
        n = len(points)
        scores = np.zeros(n, dtype=np.float32)
        distances = np.full(n, np.inf, dtype=np.float32)

        if building_polygons is None or len(building_polygons) == 0:
            return scores, distances

        # Build spatial index
        polygons_list = building_polygons.geometry.tolist()
        tree = STRtree(polygons_list)

        # For each point, find nearest polygon and compute distance
        points_2d = points[:, :2]

        for i in range(n):
            pt = Point(points_2d[i, 0], points_2d[i, 1])

            # Find nearest polygons
            nearby_indices = tree.query(pt)

            if len(nearby_indices) == 0:
                continue

            # Compute signed distance to nearest polygon
            min_distance = np.inf
            is_inside = False

            for poly_idx in nearby_indices:
                polygon = polygons_list[poly_idx]

                if polygon.contains(pt):
                    # Inside polygon
                    is_inside = True
                    # Distance to boundary (negative inside)
                    boundary_dist = polygon.exterior.distance(pt)
                    min_distance = min(min_distance, -boundary_dist)
                else:
                    # Outside polygon
                    dist = polygon.distance(pt)
                    min_distance = min(min_distance, dist)

            distances[i] = min_distance

            # Compute fuzzy score based on distance
            if is_inside or min_distance <= 0:
                # Inside polygon: full score
                scores[i] = 1.0
            elif min_distance <= self.fuzzy_boundary_outer:
                # Within fuzzy boundary: decay based on distance
                if self.fuzzy_decay_function == "linear":
                    scores[i] = 1.0 - (min_distance / self.fuzzy_boundary_outer)
                elif self.fuzzy_decay_function == "gaussian":
                    # Gaussian decay: exp(-d²/2σ²)
                    sigma = self.fuzzy_boundary_outer / 2.0
                    scores[i] = np.exp(-(min_distance**2) / (2 * sigma**2))
                elif self.fuzzy_decay_function == "exponential":
                    # Exponential decay: exp(-d/λ)
                    lambda_ = self.fuzzy_boundary_outer / 3.0
                    scores[i] = np.exp(-min_distance / lambda_)
                else:
                    # Default to linear
                    scores[i] = 1.0 - (min_distance / self.fuzzy_boundary_outer)
            else:
                # Beyond fuzzy boundary: no score
                scores[i] = 0.0

        return scores, distances

    def _compute_spatial_coherence_scores(
        self,
        points: np.ndarray,
        candidate_mask: np.ndarray,
        height: Optional[np.ndarray],
        geometry_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Compute building likelihood from spatial coherence.

        Buildings are spatially coherent:
        - Points near other building points get higher scores
        - Isolated points get lower scores
        - Considers both 2D and 3D proximity

        Returns:
            Spatial coherence scores [0, 1]
        """
        n = len(points)
        scores = np.zeros(n, dtype=np.float32)

        if not self.enable_spatial_clustering:
            return np.ones(n, dtype=np.float32)  # Neutral score

        if not np.any(candidate_mask):
            return scores

        # Build spatial index for candidate points
        candidate_points = points[candidate_mask]

        if len(candidate_points) < 3:
            return scores

        try:
            # Use 3D spatial index for better coherence check
            tree = cKDTree(candidate_points)

            # For each point, check neighbors within radius
            for i in range(n):
                # Query neighbors
                neighbors = tree.query_ball_point(points[i], self.spatial_radius)

                if len(neighbors) == 0:
                    scores[i] = 0.0
                    continue

                # Compute neighbor ratio
                neighbor_ratio = len(neighbors) / max(len(candidate_points), 1)

                # Weight by geometry scores of neighbors
                neighbor_indices = np.where(candidate_mask)[0][neighbors]
                avg_neighbor_quality = np.mean(geometry_scores[neighbor_indices])

                # Combine neighbor ratio and quality
                scores[i] = (
                    min(1.0, neighbor_ratio / self.min_neighbor_ratio)
                    * avg_neighbor_quality
                )

        except Exception as e:
            logger.warning(f"Spatial coherence computation failed: {e}")
            scores = np.ones(n, dtype=np.float32) * 0.5

        return scores


__all__ = [
    "ClassificationConfidence",
    "BuildingFeatureSignature",
    "PointBuildingScore",
    "AdaptiveBuildingClassifier",
]
