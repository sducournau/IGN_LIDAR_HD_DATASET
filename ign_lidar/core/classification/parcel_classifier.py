"""
Parcel-Based Classification Module

This module provides intelligent classification by grouping points into cadastral
parcels and processing them as coherent units. This approach provides:
- 10-100× faster processing through batch operations
- Spatially coherent results within each parcel
- Natural integration with ground truth data (cadastre, BD Forêt, RPG)
- Intelligent land use detection at parcel level

v3.2+ Changes:
    - Now inherits from BaseClassifier for API consistency
    - Added classify() method following BaseClassifier interface
    - classify_by_parcels() maintains original functionality
    - Returns base ClassificationResult for compatibility

Author: Classification Optimization Team
Date: October 25, 2025
Version: 1.1
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Import BaseClassifier for v3.2+ unified interface
from .base import BaseClassifier
from .base import ClassificationResult as BaseClassificationResult
from .constants import ASPRSClass

logger = logging.getLogger(__name__)

try:
    import geopandas as gpd
    from shapely.geometry import MultiPolygon, Point, Polygon
    from shapely.strtree import STRtree

    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False
    logger.warning("Spatial libraries not available for parcel classification")


# ============================================================================
# Parcel Statistics and Metadata
# ============================================================================


@dataclass
class ParcelStatistics:
    """Aggregated statistics for a parcel."""

    # Basic info
    parcel_id: str
    n_points: int
    area_m2: float
    point_density: float

    # Feature aggregates
    mean_ndvi: float
    std_ndvi: float
    mean_height: float
    std_height: float
    height_range: float
    mean_planarity: float
    mean_verticality: float
    mean_curvature: float
    dominant_normal_z: float

    # Ground truth matches
    bd_foret_match: Optional[Dict] = None
    rpg_match: Optional[Dict] = None

    # Classification results
    parcel_type: str = "unknown"
    confidence_scores: Dict[str, float] = None

    def __post_init__(self):
        if self.confidence_scores is None:
            self.confidence_scores = {}


@dataclass
class ParcelClassificationConfig:
    """Configuration for parcel-based classification."""

    # Parcel filtering
    min_parcel_points: int = 20
    min_parcel_area: float = 10.0  # m²

    # Classification thresholds
    parcel_confidence_threshold: float = 0.6

    # Feature validation
    require_feature_validation: bool = True

    # Point refinement
    refine_points: bool = True
    refinement_method: str = "feature_based"  # 'feature_based' or 'clustering'

    # NDVI thresholds for parcel-level classification
    forest_ndvi_min: float = 0.5
    agriculture_ndvi_min: float = 0.2
    agriculture_ndvi_max: float = 0.6
    building_ndvi_max: float = 0.15
    road_ndvi_max: float = 0.15
    water_ndvi_max: float = -0.05

    # Geometric thresholds
    forest_curvature_min: float = 0.25
    forest_planarity_max: float = 0.6
    building_verticality_min: float = 0.6
    building_planarity_min: float = 0.7
    road_planarity_min: float = 0.85
    water_planarity_min: float = 0.9


# ============================================================================
# Parcel Type Constants
# ============================================================================


class ParcelType:
    """Parcel type classification constants."""

    FOREST = "forest"
    AGRICULTURE = "agriculture"
    BUILDING = "building"
    ROAD = "road"
    WATER = "water"
    MIXED = "mixed"
    UNKNOWN = "unknown"

    @classmethod
    def all_types(cls) -> List[str]:
        """Get all parcel types."""
        return [
            cls.FOREST,
            cls.AGRICULTURE,
            cls.BUILDING,
            cls.ROAD,
            cls.WATER,
            cls.MIXED,
            cls.UNKNOWN,
        ]


# ============================================================================
# Main Parcel Classifier
# ============================================================================


class ParcelClassifier(BaseClassifier):
    """
    Classify point clouds using parcel-based clustering.

    v3.2+ Changes:
    - Now inherits from BaseClassifier for API consistency
    - Added classify() method following BaseClassifier interface
    - classify_by_parcels() wraps original functionality

    This classifier groups points by cadastral parcels and processes them
    as coherent units, providing faster and more spatially consistent results.

    Features:
    - Parcel-level feature aggregation
    - Ground truth integration (cadastre, BD Forêt, RPG)
    - Intelligent parcel type classification
    - Point-level refinement within parcels

    Example (v3.2+ unified interface):
        >>> classifier = ParcelClassifier()
        >>> features = {
        ...     'ndvi': ndvi_array,
        ...     'height': height_array,
        ...     'cadastre': cadastre_gdf  # Required
        ... }
        >>> result = classifier.classify(points, features)
        >>> labels = result.labels

    Example (legacy interface - still works):
        >>> classifier = ParcelClassifier()
        >>> labels = classifier.classify_by_parcels(
        ...     points=points,
        ...     features=features,
        ...     cadastre=cadastre_gdf
        ... )
    """

    # ASPRS Classification codes
    # Use ASPRSClass from constants module

    def __init__(self, config: Optional[ParcelClassificationConfig] = None):
        """
        Initialize parcel classifier.

        Args:
            config: Configuration object (uses defaults if None)
        """
        if not HAS_SPATIAL:
            raise ImportError(
                "shapely and geopandas required for parcel classification. "
                "Install with: pip install shapely geopandas"
            )

        self.config = config or ParcelClassificationConfig()
        self._parcel_stats_cache: Dict[str, ParcelStatistics] = {}

    # ========================================================================
    # v3.2+ Unified Interface (BaseClassifier compatibility)
    # ========================================================================

    def classify(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        ground_truth: Optional[Union[gpd.GeoDataFrame, Dict[str, Any]]] = None,
        **kwargs,
    ) -> BaseClassificationResult:
        """
        Classify point cloud using BaseClassifier interface (v3.2+).

        This method requires 'cadastre' in either features or ground_truth,
        as parcel-based classification depends on cadastral parcels.

        Args:
            points: Point cloud [N, 3] with XYZ coordinates
            features: Dictionary mapping feature name → array [N]
                REQUIRED (one of):
                - 'cadastre': GeoDataFrame with cadastral parcels
                Optional features:
                - 'ndvi': NDVI values [N]
                - 'height': Height above ground [N]
                - 'planarity': Planarity [N]
                - 'verticality': Verticality [N]
                - 'curvature': Curvature [N]
                - 'normals': Normal vectors [N, 3]
            ground_truth: Optional ground truth data:
                - GeoDataFrame with cadastral parcels (if not in features)
                - Dictionary with 'cadastre', 'bd_foret', 'rpg' GeoDataFrames
            **kwargs: Additional parameters passed to classify_by_parcels()

        Returns:
            BaseClassificationResult with labels, confidence (None), and metadata

        Raises:
            ValueError: If 'cadastre' not provided

        Example:
            >>> classifier = ParcelClassifier()
            >>> features = {
            ...     'cadastre': cadastre_gdf,
            ...     'ndvi': ndvi_array,
            ...     'height': height_array
            ... }
            >>> result = classifier.classify(points, features)
            >>> labels = result.labels
        """
        # Validate inputs
        self.validate_inputs(points, features)

        # Extract cadastre from features or ground_truth
        cadastre = None
        bd_foret = None
        rpg = None

        # Check features first
        if "cadastre" in features:
            cadastre = features["cadastre"]
            # Remove from features dict for processing
            features = {k: v for k, v in features.items() if k != "cadastre"}

        # Check ground_truth
        if ground_truth is not None:
            if HAS_SPATIAL and isinstance(ground_truth, gpd.GeoDataFrame):
                # Assume it's cadastre if not already set
                if cadastre is None:
                    cadastre = ground_truth
            elif isinstance(ground_truth, dict):
                if "cadastre" in ground_truth and cadastre is None:
                    cadastre = ground_truth["cadastre"]
                if "bd_foret" in ground_truth:
                    bd_foret = ground_truth["bd_foret"]
                if "rpg" in ground_truth:
                    rpg = ground_truth["rpg"]

        # Validate cadastre is provided
        if cadastre is None:
            raise ValueError(
                "ParcelClassifier requires 'cadastre' (GeoDataFrame with parcels). "
                "Provide it in features dict as features['cadastre'] = cadastre_gdf, "
                "or in ground_truth parameter. "
                "Example: result = classifier.classify(points, features, ground_truth=cadastre_gdf)"
            )

        # Call original classify_by_parcels method
        labels = self.classify_by_parcels(
            points=points,
            features=features,
            cadastre=cadastre,
            bd_foret=bd_foret,
            rpg=rpg,
        )

        # Compute parcel-level confidence (if cached)
        parcel_confidences = {}
        for parcel_id, stats in self._parcel_stats_cache.items():
            if stats.confidence_scores:
                avg_conf = np.mean(list(stats.confidence_scores.values()))
                parcel_confidences[parcel_id] = avg_conf

        # Return standardized result
        return BaseClassificationResult(
            labels=labels,
            confidence=None,  # ParcelClassifier doesn't provide point-level confidence
            metadata={
                "method": "parcel_based",
                "num_parcels": len(self._parcel_stats_cache),
                "parcel_confidences": parcel_confidences,
                "config": {
                    "min_parcel_points": self.config.min_parcel_points,
                    "refine_points": self.config.refine_points,
                },
            },
        )

    # ========================================================================
    # Original Interface (maintained for backward compatibility)
    # ========================================================================

    def classify_by_parcels(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        cadastre: gpd.GeoDataFrame,
        bd_foret: Optional[gpd.GeoDataFrame] = None,
        rpg: Optional[gpd.GeoDataFrame] = None,
    ) -> np.ndarray:
        """
        Classify points using parcel-based clustering.

        Args:
            points: Point coordinates [N, 3] (X, Y, Z)
            features: Dictionary of feature arrays:
                - 'ndvi': NDVI values [N]
                - 'height': Height above ground [N]
                - 'planarity': Planarity [N]
                - 'verticality': Verticality [N]
                - 'curvature': Curvature [N]
                - 'normals': Normal vectors [N, 3]
            cadastre: GeoDataFrame with cadastral parcels
            bd_foret: Optional GeoDataFrame with forest data
            rpg: Optional GeoDataFrame with agricultural data

        Returns:
            ASPRS classification labels [N]
        """
        n_points = len(points)
        labels = np.full(n_points, int(ASPRSClass.UNCLASSIFIED), dtype=np.uint8)

        logger.info(f"🏘️  Starting parcel-based classification for {n_points:,} points")
        logger.info(f"   Cadastral parcels: {len(cadastre):,}")
        if bd_foret is not None:
            logger.info(f"   Forest parcels: {len(bd_foret):,}")
        if rpg is not None:
            logger.info(f"   Agricultural parcels: {len(rpg):,}")

        # Step 1: Group points by parcel
        parcel_groups = self._group_by_parcels(points, cadastre)

        if not parcel_groups:
            logger.warning(
                "   No points assigned to parcels, falling back to unclassified"
            )
            return labels

        logger.info(f"   Grouped into {len(parcel_groups)} parcels")

        # Step 2: Classify each parcel
        n_classified = 0
        parcel_type_counts = defaultdict(int)

        for parcel_id, point_indices in parcel_groups.items():
            if len(point_indices) < self.config.min_parcel_points:
                continue

            # Extract parcel data
            parcel_points = points[point_indices]
            parcel_features = {
                key: value[point_indices]
                for key, value in features.items()
                if value is not None
            }

            # Compute parcel statistics
            parcel_stats = self.compute_parcel_features(
                parcel_points, parcel_features, parcel_id
            )

            # Match with ground truth
            if bd_foret is not None:
                parcel_stats.bd_foret_match = self._match_bd_foret(
                    parcel_id, parcel_points, bd_foret
                )

            if rpg is not None:
                parcel_stats.rpg_match = self._match_rpg(parcel_id, parcel_points, rpg)

            # Classify parcel type
            parcel_type, confidence = self.classify_parcel_type(parcel_stats)
            parcel_stats.parcel_type = parcel_type
            parcel_stats.confidence_scores = confidence

            # Cache statistics
            self._parcel_stats_cache[parcel_id] = parcel_stats

            # Count parcel types
            parcel_type_counts[parcel_type] += 1

            # Refine point-level labels
            if self.config.refine_points:
                parcel_labels = self.refine_parcel_points(
                    parcel_type, parcel_points, parcel_features, parcel_stats
                )
            else:
                # Use simple parcel-level classification
                parcel_labels = self._get_default_label_for_parcel_type(
                    parcel_type, parcel_features.get("height")
                )

            # Assign labels
            labels[point_indices] = parcel_labels
            n_classified += len(point_indices)

        # Summary statistics
        coverage = 100 * n_classified / n_points
        logger.info(f"   Classified {n_classified:,} points ({coverage:.1f}%)")
        logger.info(f"   Parcel type distribution:")
        for ptype, count in sorted(parcel_type_counts.items()):
            logger.info(f"     - {ptype}: {count} parcels")

        return labels

    def _group_by_parcels(
        self, points: np.ndarray, cadastre: gpd.GeoDataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Group points by cadastral parcel.

        Args:
            points: Point coordinates [N, 3]
            cadastre: GeoDataFrame with parcel polygons

        Returns:
            Dictionary mapping parcel_id -> point_indices
        """
        from ign_lidar.io.cadastre import CadastreFetcher

        # Use existing cadastre fetcher grouping method
        fetcher = CadastreFetcher()
        parcel_groups = fetcher.group_points_by_parcel(
            points=points, parcels_gdf=cadastre, labels=None
        )

        # Convert to simple dict of parcel_id -> indices
        result = {}
        for parcel_id, info in parcel_groups.items():
            result[parcel_id] = info["indices"]

        return result

    def compute_parcel_features(
        self,
        parcel_points: np.ndarray,
        parcel_features: Dict[str, np.ndarray],
        parcel_id: str,
    ) -> ParcelStatistics:
        """
        Compute aggregated features for a parcel.

        Args:
            parcel_points: Points in this parcel [M, 3]
            parcel_features: Features for points in parcel
            parcel_id: Unique parcel identifier

        Returns:
            ParcelStatistics object with aggregated features
        """
        n_points = len(parcel_points)

        # Compute area (rough estimate from point extent)
        x_extent = np.ptp(parcel_points[:, 0])
        y_extent = np.ptp(parcel_points[:, 1])
        area_m2 = x_extent * y_extent if x_extent > 0 and y_extent > 0 else 1.0
        point_density = n_points / area_m2

        # Extract features with safe defaults
        ndvi = parcel_features.get("ndvi", np.zeros(n_points))
        height = parcel_features.get("height", np.zeros(n_points))
        planarity = parcel_features.get("planarity", np.zeros(n_points))
        verticality = parcel_features.get("verticality", np.zeros(n_points))
        curvature = parcel_features.get("curvature", np.zeros(n_points))
        normals = parcel_features.get("normals", np.zeros((n_points, 3)))

        # Compute statistics
        stats = ParcelStatistics(
            parcel_id=parcel_id,
            n_points=n_points,
            area_m2=area_m2,
            point_density=point_density,
            mean_ndvi=float(np.mean(ndvi)),
            std_ndvi=float(np.std(ndvi)),
            mean_height=float(np.mean(height)),
            std_height=float(np.std(height)),
            height_range=float(np.ptp(height)),
            mean_planarity=float(np.mean(planarity)),
            mean_verticality=float(np.mean(verticality)),
            mean_curvature=float(np.mean(curvature)),
            dominant_normal_z=(
                float(np.median(normals[:, 2])) if len(normals) > 0 else 0.0
            ),
        )

        return stats

    def classify_parcel_type(
        self, stats: ParcelStatistics
    ) -> Tuple[str, Dict[str, float]]:
        """
        Classify parcel type using multi-feature decision tree.

        Args:
            stats: Aggregated parcel statistics

        Returns:
            Tuple of (parcel_type, confidence_scores)
        """
        confidence = {}

        # DECISION 1: Forest Parcel
        forest_score = 0.0
        if stats.bd_foret_match is not None:
            forest_score += 0.4  # BD Forêt ground truth
        if stats.mean_ndvi >= self.config.forest_ndvi_min:
            forest_score += 0.3  # High NDVI
        if stats.mean_curvature >= self.config.forest_curvature_min:
            forest_score += 0.2  # Irregular surface
        if stats.mean_planarity <= self.config.forest_planarity_max:
            forest_score += 0.1  # Low planarity
        confidence[ParcelType.FOREST] = forest_score

        # DECISION 2: Agricultural Parcel
        agri_score = 0.0
        if stats.rpg_match is not None:
            agri_score += 0.5  # RPG ground truth
        if (
            self.config.agriculture_ndvi_min
            <= stats.mean_ndvi
            <= self.config.agriculture_ndvi_max
        ):
            agri_score += 0.3  # Moderate vegetation
        if stats.mean_planarity >= 0.7:
            agri_score += 0.2  # Relatively flat
        confidence[ParcelType.AGRICULTURE] = agri_score

        # DECISION 3: Building Parcel
        building_score = 0.0
        if stats.mean_verticality >= self.config.building_verticality_min:
            building_score += 0.4  # High verticality (walls)
        if stats.mean_ndvi <= self.config.building_ndvi_max:
            building_score += 0.3  # Not vegetation
        if stats.mean_planarity >= self.config.building_planarity_min:
            building_score += 0.2  # Planar surfaces
        if stats.height_range > 3.0:
            building_score += 0.1  # Multi-story
        confidence[ParcelType.BUILDING] = building_score

        # DECISION 4: Road Parcel
        road_score = 0.0
        if stats.mean_planarity >= self.config.road_planarity_min:
            road_score += 0.4  # Very flat
        if stats.mean_ndvi <= self.config.road_ndvi_max:
            road_score += 0.3  # Not vegetation
        if abs(stats.dominant_normal_z) > 0.9:
            road_score += 0.2  # Horizontal
        if stats.mean_height < 1.0:
            road_score += 0.1  # Near ground
        confidence[ParcelType.ROAD] = road_score

        # DECISION 5: Water Parcel
        water_score = 0.0
        if stats.mean_ndvi <= self.config.water_ndvi_max:
            water_score += 0.5  # Negative NDVI
        if stats.mean_planarity >= self.config.water_planarity_min:
            water_score += 0.3  # Very flat
        if stats.mean_height < 0.5:
            water_score += 0.2  # Very low
        confidence[ParcelType.WATER] = water_score

        # Select parcel type with highest confidence
        if max(confidence.values()) < self.config.parcel_confidence_threshold:
            return ParcelType.MIXED, confidence

        parcel_type = max(confidence, key=confidence.get)
        return parcel_type, confidence

    def refine_parcel_points(
        self,
        parcel_type: str,
        parcel_points: np.ndarray,
        parcel_features: Dict[str, np.ndarray],
        parcel_stats: ParcelStatistics,
    ) -> np.ndarray:
        """
        Refine classification for individual points within parcel.

        Args:
            parcel_type: Classified parcel type
            parcel_points: Points in this parcel [M, 3]
            parcel_features: Features for points in parcel
            parcel_stats: Parcel statistics

        Returns:
            ASPRS labels for parcel points [M]
        """
        n_points = len(parcel_points)
        labels = np.zeros(n_points, dtype=np.uint8)

        if parcel_type == ParcelType.FOREST:
            labels = self._refine_forest_parcel(parcel_features, parcel_stats)

        elif parcel_type == ParcelType.AGRICULTURE:
            labels = self._refine_agriculture_parcel(parcel_features, parcel_stats)

        elif parcel_type == ParcelType.BUILDING:
            labels = self._refine_building_parcel(parcel_features, parcel_stats)

        elif parcel_type == ParcelType.ROAD:
            labels = self._refine_road_parcel(parcel_features, parcel_stats)

        elif parcel_type == ParcelType.WATER:
            labels[:] = int(ASPRSClass.WATER)

        else:  # MIXED or UNKNOWN
            # Fall back to feature-based classification
            labels = self._classify_points_by_features(parcel_features)

        return labels

    def _refine_forest_parcel(
        self, features: Dict[str, np.ndarray], stats: ParcelStatistics
    ) -> np.ndarray:
        """Refine classification for forest parcel using height-NDVI stratification."""
        n_points = len(features.get("ndvi", []))
        labels = np.zeros(n_points, dtype=np.uint8)

        ndvi = features.get("ndvi", np.zeros(n_points))
        height = features.get("height", np.zeros(n_points))

        # Multi-level vegetation classification
        # Level 1: Dense forest (NDVI >= 0.6)
        mask = ndvi >= 0.60
        labels[mask] = int(ASPRSClass.HIGH_VEGETATION)

        # Level 2: Healthy trees (0.5 <= NDVI < 0.6)
        mask = (ndvi >= 0.50) & (ndvi < 0.60) & (labels == 0)
        high_mask = mask & (height > 2.0)
        med_mask = mask & (height <= 2.0)
        labels[high_mask] = int(ASPRSClass.HIGH_VEGETATION)
        labels[med_mask] = int(ASPRSClass.MEDIUM_VEGETATION)

        # Level 3: Moderate vegetation (0.4 <= NDVI < 0.5)
        mask = (ndvi >= 0.40) & (ndvi < 0.50) & (labels == 0)
        med_mask = mask & (height > 1.0)
        low_mask = mask & (height <= 1.0)
        labels[med_mask] = int(ASPRSClass.MEDIUM_VEGETATION)
        labels[low_mask] = int(ASPRSClass.LOW_VEGETATION)

        # Level 4: Grass/understory (0.3 <= NDVI < 0.4)
        mask = (ndvi >= 0.30) & (ndvi < 0.40) & (labels == 0)
        labels[mask] = int(ASPRSClass.LOW_VEGETATION)

        # Level 5: Forest floor/bare soil (NDVI < 0.3)
        mask = (ndvi < 0.30) & (labels == 0)
        labels[mask] = int(ASPRSClass.GROUND)

        return labels

    def _refine_agriculture_parcel(
        self, features: Dict[str, np.ndarray], stats: ParcelStatistics
    ) -> np.ndarray:
        """Refine classification for agricultural parcel."""
        n_points = len(features.get("ndvi", []))
        labels = np.zeros(n_points, dtype=np.uint8)

        ndvi = features.get("ndvi", np.zeros(n_points))
        height = features.get("height", np.zeros(n_points))

        # Crops: mostly low-medium vegetation
        mask = ndvi >= 0.4
        high_mask = mask & (height > 0.5)
        low_mask = mask & (height <= 0.5)
        labels[high_mask] = int(ASPRSClass.MEDIUM_VEGETATION)  # Tall crops
        labels[low_mask] = int(ASPRSClass.LOW_VEGETATION)  # Short crops

        mask = (ndvi >= 0.2) & (ndvi < 0.4)
        labels[mask] = int(ASPRSClass.LOW_VEGETATION)  # Sparse crops

        mask = ndvi < 0.2
        labels[mask] = int(ASPRSClass.GROUND)  # Bare soil

        return labels

    def _refine_building_parcel(
        self, features: Dict[str, np.ndarray], stats: ParcelStatistics
    ) -> np.ndarray:
        """Refine classification for building parcel using verticality."""
        n_points = len(features.get("verticality", []))
        labels = np.full(n_points, int(ASPRSClass.BUILDING), dtype=np.uint8)

        verticality = features.get("verticality", np.zeros(n_points))
        planarity = features.get("planarity", np.zeros(n_points))
        normals = features.get("normals", np.zeros((n_points, 3)))
        normal_z = normals[:, 2] if len(normals) > 0 else np.zeros(n_points)

        # Walls: high verticality + low normal_z
        wall_mask = (verticality > 0.7) & (np.abs(normal_z) < 0.3)
        labels[wall_mask] = int(ASPRSClass.BUILDING)

        # Roofs: high planarity + high normal_z
        roof_mask = (planarity > 0.7) & (np.abs(normal_z) > 0.85)
        labels[roof_mask] = int(ASPRSClass.BUILDING)

        return labels

    def _refine_road_parcel(
        self, features: Dict[str, np.ndarray], stats: ParcelStatistics
    ) -> np.ndarray:
        """Refine classification for road parcel, detecting tree canopy."""
        n_points = len(features.get("ndvi", []))
        labels = np.zeros(n_points, dtype=np.uint8)

        ndvi = features.get("ndvi", np.zeros(n_points))
        height = features.get("height", np.zeros(n_points))
        planarity = features.get("planarity", np.zeros(n_points))

        # Tree canopy over road
        canopy_mask = (ndvi > 0.3) & (height > 2.0)
        high_mask = canopy_mask & (height > 5.0)
        med_mask = canopy_mask & (height <= 5.0)
        labels[high_mask] = int(ASPRSClass.HIGH_VEGETATION)
        labels[med_mask] = int(ASPRSClass.MEDIUM_VEGETATION)

        # Road surface
        road_mask = (planarity > 0.8) & (ndvi < 0.15) & (~canopy_mask)
        labels[road_mask] = int(ASPRSClass.ROAD_SURFACE)

        # Ground (unpaved shoulders)
        ground_mask = labels == 0
        labels[ground_mask] = int(ASPRSClass.GROUND)

        return labels

    def _classify_points_by_features(
        self, features: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Fall-back feature-based classification for mixed/unknown parcels.

        Uses multi-feature decision logic.
        """
        n_points = len(features.get("ndvi", []))
        labels = np.full(n_points, int(ASPRSClass.UNCLASSIFIED), dtype=np.uint8)

        ndvi = features.get("ndvi", np.zeros(n_points))
        height = features.get("height", np.zeros(n_points))
        planarity = features.get("planarity", np.zeros(n_points))
        curvature = features.get("curvature", np.zeros(n_points))

        # Vegetation: high NDVI + low planarity + high curvature
        veg_mask = (ndvi > 0.3) & (planarity < 0.6) & (curvature > 0.2)
        high_mask = veg_mask & (height > 2.0)
        med_mask = veg_mask & (height > 0.5) & (height <= 2.0)
        low_mask = veg_mask & (height <= 0.5)
        labels[high_mask] = int(ASPRSClass.HIGH_VEGETATION)
        labels[med_mask] = int(ASPRSClass.MEDIUM_VEGETATION)
        labels[low_mask] = int(ASPRSClass.LOW_VEGETATION)

        # Buildings: low NDVI + high planarity + low curvature
        building_mask = (ndvi < 0.15) & (planarity > 0.7) & (curvature < 0.1)
        labels[building_mask] = int(ASPRSClass.BUILDING)

        # Ground: everything else with high planarity
        ground_mask = (labels == int(ASPRSClass.UNCLASSIFIED)) & (planarity > 0.75)
        labels[ground_mask] = int(ASPRSClass.GROUND)

        return labels

    def _get_default_label_for_parcel_type(
        self, parcel_type: str, height: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Get default labels based on parcel type."""
        if height is None:
            n_points = 1
        else:
            n_points = len(height)

        if parcel_type == ParcelType.FOREST:
            if height is not None:
                labels = np.where(
                    height > 2.0,
                    int(ASPRSClass.HIGH_VEGETATION),
                    int(ASPRSClass.MEDIUM_VEGETATION),
                )
            else:
                labels = np.full(
                    n_points, int(ASPRSClass.HIGH_VEGETATION), dtype=np.uint8
                )

        elif parcel_type == ParcelType.AGRICULTURE:
            labels = np.full(n_points, int(ASPRSClass.LOW_VEGETATION), dtype=np.uint8)

        elif parcel_type == ParcelType.BUILDING:
            labels = np.full(n_points, int(ASPRSClass.BUILDING), dtype=np.uint8)

        elif parcel_type == ParcelType.ROAD:
            labels = np.full(n_points, int(ASPRSClass.ROAD_SURFACE), dtype=np.uint8)

        elif parcel_type == ParcelType.WATER:
            labels = np.full(n_points, int(ASPRSClass.WATER), dtype=np.uint8)

        else:
            labels = np.full(n_points, int(ASPRSClass.UNCLASSIFIED), dtype=np.uint8)

        return labels

    def _match_bd_foret(
        self, parcel_id: str, parcel_points: np.ndarray, bd_foret: gpd.GeoDataFrame
    ) -> Optional[Dict]:
        """Match parcel with BD Forêt data."""
        # Simple centroid-based matching
        centroid = Point(np.mean(parcel_points[:, 0]), np.mean(parcel_points[:, 1]))

        for idx, row in bd_foret.iterrows():
            if row["geometry"].contains(centroid):
                return {
                    "forest_type": row.get("forest_type", "unknown"),
                    "dominant_species": row.get("dominant_species", "unknown"),
                    "density_category": row.get("density_category", "unknown"),
                    "estimated_height": row.get("estimated_height", 10.0),
                }

        return None

    def _match_rpg(
        self, parcel_id: str, parcel_points: np.ndarray, rpg: gpd.GeoDataFrame
    ) -> Optional[Dict]:
        """Match parcel with RPG data."""
        # Simple centroid-based matching
        centroid = Point(np.mean(parcel_points[:, 0]), np.mean(parcel_points[:, 1]))

        for idx, row in rpg.iterrows():
            if row["geometry"].contains(centroid):
                return {
                    "crop_code": row.get("code_cultu", "unknown"),
                    "crop_category": row.get("crop_category", "unknown"),
                    "parcel_area": row.get("surf_parc", 0.0),
                    "is_bio": row.get("bio", False),
                }

        return None

    def get_parcel_statistics(self, parcel_id: str) -> Optional[ParcelStatistics]:
        """Get cached statistics for a parcel."""
        return self._parcel_stats_cache.get(parcel_id)

    def export_parcel_statistics(self) -> List[Dict]:
        """Export all parcel statistics as list of dictionaries."""
        return [
            {
                "parcel_id": stats.parcel_id,
                "n_points": stats.n_points,
                "area_m2": stats.area_m2,
                "point_density": stats.point_density,
                "mean_ndvi": stats.mean_ndvi,
                "mean_height": stats.mean_height,
                "parcel_type": stats.parcel_type,
                "confidence": (
                    max(stats.confidence_scores.values())
                    if stats.confidence_scores
                    else 0.0
                ),
            }
            for stats in self._parcel_stats_cache.values()
        ]
