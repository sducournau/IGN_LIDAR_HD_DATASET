"""
Ground Truth Refinement Module

Refines ground truth classifications to handle:
1. Water & Roads: Should be on flat ground with specific geometric signatures
2. Vegetation: Better segmentation using curvature and NDVI
3. Buildings: Adjusted polygons to capture all building points

Author: Simon Ducournau
Date: October 19, 2025
"""

import logging
from typing import Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.strtree import STRtree

logger = logging.getLogger(__name__)


class GroundTruthRefinementConfig:
    """Configuration for ground truth refinement."""

    # Water refinement
    WATER_HEIGHT_MAX = 0.3  # Maximum height for water (meters)
    WATER_PLANARITY_MIN = 0.90  # Minimum planarity for water surfaces
    WATER_CURVATURE_MAX = 0.02  # Maximum curvature for flat water
    WATER_NORMAL_Z_MIN = 0.95  # Minimum normal Z for horizontal water

    # Road refinement (DTM-based strict filtering)
    ROAD_HEIGHT_MAX = (
        0.3  # Maximum height for roads (30cm above DTM ground, excludes vegetation)
    )
    ROAD_HEIGHT_MIN = -0.2  # Minimum height (tolerance for slight embedding in ground)
    ROAD_PLANARITY_MIN = 0.85  # Minimum planarity for roads
    ROAD_CURVATURE_MAX = 0.05  # Maximum curvature for smooth roads
    ROAD_NORMAL_Z_MIN = 0.90  # Minimum normal Z for horizontal roads
    ROAD_NDVI_MAX = 0.15  # Maximum NDVI (roads are not vegetation)

    # Vegetation refinement
    VEG_NDVI_MIN = 0.25  # Minimum NDVI for vegetation
    VEG_CURVATURE_MIN = 0.02  # Minimum curvature (complex surfaces)
    VEG_PLANARITY_MAX = 0.60  # Maximum planarity (irregular surfaces)
    VEG_LOW_HEIGHT_MAX = 0.5  # Low vegetation threshold
    VEG_MEDIUM_HEIGHT_MAX = 2.0  # Medium vegetation threshold

    # Tree canopy separation (for road refinement)
    TREE_CANOPY_HEIGHT_MIN = 2.0  # Trees typically start at 2m height above ground

    # Building polygon adjustment
    BUILDING_BUFFER_EXPAND = 0.5  # Expand building polygons by 0.5m
    BUILDING_HEIGHT_MIN = (
        0.5  # ✅ IMPROVED: Abaissé de 1.5→0.5m pour capturer façades basses
    )
    BUILDING_PLANARITY_MIN = 0.60  # ✅ IMPROVED: Abaissé de 0.65→0.60 pour surfaces de bâtiments moins planaires
    BUILDING_NDVI_MAX = (
        0.25  # ✅ IMPROVED: Augmenté de 0.20→0.25 pour tolérer végétation proche
    )
    BUILDING_VERTICAL_THRESHOLD = (
        0.5  # ✅ IMPROVED: Abaissé de 0.6→0.5 pour capturer plus de murs/façades
    )


class GroundTruthRefiner:
    """
    Refines ground truth classifications using geometric and spectral features.

    Key improvements:
    1. Water & Roads: Validates they are on flat ground with correct geometry
    2. Vegetation: Better segmentation using curvature, NDVI, and height
    3. Buildings: Expands polygons to capture all building points
    """

    # ASPRS class definitions
    ASPRS_UNCLASSIFIED = 1
    ASPRS_GROUND = 2
    ASPRS_LOW_VEGETATION = 3
    ASPRS_MEDIUM_VEGETATION = 4
    ASPRS_HIGH_VEGETATION = 5
    ASPRS_BUILDING = 6
    ASPRS_WATER = 9
    ASPRS_ROAD = 11

    def __init__(self, config: Optional[GroundTruthRefinementConfig] = None):
        """Initialize refiner with configuration."""
        self.config = config if config is not None else GroundTruthRefinementConfig()

    def refine_water_classification(
        self,
        labels: np.ndarray,
        points: np.ndarray,
        water_mask: np.ndarray,
        height: Optional[np.ndarray] = None,
        planarity: Optional[np.ndarray] = None,
        curvature: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Refine water classification to ensure points are on flat, horizontal surfaces.

        Water bodies should have:
        - Very low height (near ground level)
        - Very high planarity (flat surfaces)
        - Very low curvature (smooth)
        - Horizontal normals (pointing up)

        Args:
            labels: Current classification labels [N]
            points: Point cloud XYZ coordinates [N, 3]
            water_mask: Boolean mask for ground truth water points [N]
            height: Height above ground [N]
            planarity: Planarity feature [N]
            curvature: Curvature feature [N]
            normals: Normal vectors [N, 3]

        Returns:
            Tuple of (refined_labels, stats_dict)
        """
        stats = {"water_validated": 0, "water_rejected": 0, "water_missing": 0}

        refined = labels.copy()

        if not np.any(water_mask):
            return refined, stats

        logger.info("  Refining water classification...")

        # Extract water candidates from ground truth
        water_candidates = np.where(water_mask)[0]

        # Build validation criteria
        valid_water = np.ones(len(water_candidates), dtype=bool)
        validation_reasons = []

        # Criterion 1: Height (water should be near ground)
        if height is not None:
            height_valid = (height[water_candidates] >= -0.5) & (
                height[water_candidates] <= self.config.WATER_HEIGHT_MAX
            )
            valid_water &= height_valid
            n_height_invalid = np.sum(~height_valid)
            if n_height_invalid > 0:
                validation_reasons.append(f"height: {n_height_invalid} rejected")

        # Criterion 2: Planarity (water should be very flat)
        if planarity is not None:
            # Filter out NaN/Inf before comparison to prevent silent rejection
            candidate_planarity = planarity[water_candidates]
            is_finite = np.isfinite(candidate_planarity)

            planarity_valid = is_finite & (
                candidate_planarity >= self.config.WATER_PLANARITY_MIN
            )
            valid_water &= planarity_valid
            n_planarity_invalid = np.sum(~planarity_valid)

            # Log if artifacts detected
            n_invalid_features = np.sum(~is_finite)
            if n_invalid_features > 0:
                validation_reasons.append(
                    f"planarity: {n_planarity_invalid} rejected "
                    f"({n_invalid_features} with NaN/Inf artifacts)"
                )
                logger.warning(
                    f"      ⚠️  {n_invalid_features} water candidates have "
                    f"invalid planarity (NaN/Inf)"
                )
            elif n_planarity_invalid > 0:
                validation_reasons.append(f"planarity: {n_planarity_invalid} rejected")

        # Criterion 3: Curvature (water should be smooth)
        if curvature is not None:
            curvature_valid = (
                curvature[water_candidates] <= self.config.WATER_CURVATURE_MAX
            )
            valid_water &= curvature_valid
            n_curvature_invalid = np.sum(~curvature_valid)
            if n_curvature_invalid > 0:
                validation_reasons.append(f"curvature: {n_curvature_invalid} rejected")

        # Criterion 4: Normals (water should be horizontal)
        if normals is not None:
            normal_z = np.abs(normals[water_candidates, 2])
            normal_valid = normal_z >= self.config.WATER_NORMAL_Z_MIN
            valid_water &= normal_valid
            n_normal_invalid = np.sum(~normal_valid)
            if n_normal_invalid > 0:
                validation_reasons.append(f"normals: {n_normal_invalid} rejected")

        # Apply validated water classification
        validated_indices = water_candidates[valid_water]
        rejected_indices = water_candidates[~valid_water]

        refined[validated_indices] = self.ASPRS_WATER
        stats["water_validated"] = len(validated_indices)
        stats["water_rejected"] = len(rejected_indices)

        # Log results
        logger.info(f"    ✓ Validated: {stats['water_validated']:,} water points")
        if stats["water_rejected"] > 0:
            logger.info(f"    ✗ Rejected: {stats['water_rejected']:,} water points")
            for reason in validation_reasons:
                logger.info(f"      - {reason}")

        return refined, stats

    def refine_road_classification(
        self,
        labels: np.ndarray,
        points: np.ndarray,
        road_mask: np.ndarray,
        height: Optional[np.ndarray] = None,
        planarity: Optional[np.ndarray] = None,
        curvature: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        ndvi: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Refine road classification to ensure points are on flat, horizontal surfaces.

        Roads should have:
        - Low height (near ground level)
        - High planarity (flat surfaces)
        - Low curvature (smooth pavement)
        - Horizontal normals
        - Low NDVI (not vegetation)

        Args:
            labels: Current classification labels [N]
            points: Point cloud XYZ coordinates [N, 3]
            road_mask: Boolean mask for ground truth road points [N]
            height: Height above ground [N]
            planarity: Planarity feature [N]
            curvature: Curvature feature [N]
            normals: Normal vectors [N, 3]
            ndvi: NDVI values [N]

        Returns:
            Tuple of (refined_labels, stats_dict)
        """
        stats = {"road_validated": 0, "road_rejected": 0, "road_vegetation_override": 0}

        refined = labels.copy()

        if not np.any(road_mask):
            return refined, stats

        logger.info("  Refining road classification...")

        # Extract road candidates from ground truth
        road_candidates = np.where(road_mask)[0]

        # Build validation criteria
        valid_road = np.ones(len(road_candidates), dtype=bool)
        validation_reasons = []

        # Criterion 1: Height (roads should be near ground)
        if height is not None:
            height_valid = (height[road_candidates] >= self.config.ROAD_HEIGHT_MIN) & (
                height[road_candidates] <= self.config.ROAD_HEIGHT_MAX
            )

            # Special case: Points above road are likely tree canopy
            # CRITICAL FIX: Reclassify elevated points as vegetation
            above_road = height[road_candidates] > self.config.ROAD_HEIGHT_MAX
            if np.any(above_road) and ndvi is not None:
                above_road_indices = road_candidates[above_road]
                # Check NDVI to confirm vegetation (threshold 0.25)
                vegetation_ndvi = ndvi[above_road_indices] > 0.25
                if np.any(vegetation_ndvi):
                    veg_indices = above_road_indices[vegetation_ndvi]
                    # Classify by height: >5m = HIGH, 2-5m = MEDIUM
                    high_veg_mask = height[veg_indices] > 5.0
                    refined[veg_indices[high_veg_mask]] = self.ASPRS_HIGH_VEGETATION
                    refined[veg_indices[~high_veg_mask]] = self.ASPRS_MEDIUM_VEGETATION
                    n_reclassified = np.sum(vegetation_ndvi)
                    stats["elevated_roads_reclassified_vegetation"] = n_reclassified
                    logger.debug(
                        f"      🌳 Reclassified {n_reclassified} elevated "
                        f"road points as vegetation"
                    )

            valid_road &= height_valid
            n_height_invalid = np.sum(~height_valid)
            if n_height_invalid > 0:
                validation_reasons.append(
                    f"height: {n_height_invalid} rejected (likely tree canopy)"
                )

        # Criterion 2: Planarity (roads should be very flat)
        if planarity is not None:
            # Filter out NaN/Inf before comparison to prevent silent rejection
            candidate_planarity = planarity[road_candidates]
            is_finite = np.isfinite(candidate_planarity)

            planarity_valid = is_finite & (
                candidate_planarity >= self.config.ROAD_PLANARITY_MIN
            )
            valid_road &= planarity_valid
            n_planarity_invalid = np.sum(~planarity_valid)

            # Log if artifacts detected
            n_invalid_features = np.sum(~is_finite)
            if n_invalid_features > 0:
                validation_reasons.append(
                    f"planarity: {n_planarity_invalid} rejected "
                    f"({n_invalid_features} with NaN/Inf artifacts)"
                )
                logger.warning(
                    f"      ⚠️  {n_invalid_features} road candidates have "
                    f"invalid planarity (NaN/Inf) - features may be corrupted"
                )
            elif n_planarity_invalid > 0:
                validation_reasons.append(f"planarity: {n_planarity_invalid} rejected")

        # Criterion 3: Curvature (roads should be smooth)
        if curvature is not None:
            curvature_valid = (
                curvature[road_candidates] <= self.config.ROAD_CURVATURE_MAX
            )
            valid_road &= curvature_valid
            n_curvature_invalid = np.sum(~curvature_valid)
            if n_curvature_invalid > 0:
                validation_reasons.append(f"curvature: {n_curvature_invalid} rejected")

        # Criterion 4: Normals (roads should be horizontal)
        if normals is not None:
            normal_z = np.abs(normals[road_candidates, 2])
            normal_valid = normal_z >= self.config.ROAD_NORMAL_Z_MIN
            valid_road &= normal_valid
            n_normal_invalid = np.sum(~normal_valid)
            if n_normal_invalid > 0:
                validation_reasons.append(f"normals: {n_normal_invalid} rejected")

            # Criterion 5: NDVI (roads should not be vegetation)
        if ndvi is not None:
            ndvi_valid = ndvi[road_candidates] <= self.config.ROAD_NDVI_MAX

            # Points with high NDVI are tree canopy over road -> reclassify as vegetation
            high_ndvi = ndvi[road_candidates] > self.config.VEG_NDVI_MIN
            if height is not None:
                # High NDVI + elevated = tree canopy (use TREE_CANOPY_HEIGHT_MIN threshold)
                tree_canopy = high_ndvi & (
                    height[road_candidates] > self.config.TREE_CANOPY_HEIGHT_MIN
                )
                if np.any(tree_canopy):
                    canopy_indices = road_candidates[tree_canopy]
                    # Classify based on height
                    high_trees = height[canopy_indices] > 5.0
                    refined[canopy_indices[high_trees]] = self.ASPRS_HIGH_VEGETATION
                    refined[canopy_indices[~high_trees]] = self.ASPRS_MEDIUM_VEGETATION
                    stats["road_vegetation_override"] = np.sum(tree_canopy)

            valid_road &= ndvi_valid
            n_ndvi_invalid = np.sum(~ndvi_valid & ~high_ndvi)  # Exclude tree canopy
            if n_ndvi_invalid > 0:
                validation_reasons.append(
                    f"ndvi: {n_ndvi_invalid} rejected"
                )  # Apply validated road classification
        validated_indices = road_candidates[valid_road]
        rejected_indices = road_candidates[~valid_road]

        refined[validated_indices] = self.ASPRS_ROAD
        stats["road_validated"] = len(validated_indices)
        stats["road_rejected"] = len(rejected_indices)

        # Log results
        logger.info(f"    ✓ Validated: {stats['road_validated']:,} road points")
        if stats["road_rejected"] > 0:
            logger.info(f"    ✗ Rejected: {stats['road_rejected']:,} road points")
            for reason in validation_reasons:
                logger.info(f"      - {reason}")
        if stats["road_vegetation_override"] > 0:
            logger.info(
                f"    🌳 Tree canopy over road: {stats['road_vegetation_override']:,} points → vegetation"
            )

        return refined, stats

    def refine_vegetation_with_features(
        self,
        labels: np.ndarray,
        ndvi: Optional[np.ndarray] = None,
        height: Optional[np.ndarray] = None,
        curvature: Optional[np.ndarray] = None,
        planarity: Optional[np.ndarray] = None,
        sphericity: Optional[np.ndarray] = None,
        roughness: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Refine vegetation classification using NDVI, curvature, sphericity, and geometric features.

        Better vegetation segmentation using:
        - NDVI: Primary vegetation indicator (chlorophyll absorption)
        - Curvature: High curvature for complex vegetation surfaces (branches, leaves)
        - Sphericity: High sphericity for organic, irregular shapes
        - Planarity: Low planarity for irregular vegetation
        - Roughness: High roughness for vegetation surfaces
        - Height: Distinguish low/medium/high vegetation

        This approach does NOT use BD TOPO vegetation - it's purely feature-based.

        Args:
            labels: Current classification labels [N]
            ndvi: NDVI values [N]
            height: Height above ground [N]
            curvature: Curvature feature [N]
            planarity: Planarity feature [N]
            sphericity: Sphericity feature [N] (NEW - better organic shape detection)
            roughness: Surface roughness [N]

        Returns:
            Tuple of (refined_labels, stats_dict)
        """
        stats = {
            "vegetation_added": 0,
            "vegetation_refined": 0,
            "low_veg": 0,
            "medium_veg": 0,
            "high_veg": 0,
        }

        refined = labels.copy()

        if ndvi is None:
            logger.warning("  NDVI not available - skipping vegetation refinement")
            return refined, stats

        logger.info(
            "  Refining vegetation classification with NDVI + curvature + sphericity..."
        )

        # Build vegetation confidence score (multi-feature approach)
        veg_confidence = np.zeros(len(labels), dtype=np.float32)
        total_weight = 0.0

        # 1. NDVI contribution (primary indicator, weight: 0.40)
        ndvi_normalized = np.clip((ndvi - self.config.VEG_NDVI_MIN) / 0.5, 0, 1)
        veg_confidence += ndvi_normalized * 0.40
        total_weight += 0.40

        # 2. Curvature contribution (complex surfaces like branches, weight: 0.20)
        if curvature is not None:
            curv_normalized = np.clip(curvature / 0.1, 0, 1)
            veg_confidence += curv_normalized * 0.20
            total_weight += 0.20

        # 3. Sphericity contribution (organic shapes, weight: 0.20)
        # NEW: Sphericity is excellent for detecting vegetation's irregular, organic geometry
        if sphericity is not None:
            # High sphericity = more isotropic = more organic/vegetation-like
            spher_normalized = np.clip(sphericity, 0, 1)
            veg_confidence += spher_normalized * 0.20
            total_weight += 0.20

        # 4. Planarity contribution (irregular surfaces, weight: 0.10)
        if planarity is not None:
            # Invert: low planarity = high vegetation likelihood
            plan_normalized = 1.0 - np.clip(
                planarity / self.config.VEG_PLANARITY_MAX, 0, 1
            )
            veg_confidence += plan_normalized * 0.10
            total_weight += 0.10

        # 5. Roughness contribution (irregular surfaces, weight: 0.10)
        if roughness is not None:
            # Normalize roughness: higher = more vegetation-like
            rough_normalized = np.clip(roughness / 0.15, 0, 1)
            veg_confidence += rough_normalized * 0.10
            total_weight += 0.10

        # Normalize confidence by actual weights used
        if total_weight > 0:
            veg_confidence /= total_weight

        # Identify vegetation points (confidence > 0.6)
        is_vegetation = veg_confidence > 0.6

        # Classify by height if available
        if height is not None:
            # Low vegetation (0-0.5m)
            low_veg_mask = is_vegetation & (height <= self.config.VEG_LOW_HEIGHT_MAX)
            refined[low_veg_mask] = self.ASPRS_LOW_VEGETATION
            stats["low_veg"] = np.sum(low_veg_mask)

            # Medium vegetation (0.5-2m)
            med_veg_mask = (
                is_vegetation
                & (height > self.config.VEG_LOW_HEIGHT_MAX)
                & (height <= self.config.VEG_MEDIUM_HEIGHT_MAX)
            )
            refined[med_veg_mask] = self.ASPRS_MEDIUM_VEGETATION
            stats["medium_veg"] = np.sum(med_veg_mask)

            # High vegetation (>2m)
            high_veg_mask = is_vegetation & (height > self.config.VEG_MEDIUM_HEIGHT_MAX)
            refined[high_veg_mask] = self.ASPRS_HIGH_VEGETATION
            stats["high_veg"] = np.sum(high_veg_mask)

            stats["vegetation_added"] = np.sum(is_vegetation)
        else:
            # No height - classify as medium vegetation
            refined[is_vegetation] = self.ASPRS_MEDIUM_VEGETATION
            stats["vegetation_added"] = np.sum(is_vegetation)
            stats["medium_veg"] = stats["vegetation_added"]

        # Log results
        logger.info(f"    ✓ Total vegetation: {stats['vegetation_added']:,} points")
        if height is not None:
            logger.info(f"      - Low (0-0.5m): {stats['low_veg']:,}")
            logger.info(f"      - Medium (0.5-2m): {stats['medium_veg']:,}")
            logger.info(f"      - High (>2m): {stats['high_veg']:,}")

        return refined, stats

    def refine_building_with_expanded_polygons(
        self,
        labels: np.ndarray,
        points: np.ndarray,
        building_polygons: gpd.GeoDataFrame,
        height: Optional[np.ndarray] = None,
        planarity: Optional[np.ndarray] = None,
        verticality: Optional[np.ndarray] = None,
        ndvi: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Refine building classification with expanded polygons.

        Problem: Building polygons from BD TOPO don't exactly match point cloud,
        resulting in unclassified points that belong to buildings.

        Solution: Expand building polygons slightly and validate with geometric features.

        Args:
            labels: Current classification labels [N]
            points: Point cloud XYZ coordinates [N, 3]
            building_polygons: GeoDataFrame with building polygons
            height: Height above ground [N]
            planarity: Planarity feature [N]
            verticality: Verticality feature [N]
            ndvi: NDVI values [N]

        Returns:
            Tuple of (refined_labels, stats_dict)
        """
        stats = {
            "building_validated": 0,
            "building_expanded": 0,
            "building_rejected": 0,
        }

        refined = labels.copy()

        if building_polygons is None or len(building_polygons) == 0:
            return refined, stats

        logger.info("  Refining building classification with expanded polygons...")

        # Expand building polygons
        expanded_polygons = building_polygons.copy()
        expanded_polygons["geometry"] = building_polygons.geometry.buffer(
            self.config.BUILDING_BUFFER_EXPAND
        )

        # Build spatial index
        polygons_list = expanded_polygons.geometry.tolist()
        tree = STRtree(polygons_list)

        # Find unclassified or uncertain points
        uncertain_mask = np.isin(labels, [self.ASPRS_UNCLASSIFIED, self.ASPRS_GROUND])
        uncertain_indices = np.where(uncertain_mask)[0]

        if len(uncertain_indices) == 0:
            logger.info("    No uncertain points to refine")
            return refined, stats

        # Query spatial index for each uncertain point
        building_candidates = []
        for idx in uncertain_indices:
            pt = Point(points[idx, 0], points[idx, 1])
            nearby_polygon_indices = tree.query(pt)

            # Check if point is actually within any nearby polygon
            for poly_idx in nearby_polygon_indices:
                if polygons_list[poly_idx].contains(pt):
                    building_candidates.append(idx)
                    break

        if len(building_candidates) == 0:
            logger.info("    No building candidates found in expanded polygons")
            return refined, stats

        building_candidates = np.array(building_candidates)
        logger.info(
            f"    Found {len(building_candidates):,} candidates in expanded polygons"
        )

        # Validate building candidates with geometric features
        valid_building = np.ones(len(building_candidates), dtype=bool)

        # Criterion 1: Height (buildings should be elevated)
        if height is not None:
            height_valid = (
                height[building_candidates] >= self.config.BUILDING_HEIGHT_MIN
            )
            valid_building &= height_valid
            n_height_invalid = np.sum(~height_valid)
            if n_height_invalid > 0:
                logger.info(f"      - Height: {n_height_invalid} rejected (too low)")

        # Criterion 2: Planarity or Verticality (buildings have flat or vertical surfaces)
        if planarity is not None and verticality is not None:
            # Filter out NaN/Inf before comparison to prevent silent rejection
            candidate_planarity = planarity[building_candidates]
            candidate_verticality = verticality[building_candidates]
            is_finite_plan = np.isfinite(candidate_planarity)
            is_finite_vert = np.isfinite(candidate_verticality)

            # Either high planarity (roofs) or high verticality (walls)
            geometry_valid = (
                is_finite_plan
                & (candidate_planarity >= self.config.BUILDING_PLANARITY_MIN)
            ) | (
                is_finite_vert
                & (candidate_verticality >= self.config.BUILDING_VERTICAL_THRESHOLD)
            )
            valid_building &= geometry_valid
            n_geometry_invalid = np.sum(~geometry_valid)

            # Log if artifacts detected
            n_invalid_plan = np.sum(~is_finite_plan)
            n_invalid_vert = np.sum(~is_finite_vert)
            if n_invalid_plan > 0 or n_invalid_vert > 0:
                logger.warning(
                    f"      ⚠️  Building candidates with invalid features: "
                    f"{n_invalid_plan} planarity, {n_invalid_vert} verticality NaN/Inf"
                )
            if n_geometry_invalid > 0:
                logger.info(
                    f"      - Geometry: {n_geometry_invalid} rejected (neither flat nor vertical)"
                )

        # Criterion 3: NDVI (buildings should not be vegetation)
        if ndvi is not None:
            ndvi_valid = ndvi[building_candidates] <= self.config.BUILDING_NDVI_MAX
            valid_building &= ndvi_valid
            n_ndvi_invalid = np.sum(~ndvi_valid)
            if n_ndvi_invalid > 0:
                logger.info(
                    f"      - NDVI: {n_ndvi_invalid} rejected (likely vegetation)"
                )

        # Apply validated building classification
        validated_indices = building_candidates[valid_building]
        rejected_indices = building_candidates[~valid_building]

        refined[validated_indices] = self.ASPRS_BUILDING
        stats["building_validated"] = np.sum(
            labels[validated_indices] == self.ASPRS_BUILDING
        )
        stats["building_expanded"] = np.sum(
            labels[validated_indices] != self.ASPRS_BUILDING
        )
        stats["building_rejected"] = len(rejected_indices)

        # Log results
        logger.info(
            f"    ✓ Expanded buildings: {stats['building_expanded']:,} new building points"
        )
        logger.info(
            f"    ✓ Total validated: {stats['building_validated'] + stats['building_expanded']:,} building points"
        )
        if stats["building_rejected"] > 0:
            logger.info(f"    ✗ Rejected: {stats['building_rejected']:,} candidates")

        return refined, stats

    def resolve_road_building_conflicts(
        self,
        labels: np.ndarray,
        points: np.ndarray,
        buildings_gdf: gpd.GeoDataFrame,
        height: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Resolve conflicts where road points are elevated
        and inside building polygons.

        Road points should NOT be classified as roads when:
        1. Height above ground > 0.3m (not at ground level)
        2. Point is inside a building polygon

        These points should be reclassified as buildings instead.

        Args:
            labels: Current classification labels [N]
            points: Point cloud XYZ coordinates [N, 3]
            buildings_gdf: Building polygons from ground truth
            height: Height above ground [N] (optional)

        Returns:
            Tuple of (refined_labels, stats_dict)
        """
        stats = {"road_to_building": 0, "elevated_roads_checked": 0}

        refined = labels.copy()

        if height is None:
            logger.info(
                "  Height not available - " "skipping road/building conflict resolution"
            )
            return refined, stats

        logger.info("  Resolving road/building conflicts...")

        # Find road points
        road_mask = labels == self.ASPRS_ROAD
        road_indices = np.where(road_mask)[0]

        if len(road_indices) == 0:
            logger.info("    No road points found")
            return refined, stats

        # Find elevated road points (> 0.3m above ground)
        elevated_road_mask = road_mask & (height > 0.3)
        elevated_road_indices = np.where(elevated_road_mask)[0]
        stats["elevated_roads_checked"] = len(elevated_road_indices)

        if len(elevated_road_indices) == 0:
            logger.info("    No elevated road points found")
            return refined, stats

        logger.info(
            f"    Checking {len(elevated_road_indices):,} " f"elevated road points..."
        )

        # Create spatial index for building polygons
        building_tree = STRtree(buildings_gdf.geometry)

        # Check each elevated road point
        n_reclassified = 0
        for idx in elevated_road_indices:
            pt = Point(points[idx, 0], points[idx, 1])

            # Find intersecting building polygons
            possible_matches = building_tree.query(pt)

            for building_idx in possible_matches:
                building_geom = buildings_gdf.iloc[building_idx].geometry

                if building_geom.contains(pt):
                    # Point is inside building polygon and elevated
                    # -> reclassify as building
                    refined[idx] = self.ASPRS_BUILDING
                    n_reclassified += 1
                    break

        stats["road_to_building"] = n_reclassified

        logger.info(
            f"    ✓ Reclassified {n_reclassified:,} elevated "
            f"road points to building"
        )

        return refined, stats

    def refine_all(
        self,
        labels: np.ndarray,
        points: np.ndarray,
        ground_truth_features: Dict[str, gpd.GeoDataFrame],
        features: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Apply all ground truth refinements.

        Args:
            labels: Current classification labels [N]
            points: Point cloud XYZ coordinates [N, 3]
            ground_truth_features: Dictionary of ground truth GeoDataFrames
            features: Dictionary of computed features

        Returns:
            Tuple of (refined_labels, comprehensive_stats)
        """
        refined = labels.copy()
        all_stats = {}

        logger.info("=== Ground Truth Refinement ===")

        # Extract features
        height = features.get("height")
        planarity = features.get("planarity")
        curvature = features.get("curvature")
        normals = features.get("normals")
        ndvi = features.get("ndvi")
        verticality = features.get("verticality")
        sphericity = features.get("sphericity")  # NEW: for vegetation
        roughness = features.get("roughness")  # NEW: for vegetation

        # 1. Refine water (must be flat and near ground)
        if "water" in ground_truth_features:
            water_gdf = ground_truth_features["water"]
            if water_gdf is not None and len(water_gdf) > 0:
                # Create water mask
                water_mask = labels == self.ASPRS_WATER
                refined, water_stats = self.refine_water_classification(
                    refined, points, water_mask, height, planarity, curvature, normals
                )
                all_stats.update(water_stats)

        # 2. Refine roads (must be flat and near ground)
        if "roads" in ground_truth_features:
            roads_gdf = ground_truth_features["roads"]
            if roads_gdf is not None and len(roads_gdf) > 0:
                # Create road mask
                road_mask = labels == self.ASPRS_ROAD
                refined, road_stats = self.refine_road_classification(
                    refined,
                    points,
                    road_mask,
                    height,
                    planarity,
                    curvature,
                    normals,
                    ndvi,
                )
                all_stats.update(road_stats)

        # 3. Refine vegetation (FEATURE-BASED ONLY - no BD TOPO vegetation)
        # NEW: Use sphericity and roughness for better organic shape detection
        logger.info(
            "  Using feature-based vegetation classification (NDVI + curvature + sphericity)"
        )
        refined, veg_stats = self.refine_vegetation_with_features(
            refined, ndvi, height, curvature, planarity, sphericity, roughness
        )
        all_stats.update(veg_stats)

        # 4. Refine buildings (expand polygons to capture all building points)
        if "buildings" in ground_truth_features:
            buildings_gdf = ground_truth_features["buildings"]
            if buildings_gdf is not None and len(buildings_gdf) > 0:
                refined, building_stats = self.refine_building_with_expanded_polygons(
                    refined, points, buildings_gdf, height, planarity, verticality, ndvi
                )
                all_stats.update(building_stats)

        # 5. Resolve road/building conflicts (elevated road points inside building polygons)
        if "buildings" in ground_truth_features and "roads" in ground_truth_features:
            buildings_gdf = ground_truth_features["buildings"]
            if buildings_gdf is not None and len(buildings_gdf) > 0:
                refined, conflict_stats = self.resolve_road_building_conflicts(
                    refined, points, buildings_gdf, height
                )
                all_stats.update(conflict_stats)

        # Log summary
        logger.info("=== Refinement Summary ===")
        total_refined = np.sum(labels != refined)
        logger.info(f"Total points refined: {total_refined:,}")

        return refined, all_stats
