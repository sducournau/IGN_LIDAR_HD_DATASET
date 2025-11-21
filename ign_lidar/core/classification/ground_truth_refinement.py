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

from .constants import ASPRSClass

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
        0.20  # FURTHER REDUCED from 0.25 - Ultra-strict ground level (20cm)
    )
    ROAD_HEIGHT_MIN = -0.2  # Minimum height (tolerance for slight embedding in ground)
    ROAD_PLANARITY_MIN = 0.85  # Minimum planarity for roads
    ROAD_CURVATURE_MAX = 0.05  # Maximum curvature for smooth roads
    ROAD_NORMAL_Z_MIN = 0.90  # Minimum normal Z for horizontal roads
    ROAD_NDVI_MAX = 0.12  # REDUCED from 0.15 - Stricter vegetation exclusion

    # Vegetation refinement
    VEG_NDVI_MIN = 0.25  # Minimum NDVI for vegetation
    VEG_CURVATURE_MIN = 0.02  # Minimum curvature (complex surfaces)
    VEG_PLANARITY_MAX = 0.60  # Maximum planarity (irregular surfaces)
    VEG_LOW_HEIGHT_MAX = 0.5  # Low vegetation threshold
    VEG_MEDIUM_HEIGHT_MAX = 2.0  # Medium vegetation threshold

    # Tree canopy separation (for road refinement)
    TREE_CANOPY_HEIGHT_MIN = 2.0  # Trees typically start at 2m height above ground

    # Building polygon adjustment
    BUILDING_BUFFER_EXPAND = 0.5  # Expand building polygons by 0.5m (fixed)
    BUILDING_BUFFER_MIN = 0.5  # Minimum buffer for adaptive mode (meters)
    BUILDING_BUFFER_MAX = 4.0  # INCREASED from 3.5 - Better capture of large buildings
    BUILDING_BUFFER_SCALE = 0.08  # INCREASED from 0.06 - Scale factor: 8% of perimeter
    USE_ADAPTIVE_BUFFERS = True  # Use adaptive sizing (recommended)

    # Height-stratified validation (facades vs roofs)
    USE_FACADE_SPECIFIC_VALIDATION = True  # Separate criteria for facades
    FACADE_TRANSITION_HEIGHT = 2.5  # Better roof/facade separation

    # Facade-specific criteria (more relaxed for better capture)
    FACADE_HEIGHT_MIN = 0.15  # LOWERED from 0.2 - Capture very low foundation edges
    FACADE_VERTICAL_MIN = (
        0.25  # LOWERED from 0.30 - More permissive for complex facades
    )
    FACADE_PLANARITY_MAX = (
        0.80  # INCREASED from 0.75 - More permissive for varied facades
    )

    # Roof overhang detection
    OVERHANG_DETECTION_ENABLED = True  # Enable overhang detection
    OVERHANG_HEIGHT_MIN = 1.8  # LOWERED from 2.0 - Capture lower overhangs
    OVERHANG_PLANARITY_MIN = (
        0.45  # LOWERED from 0.50 - More permissive for complex roofs
    )
    OVERHANG_VERTICAL_MAX = 0.65  # INCREASED from 0.60 - Allow steeper roof angles

    # Roof-specific criteria (stricter for horizontal surfaces)
    ROOF_HEIGHT_MIN = 0.5  # Standard minimum
    ROOF_PLANARITY_MIN = 0.60  # Roofs should be planar

    # Legacy thresholds (used when stratification disabled)
    BUILDING_HEIGHT_MIN = 0.5
    BUILDING_PLANARITY_MIN = 0.60
    BUILDING_NDVI_MAX = 0.25
    BUILDING_VERTICAL_THRESHOLD = 0.5


class GroundTruthRefiner:
    """
    Refines ground truth classifications using geometric and spectral features.

    Key improvements:
    1. Water & Roads: Validates they are on flat ground with correct geometry
    2. Vegetation: Better segmentation using curvature, NDVI, and height
    3. Buildings: Expands polygons to capture all building points
    """

    # ASPRS class definitions
    # Use ASPRSClass from constants module

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

        refined[validated_indices] = int(ASPRSClass.WATER)
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
        Refine road classification to ensure ALL points are on flat, horizontal surfaces.

        CRITICAL: Roads MUST be on ground level. Any elevated point is reclassified.

        Roads should have:
        - Very low height (near ground level, max 25cm)
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
        stats = {
            "road_validated": 0,
            "road_rejected": 0,
            "elevated_to_vegetation": 0,
            "elevated_to_unclassified": 0,
        }

        refined = labels.copy()

        if not np.any(road_mask):
            return refined, stats

        logger.info(
            "  Refining road classification (strict ground-level enforcement)..."
        )

        # Extract road candidates from ground truth
        road_candidates = np.where(road_mask)[0]

        # Build validation criteria
        valid_road = np.ones(len(road_candidates), dtype=bool)
        validation_reasons = []

        # CRITICAL Criterion 1: Height - MUST be near ground
        if height is not None:
            candidate_height = height[road_candidates]

            # Strict height validation
            height_valid = (candidate_height >= self.config.ROAD_HEIGHT_MIN) & (
                candidate_height <= self.config.ROAD_HEIGHT_MAX
            )

            # AGGRESSIVE RECLASSIFICATION: Points above road threshold
            above_road = candidate_height > self.config.ROAD_HEIGHT_MAX

            if np.any(above_road):
                above_indices = road_candidates[above_road]
                n_above = len(above_indices)

                logger.info(f"      ⚠️  Found {n_above:,} elevated points above roads")

                # Classify elevated points based on height and NDVI
                if ndvi is not None:
                    # Check NDVI to determine if vegetation or other
                    above_ndvi = ndvi[above_indices]

                    # High NDVI = vegetation
                    is_vegetation = above_ndvi > 0.20  # Lower threshold for safety

                    veg_indices = above_indices[is_vegetation]
                    non_veg_indices = above_indices[~is_vegetation]

                    if len(veg_indices) > 0:
                        # Classify by height: >5m = HIGH, 2-5m = MEDIUM, <2m = LOW
                        veg_heights = height[veg_indices]
                        high_veg = veg_heights > 5.0
                        med_veg = (veg_heights >= 2.0) & (veg_heights <= 5.0)
                        low_veg = veg_heights < 2.0

                        refined[veg_indices[high_veg]] = int(ASPRSClass.HIGH_VEGETATION)
                        refined[veg_indices[med_veg]] = int(
                            ASPRSClass.MEDIUM_VEGETATION
                        )
                        refined[veg_indices[low_veg]] = int(ASPRSClass.LOW_VEGETATION)

                        stats["elevated_to_vegetation"] = len(veg_indices)
                        logger.info(
                            f"         └─ Reclassified {len(veg_indices):,} as vegetation "
                            f"(H:{np.sum(high_veg)}, M:{np.sum(med_veg)}, L:{np.sum(low_veg)})"
                        )

                    if len(non_veg_indices) > 0:
                        # Non-vegetation elevated points → unclassified (may be poles, signs, etc.)
                        refined[non_veg_indices] = int(ASPRSClass.UNCLASSIFIED)
                        stats["elevated_to_unclassified"] = len(non_veg_indices)
                        logger.info(
                            f"         └─ Reclassified {len(non_veg_indices):,} as unclassified "
                            f"(low NDVI, likely infrastructure)"
                        )
                else:
                    # No NDVI available - conservative reclassification
                    # Assume vegetation if height > 2m, otherwise unclassified
                    above_heights = height[above_indices]
                    likely_vegetation = above_heights > 2.0

                    veg_indices = above_indices[likely_vegetation]
                    other_indices = above_indices[~likely_vegetation]

                    if len(veg_indices) > 0:
                        refined[veg_indices] = int(ASPRSClass.MEDIUM_VEGETATION)
                        stats["elevated_to_vegetation"] = len(veg_indices)
                        logger.info(
                            f"         └─ Reclassified {len(veg_indices):,} as vegetation "
                            f"(height-based, no NDVI)"
                        )

                    if len(other_indices) > 0:
                        refined[other_indices] = int(ASPRSClass.UNCLASSIFIED)
                        stats["elevated_to_unclassified"] = len(other_indices)
                        logger.info(
                            f"         └─ Reclassified {len(other_indices):,} as unclassified"
                        )

            valid_road &= height_valid
            n_height_invalid = np.sum(~height_valid)
            if n_height_invalid > 0:
                validation_reasons.append(
                    f"height: {n_height_invalid} rejected (not ground-level)"
                )

        # Criterion 2: Planarity (roads should be very flat)
        if planarity is not None:
            # Filter out NaN/Inf before comparison
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
            candidate_ndvi = ndvi[road_candidates]

            # Robust NaN/Inf handling for NDVI
            is_finite_ndvi = np.isfinite(candidate_ndvi)
            ndvi_robust = np.where(is_finite_ndvi, candidate_ndvi, 0.0)

            ndvi_valid = ndvi_robust <= self.config.ROAD_NDVI_MAX

            # Vegetation detection on roads:
            # 1. Moderate NDVI (0.12-0.40) = grass, shrubs on/near road
            moderate_ndvi = (ndvi_robust > self.config.ROAD_NDVI_MAX) & (
                ndvi_robust <= 0.40
            )
            # 2. High NDVI (>0.40) = trees, dense vegetation over road
            high_ndvi = ndvi_robust > 0.40

            # 3. Very low NDVI but elevated = likely artificial structures (poles, signs)
            very_low_ndvi = ndvi_robust < 0.05

            if height is not None and np.any(moderate_ndvi):
                # Moderate NDVI: Precise height-based classification
                mod_indices = road_candidates[moderate_ndvi]
                mod_heights = height[mod_indices]

                # Low vegetation on road surface (grass, small plants)
                low_grass = mod_heights <= self.config.VEG_LOW_HEIGHT_MAX
                refined[mod_indices[low_grass]] = int(ASPRSClass.LOW_VEGETATION)

                # Medium vegetation near road (shrubs, small trees)
                med_grass = (mod_heights > self.config.VEG_LOW_HEIGHT_MAX) & (
                    mod_heights <= self.config.VEG_MEDIUM_HEIGHT_MAX
                )
                refined[mod_indices[med_grass]] = int(ASPRSClass.MEDIUM_VEGETATION)

                logger.info(
                    f"      🌱 Reclassified {np.sum(low_grass)} as low vegetation (grass on road)"
                )

            if height is not None and np.any(high_ndvi):
                # High NDVI = tree canopy over road
                tree_indices = road_candidates[high_ndvi]
                tree_heights = height[tree_indices]

                # Classify based on height
                high_trees = tree_heights > 5.0
                med_trees = (tree_heights >= 2.0) & (tree_heights <= 5.0)

                refined[tree_indices[high_trees]] = int(ASPRSClass.HIGH_VEGETATION)
                refined[tree_indices[med_trees]] = int(ASPRSClass.MEDIUM_VEGETATION)

                n_trees = len(tree_indices)
                logger.info(
                    f"      🌳 Reclassified {n_trees} as tree canopy over road "
                    f"(H:{np.sum(high_trees)}, M:{np.sum(med_trees)})"
                )

            valid_road &= ndvi_valid
            n_ndvi_invalid = np.sum(~ndvi_valid)
            if n_ndvi_invalid > 0:
                validation_reasons.append(f"ndvi: {n_ndvi_invalid} rejected")

        # Apply validated road classification
        validated_indices = road_candidates[valid_road]
        rejected_indices = road_candidates[~valid_road]

        refined[validated_indices] = int(ASPRSClass.ROAD_SURFACE)
        stats["road_validated"] = len(validated_indices)
        stats["road_rejected"] = len(rejected_indices)

        # Log results
        logger.info(
            f"    ✓ Validated: {stats['road_validated']:,} road points (ground-level)"
        )
        if stats["road_rejected"] > 0:
            logger.info(f"    ✗ Rejected: {stats['road_rejected']:,} road points")
            for reason in validation_reasons:
                logger.info(f"      - {reason}")

        # Summary of reclassifications
        n_reclassified = (
            stats["elevated_to_vegetation"] + stats["elevated_to_unclassified"]
        )
        if n_reclassified > 0:
            logger.info(
                f"    ♻️  Reclassified {n_reclassified:,} elevated points "
                f"(vegetation: {stats['elevated_to_vegetation']:,}, "
                f"other: {stats['elevated_to_unclassified']:,})"
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
            sphericity: Sphericity feature [N] (organic shape detection)
            roughness: Surface roughness [N]

        Returns:
            Tuple of (refined_labels, stats_dict)
        """
        stats = {
            "vegetation_added": 0,
            "vegetation_corrected": 0,
            "low_veg": 0,
            "medium_veg": 0,
            "high_veg": 0,
            "false_buildings_corrected": 0,
        }

        refined = labels.copy()

        if ndvi is None:
            logger.warning("  NDVI not available - skipping vegetation refinement")
            return refined, stats

        logger.info(
            "  Refining vegetation classification with NDVI + geometric features..."
        )

        # Build vegetation confidence score (multi-feature approach)
        veg_confidence = np.zeros(len(labels), dtype=np.float32)
        total_weight = 0.0

        # 1. NDVI contribution (primary indicator, weight: 0.45 - increased)
        # Robust handling of NaN/Inf
        ndvi_valid = np.isfinite(ndvi)
        ndvi_safe = np.where(ndvi_valid, ndvi, 0.0)
        ndvi_normalized = np.clip((ndvi_safe - self.config.VEG_NDVI_MIN) / 0.5, 0, 1)
        veg_confidence += ndvi_normalized * 0.45
        total_weight += 0.45

        # 2. Curvature contribution (complex surfaces like branches, weight: 0.20)
        if curvature is not None:
            curv_valid = np.isfinite(curvature)
            curv_safe = np.where(curv_valid, curvature, 0.0)
            curv_normalized = np.clip(curv_safe / 0.1, 0, 1)
            veg_confidence += curv_normalized * 0.20
            total_weight += 0.20

        # 3. Sphericity contribution (organic shapes, weight: 0.15)
        if sphericity is not None:
            # High sphericity = more isotropic = more organic/vegetation-like
            spher_valid = np.isfinite(sphericity)
            spher_safe = np.where(spher_valid, sphericity, 0.0)
            spher_normalized = np.clip(spher_safe, 0, 1)
            veg_confidence += spher_normalized * 0.15
            total_weight += 0.15

        # 4. Planarity contribution (irregular surfaces, weight: 0.10)
        if planarity is not None:
            plan_valid = np.isfinite(planarity)
            plan_safe = np.where(plan_valid, planarity, 1.0)
            # Invert: low planarity = high vegetation likelihood
            plan_normalized = 1.0 - np.clip(
                plan_safe / self.config.VEG_PLANARITY_MAX, 0, 1
            )
            veg_confidence += plan_normalized * 0.10
            total_weight += 0.10

        # 5. Roughness contribution (irregular surfaces, weight: 0.10)
        if roughness is not None:
            rough_valid = np.isfinite(roughness)
            rough_safe = np.where(rough_valid, roughness, 0.0)
            # Normalize roughness: higher = more vegetation-like
            rough_normalized = np.clip(rough_safe / 0.15, 0, 1)
            veg_confidence += rough_normalized * 0.10
            total_weight += 0.10

        # Normalize confidence by actual weights used
        if total_weight > 0:
            veg_confidence /= total_weight

        # Two-tier classification strategy:
        # Tier 1: High confidence vegetation (confidence > 0.65)
        high_confidence_veg = veg_confidence > 0.65

        # Tier 2: Moderate confidence + unclassified/ground (confidence > 0.50)
        # This captures vegetation that was missed but has reasonable evidence
        moderate_confidence_veg = (veg_confidence > 0.50) & (veg_confidence <= 0.65)
        can_reclassify = np.isin(
            labels, [int(ASPRSClass.UNCLASSIFIED), int(ASPRSClass.GROUND)]
        )
        moderate_safe = moderate_confidence_veg & can_reclassify

        # Combined vegetation mask
        is_vegetation = high_confidence_veg | moderate_safe

        # Correction: Find buildings/roads misclassified as vegetation
        if height is not None and planarity is not None:
            # Buildings with high NDVI but also high planarity + elevated → likely green roofs
            potential_green_roofs = (
                (labels == int(ASPRSClass.BUILDING))
                & (ndvi_safe > 0.30)
                & (planarity > 0.70)
                & (height > 2.0)
            )
            # Keep as buildings but log
            if np.any(potential_green_roofs):
                n_green_roofs = np.sum(potential_green_roofs)
                logger.info(
                    f"      ℹ️  Detected {n_green_roofs:,} potential green roofs "
                    f"(high NDVI + planar + elevated)"
                )

        # Classify by height if available
        if height is not None:
            height_safe = np.where(np.isfinite(height), height, 0.0)

            # Low vegetation (0-0.5m)
            low_veg_mask = is_vegetation & (
                height_safe <= self.config.VEG_LOW_HEIGHT_MAX
            )
            refined[low_veg_mask] = int(ASPRSClass.LOW_VEGETATION)
            stats["low_veg"] = np.sum(low_veg_mask)

            # Medium vegetation (0.5-2m)
            med_veg_mask = (
                is_vegetation
                & (height_safe > self.config.VEG_LOW_HEIGHT_MAX)
                & (height_safe <= self.config.VEG_MEDIUM_HEIGHT_MAX)
            )
            refined[med_veg_mask] = int(ASPRSClass.MEDIUM_VEGETATION)
            stats["medium_veg"] = np.sum(med_veg_mask)

            # High vegetation (>2m)
            high_veg_mask = is_vegetation & (
                height_safe > self.config.VEG_MEDIUM_HEIGHT_MAX
            )
            refined[high_veg_mask] = int(ASPRSClass.HIGH_VEGETATION)
            stats["high_veg"] = np.sum(high_veg_mask)

            stats["vegetation_added"] = np.sum(is_vegetation)

            # Count corrections
            was_unclassified = np.isin(
                labels[is_vegetation],
                [int(ASPRSClass.UNCLASSIFIED), int(ASPRSClass.GROUND)],
            )
            stats["vegetation_corrected"] = np.sum(~was_unclassified)

        else:
            # No height - classify as medium vegetation
            refined[is_vegetation] = int(ASPRSClass.MEDIUM_VEGETATION)
            stats["vegetation_added"] = np.sum(is_vegetation)
            stats["medium_veg"] = stats["vegetation_added"]

        # Log results
        logger.info(f"    ✓ Total vegetation: {stats['vegetation_added']:,} points")
        if height is not None:
            logger.info(f"      - Low (≤0.5m): {stats['low_veg']:,}")
            logger.info(f"      - Medium (0.5-2m): {stats['medium_veg']:,}")
            logger.info(f"      - High (>2m): {stats['high_veg']:,}")
        if stats["vegetation_corrected"] > 0:
            logger.info(
                f"      - Corrected: {stats['vegetation_corrected']:,} previously misclassified"
            )

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
        Uses height-stratified validation to separately handle facades, roofs, and overhangs.

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
            "facades_captured": 0,
            "roofs_captured": 0,
            "overhangs_captured": 0,
        }

        refined = labels.copy()

        if building_polygons is None or len(building_polygons) == 0:
            return refined, stats

        logger.info("  Refining building classification with expanded polygons...")

        # Compute buffer distances (adaptive or fixed)
        expanded_polygons = building_polygons.copy()
        if self.config.USE_ADAPTIVE_BUFFERS:
            # Adaptive buffer based on building area
            areas = building_polygons.geometry.area
            # Buffer = 6% of perimeter (sqrt(area) approximates side length)
            buffer_distances = np.clip(
                (areas**0.5) * self.config.BUILDING_BUFFER_SCALE,
                self.config.BUILDING_BUFFER_MIN,
                self.config.BUILDING_BUFFER_MAX,
            )
            # Apply per-building buffers
            expanded_geoms = [
                geom.buffer(dist)
                for geom, dist in zip(building_polygons.geometry, buffer_distances)
            ]
            expanded_polygons["geometry"] = expanded_geoms
            logger.info(
                f"    Using adaptive buffers: "
                f"{buffer_distances.min():.2f}m - "
                f"{buffer_distances.max():.2f}m "
                f"(mean: {buffer_distances.mean():.2f}m)"
            )
        else:
            # Fixed buffer for all buildings
            expanded_polygons["geometry"] = building_polygons.geometry.buffer(
                self.config.BUILDING_BUFFER_EXPAND
            )
            logger.info(
                f"    Using fixed buffer: " f"{self.config.BUILDING_BUFFER_EXPAND}m"
            )

        # Build spatial index
        polygons_list = expanded_polygons.geometry.tolist()
        tree = STRtree(polygons_list)

        # Find unclassified or uncertain points
        uncertain_mask = np.isin(
            labels, [int(ASPRSClass.UNCLASSIFIED), int(ASPRSClass.GROUND)]
        )
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

        # Height-stratified validation
        if (
            height is not None
            and planarity is not None
            and verticality is not None
            and self.config.USE_FACADE_SPECIFIC_VALIDATION
        ):
            logger.info(
                "    Using height-stratified validation (facades/roofs/overhangs)"
            )

            # Stratify by height
            candidate_heights = height[building_candidates]

            # Facades: low points (ground level to ~2.5m)
            facade_mask = candidate_heights < self.config.FACADE_TRANSITION_HEIGHT

            # Roofs: higher points with high planarity
            roof_mask = candidate_heights >= self.config.FACADE_TRANSITION_HEIGHT

            # Overhangs: high points with mixed planarity/verticality
            overhang_mask = (
                roof_mask
                & (candidate_heights >= self.config.OVERHANG_HEIGHT_MIN)
                & self.config.OVERHANG_DETECTION_ENABLED
            )

            valid_building = np.zeros(len(building_candidates), dtype=bool)

            # Process facades (multi-criteria validation)
            if np.any(facade_mask):
                candidate_vert = verticality[building_candidates[facade_mask]]
                candidate_plan = planarity[building_candidates[facade_mask]]
                candidate_ndvi = (
                    ndvi[building_candidates[facade_mask]] if ndvi is not None else None
                )

                # Robust NaN/Inf handling with intelligent fallback
                is_finite_vert = np.isfinite(candidate_vert)
                is_finite_plan = np.isfinite(candidate_plan)

                # Smart fallback: if verticality is NaN but planarity is low, likely vertical
                vert_robust = np.where(
                    is_finite_vert,
                    candidate_vert,
                    np.where(is_finite_plan & (candidate_plan < 0.3), 0.8, 0.0),
                )
                plan_robust = np.where(
                    is_finite_plan, candidate_plan, 1.0
                )  # Assume non-planar if unknown

                # Very relaxed criteria for facades to maximize capture
                height_valid = (
                    candidate_heights[facade_mask] >= self.config.FACADE_HEIGHT_MIN
                )

                # Multi-criteria geometry validation (more permissive)
                # Accept if: vertical walls OR planar elements OR reasonable geometry
                is_vertical_wall = vert_robust >= self.config.FACADE_VERTICAL_MIN
                is_planar_element = plan_robust <= self.config.FACADE_PLANARITY_MAX
                has_reasonable_geometry = (vert_robust > 0.15) | (
                    plan_robust < 0.85
                )  # Very permissive

                geometry_valid = (
                    is_vertical_wall | is_planar_element | has_reasonable_geometry
                )

                # NDVI check (not vegetation) - relaxed for facades near vegetation
                if candidate_ndvi is not None:
                    # Allow slightly higher NDVI for facades (may have climbing plants)
                    ndvi_valid = candidate_ndvi <= self.config.BUILDING_NDVI_MAX + 0.05
                else:
                    ndvi_valid = True

                facade_valid = height_valid & geometry_valid & ndvi_valid
                valid_building[facade_mask] = facade_valid
                stats["facades_captured"] = np.sum(facade_valid)

                if stats["facades_captured"] > 0:
                    pct = 100.0 * stats["facades_captured"] / np.sum(facade_mask)
                    logger.info(
                        f"      🏢 Facades: {stats['facades_captured']:,} points validated ({pct:.1f}% of candidates)"
                    )

            # Process roofs (non-overhangs)
            roof_only_mask = roof_mask & ~overhang_mask
            if np.any(roof_only_mask):
                candidate_plan = planarity[building_candidates[roof_only_mask]]
                candidate_ndvi = (
                    ndvi[building_candidates[roof_only_mask]]
                    if ndvi is not None
                    else None
                )

                # Robust NaN handling
                is_finite_plan = np.isfinite(candidate_plan)
                plan_robust = np.where(is_finite_plan, candidate_plan, 0.0)

                # Stricter criteria for roofs
                height_valid = (
                    candidate_heights[roof_only_mask] >= self.config.ROOF_HEIGHT_MIN
                )
                geometry_valid = plan_robust >= self.config.ROOF_PLANARITY_MIN
                ndvi_valid = (
                    (candidate_ndvi <= self.config.BUILDING_NDVI_MAX)
                    if candidate_ndvi is not None
                    else True
                )

                roof_valid = height_valid & geometry_valid & ndvi_valid
                valid_building[roof_only_mask] = roof_valid
                stats["roofs_captured"] = np.sum(roof_valid)

                if stats["roofs_captured"] > 0:
                    logger.info(
                        f"      🏠 Roofs: {stats['roofs_captured']:,} points validated"
                    )

            # Process overhangs (roof edges extending beyond building footprint)
            if np.any(overhang_mask):
                candidate_plan = planarity[building_candidates[overhang_mask]]
                candidate_vert = verticality[building_candidates[overhang_mask]]
                candidate_ndvi = (
                    ndvi[building_candidates[overhang_mask]]
                    if ndvi is not None
                    else None
                )

                # Robust NaN handling with intelligent defaults
                is_finite_plan = np.isfinite(candidate_plan)
                is_finite_vert = np.isfinite(candidate_vert)

                # For overhangs, default to moderate planarity if unknown
                plan_robust = np.where(is_finite_plan, candidate_plan, 0.5)
                vert_robust = np.where(is_finite_vert, candidate_vert, 0.3)

                # Relaxed criteria for overhangs (can be sloped, complex geometry)
                height_valid = (
                    candidate_heights[overhang_mask] >= self.config.OVERHANG_HEIGHT_MIN
                )

                # Multi-criteria validation for overhangs:
                # 1. Moderate planarity (horizontal-ish roof edges)
                has_planar_geometry = plan_robust >= self.config.OVERHANG_PLANARITY_MIN
                # 2. Low to moderate verticality (sloped roofs, gutters)
                has_sloped_geometry = vert_robust <= self.config.OVERHANG_VERTICAL_MAX
                # 3. Mixed geometry (transition zones between roof and facade)
                has_transition_geometry = (plan_robust > 0.35) & (vert_robust < 0.70)

                geometry_valid = (
                    has_planar_geometry | has_sloped_geometry | has_transition_geometry
                )

                # NDVI validation (allow slightly higher for roof vegetation like moss)
                if candidate_ndvi is not None:
                    ndvi_valid = candidate_ndvi <= self.config.BUILDING_NDVI_MAX + 0.03
                else:
                    ndvi_valid = True

                overhang_valid = height_valid & geometry_valid & ndvi_valid
                valid_building[overhang_mask] = overhang_valid
                stats["overhangs_captured"] = np.sum(overhang_valid)

                if stats["overhangs_captured"] > 0:
                    pct = 100.0 * stats["overhangs_captured"] / np.sum(overhang_mask)
                    logger.info(
                        f"      🏘️  Overhangs: {stats['overhangs_captured']:,} points validated ({pct:.1f}% of candidates)"
                    )

        else:
            # Fallback: Legacy validation without height stratification
            logger.info("    Using legacy validation (no height stratification)")
            valid_building = np.ones(len(building_candidates), dtype=bool)

            # Height criterion
            if height is not None:
                height_valid = (
                    height[building_candidates] >= self.config.BUILDING_HEIGHT_MIN
                )
                valid_building &= height_valid
                n_height_invalid = np.sum(~height_valid)
                if n_height_invalid > 0:
                    logger.info(
                        f"      - Height: {n_height_invalid} rejected (too low)"
                    )

            # Geometry criterion
            if planarity is not None and verticality is not None:
                candidate_planarity = planarity[building_candidates]
                candidate_verticality = verticality[building_candidates]
                is_finite_plan = np.isfinite(candidate_planarity)
                is_finite_vert = np.isfinite(candidate_verticality)

                planarity_robust = np.where(is_finite_plan, candidate_planarity, 0.0)
                verticality_robust = np.where(
                    is_finite_vert,
                    candidate_verticality,
                    np.maximum(0.0, 1.0 - planarity_robust),
                )

                geometry_valid = (
                    planarity_robust >= self.config.BUILDING_PLANARITY_MIN
                ) | (verticality_robust >= self.config.BUILDING_VERTICAL_THRESHOLD)
                valid_building &= geometry_valid

            # NDVI criterion
            if ndvi is not None:
                ndvi_valid = ndvi[building_candidates] <= self.config.BUILDING_NDVI_MAX
                valid_building &= ndvi_valid

        # Apply validated building classification
        validated_indices = building_candidates[valid_building]
        rejected_indices = building_candidates[~valid_building]

        refined[validated_indices] = int(ASPRSClass.BUILDING)
        stats["building_validated"] = np.sum(
            labels[validated_indices] == int(ASPRSClass.BUILDING)
        )
        stats["building_expanded"] = np.sum(
            labels[validated_indices] != int(ASPRSClass.BUILDING)
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

    def recover_missing_facades(
        self,
        labels: np.ndarray,
        points: np.ndarray,
        building_polygons: gpd.GeoDataFrame,
        height: np.ndarray,
        verticality: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Aggressive recovery pass for missing facade points.

        Finds unclassified vertical points near buildings that were
        missed by polygon expansion. This is a final recovery pass
        after standard classification.

        Strategy:
        1. Find unclassified points with vertical geometry
        2. Check if they're within search radius of building polygons
        3. Validate with height constraints
        4. Classify as building

        Args:
            labels: Current classification labels [N]
            points: Point cloud XYZ coordinates [N, 3]
            building_polygons: Building polygons from ground truth
            height: Height above ground [N]
            verticality: Verticality feature [N]

        Returns:
            Tuple of (refined_labels, stats_dict)
        """
        stats = {
            "facades_recovered": 0,
            "low_walls_recovered": 0,
            "vertical_elements_recovered": 0,
        }
        refined = labels.copy()

        if building_polygons is None or len(building_polygons) == 0:
            return refined, stats

        logger.info("  🔍 Aggressive facade recovery pass...")

        # Find candidates: unclassified + vertical + reasonable height
        unclassified = labels == int(ASPRSClass.UNCLASSIFIED)

        # Multi-tier height validation for different building elements
        # Tier 1: Very low walls and foundations (0.1-1.0m)
        low_wall_mask = (height > 0.1) & (height <= 1.0)
        # Tier 2: Normal facades (1.0-10.0m)
        facade_mask = (height > 1.0) & (height <= 10.0)
        # Tier 3: High facades and chimneys (10.0-20.0m)
        high_mask = (height > 10.0) & (height <= 20.0)

        height_valid = low_wall_mask | facade_mask | high_mask

        # Relaxed verticality threshold with NaN handling
        vert_valid = np.isfinite(verticality) & (verticality > 0.25)  # Very relaxed

        facade_candidates = unclassified & height_valid & vert_valid
        n_candidates = np.sum(facade_candidates)

        if n_candidates == 0:
            logger.info("    No facade candidates found")
            return refined, stats

        logger.info(f"    Found {n_candidates:,} potential facade/wall points")

        # Separate candidates by type
        low_wall_candidates = facade_candidates & low_wall_mask
        facade_candidates_normal = facade_candidates & facade_mask
        high_candidates = facade_candidates & high_mask

        n_low = np.sum(low_wall_candidates)
        n_normal = np.sum(facade_candidates_normal)
        n_high = np.sum(high_candidates)

        logger.info(f"      - Low walls: {n_low:,} candidates")
        logger.info(f"      - Facades: {n_normal:,} candidates")
        logger.info(f"      - High elements: {n_high:,} candidates")

        # Build spatial index with adaptive search buffers
        # Larger buildings get larger search radius
        areas = building_polygons.geometry.area
        base_radius = 2.0  # Base 2m search
        adaptive_radius = np.clip(
            base_radius + (areas**0.5) * 0.02,  # Add 2% of perimeter
            2.0,  # Min 2m
            5.0,  # Max 5m for very large buildings
        )

        # Create buffered polygons with adaptive radius
        buffered_polygons = [
            geom.buffer(radius)
            for geom, radius in zip(building_polygons.geometry, adaptive_radius)
        ]
        tree = STRtree(buffered_polygons)

        # Check each candidate point
        recovered_count = 0
        low_wall_count = 0
        high_element_count = 0
        candidate_indices = np.where(facade_candidates)[0]

        for idx in candidate_indices:
            pt = Point(points[idx, 0], points[idx, 1])
            nearby = tree.query(pt)

            if len(nearby) > 0:
                # Point is near a building → classify as building
                refined[idx] = int(ASPRSClass.BUILDING)
                recovered_count += 1

                # Track by type
                if low_wall_candidates[idx]:
                    low_wall_count += 1
                elif high_candidates[idx]:
                    high_element_count += 1

        stats["facades_recovered"] = recovered_count
        stats["low_walls_recovered"] = low_wall_count
        stats["vertical_elements_recovered"] = high_element_count

        if recovered_count > 0:
            pct = 100.0 * recovered_count / n_candidates
            logger.info(
                f"    ✅ Recovered {recovered_count:,} facade/wall points "
                f"({pct:.1f}% of candidates)"
            )
            if low_wall_count > 0:
                logger.info(f"       └─ Low walls: {low_wall_count:,}")
            if high_element_count > 0:
                logger.info(f"       └─ High elements: {high_element_count:,}")
        else:
            logger.info("    No additional facades recovered")

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
        road_mask = labels == int(ASPRSClass.ROAD_SURFACE)
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
                    refined[idx] = int(ASPRSClass.BUILDING)
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
                water_mask = labels == int(ASPRSClass.WATER)
                refined, water_stats = self.refine_water_classification(
                    refined, points, water_mask, height, planarity, curvature, normals
                )
                all_stats.update(water_stats)

        # 2. Refine roads (must be flat and near ground)
        if "roads" in ground_truth_features:
            roads_gdf = ground_truth_features["roads"]
            if roads_gdf is not None and len(roads_gdf) > 0:
                # Create road mask
                road_mask = labels == int(ASPRSClass.ROAD_SURFACE)
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

                # 4b. Aggressive facade recovery (new post-processing step)
                if verticality is not None:
                    refined, facade_stats = self.recover_missing_facades(
                        refined, points, buildings_gdf, height, verticality
                    )
                    all_stats.update(facade_stats)

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
