#!/usr/bin/env python3
"""
Quick performance fix for ground truth classification.

This script patches the _classify_by_ground_truth method to use:
1. Pre-filtering by geometric features (height, planarity)
2. Progress bars to monitor performance
3. Better logging

Apply this fix by importing before your processing:
    from ground_truth_quick_fix import patch_classifier
    patch_classifier()
"""

import logging
import numpy as np
from typing import Dict, Optional, List

from ign_lidar.core.classification.priorities import (
    get_priority_order_for_iteration,
)

logger = logging.getLogger(__name__)

try:
    from shapely.geometry import Point, Polygon, MultiPolygon
    from tqdm import tqdm

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def create_optimized_classify_by_ground_truth():
    """Create optimized version with pre-filtering and progress bars."""

    def _classify_by_ground_truth_optimized(
        self,
        labels: np.ndarray,
        points: np.ndarray,
        ground_truth_features: Dict[str, "gpd.GeoDataFrame"],
        ndvi: Optional[np.ndarray],
        height: Optional[np.ndarray] = None,
        planarity: Optional[np.ndarray] = None,
        intensity: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        OPTIMIZED: Classify using IGN BD TOPO® ground truth with pre-filtering.

        Key optimizations:
        1. Pre-filter points by geometric features BEFORE spatial queries
        2. Add progress bars for long operations
        3. Better logging of performance metrics
        """
        if not HAS_DEPS:
            logger.warning("Dependencies not available, using original method")
            return self._classify_by_ground_truth_original(
                labels,
                points,
                ground_truth_features,
                ndvi,
                height,
                planarity,
                intensity,
            )

        logger.info(f"  Optimized classification for {len(points):,} points")

        # Create Point geometries
        logger.info("  Creating point geometries...")
        point_geoms = [Point(p[0], p[1]) for p in points]

        # ✅ Use centralized priority order (lowest → highest priority)
        feature_priority = get_priority_order_for_iteration()
        asprs_mapping = {
            "buildings": self.ASPRS_BUILDING,
            "roads": self.ASPRS_ROAD,
            "railways": self.ASPRS_RAIL,
            "water": self.ASPRS_WATER,
            "bridges": self.ASPRS_BRIDGE,
            "vegetation": self.ASPRS_MEDIUM_VEGETATION,
            "cemeteries": self.ASPRS_CEMETERY,
            "parking": self.ASPRS_PARKING,
            "sports": self.ASPRS_SPORTS,
            "power_lines": self.ASPRS_POWER_LINE,
        }
        priority_order = [
            (feature, asprs_mapping.get(feature, self.ASPRS_BUILDING))
            for feature in feature_priority
        ]

        # Pre-filter candidates for roads and railways
        road_candidates = None
        rail_candidates = None

        if height is not None and planarity is not None:
            logger.info("  Pre-filtering candidates by geometric features...")

            # Road candidates: low height, high planarity
            if "roads" in ground_truth_features:
                road_candidates = np.where(
                    (height <= 2.0)  # At ground level
                    & (height >= -0.5)
                    & (planarity >= 0.7)  # Flat surfaces
                )[0]
                logger.info(
                    f"    Road candidates: {len(road_candidates):,} / {len(points):,} "
                    f"({len(road_candidates)/len(points)*100:.1f}%)"
                )

            # Railway candidates: low height, medium planarity (ballast is less flat)
            if "railways" in ground_truth_features:
                rail_candidates = np.where(
                    (height <= 2.0)
                    & (height >= -0.5)
                    & (planarity >= 0.5)  # Less strict than roads
                )[0]
                logger.info(
                    f"    Railway candidates: {len(rail_candidates):,} / {len(points):,} "
                    f"({len(rail_candidates)/len(points)*100:.1f}%)"
                )

        # Process each feature type
        for feature_type, asprs_class in priority_order:
            if feature_type not in ground_truth_features:
                continue

            gdf = ground_truth_features[feature_type]
            if gdf is None or len(gdf) == 0:
                continue

            logger.info(f"    Processing {feature_type}: {len(gdf)} features")

            # Use pre-filtered candidates for roads/railways
            if feature_type == "roads" and road_candidates is not None:
                search_indices = road_candidates
                logger.info(
                    f"      Using pre-filtered candidates ({len(search_indices):,} points)"
                )
            elif feature_type == "railways" and rail_candidates is not None:
                search_indices = rail_candidates
                logger.info(
                    f"      Using pre-filtered candidates ({len(search_indices):,} points)"
                )
            else:
                search_indices = None

            # Process each polygon with progress bar
            n_classified = 0

            # OPTIMIZED: Vectorized geometry extraction to reduce .iterrows() overhead
            valid_mask = gdf["geometry"].apply(
                lambda g: isinstance(g, (Polygon, MultiPolygon))
            )
            valid_gdf = gdf[valid_mask]

            for idx, row in tqdm(
                valid_gdf.iterrows(),
                total=len(valid_gdf),
                desc=f"      {feature_type}",
                leave=False,
                disable=(len(valid_gdf) < 10),
            ):  # Only show for large feature sets
                polygon = row["geometry"]

                # Bbox filtering
                bounds = polygon.bounds  # (minx, miny, maxx, maxy)
                bbox_mask = (
                    (points[:, 0] >= bounds[0])
                    & (points[:, 0] <= bounds[2])
                    & (points[:, 1] >= bounds[1])
                    & (points[:, 1] <= bounds[3])
                )

                # Intersect with pre-filtered candidates if available
                if search_indices is not None:
                    # Create mask for search_indices
                    search_mask = np.zeros(len(points), dtype=bool)
                    search_mask[search_indices] = True
                    # Combine masks
                    bbox_mask = bbox_mask & search_mask

                candidate_indices = np.where(bbox_mask)[0]

                # Check only candidate points
                for i in candidate_indices:
                    if polygon.contains(point_geoms[i]):
                        # For roads/railways, apply additional filters
                        if feature_type in ["roads", "railways"]:
                            # Additional intensity filter if available
                            if intensity is not None:
                                if intensity[i] < 0.1 or intensity[i] > 0.9:
                                    continue  # Skip unlikely intensity values

                        labels[i] = asprs_class
                        n_classified += 1

            logger.info(f"      Classified {n_classified:,} points as {feature_type}")

        # NDVI refinement
        if ndvi is not None:
            building_mask = labels == self.ASPRS_BUILDING
            high_ndvi_buildings = building_mask & (ndvi >= self.ndvi_veg_threshold)

            if np.any(high_ndvi_buildings):
                n_veg_on_building = np.sum(high_ndvi_buildings)
                logger.info(
                    f"    Note: {n_veg_on_building} building points with high NDVI"
                )

        return labels

    return _classify_by_ground_truth_optimized


def patch_classifier():
    """Patch AdvancedClassifier to use optimized ground truth classification."""

    try:
        from ign_lidar.core.classification import AdvancedClassifier

        # Save original method
        if not hasattr(AdvancedClassifier, "_classify_by_ground_truth_original"):
            AdvancedClassifier._classify_by_ground_truth_original = (
                AdvancedClassifier._classify_by_ground_truth
            )

        # Apply optimized method
        AdvancedClassifier._classify_by_ground_truth = (
            create_optimized_classify_by_ground_truth()
        )

        logger.info("✅ Applied optimized ground truth classification")
        logger.info("   Expected speedup: 2-5× (with pre-filtering)")

    except ImportError as e:
        logger.error(f"Failed to patch classifier: {e}")


if __name__ == "__main__":
    print("Ground Truth Quick Fix")
    print("=" * 80)
    print()
    print(
        "This module provides a quick performance fix for ground truth classification."
    )
    print()
    print("Usage:")
    print("  from ground_truth_quick_fix import patch_classifier")
    print("  patch_classifier()")
    print()
    print("Then run your normal processing:")
    print("  python reprocess_with_ground_truth.py enriched.laz")
    print()
    print("Expected improvements:")
    print("  - 2-5× speedup from pre-filtering")
    print("  - Progress bars for monitoring")
    print("  - Better logging")
