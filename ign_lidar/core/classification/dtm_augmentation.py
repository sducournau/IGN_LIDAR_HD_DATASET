"""
DTM Ground Point Augmentation Module

This module augments LiDAR point clouds with synthetic ground points derived from
IGN RGE ALTI DTM (Digital Terrain Model). This is critical for:

1. Buildings: Add ground-level points under building footprints where LiDAR cannot penetrate
2. Vegetation: Add ground points under dense tree canopy for accurate height computation
3. Gaps: Fill coverage gaps in sparse areas

Key Features:
- Intelligent placement: Only add points where needed (under buildings/vegetation)
- Quality validation: Ensure synthetic points are consistent with nearby real ground
- Height accuracy: ±0.15m from 1m resolution RGE ALTI DTM
- Spatial filtering: Avoid placing points too close to existing ground points
- Classification: Mark synthetic points with special flag for transparency

Use Cases:
- Height normalization: Better ground reference → more accurate heights above ground
- Classification: Better ground/non-ground separation
- Terrain analysis: Fill gaps for complete terrain coverage
- Feature computation: More reliable neighbor queries near ground level

Performance:
- Typical tile (18M points): Add 0.9-2.8M synthetic ground points (5-15% increase)
- Processing time: ~1-2 minutes per tile (DTM download + point generation + validation)
- Memory: Minimal overhead (<100MB for typical tiles)

Author: Simon Ducournau
Date: October 23, 2025
Version: 3.1.0
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .constants import ASPRSClass

logger = logging.getLogger(__name__)

try:
    import geopandas as gpd
    from shapely.geometry import MultiPolygon, Point, Polygon, box
    from shapely.strtree import STRtree

    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    gpd = None

try:
    from scipy.spatial import cKDTree

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    cKDTree = None


class AugmentationStrategy(Enum):
    """Strategy for ground point augmentation."""

    FULL = "full"  # Add points everywhere (dense but slow)
    GAPS = "gaps"  # Only fill areas with no existing ground
    INTELLIGENT = "intelligent"  # Prioritize under buildings/vegetation (RECOMMENDED)


class AugmentationArea(Enum):
    """Areas where synthetic ground points should be added."""

    VEGETATION = "vegetation"  # Under trees and vegetation (CRITICAL)
    BUILDINGS = "buildings"  # Under building footprints
    WATER = "water"  # Under water bodies (usually not needed)
    ROADS = "roads"  # Roads usually have good coverage
    GAPS = "gaps"  # General coverage gaps


@dataclass
class DTMAugmentationConfig:
    """Configuration for DTM ground augmentation."""

    # Enable augmentation
    enabled: bool = True

    # Strategy
    strategy: AugmentationStrategy = AugmentationStrategy.INTELLIGENT

    # Grid spacing for synthetic points (meters)
    spacing: float = 2.0  # Balance between coverage and performance

    # Minimum distance to existing ground points (meters)
    min_spacing_to_existing: float = 1.5  # Avoid clustering

    # Priority areas (which areas to augment)
    augment_vegetation: bool = True  # CRITICAL for accurate vegetation height
    augment_buildings: bool = True  # Ground level under buildings
    augment_water: bool = False  # Usually not needed
    augment_roads: bool = False  # Roads have good coverage
    augment_gaps: bool = True  # Fill sparse areas

    # Validation thresholds
    max_height_difference: float = (
        5.0  # Max difference from nearby real ground (meters)
    )
    validate_against_neighbors: bool = True  # Check consistency with neighbors
    min_neighbors_for_validation: int = 3  # Minimum neighbors for validation
    neighbor_search_radius: float = 10.0  # Search radius for validation (meters)

    # Classification
    synthetic_ground_class: int = 2  # ASPRS Ground class
    mark_as_synthetic: bool = True  # Add flag to distinguish from real LiDAR

    # Performance
    use_spatial_index: bool = True  # Use KD-tree for fast neighbor search
    batch_size: int = 100000  # Process points in batches

    # Logging
    verbose: bool = True


class DTMAugmentationStats:
    """Statistics for DTM augmentation process."""

    def __init__(self):
        self.total_generated = 0
        self.total_validated = 0
        self.total_rejected = 0

        # Per-area statistics
        self.vegetation_points = 0
        self.building_points = 0
        self.water_points = 0
        self.road_points = 0
        self.gap_points = 0

        # Validation statistics
        self.rejected_height_diff = 0
        self.rejected_no_neighbors = 0
        self.rejected_spacing = 0

    def log_summary(self, logger):
        """Log summary statistics."""
        logger.info("=== DTM Ground Augmentation Summary ===")
        logger.info(f"  Generated: {self.total_generated:,} synthetic points")
        logger.info(f"  Validated: {self.total_validated:,} points added")
        logger.info(f"  Rejected: {self.total_rejected:,} points filtered")

        if self.total_validated > 0:
            logger.info("  Distribution:")
            if self.vegetation_points > 0:
                pct = 100 * self.vegetation_points / self.total_validated
                logger.info(
                    f"    - Vegetation: {self.vegetation_points:,} ({pct:.1f}%)"
                )
            if self.building_points > 0:
                pct = 100 * self.building_points / self.total_validated
                logger.info(f"    - Buildings: {self.building_points:,} ({pct:.1f}%)")
            if self.gap_points > 0:
                pct = 100 * self.gap_points / self.total_validated
                logger.info(f"    - Coverage gaps: {self.gap_points:,} ({pct:.1f}%)")

        if self.total_rejected > 0:
            logger.info("  Rejection reasons:")
            if self.rejected_height_diff > 0:
                pct = 100 * self.rejected_height_diff / self.total_rejected
                logger.info(
                    f"    - Height difference: {self.rejected_height_diff:,} ({pct:.1f}%)"
                )
            if self.rejected_spacing > 0:
                pct = 100 * self.rejected_spacing / self.total_rejected
                logger.info(
                    f"    - Too close to existing: {self.rejected_spacing:,} ({pct:.1f}%)"
                )
            if self.rejected_no_neighbors > 0:
                pct = 100 * self.rejected_no_neighbors / self.total_rejected
                logger.info(
                    f"    - No neighbors for validation: {self.rejected_no_neighbors:,} ({pct:.1f}%)"
                )


class DTMAugmenter:
    """
    Augment LiDAR point cloud with synthetic ground points from DTM.

    This class implements intelligent ground point augmentation to improve:
    1. Height computation accuracy (especially under vegetation)
    2. Ground/non-ground classification
    3. Terrain coverage in sparse areas
    """

    # ASPRS class codes

    def __init__(self, config: Optional[DTMAugmentationConfig] = None):
        """
        Initialize DTM augmenter.

        Args:
            config: Augmentation configuration (default if None)
        """
        self.config = config if config is not None else DTMAugmentationConfig()
        self.stats = DTMAugmentationStats()

        if not HAS_GEOPANDAS:
            logger.warning("GeoPandas not available - spatial filtering disabled")

        if not HAS_SCIPY:
            logger.warning("SciPy not available - neighbor validation disabled")

    def augment_point_cloud(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        dtm_fetcher,
        bbox: Tuple[float, float, float, float],
        building_polygons: Optional[gpd.GeoDataFrame] = None,
        crs: str = "EPSG:2154",
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Augment point cloud with synthetic ground points from DTM.

        Args:
            points: Original point cloud [N, 3] (X, Y, Z)
            labels: Original classifications [N]
            dtm_fetcher: RGEALTIFetcher instance for DTM access
            bbox: Bounding box (minx, miny, maxx, maxy)
            building_polygons: Optional building footprints for targeted augmentation
            crs: Coordinate reference system

        Returns:
            Tuple of:
            - augmented_points: Combined point cloud [N+M, 3]
            - augmented_labels: Combined classifications [N+M]
            - augmentation_attrs: Dict with synthetic point attributes
        """
        if not self.config.enabled:
            logger.info("DTM augmentation disabled")
            return points, labels, {}

        logger.info("=== DTM Ground Point Augmentation ===")
        logger.info(f"  Strategy: {self.config.strategy.value}")
        logger.info(f"  Grid spacing: {self.config.spacing}m")

        # Reset statistics
        self.stats = DTMAugmentationStats()

        # Step 1: Generate candidate synthetic points from DTM
        synthetic_points = self._generate_synthetic_points(dtm_fetcher, bbox, crs)

        if synthetic_points is None or len(synthetic_points) == 0:
            logger.warning("  Failed to generate synthetic points from DTM")
            return points, labels, {}

        self.stats.total_generated = len(synthetic_points)
        logger.info(f"  Generated {len(synthetic_points):,} candidate points from DTM")

        # Step 2: Filter synthetic points based on strategy and areas
        filtered_points, area_labels = self._filter_by_strategy(
            synthetic_points, points, labels, building_polygons, bbox
        )

        if len(filtered_points) == 0:
            logger.warning("  All synthetic points filtered out")
            return points, labels, {}

        logger.info(f"  Kept {len(filtered_points):,} points after strategy filtering")

        # Step 3: Validate synthetic points against real ground points
        if self.config.validate_against_neighbors:
            validated_points, validated_areas = self._validate_against_neighbors(
                filtered_points, area_labels, points, labels
            )
        else:
            validated_points = filtered_points
            validated_areas = area_labels

        if len(validated_points) == 0:
            logger.warning("  All synthetic points rejected during validation")
            return points, labels, {}

        self.stats.total_validated = len(validated_points)
        self.stats.total_rejected = (
            self.stats.total_generated - self.stats.total_validated
        )

        logger.info(f"  Validated {len(validated_points):,} points for augmentation")

        # Step 4: Combine with original point cloud
        augmented_points = np.vstack([points, validated_points])

        # Create labels for synthetic points
        synthetic_labels = np.full(
            len(validated_points),
            self.config.synthetic_ground_class,
            dtype=labels.dtype,
        )
        augmented_labels = np.concatenate([labels, synthetic_labels])

        # Create synthetic point attributes
        augmentation_attrs = {
            "is_synthetic": np.concatenate(
                [
                    np.zeros(len(points), dtype=bool),
                    np.ones(len(validated_points), dtype=bool),
                ]
            ),
            "augmentation_area": validated_areas,
            "dtm_elevation": validated_points[:, 2],  # Z is DTM ground elevation
        }

        # Log statistics
        self.stats.log_summary(logger)

        return augmented_points, augmented_labels, augmentation_attrs

    def _generate_synthetic_points(
        self, dtm_fetcher, bbox: Tuple[float, float, float, float], crs: str
    ) -> Optional[np.ndarray]:
        """
        Generate synthetic ground points from DTM on a regular grid.

        Args:
            dtm_fetcher: RGEALTIFetcher instance
            bbox: Bounding box
            crs: Coordinate reference system

        Returns:
            Synthetic points [M, 3] or None if failed
        """
        try:
            synthetic_points = dtm_fetcher.generate_ground_points(
                bbox=bbox, spacing=self.config.spacing, crs=crs
            )

            # No hard limit needed for 64GB systems
            # 1M points @ 24 bytes = ~24MB (trivial for 64GB)
            # Allow natural generation from spacing parameter

            return synthetic_points
        except Exception as e:
            logger.error(f"Failed to generate synthetic points: {e}")
            return None

    def _filter_by_strategy(
        self,
        synthetic_points: np.ndarray,
        real_points: np.ndarray,
        labels: np.ndarray,
        building_polygons: Optional[gpd.GeoDataFrame],
        bbox: Tuple[float, float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter synthetic points based on augmentation strategy.

        Args:
            synthetic_points: Candidate synthetic points [M, 3]
            real_points: Original point cloud [N, 3]
            labels: Original classifications [N]
            building_polygons: Optional building footprints
            bbox: Bounding box

        Returns:
            Tuple of (filtered_points, area_labels)
            - filtered_points: Points to keep [K, 3]
            - area_labels: Which area each point belongs to [K] (enum indices)
        """
        strategy = self.config.strategy

        if strategy == AugmentationStrategy.FULL:
            # Add all synthetic points
            area_labels = np.full(len(synthetic_points), AugmentationArea.GAPS.value)
            return synthetic_points, area_labels

        elif strategy == AugmentationStrategy.GAPS:
            # Only add where no existing ground points
            return self._filter_gaps_only(synthetic_points, real_points, labels)

        elif strategy == AugmentationStrategy.INTELLIGENT:
            # Prioritize under buildings and vegetation (RECOMMENDED)
            return self._filter_intelligent(
                synthetic_points, real_points, labels, building_polygons, bbox
            )

        else:
            logger.warning(f"Unknown strategy: {strategy}, using INTELLIGENT")
            return self._filter_intelligent(
                synthetic_points, real_points, labels, building_polygons, bbox
            )

    def _filter_gaps_only(
        self, synthetic_points: np.ndarray, real_points: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter to only add points in coverage gaps.

        OPTIMIZED VERSION:
        - Uses chunked processing to avoid memory overflow
        - Reduces memory footprint for large point clouds
        """
        if not HAS_SCIPY:
            logger.warning("SciPy not available - cannot filter gaps")
            return synthetic_points, np.full(
                len(synthetic_points), AugmentationArea.GAPS.value
            )

        # Find existing ground points
        ground_mask = labels == int(ASPRSClass.GROUND)
        ground_points = real_points[ground_mask]

        if len(ground_points) == 0:
            # No ground points - keep all synthetic
            return synthetic_points, np.full(
                len(synthetic_points), AugmentationArea.GAPS.value
            )

        # Build KD-tree of existing ground
        tree = cKDTree(ground_points[:, :2])  # Use XY only

        # MEMORY OPTIMIZATION: Process synthetic points in chunks
        chunk_size = 100000  # 100k points at a time
        n_synthetic = len(synthetic_points)
        gap_mask = np.zeros(n_synthetic, dtype=bool)

        logger.debug(
            f"  Filtering gaps for {n_synthetic:,} points in chunks of {chunk_size:,}..."
        )

        for start_idx in range(0, n_synthetic, chunk_size):
            end_idx = min(start_idx + chunk_size, n_synthetic)
            chunk = synthetic_points[start_idx:end_idx, :2]

            # Query nearest ground point for each synthetic point in chunk
            distances, _ = tree.query(chunk)

            # Keep only points far from existing ground
            chunk_gap_mask = distances > self.config.min_spacing_to_existing
            gap_mask[start_idx:end_idx] = chunk_gap_mask

            rejected_in_chunk = np.sum(~chunk_gap_mask)
            if rejected_in_chunk > 0:
                self.stats.rejected_spacing += rejected_in_chunk

        filtered = synthetic_points[gap_mask]
        area_labels = np.full(len(filtered), AugmentationArea.GAPS.value)

        return filtered, area_labels

    def _filter_intelligent(
        self,
        synthetic_points: np.ndarray,
        real_points: np.ndarray,
        labels: np.ndarray,
        building_polygons: Optional[gpd.GeoDataFrame],
        bbox: Tuple[float, float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Intelligent filtering: prioritize under buildings and vegetation.

        This is the RECOMMENDED strategy for best results.
        """
        kept_points = []
        area_labels_list = []

        # Priority 1: Under vegetation (CRITICAL for height accuracy)
        if self.config.augment_vegetation:
            veg_points, n_veg = self._filter_under_vegetation(
                synthetic_points, real_points, labels
            )
            if len(veg_points) > 0:
                kept_points.append(veg_points)
                area_labels_list.append(
                    np.full(len(veg_points), AugmentationArea.VEGETATION.value)
                )
                self.stats.vegetation_points = len(veg_points)

        # Priority 2: Under buildings (ground level reference)
        if self.config.augment_buildings and building_polygons is not None:
            building_points, n_bldg = self._filter_under_buildings(
                synthetic_points, building_polygons, real_points, labels
            )
            if len(building_points) > 0:
                kept_points.append(building_points)
                area_labels_list.append(
                    np.full(len(building_points), AugmentationArea.BUILDINGS.value)
                )
                self.stats.building_points = len(building_points)

        # Priority 3: Coverage gaps (general improvement)
        if self.config.augment_gaps:
            gap_points, n_gaps = self._filter_coverage_gaps(
                synthetic_points, real_points, labels, kept_points
            )
            if len(gap_points) > 0:
                kept_points.append(gap_points)
                area_labels_list.append(
                    np.full(len(gap_points), AugmentationArea.GAPS.value)
                )
                self.stats.gap_points = len(gap_points)

        if len(kept_points) == 0:
            return np.array([]).reshape(0, 3), np.array([])

        # Combine all filtered points
        filtered_points = np.vstack(kept_points)
        area_labels = np.concatenate(area_labels_list)

        return filtered_points, area_labels

    def _filter_under_vegetation(
        self, synthetic_points: np.ndarray, real_points: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """
        Filter synthetic points to only those under vegetation.

        Critical for accurate vegetation height computation.
        """
        if not HAS_SCIPY:
            return np.array([]).reshape(0, 3), 0

        # Find vegetation points
        veg_mask = np.isin(
            labels,
            [
                int(ASPRSClass.LOW_VEGETATION),
                int(ASPRSClass.MEDIUM_VEGETATION),
                int(ASPRSClass.HIGH_VEGETATION),
            ],
        )

        if not np.any(veg_mask):
            return np.array([]).reshape(0, 3), 0

        veg_points = real_points[veg_mask]

        # Build KD-tree of vegetation points
        tree = cKDTree(veg_points[:, :2])

        # Find synthetic points within vegetation footprint
        # Use larger radius to capture area under tree canopy
        radius = 5.0  # meters
        neighbor_indices = tree.query_ball_point(synthetic_points[:, :2], radius)

        # Keep points that have vegetation above them
        under_veg_mask = np.array(
            [len(neighbors) > 0 for neighbors in neighbor_indices]
        )

        # Also ensure not too close to existing ground points
        ground_mask = labels == int(ASPRSClass.GROUND)
        if np.any(ground_mask):
            ground_points = real_points[ground_mask]
            ground_tree = cKDTree(ground_points[:, :2])
            distances, _ = ground_tree.query(synthetic_points[:, :2])
            spacing_ok = distances > self.config.min_spacing_to_existing
            under_veg_mask &= spacing_ok

        return synthetic_points[under_veg_mask], np.sum(under_veg_mask)

    def _filter_under_buildings(
        self,
        synthetic_points: np.ndarray,
        building_polygons: gpd.GeoDataFrame,
        real_points: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """
        Filter synthetic points to only those under building footprints.

        Provides ground-level reference under buildings.
        """
        if (
            not HAS_GEOPANDAS
            or building_polygons is None
            or len(building_polygons) == 0
        ):
            return np.array([]).reshape(0, 3), 0

        # Build spatial index
        polygons_list = building_polygons.geometry.tolist()
        tree = STRtree(polygons_list)

        # Check each synthetic point
        under_building_mask = np.zeros(len(synthetic_points), dtype=bool)

        for i, pt_coords in enumerate(synthetic_points[:, :2]):
            pt = Point(pt_coords[0], pt_coords[1])
            nearby_indices = tree.query(pt)

            # Check if point is within any building
            for poly_idx in nearby_indices:
                if polygons_list[poly_idx].contains(pt):
                    under_building_mask[i] = True
                    break

        # Also ensure not too close to existing ground points
        if HAS_SCIPY:
            ground_mask = labels == int(ASPRSClass.GROUND)
            if np.any(ground_mask):
                ground_points = real_points[ground_mask]
                ground_tree = cKDTree(ground_points[:, :2])
                distances, _ = ground_tree.query(synthetic_points[:, :2])
                spacing_ok = distances > self.config.min_spacing_to_existing
                under_building_mask &= spacing_ok

        return synthetic_points[under_building_mask], np.sum(under_building_mask)

    def _filter_coverage_gaps(
        self,
        synthetic_points: np.ndarray,
        real_points: np.ndarray,
        labels: np.ndarray,
        already_kept: List[np.ndarray],
    ) -> Tuple[np.ndarray, int]:
        """
        Filter synthetic points to fill general coverage gaps.

        Excludes points already kept for vegetation/buildings.
        """
        if not HAS_SCIPY:
            return np.array([]).reshape(0, 3), 0

        # Exclude already kept points
        kept_coords = set()
        for kept_array in already_kept:
            for pt in kept_array[:, :2]:
                kept_coords.add((round(pt[0], 2), round(pt[1], 2)))

        # Filter out already kept
        not_kept_mask = np.array(
            [
                (round(pt[0], 2), round(pt[1], 2)) not in kept_coords
                for pt in synthetic_points[:, :2]
            ]
        )
        remaining = synthetic_points[not_kept_mask]

        if len(remaining) == 0:
            return np.array([]).reshape(0, 3), 0

        # Find ground points
        ground_mask = labels == int(ASPRSClass.GROUND)
        if not np.any(ground_mask):
            # No existing ground - all remaining are gaps
            return remaining, len(remaining)

        ground_points = real_points[ground_mask]
        tree = cKDTree(ground_points[:, :2])

        # Find points far from existing ground
        distances, _ = tree.query(remaining[:, :2])
        gap_mask = distances > self.config.min_spacing_to_existing

        return remaining[gap_mask], np.sum(gap_mask)

    def _validate_against_neighbors(
        self,
        synthetic_points: np.ndarray,
        area_labels: np.ndarray,
        real_points: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate synthetic points against nearby real ground points.

        Rejects points that are too different from nearby ground elevation.

        OPTIMIZED VERSION FOR 64GB SYSTEMS:
        - Processes in chunks to avoid memory overflow
        - Larger chunks (50k) suitable for high-memory systems
        - Uses vectorized operations where possible
        - Handles large point clouds (21M+ points) efficiently
        """
        if not HAS_SCIPY:
            logger.warning("SciPy not available - skipping neighbor validation")
            return synthetic_points, area_labels

        # Find real ground points
        ground_mask = labels == int(ASPRSClass.GROUND)
        if not np.any(ground_mask):
            logger.warning("No real ground points for validation")
            return synthetic_points, area_labels

        ground_points = real_points[ground_mask]

        # Build KD-tree (this is fast and memory-efficient)
        tree = cKDTree(ground_points[:, :2])

        # Process in chunks to avoid memory issues
        # 64GB system can handle larger chunks
        chunk_size = 50000  # Balanced for 64GB systems
        n_synthetic = len(synthetic_points)
        valid_mask = np.ones(n_synthetic, dtype=bool)

        logger.debug(
            f"  Validating {n_synthetic:,} synthetic points in chunks of {chunk_size:,}..."
        )

        for start_idx in range(0, n_synthetic, chunk_size):
            end_idx = min(start_idx + chunk_size, n_synthetic)
            chunk = synthetic_points[start_idx:end_idx]

            # Vectorized neighbor count query (much faster than query_ball_point in loop)
            # First, do a quick k-nearest neighbor search to estimate neighbor count
            distances, indices = tree.query(
                chunk[:, :2],
                k=min(self.config.min_neighbors_for_validation + 5, len(ground_points)),
                distance_upper_bound=self.config.neighbor_search_radius,
            )

            # Process each point in chunk
            for i, syn_pt in enumerate(chunk):
                global_idx = start_idx + i

                # Count valid neighbors (within radius)
                if distances.ndim == 2:
                    valid_neighbors = distances[i] <= self.config.neighbor_search_radius
                    neighbor_indices = indices[i][valid_neighbors]
                    n_neighbors = np.sum(valid_neighbors & (distances[i] < np.inf))
                else:
                    # Single neighbor case
                    valid_neighbors = distances[i] <= self.config.neighbor_search_radius
                    neighbor_indices = [indices[i]] if valid_neighbors else []
                    n_neighbors = 1 if valid_neighbors else 0

                # Check minimum neighbor requirement
                if n_neighbors < self.config.min_neighbors_for_validation:
                    valid_mask[global_idx] = False
                    self.stats.rejected_no_neighbors += 1
                    continue

                # Check height consistency with neighbors
                if len(neighbor_indices) > 0:
                    neighbor_elevations = ground_points[neighbor_indices, 2]
                    mean_elevation = np.mean(neighbor_elevations)
                    height_diff = abs(syn_pt[2] - mean_elevation)

                    if height_diff > self.config.max_height_difference:
                        valid_mask[global_idx] = False
                        self.stats.rejected_height_diff += 1
                        continue

            # MEMORY SAFETY: Explicit cleanup after each chunk
            del distances, indices

        validated_points = synthetic_points[valid_mask]
        validated_areas = area_labels[valid_mask]

        # MEMORY SAFETY: Final cleanup
        del valid_mask, tree, ground_points
        import gc

        gc.collect()

        return validated_points, validated_areas


# ============================================================================
# Convenience Functions
# ============================================================================


def augment_with_dtm(
    points: np.ndarray,
    labels: np.ndarray,
    dtm_fetcher,
    bbox: Tuple[float, float, float, float],
    building_polygons: Optional[gpd.GeoDataFrame] = None,
    config: Optional[DTMAugmentationConfig] = None,
    crs: str = "EPSG:2154",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Convenience function to augment point cloud with DTM ground points.

    Args:
        points: Original point cloud [N, 3]
        labels: Original classifications [N]
        dtm_fetcher: RGEALTIFetcher instance
        bbox: Bounding box (minx, miny, maxx, maxy)
        building_polygons: Optional building footprints
        config: Augmentation configuration (default if None)
        crs: Coordinate reference system

    Returns:
        Tuple of (augmented_points, augmented_labels, augmentation_attrs)
    """
    augmenter = DTMAugmenter(config)
    return augmenter.augment_point_cloud(
        points, labels, dtm_fetcher, bbox, building_polygons, crs
    )


__all__ = [
    "DTMAugmenter",
    "DTMAugmentationConfig",
    "DTMAugmentationStats",
    "AugmentationStrategy",
    "AugmentationArea",
    "augment_with_dtm",
]
