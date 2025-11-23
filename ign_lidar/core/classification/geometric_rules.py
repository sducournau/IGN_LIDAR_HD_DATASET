"""
Geometric Rules Module for Intelligent Reclassification

This module provides geometric and spectral rules to improve classification accuracy:
1. Road-vegetation overlap detection using vertical separation and NDVI
2. Building buffer zone classification for nearby unclassified points
3. Height-based disambiguation for overlapping features
4. NDVI-based vegetation refinement

Author: Data Processing Team
Date: October 16, 2025
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

from ign_lidar.optimization.gpu_accelerated_ops import knn  # GPU-accelerated KNN
from .constants import ASPRSClass

logger = logging.getLogger(__name__)

try:
    import geopandas as gpd
    from shapely.geometry import MultiPolygon, Point, Polygon
    from shapely.strtree import STRtree

    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False
    logger.warning("Spatial libraries not available for geometric rules")

try:
    from .spectral_rules import SpectralRulesEngine

    HAS_SPECTRAL_RULES = True
except ImportError:
    HAS_SPECTRAL_RULES = False
    logger.warning("Spectral rules module not available")

try:
    from sklearn.cluster import DBSCAN

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not available for clustering optimization")


class GeometricRulesEngine:
    """
    Engine for applying geometric and spectral rules to improve classification.

    Features:
    - Road-vegetation disambiguation using height and NDVI
    - Building buffer zones for nearby unclassified points
    - Vertical separation analysis for overlapping geometries
    - Multi-level NDVI-based refinement for vegetation/non-vegetation
    """

    # ASPRS Classification codes - expose as class attributes for convenience
    ASPRS_UNCLASSIFIED = int(ASPRSClass.UNCLASSIFIED)
    ASPRS_GROUND = int(ASPRSClass.GROUND)
    ASPRS_LOW_VEGETATION = int(ASPRSClass.LOW_VEGETATION)
    ASPRS_MEDIUM_VEGETATION = int(ASPRSClass.MEDIUM_VEGETATION)
    ASPRS_HIGH_VEGETATION = int(ASPRSClass.HIGH_VEGETATION)
    ASPRS_BUILDING = int(ASPRSClass.BUILDING)
    ASPRS_LOW_POINT = int(ASPRSClass.LOW_POINT)
    ASPRS_WATER = int(ASPRSClass.WATER)
    ASPRS_RAIL = int(ASPRSClass.RAIL)
    ASPRS_ROAD_SURFACE = int(ASPRSClass.ROAD_SURFACE)
    ASPRS_ROAD = int(ASPRSClass.ROAD_SURFACE)  # Alias for compatibility

    # Multi-level NDVI thresholds (aligned with advanced_classification.py)
    NDVI_DENSE_FOREST = 0.60  # Dense forest, high vegetation
    NDVI_HEALTHY_TREES = 0.50  # Healthy trees, high/medium vegetation
    NDVI_MODERATE_VEG = 0.40  # Moderate vegetation, medium vegetation
    NDVI_GRASS = 0.30  # Grass/shrubs, low/medium vegetation
    NDVI_SPARSE_VEG = 0.20  # Sparse vegetation, low vegetation
    NDVI_ROAD = 0.15  # Road/impervious surfaces

    def __init__(
        self,
        ndvi_vegetation_threshold: float = 0.3,
        ndvi_road_threshold: float = 0.15,
        road_vegetation_height_threshold: float = 0.5,
        building_buffer_distance: float = 2.0,
        max_building_height_difference: float = 3.0,
        verticality_threshold: float = 0.7,
        verticality_search_radius: float = 1.0,
        min_vertical_neighbors: int = 5,
        use_spectral_rules: bool = True,
        nir_vegetation_threshold: float = 0.4,
        nir_building_threshold: float = 0.3,
        use_clustering: bool = True,
        spatial_cluster_eps: float = 0.5,
        min_cluster_size: int = 10,
    ):
        """
        Initialize geometric rules engine.

        Note: Multi-level NDVI thresholds are defined as class constants:
        - NDVI_DENSE_FOREST = 0.60 (HIGH vegetation)
        - NDVI_HEALTHY_TREES = 0.50 (HIGH/MED vegetation)
        - NDVI_MODERATE_VEG = 0.40 (MEDIUM vegetation)
        - NDVI_GRASS = 0.30 (LOW/MED vegetation)
        - NDVI_SPARSE_VEG = 0.20 (LOW vegetation)
        - NDVI_ROAD = 0.15 (Road/impervious surfaces)

        Args:
            ndvi_vegetation_threshold: Legacy threshold for backward compatibility (default 0.3)
            ndvi_road_threshold: NDVI threshold for roads (<= this = likely road/impervious)
            road_vegetation_height_threshold: Height above road surface to classify as
                vegetation (default: 0.5m). Points within 50cm of DTM ground are part
                of road surface; points higher are vegetation (trees, bushes).
            building_buffer_distance: Buffer distance around buildings for unclassified points (meters)
            max_building_height_difference: Max height diff for points to be part of building (meters)
            verticality_threshold: Verticality score threshold for building classification (0-1)
            verticality_search_radius: Search radius for computing verticality (meters)
            min_vertical_neighbors: Minimum neighbors required for verticality computation
            use_spectral_rules: Enable advanced spectral classification rules
            nir_vegetation_threshold: Minimum NIR for vegetation (spectral rules)
            nir_building_threshold: Minimum NIR for building materials (spectral rules)
            use_clustering: Enable clustering-based building buffer classification (10-100x faster)
            spatial_cluster_eps: Spatial clustering epsilon (meters)
            min_cluster_size: Minimum points per cluster
        """
        if not HAS_SPATIAL:
            raise ImportError(
                "Spatial libraries required for geometric rules. "
                "Install: pip install shapely geopandas scipy"
            )

        self.ndvi_vegetation_threshold = ndvi_vegetation_threshold
        self.ndvi_road_threshold = ndvi_road_threshold
        self.road_vegetation_height_threshold = road_vegetation_height_threshold
        self.building_buffer_distance = building_buffer_distance
        self.max_building_height_difference = max_building_height_difference
        self.verticality_threshold = verticality_threshold
        self.verticality_search_radius = verticality_search_radius
        self.min_vertical_neighbors = min_vertical_neighbors
        self.use_clustering = use_clustering and HAS_SKLEARN
        self.spatial_cluster_eps = spatial_cluster_eps
        self.min_cluster_size = min_cluster_size

        # Initialize spectral rules engine if available and enabled
        self.spectral_rules = None
        self.use_spectral_rules = use_spectral_rules
        if use_spectral_rules and HAS_SPECTRAL_RULES:
            self.spectral_rules = SpectralRulesEngine(
                nir_vegetation_threshold=nir_vegetation_threshold,
                nir_building_threshold=nir_building_threshold,
            )
            logger.info("   Spectral rules engine enabled")
        elif use_spectral_rules and not HAS_SPECTRAL_RULES:
            logger.warning("   Spectral rules requested but module not available")

        if self.use_clustering and not HAS_SKLEARN:
            logger.warning("   Clustering requested but scikit-learn not available")
            self.use_clustering = False

        logger.info("🔧 Geometric Rules Engine initialized")
        logger.info(f"   NDVI vegetation threshold: {ndvi_vegetation_threshold}")
        logger.info(f"   NDVI road threshold: {ndvi_road_threshold}")
        logger.info(
            f"   Road-vegetation height separation: {road_vegetation_height_threshold}m"
        )
        logger.info(f"   Building buffer distance: {building_buffer_distance}m")
        logger.info(f"   Verticality threshold: {verticality_threshold}")
        logger.info(f"   Verticality search radius: {verticality_search_radius}m")
        if self.use_clustering:
            logger.info(
                f"   Clustering enabled: eps={spatial_cluster_eps}m, min_size={min_cluster_size}"
            )

    def apply_all_rules(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        ground_truth_features: Dict[str, gpd.GeoDataFrame],
        ndvi: Optional[np.ndarray] = None,
        intensities: Optional[np.ndarray] = None,
        rgb: Optional[np.ndarray] = None,
        nir: Optional[np.ndarray] = None,
        verticality: Optional[np.ndarray] = None,
        preserve_ground_truth: bool = True,  # ✅ NEW: Preserve GT labels
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Apply all geometric rules to improve classification.

        Args:
            points: XYZ coordinates [N, 3]
            labels: Current classification labels [N] (will be modified)
            ground_truth_features: Dict of feature_type -> GeoDataFrame
            ndvi: Optional NDVI values [N] for each point (-1 to 1)
            intensities: Optional intensity values [N]
            rgb: Optional RGB values [N, 3] normalized to [0, 1]
            nir: Optional NIR values [N] normalized to [0, 1]
            verticality: Optional verticality values [N] (0-1) for relaxed
                classification rules
            preserve_ground_truth: If True, only modify unclassified points
                (ASPRS code 1). If False, all points can be reclassified.

        Returns:
            Tuple of:
            - Updated labels [N]
            - Statistics dict with counts per rule
        """
        stats = {}
        updated_labels = labels.copy()

        logger.info("🔧 Applying geometric rules for classification refinement...")

        # ✅ NEW: Create mask for points that can be modified
        if preserve_ground_truth:
            # Only unclassified points (code 1) can be modified
            modifiable_mask = updated_labels == int(ASPRSClass.UNCLASSIFIED)
            n_modifiable = np.sum(modifiable_mask)
            logger.info(
                f"  GT preservation enabled: {n_modifiable:,} modifiable points"
            )
        else:
            # All points can be modified
            modifiable_mask = np.ones(len(points), dtype=bool)

        # ✅ FIX Bug #3: Apply NDVI refinement FIRST (before geometric rules)
        # This ensures NDVI-based labels are not overwritten by verticality rules
        if ndvi is not None:
            # Calculate height above ground for better vegetation sub-classification
            height = None
            ground_mask = updated_labels == int(ASPRSClass.GROUND)
            if np.any(ground_mask):
                height = self.get_height_above_ground(
                    points=points, labels=updated_labels, search_radius=5.0
                )

            n_refined = self.apply_ndvi_refinement(
                points=points,
                labels=updated_labels,
                ndvi=ndvi,
                height=height,
                modifiable_mask=modifiable_mask,
            )
            stats["ndvi_refined"] = n_refined
            if n_refined > 0:
                logger.info(
                    f"  ✓ NDVI Refinement (First): Refined {n_refined:,} points"
                )

            # ✅ NEW: Protect NDVI-modified labels from future rules
            # Points that were reclassified by NDVI should not be touched
            # by geometric rules
            ndvi_modified_mask = labels != updated_labels
            if np.any(ndvi_modified_mask):
                # Remove NDVI-modified points from modifiable mask
                modifiable_mask = modifiable_mask & ~ndvi_modified_mask
                n_protected = np.sum(ndvi_modified_mask)
                logger.info(
                    f"  NDVI labels protected: {n_protected:,} points "
                    "locked from further modification"
                )

        # Rule 1: Road-vegetation disambiguation
        if "roads" in ground_truth_features and ndvi is not None:
            n_fixed = self.fix_road_vegetation_overlap(
                points=points,
                labels=updated_labels,
                road_geometries=ground_truth_features["roads"],
                ndvi=ndvi,
                modifiable_mask=modifiable_mask,
            )
            stats["road_vegetation_fixed"] = n_fixed
            if n_fixed > 0:
                logger.info(
                    f"  ✓ Rule 1 (Road-Vegetation): " f"Fixed {n_fixed:,} points"
                )

        # Rule 2: Building buffer zone classification with verticality
        if "buildings" in ground_truth_features:
            # Use clustered version if enabled (10-100x faster)
            if self.use_clustering:
                n_added = self.classify_building_buffer_zone_clustered(
                    points=points,
                    labels=updated_labels,
                    building_geometries=ground_truth_features["buildings"],
                    modifiable_mask=modifiable_mask,
                )
            else:
                n_added = self.classify_building_buffer_zone(
                    points=points,
                    labels=updated_labels,
                    building_geometries=ground_truth_features["buildings"],
                    modifiable_mask=modifiable_mask,
                )
            stats["building_buffer_added"] = n_added
            if n_added > 0:
                method = "Clustered" if self.use_clustering else "Standard"
                logger.info(
                    f"  ✓ Rule 2 (Building Buffer - {method}): "
                    f"Added {n_added:,} points"
                )

        # Rule 2b: Verticality-based building classification
        n_vertical = self.classify_by_verticality(
            points=points,
            labels=updated_labels,
            ndvi=ndvi,
            modifiable_mask=modifiable_mask,
        )
        stats["verticality_buildings_added"] = n_vertical
        if n_vertical > 0:
            logger.info(
                f"  ✓ Rule 2b (Verticality): " f"Added {n_vertical:,} building points"
            )

        # Rule 4: Advanced spectral classification (if RGB + NIR available)
        if self.spectral_rules is not None and rgb is not None and nir is not None:
            logger.info("  Applying advanced spectral classification rules...")
            updated_labels, spectral_stats = (
                self.spectral_rules.classify_by_spectral_signature(
                    rgb=rgb,
                    nir=nir,
                    current_labels=updated_labels,
                    ndvi=ndvi,
                    apply_to_unclassified_only=True,
                )
            )
            stats.update(spectral_stats)
            if spectral_stats.get("total_reclassified", 0) > 0:
                logger.info(
                    f"  ✓ Rule 4 (Spectral): Classified {spectral_stats['total_reclassified']:,} points"
                )

            # ✅ NOUVEAU - Rule 4b: Classification relaxée pour points restants non classifiés
            logger.info(
                "  Applying relaxed classification rules for remaining unclassified points..."
            )
            updated_labels, relaxed_stats = (
                self.spectral_rules.classify_unclassified_relaxed(
                    rgb=rgb,
                    nir=nir,
                    current_labels=updated_labels,
                    ndvi=ndvi,
                    verticality=verticality,
                    heights=height,
                )
            )
            stats.update(relaxed_stats)
            if relaxed_stats.get("total_relaxed", 0) > 0:
                logger.info(
                    f"  ✓ Rule 4b (Relaxed): Classified {relaxed_stats['total_relaxed']:,} additional points"
                )

        # Calculate total changes
        n_changed = np.sum(labels != updated_labels)
        stats["total_changed"] = n_changed

        logger.info(f"📊 Geometric rules applied: {n_changed:,} points modified")

        return updated_labels, stats

    def fix_road_vegetation_overlap(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        road_geometries: gpd.GeoDataFrame,
        ndvi: np.ndarray,
        modifiable_mask: np.ndarray,
    ) -> int:
        """
        Fix misclassified vegetation points on roads using vertical
        separation and NDVI.

        Logic:
        - Points above roads (e.g., tree canopy) with high NDVI
          -> keep as vegetation
        - Points on road surface with low NDVI -> reclassify to road
        - Uses height difference and NDVI to determine if vegetation
          is ON road or ABOVE road

        Args:
            points: XYZ coordinates [N, 3]
            labels: Classification labels [N] (modified in-place)
            road_geometries: GeoDataFrame with road polygons
            ndvi: NDVI values [N] for each point
            modifiable_mask: Boolean mask [N] indicating which points
                can be modified

        Returns:
            Number of points reclassified
        """
        if len(road_geometries) == 0:
            return 0

        n_fixed = 0

        # Find points currently classified as vegetation AND modifiable
        vegetation_mask = np.isin(
            labels,
            [
                int(ASPRSClass.LOW_VEGETATION),
                int(ASPRSClass.MEDIUM_VEGETATION),
                int(ASPRSClass.HIGH_VEGETATION),
            ],
        )
        # ✅ NEW: Only process modifiable vegetation points
        vegetation_mask = vegetation_mask & modifiable_mask

        if not np.any(vegetation_mask):
            return 0

        vegetation_indices = np.where(vegetation_mask)[0]
        vegetation_points = points[vegetation_mask]
        vegetation_ndvi = ndvi[vegetation_mask]

        # Build spatial index for road geometries
        tree = STRtree(road_geometries.geometry.values)

        # For each vegetation point, check if it's on a road
        for i, (pt, pt_ndvi) in enumerate(zip(vegetation_points, vegetation_ndvi)):
            global_idx = vegetation_indices[i]
            pt_geom = Point(pt[0], pt[1])

            # Query nearby road polygons
            possible_roads = tree.query(pt_geom)

            for road_idx in possible_roads:
                road_geom = road_geometries.geometry.iloc[road_idx]

                if not road_geom.contains(pt_geom):
                    continue

                # Point is within road footprint
                # Decision logic:
                # 1. Low NDVI (<= threshold) -> likely road surface (e.g., painted lines, asphalt)
                # 2. High NDVI but low height difference -> vegetation ON road (e.g., weeds)
                # 3. High NDVI and high height difference -> vegetation ABOVE road (e.g., tree)

                if pt_ndvi <= self.ndvi_road_threshold:
                    # Low NDVI: definitely road, not vegetation
                    labels[global_idx] = int(ASPRSClass.ROAD_SURFACE)
                    n_fixed += 1
                    break

                # For medium NDVI (ambiguous), we need height information
                # Estimate ground height from nearby ground points
                # Find ground points within 5m radius
                ground_mask = labels == int(ASPRSClass.GROUND)
                if np.any(ground_mask):
                    ground_points = points[ground_mask]

                    # Find nearby ground points
                    distances = np.sqrt(
                        (ground_points[:, 0] - pt[0]) ** 2
                        + (ground_points[:, 1] - pt[1]) ** 2
                    )
                    nearby_ground = distances < 5.0

                    if np.any(nearby_ground):
                        # Estimate ground height (median of nearby ground points)
                        ground_height = np.median(ground_points[nearby_ground, 2])
                        height_above_ground = pt[2] - ground_height

                        # If vegetation is low (<2m) and on road, likely misclassified
                        if height_above_ground < self.road_vegetation_height_threshold:
                            # Low vegetation on road -> reclassify to road
                            labels[global_idx] = int(ASPRSClass.ROAD_SURFACE)
                            n_fixed += 1
                            break
                        # else: high vegetation (tree canopy) above road -> keep as vegetation

                break  # Stop after first matching road

        return n_fixed

    def classify_building_buffer_zone(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        building_geometries: gpd.GeoDataFrame,
        modifiable_mask: np.ndarray,
    ) -> int:
        """
        Classify unclassified points near buildings as building points.

        Logic:
        - Find unclassified points within buffer distance of buildings
        - Check height similarity to nearby building points
        - If height is similar, classify as building

        Args:
            points: XYZ coordinates [N, 3]
            labels: Classification labels [N] (modified in-place)
            building_geometries: GeoDataFrame with building polygons
            modifiable_mask: Boolean mask [N] indicating which points
                can be modified

        Returns:
            Number of points classified as building
        """
        if len(building_geometries) == 0:
            return 0

        n_added = 0

        # Find unclassified points that are modifiable
        unclassified_mask = (labels == int(ASPRSClass.UNCLASSIFIED)) & modifiable_mask

        if not np.any(unclassified_mask):
            return 0

        unclassified_indices = np.where(unclassified_mask)[0]
        unclassified_points = points[unclassified_mask]

        # Find existing building points (for height reference)
        building_mask = labels == int(ASPRSClass.BUILDING)
        building_points = points[building_mask]

        if len(building_points) == 0:
            logger.debug("No existing building points for reference")
            return 0

        # Create buffered building geometries
        buffered_buildings = building_geometries.geometry.buffer(
            self.building_buffer_distance
        )
        tree = STRtree(buffered_buildings.values)

        # For each unclassified point, check if it's near a building
        for i, pt in enumerate(unclassified_points):
            global_idx = unclassified_indices[i]
            pt_geom = Point(pt[0], pt[1])

            # Query nearby buffered buildings
            possible_buildings = tree.query(pt_geom)

            for building_idx in possible_buildings:
                buffered_geom = buffered_buildings.iloc[building_idx]

                if not buffered_geom.contains(pt_geom):
                    continue

                # Point is within buffer zone
                # Check height similarity to nearby building points (GPU-accelerated KNN)
                # Find nearest building points (within 5m)
                distances, indices = knn(
                    building_points[:, :2],  # XY only
                    pt[:2].reshape(1, -1),
                    k=min(10, len(building_points))
                )
                distances, indices = distances[0], indices[0]  # Extract from batch
                # Manual distance threshold
                distances = distances.copy()  # Ensure we can filter
                distances[distances > 5.0] = float("inf")

                valid_neighbors = distances < float("inf")
                if not np.any(valid_neighbors):
                    continue

                neighbor_heights = building_points[indices[valid_neighbors], 2]
                median_building_height = np.median(neighbor_heights)
                height_diff = abs(pt[2] - median_building_height)

                # If height is similar to nearby building points, classify as building
                if height_diff < self.max_building_height_difference:
                    labels[global_idx] = int(ASPRSClass.BUILDING)
                    n_added += 1
                    break

        return n_added

    def classify_building_buffer_zone_clustered(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        building_geometries: gpd.GeoDataFrame,
        modifiable_mask: np.ndarray,
    ) -> int:
        """
        Classify unclassified points near buildings using spatial
        clustering (10-100x faster).

        Logic:
        1. Extract all unclassified points within building buffers
        2. Cluster points by spatial proximity
        3. For each cluster, check height consistency with nearby
           buildings
        4. Classify entire clusters at once (batch processing)

        Args:
            points: XYZ coordinates [N, 3]
            labels: Classification labels [N] (modified in-place)
            building_geometries: GeoDataFrame with building polygons
            modifiable_mask: Boolean mask [N] indicating which points
                can be modified

        Returns:
            Number of points classified as building
        """
        if len(building_geometries) == 0:
            return 0

        # Find unclassified points that are modifiable
        unclassified_mask = (labels == int(ASPRSClass.UNCLASSIFIED)) & modifiable_mask
        if not np.any(unclassified_mask):
            return 0

        unclassified_indices = np.where(unclassified_mask)[0]
        unclassified_points = points[unclassified_mask]

        # Step 1: Filter points within buffer zone
        buffered_buildings = building_geometries.geometry.buffer(
            self.building_buffer_distance
        )
        tree = STRtree(buffered_buildings.values)

        buffer_zone_mask = np.zeros(len(unclassified_points), dtype=bool)
        for i, pt in enumerate(unclassified_points):
            pt_geom = Point(pt[0], pt[1])
            possible = tree.query(pt_geom)
            for idx in possible:
                if buffered_buildings.iloc[idx].contains(pt_geom):
                    buffer_zone_mask[i] = True
                    break

        if not np.any(buffer_zone_mask):
            return 0

        buffer_points = unclassified_points[buffer_zone_mask]
        buffer_indices = unclassified_indices[buffer_zone_mask]

        logger.info(f"  Found {len(buffer_points):,} points in buffer zone")

        # Step 2: Cluster buffer zone points by spatial proximity
        logger.info(f"  Clustering buffer zone points...")
        clustering = DBSCAN(
            eps=self.spatial_cluster_eps, min_samples=self.min_cluster_size
        )
        cluster_labels = clustering.fit_predict(buffer_points[:, :3])  # Use XYZ

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        logger.info(f"  Found {n_clusters} clusters")

        # Step 3: Classify each cluster
        n_added = 0
        building_mask = labels == int(ASPRSClass.BUILDING)

        if not np.any(building_mask):
            return 0

        building_points = points[building_mask]

        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Skip noise points
                continue

            cluster_mask = cluster_labels == cluster_id
            cluster_pts = buffer_points[cluster_mask]
            cluster_global_idx = buffer_indices[cluster_mask]

            # Get cluster statistics
            cluster_center = np.mean(cluster_pts, axis=0)
            cluster_mean_height = cluster_center[2]

            # Find nearest building points (GPU-accelerated KNN)
            distances, indices = knn(
                building_points[:, :2],  # XY only
                cluster_center[:2].reshape(1, -1),
                k=min(10, len(building_points))
            )
            distances, indices = distances[0], indices[0]  # Extract from batch
            # Manual distance threshold
            distances = distances.copy()
            distances[distances > 5.0] = float("inf")

            valid = distances < float("inf")
            if not np.any(valid):
                continue

            # Check height consistency
            neighbor_heights = building_points[indices[valid], 2]
            median_building_height = np.median(neighbor_heights)
            height_diff = abs(cluster_mean_height - median_building_height)

            if height_diff < self.max_building_height_difference:
                # Classify entire cluster as building
                labels[cluster_global_idx] = int(ASPRSClass.BUILDING)
                n_added += len(cluster_global_idx)

        return n_added

    def apply_ndvi_refinement(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        ndvi: np.ndarray,
        height: Optional[np.ndarray] = None,
        modifiable_mask: Optional[np.ndarray] = None,
    ) -> int:
        """
        Apply multi-level NDVI-based refinement for all classification
        types.

        Uses multi-level NDVI thresholds aligned with
        advanced_classification.py:
        - Dense forest (≥0.60): HIGH vegetation
        - Healthy trees (≥0.50): HIGH/MED vegetation
        - Moderate vegetation (≥0.40): MEDIUM vegetation
        - Grass/shrubs (≥0.30): LOW/MED vegetation
        - Sparse vegetation (≥0.20): LOW vegetation
        - Road/impervious (≤0.15): Non-vegetation

        Logic:
        - High NDVI points classified as non-vegetation -> reclassify
          to vegetation (multi-level)
        - Low NDVI points classified as vegetation -> check if should
          be non-vegetation
        - Unclassified points with clear NDVI signal -> classify
          (multi-level)

        Args:
            points: XYZ coordinates [N, 3]
            labels: Classification labels [N] (modified in-place)
            ndvi: NDVI values [N] for each point
            height: Optional height above ground [N] for
                sub-classification
            modifiable_mask: Optional boolean mask [N] indicating which
                points can be modified. If None, all points can be
                modified.

        Returns:
            Number of points refined
        """
        n_refined = 0

        # Create modifiable mask if not provided
        if modifiable_mask is None:
            modifiable_mask = np.ones(len(points), dtype=bool)

        # Rule 1: Non-vegetation with high NDVI -> vegetation
        # (multi-level)
        non_veg_mask = (
            ~np.isin(
                labels,
                [
                    int(ASPRSClass.LOW_VEGETATION),
                    int(ASPRSClass.MEDIUM_VEGETATION),
                    int(ASPRSClass.HIGH_VEGETATION),
                    int(ASPRSClass.UNCLASSIFIED),
                    int(ASPRSClass.GROUND),
                ],
            )
            & modifiable_mask
        )

        # Multi-level NDVI classification for non-vegetation
        # Dense forest (NDVI ≥ 0.60) -> HIGH vegetation
        dense_forest = non_veg_mask & (ndvi >= self.NDVI_DENSE_FOREST)
        if np.any(dense_forest):
            labels[dense_forest] = int(ASPRSClass.HIGH_VEGETATION)
            n_refined += np.sum(dense_forest)

        # Healthy trees (0.50 ≤ NDVI < 0.60) -> HIGH/MED vegetation
        healthy_trees = (
            non_veg_mask
            & (ndvi >= self.NDVI_HEALTHY_TREES)
            & (ndvi < self.NDVI_DENSE_FOREST)
        )
        if np.any(healthy_trees):
            if height is not None:
                # Use height to sub-classify
                high_trees = healthy_trees & (height >= 3.0)
                med_trees = healthy_trees & (height < 3.0)
                labels[high_trees] = int(ASPRSClass.HIGH_VEGETATION)
                labels[med_trees] = int(ASPRSClass.MEDIUM_VEGETATION)
                n_refined += np.sum(healthy_trees)
            else:
                # Default to high vegetation
                labels[healthy_trees] = int(ASPRSClass.HIGH_VEGETATION)
                n_refined += np.sum(healthy_trees)

        # Moderate vegetation (0.40 ≤ NDVI < 0.50) -> MEDIUM vegetation
        moderate_veg = (
            non_veg_mask
            & (ndvi >= self.NDVI_MODERATE_VEG)
            & (ndvi < self.NDVI_HEALTHY_TREES)
        )
        if np.any(moderate_veg):
            labels[moderate_veg] = int(ASPRSClass.MEDIUM_VEGETATION)
            n_refined += np.sum(moderate_veg)

        # Grass/shrubs (0.30 ≤ NDVI < 0.40) -> LOW/MED vegetation
        grass = (
            non_veg_mask & (ndvi >= self.NDVI_GRASS) & (ndvi < self.NDVI_MODERATE_VEG)
        )
        if np.any(grass):
            if height is not None:
                # Use height to sub-classify
                tall_grass = grass & (height >= 1.0)
                short_grass = grass & (height < 1.0)
                labels[tall_grass] = int(ASPRSClass.MEDIUM_VEGETATION)
                labels[short_grass] = int(ASPRSClass.LOW_VEGETATION)
                n_refined += np.sum(grass)
            else:
                # Default to low vegetation
                labels[grass] = int(ASPRSClass.LOW_VEGETATION)
                n_refined += np.sum(grass)

        # Sparse vegetation (0.20 ≤ NDVI < 0.30) -> LOW vegetation
        sparse_veg = (
            non_veg_mask & (ndvi >= self.NDVI_SPARSE_VEG) & (ndvi < self.NDVI_GRASS)
        )
        if np.any(sparse_veg):
            labels[sparse_veg] = int(ASPRSClass.LOW_VEGETATION)
            n_refined += np.sum(sparse_veg)

        # Rule 2: Vegetation with very low NDVI -> might be
        # misclassified
        # (Be conservative here - only very low NDVI, below road
        # threshold)
        veg_mask = (
            np.isin(
                labels,
                [
                    int(ASPRSClass.LOW_VEGETATION),
                    int(ASPRSClass.MEDIUM_VEGETATION),
                    int(ASPRSClass.HIGH_VEGETATION),
                ],
            )
            & modifiable_mask
        )
        very_low_ndvi_veg = veg_mask & (ndvi <= self.NDVI_ROAD)  # Below road threshold

        if np.any(very_low_ndvi_veg):
            # Reclassify to unclassified (let other rules handle it)
            labels[very_low_ndvi_veg] = int(ASPRSClass.UNCLASSIFIED)
            n_refined += np.sum(very_low_ndvi_veg)

        # Rule 3: Unclassified with clear NDVI signal -> classify
        # (multi-level)
        unclassified_mask = (labels == int(ASPRSClass.UNCLASSIFIED)) & modifiable_mask

        # Dense forest (NDVI ≥ 0.60) -> HIGH vegetation
        dense_forest_unc = unclassified_mask & (ndvi >= self.NDVI_DENSE_FOREST)
        if np.any(dense_forest_unc):
            labels[dense_forest_unc] = int(ASPRSClass.HIGH_VEGETATION)
            n_refined += np.sum(dense_forest_unc)

        # Healthy trees (0.50 ≤ NDVI < 0.60) -> HIGH/MED vegetation
        healthy_trees_unc = (
            unclassified_mask
            & (ndvi >= self.NDVI_HEALTHY_TREES)
            & (ndvi < self.NDVI_DENSE_FOREST)
        )
        if np.any(healthy_trees_unc):
            if height is not None:
                high_trees = healthy_trees_unc & (height >= 3.0)
                med_trees = healthy_trees_unc & (height < 3.0)
                labels[high_trees] = int(ASPRSClass.HIGH_VEGETATION)
                labels[med_trees] = int(ASPRSClass.MEDIUM_VEGETATION)
                n_refined += np.sum(healthy_trees_unc)
            else:
                labels[healthy_trees_unc] = int(ASPRSClass.HIGH_VEGETATION)
                n_refined += np.sum(healthy_trees_unc)

        # Moderate vegetation (0.40 ≤ NDVI < 0.50) -> MEDIUM vegetation
        moderate_veg_unc = (
            unclassified_mask
            & (ndvi >= self.NDVI_MODERATE_VEG)
            & (ndvi < self.NDVI_HEALTHY_TREES)
        )
        if np.any(moderate_veg_unc):
            labels[moderate_veg_unc] = int(ASPRSClass.MEDIUM_VEGETATION)
            n_refined += np.sum(moderate_veg_unc)

        # Grass/shrubs (0.30 ≤ NDVI < 0.40) -> LOW/MED vegetation
        grass_unc = (
            unclassified_mask
            & (ndvi >= self.NDVI_GRASS)
            & (ndvi < self.NDVI_MODERATE_VEG)
        )
        if np.any(grass_unc):
            if height is not None:
                tall_grass = grass_unc & (height >= 1.0)
                short_grass = grass_unc & (height < 1.0)
                labels[tall_grass] = int(ASPRSClass.MEDIUM_VEGETATION)
                labels[short_grass] = int(ASPRSClass.LOW_VEGETATION)
                n_refined += np.sum(grass_unc)
            else:
                labels[grass_unc] = int(ASPRSClass.LOW_VEGETATION)
                n_refined += np.sum(grass_unc)

        # Sparse vegetation (0.20 ≤ NDVI < 0.30) -> LOW vegetation
        sparse_veg_unc = (
            unclassified_mask
            & (ndvi >= self.NDVI_SPARSE_VEG)
            & (ndvi < self.NDVI_GRASS)
        )
        if np.any(sparse_veg_unc):
            labels[sparse_veg_unc] = int(ASPRSClass.LOW_VEGETATION)
            n_refined += np.sum(sparse_veg_unc)

        return n_refined

    def classify_by_verticality(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        ndvi: Optional[np.ndarray] = None,
        modifiable_mask: Optional[np.ndarray] = None,
    ) -> int:
        """
        Classify unclassified points as buildings based on verticality
        analysis.

        Verticality measures how vertically aligned points are in a
        local neighborhood,
        which is a strong indicator of building walls and structures.

        Logic:
        - Compute verticality score for unclassified points
        - High verticality + low NDVI = likely building
        - Check height consistency with nearby classified building
          points

        Args:
            points: XYZ coordinates [N, 3]
            labels: Classification labels [N] (modified in-place)
            ndvi: Optional NDVI values [N] to exclude vegetation
            modifiable_mask: Optional boolean mask [N] indicating which
                points can be modified. If None, all points can be
                modified.

        Returns:
            Number of points classified as building
        """
        n_added = 0

        # Find unclassified points
        unclassified_mask = labels == int(ASPRSClass.UNCLASSIFIED)

        # Apply modifiable mask if provided
        if modifiable_mask is not None:
            unclassified_mask = unclassified_mask & modifiable_mask

        if not np.any(unclassified_mask):
            return 0

        unclassified_indices = np.where(unclassified_mask)[0]
        unclassified_points = points[unclassified_mask]

        # Compute verticality for unclassified points (KNN computed inside)
        verticality_scores = self.compute_verticality(
            points=points, query_points=unclassified_points
        )

        # Filter by verticality threshold
        high_verticality_mask = verticality_scores >= self.verticality_threshold

        if not np.any(high_verticality_mask):
            return 0

        # Further filter by NDVI if available (exclude high NDVI = vegetation)
        if ndvi is not None:
            unclassified_ndvi = ndvi[unclassified_mask]
            # Keep points with low NDVI (not vegetation)
            low_ndvi_mask = unclassified_ndvi < self.ndvi_vegetation_threshold
            candidate_mask = high_verticality_mask & low_ndvi_mask
        else:
            candidate_mask = high_verticality_mask

        if not np.any(candidate_mask):
            return 0

        # Get candidate points
        candidate_indices = unclassified_indices[candidate_mask]
        candidate_points = unclassified_points[candidate_mask]

        # Check if there are nearby building points for validation
        building_mask = labels == int(ASPRSClass.BUILDING)

        if np.any(building_mask):
            building_points = points[building_mask]

            # For each candidate, check proximity and height consistency with buildings
            for i, pt in enumerate(candidate_points):
                global_idx = candidate_indices[i]

                # Find nearby building points (within 10m) - GPU-accelerated KNN
                distances, indices = knn(
                    building_points[:, :2],  # XY only
                    pt[:2].reshape(1, -1),
                    k=10
                )
                distances, indices = distances[0], indices[0]  # Extract from batch

                # Manual distance threshold (distance_upper_bound=10.0)
                valid_neighbors = distances < 10.0

                if np.any(valid_neighbors):
                    # Check height consistency
                    neighbor_heights = building_points[indices[valid_neighbors], 2]
                    median_building_height = np.median(neighbor_heights)
                    height_diff = abs(pt[2] - median_building_height)

                    # If height is consistent, classify as building
                    if (
                        height_diff < self.max_building_height_difference * 2
                    ):  # More lenient
                        labels[global_idx] = int(ASPRSClass.BUILDING)
                        n_added += 1
                else:
                    # No nearby buildings, but high verticality suggests new building
                    # Be more conservative: require very high verticality
                    vert_score = verticality_scores[candidate_mask][i]
                    if vert_score >= 0.85:  # Very high verticality
                        labels[global_idx] = int(ASPRSClass.BUILDING)
                        n_added += 1
        else:
            # No existing building points, classify based on verticality alone
            # Require very high verticality score
            very_high_vert = verticality_scores[candidate_mask] >= 0.85
            very_high_indices = candidate_indices[very_high_vert]

            labels[very_high_indices] = int(ASPRSClass.BUILDING)
            n_added = len(very_high_indices)

        return n_added

    def compute_verticality(
        self,
        points: np.ndarray,
        query_points: np.ndarray,
    ) -> np.ndarray:
        """
        Compute verticality score for query points.

        Verticality measures the ratio of vertical extent to horizontal extent
        in a local neighborhood. High verticality indicates vertical structures
        like building walls.

        Algorithm:
        1. For each query point, find neighbors within search radius (GPU KNN)
        2. Compute vertical extent (max Z - min Z)
        3. Compute horizontal extent (max of XY spread)
        4. Verticality = vertical_extent / (horizontal_extent + epsilon)
        5. Normalize to [0, 1] range

        Args:
            points: All points [N, 3] for neighborhood search
            query_points: Points to compute verticality for [M, 3]

        Returns:
            Verticality scores [M] in range [0, 1]
        """
        verticality_scores = np.zeros(len(query_points))

        for i, query_pt in enumerate(query_points):
            # Find neighbors within search radius (GPU-accelerated KNN + filter)
            # Use k=50 as approximate radius search, then filter by distance
            distances, indices = knn(
                points,
                query_pt.reshape(1, -1),
                k=min(50, len(points))
            )
            distances, indices = distances[0], indices[0]  # Extract from batch

            # Filter by radius
            radius_mask = distances <= self.verticality_search_radius
            indices = indices[radius_mask]

            if len(indices) < self.min_vertical_neighbors:
                # Not enough neighbors, verticality = 0
                verticality_scores[i] = 0.0
                continue

            # Get neighbor points
            neighbors = points[indices]

            # Compute vertical extent
            z_min = neighbors[:, 2].min()
            z_max = neighbors[:, 2].max()
            vertical_extent = z_max - z_min

            # Compute horizontal extent (max of X and Y spread)
            x_extent = neighbors[:, 0].max() - neighbors[:, 0].min()
            y_extent = neighbors[:, 1].max() - neighbors[:, 1].min()
            horizontal_extent = max(x_extent, y_extent)

            # Avoid division by zero
            if horizontal_extent < 0.01:  # Less than 1cm
                # Very small horizontal extent, check if vertical
                if vertical_extent > 0.5:  # At least 50cm vertical
                    verticality_scores[i] = 1.0
                else:
                    verticality_scores[i] = 0.0
                continue

            # Compute verticality ratio
            verticality_ratio = vertical_extent / horizontal_extent

            # Normalize to [0, 1] range
            # Typical building walls have ratio > 2 (e.g., 3m height, 1m width)
            # Normalize such that ratio of 2 = 0.7, ratio of 5+ = 1.0
            verticality_score = min(1.0, verticality_ratio / 5.0)

            verticality_scores[i] = verticality_score

        return verticality_scores

    def get_height_above_ground(
        self, points: np.ndarray, labels: np.ndarray, search_radius: float = 5.0
    ) -> np.ndarray:
        """
        Calculate height above ground for all points.

        Args:
            points: XYZ coordinates [N, 3]
            labels: Classification labels [N]
            search_radius: Radius to search for ground points (meters)

        Returns:
            Height above ground [N] for each point
        """
        heights = np.zeros(len(points))

        # Find ground points
        ground_mask = labels == int(ASPRSClass.GROUND)

        if not np.any(ground_mask):
            logger.warning("No ground points available for height calculation")
            return heights

        ground_points = points[ground_mask]

        # For each point, find nearby ground points and estimate ground height
        for i, pt in enumerate(points):
            # Query nearby ground points (GPU-accelerated KNN)
            distances, indices = knn(
                ground_points[:, :2],  # XY only
                pt[:2].reshape(1, -1),
                k=min(5, len(ground_points))
            )
            distances, indices = distances[0], indices[0]  # Extract from batch

            # Manual distance threshold (distance_upper_bound=search_radius)
            valid_neighbors = distances < search_radius

            if np.any(valid_neighbors):
                neighbor_heights = ground_points[indices[valid_neighbors], 2]
                ground_height = np.median(neighbor_heights)
                heights[i] = pt[2] - ground_height
            else:
                # No nearby ground points, use point height directly
                heights[i] = pt[2]

        return heights
