"""
ASPRS Class-Specific Rules Engine (Phase 3)

This module implements detection rules for specific ASPRS classes that require
specialized geometric, spectral, and spatial analysis beyond basic ground truth
containment.

Author: Simon Ducournau
Date: October 25, 2025
Version: 6.0.0
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from ign_lidar.optimization import cKDTree  # GPU-accelerated drop-in replacement
from ign_lidar.optimization.gpu_accelerated_ops import knn
from ign_lidar.classification_schema import ASPRSClass

logger = logging.getLogger(__name__)


@dataclass
class WaterDetectionConfig:
    """Configuration for water body detection."""

    enabled: bool = True
    planarity_min: float = 0.85
    ndvi_max: float = 0.15
    ndwi_min: float = 0.20
    height_max: float = 0.5
    curvature_max: float = 0.05
    min_cluster_size: int = 100
    use_bd_topo_water: bool = True


@dataclass
class BridgeDetectionConfig:
    """Configuration for bridge detection."""

    enabled: bool = True
    height_min: float = 3.0
    height_max: float = 50.0
    planarity_min: float = 0.75
    verticality_max: float = 0.30
    road_proximity_max: float = 5.0
    water_proximity_max: float = 30.0
    min_length: float = 10.0
    max_width: float = 50.0
    linearity_min: float = 0.65
    use_road_alignment: bool = True


@dataclass
class RailwayDetectionConfig:
    """Configuration for railway detection."""

    enabled: bool = True
    height_min: float = -0.2
    height_max: float = 2.0
    planarity_min: float = 0.70
    linearity_min: float = 0.75
    ndvi_max: float = 0.25
    width_min: float = 2.0
    width_max: float = 6.0
    parallel_track_detection: bool = True
    track_spacing: List[float] = None
    use_bd_topo_railways: bool = True

    def __post_init__(self):
        if self.track_spacing is None:
            self.track_spacing = [1.4, 1.5, 1.6]  # Standard gauge ± tolerance


@dataclass
class OverheadStructureDetectionConfig:
    """Configuration for overhead structure detection (power lines, cables)."""

    enabled: bool = True
    height_min: float = 5.0
    height_max: float = 100.0
    linearity_min: float = 0.80
    planarity_max: float = 0.40
    verticality_max: float = 0.30
    min_length: float = 20.0
    max_thickness: float = 0.5
    near_building_distance: float = 50.0
    exclude_vegetation: bool = True


@dataclass
class NoiseClassificationConfig:
    """Configuration for noise classification."""

    enabled: bool = True
    isolation_threshold: float = 2.0
    min_neighbors: int = 3
    height_deviation_max: float = 5.0
    classify_as_noise: bool = True


class ASPRSClassRulesEngine:
    """
    ASPRS Class-Specific Rules Engine.

    Implements detection rules for ASPRS classes that require specialized
    analysis beyond basic ground truth containment.

    Supported classes:
    - Water bodies (ASPRS 9)
    - Bridges (ASPRS 17)
    - Railways (railway-related)
    - Overhead structures (power lines, cables)
    - Noise (ASPRS 7)
    """

    def __init__(
        self,
        water_config: Optional[WaterDetectionConfig] = None,
        bridge_config: Optional[BridgeDetectionConfig] = None,
        railway_config: Optional[RailwayDetectionConfig] = None,
        overhead_config: Optional[OverheadStructureDetectionConfig] = None,
        noise_config: Optional[NoiseClassificationConfig] = None,
    ):
        """
        Initialize the ASPRS class rules engine.

        Args:
            water_config: Water body detection configuration
            bridge_config: Bridge detection configuration
            railway_config: Railway detection configuration
            overhead_config: Overhead structure detection configuration
            noise_config: Noise classification configuration
        """
        self.water_config = water_config or WaterDetectionConfig()
        self.bridge_config = bridge_config or BridgeDetectionConfig()
        self.railway_config = railway_config or RailwayDetectionConfig()
        self.overhead_config = overhead_config or OverheadStructureDetectionConfig()
        self.noise_config = noise_config or NoiseClassificationConfig()

        logger.info("ASPRS Class Rules Engine initialized")

    def _check_spatial_containment(
        self,
        points: np.ndarray,
        mask: np.ndarray,
        polygons: Any,
        buffer_m: float = 0.0,
        use_strtree: bool = True,
    ) -> np.ndarray:
        """
        Check if points are within polygons (with optional buffer).

        Uses STRtree spatial indexing for efficient querying when
        available.

        Args:
            points: [N, 3] point coordinates (XYZ)
            mask: [N] boolean mask of candidate points to check
            polygons: GeoDataFrame or list of shapely geometries
            buffer_m: Buffer distance in meters
                (positive=expand, negative=inset)
            use_strtree: Whether to use STRtree spatial index (faster)

        Returns:
            [N] refined boolean mask (subset of input mask)
        """
        try:
            import geopandas as gpd
            from shapely.geometry import Point
            from shapely.strtree import STRtree
        except ImportError:
            logger.warning(
                "shapely/geopandas not available, " "skipping spatial containment"
            )
            return mask

        if polygons is None or len(polygons) == 0:
            return mask

        # Convert to GeoDataFrame if needed
        if not isinstance(polygons, gpd.GeoDataFrame):
            polygons = gpd.GeoDataFrame(
                geometry=list(polygons),
                crs="EPSG:2154",  # Lambert 93 (French projection)
            )

        # Apply buffer if specified
        if buffer_m != 0.0:
            geoms = polygons.geometry.buffer(buffer_m)
        else:
            geoms = polygons.geometry

        # Build spatial index if requested and beneficial
        if use_strtree and len(geoms) > 10:
            tree = STRtree(geoms)
            use_index = True
        else:
            use_index = False
            tree = None

        # Check each candidate point
        refined_mask = mask.copy()
        candidate_indices = np.where(mask)[0]

        n_checked = 0
        n_contained = 0

        for idx in candidate_indices:
            point = Point(points[idx, 0], points[idx, 1])  # XY only

            # Query spatial index or check all geometries
            if use_index:
                # STRtree query returns indices of potential matches
                potential_matches = tree.query(point)
                contained = any(
                    geoms.iloc[i].contains(point) for i in potential_matches
                )
            else:
                # Check all geometries
                contained = any(geom.contains(point) for geom in geoms)

            if not contained:
                refined_mask[idx] = False
            else:
                n_contained += 1

            n_checked += 1

        logger.debug(
            f"Spatial containment: {n_contained}/{n_checked} " f"points contained"
        )

        return refined_mask

    def apply_all_rules(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        classification: np.ndarray,
        ground_truth: Optional[Dict[str, any]] = None,
    ) -> np.ndarray:
        """
        Apply all enabled class-specific rules.

        Args:
            points: Point cloud [N, 3] with XYZ coordinates
            features: Dictionary of computed features
            classification: Current classification array [N]
            ground_truth: Optional ground truth data (roads, water, etc.)

        Returns:
            Updated classification array [N]
        """
        n_points = len(points)
        logger.info(f"Applying ASPRS class-specific rules to {n_points:,} points")

        # Apply rules in order
        if self.water_config.enabled:
            classification = self.classify_water_bodies(
                points, features, classification, ground_truth
            )

        if self.bridge_config.enabled:
            classification = self.classify_bridges(
                points, features, classification, ground_truth
            )

        if self.railway_config.enabled:
            classification = self.classify_railways(
                points, features, classification, ground_truth
            )

        if self.overhead_config.enabled:
            classification = self.classify_overhead_structures(
                points, features, classification
            )

        if self.noise_config.enabled:
            classification = self.classify_noise(points, features, classification)

        logger.info("ASPRS class-specific rules applied successfully")
        return classification

    def classify_water_bodies(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        classification: np.ndarray,
        ground_truth: Optional[Dict[str, any]] = None,
    ) -> np.ndarray:
        """
        Classify water bodies using planarity, NDVI, NDWI, and height.

        Water characteristics:
        - High planarity (flat surface)
        - Low NDVI (no vegetation)
        - High NDWI (water index)
        - Near ground level
        - Low curvature

        Args:
            points: Point cloud [N, 3]
            features: Feature dictionary
            classification: Current classification [N]
            ground_truth: Optional ground truth with water polygons

        Returns:
            Updated classification [N]
        """
        cfg = self.water_config

        # Extract required features
        planarity = features.get("planarity", np.zeros(len(points)))
        height = features.get("height_above_ground", points[:, 2])
        curvature = features.get("curvature", np.zeros(len(points)))
        ndvi = features.get("ndvi", np.zeros(len(points)))

        # Check if NDWI is available
        ndwi = features.get("ndwi", None)

        # Build water candidate mask
        water_mask = (
            (planarity >= cfg.planarity_min)
            & (height <= cfg.height_max)
            & (curvature <= cfg.curvature_max)
            & (ndvi <= cfg.ndvi_max)
        )

        # Add NDWI constraint if available
        if ndwi is not None:
            water_mask &= ndwi >= cfg.ndwi_min

        # Use BD TOPO water polygons if available
        if cfg.use_bd_topo_water and ground_truth is not None:
            water_polygons = ground_truth.get("water", None)
            if water_polygons is not None:
                logger.info("Refining water detection with BD TOPO polygons")
                water_mask = self._check_spatial_containment(
                    points,
                    water_mask,
                    water_polygons,
                    buffer_m=2.0,  # Small buffer for edge tolerance
                )

        # Cluster water points to filter isolated detections
        if water_mask.sum() > 0:
            water_indices = np.where(water_mask)[0]
            water_points = points[water_indices]

            # DBSCAN clustering to remove isolated points
            clustering = DBSCAN(eps=2.0, min_samples=cfg.min_cluster_size).fit(
                water_points[:, :2]
            )  # Only XY
            valid_clusters = clustering.labels_ >= 0

            # Update mask to only include valid clusters
            water_mask[water_indices[~valid_clusters]] = False

        n_water = water_mask.sum()
        if n_water > 0:
            classification[water_mask] = int(ASPRSClass.WATER)
            logger.info(f"  Water: {n_water:,} points classified")

        return classification

    def classify_bridges(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        classification: np.ndarray,
        ground_truth: Optional[Dict[str, any]] = None,
    ) -> np.ndarray:
        """
        Classify bridges using elevation, planarity, and proximity to roads/water.

        Bridge characteristics:
        - Elevated above ground
        - Planar (bridge deck)
        - Near roads (alignment)
        - Often crosses water or other roads
        - Linear structure

        Args:
            points: Point cloud [N, 3]
            features: Feature dictionary
            classification: Current classification [N]
            ground_truth: Optional ground truth with roads/water

        Returns:
            Updated classification [N]
        """
        cfg = self.bridge_config

        # Extract required features
        height = features.get("height_above_ground", points[:, 2])
        planarity = features.get("planarity", np.zeros(len(points)))
        verticality = features.get("verticality", np.zeros(len(points)))
        linearity = features.get("linearity", np.zeros(len(points)))

        # Build bridge candidate mask
        bridge_mask = (
            (height >= cfg.height_min)
            & (height <= cfg.height_max)
            & (planarity >= cfg.planarity_min)
            & (verticality <= cfg.verticality_max)
            & (linearity >= cfg.linearity_min)
        )

        # Check proximity to roads if ground truth available
        if cfg.use_road_alignment and ground_truth is not None:
            roads = ground_truth.get("roads", None)
            if roads is not None:
                logger.info("Refining bridge detection with road proximity")
                # Check if elevated points are near/aligned with roads
                bridge_mask = self._check_spatial_containment(
                    points,
                    bridge_mask,
                    roads,
                    buffer_m=10.0,  # 10m buffer for road alignment
                )

        # Check proximity to water if available
        if ground_truth is not None:
            water = ground_truth.get("water", None)
            if water is not None:
                logger.info("Checking bridge proximity to water")
                # Bridges often cross water bodies
                # Expand mask to points near water (not just over it)
                bridge_mask = self._check_spatial_containment(
                    points,
                    bridge_mask,
                    water,
                    buffer_m=25.0,  # 25m buffer for water proximity
                )

        # Cluster to identify bridge-like structures
        if bridge_mask.sum() > 0:
            bridge_indices = np.where(bridge_mask)[0]
            bridge_points = points[bridge_indices]

            # DBSCAN to find connected components
            clustering = DBSCAN(eps=3.0, min_samples=50).fit(bridge_points)
            valid_clusters = clustering.labels_ >= 0

            # Filter by cluster size and geometry
            for cluster_id in np.unique(clustering.labels_[valid_clusters]):
                cluster_mask = clustering.labels_ == cluster_id
                cluster_pts = bridge_points[cluster_mask]

                # Check cluster dimensions (length vs width)
                xy_range = cluster_pts[:, :2].max(axis=0) - cluster_pts[:, :2].min(
                    axis=0
                )
                length = xy_range.max()
                width = xy_range.min()

                # Valid bridge: length > min_length, width < max_width
                if length >= cfg.min_length and width <= cfg.max_width:
                    global_indices = bridge_indices[cluster_mask]
                    classification[global_indices] = int(ASPRSClass.BRIDGE_DECK)

        n_bridge = (classification == int(ASPRSClass.BRIDGE_DECK)).sum()
        if n_bridge > 0:
            logger.info(f"  Bridges: {n_bridge:,} points classified")

        return classification

    def classify_railways(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        classification: np.ndarray,
        ground_truth: Optional[Dict[str, any]] = None,
    ) -> np.ndarray:
        """
        Classify railways using linearity, planarity, and parallel track detection.

        Railway characteristics:
        - Very linear (tracks)
        - Relatively flat
        - Low vegetation (NDVI)
        - Specific width (track gauge)
        - Often parallel tracks

        Args:
            points: Point cloud [N, 3]
            features: Feature dictionary
            classification: Current classification [N]
            ground_truth: Optional ground truth with railway lines

        Returns:
            Updated classification [N]
        """
        cfg = self.railway_config

        # Extract required features
        height = features.get("height_above_ground", points[:, 2])
        planarity = features.get("planarity", np.zeros(len(points)))
        linearity = features.get("linearity", np.zeros(len(points)))
        ndvi = features.get("ndvi", np.zeros(len(points)))

        # Build railway candidate mask
        railway_mask = (
            (height >= cfg.height_min)
            & (height <= cfg.height_max)
            & (planarity >= cfg.planarity_min)
            & (linearity >= cfg.linearity_min)
            & (ndvi <= cfg.ndvi_max)
        )

        # Use BD TOPO railway lines if available
        if cfg.use_bd_topo_railways and ground_truth is not None:
            railways = ground_truth.get("railways", None)
            if railways is not None:
                logger.info("Refining railway detection with BD TOPO data")
                # Assign points within buffer of railway lines
                railway_mask = self._check_spatial_containment(
                    points,
                    railway_mask,
                    railways,
                    buffer_m=5.0,  # 5m buffer for railway alignment
                )

        # Detect parallel tracks if enabled
        if cfg.parallel_track_detection and railway_mask.sum() > 0:
            railway_indices = np.where(railway_mask)[0]
            railway_points = points[railway_indices]

            # Cluster railway points
            clustering = DBSCAN(eps=1.5, min_samples=30).fit(
                railway_points[:, :2]
            )  # XY only
            valid_clusters = clustering.labels_ >= 0

            # Check for parallel tracks (clusters at standard gauge distance)
            cluster_ids = np.unique(clustering.labels_[valid_clusters])
            if len(cluster_ids) >= 2:
                # Compute centroids of clusters
                centroids = []
                for cid in cluster_ids:
                    cluster_pts = railway_points[clustering.labels_ == cid]
                    centroid = cluster_pts[:, :2].mean(axis=0)
                    centroids.append(centroid)

                centroids = np.array(centroids)

                # Check distances between centroids
                from scipy.spatial.distance import pdist, squareform

                distances = squareform(pdist(centroids))

                # Look for pairs at standard gauge distance
                for spacing in cfg.track_spacing:
                    parallel_pairs = np.abs(distances - spacing) < 0.3  # ±30cm
                    if parallel_pairs.sum() > 0:
                        logger.info(
                            f"  Detected parallel railway tracks (spacing ~{spacing}m)"
                        )
                        break

        n_railway = railway_mask.sum()
        if n_railway > 0:
            # Use ASPRS standard railway classification
            # RAIL (10) for standard ASPRS, or RAILWAY_TRACK (90) for extended
            classification[railway_mask] = int(ASPRSClass.RAIL)
            logger.info(f"  Railways: {n_railway:,} points classified as RAIL (10)")

        return classification

    def classify_overhead_structures(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        classification: np.ndarray,
    ) -> np.ndarray:
        """
        Classify overhead structures (power lines, cables, wires).

        Overhead structure characteristics:
        - High elevation
        - Very linear (cables)
        - Not planar (thin structures)
        - Low verticality (horizontal/diagonal)
        - Long spans

        Args:
            points: Point cloud [N, 3]
            features: Feature dictionary
            classification: Current classification [N]

        Returns:
            Updated classification [N]
        """
        cfg = self.overhead_config

        # Extract required features
        height = features.get("height_above_ground", points[:, 2])
        linearity = features.get("linearity", np.zeros(len(points)))
        planarity = features.get("planarity", np.zeros(len(points)))
        verticality = features.get("verticality", np.zeros(len(points)))
        ndvi = features.get("ndvi", np.zeros(len(points)))

        # Build overhead structure candidate mask
        overhead_mask = (
            (height >= cfg.height_min)
            & (height <= cfg.height_max)
            & (linearity >= cfg.linearity_min)
            & (planarity <= cfg.planarity_max)
            & (verticality <= cfg.verticality_max)
        )

        # Exclude vegetation if enabled
        if cfg.exclude_vegetation:
            overhead_mask &= ndvi < 0.30  # Not vegetation

        # Cluster to identify cable spans
        if overhead_mask.sum() > 0:
            overhead_indices = np.where(overhead_mask)[0]
            overhead_points = points[overhead_indices]

            # DBSCAN with larger eps for cable spans
            clustering = DBSCAN(eps=5.0, min_samples=20).fit(overhead_points)
            valid_clusters = clustering.labels_ >= 0

            # Filter by cluster length
            for cluster_id in np.unique(clustering.labels_[valid_clusters]):
                cluster_mask = clustering.labels_ == cluster_id
                cluster_pts = overhead_points[cluster_mask]

                # Check cluster length
                xy_range = cluster_pts[:, :2].max(axis=0) - cluster_pts[:, :2].min(
                    axis=0
                )
                length = np.linalg.norm(xy_range)

                if length >= cfg.min_length:
                    global_indices = overhead_indices[cluster_mask]
                    classification[global_indices] = int(
                        ASPRSClass.WIRE_CONDUCTOR_OVERHEAD
                    )

        n_overhead = (classification == int(ASPRSClass.WIRE_CONDUCTOR_OVERHEAD)).sum()
        if n_overhead > 0:
            logger.info(f"  Overhead structures: {n_overhead:,} points classified")

        return classification

    def classify_noise(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        classification: np.ndarray,
    ) -> np.ndarray:
        """
        Classify noise (isolated points, outliers).

        Noise characteristics:
        - Isolated (few neighbors)
        - Large height deviation from local neighborhood
        - Not part of recognized structures

        Args:
            points: Point cloud [N, 3]
            features: Feature dictionary
            classification: Current classification [N]

        Returns:
            Updated classification [N]
        """
        cfg = self.noise_config

        if not cfg.classify_as_noise:
            return classification

        # 🔥 GPU-accelerated KNN for isolated point detection
        k_neighbors = max(cfg.min_neighbors + 5, 10)  # Extra neighbors for robust check
        
        distances, neighbors_indices = knn(
            points,
            points,
            k=k_neighbors
        )
        
        # Count neighbors within isolation threshold
        neighbor_counts = np.array([
            np.sum(distances[i] <= cfg.isolation_threshold) for i in range(len(points))
        ])
        isolated_mask = neighbor_counts < cfg.min_neighbors

        # Check height deviation
        height = features.get("height_above_ground", points[:, 2])
        height_deviation = np.abs(height - np.median(height))
        height_outlier_mask = height_deviation > cfg.height_deviation_max

        # Combine criteria
        noise_mask = isolated_mask | height_outlier_mask

        # Only classify currently unclassified points as noise
        unclassified_mask = classification == int(ASPRSClass.UNCLASSIFIED)
        noise_mask &= unclassified_mask

        n_noise = noise_mask.sum()
        if n_noise > 0:
            classification[noise_mask] = int(ASPRSClass.LOW_POINT_NOISE)
            logger.info(f"  Noise: {n_noise:,} points classified")

        return classification


def create_asprs_rules_from_config(config: Dict) -> ASPRSClassRulesEngine:
    """
    Create ASPRSClassRulesEngine from Hydra configuration.

    Args:
        config: Hydra configuration dictionary

    Returns:
        Configured ASPRSClassRulesEngine instance
    """
    asprs_config = config.get("asprs_class_rules", {})

    if not asprs_config.get("enabled", False):
        logger.info("ASPRS class-specific rules disabled")
        return None

    # Water detection config
    water_cfg_dict = asprs_config.get("water_detection", {})
    water_config = WaterDetectionConfig(**water_cfg_dict)

    # Bridge detection config
    bridge_cfg_dict = asprs_config.get("bridge_detection", {})
    bridge_config = BridgeDetectionConfig(**bridge_cfg_dict)

    # Railway detection config
    railway_cfg_dict = asprs_config.get("railway_detection", {})
    railway_config = RailwayDetectionConfig(**railway_cfg_dict)

    # Overhead structure config
    overhead_cfg_dict = asprs_config.get("overhead_structure_detection", {})
    overhead_config = OverheadStructureDetectionConfig(**overhead_cfg_dict)

    # Noise classification config
    noise_cfg_dict = asprs_config.get("noise_classification", {})
    noise_config = NoiseClassificationConfig(**noise_cfg_dict)

    return ASPRSClassRulesEngine(
        water_config=water_config,
        bridge_config=bridge_config,
        railway_config=railway_config,
        overhead_config=overhead_config,
        noise_config=noise_config,
    )
