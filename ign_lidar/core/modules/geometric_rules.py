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
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon, MultiPolygon
    from shapely.strtree import STRtree
    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False
    logger.warning("Spatial libraries not available for geometric rules")


class GeometricRulesEngine:
    """
    Engine for applying geometric and spectral rules to improve classification.
    
    Features:
    - Road-vegetation disambiguation using height and NDVI
    - Building buffer zones for nearby unclassified points
    - Vertical separation analysis for overlapping geometries
    - NDVI-based refinement for vegetation/non-vegetation
    """
    
    # ASPRS Classification codes
    ASPRS_UNCLASSIFIED = 1
    ASPRS_GROUND = 2
    ASPRS_LOW_VEGETATION = 3
    ASPRS_MEDIUM_VEGETATION = 4
    ASPRS_HIGH_VEGETATION = 5
    ASPRS_BUILDING = 6
    ASPRS_WATER = 9
    ASPRS_RAIL = 10
    ASPRS_ROAD = 11
    ASPRS_BRIDGE = 17
    
    def __init__(
        self,
        ndvi_vegetation_threshold: float = 0.3,
        ndvi_road_threshold: float = 0.15,
        road_vegetation_height_threshold: float = 2.0,
        building_buffer_distance: float = 2.0,
        max_building_height_difference: float = 3.0,
        verticality_threshold: float = 0.7,
        verticality_search_radius: float = 1.0,
        min_vertical_neighbors: int = 5
    ):
        """
        Initialize geometric rules engine.
        
        Args:
            ndvi_vegetation_threshold: NDVI threshold for vegetation (>= this = vegetation)
            ndvi_road_threshold: NDVI threshold for roads (<= this = likely road/impervious)
            road_vegetation_height_threshold: Height above road to classify as vegetation (meters)
            building_buffer_distance: Buffer distance around buildings for unclassified points (meters)
            max_building_height_difference: Max height diff for points to be part of building (meters)
            verticality_threshold: Verticality score threshold for building classification (0-1)
            verticality_search_radius: Search radius for computing verticality (meters)
            min_vertical_neighbors: Minimum neighbors required for verticality computation
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
        
        logger.info("🔧 Geometric Rules Engine initialized")
        logger.info(f"   NDVI vegetation threshold: {ndvi_vegetation_threshold}")
        logger.info(f"   NDVI road threshold: {ndvi_road_threshold}")
        logger.info(f"   Road-vegetation height separation: {road_vegetation_height_threshold}m")
        logger.info(f"   Building buffer distance: {building_buffer_distance}m")
        logger.info(f"   Verticality threshold: {verticality_threshold}")
        logger.info(f"   Verticality search radius: {verticality_search_radius}m")
    
    def apply_all_rules(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        ground_truth_features: Dict[str, gpd.GeoDataFrame],
        ndvi: Optional[np.ndarray] = None,
        intensities: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Apply all geometric rules to improve classification.
        
        Args:
            points: XYZ coordinates [N, 3]
            labels: Current classification labels [N] (will be modified)
            ground_truth_features: Dict of feature_type -> GeoDataFrame
            ndvi: Optional NDVI values [N] for each point (-1 to 1)
            intensities: Optional intensity values [N]
            
        Returns:
            Tuple of:
            - Updated labels [N]
            - Statistics dict with counts per rule
        """
        stats = {}
        updated_labels = labels.copy()
        
        logger.info("🔧 Applying geometric rules for classification refinement...")
        
        # Rule 1: Road-vegetation disambiguation
        if 'roads' in ground_truth_features and ndvi is not None:
            n_fixed = self.fix_road_vegetation_overlap(
                points=points,
                labels=updated_labels,
                road_geometries=ground_truth_features['roads'],
                ndvi=ndvi
            )
            stats['road_vegetation_fixed'] = n_fixed
            if n_fixed > 0:
                logger.info(f"  ✓ Rule 1 (Road-Vegetation): Fixed {n_fixed:,} points")
        
        # Rule 2: Building buffer zone classification with verticality
        if 'buildings' in ground_truth_features:
            n_added = self.classify_building_buffer_zone(
                points=points,
                labels=updated_labels,
                building_geometries=ground_truth_features['buildings']
            )
            stats['building_buffer_added'] = n_added
            if n_added > 0:
                logger.info(f"  ✓ Rule 2 (Building Buffer): Added {n_added:,} points")
        
        # Rule 2b: Verticality-based building classification
        n_vertical = self.classify_by_verticality(
            points=points,
            labels=updated_labels,
            ndvi=ndvi
        )
        stats['verticality_buildings_added'] = n_vertical
        if n_vertical > 0:
            logger.info(f"  ✓ Rule 2b (Verticality): Added {n_vertical:,} building points")
        
        # Rule 3: NDVI-based general refinement
        if ndvi is not None:
            n_refined = self.apply_ndvi_refinement(
                points=points,
                labels=updated_labels,
                ndvi=ndvi
            )
            stats['ndvi_refined'] = n_refined
            if n_refined > 0:
                logger.info(f"  ✓ Rule 3 (NDVI Refinement): Refined {n_refined:,} points")
        
        # Calculate total changes
        n_changed = np.sum(labels != updated_labels)
        stats['total_changed'] = n_changed
        
        logger.info(f"📊 Geometric rules applied: {n_changed:,} points modified")
        
        return updated_labels, stats
    
    def fix_road_vegetation_overlap(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        road_geometries: gpd.GeoDataFrame,
        ndvi: np.ndarray
    ) -> int:
        """
        Fix misclassified vegetation points on roads using vertical separation and NDVI.
        
        Logic:
        - Points above roads (e.g., tree canopy) with high NDVI -> keep as vegetation
        - Points on road surface with low NDVI -> reclassify to road
        - Uses height difference and NDVI to determine if vegetation is ON road or ABOVE road
        
        Args:
            points: XYZ coordinates [N, 3]
            labels: Classification labels [N] (modified in-place)
            road_geometries: GeoDataFrame with road polygons
            ndvi: NDVI values [N] for each point
            
        Returns:
            Number of points reclassified
        """
        if len(road_geometries) == 0:
            return 0
        
        n_fixed = 0
        
        # Find points currently classified as vegetation
        vegetation_mask = np.isin(labels, [
            self.ASPRS_LOW_VEGETATION,
            self.ASPRS_MEDIUM_VEGETATION,
            self.ASPRS_HIGH_VEGETATION
        ])
        
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
                    labels[global_idx] = self.ASPRS_ROAD
                    n_fixed += 1
                    break
                
                # For medium NDVI (ambiguous), we need height information
                # Estimate ground height from nearby ground points
                # Find ground points within 5m radius
                ground_mask = (labels == self.ASPRS_GROUND)
                if np.any(ground_mask):
                    ground_points = points[ground_mask]
                    
                    # Find nearby ground points
                    distances = np.sqrt(
                        (ground_points[:, 0] - pt[0])**2 +
                        (ground_points[:, 1] - pt[1])**2
                    )
                    nearby_ground = distances < 5.0
                    
                    if np.any(nearby_ground):
                        # Estimate ground height (median of nearby ground points)
                        ground_height = np.median(ground_points[nearby_ground, 2])
                        height_above_ground = pt[2] - ground_height
                        
                        # If vegetation is low (<2m) and on road, likely misclassified
                        if height_above_ground < self.road_vegetation_height_threshold:
                            # Low vegetation on road -> reclassify to road
                            labels[global_idx] = self.ASPRS_ROAD
                            n_fixed += 1
                            break
                        # else: high vegetation (tree canopy) above road -> keep as vegetation
                
                break  # Stop after first matching road
        
        return n_fixed
    
    def classify_building_buffer_zone(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        building_geometries: gpd.GeoDataFrame
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
            
        Returns:
            Number of points classified as building
        """
        if len(building_geometries) == 0:
            return 0
        
        n_added = 0
        
        # Find unclassified points
        unclassified_mask = (labels == self.ASPRS_UNCLASSIFIED)
        
        if not np.any(unclassified_mask):
            return 0
        
        unclassified_indices = np.where(unclassified_mask)[0]
        unclassified_points = points[unclassified_mask]
        
        # Find existing building points (for height reference)
        building_mask = (labels == self.ASPRS_BUILDING)
        building_points = points[building_mask]
        
        if len(building_points) == 0:
            logger.debug("No existing building points for reference")
            return 0
        
        # Build KD-Tree for efficient nearest neighbor search
        building_tree = cKDTree(building_points[:, :2])  # XY only
        
        # Create buffered building geometries
        buffered_buildings = building_geometries.geometry.buffer(self.building_buffer_distance)
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
                # Check height similarity to nearby building points
                # Find nearest building points (within 5m)
                distances, indices = building_tree.query(pt[:2], k=10, distance_upper_bound=5.0)
                
                valid_neighbors = distances < float('inf')
                if not np.any(valid_neighbors):
                    continue
                
                neighbor_heights = building_points[indices[valid_neighbors], 2]
                median_building_height = np.median(neighbor_heights)
                height_diff = abs(pt[2] - median_building_height)
                
                # If height is similar to nearby building points, classify as building
                if height_diff < self.max_building_height_difference:
                    labels[global_idx] = self.ASPRS_BUILDING
                    n_added += 1
                    break
        
        return n_added
    
    def apply_ndvi_refinement(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        ndvi: np.ndarray
    ) -> int:
        """
        Apply NDVI-based refinement for all classification types.
        
        Logic:
        - High NDVI points classified as non-vegetation -> check if should be vegetation
        - Low NDVI points classified as vegetation -> check if should be non-vegetation
        - Unclassified points with clear NDVI signal -> add classification
        
        Args:
            points: XYZ coordinates [N, 3]
            labels: Classification labels [N] (modified in-place)
            ndvi: NDVI values [N] for each point
            
        Returns:
            Number of points refined
        """
        n_refined = 0
        
        # Rule 1: Non-vegetation with high NDVI -> vegetation
        non_veg_mask = ~np.isin(labels, [
            self.ASPRS_LOW_VEGETATION,
            self.ASPRS_MEDIUM_VEGETATION,
            self.ASPRS_HIGH_VEGETATION,
            self.ASPRS_UNCLASSIFIED,
            self.ASPRS_GROUND,
            self.ASPRS_WATER  # Keep water as water
        ])
        high_ndvi_non_veg = non_veg_mask & (ndvi >= self.ndvi_vegetation_threshold)
        
        if np.any(high_ndvi_non_veg):
            # Reclassify to medium vegetation (can be refined by height later)
            labels[high_ndvi_non_veg] = self.ASPRS_MEDIUM_VEGETATION
            n_refined += np.sum(high_ndvi_non_veg)
        
        # Rule 2: Vegetation with low NDVI -> might be misclassified
        # (Be conservative here - only very low NDVI)
        veg_mask = np.isin(labels, [
            self.ASPRS_LOW_VEGETATION,
            self.ASPRS_MEDIUM_VEGETATION,
            self.ASPRS_HIGH_VEGETATION
        ])
        very_low_ndvi_veg = veg_mask & (ndvi <= 0.0)  # Very conservative
        
        if np.any(very_low_ndvi_veg):
            # Reclassify to unclassified (let other rules handle it)
            labels[very_low_ndvi_veg] = self.ASPRS_UNCLASSIFIED
            n_refined += np.sum(very_low_ndvi_veg)
        
        # Rule 3: Unclassified with very high NDVI -> vegetation
        unclassified_mask = (labels == self.ASPRS_UNCLASSIFIED)
        very_high_ndvi_unclassified = unclassified_mask & (ndvi >= 0.5)  # Very high NDVI
        
        if np.any(very_high_ndvi_unclassified):
            # Classify as medium vegetation
            labels[very_high_ndvi_unclassified] = self.ASPRS_MEDIUM_VEGETATION
            n_refined += np.sum(very_high_ndvi_unclassified)
        
        return n_refined
    
    def classify_by_verticality(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        ndvi: Optional[np.ndarray] = None
    ) -> int:
        """
        Classify unclassified points as buildings based on verticality analysis.
        
        Verticality measures how vertically aligned points are in a local neighborhood,
        which is a strong indicator of building walls and structures.
        
        Logic:
        - Compute verticality score for unclassified points
        - High verticality + low NDVI = likely building
        - Check height consistency with nearby classified building points
        
        Args:
            points: XYZ coordinates [N, 3]
            labels: Classification labels [N] (modified in-place)
            ndvi: Optional NDVI values [N] to exclude vegetation
            
        Returns:
            Number of points classified as building
        """
        n_added = 0
        
        # Find unclassified points
        unclassified_mask = (labels == self.ASPRS_UNCLASSIFIED)
        
        if not np.any(unclassified_mask):
            return 0
        
        unclassified_indices = np.where(unclassified_mask)[0]
        unclassified_points = points[unclassified_mask]
        
        # Build KD-Tree for all points (for verticality computation)
        all_points_tree = cKDTree(points)
        
        # Compute verticality for unclassified points
        verticality_scores = self.compute_verticality(
            points=points,
            query_points=unclassified_points,
            kdtree=all_points_tree
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
        building_mask = (labels == self.ASPRS_BUILDING)
        
        if np.any(building_mask):
            building_points = points[building_mask]
            building_tree = cKDTree(building_points[:, :2])  # XY only
            
            # For each candidate, check proximity and height consistency with buildings
            for i, pt in enumerate(candidate_points):
                global_idx = candidate_indices[i]
                
                # Find nearby building points (within 10m)
                distances, indices = building_tree.query(
                    pt[:2],
                    k=10,
                    distance_upper_bound=10.0
                )
                
                valid_neighbors = distances < float('inf')
                
                if np.any(valid_neighbors):
                    # Check height consistency
                    neighbor_heights = building_points[indices[valid_neighbors], 2]
                    median_building_height = np.median(neighbor_heights)
                    height_diff = abs(pt[2] - median_building_height)
                    
                    # If height is consistent, classify as building
                    if height_diff < self.max_building_height_difference * 2:  # More lenient
                        labels[global_idx] = self.ASPRS_BUILDING
                        n_added += 1
                else:
                    # No nearby buildings, but high verticality suggests new building
                    # Be more conservative: require very high verticality
                    vert_score = verticality_scores[candidate_mask][i]
                    if vert_score >= 0.85:  # Very high verticality
                        labels[global_idx] = self.ASPRS_BUILDING
                        n_added += 1
        else:
            # No existing building points, classify based on verticality alone
            # Require very high verticality score
            very_high_vert = verticality_scores[candidate_mask] >= 0.85
            very_high_indices = candidate_indices[very_high_vert]
            
            labels[very_high_indices] = self.ASPRS_BUILDING
            n_added = len(very_high_indices)
        
        return n_added
    
    def compute_verticality(
        self,
        points: np.ndarray,
        query_points: np.ndarray,
        kdtree: Optional[cKDTree] = None
    ) -> np.ndarray:
        """
        Compute verticality score for query points.
        
        Verticality measures the ratio of vertical extent to horizontal extent
        in a local neighborhood. High verticality indicates vertical structures
        like building walls.
        
        Algorithm:
        1. For each query point, find neighbors within search radius
        2. Compute vertical extent (max Z - min Z)
        3. Compute horizontal extent (max of XY spread)
        4. Verticality = vertical_extent / (horizontal_extent + epsilon)
        5. Normalize to [0, 1] range
        
        Args:
            points: All points [N, 3] for neighborhood search
            query_points: Points to compute verticality for [M, 3]
            kdtree: Pre-computed KD-Tree (optional, will create if None)
            
        Returns:
            Verticality scores [M] in range [0, 1]
        """
        if kdtree is None:
            kdtree = cKDTree(points)
        
        verticality_scores = np.zeros(len(query_points))
        
        for i, query_pt in enumerate(query_points):
            # Find neighbors within search radius
            indices = kdtree.query_ball_point(
                query_pt,
                r=self.verticality_search_radius
            )
            
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
        self,
        points: np.ndarray,
        labels: np.ndarray,
        search_radius: float = 5.0
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
        ground_mask = (labels == self.ASPRS_GROUND)
        
        if not np.any(ground_mask):
            logger.warning("No ground points available for height calculation")
            return heights
        
        ground_points = points[ground_mask]
        
        # Build KD-Tree for ground points
        ground_tree = cKDTree(ground_points[:, :2])  # XY only
        
        # For each point, find nearby ground points and estimate ground height
        for i, pt in enumerate(points):
            # Query nearby ground points
            distances, indices = ground_tree.query(
                pt[:2],
                k=5,
                distance_upper_bound=search_radius
            )
            
            valid_neighbors = distances < float('inf')
            
            if np.any(valid_neighbors):
                neighbor_heights = ground_points[indices[valid_neighbors], 2]
                ground_height = np.median(neighbor_heights)
                heights[i] = pt[2] - ground_height
            else:
                # No nearby ground points, use point height directly
                heights[i] = pt[2]
        
        return heights
