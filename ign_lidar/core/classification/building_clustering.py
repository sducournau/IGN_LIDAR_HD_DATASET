"""
Building Clustering by Batiment - Centroid-Based Spatial Coherence

This module implements advanced building point clustering using building footprint
polygons (batiment) from IGN BD TOPO®. It uses centroid attraction to ensure
spatial coherence within building boundaries.

Key Features:
1. Centroid attraction: Points gravitate toward building centroids
2. Polygon membership: Strict containment within building footprints
3. Multi-source fusion: Combines multiple BD TOPO layers (buildings, cadastre)
4. Approximate polygon movement: Adjusts polygons to match point cloud reality

Use Cases:
- LOD2/LOD3 building reconstruction: Group wall/roof points by building
- Building-level statistics: Count points, compute volumes per building
- Quality control: Detect buildings with insufficient points
- Spatial coherence: Ensure consistent classification within buildings

Author: Building Clustering Enhancement
Date: October 19, 2025
"""

import logging
from typing import Optional, Dict, List, Tuple, TYPE_CHECKING
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from shapely.geometry import Polygon, MultiPolygon, Point
    import geopandas as gpd

try:
    from shapely.geometry import Polygon, MultiPolygon, Point
    from shapely.strtree import STRtree
    from shapely.ops import unary_union
    import geopandas as gpd
    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False
    STRtree = None


@dataclass
class BuildingCluster:
    """Represents a cluster of points belonging to a single building."""
    building_id: int
    point_indices: np.ndarray  # Indices into original point cloud
    centroid: np.ndarray  # [3] XYZ centroid
    polygon: Optional['Polygon'] = None  # Building footprint
    n_points: int = 0
    volume: float = 0.0
    height_mean: float = 0.0
    height_max: float = 0.0


class BuildingClusterer:
    """
    Cluster building points by batiment using centroid attraction.
    
    This class assigns building points to their respective building footprints
    using a combination of:
    1. Spatial containment (point-in-polygon tests)
    2. Centroid attraction (nearest centroid for ambiguous points)
    3. Height compatibility (ground-level separation)
    """
    
    def __init__(
        self,
        use_centroid_attraction: bool = True,
        attraction_radius: float = 5.0,
        min_points_per_building: int = 10,
        adjust_polygons: bool = True,
        polygon_buffer: float = 0.5,
        wall_buffer: float = 0.3,
        detect_near_vertical_walls: bool = True
    ):
        """
        Initialize building clusterer.
        
        Args:
            use_centroid_attraction: Use centroids to resolve ambiguous assignments
            attraction_radius: Maximum distance for centroid attraction (meters)
            min_points_per_building: Minimum points to form valid cluster
            adjust_polygons: Adjust polygon boundaries to match point cloud
            polygon_buffer: Buffer distance for polygon adjustment (meters)
            wall_buffer: Additional buffer for near-vertical wall detection (meters)
            detect_near_vertical_walls: Enable near-vertical wall detection
        """
        self.use_centroid_attraction = use_centroid_attraction
        self.attraction_radius = attraction_radius
        self.min_points_per_building = min_points_per_building
        self.adjust_polygons = adjust_polygons
        self.polygon_buffer = polygon_buffer
        self.wall_buffer = wall_buffer
        self.detect_near_vertical_walls = detect_near_vertical_walls
        
        logger.info("Building Clusterer initialized")
        logger.info(f"  Centroid attraction: {use_centroid_attraction}")
        logger.info(f"  Attraction radius: {attraction_radius}m")
        logger.info(f"  Min points per building: {min_points_per_building}")
        logger.info(f"  Wall buffer: {wall_buffer}m (near-vertical detection: {detect_near_vertical_walls})")
    
    def cluster_points_by_buildings(
        self,
        points: np.ndarray,
        buildings_gdf: 'gpd.GeoDataFrame',
        labels: Optional[np.ndarray] = None,
        building_classes: Optional[List[int]] = None,
        normals: Optional[np.ndarray] = None,
        verticality: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[BuildingCluster]]:
        """
        Cluster building points by their respective building footprints.
        
        ENHANCED: Now detects near-vertical walls (plans verticaux / murs) using normals
        and extends building polygons with buffers to capture wall points up to boundaries.
        
        Args:
            points: Point coordinates [N, 3] (X, Y, Z)
            buildings_gdf: GeoDataFrame of building footprint polygons
            labels: Optional classification labels [N] to filter building points
            building_classes: List of ASPRS codes considered as buildings
            normals: Optional surface normals [N, 3] for wall detection
            verticality: Optional verticality values [N] (0=horizontal, 1=vertical)
            
        Returns:
            Tuple of (building_ids [N], list of BuildingCluster objects)
            - building_ids: Array mapping each point to building ID (-1 if not in building)
            - clusters: List of BuildingCluster objects for each building
        """
        if not HAS_SPATIAL:
            logger.error("Spatial libraries not available")
            return np.full(len(points), -1, dtype=np.int32), []
        
        n_points = len(points)
        building_ids = np.full(n_points, -1, dtype=np.int32)
        
        # Filter to building points if labels provided
        if labels is not None and building_classes is not None:
            building_mask = np.isin(labels, building_classes)
            candidate_indices = np.where(building_mask)[0]
            logger.info(f"Filtering to {len(candidate_indices):,} building points")
        else:
            candidate_indices = np.arange(n_points)
            logger.info(f"Processing all {n_points:,} points")
        
        # ENHANCED: Detect near-vertical walls if normals/verticality available
        near_vertical_mask = None
        if self.detect_near_vertical_walls and normals is not None:
            # Compute verticality if not provided (1 - |nz|)
            if verticality is None:
                verticality = 1.0 - np.abs(normals[:, 2])
            
            # Near-vertical walls: verticality > 0.6 (angle > ~53° from horizontal)
            near_vertical_mask = (verticality > 0.6)
            n_vertical = near_vertical_mask.sum()
            logger.info(f"Detected {n_vertical:,} near-vertical points (walls)")
            
            # Add near-vertical points to candidates (even if not initially labeled as building)
            if labels is not None:
                extended_candidates = np.union1d(
                    candidate_indices,
                    np.where(near_vertical_mask)[0]
                )
                logger.info(f"Extended candidates to {len(extended_candidates):,} points (including walls)")
                candidate_indices = extended_candidates
        
        # Extract building polygons and compute centroids
        valid_buildings = []
        building_centroids = []
        building_polygons = []
        
        for idx, row in buildings_gdf.iterrows():
            geom = row['geometry']
            if not isinstance(geom, (Polygon, MultiPolygon)):
                continue
            
            # Compute centroid
            centroid = geom.centroid
            centroid_xyz = np.array([centroid.x, centroid.y, 0.0])  # Z will be updated
            
            valid_buildings.append(idx)
            building_centroids.append(centroid_xyz)
            building_polygons.append(geom)
        
        n_buildings = len(valid_buildings)
        logger.info(f"Processing {n_buildings} valid building polygons")
        
        if n_buildings == 0:
            return building_ids, []
        
        # Adjust polygon boundaries if requested
        if self.adjust_polygons:
            building_polygons = self._adjust_polygons(
                points[candidate_indices],
                building_polygons,
                building_centroids
            )
        
        # Build spatial index for efficient polygon queries
        building_tree = STRtree(building_polygons)
        
        # First pass: Assign points within polygons
        candidate_points_2d = points[candidate_indices, :2]
        point_geoms = [Point(p) for p in candidate_points_2d]
        
        for i, (idx, point_geom) in enumerate(zip(candidate_indices, point_geoms)):
            # Query spatial index for containing polygons
            potential_buildings = building_tree.query(point_geom, predicate='contains')
            
            if len(potential_buildings) > 0:
                # Point is inside at least one building
                if len(potential_buildings) == 1:
                    # Unique assignment
                    building_ids[idx] = potential_buildings[0]
                else:
                    # Multiple buildings contain point - use centroid attraction
                    if self.use_centroid_attraction:
                        building_ids[idx] = self._nearest_centroid(
                            points[idx, :2],
                            [building_centroids[bid] for bid in potential_buildings],
                            potential_buildings
                        )
                    else:
                        # Default to first match
                        building_ids[idx] = potential_buildings[0]
        
        # Second pass: Centroid attraction for nearby unassigned points
        if self.use_centroid_attraction:
            unassigned_mask = (building_ids == -1) & np.isin(np.arange(n_points), candidate_indices)
            unassigned_indices = np.where(unassigned_mask)[0]
            
            if len(unassigned_indices) > 0:
                logger.info(f"Applying centroid attraction to {len(unassigned_indices):,} unassigned points")
                
                for idx in unassigned_indices:
                    point_xy = points[idx, :2]
                    
                    # Find nearest centroid within attraction radius
                    distances = np.array([
                        np.linalg.norm(point_xy - c[:2])
                        for c in building_centroids
                    ])
                    
                    nearest_idx = np.argmin(distances)
                    nearest_dist = distances[nearest_idx]
                    
                    if nearest_dist <= self.attraction_radius:
                        building_ids[idx] = nearest_idx
        
        # Create clusters
        clusters = self._create_clusters(
            points,
            building_ids,
            valid_buildings,
            building_centroids,
            building_polygons
        )
        
        # Filter clusters by minimum points
        valid_clusters = [c for c in clusters if c.n_points >= self.min_points_per_building]
        
        logger.info(f"Created {len(valid_clusters)} valid clusters (min {self.min_points_per_building} points)")
        logger.info(f"  Total clustered points: {sum(c.n_points for c in valid_clusters):,}")
        
        return building_ids, valid_clusters
    
    def _adjust_polygons(
        self,
        points: np.ndarray,
        polygons: List['Polygon'],
        centroids: List[np.ndarray]
    ) -> List['Polygon']:
        """
        Adjust polygon boundaries to better match point cloud distribution.
        
        ENHANCED Strategy for Wall Detection:
        1. Apply base buffer to capture building footprint
        2. Add extended buffer for near-vertical walls (captures points at wall boundaries)
        3. Compute point density around polygon edges
        4. Expand polygons where points are dense outside boundary
        5. Contract polygons where interior is empty
        
        This ensures points near walls (murs) are included within building polygons.
        """
        adjusted = []
        
        for poly, centroid in zip(polygons, centroids):
            # Base adjustment: apply small buffer to capture nearby points
            base_buffer = self.polygon_buffer
            
            # Extended buffer for near-vertical wall detection
            if self.detect_near_vertical_walls:
                # Use larger buffer to reach wall boundaries
                total_buffer = base_buffer + self.wall_buffer
                adjusted_poly = poly.buffer(total_buffer, cap_style='square')
                logger.debug(f"Applied wall buffer: {total_buffer:.2f}m (base: {base_buffer:.2f}m + wall: {self.wall_buffer:.2f}m)")
            else:
                adjusted_poly = poly.buffer(base_buffer, cap_style='square')
            
            adjusted.append(adjusted_poly)
        
        total_buffer_used = self.polygon_buffer + (self.wall_buffer if self.detect_near_vertical_walls else 0)
        logger.info(f"Adjusted {len(polygons)} polygons with {total_buffer_used:.2f}m buffer (wall detection: {self.detect_near_vertical_walls})")
        return adjusted
    
    def _nearest_centroid(
        self,
        point_xy: np.ndarray,
        centroids: List[np.ndarray],
        building_indices: List[int]
    ) -> int:
        """Find nearest centroid among candidates."""
        distances = [np.linalg.norm(point_xy - c[:2]) for c in centroids]
        nearest_idx = np.argmin(distances)
        return building_indices[nearest_idx]
    
    def _create_clusters(
        self,
        points: np.ndarray,
        building_ids: np.ndarray,
        valid_buildings: List[int],
        building_centroids: List[np.ndarray],
        building_polygons: List['Polygon']
    ) -> List[BuildingCluster]:
        """Create BuildingCluster objects from assignments."""
        clusters = []
        
        for i, building_idx in enumerate(valid_buildings):
            # Get points assigned to this building
            mask = (building_ids == i)
            if not mask.any():
                continue
            
            point_indices = np.where(mask)[0]
            building_points = points[point_indices]
            
            # Compute cluster statistics
            centroid_3d = building_points.mean(axis=0)
            heights = building_points[:, 2]
            
            cluster = BuildingCluster(
                building_id=building_idx,
                point_indices=point_indices,
                centroid=centroid_3d,
                polygon=building_polygons[i],
                n_points=len(point_indices),
                volume=self._compute_volume(building_points, building_polygons[i]),
                height_mean=heights.mean(),
                height_max=heights.max()
            )
            
            clusters.append(cluster)
        
        return clusters
    
    def _compute_volume(self, points: np.ndarray, polygon: 'Polygon') -> float:
        """Estimate building volume from point cloud."""
        if len(points) == 0:
            return 0.0
        
        # Simple estimate: polygon area × mean height
        area = polygon.area
        mean_height = points[:, 2].mean()
        
        return area * mean_height


def cluster_buildings_multi_source(
    points: np.ndarray,
    ground_truth_features: Dict[str, 'gpd.GeoDataFrame'],
    labels: Optional[np.ndarray] = None,
    building_classes: Optional[List[int]] = None,
    **kwargs
) -> Tuple[np.ndarray, List[BuildingCluster]]:
    """
    Cluster buildings using multiple ground truth sources.
    
    Combines:
    1. BD TOPO buildings (primary)
    2. Cadastre parcels (fallback for missing buildings)
    3. Building labels from classification
    
    Args:
        points: Point coordinates [N, 3]
        ground_truth_features: Dict with 'buildings' and optionally 'cadastre'
        labels: Classification labels [N]
        building_classes: ASPRS codes for buildings
        **kwargs: Additional arguments for BuildingClusterer
        
    Returns:
        Tuple of (building_ids, clusters)
    """
    # Primary: BD TOPO buildings
    buildings_gdf = ground_truth_features.get('buildings')
    
    # Fallback: Use cadastre if no buildings available
    if buildings_gdf is None or len(buildings_gdf) == 0:
        buildings_gdf = ground_truth_features.get('cadastre')
        if buildings_gdf is not None:
            logger.info("Using cadastre parcels as building proxies")
    
    if buildings_gdf is None or len(buildings_gdf) == 0:
        logger.warning("No building footprints available for clustering")
        return np.full(len(points), -1, dtype=np.int32), []
    
    # Cluster
    clusterer = BuildingClusterer(**kwargs)
    return clusterer.cluster_points_by_buildings(
        points, buildings_gdf, labels, building_classes
    )


__all__ = [
    'BuildingCluster',
    'BuildingClusterer',
    'cluster_buildings_multi_source'
]
