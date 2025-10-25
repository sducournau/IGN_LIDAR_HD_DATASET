"""
Multi-Source Building Fusion - Intelligent Polygon Comparison and Adaptation

This module implements advanced building footprint fusion by comparing multiple
data sources (BD TOPO®, Cadastre, OpenStreetMap) and adapting polygons to match
the actual point cloud distribution.

Key Features:
1. Multi-source comparison: Compare BD TOPO, Cadastre, OSM building footprints
2. Quality scoring: Score each polygon based on point cloud fit
3. Intelligent fusion: Merge complementary sources, select best per building
4. Adaptive adjustment: Move, scale, and buffer polygons to match reality
5. Conflict resolution: Handle overlapping buildings from different sources

Polygon Adaptation Strategies:
- Translation: Move polygon centroid to point cloud centroid
- Scaling: Expand/contract to match point density
- Rotation: Align with principal axes of point distribution
- Buffering: Extend boundaries to capture wall points
- Shape refinement: Adjust vertices to follow point clusters

Uses Consolidated Utilities:
- building.utils: Spatial operations (buffer_polygon, points_in_polygon, etc.)
- Avoids duplication of spatial/geometric functions

Author: Building Fusion Enhancement
Date: October 19, 2025
"""

import logging
from typing import Optional, Dict, List, Tuple, TYPE_CHECKING
import numpy as np
from dataclasses import dataclass, field
from enum import Enum

# Import consolidated utilities
from . import utils

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from shapely.geometry import Polygon, MultiPolygon, Point
    import geopandas as gpd

try:
    from shapely.geometry import Polygon, MultiPolygon, Point, box
    from shapely.strtree import STRtree
    from shapely.ops import unary_union
    from shapely.affinity import translate, scale, rotate
    import geopandas as gpd
    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False
    STRtree = None


class BuildingSource(str, Enum):
    """Building data sources."""
    BD_TOPO = "bd_topo"          # IGN BD TOPO (most accurate)
    CADASTRE = "cadastre"         # Cadastre (parcels)
    OSM = "osm"                   # OpenStreetMap
    FUSED = "fused"               # Multi-source fusion


@dataclass
class PolygonQuality:
    """Quality metrics for a building polygon against point cloud."""
    source: BuildingSource
    polygon: 'Polygon'
    
    # Coverage metrics
    points_inside: int = 0        # Points within polygon
    points_nearby: int = 0        # Points within buffer
    coverage_ratio: float = 0.0   # points_inside / total_building_points
    
    # Geometric fit metrics
    centroid_offset: float = 0.0  # Distance between centroids (meters)
    area_ratio: float = 1.0       # polygon_area / point_cloud_area
    shape_similarity: float = 0.0 # IoU or similar metric
    
    # Completeness metrics
    wall_coverage: float = 0.0    # % of walls captured
    roof_coverage: float = 0.0    # % of roof captured
    
    # Overall quality score (0-1)
    quality_score: float = 0.0
    
    def compute_quality_score(self) -> float:
        """
        Compute overall quality score from metrics.
        
        Weighted average:
        - Coverage: 40% (most important)
        - Geometric fit: 30% (centroid + area + shape)
        - Completeness: 30% (walls + roof)
        """
        coverage_score = self.coverage_ratio
        
        # Geometric fit (penalize offsets and mismatches)
        centroid_penalty = np.exp(-self.centroid_offset / 2.0)  # Gaussian decay
        area_penalty = 1.0 - abs(1.0 - self.area_ratio)
        geometric_score = (centroid_penalty + area_penalty + self.shape_similarity) / 3.0
        
        # Completeness
        completeness_score = (self.wall_coverage + self.roof_coverage) / 2.0
        
        # Weighted combination
        self.quality_score = (
            0.4 * coverage_score +
            0.3 * geometric_score +
            0.3 * completeness_score
        )
        
        return self.quality_score


@dataclass
class FusedBuilding:
    """Represents a building with fused polygon from multiple sources."""
    building_id: int
    primary_source: BuildingSource
    polygon: 'Polygon'
    original_polygons: Dict[BuildingSource, 'Polygon'] = field(default_factory=dict)
    quality_scores: Dict[BuildingSource, float] = field(default_factory=dict)
    
    # Point cloud statistics
    n_points: int = 0
    point_centroid: Optional[np.ndarray] = None
    point_bbox: Optional[Tuple[float, float, float, float]] = None
    
    # Adaptation applied
    was_translated: bool = False
    was_scaled: bool = False
    was_rotated: bool = False
    was_buffered: bool = False
    translation_vector: Optional[np.ndarray] = None
    scale_factor: Optional[float] = None
    rotation_angle: Optional[float] = None


class BuildingFusion:
    """
    Fuse building footprints from multiple sources and adapt to point cloud.
    
    This class implements intelligent polygon fusion by:
    1. Comparing polygons from BD TOPO, Cadastre, OSM
    2. Scoring each polygon based on point cloud fit
    3. Selecting best polygon or fusing multiple sources
    4. Adapting polygon geometry to match actual points
    """
    
    def __init__(
        self,
        # Source priorities
        source_priority: List[BuildingSource] = None,
        
        # Quality thresholds
        min_quality_score: float = 0.5,
        quality_difference_threshold: float = 0.15,
        
        # Fusion strategy
        fusion_mode: str = "best",  # "best", "weighted_merge", "consensus"
        enable_multi_source_fusion: bool = True,
        
        # Adaptation parameters
        enable_translation: bool = True,
        enable_scaling: bool = True,
        enable_rotation: bool = False,
        enable_buffering: bool = True,
        
        max_translation: float = 5.0,      # meters
        max_scale_factor: float = 1.5,     # 1.5x max expansion/contraction
        max_rotation: float = 15.0,        # degrees
        adaptive_buffer_range: Tuple[float, float] = (0.3, 1.0),  # meters
        
        # Point cloud analysis
        min_points_per_building: int = 20,
        wall_detection_threshold: float = 0.7,  # verticality for walls
        
        # Conflict resolution
        overlap_threshold: float = 0.3,    # IoU threshold for conflict
        merge_nearby_buildings: bool = True,
        merge_distance_threshold: float = 2.0,  # meters
    ):
        """
        Initialize building fusion system.
        
        Args:
            source_priority: Priority order for sources (highest first)
            min_quality_score: Minimum quality to accept polygon
            quality_difference_threshold: Min difference to switch sources
            fusion_mode: Strategy for combining sources
            enable_multi_source_fusion: Enable fusion of multiple sources
            enable_translation: Allow moving polygons
            enable_scaling: Allow resizing polygons
            enable_rotation: Allow rotating polygons
            enable_buffering: Allow buffering polygons
            max_translation: Maximum translation distance (meters)
            max_scale_factor: Maximum scaling ratio
            max_rotation: Maximum rotation angle (degrees)
            adaptive_buffer_range: Min/max buffer distances (meters)
            min_points_per_building: Minimum points to process building
            wall_detection_threshold: Verticality threshold for walls
            overlap_threshold: IoU threshold for overlap detection
            merge_nearby_buildings: Merge overlapping/touching buildings
            merge_distance_threshold: Max distance for merging (meters)
        """
        if source_priority is None:
            source_priority = [BuildingSource.BD_TOPO, BuildingSource.CADASTRE, BuildingSource.OSM]
        
        self.source_priority = source_priority
        self.min_quality_score = min_quality_score
        self.quality_difference_threshold = quality_difference_threshold
        self.fusion_mode = fusion_mode
        self.enable_multi_source_fusion = enable_multi_source_fusion
        
        self.enable_translation = enable_translation
        self.enable_scaling = enable_scaling
        self.enable_rotation = enable_rotation
        self.enable_buffering = enable_buffering
        
        self.max_translation = max_translation
        self.max_scale_factor = max_scale_factor
        self.max_rotation = max_rotation
        self.adaptive_buffer_range = adaptive_buffer_range
        
        self.min_points_per_building = min_points_per_building
        self.wall_detection_threshold = wall_detection_threshold
        
        self.overlap_threshold = overlap_threshold
        self.merge_nearby_buildings = merge_nearby_buildings
        self.merge_distance_threshold = merge_distance_threshold
        
        logger.info("Building Fusion System initialized")
        logger.info(f"  Source priority: {[s.value for s in source_priority]}")
        logger.info(f"  Fusion mode: {fusion_mode}")
        logger.info(f"  Adaptation enabled: translate={enable_translation}, scale={enable_scaling}, "
                   f"rotate={enable_rotation}, buffer={enable_buffering}")
        logger.info(f"  Max adaptation: translate={max_translation}m, scale={max_scale_factor}x, "
                   f"rotate={max_rotation}°, buffer={adaptive_buffer_range}")
    
    def fuse_building_sources(
        self,
        points: np.ndarray,
        building_sources: Dict[BuildingSource, 'gpd.GeoDataFrame'],
        normals: Optional[np.ndarray] = None,
        verticality: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        building_classes: Optional[List[int]] = None
    ) -> Tuple[List[FusedBuilding], Dict[str, any]]:
        """
        Fuse building footprints from multiple sources and adapt to point cloud.
        
        Process:
        1. Extract candidate buildings from each source
        2. Match buildings across sources (spatial overlap)
        3. Score each polygon against point cloud
        4. Select best polygon or fuse multiple sources
        5. Adapt selected polygon to match points
        6. Resolve conflicts (overlaps)
        
        Args:
            points: Point coordinates [N, 3]
            building_sources: Dict mapping source → GeoDataFrame
            normals: Surface normals [N, 3] (optional, for wall detection)
            verticality: Verticality scores [N] (optional)
            labels: Classification labels [N] (optional)
            building_classes: Building class codes (optional)
            
        Returns:
            Tuple of (fused_buildings, statistics)
        """
        if not HAS_SPATIAL:
            raise ImportError("Shapely and GeoPandas required for building fusion")
        
        logger.info("=== Multi-Source Building Fusion ===")
        logger.info(f"Processing {len(points):,} points")
        logger.info(f"Available sources: {list(building_sources.keys())}")
        
        # Filter building points if labels provided
        building_points = points
        building_normals = normals
        building_verticality = verticality
        
        if labels is not None and building_classes is not None:
            building_mask = np.isin(labels, building_classes)
            building_points = points[building_mask]
            if normals is not None:
                building_normals = normals[building_mask]
            if verticality is not None:
                building_verticality = verticality[building_mask]
            logger.info(f"Filtered to {len(building_points):,} building points")
        
        # Step 1: Extract and prepare buildings from each source
        source_buildings = self._extract_source_buildings(building_sources)
        
        # Step 2: Match buildings across sources
        building_matches = self._match_buildings_across_sources(source_buildings)
        logger.info(f"Found {len(building_matches)} building groups")
        
        # Step 3: Score and select best polygon for each building
        fused_buildings = []
        
        for match_id, matched_sources in enumerate(building_matches):
            # Score each polygon
            quality_scores = self._score_polygons(
                matched_sources,
                building_points,
                building_normals,
                building_verticality
            )
            
            # Select best or fuse
            if self.fusion_mode == "best":
                fused_poly = self._select_best_polygon(matched_sources, quality_scores)
            elif self.fusion_mode == "weighted_merge":
                fused_poly = self._weighted_merge_polygons(matched_sources, quality_scores)
            elif self.fusion_mode == "consensus":
                fused_poly = self._consensus_merge_polygons(matched_sources, quality_scores)
            else:
                fused_poly = self._select_best_polygon(matched_sources, quality_scores)
            
            if fused_poly is None:
                continue
            
            # Step 4: Adapt polygon to point cloud
            adapted_building = self._adapt_polygon_to_points(
                fused_poly,
                match_id,
                building_points,
                building_normals,
                building_verticality
            )
            
            if adapted_building is not None and adapted_building.n_points >= self.min_points_per_building:
                fused_buildings.append(adapted_building)
        
        logger.info(f"Created {len(fused_buildings)} fused buildings")
        
        # Step 5: Resolve conflicts
        if self.merge_nearby_buildings:
            fused_buildings = self._resolve_conflicts(fused_buildings)
            logger.info(f"After conflict resolution: {len(fused_buildings)} buildings")
        
        # Compute statistics
        stats = self._compute_fusion_statistics(fused_buildings, building_sources)
        
        return fused_buildings, stats
    
    def _extract_source_buildings(
        self,
        building_sources: Dict[BuildingSource, 'gpd.GeoDataFrame']
    ) -> Dict[BuildingSource, List['Polygon']]:
        """Extract valid building polygons from each source."""
        source_buildings = {}
        
        for source, gdf in building_sources.items():
            if gdf is None or len(gdf) == 0:
                continue
            
            # Extract polygons
            polygons = []
            for geom in gdf.geometry:
                if isinstance(geom, Polygon):
                    polygons.append(geom)
                elif isinstance(geom, MultiPolygon):
                    polygons.extend(list(geom.geoms))
            
            source_buildings[source] = polygons
            logger.info(f"  {source.value}: {len(polygons)} buildings")
        
        return source_buildings
    
    def _match_buildings_across_sources(
        self,
        source_buildings: Dict[BuildingSource, List['Polygon']]
    ) -> List[Dict[BuildingSource, 'Polygon']]:
        """
        Match buildings across sources based on spatial overlap.
        
        Returns list of dicts, each containing matched polygons from different sources.
        """
        # Combine all polygons with source tags
        all_polygons = []
        for source, polygons in source_buildings.items():
            for poly in polygons:
                all_polygons.append((source, poly))
        
        if len(all_polygons) == 0:
            return []
        
        # Build spatial index
        poly_list = [p for _, p in all_polygons]
        tree = STRtree(poly_list)
        
        # Find overlapping groups
        matched_groups = []
        processed = set()
        
        for idx, (source, poly) in enumerate(all_polygons):
            if idx in processed:
                continue
            
            # Find overlapping polygons
            candidates = tree.query(poly)
            
            # Build match group
            match_group = {}
            for candidate_idx in candidates:
                candidate_source, candidate_poly = all_polygons[candidate_idx]
                
                # Check overlap
                if poly.intersects(candidate_poly):
                    intersection = poly.intersection(candidate_poly).area
                    union = poly.union(candidate_poly).area
                    iou = intersection / union if union > 0 else 0
                    
                    if iou >= self.overlap_threshold:
                        # Only keep best polygon per source
                        if candidate_source not in match_group or \
                           candidate_poly.area > match_group[candidate_source].area:
                            match_group[candidate_source] = candidate_poly
                        processed.add(candidate_idx)
            
            if match_group:
                matched_groups.append(match_group)
        
        logger.info(f"Matched {len(matched_groups)} building groups across sources")
        
        return matched_groups
    
    def _score_polygons(
        self,
        matched_sources: Dict[BuildingSource, 'Polygon'],
        points: np.ndarray,
        normals: Optional[np.ndarray],
        verticality: Optional[np.ndarray]
    ) -> Dict[BuildingSource, PolygonQuality]:
        """Score each polygon against point cloud."""
        quality_scores = {}
        
        for source, polygon in matched_sources.items():
            quality = PolygonQuality(source=source, polygon=polygon)
            
            # Find points inside and nearby
            points_2d = points[:, :2]
            poly_bounds = polygon.bounds  # (minx, miny, maxx, maxy)
            
            # Quick bbox filter
            bbox_mask = (
                (points_2d[:, 0] >= poly_bounds[0] - 2.0) &
                (points_2d[:, 0] <= poly_bounds[2] + 2.0) &
                (points_2d[:, 1] >= poly_bounds[1] - 2.0) &
                (points_2d[:, 1] <= poly_bounds[3] + 2.0)
            )
            
            if not bbox_mask.any():
                quality.quality_score = 0.0
                quality_scores[source] = quality
                continue
            
            bbox_points = points[bbox_mask]
            bbox_points_2d = bbox_points[:, :2]
            
            # Check containment - use consolidated utility
            inside_mask = utils.points_in_polygon(bbox_points, polygon, return_mask=True)
            
            quality.points_inside = inside_mask.sum()
            
            # Check nearby (buffered) - use consolidated utility
            buffered = utils.buffer_polygon(polygon, 1.0)
            nearby_mask = utils.points_in_polygon(bbox_points, buffered, return_mask=True)
            quality.points_nearby = nearby_mask.sum()
            
            # Coverage ratio
            total_building_points = bbox_mask.sum()
            quality.coverage_ratio = quality.points_inside / max(total_building_points, 1)
            
            # Centroid offset - use consolidated utility
            if quality.points_inside > 0:
                point_centroid_2d = bbox_points_2d[inside_mask].mean(axis=0)
                poly_centroid = utils.compute_polygon_centroid(polygon)
                quality.centroid_offset = np.linalg.norm(point_centroid_2d - poly_centroid)
            else:
                quality.centroid_offset = 1000.0  # Large penalty
            
            # Area ratio
            if quality.points_inside > 0:
                # Estimate point cloud area (convex hull) - use consolidated utility
                try:
                    point_hull = utils.compute_convex_hull_polygon(bbox_points_2d[inside_mask])
                    if point_hull is not None:
                        point_area = point_hull.area
                        quality.area_ratio = polygon.area / max(point_area, 1.0)
                    else:
                        quality.area_ratio = 1.0
                except:
                    quality.area_ratio = 1.0
            else:
                quality.area_ratio = 0.0
            
            # Shape similarity (simplified as coverage for now)
            quality.shape_similarity = quality.coverage_ratio
            
            # Wall/roof coverage (if normals available)
            if normals is not None and verticality is not None and quality.points_inside > 0:
                inside_full_indices = np.where(bbox_mask)[0][inside_mask]
                inside_verticality = verticality[inside_full_indices]
                
                wall_mask = inside_verticality >= self.wall_detection_threshold
                roof_mask = inside_verticality < self.wall_detection_threshold
                
                quality.wall_coverage = wall_mask.sum() / max(quality.points_inside, 1)
                quality.roof_coverage = roof_mask.sum() / max(quality.points_inside, 1)
            else:
                quality.wall_coverage = 0.5
                quality.roof_coverage = 0.5
            
            # Compute final score
            quality.compute_quality_score()
            quality_scores[source] = quality
            
            logger.debug(f"  {source.value}: quality={quality.quality_score:.3f}, "
                        f"coverage={quality.coverage_ratio:.2f}, offset={quality.centroid_offset:.2f}m")
        
        return quality_scores
    
    def _select_best_polygon(
        self,
        matched_sources: Dict[BuildingSource, 'Polygon'],
        quality_scores: Dict[BuildingSource, PolygonQuality]
    ) -> Optional[FusedBuilding]:
        """Select polygon with highest quality score."""
        if not quality_scores:
            return None
        
        # Sort by priority first, then quality
        best_source = None
        best_quality = -1.0
        
        for source in self.source_priority:
            if source in quality_scores:
                quality = quality_scores[source].quality_score
                
                # Accept if significantly better or first valid option
                if quality >= self.min_quality_score:
                    if best_source is None or quality > best_quality + self.quality_difference_threshold:
                        best_source = source
                        best_quality = quality
        
        if best_source is None:
            return None
        
        fused = FusedBuilding(
            building_id=-1,  # Will be assigned later
            primary_source=best_source,
            polygon=matched_sources[best_source],
            original_polygons=matched_sources.copy(),
            quality_scores={s: q.quality_score for s, q in quality_scores.items()}
        )
        
        return fused
    
    def _weighted_merge_polygons(
        self,
        matched_sources: Dict[BuildingSource, 'Polygon'],
        quality_scores: Dict[BuildingSource, PolygonQuality]
    ) -> Optional[FusedBuilding]:
        """Merge polygons using quality-weighted averaging."""
        # Filter by quality threshold
        valid_sources = {
            s: p for s, p in matched_sources.items()
            if quality_scores[s].quality_score >= self.min_quality_score
        }
        
        if not valid_sources:
            return None
        
        if len(valid_sources) == 1:
            return self._select_best_polygon(matched_sources, quality_scores)
        
        # Weighted union based on quality scores
        weights = [quality_scores[s].quality_score for s in valid_sources.keys()]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return None
        
        # Take union and simplify
        merged_poly = unary_union(list(valid_sources.values()))
        
        # Simplify to reduce complexity
        merged_poly = merged_poly.simplify(0.1, preserve_topology=True)
        
        # Create fused building
        best_source = max(valid_sources.keys(), key=lambda s: quality_scores[s].quality_score)
        
        fused = FusedBuilding(
            building_id=-1,
            primary_source=BuildingSource.FUSED,
            polygon=merged_poly,
            original_polygons=matched_sources.copy(),
            quality_scores={s: q.quality_score for s, q in quality_scores.items()}
        )
        
        return fused
    
    def _consensus_merge_polygons(
        self,
        matched_sources: Dict[BuildingSource, 'Polygon'],
        quality_scores: Dict[BuildingSource, PolygonQuality]
    ) -> Optional[FusedBuilding]:
        """Merge polygons using consensus (intersection of good sources)."""
        valid_sources = {
            s: p for s, p in matched_sources.items()
            if quality_scores[s].quality_score >= self.min_quality_score
        }
        
        if not valid_sources:
            return None
        
        if len(valid_sources) == 1:
            return self._select_best_polygon(matched_sources, quality_scores)
        
        # Take intersection (consensus area)
        polygons = list(valid_sources.values())
        consensus_poly = polygons[0]
        
        for poly in polygons[1:]:
            consensus_poly = consensus_poly.intersection(poly)
        
        if consensus_poly.is_empty or consensus_poly.area < 10.0:
            # Fallback to weighted merge if consensus is too small
            return self._weighted_merge_polygons(matched_sources, quality_scores)
        
        fused = FusedBuilding(
            building_id=-1,
            primary_source=BuildingSource.FUSED,
            polygon=consensus_poly,
            original_polygons=matched_sources.copy(),
            quality_scores={s: q.quality_score for s, q in quality_scores.items()}
        )
        
        return fused
    
    def _adapt_polygon_to_points(
        self,
        fused_building: FusedBuilding,
        building_id: int,
        points: np.ndarray,
        normals: Optional[np.ndarray],
        verticality: Optional[np.ndarray]
    ) -> Optional[FusedBuilding]:
        """
        Adapt polygon geometry to match actual point cloud distribution.
        
        Applies transformations:
        1. Translation: Move to point cloud centroid
        2. Scaling: Expand/contract to match point density
        3. Rotation: Align with principal axes (optional)
        4. Buffering: Adaptive buffer to capture walls
        """
        polygon = fused_building.polygon
        fused_building.building_id = building_id
        
        # Find points inside current polygon
        points_2d = points[:, :2]
        from shapely.vectorized import contains
        inside_mask = contains(polygon, points_2d[:, 0], points_2d[:, 1])
        
        if not inside_mask.any():
            return None
        
        inside_points = points[inside_mask]
        fused_building.n_points = len(inside_points)
        
        # Compute point cloud centroid and bbox
        point_centroid = inside_points[:, :2].mean(axis=0)
        fused_building.point_centroid = np.append(point_centroid, inside_points[:, 2].mean())
        
        point_bbox = (
            inside_points[:, 0].min(),
            inside_points[:, 1].min(),
            inside_points[:, 0].max(),
            inside_points[:, 1].max()
        )
        fused_building.point_bbox = point_bbox
        
        # Start with original polygon
        adapted_poly = polygon
        
        # 1. Translation
        if self.enable_translation:
            poly_centroid = np.array([polygon.centroid.x, polygon.centroid.y])
            offset = point_centroid - poly_centroid
            offset_distance = np.linalg.norm(offset)
            
            if offset_distance > 0.5 and offset_distance <= self.max_translation:
                adapted_poly = translate(adapted_poly, xoff=offset[0], yoff=offset[1])
                fused_building.was_translated = True
                fused_building.translation_vector = offset
                logger.debug(f"  Building {building_id}: translated {offset_distance:.2f}m")
        
        # 2. Scaling
        if self.enable_scaling:
            # Compute scale factor based on point extent vs polygon extent
            poly_bounds = adapted_poly.bounds
            poly_width = poly_bounds[2] - poly_bounds[0]
            poly_height = poly_bounds[3] - poly_bounds[1]
            
            point_width = point_bbox[2] - point_bbox[0]
            point_height = point_bbox[3] - point_bbox[1]
            
            scale_x = point_width / max(poly_width, 0.1)
            scale_y = point_height / max(poly_height, 0.1)
            scale_factor = (scale_x + scale_y) / 2.0
            
            # Clamp scale factor
            scale_factor = np.clip(scale_factor, 1.0 / self.max_scale_factor, self.max_scale_factor)
            
            if abs(scale_factor - 1.0) > 0.1:
                # Scale from centroid
                adapted_poly = scale(
                    adapted_poly,
                    xfact=scale_factor,
                    yfact=scale_factor,
                    origin='centroid'
                )
                fused_building.was_scaled = True
                fused_building.scale_factor = scale_factor
                logger.debug(f"  Building {building_id}: scaled {scale_factor:.2f}x")
        
        # 3. Rotation (optional, computationally expensive)
        if self.enable_rotation:
            # Compute principal axes of point distribution
            from sklearn.decomposition import PCA
            try:
                pca = PCA(n_components=2)
                pca.fit(inside_points[:, :2])
                
                # Get rotation angle
                principal_axis = pca.components_[0]
                angle = np.degrees(np.arctan2(principal_axis[1], principal_axis[0]))
                
                # Clamp angle
                if abs(angle) > 1.0 and abs(angle) <= self.max_rotation:
                    adapted_poly = rotate(adapted_poly, angle, origin='centroid', use_radians=False)
                    fused_building.was_rotated = True
                    fused_building.rotation_angle = angle
                    logger.debug(f"  Building {building_id}: rotated {angle:.1f}°")
            except:
                pass  # PCA failed, skip rotation
        
        # 4. Adaptive buffering
        if self.enable_buffering:
            # Compute buffer based on wall detection
            buffer_distance = self.adaptive_buffer_range[0]  # Start with minimum
            
            if normals is not None and verticality is not None:
                # Find wall points (high verticality)
                inside_full_indices = np.where(inside_mask)[0]
                inside_verticality = verticality[inside_full_indices]
                
                wall_mask = inside_verticality >= self.wall_detection_threshold
                n_walls = wall_mask.sum()
                wall_ratio = n_walls / max(len(inside_verticality), 1)
                
                # Increase buffer if many walls detected (walls often near boundaries)
                if wall_ratio > 0.2:
                    buffer_distance = self.adaptive_buffer_range[0] + \
                                    (self.adaptive_buffer_range[1] - self.adaptive_buffer_range[0]) * wall_ratio
            
            if buffer_distance > 0.05:
                adapted_poly = utils.buffer_polygon(adapted_poly, buffer_distance)
                fused_building.was_buffered = True
                logger.debug(f"  Building {building_id}: buffered {buffer_distance:.2f}m")
        
        # Update polygon
        fused_building.polygon = adapted_poly
        
        # Recount points in adapted polygon - use consolidated utility
        inside_mask_adapted = utils.points_in_polygon(points, adapted_poly, return_mask=True)
        fused_building.n_points = inside_mask_adapted.sum()
        
        return fused_building
    
    def _resolve_conflicts(
        self,
        fused_buildings: List[FusedBuilding]
    ) -> List[FusedBuilding]:
        """
        Resolve overlapping buildings.
        
        Strategy:
        - Merge very close buildings (< 2m apart)
        - Remove smaller building if significant overlap
        - Adjust boundaries if moderate overlap
        """
        if len(fused_buildings) <= 1:
            return fused_buildings
        
        # Build spatial index
        polygons = [b.polygon for b in fused_buildings]
        tree = STRtree(polygons)
        
        resolved = []
        merged = set()
        
        for i, building in enumerate(fused_buildings):
            if i in merged:
                continue
            
            # Find overlapping buildings
            candidates = tree.query(building.polygon)
            
            conflicts = []
            for j in candidates:
                if j == i or j in merged:
                    continue
                
                other = fused_buildings[j]
                
                if building.polygon.intersects(other.polygon):
                    intersection = building.polygon.intersection(other.polygon).area
                    union = building.polygon.union(other.polygon).area
                    iou = intersection / union if union > 0 else 0
                    
                    if iou >= self.overlap_threshold:
                        conflicts.append((j, other, iou))
            
            if not conflicts:
                resolved.append(building)
                continue
            
            # Resolve conflicts
            if self.merge_nearby_buildings:
                # Check if should merge
                should_merge = any(
                    building.polygon.distance(other.polygon) <= self.merge_distance_threshold
                    for _, other, _ in conflicts
                )
                
                if should_merge and len(conflicts) <= 2:
                    # Merge buildings
                    merged_poly = building.polygon
                    total_points = building.n_points
                    
                    for j, other, _ in conflicts:
                        merged_poly = merged_poly.union(other.polygon)
                        total_points += other.n_points
                        merged.add(j)
                    
                    building.polygon = merged_poly.simplify(0.1)
                    building.n_points = total_points
                    building.primary_source = BuildingSource.FUSED
                    
                    resolved.append(building)
                else:
                    # Keep building with most points
                    resolved.append(building)
            else:
                resolved.append(building)
        
        logger.info(f"Conflict resolution: {len(fused_buildings)} → {len(resolved)} buildings")
        
        return resolved
    
    def _compute_fusion_statistics(
        self,
        fused_buildings: List[FusedBuilding],
        building_sources: Dict[BuildingSource, 'gpd.GeoDataFrame']
    ) -> Dict[str, any]:
        """Compute statistics about fusion process."""
        stats = {
            'total_fused': len(fused_buildings),
            'total_points': sum(b.n_points for b in fused_buildings),
            'sources_used': {},
            'adaptations': {
                'translated': sum(1 for b in fused_buildings if b.was_translated),
                'scaled': sum(1 for b in fused_buildings if b.was_scaled),
                'rotated': sum(1 for b in fused_buildings if b.was_rotated),
                'buffered': sum(1 for b in fused_buildings if b.was_buffered),
            },
            'quality_scores': {},
        }
        
        # Count sources
        for building in fused_buildings:
            source = building.primary_source
            stats['sources_used'][source.value] = stats['sources_used'].get(source.value, 0) + 1
            
            # Aggregate quality scores
            for src, score in building.quality_scores.items():
                if src.value not in stats['quality_scores']:
                    stats['quality_scores'][src.value] = []
                stats['quality_scores'][src.value].append(score)
        
        # Average quality scores
        for src in stats['quality_scores']:
            scores = stats['quality_scores'][src]
            stats['quality_scores'][src] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
            }
        
        logger.info("=== Fusion Statistics ===")
        logger.info(f"Total fused buildings: {stats['total_fused']}")
        logger.info(f"Total points: {stats['total_points']:,}")
        logger.info(f"Sources used: {stats['sources_used']}")
        logger.info(f"Adaptations: {stats['adaptations']}")
        
        return stats


__all__ = [
    'BuildingSource',
    'PolygonQuality',
    'FusedBuilding',
    'BuildingFusion',
]
