"""
3D Building Extrusion - Volumetric Bounding Box Classification

This module implements 3D volumetric bounding boxes for building classification
by extruding 2D ground truth polygons using height information from point clouds.

Key Features:
1. 3D Bounding Box Creation: Extrude 2D polygons vertically using height data
2. Multi-Level Analysis: Segment buildings by floor (ground, mid, top)
3. Volumetric Containment: Check if points are inside 3D building volume
4. Height-Aware Buffering: Apply different buffers at different heights
5. Wall Plane Detection: Fit 3D planes to walls and extend vertically

Philosophy:
- 2D polygons only capture footprint → miss points above/below
- 3D extrusion captures full building volume including overhangs, balconies
- Height-based segmentation detects setbacks and roof terraces
- Vertical plane extension captures wall points at all heights

Uses Consolidated Utilities:
- building.utils: Spatial operations, bbox computations, polygon operations
- Avoids code duplication across building classification modules

Author: Building Classification Enhancement - 3D Extrusion
Date: October 25, 2025
"""

import logging
from typing import Optional, Dict, List, Tuple, TYPE_CHECKING
import numpy as np
from dataclasses import dataclass

# Import consolidated utilities from building.utils
from . import utils

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from shapely.geometry import Polygon, MultiPolygon
    import geopandas as gpd

try:
    from shapely.geometry import Polygon, MultiPolygon, box
    from shapely.ops import unary_union
    from shapely.affinity import translate, scale
    import geopandas as gpd
    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False


@dataclass
class BoundingBox3D:
    """Represents a 3D bounding box for a building."""
    # 2D footprint
    polygon: 'Polygon'
    
    # Vertical extent
    z_min: float  # Ground level (meters)
    z_max: float  # Roof height (meters)
    
    # Building metadata
    building_id: int
    n_floors: int = 1
    floor_height: float = 3.0  # Typical floor height
    
    # Point statistics
    n_points: int = 0
    point_density: float = 0.0  # points per cubic meter
    
    # Wall detection statistics (NEW)
    wall_points: int = 0  # Number of vertical facade points
    wall_coverage: float = 0.0  # % of perimeter with detected walls
    adaptive_buffer_distance: float = 0.0  # Computed adaptive buffer (meters)
    has_missing_walls: bool = False  # True if walls under-represented
    
    # Gap/void detection statistics (NEW - v3.3.3)
    detected_gaps: Optional[List[Tuple[float, float, float]]] = None  # List of gap segments (start_angle, end_angle, width_m)
    gap_total_length: float = 0.0  # Total perimeter length with gaps (meters)
    gap_coverage_ratio: float = 0.0  # Ratio of perimeter with gaps (0-1)
    has_significant_gaps: bool = False  # True if gaps > 20% of perimeter
    
    # Floor-specific footprints (optional)
    floor_polygons: Optional[List['Polygon']] = None  # One per floor
    floor_heights: Optional[List[Tuple[float, float]]] = None  # Height ranges per floor


@dataclass
class FloorSegment:
    """Represents a single floor/level of a building."""
    floor_index: int  # 0 = ground, 1 = first floor, etc.
    z_min: float
    z_max: float
    footprint: 'Polygon'
    n_points: int = 0
    has_setback: bool = False  # True if smaller than floor below
    is_roof: bool = False


class Building3DExtruder:
    """
    Extrude 2D building polygons into 3D volumetric bounding boxes.
    
    This class creates 3D bounding boxes from 2D ground truth polygons
    by analyzing point cloud height distribution and extruding vertically.
    """
    
    def __init__(
        self,
        floor_height: float = 3.0,
        min_building_height: float = 2.0,
        max_building_height: float = 100.0,
        detect_setbacks: bool = True,
        detect_overhangs: bool = True,
        vertical_buffer: float = 0.5,  # Buffer above/below (meters)
        horizontal_buffer_ground: float = 0.8,  # Buffer at ground level
        horizontal_buffer_upper: float = 1.2,  # Buffer at upper levels (balconies)
        enable_floor_segmentation: bool = True,
        # NEW: Adaptive wall detection parameters
        enable_adaptive_buffer: bool = True,
        adaptive_buffer_min: float = 0.5,  # Minimum lateral buffer (meters)
        adaptive_buffer_max: float = 5.0,  # Maximum lateral buffer (meters)
        wall_verticality_threshold: float = 0.65,  # Minimum verticality for walls
        wall_detection_enabled: bool = True,
        missing_wall_threshold: float = 0.3,  # If <30% wall coverage, flag as missing
    ):
        """
        Initialize 3D building extruder with adaptive wall detection.
        
        Args:
            floor_height: Typical floor height in meters (default: 3.0m)
            min_building_height: Minimum building height to consider
            max_building_height: Maximum building height (filter outliers)
            detect_setbacks: Enable setback detection (upper floors smaller)
            detect_overhangs: Enable overhang detection (balconies, eaves)
            vertical_buffer: Additional buffer above/below building
            horizontal_buffer_ground: Horizontal buffer at ground level
            horizontal_buffer_upper: Horizontal buffer at upper levels
            enable_floor_segmentation: Analyze each floor separately
            enable_adaptive_buffer: Use adaptive lateral buffer based on wall detection
            adaptive_buffer_min: Minimum lateral buffer distance
            adaptive_buffer_max: Maximum lateral buffer distance (up to 5m for facades)
            wall_verticality_threshold: Minimum verticality to consider as wall
            wall_detection_enabled: Enable wall point detection and statistics
            missing_wall_threshold: Wall coverage threshold to flag missing walls
        """
        self.floor_height = floor_height
        self.min_building_height = min_building_height
        self.max_building_height = max_building_height
        self.detect_setbacks = detect_setbacks
        self.detect_overhangs = detect_overhangs
        self.vertical_buffer = vertical_buffer
        self.horizontal_buffer_ground = horizontal_buffer_ground
        self.horizontal_buffer_upper = horizontal_buffer_upper
        self.enable_floor_segmentation = enable_floor_segmentation
        
        # NEW: Adaptive wall detection
        self.enable_adaptive_buffer = enable_adaptive_buffer
        self.adaptive_buffer_min = adaptive_buffer_min
        self.adaptive_buffer_max = adaptive_buffer_max
        self.wall_verticality_threshold = wall_verticality_threshold
        self.wall_detection_enabled = wall_detection_enabled
        self.missing_wall_threshold = missing_wall_threshold
        
        # NEW v3.3.3: Gap detection parameters
        self.enable_gap_detection = True  # Enable gap/void detection
        self.gap_detection_resolution = 36  # Number of angular sectors (10° each)
        self.gap_detection_band_width = 1.5  # Width of perimeter band for gap detection (m)
        self.gap_min_points_per_sector = 5  # Minimum points per sector to consider covered
        self.gap_significant_threshold = 0.2  # If >20% perimeter has gaps, flag as significant
        
        logger.info("3D Building Extruder initialized (Enhanced Wall Detection)")
        logger.info(f"  Floor height: {floor_height}m")
        logger.info(f"  Vertical buffer: ±{vertical_buffer}m")
        logger.info(f"  Horizontal buffers: ground={horizontal_buffer_ground}m, upper={horizontal_buffer_upper}m")
        logger.info(f"  Adaptive buffer: {enable_adaptive_buffer} (range: {adaptive_buffer_min}-{adaptive_buffer_max}m)")
        logger.info(f"  Wall detection: verticality>{wall_verticality_threshold}, missing_threshold<{missing_wall_threshold}")
    
    def extrude_buildings(
        self,
        polygons: List['Polygon'],
        points: np.ndarray,
        heights: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        building_classes: Optional[List[int]] = None,
        normals: Optional[np.ndarray] = None,  # NEW: Surface normals for wall detection
        verticality: Optional[np.ndarray] = None,  # NEW: Verticality values
    ) -> List[BoundingBox3D]:
        """
        Create 3D bounding boxes for each building polygon with enhanced wall detection.
        
        Args:
            polygons: List of 2D building footprint polygons
            points: Point cloud array [N, 3] (X, Y, Z)
            heights: Height above ground [N] (if None, computed from Z)
            labels: Classification labels [N] (optional, for filtering)
            building_classes: List of class codes considered as buildings
            normals: Surface normals [N, 3] for wall detection (optional)
            verticality: Verticality values [N] for wall detection (optional)
            
        Returns:
            List of 3D bounding boxes with adaptive buffers and wall statistics
        """
        if not HAS_SPATIAL:
            logger.error("Shapely not available. Cannot perform 3D extrusion.")
            return []
        
        if points.shape[1] < 3:
            logger.error("Points must have Z coordinate for 3D extrusion")
            return []
        
        # Compute heights if not provided
        if heights is None:
            # Use Z coordinate as height (assumes points are already height-normalized)
            heights = points[:, 2]
            logger.info("Using Z coordinate as height")
        
        # Compute verticality from normals if needed
        if self.wall_detection_enabled and verticality is None and normals is not None:
            # Verticality = 1 - |nz| (high when normal is horizontal = vertical surface)
            verticality = 1.0 - np.abs(normals[:, 2])
            logger.info("Computed verticality from normals")
        
        # Filter building points if labels provided
        if labels is not None and building_classes is not None:
            building_mask = np.isin(labels, building_classes)
            building_points = points[building_mask]
            building_heights = heights[building_mask]
            building_normals = normals[building_mask] if normals is not None else None
            building_verticality = verticality[building_mask] if verticality is not None else None
            logger.info(f"Filtered to {len(building_points):,} building points")
        else:
            building_points = points
            building_heights = heights
            building_normals = normals
            building_verticality = verticality
        
        # Create 3D bounding boxes
        bboxes_3d = []
        
        for i, polygon in enumerate(polygons):
            bbox_3d = self._extrude_single_building(
                polygon=polygon,
                building_id=i,
                points=building_points,
                heights=building_heights,
                normals=building_normals,
                verticality=building_verticality,
            )
            
            if bbox_3d is not None:
                bboxes_3d.append(bbox_3d)
        
        logger.info(f"Created {len(bboxes_3d)} 3D bounding boxes")
        if self.wall_detection_enabled:
            total_walls = sum(b.wall_points for b in bboxes_3d)
            avg_coverage = np.mean([b.wall_coverage for b in bboxes_3d]) if bboxes_3d else 0
            missing_walls = sum(1 for b in bboxes_3d if b.has_missing_walls)
            logger.info(f"  Wall statistics: {total_walls:,} wall points, "
                       f"avg coverage: {avg_coverage:.1%}, {missing_walls} buildings with missing walls")
        
        if self.enable_gap_detection:
            total_gaps = sum(len(b.detected_gaps) if b.detected_gaps else 0 for b in bboxes_3d)
            avg_gap_ratio = np.mean([b.gap_coverage_ratio for b in bboxes_3d]) if bboxes_3d else 0
            significant_gaps = sum(1 for b in bboxes_3d if b.has_significant_gaps)
            logger.info(f"  Gap statistics: {total_gaps} gaps detected, "
                       f"avg gap coverage: {avg_gap_ratio:.1%}, {significant_gaps} buildings with significant gaps")
        
        return bboxes_3d
    
    def _extrude_single_building(
        self,
        polygon: 'Polygon',
        building_id: int,
        points: np.ndarray,
        heights: np.ndarray,
        normals: Optional[np.ndarray] = None,
        verticality: Optional[np.ndarray] = None,
    ) -> Optional[BoundingBox3D]:
        """
        Extrude a single 2D polygon into 3D bounding box with adaptive wall detection.
        
        Enhanced Algorithm:
        1. Compute adaptive buffer based on wall point density
        2. Find all points inside buffered polygon (2D)
        3. Analyze height distribution to determine building extent
        4. Detect walls using verticality and compute coverage
        5. Flag buildings with missing walls
        6. Create 3D bounding box with vertical extent
        7. Optionally segment by floor if enabled
        
        Args:
            polygon: 2D building footprint
            building_id: Unique building identifier
            points: All points [N, 3]
            heights: Heights above ground [N]
            normals: Surface normals [N, 3] (optional, for wall detection)
            verticality: Verticality values [N] (optional, for wall detection)
            
        Returns:
            3D bounding box with wall statistics or None if insufficient points
        """
        # Step 1: Compute adaptive buffer based on initial wall detection
        if self.enable_adaptive_buffer and verticality is not None:
            adaptive_buffer = self._compute_adaptive_buffer(
                polygon, points, heights, verticality
            )
        else:
            adaptive_buffer = self.horizontal_buffer_ground
        
        # Buffer polygon to capture nearby points (using adaptive buffer)
        buffered_polygon = utils.buffer_polygon(polygon, adaptive_buffer)
        
        # Find points inside buffered polygon (2D) - use consolidated utility
        inside_mask = utils.points_in_polygon(points, buffered_polygon, return_mask=True)
        
        if not inside_mask.any():
            logger.debug(f"Building {building_id}: No points found inside polygon (buffer={adaptive_buffer:.2f}m)")
            return None
        
        # Get data for points inside polygon
        inside_heights = heights[inside_mask]
        inside_points = points[inside_mask]
        inside_verticality = verticality[inside_mask] if verticality is not None else None
        
        # Filter by height range
        valid_height_mask = (
            (inside_heights >= self.min_building_height) &
            (inside_heights <= self.max_building_height)
        )
        
        if not valid_height_mask.any():
            logger.debug(f"Building {building_id}: No points in valid height range")
            return None
        
        valid_heights = inside_heights[valid_height_mask]
        valid_points = inside_points[valid_height_mask]
        valid_verticality = inside_verticality[valid_height_mask] if inside_verticality is not None else None
        
        # Determine vertical extent
        # Use percentiles to be robust to outliers
        z_min = np.percentile(valid_heights, 5)  # 5th percentile
        z_max = np.percentile(valid_heights, 95)  # 95th percentile
        
        # Add vertical buffer
        z_min = max(0.0, z_min - self.vertical_buffer)
        z_max = z_max + self.vertical_buffer
        
        building_height = z_max - z_min
        
        # Estimate number of floors
        n_floors = max(1, int(np.ceil(building_height / self.floor_height)))
        
        # Step 2: Detect walls and compute statistics
        wall_points_count = 0
        wall_coverage = 0.0
        has_missing_walls = False
        
        # NEW v3.3.3: Gap detection
        detected_gaps = []
        gap_total_length = 0.0
        gap_coverage_ratio = 0.0
        has_significant_gaps = False
        
        if self.wall_detection_enabled and valid_verticality is not None:
            # Detect wall points (high verticality)
            wall_mask = valid_verticality >= self.wall_verticality_threshold
            wall_points_count = wall_mask.sum()
            
            # Compute wall coverage (ratio of wall points to expected wall points)
            # Expected: perimeter * building_height * typical_wall_density
            perimeter = polygon.length  # meters
            expected_wall_volume = perimeter * 0.5 * building_height  # 0.5m thickness estimate
            typical_wall_density = 50  # points per cubic meter for walls
            expected_wall_points = expected_wall_volume * typical_wall_density
            
            if expected_wall_points > 0:
                wall_coverage = wall_points_count / expected_wall_points
                wall_coverage = min(wall_coverage, 1.0)  # Cap at 100%
            
            # Flag if walls are under-represented
            has_missing_walls = wall_coverage < self.missing_wall_threshold
            
            if has_missing_walls:
                logger.debug(
                    f"Building {building_id}: Missing walls detected "
                    f"(coverage: {wall_coverage:.1%}, threshold: {self.missing_wall_threshold:.1%})"
                )
            
            # NEW v3.3.3: Detect perimeter gaps (zones without points)
            if self.enable_gap_detection:
                detected_gaps, gap_total_length, gap_coverage_ratio, has_significant_gaps = (
                    self._detect_perimeter_gaps(
                        polygon=polygon,
                        points=inside_points,
                        heights=inside_heights,
                        verticality=inside_verticality,
                        building_height=building_height
                    )
                )
                
                if has_significant_gaps:
                    logger.debug(
                        f"Building {building_id}: Significant gaps detected "
                        f"({len(detected_gaps)} gaps, {gap_coverage_ratio:.1%} of perimeter)"
                    )
                    
                    # If significant gaps detected, consider increasing buffer adaptively
                    # Gaps indicate occlusion or missing data → need larger search radius
                    if self.enable_adaptive_buffer and gap_coverage_ratio > 0.3:
                        # Severe gaps (>30% perimeter) → increase buffer towards maximum
                        gap_penalty = min(gap_coverage_ratio - 0.3, 0.5) / 0.5  # 0 to 1
                        adaptive_buffer = min(
                            adaptive_buffer + gap_penalty * (self.adaptive_buffer_max - adaptive_buffer),
                            self.adaptive_buffer_max
                        )
                        logger.debug(
                            f"Building {building_id}: Increased buffer to {adaptive_buffer:.2f}m due to gaps"
                        )
        
        # Create base 3D bounding box
        bbox_3d = BoundingBox3D(
            polygon=polygon,
            z_min=z_min,
            z_max=z_max,
            building_id=building_id,
            n_floors=n_floors,
            floor_height=self.floor_height,
            n_points=valid_height_mask.sum(),
            # Wall statistics
            wall_points=wall_points_count,
            wall_coverage=wall_coverage,
            adaptive_buffer_distance=adaptive_buffer,
            has_missing_walls=has_missing_walls,
            # NEW v3.3.3: Gap statistics
            detected_gaps=detected_gaps if detected_gaps else None,
            gap_total_length=gap_total_length,
            gap_coverage_ratio=gap_coverage_ratio,
            has_significant_gaps=has_significant_gaps,
        )
        
        # Compute point density (points per cubic meter)
        polygon_area = polygon.area  # square meters
        volume = utils.compute_bounding_box_volume(valid_points)
        bbox_3d.point_density = bbox_3d.n_points / max(volume, 1.0)
        
        # Floor segmentation (optional)
        if self.enable_floor_segmentation and n_floors > 1:
            floor_segments = self._segment_by_floors(
                polygon=polygon,
                building_id=building_id,
                points=valid_points,
                heights=valid_heights,
                n_floors=n_floors,
                z_min=z_min,
                z_max=z_max
            )
            
            if floor_segments:
                bbox_3d.floor_polygons = [seg.footprint for seg in floor_segments]
                bbox_3d.floor_heights = [(seg.z_min, seg.z_max) for seg in floor_segments]
        
        logger.debug(
            f"Building {building_id}: "
            f"height={building_height:.1f}m, "
            f"floors={n_floors}, "
            f"points={bbox_3d.n_points}, "
            f"density={bbox_3d.point_density:.1f} pts/m³"
        )
        
        return bbox_3d
    
    def _compute_adaptive_buffer(
        self,
        polygon: 'Polygon',
        points: np.ndarray,
        heights: np.ndarray,
        verticality: np.ndarray,
    ) -> float:
        """
        Compute adaptive lateral buffer distance based on wall point density.
        
        Strategy:
        1. Create narrow search band around polygon perimeter (0.5m-2m)
        2. Count wall points (high verticality) in this band
        3. Compute wall density along perimeter
        4. If low wall density → increase buffer (walls may be further out)
        5. If high wall density → use minimum buffer (walls well captured)
        6. Range: adaptive_buffer_min (0.5m) to adaptive_buffer_max (5.0m)
        
        This adapts to:
        - Missing wall points → larger buffer to search further
        - Thick walls → larger buffer to capture full facade
        - Overhangs/balconies → larger buffer at upper levels
        - Sparse LiDAR → larger buffer to compensate
        
        Args:
            polygon: Building footprint polygon
            points: All points [N, 3]
            heights: Heights above ground [N]
            verticality: Verticality values [N]
            
        Returns:
            Adaptive buffer distance in meters
        """
        # Create narrow search band around perimeter (1.5m band)
        search_buffer = 1.5
        outer_ring = utils.buffer_polygon(polygon, search_buffer)
        inner_ring = polygon  # Original polygon
        
        # Find points in search band
        outer_mask = utils.points_in_polygon(points, outer_ring, return_mask=True)
        inner_mask = utils.points_in_polygon(points, inner_ring, return_mask=True)
        band_mask = outer_mask & ~inner_mask  # Points between inner and outer
        
        if not band_mask.any():
            # No points in band → likely missing walls, use large buffer
            logger.debug(f"No points in perimeter band, using max buffer: {self.adaptive_buffer_max}m")
            return self.adaptive_buffer_max
        
        # Get verticality of points in band
        band_verticality = verticality[band_mask]
        band_heights = heights[band_mask]
        
        # Count wall points (high verticality, building height range)
        wall_mask = (
            (band_verticality >= self.wall_verticality_threshold) &
            (band_heights >= self.min_building_height) &
            (band_heights <= self.max_building_height)
        )
        n_wall_points = wall_mask.sum()
        n_band_points = band_mask.sum()
        
        # Compute wall density
        wall_density = n_wall_points / max(n_band_points, 1)
        
        # Compute perimeter-normalized wall density
        perimeter = polygon.length  # meters
        wall_points_per_meter = n_wall_points / max(perimeter, 1.0)
        
        # Adaptive buffer logic:
        # - High wall density (>50%) → minimum buffer (walls well captured)
        # - Medium wall density (20-50%) → moderate buffer
        # - Low wall density (<20%) → maximum buffer (search further for walls)
        # - Very low point density (<5 pts/m) → increase buffer for sparse data
        
        if wall_density >= 0.5 and wall_points_per_meter >= 10:
            # Excellent wall coverage, use minimum buffer
            buffer_distance = self.adaptive_buffer_min
            reason = f"high wall density ({wall_density:.1%})"
        elif wall_density >= 0.3 and wall_points_per_meter >= 5:
            # Good wall coverage, use moderate buffer
            buffer_distance = (self.adaptive_buffer_min + self.adaptive_buffer_max) / 2
            reason = f"medium wall density ({wall_density:.1%})"
        elif wall_points_per_meter < 5:
            # Sparse data, use large buffer
            buffer_distance = self.adaptive_buffer_max
            reason = f"sparse data ({wall_points_per_meter:.1f} pts/m)"
        else:
            # Low wall density, interpolate based on density
            # Map wall_density [0, 0.3] → buffer [max, (min+max)/2]
            t = wall_density / 0.3  # 0 to 1
            buffer_distance = self.adaptive_buffer_max - t * (self.adaptive_buffer_max - (self.adaptive_buffer_min + self.adaptive_buffer_max) / 2)
            reason = f"low wall density ({wall_density:.1%})"
        
        logger.debug(
            f"Adaptive buffer: {buffer_distance:.2f}m ({reason}, "
            f"{n_wall_points} wall pts, {wall_points_per_meter:.1f} pts/m)"
        )
        
        return buffer_distance
    
    def _detect_perimeter_gaps(
        self,
        polygon: 'Polygon',
        points: np.ndarray,
        heights: np.ndarray,
        verticality: Optional[np.ndarray] = None,
        building_height: float = 10.0
    ) -> Tuple[List[Tuple[float, float, float]], float, float, bool]:
        """
        Detect gaps/voids in building perimeter (zones without wall points).
        
        This identifies areas where walls are missing or occluded in the LiDAR data.
        Critical for:
        - Identifying incomplete building captures
        - Adjusting buffer strategy per building side
        - Quality assessment of building detection
        - Detecting occlusions (trees, adjacent buildings)
        
        Algorithm:
        1. Divide building perimeter into angular sectors (e.g., 36 sectors = 10° each)
        2. For each sector, count points in a narrow perimeter band
        3. Identify sectors with insufficient points as "gaps"
        4. Merge adjacent gap sectors into continuous gap segments
        5. Compute gap metrics (total length, coverage ratio)
        
        Args:
            polygon: Building footprint polygon
            points: All points [N, 3]
            heights: Heights above ground [N]
            verticality: Verticality values [N] (optional, prioritizes wall points)
            building_height: Building height for filtering (meters)
            
        Returns:
            Tuple of:
            - detected_gaps: List of (start_angle, end_angle, width_m) for each gap
            - gap_total_length: Total perimeter length with gaps (meters)
            - gap_coverage_ratio: Ratio of perimeter with gaps (0-1)
            - has_significant_gaps: True if gaps > threshold
        """
        if not HAS_SPATIAL:
            return [], 0.0, 0.0, False
        
        # Get building centroid for angular analysis
        centroid = polygon.centroid
        cx, cy = centroid.x, centroid.y
        
        # Create perimeter search band
        outer_ring = utils.buffer_polygon(polygon, self.gap_detection_band_width)
        inner_ring = polygon
        
        # Find points in search band
        outer_mask = utils.points_in_polygon(points, outer_ring, return_mask=True)
        inner_mask = utils.points_in_polygon(points, inner_ring, return_mask=True)
        band_mask = outer_mask & ~inner_mask
        
        if not band_mask.any():
            # No points in band → entire perimeter is a gap
            perimeter = polygon.length
            return [(0.0, 360.0, perimeter)], perimeter, 1.0, True
        
        # Filter by height (building height range)
        band_mask = band_mask & (heights >= self.min_building_height) & (heights <= building_height + 2.0)
        
        if not band_mask.any():
            # No valid height points → entire perimeter is a gap
            perimeter = polygon.length
            return [(0.0, 360.0, perimeter)], perimeter, 1.0, True
        
        # Get points in band
        band_points = points[band_mask]
        band_verticality = verticality[band_mask] if verticality is not None else None
        
        # Compute angles from centroid to each point
        dx = band_points[:, 0] - cx
        dy = band_points[:, 1] - cy
        angles = np.arctan2(dy, dx) * 180.0 / np.pi  # Convert to degrees [-180, 180]
        angles = (angles + 360.0) % 360.0  # Normalize to [0, 360]
        
        # Divide into angular sectors
        sector_size = 360.0 / self.gap_detection_resolution
        sector_counts = np.zeros(self.gap_detection_resolution, dtype=int)
        
        # Count points per sector (prioritize wall points if verticality available)
        for i, angle in enumerate(angles):
            sector_idx = int(angle / sector_size) % self.gap_detection_resolution
            
            # Weight by verticality if available (walls are more important)
            if band_verticality is not None and band_verticality[i] >= self.wall_verticality_threshold:
                sector_counts[sector_idx] += 2  # Double weight for wall points
            else:
                sector_counts[sector_idx] += 1
        
        # Identify gap sectors (insufficient points)
        gap_sectors = sector_counts < self.gap_min_points_per_sector
        
        # Merge adjacent gap sectors into continuous gaps
        detected_gaps = []
        perimeter = polygon.length
        sector_arc_length = perimeter / self.gap_detection_resolution
        
        in_gap = False
        gap_start_sector = -1
        
        # Scan sectors twice to handle wraparound at 0°/360°
        for i in range(self.gap_detection_resolution * 2):
            sector_idx = i % self.gap_detection_resolution
            is_gap = gap_sectors[sector_idx]
            
            if is_gap and not in_gap:
                # Start of new gap
                in_gap = True
                gap_start_sector = sector_idx
            elif not is_gap and in_gap:
                # End of gap
                gap_end_sector = sector_idx - 1
                if gap_end_sector < 0:
                    gap_end_sector += self.gap_detection_resolution
                
                # Compute gap angular extent and arc length
                start_angle = gap_start_sector * sector_size
                end_angle = gap_end_sector * sector_size
                
                # Handle wraparound
                if end_angle < start_angle:
                    end_angle += 360.0
                
                angular_extent = end_angle - start_angle
                gap_arc_length = (angular_extent / 360.0) * perimeter
                
                # Only record if we haven't wrapped around (avoid duplicates)
                if i < self.gap_detection_resolution or gap_start_sector == 0:
                    detected_gaps.append((start_angle, end_angle % 360.0, gap_arc_length))
                
                in_gap = False
        
        # Close final gap if still open
        if in_gap and gap_start_sector >= 0:
            gap_end_sector = (self.gap_detection_resolution - 1)
            start_angle = gap_start_sector * sector_size
            end_angle = gap_end_sector * sector_size
            
            if start_angle > 0:  # Not a full perimeter gap
                angular_extent = end_angle - start_angle
                if angular_extent < 0:
                    angular_extent += 360.0
                gap_arc_length = (angular_extent / 360.0) * perimeter
                detected_gaps.append((start_angle, end_angle, gap_arc_length))
        
        # Compute gap statistics
        gap_total_length = sum(gap[2] for gap in detected_gaps)
        gap_coverage_ratio = gap_total_length / perimeter if perimeter > 0 else 0.0
        has_significant_gaps = gap_coverage_ratio > self.gap_significant_threshold
        
        logger.debug(
            f"Gap detection: {len(detected_gaps)} gaps, "
            f"total_length={gap_total_length:.1f}m, "
            f"coverage={gap_coverage_ratio:.1%}, "
            f"significant={has_significant_gaps}"
        )
        
        return detected_gaps, gap_total_length, gap_coverage_ratio, has_significant_gaps
    
    def _segment_by_floors(
        self,
        polygon: 'Polygon',
        building_id: int,
        points: np.ndarray,
        heights: np.ndarray,
        n_floors: int,
        z_min: float,
        z_max: float
    ) -> List[FloorSegment]:
        """
        Segment building into floors and detect setbacks.
        
        Setbacks: Upper floors may have smaller footprints than ground floor.
        Common in: apartment buildings, terraced buildings, stepped facades.
        
        Algorithm:
        1. Divide height range into floor segments
        2. For each floor, compute convex hull of points
        3. Compare floor footprints to detect setbacks
        4. Apply different buffers per floor (ground vs upper)
        
        Args:
            polygon: Original 2D footprint
            building_id: Building identifier
            points: Points inside building [M, 3]
            heights: Heights of these points [M]
            n_floors: Number of floors detected
            z_min: Minimum building height
            z_max: Maximum building height
            
        Returns:
            List of floor segments
        """
        floor_segments = []
        building_height = z_max - z_min
        floor_height = building_height / n_floors
        
        previous_area = None
        
        for floor_idx in range(n_floors):
            # Height range for this floor
            floor_z_min = z_min + (floor_idx * floor_height)
            floor_z_max = z_min + ((floor_idx + 1) * floor_height)
            
            # Points in this floor
            floor_mask = (heights >= floor_z_min) & (heights < floor_z_max)
            
            if not floor_mask.any() or floor_mask.sum() < 10:
                # Insufficient points, use base polygon
                floor_footprint = polygon
            else:
                # Compute convex hull of floor points
                floor_points_2d = points[floor_mask, :2]
                
                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(floor_points_2d)
                    hull_points = floor_points_2d[hull.vertices]
                    floor_footprint = Polygon(hull_points)
                    
                    # Apply appropriate buffer
                    if floor_idx == 0:
                        # Ground floor: use ground buffer
                        buffer_dist = self.horizontal_buffer_ground
                    else:
                        # Upper floors: use upper buffer (captures balconies)
                        buffer_dist = self.horizontal_buffer_upper
                    
                    floor_footprint = floor_footprint.buffer(buffer_dist)
                    
                except Exception as e:
                    logger.debug(f"Building {building_id}, floor {floor_idx}: "
                               f"Convex hull failed, using base polygon")
                    floor_footprint = polygon
            
            # Detect setback
            has_setback = False
            if previous_area is not None and self.detect_setbacks:
                current_area = floor_footprint.area
                # Setback if current floor is >10% smaller than previous
                if current_area < 0.9 * previous_area:
                    has_setback = True
                    logger.debug(f"Building {building_id}, floor {floor_idx}: "
                               f"Setback detected (area: {current_area:.1f} < {previous_area:.1f})")
            
            # Create floor segment
            is_roof = (floor_idx == n_floors - 1)
            segment = FloorSegment(
                floor_index=floor_idx,
                z_min=floor_z_min,
                z_max=floor_z_max,
                footprint=floor_footprint,
                n_points=floor_mask.sum(),
                has_setback=has_setback,
                is_roof=is_roof
            )
            
            floor_segments.append(segment)
            previous_area = floor_footprint.area
        
        return floor_segments
    
    def classify_points_3d(
        self,
        points: np.ndarray,
        heights: np.ndarray,
        bboxes_3d: List[BoundingBox3D],
        building_class: int = 6
    ) -> np.ndarray:
        """
        Classify points using 3D bounding boxes.
        
        This is the key method that uses volumetric containment to classify
        building points. Much more accurate than 2D polygon containment.
        
        Algorithm:
        1. For each point, check all 3D bounding boxes
        2. Point is classified as building if:
           - XY coordinates inside 2D footprint (or floor-specific footprint)
           - Z coordinate within building height range [z_min, z_max]
        3. Handle overlapping buildings (use nearest centroid)
        
        Args:
            points: Point cloud [N, 3]
            heights: Heights above ground [N]
            bboxes_3d: List of 3D bounding boxes
            building_class: ASPRS class code for buildings (default: 6)
            
        Returns:
            Classification labels [N]
        """
        n_points = len(points)
        labels = np.zeros(n_points, dtype=np.uint8)
        
        if not bboxes_3d:
            logger.warning("No 3D bounding boxes provided")
            return labels
        
        points_2d = points[:, :2]
        
        logger.info(f"Classifying {n_points:,} points using {len(bboxes_3d)} 3D bounding boxes")
        
        # Process each bounding box
        for bbox_3d in bboxes_3d:
            # Quick height filter
            height_mask = (heights >= bbox_3d.z_min) & (heights <= bbox_3d.z_max)
            
            if not height_mask.any():
                continue
            
            # Check 2D containment for points in height range
            candidate_points = points_2d[height_mask]
            
            if self.enable_floor_segmentation and bbox_3d.floor_polygons is not None:
                # Use floor-specific footprints for more accurate classification
                for floor_idx, (floor_poly, (floor_z_min, floor_z_max)) in enumerate(
                    zip(bbox_3d.floor_polygons, bbox_3d.floor_heights)
                ):
                    # Points in this floor's height range
                    floor_mask = height_mask & (heights >= floor_z_min) & (heights <= floor_z_max)
                    
                    if not floor_mask.any():
                        continue
                    
                    # Check 2D containment in floor polygon
                    from shapely.vectorized import contains
                    floor_points_2d = points_2d[floor_mask]
                    inside_floor = contains(floor_poly, floor_points_2d[:, 0], floor_points_2d[:, 1])
                    
                    # Update labels
                    floor_indices = np.where(floor_mask)[0]
                    labels[floor_indices[inside_floor]] = building_class
            
            else:
                # Use single footprint for entire building
                from shapely.vectorized import contains
                inside_2d = contains(bbox_3d.polygon, candidate_points[:, 0], candidate_points[:, 1])
                
                # Update labels
                candidate_indices = np.where(height_mask)[0]
                labels[candidate_indices[inside_2d]] = building_class
        
        n_classified = (labels == building_class).sum()
        logger.info(f"Classified {n_classified:,} points as buildings ({100*n_classified/n_points:.1f}%)")
        
        return labels
    
    def export_3d_bboxes_to_gdf(
        self,
        bboxes_3d: List[BoundingBox3D],
        crs: str = "EPSG:2154"
    ) -> Optional['gpd.GeoDataFrame']:
        """
        Export 3D bounding boxes to GeoDataFrame for visualization.
        
        Args:
            bboxes_3d: List of 3D bounding boxes
            crs: Coordinate reference system
            
        Returns:
            GeoDataFrame with bounding box geometries and attributes
        """
        if not HAS_SPATIAL:
            logger.error("Geopandas not available")
            return None
        
        records = []
        
        for bbox_3d in bboxes_3d:
            record = {
                'geometry': bbox_3d.polygon,
                'building_id': bbox_3d.building_id,
                'z_min': bbox_3d.z_min,
                'z_max': bbox_3d.z_max,
                'height': bbox_3d.z_max - bbox_3d.z_min,
                'n_floors': bbox_3d.n_floors,
                'n_points': bbox_3d.n_points,
                'point_density': bbox_3d.point_density,
                'area': bbox_3d.polygon.area,
                'volume': bbox_3d.polygon.area * (bbox_3d.z_max - bbox_3d.z_min)
            }
            records.append(record)
        
        gdf = gpd.GeoDataFrame(records, crs=crs)
        
        logger.info(f"Exported {len(gdf)} 3D bounding boxes to GeoDataFrame")
        
        return gdf


def create_3d_bboxes_from_ground_truth(
    buildings_gdf: 'gpd.GeoDataFrame',
    points: np.ndarray,
    heights: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    building_classes: Optional[List[int]] = None,
    normals: Optional[np.ndarray] = None,  # NEW: For wall detection
    verticality: Optional[np.ndarray] = None,  # NEW: For wall detection
    **extruder_kwargs
) -> List[BoundingBox3D]:
    """
    Convenience function to create 3D bounding boxes from ground truth with wall detection.
    
    Args:
        buildings_gdf: GeoDataFrame with building polygons
        points: Point cloud [N, 3]
        heights: Heights above ground [N]
        labels: Classification labels [N]
        building_classes: List of building class codes
        normals: Surface normals [N, 3] for wall detection (optional)
        verticality: Verticality values [N] for wall detection (optional)
        **extruder_kwargs: Additional arguments for Building3DExtruder
                          (e.g., adaptive_buffer_max=5.0, wall_verticality_threshold=0.65)
        
    Returns:
        List of 3D bounding boxes with adaptive buffers and wall statistics
        
    Example:
        >>> # Create bboxes with enhanced wall detection
        >>> bboxes = create_3d_bboxes_from_ground_truth(
        ...     buildings_gdf=my_buildings,
        ...     points=point_cloud,
        ...     heights=heights_above_ground,
        ...     normals=surface_normals,
        ...     enable_adaptive_buffer=True,
        ...     adaptive_buffer_max=5.0,  # Search up to 5m for walls
        ...     wall_verticality_threshold=0.65,
        ...     missing_wall_threshold=0.3
        ... )
        >>> 
        >>> # Check wall statistics
        >>> for bbox in bboxes:
        ...     print(f"Building {bbox.building_id}: "
        ...           f"{bbox.wall_points} wall points, "
        ...           f"coverage: {bbox.wall_coverage:.1%}, "
        ...           f"buffer: {bbox.adaptive_buffer_distance:.2f}m")
    """
    # Extract polygons from GeoDataFrame
    polygons = []
    for idx, row in buildings_gdf.iterrows():
        geom = row['geometry']
        if isinstance(geom, Polygon):
            polygons.append(geom)
        elif isinstance(geom, MultiPolygon):
            # Flatten MultiPolygon into individual Polygons
            polygons.extend(list(geom.geoms))
    
    # Create extruder
    extruder = Building3DExtruder(**extruder_kwargs)
    
    # Extrude buildings with wall detection
    bboxes_3d = extruder.extrude_buildings(
        polygons=polygons,
        points=points,
        heights=heights,
        labels=labels,
        building_classes=building_classes,
        normals=normals,
        verticality=verticality,
    )
    
    return bboxes_3d


# Backward compatibility exports
__all__ = [
    'Building3DExtruder',
    'BoundingBox3D',
    'FloorSegment',
    'create_3d_bboxes_from_ground_truth'
]
