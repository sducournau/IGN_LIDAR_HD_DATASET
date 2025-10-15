"""
Advanced Classification Module with Geometric Features, NDVI, and Ground Truth

This module provides enhanced classification using:
- Geometric features (height, normals, planarity, curvature)
- NDVI for vegetation detection
- IGN BD TOPO® ground truth with intelligent road buffers
- Multi-criteria decision fusion
- Mode-aware building detection (ASPRS, LOD2, LOD3)

Author: Classification Enhancement
Date: October 15, 2025
"""

import logging
from typing import Dict, Optional, Tuple, List, TYPE_CHECKING
import numpy as np

# Import building detection module
from .building_detection import (
    BuildingDetectionMode,
    BuildingDetectionConfig,
    detect_buildings_multi_mode
)

# Import transport detection module
from .transport_detection import (
    TransportDetectionMode,
    TransportDetectionConfig,
    detect_transport_multi_mode
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from shapely.geometry import Point, Polygon, MultiPolygon
    import geopandas as gpd

try:
    from shapely.geometry import Point, Polygon, MultiPolygon
    import geopandas as gpd
    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False


class AdvancedClassifier:
    """
    Advanced point cloud classifier combining multiple data sources.
    
    Classification hierarchy:
    1. Ground Truth (IGN BD TOPO®) - highest priority
    2. NDVI-based vegetation detection
    3. Geometric feature analysis
    4. Height-based classification
    5. Default/fallback classification
    """
    
    # ASPRS Classification codes (standard + extended)
    ASPRS_UNCLASSIFIED = 1
    ASPRS_GROUND = 2
    ASPRS_LOW_VEGETATION = 3
    ASPRS_MEDIUM_VEGETATION = 4
    ASPRS_HIGH_VEGETATION = 5
    ASPRS_BUILDING = 6
    ASPRS_LOW_POINT = 7
    ASPRS_WATER = 9
    ASPRS_RAIL = 10  # Rail/Railway
    ASPRS_ROAD = 11
    ASPRS_BRIDGE = 17  # Bridge deck
    
    # Extended codes for additional BD TOPO classes
    ASPRS_PARKING = 40  # Parking area (custom)
    ASPRS_SPORTS = 41  # Sports facility (custom)
    ASPRS_CEMETERY = 42  # Cemetery (custom)
    ASPRS_POWER_LINE = 43  # Power line corridor (custom)
    ASPRS_AGRICULTURE = 44  # Agricultural land (custom)
    
    def __init__(
        self,
        use_ground_truth: bool = True,
        use_ndvi: bool = True,
        use_geometric: bool = True,
        road_buffer_tolerance: float = 0.5,
        ndvi_veg_threshold: float = 0.35,
        ndvi_building_threshold: float = 0.15,
        height_low_veg_threshold: float = 0.5,
        height_medium_veg_threshold: float = 2.0,
        planarity_road_threshold: float = 0.8,
        planarity_building_threshold: float = 0.7,
        building_detection_mode: str = 'asprs',
        transport_detection_mode: str = 'asprs_standard'
    ):
        """
        Initialize advanced classifier.
        
        Args:
            use_ground_truth: Use IGN BD TOPO® ground truth
            use_ndvi: Use NDVI for vegetation refinement
            use_geometric: Use geometric features
            road_buffer_tolerance: Additional buffer around roads/rails (meters)
            ndvi_veg_threshold: NDVI threshold for vegetation (>= value)
            ndvi_building_threshold: NDVI threshold for non-vegetation (<= value)
            height_low_veg_threshold: Height threshold for low vegetation
            height_medium_veg_threshold: Height threshold for medium vegetation
            planarity_road_threshold: Planarity for road surfaces
            planarity_building_threshold: Planarity for building surfaces
            building_detection_mode: Mode for building detection ('asprs', 'lod2', or 'lod3')
            transport_detection_mode: Mode for transport detection ('asprs_standard', 'asprs_extended', or 'lod2')
        """
        self.use_ground_truth = use_ground_truth
        self.use_ndvi = use_ndvi
        self.use_geometric = use_geometric
        self.building_detection_mode = building_detection_mode.lower()
        self.transport_detection_mode = transport_detection_mode.lower()
        
        # Thresholds
        self.road_buffer_tolerance = road_buffer_tolerance
        self.ndvi_veg_threshold = ndvi_veg_threshold
        self.ndvi_building_threshold = ndvi_building_threshold
        self.height_low_veg = height_low_veg_threshold
        self.height_medium_veg = height_medium_veg_threshold
        self.planarity_road = planarity_road_threshold
        self.planarity_building = planarity_building_threshold
        
        logger.info("🎯 Advanced Classifier initialized")
        logger.info(f"  Ground truth: {use_ground_truth}")
        logger.info(f"  NDVI refinement: {use_ndvi}")
        logger.info(f"  Geometric features: {use_geometric}")
        logger.info(f"  Building detection mode: {self.building_detection_mode.upper()}")
        logger.info(f"  Transport detection mode: {self.transport_detection_mode.upper()}")
    
    def classify_points(
        self,
        points: np.ndarray,
        ground_truth_features: Optional[Dict[str, 'gpd.GeoDataFrame']] = None,
        ndvi: Optional[np.ndarray] = None,
        height: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        planarity: Optional[np.ndarray] = None,
        curvature: Optional[np.ndarray] = None,
        intensity: Optional[np.ndarray] = None,
        return_number: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Classify points using all available data sources.
        
        Args:
            points: Point coordinates [N, 3] (X, Y, Z)
            ground_truth_features: Dictionary of feature type -> GeoDataFrame
            ndvi: NDVI values [N] in range [-1, 1]
            height: Height above ground [N]
            normals: Surface normals [N, 3]
            planarity: Planarity values [N] in range [0, 1]
            curvature: Surface curvature [N]
            intensity: LiDAR intensity [N]
            return_number: Return number [N]
            
        Returns:
            Classification labels [N] in ASPRS format
        """
        n_points = len(points)
        logger.info(f"🎯 Classifying {n_points:,} points with advanced method")
        
        # Initialize with unclassified
        labels = np.full(n_points, self.ASPRS_UNCLASSIFIED, dtype=np.uint8)
        
        # Track confidence scores for each classification
        confidence = np.zeros(n_points, dtype=np.float32)
        
        # Stage 1: Geometric-based classification (if available)
        if self.use_geometric and height is not None:
            logger.info("  Stage 1: Geometric feature classification")
            labels, confidence = self._classify_by_geometry(
                labels, confidence, points, height, normals, planarity, 
                curvature, intensity, return_number
            )
        
        # Stage 2: NDVI-based vegetation detection
        if self.use_ndvi and ndvi is not None:
            logger.info("  Stage 2: NDVI-based vegetation refinement")
            labels, confidence = self._classify_by_ndvi(
                labels, confidence, ndvi, height
            )
        
        # Stage 3: Ground truth (highest priority - overwrites previous)
        if self.use_ground_truth and ground_truth_features:
            logger.info("  Stage 3: Ground truth classification (highest priority)")
            labels = self._classify_by_ground_truth(
                labels, points, ground_truth_features, ndvi
            )
        
        # Log final distribution
        self._log_distribution(labels)
        
        return labels
    
    def _classify_by_geometry(
        self,
        labels: np.ndarray,
        confidence: np.ndarray,
        points: np.ndarray,
        height: np.ndarray,
        normals: Optional[np.ndarray],
        planarity: Optional[np.ndarray],
        curvature: Optional[np.ndarray],
        intensity: Optional[np.ndarray],
        return_number: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Classify based on geometric features."""
        
        # Ground detection (low height + high planarity)
        if height is not None and planarity is not None:
            ground_mask = (height < 0.2) & (planarity > 0.85)
            labels[ground_mask] = self.ASPRS_GROUND
            confidence[ground_mask] = 0.9
            logger.info(f"    Ground: {np.sum(ground_mask):,} points")
        
        # Road detection (low height + very high planarity + horizontal normals)
        if height is not None and planarity is not None:
            road_mask = (
                (height >= 0.2) & 
                (height < 2.0) & 
                (planarity > self.planarity_road)
            )
            
            # Refine with horizontal normals if available
            if normals is not None:
                # Normals pointing up (z component close to 1)
                horizontal_mask = np.abs(normals[:, 2]) > 0.9
                road_mask = road_mask & horizontal_mask
            
            labels[road_mask] = self.ASPRS_ROAD
            confidence[road_mask] = 0.7
            logger.info(f"    Roads (geometric): {np.sum(road_mask):,} points")
        
        # Building detection (medium height + high planarity)
        # Use mode-aware building detection if enabled
        if height is not None and planarity is not None:
            if self.building_detection_mode in ['asprs', 'lod2', 'lod3']:
                # Use new building detection system
                try:
                    # Compute verticality if normals available
                    verticality = None
                    if normals is not None:
                        verticality = 1.0 - np.abs(normals[:, 2])  # 1 - |nz|
                    
                    # Prepare features for building detection
                    features_dict = {
                        'height': height,
                        'planarity': planarity,
                        'verticality': verticality,
                        'normals': normals,
                        'curvature': curvature,
                        'intensity': intensity,
                        'points': points
                    }
                    
                    # Detect buildings using mode-aware detection
                    labels_updated, stats = detect_buildings_multi_mode(
                        labels=labels,
                        features=features_dict,
                        mode=self.building_detection_mode,
                        ground_truth_mask=None,  # Ground truth handled separately
                        config=None  # Use default mode-specific config
                    )
                    
                    # Update labels where buildings detected
                    building_mask = labels_updated != labels
                    labels = labels_updated
                    confidence[building_mask] = 0.7
                    
                    logger.info(f"    Buildings ({self.building_detection_mode.upper()}): {stats.get('total_building', 0):,} points")
                    
                except Exception as e:
                    logger.warning(f"    Mode-aware building detection failed: {e}, using fallback")
                    # Fall back to simple detection
                    building_mask = (
                        (height >= 2.0) &
                        (planarity > self.planarity_building)
                    )
                    
                    if normals is not None:
                        roof_mask = np.abs(normals[:, 2]) > 0.7
                        wall_mask = np.abs(normals[:, 2]) < 0.3
                        building_mask = building_mask & (roof_mask | wall_mask)
                    
                    labels[building_mask] = self.ASPRS_BUILDING
                    confidence[building_mask] = 0.6
                    logger.info(f"    Buildings (geometric fallback): {np.sum(building_mask):,} points")
            else:
                # Legacy simple detection
                building_mask = (
                    (height >= 2.0) &
                    (planarity > self.planarity_building)
                )
                
                # Refine with vertical/horizontal normals if available
                if normals is not None:
                    # Either horizontal (roofs) or vertical (walls)
                    roof_mask = np.abs(normals[:, 2]) > 0.7  # Horizontal
                    wall_mask = np.abs(normals[:, 2]) < 0.3  # Vertical
                    building_mask = building_mask & (roof_mask | wall_mask)
                
                labels[building_mask] = self.ASPRS_BUILDING
                confidence[building_mask] = 0.6
                logger.info(f"    Buildings (geometric): {np.sum(building_mask):,} points")
        
        # Vegetation detection (low planarity + variable height)
        if height is not None and planarity is not None:
            # Low planarity suggests organic/irregular surfaces
            veg_mask = (planarity < 0.4) & (height > 0.2)
            
            # Classify by height
            low_veg_mask = veg_mask & (height < self.height_low_veg)
            medium_veg_mask = veg_mask & (height >= self.height_low_veg) & (height < self.height_medium_veg)
            high_veg_mask = veg_mask & (height >= self.height_medium_veg)
            
            labels[low_veg_mask] = self.ASPRS_LOW_VEGETATION
            labels[medium_veg_mask] = self.ASPRS_MEDIUM_VEGETATION
            labels[high_veg_mask] = self.ASPRS_HIGH_VEGETATION
            
            confidence[low_veg_mask] = 0.5
            confidence[medium_veg_mask] = 0.5
            confidence[high_veg_mask] = 0.5
            
            n_veg = np.sum(low_veg_mask) + np.sum(medium_veg_mask) + np.sum(high_veg_mask)
            logger.info(f"    Vegetation (geometric): {n_veg:,} points")
        
        return labels, confidence
    
    def _classify_by_ndvi(
        self,
        labels: np.ndarray,
        confidence: np.ndarray,
        ndvi: np.ndarray,
        height: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Classify and refine using NDVI."""
        
        # High NDVI = vegetation (overwrite low-confidence classifications)
        high_ndvi_mask = (ndvi >= self.ndvi_veg_threshold) & (confidence < 0.8)
        
        if height is not None:
            # Classify by height
            low_veg = high_ndvi_mask & (height < self.height_low_veg)
            medium_veg = high_ndvi_mask & (height >= self.height_low_veg) & (height < self.height_medium_veg)
            high_veg = high_ndvi_mask & (height >= self.height_medium_veg)
            
            labels[low_veg] = self.ASPRS_LOW_VEGETATION
            labels[medium_veg] = self.ASPRS_MEDIUM_VEGETATION
            labels[high_veg] = self.ASPRS_HIGH_VEGETATION
            
            confidence[high_ndvi_mask] = 0.85
            
            n_veg = np.sum(high_ndvi_mask)
            logger.info(f"    Vegetation (NDVI): {n_veg:,} points")
        else:
            # Default to medium vegetation if no height
            labels[high_ndvi_mask] = self.ASPRS_MEDIUM_VEGETATION
            confidence[high_ndvi_mask] = 0.8
            logger.info(f"    Vegetation (NDVI, no height): {np.sum(high_ndvi_mask):,} points")
        
        # Low NDVI = non-vegetation (buildings/roads/ground)
        # Refine building/road classifications
        low_ndvi_mask = (ndvi <= self.ndvi_building_threshold)
        
        # If labeled as vegetation but low NDVI, reclassify
        veg_classes = [self.ASPRS_LOW_VEGETATION, self.ASPRS_MEDIUM_VEGETATION, self.ASPRS_HIGH_VEGETATION]
        incorrect_veg = low_ndvi_mask & np.isin(labels, veg_classes)
        
        if np.any(incorrect_veg):
            # Reclassify based on height if available
            if height is not None:
                low_height = incorrect_veg & (height < 2.0)
                high_height = incorrect_veg & (height >= 2.0)
                
                labels[low_height] = self.ASPRS_ROAD  # or ground
                labels[high_height] = self.ASPRS_BUILDING
                
                confidence[incorrect_veg] = 0.7
                logger.info(f"    Reclassified low-NDVI vegetation: {np.sum(incorrect_veg):,} points")
            else:
                labels[incorrect_veg] = self.ASPRS_UNCLASSIFIED
        
        return labels, confidence
    
    def _classify_by_ground_truth(
        self,
        labels: np.ndarray,
        points: np.ndarray,
        ground_truth_features: Dict[str, 'gpd.GeoDataFrame'],
        ndvi: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Classify using IGN BD TOPO® ground truth with intelligent road buffers.
        
        This is the highest priority classification and overwrites previous labels.
        """
        if not HAS_SPATIAL:
            logger.warning("Spatial libraries not available, skipping ground truth")
            return labels
        
        # Create Point geometries
        point_geoms = [Point(p[0], p[1]) for p in points]
        
        # Classification priority (reverse order - last wins)
        # Lower in list = higher priority (overwrites previous)
        priority_order = [
            ('vegetation', self.ASPRS_MEDIUM_VEGETATION),
            ('water', self.ASPRS_WATER),
            ('cemeteries', self.ASPRS_CEMETERY),
            ('parking', self.ASPRS_PARKING),
            ('sports', self.ASPRS_SPORTS),
            ('power_lines', self.ASPRS_POWER_LINE),
            ('railways', self.ASPRS_RAIL),
            ('roads', self.ASPRS_ROAD),
            ('bridges', self.ASPRS_BRIDGE),
            ('buildings', self.ASPRS_BUILDING)
        ]
        
        for feature_type, asprs_class in priority_order:
            if feature_type not in ground_truth_features:
                continue
            
            gdf = ground_truth_features[feature_type]
            if gdf is None or len(gdf) == 0:
                continue
            
            logger.info(f"    Processing {feature_type}: {len(gdf)} features")
            
            # Special handling for roads and railways with intelligent buffers
            if feature_type == 'roads':
                labels = self._classify_roads_with_buffer(
                    labels, point_geoms, gdf, asprs_class
                )
            elif feature_type == 'railways':
                labels = self._classify_railways_with_buffer(
                    labels, point_geoms, gdf, asprs_class
                )
            else:
                # Standard polygon intersection
                for idx, row in gdf.iterrows():
                    polygon = row['geometry']
                    
                    if not isinstance(polygon, (Polygon, MultiPolygon)):
                        continue
                    
                    for i, point_geom in enumerate(point_geoms):
                        if polygon.contains(point_geom):
                            labels[i] = asprs_class
        
        # NDVI refinement for vegetation vs building confusion
        if ndvi is not None:
            # Buildings with high NDVI might be vegetation on roofs
            building_mask = (labels == self.ASPRS_BUILDING)
            high_ndvi_buildings = building_mask & (ndvi >= self.ndvi_veg_threshold)
            
            if np.any(high_ndvi_buildings):
                # Keep as building but log for review
                n_veg_on_building = np.sum(high_ndvi_buildings)
                logger.info(f"    Note: {n_veg_on_building} building points with high NDVI (roof vegetation?)")
        
        return labels
    
    def _classify_roads_with_buffer(
        self,
        labels: np.ndarray,
        point_geoms: List[Point],
        roads_gdf: 'gpd.GeoDataFrame',
        asprs_class: int
    ) -> np.ndarray:
        """
        Classify road points using intelligent buffering based on road width.
        
        The IGN BD TOPO® contains road centerlines with width attributes.
        We use the width to create appropriate buffers for each road.
        """
        logger.info(f"      Using intelligent road buffers (tolerance={self.road_buffer_tolerance}m)")
        
        # Track road statistics
        road_widths = []
        points_per_road = []
        
        for idx, row in roads_gdf.iterrows():
            # Get road polygon (already buffered by width/2 in fetch_roads_with_polygons)
            polygon = row['geometry']
            
            if not isinstance(polygon, (Polygon, MultiPolygon)):
                continue
            
            # Get road width for logging
            road_width = row.get('width_m', 'unknown')
            road_widths.append(road_width if road_width != 'unknown' else 0)
            
            # Apply additional tolerance buffer if specified
            if self.road_buffer_tolerance > 0:
                polygon = polygon.buffer(self.road_buffer_tolerance)
            
            # Classify points within this road polygon
            n_classified = 0
            for i, point_geom in enumerate(point_geoms):
                if polygon.contains(point_geom):
                    labels[i] = asprs_class
                    n_classified += 1
            
            points_per_road.append(n_classified)
        
        # Log statistics
        if road_widths:
            valid_widths = [w for w in road_widths if w > 0]
            if valid_widths:
                logger.info(f"      Road widths: {min(valid_widths):.1f}m - {max(valid_widths):.1f}m (avg: {np.mean(valid_widths):.1f}m)")
            
            total_road_points = sum(points_per_road)
            logger.info(f"      Classified {total_road_points:,} road points from {len(roads_gdf)} roads")
            logger.info(f"      Avg points per road: {np.mean(points_per_road):.0f}")
        
        return labels
    
    def _classify_railways_with_buffer(
        self,
        labels: np.ndarray,
        point_geoms: List[Point],
        railways_gdf: 'gpd.GeoDataFrame',
        asprs_class: int
    ) -> np.ndarray:
        """
        Classify railway points using intelligent buffering based on track width.
        
        The IGN BD TOPO® contains railway centerlines with width attributes.
        We use the width to create appropriate buffers for each railway.
        Similar to road buffering but typically narrower (default 3.5m for single track).
        """
        logger.info(f"      Using intelligent railway buffers (tolerance={self.road_buffer_tolerance}m)")
        
        # Track railway statistics
        railway_widths = []
        points_per_railway = []
        track_counts = []
        
        for idx, row in railways_gdf.iterrows():
            # Get railway polygon (already buffered by width/2 in fetch_railways_with_polygons)
            polygon = row['geometry']
            
            if not isinstance(polygon, (Polygon, MultiPolygon)):
                continue
            
            # Get railway width and track count for logging
            railway_width = row.get('width_m', 'unknown')
            n_tracks = row.get('nombre_voies', 1)
            railway_widths.append(railway_width if railway_width != 'unknown' else 0)
            track_counts.append(n_tracks)
            
            # Apply additional tolerance buffer if specified
            if self.road_buffer_tolerance > 0:
                polygon = polygon.buffer(self.road_buffer_tolerance)
            
            # Classify points within this railway polygon
            n_classified = 0
            for i, point_geom in enumerate(point_geoms):
                if polygon.contains(point_geom):
                    labels[i] = asprs_class
                    n_classified += 1
            
            points_per_railway.append(n_classified)
        
        # Log statistics
        if railway_widths:
            valid_widths = [w for w in railway_widths if w > 0]
            if valid_widths:
                logger.info(f"      Railway widths: {min(valid_widths):.1f}m - {max(valid_widths):.1f}m (avg: {np.mean(valid_widths):.1f}m)")
            
            total_railway_points = sum(points_per_railway)
            logger.info(f"      Classified {total_railway_points:,} railway points from {len(railways_gdf)} railways")
            logger.info(f"      Avg points per railway: {np.mean(points_per_railway):.0f}")
            
            if track_counts:
                unique_tracks = sorted(set(track_counts))
                logger.info(f"      Track counts: {unique_tracks} (single, double, etc.)")
        
        return labels
    
    def _log_distribution(self, labels: np.ndarray):
        """Log the final classification distribution."""
        unique, counts = np.unique(labels, return_counts=True)
        
        class_names = {
            self.ASPRS_UNCLASSIFIED: 'Unclassified',
            self.ASPRS_GROUND: 'Ground',
            self.ASPRS_LOW_VEGETATION: 'Low Vegetation',
            self.ASPRS_MEDIUM_VEGETATION: 'Medium Vegetation',
            self.ASPRS_HIGH_VEGETATION: 'High Vegetation',
            self.ASPRS_BUILDING: 'Building',
            self.ASPRS_LOW_POINT: 'Low Point',
            self.ASPRS_WATER: 'Water',
            self.ASPRS_RAIL: 'Rail',
            self.ASPRS_ROAD: 'Road',
            self.ASPRS_BRIDGE: 'Bridge',
            self.ASPRS_PARKING: 'Parking',
            self.ASPRS_SPORTS: 'Sports Facility',
            self.ASPRS_CEMETERY: 'Cemetery',
            self.ASPRS_POWER_LINE: 'Power Line',
            self.ASPRS_AGRICULTURE: 'Agriculture'
        }
        
        logger.info("📊 Final classification distribution:")
        total = len(labels)
        for label_val, count in zip(unique, counts):
            name = class_names.get(label_val, f'Unknown_{label_val}')
            percentage = 100 * count / total
            logger.info(f"  {name:20s}: {count:8,} ({percentage:5.1f}%)")


def classify_with_all_features(
    points: np.ndarray,
    ground_truth_fetcher=None,
    bd_foret_fetcher=None,
    rpg_fetcher=None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    ndvi: Optional[np.ndarray] = None,
    height: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    planarity: Optional[np.ndarray] = None,
    curvature: Optional[np.ndarray] = None,
    intensity: Optional[np.ndarray] = None,
    return_number: Optional[np.ndarray] = None,
    include_railways: bool = True,
    include_forest: bool = True,
    include_agriculture: bool = True,
    include_bridges: bool = False,
    include_parking: bool = False,
    include_sports: bool = False,
    **kwargs
) -> Tuple[np.ndarray, Optional[Dict], Optional[Dict]]:
    """
    Convenience function for classification with all features including railways and BD Forêt®.
    
    Args:
        points: Point coordinates [N, 3]
        ground_truth_fetcher: IGNGroundTruthFetcher instance
        bd_foret_fetcher: BDForetFetcher instance for forest type classification
        bbox: Bounding box for fetching ground truth
        ndvi: NDVI values
        height: Height above ground
        normals: Surface normals
        planarity: Planarity values
        curvature: Surface curvature
        intensity: LiDAR intensity
        return_number: Return number
        include_railways: Whether to include railway classification
        include_forest: Whether to include BD Forêt® forest type refinement
        **kwargs: Additional arguments for AdvancedClassifier
        
    Returns:
        Tuple of (classification labels [N], forest_attributes dict or None)
        
    Example:
        >>> labels, forest_attrs = classify_with_all_features(
        ...     points=points,
        ...     ground_truth_fetcher=gt_fetcher,
        ...     bd_foret_fetcher=forest_fetcher,
        ...     bbox=bbox,
        ...     ndvi=ndvi,
        ...     height=height,
        ...     include_railways=True,
        ...     include_forest=True
        ... )
    """
    # Fetch ground truth if fetcher provided
    ground_truth_features = None
    if ground_truth_fetcher and bbox:
        logger.info("Fetching ground truth from IGN BD TOPO®...")
        ground_truth_features = ground_truth_fetcher.fetch_all_features(
            bbox=bbox,
            include_buildings=True,
            include_roads=True,
            include_water=True,
            include_vegetation=True,
            include_railways=include_railways,
            include_bridges=include_bridges,
            include_parking=include_parking,
            include_sports=include_sports
        )
    
    # Create classifier
    classifier = AdvancedClassifier(
        use_ground_truth=ground_truth_features is not None,
        use_ndvi=ndvi is not None,
        use_geometric=(height is not None or normals is not None or planarity is not None),
        **kwargs
    )
    
    # Classify
    labels = classifier.classify_points(
        points=points,
        ground_truth_features=ground_truth_features,
        ndvi=ndvi,
        height=height,
        normals=normals,
        planarity=planarity,
        curvature=curvature,
        intensity=intensity,
        return_number=return_number
    )
    
    # Refine vegetation classification with BD Forêt® if available
    forest_attributes = None
    if include_forest and bd_foret_fetcher and bbox:
        logger.info("Refining vegetation classification with BD Forêt® V2...")
        try:
            # Fetch forest polygons
            forest_gdf = bd_foret_fetcher.fetch_forest_polygons(bbox)
            
            if forest_gdf is not None and len(forest_gdf) > 0:
                # Label vegetation points with forest types
                forest_attributes = bd_foret_fetcher.label_points_with_forest_type(
                    points=points,
                    labels=labels,
                    forest_gdf=forest_gdf
                )
                
                # Log forest type statistics
                if forest_attributes:
                    n_labeled = sum(1 for t in forest_attributes.get('forest_type', []) if t and t != 'unknown')
                    logger.info(f"  Labeled {n_labeled:,} vegetation points with forest types")
                    
                    # Count forest types
                    from collections import Counter
                    type_counts = Counter(forest_attributes.get('forest_type', []))
                    for ftype, count in type_counts.most_common(5):
                        if ftype and ftype != 'unknown':
                            logger.info(f"    {ftype}: {count:,} points")
            else:
                logger.info("  No forest data found in BD Forêt® for this area")
                
        except Exception as e:
            logger.warning(f"  Failed to fetch/apply BD Forêt® data: {e}")
    
    # Refine ground/vegetation with RPG agricultural parcels if available
    rpg_attributes = None
    if include_agriculture and rpg_fetcher and bbox:
        logger.info("Refining classification with RPG agricultural parcels...")
        try:
            # Fetch agricultural parcels
            parcels_gdf = rpg_fetcher.fetch_parcels(bbox)
            
            if parcels_gdf is not None and len(parcels_gdf) > 0:
                # Label ground/vegetation points with crop types
                rpg_attributes = rpg_fetcher.label_points_with_crops(
                    points=points,
                    labels=labels,
                    parcels_gdf=parcels_gdf
                )
                
                # Optionally update labels for agricultural areas
                if rpg_attributes:
                    is_agri = rpg_attributes.get('is_agricultural', [])
                    n_agri = sum(is_agri)
                    
                    if n_agri > 0:
                        logger.info(f"  Labeled {n_agri:,} points as agricultural")
                        
                        # Optionally change ground/low veg to AGRICULTURE code
                        for i, is_ag in enumerate(is_agri):
                            if is_ag and labels[i] in [2, 3]:  # Ground or low veg
                                labels[i] = AdvancedClassifier.ASPRS_AGRICULTURE
                        
                        # Count crop types
                        from collections import Counter
                        crop_cats = [c for c in rpg_attributes.get('crop_category', []) if c != 'unknown']
                        if crop_cats:
                            cat_counts = Counter(crop_cats)
                            logger.info(f"  Crop categories:")
                            for cat, count in cat_counts.most_common():
                                logger.info(f"    {cat}: {count:,} points")
            else:
                logger.info("  No RPG parcels found in this area")
                
        except Exception as e:
            logger.warning(f"  Failed to fetch/apply RPG data: {e}")
    
    return labels, forest_attributes, rpg_attributes
