"""
WFS Ground Truth Fetcher for IGN Topo Data

This module provides functionality to fetch ground truth vector data from IGN's
WFS services for generating labeled training patches. It retrieves:
- Building footprints (emprise de bâtiment)
- Road polygons generated from centerlines + width (largeur)
- Other topographic features from BD TOPO®

The ground truth data is used to label point clouds and generate training datasets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union, TYPE_CHECKING
import json
import requests
from urllib.parse import urlencode
from dataclasses import dataclass
import numpy as np

if TYPE_CHECKING:
    import geopandas as gpd

logger = logging.getLogger(__name__)

try:
    from shapely.geometry import shape, Point, Polygon, LineString, MultiPolygon, box
    from shapely.ops import unary_union
    import geopandas as gpd
    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False
    logger.warning("shapely/geopandas not available - WFS ground truth fetching disabled")


# ============================================================================
# IGN WFS Service Configuration
# ============================================================================

@dataclass
class IGNWFSConfig:
    """Configuration for IGN WFS services."""
    
    # BD TOPO® WFS Service
    WFS_URL = "https://data.geopf.fr/wfs"
    
    # Layer names for BD TOPO® V3
    BUILDINGS_LAYER = "BDTOPO_V3:batiment"  # Building footprints
    ROADS_LAYER = "BDTOPO_V3:troncon_de_route"  # Road segments with width
    RAILWAYS_LAYER = "BDTOPO_V3:troncon_de_voie_ferree"  # Railway tracks
    WATER_LAYER = "BDTOPO_V3:surface_hydrographique"  # Water surfaces
    VEGETATION_LAYER = "BDTOPO_V3:zone_de_vegetation"  # Vegetation zones
    TERRAIN_LAYER = "BDTOPO_V3:terrain_de_sport"  # Sports grounds
    
    # Additional BD TOPO® layers
    BRIDGE_LAYER = "BDTOPO_V3:pont"  # Bridges
    PARKING_LAYER = "BDTOPO_V3:parking"  # Parking areas
    CEMETERY_LAYER = "BDTOPO_V3:cimetiere"  # Cemeteries
    POWER_LINE_LAYER = "BDTOPO_V3:ligne_electrique"  # Power lines
    CONSTRUCTION_LAYER = "BDTOPO_V3:construction_surfacique"  # Surface constructions
    RESERVOIR_LAYER = "BDTOPO_V3:reservoir"  # Water reservoirs/tanks
    
    # Default CRS
    CRS = "EPSG:2154"  # Lambert 93
    
    # Request parameters
    VERSION = "2.0.0"
    OUTPUT_FORMAT = "application/json"
    MAX_FEATURES = 10000


# ============================================================================
# Ground Truth Fetcher
# ============================================================================

class IGNGroundTruthFetcher:
    """
    Fetch ground truth vector data from IGN BD TOPO® WFS service.
    
    This class provides methods to:
    1. Fetch building footprints
    2. Fetch road segments and generate polygons using width attribute
    3. Fetch other topographic features
    4. Generate ground truth labels for point clouds
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        config: Optional[IGNWFSConfig] = None
    ):
        """
        Initialize ground truth fetcher.
        
        Args:
            cache_dir: Directory to cache fetched data
            config: WFS configuration (uses default if None)
        """
        if not HAS_SPATIAL:
            raise ImportError(
                "shapely and geopandas required for ground truth fetching. "
                "Install with: pip install shapely geopandas"
            )
        
        self.cache_dir = cache_dir
        if self.cache_dir:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or IGNWFSConfig()
        self._cache: Dict[str, Any] = {}
    
    def fetch_buildings(
        self,
        bbox: Tuple[float, float, float, float],
        use_cache: bool = True
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Fetch building footprints from BD TOPO®.
        
        Args:
            bbox: Bounding box (xmin, ymin, xmax, ymax) in Lambert 93
            use_cache: Whether to use cached data if available
            
        Returns:
            GeoDataFrame with building polygons and attributes
        """
        cache_key = f"buildings_{bbox}"
        
        if use_cache and cache_key in self._cache:
            logger.debug(f"Using cached buildings for {bbox}")
            return self._cache[cache_key]
        
        logger.info(f"Fetching buildings from WFS for bbox {bbox}")
        
        try:
            gdf = self._fetch_wfs_layer(
                layer_name=self.config.BUILDINGS_LAYER,
                bbox=bbox
            )
            
            if gdf is not None:
                # Add building-specific processing
                if 'hauteur' in gdf.columns:
                    gdf['height_m'] = gdf['hauteur']
                if 'nature' in gdf.columns:
                    gdf['building_type'] = gdf['nature']
                
                logger.info(f"Retrieved {len(gdf)} buildings")
                self._cache[cache_key] = gdf
            
            return gdf
            
        except Exception as e:
            logger.error(f"Failed to fetch buildings: {e}")
            return None
    
    def fetch_roads_with_polygons(
        self,
        bbox: Tuple[float, float, float, float],
        use_cache: bool = True,
        default_width: float = 4.0
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Fetch road segments and generate polygons using width attribute.
        
        The BD TOPO® roads layer contains centerlines with width attributes.
        This method generates road polygons by buffering the centerlines.
        
        Args:
            bbox: Bounding box (xmin, ymin, xmax, ymax) in Lambert 93
            use_cache: Whether to use cached data if available
            default_width: Default road width in meters if attribute missing
            
        Returns:
            GeoDataFrame with road polygons and attributes
        """
        cache_key = f"roads_{bbox}"
        
        if use_cache and cache_key in self._cache:
            logger.debug(f"Using cached roads for {bbox}")
            return self._cache[cache_key]
        
        logger.info(f"Fetching roads from WFS for bbox {bbox}")
        
        try:
            gdf = self._fetch_wfs_layer(
                layer_name=self.config.ROADS_LAYER,
                bbox=bbox
            )
            
            if gdf is None or len(gdf) == 0:
                return None
            
            # Generate road polygons from centerlines using width
            logger.info("Generating road polygons from centerlines...")
            
            road_polygons = []
            for idx, row in gdf.iterrows():
                geometry = row['geometry']
                
                # Get road width (largeur in meters)
                width = None
                if 'largeur' in gdf.columns and row['largeur'] is not None:
                    try:
                        width = float(row['largeur'])
                        # Check if valid finite number
                        if not np.isfinite(width) or width <= 0:
                            width = None
                    except (ValueError, TypeError):
                        width = None
                
                if width is None and 'largeur_de_chaussee' in gdf.columns and row['largeur_de_chaussee'] is not None:
                    try:
                        width = float(row['largeur_de_chaussee'])
                        if not np.isfinite(width) or width <= 0:
                            width = None
                    except (ValueError, TypeError):
                        width = None
                
                if width is None:
                    width = default_width
                
                # Buffer centerline by half width on each side
                buffer_distance = width / 2.0
                
                try:
                    if isinstance(geometry, LineString):
                        road_polygon = geometry.buffer(buffer_distance, cap_style=2)  # Flat cap
                        
                        road_polygons.append({
                            'geometry': road_polygon,
                            'width_m': width,
                            'nature': row.get('nature', 'unknown'),
                            'importance': row.get('importance', 'unknown'),
                            'road_type': row.get('nature', 'road'),
                            'original_geometry': geometry  # Keep centerline
                        })
                except Exception as e:
                    logger.warning(f"Failed to buffer road geometry: {e}")
                    continue
            
            if road_polygons:
                result_gdf = gpd.GeoDataFrame(road_polygons, crs=self.config.CRS)
                logger.info(f"Generated {len(result_gdf)} road polygons")
                self._cache[cache_key] = result_gdf
                return result_gdf
            else:
                logger.warning("No valid road polygons generated")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch/process roads: {e}")
            return None
    
    def fetch_railways_with_polygons(
        self,
        bbox: Tuple[float, float, float, float],
        use_cache: bool = True,
        default_width: float = 3.5
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Fetch railway tracks and generate polygons using width attribute.
        
        Similar to roads, railway tracks in BD TOPO® are represented as
        centerlines. This method generates railway polygons by buffering.
        
        Args:
            bbox: Bounding box (xmin, ymin, xmax, ymax) in Lambert 93
            use_cache: Whether to use cached data if available
            default_width: Default railway width in meters (default: 3.5m for single track)
            
        Returns:
            GeoDataFrame with railway polygons and attributes
        """
        cache_key = f"railways_{bbox}"
        
        if use_cache and cache_key in self._cache:
            logger.debug(f"Using cached railways for {bbox}")
            return self._cache[cache_key]
        
        logger.info(f"Fetching railways from WFS for bbox {bbox}")
        
        try:
            gdf = self._fetch_wfs_layer(
                layer_name=self.config.RAILWAYS_LAYER,
                bbox=bbox
            )
            
            if gdf is None or len(gdf) == 0:
                return None
            
            # Generate railway polygons from centerlines
            logger.info("Generating railway polygons from centerlines...")
            
            railway_polygons = []
            for idx, row in gdf.iterrows():
                geometry = row['geometry']
                
                # Get railway width (or use default)
                width = default_width
                
                # Check for number of tracks to estimate width
                if 'nombre_voies' in gdf.columns and row['nombre_voies'] is not None:
                    try:
                        n_tracks = int(row['nombre_voies'])
                        if n_tracks > 1:
                            width = default_width * n_tracks  # Multiple tracks
                    except (ValueError, TypeError):
                        pass
                
                # Buffer centerline by half width on each side
                buffer_distance = width / 2.0
                
                try:
                    if isinstance(geometry, LineString):
                        railway_polygon = geometry.buffer(buffer_distance, cap_style=2)  # Flat cap
                        
                        railway_polygons.append({
                            'geometry': railway_polygon,
                            'width_m': width,
                            'nature': row.get('nature', 'voie_ferree'),
                            'importance': row.get('importance', 'unknown'),
                            'n_tracks': row.get('nombre_voies', 1),
                            'electrified': row.get('electrifie', 'unknown'),
                            'railway_type': row.get('nature', 'railway'),
                            'original_geometry': geometry  # Keep centerline
                        })
                except Exception as e:
                    logger.warning(f"Failed to buffer railway geometry: {e}")
                    continue
            
            if railway_polygons:
                result_gdf = gpd.GeoDataFrame(railway_polygons, crs=self.config.CRS)
                logger.info(f"Generated {len(result_gdf)} railway polygons")
                self._cache[cache_key] = result_gdf
                return result_gdf
            else:
                logger.warning("No valid railway polygons generated")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch/process railways: {e}")
            return None
    
    def fetch_water_surfaces(
        self,
        bbox: Tuple[float, float, float, float],
        use_cache: bool = True
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Fetch water surface polygons from BD TOPO®.
        
        Args:
            bbox: Bounding box (xmin, ymin, xmax, ymax) in Lambert 93
            use_cache: Whether to use cached data if available
            
        Returns:
            GeoDataFrame with water polygons
        """
        cache_key = f"water_{bbox}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        logger.info(f"Fetching water surfaces from WFS for bbox {bbox}")
        
        try:
            gdf = self._fetch_wfs_layer(
                layer_name=self.config.WATER_LAYER,
                bbox=bbox
            )
            
            if gdf is not None:
                logger.info(f"Retrieved {len(gdf)} water surfaces")
                self._cache[cache_key] = gdf
            
            return gdf
            
        except Exception as e:
            logger.error(f"Failed to fetch water surfaces: {e}")
            return None
    
    def fetch_vegetation_zones(
        self,
        bbox: Tuple[float, float, float, float],
        use_cache: bool = True
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Fetch vegetation zones from BD TOPO®.
        
        Args:
            bbox: Bounding box (xmin, ymin, xmax, ymax) in Lambert 93
            use_cache: Whether to use cached data if available
            
        Returns:
            GeoDataFrame with vegetation polygons
        """
        cache_key = f"vegetation_{bbox}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        logger.info(f"Fetching vegetation zones from WFS for bbox {bbox}")
        
        try:
            gdf = self._fetch_wfs_layer(
                layer_name=self.config.VEGETATION_LAYER,
                bbox=bbox
            )
            
            if gdf is not None:
                logger.info(f"Retrieved {len(gdf)} vegetation zones")
                self._cache[cache_key] = gdf
            
            return gdf
            
        except Exception as e:
            logger.error(f"Failed to fetch vegetation zones: {e}")
            return None
    
    def fetch_bridges(
        self,
        bbox: Tuple[float, float, float, float],
        use_cache: bool = True
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Fetch bridge polygons from BD TOPO®.
        
        Bridges are important structural elements that may appear as elevated
        surfaces in LiDAR data.
        
        Args:
            bbox: Bounding box (xmin, ymin, xmax, ymax) in Lambert 93
            use_cache: Whether to use cached data if available
            
        Returns:
            GeoDataFrame with bridge polygons
        """
        cache_key = f"bridges_{bbox}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        logger.info(f"Fetching bridges from WFS for bbox {bbox}")
        
        try:
            gdf = self._fetch_wfs_layer(
                layer_name=self.config.BRIDGE_LAYER,
                bbox=bbox
            )
            
            if gdf is not None:
                logger.info(f"Retrieved {len(gdf)} bridges")
                self._cache[cache_key] = gdf
            
            return gdf
            
        except Exception as e:
            logger.error(f"Failed to fetch bridges: {e}")
            return None
    
    def fetch_parking(
        self,
        bbox: Tuple[float, float, float, float],
        use_cache: bool = True
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Fetch parking area polygons from BD TOPO®.
        
        Parking areas are typically flat, paved surfaces.
        
        Args:
            bbox: Bounding box (xmin, ymin, xmax, ymax) in Lambert 93
            use_cache: Whether to use cached data if available
            
        Returns:
            GeoDataFrame with parking polygons
        """
        cache_key = f"parking_{bbox}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        logger.info(f"Fetching parking areas from WFS for bbox {bbox}")
        
        try:
            gdf = self._fetch_wfs_layer(
                layer_name=self.config.PARKING_LAYER,
                bbox=bbox
            )
            
            if gdf is not None:
                logger.info(f"Retrieved {len(gdf)} parking areas")
                self._cache[cache_key] = gdf
            
            return gdf
            
        except Exception as e:
            logger.error(f"Failed to fetch parking areas: {e}")
            return None
    
    def fetch_cemeteries(
        self,
        bbox: Tuple[float, float, float, float],
        use_cache: bool = True
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Fetch cemetery polygons from BD TOPO®.
        
        Cemeteries are vegetated areas with monuments/headstones.
        
        Args:
            bbox: Bounding box (xmin, ymin, xmax, ymax) in Lambert 93
            use_cache: Whether to use cached data if available
            
        Returns:
            GeoDataFrame with cemetery polygons
        """
        cache_key = f"cemeteries_{bbox}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        logger.info(f"Fetching cemeteries from WFS for bbox {bbox}")
        
        try:
            gdf = self._fetch_wfs_layer(
                layer_name=self.config.CEMETERY_LAYER,
                bbox=bbox
            )
            
            if gdf is not None:
                logger.info(f"Retrieved {len(gdf)} cemeteries")
                self._cache[cache_key] = gdf
            
            return gdf
            
        except Exception as e:
            logger.error(f"Failed to fetch cemeteries: {e}")
            return None
    
    def fetch_power_lines(
        self,
        bbox: Tuple[float, float, float, float],
        use_cache: bool = True,
        buffer_width: float = 2.0
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Fetch power lines from BD TOPO® and convert to polygons with intelligent buffering.
        
        Power lines are linear features that we buffer to create corridors.
        Buffer width is intelligently determined based on voltage level:
        - High voltage (>63kV): 10-15m corridor
        - Medium voltage (1-63kV): 4-6m corridor  
        - Low voltage (<1kV): 2-3m corridor
        - Unknown: use default buffer_width
        
        Args:
            bbox: Bounding box (xmin, ymin, xmax, ymax) in Lambert 93
            use_cache: Whether to use cached data if available
            buffer_width: Default buffer width in meters (default: 2.0m)
            
        Returns:
            GeoDataFrame with power line corridor polygons
        """
        cache_key = f"power_lines_{bbox}_{buffer_width}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        logger.info(f"Fetching power lines from WFS for bbox {bbox}")
        
        try:
            gdf = self._fetch_wfs_layer(
                layer_name=self.config.POWER_LINE_LAYER,
                bbox=bbox
            )
            
            if gdf is None or len(gdf) == 0:
                logger.info("No power lines found in this area")
                return None
            
            logger.info(f"Retrieved {len(gdf)} power lines, applying intelligent buffering...")
            
            # Generate power line corridor polygons with intelligent buffering
            power_line_polygons = []
            buffer_stats = {'high': 0, 'medium': 0, 'low': 0, 'unknown': 0}
            
            for idx, row in gdf.iterrows():
                geometry = row['geometry']
                
                # Determine intelligent buffer width based on voltage/nature
                voltage_level = 'unknown'
                intelligent_buffer = buffer_width
                
                # Check for voltage attribute (tension in kV)
                if 'tension' in gdf.columns and row['tension'] is not None:
                    try:
                        voltage = float(row['tension'])
                        if voltage >= 63:  # High voltage (HTB)
                            voltage_level = 'high'
                            intelligent_buffer = 12.0  # 12m corridor for HV lines
                        elif voltage >= 1:  # Medium voltage (HTA)
                            voltage_level = 'medium'
                            intelligent_buffer = 5.0  # 5m corridor for MV lines
                        else:  # Low voltage (BT)
                            voltage_level = 'low'
                            intelligent_buffer = 2.5  # 2.5m corridor for LV lines
                    except (ValueError, TypeError):
                        pass
                
                # Check nature attribute if voltage not available
                if voltage_level == 'unknown' and 'nature' in gdf.columns:
                    nature = str(row.get('nature', '')).lower()
                    if 'haute tension' in nature or 'htb' in nature:
                        voltage_level = 'high'
                        intelligent_buffer = 12.0
                    elif 'moyenne tension' in nature or 'hta' in nature:
                        voltage_level = 'medium'
                        intelligent_buffer = 5.0
                    elif 'basse tension' in nature or 'bt' in nature:
                        voltage_level = 'low'
                        intelligent_buffer = 2.5
                
                buffer_stats[voltage_level] += 1
                
                # Buffer centerline to create corridor
                try:
                    if isinstance(geometry, LineString):
                        corridor_polygon = geometry.buffer(intelligent_buffer, cap_style=2)  # Flat cap
                        
                        power_line_polygons.append({
                            'geometry': corridor_polygon,
                            'buffer_width': intelligent_buffer,
                            'voltage_level': voltage_level,
                            'nature': row.get('nature', 'unknown'),
                            'tension': row.get('tension', None),
                            'power_line_type': row.get('nature', 'power_line'),
                            'original_geometry': geometry  # Keep centerline
                        })
                except Exception as e:
                    logger.warning(f"Failed to buffer power line geometry: {e}")
                    continue
            
            if power_line_polygons:
                result_gdf = gpd.GeoDataFrame(power_line_polygons, crs=self.config.CRS)
                logger.info(f"Generated {len(result_gdf)} power line corridors with intelligent buffering:")
                logger.info(f"  - High voltage (12m): {buffer_stats['high']} lines")
                logger.info(f"  - Medium voltage (5m): {buffer_stats['medium']} lines")
                logger.info(f"  - Low voltage (2.5m): {buffer_stats['low']} lines")
                logger.info(f"  - Unknown voltage ({buffer_width}m): {buffer_stats['unknown']} lines")
                self._cache[cache_key] = result_gdf
                return result_gdf
            else:
                logger.warning("No valid power line polygons generated")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch power lines: {e}")
            return None
    
    def fetch_sports_facilities(
        self,
        bbox: Tuple[float, float, float, float],
        use_cache: bool = True
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Fetch sports facility polygons from BD TOPO®.
        
        Sports facilities include stadiums, playing fields, courts, etc.
        
        Args:
            bbox: Bounding box (xmin, ymin, xmax, ymax) in Lambert 93
            use_cache: Whether to use cached data if available
            
        Returns:
            GeoDataFrame with sports facility polygons
        """
        cache_key = f"sports_{bbox}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        logger.info(f"Fetching sports facilities from WFS for bbox {bbox}")
        
        try:
            gdf = self._fetch_wfs_layer(
                layer_name=self.config.TERRAIN_LAYER,
                bbox=bbox
            )
            
            if gdf is not None:
                logger.info(f"Retrieved {len(gdf)} sports facilities")
                self._cache[cache_key] = gdf
            
            return gdf
            
        except Exception as e:
            logger.error(f"Failed to fetch sports facilities: {e}")
            return None
    
    def fetch_all_features(
        self,
        bbox: Tuple[float, float, float, float],
        use_cache: bool = True,
        include_roads: bool = True,
        include_railways: bool = True,
        include_buildings: bool = True,
        include_water: bool = True,
        include_vegetation: bool = True,
        include_bridges: bool = False,
        include_parking: bool = False,
        include_cemeteries: bool = False,
        include_power_lines: bool = False,
        include_sports: bool = False,
        road_width_fallback: float = 4.0,
        railway_width_fallback: float = 3.5,
        power_line_buffer: float = 2.0
    ) -> Dict[str, gpd.GeoDataFrame]:
        """
        Fetch all available ground truth features for a bounding box.
        
        Args:
            bbox: Bounding box (xmin, ymin, xmax, ymax) in Lambert 93
            use_cache: Whether to use cached data
            include_roads: Include road polygons
            include_railways: Include railway polygons
            include_buildings: Include building footprints
            include_water: Include water surfaces
            include_vegetation: Include vegetation zones
            include_bridges: Include bridge structures
            include_parking: Include parking areas
            include_cemeteries: Include cemetery zones
            include_power_lines: Include power line corridors
            include_sports: Include sports facilities
            road_width_fallback: Default road width in meters when 'largeur' field is missing or invalid
            railway_width_fallback: Default railway width in meters (single track)
            power_line_buffer: Buffer width for power lines in meters
            
        Returns:
            Dictionary mapping feature types to GeoDataFrames
        """
        features = {}
        
        if include_buildings:
            buildings = self.fetch_buildings(bbox, use_cache)
            if buildings is not None:
                features['buildings'] = buildings
        
        if include_roads:
            roads = self.fetch_roads_with_polygons(bbox, use_cache, default_width=road_width_fallback)
            if roads is not None:
                features['roads'] = roads
        
        if include_railways:
            railways = self.fetch_railways_with_polygons(bbox, use_cache, default_width=railway_width_fallback)
            if railways is not None:
                features['railways'] = railways
        
        if include_water:
            water = self.fetch_water_surfaces(bbox, use_cache)
            if water is not None:
                features['water'] = water
        
        if include_vegetation:
            vegetation = self.fetch_vegetation_zones(bbox, use_cache)
            if vegetation is not None:
                features['vegetation'] = vegetation
        
        if include_bridges:
            bridges = self.fetch_bridges(bbox, use_cache)
            if bridges is not None:
                features['bridges'] = bridges
        
        if include_parking:
            parking = self.fetch_parking(bbox, use_cache)
            if parking is not None:
                features['parking'] = parking
        
        if include_cemeteries:
            cemeteries = self.fetch_cemeteries(bbox, use_cache)
            if cemeteries is not None:
                features['cemeteries'] = cemeteries
        
        if include_power_lines:
            power_lines = self.fetch_power_lines(bbox, use_cache, buffer_width=power_line_buffer)
            if power_lines is not None:
                features['power_lines'] = power_lines
        
        if include_sports:
            sports = self.fetch_sports_facilities(bbox, use_cache)
            if sports is not None:
                features['sports'] = sports
        
        logger.info(f"Fetched {len(features)} feature types: {list(features.keys())}")
        return features
    
    def label_points_with_ground_truth(
        self,
        points: np.ndarray,
        ground_truth_features: Dict[str, gpd.GeoDataFrame],
        label_priority: Optional[List[str]] = None,
        ndvi: Optional[np.ndarray] = None,
        use_ndvi_refinement: bool = True,
        ndvi_vegetation_threshold: float = 0.3,
        ndvi_building_threshold: float = 0.15
    ) -> np.ndarray:
        """
        Label point cloud points based on ground truth vector data.
        
        Uses spatial intersection to assign labels. When multiple features
        overlap, uses priority order. Optionally uses NDVI to refine 
        building/vegetation classification.
        
        Args:
            points: Point cloud [N, 3] with XYZ coordinates in Lambert 93
            ground_truth_features: Dictionary of feature type -> GeoDataFrame
            label_priority: Priority order for overlapping features
                          (default: ['buildings', 'roads', 'water', 'vegetation'])
            ndvi: Optional NDVI values [N] for each point (range -1 to 1)
            use_ndvi_refinement: Use NDVI to refine building/vegetation labels
            ndvi_vegetation_threshold: NDVI threshold for vegetation (>= threshold)
            ndvi_building_threshold: NDVI threshold for buildings (<= threshold)
            
        Returns:
            Labels array [N] with feature type indices:
            - 0: unlabeled/ground
            - 1: buildings
            - 2: roads
            - 3: water
            - 4: vegetation
        """
        if label_priority is None:
            label_priority = ['buildings', 'roads', 'water', 'vegetation']
        
        # Initialize labels as 0 (unlabeled/ground)
        labels = np.zeros(len(points), dtype=np.int32)
        
        # Create shapely Points for all points
        point_geoms = [Point(p[0], p[1]) for p in points]
        
        # Label mapping
        label_map = {
            'buildings': 1,
            'roads': 2,
            'water': 3,
            'vegetation': 4
        }
        
        logger.info(f"Labeling {len(points)} points with ground truth data...")
        if ndvi is not None and use_ndvi_refinement:
            logger.info(f"  Using NDVI refinement (veg_threshold={ndvi_vegetation_threshold}, "
                       f"building_threshold={ndvi_building_threshold})")
        
        # Apply labels in reverse priority order (so higher priority overwrites)
        for feature_type in reversed(label_priority):
            if feature_type not in ground_truth_features:
                continue
            
            gdf = ground_truth_features[feature_type]
            if len(gdf) == 0:
                continue
            
            label_value = label_map.get(feature_type, 0)
            logger.debug(f"Processing {feature_type} ({len(gdf)} features)")
            
            # For each feature polygon, find intersecting points
            for idx, row in gdf.iterrows():
                polygon = row['geometry']
                
                if not isinstance(polygon, (Polygon, MultiPolygon)):
                    continue
                
                # Find points within this polygon
                for i, point_geom in enumerate(point_geoms):
                    if polygon.contains(point_geom):
                        labels[i] = label_value
        
        # NDVI-based refinement for building/vegetation confusion
        if ndvi is not None and use_ndvi_refinement:
            logger.info("  Applying NDVI-based refinement...")
            
            # Find points labeled as buildings or vegetation
            building_mask = (labels == label_map['buildings'])
            vegetation_mask = (labels == label_map['vegetation'])
            
            # Refine buildings: if NDVI is high (green), likely vegetation
            if np.any(building_mask):
                high_ndvi_buildings = building_mask & (ndvi >= ndvi_vegetation_threshold)
                n_reclassified_to_veg = np.sum(high_ndvi_buildings)
                if n_reclassified_to_veg > 0:
                    labels[high_ndvi_buildings] = label_map['vegetation']
                    logger.info(f"    Reclassified {n_reclassified_to_veg} high-NDVI building points → vegetation")
            
            # Refine vegetation: if NDVI is low (not green), likely building/road
            if np.any(vegetation_mask):
                low_ndvi_vegetation = vegetation_mask & (ndvi <= ndvi_building_threshold)
                n_reclassified_to_building = np.sum(low_ndvi_vegetation)
                if n_reclassified_to_building > 0:
                    labels[low_ndvi_vegetation] = label_map['buildings']
                    logger.info(f"    Reclassified {n_reclassified_to_building} low-NDVI vegetation points → building")
            
            # Additional refinement: ambiguous points (no ground truth label)
            # If unlabeled but high NDVI, likely vegetation
            unlabeled_mask = (labels == 0)
            high_ndvi_unlabeled = unlabeled_mask & (ndvi >= ndvi_vegetation_threshold)
            n_unlabeled_to_veg = np.sum(high_ndvi_unlabeled)
            if n_unlabeled_to_veg > 0:
                labels[high_ndvi_unlabeled] = label_map['vegetation']
                logger.info(f"    Labeled {n_unlabeled_to_veg} high-NDVI unlabeled points → vegetation")
        
        # Log label distribution
        unique, counts = np.unique(labels, return_counts=True)
        label_names = {0: 'unlabeled', 1: 'building', 2: 'road', 3: 'water', 4: 'vegetation'}
        logger.info("Label distribution:")
        for label_val, count in zip(unique, counts):
            label_name = label_names.get(label_val, f'unknown_{label_val}')
            percentage = 100 * count / len(labels)
            logger.info(f"  {label_name}: {count} ({percentage:.1f}%)")
        
        return labels
    
    def create_road_mask(
        self,
        points: np.ndarray,
        bbox: Tuple[float, float, float, float],
        buffer_tolerance: float = 0.5
    ) -> Optional[np.ndarray]:
        """
        Create a boolean mask for road points using ground truth road polygons.
        
        Args:
            points: Point cloud coordinates [N, 3] (X, Y, Z)
            bbox: Bounding box (xmin, ymin, xmax, ymax) in Lambert 93
            buffer_tolerance: Additional buffer around roads in meters
            
        Returns:
            Boolean array [N] where True = road point, or None if failed
        """
        try:
            # Fetch road polygons
            roads_gdf = self.fetch_roads_with_polygons(bbox=bbox)
            
            if roads_gdf is None or len(roads_gdf) == 0:
                logger.debug("No road polygons available for masking")
                return None
            
            # Initialize mask as False (not road)
            road_mask = np.zeros(len(points), dtype=bool)
            
            # Create Point geometries for all points
            logger.debug(f"Creating road mask for {len(points)} points...")
            point_geoms = [Point(p[0], p[1]) for p in points]
            
            # Optionally apply buffer for tolerance
            if buffer_tolerance > 0:
                roads_gdf = roads_gdf.copy()
                roads_gdf['geometry'] = roads_gdf['geometry'].buffer(buffer_tolerance)
            
            # Check each point against road polygons
            for idx, row in roads_gdf.iterrows():
                polygon = row['geometry']
                
                if not isinstance(polygon, (Polygon, MultiPolygon)):
                    continue
                
                # Find points within this road polygon
                for i, point_geom in enumerate(point_geoms):
                    if not road_mask[i]:  # Only check if not already marked as road
                        if polygon.contains(point_geom):
                            road_mask[i] = True
            
            n_road_points = road_mask.sum()
            pct = (n_road_points / len(points)) * 100 if len(points) > 0 else 0
            logger.info(f"  Road mask: {n_road_points:,} points ({pct:.1f}%) marked as roads")
            
            return road_mask
            
        except Exception as e:
            logger.error(f"Failed to create road mask: {e}")
            return None
    
    def _fetch_wfs_layer(
        self,
        layer_name: str,
        bbox: Tuple[float, float, float, float]
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Fetch a WFS layer for a bounding box.
        
        Args:
            layer_name: WFS layer name
            bbox: Bounding box (xmin, ymin, xmax, ymax)
            
        Returns:
            GeoDataFrame with features
        """
        # Build WFS GetFeature request
        params = {
            'SERVICE': 'WFS',
            'VERSION': self.config.VERSION,
            'REQUEST': 'GetFeature',
            'TYPENAME': layer_name,
            'OUTPUTFORMAT': self.config.OUTPUT_FORMAT,
            'SRSNAME': self.config.CRS,
            'BBOX': f'{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{self.config.CRS}',
            'COUNT': self.config.MAX_FEATURES
        }
        
        url = f"{self.config.WFS_URL}?{urlencode(params)}"
        
        try:
            logger.debug(f"WFS request: {layer_name}")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            # Parse GeoJSON response
            data = response.json()
            
            if 'features' not in data or len(data['features']) == 0:
                logger.warning(f"No features found for {layer_name} in bbox {bbox}")
                return None
            
            # Convert to GeoDataFrame
            gdf = gpd.GeoDataFrame.from_features(data['features'], crs=self.config.CRS)
            
            # Save to cache if enabled
            if self.cache_dir:
                cache_file = self.cache_dir / f"{layer_name.replace(':', '_')}_{hash(bbox)}.geojson"
                gdf.to_file(cache_file, driver='GeoJSON')
                logger.debug(f"Cached to {cache_file}")
            
            return gdf
            
        except requests.RequestException as e:
            logger.error(f"WFS request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to process WFS response: {e}")
            return None
    
    def save_ground_truth(
        self,
        features: Dict[str, gpd.GeoDataFrame],
        output_dir: Path,
        bbox: Tuple[float, float, float, float]
    ):
        """
        Save ground truth features to disk.
        
        Args:
            features: Dictionary of feature type -> GeoDataFrame
            output_dir: Output directory
            bbox: Bounding box for filename
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        bbox_str = f"{int(bbox[0])}_{int(bbox[1])}"
        
        for feature_type, gdf in features.items():
            output_file = output_dir / f"ground_truth_{feature_type}_{bbox_str}.geojson"
            gdf.to_file(output_file, driver='GeoJSON')
            logger.info(f"Saved {feature_type} ground truth to {output_file}")


# ============================================================================
# Convenience Functions
# ============================================================================

def fetch_ground_truth_for_tile(
    tile_bbox: Tuple[float, float, float, float],
    cache_dir: Optional[Path] = None,
    include_roads: bool = True,
    include_buildings: bool = True,
    include_water: bool = True,
    include_vegetation: bool = True
) -> Dict[str, gpd.GeoDataFrame]:
    """
    Fetch all ground truth features for a LiDAR tile.
    
    Args:
        tile_bbox: Tile bounding box in Lambert 93
        cache_dir: Cache directory for fetched data
        include_roads: Include road polygons
        include_buildings: Include building footprints
        include_water: Include water surfaces
        include_vegetation: Include vegetation zones
        
    Returns:
        Dictionary of feature type -> GeoDataFrame
    """
    fetcher = IGNGroundTruthFetcher(cache_dir=cache_dir)
    
    features = fetcher.fetch_all_features(
        bbox=tile_bbox,
        include_roads=include_roads,
        include_buildings=include_buildings,
        include_water=include_water,
        include_vegetation=include_vegetation
    )
    
    return features


def generate_patches_with_ground_truth(
    points: np.ndarray,
    features: Dict[str, np.ndarray],
    tile_bbox: Tuple[float, float, float, float],
    patch_size: float = 150.0,
    cache_dir: Optional[Path] = None,
    use_ndvi_refinement: bool = True,
    compute_ndvi_if_missing: bool = True
) -> List[Dict[str, np.ndarray]]:
    """
    Generate patches with ground truth labels from IGN BD TOPO®.
    
    This function:
    1. Fetches ground truth vector data (buildings, roads, etc.)
    2. Computes NDVI if RGB and NIR available
    3. Labels points based on spatial intersection + NDVI refinement
    4. Extracts patches from labeled point cloud
    
    Args:
        points: Point cloud [N, 3]
        features: Dictionary of point features (may include 'rgb', 'nir', 'ndvi')
        tile_bbox: Tile bounding box in Lambert 93
        patch_size: Patch size in meters
        cache_dir: Cache directory for ground truth data
        use_ndvi_refinement: Use NDVI to refine building/vegetation labels
        compute_ndvi_if_missing: Compute NDVI from RGB/NIR if not present
        
    Returns:
        List of patches with ground truth labels
    """
    # Import patch extraction
    from ..core.modules.patch_extractor import extract_patches
    
    # Fetch ground truth
    logger.info("Fetching ground truth data from IGN BD TOPO®...")
    fetcher = IGNGroundTruthFetcher(cache_dir=cache_dir)
    ground_truth_features = fetcher.fetch_all_features(bbox=tile_bbox)
    
    if not ground_truth_features:
        logger.warning("No ground truth features found for tile")
        return []
    
    # Check for NDVI or compute it
    ndvi = None
    if use_ndvi_refinement:
        if 'ndvi' in features:
            ndvi = features['ndvi']
            logger.info("Using existing NDVI values for refinement")
        elif compute_ndvi_if_missing and 'rgb' in features and 'nir' in features:
            logger.info("Computing NDVI from RGB and NIR for refinement...")
            try:
                from ..core.modules.enrichment import compute_ndvi
                rgb = features['rgb']
                nir = features['nir']
                ndvi = compute_ndvi(rgb, nir)
                features['ndvi'] = ndvi  # Add to features dict
                logger.info(f"  Computed NDVI (range: {ndvi.min():.3f} to {ndvi.max():.3f})")
            except ImportError:
                logger.warning("Cannot compute NDVI: enrichment module not available")
            except Exception as e:
                logger.warning(f"Failed to compute NDVI: {e}")
        else:
            logger.info("NDVI refinement requested but no NDVI/RGB/NIR data available")
    
    # Label points with ground truth (with optional NDVI refinement)
    logger.info("Labeling points with ground truth...")
    labels = fetcher.label_points_with_ground_truth(
        points=points,
        ground_truth_features=ground_truth_features,
        ndvi=ndvi,
        use_ndvi_refinement=use_ndvi_refinement and ndvi is not None
    )
    
    # Extract patches
    logger.info(f"Extracting patches (size={patch_size}m)...")
    patches = extract_patches(
        points=points,
        features=features,
        labels=labels,
        patch_size=patch_size,
        min_points=5000
    )
    
    logger.info(f"Generated {len(patches)} patches with ground truth labels")
    return patches


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'IGNWFSConfig',
    'IGNGroundTruthFetcher',
    'fetch_ground_truth_for_tile',
    'generate_patches_with_ground_truth',
]
