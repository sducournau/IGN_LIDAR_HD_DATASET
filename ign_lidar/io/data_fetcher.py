"""
Geographic Data Fetcher

This module consolidates all IGN geographic data fetchers into a single
interface for ease of use and consistency.

Supported data sources:
- BD TOPO® V3: Buildings, roads, railways, water, vegetation, bridges, etc.
- BD Forêt® V2: Forest types and species
- RPG: Agricultural parcels and crop types
- BD PARCELLAIRE: Cadastral parcels

Author: Data Integration Team
Date: October 15, 2025
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    import geopandas as gpd
    import numpy as np

logger = logging.getLogger(__name__)

# Import individual fetchers
try:
    from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
    from ign_lidar.io.bd_foret import BDForetFetcher
    from ign_lidar.io.rpg import RPGFetcher
    from ign_lidar.io.cadastre import CadastreFetcher
    HAS_FETCHERS = True
except ImportError as e:
    logger.warning(f"Could not import all fetchers: {e}")
    HAS_FETCHERS = False


@dataclass
class DataFetchConfig:
    """
    Configuration for data fetching.
    
    Attributes:
        # BD TOPO® features
        include_buildings: Include building footprints
        include_roads: Include road polygons
        include_railways: Include railway tracks
        include_water: Include water surfaces
        include_vegetation: Include vegetation zones
        include_bridges: Include bridge structures
        include_parking: Include parking areas
        include_cemeteries: Include cemeteries
        include_power_lines: Include power line corridors
        include_sports: Include sports facilities
        
        # Forest data
        include_forest: Include BD Forêt® forest types
        
        # Agriculture data
        include_agriculture: Include RPG agricultural parcels
        rpg_year: Year for RPG data (2020-2023)
        
        # Cadastre data
        include_cadastre: Include cadastral parcels
        group_by_parcel: Group points by cadastral parcel
        
        # Buffer parameters
        road_width_fallback: Default road width (m)
        railway_width_fallback: Default railway width (m)
        power_line_buffer: Power line buffer width (m)
    """
    # BD TOPO® features
    include_buildings: bool = True
    include_roads: bool = True
    include_railways: bool = True
    include_water: bool = True
    include_vegetation: bool = True
    include_bridges: bool = False
    include_parking: bool = False
    include_cemeteries: bool = False
    include_power_lines: bool = False
    include_sports: bool = False
    
    # Forest data
    include_forest: bool = True
    
    # Agriculture data
    include_agriculture: bool = True
    rpg_year: int = 2023
    
    # Cadastre data
    include_cadastre: bool = True
    group_by_parcel: bool = True
    
    # Buffer parameters
    road_width_fallback: float = 4.0
    railway_width_fallback: float = 3.5
    power_line_buffer: float = 2.0


class DataFetcher:
    """
    Interface for fetching all IGN geographic data.
    
    This class consolidates all individual fetchers (BD TOPO®, BD Forêt®,
    RPG, Cadastre) into a single easy-to-use interface.
    
    Example:
        >>> config = DataFetchConfig(
        ...     include_forest=True,
        ...     include_agriculture=True,
        ...     include_cadastre=True
        ... )
        >>> fetcher = DataFetcher(
        ...     cache_dir="cache",
        ...     config=config
        ... )
        >>> data = fetcher.fetch_all(bbox=(xmin, ymin, xmax, ymax))
        >>> # Access different data sources
        >>> buildings = data['ground_truth']['buildings']
        >>> forest = data['forest']
        >>> parcels = data['cadastre']
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        config: Optional[DataFetchConfig] = None,
        crs: str = "EPSG:2154"
    ):
        """
        Initialize data fetcher.
        
        Args:
            cache_dir: Root cache directory (subdirectories created per source)
            config: Data fetch configuration
            crs: Coordinate reference system (default: Lambert 93)
        """
        if not HAS_FETCHERS:
            raise ImportError("Required fetcher modules not available")
        
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.config = config or DataFetchConfig()
        self.crs = crs
        
        # Initialize individual fetchers
        self._init_fetchers()
        
        logger.info("Initialized DataFetcher")
        logger.info(f"  Cache dir: {self.cache_dir}")
        logger.info(f"  CRS: {self.crs}")
    
    def _init_fetchers(self):
        """Initialize all individual fetchers."""
        # BD TOPO® Ground Truth
        gt_cache = self.cache_dir / "ground_truth" if self.cache_dir else None
        self.ground_truth_fetcher = IGNGroundTruthFetcher(
            cache_dir=gt_cache
        )
        
        # BD Forêt®
        if self.config.include_forest:
            forest_cache = self.cache_dir / "bd_foret" if self.cache_dir else None
            self.forest_fetcher = BDForetFetcher(
                cache_dir=forest_cache
            )
        else:
            self.forest_fetcher = None
        
        # RPG Agriculture
        if self.config.include_agriculture:
            rpg_cache = self.cache_dir / "rpg" if self.cache_dir else None
            self.rpg_fetcher = RPGFetcher(
                cache_dir=rpg_cache,
                year=self.config.rpg_year,
                crs=self.crs
            )
        else:
            self.rpg_fetcher = None
        
        # Cadastre
        if self.config.include_cadastre:
            cadastre_cache = self.cache_dir / "cadastre" if self.cache_dir else None
            self.cadastre_fetcher = CadastreFetcher(
                cache_dir=cadastre_cache,
                crs=self.crs
            )
        else:
            self.cadastre_fetcher = None
    
    def fetch_all(
        self,
        bbox: Tuple[float, float, float, float],
        use_cache: bool = True
    ) -> Dict[str, any]:
        """
        Fetch all configured geographic data sources.
        
        Args:
            bbox: Bounding box (xmin, ymin, xmax, ymax) in specified CRS
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary with all fetched data:
            {
                'ground_truth': {
                    'buildings': GeoDataFrame,
                    'roads': GeoDataFrame,
                    'railways': GeoDataFrame,
                    'water': GeoDataFrame,
                    'vegetation': GeoDataFrame,
                    'bridges': GeoDataFrame,
                    'parking': GeoDataFrame,
                    'cemeteries': GeoDataFrame,
                    'power_lines': GeoDataFrame,
                    'sports': GeoDataFrame
                },
                'forest': GeoDataFrame,
                'agriculture': GeoDataFrame,
                'cadastre': GeoDataFrame
            }
        """
        logger.info(f"Fetching all data for bbox {bbox}")
        
        result = {}
        
        # OPTIMIZATION: Fetch all data sources in parallel
        # This reduces total fetch time from sum(T_i) to max(T_i)
        # Expected speedup: 2-4× for typical use cases
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        fetch_tasks = []
        
        # 1. BD TOPO® Ground Truth (includes internal parallelization)
        def fetch_ground_truth():
            logger.info("📍 Fetching BD TOPO® ground truth...")
            return self.ground_truth_fetcher.fetch_all_features(
                bbox=bbox,
                use_cache=use_cache,
                include_buildings=self.config.include_buildings,
                include_roads=self.config.include_roads,
                include_railways=self.config.include_railways,
                include_water=self.config.include_water,
                include_vegetation=self.config.include_vegetation,
                include_bridges=self.config.include_bridges,
                include_parking=self.config.include_parking,
                include_cemeteries=self.config.include_cemeteries,
                include_power_lines=self.config.include_power_lines,
                include_sports=self.config.include_sports,
                road_width_fallback=self.config.road_width_fallback,
                railway_width_fallback=self.config.railway_width_fallback,
                power_line_buffer=self.config.power_line_buffer
            )
        fetch_tasks.append(('ground_truth', fetch_ground_truth))
        
        # 2. BD Forêt®
        if self.config.include_forest and self.forest_fetcher:
            def fetch_forest():
                logger.info("🌲 Fetching BD Forêt® forest data...")
                return self.forest_fetcher.fetch_forest_polygons(bbox)
            fetch_tasks.append(('forest', fetch_forest))
        
        # 3. RPG Agriculture
        if self.config.include_agriculture and self.rpg_fetcher:
            def fetch_agriculture():
                logger.info("🌾 Fetching RPG agricultural parcels...")
                return self.rpg_fetcher.fetch_parcels(bbox)
            fetch_tasks.append(('agriculture', fetch_agriculture))
        
        # 4. Cadastre
        if self.config.include_cadastre and self.cadastre_fetcher:
            def fetch_cadastre():
                logger.info("🗺️  Fetching cadastral parcels...")
                return self.cadastre_fetcher.fetch_parcels(bbox)
            fetch_tasks.append(('cadastre', fetch_cadastre))
        
        # Execute all fetches in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_name = {executor.submit(task): name for name, task in fetch_tasks}
            
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    result[name] = future.result()
                except Exception as e:
                    logger.error(f"Failed to fetch {name}: {e}")
                    result[name] = None
        
        # Ensure all keys exist (for sources that weren't requested)
        if 'ground_truth' not in result:
            result['ground_truth'] = {}
        if 'forest' not in result:
            result['forest'] = None
        if 'agriculture' not in result:
            result['agriculture'] = None
        if 'cadastre' not in result:
            result['cadastre'] = None
        
        # Log summary
        self._log_fetch_summary(result)
        
        return result
    
    def process_points(
        self,
        points: 'np.ndarray',
        bbox: Tuple[float, float, float, float],
        labels: Optional['np.ndarray'] = None,
        ndvi: Optional['np.ndarray'] = None,
        height: Optional['np.ndarray'] = None,
        use_cache: bool = True
    ) -> Dict[str, any]:
        """
        Fetch all data and process points with comprehensive labeling.
        
        Args:
            points: Point coordinates [N, 3] (X, Y, Z)
            bbox: Bounding box for fetching data
            labels: Optional existing classification labels [N]
            ndvi: Optional NDVI values [N]
            height: Optional height above ground [N]
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary with processed results:
            {
                'labels': Classification labels (if provided),
                'forest_attributes': Forest type attributes,
                'agriculture_attributes': Crop type attributes,
                'cadastre_groups': Points grouped by cadastral parcel,
                'cadastre_labels': Parcel ID for each point,
                'data_sources': Raw fetched data
            }
        """
        logger.info("Processing points with data fetcher")
        logger.info(f"  Points: {len(points):,}")
        
        # Fetch all data
        data = self.fetch_all(bbox=bbox, use_cache=use_cache)
        
        result = {
            'labels': labels,
            'forest_attributes': None,
            'agriculture_attributes': None,
            'cadastre_groups': None,
            'cadastre_labels': None,
            'data_sources': data
        }
        
        # Process forest data
        if data['forest'] is not None and self.forest_fetcher and labels is not None:
            logger.info("🌲 Labeling points with forest types...")
            result['forest_attributes'] = self.forest_fetcher.label_points_with_forest_type(
                points=points,
                labels=labels,
                forest_gdf=data['forest']
            )
        
        # Process agriculture data
        if data['agriculture'] is not None and self.rpg_fetcher and labels is not None:
            logger.info("🌾 Labeling points with crop types...")
            result['agriculture_attributes'] = self.rpg_fetcher.label_points_with_crops(
                points=points,
                labels=labels,
                parcels_gdf=data['agriculture']
            )
        
        # Process cadastre data
        if data['cadastre'] is not None and self.cadastre_fetcher:
            logger.info("🗺️  Grouping points by cadastral parcel...")
            
            # Group points by parcel
            if self.config.group_by_parcel:
                result['cadastre_groups'] = self.cadastre_fetcher.group_points_by_parcel(
                    points=points,
                    parcels_gdf=data['cadastre'],
                    labels=labels
                )
            
            # Label each point with parcel ID
            result['cadastre_labels'] = self.cadastre_fetcher.label_points_with_parcel_id(
                points=points,
                parcels_gdf=data['cadastre']
            )
        
        return result
    
    def _log_fetch_summary(self, data: Dict):
        """Log summary of fetched data."""
        logger.info("📊 Data fetch summary:")
        
        # Ground truth
        gt = data.get('ground_truth', {})
        if gt:
            logger.info(f"  BD TOPO®: {len(gt)} feature types")
            for ftype, gdf in gt.items():
                if gdf is not None:
                    logger.info(f"    {ftype}: {len(gdf)} features")
        
        # Forest
        if data.get('forest') is not None:
            logger.info(f"  BD Forêt®: {len(data['forest'])} forest formations")
        
        # Agriculture
        if data.get('agriculture') is not None:
            logger.info(f"  RPG: {len(data['agriculture'])} agricultural parcels")
        
        # Cadastre
        if data.get('cadastre') is not None:
            logger.info(f"  Cadastre: {len(data['cadastre'])} cadastral parcels")


# ============================================================================
# Convenience Functions
# ============================================================================

def create_default_fetcher(cache_dir: Optional[Path] = None) -> DataFetcher:
    """
    Create a DataFetcher with default configuration.
    
    Includes:
    - All BD TOPO® basic features (buildings, roads, railways, water, vegetation)
    - BD Forêt® forest types
    - RPG agricultural parcels
    - Cadastral parcels
    
    Args:
        cache_dir: Cache directory (default: None)
        
    Returns:
        DataFetcher instance
    """
    config = DataFetchConfig(
        include_buildings=True,
        include_roads=True,
        include_railways=True,
        include_water=True,
        include_vegetation=True,
        include_forest=True,
        include_agriculture=True,
        include_cadastre=True
    )
    
    return DataFetcher(cache_dir=cache_dir, config=config)


def create_full_fetcher(cache_dir: Optional[Path] = None) -> DataFetcher:
    """
    Create a DataFetcher with ALL features enabled.
    
    Includes everything: BD TOPO® extended, BD Forêt®, RPG, Cadastre.
    
    Args:
        cache_dir: Cache directory (default: None)
        
    Returns:
        DataFetcher instance
    """
    config = DataFetchConfig(
        # Enable ALL BD TOPO® features
        include_buildings=True,
        include_roads=True,
        include_railways=True,
        include_water=True,
        include_vegetation=True,
        include_bridges=True,
        include_parking=True,
        include_cemeteries=True,
        include_power_lines=True,
        include_sports=True,
        # Enable all other sources
        include_forest=True,
        include_agriculture=True,
        include_cadastre=True,
        group_by_parcel=True
    )
    
    return DataFetcher(cache_dir=cache_dir, config=config)


# Deprecated aliases for backward compatibility
UnifiedDataFetcher = DataFetcher  # Deprecated: use DataFetcher
