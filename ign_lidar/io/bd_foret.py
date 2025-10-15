"""
BD Forêt® V2 Integration - Precise Vegetation Classification

This module provides functionality to fetch detailed forest information from IGN's
BD Forêt® V2 database, including:
- Forest types (coniferous, deciduous, mixed)
- Tree species composition
- Forest density and structure
- Age classes

The data is used to refine vegetation classification with precise forest types.

BD Forêt® V2 provides:
- Detailed forest polygons with species composition
- Forest structure (futaie, taillis, etc.)
- Density information
- Age class estimates

Author: Forest Classification Enhancement
Date: October 15, 2025
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING
import json
import requests
from urllib.parse import urlencode
from dataclasses import dataclass
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import geopandas as gpd

logger = logging.getLogger(__name__)

try:
    from shapely.geometry import Point, Polygon, MultiPolygon
    import geopandas as gpd
    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False
    logger.warning("shapely/geopandas not available - BD Forêt® fetching disabled")


# ============================================================================
# IGN BD Forêt® Configuration
# ============================================================================

@dataclass
class BDForetConfig:
    """Configuration for IGN BD Forêt® V2 WFS service."""
    
    # BD Forêt® V2 WFS Service
    WFS_URL = "https://data.geopf.fr/wfs"
    
    # Layer names for BD Forêt® V2
    FOREST_LAYER = "BDFORET_V2:formation_vegetale"  # Forest formations
    
    # Default CRS
    CRS = "EPSG:2154"  # Lambert 93
    
    # Request parameters
    VERSION = "2.0.0"
    OUTPUT_FORMAT = "application/json"
    MAX_FEATURES = 5000


# ============================================================================
# Forest Type Mappings
# ============================================================================

class ForestType:
    """Forest type classification based on BD Forêt® attributes."""
    
    # Main forest types
    CONIFEROUS = "coniferous"  # Résineux
    DECIDUOUS = "deciduous"    # Feuillus
    MIXED = "mixed"            # Mélangé
    OPEN_FOREST = "open"       # Forêt ouverte
    CLOSED_FOREST = "closed"   # Forêt fermée
    YOUNG = "young"            # Jeune formation
    MATURE = "mature"          # Formation mature
    
    # Common species groups
    PINE = "pine"              # Pins
    FIR_SPRUCE = "fir_spruce"  # Sapins/Épicéas
    OAK = "oak"                # Chênes
    BEECH = "beech"            # Hêtre
    CHESTNUT = "chestnut"      # Châtaignier
    POPLAR = "poplar"          # Peuplier
    
    @staticmethod
    def get_asprs_code(forest_type: str, height: float = None) -> int:
        """
        Map forest type to ASPRS classification code.
        
        Args:
            forest_type: Forest type string
            height: Tree height in meters (for height-based classification)
            
        Returns:
            ASPRS code (3: low veg, 4: medium veg, 5: high veg)
        """
        # Use height if available
        if height is not None:
            if height < 0.5:
                return 3  # Low vegetation
            elif height < 2.0:
                return 4  # Medium vegetation
            else:
                return 5  # High vegetation
        
        # Otherwise use forest type
        young_types = [ForestType.YOUNG, ForestType.OPEN_FOREST]
        if forest_type in young_types:
            return 4  # Medium vegetation
        else:
            return 5  # High vegetation (mature forest)


# ============================================================================
# BD Forêt® Fetcher
# ============================================================================

class BDForetFetcher:
    """
    Fetch detailed forest information from IGN BD Forêt® V2.
    
    This class provides methods to:
    1. Fetch forest polygons with species composition
    2. Extract forest type (coniferous, deciduous, mixed)
    3. Get density and structure information
    4. Generate detailed vegetation labels for point clouds
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        config: Optional[BDForetConfig] = None
    ):
        """
        Initialize BD Forêt® fetcher.
        
        Args:
            cache_dir: Directory to cache fetched data
            config: WFS configuration (uses default if None)
        """
        if not HAS_SPATIAL:
            raise ImportError(
                "shapely and geopandas required for BD Forêt® fetching. "
                "Install with: pip install shapely geopandas"
            )
        
        self.cache_dir = cache_dir
        if self.cache_dir:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or BDForetConfig()
        self._cache: Dict[str, Any] = {}
    
    def fetch_forest_polygons(
        self,
        bbox: Tuple[float, float, float, float],
        use_cache: bool = True
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Fetch forest formations from BD Forêt® V2.
        
        Args:
            bbox: Bounding box (xmin, ymin, xmax, ymax) in Lambert 93
            use_cache: Whether to use cached data if available
            
        Returns:
            GeoDataFrame with forest polygons and attributes
        """
        cache_key = f"forest_{bbox}"
        
        if use_cache and cache_key in self._cache:
            logger.debug(f"Using cached forest data for {bbox}")
            return self._cache[cache_key]
        
        logger.info(f"Fetching forest data from BD Forêt® for bbox {bbox}")
        
        try:
            gdf = self._fetch_wfs_layer(bbox=bbox)
            
            if gdf is not None and len(gdf) > 0:
                # Process forest attributes
                gdf = self._process_forest_attributes(gdf)
                logger.info(f"Retrieved {len(gdf)} forest formations")
                self._cache[cache_key] = gdf
                return gdf
            else:
                logger.info("No forest formations found in this area")
                return None
            
        except Exception as e:
            logger.error(f"Failed to fetch forest data: {e}")
            return None
    
    def _fetch_wfs_layer(
        self,
        bbox: Tuple[float, float, float, float]
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Fetch WFS layer from BD Forêt®.
        
        Args:
            bbox: Bounding box (xmin, ymin, xmax, ymax)
            
        Returns:
            GeoDataFrame or None if request failed
        """
        # Build WFS GetFeature request
        params = {
            'SERVICE': 'WFS',
            'VERSION': self.config.VERSION,
            'REQUEST': 'GetFeature',
            'TYPENAME': self.config.FOREST_LAYER,
            'OUTPUTFORMAT': self.config.OUTPUT_FORMAT,
            'SRSNAME': self.config.CRS,
            'BBOX': f'{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{self.config.CRS}',
            'COUNT': self.config.MAX_FEATURES
        }
        
        try:
            response = requests.get(
                self.config.WFS_URL,
                params=params,
                timeout=60
            )
            response.raise_for_status()
            
            # Parse GeoJSON response
            geojson_data = response.json()
            
            if 'features' not in geojson_data or len(geojson_data['features']) == 0:
                return None
            
            # Convert to GeoDataFrame
            gdf = gpd.GeoDataFrame.from_features(
                geojson_data['features'],
                crs=self.config.CRS
            )
            
            return gdf
            
        except requests.exceptions.RequestException as e:
            logger.error(f"WFS request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to parse WFS response: {e}")
            return None
    
    def _process_forest_attributes(
        self,
        gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Process and enrich forest attributes from BD Forêt®.
        
        BD Forêt® V2 attributes include:
        - code_tfv: Forest type code
        - lib_tfv: Forest type label
        - essence_1, essence_2, essence_3: Species codes
        - taux_1, taux_2, taux_3: Species percentages
        - densite: Forest density
        - structure: Forest structure
        
        Args:
            gdf: Raw GeoDataFrame from WFS
            
        Returns:
            Enriched GeoDataFrame with processed attributes
        """
        # Classify forest type
        gdf['forest_type'] = gdf.apply(self._classify_forest_type, axis=1)
        
        # Extract dominant species
        gdf['dominant_species'] = gdf.apply(self._get_dominant_species, axis=1)
        
        # Extract density category
        if 'densite' in gdf.columns:
            gdf['density_category'] = gdf['densite'].apply(self._classify_density)
        else:
            gdf['density_category'] = 'unknown'
        
        # Extract forest height estimate
        gdf['estimated_height'] = gdf.apply(self._estimate_height, axis=1)
        
        logger.debug(f"Forest types: {gdf['forest_type'].value_counts().to_dict()}")
        
        return gdf
    
    def _classify_forest_type(self, row: pd.Series) -> str:
        """Classify forest type from BD Forêt® attributes."""
        # Check lib_tfv or code_tfv for forest type
        if 'lib_tfv' in row and row['lib_tfv']:
            label = str(row['lib_tfv']).lower()
            
            if 'résineux' in label or 'conifère' in label:
                return ForestType.CONIFEROUS
            elif 'feuillu' in label:
                return ForestType.DECIDUOUS
            elif 'mélangé' in label or 'mixte' in label:
                return ForestType.MIXED
            elif 'jeune' in label or 'taillis' in label:
                return ForestType.YOUNG
            elif 'mature' in label or 'futaie' in label:
                return ForestType.MATURE
        
        # Fallback: analyze species composition
        if 'essence_1' in row and row['essence_1']:
            essence = str(row['essence_1']).upper()
            
            # Coniferous species codes
            if essence in ['PIN', 'SAP', 'EPI', 'MEL', 'CED', 'DOU', 'IF']:
                return ForestType.CONIFEROUS
            # Deciduous species codes
            elif essence in ['CHE', 'HET', 'CHA', 'FRA', 'ERA', 'PEU', 'ALI', 'BOU']:
                return ForestType.DECIDUOUS
        
        return 'unknown'
    
    def _get_dominant_species(self, row: pd.Series) -> str:
        """Extract dominant tree species."""
        if 'essence_1' in row and row['essence_1']:
            return str(row['essence_1'])
        return 'unknown'
    
    def _classify_density(self, density_value) -> str:
        """Classify forest density category."""
        if density_value is None or density_value == '':
            return 'unknown'
        
        try:
            density_str = str(density_value).lower()
            if 'ouvert' in density_str or 'faible' in density_str:
                return 'open'
            elif 'fermé' in density_str or 'dense' in density_str:
                return 'closed'
            elif 'moyen' in density_str:
                return 'medium'
        except:
            pass
        
        return 'unknown'
    
    def _estimate_height(self, row: pd.Series) -> float:
        """
        Estimate tree height from forest type and structure.
        
        Returns:
            Estimated height in meters
        """
        # Default heights by forest type
        heights = {
            ForestType.CONIFEROUS: 15.0,
            ForestType.DECIDUOUS: 12.0,
            ForestType.MIXED: 13.0,
            ForestType.YOUNG: 5.0,
            ForestType.MATURE: 20.0,
            ForestType.OPEN_FOREST: 8.0,
            ForestType.CLOSED_FOREST: 18.0
        }
        
        forest_type = row.get('forest_type', 'unknown')
        return heights.get(forest_type, 10.0)  # Default 10m
    
    def label_points_with_forest_type(
        self,
        points: np.ndarray,
        forest_gdf: gpd.GeoDataFrame,
        existing_labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Label vegetation points with detailed forest types.
        
        Args:
            points: Point cloud [N, 3] with XYZ coordinates
            forest_gdf: GeoDataFrame with forest polygons from BD Forêt®
            existing_labels: Optional existing ASPRS labels [N]
            
        Returns:
            Tuple of:
            - Updated ASPRS labels [N]
            - Dictionary of forest attributes (forest_type, species, density)
        """
        n_points = len(points)
        
        # Initialize forest attributes
        forest_types = np.full(n_points, '', dtype='U20')
        species = np.full(n_points, '', dtype='U10')
        densities = np.full(n_points, '', dtype='U10')
        estimated_heights = np.zeros(n_points, dtype=np.float32)
        
        # Create Point geometries
        logger.info(f"Labeling {n_points:,} points with BD Forêt® data...")
        point_geoms = [Point(p[0], p[1]) for p in points]
        
        # Process each forest polygon
        n_labeled = 0
        for idx, row in forest_gdf.iterrows():
            polygon = row['geometry']
            
            if not isinstance(polygon, (Polygon, MultiPolygon)):
                continue
            
            # Find points within this forest polygon
            for i, point_geom in enumerate(point_geoms):
                if polygon.contains(point_geom):
                    forest_types[i] = row.get('forest_type', '')
                    species[i] = row.get('dominant_species', '')
                    densities[i] = row.get('density_category', '')
                    estimated_heights[i] = row.get('estimated_height', 0.0)
                    n_labeled += 1
        
        logger.info(f"  Labeled {n_labeled:,} points ({100*n_labeled/n_points:.1f}%) with forest data")
        
        # Update ASPRS labels based on forest type
        if existing_labels is not None:
            labels = existing_labels.copy()
        else:
            labels = np.zeros(n_points, dtype=np.uint8)
        
        # Refine vegetation classification with forest height
        vegetation_mask = (forest_types != '')
        for i in np.where(vegetation_mask)[0]:
            height = estimated_heights[i]
            labels[i] = ForestType.get_asprs_code(forest_types[i], height)
        
        # Return labels and attributes
        forest_attributes = {
            'forest_type': forest_types,
            'species': species,
            'density': densities,
            'estimated_height': estimated_heights
        }
        
        # Log statistics
        if n_labeled > 0:
            unique_types = np.unique(forest_types[forest_types != ''])
            logger.info(f"  Forest types found: {', '.join(unique_types)}")
        
        return labels, forest_attributes


# ============================================================================
# Convenience Functions
# ============================================================================

def fetch_and_label_forest(
    points: np.ndarray,
    bbox: Tuple[float, float, float, float],
    cache_dir: Optional[Path] = None,
    existing_labels: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Optional[gpd.GeoDataFrame]]:
    """
    Convenience function to fetch BD Forêt® data and label points.
    
    Args:
        points: Point cloud [N, 3]
        bbox: Bounding box (xmin, ymin, xmax, ymax)
        cache_dir: Cache directory
        existing_labels: Optional existing labels to refine
        
    Returns:
        Tuple of:
        - Updated labels [N]
        - Forest attributes dictionary
        - Forest GeoDataFrame (for visualization/analysis)
    """
    fetcher = BDForetFetcher(cache_dir=cache_dir)
    
    # Fetch forest data
    forest_gdf = fetcher.fetch_forest_polygons(bbox=bbox)
    
    if forest_gdf is None or len(forest_gdf) == 0:
        logger.warning("No BD Forêt® data available for this area")
        # Return unchanged
        if existing_labels is not None:
            return existing_labels, {}, None
        else:
            return np.zeros(len(points), dtype=np.uint8), {}, None
    
    # Label points
    labels, forest_attrs = fetcher.label_points_with_forest_type(
        points=points,
        forest_gdf=forest_gdf,
        existing_labels=existing_labels
    )
    
    return labels, forest_attrs, forest_gdf
