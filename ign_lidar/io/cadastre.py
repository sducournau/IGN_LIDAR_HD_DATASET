"""
Cadastre (BD PARCELLAIRE) Fetcher

This module provides functionality to fetch cadastral parcel data from
the French cadastre (BD PARCELLAIRE) which contains:
- Cadastral parcel boundaries
- Parcel identifiers (section, numero)
- Commune information
- Parcel areas

The cadastre data is essential for grouping LiDAR points by land ownership
and administrative divisions.

Data source: https://data.geopf.fr/ (IGN Géoplateforme)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List, TYPE_CHECKING
import json
import requests
from urllib.parse import urlencode
from collections import defaultdict
import time

if TYPE_CHECKING:
    import geopandas as gpd
    import numpy as np

logger = logging.getLogger(__name__)

try:
    from shapely.geometry import Point, Polygon, MultiPolygon, box
    from shapely.strtree import STRtree
    import geopandas as gpd
    import pandas as pd
    import numpy as np
    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False
    logger.warning("shapely/geopandas not available - Cadastre fetching disabled")


# ============================================================================
# Cadastre Fetcher
# ============================================================================

class CadastreFetcher:
    """
    Fetch cadastral parcel data from BD PARCELLAIRE (French cadastre).
    
    The cadastre provides detailed parcel boundaries for land ownership
    and administrative purposes. Each parcel is uniquely identified by:
    - Commune code (INSEE)
    - Section (letter/number)
    - Numero (parcel number)
    
    Example:
        >>> fetcher = CadastreFetcher(cache_dir="cache/cadastre")
        >>> parcels = fetcher.fetch_parcels(bbox=(xmin, ymin, xmax, ymax))
        >>> # Group points by parcel
        >>> groups = fetcher.group_points_by_parcel(points, parcels)
    """
    
    # WFS endpoint for cadastre data
    WFS_URL = "https://data.geopf.fr/wfs"
    
    # Layer name for cadastral parcels
    PARCELS_LAYER = "CADASTRALPARCELS.PARCELLAIRE_EXPRESS:parcelle"
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        crs: str = "EPSG:2154"
    ):
        """
        Initialize Cadastre fetcher.
        
        Args:
            cache_dir: Directory for caching WFS responses
            crs: Coordinate reference system (default: Lambert 93)
        """
        if not HAS_SPATIAL:
            raise ImportError("geopandas and shapely required for Cadastre fetching")
        
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.crs = crs
        
        logger.info("Initialized Cadastre fetcher")
    
    def fetch_parcels(
        self,
        bbox: Tuple[float, float, float, float],
        max_features: int = 10000
    ) -> Optional['gpd.GeoDataFrame']:
        """
        Fetch cadastral parcels from BD PARCELLAIRE within bounding box.
        
        Args:
            bbox: Bounding box (xmin, ymin, xmax, ymax) in Lambert 93
            max_features: Maximum number of features to retrieve
            
        Returns:
            GeoDataFrame with parcel polygons and attributes, or None if error
            
        Attributes in returned GeoDataFrame:
            - id_parcelle: Unique parcel ID
            - numero: Parcel number
            - section: Section identifier
            - commune: Commune code (INSEE)
            - prefixe: Prefix code
            - contenance: Parcel area (m²)
            - geometry: Polygon geometry
        """
        logger.info(f"Fetching cadastral parcels in bbox {bbox}...")
        
        # Check cache
        if self.cache_dir:
            cache_file = self._get_cache_path(bbox)
            if cache_file.exists():
                try:
                    logger.info(f"  Loading from cache: {cache_file.name}")
                    return gpd.read_file(cache_file)
                except Exception as e:
                    logger.warning(f"  Failed to load cache: {e}")
        
        # Build WFS request
        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{self.crs}"
        
        params = {
            'service': 'WFS',
            'version': '2.0.0',
            'request': 'GetFeature',
            'typename': self.PARCELS_LAYER,
            'outputFormat': 'application/json',
            'srsname': self.crs,
            'bbox': bbox_str,
            'count': max_features
        }
        
        url = f"{self.WFS_URL}?{urlencode(params)}"
        
        # Retry logic with exponential backoff for rate limiting
        max_retries = 5
        base_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(f"  Retry attempt {attempt}/{max_retries-1}")
                else:
                    logger.info(f"  Requesting WFS: {self.PARCELS_LAYER}")
                
                response = requests.get(url, timeout=60)
                
                # Handle rate limiting (429 Too Many Requests)
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        # Exponential backoff: 2s, 4s, 8s, 16s, 32s
                        delay = base_delay * (2 ** attempt)
                        # Add jitter to avoid thundering herd
                        import random
                        delay = delay + random.uniform(0, delay * 0.1)
                        logger.warning(f"  ⚠️  Rate limited (429). Waiting {delay:.1f}s before retry...")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"  ❌ Rate limit exceeded after {max_retries} attempts")
                        return None
                
                response.raise_for_status()
                
                # Parse GeoJSON
                data = response.json()
                
                if 'features' not in data or len(data['features']) == 0:
                    logger.info(f"  No cadastral parcels found in this area")
                    return None
                
                # Convert to GeoDataFrame
                gdf = gpd.GeoDataFrame.from_features(data['features'], crs=self.crs)
                
                logger.info(f"  Retrieved {len(gdf)} cadastral parcels")
                
                # Log commune distribution
                if 'commune' in gdf.columns:
                    n_communes = gdf['commune'].nunique()
                    logger.info(f"  Spanning {n_communes} commune(s)")
                
                # Cache the result
                if self.cache_dir and cache_file:
                    try:
                        gdf.to_file(cache_file, driver='GeoJSON')
                        logger.info(f"  Cached to: {cache_file.name}")
                    except Exception as e:
                        logger.warning(f"  Failed to cache: {e}")
                
                return gdf
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"  ⚠️  Request failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"  WFS request failed after {max_retries} attempts: {e}")
                    return None
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"  ⚠️  Parse error: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"  Failed to parse cadastre data after {max_retries} attempts: {e}")
                    return None
        
        # Should never reach here, but just in case
        return None
    
    def group_points_by_parcel(
        self,
        points: 'np.ndarray',
        parcels_gdf: 'gpd.GeoDataFrame',
        labels: Optional['np.ndarray'] = None
    ) -> Dict[str, Dict]:
        """
        Group points by cadastral parcel.
        
        Creates a dictionary mapping parcel IDs to point indices and statistics.
        Useful for per-parcel analysis, segmentation, and processing.
        
        Args:
            points: Point coordinates [N, 3] (X, Y, Z)
            parcels_gdf: GeoDataFrame with cadastral parcels
            labels: Optional classification labels [N]
            
        Returns:
            Dictionary mapping parcel ID to:
            {
                'parcel_id': {
                    'indices': [array of point indices],
                    'n_points': int,
                    'commune': str,
                    'section': str,
                    'numero': str,
                    'area_m2': float,
                    'point_density': float (points/m²),
                    'bounds': (xmin, ymin, xmax, ymax),
                    'class_distribution': {class_code: count} (if labels provided)
                }
            }
        """
        if parcels_gdf is None or len(parcels_gdf) == 0:
            return {}
        
        n_points = len(points)
        logger.info(f"  Grouping {n_points:,} points by cadastral parcel...")
        
        # Initialize result dictionary
        parcel_groups = {}
        
        # Create spatial index for parcels
        logger.info(f"  Creating spatial index for {len(parcels_gdf)} parcels...")
        
        # Track point assignment
        point_assigned = np.zeros(n_points, dtype=bool)
        
        # OPTIMIZED: Use STRtree spatial indexing for O(log N) point-in-polygon queries
        # Performance gain: 10-100x faster than nested loops with .iterrows()
        try:
            # VECTORIZED: Filter valid parcels without iterrows()
            mask_valid_geom = parcels_gdf.geometry.apply(
                lambda g: isinstance(g, (Polygon, MultiPolygon))
            )
            valid_gdf = parcels_gdf[mask_valid_geom].copy()
            
            if len(valid_gdf) == 0:
                logger.warning("  No valid parcel geometries found")
                return parcel_info_list
            
            # VECTORIZED: Extract all metadata at once
            valid_gdf['parcel_id'] = valid_gdf.get('id_parcelle', pd.Series(dtype=str)).fillna(
                pd.Series([f'parcel_{i}' for i in range(len(valid_gdf))], index=valid_gdf.index)
            ).astype(str)
            valid_gdf['commune'] = valid_gdf.get('commune', 'unknown').fillna('unknown').astype(str)
            valid_gdf['section'] = valid_gdf.get('section', 'unknown').fillna('unknown').astype(str)
            valid_gdf['numero'] = valid_gdf.get('numero', 'unknown').fillna('unknown').astype(str)
            valid_gdf['area_m2'] = valid_gdf.get('contenance', 0.0).fillna(0.0).astype(float)
            
            # Extract geometries and metadata as lists (O(N) operation)
            valid_parcels = valid_gdf.geometry.tolist()
            parcel_metadata = valid_gdf[['parcel_id', 'commune', 'section', 'numero', 'area_m2']].to_dict('records')
            
            # Add bounds to metadata (vectorized)
            for i, geom in enumerate(valid_parcels):
                parcel_metadata[i]['bounds'] = geom.bounds
            
            if valid_parcels:
                # Build R-tree spatial index
                parcel_tree = STRtree(valid_parcels)
                
                # Query each point (O(log N) per query instead of O(N))
                for i in range(n_points):
                    if point_assigned[i]:
                        continue
                    
                    point = Point(points[i, 0], points[i, 1])
                    
                    # Query spatial index for parcels containing this point
                    candidate_parcels = parcel_tree.query(point, predicate='contains')
                    
                    if len(candidate_parcels) > 0:
                        # Point is inside at least one parcel (use first match)
                        parcel_geom = candidate_parcels[0]
                        parcel_idx = valid_parcels.index(parcel_geom)
                        metadata = parcel_metadata[parcel_idx]
                        
                        parcel_id = metadata['id']
                        
                        # Initialize parcel group if needed
                        if parcel_id not in parcel_groups:
                            parcel_groups[parcel_id] = {
                                'indices': [],
                                'n_points': 0,
                                'commune': metadata['commune'],
                                'section': metadata['section'],
                                'numero': metadata['numero'],
                                'area_m2': metadata['area_m2'],
                                'bounds': metadata['bounds'],
                                'class_distribution': {}
                            }
                        
                        # Add point to parcel
                        parcel_groups[parcel_id]['indices'].append(i)
                        parcel_groups[parcel_id]['n_points'] += 1
                        point_assigned[i] = True
                
                # Convert indices lists to numpy arrays and compute statistics
                for parcel_id in parcel_groups:
                    parcel_info = parcel_groups[parcel_id]
                    parcel_info['indices'] = np.array(parcel_info['indices'])
                    
                    # Calculate point density
                    area_m2 = parcel_info['area_m2']
                    n_points_parcel = parcel_info['n_points']
                    parcel_info['point_density'] = n_points_parcel / area_m2 if area_m2 > 0 else 0.0
                    
                    # Class distribution if labels provided
                    if labels is not None:
                        parcel_labels = labels[parcel_info['indices']]
                        unique, counts = np.unique(parcel_labels, return_counts=True)
                        parcel_info['class_distribution'] = {int(c): int(n) for c, n in zip(unique, counts)}
        
        except Exception as e:
            logger.warning(f"  STRtree optimization failed ({e}), falling back to bbox filtering")
            
            # FALLBACK: Vectorized bbox filtering (no iterrows())
            mask_valid_geom = parcels_gdf.geometry.apply(
                lambda g: isinstance(g, (Polygon, MultiPolygon))
            )
            valid_gdf = parcels_gdf[mask_valid_geom].copy()
            
            # Extract metadata vectorially
            valid_gdf['parcel_id'] = valid_gdf.get('id_parcelle', pd.Series(dtype=str)).fillna(
                pd.Series([f'parcel_{i}' for i in range(len(valid_gdf))], index=valid_gdf.index)
            ).astype(str)
            valid_gdf['commune'] = valid_gdf.get('commune', 'unknown').fillna('unknown').astype(str)
            valid_gdf['section'] = valid_gdf.get('section', 'unknown').fillna('unknown').astype(str)
            valid_gdf['numero'] = valid_gdf.get('numero', 'unknown').fillna('unknown').astype(str)
            valid_gdf['area_m2'] = valid_gdf.get('contenance', 0.0).fillna(0.0).astype(float)
            
            # Process each parcel (still need loop for contains check)
            for idx in valid_gdf.index:
                parcel_row = valid_gdf.loc[idx]
                parcel_geom = parcel_row['geometry']
                
                # Get parcel attributes (already extracted vectorially)
                parcel_id = parcel_row['parcel_id']
                commune = parcel_row['commune']
                section = parcel_row['section']
                numero = parcel_row['numero']
                area_m2 = parcel_row['area_m2']
                
                # Get parcel bounds for quick filtering
                bounds = parcel_geom.bounds
                
                # Find points in parcel bounds (quick filter)
                in_bounds = (
                    (points[:, 0] >= bounds[0]) & (points[:, 0] <= bounds[2]) &
                    (points[:, 1] >= bounds[1]) & (points[:, 1] <= bounds[3])
                )
                
                if not np.any(in_bounds):
                    continue
                
                # Get candidate points
                candidate_indices = np.where(in_bounds)[0]
                
                # Check which candidates are actually inside polygon
                parcel_point_indices = []
                for i in candidate_indices:
                    if point_assigned[i]:
                        continue
                    point = Point(points[i, 0], points[i, 1])
                    if parcel_geom.contains(point):
                        parcel_point_indices.append(i)
                        point_assigned[i] = True
                
                if len(parcel_point_indices) == 0:
                    continue
                
                # Calculate statistics
                n_points_parcel = len(parcel_point_indices)
                point_density = n_points_parcel / area_m2 if area_m2 > 0 else 0.0
                
                # Class distribution if labels provided
                class_dist = {}
                if labels is not None:
                    parcel_labels = labels[parcel_point_indices]
                    unique, counts = np.unique(parcel_labels, return_counts=True)
                    class_dist = {int(c): int(n) for c, n in zip(unique, counts)}
                
                # Store parcel info
                parcel_groups[parcel_id] = {
                    'indices': np.array(parcel_point_indices),
                    'n_points': n_points_parcel,
                    'commune': commune,
                    'section': section,
                    'numero': numero,
                    'area_m2': area_m2,
                    'point_density': point_density,
                    'bounds': bounds,
                    'class_distribution': class_dist
                }
        
        # Log statistics
        n_parcels = len(parcel_groups)
        n_assigned = point_assigned.sum()
        pct_assigned = 100.0 * n_assigned / n_points if n_points > 0 else 0.0
        
        logger.info(f"  Grouped into {n_parcels} parcels")
        logger.info(f"  Assigned {n_assigned:,} / {n_points:,} points ({pct_assigned:.1f}%)")
        
        if n_parcels > 0:
            # Compute density statistics
            densities = [p['point_density'] for p in parcel_groups.values()]
            logger.info(f"  Point density: {min(densities):.1f} - {max(densities):.1f} pts/m² (avg: {np.mean(densities):.1f})")
            
            # Points per parcel statistics
            points_per_parcel = [p['n_points'] for p in parcel_groups.values()]
            logger.info(f"  Points per parcel: {min(points_per_parcel)} - {max(points_per_parcel)} (avg: {int(np.mean(points_per_parcel))})")
        
        return parcel_groups
    
    def label_points_with_parcel_id(
        self,
        points: 'np.ndarray',
        parcels_gdf: 'gpd.GeoDataFrame'
    ) -> Optional[List[str]]:
        """
        Label each point with its cadastral parcel ID.
        
        Args:
            points: Point coordinates [N, 3] (X, Y, Z)
            parcels_gdf: GeoDataFrame with cadastral parcels
            
        Returns:
            List of parcel IDs for each point (or 'unassigned')
        """
        if parcels_gdf is None or len(parcels_gdf) == 0:
            return None
        
        n_points = len(points)
        logger.info(f"  Labeling {n_points:,} points with parcel IDs...")
        
        # Initialize result
        parcel_ids = ['unassigned'] * n_points
        
        # OPTIMIZED: Use STRtree spatial indexing for O(log N) lookups
        try:
            # VECTORIZED: Filter and extract data without iterrows()
            mask_valid_geom = parcels_gdf.geometry.apply(
                lambda g: isinstance(g, (Polygon, MultiPolygon))
            )
            valid_gdf = parcels_gdf[mask_valid_geom].copy()
            
            if len(valid_gdf) == 0:
                logger.warning("  No valid parcel geometries found")
                return parcel_ids
            
            # Extract parcel IDs vectorially
            valid_gdf['parcel_id'] = valid_gdf.get('id_parcelle', pd.Series(dtype=str)).fillna(
                pd.Series([f'parcel_{i}' for i in range(len(valid_gdf))], index=valid_gdf.index)
            ).astype(str)
            
            # Build spatial index
            valid_parcels = valid_gdf.geometry.tolist()
            parcel_id_map = valid_gdf['parcel_id'].tolist()
            
            if valid_parcels:
                parcel_tree = STRtree(valid_parcels)
                
                # Query each unassigned point
                for i in range(n_points):
                    if parcel_ids[i] != 'unassigned':
                        continue
                    
                    point = Point(points[i, 0], points[i, 1])
                    candidates = parcel_tree.query(point, predicate='contains')
                    
                    if len(candidates) > 0:
                        # Use first match
                        parcel_geom = candidates[0]
                        parcel_idx = valid_parcels.index(parcel_geom)
                        parcel_ids[i] = parcel_id_map[parcel_idx]
        
        except Exception as e:
            logger.warning(f"  STRtree optimization failed ({e}), falling back to bbox filtering")
            
            # FALLBACK: Vectorized bbox filtering
            mask_valid_geom = parcels_gdf.geometry.apply(
                lambda g: isinstance(g, (Polygon, MultiPolygon))
            )
            valid_gdf = parcels_gdf[mask_valid_geom].copy()
            
            # Extract parcel IDs vectorially
            valid_gdf['parcel_id'] = valid_gdf.get('id_parcelle', pd.Series(dtype=str)).fillna(
                pd.Series([f'parcel_{i}' for i in range(len(valid_gdf))], index=valid_gdf.index)
            ).astype(str)
            
            # Process each parcel
            for idx in valid_gdf.index:
                parcel_row = valid_gdf.loc[idx]
                parcel_geom = parcel_row['geometry']
                parcel_id = parcel_row['parcel_id']
                
                # Get parcel bounds for quick filtering
                bounds = parcel_geom.bounds
                
                # Find points in bounds
                in_bounds = (
                    (points[:, 0] >= bounds[0]) & (points[:, 0] <= bounds[2]) &
                    (points[:, 1] >= bounds[1]) & (points[:, 1] <= bounds[3])
                )
                
                if not np.any(in_bounds):
                    continue
                
                # Check actual containment
                for i in np.where(in_bounds)[0]:
                    if parcel_ids[i] != 'unassigned':
                        continue
                    point = Point(points[i, 0], points[i, 1])
                    if parcel_geom.contains(point):
                        parcel_ids[i] = parcel_id
        
        n_labeled = sum(1 for pid in parcel_ids if pid != 'unassigned')
        pct = 100.0 * n_labeled / n_points
        logger.info(f"  Labeled {n_labeled:,} / {n_points:,} points ({pct:.1f}%)")
        
        return parcel_ids
    
    def get_parcel_statistics(
        self,
        parcel_groups: Dict[str, Dict],
        labels: Optional['np.ndarray'] = None
    ) -> pd.DataFrame:
        """
        Generate statistics DataFrame for parcel groups.
        
        Args:
            parcel_groups: Dictionary from group_points_by_parcel()
            labels: Optional classification labels
            
        Returns:
            DataFrame with per-parcel statistics
        """
        if not parcel_groups:
            return pd.DataFrame()
        
        # Build statistics rows
        rows = []
        for parcel_id, info in parcel_groups.items():
            row = {
                'parcel_id': parcel_id,
                'commune': info['commune'],
                'section': info['section'],
                'numero': info['numero'],
                'area_m2': info['area_m2'],
                'n_points': info['n_points'],
                'point_density': info['point_density'],
                'xmin': info['bounds'][0],
                'ymin': info['bounds'][1],
                'xmax': info['bounds'][2],
                'ymax': info['bounds'][3]
            }
            
            # Add class distribution if available
            if info['class_distribution']:
                for class_code, count in info['class_distribution'].items():
                    row[f'class_{class_code}'] = count
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def _get_cache_path(self, bbox: Tuple[float, float, float, float]) -> Path:
        """Generate cache file path for a bbox."""
        bbox_str = f"{int(bbox[0])}_{int(bbox[1])}_{int(bbox[2])}_{int(bbox[3])}"
        return self.cache_dir / f"cadastre_{bbox_str}.geojson"


# ============================================================================
# Utility Functions
# ============================================================================

def export_parcel_groups_to_geojson(
    parcel_groups: Dict[str, Dict],
    parcels_gdf: 'gpd.GeoDataFrame',
    output_path: Path
):
    """
    Export parcel groups with statistics to GeoJSON.
    
    Args:
        parcel_groups: Dictionary from group_points_by_parcel()
        parcels_gdf: Original parcels GeoDataFrame
        output_path: Output GeoJSON file path
    """
    # Create new GeoDataFrame with statistics
    features = []
    
    for parcel_id, info in parcel_groups.items():
        # Find original parcel geometry
        parcel_row = parcels_gdf[parcels_gdf['id_parcelle'] == parcel_id]
        if len(parcel_row) == 0:
            continue
        
        geom = parcel_row.iloc[0]['geometry']
        
        # Add statistics
        properties = {
            'parcel_id': parcel_id,
            'commune': info['commune'],
            'section': info['section'],
            'numero': info['numero'],
            'area_m2': info['area_m2'],
            'n_points': info['n_points'],
            'point_density': info['point_density']
        }
        
        # Add class distribution
        for class_code, count in info['class_distribution'].items():
            properties[f'class_{class_code}'] = count
        
        features.append({
            'geometry': geom,
            'properties': properties
        })
    
    # Create GeoDataFrame
    result_gdf = gpd.GeoDataFrame.from_features(features, crs=parcels_gdf.crs)
    
    # Export
    result_gdf.to_file(output_path, driver='GeoJSON')
    logger.info(f"Exported {len(result_gdf)} parcel groups to {output_path}")
