"""
IGN RGE ALTI® Fetcher - Digital Terrain Model (DTM/MNT) Integration

This module provides access to the IGN RGE ALTI® (Référentiel à Grande Échelle 
pour l'Altimétrie), France's national high-resolution digital terrain model.

RGE ALTI® provides:
- 1m resolution DTM for mainland France and DOM-TOM
- Derived from LiDAR and other elevation sources
- Regular grid format (ASC, GeoTIFF)
- Available via IGN Géoplateforme WMS (Web Map Service)

Data Sources (in priority order):
1. Cache - Locally cached GeoTIFF files from previous requests
2. Local files - Pre-downloaded DTM tiles
3. WMS - Live download from https://data.geopf.fr/wms-r/wms

Use cases:
1. Ground point augmentation: Add synthetic ground points from DTM
2. Height normalization: Compute height above ground using DTM reference
3. Terrain analysis: Slope, aspect, roughness from high-quality DTM
4. Classification improvement: Better ground/non-ground separation

Migration Note (October 2025):
- Old WCS service (wxs.ign.fr) is deprecated and non-functional
- New WMS service (data.geopf.fr) is used for online DTM fetching
- WMS caching ensures efficient repeated access

Author: MNT Integration Enhancement
Date: October 19, 2025
"""

import logging
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

try:
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    rasterio = None
    
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    requests = None


class RGEALTIFetcher:
    """
    Fetcher for IGN digital terrain model data (LiDAR HD MNT preferred, RGE ALTI fallback).
    
    Supports multiple data sources:
    1. WMS (Web Map Service) - Download from IGN Géoplateforme with caching
    2. Local files - Pre-downloaded GeoTIFF files
    3. Cache - Local cache of previously fetched tiles
    
    Default DTM: LiDAR HD MNT (1m resolution from LiDAR, best quality for LiDAR projects)
    Fallback: RGE ALTI (1m-5m resolution, broader coverage)
    """
    
    # IGN Géoplateforme WMS endpoint (replaces deprecated WCS)
    WMS_ENDPOINT = "https://data.geopf.fr/wms-r/wms"
    WMS_VERSION = "1.3.0"
    
    # Layer names for different DTM sources
    LAYER_LIDAR_HD_MNT = "IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.SHADOW"  # LiDAR HD MNT (preferred)
    LAYER_RGE_ALTI = "ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES"  # RGE ALTI (fallback)
    
    # Legacy WCS endpoint (no longer functional)
    WCS_ENDPOINT = "https://wxs.ign.fr/altimetrie/geoportail/r/wcs"  # DEPRECATED
    WCS_VERSION = "2.0.1"
    COVERAGE_ID = "ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES"
    
    # Grid resolution
    RESOLUTION_1M = 1.0  # 1 meter resolution
    RESOLUTION_5M = 5.0  # 5 meter resolution (fallback)
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        resolution: float = RESOLUTION_1M,
        use_wcs: bool = True,
        local_dtm_dir: Optional[str] = None,
        api_key: Optional[str] = None,
        prefer_lidar_hd: bool = True
    ):
        """
        Initialize DTM fetcher (LiDAR HD MNT preferred, RGE ALTI fallback).
        
        Args:
            cache_dir: Directory for caching downloaded tiles
            resolution: Grid resolution in meters (1.0 or 5.0)
            use_wcs: Enable WMS download (parameter name kept for compatibility)
            local_dtm_dir: Directory containing local DTM files
            api_key: Legacy parameter (no longer needed for WMS)
            prefer_lidar_hd: Use LiDAR HD MNT layer (default: True, best quality)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.resolution = resolution
        self.local_dtm_dir = Path(local_dtm_dir) if local_dtm_dir else None
        self.prefer_lidar_hd = prefer_lidar_hd
        
        # Select WMS layer based on preference
        # LiDAR HD MNT: 1m resolution, best quality, derived from LiDAR
        # RGE ALTI: broader coverage but may be lower quality
        if prefer_lidar_hd:
            self.wms_layer = self.LAYER_LIDAR_HD_MNT
            dtm_source = "LiDAR HD MNT (1m, best quality)"
        else:
            self.wms_layer = self.LAYER_RGE_ALTI
            dtm_source = "RGE ALTI (broader coverage)"
        
        # Use WMS instead of deprecated WCS (October 2025 migration)
        # WMS provides elevation data via GetMap requests with automatic caching
        self.use_wms = use_wcs and HAS_REQUESTS and HAS_RASTERIO
        self.use_wcs = False  # WCS is deprecated
        
        if not HAS_RASTERIO:
            logger.warning("rasterio not available - DTM fetching disabled")
            self.use_wms = False
            
        if not HAS_REQUESTS:
            logger.warning("requests not available - WMS fetching disabled")
            self.use_wms = False
            
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"DTM cache: {self.cache_dir}")
        
        if self.local_dtm_dir and self.local_dtm_dir.exists():
            n_files = len(list(self.local_dtm_dir.glob("*.tif")))
            logger.info(f"Local DTM directory: {self.local_dtm_dir} ({n_files} files)")
        
        logger.info(
            f"DTM Fetcher initialized: {dtm_source}, resolution={resolution}m, "
            f"WMS={'enabled' if self.use_wms else 'disabled'}, "
            f"local_dir={local_dtm_dir}"
        )
    
    def fetch_dtm_for_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        crs: str = "EPSG:2154"
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Fetch DTM (MNT) data for a bounding box.
        
        Args:
            bbox: Bounding box (minx, miny, maxx, maxy) in CRS coordinates
            crs: Coordinate reference system (default: Lambert-93 EPSG:2154)
            
        Returns:
            Tuple of (elevation_grid, metadata) or None if failed
            - elevation_grid: 2D array of elevation values [H, W]
            - metadata: Dict with 'transform', 'crs', 'resolution', 'bounds'
        """
        # Try cache first
        if self.cache_dir:
            cached = self._load_from_cache(bbox, crs)
            if cached is not None:
                logger.info("Loaded DTM from cache")
                return cached
        
        # Try local files
        if self.local_dtm_dir:
            local_dtm = self._load_from_local(bbox, crs)
            if local_dtm is not None:
                logger.info("Loaded DTM from local files")
                # Cache for future use
                if self.cache_dir:
                    self._save_to_cache(bbox, crs, local_dtm)
                return local_dtm
        
        # Try WMS download (replaces deprecated WCS)
        if self.use_wms:
            wms_dtm = self._fetch_from_wms(bbox, crs)
            if wms_dtm is not None:
                logger.info("Fetched DTM from IGN WMS")
                # Cache for future use
                if self.cache_dir:
                    self._save_to_cache(bbox, crs, wms_dtm)
                return wms_dtm
        
        logger.warning(f"Failed to fetch DTM for bbox {bbox}")
        return None
    
    def sample_elevation_at_points(
        self,
        points: np.ndarray,
        dtm_data: Optional[Tuple[np.ndarray, Dict[str, Any]]] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        crs: str = "EPSG:2154"
    ) -> Optional[np.ndarray]:
        """
        Sample DTM elevation at given point locations.
        
        Args:
            points: Point coordinates [N, 2] or [N, 3] (X, Y, [Z])
            dtm_data: Pre-loaded DTM data (grid, metadata) or None to auto-fetch
            bbox: Bounding box for fetching DTM if not provided
            crs: Coordinate reference system
            
        Returns:
            Elevation values [N] sampled from DTM, or None if failed
        """
        if not HAS_RASTERIO:
            logger.warning("rasterio not available - cannot sample DTM")
            return None
        
        # Fetch DTM if not provided
        if dtm_data is None:
            if bbox is None:
                # Compute bbox from points
                minx, miny = points[:, :2].min(axis=0)
                maxx, maxy = points[:, :2].max(axis=0)
                bbox = (minx, miny, maxx, maxy)
            
            dtm_data = self.fetch_dtm_for_bbox(bbox, crs)
            if dtm_data is None:
                return None
        
        grid, metadata = dtm_data
        transform = metadata['transform']
        
        # Convert point coordinates to pixel indices
        from rasterio.transform import rowcol
        
        rows, cols = rowcol(
            transform,
            points[:, 0],  # X coordinates
            points[:, 1],  # Y coordinates
            op=np.floor
        )
        
        # Clip to grid bounds and convert to integers
        rows = np.clip(rows, 0, grid.shape[0] - 1).astype(np.int32)
        cols = np.clip(cols, 0, grid.shape[1] - 1).astype(np.int32)
        
        # Sample elevations
        elevations = grid[rows, cols]
        
        # Handle nodata
        nodata = metadata.get('nodata', -9999.0)
        valid_mask = elevations != nodata
        
        # Interpolate nodata values if possible
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            logger.warning(f"{n_invalid} points outside DTM coverage or with nodata")
            # TODO: Implement nearest-neighbor interpolation for nodata values
        
        return elevations
    
    def compute_height_above_ground(
        self,
        points: np.ndarray,
        dtm_data: Optional[Tuple[np.ndarray, Dict[str, Any]]] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        crs: str = "EPSG:2154"
    ) -> Optional[np.ndarray]:
        """
        Compute height above ground using DTM reference.
        
        Args:
            points: Point coordinates [N, 3] (X, Y, Z)
            dtm_data: Pre-loaded DTM data or None to auto-fetch
            bbox: Bounding box for fetching DTM
            crs: Coordinate reference system
            
        Returns:
            Height above ground [N] or None if failed
        """
        if points.shape[1] < 3:
            logger.error("Points must have Z coordinate for height computation")
            return None
        
        # Sample ground elevation from DTM
        ground_elevation = self.sample_elevation_at_points(points, dtm_data, bbox, crs)
        if ground_elevation is None:
            return None
        
        # Compute height = Z - DTM_elevation
        height_above_ground = points[:, 2] - ground_elevation
        
        return height_above_ground
    
    def generate_ground_points(
        self,
        bbox: Tuple[float, float, float, float],
        spacing: float = 1.0,
        crs: str = "EPSG:2154"
    ) -> Optional[np.ndarray]:
        """
        Generate synthetic ground points from DTM at regular grid spacing.
        
        Args:
            bbox: Bounding box (minx, miny, maxx, maxy)
            spacing: Grid spacing in meters (default: 1m matching DTM resolution)
            crs: Coordinate reference system
            
        Returns:
            Ground points [N, 3] (X, Y, Z) or None if failed
        """
        # Fetch DTM
        dtm_data = self.fetch_dtm_for_bbox(bbox, crs)
        if dtm_data is None:
            return None
        
        grid, metadata = dtm_data
        transform = metadata['transform']
        nodata = metadata.get('nodata', -9999.0)
        
        # Create regular grid of XY coordinates
        minx, miny, maxx, maxy = bbox
        x_coords = np.arange(minx, maxx, spacing)
        y_coords = np.arange(miny, maxy, spacing)
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        # Flatten to point array
        xy_points = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Sample elevations from DTM
        z_values = self.sample_elevation_at_points(xy_points, dtm_data)
        if z_values is None:
            return None
        
        # Filter out nodata points
        valid_mask = z_values != nodata
        valid_points = np.column_stack([
            xy_points[valid_mask, 0],  # X
            xy_points[valid_mask, 1],  # Y
            z_values[valid_mask]        # Z (ground elevation)
        ])
        
        logger.info(f"Generated {len(valid_points):,} ground points from DTM")
        return valid_points
    
    # ========================================================================
    # Private methods for data loading
    # ========================================================================
    
    def _load_from_cache(
        self,
        bbox: Tuple[float, float, float, float],
        crs: str
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Load DTM from local cache."""
        if not self.cache_dir or not HAS_RASTERIO:
            return None
        
        # Generate cache filename from bbox
        cache_file = self._get_cache_filename(bbox, crs)
        cache_path = self.cache_dir / cache_file
        
        if not cache_path.exists():
            return None
        
        try:
            with rasterio.open(cache_path) as src:
                grid = src.read(1)  # Read first band
                metadata = {
                    'transform': src.transform,
                    'crs': src.crs,
                    'resolution': (src.res[0], src.res[1]),
                    'bounds': src.bounds,
                    'nodata': src.nodata
                }
            return grid, metadata
        except Exception as e:
            logger.warning(f"Failed to load cached DTM: {e}")
            return None
    
    def _save_to_cache(
        self,
        bbox: Tuple[float, float, float, float],
        crs: str,
        dtm_data: Tuple[np.ndarray, Dict[str, Any]]
    ):
        """Save DTM to local cache."""
        if not self.cache_dir or not HAS_RASTERIO:
            return
        
        cache_file = self._get_cache_filename(bbox, crs)
        cache_path = self.cache_dir / cache_file
        
        try:
            grid, metadata = dtm_data
            
            # Write to GeoTIFF
            with rasterio.open(
                cache_path,
                'w',
                driver='GTiff',
                height=grid.shape[0],
                width=grid.shape[1],
                count=1,
                dtype=grid.dtype,
                crs=metadata['crs'],
                transform=metadata['transform'],
                nodata=metadata.get('nodata', -9999.0),
                compress='lzw'
            ) as dst:
                dst.write(grid, 1)
            
            logger.debug(f"Cached DTM to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache DTM: {e}")
    
    def _load_from_local(
        self,
        bbox: Tuple[float, float, float, float],
        crs: str
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Load DTM from local directory."""
        if not self.local_dtm_dir or not HAS_RASTERIO:
            return None
        
        # Find matching GeoTIFF files
        tif_files = list(self.local_dtm_dir.glob("*.tif")) + list(self.local_dtm_dir.glob("*.tiff"))
        
        if not tif_files:
            return None
        
        # TODO: Implement spatial index to find overlapping files efficiently
        # For now, check each file
        for tif_file in tif_files:
            try:
                with rasterio.open(tif_file) as src:
                    # Check if bbox intersects file bounds
                    src_bounds = src.bounds
                    if self._bbox_intersects(bbox, src_bounds):
                        # Read subset
                        window = self._compute_window(src, bbox)
                        grid = src.read(1, window=window)
                        
                        # Compute transform for window
                        window_transform = src.window_transform(window)
                        
                        metadata = {
                            'transform': window_transform,
                            'crs': src.crs,
                            'resolution': (src.res[0], src.res[1]),
                            'bounds': bbox,
                            'nodata': src.nodata
                        }
                        return grid, metadata
            except Exception as e:
                logger.warning(f"Failed to read {tif_file}: {e}")
                continue
        
        return None
    
    def _fetch_from_wms(
        self,
        bbox: Tuple[float, float, float, float],
        crs: str
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Fetch DTM from IGN WMS service using GetMap request.
        
        Implements automatic fallback:
        1. Try LiDAR HD MNT (1m, best quality)
        2. If that fails, fall back to RGE ALTI (broader coverage)
        """
        if not self.use_wms or not HAS_REQUESTS or not HAS_RASTERIO:
            return None
        
        # Build WMS GetMap request for elevation data
        minx, miny, maxx, maxy = bbox
        
        # Calculate pixel dimensions based on resolution
        width = int((maxx - minx) / self.resolution)
        height = int((maxy - miny) / self.resolution)
        
        # Ensure reasonable image size (max 2048x2048)
        max_size = 2048
        if width > max_size or height > max_size:
            scale = max(width / max_size, height / max_size)
            width = int(width / scale)
            height = int(height / scale)
            logger.warning(f"Downsampling WMS request to {width}x{height} to stay within limits")
        
        # Define layers to try (primary + fallback)
        layers_to_try = []
        if self.prefer_lidar_hd:
            layers_to_try = [
                (self.LAYER_LIDAR_HD_MNT, "LiDAR HD MNT"),
                (self.LAYER_RGE_ALTI, "RGE ALTI (fallback)")
            ]
        else:
            layers_to_try = [
                (self.LAYER_RGE_ALTI, "RGE ALTI"),
                (self.LAYER_LIDAR_HD_MNT, "LiDAR HD MNT (fallback)")
            ]
        
        # Try each layer in order
        for layer_name, layer_desc in layers_to_try:
            params = {
                'SERVICE': 'WMS',
                'VERSION': self.WMS_VERSION,
                'REQUEST': 'GetMap',
                'LAYERS': layer_name,
                'STYLES': '',
                'FORMAT': 'image/geotiff',
                'BBOX': f'{minx},{miny},{maxx},{maxy}',
                'WIDTH': width,
                'HEIGHT': height,
                'CRS': crs
            }
            
            try:
                logger.info(f"Fetching DTM from IGN WMS ({layer_desc}): {bbox} ({width}x{height})")
                response = requests.get(self.WMS_ENDPOINT, params=params, timeout=60)
                response.raise_for_status()
                
                # Check if we got an error message instead of image
                content_type = response.headers.get('Content-Type', '')
                if 'xml' in content_type or 'text' in content_type:
                    error_text = response.text[:500]
                    logger.warning(f"WMS returned error for {layer_desc}: {error_text}")
                    continue  # Try next layer
                
                # Save to temporary file and read with rasterio
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                    tmp.write(response.content)
                    tmp_path = tmp.name
                
                try:
                    with rasterio.open(tmp_path) as src:
                        grid = src.read(1)
                        
                        # Create proper geotransform
                        from rasterio.transform import from_bounds as create_transform
                        transform = create_transform(minx, miny, maxx, maxy, width, height)
                        
                        metadata = {
                            'transform': transform,
                            'crs': crs,
                            'resolution': (self.resolution, self.resolution),
                            'bounds': bbox,
                            'nodata': src.nodata if src.nodata is not None else -9999.0,
                            'source': layer_desc  # Track which layer was used
                        }
                    logger.info(f"✅ Successfully fetched DTM using {layer_desc}")
                    return grid, metadata
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
                    
            except Exception as e:
                logger.warning(f"WMS fetch failed for {layer_desc}: {e}")
                # Continue to next layer instead of returning None immediately
                continue
        
        # All layers failed
        logger.error(f"Failed to fetch DTM from all WMS layers for bbox {bbox}")
        return None
    
    def _get_cache_filename(self, bbox: Tuple[float, float, float, float], crs: str) -> str:
        """Generate cache filename from bbox."""
        minx, miny, maxx, maxy = bbox
        crs_code = crs.split(':')[-1]
        return f"rge_alti_{crs_code}_{minx:.0f}_{miny:.0f}_{maxx:.0f}_{maxy:.0f}.tif"
    
    def _bbox_intersects(
        self,
        bbox1: Tuple[float, float, float, float],
        bbox2
    ) -> bool:
        """Check if two bounding boxes intersect."""
        minx1, miny1, maxx1, maxy1 = bbox1
        minx2, miny2, maxx2, maxy2 = bbox2.left, bbox2.bottom, bbox2.right, bbox2.top
        
        return not (maxx1 < minx2 or maxx2 < minx1 or maxy1 < miny2 or maxy2 < miny1)
    
    def _compute_window(self, src, bbox: Tuple[float, float, float, float]):
        """Compute rasterio Window for bbox subset."""
        from rasterio.windows import from_bounds
        minx, miny, maxx, maxy = bbox
        return from_bounds(minx, miny, maxx, maxy, src.transform)


# ============================================================================
# Convenience Functions
# ============================================================================

def augment_ground_with_rge_alti(
    points: np.ndarray,
    labels: np.ndarray,
    bbox: Tuple[float, float, float, float],
    fetcher: Optional[RGEALTIFetcher] = None,
    ground_class: int = 2,
    spacing: float = 2.0,
    crs: str = "EPSG:2154"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment ground points with synthetic points from RGE ALTI DTM.
    
    Args:
        points: Original point cloud [N, 3]
        labels: Original classifications [N]
        bbox: Bounding box for DTM fetching
        fetcher: RGEALTIFetcher instance or None to create default
        ground_class: ASPRS ground class code (default: 2)
        spacing: Grid spacing for synthetic points (meters)
        crs: Coordinate reference system
        
    Returns:
        Tuple of (augmented_points, augmented_labels)
    """
    if fetcher is None:
        fetcher = RGEALTIFetcher()
    
    # Generate synthetic ground points from DTM
    synthetic_ground = fetcher.generate_ground_points(bbox, spacing, crs)
    if synthetic_ground is None:
        logger.warning("Failed to generate synthetic ground points")
        return points, labels
    
    # Combine with original points
    augmented_points = np.vstack([points, synthetic_ground])
    synthetic_labels = np.full(len(synthetic_ground), ground_class, dtype=labels.dtype)
    augmented_labels = np.concatenate([labels, synthetic_labels])
    
    logger.info(f"Augmented with {len(synthetic_ground):,} synthetic ground points from RGE ALTI")
    return augmented_points, augmented_labels


__all__ = [
    'RGEALTIFetcher',
    'augment_ground_with_rge_alti'
]
