"""
Tile Loading and I/O Module

Handles LAZ file loading, point cloud filtering, coordinate transformations,
and data validation. Extracted from processor.py as part of Phase 3.4 refactoring.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import gc

import numpy as np
import laspy
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class TileLoader:
    """Handles tile loading and preprocessing operations."""
    
    def __init__(self, config: DictConfig):
        """
        Initialize tile loader with configuration.
        
        Args:
            config: Configuration object containing tile loading settings
        """
        self.config = config
        # bbox is at root level, not in processor section
        self.bbox = config.get('bbox')
        self.preprocess = config.get('processor', {}).get('preprocess', False)
        self.preprocess_config = config.get('preprocess')
        self.chunk_size_mb = config.get('processor', {}).get('chunk_size_mb', 500)
        
    def load_tile(
        self, 
        tile_path: Path,
        max_retries: int = 2
    ) -> Optional[Dict[str, Any]]:
        """
        Load a LAZ tile with automatic corruption recovery.
        
        Args:
            tile_path: Path to LAZ file
            max_retries: Maximum number of retry attempts for corrupted files
            
        Returns:
            Dictionary containing:
                - points: (N, 3) XYZ coordinates
                - intensity: (N,) intensity values [0-1]
                - return_number: (N,) return numbers
                - classification: (N,) classification codes
                - input_rgb: (N, 3) RGB values [0-1] if available
                - input_nir: (N,) NIR values [0-1] if available
                - input_ndvi: (N,) NDVI values [-1, 1] if available
                - enriched_features: Dict of enriched features if available
                - las: Original laspy object (for header info)
            Returns None if loading fails
        """
        logger.info(f"  📂 Loading tile: {tile_path.name}")
        
        # Check file size to determine loading strategy
        file_size_mb = tile_path.stat().st_size / (1024 * 1024)
        use_chunked = file_size_mb > self.chunk_size_mb
        
        if use_chunked:
            logger.info(f"  ⚠️  Large file ({file_size_mb:.1f}MB), using chunked loading...")
            return self._load_tile_chunked(tile_path, max_retries)
        else:
            return self._load_tile_standard(tile_path, max_retries)
    
    def _load_tile_standard(
        self, 
        tile_path: Path,
        max_retries: int
    ) -> Optional[Dict[str, Any]]:
        """Load tile using standard laspy.read (for files < chunk_size_mb)."""
        las = None
        
        for attempt in range(max_retries):
            try:
                las = laspy.read(str(tile_path))
                break  # Success
            except Exception as e:
                error_msg = str(e)
                is_corruption_error = (
                    'failed to fill whole buffer' in error_msg.lower() or
                    'ioerror' in error_msg.lower() or
                    'unexpected end of file' in error_msg.lower() or
                    'invalid' in error_msg.lower()
                )
                
                if is_corruption_error and attempt < max_retries - 1:
                    logger.warning(f"  ⚠️  Corrupted LAZ detected: {error_msg}")
                    logger.info(f"  🔄 Re-download attempt {attempt + 2}/{max_retries}...")
                    # Note: Actual re-download would be handled by caller
                    continue
                else:
                    logger.error(f"  ✗ Failed to read {tile_path.name}: {e}")
                    return None
        
        if las is None:
            logger.error(f"  ✗ Failed to load LAZ after {max_retries} attempts")
            return None
        
        # Extract basic data
        points = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
        intensity = np.array(las.intensity, dtype=np.float32) / 65535.0
        return_number = np.array(las.return_number, dtype=np.float32)
        classification = np.array(las.classification, dtype=np.uint8)
        
        # Extract RGB if present
        input_rgb = self._extract_rgb(las)
        if input_rgb is not None:
            logger.info(f"  🎨 RGB data found in LAZ (will be preserved)")
        
        # Extract NIR if present
        input_nir = self._extract_nir(las)
        if input_nir is not None:
            logger.info(f"  🌿 NIR data found in LAZ (will be preserved)")
        
        # Extract NDVI if present
        input_ndvi = self._extract_ndvi(las)
        if input_ndvi is not None:
            logger.info(f"  🌱 NDVI data found in LAZ (will be preserved)")
        
        # Extract enriched features if present
        enriched_features = self._extract_enriched_features(las)
        if enriched_features:
            logger.info(f"  ✨ Enriched features found: {list(enriched_features.keys())}")
        
        logger.info(f"  📊 Loaded {len(points):,} points | "
                   f"Classes: {len(np.unique(classification))}")
        
        return {
            'points': points,
            'intensity': intensity,
            'return_number': return_number,
            'classification': classification,
            'input_rgb': input_rgb,
            'input_nir': input_nir,
            'input_ndvi': input_ndvi,
            'enriched_features': enriched_features,
            'las': las
        }
    
    def _load_tile_chunked(
        self, 
        tile_path: Path,
        max_retries: int
    ) -> Optional[Dict[str, Any]]:
        """Load tile using memory-mapped chunked reading (for large files)."""
        for attempt in range(max_retries):
            try:
                with laspy.open(str(tile_path)) as laz_reader:
                    header = laz_reader.header
                    total_points = header.point_count
                    logger.info(f"  📊 {total_points:,} points - loading in chunks...")
                    
                    chunk_size = 10_000_000  # 10M points per chunk
                    num_chunks = (total_points + chunk_size - 1) // chunk_size
                    
                    # Collect chunks
                    all_points = []
                    all_intensity = []
                    all_return_number = []
                    all_classification = []
                    all_rgb = []
                    all_nir = []
                    
                    for i, chunk in enumerate(laz_reader.chunk_iterator(chunk_size)):
                        logger.info(f"    📦 Chunk {i+1}/{num_chunks}...")
                        
                        # Basic data
                        chunk_xyz = np.vstack([chunk.x, chunk.y, chunk.z]).T.astype(np.float32)
                        all_points.append(chunk_xyz)
                        all_intensity.append(np.array(chunk.intensity, dtype=np.float32) / 65535.0)
                        all_return_number.append(np.array(chunk.return_number, dtype=np.float32))
                        all_classification.append(np.array(chunk.classification, dtype=np.uint8))
                        
                        # RGB if available
                        chunk_rgb = self._extract_rgb(chunk)
                        if chunk_rgb is not None:
                            all_rgb.append(chunk_rgb)
                        
                        # NIR if available
                        chunk_nir = self._extract_nir(chunk)
                        if chunk_nir is not None:
                            all_nir.append(chunk_nir)
                        
                        # Clean up
                        del chunk, chunk_xyz
                        gc.collect()
                    
                    # Concatenate chunks
                    logger.info(f"  🔗 Concatenating {len(all_points)} chunks...")
                    points = np.vstack(all_points)
                    intensity = np.concatenate(all_intensity)
                    return_number = np.concatenate(all_return_number)
                    classification = np.concatenate(all_classification)
                    
                    input_rgb = np.vstack(all_rgb) if all_rgb else None
                    input_nir = np.concatenate(all_nir) if all_nir else None
                    
                    if input_rgb is not None:
                        logger.info(f"  🎨 RGB data loaded from chunks")
                    if input_nir is not None:
                        logger.info(f"  🌿 NIR data loaded from chunks")
                    
                    logger.info(f"  📊 Loaded {len(points):,} points total")
                    
                    # Store header info for later use (e.g., when saving enriched tiles)
                    return {
                        'points': points,
                        'intensity': intensity,
                        'return_number': return_number,
                        'classification': classification,
                        'input_rgb': input_rgb,
                        'input_nir': input_nir,
                        'input_ndvi': None,  # Not loaded in chunked mode
                        'enriched_features': {},
                        'las': None,  # No full LAS object in chunked mode
                        'header': header  # Store header for recreating LAZ files
                    }
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"  ⚠️  Chunked loading failed: {e}")
                    logger.info(f"  🔄 Retry {attempt + 2}/{max_retries}...")
                    continue
                else:
                    logger.error(f"  ✗ Failed to load chunked: {e}")
                    return None
        
        return None
    
    def _extract_rgb(self, las_obj) -> Optional[np.ndarray]:
        """Extract RGB data from LAS object if available."""
        if hasattr(las_obj, 'red') and hasattr(las_obj, 'green') and hasattr(las_obj, 'blue'):
            try:
                return np.vstack([
                    np.array(las_obj.red, dtype=np.float32) / 65535.0,
                    np.array(las_obj.green, dtype=np.float32) / 65535.0,
                    np.array(las_obj.blue, dtype=np.float32) / 65535.0
                ]).T
            except Exception as e:
                logger.debug(f"  Failed to extract RGB: {e}")
                return None
        return None
    
    def _extract_nir(self, las_obj) -> Optional[np.ndarray]:
        """Extract NIR data from LAS object if available."""
        if hasattr(las_obj, 'nir'):
            nir_data = np.array(las_obj.nir, dtype=np.float32)
            if nir_data.max() > 1.0:
                nir_data = nir_data / 65535.0
            return nir_data
        elif hasattr(las_obj, 'near_infrared'):
            nir_data = np.array(las_obj.near_infrared, dtype=np.float32)
            if nir_data.max() > 1.0:
                nir_data = nir_data / 65535.0
            return nir_data
        return None
    
    def _extract_ndvi(self, las_obj) -> Optional[np.ndarray]:
        """Extract NDVI data from LAS object if available."""
        if hasattr(las_obj, 'ndvi'):
            ndvi_data = np.array(las_obj.ndvi, dtype=np.float32)
            # NDVI should be [-1, 1], normalize if needed
            if ndvi_data.max() > 1.0:
                ndvi_data = ndvi_data / 65535.0 * 2.0 - 1.0
            return ndvi_data
        return None
    
    def _extract_enriched_features(self, las_obj) -> Dict[str, np.ndarray]:
        """Extract enriched features from LAS object if available."""
        enriched_features = {}
        
        feature_names = [
            'planarity', 'linearity', 'sphericity', 'anisotropy',
            'roughness', 'density', 'curvature', 'verticality',
            'height', 'z_normalized', 'z_from_ground', 'z_from_median'
        ]
        
        for feature_name in feature_names:
            if hasattr(las_obj, feature_name):
                enriched_features[feature_name] = np.array(
                    getattr(las_obj, feature_name), dtype=np.float32
                )
        
        # Extract normals if present
        if (hasattr(las_obj, 'normal_x') and 
            hasattr(las_obj, 'normal_y') and 
            hasattr(las_obj, 'normal_z')):
            enriched_features['normals'] = np.vstack([
                np.array(las_obj.normal_x, dtype=np.float32),
                np.array(las_obj.normal_y, dtype=np.float32),
                np.array(las_obj.normal_z, dtype=np.float32)
            ]).T
        
        return enriched_features
    
    def apply_bbox_filter(
        self, 
        tile_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply bounding box filter to tile data.
        
        Args:
            tile_data: Dictionary from load_tile()
            
        Returns:
            Filtered tile data dictionary
        """
        if self.bbox is None:
            return tile_data
        
        points = tile_data['points']
        xmin, ymin, xmax, ymax = self.bbox
        
        mask = (
            (points[:, 0] >= xmin) & (points[:, 0] <= xmax) &
            (points[:, 1] >= ymin) & (points[:, 1] <= ymax)
        )
        
        # Filter all arrays
        tile_data['points'] = points[mask]
        tile_data['intensity'] = tile_data['intensity'][mask]
        tile_data['return_number'] = tile_data['return_number'][mask]
        tile_data['classification'] = tile_data['classification'][mask]
        
        # Filter optional arrays
        if tile_data['input_rgb'] is not None:
            tile_data['input_rgb'] = tile_data['input_rgb'][mask]
        if tile_data['input_nir'] is not None:
            tile_data['input_nir'] = tile_data['input_nir'][mask]
        if tile_data['input_ndvi'] is not None:
            tile_data['input_ndvi'] = tile_data['input_ndvi'][mask]
        
        # Filter enriched features
        for key in list(tile_data['enriched_features'].keys()):
            tile_data['enriched_features'][key] = tile_data['enriched_features'][key][mask]
        
        logger.info(f"  📍 BBox filter: {len(tile_data['points']):,} points remaining")
        
        return tile_data
    
    def apply_preprocessing(
        self, 
        tile_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply preprocessing (outlier removal, downsampling) to tile data.
        
        Args:
            tile_data: Dictionary from load_tile()
            
        Returns:
            Preprocessed tile data dictionary
        """
        if not self.preprocess:
            return tile_data
        
        logger.info("  🧹 Preprocessing (artifact mitigation)...")
        
        from ...preprocessing.preprocessing import (
            statistical_outlier_removal,
            radius_outlier_removal,
            voxel_downsample
        )
        
        cfg = self.preprocess_config or {}
        sor_cfg = cfg.get('sor', {'enable': True})
        ror_cfg = cfg.get('ror', {'enable': True})
        voxel_cfg = cfg.get('voxel', {'enable': False})
        
        points = tile_data['points']
        original_count = len(points)
        cumulative_mask = np.ones(len(points), dtype=bool)
        
        # Statistical Outlier Removal
        if sor_cfg.get('enable', True):
            _, sor_mask = statistical_outlier_removal(
                points,
                k=sor_cfg.get('k', 12),
                std_multiplier=sor_cfg.get('std_multiplier', 2.0)
            )
            cumulative_mask &= sor_mask
        
        # Radius Outlier Removal
        if ror_cfg.get('enable', True):
            _, ror_mask = radius_outlier_removal(
                points,
                radius=ror_cfg.get('radius', 1.0),
                min_neighbors=ror_cfg.get('min_neighbors', 4)
            )
            cumulative_mask &= ror_mask
        
        # Apply cumulative filter
        tile_data['points'] = points[cumulative_mask]
        tile_data['intensity'] = tile_data['intensity'][cumulative_mask]
        tile_data['return_number'] = tile_data['return_number'][cumulative_mask]
        tile_data['classification'] = tile_data['classification'][cumulative_mask]
        
        if tile_data['input_rgb'] is not None:
            tile_data['input_rgb'] = tile_data['input_rgb'][cumulative_mask]
        if tile_data['input_nir'] is not None:
            tile_data['input_nir'] = tile_data['input_nir'][cumulative_mask]
        if tile_data['input_ndvi'] is not None:
            tile_data['input_ndvi'] = tile_data['input_ndvi'][cumulative_mask]
        
        for key in list(tile_data['enriched_features'].keys()):
            tile_data['enriched_features'][key] = tile_data['enriched_features'][key][cumulative_mask]
        
        # Voxel Downsampling
        if voxel_cfg.get('enable', False):
            points_filtered, voxel_indices = voxel_downsample(
                tile_data['points'],
                voxel_size=voxel_cfg.get('voxel_size', 0.5),
                method=voxel_cfg.get('method', 'centroid')
            )
            
            tile_data['points'] = points_filtered
            tile_data['intensity'] = tile_data['intensity'][voxel_indices]
            tile_data['return_number'] = tile_data['return_number'][voxel_indices]
            tile_data['classification'] = tile_data['classification'][voxel_indices]
            
            if tile_data['input_rgb'] is not None:
                tile_data['input_rgb'] = tile_data['input_rgb'][voxel_indices]
            if tile_data['input_nir'] is not None:
                tile_data['input_nir'] = tile_data['input_nir'][voxel_indices]
            if tile_data['input_ndvi'] is not None:
                tile_data['input_ndvi'] = tile_data['input_ndvi'][voxel_indices]
            
            for key in list(tile_data['enriched_features'].keys()):
                tile_data['enriched_features'][key] = tile_data['enriched_features'][key][voxel_indices]
        
        final_count = len(tile_data['points'])
        reduction = 1 - final_count / original_count
        
        logger.info(
            f"  ✓ Preprocessing: {final_count:,}/{original_count:,} "
            f"({reduction:.1%} reduction)"
        )
        
        return tile_data
    
    def validate_tile(self, tile_data: Dict[str, Any], min_points: int = 1000) -> bool:
        """
        Validate tile has sufficient points for processing.
        
        Args:
            tile_data: Dictionary from load_tile()
            min_points: Minimum number of points required
            
        Returns:
            True if tile is valid, False otherwise
        """
        if tile_data is None:
            return False
        
        num_points = len(tile_data['points'])
        
        if num_points < min_points:
            logger.warning(
                f"  ⚠️  Insufficient points: {num_points:,} < {min_points:,}"
            )
            return False
        
        return True
