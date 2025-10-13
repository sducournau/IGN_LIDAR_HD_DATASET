"""
Feature Computation Module

Handles geometric and spectral feature computation for point clouds.
Extracted from processor.py as part of Phase 3.4 refactoring.
"""

import logging
from typing import Dict, Optional, Any
import time

import numpy as np
from omegaconf import DictConfig

from ...features.factory import FeatureComputerFactory

logger = logging.getLogger(__name__)


class FeatureComputer:
    """Handles feature computation for point clouds."""
    
    def __init__(
        self, 
        config: DictConfig,
        feature_manager=None
    ):
        """
        Initialize feature computer with configuration.
        
        Args:
            config: Configuration object containing feature settings
            feature_manager: FeatureManager instance for feature tracking
        """
        self.config = config
        self.feature_manager = feature_manager
        
        # Extract feature settings
        processor_cfg = config.get('processor', {})
        features_cfg = config.get('features', {})
        
        self.use_gpu = processor_cfg.get('use_gpu', False)
        self.use_gpu_chunked = processor_cfg.get('use_gpu_chunked', False)
        self.k_neighbors = features_cfg.get('k_neighbors')
        self.feature_mode = features_cfg.get('mode', 'full')
        self.include_extra_features = processor_cfg.get('include_extra_features', True)
        self.include_rgb = processor_cfg.get('include_rgb', False)
        self.include_infrared = processor_cfg.get('include_infrared', False)
        self.compute_ndvi = processor_cfg.get('compute_ndvi', False)
        self.include_architectural_style = processor_cfg.get('include_architectural_style', False)
        self.style_encoding = processor_cfg.get('style_encoding', 'constant')
        
        # RGB fetcher (if needed)
        self.rgb_fetcher = processor_cfg.get('rgb_fetcher')
    
    def compute_features(
        self,
        tile_data: Dict[str, Any],
        use_enriched: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Compute all features for a point cloud.
        
        Args:
            tile_data: Dictionary containing:
                - points: (N, 3) XYZ coordinates
                - classification: (N,) classification codes
                - intensity: (N,) intensity values
                - return_number: (N,) return numbers
                - input_rgb: Optional (N, 3) RGB from input LAZ
                - input_nir: Optional (N,) NIR from input LAZ
                - input_ndvi: Optional (N,) NDVI from input LAZ
                - enriched_features: Optional dict of enriched features
            use_enriched: If True and enriched features exist, use them instead of recomputing
            
        Returns:
            Dictionary of all computed features:
                - normals: (N, 3)
                - curvature: (N,)
                - height: (N,)
                - intensity: (N,)
                - return_number: (N,)
                - [geometric features]: (N,) each
                - rgb: (N, 3) if requested
                - nir: (N,) if requested
                - ndvi: (N,) if requested
                - architectural_style: (N,) or (N, K) if requested
        """
        points = tile_data['points']
        classification = tile_data['classification']
        intensity = tile_data['intensity']
        return_number = tile_data['return_number']
        enriched_features = tile_data.get('enriched_features', {})
        
        all_features = {}
        
        # Check if we should use existing enriched features
        if use_enriched and enriched_features:
            logger.info(f"  ♻️  Using existing enriched features from input LAZ")
            
            normals = enriched_features.get('normals')
            curvature = enriched_features.get('curvature')
            # Use explicit None check to avoid numpy array truthiness issues
            height = enriched_features.get('height')
            if height is None:
                height = enriched_features.get('z_normalized')
            
            # Build geo_features from enriched
            geo_features = {
                k: v for k, v in enriched_features.items() 
                if k not in ['normals', 'curvature', 'height']
            }
        else:
            # Compute features
            normals, curvature, height, geo_features = self._compute_geometric_features(
                points, classification
            )
        
        # Add main features
        all_features['normals'] = normals
        all_features['curvature'] = curvature
        all_features['height'] = height
        all_features['intensity'] = intensity
        all_features['return_number'] = return_number
        
        # Add geometric features
        if isinstance(geo_features, dict):
            all_features.update(geo_features)
        
        # Add enriched features if present (alongside recomputed ones)
        if enriched_features:
            for feat_name, feat_data in enriched_features.items():
                # Use prefix to distinguish from recomputed features
                enriched_key = f"enriched_{feat_name}" if feat_name in all_features else feat_name
                all_features[enriched_key] = feat_data
            logger.info(f"  ✓ Added {len(enriched_features)} enriched features from input")
        
        # Add RGB features
        rgb_added = self._add_rgb_features(tile_data, all_features)
        
        # Add NIR features
        nir_added = self._add_nir_features(tile_data, all_features)
        
        # Add or compute NDVI
        self._add_ndvi_features(tile_data, all_features, rgb_added, nir_added)
        
        return all_features
    
    def _compute_geometric_features(
        self,
        points: np.ndarray,
        classification: np.ndarray
    ) -> tuple:
        """
        Compute geometric features using feature factory.
        
        Returns:
            Tuple of (normals, curvature, height, geo_features_dict)
        """
        feature_mode = "FULL" if self.include_extra_features else "CORE"
        k_display = self.k_neighbors if self.k_neighbors else "auto"
        
        logger.info(f"  🔧 Computing features | k={k_display} | mode={feature_mode}")
        
        feature_start = time.time()
        
        # Compute patch center for distance_to_center feature
        patch_center = (
            np.mean(points, axis=0) if self.include_extra_features else None
        )
        
        # Use manual k if specified, otherwise auto-estimate
        use_auto_k = self.k_neighbors is None
        k_value = self.k_neighbors if self.k_neighbors is not None else 20
        
        # Create feature computer using factory
        computer = FeatureComputerFactory.create(
            use_gpu=self.use_gpu,
            use_chunked=self.use_gpu_chunked,
            k_neighbors=k_value
        )
        
        # Compute features
        feature_dict = computer.compute_features(
            points=points,
            classification=classification,
            auto_k=use_auto_k,
            include_extra=self.include_extra_features,
            patch_center=patch_center,
            mode=self.feature_mode
        )
        
        # Extract main features
        normals = feature_dict.get('normals')
        curvature = feature_dict.get('curvature')
        height = feature_dict.get('height')
        
        # Extract geometric features
        main_features = {'normals', 'curvature', 'height'}
        geo_features = {
            k: v for k, v in feature_dict.items() if k not in main_features
        }
        
        # Debug logging
        logger.debug(
            f"🔍 [FEATURE_FLOW] Extracted {len(geo_features)} geometric features"
        )
        logger.debug(f"🔍 [FEATURE_FLOW] Feature names: {sorted(geo_features.keys())}")
        
        if len(geo_features) == 0:
            logger.warning(f"⚠️  [FEATURE_FLOW] geo_features is EMPTY after extraction!")
            logger.warning(f"⚠️  [FEATURE_FLOW] feature_dict keys: {list(feature_dict.keys())}")
        
        feature_time = time.time() - feature_start
        logger.info(f"  ⏱️  Features computed in {feature_time:.1f}s")
        
        return normals, curvature, height, geo_features
    
    def _add_rgb_features(
        self,
        tile_data: Dict[str, Any],
        all_features: Dict[str, np.ndarray]
    ) -> bool:
        """
        Add RGB features to feature dictionary.
        
        Returns:
            True if RGB was added, False otherwise
        """
        if not self.include_rgb:
            return False
        
        points = tile_data['points']
        input_rgb = tile_data.get('input_rgb')
        
        # Priority 1: Use RGB from input LAZ
        if input_rgb is not None:
            all_features['rgb'] = input_rgb
            logger.info(f"  ✅ Using RGB from input LAZ ({len(input_rgb):,} points)")
            return True
        
        # Priority 2: Fetch from IGN orthophotos
        if self.rgb_fetcher:
            logger.info("  🎨 Fetching RGB from IGN orthophotos...")
            rgb_start = time.time()
            
            tile_bbox = (
                points[:, 0].min(),
                points[:, 1].min(),
                points[:, 0].max(),
                points[:, 1].max()
            )
            
            try:
                rgb_tile = self.rgb_fetcher.augment_points_with_rgb(
                    points,
                    bbox=tile_bbox
                )
                all_features['rgb'] = rgb_tile.astype(np.float32) / 255.0
                
                rgb_time = time.time() - rgb_start
                logger.info(f"  ✓ RGB augmentation completed in {rgb_time:.2f}s")
                return True
            except Exception as e:
                logger.warning(f"  ⚠️  RGB augmentation failed: {e}")
                # Add default gray color
                all_features['rgb'] = np.full((len(points), 3), 0.5, dtype=np.float32)
                return True
        
        logger.warning("  ⚠️  RGB requested but no source available")
        return False
    
    def _add_nir_features(
        self,
        tile_data: Dict[str, Any],
        all_features: Dict[str, np.ndarray]
    ) -> bool:
        """
        Add NIR features to feature dictionary.
        
        Returns:
            True if NIR was added, False otherwise
        """
        if not self.include_infrared:
            return False
        
        input_nir = tile_data.get('input_nir')
        
        if input_nir is not None:
            all_features['nir'] = input_nir
            logger.info(f"  ✅ Using NIR from input LAZ ({len(input_nir):,} points)")
            return True
        
        logger.warning("  ⚠️  NIR requested but not available in input LAZ")
        return False
    
    def _add_ndvi_features(
        self,
        tile_data: Dict[str, Any],
        all_features: Dict[str, np.ndarray],
        rgb_available: bool,
        nir_available: bool
    ):
        """Add NDVI features to feature dictionary."""
        if not self.compute_ndvi:
            return
        
        input_ndvi = tile_data.get('input_ndvi')
        
        # Priority 1: Use NDVI from input LAZ
        if input_ndvi is not None:
            all_features['ndvi'] = input_ndvi
            logger.info(f"  ✅ Using NDVI from input LAZ ({len(input_ndvi):,} points)")
            return
        
        # Priority 2: Compute from RGB and NIR
        if rgb_available and nir_available:
            logger.info("  🌱 Computing NDVI from RGB and NIR...")
            
            try:
                rgb = all_features['rgb']
                nir = all_features['nir']
                
                # NDVI = (NIR - Red) / (NIR + Red)
                red = rgb[:, 0]
                ndvi = (nir - red) / (nir + red + 1e-8)
                all_features['ndvi'] = ndvi
                
                logger.info("  ✓ NDVI computed successfully")
            except Exception as e:
                logger.warning(f"  ⚠️  NDVI computation failed: {e}")
        else:
            logger.warning("  ⚠️  NDVI requested but cannot compute (missing RGB or NIR)")
    
    def add_architectural_style(
        self,
        all_features: Dict[str, np.ndarray],
        tile_metadata: Optional[Dict] = None
    ):
        """
        Add architectural style features.
        
        Args:
            all_features: Feature dictionary to update
            tile_metadata: Tile metadata containing style information
        """
        if not self.include_architectural_style:
            return
        
        from ...features.architectural_styles import (
            get_architectural_style_id,
            encode_style_as_feature,
            encode_multi_style_feature
        )
        
        num_points = len(next(iter(all_features.values())))
        
        # Default: unknown style
        architectural_style_id = 0
        multi_styles = None
        
        if tile_metadata:
            # Check for multi-label styles
            if "architectural_styles" in tile_metadata:
                multi_styles = tile_metadata["architectural_styles"]
                style_names = [s.get("style_name", "?") for s in multi_styles]
                logger.info(f"  🏛️  Multi-style: {', '.join(style_names)}")
            else:
                # Legacy single style
                characteristics = tile_metadata.get("characteristics", [])
                category = tile_metadata.get("location", {}).get("category")
                architectural_style_id = get_architectural_style_id(
                    characteristics=characteristics,
                    category=category
                )
                loc_name = tile_metadata.get("location", {}).get("name", "?")
                logger.info(f"  🏛️  Style: {architectural_style_id} ({loc_name})")
        
        # Encode style
        if multi_styles and self.style_encoding == 'multihot':
            style_ids = [s["style_id"] for s in multi_styles]
            weights = [s.get("weight", 1.0) for s in multi_styles]
            architectural_style = encode_multi_style_feature(
                style_ids=style_ids,
                weights=weights,
                num_points=num_points,
                encoding="multihot"
            )
        else:
            architectural_style = encode_style_as_feature(
                style_id=architectural_style_id,
                num_points=num_points,
                encoding="constant"
            )
        
        all_features['architectural_style'] = architectural_style
