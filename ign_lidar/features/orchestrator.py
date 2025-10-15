"""
Feature Orchestrator - Unified Feature Computation System

Consolidates FeatureManager, FeatureComputer, and FeatureComputerFactory into a
single, cohesive orchestrator that manages all aspects of feature computation.

This module is part of Phase 4 consolidation effort to simplify the feature system
architecture and reduce code duplication.

Key Responsibilities:
- Resource Management: RGB, NIR fetchers and GPU validation
- Strategy Selection: CPU, GPU, Chunked, or Boundary-aware computation
- Feature Mode Enforcement: MINIMAL, LOD2, LOD3, FULL modes
- Computation Coordination: Geometric, spectral, and architectural features

Example:
    >>> from ign_lidar.features.orchestrator import FeatureOrchestrator
    >>> orchestrator = FeatureOrchestrator(config)
    >>> features = orchestrator.compute_features(tile_data)
    >>> print(f"Computed {len(features)} feature arrays")

Author: Phase 4 Consolidation
Date: October 13, 2025
"""

import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Any, List
import numpy as np
from omegaconf import DictConfig

from .factory import BaseFeatureComputer, CPUFeatureComputer, GPUFeatureComputer
from .factory import GPUChunkedFeatureComputer, BoundaryAwareFeatureComputer
from .feature_modes import FeatureMode, get_feature_config

logger = logging.getLogger(__name__)

__all__ = ['FeatureOrchestrator']


class FeatureOrchestrator:
    """
    Unified orchestrator for all feature computation operations.
    
    This class consolidates three previously separate concerns:
    1. Resource Management (RGB, NIR, GPU) - from FeatureManager
    2. Strategy Selection (CPU/GPU/Chunked) - from FeatureComputerFactory
    3. Computation Coordination - from FeatureComputer
    
    Benefits of consolidation:
    - Single entry point for all feature operations
    - Clear ownership of resources and strategy
    - Consistent feature mode enforcement
    - Easier to test and maintain
    - Less indirection and complexity
    
    Architecture:
        FeatureOrchestrator
            ├── Resource Management
            │   ├── rgb_fetcher: IGNOrthophotoFetcher
            │   ├── infrared_fetcher: IGNInfraredFetcher
            │   └── gpu_available: bool
            ├── Strategy Selection
            │   └── computer: BaseFeatureComputer
            ├── Mode Management
            │   └── feature_mode: FeatureMode
            └── Computation Methods
                ├── compute_features()
                ├── compute_geometric_features()
                └── add_spectral_features()
    
    Example:
        >>> config = OmegaConf.load("config.yaml")
        >>> orchestrator = FeatureOrchestrator(config)
        >>> 
        >>> # Compute features for a tile
        >>> features = orchestrator.compute_features(tile_data)
        >>> 
        >>> # Check what was computed
        >>> print(f"Features: {list(features.keys())}")
        >>> print(f"Strategy: {orchestrator.strategy_name}")
        >>> print(f"Mode: {orchestrator.feature_mode}")
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize feature orchestrator from configuration.
        
        This performs three main initialization steps:
        1. Initialize external resources (RGB, NIR, GPU)
        2. Select and create appropriate feature computer
        3. Setup feature mode and validation
        
        Args:
            config: Hydra/OmegaConf configuration object with sections:
                - processor: use_gpu, use_gpu_chunked, etc.
                - features: mode, k_neighbors, use_rgb, use_infrared, etc.
        
        Raises:
            ValueError: If configuration is invalid
            ImportError: If required dependencies are missing
        """
        self.config = config
        
        # Initialize resources (RGB, NIR, GPU)
        self._init_resources()
        
        # Select and create feature computer
        self._init_computer()
        
        # Setup feature mode
        self._init_feature_mode()
        
        # Log feature configuration with data availability
        self._log_feature_config()
        
        logger.info(
            f"FeatureOrchestrator initialized | "
            f"strategy={self.strategy_name} | "
            f"mode={self.feature_mode} | "
            f"gpu={self.gpu_available}"
        )
    
    # =========================================================================
    # RESOURCE MANAGEMENT (from FeatureManager)
    # =========================================================================
    
    def _init_resources(self):
        """
        Initialize external resources for feature computation.
        
        This includes:
        - RGB orthophoto fetcher (if use_rgb enabled)
        - Infrared (NIR) fetcher (if use_infrared enabled)
        - GPU availability validation (if use_gpu enabled)
        
        Resources are initialized lazily - they're only created if actually
        needed based on configuration.
        """
        processor_cfg = self.config.get('processor', {})
        features_cfg = self.config.get('features', {})
        
        # Initialize RGB fetcher if needed
        self.use_rgb = features_cfg.get('use_rgb', False)
        self.rgb_fetcher = None
        if self.use_rgb:
            self.rgb_fetcher = self._init_rgb_fetcher()
        
        # Initialize NIR fetcher if needed
        self.use_infrared = features_cfg.get('use_infrared', False)
        self.infrared_fetcher = None
        if self.use_infrared:
            self.infrared_fetcher = self._init_infrared_fetcher()
        
        # Validate GPU availability if needed
        self.use_gpu = processor_cfg.get('use_gpu', False)
        self.gpu_available = False
        if self.use_gpu:
            self.gpu_available = self._validate_gpu()
        
        logger.debug(
            f"Resources initialized | "
            f"rgb={self.rgb_fetcher is not None} | "
            f"nir={self.infrared_fetcher is not None} | "
            f"gpu={self.gpu_available}"
        )
    
    def _init_rgb_fetcher(self):
        """
        Initialize RGB orthophoto fetcher.
        
        Returns:
            IGNOrthophotoFetcher instance or None if initialization fails
        """
        try:
            from ..preprocessing.rgb_augmentation import IGNOrthophotoFetcher
            
            # Determine cache directory
            rgb_cache_dir = self.config.features.get('rgb_cache_dir')
            if rgb_cache_dir is None:
                rgb_cache_dir = Path(tempfile.gettempdir()) / "ign_lidar_cache" / "orthophotos"
                rgb_cache_dir.mkdir(parents=True, exist_ok=True)
            else:
                rgb_cache_dir = Path(rgb_cache_dir)
            
            fetcher = IGNOrthophotoFetcher(cache_dir=rgb_cache_dir)
            logger.info(
                "RGB enabled (will use from input LAZ if present, "
                "otherwise fetch from IGN orthophotos)"
            )
            return fetcher
            
        except ImportError as e:
            logger.error(f"RGB augmentation requires additional packages: {e}")
            logger.error("Install with: pip install requests Pillow")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize RGB fetcher: {e}")
            return None
    
    def _init_infrared_fetcher(self):
        """
        Initialize infrared (NIR) fetcher.
        
        Returns:
            IGNInfraredFetcher instance or None if initialization fails
        """
        try:
            from ..preprocessing.infrared_augmentation import IGNInfraredFetcher
            
            # Determine cache directory
            rgb_cache_dir = self.config.features.get('rgb_cache_dir')
            if rgb_cache_dir is None:
                infrared_cache_dir = Path(tempfile.gettempdir()) / "ign_lidar_cache" / "infrared"
            else:
                infrared_cache_dir = Path(rgb_cache_dir).parent / "infrared"
            
            infrared_cache_dir.mkdir(parents=True, exist_ok=True)
            
            fetcher = IGNInfraredFetcher(cache_dir=infrared_cache_dir)
            logger.info(
                "NIR enabled (will use from input LAZ if present, "
                "otherwise fetch from IGN IRC)"
            )
            return fetcher
            
        except ImportError as e:
            logger.error(f"Infrared augmentation requires additional packages: {e}")
            logger.error("Install with: pip install requests Pillow")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize infrared fetcher: {e}")
            return None
    
    def _validate_gpu(self) -> bool:
        """
        Validate GPU availability.
        
        Returns:
            bool: True if GPU is available and working
        """
        try:
            from .features_gpu import GPU_AVAILABLE
            
            if not GPU_AVAILABLE:
                logger.warning(
                    "GPU requested but CuPy not available. Using CPU."
                )
                return False
            
            logger.info("GPU acceleration enabled")
            return True
            
        except ImportError:
            logger.warning("GPU module not available. Using CPU.")
            return False
        except Exception as e:
            logger.error(f"GPU validation failed: {e}")
            return False
    
    # =========================================================================
    # STRATEGY SELECTION (from FeatureComputerFactory)
    # =========================================================================
    
    def _init_computer(self):
        """
        Select and create appropriate feature computer based on configuration.
        
        Strategy selection logic:
        1. Boundary-aware if use_boundary_aware=True
        2. GPU chunked if use_gpu=True and use_gpu_chunked=True
        3. GPU basic if use_gpu=True
        4. CPU otherwise (fallback)
        
        The created computer is cached in self.computer for reuse.
        """
        processor_cfg = self.config.get('processor', {})
        features_cfg = self.config.get('features', {})
        
        # Extract parameters
        k_neighbors = features_cfg.get('k_neighbors', 20)
        use_boundary_aware = processor_cfg.get('use_boundary_aware', False)
        use_gpu_chunked = processor_cfg.get('use_gpu_chunked', False)
        
        # Select strategy
        if use_boundary_aware:
            self.strategy_name = "boundary_aware"
            buffer_size = processor_cfg.get('buffer_size', 10.0)
            self.computer = BoundaryAwareFeatureComputer(
                k_neighbors=k_neighbors,
                buffer_size=buffer_size
            )
        elif self.gpu_available and use_gpu_chunked:
            self.strategy_name = "gpu_chunked"
            chunk_size = processor_cfg.get('gpu_batch_size', 100000)
            self.computer = GPUChunkedFeatureComputer(
                k_neighbors=k_neighbors,
                gpu_batch_size=chunk_size
            )
        elif self.gpu_available:
            self.strategy_name = "gpu"
            self.computer = GPUFeatureComputer(
                k_neighbors=k_neighbors
            )
        else:
            self.strategy_name = "cpu"
            self.computer = CPUFeatureComputer(
                k_neighbors=k_neighbors
            )
        
        logger.debug(f"Selected strategy: {self.strategy_name}")
    
    # =========================================================================
    # FEATURE MODE MANAGEMENT (enhanced)
    # =========================================================================
    
    def _init_feature_mode(self):
        """
        Initialize and validate feature mode.
        
        Feature modes control which features are computed:
        - MINIMAL: ~8 features (fast)
        - LOD2_SIMPLIFIED: ~11 features (building classification)
        - LOD3_FULL: ~35 features (detailed modeling)
        - FULL: All available features
        - CUSTOM: User-defined selection
        """
        features_cfg = self.config.get('features', {})
        # Check both 'mode' (Hydra configs) and 'feature_mode' (legacy kwargs)
        mode_str = features_cfg.get('mode') or features_cfg.get('feature_mode', 'lod2')
        
        # Debug: Log the mode being loaded (with full config inspection)
        logger.info(f"🔍 DEBUG: features_cfg = {dict(features_cfg)}")
        logger.info(f"🔍 DEBUG: mode_str = '{mode_str}' (type: {type(mode_str).__name__})")
        
        # Parse mode
        try:
            if isinstance(mode_str, FeatureMode):
                self.feature_mode = mode_str
                logger.info(f"🔍 DEBUG: mode_str is already a FeatureMode: {self.feature_mode}")
            else:
                self.feature_mode = FeatureMode(mode_str.lower())
                logger.info(f"🔍 DEBUG: Parsed mode string '{mode_str}' to {self.feature_mode}")
        except ValueError:
            logger.warning(f"Invalid feature mode '{mode_str}', using FULL")
            self.feature_mode = FeatureMode.FULL
        
        # Get feature configuration for this mode
        # Note: We don't know data availability yet during initialization,
        # so we suppress the detailed logging here (will be shown during processing)
        self.feature_config = get_feature_config(
            self.feature_mode.value,
            has_rgb=None,
            has_nir=None,
            log_config=False  # Suppress logging during init
        )
        
        logger.debug(
            f"Feature mode: {self.feature_mode.value} | "
            f"features: {len(self.feature_config.feature_names)}"
        )
    
    def _log_feature_config(self):
        """Log feature configuration with data availability information."""
        # Check if RGB/NIR fetchers are available (not necessarily that data exists yet)
        has_rgb_fetcher = self.rgb_fetcher is not None
        has_nir_fetcher = self.infrared_fetcher is not None
        
        logger.info(f"📊 Feature Configuration: {self.feature_config.get_description()}")
        logger.info(f"   Features: {', '.join(self.feature_config.feature_names)}")
        
        # Log RGB/NIR status based on whether fetchers are available
        if self.feature_config.requires_rgb:
            if has_rgb_fetcher or self.use_rgb:
                logger.info("   ✓ RGB channels enabled (will use from input LAZ or fetch if needed)")
            elif has_rgb_fetcher is False and not self.use_rgb:
                logger.debug("   ⚠️  RGB channels required but RGB fetcher not available (will attempt to load from LAZ)")
        
        if self.feature_config.requires_nir:
            if has_nir_fetcher or self.use_infrared:
                logger.info("   ✓ NIR channel enabled (will use from input LAZ or fetch if needed)")
            elif has_nir_fetcher is False and not self.use_infrared:
                logger.debug("   ⚠️  NIR channel required but NIR fetcher not available (will attempt to load from LAZ)")
    
    def validate_mode(self, mode: FeatureMode) -> bool:
        """
        Validate that a feature mode is supported.
        
        Args:
            mode: FeatureMode to validate
            
        Returns:
            bool: True if mode is valid and supported
        """
        return isinstance(mode, FeatureMode)
    
    def get_feature_list(self, mode: Optional[FeatureMode] = None) -> List[str]:
        """
        Get list of features for a given mode.
        
        Args:
            mode: FeatureMode to query (defaults to current mode)
            
        Returns:
            List of feature names that will be computed
        """
        if mode is None:
            mode = self.feature_mode
        
        mode_str = mode.value if isinstance(mode, FeatureMode) else mode
        feature_config = get_feature_config(mode_str)
        return list(feature_config.feature_names)
    
    def filter_features(
        self, 
        features: Dict[str, np.ndarray],
        mode: Optional[FeatureMode] = None
    ) -> Dict[str, np.ndarray]:
        """
        Filter features dict to only include features for given mode.
        
        This is used to enforce feature mode consistency across all
        computation strategies.
        
        Args:
            features: Dict of all computed features
            mode: FeatureMode to filter by (defaults to current mode)
            
        Returns:
            Filtered dict with only features for specified mode
        """
        if mode is None:
            mode = self.feature_mode
        
        # Get allowed features for this mode
        allowed_features = set(self.get_feature_list(mode))
        
        # Always keep core features regardless of mode
        core_features = {'normals', 'curvature', 'height', 'intensity', 'return_number'}
        allowed_features.update(core_features)
        
        # Handle spectral features: if mode defines 'red', 'green', 'blue' individually,
        # also allow 'rgb' as a combined feature (and vice versa)
        if 'red' in allowed_features or 'green' in allowed_features or 'blue' in allowed_features:
            allowed_features.add('rgb')
        if 'rgb' in allowed_features:
            allowed_features.update(['red', 'green', 'blue'])
        
        # Same for NIR and NDVI
        if 'nir' in allowed_features or self.use_infrared:
            allowed_features.add('nir')
        if 'ndvi' in allowed_features or (self.use_rgb and self.use_infrared):
            allowed_features.add('ndvi')
        
        # Filter
        filtered = {
            k: v for k, v in features.items()
            if k in allowed_features or k.startswith('enriched_')
        }
        
        return filtered
    
    # =========================================================================
    # COMPUTATION COORDINATION (from FeatureComputer)
    # =========================================================================
    
    def compute_features(
        self,
        tile_data: Dict[str, Any],
        use_enriched: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Compute all features for a point cloud tile.
        
        This is the main entry point for feature computation. It orchestrates:
        1. Checking for existing enriched features
        2. Computing geometric features (if needed)
        3. Adding spectral features (RGB, NIR, NDVI)
        4. Adding architectural style features
        5. Enforcing feature mode
        
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
            use_enriched: If True and enriched features exist, use them
            
        Returns:
            Dictionary of computed features with keys like:
                - normals: (N, 3) surface normals
                - curvature: (N,) curvature values
                - height: (N,) height above ground
                - intensity: (N,) intensity values
                - return_number: (N,) return numbers
                - [geometric features]: (N,) each
                - rgb: (N, 3) if requested
                - nir: (N,) if requested
                - ndvi: (N,) if requested
                - architectural_style: (N,) or (N, K) if requested
        
        Example:
            >>> features = orchestrator.compute_features(tile_data)
            >>> print(f"Computed {len(features)} feature arrays")
            >>> print(f"Point cloud size: {features['normals'].shape[0]}")
        """
        points = tile_data['points']
        classification = tile_data['classification']
        intensity = tile_data['intensity']
        return_number = tile_data['return_number']
        enriched_features = tile_data.get('enriched_features', {})
        
        all_features = {}
        
        # Check if we should use existing enriched features
        if use_enriched and enriched_features:
            logger.info("  ♻️  Using existing enriched features from input LAZ")
            
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
            # Check data availability for better logging
            has_rgb = tile_data.get('input_rgb') is not None or (self.rgb_fetcher is not None)
            has_nir = tile_data.get('input_nir') is not None or (self.infrared_fetcher is not None)
            
            # Compute features
            normals, curvature, height, geo_features = self._compute_geometric_features(
                points, classification, has_rgb=has_rgb, has_nir=has_nir
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
        
        # Add spectral features (RGB, NIR, NDVI)
        rgb_added = self._add_rgb_features(tile_data, all_features)
        nir_added = self._add_nir_features(tile_data, all_features)
        self._add_ndvi_features(tile_data, all_features, rgb_added, nir_added)
        
        # Add architectural style if requested
        self._add_architectural_style(tile_data, all_features)
        
        # Enforce feature mode (filter to only allowed features)
        if self.feature_mode != FeatureMode.FULL:
            all_features = self.filter_features(all_features, self.feature_mode)
            logger.debug(
                f"  🔽 Filtered to {len(all_features)} features for mode {self.feature_mode.value}"
            )
        
        return all_features
    
    def _compute_geometric_features(
        self,
        points: np.ndarray,
        classification: np.ndarray,
        has_rgb: bool = False,
        has_nir: bool = False
    ) -> tuple:
        """
        Compute geometric features using selected strategy.
        
        Args:
            points: (N, 3) XYZ coordinates
            classification: (N,) classification codes
            has_rgb: Whether RGB data is available
            has_nir: Whether NIR data is available
            
        Returns:
            Tuple of (normals, curvature, height, geo_features_dict)
        """
        features_cfg = self.config.get('features', {})
        processor_cfg = self.config.get('processor', {})
        
        k_neighbors = features_cfg.get('k_neighbors')
        search_radius = features_cfg.get('search_radius', None)
        
        # Derive include_extra from feature mode (not from config)
        # Only MINIMAL mode should exclude extra features
        include_extra = self.feature_mode != FeatureMode.MINIMAL
        
        k_display = k_neighbors if k_neighbors else "auto"
        
        # Log search strategy
        if search_radius is not None and search_radius > 0:
            logger.info(f"  🔧 Computing features | radius={search_radius:.2f}m (avoids scan line artifacts) | mode={self.feature_mode.value}")
        else:
            logger.info(f"  🔧 Computing features | k={k_display} | mode={self.feature_mode.value}")
        
        feature_start = time.time()
        
        # Compute patch center for distance_to_center feature
        patch_center = (
            np.mean(points, axis=0) if include_extra else None
        )
        
        # Use manual k if specified, otherwise auto-estimate
        use_auto_k = k_neighbors is None
        k_value = k_neighbors if k_neighbors is not None else 20
        
        # Compute features using selected computer
        feature_dict = self.computer.compute_features(
            points=points,
            classification=classification,
            auto_k=use_auto_k,
            include_extra=include_extra,
            patch_center=patch_center,
            mode=self.feature_mode.value,
            radius=search_radius  # Pass radius parameter
        )
        
        # Extract main features
        normals = feature_dict.get('normals')
        curvature = feature_dict.get('curvature')
        height = feature_dict.get('height')
        
        # Extract geometric features
        main_features = {'normals', 'curvature', 'height'}
        geo_features = {
            k: v for k, v in feature_dict.items()
            if k not in main_features
        }
        
        elapsed = time.time() - feature_start
        logger.info(
            f"  ✓ Computed {len(geo_features)} geometric features "
            f"in {elapsed:.2f}s using {self.strategy_name}"
        )
        
        return normals, curvature, height, geo_features
    
    # =========================================================================
    # SPECTRAL FEATURES (RGB, NIR, NDVI)
    # =========================================================================
    
    def _add_rgb_features(
        self,
        tile_data: Dict[str, Any],
        all_features: Dict[str, np.ndarray]
    ) -> bool:
        """
        Add RGB features from input LAZ or fetch from orthophotos.
        
        Args:
            tile_data: Tile data dict
            all_features: Features dict to update
            
        Returns:
            bool: True if RGB was added
        """
        if not self.use_rgb:
            return False
        
        # Check if RGB already in input
        input_rgb = tile_data.get('input_rgb')
        if input_rgb is not None:
            all_features['rgb'] = input_rgb
            logger.info("  ✓ Using RGB from input LAZ")
            return True
        
        # Try to fetch from orthophotos
        if self.rgb_fetcher is not None:
            try:
                points = tile_data['points']
                rgb = self.rgb_fetcher.augment_points_with_rgb(points)
                if rgb is not None:
                    all_features['rgb'] = rgb
                    logger.info("  ✓ Fetched RGB from IGN orthophotos")
                    return True
            except Exception as e:
                logger.warning(f"  ⚠️  Failed to fetch RGB: {e}")
        
        return False
    
    def _add_nir_features(
        self,
        tile_data: Dict[str, Any],
        all_features: Dict[str, np.ndarray]
    ) -> bool:
        """
        Add NIR (near-infrared) features from input LAZ or fetch.
        
        Args:
            tile_data: Tile data dict
            all_features: Features dict to update
            
        Returns:
            bool: True if NIR was added
        """
        if not self.use_infrared:
            return False
        
        # Check if NIR already in input
        input_nir = tile_data.get('input_nir')
        if input_nir is not None:
            all_features['nir'] = input_nir
            logger.info("  ✓ Using NIR from input LAZ")
            return True
        
        # Try to fetch from infrared service
        if self.infrared_fetcher is not None:
            try:
                points = tile_data['points']
                nir = self.infrared_fetcher.augment_points_with_infrared(points)
                if nir is not None:
                    all_features['nir'] = nir
                    logger.info("  ✓ Fetched NIR from IGN IRC")
                    return True
            except Exception as e:
                logger.warning(f"  ⚠️  Failed to fetch NIR: {e}")
        
        return False
    
    def _add_ndvi_features(
        self,
        tile_data: Dict[str, Any],
        all_features: Dict[str, np.ndarray],
        rgb_added: bool,
        nir_added: bool
    ):
        """
        Add or compute NDVI (Normalized Difference Vegetation Index).
        
        NDVI = (NIR - Red) / (NIR + Red)
        
        Args:
            tile_data: Tile data dict
            all_features: Features dict to update
            rgb_added: Whether RGB was successfully added
            nir_added: Whether NIR was successfully added
        """
        # Check both processor and features sections for compute_ndvi flag
        processor_cfg = self.config.get('processor', {})
        features_cfg = self.config.get('features', {})
        compute_ndvi = features_cfg.get('compute_ndvi', processor_cfg.get('compute_ndvi', False))
        
        if not compute_ndvi:
            return
        
        # Check if NDVI already in input
        input_ndvi = tile_data.get('input_ndvi')
        if input_ndvi is not None:
            all_features['ndvi'] = input_ndvi
            logger.info("  ✓ Using NDVI from input LAZ")
            return
        
        # Compute NDVI if we have both RGB and NIR
        if rgb_added and nir_added:
            rgb = all_features['rgb']
            nir = all_features['nir']
            
            # Extract red channel (index 0)
            red = rgb[:, 0].astype(np.float32)
            nir_float = nir.astype(np.float32)
            
            # Compute NDVI with safe division
            denominator = nir_float + red
            ndvi = np.zeros_like(nir_float)
            mask = denominator > 0
            ndvi[mask] = (nir_float[mask] - red[mask]) / denominator[mask]
            
            # Validate NDVI: clip to valid range [-1, 1] and fix any NaN/Inf
            ndvi = np.clip(ndvi, -1.0, 1.0)
            ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=1.0, neginf=-1.0)
            
            all_features['ndvi'] = ndvi
            logger.info("  ✓ Computed NDVI from RGB and NIR")
        else:
            # Don't add NDVI to features if we can't compute it
            # This prevents None/empty values from being saved
            logger.debug("  ⚠️  Cannot compute NDVI (need both RGB and NIR) - not adding to features")
    
    def _add_architectural_style(
        self,
        tile_data: Dict[str, Any],
        all_features: Dict[str, np.ndarray]
    ):
        """
        Add architectural style features if requested.
        
        Args:
            tile_data: Tile data dict
            all_features: Features dict to update
        """
        processor_cfg = self.config.get('processor', {})
        include_style = processor_cfg.get('include_architectural_style', False)
        
        if not include_style:
            return
        
        try:
            from .architectural_styles import compute_architectural_style_features
            
            points = tile_data['points']
            classification = tile_data['classification']
            encoding = processor_cfg.get('style_encoding', 'constant')
            
            style_features = compute_architectural_style_features(
                points=points,
                classification=classification,
                encoding=encoding
            )
            
            all_features['architectural_style'] = style_features
            logger.info(f"  ✓ Added architectural style features (encoding={encoding})")
            
        except ImportError:
            logger.warning("  ⚠️  Architectural style module not available")
        except Exception as e:
            logger.error(f"  ❌ Failed to compute architectural style: {e}")
    
    # =========================================================================
    # PROPERTIES & UTILITIES
    # =========================================================================
    
    @property
    def has_rgb(self) -> bool:
        """Check if RGB fetcher is available."""
        return self.rgb_fetcher is not None
    
    @property
    def has_infrared(self) -> bool:
        """Check if infrared fetcher is available."""
        return self.infrared_fetcher is not None
    
    @property
    def has_gpu(self) -> bool:
        """Check if GPU is available."""
        return self.gpu_available
    
    def __repr__(self) -> str:
        """String representation of orchestrator."""
        return (
            f"FeatureOrchestrator("
            f"strategy={self.strategy_name}, "
            f"mode={self.feature_mode.value}, "
            f"gpu={self.gpu_available}, "
            f"rgb={self.has_rgb}, "
            f"nir={self.has_infrared})"
        )
