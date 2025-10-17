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

# NEW: Strategy Pattern (Week 2 refactoring)
from .strategies import BaseFeatureStrategy, FeatureComputeMode
from .strategy_cpu import CPUStrategy
try:
    from .strategy_gpu import GPUStrategy
    from .strategy_gpu_chunked import GPUChunkedStrategy
except ImportError:
    GPUStrategy = None
    GPUChunkedStrategy = None
from .strategy_boundary import BoundaryAwareStrategy

# LEGACY: Old factory pattern (deprecated, for backward compatibility)
try:
    from .factory import BaseFeatureComputer, CPUFeatureComputer, GPUFeatureComputer
    from .factory import GPUChunkedFeatureComputer, BoundaryAwareFeatureComputer
    LEGACY_FACTORY_AVAILABLE = True
except ImportError:
    LEGACY_FACTORY_AVAILABLE = False
    
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
        
        # ✅ OPTIMIZATION: Cache frequently accessed config values (Phase 1 - Quick Win)
        self._cache_config_values()
        
        # ✅ OPTIMIZATION: Initialize advanced optimizations (V5 consolidation)
        self._init_optimizations()
        
        # Log feature configuration with data availability
        self._log_feature_config()
        
        logger.info(
            f"FeatureOrchestrator V5 initialized | "
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
        # Support both 'use_nir' (current) and 'use_infrared' (legacy) for backward compatibility
        self.use_infrared = features_cfg.get('use_nir', features_cfg.get('use_infrared', False))
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
        Select and create appropriate feature strategy based on configuration.
        
        NEW (Week 2): Uses Strategy Pattern instead of Factory Pattern
        
        Strategy selection logic:
        1. Boundary-aware if use_boundary_aware=True (wraps base strategy)
        2. GPU chunked if use_gpu=True and use_gpu_chunked=True
        3. GPU basic if use_gpu=True
        4. CPU otherwise (fallback)
        
        The created strategy is cached in self.computer for reuse.
        
        Note: self.computer can be either:
        - New Strategy Pattern: BaseFeatureStrategy (Week 2)
        - Old Factory Pattern: BaseFeatureComputer (legacy, deprecated)
        """
        processor_cfg = self.config.get('processor', {})
        features_cfg = self.config.get('features', {})
        
        # Extract parameters
        k_neighbors = features_cfg.get('k_neighbors', 20)
        radius = features_cfg.get('search_radius', 1.0)
        use_boundary_aware = processor_cfg.get('use_boundary_aware', False)
        use_gpu_chunked = processor_cfg.get('use_gpu_chunked', False)
        use_strategy_pattern = processor_cfg.get('use_strategy_pattern', True)  # NEW: opt-in
        
        # Initialize size tracking for logging
        gpu_size = None
        
        # NEW (Week 2): Strategy Pattern implementation
        if use_strategy_pattern:
            logger.info("🆕 Using Strategy Pattern (Week 2 refactoring)")
            
            # Select base strategy
            if self.gpu_available and use_gpu_chunked:
                self.strategy_name = "gpu_chunked"
                chunk_size = processor_cfg.get('gpu_batch_size', 5_000_000)
                batch_size = 250_000  # Week 1 optimized batch size
                gpu_size = chunk_size
                
                if GPUChunkedStrategy is None:
                    logger.warning("GPU chunked strategy not available, falling back to CPU")
                    base_strategy = CPUStrategy(k_neighbors=k_neighbors, radius=radius)
                else:
                    base_strategy = GPUChunkedStrategy(
                        k_neighbors=k_neighbors,
                        radius=radius,
                        chunk_size=chunk_size,
                        batch_size=batch_size
                    )
                    
            elif self.gpu_available:
                self.strategy_name = "gpu"
                batch_size = features_cfg.get('gpu_batch_size', processor_cfg.get('gpu_batch_size', 8_000_000))
                gpu_size = batch_size
                
                if GPUStrategy is None:
                    logger.warning("GPU strategy not available, falling back to CPU")
                    base_strategy = CPUStrategy(k_neighbors=k_neighbors, radius=radius)
                else:
                    base_strategy = GPUStrategy(
                        k_neighbors=k_neighbors,
                        radius=radius,
                        batch_size=batch_size
                    )
            else:
                self.strategy_name = "cpu"
                base_strategy = CPUStrategy(k_neighbors=k_neighbors, radius=radius)
            
            # Wrap with boundary-aware strategy if needed
            if use_boundary_aware:
                buffer_size = processor_cfg.get('buffer_size', 10.0)
                self.computer = BoundaryAwareStrategy(
                    base_strategy=base_strategy,
                    boundary_buffer=buffer_size
                )
                self.strategy_name = f"boundary_aware({self.strategy_name})"
            else:
                self.computer = base_strategy
        
        # LEGACY: Old factory pattern (deprecated)
        else:
            logger.warning("⚠️  Using legacy Factory Pattern (deprecated, will be removed)")
            
            if not LEGACY_FACTORY_AVAILABLE:
                raise ImportError(
                    "Legacy factory pattern not available. "
                    "Set processor.use_strategy_pattern=true in config to use new Strategy Pattern."
                )
            
            # Select strategy (old way)
            if use_boundary_aware:
                self.strategy_name = "boundary_aware"
                buffer_size = processor_cfg.get('buffer_size', 10.0)
                self.computer = BoundaryAwareFeatureComputer(
                    k_neighbors=k_neighbors,
                    buffer_size=buffer_size
                )
            elif self.gpu_available and use_gpu_chunked:
                self.strategy_name = "gpu_chunked"
                chunk_size = processor_cfg.get('gpu_batch_size', 1_000_000)
                gpu_size = chunk_size
                self.computer = GPUChunkedFeatureComputer(
                    k_neighbors=k_neighbors,
                    gpu_batch_size=chunk_size
                )
            elif self.gpu_available:
                self.strategy_name = "gpu"
                batch_size = features_cfg.get('gpu_batch_size', processor_cfg.get('gpu_batch_size', 1_000_000))
                gpu_size = batch_size
                self.computer = GPUFeatureComputer(
                    k_neighbors=k_neighbors,
                    gpu_batch_size=batch_size
                )
            else:
                self.strategy_name = "cpu"
                self.computer = CPUFeatureComputer(
                    k_neighbors=k_neighbors
                )
        
        logger.debug(f"Selected strategy: {self.strategy_name}")
        if gpu_size is not None:
            logger.info(f"  💾 GPU batch/chunk size: {gpu_size:,} points")
    
    def _cache_config_values(self):
        """
        Cache frequently accessed config values to avoid repeated lookups.
        
        OPTIMIZATION: Phase 1 - Quick Win
        - Caches ~10 most frequently accessed values
        - Reduces dict lookup overhead (~50-100ns per lookup)
        - Saves ~10ms total across thousands of accesses
        - Negligible impact individually, but good practice
        
        This is a micro-optimization but demonstrates proper config handling.
        """
        features_cfg = self.config.get('features', {})
        processor_cfg = self.config.get('processor', {})
        
        # Cache common values
        self._k_neighbors_cached = features_cfg.get('k_neighbors', 20)
        self._use_gpu_chunked_cached = processor_cfg.get('use_gpu_chunked', False)
        self._gpu_batch_size_cached = features_cfg.get('gpu_batch_size', 1_000_000)
        self._search_radius_cached = features_cfg.get('search_radius', 1.0)
        self._use_boundary_aware_cached = processor_cfg.get('use_boundary_aware', False)
        self._buffer_size_cached = processor_cfg.get('buffer_size', 10.0)
        
        logger.debug("Config values cached for fast access")
    
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
    
    # =========================================================================
    # OPTIMIZATION FEATURES (V5 Consolidation)
    # =========================================================================
    
    def _init_optimizations(self):
        """Initialize optimization subsystems consolidated from EnhancedFeatureOrchestrator."""
        self._init_caching()
        self._init_parallel_processing()
        self._init_adaptive_parameters()
        self._init_performance_monitoring()
        
        logger.debug("FeatureOrchestrator V5 optimizations initialized")
    
    def _init_caching(self):
        """Initialize intelligent feature caching system."""
        features_cfg = self.config.get('features', {})
        
        self._enable_feature_cache = features_cfg.get('enable_caching', True)
        self._cache_max_size = features_cfg.get('cache_max_size', 100)  # MB
        self._feature_cache = {}
        self._current_cache_size = 0
        
        if self._enable_feature_cache:
            logger.debug("Feature caching enabled")
    
    def _init_parallel_processing(self):
        """Initialize parallel processing for RGB/NIR operations."""
        from concurrent.futures import ThreadPoolExecutor
        
        processor_cfg = self.config.get('processor', {})
        num_workers = processor_cfg.get('num_workers', 1)
        
        # Create thread pools for different operations
        self._rgb_nir_executor = ThreadPoolExecutor(
            max_workers=min(4, max(1, num_workers // 2)),
            thread_name_prefix="rgb_nir"
        )
        self._feature_executor = ThreadPoolExecutor(
            max_workers=min(2, max(1, num_workers)),
            thread_name_prefix="features"
        )
        
        logger.debug(f"Parallel processing initialized: "
                    f"rgb_nir_workers={self._rgb_nir_executor._max_workers}, "
                    f"feature_workers={self._feature_executor._max_workers}")
    
    def _init_adaptive_parameters(self):
        """Initialize adaptive parameter tuning."""
        features_cfg = self.config.get('features', {})
        
        # Base parameters that can be adapted
        self._adaptive_parameters = {
            'k_neighbors': features_cfg.get('k_neighbors', 20),
            'search_radius': features_cfg.get('search_radius', 1.0),
            'batch_size': features_cfg.get('gpu_batch_size', 1_000_000)
        }
        
        # Adaptation history
        self._parameter_history = {}
        self._adaptation_enabled = features_cfg.get('enable_auto_tuning', True)
        
        if self._adaptation_enabled:
            logger.debug("Adaptive parameter tuning enabled")
    
    def _init_performance_monitoring(self):
        """Initialize performance monitoring and metrics collection."""
        monitoring_cfg = self.config.get('monitoring', {})
        
        self._enable_profiling = monitoring_cfg.get('enable_profiling', False)
        self._enable_performance_metrics = monitoring_cfg.get('enable_performance_metrics', True)
        
        # Performance tracking
        self._processing_times = []
        self._memory_usage = []
        self._gpu_utilization = []
        
        if self._enable_performance_metrics:
            logger.debug("Performance monitoring enabled")
    
    def _generate_cache_key(self, points, classification, kwargs):
        """Generate a unique cache key for the given inputs."""
        import hashlib
        
        # Create a hash from point statistics and parameters
        data_hash = hashlib.md5()
        data_hash.update(str(points.shape).encode())
        data_hash.update(str(np.mean(points, axis=0)).encode())
        data_hash.update(str(np.std(points, axis=0)).encode())
        data_hash.update(str(np.unique(classification)).encode())
        data_hash.update(str(sorted(kwargs.items())).encode())
        
        return data_hash.hexdigest()
    
    def _optimize_parameters_for_data(self, points, classification):
        """
        Optimize parameters based on data characteristics.
        
        Args:
            points: Point cloud data
            classification: Classification codes
            
        Returns:
            Optimized parameters
        """
        n_points = len(points)
        
        # Analyze point density
        bbox_volume = self._estimate_point_cloud_volume(points)
        point_density = n_points / bbox_volume if bbox_volume > 0 else 1000
        
        # Base parameters
        optimized = self._adaptive_parameters.copy()
        
        # Optimize k_neighbors based on density
        base_k = optimized['k_neighbors']
        if point_density > 10000:  # Very dense
            optimized['k_neighbors'] = min(base_k + 10, 50)
        elif point_density > 1000:  # Dense
            optimized['k_neighbors'] = base_k
        else:  # Sparse
            optimized['k_neighbors'] = max(base_k - 5, 10)
        
        # Optimize search radius based on data distribution
        z_range = np.ptp(points[:, 2])  # Z-axis range
        base_radius = optimized['search_radius']
        if z_range > 100:  # Building/urban area
            optimized['search_radius'] = base_radius * 1.5
        elif z_range < 10:  # Flat area
            optimized['search_radius'] = base_radius * 0.8
        
        # Optimize batch size for GPU processing
        base_batch_size = optimized['batch_size']
        if n_points > 10_000_000:  # Very large
            optimized['batch_size'] = min(base_batch_size, 500_000)
        elif n_points > 5_000_000:  # Large
            optimized['batch_size'] = min(base_batch_size, 1_000_000)
        
        logger.debug(f"Optimized parameters: k={optimized['k_neighbors']}, "
                    f"radius={optimized['search_radius']:.2f}, "
                    f"batch_size={optimized['batch_size']:,}")
        
        return optimized
    
    def _estimate_point_cloud_volume(self, points):
        """Estimate the volume of the point cloud bounding box."""
        if len(points) == 0:
            return 1.0
        
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        dimensions = max_coords - min_coords
        
        # Avoid zero volume
        dimensions = np.maximum(dimensions, 0.1)
        
        return np.prod(dimensions)
    
    def _should_cache_features(self, points, features):
        """Determine if features should be cached."""
        if not self._enable_feature_cache:
            return False
        
        # Estimate feature size
        feature_size = sum(arr.nbytes for arr in features.values()) / (1024 * 1024)  # MB
        
        # Don't cache if it would exceed max size
        if self._current_cache_size + feature_size > self._cache_max_size:
            return False
        
        # Cache if it's a reasonably sized result
        return 1 < feature_size < 50  # Cache features between 1MB and 50MB
    
    def _cache_features(self, cache_key, features):
        """Cache computed features."""
        feature_size = sum(arr.nbytes for arr in features.values()) / (1024 * 1024)  # MB
        
        # Clean cache if needed
        while self._current_cache_size + feature_size > self._cache_max_size and self._feature_cache:
            oldest_key = next(iter(self._feature_cache))
            old_features = self._feature_cache.pop(oldest_key)
            old_size = sum(arr.nbytes for arr in old_features.values()) / (1024 * 1024)
            self._current_cache_size -= old_size
        
        # Cache features
        self._feature_cache[cache_key] = features
        self._current_cache_size += feature_size
        
        logger.debug(f"Cached features ({feature_size:.1f}MB), total cache: {self._current_cache_size:.1f}MB")
    
    def _update_performance_metrics(self, processing_time, num_points):
        """Update performance metrics."""
        if not self._enable_performance_metrics:
            return
        
        self._processing_times.append(processing_time)
        
        # Keep only recent metrics (last 100 computations)
        if len(self._processing_times) > 100:
            self._processing_times.pop(0)
        
        points_per_second = num_points / processing_time if processing_time > 0 else 0
        logger.debug(f"Performance: {processing_time:.2f}s, {points_per_second:.0f} points/sec")
    
    def clear_cache(self):
        """Clear feature cache."""
        self._feature_cache.clear()
        self._current_cache_size = 0
        logger.info("Feature cache cleared")
    
    def get_performance_summary(self):
        """Get performance summary."""
        if not hasattr(self, '_processing_times') or not self._processing_times:
            return {}
        
        return {
            'total_computations': len(self._processing_times),
            'avg_processing_time': np.mean(self._processing_times),
            'min_processing_time': np.min(self._processing_times),
            'max_processing_time': np.max(self._processing_times),
            'cache_hit_ratio': len(self._feature_cache) / len(self._processing_times) if self._processing_times else 0,
            'current_cache_size_mb': getattr(self, '_current_cache_size', 0),
            'strategy': self.strategy_name,
            'feature_mode': str(self.feature_mode),
            'adaptive_parameters': getattr(self, '_adaptive_parameters', {}).copy()
        }
    
    def _start_parallel_rgb_nir_processing(self, tile_data):
        """Start parallel RGB/NIR processing."""
        def fetch_rgb_nir():
            """Fetch RGB/NIR data in parallel."""
            results = {}
            points = tile_data['points']
            
            if self.use_rgb and self.rgb_fetcher:
                try:
                    rgb_data = self.rgb_fetcher.fetch_for_points(points)
                    results['rgb'] = rgb_data
                except Exception as e:
                    logger.warning(f"RGB fetch failed: {e}")
            
            if self.use_infrared and self.infrared_fetcher:
                try:
                    nir_data = self.infrared_fetcher.fetch_for_points(points)
                    results['nir'] = nir_data
                except Exception as e:
                    logger.warning(f"NIR fetch failed: {e}")
            
            return results
        
        return self._rgb_nir_executor.submit(fetch_rgb_nir)
    
    def _compute_geometric_features_optimized(self, points, classification, optimized_params, **kwargs):
        """Compute geometric features with optimized parameters."""
        # Apply optimized parameters temporarily
        if optimized_params and hasattr(self.computer, 'k_neighbors'):
            original_k = getattr(self.computer, 'k_neighbors', 20)
            setattr(self.computer, 'k_neighbors', optimized_params.get('k_neighbors', original_k))
        
        try:
            # Use the existing _compute_geometric_features method
            return self._compute_geometric_features(points, classification, **kwargs)
        finally:
            # Restore original parameters
            if optimized_params and hasattr(self.computer, 'k_neighbors'):
                setattr(self.computer, 'k_neighbors', original_k)
    
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
        Compute all features for a point cloud tile with V5 optimizations.
        
        This enhanced version includes:
        1. Intelligent caching with memory management
        2. Automatic parameter optimization based on data characteristics
        3. Parallel RGB/NIR processing
        4. Performance monitoring and adaptive tuning
        5. All original feature computation capabilities
        
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
            Dictionary of computed features with optimizations applied
        """
        import time
        start_time = time.time()
        
        points = tile_data['points']
        classification = tile_data['classification']
        intensity = tile_data['intensity']
        return_number = tile_data['return_number']
        enriched_features = tile_data.get('enriched_features', {})
        
        # V5 OPTIMIZATION: Check cache first
        cache_key = None
        if hasattr(self, '_enable_feature_cache') and self._enable_feature_cache:
            cache_key = self._generate_cache_key(points, classification, {
                'use_enriched': use_enriched, 
                'mode': str(self.feature_mode)
            })
            if cache_key in self._feature_cache:
                logger.debug("Features retrieved from cache")
                return self._feature_cache[cache_key]
        
        # V5 OPTIMIZATION: Optimize parameters based on data characteristics
        optimized_params = {}
        if hasattr(self, '_adaptive_parameters'):
            optimized_params = self._optimize_parameters_for_data(points, classification)
        
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
            
            # V5 OPTIMIZATION: Start parallel RGB/NIR processing if available
            rgb_nir_future = None
            if hasattr(self, '_rgb_nir_executor') and (self.use_rgb or self.use_infrared):
                rgb_nir_future = self._start_parallel_rgb_nir_processing(tile_data)
            
            # Compute features with optimized parameters
            normals, curvature, height, geo_features = self._compute_geometric_features_optimized(
                points, classification, optimized_params, has_rgb=has_rgb, has_nir=has_nir
            )
            
            # V5 OPTIMIZATION: Integrate parallel RGB/NIR results
            if rgb_nir_future is not None:
                try:
                    parallel_rgb_nir = rgb_nir_future.result(timeout=30)
                    # Merge parallel results into tile_data for later processing
                    if 'rgb' in parallel_rgb_nir:
                        tile_data['fetched_rgb'] = parallel_rgb_nir['rgb']
                    if 'nir' in parallel_rgb_nir:
                        tile_data['fetched_nir'] = parallel_rgb_nir['nir']
                except Exception as e:
                    logger.warning(f"Parallel RGB/NIR processing failed: {e}")
        
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
        
        # V5 OPTIMIZATION: Cache results and update performance metrics
        processing_time = time.time() - start_time
        
        if hasattr(self, '_enable_feature_cache') and cache_key and self._should_cache_features(points, all_features):
            self._cache_features(cache_key, all_features)
        
        if hasattr(self, '_enable_performance_metrics') and self._enable_performance_metrics:
            self._update_performance_metrics(processing_time, len(points))
        
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
                    # Normalize RGB from [0, 255] to [0, 1] for consistency
                    all_features['rgb'] = rgb.astype(np.float32) / 255.0
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
                    # Normalize NIR from [0, 255] to [0, 1] for consistency
                    all_features['nir'] = nir.astype(np.float32) / 255.0
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
    
    def __del__(self):
        """Cleanup resources on deletion."""
        try:
            if hasattr(self, '_rgb_nir_executor'):
                self._rgb_nir_executor.shutdown(wait=False)
            if hasattr(self, '_feature_executor'):
                self._feature_executor.shutdown(wait=False)
        except Exception:
            pass  # Ignore cleanup errors
