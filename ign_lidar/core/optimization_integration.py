"""
Phase 4 Optimization Integration Module

This module provides integrated access to all Phase 4 optimizations:
- Phase 4.1: WFS Memory Cache
- Phase 4.2: Preprocessing GPU Pipeline  
- Phase 4.3: GPU Memory Pooling
- Phase 4.4: Batch Multi-Tile Processing
- Phase 4.5: Async I/O Pipeline

Author: IGN LiDAR HD Development Team
Date: November 23, 2025
Version: 3.5.0
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try to import async I/O components (Phase 4.5)
try:
    from ..io.async_loader import AsyncTileLoader, AsyncPipeline
    ASYNC_IO_AVAILABLE = True
except ImportError:
    ASYNC_IO_AVAILABLE = False
    logger.warning("⚠️  Async I/O (Phase 4.5) not available")

# Try to import GPU components (Phase 4.3, 4.4)
try:
    from ..optimization.gpu_memory import GPUMemoryPool
    GPU_MEMORY_POOL_AVAILABLE = True
except ImportError:
    GPU_MEMORY_POOL_AVAILABLE = False
    logger.warning("⚠️  GPU Memory Pooling (Phase 4.3) not available")

try:
    from ..features.gpu_processor import GPUProcessor
    GPU_PROCESSOR_AVAILABLE = True
except ImportError:
    GPU_PROCESSOR_AVAILABLE = False
    logger.warning("⚠️  GPU Processor (Phase 4.4) not available")


class OptimizationManager:
    """
    Centralized manager for all Phase 4 optimizations.
    
    Provides a unified interface to enable/disable optimizations and
    coordinate their usage across the processing pipeline.
    
    Features:
    - Phase 4.1: WFS Memory Cache (automatic via WFSGroundTruth)
    - Phase 4.2: GPU Preprocessing (automatic via preprocessing module)
    - Phase 4.3: GPU Memory Pooling (managed here)
    - Phase 4.4: Batch Multi-Tile Processing (managed here)
    - Phase 4.5: Async I/O Pipeline (managed here)
    
    Usage:
        >>> opt_mgr = OptimizationManager(
        ...     enable_async_io=True,
        ...     enable_batch_processing=True,
        ...     enable_gpu_pooling=True
        ... )
        >>> opt_mgr.initialize()
        >>> 
        >>> # Process tiles with all optimizations
        >>> results = opt_mgr.process_tiles_optimized(
        ...     tile_paths=tiles,
        ...     processor_func=my_processor
        ... )
    """
    
    def __init__(
        self,
        enable_async_io: bool = True,
        enable_batch_processing: bool = True,
        enable_gpu_pooling: bool = True,
        async_workers: int = 2,
        tile_cache_size: int = 3,
        batch_size: int = 4,
        gpu_pool_max_size_gb: float = 4.0,
        show_progress: bool = True,
    ):
        """
        Initialize optimization manager.
        
        Args:
            enable_async_io: Enable Phase 4.5 async I/O pipeline
            enable_batch_processing: Enable Phase 4.4 batch multi-tile processing
            enable_gpu_pooling: Enable Phase 4.3 GPU memory pooling
            async_workers: Number of background I/O threads (Phase 4.5)
            tile_cache_size: Tile LRU cache size (Phase 4.5)
            batch_size: Number of tiles per batch (Phase 4.4)
            gpu_pool_max_size_gb: GPU memory pool size limit (Phase 4.3)
            show_progress: Show progress bars
        """
        self.enable_async_io = enable_async_io and ASYNC_IO_AVAILABLE
        self.enable_batch_processing = enable_batch_processing and GPU_PROCESSOR_AVAILABLE
        self.enable_gpu_pooling = enable_gpu_pooling and GPU_MEMORY_POOL_AVAILABLE
        
        # Configuration
        self.async_workers = async_workers
        self.tile_cache_size = tile_cache_size
        self.batch_size = batch_size
        self.gpu_pool_max_size_gb = gpu_pool_max_size_gb
        self.show_progress = show_progress
        
        # Components (initialized later)
        self.async_pipeline = None
        self.gpu_pool = None
        self.gpu_processor = None
        
        # Statistics
        self.stats = {
            'tiles_processed': 0,
            'batches_processed': 0,
            'async_cache_hits': 0,
            'async_cache_misses': 0,
            'gpu_pool_hits': 0,
            'gpu_pool_misses': 0,
        }
        
        # Log enabled optimizations
        self._log_status()
    
    def _log_status(self):
        """Log enabled optimization status."""
        logger.info("🚀 Phase 4 Optimization Manager")
        logger.info("=" * 60)
        
        if self.enable_async_io:
            logger.info("✅ Phase 4.5: Async I/O Pipeline ENABLED")
            logger.info(f"   ├─ Workers: {self.async_workers}")
            logger.info(f"   └─ Cache size: {self.tile_cache_size} tiles")
        else:
            logger.info("❌ Phase 4.5: Async I/O Pipeline DISABLED")
        
        if self.enable_batch_processing:
            logger.info("✅ Phase 4.4: Batch Multi-Tile ENABLED")
            logger.info(f"   └─ Batch size: {self.batch_size} tiles")
        else:
            logger.info("❌ Phase 4.4: Batch Multi-Tile DISABLED")
        
        if self.enable_gpu_pooling:
            logger.info("✅ Phase 4.3: GPU Memory Pooling ENABLED")
            logger.info(f"   └─ Pool size: {self.gpu_pool_max_size_gb:.1f} GB")
        else:
            logger.info("❌ Phase 4.3: GPU Memory Pooling DISABLED")
        
        logger.info("=" * 60)
    
    def initialize(self, feature_orchestrator=None):
        """
        Initialize optimization components.
        
        Args:
            feature_orchestrator: FeatureOrchestrator instance for GPU processor
        """
        # Initialize async I/O pipeline (Phase 4.5)
        if self.enable_async_io:
            logger.info("🔄 Initializing Async I/O Pipeline...")
            self.async_pipeline = AsyncPipeline(
                num_workers=self.async_workers,
                enable_wfs=True,
                cache_size=self.tile_cache_size,
                show_progress=self.show_progress,
            )
            logger.info("✅ Async I/O Pipeline initialized")
        
        # Initialize GPU memory pool (Phase 4.3)
        if self.enable_gpu_pooling:
            logger.info("🔄 Initializing GPU Memory Pool...")
            try:
                self.gpu_pool = GPUMemoryPool(
                    max_size_gb=self.gpu_pool_max_size_gb,
                    enable_stats=True,
                )
                logger.info("✅ GPU Memory Pool initialized")
            except Exception as e:
                logger.warning(f"⚠️  GPU Memory Pool initialization failed: {e}")
                self.enable_gpu_pooling = False
        
        # Initialize GPU processor for batch processing (Phase 4.4)
        if self.enable_batch_processing and feature_orchestrator:
            logger.info("🔄 Initializing GPU Batch Processor...")
            try:
                # Get GPU processor from feature orchestrator
                if hasattr(feature_orchestrator, 'gpu_processor'):
                    self.gpu_processor = feature_orchestrator.gpu_processor
                    
                    # Integrate memory pool if available
                    if self.gpu_pool and hasattr(self.gpu_processor, 'gpu_pool'):
                        self.gpu_processor.gpu_pool = self.gpu_pool
                        logger.info("✅ GPU Memory Pool integrated with GPU Processor")
                    
                    logger.info("✅ GPU Batch Processor initialized")
                else:
                    logger.warning("⚠️  GPU processor not available in feature orchestrator")
                    self.enable_batch_processing = False
            except Exception as e:
                logger.warning(f"⚠️  GPU Batch Processor initialization failed: {e}")
                self.enable_batch_processing = False
        
        logger.info("🎯 Optimization Manager ready")
    
    def process_tiles_optimized(
        self,
        tile_paths: List[Path],
        processor_func: callable,
        fetch_ground_truth: bool = True,
    ) -> List[Dict]:
        """
        Process tiles with all enabled optimizations.
        
        This method orchestrates:
        - Async I/O (Phase 4.5): Load tiles in background
        - Batch processing (Phase 4.4): Process multiple tiles in GPU batch
        - GPU pooling (Phase 4.3): Reuse GPU memory allocations
        
        Args:
            tile_paths: List of tile file paths
            processor_func: Function(tile_data, ground_truth) -> result
            fetch_ground_truth: Enable WFS ground truth fetching
        
        Returns:
            List of processing results
        """
        results = []
        
        if self.enable_async_io and self.async_pipeline:
            # Use async I/O pipeline (Phase 4.5)
            logger.info(f"🚀 Processing {len(tile_paths)} tiles with Async I/O Pipeline")
            results = self.async_pipeline.process_tiles(
                tile_paths=tile_paths,
                processor_func=processor_func,
                fetch_ground_truth=fetch_ground_truth,
            )
            
            # Update statistics
            loader_stats = self.async_pipeline.loader.get_stats()
            self.stats['async_cache_hits'] += loader_stats.get('cache_hits', 0)
            self.stats['async_cache_misses'] += loader_stats.get('cache_misses', 0)
        
        elif self.enable_batch_processing and self.gpu_processor:
            # Use batch processing without async I/O (Phase 4.4)
            logger.info(f"🚀 Processing {len(tile_paths)} tiles with Batch Processing")
            
            # Process in batches
            for i in range(0, len(tile_paths), self.batch_size):
                batch_paths = tile_paths[i:i + self.batch_size]
                
                # Load tiles synchronously
                tile_data_list = []
                for tile_path in batch_paths:
                    from ..core.classification.io import load_laz_file
                    tile_data = load_laz_file(tile_path)
                    tile_data_list.append(tile_data)
                
                # Process batch (if processor supports it)
                if hasattr(self.gpu_processor, 'process_tile_batch'):
                    batch_results = self.gpu_processor.process_tile_batch(
                        tile_data_list,
                        k=20,  # TODO: Make configurable
                    )
                    results.extend(batch_results)
                else:
                    # Fall back to sequential
                    for tile_data in tile_data_list:
                        result = processor_func(tile_data, None)
                        results.append(result)
                
                self.stats['batches_processed'] += 1
        
        else:
            # Sequential processing (no optimizations)
            logger.info(f"🚀 Processing {len(tile_paths)} tiles sequentially")
            for tile_path in tile_paths:
                from ..core.classification.io import load_laz_file
                tile_data = load_laz_file(tile_path)
                result = processor_func(tile_data, None)
                results.append(result)
        
        self.stats['tiles_processed'] += len(tile_paths)
        
        return results
    
    def get_stats(self) -> Dict:
        """Get optimization statistics."""
        stats = self.stats.copy()
        
        # Add async I/O stats
        if self.async_pipeline:
            loader_stats = self.async_pipeline.loader.get_stats()
            stats.update({
                'async_tiles_loaded': loader_stats.get('tiles_loaded', 0),
                'async_wfs_fetches': loader_stats.get('wfs_fetches', 0),
                'async_cache_hit_rate': loader_stats.get('cache_hit_rate', 0.0),
                'async_avg_io_time_ms': loader_stats.get('avg_io_time', 0.0) * 1000,
                'async_avg_wait_time_ms': loader_stats.get('avg_wait_time', 0.0) * 1000,
            })
        
        # Add GPU pool stats
        if self.gpu_pool:
            pool_stats = self.gpu_pool.get_stats()
            stats.update({
                'gpu_pool_hits': pool_stats.get('hits', 0),
                'gpu_pool_misses': pool_stats.get('misses', 0),
                'gpu_pool_hit_rate': pool_stats.get('hit_rate', 0.0),
                'gpu_pool_current_size_gb': pool_stats.get('current_size_gb', 0.0),
            })
        
        return stats
    
    def print_stats(self):
        """Print optimization statistics."""
        stats = self.get_stats()
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("📊 Phase 4 Optimization Statistics")
        logger.info("=" * 60)
        
        logger.info(f"Tiles Processed: {stats['tiles_processed']}")
        
        if self.enable_async_io:
            logger.info("")
            logger.info("🔄 Async I/O Pipeline (Phase 4.5):")
            logger.info(f"   Cache Hit Rate: {stats.get('async_cache_hit_rate', 0):.1%}")
            logger.info(f"   Avg I/O Time: {stats.get('async_avg_io_time_ms', 0):.1f} ms")
            logger.info(f"   Avg Wait Time: {stats.get('async_avg_wait_time_ms', 0):.1f} ms")
            logger.info(f"   WFS Fetches: {stats.get('async_wfs_fetches', 0)}")
        
        if self.enable_batch_processing:
            logger.info("")
            logger.info("⚡ Batch Multi-Tile (Phase 4.4):")
            logger.info(f"   Batches Processed: {stats['batches_processed']}")
            logger.info(f"   Avg Tiles/Batch: {stats['tiles_processed'] / max(1, stats['batches_processed']):.1f}")
        
        if self.enable_gpu_pooling:
            logger.info("")
            logger.info("💾 GPU Memory Pooling (Phase 4.3):")
            logger.info(f"   Pool Hit Rate: {stats.get('gpu_pool_hit_rate', 0):.1%}")
            logger.info(f"   Current Size: {stats.get('gpu_pool_current_size_gb', 0):.2f} GB")
        
        logger.info("=" * 60)
    
    def shutdown(self):
        """Shutdown optimization components and print final statistics."""
        logger.info("🔚 Shutting down optimization manager...")
        
        # Print final statistics
        self.print_stats()
        
        # Shutdown async I/O
        if self.async_pipeline:
            self.async_pipeline.shutdown()
            logger.info("✅ Async I/O Pipeline shutdown complete")
        
        # Clear GPU pool
        if self.gpu_pool:
            self.gpu_pool.clear()
            logger.info("✅ GPU Memory Pool cleared")
        
        logger.info("✅ Optimization Manager shutdown complete")


# Convenience function for quick setup
def create_optimization_manager(
    use_gpu: bool = True,
    enable_all: bool = True,
    **kwargs
) -> OptimizationManager:
    """
    Create OptimizationManager with sensible defaults.
    
    Args:
        use_gpu: Enable GPU optimizations (Phase 4.3, 4.4)
        enable_all: Enable all available optimizations
        **kwargs: Additional OptimizationManager arguments
    
    Returns:
        Configured OptimizationManager instance
    
    Example:
        >>> opt_mgr = create_optimization_manager(use_gpu=True)
        >>> opt_mgr.initialize(feature_orchestrator)
        >>> results = opt_mgr.process_tiles_optimized(tiles, processor)
    """
    # Default settings
    defaults = {
        'enable_async_io': enable_all,
        'enable_batch_processing': enable_all and use_gpu,
        'enable_gpu_pooling': enable_all and use_gpu,
        'async_workers': 2,
        'tile_cache_size': 3,
        'batch_size': 4,
        'gpu_pool_max_size_gb': 4.0,
        'show_progress': True,
    }
    
    # Override with user kwargs
    defaults.update(kwargs)
    
    return OptimizationManager(**defaults)
