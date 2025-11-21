"""
Configuration integration for tile stitching.

This module provides configuration loaders and validators for the advanced
tile stitching functionality.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class StitchingConfigManager:
    """
    Manager for stitching configurations with validation and presets.
    """
    
    PRESETS = {
        'disabled': {
            'enabled': False,
            'buffer_size': 0.0
        },
        
        'basic': {
            'enabled': True,
            'buffer_size': 10.0,
            'auto_detect_neighbors': True,
            'cache_enabled': True
        },
        
        'standard': {
            'enabled': True,
            'buffer_size': 15.0,
            'adaptive_buffer': True,
            'min_buffer': 5.0,
            'max_buffer': 25.0,
            'auto_detect_neighbors': True,
            'neighbor_search_radius': 50.0,
            'max_neighbors': 8,
            'use_grid_pattern': True,
            'cache_enabled': True,
            'cache_size': 1000,
            'parallel_loading': True,
            'boundary_smoothing': True,
            'edge_artifact_removal': True,
            'compute_boundary_features': True,
            'cross_tile_neighborhoods': True
        },
        
        'advanced': {
            'enabled': True,
            'buffer_size': 20.0,
            'multi_scale_buffers': True,
            'geometric_buffer': 10.0,
            'contextual_buffer': 20.0,
            'semantic_buffer': 30.0,
            'auto_detect_neighbors': True,
            'neighbor_detection_method': 'smart_grid',
            'grid_tile_size': 1000.0,
            'spatial_index_type': 'rtree',
            'cache_enabled': True,
            'cache_strategy': 'lru',
            'memory_cache_size': 2048,
            'disk_cache_enabled': True,
            'cache_compression': True,
            'parallel_loading': True,
            'max_parallel_tiles': 4,
            'worker_threads': 8,
            'boundary_detection_method': 'adaptive',
            'boundary_smoothing_kernel': 'gaussian',
            'smoothing_sigma': 1.0,
            'edge_preservation': True,
            'cross_tile_knn': True,
            'boundary_feature_enhancement': True,
            'multi_resolution_features': True,
            'overlap_validation': True,
            'gap_detection': True,
            'continuity_checks': True,
            'quality_metrics': True
        }
    }
    
    @classmethod
    def load_config(cls, config_name: str) -> Dict[str, Any]:
        """
        Load a stitching configuration by name.
        
        Args:
            config_name: Configuration name ('disabled', 'basic', 'standard', 'advanced')
            
        Returns:
            Configuration dictionary
        """
        if config_name in cls.PRESETS:
            config = cls.PRESETS[config_name].copy()
            logger.info(f"Loaded stitching preset: {config_name}")
            return config
        else:
            logger.warning(f"Unknown stitching preset: {config_name}, using 'basic'")
            return cls.PRESETS['basic'].copy()
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize stitching configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Validated configuration
        """
        validated = config.copy()
        
        # Required fields with defaults
        defaults = {
            'enabled': True,
            'buffer_size': 10.0,
            'auto_detect_neighbors': True,
            'cache_enabled': True
        }
        
        for key, default_value in defaults.items():
            if key not in validated:
                validated[key] = default_value
                logger.debug(f"Added default value for {key}: {default_value}")
        
        # Validate numeric ranges
        if validated.get('buffer_size', 0) < 0:
            validated['buffer_size'] = 10.0
            logger.warning("buffer_size must be >= 0, set to 10.0")
        
        if validated.get('max_neighbors', 0) < 0:
            validated['max_neighbors'] = 8
            logger.warning("max_neighbors must be >= 0, set to 8")
        
        if validated.get('cache_size', 0) < 0:
            validated['cache_size'] = 1000
            logger.warning("cache_size must be >= 0, set to 1000")
        
        # Validate adaptive buffer settings
        if validated.get('adaptive_buffer', False):
            min_buffer = validated.get('min_buffer', 5.0)
            max_buffer = validated.get('max_buffer', 25.0)
            base_buffer = validated.get('buffer_size', 10.0)
            
            if min_buffer > max_buffer:
                validated['min_buffer'] = 5.0
                validated['max_buffer'] = 25.0
                logger.warning("min_buffer > max_buffer, reset to defaults")
            
            if base_buffer < min_buffer or base_buffer > max_buffer:
                validated['buffer_size'] = (min_buffer + max_buffer) / 2
                logger.warning(f"buffer_size outside min/max range, set to {validated['buffer_size']}")
        
        return validated
    
    @classmethod
    def create_hybrid_config(cls, base_preset: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a hybrid configuration by combining a preset with overrides.
        
        Args:
            base_preset: Base preset name
            overrides: Dictionary of values to override
            
        Returns:
            Hybrid configuration
        """
        config = cls.load_config(base_preset)
        config.update(overrides)
        return cls.validate_config(config)


def get_stitching_config_for_processor(
    stitching_preset: str = 'standard',
    custom_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get stitching configuration optimized for the processor.
    
    Args:
        stitching_preset: Preset to use as base
        custom_overrides: Custom parameter overrides
        
    Returns:
        Processor-ready stitching configuration
    """
    manager = StitchingConfigManager()
    
    if custom_overrides:
        config = manager.create_hybrid_config(stitching_preset, custom_overrides)
    else:
        config = manager.load_config(stitching_preset)
    
    # Add processor-specific optimizations
    config['processor_optimized'] = True
    config['verbose_logging'] = True
    
    return config


def get_recommended_stitching_preset(
    tile_count: int,
    memory_gb: float,
    use_gpu: bool = False
) -> str:
    """
    Get recommended stitching preset based on processing parameters.
    
    Args:
        tile_count: Number of tiles to process
        memory_gb: Available memory in GB
        use_gpu: Whether GPU processing is enabled
        
    Returns:
        Recommended preset name
    """
    if tile_count == 1:
        return 'disabled'  # Single tile doesn't need stitching
    
    if memory_gb < 4:
        return 'basic'  # Conservative for low memory
    
    if tile_count > 100 and memory_gb >= 16:
        return 'advanced'  # High performance for large datasets
    
    return 'standard'  # Good balance for most cases