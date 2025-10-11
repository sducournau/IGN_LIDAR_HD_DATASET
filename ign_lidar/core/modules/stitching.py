"""
Tile Stitching Module

This module provides a clean interface to tile stitching functionality for
seamless feature computation at tile boundaries.

The core TileStitcher class is implemented in tile_stitcher.py. This module
provides a simplified interface and helper functions for common stitching operations.

Extracted as part of Phase 4 refactoring for consistent module organization.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Import the main TileStitcher class
from ..tile_stitcher import TileStitcher

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Dataclass
# ============================================================================

@dataclass
class StitchingConfig:
    """Configuration for tile stitching operations.
    
    Attributes:
        enabled: Whether stitching is enabled
        buffer_size: Buffer zone width in meters
        adaptive_buffer: Whether to use adaptive buffer sizing
        min_buffer: Minimum buffer size in meters
        max_buffer: Maximum buffer size in meters
        auto_detect_neighbors: Automatically detect neighboring tiles
        parallel_loading: Load neighbor tiles in parallel
        boundary_smoothing: Apply smoothing at tile boundaries
        verbose_logging: Enable detailed logging
    """
    enabled: bool = False
    buffer_size: float = 15.0
    adaptive_buffer: bool = True
    min_buffer: float = 5.0
    max_buffer: float = 25.0
    auto_detect_neighbors: bool = True
    parallel_loading: bool = False
    boundary_smoothing: bool = False
    verbose_logging: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for TileStitcher."""
        return {
            'buffer_size': self.buffer_size,
            'adaptive_buffer': self.adaptive_buffer,
            'min_buffer': self.min_buffer,
            'max_buffer': self.max_buffer,
            'auto_detect_neighbors': self.auto_detect_neighbors,
            'parallel_loading': self.parallel_loading,
            'boundary_smoothing': self.boundary_smoothing,
            'verbose_logging': self.verbose_logging,
            'enable_caching': True  # Always enable caching
        }


# ============================================================================
# Helper Functions
# ============================================================================

def create_stitcher(
    config: Optional[StitchingConfig] = None,
    buffer_size: float = 15.0,
    enable_caching: bool = True
) -> Optional[TileStitcher]:
    """Create a TileStitcher instance with configuration.
    
    Args:
        config: StitchingConfig instance (preferred)
        buffer_size: Buffer size in meters (fallback if no config)
        enable_caching: Whether to enable caching (fallback if no config)
        
    Returns:
        TileStitcher instance, or None if stitching disabled
    """
    if config is not None:
        if not config.enabled:
            return None
        return TileStitcher(config=config.to_dict())
    else:
        # Fallback: create with basic parameters
        return TileStitcher(buffer_size=buffer_size, enable_caching=enable_caching)


def check_neighbors_available(
    stitcher: TileStitcher,
    laz_file: Path,
    logger_instance: Optional[logging.Logger] = None
) -> bool:
    """Check if neighboring tiles are available for stitching.
    
    Args:
        stitcher: TileStitcher instance
        laz_file: Path to the LAZ file
        logger_instance: Optional logger instance
        
    Returns:
        True if neighbors exist, False otherwise
    """
    log = logger_instance or logger
    
    if stitcher is None:
        return False
    
    try:
        return stitcher.check_neighbors_exist(laz_file)
    except Exception as e:
        log.warning(f"  ⚠️  Error checking neighbors: {e}")
        return False


def compute_boundary_aware_features(
    stitcher: TileStitcher,
    laz_file: Path,
    k_neighbors: int = 20,
    logger_instance: Optional[logging.Logger] = None
) -> Dict[str, np.ndarray]:
    """Compute features with boundary-aware tile stitching.
    
    This function loads the core tile plus buffer zones from neighboring tiles,
    then computes features with complete neighborhoods even at tile edges.
    
    Args:
        stitcher: TileStitcher instance
        laz_file: Path to the LAZ file
        k_neighbors: Number of neighbors for feature computation
        logger_instance: Optional logger instance
        
    Returns:
        Dictionary containing:
        - 'normals': Surface normals [N, 3]
        - 'curvature': Curvature values [N]
        - 'geometric_features': Dict or array of geometric features
        - 'num_boundary_points': Number of points affected by stitching
        - Additional features depending on stitcher configuration
        
    Raises:
        Exception: If stitching fails
    """
    log = logger_instance or logger
    
    log.info("  🔗 Using tile stitching for boundary features...")
    
    try:
        features = stitcher.compute_boundary_aware_features(
            laz_file=laz_file,
            k=k_neighbors
        )
        
        num_boundary = features.get('num_boundary_points', 0)
        log.info(
            f"  ✓ Boundary-aware features computed "
            f"({num_boundary} boundary points affected)"
        )
        
        return features
        
    except Exception as e:
        log.error(f"  ❌ Tile stitching failed: {e}")
        raise


def extract_and_normalize_features(
    features: Dict[str, Any],
    points: np.ndarray,
    logger_instance: Optional[logging.Logger] = None
) -> Dict[str, np.ndarray]:
    """Extract and normalize features from stitcher output.
    
    Handles different feature formats and converts to standardized dictionary.
    
    Args:
        features: Raw features from stitcher
        points: Point cloud [N, 3] for height computation
        logger_instance: Optional logger instance
        
    Returns:
        Dictionary with standardized features:
        - 'normals': [N, 3]
        - 'curvature': [N]
        - 'height': [N]
        - 'geo_features': Dict with 'planarity', 'linearity', etc.
    """
    log = logger_instance or logger
    
    result = {}
    
    # Extract normals
    if 'normals' in features:
        result['normals'] = features['normals']
    else:
        log.warning("  ⚠️  No normals in stitcher output")
    
    # Extract curvature
    if 'curvature' in features:
        result['curvature'] = features['curvature']
    else:
        log.warning("  ⚠️  No curvature in stitcher output")
    
    # Extract or compute height
    if 'height' in features:
        result['height'] = features['height']
    else:
        # Fallback: compute from points
        result['height'] = points[:, 2] - points[:, 2].min()
    
    # Extract geometric features with format normalization
    if 'geometric_features' in features:
        geo_dict = features['geometric_features']
        
        # Convert to standardized dict format
        if isinstance(geo_dict, dict):
            result['geo_features'] = geo_dict
        elif isinstance(geo_dict, np.ndarray):
            # Assume array format: [planarity, linearity, sphericity, verticality]
            normals = result.get('normals')
            result['geo_features'] = {
                'planarity': geo_dict[:, 0],
                'linearity': geo_dict[:, 1],
                'sphericity': geo_dict[:, 2],
                'verticality': (geo_dict[:, 3] if geo_dict.shape[1] > 3 
                               else np.abs(normals[:, 2]) if normals is not None
                               else None)
            }
        else:
            log.warning(f"  ⚠️  Unknown geometric features format: {type(geo_dict)}")
    else:
        log.debug("  ℹ️  No geometric features in stitcher output")
    
    return result


def should_use_stitching(
    stitcher: Optional[TileStitcher],
    laz_file: Path,
    logger_instance: Optional[logging.Logger] = None
) -> bool:
    """Determine if stitching should be used for this tile.
    
    Checks if:
    1. Stitcher is available
    2. Neighboring tiles exist
    3. Stitching is worth the overhead
    
    Args:
        stitcher: TileStitcher instance (or None)
        laz_file: Path to the LAZ file
        logger_instance: Optional logger instance
        
    Returns:
        True if stitching should be used, False otherwise
    """
    log = logger_instance or logger
    
    # No stitcher available
    if stitcher is None:
        return False
    
    # Check if neighbors exist
    neighbors_exist = check_neighbors_available(stitcher, laz_file, log)
    if not neighbors_exist:
        log.debug(f"  ℹ️  No neighbors found for {laz_file.name}, using standard processing")
        return False
    
    return True


def get_stitching_stats(features: Dict[str, Any]) -> Dict[str, Any]:
    """Extract statistics from stitching results.
    
    Args:
        features: Features dictionary from stitcher
        
    Returns:
        Dictionary with statistics:
        - 'num_boundary_points': Points affected by stitching
        - 'boundary_ratio': Ratio of boundary to total points
        - 'total_points': Total points processed
    """
    stats = {}
    
    num_boundary = features.get('num_boundary_points', 0)
    stats['num_boundary_points'] = num_boundary
    
    # Try to get total points from features
    if 'normals' in features:
        total_points = len(features['normals'])
        stats['total_points'] = total_points
        stats['boundary_ratio'] = num_boundary / total_points if total_points > 0 else 0
    elif 'points' in features:
        total_points = len(features['points'])
        stats['total_points'] = total_points
        stats['boundary_ratio'] = num_boundary / total_points if total_points > 0 else 0
    else:
        stats['total_points'] = None
        stats['boundary_ratio'] = None
    
    return stats


# ============================================================================
# Export List
# ============================================================================

__all__ = [
    # Main class (re-exported)
    'TileStitcher',
    
    # Configuration
    'StitchingConfig',
    
    # Helper functions
    'create_stitcher',
    'check_neighbors_available',
    'compute_boundary_aware_features',
    'extract_and_normalize_features',
    'should_use_stitching',
    'get_stitching_stats',
]
