"""
Transport Classification Module

This module provides comprehensive transport detection and enhancement for
LiDAR point cloud classification, supporting roads and railways with multiple
detection modes and advanced buffering strategies.

Features:
- Multi-mode detection (ASPRS standard/extended, LOD2)
- Adaptive buffering based on curvature
- Spatial indexing for fast classification (R-tree)
- Quality metrics and confidence scoring
- Type-specific tolerances for different road/railway types

Author: Transport Module Consolidation (Phase 3)
Date: October 22, 2025
Version: 3.1.0

Example:
    Basic transport detection (ASPRS standard mode):
    
    >>> from ign_lidar.core.classification.transport import (
    ...     TransportDetector, DetectionConfig, TransportMode
    ... )
    >>> 
    >>> config = DetectionConfig(mode=TransportMode.ASPRS_STANDARD)
    >>> detector = TransportDetector(config)
    >>> result = detector.detect_transport(
    ...     labels=labels,
    ...     height=height,
    ...     planarity=planarity,
    ...     road_ground_truth_mask=road_mask
    ... )
    >>> print(result.get_summary())
    
    Advanced buffering with curvature awareness:
    
    >>> from ign_lidar.core.classification.transport import (
    ...     AdaptiveTransportBuffer, BufferingConfig
    ... )
    >>> 
    >>> config = BufferingConfig(curvature_aware=True, curvature_factor=0.3)
    >>> buffer = AdaptiveTransportBuffer(config)
    >>> enhanced_roads = buffer.process_roads(roads_gdf)
    
    Fast spatial classification with R-tree indexing:
    
    >>> from ign_lidar.core.classification.transport import (
    ...     SpatialTransportClassifier, IndexingConfig
    ... )
    >>> 
    >>> config = IndexingConfig(enabled=True, cache_index=True)
    >>> classifier = SpatialTransportClassifier(config)
    >>> classifier.index_roads(roads_gdf)
    >>> labels = classifier.classify_points_fast(points, labels)
"""

# ============================================================================
# Public API Exports
# ============================================================================

# Base classes and enums
from .base import (
    # Enums
    TransportMode,
    TransportType,
    DetectionStrategy,
    
    # Configuration classes
    TransportConfigBase,
    DetectionConfig,
    BufferingConfig,
    IndexingConfig,
    QualityMetricsConfig,
    
    # Result types
    TransportStats,
    TransportDetectionResult,
    
    # Abstract base classes
    TransportDetectorBase,
    TransportBufferBase,
    TransportClassifierBase,
)

# Utility functions
from .utils import (
    # Validation functions
    validate_transport_height,
    check_transport_planarity,
    filter_by_roughness,
    filter_by_intensity,
    check_horizontality,
    
    # Curvature functions
    calculate_curvature,
    compute_adaptive_width,
    
    # Type-specific functions
    get_road_type_tolerance,
    get_railway_type_tolerance,
    
    # Intersection detection
    detect_intersections,
    
    # Geometric helpers
    create_adaptive_buffer,
    calculate_distance_to_centerline,
    
    # Availability flags
    HAS_SHAPELY,
    HAS_SCIPY,
)

# Import implementation classes (migrated in Phase 3C)
from .detection import (
    TransportDetector,
    detect_transport_multi_mode,
)

from .enhancement import (
    AdaptiveTransportBuffer,
    SpatialTransportClassifier,
    TransportClassificationScore,
    TransportCoverageStats,
)


# ============================================================================
# Version and Module Info
# ============================================================================

__version__ = "3.1.0"
__author__ = "Transport Module Consolidation (Phase 3)"
__date__ = "October 22, 2025"


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # === Enums ===
    'TransportMode',
    'TransportType',
    'DetectionStrategy',
    
    # === Configuration Classes ===
    'TransportConfigBase',
    'DetectionConfig',
    'BufferingConfig',
    'IndexingConfig',
    'QualityMetricsConfig',
    
    # === Result Types ===
    'TransportStats',
    'TransportDetectionResult',
    'TransportClassificationScore',
    'TransportCoverageStats',
    
    # === Abstract Base Classes ===
    'TransportDetectorBase',
    'TransportBufferBase',
    'TransportClassifierBase',
    
    # === Detection Classes (Phase 3C) ===
    'TransportDetector',
    'detect_transport_multi_mode',
    
    # === Enhancement Classes (Phase 3C) ===
    'AdaptiveTransportBuffer',
    'SpatialTransportClassifier',
    
    # === Utility Functions ===
    'validate_transport_height',
    'check_transport_planarity',
    'filter_by_roughness',
    'filter_by_intensity',
    'check_horizontality',
    'calculate_curvature',
    'compute_adaptive_width',
    'get_road_type_tolerance',
    'get_railway_type_tolerance',
    'detect_intersections',
    'create_adaptive_buffer',
    'calculate_distance_to_centerline',
    
    # === Availability Flags ===
    'HAS_SHAPELY',
    'HAS_SCIPY',
]


# ============================================================================
# Module Status
# ============================================================================

def get_module_status() -> dict:
    """
    Get current module status and availability.
    
    Returns:
        Dictionary with module component availability
    """
    return {
        'version': __version__,
        'detection_available': True,  # Migrated in Phase 3C
        'enhancement_available': True,  # Migrated in Phase 3C
        'shapely_available': HAS_SHAPELY,
        'scipy_available': HAS_SCIPY,
        'phase_3_complete': True,  # Phase 3C migration complete
    }
