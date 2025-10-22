"""
Transport Enhancement Module - Backward Compatibility Wrapper

**DEPRECATED:** This module has been migrated to the `transport` subdirectory.
Please update your imports:

    OLD: from ign_lidar.core.classification.transport_enhancement import AdaptiveTransportBuffer
    NEW: from ign_lidar.core.classification.transport import AdaptiveTransportBuffer

This compatibility wrapper will be removed in v4.0.0 (mid-2026).

For migration guide, see: docs/TRANSPORT_MODULE_MIGRATION_GUIDE.md

Author: Transport Enhancement Team
Date: October 15, 2025
Updated: October 22, 2025 - Deprecated, migrated to transport/ (Phase 3C)
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "transport_enhancement module is deprecated and will be removed in v4.0.0. "
    "Use 'from ign_lidar.core.classification.transport import AdaptiveTransportBuffer' instead. "
    "See docs/TRANSPORT_MODULE_MIGRATION_GUIDE.md for details.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from new location
from .transport.enhancement import *  # noqa: F401, F403
from .transport.base import BufferingConfig as AdaptiveBufferConfig  # noqa: F401
from .transport.base import IndexingConfig as SpatialIndexConfig  # noqa: F401

# Maintain backward compatibility for old names
__all__ = [
    'AdaptiveBufferConfig',
    'SpatialIndexConfig',
    'QualityMetricsConfig',
    'AdaptiveTransportBuffer',
    'SpatialTransportClassifier',
    'TransportClassificationScore',
    'TransportCoverageStats',
]
