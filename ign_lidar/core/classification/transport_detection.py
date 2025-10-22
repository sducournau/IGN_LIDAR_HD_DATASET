"""
Transport Detection Module - Backward Compatibility Wrapper

**DEPRECATED:** This module has been migrated to the `transport` subdirectory.
Please update your imports:

    OLD: from ign_lidar.core.classification.transport_detection import TransportDetector
    NEW: from ign_lidar.core.classification.transport import TransportDetector

This compatibility wrapper will be removed in v4.0.0 (mid-2026).

For migration guide, see: docs/TRANSPORT_MODULE_MIGRATION_GUIDE.md

Author: Transport Detection Enhancement
Date: October 15, 2025
Updated: October 22, 2025 - Deprecated, migrated to transport/ (Phase 3C)
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "transport_detection module is deprecated and will be removed in v4.0.0. "
    "Use 'from ign_lidar.core.classification.transport import TransportDetector' instead. "
    "See docs/TRANSPORT_MODULE_MIGRATION_GUIDE.md for details.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from new location
from .transport.detection import *  # noqa: F401, F403
from .transport.base import TransportMode as TransportDetectionMode  # noqa: F401
from .transport.base import DetectionConfig as TransportDetectionConfig  # noqa: F401

# Maintain backward compatibility for old names
__all__ = [
    'TransportDetectionMode',
    'TransportDetectionConfig',
    'TransportDetector',
    'detect_transport_multi_mode',
]
