"""
Configuration module for IGN LiDAR HD v2.0.

Provides Hydra-based structured configuration with:
- Type-safe configuration schemas
- Hierarchical composition
- Command-line overrides
- Validation
"""

from .schema import (
    ProcessorConfig,
    FeaturesConfig,
    PreprocessConfig,
    StitchingConfig,
    OutputConfig,
    IGNLiDARConfig,
)
from .defaults import DEFAULT_CONFIG

__all__ = [
    "ProcessorConfig",
    "FeaturesConfig",
    "PreprocessConfig",
    "StitchingConfig",
    "OutputConfig",
    "IGNLiDARConfig",
    "DEFAULT_CONFIG",
]
