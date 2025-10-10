"""
Formatters for multi-architecture deep learning support.

This module provides formatters to convert IGN LiDAR HD patches
into architecture-specific formats for deep learning.

Supported architectures:
- PointNet++ (Set Abstraction)
- Octree-CNN / OctFormer (Hierarchical)
- Point Transformer / PCT (Attention-based)
- Sparse Convolutions (Voxel-based)
- Hybrid Models (Combinations)
"""

from .multi_arch_formatter import MultiArchitectureFormatter
from .hybrid_formatter import HybridFormatter
from .base_formatter import BaseFormatter

__all__ = [
    'MultiArchitectureFormatter',
    'HybridFormatter',
    'BaseFormatter',
]

__version__ = '2.0.0'
