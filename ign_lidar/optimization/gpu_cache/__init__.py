"""
GPU Cache & Memory Optimization

GPU array caching, transfer optimization, and memory pool management.

Reorganized in v3.3.0 for better clarity:
- arrays.py: GPUArrayCache
- transfer.py: TransferOptimizer, GPUMemoryPool, utility functions

Version: 1.0.0
"""

from .arrays import GPUArrayCache
from .transfer import (
    TransferOptimizer,
    GPUMemoryPool,
    estimate_gpu_memory_for_features,
    optimize_chunk_size_for_vram
)

__all__ = [
    'GPUArrayCache',
    'TransferOptimizer',
    'GPUMemoryPool',
    'estimate_gpu_memory_for_features',
    'optimize_chunk_size_for_vram',
]
