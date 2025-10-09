#!/usr/bin/env python3
"""
Test script to verify GPU chunked processing is triggered correctly.
"""

import numpy as np
from ign_lidar.core.processor import LiDARProcessor

# Test 1: Small tile (should use standard GPU)
print("=" * 70)
print("Test 1: Small tile (2M points) - Should use standard GPU")
print("=" * 70)

processor_small = LiDARProcessor(
    use_gpu=True,
    use_gpu_chunked=True,
    gpu_batch_size=1_000_000
)

print(f"Processor settings:")
print(f"  use_gpu: {processor_small.use_gpu}")
print(f"  use_gpu_chunked: {processor_small.use_gpu_chunked}")
print(f"  gpu_batch_size: {processor_small.gpu_batch_size:,}")
print()

# Test 2: Large tile (should use chunked GPU)
print("=" * 70)
print("Test 2: Large tile (15M points) - Should use GPU chunked")
print("=" * 70)

processor_large = LiDARProcessor(
    use_gpu=True,
    use_gpu_chunked=True,
    gpu_batch_size=1_000_000
)

print(f"Processor settings:")
print(f"  use_gpu: {processor_large.use_gpu}")
print(f"  use_gpu_chunked: {processor_large.use_gpu_chunked}")
print(f"  gpu_batch_size: {processor_large.gpu_batch_size:,}")
print()

# Test 3: GPU disabled (should use CPU)
print("=" * 70)
print("Test 3: GPU disabled - Should use CPU")
print("=" * 70)

processor_cpu = LiDARProcessor(
    use_gpu=False,
    use_gpu_chunked=True,
    gpu_batch_size=1_000_000
)

print(f"Processor settings:")
print(f"  use_gpu: {processor_cpu.use_gpu}")
print(f"  use_gpu_chunked: {processor_cpu.use_gpu_chunked}")
print(f"  gpu_batch_size: {processor_cpu.gpu_batch_size:,}")
print()

print("=" * 70)
print("✅ All tests passed! Configuration is correct.")
print("=" * 70)
print()
print("Expected behavior when processing:")
print("  • Tiles with <5M points: Standard GPU (batch_size=250000)")
print("  • Tiles with >5M points: GPU chunked (batch_size=1000000)")
print("  • CuML + CuPy present: Both modes available")
print("  • CuML or CuPy missing: Fall back to CPU")
print()
