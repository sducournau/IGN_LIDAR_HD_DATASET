#!/usr/bin/env python3
"""
Quick test of the fast normal computation optimization.

Tests a single small batch to verify the optimization is working.
"""

import sys
import time
import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    print("❌ CuPy not available - cannot test GPU optimizations")
    sys.exit(1)

# Import the optimized computer
sys.path.insert(0, '/mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET')
from ign_lidar.features.features_gpu_chunked import GPUChunkedFeatureComputer

def main():
    print("=" * 70)
    print("Quick GPU Normal Optimization Test")
    print("=" * 70)
    
    # Generate test points
    n_points = 100000
    print(f"\n📝 Generating {n_points:,} test points...")
    np.random.seed(42)
    points = np.random.randn(n_points, 3).astype(np.float32)
    points[:, 2] = 0.1 * points[:, 0] + 0.2 * points[:, 1]
    
    # Create GPU computer
    print("🚀 Initializing GPU chunked computer...")
    computer = GPUChunkedFeatureComputer(
        chunk_size=2_000_000,
        use_gpu=True,
        show_progress=True,
        auto_optimize=True
    )
    
    # Test normal computation
    print(f"\n⚡ Computing normals with k=20...")
    start = time.time()
    normals = computer.compute_normals_chunked(points, k=20)
    elapsed = time.time() - start
    
    print(f"\n{'=' * 70}")
    print(f"✅ Success!")
    print(f"   Points processed: {n_points:,}")
    print(f"   Time:            {elapsed:.3f}s")
    print(f"   Throughput:      {n_points/elapsed:,.0f} points/sec")
    print(f"   Normal shape:    {normals.shape}")
    print(f"   Normal range:    [{normals.min():.3f}, {normals.max():.3f}]")
    print(f"{'=' * 70}")
    
    # Verify normals are normalized
    norms = np.linalg.norm(normals, axis=1)
    print(f"\n🔍 Verification:")
    print(f"   Mean norm:       {norms.mean():.6f} (should be ~1.0)")
    print(f"   Min/max norm:    [{norms.min():.6f}, {norms.max():.6f}]")
    print(f"   Upward oriented: {(normals[:, 2] >= 0).sum() / len(normals) * 100:.1f}%")
    
    if abs(norms.mean() - 1.0) < 0.01:
        print("\n✅ Normals are properly normalized!")
    else:
        print("\n⚠️  Warning: Normals may not be properly normalized")

if __name__ == "__main__":
    main()
