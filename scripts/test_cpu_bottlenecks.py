#!/usr/bin/env python3
"""
Test script to validate CPU bottlenecks in GPU code.
Measures performance of critical operations.
"""

import time
import numpy as np
from ign_lidar.features.features_gpu import GPUFeatureComputer

def test_curvature_bottleneck():
    """Test curvature computation bottleneck."""
    print("=" * 60)
    print("TEST 1: Curvature Bottleneck")
    print("=" * 60)
    
    computer = GPUFeatureComputer(use_gpu=True)
    
    # Small test (100K points)
    N = 100_000
    points = np.random.rand(N, 3).astype(np.float32)
    
    print(f"\nComputing normals for {N:,} points...")
    start = time.time()
    normals = computer.compute_normals(points, k=20)
    normals_time = time.time() - start
    print(f"✅ Normals: {normals_time:.3f}s")
    
    print(f"\nComputing curvature for {N:,} points...")
    start = time.time()
    curvature = computer.compute_curvature(points, normals, k=20)
    curv_time = time.time() - start
    print(f"✅ Curvature: {curv_time:.3f}s")
    
    # Check if curvature is on CPU (bottleneck)
    if curv_time > normals_time:
        print(f"⚠️ BOTTLENECK: Curvature ({curv_time:.3f}s) slower than normals ({normals_time:.3f}s)")
        print(f"   Expected: Curvature should be ~same speed (both GPU)")
        print(f"   Actual: {curv_time/normals_time:.1f}x slower (likely CPU!)")
        return False
    else:
        print(f"✅ OK: Curvature fast enough")
        return True

def test_geometric_features_bottleneck():
    """Test geometric features bottleneck."""
    print("\n" + "=" * 60)
    print("TEST 2: Geometric Features Bottleneck")
    print("=" * 60)
    
    computer = GPUFeatureComputer(use_gpu=True)
    
    # Small test (100K points)
    N = 100_000
    points = np.random.rand(N, 3).astype(np.float32)
    
    print(f"\nComputing geometric features for {N:,} points...")
    features = ['planarity', 'linearity', 'sphericity', 'anisotropy']
    
    start = time.time()
    result = computer.compute_geometric_features(points, features, k=20)
    geo_time = time.time() - start
    
    print(f"✅ Geometric features: {geo_time:.3f}s")
    print(f"   Features computed: {list(result.keys())}")
    
    # Expected: <1s for 100K points on GPU
    # Actual with bottleneck: 3-5s
    if geo_time > 2.0:
        print(f"⚠️ BOTTLENECK: Geometric features too slow ({geo_time:.3f}s)")
        print(f"   Expected: <1.0s on GPU")
        print(f"   Actual: {geo_time:.1f}x slower than expected")
        return False
    else:
        print(f"✅ OK: Geometric features fast enough")
        return True

def test_eigenvalue_transfers():
    """Test eigenvalue feature transfer bottleneck."""
    print("\n" + "=" * 60)
    print("TEST 3: Eigenvalue Transfer Bottleneck")
    print("=" * 60)
    
    computer = GPUFeatureComputer(use_gpu=True)
    
    # Medium test (500K points)
    N = 500_000
    points = np.random.rand(N, 3).astype(np.float32)
    
    print(f"\nComputing geometric features for {N:,} points...")
    features = ['planarity', 'linearity', 'sphericity', 'anisotropy']
    
    start = time.time()
    result = computer.compute_geometric_features(points, features, k=20)
    geo_time = time.time() - start
    
    print(f"✅ Geometric features: {geo_time:.3f}s")
    
    # With per-feature transfers: ~15-20s for 500K points
    # Expected optimized: ~3-5s
    throughput = N / geo_time
    print(f"   Throughput: {throughput:,.0f} points/sec")
    
    if throughput < 50_000:
        print(f"⚠️ BOTTLENECK: Low throughput ({throughput:,.0f} pts/sec)")
        print(f"   Expected: >100,000 pts/sec with GPU optimization")
        print(f"   Likely cause: Per-feature GPU→CPU transfers")
        return False
    else:
        print(f"✅ OK: Good throughput")
        return True

def main():
    """Run all bottleneck tests."""
    print("\n🔍 CPU BOTTLENECK DETECTION TEST")
    print("Testing for CPU operations in GPU code paths\n")
    
    results = []
    
    # Test 1: Curvature
    try:
        results.append(("Curvature", test_curvature_bottleneck()))
    except Exception as e:
        print(f"❌ Curvature test failed: {e}")
        results.append(("Curvature", False))
    
    # Test 2: Geometric features
    try:
        results.append(("Geometric", test_geometric_features_bottleneck()))
    except Exception as e:
        print(f"❌ Geometric test failed: {e}")
        results.append(("Geometric", False))
    
    # Test 3: Eigenvalue transfers
    try:
        results.append(("Eigenvalue", test_eigenvalue_transfers()))
    except Exception as e:
        print(f"❌ Eigenvalue test failed: {e}")
        results.append(("Eigenvalue", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "⚠️ BOTTLENECK FOUND"
        print(f"{name:20s} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed < total:
        print("\n⚠️ BOTTLENECKS DETECTED!")
        print("See CRITICAL_CPU_BOTTLENECKS_FOUND.md for details and fixes")
        return 1
    else:
        print("\n✅ All tests passed!")
        return 0

if __name__ == "__main__":
    exit(main())
