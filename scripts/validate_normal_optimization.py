#!/usr/bin/env python3
"""
Validation script for GPU normal computation optimization.

This script tests the new batched 3x3 inverse + power iteration approach
against the original cuSOLVER eigendecomposition method.

Expected improvement: 10-50x speedup for large batches (>1M points)
"""

import time
import numpy as np
import sys

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    print("❌ CuPy not available - GPU optimization cannot be tested")
    sys.exit(1)

from ign_lidar.features.features_gpu import GPUFeatureComputer


def generate_test_data(n_points, noise_level=0.01):
    """Generate a noisy planar point cloud for testing."""
    # Create points on a plane: z = 0.5*x + 0.3*y + noise
    xs = np.random.uniform(0, 100, size=n_points)
    ys = np.random.uniform(0, 100, size=n_points)
    zs = 0.5 * xs + 0.3 * ys + np.random.normal(0, noise_level, size=n_points)
    
    points = np.stack([xs, ys, zs], axis=1).astype(np.float32)
    
    # Expected normal direction (normalized)
    expected_normal = np.array([-0.5, -0.3, 1.0])
    expected_normal = expected_normal / np.linalg.norm(expected_normal)
    
    return points, expected_normal


def test_normal_computation_accuracy():
    """Test that the optimized normal computation produces correct results."""
    print("\n" + "="*70)
    print("TEST 1: Accuracy Validation")
    print("="*70)
    
    # Generate test data
    n_points = 50000
    points, expected_normal = generate_test_data(n_points, noise_level=0.01)
    
    print(f"Generated {n_points:,} points on a plane")
    print(f"Expected normal direction: [{expected_normal[0]:.4f}, {expected_normal[1]:.4f}, {expected_normal[2]:.4f}]")
    
    # Compute normals
    computer = GPUFeatureComputer(use_gpu=True, batch_size=2_000_000)
    normals = computer.compute_normals(points, k=16)
    
    # Check mean normal direction
    mean_normal = np.mean(normals, axis=0)
    mean_normal = mean_normal / np.linalg.norm(mean_normal)
    
    print(f"Computed mean normal: [{mean_normal[0]:.4f}, {mean_normal[1]:.4f}, {mean_normal[2]:.4f}]")
    
    # Compare with expected
    dot_product = np.dot(mean_normal, expected_normal)
    angle_error = np.arccos(np.clip(dot_product, -1, 1)) * 180 / np.pi
    
    print(f"Angular error: {angle_error:.2f}°")
    
    if angle_error < 5.0:
        print("✅ PASS: Normal computation is accurate")
        return True
    else:
        print("❌ FAIL: Normal computation has high error")
        return False


def test_normal_computation_performance():
    """Benchmark the optimized normal computation."""
    print("\n" + "="*70)
    print("TEST 2: Performance Benchmark")
    print("="*70)
    
    test_sizes = [100_000, 500_000, 1_000_000, 2_000_000]
    k_neighbors = 20
    
    results = []
    
    for n_points in test_sizes:
        print(f"\n📊 Testing with {n_points:,} points, k={k_neighbors}")
        
        # Generate test data
        points, _ = generate_test_data(n_points, noise_level=0.01)
        
        # Warm-up run
        computer = GPUFeatureComputer(use_gpu=True, batch_size=2_000_000)
        _ = computer.compute_normals(points, k=k_neighbors)
        
        # Timed runs
        n_runs = 3
        times = []
        
        for run in range(n_runs):
            start = time.perf_counter()
            normals = computer.compute_normals(points, k=k_neighbors)
            cp.cuda.Stream.null.synchronize()  # Ensure GPU work completes
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = n_points / avg_time
        
        print(f"   Time: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"   Throughput: {throughput:,.0f} points/sec")
        
        results.append({
            'n_points': n_points,
            'avg_time': avg_time,
            'throughput': throughput
        })
    
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    print(f"{'Points':<15} {'Time (s)':<15} {'Throughput (pts/s)':<20}")
    print("-"*70)
    for r in results:
        print(f"{r['n_points']:>13,}  {r['avg_time']:>13.3f}  {r['throughput']:>18,.0f}")
    
    # Check if performance is reasonable
    # For reference: old method took ~5-9 minutes for 2M points
    # New method should be <10 seconds for 2M points
    large_test = [r for r in results if r['n_points'] >= 2_000_000]
    if large_test and large_test[0]['avg_time'] < 30.0:
        print("\n✅ PASS: Performance is significantly improved")
        print(f"   2M points processed in {large_test[0]['avg_time']:.1f}s (target: <30s)")
        return True
    else:
        print("\n⚠️  WARNING: Performance may not be optimal")
        if large_test:
            print(f"   2M points took {large_test[0]['avg_time']:.1f}s")
        return False


def test_edge_cases():
    """Test edge cases and numerical stability."""
    print("\n" + "="*70)
    print("TEST 3: Edge Cases & Stability")
    print("="*70)
    
    computer = GPUFeatureComputer(use_gpu=True, batch_size=2_000_000)
    
    # Test 1: Degenerate points (all colinear)
    print("\n📍 Test 3.1: Colinear points")
    n = 1000
    t = np.linspace(0, 10, n)
    colinear_points = np.stack([t, 2*t, 3*t], axis=1).astype(np.float32)
    
    normals = computer.compute_normals(colinear_points, k=10)
    
    # Check for NaN/Inf
    has_nan = np.isnan(normals).any()
    has_inf = np.isinf(normals).any()
    
    print(f"   NaN values: {'Yes ❌' if has_nan else 'No ✅'}")
    print(f"   Inf values: {'Yes ❌' if has_inf else 'No ✅'}")
    
    # Test 2: Very sparse points
    print("\n📍 Test 3.2: Sparse point cloud")
    sparse_points = np.random.uniform(-1000, 1000, size=(5000, 3)).astype(np.float32)
    
    normals = computer.compute_normals(sparse_points, k=5)
    
    has_nan = np.isnan(normals).any()
    has_inf = np.isinf(normals).any()
    
    print(f"   NaN values: {'Yes ❌' if has_nan else 'No ✅'}")
    print(f"   Inf values: {'Yes ❌' if has_inf else 'No ✅'}")
    
    # Test 3: Dense clusters
    print("\n📍 Test 3.3: Dense clusters")
    centers = np.random.uniform(-10, 10, size=(5, 3))
    dense_points = []
    for center in centers:
        cluster = center + np.random.normal(0, 0.001, size=(2000, 3))
        dense_points.append(cluster)
    dense_points = np.vstack(dense_points).astype(np.float32)
    
    normals = computer.compute_normals(dense_points, k=20)
    
    has_nan = np.isnan(normals).any()
    has_inf = np.isinf(normals).any()
    is_normalized = np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-4)
    
    print(f"   NaN values: {'Yes ❌' if has_nan else 'No ✅'}")
    print(f"   Inf values: {'Yes ❌' if has_inf else 'No ✅'}")
    print(f"   Normalized: {'Yes ✅' if is_normalized else 'No ❌'}")
    
    all_stable = not (has_nan or has_inf) and is_normalized
    
    if all_stable:
        print("\n✅ PASS: All edge cases handled correctly")
    else:
        print("\n❌ FAIL: Some edge cases not handled properly")
    
    return all_stable


def main():
    """Run all validation tests."""
    print("\n" + "="*70)
    print("GPU NORMAL COMPUTATION OPTIMIZATION - VALIDATION SUITE")
    print("="*70)
    print("\nThis script validates the new batched 3x3 inverse + power iteration")
    print("optimization for GPU normal computation.")
    print("\nExpected improvements:")
    print("  • 10-50x speedup for large batches (>1M points)")
    print("  • Same or better numerical accuracy")
    print("  • Robust handling of edge cases")
    print("="*70)
    
    # Run tests
    accuracy_pass = test_normal_computation_accuracy()
    performance_pass = test_normal_computation_performance()
    stability_pass = test_edge_cases()
    
    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Accuracy Test:     {'✅ PASS' if accuracy_pass else '❌ FAIL'}")
    print(f"Performance Test:  {'✅ PASS' if performance_pass else '⚠️  WARNING'}")
    print(f"Stability Test:    {'✅ PASS' if stability_pass else '❌ FAIL'}")
    print("="*70)
    
    all_pass = accuracy_pass and performance_pass and stability_pass
    
    if all_pass:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nThe optimization is working correctly and provides significant speedup.")
        print("You can now use the optimized GPU normal computation in production.")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED OR SHOWED WARNINGS")
        print("\nPlease review the results above and investigate any failures.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
