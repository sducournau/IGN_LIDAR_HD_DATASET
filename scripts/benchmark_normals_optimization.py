#!/usr/bin/env python3
"""
Benchmark script for Phase 3 Sprint 1: Normals Optimization

This script compares the performance of the original vs optimized
normals computation using Numba JIT compilation.

Expected improvement: 2-5x faster (50K → 100-250K pts/sec)
"""

import numpy as np
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ign_lidar.features.core.normals import compute_normals as compute_normals_original
from ign_lidar.features.core.normals_optimized import (
    compute_normals_optimized,
    NUMBA_AVAILABLE
)


def generate_test_data(n_points: int = 100000) -> np.ndarray:
    """Generate realistic test point cloud."""
    print(f"📊 Generating {n_points:,} test points...")
    np.random.seed(42)
    
    # Create a mix of structures: planes, edges, spheres
    points_list = []
    
    # Planar surface (roof)
    n_plane = n_points // 3
    x = np.random.rand(n_plane) * 20
    y = np.random.rand(n_plane) * 20
    z = np.ones(n_plane) * 5.0 + np.random.randn(n_plane) * 0.1
    points_list.append(np.column_stack([x, y, z]))
    
    # Linear structure (edge)
    n_line = n_points // 3
    t = np.random.rand(n_line) * 20
    x = t
    y = np.ones(n_line) * 10.0 + np.random.randn(n_line) * 0.1
    z = np.ones(n_line) * 5.0 + np.random.randn(n_line) * 0.1
    points_list.append(np.column_stack([x, y, z]))
    
    # Scattered points (vegetation)
    n_scatter = n_points - n_plane - n_line
    x = np.random.rand(n_scatter) * 20
    y = np.random.rand(n_scatter) * 20
    z = np.random.rand(n_scatter) * 10
    points_list.append(np.column_stack([x, y, z]))
    
    points = np.vstack(points_list).astype(np.float32)
    np.random.shuffle(points)
    
    print(f"✅ Generated {len(points):,} points\n")
    return points


def benchmark_comparison(
    points: np.ndarray,
    k_neighbors: int = 20,
    n_runs: int = 3
):
    """Compare original vs optimized implementation."""
    print("=" * 70)
    print("🔬 PHASE 3 SPRINT 1: NORMALS OPTIMIZATION BENCHMARK")
    print("=" * 70)
    print(f"\n📋 Configuration:")
    print(f"   Points: {len(points):,}")
    print(f"   k_neighbors: {k_neighbors}")
    print(f"   Runs: {n_runs}")
    print(f"   Numba available: {NUMBA_AVAILABLE}\n")
    
    if not NUMBA_AVAILABLE:
        print("❌ ERROR: Numba not available!")
        print("   Install with: conda install -c conda-forge numba")
        return
    
    # Warm up JIT compiler
    print("⏳ Warming up JIT compiler (first run is slow)...")
    sample = points[:1000].copy()
    _ = compute_normals_optimized(sample, k_neighbors=min(k_neighbors, 100))
    print("✅ JIT warmup complete\n")
    
    # Benchmark original implementation
    print("-" * 70)
    print("📊 BENCHMARK 1: Original Implementation (CPU)")
    print("-" * 70)
    times_original = []
    
    for run in range(n_runs):
        start = time.perf_counter()
        normals_orig, eigvals_orig = compute_normals_original(
            points, k_neighbors=k_neighbors
        )
        elapsed = time.perf_counter() - start
        times_original.append(elapsed)
        throughput = len(points) / elapsed
        print(f"   Run {run+1}/{n_runs}: {elapsed:6.3f}s  →  {throughput:>10,.0f} pts/sec")
    
    avg_time_orig = np.mean(times_original)
    std_time_orig = np.std(times_original)
    throughput_orig = len(points) / avg_time_orig
    
    print(f"\n   Average: {avg_time_orig:6.3f}s ± {std_time_orig:.3f}s")
    print(f"   Throughput: {throughput_orig:>10,.0f} pts/sec\n")
    
    # Benchmark optimized implementation
    print("-" * 70)
    print("📊 BENCHMARK 2: Optimized Implementation (Numba JIT + Parallel)")
    print("-" * 70)
    times_optimized = []
    
    for run in range(n_runs):
        start = time.perf_counter()
        normals_opt, eigvals_opt = compute_normals_optimized(
            points, k_neighbors=k_neighbors
        )
        elapsed = time.perf_counter() - start
        times_optimized.append(elapsed)
        throughput = len(points) / elapsed
        print(f"   Run {run+1}/{n_runs}: {elapsed:6.3f}s  →  {throughput:>10,.0f} pts/sec")
    
    avg_time_opt = np.mean(times_optimized)
    std_time_opt = np.std(times_optimized)
    throughput_opt = len(points) / avg_time_opt
    speedup = avg_time_orig / avg_time_opt
    
    print(f"\n   Average: {avg_time_opt:6.3f}s ± {std_time_opt:.3f}s")
    print(f"   Throughput: {throughput_opt:>10,.0f} pts/sec\n")
    
    # Results comparison
    print("=" * 70)
    print("🎯 RESULTS COMPARISON")
    print("=" * 70)
    print(f"\n   Original Implementation:")
    print(f"      Time:       {avg_time_orig:8.3f}s")
    print(f"      Throughput: {throughput_orig:>10,.0f} pts/sec")
    print(f"\n   Optimized Implementation:")
    print(f"      Time:       {avg_time_opt:8.3f}s")
    print(f"      Throughput: {throughput_opt:>10,.0f} pts/sec")
    print(f"\n   Performance Gain:")
    print(f"      Speedup:    {speedup:8.2f}x faster")
    print(f"      Improvement: {(speedup-1)*100:6.1f}% faster")
    print(f"      Time saved:  {avg_time_orig - avg_time_opt:6.3f}s ({(1-avg_time_opt/avg_time_orig)*100:.1f}%)")
    
    # Verify correctness
    print("\n" + "=" * 70)
    print("✓ CORRECTNESS VALIDATION")
    print("=" * 70)
    
    # Check normals - handle sign ambiguity
    # Normals can point in either direction (+/-), so check dot product instead
    dots = np.sum(normals_orig * normals_opt, axis=1)
    parallel_percent = np.sum(np.abs(dots) > 0.999) / len(dots) * 100
    avg_dot = np.mean(np.abs(dots))
    
    print(f"\n   Normals (accounting for sign ambiguity):")
    print(f"      Parallel (|dot| > 0.999): {parallel_percent:.1f}%")
    print(f"      Average |dot product|:    {avg_dot:.6f}")
    
    if parallel_percent > 99.9:
        print(f"      Status: ✅ PASS (normals are parallel)")
    else:
        print(f"      Status: ⚠️  WARNING (some normals differ)")
        # Also show raw difference for debugging
        diff_normals = np.abs(normals_orig - normals_opt)
        print(f"      Raw max difference:     {diff_normals.max():.6f}")
        print(f"      Raw mean difference:    {diff_normals.mean():.6f}")
    
    # Check eigenvalues
    diff_eigvals = np.abs(eigvals_orig - eigvals_opt)
    max_diff_eig = diff_eigvals.max()
    mean_diff_eig = diff_eigvals.mean()
    
    print(f"\n   Eigenvalues:")
    print(f"      Max difference:  {max_diff_eig:.6f}")
    print(f"      Mean difference: {mean_diff_eig:.6f}")
    
    if max_diff_eig < 0.01:
        print(f"      Status: ✅ PASS (difference < 0.01)")
    else:
        print(f"      Status: ⚠️  WARNING (difference >= 0.01)")
    
    # Overall verdict
    print("\n" + "=" * 70)
    if speedup >= 2.0 and parallel_percent > 99.9:
        print("🎉 SUCCESS! Optimization achieved 2x+ speedup with correct results!")
    elif speedup >= 1.5:
        print("✅ GOOD! Optimization achieved 1.5x+ speedup")
    elif speedup >= 1.2:
        print("👍 OK! Optimization achieved 1.2x+ speedup")
    else:
        print("⚠️  WARNING! Speedup less than 1.2x - may need further optimization")
    print("=" * 70 + "\n")
    
    return {
        'throughput_original': throughput_orig,
        'throughput_optimized': throughput_opt,
        'speedup': speedup,
        'time_original': avg_time_orig,
        'time_optimized': avg_time_opt,
        'normals_parallel_percent': parallel_percent,
        'normals_avg_dot': avg_dot,
        'max_diff_eigenvalues': max_diff_eig
    }


def main():
    """Main benchmark execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Benchmark normals optimization (Phase 3 Sprint 1)"
    )
    parser.add_argument(
        '--points', '-n',
        type=int,
        default=100000,
        help='Number of test points (default: 100000)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=20,
        help='Number of neighbors (default: 20)'
    )
    parser.add_argument(
        '--runs', '-r',
        type=int,
        default=3,
        help='Number of benchmark runs (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Generate test data
    points = generate_test_data(args.points)
    
    # Run benchmark
    results = benchmark_comparison(points, k_neighbors=args.k, n_runs=args.runs)
    
    # Save results
    import json
    output_file = 'benchmark_normals_optimization.json'
    # Convert numpy types to Python types for JSON serialization
    results_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                    for k, v in results.items()}
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"📁 Results saved to: {output_file}\n")


if __name__ == '__main__':
    main()
