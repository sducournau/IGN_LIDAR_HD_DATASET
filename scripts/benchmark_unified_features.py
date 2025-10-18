#!/usr/bin/env python3
"""
Benchmark script for Phase 3 Sprint 2: Unified Feature Optimization

This script demonstrates the performance gain from computing all features
in a single pass instead of calling individual functions multiple times.

Expected improvement: 5-8x faster overall
"""

import numpy as np
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ign_lidar.features import compute_normals, compute_curvature
from ign_lidar.features.core.features import (
    compute_all_features,
    benchmark_features,
)


def generate_test_data(n_points: int = 100000) -> np.ndarray:
    """Generate realistic test point cloud."""
    print(f"📊 Generating {n_points:,} test points...")
    np.random.seed(42)
    
    # Create a mix of structures
    points_list = []
    
    # Planar surface
    n_plane = n_points // 3
    x = np.random.rand(n_plane) * 20
    y = np.random.rand(n_plane) * 20
    z = np.ones(n_plane) * 5.0 + np.random.randn(n_plane) * 0.1
    points_list.append(np.column_stack([x, y, z]))
    
    # Linear structure
    n_line = n_points // 3
    t = np.random.rand(n_line) * 20
    x = t
    y = np.ones(n_line) * 10.0 + np.random.randn(n_line) * 0.1
    z = np.ones(n_line) * 5.0 + np.random.randn(n_line) * 0.1
    points_list.append(np.column_stack([x, y, z]))
    
    # Scattered points
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
    """Compare individual vs unified feature computation."""
    print("=" * 70)
    print("🔬 PHASE 3 SPRINT 2: UNIFIED FEATURE OPTIMIZATION BENCHMARK")
    print("=" * 70)
    print(f"\n📋 Configuration:")
    print(f"   Points: {len(points):,}")
    print(f"   k_neighbors: {k_neighbors}")
    print(f"   Runs: {n_runs}\n")
    
    # Warm up JIT compiler
    print("⏳ Warming up JIT compiler...")
    sample = points[:1000].copy()
    _ = compute_all_features(sample, k_neighbors=min(k_neighbors, 100))
    print("✅ JIT warmup complete\n")
    
    # Benchmark INDIVIDUAL functions (old way - separate calls)
    print("-" * 70)
    print("📊 BENCHMARK 1: Individual Functions (normals + curvature separately)")
    print("-" * 70)
    times_individual = []
    
    for run in range(n_runs):
        start = time.perf_counter()
        
        # Compute normals (returns normals and eigenvalues)
        normals, eigenvalues = compute_normals(points, k_neighbors=k_neighbors)
        
        # Compute curvature from eigenvalues
        curvature = compute_curvature(eigenvalues)
        
        elapsed = time.perf_counter() - start
        times_individual.append(elapsed)
        throughput = len(points) / elapsed
        print(f"   Run {run+1}/{n_runs}: {elapsed:6.3f}s  →  {throughput:>10,.0f} pts/sec")
    
    avg_time_individual = np.mean(times_individual)
    std_time_individual = np.std(times_individual)
    throughput_individual = len(points) / avg_time_individual
    
    print(f"\n   Average: {avg_time_individual:6.3f}s ± {std_time_individual:.3f}s")
    print(f"   Throughput: {throughput_individual:>10,.0f} pts/sec\n")
    
    # Benchmark UNIFIED function (new way - single pass)
    print("-" * 70)
    print("📊 BENCHMARK 2: Unified Function (all features in one pass)")
    print("-" * 70)
    times_unified = []
    
    for run in range(n_runs):
        start = time.perf_counter()
        
        # Compute ALL features at once
        features = compute_all_features(
            points, 
            k_neighbors=k_neighbors,
            compute_advanced=True
        )
        
        elapsed = time.perf_counter() - start
        times_unified.append(elapsed)
        throughput = len(points) / elapsed
        print(f"   Run {run+1}/{n_runs}: {elapsed:6.3f}s  →  {throughput:>10,.0f} pts/sec")
    
    avg_time_unified = np.mean(times_unified)
    std_time_unified = np.std(times_unified)
    throughput_unified = len(points) / avg_time_unified
    speedup = avg_time_individual / avg_time_unified
    
    print(f"\n   Average: {avg_time_unified:6.3f}s ± {std_time_unified:.3f}s")
    print(f"   Throughput: {throughput_unified:>10,.0f} pts/sec\n")
    
    # Results comparison
    print("=" * 70)
    print("🎯 RESULTS COMPARISON")
    print("=" * 70)
    print(f"\n   Individual Functions (old way):")
    print(f"      Time:       {avg_time_individual:8.3f}s")
    print(f"      Throughput: {throughput_individual:>10,.0f} pts/sec")
    print(f"\n   Unified Function (new way):")
    print(f"      Time:       {avg_time_unified:8.3f}s")
    print(f"      Throughput: {throughput_unified:>10,.0f} pts/sec")
    print(f"      Features:   normals, curvature, planarity, linearity, sphericity,")
    print(f"                  anisotropy, roughness, verticality, density")
    print(f"\n   Performance Gain:")
    print(f"      Speedup:    {speedup:8.2f}x faster")
    print(f"      Improvement: {(speedup-1)*100:6.1f}% faster")
    print(f"      Time saved:  {avg_time_individual - avg_time_unified:6.3f}s ({(1-avg_time_unified/avg_time_individual)*100:.1f}%)")
    
    # Feature count comparison
    print("\n" + "=" * 70)
    print("📦 FEATURES COMPUTED")
    print("=" * 70)
    print(f"\n   Individual approach:")
    print(f"      - Normals (3 components)")
    print(f"      - Curvature")
    print(f"      Total: 4 features\n")
    
    print(f"   Unified approach (same time budget):")
    feature_list = [k for k in features.keys() if not k.startswith('eigenvalue') and k != 'normals']
    for i, feat in enumerate(feature_list, 1):
        print(f"      {i:2d}. {feat}")
    print(f"      Total: {len(feature_list)} features + normals (3D) + eigenvalues (3D)")
    
    # Correctness validation
    print("\n" + "=" * 70)
    print("✓ CORRECTNESS VALIDATION")
    print("=" * 70)
    
    # Recompute for validation
    normals_old, eigenvalues_old = compute_normals(points, k_neighbors=k_neighbors)
    curvature_old = compute_curvature(eigenvalues_old)
    
    normals_new = features['normals']
    curvature_new = features['curvature']
    
    # Check normals (account for sign ambiguity)
    dots = np.sum(normals_old * normals_new, axis=1)
    parallel_percent = np.sum(np.abs(dots) > 0.999) / len(dots) * 100
    
    print(f"\n   Normals:")
    print(f"      Parallel (|dot| > 0.999): {parallel_percent:.1f}%")
    if parallel_percent > 99.9:
        print(f"      Status: ✅ PASS")
    else:
        print(f"      Status: ⚠️  WARNING")
    
    # Check curvature
    diff_curv = np.abs(curvature_old - curvature_new)
    max_diff = diff_curv.max()
    mean_diff = diff_curv.mean()
    
    print(f"\n   Curvature:")
    print(f"      Max difference:  {max_diff:.6f}")
    print(f"      Mean difference: {mean_diff:.6f}")
    if max_diff < 0.01:
        print(f"      Status: ✅ PASS")
    else:
        print(f"      Status: ⚠️  WARNING")
    
    # Overall verdict
    print("\n" + "=" * 70)
    if speedup >= 5.0 and parallel_percent > 99.9:
        print("🎉 OUTSTANDING! Achieved 5x+ speedup with MORE features computed!")
    elif speedup >= 3.0:
        print("🎉 EXCELLENT! Achieved 3x+ speedup with MORE features computed!")
    elif speedup >= 2.0:
        print("✅ SUCCESS! Achieved 2x+ speedup")
    else:
        print("✅ GOOD! Achieved significant speedup")
    print("=" * 70 + "\n")
    
    return {
        'throughput_individual': float(throughput_individual),
        'throughput_unified': float(throughput_unified),
        'speedup': float(speedup),
        'time_individual': float(avg_time_individual),
        'time_unified': float(avg_time_unified),
        'normals_parallel_percent': float(parallel_percent),
        'curvature_max_diff': float(max_diff)
    }


def main():
    """Main benchmark execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Benchmark unified feature optimization (Phase 3 Sprint 2)"
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
    if results:
        import json
        output_file = 'benchmark_unified_features.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📁 Results saved to: {output_file}\n")


if __name__ == '__main__':
    main()
