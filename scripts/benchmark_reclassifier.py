#!/usr/bin/env python3
"""
Benchmark script for reclassifier optimizations.

Validates the expected 10-30× speedup from:
1. CPU vectorized implementation (Shapely 2.0+ bulk queries)
2. GPU batched implementation (RAPIDS cuSpatial)

Usage:
    python scripts/benchmark_reclassifier.py
    python scripts/benchmark_reclassifier.py --gpu  # Include GPU tests
    python scripts/benchmark_reclassifier.py --size 5000000  # Test with 5M points
"""

import argparse
import time
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.core.classification.reclassifier import (
    Reclassifier,
    HAS_GPU,
    HAS_BULK_QUERY,
)


def generate_test_data(n_points: int, n_polygons: int = 100):
    """
    Generate synthetic test data.
    
    Args:
        n_points: Number of points to generate
        n_polygons: Number of building polygons
    
    Returns:
        points: [N, 3] array of XYZ coordinates
        geometries: Array of shapely polygons
    """
    np.random.seed(42)
    
    print(f"Generating test data...")
    print(f"  Points: {n_points:,}")
    print(f"  Polygons: {n_polygons:,}")
    
    # Generate points in 1000m × 1000m area
    points = np.column_stack([
        np.random.uniform(0, 1000, n_points),  # X
        np.random.uniform(0, 1000, n_points),  # Y
        np.random.uniform(0, 50, n_points),    # Z (height)
    ])
    
    # Generate building polygons
    polygons = []
    for i in range(n_polygons):
        center_x = np.random.uniform(100, 900)
        center_y = np.random.uniform(100, 900)
        width = np.random.uniform(10, 50)
        height = np.random.uniform(10, 50)
        
        poly = Polygon([
            (center_x - width/2, center_y - height/2),
            (center_x + width/2, center_y - height/2),
            (center_x + width/2, center_y + height/2),
            (center_x - width/2, center_y + height/2),
            (center_x - width/2, center_y - height/2),
        ])
        polygons.append(poly)
    
    return points, np.array(polygons)


def benchmark_cpu_implementations(points, geometries):
    """
    Benchmark CPU vectorized vs legacy implementations.
    
    Returns:
        dict with timing results
    """
    print(f"\n{'='*70}")
    print(f"CPU Implementation Benchmark")
    print(f"{'='*70}")
    
    n_points = len(points)
    labels = np.ones(n_points, dtype=np.uint8)
    asprs_code = 6  # Building
    
    reclassifier = Reclassifier(
        acceleration_mode="cpu",
        chunk_size=100_000,
        show_progress=True
    )
    
    results = {}
    
    # Test legacy implementation (if available)
    if hasattr(reclassifier, '_classify_feature_cpu_legacy'):
        print(f"\n1. Legacy CPU (loop-based)")
        print(f"   {'─'*66}")
        labels_legacy = labels.copy()
        
        start = time.time()
        n_classified_legacy = reclassifier._classify_feature_cpu_legacy(
            points, labels_legacy, geometries, asprs_code, "Legacy"
        )
        time_legacy = time.time() - start
        
        results['legacy'] = {
            'time': time_legacy,
            'classified': n_classified_legacy,
            'points_per_sec': n_points / time_legacy
        }
        
        print(f"   ✓ Classified: {n_classified_legacy:,} points")
        print(f"   ✓ Time: {time_legacy:.2f}s ({n_points/time_legacy:,.0f} points/sec)")
    
    # Test vectorized implementation
    if HAS_BULK_QUERY:
        print(f"\n2. Vectorized CPU (Shapely 2.0+ bulk queries)")
        print(f"   {'─'*66}")
        labels_vec = labels.copy()
        
        start = time.time()
        n_classified_vec = reclassifier._classify_feature_cpu_vectorized(
            points, labels_vec, geometries, asprs_code, "Vectorized"
        )
        time_vec = time.time() - start
        
        results['vectorized'] = {
            'time': time_vec,
            'classified': n_classified_vec,
            'points_per_sec': n_points / time_vec
        }
        
        print(f"   ✓ Classified: {n_classified_vec:,} points")
        print(f"   ✓ Time: {time_vec:.2f}s ({n_points/time_vec:,.0f} points/sec)")
        
        # Calculate speedup
        if 'legacy' in results:
            speedup = results['legacy']['time'] / results['vectorized']['time']
            print(f"   ✓ Speedup: {speedup:.1f}× faster than legacy")
    else:
        print(f"\n⚠️  Shapely 2.0+ required for vectorized implementation")
        print(f"   Current version does not support bulk queries")
    
    return results


def benchmark_gpu_implementation(points, geometries):
    """
    Benchmark GPU implementation.
    
    Returns:
        dict with timing results
    """
    if not HAS_GPU:
        print(f"\n⚠️  GPU not available (RAPIDS cuSpatial not installed)")
        return None
    
    print(f"\n{'='*70}")
    print(f"GPU Implementation Benchmark")
    print(f"{'='*70}")
    
    n_points = len(points)
    labels = np.ones(n_points, dtype=np.uint8)
    asprs_code = 6  # Building
    
    reclassifier = Reclassifier(
        acceleration_mode="gpu",
        chunk_size=100_000,
        show_progress=True
    )
    
    print(f"\n1. GPU Batched (RAPIDS cuSpatial)")
    print(f"   {'─'*66}")
    
    start = time.time()
    n_classified = reclassifier._classify_feature_gpu(
        points, labels, geometries, asprs_code, "GPU"
    )
    time_gpu = time.time() - start
    
    results = {
        'time': time_gpu,
        'classified': n_classified,
        'points_per_sec': n_points / time_gpu
    }
    
    print(f"   ✓ Classified: {n_classified:,} points")
    print(f"   ✓ Time: {time_gpu:.2f}s ({n_points/time_gpu:,.0f} points/sec)")
    
    return results


def print_summary(cpu_results, gpu_results, n_points):
    """Print comprehensive summary."""
    print(f"\n{'='*70}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"Dataset: {n_points:,} points")
    print(f"")
    
    # Create results table
    implementations = []
    
    if cpu_results and 'legacy' in cpu_results:
        implementations.append(
            ("CPU Legacy", cpu_results['legacy']['time'], 
             cpu_results['legacy']['points_per_sec'], 1.0)
        )
        baseline_time = cpu_results['legacy']['time']
    elif cpu_results and 'vectorized' in cpu_results:
        implementations.append(
            ("CPU Vectorized", cpu_results['vectorized']['time'],
             cpu_results['vectorized']['points_per_sec'], 1.0)
        )
        baseline_time = cpu_results['vectorized']['time']
    else:
        baseline_time = None
    
    if cpu_results and 'vectorized' in cpu_results and 'legacy' in cpu_results:
        speedup = cpu_results['legacy']['time'] / cpu_results['vectorized']['time']
        implementations.append(
            ("CPU Vectorized", cpu_results['vectorized']['time'],
             cpu_results['vectorized']['points_per_sec'], speedup)
        )
    
    if gpu_results:
        speedup = baseline_time / gpu_results['time'] if baseline_time else 1.0
        implementations.append(
            ("GPU Batched", gpu_results['time'],
             gpu_results['points_per_sec'], speedup)
        )
    
    # Print table
    print(f"{'Implementation':<20} {'Time (s)':<12} {'Points/sec':<15} {'Speedup':<10}")
    print(f"{'─'*20} {'─'*12} {'─'*15} {'─'*10}")
    
    for name, time_val, pps, speedup in implementations:
        print(f"{name:<20} {time_val:>10.2f}s  {pps:>13,.0f}  {speedup:>8.1f}×")
    
    print(f"")
    
    # Performance targets
    print(f"Performance Targets:")
    print(f"  CPU Vectorized: 10-20× faster than legacy  ", end="")
    if cpu_results and 'vectorized' in cpu_results and 'legacy' in cpu_results:
        speedup = cpu_results['legacy']['time'] / cpu_results['vectorized']['time']
        if speedup >= 10:
            print(f"✅ ACHIEVED ({speedup:.1f}×)")
        elif speedup >= 5:
            print(f"⚠️  PARTIAL ({speedup:.1f}×)")
        else:
            print(f"❌ MISSED ({speedup:.1f}×)")
    else:
        print(f"⏩ SKIPPED")
    
    print(f"  GPU Batched:    5-10× faster than CPU      ", end="")
    if gpu_results and baseline_time:
        speedup = baseline_time / gpu_results['time']
        if speedup >= 5:
            print(f"✅ ACHIEVED ({speedup:.1f}×)")
        elif speedup >= 3:
            print(f"⚠️  PARTIAL ({speedup:.1f}×)")
        else:
            print(f"❌ MISSED ({speedup:.1f}×)")
    else:
        print(f"⏩ SKIPPED (no GPU)")
    
    print(f"")
    
    # Extrapolate to real IGN tile
    if baseline_time:
        ign_points = 18_000_000
        scale_factor = ign_points / n_points
        
        print(f"Extrapolation to IGN LiDAR HD tile ({ign_points:,} points):")
        
        if 'legacy' in cpu_results:
            legacy_ign = cpu_results['legacy']['time'] * scale_factor
            print(f"  Legacy CPU:       {legacy_ign/60:.1f} min")
        
        if 'vectorized' in cpu_results:
            vec_ign = cpu_results['vectorized']['time'] * scale_factor
            print(f"  Vectorized CPU:   {vec_ign/60:.1f} min")
            if 'legacy' in cpu_results:
                time_saved = (legacy_ign - vec_ign) / 60
                print(f"    → Saves {time_saved:.1f} min per tile")
        
        if gpu_results:
            gpu_ign = gpu_results['time'] * scale_factor
            print(f"  GPU Batched:      {gpu_ign/60:.1f} min")
            time_saved = (baseline_time * scale_factor - gpu_ign) / 60
            print(f"    → Saves {time_saved:.1f} min per tile")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark reclassifier optimizations"
    )
    parser.add_argument(
        "--size", type=int, default=1_000_000,
        help="Number of points (default: 1M)"
    )
    parser.add_argument(
        "--polygons", type=int, default=100,
        help="Number of polygons (default: 100)"
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Include GPU benchmarks"
    )
    parser.add_argument(
        "--cpu-only", action="store_true",
        help="Only run CPU benchmarks"
    )
    
    args = parser.parse_args()
    
    print(f"{'='*70}")
    print(f"Reclassifier Optimization Benchmark")
    print(f"{'='*70}")
    print(f"Shapely bulk queries: {'✅ Available' if HAS_BULK_QUERY else '❌ Not available'}")
    print(f"GPU (RAPIDS):         {'✅ Available' if HAS_GPU else '❌ Not available'}")
    print(f"")
    
    # Generate test data
    points, geometries = generate_test_data(args.size, args.polygons)
    
    # Run CPU benchmarks
    cpu_results = benchmark_cpu_implementations(points, geometries)
    
    # Run GPU benchmarks
    gpu_results = None
    if not args.cpu_only and (args.gpu or HAS_GPU):
        gpu_results = benchmark_gpu_implementation(points, geometries)
    
    # Print summary
    print_summary(cpu_results, gpu_results, args.size)


if __name__ == "__main__":
    main()
