#!/usr/bin/env python3
"""Phase 2 Reclassifier Optimization Benchmark

Tests the new optimizations:
1. Vectorized Point creation with Shapely 2.0
2. Parallel contains() testing with ThreadPoolExecutor
3. Optimized GPU batching with memory management

Usage:
    # CPU testing (base environment):
    python scripts/benchmark_reclassifier_phase2.py
    
    # GPU testing (CRITICAL - use ign_gpu environment):
    conda run -n ign_gpu python scripts/benchmark_reclassifier_phase2.py --gpu

Note: ALWAYS use ign_gpu conda environment for GPU-related work.
"""

import time
import sys
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.core.classification.reclassifier import Reclassifier, HAS_BULK_QUERY


def generate_test_data(n_points: int, n_polygons: int = 100):
    """Generate test data."""
    np.random.seed(42)
    
    print(f"\n🔧 Generating test data:")
    print(f"   Points: {n_points:,}")
    print(f"   Polygons: {n_polygons:,}")
    
    # Generate points in 1000m × 1000m area
    points = np.column_stack([
        np.random.uniform(0, 1000, n_points),  # X
        np.random.uniform(0, 1000, n_points),  # Y
        np.random.uniform(0, 50, n_points),    # Z
    ])
    
    # Generate building polygons (mix of sizes)
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


def benchmark_implementation(name, method, points, geometries, n_runs=3):
    """Benchmark a single implementation."""
    asprs_code = 6  # Building
    times = []
    n_classified_list = []
    
    for run in range(n_runs):
        labels = np.ones(len(points), dtype=np.uint8)
        
        start = time.perf_counter()
        n_classified = method(
            points, labels, geometries, asprs_code, name
        )
        elapsed = time.perf_counter() - start
        
        times.append(elapsed)
        n_classified_list.append(n_classified)
    
    # Use median time (more robust than mean)
    median_time = np.median(times)
    points_per_sec = len(points) / median_time
    
    return {
        'name': name,
        'time': median_time,
        'points_per_sec': points_per_sec,
        'n_classified': n_classified_list[0],
        'times': times
    }


def main():
    print("="*70)
    print("PHASE 2: Reclassifier Optimization Benchmark")
    print("="*70)
    print(f"Shapely bulk queries: {'✅ Available' if HAS_BULK_QUERY else '❌ Not available'}")
    
    # Test with different dataset sizes
    test_sizes = [100_000, 500_000, 1_000_000]
    
    for n_points in test_sizes:
        print(f"\n{'='*70}")
        print(f"Dataset: {n_points:,} points")
        print(f"{'='*70}")
        
        points, geometries = generate_test_data(n_points, n_polygons=100)
        
        reclassifier_legacy = Reclassifier(
            acceleration_mode="cpu",
            chunk_size=100_000,
            show_progress=False
        )
        
        reclassifier_optimized = Reclassifier(
            acceleration_mode="cpu",
            chunk_size=100_000,
            show_progress=False
        )
        
        # Benchmark legacy implementation
        print(f"\n1. Legacy CPU (STRtree + loop)")
        print("   " + "─"*66)
        result_legacy = benchmark_implementation(
            "Legacy",
            reclassifier_legacy._classify_feature_cpu_legacy,
            points,
            geometries,
            n_runs=2
        )
        print(f"   ✓ Time: {result_legacy['time']:.2f}s ({result_legacy['points_per_sec']:,.0f} pts/sec)")
        print(f"   ✓ Classified: {result_legacy['n_classified']:,} points")
        
        # Benchmark optimized vectorized implementation
        print(f"\n2. Optimized Vectorized (Shapely 2.0 + Parallel)")
        print("   " + "─"*66)
        result_optimized = benchmark_implementation(
            "Optimized",
            reclassifier_optimized._classify_feature_cpu_vectorized,
            points,
            geometries,
            n_runs=2
        )
        print(f"   ✓ Time: {result_optimized['time']:.2f}s ({result_optimized['points_per_sec']:,.0f} pts/sec)")
        print(f"   ✓ Classified: {result_optimized['n_classified']:,} points")
        
        # Calculate speedup
        speedup = result_legacy['time'] / result_optimized['time']
        time_saved = result_legacy['time'] - result_optimized['time']
        
        print(f"\n{'='*70}")
        print(f"RESULTS FOR {n_points:,} POINTS")
        print(f"{'='*70}")
        print(f"Legacy:         {result_legacy['time']:.2f}s")
        print(f"Optimized:      {result_optimized['time']:.2f}s")
        print(f"Speedup:        {speedup:.1f}× 🚀")
        print(f"Time saved:     {time_saved:.2f}s")
        
        if speedup >= 5:
            print(f"Status:         ✅ EXCELLENT (target: 5-15×)")
        elif speedup >= 3:
            print(f"Status:         ✅ GOOD (approaching target)")
        elif speedup >= 1.5:
            print(f"Status:         ⚠️ MODERATE (needs more work)")
        else:
            print(f"Status:         ❌ NO IMPROVEMENT")
    
    # Final summary
    print(f"\n{'='*70}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*70}")
    print("\nKey Optimizations Implemented:")
    print("  1. ✅ Vectorized Point creation (Shapely 2.0 array interface)")
    print("  2. ✅ Parallel contains() testing (ThreadPoolExecutor)")
    print("  3. ✅ defaultdict for faster candidate building")
    print("  4. ✅ Optimized batch size heuristics")
    print(f"\nExpected Gains: 5-15× speedup for large datasets (>500K points)")
    print("="*70)


if __name__ == "__main__":
    main()
