#!/usr/bin/env python3
"""
Benchmark script for Phase 3 Sprint 3: Parallel Tile Processing

This script tests the speedup achieved by processing tiles in parallel
vs sequential processing.

Expected improvement: 2-4x on multi-core systems
"""

import numpy as np
import time
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("📦 Checking dependencies...")
try:
    from joblib import Parallel, delayed
    print("   ✅ joblib available")
    JOBLIB_AVAILABLE = True
except ImportError:
    print("   ❌ joblib not available - install with: pip install joblib")
    JOBLIB_AVAILABLE = False

import multiprocessing as mp

def simulate_tile_processing(tile_id: int, processing_time: float = 0.5) -> dict:
    """
    Simulate processing a single tile.
    
    This mimics the actual workload without needing real data.
    """
    import time
    import numpy as np
    
    # Simulate some computation (feature extraction, etc.)
    start = time.perf_counter()
    
    # Simulate feature computation
    n_points = 100000
    points = np.random.rand(n_points, 3).astype(np.float32)
    
    # Simulate normals computation (simplified)
    normals = np.zeros((n_points, 3), dtype=np.float32)
    for i in range(min(1000, n_points)):  # Just a small sample
        normals[i] = np.random.randn(3)
        normals[i] /= np.linalg.norm(normals[i])
    
    # Sleep to simulate I/O and remaining computation
    remaining_time = max(0, processing_time - (time.perf_counter() - start))
    if remaining_time > 0:
        time.sleep(remaining_time)
    
    elapsed = time.perf_counter() - start
    
    return {
        'tile_id': tile_id,
        'time': elapsed,
        'num_patches': np.random.randint(5, 15),
        'success': True
    }


def benchmark_sequential(n_tiles: int, processing_time: float = 0.5) -> dict:
    """Benchmark sequential processing."""
    print(f"\n📊 Benchmarking SEQUENTIAL processing ({n_tiles} tiles)...")
    
    start_time = time.perf_counter()
    results = []
    
    for i in range(n_tiles):
        result = simulate_tile_processing(i, processing_time)
        results.append(result)
        if (i + 1) % 5 == 0:
            print(f"   Processed {i+1}/{n_tiles} tiles...")
    
    elapsed = time.perf_counter() - start_time
    
    total_patches = sum(r['num_patches'] for r in results)
    
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Throughput: {n_tiles/elapsed:.2f} tiles/sec")
    print(f"   Total patches: {total_patches}")
    
    return {
        'time': elapsed,
        'tiles_per_sec': n_tiles / elapsed,
        'total_patches': total_patches,
        'results': results
    }


def benchmark_parallel(n_tiles: int, n_jobs: int = -1, processing_time: float = 0.5) -> dict:
    """Benchmark parallel processing."""
    if not JOBLIB_AVAILABLE:
        print("❌ Cannot benchmark parallel - joblib not available")
        return None
    
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    print(f"\n📊 Benchmarking PARALLEL processing ({n_tiles} tiles, {n_jobs} jobs)...")
    
    start_time = time.perf_counter()
    
    # Process in parallel
    results = Parallel(n_jobs=n_jobs, prefer='processes', verbose=5)(
        delayed(simulate_tile_processing)(i, processing_time)
        for i in range(n_tiles)
    )
    
    elapsed = time.perf_counter() - start_time
    
    total_patches = sum(r['num_patches'] for r in results)
    
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Throughput: {n_tiles/elapsed:.2f} tiles/sec")
    print(f"   Total patches: {total_patches}")
    
    return {
        'time': elapsed,
        'tiles_per_sec': n_tiles / elapsed,
        'total_patches': total_patches,
        'results': results,
        'n_jobs': n_jobs
    }


def main():
    """Main benchmark execution."""
    print("=" * 70)
    print("🚀 PHASE 3 SPRINT 3: PARALLEL TILE PROCESSING BENCHMARK")
    print("=" * 70)
    
    n_cpus = mp.cpu_count()
    print(f"\n💻 System Info:")
    print(f"   CPU cores: {n_cpus}")
    print(f"   Joblib available: {JOBLIB_AVAILABLE}")
    
    if not JOBLIB_AVAILABLE:
        print("\n❌ Joblib not available. Install with:")
        print("   pip install joblib")
        return
    
    # Configuration
    n_tiles = 20  # Number of tiles to simulate
    processing_time_per_tile = 0.5  # seconds
    
    print(f"\n📋 Benchmark Configuration:")
    print(f"   Tiles to process: {n_tiles}")
    print(f"   Processing time per tile: {processing_time_per_tile}s")
    print(f"   Expected sequential time: ~{n_tiles * processing_time_per_tile:.1f}s")
    
    # Benchmark sequential
    seq_results = benchmark_sequential(n_tiles, processing_time_per_tile)
    
    # Benchmark parallel with different job counts
    job_configs = [2, 4, n_cpus]
    if n_cpus > 8:
        job_configs.append(8)  # Also test with 8 jobs
    
    parallel_results = {}
    for n_jobs in sorted(set(job_configs)):
        if n_jobs <= n_cpus:
            parallel_results[n_jobs] = benchmark_parallel(
                n_tiles, n_jobs, processing_time_per_tile
            )
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Configuration':<25} {'Time':<12} {'Throughput':<18} {'Speedup':<10}")
    print("-" * 70)
    
    baseline_time = seq_results['time']
    
    print(f"{'Sequential (1 core)':<25} "
          f"{seq_results['time']:>8.2f}s   "
          f"{seq_results['tiles_per_sec']:>8.2f} tiles/s   "
          f"{'1.00x':>8}")
    
    for n_jobs in sorted(parallel_results.keys()):
        result = parallel_results[n_jobs]
        speedup = baseline_time / result['time']
        efficiency = (speedup / n_jobs) * 100
        
        print(f"{'Parallel (' + str(n_jobs) + ' cores)':<25} "
              f"{result['time']:>8.2f}s   "
              f"{result['tiles_per_sec']:>8.2f} tiles/s   "
              f"{speedup:>7.2f}x")
    
    # Best configuration
    best_n_jobs = max(parallel_results.keys(), 
                     key=lambda k: parallel_results[k]['tiles_per_sec'])
    best_result = parallel_results[best_n_jobs]
    best_speedup = baseline_time / best_result['time']
    
    print("\n" + "=" * 70)
    print(f"🏆 BEST CONFIGURATION: {best_n_jobs} cores")
    print(f"   Speedup: {best_speedup:.2f}x faster than sequential")
    print(f"   Time: {seq_results['time']:.2f}s → {best_result['time']:.2f}s")
    print(f"   Time saved: {seq_results['time'] - best_result['time']:.2f}s "
          f"({(1 - best_result['time']/seq_results['time'])*100:.1f}%)")
    
    # Scaling efficiency
    print(f"\n📈 PARALLEL SCALING EFFICIENCY:")
    for n_jobs in sorted(parallel_results.keys()):
        result = parallel_results[n_jobs]
        speedup = baseline_time / result['time']
        efficiency = (speedup / n_jobs) * 100
        
        efficiency_bar = "█" * int(efficiency / 5) + "░" * (20 - int(efficiency / 5))
        print(f"   {n_jobs:2d} cores: {efficiency:>5.1f}% [{efficiency_bar}]")
    
    print("\n" + "=" * 70)
    
    if best_speedup >= 2.0:
        print("🎉 EXCELLENT! Achieved 2x+ speedup with parallel processing!")
    elif best_speedup >= 1.5:
        print("✅ GOOD! Achieved 1.5x+ speedup with parallel processing")
    else:
        print("⚠️  Speedup below 1.5x - may be limited by I/O or overhead")
    
    print("=" * 70)
    
    # Save results
    import json
    output_file = 'benchmark_parallel_processing.json'
    
    results_json = {
        'sequential': {
            'time': float(seq_results['time']),
            'tiles_per_sec': float(seq_results['tiles_per_sec'])
        },
        'parallel': {
            str(k): {
                'time': float(v['time']),
                'tiles_per_sec': float(v['tiles_per_sec']),
                'n_jobs': int(v['n_jobs']),
                'speedup': float(baseline_time / v['time'])
            }
            for k, v in parallel_results.items()
        },
        'best_configuration': {
            'n_jobs': int(best_n_jobs),
            'speedup': float(best_speedup)
        },
        'system': {
            'cpu_cores': int(n_cpus)
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_file}")


if __name__ == '__main__':
    main()
