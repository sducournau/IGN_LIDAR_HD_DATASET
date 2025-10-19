"""
Benchmark script for GPU-Core Bridge Module.

This script measures performance improvements from GPU acceleration
and validates that the bridge pattern maintains performance targets.

Usage:
    python scripts/benchmark_gpu_bridge.py
    python scripts/benchmark_gpu_bridge.py --sizes 10000 100000 500000
    python scripts/benchmark_gpu_bridge.py --runs 10

Author: IGN LiDAR HD Dataset Team
Date: October 2025
"""

import numpy as np
import time
import argparse
from typing import List, Dict
import sys

# Add parent directory to path
sys.path.insert(0, '.')

from ign_lidar.features.core.gpu_bridge import (
    GPUCoreBridge,
    CUPY_AVAILABLE,
)


def generate_test_data(n_points: int, k_neighbors: int = 20):
    """Generate test point cloud and neighbors."""
    np.random.seed(42)
    points = np.random.rand(n_points, 3).astype(np.float32)
    neighbors = np.random.randint(0, n_points, size=(n_points, k_neighbors), dtype=np.int32)
    return points, neighbors


def benchmark_eigenvalues(
    points: np.ndarray,
    neighbors: np.ndarray,
    use_gpu: bool,
    n_runs: int = 5
) -> Dict[str, float]:
    """
    Benchmark eigenvalue computation.
    
    Returns:
        dict: {'mean': mean_time, 'std': std_time, 'min': min_time, 'max': max_time}
    """
    bridge = GPUCoreBridge(use_gpu=use_gpu)
    times = []
    
    # Warmup run
    _ = bridge.compute_eigenvalues_gpu(points, neighbors)
    
    # Timed runs
    for _ in range(n_runs):
        start = time.time()
        _ = bridge.compute_eigenvalues_gpu(points, neighbors)
        elapsed = time.time() - start
        times.append(elapsed)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
    }


def benchmark_features(
    points: np.ndarray,
    neighbors: np.ndarray,
    use_gpu: bool,
    n_runs: int = 5
) -> Dict[str, float]:
    """
    Benchmark full feature computation.
    
    Returns:
        dict: {'mean': mean_time, 'std': std_time, 'min': min_time, 'max': max_time}
    """
    bridge = GPUCoreBridge(use_gpu=use_gpu)
    times = []
    
    # Warmup run
    _ = bridge.compute_eigenvalue_features_gpu(points, neighbors)
    
    # Timed runs
    for _ in range(n_runs):
        start = time.time()
        _ = bridge.compute_eigenvalue_features_gpu(points, neighbors)
        elapsed = time.time() - start
        times.append(elapsed)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
    }


def print_results(results: Dict, n_points: int):
    """Print benchmark results in a formatted table."""
    print(f"\n{'=' * 80}")
    print(f"Dataset: {n_points:,} points, k=20 neighbors")
    print(f"{'=' * 80}")
    
    # Eigenvalue computation
    print("\n1. EIGENVALUE COMPUTATION")
    print(f"   CPU: {results['cpu_eigenvalues']['mean']:.3f}s ± {results['cpu_eigenvalues']['std']:.3f}s")
    if CUPY_AVAILABLE and 'gpu_eigenvalues' in results:
        print(f"   GPU: {results['gpu_eigenvalues']['mean']:.3f}s ± {results['gpu_eigenvalues']['std']:.3f}s")
        speedup = results['cpu_eigenvalues']['mean'] / results['gpu_eigenvalues']['mean']
        print(f"   Speedup: {speedup:.1f}×")
        
        if speedup >= 8.0:
            print(f"   ✅ Performance target met (>= 8×)")
        else:
            print(f"   ⚠️  Below target speedup (expected >= 8×)")
    else:
        print("   GPU: Not available")
    
    # Feature computation
    print("\n2. FULL FEATURE COMPUTATION")
    print(f"   CPU: {results['cpu_features']['mean']:.3f}s ± {results['cpu_features']['std']:.3f}s")
    if CUPY_AVAILABLE and 'gpu_features' in results:
        print(f"   GPU: {results['gpu_features']['mean']:.3f}s ± {results['gpu_features']['std']:.3f}s")
        speedup = results['cpu_features']['mean'] / results['gpu_features']['mean']
        print(f"   Speedup: {speedup:.1f}×")
    
    # Overhead analysis
    print("\n3. OVERHEAD ANALYSIS")
    cpu_overhead = results['cpu_features']['mean'] - results['cpu_eigenvalues']['mean']
    cpu_overhead_pct = (cpu_overhead / results['cpu_features']['mean']) * 100
    print(f"   CPU feature overhead: {cpu_overhead:.3f}s ({cpu_overhead_pct:.1f}%)")
    
    if CUPY_AVAILABLE and 'gpu_features' in results:
        gpu_overhead = results['gpu_features']['mean'] - results['gpu_eigenvalues']['mean']
        gpu_overhead_pct = (gpu_overhead / results['gpu_features']['mean']) * 100
        print(f"   GPU feature overhead: {gpu_overhead:.3f}s ({gpu_overhead_pct:.1f}%)")
        
        transfer_overhead = gpu_overhead - cpu_overhead
        print(f"   GPU transfer overhead: ~{transfer_overhead:.3f}s")


def run_benchmark_suite(
    dataset_sizes: List[int],
    n_runs: int = 5,
    k_neighbors: int = 20
):
    """Run complete benchmark suite."""
    print("=" * 80)
    print("GPU-CORE BRIDGE BENCHMARK SUITE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  - Dataset sizes: {dataset_sizes}")
    print(f"  - Runs per test: {n_runs}")
    print(f"  - Neighbors (k): {k_neighbors}")
    print(f"  - CuPy available: {CUPY_AVAILABLE}")
    
    all_results = []
    
    for n_points in dataset_sizes:
        print(f"\n{'=' * 80}")
        print(f"Generating test data: {n_points:,} points...")
        points, neighbors = generate_test_data(n_points, k_neighbors)
        
        results = {}
        
        # Benchmark CPU eigenvalues
        print("Benchmarking CPU eigenvalue computation...")
        results['cpu_eigenvalues'] = benchmark_eigenvalues(points, neighbors, use_gpu=False, n_runs=n_runs)
        
        # Benchmark GPU eigenvalues
        if CUPY_AVAILABLE:
            print("Benchmarking GPU eigenvalue computation...")
            results['gpu_eigenvalues'] = benchmark_eigenvalues(points, neighbors, use_gpu=True, n_runs=n_runs)
        
        # Benchmark CPU features
        print("Benchmarking CPU feature computation...")
        results['cpu_features'] = benchmark_features(points, neighbors, use_gpu=False, n_runs=n_runs)
        
        # Benchmark GPU features
        if CUPY_AVAILABLE:
            print("Benchmarking GPU feature computation...")
            results['gpu_features'] = benchmark_features(points, neighbors, use_gpu=True, n_runs=n_runs)
        
        print_results(results, n_points)
        all_results.append((n_points, results))
    
    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    
    if CUPY_AVAILABLE:
        print("\nGPU Speedups (Eigenvalue Computation):")
        for n_points, results in all_results:
            if 'gpu_eigenvalues' in results:
                speedup = results['cpu_eigenvalues']['mean'] / results['gpu_eigenvalues']['mean']
                status = "✅" if speedup >= 8.0 else "⚠️"
                print(f"  {status} {n_points:>10,} points: {speedup:>5.1f}× speedup")
    else:
        print("\n⚠️  CuPy not available - install with: pip install cupy-cuda11x")
    
    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Benchmark GPU-Core Bridge performance',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--sizes',
        type=int,
        nargs='+',
        default=[10_000, 100_000, 500_000],
        help='Dataset sizes to test'
    )
    parser.add_argument(
        '--runs',
        type=int,
        default=5,
        help='Number of runs per test'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=20,
        help='Number of neighbors'
    )
    
    args = parser.parse_args()
    
    run_benchmark_suite(
        dataset_sizes=args.sizes,
        n_runs=args.runs,
        k_neighbors=args.k
    )


if __name__ == '__main__':
    main()
