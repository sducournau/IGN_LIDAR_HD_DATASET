#!/usr/bin/env python3
"""
Benchmark GPU kernel fusion performance.

Compares:
- Sequential GPU kernels (covariance → eigenvalues → normals)
- Fused GPU kernel (all in one pass)

Measures:
- Execution time
- Memory transfers
- Speedup factor
- GPU utilization

Usage:
    python scripts/benchmark_gpu_fusion.py
    python scripts/benchmark_gpu_fusion.py --points 5000000 --k 30
    python scripts/benchmark_gpu_fusion.py --output results.json

Author: IGN LiDAR HD Team
Date: November 23, 2025
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.optimization.gpu_kernels import CUDAKernels, HAS_CUPY

logger = logging.getLogger(__name__)


def generate_test_data(n_points: int, k_neighbors: int) -> tuple:
    """
    Generate synthetic test data.
    
    Args:
        n_points: Number of points
        k_neighbors: Number of neighbors
    
    Returns:
        points: Point cloud (N, 3)
        knn_indices: KNN indices (N, k)
    """
    logger.info(f"Generating test data: {n_points:,} points, k={k_neighbors}")
    
    # Generate random point cloud
    points = np.random.randn(n_points, 3).astype(np.float32)
    
    # Generate random KNN indices (in practice, these would come from KNN search)
    knn_indices = np.random.randint(0, n_points, (n_points, k_neighbors), dtype=np.int32)
    
    return points, knn_indices


def benchmark_sequential(
    kernels: CUDAKernels,
    points: np.ndarray,
    knn_indices: np.ndarray,
    k: int,
    n_iterations: int = 10
) -> Dict[str, float]:
    """
    Benchmark sequential kernel approach.
    
    Computes covariance, eigenvalues, and normals in separate kernel launches.
    
    Args:
        kernels: CUDAKernels instance
        points: Point cloud
        knn_indices: KNN indices
        k: Number of neighbors
        n_iterations: Number of benchmark iterations
    
    Returns:
        Statistics dictionary
    """
    if not HAS_CUPY:
        raise RuntimeError("CuPy required for GPU benchmarking")
    
    import cupy as cp
    
    logger.info("Warming up GPU (sequential approach)...")
    # Warm up
    covariance, _ = kernels.compute_covariance(points, knn_indices, k)
    normals, eigenvalues = kernels.compute_normals_and_eigenvalues(covariance)
    cp.cuda.Device().synchronize()
    
    logger.info(f"Benchmarking sequential approach ({n_iterations} iterations)...")
    times = []
    
    for i in range(n_iterations):
        start = time.perf_counter()
        
        # Step 1: Compute covariance
        covariance, centroids = kernels.compute_covariance(points, knn_indices, k)
        
        # Step 2: Compute normals and eigenvalues
        normals, eigenvalues = kernels.compute_normals_and_eigenvalues(covariance)
        
        # Synchronize to ensure completion
        cp.cuda.Device().synchronize()
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        if (i + 1) % 5 == 0:
            logger.info(f"  Iteration {i+1}/{n_iterations}: {elapsed*1000:.2f}ms")
    
    times = np.array(times)
    
    return {
        'mean_ms': float(np.mean(times) * 1000),
        'std_ms': float(np.std(times) * 1000),
        'min_ms': float(np.min(times) * 1000),
        'max_ms': float(np.max(times) * 1000),
        'median_ms': float(np.median(times) * 1000)
    }


def benchmark_fused(
    kernels: CUDAKernels,
    points: np.ndarray,
    knn_indices: np.ndarray,
    k: int,
    n_iterations: int = 10
) -> Dict[str, float]:
    """
    Benchmark fused kernel approach.
    
    Computes covariance, eigenvalues, normals, and curvature in single kernel.
    
    Args:
        kernels: CUDAKernels instance
        points: Point cloud
        knn_indices: KNN indices
        k: Number of neighbors
        n_iterations: Number of benchmark iterations
    
    Returns:
        Statistics dictionary
    """
    if not HAS_CUPY:
        raise RuntimeError("CuPy required for GPU benchmarking")
    
    import cupy as cp
    
    logger.info("Warming up GPU (fused approach)...")
    # Warm up
    normals, eigenvalues, curvature = kernels.compute_normals_eigenvalues_fused(
        points, knn_indices, k
    )
    cp.cuda.Device().synchronize()
    
    logger.info(f"Benchmarking fused approach ({n_iterations} iterations)...")
    times = []
    
    for i in range(n_iterations):
        start = time.perf_counter()
        
        # Single fused kernel call
        normals, eigenvalues, curvature = kernels.compute_normals_eigenvalues_fused(
            points, knn_indices, k
        )
        
        # Synchronize to ensure completion
        cp.cuda.Device().synchronize()
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        if (i + 1) % 5 == 0:
            logger.info(f"  Iteration {i+1}/{n_iterations}: {elapsed*1000:.2f}ms")
    
    times = np.array(times)
    
    return {
        'mean_ms': float(np.mean(times) * 1000),
        'std_ms': float(np.std(times) * 1000),
        'min_ms': float(np.min(times) * 1000),
        'max_ms': float(np.max(times) * 1000),
        'median_ms': float(np.median(times) * 1000)
    }


def calculate_speedup(sequential: Dict[str, float], fused: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate speedup metrics.
    
    Args:
        sequential: Sequential benchmark results
        fused: Fused benchmark results
    
    Returns:
        Speedup statistics
    """
    speedup_mean = sequential['mean_ms'] / fused['mean_ms']
    speedup_median = sequential['median_ms'] / fused['median_ms']
    speedup_min = sequential['min_ms'] / fused['min_ms']
    
    improvement_pct = ((sequential['mean_ms'] - fused['mean_ms']) / sequential['mean_ms']) * 100
    
    return {
        'speedup_mean': speedup_mean,
        'speedup_median': speedup_median,
        'speedup_min': speedup_min,
        'improvement_percent': improvement_pct
    }


def print_results(results: Dict[str, Any]) -> None:
    """Print benchmark results in formatted table."""
    print("\n" + "="*80)
    print("GPU KERNEL FUSION BENCHMARK RESULTS")
    print("="*80)
    
    print(f"\nTest Configuration:")
    print(f"  Points:          {results['n_points']:>12,}")
    print(f"  Neighbors (k):   {results['k_neighbors']:>12}")
    print(f"  Iterations:      {results['n_iterations']:>12}")
    
    print(f"\nSequential Approach (3 kernel launches):")
    seq = results['sequential']
    print(f"  Mean:            {seq['mean_ms']:>12.2f} ms")
    print(f"  Median:          {seq['median_ms']:>12.2f} ms")
    print(f"  Std Dev:         {seq['std_ms']:>12.2f} ms")
    print(f"  Min:             {seq['min_ms']:>12.2f} ms")
    print(f"  Max:             {seq['max_ms']:>12.2f} ms")
    
    print(f"\nFused Approach (1 kernel launch):")
    fused = results['fused']
    print(f"  Mean:            {fused['mean_ms']:>12.2f} ms")
    print(f"  Median:          {fused['median_ms']:>12.2f} ms")
    print(f"  Std Dev:         {fused['std_ms']:>12.2f} ms")
    print(f"  Min:             {fused['min_ms']:>12.2f} ms")
    print(f"  Max:             {fused['max_ms']:>12.2f} ms")
    
    print(f"\nPerformance Improvement:")
    speedup = results['speedup']
    print(f"  Speedup (mean):  {speedup['speedup_mean']:>12.2f}x")
    print(f"  Speedup (median):{speedup['speedup_median']:>12.2f}x")
    print(f"  Improvement:     {speedup['improvement_percent']:>12.1f}%")
    
    print(f"\nThroughput:")
    throughput = results['throughput']
    print(f"  Sequential:      {throughput['sequential_points_per_sec']:>12,.0f} points/sec")
    print(f"  Fused:           {throughput['fused_points_per_sec']:>12,.0f} points/sec")
    
    print("\n" + "="*80)
    
    # Success message
    if speedup['improvement_percent'] >= 30:
        print("✅ EXCELLENT: Achieved target 30%+ performance improvement!")
    elif speedup['improvement_percent'] >= 20:
        print("✅ GOOD: Significant performance improvement achieved.")
    else:
        print("⚠️  WARNING: Performance improvement below expected 30% target.")
    
    print("="*80 + "\n")


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(
        description="Benchmark GPU kernel fusion performance"
    )
    parser.add_argument(
        '--points',
        type=int,
        default=5_000_000,
        help='Number of points to test (default: 5M)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=30,
        help='Number of neighbors (default: 30)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=10,
        help='Number of benchmark iterations (default: 10)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Check GPU availability
    if not HAS_CUPY:
        logger.error("CuPy not available - GPU benchmarking requires CuPy")
        logger.error("Install with: pip install cupy-cuda11x or cupy-cuda12x")
        sys.exit(1)
    
    import cupy as cp
    logger.info(f"CUDA Device: {cp.cuda.Device().name}")
    logger.info(f"CUDA Compute Capability: {cp.cuda.Device().compute_capability}")
    
    # Initialize kernels
    logger.info("Initializing CUDA kernels...")
    kernels = CUDAKernels()
    
    if not kernels.available:
        logger.error("Failed to initialize CUDA kernels")
        sys.exit(1)
    
    # Generate test data
    points, knn_indices = generate_test_data(args.points, args.k)
    
    # Benchmark sequential approach
    sequential_results = benchmark_sequential(
        kernels, points, knn_indices, args.k, args.iterations
    )
    
    # Benchmark fused approach
    fused_results = benchmark_fused(
        kernels, points, knn_indices, args.k, args.iterations
    )
    
    # Calculate speedup
    speedup_results = calculate_speedup(sequential_results, fused_results)
    
    # Calculate throughput
    throughput = {
        'sequential_points_per_sec': args.points / (sequential_results['mean_ms'] / 1000),
        'fused_points_per_sec': args.points / (fused_results['mean_ms'] / 1000)
    }
    
    # Compile results
    results = {
        'n_points': args.points,
        'k_neighbors': args.k,
        'n_iterations': args.iterations,
        'sequential': sequential_results,
        'fused': fused_results,
        'speedup': speedup_results,
        'throughput': throughput,
        'cuda_device': str(cp.cuda.Device().name),
        'cuda_compute_capability': str(cp.cuda.Device().compute_capability)
    }
    
    # Print results
    print_results(results)
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_path}")
    
    # Exit with appropriate code
    if speedup_results['improvement_percent'] >= 20:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
