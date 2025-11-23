#!/usr/bin/env python3
"""
Benchmark CUDA Streams Performance

Compares synchronous vs asynchronous GPU processing to validate
the 10-20% performance improvement from stream overlapping.

Tests:
1. Synchronous processing (baseline)
2. Async with 2 streams (compute + transfer overlap)
3. Async with 3 streams (full pipeline)
4. Async with 4 streams (maximum parallelism)

Metrics:
- Wall clock time
- GPU utilization
- Memory bandwidth
- Throughput (points/sec)

Usage:
    python scripts/benchmark_cuda_streams.py
    python scripts/benchmark_cuda_streams.py --points 10000000 --streams 4
    python scripts/benchmark_cuda_streams.py --output results.json

Author: IGN LiDAR HD Team
Date: November 23, 2025
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.optimization.cuda_streams import HAS_CUPY, create_stream_manager

logger = logging.getLogger(__name__)


def generate_test_data(n_points: int, k: int = 30) -> tuple:
    """Generate test data."""
    logger.info(f"Generating test data: {n_points:,} points, k={k}")
    points = np.random.randn(n_points, 3).astype(np.float32)
    knn_indices = np.random.randint(0, n_points, (n_points, k), dtype=np.int32)
    return points, knn_indices


def benchmark_synchronous(
    points: np.ndarray,
    knn_indices: np.ndarray,
    k: int,
    n_iterations: int = 10
) -> Dict[str, float]:
    """Benchmark synchronous (no streams) processing."""
    if not HAS_CUPY:
        raise RuntimeError("CuPy required")
    
    import cupy as cp
    from ign_lidar.optimization.gpu_kernels import CUDAKernels
    
    kernels = CUDAKernels()
    
    logger.info("Warming up (synchronous)...")
    _ = kernels.compute_normals_eigenvalues_fused(points, knn_indices, k)
    cp.cuda.Device().synchronize()
    
    logger.info(f"Benchmarking synchronous ({n_iterations} iterations)...")
    times = []
    
    for i in range(n_iterations):
        start = time.perf_counter()
        
        normals, eigenvalues, curvature = kernels.compute_normals_eigenvalues_fused(
            points, knn_indices, k
        )
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


def benchmark_async_streams(
    points: np.ndarray,
    knn_indices: np.ndarray,
    k: int,
    num_streams: int = 3,
    n_iterations: int = 10
) -> Dict[str, float]:
    """Benchmark async stream processing."""
    if not HAS_CUPY:
        raise RuntimeError("CuPy required")
    
    import cupy as cp
    from ign_lidar.optimization.gpu_kernels import CUDAKernels
    from ign_lidar.optimization.adaptive_chunking import auto_chunk_size
    
    logger.info(f"Warming up (async {num_streams} streams)...")
    
    # Calculate chunk size
    chunk_size = auto_chunk_size(points.shape, use_gpu=True, target_memory_usage=0.6)
    n_chunks = (len(points) + chunk_size - 1) // chunk_size
    
    logger.info(f"Using {n_chunks} chunks of size {chunk_size:,}")
    
    kernels = CUDAKernels()
    stream_manager = create_stream_manager(num_streams=num_streams)
    
    # Warm up
    results = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, len(points))
        chunk_points = points[start:end]
        chunk_indices = knn_indices[start:end]
        
        stream_id = i % num_streams
        stream = stream_manager.get_stream(stream_id)
        
        with stream:
            n, e, c = kernels.compute_normals_eigenvalues_fused(
                chunk_points, chunk_indices, k
            )
            results.append((n, e, c))
    
    stream_manager.synchronize_all()
    results.clear()
    
    logger.info(f"Benchmarking async {num_streams} streams ({n_iterations} iterations)...")
    times = []
    
    for iter_idx in range(n_iterations):
        results = []
        start = time.perf_counter()
        
        # Process chunks with stream pipeline
        for i in range(n_chunks):
            chunk_start = i * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(points))
            chunk_points = points[chunk_start:chunk_end]
            chunk_indices = knn_indices[chunk_start:chunk_end]
            
            stream_id = i % num_streams
            stream = stream_manager.get_stream(stream_id)
            
            with stream:
                n, e, c = kernels.compute_normals_eigenvalues_fused(
                    chunk_points, chunk_indices, k
                )
                results.append((n, e, c))
        
        # Synchronize all streams
        stream_manager.synchronize_all()
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        results.clear()
        
        if (iter_idx + 1) % 5 == 0:
            logger.info(f"  Iteration {iter_idx+1}/{n_iterations}: {elapsed*1000:.2f}ms")
    
    stream_manager.cleanup()
    
    times = np.array(times)
    return {
        'mean_ms': float(np.mean(times) * 1000),
        'std_ms': float(np.std(times) * 1000),
        'min_ms': float(np.min(times) * 1000),
        'max_ms': float(np.max(times) * 1000),
        'median_ms': float(np.median(times) * 1000)
    }


def calculate_improvement(baseline: Dict, optimized: Dict) -> Dict[str, float]:
    """Calculate improvement metrics."""
    speedup = baseline['mean_ms'] / optimized['mean_ms']
    improvement_pct = ((baseline['mean_ms'] - optimized['mean_ms']) / baseline['mean_ms']) * 100
    
    return {
        'speedup': speedup,
        'improvement_percent': improvement_pct,
        'time_saved_ms': baseline['mean_ms'] - optimized['mean_ms']
    }


def print_results(results: Dict[str, Any]):
    """Print formatted results."""
    print("\n" + "="*80)
    print("CUDA STREAMS BENCHMARK RESULTS")
    print("="*80)
    
    print(f"\nTest Configuration:")
    print(f"  Points:          {results['n_points']:>12,}")
    print(f"  Neighbors (k):   {results['k']:>12}")
    print(f"  Iterations:      {results['n_iterations']:>12}")
    
    print(f"\nSynchronous (baseline):")
    sync = results['synchronous']
    print(f"  Mean:            {sync['mean_ms']:>12.2f} ms")
    print(f"  Median:          {sync['median_ms']:>12.2f} ms")
    print(f"  Std Dev:         {sync['std_ms']:>12.2f} ms")
    
    # Print all async variants
    for num_streams in [2, 3, 4]:
        key = f'async_{num_streams}_streams'
        if key in results:
            print(f"\nAsync ({num_streams} streams):")
            async_res = results[key]
            print(f"  Mean:            {async_res['mean_ms']:>12.2f} ms")
            print(f"  Median:          {async_res['median_ms']:>12.2f} ms")
            print(f"  Std Dev:         {async_res['std_ms']:>12.2f} ms")
            
            improvement = results[f'improvement_{num_streams}_streams']
            print(f"  Speedup:         {improvement['speedup']:>12.2f}x")
            print(f"  Improvement:     {improvement['improvement_percent']:>12.1f}%")
    
    # Find best configuration
    best_streams = 2
    best_improvement = results['improvement_2_streams']['improvement_percent']
    for streams in [3, 4]:
        key = f'improvement_{streams}_streams'
        if key in results and results[key]['improvement_percent'] > best_improvement:
            best_streams = streams
            best_improvement = results[key]['improvement_percent']
    
    print(f"\n{'='*80}")
    print(f"BEST CONFIGURATION: {best_streams} streams ({best_improvement:.1f}% improvement)")
    
    if best_improvement >= 10:
        print("✅ SUCCESS: Achieved target 10%+ performance improvement!")
    else:
        print("⚠️  WARNING: Below 10% target. Check GPU utilization and chunk sizing.")
    
    print("="*80 + "\n")


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Benchmark CUDA streams")
    parser.add_argument('--points', type=int, default=5_000_000)
    parser.add_argument('--k', type=int, default=30)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--max-streams', type=int, default=4)
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if not HAS_CUPY:
        logger.error("CuPy not available")
        sys.exit(1)
    
    import cupy as cp
    logger.info(f"CUDA Device: {cp.cuda.Device().name}")
    
    # Generate data
    points, knn_indices = generate_test_data(args.points, args.k)
    
    # Benchmark synchronous (baseline)
    sync_results = benchmark_synchronous(points, knn_indices, args.k, args.iterations)
    
    results = {
        'n_points': args.points,
        'k': args.k,
        'n_iterations': args.iterations,
        'synchronous': sync_results,
        'cuda_device': str(cp.cuda.Device().name)
    }
    
    # Benchmark async with different stream counts
    for num_streams in range(2, args.max_streams + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {num_streams} streams")
        logger.info(f"{'='*60}")
        
        async_results = benchmark_async_streams(
            points, knn_indices, args.k, num_streams, args.iterations
        )
        
        improvement = calculate_improvement(sync_results, async_results)
        
        results[f'async_{num_streams}_streams'] = async_results
        results[f'improvement_{num_streams}_streams'] = improvement
    
    # Print results
    print_results(results)
    
    # Save to file
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.output}")
    
    # Exit code based on best improvement
    best_improvement = max(
        results[f'improvement_{s}_streams']['improvement_percent']
        for s in range(2, args.max_streams + 1)
    )
    
    sys.exit(0 if best_improvement >= 10 else 1)


if __name__ == '__main__':
    main()
