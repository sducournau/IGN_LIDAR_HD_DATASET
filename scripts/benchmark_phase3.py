"""
Comprehensive GPU Performance Benchmark Suite - Phase 3

This script benchmarks all Phase 3 GPU optimizations:
1. CUDA Streams (async processing)
2. Kernel Fusion (Phase 2)
3. Adaptive Chunking
4. Memory Safety Checks

Generates performance reports for CI/CD integration and regression detection.

Usage:
    python scripts/benchmark_phase3.py --quick
    python scripts/benchmark_phase3.py --full --output results.json
    python scripts/benchmark_phase3.py --compare baseline.json

Author: IGN LiDAR HD Team
Date: November 23, 2025 (Phase 3)
Version: 3.8.0
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for GPU availability
try:
    import cupy as cp
    HAS_CUPY = True
    logger.info("✓ CuPy available")
except ImportError:
    HAS_CUPY = False
    cp = None
    logger.warning("✗ CuPy not available - GPU benchmarks will be skipped")


@dataclass
class BenchmarkResult:
    """Result from a single benchmark."""
    name: str
    points: int
    iterations: int
    mean_time: float
    median_time: float
    std_time: float
    min_time: float
    max_time: float
    throughput: float  # points/sec
    memory_used_gb: Optional[float] = None
    speedup: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


def generate_test_data(num_points: int, num_features: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic test data."""
    logger.info(f"Generating test data: {num_points:,} points, {num_features} features")
    
    points = np.random.randn(num_points, num_features).astype(np.float32)
    knn_indices = np.random.randint(0, num_points, (num_points, 30), dtype=np.int32)
    
    return points, knn_indices


def benchmark_function(
    func,
    *args,
    iterations: int = 10,
    warmup: int = 2,
    **kwargs
) -> Tuple[float, float, float, float, float]:
    """
    Benchmark a function with multiple iterations.
    
    Returns:
        Tuple of (mean, median, std, min, max) in seconds
    """
    # Warmup
    for _ in range(warmup):
        _ = func(*args, **kwargs)
        if HAS_CUPY:
            cp.cuda.Stream.null.synchronize()
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.time()
        _ = func(*args, **kwargs)
        if HAS_CUPY:
            cp.cuda.Stream.null.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
    
    times = np.array(times)
    return (
        np.mean(times),
        np.median(times),
        np.std(times),
        np.min(times),
        np.max(times)
    )


def benchmark_kernel_fusion(
    points: np.ndarray,
    knn_indices: np.ndarray,
    iterations: int = 10
) -> Dict[str, BenchmarkResult]:
    """Benchmark Phase 2 kernel fusion."""
    if not HAS_CUPY:
        logger.warning("Skipping kernel fusion benchmark (no CuPy)")
        return {}
    
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK: Kernel Fusion (Phase 2)")
    logger.info("="*80)
    
    from ign_lidar.optimization.gpu_kernels import CUDAKernels
    
    kernels = CUDAKernels()
    
    if not kernels.available:
        logger.warning("CUDA kernels not available")
        return {}
    
    num_points = len(points)
    
    # Benchmark fused kernel
    logger.info("Benchmarking fused kernel...")
    mean, median, std, min_t, max_t = benchmark_function(
        kernels.compute_normals_eigenvalues_fused,
        points, knn_indices, 30,
        iterations=iterations
    )
    
    fused_result = BenchmarkResult(
        name="kernel_fusion_fused",
        points=num_points,
        iterations=iterations,
        mean_time=mean,
        median_time=median,
        std_time=std,
        min_time=min_t,
        max_time=max_t,
        throughput=num_points / mean
    )
    
    logger.info(f"✓ Fused kernel: {mean*1000:.2f}ms (±{std*1000:.2f}ms)")
    logger.info(f"  Throughput: {fused_result.throughput/1e6:.1f}M points/sec")
    
    return {
        "fused": fused_result
    }


def benchmark_cuda_streams(
    chunks: List[np.ndarray],
    iterations: int = 5
) -> Dict[str, BenchmarkResult]:
    """Benchmark Phase 3 CUDA streams."""
    if not HAS_CUPY:
        logger.warning("Skipping CUDA streams benchmark (no CuPy)")
        return {}
    
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK: CUDA Streams (Phase 3)")
    logger.info("="*80)
    
    from ign_lidar.optimization.cuda_streams import CUDAStreamManager
    
    total_points = sum(len(chunk) for chunk in chunks)
    
    # Define processing function
    def process_chunk(gpu_data):
        """Simple processing: compute mean."""
        return cp.mean(gpu_data, axis=1)
    
    # Benchmark sequential (no streams)
    logger.info("Benchmarking sequential processing...")
    
    def process_sequential():
        results = []
        for chunk in chunks:
            gpu_chunk = cp.asarray(chunk)
            result = process_chunk(gpu_chunk)
            results.append(cp.asnumpy(result))
        return results
    
    mean_seq, median_seq, std_seq, min_seq, max_seq = benchmark_function(
        process_sequential,
        iterations=iterations,
        warmup=1
    )
    
    sequential_result = BenchmarkResult(
        name="cuda_streams_sequential",
        points=total_points,
        iterations=iterations,
        mean_time=mean_seq,
        median_time=median_seq,
        std_time=std_seq,
        min_time=min_seq,
        max_time=max_seq,
        throughput=total_points / mean_seq
    )
    
    logger.info(f"✓ Sequential: {mean_seq*1000:.2f}ms (±{std_seq*1000:.2f}ms)")
    
    # Benchmark with streams
    logger.info("Benchmarking pipeline processing with streams...")
    
    manager = CUDAStreamManager()
    
    mean_stream, median_stream, std_stream, min_stream, max_stream = benchmark_function(
        manager.pipeline_process,
        chunks,
        process_chunk,
        iterations=iterations,
        warmup=1
    )
    
    stream_result = BenchmarkResult(
        name="cuda_streams_pipeline",
        points=total_points,
        iterations=iterations,
        mean_time=mean_stream,
        median_time=median_stream,
        std_time=std_stream,
        min_time=min_stream,
        max_time=max_stream,
        throughput=total_points / mean_stream,
        speedup=mean_seq / mean_stream
    )
    
    logger.info(f"✓ Pipeline: {mean_stream*1000:.2f}ms (±{std_stream*1000:.2f}ms)")
    logger.info(f"  Speedup: {stream_result.speedup:.2f}x")
    logger.info(f"  Throughput: {stream_result.throughput/1e6:.1f}M points/sec")
    
    return {
        "sequential": sequential_result,
        "pipeline": stream_result
    }


def benchmark_adaptive_chunking(
    points: np.ndarray,
    iterations: int = 5
) -> Dict[str, BenchmarkResult]:
    """Benchmark Phase 3 adaptive chunking."""
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK: Adaptive Chunking (Phase 3)")
    logger.info("="*80)
    
    from ign_lidar.optimization.adaptive_chunking import (
        auto_chunk_size,
        get_recommended_strategy,
        estimate_gpu_memory_required
    )
    
    num_points = len(points)
    
    # Benchmark chunk size calculation
    logger.info("Benchmarking auto chunk size calculation...")
    
    times = []
    for _ in range(iterations * 10):  # More iterations since it's fast
        start = time.time()
        chunk_size = auto_chunk_size(points.shape, use_gpu=True)
        elapsed = time.time() - start
        times.append(elapsed)
    
    times = np.array(times)
    
    chunk_calc_result = BenchmarkResult(
        name="adaptive_chunking_calc",
        points=num_points,
        iterations=len(times),
        mean_time=np.mean(times),
        median_time=np.median(times),
        std_time=np.std(times),
        min_time=np.min(times),
        max_time=np.max(times),
        throughput=num_points / np.mean(times)
    )
    
    logger.info(f"✓ Chunk size calculation: {np.mean(times)*1000:.3f}ms")
    logger.info(f"  Recommended chunk size: {chunk_size:,} points")
    
    # Benchmark memory estimation
    logger.info("Benchmarking memory estimation...")
    
    times = []
    for _ in range(iterations * 10):
        start = time.time()
        required_gb = estimate_gpu_memory_required(num_points)
        elapsed = time.time() - start
        times.append(elapsed)
    
    times = np.array(times)
    
    memory_est_result = BenchmarkResult(
        name="adaptive_chunking_memory_est",
        points=num_points,
        iterations=len(times),
        mean_time=np.mean(times),
        median_time=np.median(times),
        std_time=np.std(times),
        min_time=np.min(times),
        max_time=np.max(times),
        throughput=num_points / np.mean(times)
    )
    
    logger.info(f"✓ Memory estimation: {np.mean(times)*1000:.3f}ms")
    logger.info(f"  Estimated memory: {required_gb:.1f}GB")
    
    return {
        "chunk_calc": chunk_calc_result,
        "memory_est": memory_est_result
    }


def benchmark_memory_safety(
    points: np.ndarray,
    iterations: int = 10
) -> Dict[str, BenchmarkResult]:
    """Benchmark Phase 3 memory safety checks."""
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK: Memory Safety Checks (Phase 3)")
    logger.info("="*80)
    
    from ign_lidar.optimization.gpu_safety import check_gpu_memory_safe
    
    num_points = len(points)
    
    # Benchmark memory check
    logger.info("Benchmarking GPU memory safety check...")
    
    times = []
    for _ in range(iterations * 5):
        start = time.time()
        result = check_gpu_memory_safe(points.shape)
        elapsed = time.time() - start
        times.append(elapsed)
    
    times = np.array(times)
    
    safety_check_result = BenchmarkResult(
        name="memory_safety_check",
        points=num_points,
        iterations=len(times),
        mean_time=np.mean(times),
        median_time=np.median(times),
        std_time=np.std(times),
        min_time=np.min(times),
        max_time=np.max(times),
        throughput=num_points / np.mean(times)
    )
    
    logger.info(f"✓ Safety check: {np.mean(times)*1000:.3f}ms")
    logger.info(f"  Strategy: {result.strategy.value}")
    logger.info(f"  Can proceed: {result.can_proceed}")
    
    return {
        "safety_check": safety_check_result
    }


def run_benchmarks(
    quick: bool = False,
    output_file: Optional[str] = None
) -> Dict:
    """Run all benchmarks and generate report."""
    logger.info("\n" + "="*80)
    logger.info("PHASE 3 GPU PERFORMANCE BENCHMARK SUITE")
    logger.info("="*80)
    logger.info(f"Mode: {'Quick' if quick else 'Full'}")
    
    # Test configurations
    if quick:
        point_counts = [100_000, 1_000_000]
        iterations = 5
    else:
        point_counts = [100_000, 1_000_000, 5_000_000]
        iterations = 10
    
    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "quick" if quick else "full",
            "gpu_available": HAS_CUPY,
            "cupy_version": cp.__version__ if HAS_CUPY else None
        },
        "benchmarks": {}
    }
    
    for num_points in point_counts:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing with {num_points:,} points")
        logger.info(f"{'='*80}")
        
        # Generate test data
        points, knn_indices = generate_test_data(num_points)
        
        # Run benchmarks
        test_key = f"{num_points}_points"
        results["benchmarks"][test_key] = {}
        
        # 1. Kernel Fusion
        kernel_results = benchmark_kernel_fusion(points, knn_indices, iterations)
        if kernel_results:
            results["benchmarks"][test_key]["kernel_fusion"] = {
                k: v.to_dict() for k, v in kernel_results.items()
            }
        
        # 2. CUDA Streams (use multiple chunks)
        if HAS_CUPY:
            chunk_size = num_points // 4
            chunks = [points[i:i+chunk_size] for i in range(0, num_points, chunk_size)]
            stream_results = benchmark_cuda_streams(chunks, iterations=max(5, iterations//2))
            if stream_results:
                results["benchmarks"][test_key]["cuda_streams"] = {
                    k: v.to_dict() for k, v in stream_results.items()
                }
        
        # 3. Adaptive Chunking
        chunking_results = benchmark_adaptive_chunking(points, iterations)
        results["benchmarks"][test_key]["adaptive_chunking"] = {
            k: v.to_dict() for k, v in chunking_results.items()
        }
        
        # 4. Memory Safety
        safety_results = benchmark_memory_safety(points, iterations)
        results["benchmarks"][test_key]["memory_safety"] = {
            k: v.to_dict() for k, v in safety_results.items()
        }
    
    # Generate summary
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK SUMMARY")
    logger.info("="*80)
    
    _print_summary(results)
    
    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✓ Results saved to {output_path}")
    
    return results


def _print_summary(results: Dict):
    """Print benchmark summary."""
    for test_key, test_results in results["benchmarks"].items():
        logger.info(f"\n{test_key}:")
        
        for category, benchmarks in test_results.items():
            logger.info(f"  {category}:")
            for name, data in benchmarks.items():
                logger.info(f"    {name}: {data['mean_time']*1000:.2f}ms")
                if data.get('speedup'):
                    logger.info(f"      Speedup: {data['speedup']:.2f}x")


def compare_with_baseline(
    current_results: Dict,
    baseline_file: str
) -> None:
    """Compare current results with baseline."""
    logger.info("\n" + "="*80)
    logger.info("REGRESSION CHECK")
    logger.info("="*80)
    
    with open(baseline_file, 'r') as f:
        baseline = json.load(f)
    
    logger.info(f"Comparing against baseline: {baseline_file}")
    logger.info(f"Baseline date: {baseline['metadata']['timestamp']}")
    
    # Compare results
    regressions = []
    improvements = []
    
    for test_key in current_results["benchmarks"]:
        if test_key not in baseline["benchmarks"]:
            continue
        
        for category in current_results["benchmarks"][test_key]:
            if category not in baseline["benchmarks"][test_key]:
                continue
            
            for name in current_results["benchmarks"][test_key][category]:
                if name not in baseline["benchmarks"][test_key][category]:
                    continue
                
                current_time = current_results["benchmarks"][test_key][category][name]["mean_time"]
                baseline_time = baseline["benchmarks"][test_key][category][name]["mean_time"]
                
                change = (current_time - baseline_time) / baseline_time * 100
                
                test_name = f"{test_key}/{category}/{name}"
                
                if change > 10:  # > 10% slower
                    regressions.append((test_name, change))
                    logger.warning(f"⚠ REGRESSION: {test_name}: +{change:.1f}%")
                elif change < -5:  # > 5% faster
                    improvements.append((test_name, change))
                    logger.info(f"✓ IMPROVEMENT: {test_name}: {change:.1f}%")
    
    logger.info(f"\nRegressions: {len(regressions)}")
    logger.info(f"Improvements: {len(improvements)}")
    
    if regressions:
        logger.warning("\n⚠ Performance regressions detected!")
        return False
    else:
        logger.info("\n✓ No regressions detected")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 GPU Performance Benchmark Suite"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark (fewer iterations, smaller datasets)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full benchmark (more iterations, larger datasets)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--compare",
        type=str,
        help="Baseline file to compare against"
    )
    
    args = parser.parse_args()
    
    # Run benchmarks
    results = run_benchmarks(
        quick=args.quick or not args.full,
        output_file=args.output
    )
    
    # Compare with baseline if provided
    if args.compare:
        success = compare_with_baseline(results, args.compare)
        exit(0 if success else 1)
    
    logger.info("\n✓ Benchmarking complete!")


if __name__ == "__main__":
    main()
