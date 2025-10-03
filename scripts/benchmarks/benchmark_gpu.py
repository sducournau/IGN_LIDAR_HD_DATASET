#!/usr/bin/env python3
"""
GPU vs CPU Performance Benchmark

Compares performance of feature computation between CPU and GPU
implementations. Tests various point cloud sizes and documents speedups.

Usage:
    python scripts/benchmarks/benchmark_gpu.py <laz_file> [--k <neighbors>]
    python scripts/benchmarks/benchmark_gpu.py --synthetic
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ign_lidar.features import compute_all_features_optimized  # noqa: E402
from ign_lidar.features_gpu import (  # noqa: E402
    GPU_AVAILABLE,
    CUML_AVAILABLE
)

if GPU_AVAILABLE:
    from ign_lidar.features import compute_all_features_with_gpu  # noqa


def generate_synthetic_pointcloud(
    num_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic point cloud for testing.
    
    Args:
        num_points: Number of points to generate
        
    Returns:
        points: [N, 3] coordinates
        classification: [N] class labels
    """
    # Generate points in a 100m x 100m x 20m volume
    points = np.random.rand(num_points, 3).astype(np.float32)
    points[:, 0] *= 100  # X: 0-100m
    points[:, 1] *= 100  # Y: 0-100m
    points[:, 2] *= 20   # Z: 0-20m
    
    # Random classification (mix of ground, building, vegetation)
    classification = np.random.choice(
        [2, 6, 5], size=num_points
    ).astype(np.uint8)
    
    return points, classification


def benchmark_cpu(
    points: np.ndarray,
    classification: np.ndarray,
    k: int,
    num_runs: int = 3
) -> Dict[str, float]:
    """
    Benchmark CPU feature computation.
    
    Args:
        points: [N, 3] point coordinates
        classification: [N] class labels
        k: number of neighbors
        num_runs: number of benchmark runs (best of)
        
    Returns:
        Benchmark results dictionary
    """
    times = []
    
    for run in range(num_runs):
        start = time.time()
        results = compute_all_features_optimized(
            points=points,
            classification=classification,
            k=k,
            auto_k=False
        )
        normals, curvature, height, geo_features = results
        elapsed = time.time() - start
        times.append(elapsed)
    
    best_time = min(times)
    
    return {
        'best_time': best_time,
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'points_per_sec': len(points) / best_time,
        'all_times': times
    }


def benchmark_gpu(
    points: np.ndarray,
    classification: np.ndarray,
    k: int,
    num_runs: int = 3
) -> Dict[str, float]:
    """
    Benchmark GPU feature computation.
    
    Args:
        points: [N, 3] point coordinates
        classification: [N] class labels
        k: number of neighbors
        num_runs: number of benchmark runs (best of)
        
    Returns:
        Benchmark results dictionary
    """
    if not GPU_AVAILABLE:
        return None
    
    times = []
    
    # Warmup run (GPU initialization)
    _ = compute_all_features_with_gpu(
        points=points,
        classification=classification,
        k=k,
        auto_k=False,
        use_gpu=True
    )
    
    # Actual benchmark runs
    for run in range(num_runs):
        start = time.time()
        results = compute_all_features_with_gpu(
            points=points,
            classification=classification,
            k=k,
            auto_k=False,
            use_gpu=True
        )
        normals, curvature, height, geo_features = results
        elapsed = time.time() - start
        times.append(elapsed)
    
    best_time = min(times)
    
    return {
        'best_time': best_time,
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'points_per_sec': len(points) / best_time,
        'all_times': times
    }


def print_results(
    num_points: int,
    k: int,
    cpu_results: Dict[str, float],
    gpu_results: Optional[Dict[str, float]] = None
):
    """Print formatted benchmark results."""
    
    print(f"\n{'='*80}")
    print("📊 BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"Dataset:        {num_points:,} points")
    print(f"K-neighbors:    {k}")
    print("")
    
    # CPU Results
    print("🖥️  CPU Performance:")
    print(f"  Best time:    {cpu_results['best_time']:.2f}s")
    avg = cpu_results['avg_time']
    std = cpu_results['std_time']
    print(f"  Average:      {avg:.2f}s ± {std:.2f}s")
    print(f"  Throughput:   {cpu_results['points_per_sec']:,.0f} points/s")
    
    # GPU Results
    if gpu_results:
        print("\n⚡ GPU Performance:")
        print(f"  Best time:    {gpu_results['best_time']:.2f}s")
        avg = gpu_results['avg_time']
        std = gpu_results['std_time']
        print(f"  Average:      {avg:.2f}s ± {std:.2f}s")
        tput = gpu_results['points_per_sec']
        print(f"  Throughput:   {tput:,.0f} points/s")
        
        # Speedup
        speedup = cpu_results['best_time'] / gpu_results['best_time']
        print(f"\n🚀 Speedup:       {speedup:.2f}x faster on GPU")
        
        # Time saved
        time_saved = cpu_results['best_time'] - gpu_results['best_time']
        pct = 100 * time_saved / cpu_results['best_time']
        print(f"   Time saved:   {time_saved:.2f}s ({pct:.1f}%)")
    else:
        print("\n⚠️  GPU not available - cannot compare performance")
    
    print(f"{'='*80}")


def benchmark_multiple_sizes(k: int = 10):
    """Benchmark multiple point cloud sizes."""
    
    sizes = [
        1_000,      # 1K points
        10_000,     # 10K points
        100_000,    # 100K points
        1_000_000,  # 1M points
        5_000_000,  # 5M points
    ]
    
    print(f"\n{'='*80}")
    print("🔬 MULTI-SIZE BENCHMARK")
    print(f"{'='*80}")
    print(f"Testing with k={k} neighbors")
    print(f"GPU Available: {GPU_AVAILABLE}")
    print(f"RAPIDS cuML: {CUML_AVAILABLE}")
    print("")
    
    results = []
    
    for num_points in sizes:
        print(f"\n{'─'*80}")
        print(f"Generating {num_points:,} points...")
        
        points, classification = generate_synthetic_pointcloud(num_points)
        
        # Benchmark CPU
        print("Running CPU benchmark...")
        cpu_results = benchmark_cpu(points, classification, k=k, num_runs=3)
        
        # Benchmark GPU
        gpu_results = None
        if GPU_AVAILABLE:
            print("Running GPU benchmark...")
            gpu_results = benchmark_gpu(
                points, classification, k=k, num_runs=3
            )
        
        # Print results for this size
        print_results(num_points, k, cpu_results, gpu_results)
        
        # Store for summary
        results.append({
            'num_points': num_points,
            'cpu': cpu_results,
            'gpu': gpu_results
        })
    
    # Print summary table
    print(f"\n{'='*80}")
    print("📈 SUMMARY TABLE")
    print(f"{'='*80}")
    header = (
        f"{'Points':<12} {'CPU Time':<12} {'GPU Time':<12} "
        f"{'Speedup':<12} {'Status':<15}"
    )
    print(header)
    print(f"{'─'*80}")
    
    for result in results:
        n = result['num_points']
        cpu_time = result['cpu']['best_time']
        
        if result['gpu']:
            gpu_time = result['gpu']['best_time']
            speedup = cpu_time / gpu_time
            status = f"{speedup:.2f}x faster"
        else:
            gpu_time = "N/A"
            speedup = "N/A"
            status = "GPU unavailable"
        
        row = (
            f"{n:>11,} {cpu_time:>11.2f}s {str(gpu_time):>11} "
            f"{str(speedup):>11} {status:<15}"
        )
        print(row)
    
    print(f"{'='*80}\n")
    
    return results


def load_laz_file(laz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load LAZ file and return points and classification."""
    try:
        import laspy
    except ImportError:
        print("❌ Error: laspy not installed. Install with: pip install laspy")
        sys.exit(1)
    
    print(f"Loading LAZ file: {laz_path}")
    las = laspy.read(str(laz_path))
    
    points = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
    classification = np.array(las.classification, dtype=np.uint8)
    
    print(f"Loaded {len(points):,} points")
    
    return points, classification


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark CPU vs GPU feature computation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'laz_file',
        nargs='?',
        type=Path,
        help='Path to LAZ file to benchmark'
    )
    
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic data instead of LAZ file'
    )
    
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='Number of neighbors for feature computation (default: 10)'
    )
    
    parser.add_argument(
        '--multi-size',
        action='store_true',
        help='Run benchmarks on multiple point cloud sizes'
    )
    
    parser.add_argument(
        '--runs',
        type=int,
        default=3,
        help='Number of benchmark runs (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Print system info
    print(f"\n{'='*80}")
    print("🔧 SYSTEM INFORMATION")
    print(f"{'='*80}")
    print(f"GPU Available:    {GPU_AVAILABLE}")
    print(f"RAPIDS cuML:      {CUML_AVAILABLE}")
    
    if GPU_AVAILABLE:
        try:
            import cupy as cp
            print(f"CuPy version:     {cp.__version__}")
            print(f"CUDA version:     {cp.cuda.runtime.runtimeGetVersion()}")
            
            # Get GPU info
            device = cp.cuda.Device()
            props = cp.cuda.runtime.getDeviceProperties(device.id)
            print(f"GPU Device:       {props['name'].decode()}")
            print(f"GPU Memory:       {props['totalGlobalMem'] / 1e9:.1f} GB")
        except Exception as e:
            print(f"⚠️  Could not get GPU info: {e}")
    
    print(f"{'='*80}")
    
    # Run appropriate benchmark
    if args.multi_size:
        benchmark_multiple_sizes(k=args.k)
    elif args.synthetic:
        print("\n📦 Generating synthetic point cloud...")
        points, classification = generate_synthetic_pointcloud(100_000)
        
        print("Running benchmarks...")
        cpu_results = benchmark_cpu(
            points, classification, k=args.k, num_runs=args.runs
        )
        if GPU_AVAILABLE:
            gpu_results = benchmark_gpu(
                points, classification, k=args.k, num_runs=args.runs
            )
        else:
            gpu_results = None
        
        print_results(len(points), args.k, cpu_results, gpu_results)
    elif args.laz_file:
        if not args.laz_file.exists():
            print(f"❌ Error: File not found: {args.laz_file}")
            sys.exit(1)
        
        points, classification = load_laz_file(args.laz_file)
        
        print("\nRunning benchmarks...")
        cpu_results = benchmark_cpu(
            points, classification, k=args.k, num_runs=args.runs
        )
        if GPU_AVAILABLE:
            gpu_results = benchmark_gpu(
                points, classification, k=args.k, num_runs=args.runs
            )
        else:
            gpu_results = None
        
        print_results(len(points), args.k, cpu_results, gpu_results)
    else:
        parser.print_help()
        print("\n💡 Tip: Use --synthetic for quick testing without LAZ files")
        print("💡 Tip: Use --multi-size to test various point cloud sizes")
        sys.exit(1)


if __name__ == '__main__':
    main()
