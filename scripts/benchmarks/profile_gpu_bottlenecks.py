#!/usr/bin/env python3
"""
GPU Bottleneck Profiler

Detailed profiling of GPU computation with breakdown of time spent
in each operation. Compares three modes:
1. CPU-only (baseline)
2. Hybrid GPU (CuPy without RAPIDS)
3. Full GPU (CuPy + RAPIDS cuML)

Usage:
    python scripts/benchmarks/profile_gpu_bottlenecks.py --points 1000000
    python scripts/benchmarks/profile_gpu_bottlenecks.py path/to/file.laz
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ign_lidar.features_gpu import (  # noqa: E402
    GPU_AVAILABLE,
    CUML_AVAILABLE,
    GPUFeatureComputer
)
from ign_lidar.features_gpu_chunked import (  # noqa: E402
    GPUChunkedFeatureComputer
)
from ign_lidar.features import (  # noqa: E402
    compute_all_features_optimized
)


class PerformanceProfiler:
    """Profile GPU computation bottlenecks."""
    
    def __init__(self):
        self.timings = {}
    
    def profile_cpu_baseline(
        self,
        points: np.ndarray,
        classification: np.ndarray,
        k: int
    ) -> Dict[str, float]:
        """Profile CPU-only computation."""
        print("\n" + "="*80)
        print("🖥️  PROFILING CPU BASELINE")
        print("="*80)
        
        timings = {}
        
        # Total time
        start_total = time.time()
        
        _ = compute_all_features_optimized(
            points=points,
            classification=classification,
            k=k,
            auto_k=False
        )
        
        timings['total'] = time.time() - start_total
        
        print("\n✓ CPU computation complete")
        print(f"  Total time: {timings['total']:.2f}s")
        
        return timings
    
    def profile_gpu_without_cuml(
        self,
        points: np.ndarray,
        classification: np.ndarray,
        k: int
    ) -> Dict[str, float]:
        """Profile GPU computation WITHOUT RAPIDS cuML (hybrid mode)."""
        print("\n" + "="*80)
        print("⚡ PROFILING HYBRID GPU (CuPy only, no RAPIDS)")
        print("="*80)
        
        if not GPU_AVAILABLE:
            print("⚠️  GPU not available, skipping")
            return None
        
        # Temporarily disable cuML to test hybrid mode
        import ign_lidar.features_gpu as gpu_module
        original_cuml = gpu_module.CUML_AVAILABLE
        gpu_module.CUML_AVAILABLE = False
        
        timings = {}
        
        try:
            computer = GPUFeatureComputer(use_gpu=True)
            
            # Profile normals computation
            print("\n📊 Profiling normals computation...")
            start = time.time()
            normals = computer.compute_normals(points, k=k)
            timings['normals'] = time.time() - start
            print(f"  Normals: {timings['normals']:.2f}s")
            
            # Profile curvature
            print("📊 Profiling curvature computation...")
            start = time.time()
            _ = computer.compute_curvature(points, normals, k=k)
            timings['curvature'] = time.time() - start
            print(f"  Curvature: {timings['curvature']:.2f}s")
            
            # Profile height
            print("📊 Profiling height computation...")
            start = time.time()
            _ = computer.compute_height_above_ground(
                points, classification
            )
            timings['height'] = time.time() - start
            print(f"  Height: {timings['height']:.2f}s")
            
            # Profile geometric features
            print("📊 Profiling geometric features...")
            start = time.time()
            _ = computer.extract_geometric_features(points, normals, k=k)
            timings['geometric'] = time.time() - start
            print(f"  Geometric: {timings['geometric']:.2f}s")
            
            timings['total'] = sum(timings.values())
            
            print("\n✓ Hybrid GPU computation complete")
            print(f"  Total time: {timings['total']:.2f}s")
            
        finally:
            # Restore cuML availability
            gpu_module.CUML_AVAILABLE = original_cuml
        
        return timings
    
    def profile_gpu_with_cuml(
        self,
        points: np.ndarray,
        classification: np.ndarray,
        k: int
    ) -> Dict[str, float]:
        """Profile GPU computation WITH RAPIDS cuML (full GPU mode)."""
        print("\n" + "="*80)
        print("🚀 PROFILING FULL GPU (CuPy + RAPIDS cuML)")
        print("="*80)
        
        if not GPU_AVAILABLE:
            print("⚠️  GPU not available, skipping")
            return None
        
        if not CUML_AVAILABLE:
            print("⚠️  RAPIDS cuML not available, skipping")
            return None
        
        timings = {}
        
        computer = GPUFeatureComputer(use_gpu=True)
        
        # Profile normals computation
        print("\n📊 Profiling normals computation...")
        start = time.time()
        normals = computer.compute_normals(points, k=k)
        timings['normals'] = time.time() - start
        print(f"  Normals: {timings['normals']:.2f}s")
        
        # Profile curvature
        print("📊 Profiling curvature computation...")
        start = time.time()
        _ = computer.compute_curvature(points, normals, k=k)
        timings['curvature'] = time.time() - start
        print(f"  Curvature: {timings['curvature']:.2f}s")
        
        # Profile height
        print("📊 Profiling height computation...")
        start = time.time()
        _ = computer.compute_height_above_ground(points, classification)
        timings['height'] = time.time() - start
        print(f"  Height: {timings['height']:.2f}s")
        
        # Profile geometric features
        print("📊 Profiling geometric features...")
        start = time.time()
        _ = computer.extract_geometric_features(points, normals, k=k)
        timings['geometric'] = time.time() - start
        print(f"  Geometric: {timings['geometric']:.2f}s")
        
        timings['total'] = sum(timings.values())
        
        print("\n✓ Full GPU computation complete")
        print(f"  Total time: {timings['total']:.2f}s")
        
        return timings
    
    def profile_chunked_without_cuml(
        self,
        points: np.ndarray,
        classification: np.ndarray,
        k: int,
        chunk_size: int = 5_000_000
    ) -> Dict[str, float]:
        """Profile chunked GPU computation WITHOUT RAPIDS (per-chunk strategy)."""
        print("\n" + "="*80)
        print("⚡ PROFILING CHUNKED GPU (Per-chunk KDTree, no RAPIDS)")
        print("="*80)
        
        if not GPU_AVAILABLE:
            print("⚠️  GPU not available, skipping")
            return None
        
        # Temporarily disable cuML
        import ign_lidar.features_gpu_chunked as chunked_module
        original_cuml = chunked_module.CUML_AVAILABLE
        chunked_module.CUML_AVAILABLE = False
        
        timings = {}
        
        try:
            computer = GPUChunkedFeatureComputer(
                chunk_size=chunk_size,
                use_gpu=True,
                show_progress=True
            )
            
            # Profile normals with chunking
            print("\n📊 Profiling chunked normals computation...")
            start = time.time()
            normals = computer.compute_normals_chunked(points, k=k)
            timings['normals'] = time.time() - start
            print(f"  Normals (chunked): {timings['normals']:.2f}s")
            
            # Profile curvature with chunking
            print("📊 Profiling chunked curvature computation...")
            start = time.time()
            _ = computer.compute_curvature_chunked(points, normals, k=k)
            timings['curvature'] = time.time() - start
            print(f"  Curvature (chunked): {timings['curvature']:.2f}s")
            
            timings['total'] = sum(timings.values())
            
            print("\n✓ Chunked GPU computation complete")
            print(f"  Total time: {timings['total']:.2f}s")
            
        finally:
            # Restore cuML availability
            chunked_module.CUML_AVAILABLE = original_cuml
        
        return timings


def generate_synthetic_pointcloud(num_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic point cloud."""
    points = np.random.rand(num_points, 3).astype(np.float32)
    points[:, 0] *= 100  # X: 0-100m
    points[:, 1] *= 100  # Y: 0-100m
    points[:, 2] *= 20   # Z: 0-20m
    
    classification = np.random.choice([2, 6, 5], size=num_points).astype(np.uint8)
    
    return points, classification


def print_comparison(
    cpu_timings: Dict[str, float],
    hybrid_timings: Dict[str, float],
    full_gpu_timings: Dict[str, float],
    chunked_timings: Dict[str, float]
):
    """Print comparative analysis."""
    print("\n" + "="*80)
    print("📊 PERFORMANCE COMPARISON")
    print("="*80)
    
    print(f"\n{'Mode':<30} {'Total Time':<15} {'Speedup':<10} {'Status'}")
    print("-" * 80)
    
    # CPU baseline
    cpu_time = cpu_timings['total']
    print(f"{'CPU-Only (Baseline)':<30} {cpu_time:>10.2f}s     {1.0:>6.2f}x     ✓")
    
    # Hybrid GPU
    if hybrid_timings:
        hybrid_time = hybrid_timings['total']
        speedup = cpu_time / hybrid_time
        msg = f"{'Hybrid GPU (CuPy only)':<30} "
        msg += f"{hybrid_time:>10.2f}s     {speedup:>6.2f}x     ✓"
        print(msg)
    else:
        print(f"{'Hybrid GPU (CuPy only)':<30} {'N/A':>10}     {'N/A':>6}     ✗")
    
    # Chunked without cuML
    if chunked_timings:
        chunked_time = chunked_timings['total']
        speedup = cpu_time / chunked_time
        msg = f"{'Chunked GPU (no RAPIDS)':<30} "
        msg += f"{chunked_time:>10.2f}s     {speedup:>6.2f}x     ✓"
        print(msg)
    else:
        print(f"{'Chunked GPU (no RAPIDS)':<30} {'N/A':>10}     {'N/A':>6}     ✗")
    
    # Full GPU
    if full_gpu_timings:
        gpu_time = full_gpu_timings['total']
        speedup = cpu_time / gpu_time
        msg = f"{'Full GPU (RAPIDS cuML)':<30} "
        msg += f"{gpu_time:>10.2f}s     {speedup:>6.2f}x     ✓"
        print(msg)
    else:
        print(f"{'Full GPU (RAPIDS cuML)':<30} {'N/A':>10}     {'N/A':>6}     ✗")
    
    print("-" * 80)
    
    # Detailed breakdown
    print("\n📋 DETAILED BREAKDOWN")
    print("="*80)
    
    operations = ['normals', 'curvature', 'height', 'geometric']
    
    for op in operations:
        print(f"\n{op.capitalize()} Computation:")
        
        if op in cpu_timings:
            print(f"  CPU:           {cpu_timings[op]:>8.2f}s  (1.00x)")
        
        if hybrid_timings and op in hybrid_timings:
            cpu_op = cpu_timings.get(op, 0)
            speedup = cpu_op / hybrid_timings[op] if cpu_op > 0 else 0
            t = hybrid_timings[op]
            print(f"  Hybrid GPU:    {t:>8.2f}s  ({speedup:.2f}x)")
        
        if chunked_timings and op in chunked_timings:
            cpu_op = cpu_timings.get(op, 0)
            speedup = cpu_op / chunked_timings[op] if cpu_op > 0 else 0
            t = chunked_timings[op]
            print(f"  Chunked GPU:   {t:>8.2f}s  ({speedup:.2f}x)")
        
        if full_gpu_timings and op in full_gpu_timings:
            cpu_op = cpu_timings.get(op, 0)
            speedup = cpu_op / full_gpu_timings[op] if cpu_op > 0 else 0
            t = full_gpu_timings[op]
            print(f"  Full GPU:      {t:>8.2f}s  ({speedup:.2f}x)")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Profile GPU computation bottlenecks"
    )
    parser.add_argument(
        'input_file',
        nargs='?',
        help='LAZ file to benchmark (optional, uses synthetic if not provided)'
    )
    parser.add_argument(
        '--points',
        type=int,
        default=1_000_000,
        help='Number of synthetic points to generate (default: 1M)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='Number of neighbors for KNN (default: 10)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=5_000_000,
        help='Chunk size for chunked processing (default: 5M)'
    )
    
    args = parser.parse_args()
    
    # Print system info
    print("="*80)
    print("🔧 SYSTEM INFORMATION")
    print("="*80)
    print(f"GPU Available:     {GPU_AVAILABLE}")
    print(f"RAPIDS cuML:       {CUML_AVAILABLE}")
    
    if GPU_AVAILABLE:
        try:
            import cupy as cp
            print(f"CuPy version:      {cp.__version__}")
            device = cp.cuda.Device()
            props = cp.cuda.runtime.getDeviceProperties(device.id)
            print(f"GPU Device:        {props['name'].decode()}")
            print(f"GPU Memory:        {props['totalGlobalMem'] / 1e9:.1f} GB")
        except Exception as e:
            print(f"⚠️  Could not get GPU info: {e}")
    
    print("="*80)
    
    # Load or generate data
    if args.input_file:
        print(f"\n📂 Loading data from {args.input_file}...")
        import laspy
        las = laspy.read(args.input_file)
        points = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
        classification = las.classification.array.astype(np.uint8)
    else:
        print(f"\n🔬 Generating {args.points:,} synthetic points...")
        points, classification = generate_synthetic_pointcloud(args.points)
    
    print(f"Dataset: {len(points):,} points")
    print(f"K-neighbors: {args.k}")
    
    # Initialize profiler
    profiler = PerformanceProfiler()
    
    # Profile all modes
    cpu_timings = profiler.profile_cpu_baseline(points, classification, args.k)
    
    hybrid_timings = profiler.profile_gpu_without_cuml(
        points, classification, args.k
    )
    
    chunked_timings = profiler.profile_chunked_without_cuml(
        points, classification, args.k, chunk_size=args.chunk_size
    )
    
    full_gpu_timings = profiler.profile_gpu_with_cuml(
        points, classification, args.k
    )
    
    # Print comparison
    print_comparison(
        cpu_timings,
        hybrid_timings,
        full_gpu_timings,
        chunked_timings
    )
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("="*80)
    
    if not GPU_AVAILABLE:
        print("📌 No GPU detected:")
        print("   1. Check NVIDIA GPU: nvidia-smi")
        print("   2. Install CuPy: pip install cupy-cuda12x")
        print("   Expected speedup: 6-8x with GPU")
    elif not CUML_AVAILABLE:
        print("📌 CuPy available but RAPIDS cuML missing:")
        print("   Current performance: 6-8x speedup (hybrid mode)")
        print("   To achieve 12-20x speedup:")
        print("   conda install -c rapidsai -c conda-forge cuml=24.10")
    else:
        print("✓ Full GPU acceleration available!")
        print("  Using optimal configuration with 12-20x speedup")
    
    print("="*80)


if __name__ == '__main__':
    main()
