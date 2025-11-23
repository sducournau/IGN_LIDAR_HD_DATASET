#!/usr/bin/env python3
"""
Benchmark Phase 3 GPU Optimizations

Comprehensive benchmark to measure actual performance gains from Phase 3:
- GPU transfer reduction (4-6 → ≤2 transfers per tile)
- Performance improvement (+20-30% target)
- Cache effectiveness
- Memory efficiency

Author: IGN LiDAR HD Optimization Team
Date: November 23, 2025
"""

import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ign_lidar.features import FeatureOrchestrator
    from ign_lidar.optimization import GPUTransferProfiler
    from ign_lidar.core.gpu import GPUManager
    from ign_lidar.optimization.gpu_memory import GPUArrayCache
    from omegaconf import OmegaConf
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Make sure you're in the ign_gpu environment")
    sys.exit(1)


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subheader(title: str):
    """Print formatted subsection header."""
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80)


def generate_realistic_data(n_points: int = 1_000_000) -> Dict:
    """Generate realistic point cloud data for benchmarking."""
    np.random.seed(42)  # Reproducible
    
    # Realistic building point cloud (50x50m tile)
    x = np.random.uniform(0, 50, n_points)
    y = np.random.uniform(0, 50, n_points)
    
    # Multiple height levels (ground + buildings)
    ground_points = int(n_points * 0.3)
    building_points = n_points - ground_points
    
    z_ground = np.random.normal(10, 0.5, ground_points)
    z_building = np.random.uniform(15, 30, building_points)
    z = np.concatenate([z_ground, z_building])
    
    points = np.column_stack([x, y, z]).astype(np.float32)
    
    # Realistic attributes
    classification = np.ones(n_points, dtype=np.uint8)
    classification[:ground_points] = 2  # Ground
    classification[ground_points:] = 6  # Building
    
    intensity = np.random.randint(0, 65536, n_points, dtype=np.uint16)
    return_number = np.ones(n_points, dtype=np.uint8)
    number_of_returns = np.ones(n_points, dtype=np.uint8)
    
    return {
        'points': points,
        'classification': classification,
        'intensity': intensity,
        'return_number': return_number,
        'number_of_returns': number_of_returns,
        'laz_file': None,
        'bbox': (0, 0, 50, 50),
    }


def benchmark_gpu_transfers(n_points: int = 1_000_000) -> Dict:
    """
    Benchmark GPU transfer count.
    
    Target: ≤2 transfers per tile (Phase 3 optimization goal)
    """
    print_subheader(f"GPU Transfer Count ({n_points:,} points)")
    
    gpu_manager = GPUManager()
    if not gpu_manager.gpu_available:
        print("⚠️  GPU not available - skipping")
        return {'error': 'GPU not available'}
    
    tile_data = generate_realistic_data(n_points)
    config = OmegaConf.create({
        'features': {'mode': 'lod2', 'k_neighbors': 20, 'search_radius': 3.0},
        'data_sources': {'use_gpu': True, 'rgb': {'enabled': False}, 'nir': {'enabled': False}},
        'processor': {'use_gpu': True, 'num_workers': 0},
    })
    
    profiler = GPUTransferProfiler(track_stacks=False)
    orchestrator = FeatureOrchestrator(config)
    
    print("Running feature computation with transfer profiling...")
    with profiler:
        start = time.time()
        features = orchestrator.compute_features(tile_data)
        elapsed = time.time() - start
    
    total_transfers = len(profiler.events)
    cpu_to_gpu = len([e for e in profiler.events if e['direction'] == 'CPU→GPU'])
    gpu_to_cpu = len([e for e in profiler.events if e['direction'] == 'GPU→CPU'])
    
    print(f"\n✓ Computed {len(features)} features in {elapsed:.2f}s")
    print(f"  Total transfers: {total_transfers}")
    print(f"  CPU→GPU: {cpu_to_gpu}")
    print(f"  GPU→CPU: {gpu_to_cpu}")
    
    if total_transfers <= 2:
        print(f"  ✅ EXCELLENT: {total_transfers} transfers (target: ≤2)")
        status = "excellent"
    elif total_transfers <= 4:
        print(f"  ✓ GOOD: {total_transfers} transfers (target: ≤2)")
        status = "good"
    else:
        print(f"  ⚠️  NEEDS WORK: {total_transfers} transfers (target: ≤2)")
        status = "needs_optimization"
    
    return {
        'n_points': n_points,
        'total_transfers': total_transfers,
        'cpu_to_gpu': cpu_to_gpu,
        'gpu_to_cpu': gpu_to_cpu,
        'time_seconds': elapsed,
        'status': status,
        'target_met': total_transfers <= 2,
    }


def benchmark_cpu_vs_gpu(n_points: int = 500_000) -> Dict:
    """
    Benchmark CPU vs GPU performance.
    
    Target: GPU should be 5-10x faster than CPU
    """
    print_subheader(f"CPU vs GPU Performance ({n_points:,} points)")
    
    tile_data = generate_realistic_data(n_points)
    
    # CPU benchmark
    print("\n🖥️  CPU Processing...")
    config_cpu = OmegaConf.create({
        'features': {'mode': 'lod2', 'k_neighbors': 20, 'search_radius': 3.0},
        'data_sources': {'use_gpu': False, 'rgb': {'enabled': False}, 'nir': {'enabled': False}},
        'processor': {'use_gpu': False, 'num_workers': 0},
    })
    orchestrator_cpu = FeatureOrchestrator(config_cpu)
    
    start = time.time()
    features_cpu = orchestrator_cpu.compute_features(tile_data)
    time_cpu = time.time() - start
    print(f"   Time: {time_cpu:.2f}s")
    
    # GPU benchmark
    gpu_manager = GPUManager()
    if not gpu_manager.gpu_available:
        print("⚠️  GPU not available - skipping GPU benchmark")
        return {'error': 'GPU not available', 'cpu_time': time_cpu}
    
    print("\n🚀 GPU Processing...")
    config_gpu = OmegaConf.create({
        'features': {'mode': 'lod2', 'k_neighbors': 20, 'search_radius': 3.0},
        'data_sources': {'use_gpu': True, 'rgb': {'enabled': False}, 'nir': {'enabled': False}},
        'processor': {'use_gpu': True, 'num_workers': 0},
    })
    orchestrator_gpu = FeatureOrchestrator(config_gpu)
    
    start = time.time()
    features_gpu = orchestrator_gpu.compute_features(tile_data)
    time_gpu = time.time() - start
    print(f"   Time: {time_gpu:.2f}s")
    
    # Results
    speedup = time_cpu / time_gpu if time_gpu > 0 else 0
    
    print(f"\n📊 Results:")
    print(f"   CPU Time:  {time_cpu:.2f}s")
    print(f"   GPU Time:  {time_gpu:.2f}s")
    print(f"   Speedup:   {speedup:.1f}x")
    
    if speedup >= 8:
        print(f"   ✅ EXCELLENT: {speedup:.1f}x speedup!")
        status = "excellent"
    elif speedup >= 5:
        print(f"   ✓ GOOD: {speedup:.1f}x speedup")
        status = "good"
    elif speedup >= 2:
        print(f"   ⚠️  ACCEPTABLE: {speedup:.1f}x speedup")
        status = "acceptable"
    else:
        print(f"   ❌ POOR: {speedup:.1f}x speedup (expected 5-10x)")
        status = "poor"
    
    return {
        'n_points': n_points,
        'cpu_time': time_cpu,
        'gpu_time': time_gpu,
        'speedup': speedup,
        'status': status,
    }


def benchmark_cache_effectiveness() -> Dict:
    """
    Benchmark GPU cache effectiveness.
    
    Target: Cache hits should be 2-3x faster than initial uploads
    """
    print_subheader("GPU Cache Effectiveness")
    
    gpu_manager = GPUManager()
    if not gpu_manager.gpu_available:
        print("⚠️  GPU not available - skipping")
        return {'error': 'GPU not available'}
    
    try:
        import cupy as cp
        from ign_lidar.optimization.gpu_memory import GPUArrayCache
    except ImportError:
        print("⚠️  CuPy not available")
        return {'error': 'CuPy not available'}
    
    cache = GPUArrayCache(max_size_gb=2.0)
    test_data = np.random.randn(1_000_000, 3).astype(np.float32)
    
    print("\n🔄 Testing cache performance...")
    
    # Cold run (no cache)
    start = time.time()
    arr1 = cache.get_or_upload('test_array', test_data)
    time_cold = time.time() - start
    print(f"   Cold run (upload):  {time_cold*1000:.2f}ms")
    
    # Warm run (cache hit)
    start = time.time()
    arr2 = cache.get_or_upload('test_array', test_data)
    time_warm = time.time() - start
    print(f"   Warm run (cached):  {time_warm*1000:.2f}ms")
    
    speedup = time_cold / time_warm if time_warm > 0 else 0
    print(f"   Cache speedup:      {speedup:.1f}x")
    
    if speedup >= 5:
        print(f"   ✅ EXCELLENT: {speedup:.1f}x faster!")
        status = "excellent"
    elif speedup >= 2:
        print(f"   ✓ GOOD: {speedup:.1f}x faster")
        status = "good"
    else:
        print(f"   ⚠️  ACCEPTABLE: {speedup:.1f}x faster")
        status = "acceptable"
    
    return {
        'cold_time_ms': time_cold * 1000,
        'warm_time_ms': time_warm * 1000,
        'speedup': speedup,
        'status': status,
    }


def benchmark_multi_scale(sizes: List[int] = [100_000, 500_000, 1_000_000, 2_000_000]) -> Dict:
    """
    Benchmark performance across multiple dataset sizes.
    """
    print_subheader("Multi-Scale Performance")
    
    results = []
    
    for n_points in sizes:
        print(f"\n📊 Testing {n_points:,} points...")
        tile_data = generate_realistic_data(n_points)
        
        config = OmegaConf.create({
            'features': {'mode': 'lod2', 'k_neighbors': 20, 'search_radius': 3.0},
            'data_sources': {'use_gpu': True, 'rgb': {'enabled': False}, 'nir': {'enabled': False}},
            'processor': {'use_gpu': True, 'num_workers': 0},
        })
        orchestrator = FeatureOrchestrator(config)
        
        start = time.time()
        features = orchestrator.compute_features(tile_data)
        elapsed = time.time() - start
        
        points_per_sec = n_points / elapsed
        print(f"   Time: {elapsed:.2f}s ({points_per_sec:,.0f} points/sec)")
        
        results.append({
            'n_points': n_points,
            'time_seconds': elapsed,
            'points_per_second': points_per_sec,
        })
    
    return {'results': results}


def main():
    """Run all benchmarks."""
    print_header("PHASE 3 GPU OPTIMIZATION BENCHMARKS")
    
    print("\n🎯 Objectives:")
    print("   - GPU transfers: ≤2 per tile (67% reduction)")
    print("   - GPU speedup: 5-10x vs CPU")
    print("   - Cache speedup: 2-3x on cache hits")
    print("   - Overall gain: +20-30% from Phase 3")
    
    results = {}
    
    # Test 1: GPU Transfer Count
    print_header("TEST 1: GPU Transfer Count")
    results['gpu_transfers'] = benchmark_gpu_transfers(n_points=1_000_000)
    
    # Test 2: CPU vs GPU Performance
    print_header("TEST 2: CPU vs GPU Performance")
    results['cpu_vs_gpu'] = benchmark_cpu_vs_gpu(n_points=500_000)
    
    # Test 3: Cache Effectiveness
    print_header("TEST 3: GPU Cache Effectiveness")
    results['cache'] = benchmark_cache_effectiveness()
    
    # Test 4: Multi-Scale Performance
    print_header("TEST 4: Multi-Scale Performance")
    results['multi_scale'] = benchmark_multi_scale([100_000, 500_000, 1_000_000, 2_000_000])
    
    # Summary
    print_header("BENCHMARK SUMMARY")
    
    print("\n📊 Results Overview:")
    
    if 'gpu_transfers' in results and 'total_transfers' in results['gpu_transfers']:
        transfers = results['gpu_transfers']
        print(f"\n   GPU Transfers: {transfers['total_transfers']} (target: ≤2)")
        print(f"   - Status: {transfers['status']}")
        print(f"   - Target met: {'✅ Yes' if transfers['target_met'] else '❌ No'}")
    
    if 'cpu_vs_gpu' in results and 'speedup' in results['cpu_vs_gpu']:
        perf = results['cpu_vs_gpu']
        print(f"\n   GPU Speedup: {perf['speedup']:.1f}x (target: 5-10x)")
        print(f"   - Status: {perf['status']}")
    
    if 'cache' in results and 'speedup' in results['cache']:
        cache = results['cache']
        print(f"\n   Cache Speedup: {cache['speedup']:.1f}x (target: 2-3x)")
        print(f"   - Status: {cache['status']}")
    
    # Save results
    output_file = Path(__file__).parent / 'benchmark_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("✅ Benchmark complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
