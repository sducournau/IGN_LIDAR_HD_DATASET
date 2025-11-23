#!/usr/bin/env python3
"""
Validation Script for Phase 3 GPU Optimizations

This script validates the GPU transfer optimizations implemented in Phase 3:
- Measures actual CPU↔GPU transfers using GPUTransferProfiler
- Benchmarks performance improvements
- Verifies cache effectiveness

Author: IGN LiDAR HD Optimization Team
Date: November 23, 2025
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ign_lidar.features import FeatureOrchestrator
    from ign_lidar.optimization import GPUTransferProfiler
    from ign_lidar.core.gpu import GPUManager
    from omegaconf import OmegaConf
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're in the correct environment")
    sys.exit(1)


def create_test_config(use_gpu: bool = True):
    """Create test configuration."""
    config = OmegaConf.create({
        'features': {
            'mode': 'lod2',
            'k_neighbors': 20,
            'search_radius': 3.0,
        },
        'data_sources': {
            'use_gpu': use_gpu,
            'rgb': {'enabled': False},
            'nir': {'enabled': False},
        },
        'processor': {
            'use_gpu': use_gpu,
            'num_workers': 0,  # Disable multiprocessing for cleaner profiling
        }
    })
    return config


def generate_test_data(n_points: int = 1_000_000):
    """Generate synthetic point cloud for testing."""
    print(f"Generating test data ({n_points:,} points)...")
    
    # Random 3D points
    points = np.random.randn(n_points, 3).astype(np.float32)
    points[:, 2] = np.abs(points[:, 2]) * 10  # Positive Z
    
    # Classification (simplified)
    classification = np.ones(n_points, dtype=np.uint8)
    
    # Intensity (required by FeatureOrchestrator)
    intensity = np.random.randint(0, 65536, n_points, dtype=np.uint16)
    
    # Return code (optional but good to have)
    return_number = np.ones(n_points, dtype=np.uint8)
    number_of_returns = np.ones(n_points, dtype=np.uint8)
    
    tile_data = {
        'points': points,
        'classification': classification,
        'intensity': intensity,
        'return_number': return_number,
        'number_of_returns': number_of_returns,
        'laz_file': None,
        'bbox': None,
    }
    
    return tile_data


def test_gpu_transfers(n_points: int = 1_000_000):
    """Test GPU transfer count and performance."""
    print("\n" + "=" * 70)
    print(f"GPU TRANSFER VALIDATION TEST ({n_points:,} points)")
    print("=" * 70)
    
    # Check GPU availability
    gpu_manager = GPUManager()
    if not gpu_manager.gpu_available:
        print("⚠️  GPU not available - skipping GPU tests")
        return
    
    print(f"✓ GPU detected: {gpu_manager.get_info()['gpu_available']}")
    
    # Create test data
    tile_data = generate_test_data(n_points)
    config = create_test_config(use_gpu=True)
    
    # Test with profiler
    print("\n📊 Running feature computation with transfer profiling...")
    profiler = GPUTransferProfiler(track_stacks=False)
    
    orchestrator = FeatureOrchestrator(config)
    
    with profiler:
        start_time = time.time()
        features = orchestrator.compute_features(tile_data)
        elapsed = time.time() - start_time
    
    # Print results
    print(f"\n✓ Features computed in {elapsed:.2f}s")
    print(f"  Features: {list(features.keys())}")
    
    # Analyze transfers
    print("\n" + "-" * 70)
    print("GPU TRANSFER ANALYSIS")
    print("-" * 70)
    profiler.print_report()
    
    # Check if target met
    total_transfers = len(profiler.events)
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    
    if total_transfers <= 2:
        print(f"✅ PASS: {total_transfers} transfers (target: ≤2)")
        print("   GPU pipeline optimized!")
    elif total_transfers <= 4:
        print(f"⚠️  ACCEPTABLE: {total_transfers} transfers (target: ≤2)")
        print("   Some optimization possible")
    else:
        print(f"❌ NEEDS OPTIMIZATION: {total_transfers} transfers (target: ≤2)")
        print("   Too many CPU↔GPU transfers detected")
    
    return total_transfers, elapsed


def benchmark_performance():
    """Benchmark CPU vs GPU performance."""
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARK: CPU vs GPU")
    print("=" * 70)
    
    n_points = 500_000
    tile_data = generate_test_data(n_points)
    
    # CPU benchmark
    print(f"\n🖥️  CPU Processing ({n_points:,} points)...")
    config_cpu = create_test_config(use_gpu=False)
    orchestrator_cpu = FeatureOrchestrator(config_cpu)
    
    start = time.time()
    features_cpu = orchestrator_cpu.compute_features(tile_data)
    time_cpu = time.time() - start
    
    print(f"   Time: {time_cpu:.2f}s")
    
    # GPU benchmark
    gpu_manager = GPUManager()
    if not gpu_manager.gpu_available:
        print("⚠️  GPU not available - skipping GPU benchmark")
        return
    
    print(f"\n🚀 GPU Processing ({n_points:,} points)...")
    config_gpu = create_test_config(use_gpu=True)
    orchestrator_gpu = FeatureOrchestrator(config_gpu)
    
    start = time.time()
    features_gpu = orchestrator_gpu.compute_features(tile_data)
    time_gpu = time.time() - start
    
    print(f"   Time: {time_gpu:.2f}s")
    
    # Results
    speedup = time_cpu / time_gpu if time_gpu > 0 else 0
    
    print("\n" + "-" * 70)
    print("PERFORMANCE RESULTS")
    print("-" * 70)
    print(f"CPU Time:  {time_cpu:.2f}s")
    print(f"GPU Time:  {time_gpu:.2f}s")
    print(f"Speedup:   {speedup:.1f}x")
    
    if speedup >= 5:
        print("✅ Excellent GPU performance!")
    elif speedup >= 2:
        print("✓ Good GPU performance")
    else:
        print("⚠️  GPU performance lower than expected")
    
    return speedup


def test_cache_effectiveness():
    """Test GPU cache effectiveness."""
    print("\n" + "=" * 70)
    print("GPU CACHE EFFECTIVENESS TEST")
    print("=" * 70)
    
    gpu_manager = GPUManager()
    if not gpu_manager.gpu_available:
        print("⚠️  GPU not available - skipping cache test")
        return
    
    # Import cache
    try:
        from ign_lidar.optimization.gpu_memory import GPUArrayCache
    except ImportError:
        print("❌ GPUArrayCache not available")
        return
    
    print("Testing cache operations...")
    
    cache = GPUArrayCache(max_size_gb=2.0)
    
    # Test 1: Upload array
    test_array = np.random.randn(100000, 3).astype(np.float32)
    
    start = time.time()
    gpu_arr1 = cache.get_or_upload('test_key', test_array)
    time_first = time.time() - start
    
    # Test 2: Get cached array (should be faster)
    start = time.time()
    gpu_arr2 = cache.get_or_upload('test_key', test_array)
    time_cached = time.time() - start
    
    print(f"\n✓ First upload:  {time_first*1000:.2f}ms")
    print(f"✓ Cached access: {time_cached*1000:.2f}ms")
    
    if time_cached < time_first * 0.5:
        print("✅ Cache working effectively!")
    else:
        print("⚠️  Cache may not be optimized")
    
    return time_first, time_cached


def main():
    """Run all validation tests."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "PHASE 3 GPU OPTIMIZATION VALIDATION" + " " * 18 + "║")
    print("╚" + "=" * 68 + "╝")
    
    try:
        # Test 1: GPU transfers
        test_gpu_transfers(n_points=1_000_000)
        
        # Test 2: Performance benchmark
        benchmark_performance()
        
        # Test 3: Cache effectiveness
        test_cache_effectiveness()
        
        print("\n" + "=" * 70)
        print("VALIDATION COMPLETE")
        print("=" * 70)
        print("\nAll tests completed successfully!")
        print("Check results above for optimization validation.")
        
    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
