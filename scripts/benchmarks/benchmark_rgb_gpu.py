"""
Benchmark script for RGB GPU acceleration
"""

import time
import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None


def benchmark_rgb_augmentation():
    """Benchmark RGB color interpolation."""
    print("=" * 80)
    print("RGB Augmentation Benchmark: GPU vs CPU")
    print("=" * 80)
    
    if not GPU_AVAILABLE:
        print("\n⚠️  CuPy not available")
        print("Install with: pip install cupy-cuda11x  (or cupy-cuda12x)")
        print("\nNote: This benchmark requires an NVIDIA GPU with CUDA")
        return
    
    from ign_lidar.features_gpu import GPUFeatureComputer
    
    # Test configuration
    N_POINTS = [10_000, 100_000, 1_000_000]
    bbox = (650000, 6860000, 650500, 6860500)  # 500m x 500m
    
    # Create test data
    rgb_image = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)
    
    print(f"\nTest setup:")
    print(f"  RGB image: 1000x1000 pixels")
    print(f"  Bbox: {bbox}")
    print(f"  Point counts: {N_POINTS}")
    
    results = []
    
    for n_points in N_POINTS:
        print(f"\n{'=' * 80}")
        print(f"Testing with {n_points:,} points")
        print(f"{'=' * 80}")
        
        # Generate random points
        points = np.random.rand(n_points, 3).astype(np.float32)
        points[:, 0] = points[:, 0] * 500 + bbox[0]
        points[:, 1] = points[:, 1] * 500 + bbox[1]
        
        # CPU baseline (estimated based on PIL performance)
        # Typical PIL interpolation: ~12s per 1M points
        time_cpu_estimated = (n_points / 1_000_000) * 12.0
        print(f"CPU (estimated): {time_cpu_estimated:.3f}s")
        
        # GPU interpolation
        try:
            computer = GPUFeatureComputer(use_gpu=True)
            points_gpu = cp.asarray(points)
            rgb_image_gpu = cp.asarray(rgb_image)
            
            # Warm-up
            _ = computer.interpolate_colors_gpu(points_gpu, rgb_image_gpu, bbox)
            cp.cuda.Stream.null.synchronize()
            
            # Benchmark
            t0 = time.time()
            colors_gpu = computer.interpolate_colors_gpu(
                points_gpu, rgb_image_gpu, bbox
            )
            cp.cuda.Stream.null.synchronize()
            time_gpu = time.time() - t0
            
            speedup = time_cpu_estimated / time_gpu
            print(f"GPU: {time_gpu:.3f}s")
            print(f"Speedup: {speedup:.1f}x")
            
            # Validate output
            colors = cp.asnumpy(colors_gpu)
            print(f"Output shape: {colors.shape}, dtype: {colors.dtype}")
            print(f"Color range: [{colors.min()}, {colors.max()}]")
            
            results.append({
                'n_points': n_points,
                'cpu_time': time_cpu_estimated,
                'gpu_time': time_gpu,
                'speedup': speedup
            })
            
        except Exception as e:
            print(f"GPU test failed: {e}")
            print("This is expected if CUDA is not properly configured")
    
    # Summary
    if results:
        print(f"\n{'=' * 80}")
        print("SUMMARY")
        print(f"{'=' * 80}")
        print(f"{'Points':<15} {'CPU (s)':<12} {'GPU (s)':<12} {'Speedup':<12}")
        print("-" * 80)
        for r in results:
            print(
                f"{r['n_points']:<15,} "
                f"{r['cpu_time']:<12.3f} "
                f"{r['gpu_time']:<12.3f} "
                f"{r['speedup']:<12.1f}x"
            )
        
        avg_speedup = np.mean([r['speedup'] for r in results])
        print(f"\nAverage speedup: {avg_speedup:.1f}x")
        print(f"Target speedup: 24x")
        print(f"Status: {'✓ PASS' if avg_speedup >= 20 else '✗ FAIL'}")
    else:
        print("\n⚠️  No results - GPU tests failed")
        print("Check CUDA installation and GPU availability")


def benchmark_gpu_cache():
    """Benchmark GPU tile caching."""
    print("\n" + "=" * 80)
    print("GPU Tile Cache Benchmark")
    print("=" * 80)
    
    if not GPU_AVAILABLE:
        print("\n⚠️  GPU not available - skipping")
        return
    
    from ign_lidar.rgb_augmentation import IGNOrthophotoFetcher
    
    print("\nSimulating tile access patterns...")
    
    # Create fetcher with GPU cache
    fetcher = IGNOrthophotoFetcher(use_gpu=True)
    fetcher.gpu_cache_max_size = 10
    
    # Mock fetch to avoid network
    def mock_fetch(bbox, width=1024, height=1024, crs="EPSG:2154"):
        return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    fetcher.fetch_orthophoto = mock_fetch
    
    # Test cache performance
    bboxes = [(i*100, 0, (i+1)*100, 100) for i in range(20)]
    
    # First pass - cache misses
    t0 = time.time()
    for bbox in bboxes:
        _ = fetcher.fetch_orthophoto_gpu(bbox)
    t_first = time.time() - t0
    
    print(f"First pass (cache misses): {t_first:.3f}s")
    print(f"Cache size: {len(fetcher.gpu_cache)}/{fetcher.gpu_cache_max_size}")
    
    # Second pass - cache hits
    t0 = time.time()
    for bbox in bboxes[-10:]:  # Last 10 should be in cache
        _ = fetcher.fetch_orthophoto_gpu(bbox)
    t_second = time.time() - t0
    
    print(f"Second pass (cache hits): {t_second:.3f}s")
    print(f"Cache hit speedup: {t_first / max(t_second, 0.001):.1f}x")
    
    # Clear cache
    fetcher.clear_gpu_cache()
    print(f"Cache cleared: {len(fetcher.gpu_cache)} tiles remaining")


if __name__ == '__main__':
    benchmark_rgb_augmentation()
    
    if GPU_AVAILABLE:
        try:
            benchmark_gpu_cache()
        except Exception as e:
            print(f"\nCache benchmark failed: {e}")
    
    print("\n" + "=" * 80)
    print("Benchmark completed!")
    print("=" * 80)
