"""
Tests for GPU-accelerated RGB augmentation (Phase 3.1)
"""

import pytest
import numpy as np
from pathlib import Path

# Try to import GPU modules
GPU_AVAILABLE = False
GPU_FUNCTIONAL = False
cp = None

try:
    import cupy as cp
    GPU_AVAILABLE = True
    # Test if GPU is actually functional (not just imported)
    try:
        test_array = cp.array([1, 2, 3])
        _ = test_array + 1
        GPU_FUNCTIONAL = True
    except Exception:
        GPU_AVAILABLE = False
        GPU_FUNCTIONAL = False
        cp = None
except ImportError:
    pass


@pytest.mark.skipif(not GPU_FUNCTIONAL, reason="GPU not functional")
def test_interpolate_colors_gpu_basic():
    """Test basic GPU color interpolation."""
    from ign_lidar.features_gpu import GPUFeatureComputer
    
    # Create test data
    computer = GPUFeatureComputer(use_gpu=True)
    
    # Simple 4x4 RGB image (gradient)
    rgb_image = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb_image[:, :, 0] = np.arange(4)[:, None] * 64  # R gradient vertical
    rgb_image[:, :, 1] = np.arange(4)[None, :] * 64  # G gradient horizontal
    rgb_image_gpu = cp.asarray(rgb_image)
    
    # Test points (in Lambert-93 coords)
    bbox = (0.0, 0.0, 100.0, 100.0)
    points = np.array([
        [50.0, 50.0, 0.0],  # Center
        [0.0, 0.0, 0.0],     # Bottom-left
        [100.0, 100.0, 0.0], # Top-right
    ], dtype=np.float32)
    points_gpu = cp.asarray(points)
    
    # Interpolate
    colors_gpu = computer.interpolate_colors_gpu(
        points_gpu, rgb_image_gpu, bbox
    )
    colors = cp.asnumpy(colors_gpu)
    
    # Validate shape
    assert colors.shape == (3, 3), f"Expected (3, 3), got {colors.shape}"
    assert colors.dtype == np.uint8, f"Expected uint8, got {colors.dtype}"
    
    # Validate center point (should be mid-range)
    assert 64 <= colors[0, 0] <= 192, f"R value out of range: {colors[0, 0]}"
    assert 64 <= colors[0, 1] <= 192, f"G value out of range: {colors[0, 1]}"
    
    print("✓ GPU color interpolation basic test passed")


@pytest.mark.skipif(not GPU_FUNCTIONAL, reason="GPU not functional")
def test_interpolate_colors_gpu_accuracy():
    """Test GPU color interpolation accuracy vs expected values."""
    from ign_lidar.features_gpu import GPUFeatureComputer
    
    computer = GPUFeatureComputer(use_gpu=True)
    
    # Create simple 2x2 checkerboard pattern
    rgb_image = np.array([
        [[255, 0, 0], [0, 255, 0]],
        [[0, 0, 255], [255, 255, 0]]
    ], dtype=np.uint8)
    rgb_image_gpu = cp.asarray(rgb_image)
    
    # Test exact corner points
    bbox = (0.0, 0.0, 10.0, 10.0)
    points = np.array([
        [0.0, 10.0, 0.0],   # Top-left (red)
        [10.0, 10.0, 0.0],  # Top-right (green)
        [0.0, 0.0, 0.0],    # Bottom-left (blue)
        [10.0, 0.0, 0.0],   # Bottom-right (yellow)
    ], dtype=np.float32)
    points_gpu = cp.asarray(points)
    
    colors_gpu = computer.interpolate_colors_gpu(
        points_gpu, rgb_image_gpu, bbox
    )
    colors = cp.asnumpy(colors_gpu)
    
    # Check corner colors (allow small interpolation error)
    tolerance = 5
    assert abs(colors[0, 0] - 255) <= tolerance  # Red R
    assert abs(colors[1, 1] - 255) <= tolerance  # Green G
    assert abs(colors[2, 2] - 255) <= tolerance  # Blue B
    assert abs(colors[3, 0] - 255) <= tolerance  # Yellow R
    assert abs(colors[3, 1] - 255) <= tolerance  # Yellow G
    
    print("✓ GPU color interpolation accuracy test passed")


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_interpolate_colors_gpu_performance():
    """Benchmark GPU vs CPU color interpolation."""
    import time
    from ign_lidar.features_gpu import GPUFeatureComputer
    
    # Create realistic test data
    computer = GPUFeatureComputer(use_gpu=True)
    
    # 1000x1000 RGB image
    rgb_image = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)
    rgb_image_gpu = cp.asarray(rgb_image)
    
    # 100K points (smaller for faster test)
    N = 100_000
    bbox = (0.0, 0.0, 100.0, 100.0)
    points = np.random.rand(N, 3).astype(np.float32) * 100
    points_gpu = cp.asarray(points)
    
    # Warm-up
    _ = computer.interpolate_colors_gpu(points_gpu, rgb_image_gpu, bbox)
    cp.cuda.Stream.null.synchronize()
    
    # Benchmark GPU
    t0 = time.time()
    colors_gpu = computer.interpolate_colors_gpu(
        points_gpu, rgb_image_gpu, bbox
    )
    cp.cuda.Stream.null.synchronize()
    t_gpu = time.time() - t0
    
    # Expected CPU time (estimated: ~1.2s for 100K points)
    t_cpu_estimated = (N / 1_000_000) * 12.0
    
    print(f"\nGPU interpolation: {t_gpu:.3f}s for {N:,} points")
    print(f"CPU (estimated): {t_cpu_estimated:.3f}s")
    print(f"Speedup: {t_cpu_estimated / t_gpu:.1f}x")
    
    # Should be significantly faster than CPU
    assert t_gpu < t_cpu_estimated * 0.5, \
        f"GPU not fast enough: {t_gpu:.3f}s vs CPU {t_cpu_estimated:.3f}s"
    
    print("✓ GPU color interpolation performance test passed")


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_gpu_tile_cache():
    """Test GPU tile caching mechanism."""
    from ign_lidar.rgb_augmentation import IGNOrthophotoFetcher
    
    # Create fetcher with GPU cache
    fetcher = IGNOrthophotoFetcher(use_gpu=True)
    fetcher.gpu_cache_max_size = 3  # Small cache for testing
    
    # Mock fetch_orthophoto to avoid network calls
    def mock_fetch(bbox, width=1024, height=1024, crs="EPSG:2154"):
        return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    fetcher.fetch_orthophoto = mock_fetch
    
    # Fetch 4 tiles (should evict first one)
    bboxes = [
        (0, 0, 100, 100),
        (100, 0, 200, 100),
        (0, 100, 100, 200),
        (100, 100, 200, 200),
    ]
    
    for bbox in bboxes:
        tile = fetcher.fetch_orthophoto_gpu(bbox)
        assert tile is not None
        assert isinstance(tile, cp.ndarray)
    
    # Cache should have 3 tiles (max size)
    assert len(fetcher.gpu_cache) == 3
    
    # First bbox should be evicted
    first_key = fetcher._get_cache_key(bboxes[0])
    assert first_key not in fetcher.gpu_cache
    
    # Last 3 should be in cache
    for bbox in bboxes[1:]:
        key = fetcher._get_cache_key(bbox)
        assert key in fetcher.gpu_cache
    
    # Fetch first bbox again (should be cache miss)
    tile = fetcher.fetch_orthophoto_gpu(bboxes[0])
    assert tile is not None
    
    # Now second bbox should be evicted
    second_key = fetcher._get_cache_key(bboxes[1])
    assert second_key not in fetcher.gpu_cache
    
    # Clear cache
    fetcher.clear_gpu_cache()
    assert len(fetcher.gpu_cache) == 0
    
    print("✓ GPU tile cache test passed")


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_gpu_fallback():
    """Test graceful fallback when GPU unavailable."""
    from ign_lidar.features_gpu import GPUFeatureComputer
    
    # Force CPU mode
    computer = GPUFeatureComputer(use_gpu=False)
    
    # Create test data on CPU
    rgb_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    points = np.random.rand(1000, 3).astype(np.float32) * 100
    bbox = (0.0, 0.0, 100.0, 100.0)
    
    # Should raise RuntimeError (fallback not implemented in GPU method)
    with pytest.raises(RuntimeError, match="GPU not available"):
        computer.interpolate_colors_gpu(
            points, rgb_image, bbox
        )
    
    print("✓ GPU fallback test passed")


def test_cpu_only_mode():
    """Test that RGB augmentation works without GPU."""
    from ign_lidar.rgb_augmentation import IGNOrthophotoFetcher
    
    # Create fetcher without GPU
    fetcher = IGNOrthophotoFetcher(use_gpu=False)
    
    # Should not have GPU capabilities
    assert not fetcher.use_gpu
    assert fetcher.gpu_cache is None
    
    print("✓ CPU-only mode test passed")


if __name__ == '__main__':
    # Run tests
    print("=" * 80)
    print("GPU RGB Augmentation Tests")
    print("=" * 80)
    
    if not GPU_AVAILABLE:
        print("\n⚠️  CuPy not installed - skipping GPU tests")
        print("Install with: pip install cupy-cuda11x (or cupy-cuda12x)")
        test_cpu_only_mode()
    elif not GPU_FUNCTIONAL:
        print("\n⚠️  GPU hardware not available or CUDA not configured")
        print("CuPy is installed but cannot access GPU")
        print("This is expected in environments without NVIDIA GPU")
        print("\nRunning CPU-only tests...\n")
        test_cpu_only_mode()
        test_gpu_fallback()
        print("\n✓ CPU-only tests passed")
        print("\nNote: GPU tests require:")
        print("  - NVIDIA GPU with CUDA support")
        print("  - CUDA Toolkit installed (libnvrtc.so)")
        print("  - Proper GPU drivers")
    else:
        print("\n✓ GPU available and functional - running all tests\n")
        test_interpolate_colors_gpu_basic()
        test_interpolate_colors_gpu_accuracy()
        test_interpolate_colors_gpu_performance()
        test_gpu_tile_cache()
        test_gpu_fallback()
        test_cpu_only_mode()
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
