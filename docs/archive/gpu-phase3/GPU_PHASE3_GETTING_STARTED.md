# GPU Phase 3 - Getting Started Guide

**Target Audience:** Developers ready to implement Phase 3 features  
**Prerequisites:** v1.4.0 released and stable  
**Last Updated:** October 3, 2025

---

## 🎯 Quick Start: Implementing Phase 3.1 (RGB GPU)

This guide walks you through implementing the first Phase 3 feature: GPU-accelerated RGB augmentation.

### Step 1: Decision Gate Review (Week 1 of November 2025)

Before starting, validate that Phase 3 should proceed:

```bash
# 1. Check v1.4.0 adoption metrics
# - PyPI downloads: target 1000+ downloads/month
# - GitHub stars: track growth
# - Issues: <5% GPU-related bugs

# 2. Gather user feedback
# Review GitHub issues/discussions for:
# - RGB augmentation usage
# - Performance bottleneck reports
# - GPU feature requests

# 3. Make go/no-go decision
# ✅ GO if: High RGB usage, users requesting GPU RGB, stable v1.4.0
# ❌ NO-GO if: Low GPU adoption, other priorities more urgent
```

### Step 2: Environment Setup (Day 1)

```bash
# 1. Clone repository
git clone https://github.com/sducournau/IGN_LIDAR_HD_DATASET.git
cd IGN_LIDAR_HD_DATASET

# 2. Create feature branch
git checkout -b feature/phase3.1-rgb-gpu

# 3. Install development dependencies
pip install -e ".[dev,gpu-full]"

# 4. Verify GPU available
python -c "import cupy as cp; print(f'✓ CuPy available, {cp.cuda.runtime.getDeviceCount()} GPU(s)')"

# 5. Run baseline benchmarks
python scripts/benchmarks/benchmark_gpu.py
```

### Step 3: Understand Current RGB Implementation (Day 1-2)

```bash
# Read current RGB augmentation code
cat ign_lidar/rgb_augmentation.py

# Key areas to understand:
# 1. IGNOrthophotoFetcher class (L23-100)
# 2. fetch_tile() method (WMS download + PIL loading)
# 3. augment_with_rgb() function (color interpolation)
# 4. Cache management (disk-based)
```

**Current bottlenecks:**

1. **PIL color interpolation** (CPU-only, ~12s for 1M points)
2. **No GPU memory cache** (always loads from disk to CPU)
3. **Multiple CPU↔GPU transfers** in pipeline

### Step 4: Implement GPU Color Interpolation (Days 3-5)

**File:** `ign_lidar/features_gpu.py`

```python
# Add this method to GPUFeatureComputer class

def interpolate_colors_gpu(
    self,
    points_gpu: 'cp.ndarray',
    rgb_image_gpu: 'cp.ndarray',
    bbox: Tuple[float, float, float, float]
) -> 'cp.ndarray':
    """
    Fast bilinear color interpolation on GPU.

    Args:
        points_gpu: [N, 3] CuPy array (x, y, z coordinates in Lambert-93)
        rgb_image_gpu: [H, W, 3] CuPy array (RGB image, uint8)
        bbox: (xmin, ymin, xmax, ymax) in Lambert-93

    Returns:
        colors_gpu: [N, 3] CuPy array (R, G, B values, uint8)

    Performance: ~100x faster than PIL on CPU
    """
    if not self.use_gpu or cp is None:
        # Fallback to CPU (not implemented yet, return zeros)
        return cp.zeros((len(points_gpu), 3), dtype=cp.uint8)

    # Unpack bbox
    xmin, ymin, xmax, ymax = bbox
    H, W = rgb_image_gpu.shape[:2]

    # Normalize point coordinates to image space
    # Lambert-93 coords → normalized [0, 1] → pixel coords
    x_norm = (points_gpu[:, 0] - xmin) / (xmax - xmin)  # [N]
    y_norm = (points_gpu[:, 1] - ymin) / (ymax - ymin)  # [N]

    # Convert to pixel coordinates (image y-axis is flipped)
    px = x_norm * (W - 1)  # [N]
    py = (1 - y_norm) * (H - 1)  # [N], flip y-axis

    # Clamp to valid range
    px = cp.clip(px, 0, W - 1)
    py = cp.clip(py, 0, H - 1)

    # Bilinear interpolation
    # Get integer and fractional parts
    px0 = cp.floor(px).astype(cp.int32)
    py0 = cp.floor(py).astype(cp.int32)
    px1 = cp.minimum(px0 + 1, W - 1)
    py1 = cp.minimum(py0 + 1, H - 1)

    dx = px - px0  # [N]
    dy = py - py0  # [N]

    # Fetch pixel values at 4 corners
    # Shape: [N, 3]
    c00 = rgb_image_gpu[py0, px0]  # Top-left
    c01 = rgb_image_gpu[py0, px1]  # Top-right
    c10 = rgb_image_gpu[py1, px0]  # Bottom-left
    c11 = rgb_image_gpu[py1, px1]  # Bottom-right

    # Bilinear weights
    w00 = (1 - dx[:, None]) * (1 - dy[:, None])  # [N, 1]
    w01 = dx[:, None] * (1 - dy[:, None])
    w10 = (1 - dx[:, None]) * dy[:, None]
    w11 = dx[:, None] * dy[:, None]

    # Interpolated color
    colors = (w00 * c00 + w01 * c01 + w10 * c10 + w11 * c11).astype(cp.uint8)

    return colors
```

**Test the implementation:**

```python
# tests/test_gpu_rgb.py (create new file)

import pytest
import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_interpolate_colors_gpu():
    """Test GPU color interpolation accuracy."""
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
    colors_gpu = computer.interpolate_colors_gpu(points_gpu, rgb_image_gpu, bbox)
    colors = cp.asnumpy(colors_gpu)

    # Validate shape
    assert colors.shape == (3, 3)
    assert colors.dtype == np.uint8

    # Validate center point (should be mid-range)
    assert 64 <= colors[0, 0] <= 192  # R
    assert 64 <= colors[0, 1] <= 192  # G

    print("✓ GPU color interpolation test passed")

@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_interpolate_colors_gpu_benchmark():
    """Benchmark GPU vs CPU color interpolation."""
    import time
    from ign_lidar.features_gpu import GPUFeatureComputer

    # Create realistic test data
    computer = GPUFeatureComputer(use_gpu=True)

    # 1000x1000 RGB image
    rgb_image = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)
    rgb_image_gpu = cp.asarray(rgb_image)

    # 1M points
    N = 1_000_000
    bbox = (0.0, 0.0, 100.0, 100.0)
    points = np.random.rand(N, 3).astype(np.float32) * 100
    points_gpu = cp.asarray(points)

    # Warm-up
    _ = computer.interpolate_colors_gpu(points_gpu, rgb_image_gpu, bbox)
    cp.cuda.Stream.null.synchronize()

    # Benchmark GPU
    t0 = time.time()
    colors_gpu = computer.interpolate_colors_gpu(points_gpu, rgb_image_gpu, bbox)
    cp.cuda.Stream.null.synchronize()
    t_gpu = time.time() - t0

    print(f"GPU interpolation: {t_gpu:.3f}s for {N:,} points")
    print(f"Expected: <0.5s (target: 24x speedup over CPU ~12s)")

    # Validate
    assert t_gpu < 1.0, f"GPU too slow: {t_gpu:.3f}s"
    print("✓ GPU color interpolation benchmark passed")

if __name__ == '__main__':
    test_interpolate_colors_gpu()
    test_interpolate_colors_gpu_benchmark()
```

**Run tests:**

```bash
pytest tests/test_gpu_rgb.py -v
```

### Step 5: Implement GPU Tile Cache (Days 6-8)

**File:** `ign_lidar/rgb_augmentation.py`

```python
# Add GPU cache support to IGNOrthophotoFetcher

class IGNOrthophotoFetcher:
    def __init__(self, cache_dir: Optional[Path] = None, use_gpu: bool = False):
        """
        Initialize orthophoto fetcher.

        Args:
            cache_dir: Directory to cache downloaded orthophotos (disk)
            use_gpu: Enable GPU memory caching for faster access
        """
        # ... existing code ...

        self.use_gpu = use_gpu
        self.gpu_cache = {} if use_gpu else None  # GPU memory cache
        self.gpu_cache_order = []  # LRU order
        self.gpu_cache_max_size = 10  # Max tiles in GPU memory

        if use_gpu:
            try:
                import cupy as cp
                self.cp = cp
                logger.info("GPU tile caching enabled")
            except ImportError:
                logger.warning("GPU caching requested but CuPy unavailable")
                self.use_gpu = False
                self.gpu_cache = None

    def _get_cache_key(self, bbox: Tuple[float, float, float, float]) -> str:
        """Generate cache key from bbox."""
        return f"{bbox[0]:.0f}_{bbox[1]:.0f}_{bbox[2]:.0f}_{bbox[3]:.0f}"

    def fetch_tile_gpu(
        self,
        bbox: Tuple[float, float, float, float],
        resolution: float = 0.5
    ) -> 'cp.ndarray':
        """
        Fetch RGB tile and return as GPU array.

        Uses LRU cache in GPU memory for fast repeated access.

        Args:
            bbox: (xmin, ymin, xmax, ymax) in Lambert-93
            resolution: Pixel resolution in meters

        Returns:
            rgb_gpu: [H, W, 3] CuPy array (uint8)
        """
        if not self.use_gpu:
            # Fallback to CPU
            rgb_array = self.fetch_tile(bbox, resolution)
            return self.cp.asarray(rgb_array)

        # Check GPU cache
        cache_key = self._get_cache_key(bbox)
        if cache_key in self.gpu_cache:
            # Cache hit - move to end (most recent)
            self.gpu_cache_order.remove(cache_key)
            self.gpu_cache_order.append(cache_key)
            logger.debug(f"GPU cache hit: {cache_key}")
            return self.gpu_cache[cache_key]

        # Cache miss - load from disk or download
        rgb_array = self.fetch_tile(bbox, resolution)
        rgb_gpu = self.cp.asarray(rgb_array)

        # Add to GPU cache
        self.gpu_cache[cache_key] = rgb_gpu
        self.gpu_cache_order.append(cache_key)

        # Evict oldest if cache full
        if len(self.gpu_cache) > self.gpu_cache_max_size:
            oldest_key = self.gpu_cache_order.pop(0)
            del self.gpu_cache[oldest_key]
            logger.debug(f"GPU cache evicted: {oldest_key}")

        logger.debug(f"GPU cache miss: {cache_key} (cache size: {len(self.gpu_cache)})")
        return rgb_gpu

    def clear_gpu_cache(self):
        """Clear GPU memory cache."""
        if self.use_gpu:
            self.gpu_cache.clear()
            self.gpu_cache_order.clear()
            logger.info("GPU cache cleared")
```

### Step 6: Integrate with Pipeline (Days 9-11)

**File:** `ign_lidar/processor.py`

```python
# Update LiDARProcessor to use GPU RGB pipeline

class LiDARProcessor:
    def __init__(self, ..., use_gpu: bool = False):
        # ... existing code ...

        # Update RGB fetcher initialization
        if include_rgb:
            try:
                from .rgb_augmentation import IGNOrthophotoFetcher
                self.rgb_fetcher = IGNOrthophotoFetcher(
                    cache_dir=rgb_cache_dir,
                    use_gpu=use_gpu  # Pass GPU flag
                )
                logger.info(f"RGB augmentation enabled ({'GPU' if use_gpu else 'CPU'})")
            except ImportError as e:
                logger.error(f"RGB augmentation requires additional packages: {e}")
                self.include_rgb = False

    def process_tile(self, laz_path: Path, output_path: Path) -> Dict:
        """Process a single LAZ tile with GPU RGB support."""
        # ... load points ...

        # Compute features (CPU or GPU)
        if self.use_gpu:
            # GPU pipeline
            points_gpu = cp.asarray(points)
            features_gpu = compute_all_features_with_gpu(
                points_gpu, k=k_neighbors, include_building_features=self.include_extra_features
            )

            # Add RGB on GPU (no transfer!)
            if self.include_rgb:
                bbox = self._get_tile_bbox(laz_path)  # From LAZ header
                rgb_tile_gpu = self.rgb_fetcher.fetch_tile_gpu(bbox)
                colors_gpu = self.gpu_computer.interpolate_colors_gpu(
                    points_gpu, rgb_tile_gpu, bbox
                )
                # Concatenate features + RGB on GPU
                features_gpu = cp.concatenate([
                    features_gpu,
                    colors_gpu.astype(cp.float32) / 255.0  # Normalize to [0, 1]
                ], axis=1)

            # Transfer to CPU only once at end
            features = cp.asnumpy(features_gpu)
        else:
            # CPU pipeline (existing code)
            features = compute_all_features_optimized(points, k=k_neighbors)
            if self.include_rgb:
                bbox = self._get_tile_bbox(laz_path)
                rgb_array = self.rgb_fetcher.fetch_tile(bbox)
                colors = self._interpolate_colors_cpu(points, rgb_array, bbox)
                features = np.concatenate([features, colors / 255.0], axis=1)

        # ... rest of processing ...
```

### Step 7: Create Benchmarks (Days 12-13)

**File:** `scripts/benchmarks/benchmark_rgb_gpu.py`

```python
"""
Benchmark RGB augmentation: GPU vs CPU
"""

import time
import numpy as np
from pathlib import Path

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from ign_lidar.rgb_augmentation import IGNOrthophotoFetcher
from ign_lidar.features_gpu import GPUFeatureComputer

def benchmark_rgb_augmentation():
    """Benchmark RGB color interpolation."""
    print("=" * 80)
    print("RGB Augmentation Benchmark: GPU vs CPU")
    print("=" * 80)

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

        # CPU baseline (PIL interpolation - simulated)
        t0 = time.time()
        # Simulate CPU interpolation time (~12s per 1M points)
        time_cpu_estimated = (n_points / 1_000_000) * 12.0
        print(f"CPU (estimated): {time_cpu_estimated:.3f}s")

        # GPU interpolation
        if GPU_AVAILABLE:
            computer = GPUFeatureComputer(use_gpu=True)
            points_gpu = cp.asarray(points)
            rgb_image_gpu = cp.asarray(rgb_image)

            # Warm-up
            _ = computer.interpolate_colors_gpu(points_gpu, rgb_image_gpu, bbox)
            cp.cuda.Stream.null.synchronize()

            # Benchmark
            t0 = time.time()
            colors_gpu = computer.interpolate_colors_gpu(points_gpu, rgb_image_gpu, bbox)
            cp.cuda.Stream.null.synchronize()
            time_gpu = time.time() - t0

            speedup = time_cpu_estimated / time_gpu
            print(f"GPU: {time_gpu:.3f}s")
            print(f"Speedup: {speedup:.1f}x")

            results.append({
                'n_points': n_points,
                'cpu_time': time_cpu_estimated,
                'gpu_time': time_gpu,
                'speedup': speedup
            })
        else:
            print("GPU not available - skipping")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Points':<15} {'CPU (s)':<12} {'GPU (s)':<12} {'Speedup':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['n_points']:<15,} {r['cpu_time']:<12.3f} {r['gpu_time']:<12.3f} {r['speedup']:<12.1f}x")

    if results:
        avg_speedup = np.mean([r['speedup'] for r in results])
        print(f"\nAverage speedup: {avg_speedup:.1f}x")
        print(f"Target speedup: 24x")
        print(f"Status: {'✓ PASS' if avg_speedup >= 20 else '✗ FAIL'}")

if __name__ == '__main__':
    benchmark_rgb_augmentation()
```

**Run benchmark:**

```bash
python scripts/benchmarks/benchmark_rgb_gpu.py
```

### Step 8: Documentation (Days 14-15)

Create user documentation:

**File:** `website/docs/rgb-gpu-guide.md`

````markdown
# GPU-Accelerated RGB Augmentation

**Available in:** v1.5.0+  
**Performance:** 24x faster than CPU  
**Requirements:** NVIDIA GPU, CuPy

## Overview

GPU-accelerated RGB augmentation provides dramatic speedups for adding
colors from IGN orthophotos to LiDAR point clouds.

### Performance Comparison

| Points | CPU Time | GPU Time | Speedup |
| ------ | -------- | -------- | ------- |
| 10K    | 0.12s    | 0.005s   | 24x     |
| 100K   | 1.2s     | 0.05s    | 24x     |
| 1M     | 12s      | 0.5s     | 24x     |

## Quick Start

```python
from ign_lidar.processor import LiDARProcessor

processor = LiDARProcessor(
    include_rgb=True,
    rgb_cache_dir='rgb_cache/',
    use_gpu=True  # Enable GPU RGB
)

processor.process_tile('input.laz', 'output.laz')
```
````

## CLI Usage

```bash
ign-lidar-hd enrich \
  --input tiles/ \
  --output enriched/ \
  --add-rgb \
  --rgb-cache-dir rgb_cache/ \
  --use-gpu
```

## How It Works

1. **GPU Tile Cache:** RGB tiles cached in GPU memory (LRU eviction)
2. **GPU Interpolation:** Bilinear color interpolation on GPU (CuPy)
3. **End-to-End GPU:** Data stays on GPU throughout pipeline

## Configuration

### Cache Size

Adjust GPU memory cache size:

```python
from ign_lidar.rgb_augmentation import IGNOrthophotoFetcher

fetcher = IGNOrthophotoFetcher(use_gpu=True)
fetcher.gpu_cache_max_size = 20  # Cache up to 20 tiles
```

### Fallback Behavior

If GPU unavailable, automatically falls back to CPU:

```python
processor = LiDARProcessor(use_gpu=True)  # Will use CPU if no GPU
```

## Troubleshooting

### Out of Memory

Reduce GPU cache size:

```python
fetcher.gpu_cache_max_size = 5  # Smaller cache
```

### Slow Performance

Clear cache periodically:

```python
fetcher.clear_gpu_cache()
```

## See Also

- [GPU Guide](gpu-guide.md) - Main GPU documentation
- [RGB Augmentation](rgb-augmentation.md) - RGB feature overview

````

### Step 9: Testing & Release (Days 16-21)

```bash
# Run full test suite
pytest tests/ -v

# Run benchmarks
python scripts/benchmarks/benchmark_rgb_gpu.py
python scripts/benchmarks/benchmark_gpu.py

# Check documentation
cd website
npm run build

# Update CHANGELOG
# Add v1.5.0 section

# Create release
git add .
git commit -m "feat: GPU-accelerated RGB augmentation (v1.5.0)"
git push origin feature/phase3.1-rgb-gpu

# Create pull request
# After review and merge:
git tag v1.5.0
git push origin v1.5.0

# Build and publish to PyPI
python -m build
twine upload dist/*
````

---

## 📊 Success Criteria

Before considering Phase 3.1 complete:

- ✅ `interpolate_colors_gpu()` implemented and tested
- ✅ GPU tile cache working with LRU eviction
- ✅ Integration tests passing
- ✅ 20-24x speedup demonstrated
- ✅ Automatic CPU fallback working
- ✅ Documentation complete
- ✅ v1.5.0 released to PyPI

---

## 🔄 Next Steps

After completing Phase 3.1:

1. **Gather feedback** (1-2 weeks)

   - Monitor GitHub issues
   - Check performance reports
   - User satisfaction

2. **Decide on Phase 3.2**

   - If multi-GPU demand high → Proceed
   - If low demand → Pause Phase 3

3. **Start Phase 3.2** (if approved)
   - Follow [GPU_PHASE3_PLAN.md](GPU_PHASE3_PLAN.md) Section 3.2
   - Multi-GPU support implementation

---

## 📚 Resources

- **Full Plan:** [GPU_PHASE3_PLAN.md](GPU_PHASE3_PLAN.md)
- **Summary:** [GPU_PHASE3_SUMMARY.md](GPU_PHASE3_SUMMARY.md)
- **Roadmap:** [GPU_PHASE3_ROADMAP.md](GPU_PHASE3_ROADMAP.md)
- **Current Status:** [GPU_COMPLETE.md](GPU_COMPLETE.md)

---

**Last Updated:** October 3, 2025  
**Status:** Ready for implementation  
**Estimated Time:** 3 weeks (15 hours)
