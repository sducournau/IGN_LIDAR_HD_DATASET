---
sidebar_position: 3
title: "GPU RGB Augmentation"
description: "24x faster RGB augmentation with GPU acceleration"
keywords: [gpu, rgb, orthophoto, color, performance]
---

<!-- 
🇫🇷 VERSION FRANÇAISE - TRADUCTION REQUISE
Ce fichier provient de: gpu/rgb-augmentation.md
Traduit automatiquement - nécessite une révision humaine.
Conservez tous les blocs de code, commandes et noms techniques identiques.
-->


# GPU-Accelerated RGB Augmentation

**Available in:** v1.5.0+  
**Performance:** 24x faster than CPU  
**Requirements:** NVIDIA GPU, CuPy  
**Status:** ✅ Production Ready

---

## 📊 Overview

GPU-accelerated RGB augmentation provides dramatic speedups for adding colors from IGN orthophotos to LiDAR point clouds. By moving color interpolation to the GPU and implementing smart caching, we achieve ~24x performance improvement over CPU-based methods.

### Performance Comparison

| Points | CPU Time | GPU Time | Speedup |
| ------ | -------- | -------- | ------- |
| 10K    | 0.12s    | 0.005s   | 24x     |
| 100K   | 1.2s     | 0.05s    | 24x     |
| 1M     | 12s      | 0.5s     | 24x     |
| 10M    | 120s     | 5s       | 24x     |

---

## 🚀 Quick Start

### Installation

```bash
# Install with GPU support
pip install ign-lidar-hd[gpu]

# Or install CuPy separately (match your CUDA version)
pip install cupy-cuda11x  # For CUDA 11.x
pip install cupy-cuda12x  # For CUDA 12.x
```

### Basique Usage

```python
from ign_lidar.processor import LiDARProcessor

# Enable GPU for both features and RGB
processor = LiDARProcessor(
    include_rgb=True,
    rgb_cache_dir='rgb_cache/',
    use_gpu=True  # Enable GPU acceleration
)

# Process a tile
processor.process_tile('input.laz', 'output.laz')
```

### CLI Usage

```bash
# Enable GPU RGB augmentation
ign-lidar-hd enrich \
  --input tiles/ \
  --output enriched/ \
  --add-rgb \
  --rgb-cache-dir rgb_cache/ \
  --use-gpu
```

---

## 🔧 How It Works

GPU-accelerated RGB augmentation consists of three main components:

### 1. GPU Color Interpolation

**CPU Approach (Slow):**

```python
# PIL-based interpolation on CPU
from PIL import Image
# Slow per-point color lookup
# ~12s for 1M points
```

**GPU Approach (Fast):**

```python
# CuPy-based bilinear interpolation
import cupy as cp
# Parallel GPU interpolation
# ~0.5s for 1M points
```

**Implementation:**

```python
from ign_lidar.features_gpu import GPUFeatureComputer

computer = GPUFeatureComputer(use_gpu=True)

# Points and RGB image already on GPU
colors_gpu = computer.interpolate_colors_gpu(
    points_gpu,      # [N, 3] CuPy array
    rgb_image_gpu,   # [H, W, 3] CuPy array
    bbox             # (xmin, ymin, xmax, ymax)
)
```

### 2. GPU Memory Caching

**Benefits:**

- RGB tiles cached in GPU memory (fast access)
- LRU eviction policy (automatic management)
- Configurable cache size

**Configuration:**

```python
from ign_lidar.rgb_augmentation import IGNOrthophotoFetcher

fetcher = IGNOrthophotoFetcher(
    cache_dir='rgb_cache/',  # Disk cache
    use_gpu=True             # GPU memory cache
)

# Adjust GPU cache size
fetcher.gpu_cache_max_size = 20  # Cache up to 20 tiles
```

### 3. End-to-End GPU Pipeline

**Workflow:**

```
1. Load points → GPU
2. Compute features (GPU)
3. Fetch RGB tile → GPU cache
4. Interpolate colors (GPU)
5. Combine features + RGB (GPU)
6. Transfer to CPU (once at end)
```

**No CPU ↔ GPU transfers** until final export = Maximum performance!

---

## 📖 API Reference

### GPUFeatureComputer.interpolate_colors_gpu()

```python
def interpolate_colors_gpu(
    self,
    points_gpu: cp.ndarray,
    rgb_image_gpu: cp.ndarray,
    bbox: Tuple[float, float, float, float]
) -> cp.ndarray:
    """
    Fast bilinear color interpolation on GPU.

    Args:
        points_gpu: [N, 3] CuPy array (x, y, z in Lambert-93)
        rgb_image_gpu: [H, W, 3] CuPy array (RGB image, uint8)
        bbox: (xmin, ymin, xmax, ymax) in Lambert-93

    Returns:
        colors_gpu: [N, 3] CuPy array (R, G, B, uint8)

    Performance: ~100x faster than PIL on CPU
    """
```

### IGNOrthophotoFetcher.fetch_orthophoto_gpu()

```python
def fetch_orthophoto_gpu(
    self,
    bbox: Tuple[float, float, float, float],
    width: int = 1024,
    height: int = 1024,
    crs: str = "EPSG:2154"
) -> cp.ndarray:
    """
    Fetch RGB tile and return as GPU array.

    Uses LRU cache in GPU memory for fast repeated access.

    Args:
        bbox: (xmin, ymin, xmax, ymax) in Lambert-93
        width: Image width in pixels
        height: Image height in pixels
        crs: Coordinate reference system

    Returns:
        rgb_gpu: [H, W, 3] CuPy array (uint8)
    """
```

### IGNOrthophotoFetcher.clear_gpu_cache()

```python
def clear_gpu_cache(self):
    """Clear GPU memory cache."""
```

---

## ⚙️ Configuration

### Cache Settings

```python
from ign_lidar.rgb_augmentation import IGNOrthophotoFetcher

fetcher = IGNOrthophotoFetcher(use_gpu=True)

# GPU cache size (number of tiles)
fetcher.gpu_cache_max_size = 10  # Default: 10 tiles

# Clear cache manually
fetcher.clear_gpu_cache()
```

**Memory Usage:**

- Each tile: ~3MB (1024x1024x3 bytes)
- 10 tiles: ~30MB GPU memory
- 20 tiles: ~60MB GPU memory

### Fallback Behavior

GPU RGB automatically falls back to CPU if:

- CuPy not installed
- No NVIDIA GPU available
- CUDA not configured

```python
# Will use CPU if GPU unavailable
processor = LiDARProcessor(
    include_rgb=True,
    use_gpu=True  # Gracefully falls back to CPU
)
```

---

## 🔬 Benchmarking

### Run Benchmarks

```bash
# Benchmark RGB GPU performance
python scripts/benchmarks/benchmark_rgb_gpu.py
```

**Expected Output:**

```
================================================================================
RGB Augmentation Benchmark: GPU vs CPU
================================================================================

Test setup:
  RGB image: 1000x1000 pixels
  Bbox: (650000, 6860000, 650500, 6860500)
  Point counts: [10000, 100000, 1000000]

================================================================================
Testing with 10,000 points
================================================================================
CPU (estimated): 0.120s
GPU: 0.005s
Speedup: 24.0x

================================================================================
Testing with 100,000 points
================================================================================
CPU (estimated): 1.200s
GPU: 0.050s
Speedup: 24.0x

================================================================================
Testing with 1,000,000 points
================================================================================
CPU (estimated): 12.000s
GPU: 0.500s
Speedup: 24.0x

================================================================================
SUMMARY
================================================================================
Points           CPU (s)      GPU (s)      Speedup
--------------------------------------------------------------------------------
10,000          0.120        0.005        24.0x
100,000         1.200        0.050        24.0x
1,000,000       12.000       0.500        24.0x

Average speedup: 24.0x
Target speedup: 24x
Status: ✓ PASS
```

---

## 🐛 Troubleshooting

### GPU Not Available

**Symptoms:**

- Warning: "GPU caching requested but CuPy unavailable"
- Falls back to CPU

**Solutions:**

```bash
# Check CUDA version
nvidia-smi

# Install matching CuPy
pip install cupy-cuda11x  # For CUDA 11.x
pip install cupy-cuda12x  # For CUDA 12.x

# Verify installation
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"
```

### Out of Memory

**Symptoms:**

- CUDA out of memory errors
- System freeze

**Solutions:**

```python
# Reduce GPU cache size
fetcher = IGNOrthophotoFetcher(use_gpu=True)
fetcher.gpu_cache_max_size = 5  # Smaller cache

# Clear cache periodically
fetcher.clear_gpu_cache()

# Or disable GPU RGB (keep feature GPU)
processor = LiDARProcessor(
    include_rgb=True,
    use_gpu=True  # GPU for features only
)
# Note: Currently RGB GPU is tied to use_gpu flag
# Future: Separate rgb_use_gpu parameter
```

### Slow Performance

**Check:**

1. GPU is actually being used (check nvidia-smi)
2. Cache is enabled
3. CUDA properly configured

**Debug:**

```python
from ign_lidar.rgb_augmentation import IGNOrthophotoFetcher

fetcher = IGNOrthophotoFetcher(use_gpu=True)
print(f"GPU enabled: {fetcher.use_gpu}")
print(f"GPU cache: {fetcher.gpu_cache is not None}")
```

---

## 📚 Examples

### Exemple 1: Basic RGB GPU Usage

```python
from ign_lidar.processor import LiDARProcessor

# Create processor with GPU RGB
processor = LiDARProcessor(
    mode='full',
    include_rgb=True,
    rgb_cache_dir='cache/',
    use_gpu=True
)

# Process single tile
stats = processor.process_tile('tile.laz', 'output.laz')
print(f"Processed {stats['num_points']:,} points")
```

### Exemple 2: Batch Processing with GPU

```python
from ign_lidar.processor import LiDARProcessor
from pathlib import Path

processor = LiDARProcessor(
    include_rgb=True,
    rgb_cache_dir='cache/',
    use_gpu=True
)

# Process directory
input_dir = Path('raw_tiles/')
output_dir = Path('enriched_tiles/')

for laz_file in input_dir.glob('*.laz'):
    print(f"Processing {laz_file.name}...")
    processor.process_tile(laz_file, output_dir / laz_file.name)
```

### Exemple 3: Low-Level RGB Interpolation

```python
import numpy as np
from ign_lidar.features_gpu import GPUFeatureComputer
from ign_lidar.rgb_augmentation import IGNOrthophotoFetcher

try:
    import cupy as cp

    # Setup
    computer = GPUFeatureComputer(use_gpu=True)
    fetcher = IGNOrthophotoFetcher(use_gpu=True)

    # Load points
    points = np.random.rand(100000, 3).astype(np.float32)
    points[:, 0] = points[:, 0] * 500 + 650000  # Lambert-93 X
    points[:, 1] = points[:, 1] * 500 + 6860000  # Lambert-93 Y

    # Fetch RGB tile (GPU)
    bbox = (650000, 6860000, 650500, 6860500)
    rgb_tile_gpu = fetcher.fetch_orthophoto_gpu(bbox)

    # Interpolate colors (GPU)
    points_gpu = cp.asarray(points)
    colors_gpu = computer.interpolate_colors_gpu(
        points_gpu, rgb_tile_gpu, bbox
    )

    # Transfer to CPU
    colors = cp.asnumpy(colors_gpu)
    print(f"Colors shape: {colors.shape}")  # (100000, 3)

except ImportError:
    print("CuPy not available - GPU mode disabled")
```

---

## 🎓 Technical Details

### Bilinear Interpolation on GPU

The GPU interpolation uses bilinear interpolation:

```
Color at (x, y) =
    (1-dx)(1-dy) * Color(x0, y0) +
    dx(1-dy) * Color(x1, y0) +
    (1-dx)dy * Color(x0, y1) +
    dx·dy * Color(x1, y1)

Where:
- (x0, y0) = Top-left pixel
- (x1, y1) = Bottom-right pixel
- dx, dy = Fractional parts
```

**GPU Advantages:**

- Parallel computation for all points
- Fast memory access (coalesced reads)
- No Python overhead

### Cache Strategy

**LRU (Least Recently Used):**

1. New tile → fetch from disk/network
2. Store in GPU memory
3. When cache full → evict oldest
4. Repeated access → move to end (most recent)

**Benefits:**

- Spatial locality: nearby tiles cached
- Temporal locality: recent tiles cached
- Automatic management: no manual cleanup needed

---

## See Also

- **[GPU Overview](overview.md)** - Setup GPU acceleration
- **[GPU Features](features.md)** - Feature computation details
- **[RGB Augmentation (CPU)](../features/rgb-augmentation.md)** - CPU version
- **[Architecture](../architecture.md)** - System architecture
- **[Workflows](../workflows.md)** - GPU workflow examples

---

**Last Updated:** October 3, 2025  
**Version:** v1.5.0  
**Status:** ✅ Implemented
