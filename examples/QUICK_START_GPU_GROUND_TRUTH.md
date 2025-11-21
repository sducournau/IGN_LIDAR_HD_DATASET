# Quick Start: GPU Training with Ground Truth

**Config:** `config_training_optimized_gpu.yaml`

## TL;DR

```bash
# 1. Activate environment with GPU support
conda activate ign_lidar_gpu  # or your GPU env

# 2. Run processing
ign-lidar process \
    --config examples/config_training_optimized_gpu.yaml \
    --input-dir /path/to/laz_tiles \
    --output-dir /path/to/output

# 3. Check output
ls output/patches/  # Training patches (.npz)
ls output/enriched_tiles/  # Enriched LAZ files
```

---

## What You Get

✅ **Ground Truth Labels** from IGN BD Topo:

- Roads (with width from centerlines)
- Buildings (footprints)
- Vegetation (zones)
- Water (surfaces)

✅ **GPU-Accelerated Processing:**

- Feature computation: 10-100× faster
- Ground truth labeling: 100-1000× faster
- Total speedup: ~5-10× for full pipeline

✅ **Smart Caching:**

- First run: fetches from WFS (~0.5s/tile)
- Cached runs: loads from disk (~0.01s/tile)
- 50× speedup after cache warm-up

---

## System Requirements

- **GPU:** NVIDIA GPU with 8GB+ VRAM (tested on RTX 3060 12GB)
- **CUDA:** 11.x or 12.x
- **Python:** 3.8+
- **Internet:** Required for BD Topo WFS (first run only)

**Install GPU dependencies:**

```bash
pip install cupy-cuda11x cuml-cu11 cuspatial  # For CUDA 11.x
# OR
pip install cupy-cuda12x cuml-cu12 cuspatial  # For CUDA 12.x
```

---

## Configuration Summary

### GPU Settings (Optimized for RTX 3060)

```yaml
processor:
  use_gpu: true
  gpu_batch_size: 2_000_000 # 2M points/batch
  vram_limit_gb: 9.0 # 75% of 12GB
  num_workers: 0 # No multiprocessing with GPU

features:
  gpu_batch_size: 2_000_000
  use_gpu_chunked: true # Chunked strategy
  force_gpu: true
```

### Ground Truth Settings

```yaml
ground_truth:
  enabled: true
  method: "auto" # Auto-select GPU
  chunk_size: 2_000_000 # Match GPU batch

  bd_topo:
    features:
      buildings: true
      roads: true # ← Road ground truth
      vegetation: true
      water: true

    parameters:
      road_width_fallback: 4.0 # Default road width (m)
```

### Cache Settings

```yaml
cache:
  enabled: true
  cache_dir: "/mnt/c/.../cache" # Fast SSD
  cache_ground_truth: true # ← Cache WFS responses
  cache_features: true
  cache_kdtrees: true
```

---

## Expected Performance

### Single Tile (~10M points)

| Component             | Time     | Notes                  |
| --------------------- | -------- | ---------------------- |
| Load LAZ              | 0.5-1s   |                        |
| Ground Truth (first)  | 0.4-0.8s | WFS fetch              |
| Ground Truth (cached) | 0.1-0.3s | Disk read              |
| Features (GPU)        | 2-4s     | 10-20× faster than CPU |
| Patches               | 1-2s     |                        |
| Save                  | 0.5-1s   |                        |
| **Total (first)**     | **5-9s** |                        |
| **Total (cached)**    | **4-8s** |                        |

### Batch (128 tiles)

| Run                  | Total Time |
| -------------------- | ---------- |
| First (builds cache) | 10-20 min  |
| Cached               | 8-15 min   |

---

## Verify It's Working

### 1. Check GPU Usage

```bash
watch -n 1 nvidia-smi
```

Look for:

- GPU Utilization: >70%
- Memory Usage: ~6-8GB
- Process: `python` running

### 2. Check Logs

```
INFO: Using GPU strategy: GPU_CHUNKED
INFO: GPU acceleration enabled (device 0)
INFO: Fetching roads from WFS for bbox ...
INFO: Generated 1234 road polygons
INFO: Label distribution:
  roads: 45678 points (12.3%)
```

### 3. Inspect Output

```python
import numpy as np

data = np.load("output/patches/patch_XXX_YYY.npz")
labels = data['labels']

# Should see road labels (class 2)
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))
# Expected: {0: XXXX, 1: XXXX, 2: XXXX, 4: XXXX}
#           unlabeled, building, road, vegetation
```

---

## Troubleshooting

### "GPU not available"

```bash
# Check CUDA
nvidia-smi

# Check CuPy
python -c "import cupy; print(cupy.cuda.is_available())"

# Install if missing
pip install cupy-cuda11x  # or cuda12x
```

### "WFS fetch failed"

1. Check internet connection
2. Verify tile is in France (Lambert 93)
3. Try clearing cache: `rm -rf cache/ground_truth`

### "Out of GPU memory"

Reduce batch size in config:

```yaml
processor:
  gpu_batch_size: 1_000_000 # Reduce from 2M

features:
  gpu_batch_size: 1_000_000

ground_truth:
  chunk_size: 1_000_000
```

### "No road labels"

Check if tile has roads:

- Urban areas: lots of roads
- Forest areas: few roads
- Water areas: no roads

View in QGIS/CloudCompare to verify.

---

## Next Steps

1. **Process your tiles**

   ```bash
   ign-lidar process --config config_training_optimized_gpu.yaml \
       --input-dir ./tiles --output-dir ./output
   ```

2. **Check label distribution**

   ```bash
   python -c "
   import numpy as np
   from pathlib import Path

   patches = list(Path('output/patches').glob('*.npz'))
   print(f'Found {len(patches)} patches')

   # Load first patch
   data = np.load(patches[0])
   print('Labels:', np.unique(data['labels'], return_counts=True))
   "
   ```

3. **Train your model**

   ```python
   from ign_lidar.datasets import MultiArchDataset

   dataset = MultiArchDataset(
       patch_dir="output/patches",
       architecture="hybrid",
       use_ground_truth=True
   )
   ```

---

## Need Help?

- **Full Guide:** `GPU_TRAINING_WITH_GROUND_TRUTH.md`
- **Ground Truth Docs:** `docs/docs/features/ground-truth-fetching.md`
- **GPU Optimization:** `PERFORMANCE_OPTIMIZATION_NOV_2025.md`
- **GitHub Issues:** https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues
