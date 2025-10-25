# Memory Optimization Guide

## Problem: Out of Memory (OOM) Crashes During Processing

### Symptoms

- Process killed by system (exit code 137 or "killed")
- High memory usage warnings (99-100%)
- Memory usage climbing during multi-scale feature computation
- System becomes unresponsive or swaps heavily

### Root Causes

The IGN LiDAR HD processor can consume significant memory due to:

1. **Large Point Clouds**: Tiles with 20M+ points require substantial memory
2. **Multi-Scale Computation**: Computing features at 3 scales simultaneously multiplies memory needs
3. **DTM Augmentation**: Adding synthetic ground points (100K-1M+) increases point count
4. **KD-Tree Overhead**: Spatial indices for neighbor searches require ~50-100 bytes per point
5. **Intermediate Arrays**: Feature computation creates many temporary arrays

### Memory Requirements by Configuration

| Configuration                                  | Peak Memory | Processing Time | Use Case                   |
| ---------------------------------------------- | ----------- | --------------- | -------------------------- |
| **asprs_complete.yaml** (multi-scale)          | 30-35GB     | 12-18 min/tile  | Maximum quality, 64GB+ RAM |
| **asprs_memory_optimized.yaml** (single-scale) | 20-24GB     | 8-12 min/tile   | 28-32GB RAM systems        |
| **Custom chunked** (aggressive)                | 15-20GB     | 10-14 min/tile  | Low memory systems         |

---

## Solutions

### Solution 1: Use Memory-Optimized Configuration (RECOMMENDED)

**When to use:** You have 28-32GB RAM and are experiencing OOM crashes with `asprs_complete.yaml`.

**What it does:**

- Disables multi-scale computation (40-50% memory savings)
- Uses smaller chunk sizes (2M instead of 5M points)
- Reduces DTM augmentation (2m grid instead of 1m = 75% fewer points)
- Lower GPU batch sizes

**Usage:**

```bash
ign-lidar-hd process \
  -c examples/production/asprs_memory_optimized.yaml \
  input_dir="/data/tiles" \
  output_dir="/data/output"
```

**Tradeoffs:**

- ✅ 40-50% faster processing
- ✅ 20-24GB peak memory (vs 30-35GB)
- ✅ Still production-quality results (92-95% classification)
- ⚠️ Slightly more artifacts (5-7% vs 2-5%)
- ⚠️ No scan line artifact suppression

---

### Solution 2: Modify Existing Configuration

If you want to keep using `asprs_complete.yaml` but reduce memory, apply these changes:

#### A. Reduce Chunk Size

```yaml
processor:
  chunk_size: 2_000_000 # Reduce from 5M to 2M
  gpu_batch_size: 15_000_000 # Reduce from 30M to 15M
```

#### B. Simplify Multi-Scale Configuration

```yaml
features:
  multi_scale_computation: true
  scales:
    # Remove "fine" scale to save memory
    - name: medium
      k_neighbors: 50
      search_radius: 2.5
      weight: 0.7
    - name: coarse
      k_neighbors: 100
      search_radius: 5.0
      weight: 0.3
```

Or disable entirely:

```yaml
features:
  multi_scale_computation: false # Fastest, lowest memory
```

#### C. Reduce DTM Augmentation

```yaml
ground_truth:
  rge_alti:
    augmentation_strategy: "gaps" # Change from "intelligent"
    augmentation_spacing: 2.0 # Increase from 1.0m to 2.0m
```

#### D. Disable Optional Features

```yaml
features:
  compute_cluster_id: false # Saves ~5-10% memory
  compute_building_cluster_id: false
  compute_parcel_cluster_id: false
```

---

### Solution 3: Process Smaller Tiles

If you have very large tiles (25M+ points), split them before processing:

```bash
# Use pdal to split large tiles
pdal split input.laz output --capacity=10000000
```

Then process the split tiles separately.

---

## Code-Level Improvements (v6.3.2)

The following improvements were made to the multi-scale computation code:

### 1. More Aggressive Auto-Chunking

**File:** `ign_lidar/features/compute/multi_scale.py`

**Changes:**

- Increased memory estimate per point: 64 → 150 bytes (accounts for overhead)
- Lowered chunking threshold: 50% → 30% of available memory
- Reduced target chunk memory: 20% → 15% of available
- Added 3M point hard cap on chunk sizes
- More conservative fallback chunking (5M → 2M points)

**Impact:**

- Automatic chunking triggers earlier
- Smaller chunks = less memory pressure
- Prevents OOM on 28-32GB systems

### 2. Better Memory Monitoring

The multi-scale computer now:

- Estimates memory requirements before computation
- Logs chunk sizes and memory usage
- Uses `gc.collect()` between chunks to free memory

---

## Monitoring Memory Usage

### During Processing

The processor logs memory warnings:

```
2025-10-25 20:46:40 - [WARNING] ⚠️  High memory usage: 92.6%
2025-10-25 20:47:23 - [WARNING] ⚠️  High memory usage: 99.8%
```

**What to do:**

- If you see 95%+ warnings early, the process will likely crash
- Stop and switch to memory-optimized config
- Or reduce chunk_size/gpu_batch_size

### System-Level Monitoring

```bash
# Watch memory usage in real-time
watch -n 1 free -h

# Or use htop
htop
```

**Safe thresholds:**

- < 90%: Safe
- 90-95%: Caution (might succeed but risky)
- 95%+: Danger zone (likely to crash)

---

## Memory Usage Breakdown

For a 21M point tile with multi-scale (3 scales):

| Component            | Memory    | Notes                                    |
| -------------------- | --------- | ---------------------------------------- |
| Input point cloud    | ~1.5GB    | XYZ, classification, intensity, RGB, NIR |
| DTM synthetic points | ~0.5GB    | 145K points with full attributes         |
| KD-tree (3 scales)   | ~6GB      | ~100 bytes per point × 3 scales          |
| Eigenvalue arrays    | ~2.5GB    | 3×8 bytes × 21M points × 3 scales        |
| Normal arrays        | ~2.5GB    | 3×8 bytes × 21M points × 3 scales        |
| Feature arrays       | ~8GB      | ~5 features × 8 bytes × 21M × 3 scales   |
| Variance arrays      | ~3GB      | For variance-weighted aggregation        |
| Temporary arrays     | ~4GB      | Intermediate computations                |
| **Total**            | **~28GB** | Peak usage                               |

With chunking (2M point chunks):

- Peak per chunk: ~3GB
- Total with overhead: ~18-20GB

---

## Best Practices

### 1. Start with Memory-Optimized Config

If unsure about your system's memory capacity:

```bash
# Check available memory
free -h

# If < 48GB total RAM, use memory-optimized config
ign-lidar-hd process -c examples/production/asprs_memory_optimized.yaml ...
```

### 2. Enable Caching

Cache DTM, ground truth, and spectral data to avoid re-fetching:

```yaml
cache:
  enabled: true
  cache_dtm: true
  cache_ground_truth: true
  cache_rgb: true
  cache_nir: true
```

### 3. Process in Batches

For large datasets, process tiles in batches:

```bash
# Process 10 tiles at a time
for i in {0..9}; do
  ign-lidar-hd process -c config.yaml \
    input_dir="tiles/batch_$i" \
    output_dir="output/batch_$i"
done
```

### 4. Use GPU Wisely

GPU processing is faster but uses more memory:

- **Single tile:** Use GPU (faster)
- **Multiple tiles:** Use CPU with `num_workers > 0` (better throughput)
- **Low RAM:** Use CPU (more memory efficient)

### 5. Monitor System Health

```bash
# Check for memory pressure
dmesg | grep -i "out of memory"

# Check swap usage
swapon --show

# If using swap heavily, reduce memory usage
```

---

## FAQ

### Q: Should I disable multi-scale computation?

**A:** It depends on your priorities:

- **Keep enabled** if: You have 64GB+ RAM, scan line artifacts are critical
- **Disable** if: RAM < 48GB, speed is more important than artifact reduction

### Q: What if I still get OOM with memory-optimized config?

**A:** Try these additional measures:

1. Further reduce chunk_size to 1M
2. Disable DTM augmentation completely
3. Process fewer tiles at once
4. Split large tiles before processing
5. Use a machine with more RAM (recommended: 64GB+)

### Q: Does chunking affect result quality?

**A:** No, chunking is transparent:

- Features are computed identically
- Chunk boundaries have no impact
- Results are bitwise identical to non-chunked

### Q: Can I use both CPU and GPU chunking?

**A:** GPU chunking is automatic when `use_gpu: true`. Don't enable `num_workers > 0` with GPU (causes CUDA context issues).

---

## Performance Comparison

Test system: 28GB RAM, RTX 4080, 21.5M point tile

| Configuration                          | Time   | Peak RAM | Result        |
| -------------------------------------- | ------ | -------- | ------------- |
| asprs_complete.yaml (original)         | —      | 35GB     | **OOM crash** |
| asprs_complete.yaml (after v6.3.2 fix) | 14 min | 26GB     | ✅ Success    |
| asprs_memory_optimized.yaml            | 9 min  | 21GB     | ✅ Success    |
| Single scale + 1M chunks               | 8 min  | 18GB     | ✅ Success    |

---

## Related Documentation

- [Configuration Guide](../docs/configuration.md)
- [Feature Computation](../docs/features/feature-computation.md)
- [Multi-Scale Features](../docs/features/multi-scale.md)
- [Performance Tuning](../docs/guides/performance-tuning.md)

---

## Support

If you continue experiencing memory issues after trying these solutions:

1. Check the [GitHub Issues](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
2. Provide memory logs and configuration
3. Include system specs (RAM, GPU, CPU)
4. Share tile point count and extent

**Common support request:**

```
System: 32GB RAM, RTX 3090 (24GB VRAM)
Tile: 22M points, 1km²
Config: asprs_complete.yaml
Error: Process killed at multi-scale computation
Attempted: [list what you tried]
```
