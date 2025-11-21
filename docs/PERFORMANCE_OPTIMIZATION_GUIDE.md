# Performance Optimization Guide

## Overview

This guide explains performance optimizations for IGN LiDAR HD processing, especially for large datasets (>15M points per tile).

## Recent Optimizations (November 2025)

### 1. FAISS GPU Batch Processing

**Problem**: Processing was slow due to small batch sizes and lack of progress visibility.

**Solution**:

- Increased GPU batch size from **2M to 5M points** per batch
- Added progress logging every 10% of batches
- Reduced GC cleanup frequency (every 20 batches instead of 10)

**Impact**: ~40% faster k-NN queries on GPU

```python
# Before
batch_size = 2_000_000  # Too small for GPU

# After
batch_size = 5_000_000  # Better GPU utilization
```

### 2. Configuration Presets

Three optimized configurations available:

#### Standard (config_training_simple_50m_stitched.yaml)

- **Speed**: 4-6 min/tile
- **Features**: Full LOD2 set
- **k_neighbors**: 30
- **Stitching**: Enabled
- **Use case**: Production quality

#### Fast (config_training_fast_50m.yaml) ⚡

- **Speed**: 2-3 min/tile (~50% faster)
- **Features**: Essential only
- **k_neighbors**: 20 (33% less computation)
- **Stitching**: Disabled
- **Use case**: Rapid prototyping, testing

#### Hybrid (config_pointnet_transformer_hybrid_training.yaml)

- **Speed**: 5-8 min/tile
- **Features**: Multi-scale + augmentation
- **k_neighbors**: 30
- **Stitching**: Enabled
- **Use case**: Research, best quality

## Performance Tips

### GPU Optimization

1. **Use larger batches**:

   ```yaml
   processor:
     gpu_batch_size: 15_000_000 # For 16GB VRAM

   features:
     gpu_batch_size: 5_000_000
   ```

2. **Maximize VRAM usage**:

   ```yaml
   processor:
     gpu_memory_target: 0.90 # Use 90% of VRAM
     vram_limit_gb: 14 # For 16GB GPU
   ```

3. **Disable workers with GPU**:
   ```yaml
   processor:
     num_workers: 0 # Always 0 when use_gpu=true
   ```

### Feature Computation

1. **Reduce k_neighbors for speed**:

   ```yaml
   features:
     k_neighbors: 20 # Instead of 30 (33% faster)
     search_radius: 2.0 # Instead of 2.5m
   ```

2. **Disable unnecessary features**:

   ```yaml
   features:
     compute_anisotropy: false # If not needed
     compute_ndvi: false # If no vegetation analysis
   ```

3. **Use essential features only**:
   ```yaml
   feature_selection:
     geometric_features:
       - normals
       - planarity
       - verticality
     height_features:
       - height_above_ground
     # Skip: omnivariance, eigenentropy, etc.
   ```

### Preprocessing

1. **Minimal preprocessing**:

   ```yaml
   preprocess:
     sor_k: 12 # Instead of 16
     sor_std: 2.5 # More permissive
     ror_neighbors: 4 # Instead of 5
   ```

2. **Skip voxel downsampling**:
   ```yaml
   preprocess:
     voxel_enabled: false
   ```

### Stitching

1. **Disable if not needed**:

   ```yaml
   stitching:
     enabled: false # ~1 min savings per tile
   ```

2. **Reduce overlap**:
   ```yaml
   processor:
     overlap: 0.10 # 10% instead of 15%
   ```

### Classification

1. **Disable reclassification**:

   ```yaml
   reclassification:
     enabled: false # Saves ~30 seconds per tile
   ```

2. **Disable cluster IDs**:
   ```yaml
   data_sources:
     bd_topo:
       assign_building_cluster_ids: false
       assign_parcel_cluster_ids: false
   ```

### Monitoring

1. **Reduce logging frequency**:

   ```yaml
   monitoring:
     progress_interval: 10 # Update every 10 seconds
   ```

2. **Disable progress bars in batch mode**:
   ```yaml
   ground_truth:
     show_progress: false
   ```

### Validation

1. **Relax validation**:
   ```yaml
   validation:
     reject_incomplete_patches: false # Flag but don't reject
     reject_patches_with_nans: false
     check_class_distribution: false
   ```

## Monitoring Performance

### Real-time Monitoring

Use the monitoring script:

```bash
chmod +x scripts/monitor_processing.sh
./scripts/monitor_processing.sh
```

Output:

```
===================================================================
System Status - 2025-11-21 03:30:00
===================================================================

--- GPU Status ---
  GPU: NVIDIA GeForce RTX 4080 | Temp: 65°C | Util: 98% | VRAM: 10240/16384 MB (62% used)

--- System Resources ---
  CPU Usage: 15.2%
  Memory: 12.5GB / 32.0GB (39.1% used)

--- IGN LiDAR HD Process ---
  PID: 12345
  CPU%: 112.3
  MEM%: 8.5
  Runtime: 00:15:23
  Status: RUNNING ✅

--- Output Progress ---
  Processed tiles: 12 / 154 (7%)
===================================================================
```

### Check Log Output

The logs show:

1. **FAISS index build time**: Should be 2-3 seconds
2. **k-NN query batches**: Progress every batch
3. **Feature computation**: Per-feature timings
4. **Memory usage**: GPU and system RAM

Example log:

```
2025-11-21 03:26:25 - [INFO]   🚀 Building FAISS index (18,651,688 points, k=25)...
2025-11-21 03:26:27 - [INFO]      ✓ Training complete
2025-11-21 03:26:30 - [INFO]      ✓ FAISS IVFFlat index ready (nprobe=64)
2025-11-21 03:26:30 - [INFO]   ⚡ Querying 18,651,688 × 25 neighbors in 4 batches...
2025-11-21 03:26:30 - [INFO]      Estimated time: 56 seconds
2025-11-21 03:26:35 - [INFO]      → Batch 1/4 (25.0%)
2025-11-21 03:26:40 - [INFO]      → Batch 2/4 (50.0%)
2025-11-21 03:26:45 - [INFO]      → Batch 3/4 (75.0%)
2025-11-21 03:26:50 - [INFO]      → Batch 4/4 (100.0%)
2025-11-21 03:26:50 - [INFO]      ✓ All neighbors found (4 batches completed)
```

## Troubleshooting

### Problem: Processing hangs at "Querying neighbors"

**Causes**:

1. GPU memory exhausted
2. Batch size too large
3. CUDA context issues

**Solutions**:

```yaml
# Reduce batch sizes
processor:
  gpu_batch_size: 10_000_000  # From 15M

features:
  gpu_batch_size: 3_000_000  # From 5M

# Or use CPU fallback
processor:
  use_gpu: false
```

### Problem: Out of Memory (OOM)

**Solutions**:

```yaml
# Reduce VRAM target
processor:
  gpu_memory_target: 0.75  # From 0.90

# Enable chunking
features:
  use_gpu_chunked: true
  gpu_batch_size: 2_000_000

# Reduce k_neighbors
features:
  k_neighbors: 20  # From 30
```

### Problem: Progress not visible

**Solutions**:

1. Check if `tqdm` is installed: `pip install tqdm`
2. Enable progress in config:
   ```yaml
   ground_truth:
     show_progress: true
   ```
3. Use monitoring script (see above)

### Problem: Very slow processing (>10 min/tile)

**Diagnosis**:

```bash
# Check GPU utilization
watch -n 1 nvidia-smi

# Expected: 80-100% GPU util during feature computation
# If low (<50%): bottleneck is CPU/disk
```

**Solutions**:

1. Use fast config preset
2. Disable stitching
3. Reduce k_neighbors
4. Check disk I/O (use SSD if possible)

## Benchmark Results

### Hardware: RTX 4080 16GB, 32GB RAM, SSD

| Configuration | Time/Tile | 154 Tiles | Quality    |
| ------------- | --------- | --------- | ---------- |
| **Standard**  | 4-6 min   | 10-15 hrs | ⭐⭐⭐⭐⭐ |
| **Fast**      | 2-3 min   | 5-8 hrs   | ⭐⭐⭐⭐   |
| **Hybrid**    | 5-8 min   | 13-20 hrs | ⭐⭐⭐⭐⭐ |

### Optimization Impact

| Optimization               | Speedup  | Quality Impact        |
| -------------------------- | -------- | --------------------- |
| Larger GPU batches (2M→5M) | +40%     | None                  |
| Reduce k_neighbors (30→20) | +33%     | Minimal               |
| Disable stitching          | +20%     | Boundary artifacts    |
| Disable reclassification   | +10%     | Less accurate classes |
| Minimal preprocessing      | +5%      | More outliers         |
| **Total (Fast config)**    | **~50%** | **Acceptable**        |

## Recommended Workflows

### Quick Testing (1-2 tiles)

```bash
ign-lidar-hd process \
  -c examples/config_training_fast_50m.yaml \
  input_dir="data/tiles" \
  output_dir="output/test"
```

### Production (100+ tiles)

```bash
# Use standard config with monitoring
ign-lidar-hd process \
  -c examples/config_training_simple_50m_stitched.yaml \
  input_dir="data/tiles" \
  output_dir="output/production" &

# Monitor in another terminal
./scripts/monitor_processing.sh
```

### Research (best quality)

```bash
ign-lidar-hd process \
  -c examples/config_pointnet_transformer_hybrid_training.yaml \
  input_dir="data/tiles" \
  output_dir="output/research"
```

## Future Optimizations

Potential improvements for v3.1:

1. **FAISS GPU batch parallelism**: Query multiple batches in parallel
2. **Persistent FAISS index**: Cache index between tiles
3. **Distributed processing**: Multi-GPU support
4. **Incremental updates**: Only reprocess changed tiles
5. **Feature caching**: Cache features across runs

## Contact

For performance issues or optimization suggestions:

- GitHub Issues: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues
- Documentation: https://sducournau.github.io/IGN_LIDAR_HD_DATASET/
