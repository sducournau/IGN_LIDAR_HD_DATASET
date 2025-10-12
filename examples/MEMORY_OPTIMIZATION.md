# Memory Optimization Guide for LOD3 Training Dataset Generation

## Problem: Out of Memory (OOM) Errors

When processing large LiDAR tiles with parallel workers, you may encounter OOM errors (Exit Code 137). This happens when:

1. **Multiple large tiles** are processed simultaneously (4 workers × 20-70M points each)
2. **Tile stitching** loads neighbor tiles into memory (buffer zones)
3. **Feature computation** requires k-NN calculations on millions of points
4. **GPU fallback** when CUDA is unavailable increases CPU memory usage

## Available Configurations

### 1. Original Config (Fastest, High Memory)

**File**: `config_lod3_training.yaml`

- **Workers**: 4 parallel
- **Batch Size**: 4
- **GPU Batch**: 1M points
- **Memory Required**: ~32GB+ RAM
- **Best for**: Systems with ≥32GB RAM and CUDA GPU

```bash
ign-lidar-hd process --config-file examples/config_lod3_training.yaml
```

---

### 2. Memory-Optimized Config (Balanced)

**File**: `config_lod3_training_memory_optimized.yaml`

**Key Changes**:

- ✅ **Workers**: 2 (reduced from 4)
- ✅ **Batch Size**: 2 (reduced from 4)
- ✅ **GPU Batch**: 500K points (reduced from 1M)
- ✅ **Prefetch Factor**: 1 (reduced from 2)
- ✅ **Pin Memory**: Disabled
- ✅ **Cache Size**: 500 (reduced from 1000)
- ✅ **Memory Limit**: 4GB per tile (reduced from 8GB)
- ✅ **GC Frequency**: Every 5 tiles (more aggressive)
- ✅ **Auto-download neighbors**: Disabled

**Memory Required**: ~16-20GB RAM  
**Speed**: 40-50% of original  
**Best for**: Systems with 16-24GB RAM

```bash
ign-lidar-hd process --config-file examples/config_lod3_training_memory_optimized.yaml
```

---

### 3. Sequential Config (Slowest, Minimal Memory)

**File**: `config_lod3_training_sequential.yaml`

**Key Changes**:

- ✅ **Workers**: 1 (sequential processing)
- ✅ **Batch Size**: 1
- ✅ **GPU Batch**: 250K points
- ✅ **Prefetch Factor**: 0 (disabled)
- ✅ **Buffer Size**: 5m (reduced from 10m)
- ✅ **Max Neighbors**: 4 (reduced from 8)
- ✅ **Cache Size**: 100 (minimal)
- ✅ **Memory Limit**: 2GB per tile
- ✅ **GC Frequency**: Every tile (most aggressive)

**Memory Required**: ~8-12GB RAM  
**Speed**: 20-25% of original  
**Best for**: Systems with ≤16GB RAM or resource-constrained environments

```bash
ign-lidar-hd process --config-file examples/config_lod3_training_sequential.yaml
```

---

## Monitoring Memory Usage

### Check available memory before processing:

```bash
free -h
```

### Monitor memory during processing:

```bash
# In another terminal
watch -n 1 free -h
```

### Check for OOM kills in system logs:

```bash
dmesg | grep -i "killed process"
```

---

## Troubleshooting CUDA Warnings

If you see: `CUDA driver initialization failed, you might not have a CUDA gpu`

**This means**: GPU acceleration is unavailable, processing falls back to CPU

**Solutions**:

1. **Check CUDA installation**:

   ```bash
   nvidia-smi
   ```

2. **If no GPU**: Use CPU-only mode by setting `use_gpu: false` in config

3. **If GPU exists but not detected**: Reinstall CUDA toolkit and PyTorch:
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

---

## Performance Comparison

| Config           | Workers | Memory  | Speed  | Time (8 tiles) |
| ---------------- | ------- | ------- | ------ | -------------- |
| Original         | 4       | 32GB+   | 100%   | ~2-3 hours     |
| Memory-Optimized | 2       | 16-20GB | 40-50% | ~5-6 hours     |
| Sequential       | 1       | 8-12GB  | 20-25% | ~10-12 hours   |

---

## Recommendations

### For 32GB+ RAM systems:

```bash
# Use original config for fastest processing
ign-lidar-hd process --config-file examples/config_lod3_training.yaml
```

### For 16-24GB RAM systems:

```bash
# Use memory-optimized config
ign-lidar-hd process --config-file examples/config_lod3_training_memory_optimized.yaml
```

### For 8-16GB RAM systems:

```bash
# Use sequential config
ign-lidar-hd process --config-file examples/config_lod3_training_sequential.yaml
```

### For systems with swap/slow disks:

Consider processing in smaller batches by using bbox filtering:

```bash
# Process specific region only
ign-lidar-hd process --config-file examples/config_lod3_training_sequential.yaml \
  --bbox 326000 6829000 327000 6830000
```

---

## Additional Memory-Saving Tips

1. **Close other applications** before processing
2. **Increase swap space** (not recommended, very slow):

   ```bash
   sudo fallocate -l 16G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

3. **Process tiles individually** using bbox filtering
4. **Use compressed output** to save disk space:

   ```yaml
   output:
     format: npz
     compression: gzip # Add this line
   ```

5. **Disable augmentation** temporarily to save memory:
   ```yaml
   processor:
     augment: false # Disable augmentation
   ```

---

## System Requirements

### Minimum (Sequential Config):

- RAM: 8GB
- CPU: 4 cores
- Disk: 100GB free
- GPU: Optional (CPU fallback available)

### Recommended (Memory-Optimized Config):

- RAM: 16-24GB
- CPU: 8 cores
- Disk: 200GB free
- GPU: NVIDIA with 6GB+ VRAM

### Optimal (Original Config):

- RAM: 32GB+
- CPU: 16+ cores
- Disk: 500GB+ SSD
- GPU: NVIDIA with 8GB+ VRAM (RTX 3060 or better)

---

## Questions?

If you continue to experience OOM errors even with the sequential config, consider:

1. Processing fewer tiles at a time (use bbox filtering)
2. Reducing `num_points` from 32768 to 16384
3. Disabling tile stitching (`stitching.enabled: false`)
4. Using a machine with more RAM (cloud VM with 64GB+ RAM)
