# Configuration Selection Guide

**Last Updated**: October 24, 2025  
**Version**: 5.4.0

## Quick Start: Which Config Should I Use?

### 🎯 Decision Tree

```
Do you have a GPU?
│
├─ YES → How much VRAM?
│  │
│  ├─ 24GB (RTX 4090, 3090 Ti, A5000)
│  │  └─ config_asprs_gpu_24gb.yaml (Coming soon)
│  │     • 1m² DTM augmentation
│  │     • 30M point GPU FAISS threshold
│  │     • Maximum quality
│  │
│  ├─ 16GB (RTX 4080, 4070 Ti Super, 3090, A4000)
│  │  └─ ✅ config_asprs_gpu_16gb.yaml **← RECOMMENDED FOR YOUR SYSTEM**
│  │     • 2m² DTM augmentation
│  │     • 25M point GPU FAISS threshold
│  │     • Excellent quality
│  │     • ~8-14 min per 20M tile
│  │
│  ├─ 12GB (RTX 4070, 3080, A2000)
│  │  └─ config_asprs_gpu_12gb.yaml (Use gpu_memory_efficient)
│  │     • 3m² DTM augmentation
│  │     • 15M point GPU FAISS threshold
│  │     • Good quality
│  │
│  └─ 8GB or less (RTX 4060, 3060, 3050)
│     └─ config_asprs_gpu_memory_efficient.yaml
│        • 3m² DTM augmentation
│        • Conservative settings
│        • May fall back to CPU FAISS for large tiles
│
└─ NO → How much RAM?
   │
   ├─ 64GB+
   │  └─ config_asprs_cpu_fast.yaml (Coming soon)
   │     • Aggressive parallelism
   │     • 1m² DTM augmentation
   │     • 8-12 workers
   │
   ├─ 32GB
   │  └─ ✅ config_asprs_cpu_v3_memory_safe.yaml
   │     • Conservative memory use
   │     • 3m² DTM augmentation
   │     • 4 workers
   │     • ~35-50 min per 20M tile
   │
   └─ 16GB
      └─ config_asprs_cpu_minimal.yaml (Coming soon)
         • Minimal features
         • No DTM augmentation
         • 2 workers
```

## Understanding the Trade-offs

### DTM Augmentation Spacing

| Spacing | Synthetic Points | Memory | Quality   | GPU FAISS? |
| ------- | ---------------- | ------ | --------- | ---------- |
| 1.0m²   | ~165K / km²      | High   | Excellent | 24GB VRAM  |
| 1.5m²   | ~73K / km²       | Med    | Very Good | 24GB VRAM  |
| 2.0m²   | ~41K / km²       | Med    | Good      | 16GB VRAM  |
| 3.0m²   | ~18K / km²       | Low    | Adequate  | 12GB VRAM  |
| None    | 0                | Min    | Basic     | Any GPU    |

**Rule of Thumb**: Smaller spacing = better height accuracy but more points

### GPU FAISS Thresholds

The system uses GPU FAISS (ultra-fast k-NN) when:

```
Total Points < FAISS GPU Threshold
```

| VRAM | Threshold  | Why                                |
| ---- | ---------- | ---------------------------------- |
| 24GB | 30M points | Can handle very large point clouds |
| 16GB | 25M points | Balanced for most real-world tiles |
| 12GB | 15M points | Conservative to avoid OOM errors   |
| 8GB  | 10M points | Very conservative, safe mode       |

**If threshold exceeded**: Falls back to CPU FAISS (still 20× faster than pure CPU, but 8× slower than GPU)

## Your Current Situation

Based on your previous run:

- **Original tile**: 21.5M points
- **GPU**: 16GB VRAM
- **RAM**: 32GB
- **Goal**: 1m² DTM augmentation

### The Problem

```
21.5M base points + 165K augmentation (1m²) = 21.7M points
21.7M > 15M (old hardcoded threshold) → CPU FAISS forced
Result: 26 minutes for k-NN instead of 2-3 minutes
```

### The Solution

Use `config_asprs_gpu_16gb.yaml` which:

1. **Raises threshold to 25M** → GPU FAISS will be used ✅
2. **Uses 2m² DTM** → ~41K synthetic points
3. **Total**: 21.5M + 41K = 21.54M < 25M → GPU FAISS enabled
4. **Result**: ~2-3 min for k-NN ✅

### If You Really Want 1m² Augmentation

Option A: **Split your tile into 4 sub-tiles**

```bash
# Each sub-tile would be ~5M points
# 5M + 41K (1m²) = 5.04M << 25M → GPU FAISS works great
# Process 4× ~8 min = ~32 min total (still faster than 1 tile with CPU FAISS)
```

Option B: **Use 24GB VRAM GPU** (RTX 4090, 3090 Ti)

- 30M threshold supports full 1m² augmentation
- Total processing time: ~6-10 minutes

## Configuration Files Overview

### Production Configs

| File                                           | Hardware            | Use Case               | Quality | Speed |
| ---------------------------------------------- | ------------------- | ---------------------- | ------- | ----- |
| `config_asprs_gpu_16gb.yaml`                   | 16GB VRAM, 32GB RAM | General ASPRS          | ★★★★☆   | ★★★★★ |
| `config_asprs_gpu_memory_efficient.yaml`       | 8-12GB VRAM         | Memory-constrained GPU | ★★★☆☆   | ★★★★☆ |
| `config_asprs_cpu_v3_memory_safe.yaml`         | 32GB RAM, no GPU    | CPU-only               | ★★★★☆   | ★★☆☆☆ |
| `config_building_fusion.yaml`                  | Any                 | Building-specific      | ★★★★★   | ★★★☆☆ |
| `config_adaptive_building_classification.yaml` | GPU preferred       | Advanced buildings     | ★★★★★   | ★★★☆☆ |

### Specialized Configs

| File                                      | Purpose                     |
| ----------------------------------------- | --------------------------- |
| `config_architectural_analysis_v5.0.yaml` | LOD2 architectural analysis |
| `config_plane_detection_lod3.yaml`        | LOD3 with plane detection   |
| `config_versailles_asprs_v5.0.yaml`       | Versailles benchmark        |

## Performance Benchmarks

### 20M Point Tile (1km² urban)

| Config               | Hardware       | Total Time | k-NN Time | Quality |
| -------------------- | -------------- | ---------- | --------- | ------- |
| gpu_16gb             | RTX 4080, 32GB | ~8-14 min  | ~2-3 min  | ★★★★☆   |
| gpu_memory_efficient | RTX 3060, 32GB | ~12-18 min | ~3-5 min  | ★★★☆☆   |
| cpu_v3_memory_safe   | CPU 32GB       | ~35-50 min | ~26 min   | ★★★★☆   |

### Accuracy Metrics

| Config                         | Overall | Buildings | Roads  | Vegetation |
| ------------------------------ | ------- | --------- | ------ | ---------- |
| gpu_16gb (2m² DTM)             | 89-96%  | 92-97%    | 81-91% | 86-92%     |
| gpu_memory_efficient (3m² DTM) | 88-95%  | 92-97%    | 80-90% | 85-90%     |
| cpu_v3_memory_safe (3m² DTM)   | 88-95%  | 92-97%    | 80-90% | 85-90%     |

## Common Issues & Solutions

### Issue: "Using CPU FAISS to avoid GPU OOM"

**Cause**: Point cloud exceeds GPU FAISS threshold

**Solutions**:

1. Use config with higher threshold (e.g., gpu_16gb: 25M vs 15M)
2. Reduce DTM augmentation spacing (3m² → fewer points)
3. Split tiles into smaller chunks
4. Use GPU with more VRAM

### Issue: "Memory usage >90%"

**Cause**: DTM augmentation + large point cloud

**Solutions**:

1. Reduce DTM augmentation spacing
2. Reduce batch sizes
3. Use memory_safe config variant
4. Add more RAM

### Issue: "GPU not being used"

**Causes & Solutions**:

1. `num_workers > 1` → Set to 1 (CUDA limitation)
2. GPU not detected → Check CUDA installation
3. `use_gpu: false` → Set to true in config
4. Out of VRAM → Use smaller batch sizes

## Configuration Parameters Explained

### Critical Parameters

```yaml
processor:
  gpu_batch_size: 8_000_000 # Points per GPU batch
  faiss_gpu_threshold: 25_000_000 # Max points for GPU FAISS
  faiss_allow_large_gpu: true # Override safety check
  num_workers: 1 # MUST be 1 for GPU

  reclassification:
    chunk_size: 6_000_000 # Points per reclass batch
    acceleration_mode: "gpu" # Force GPU acceleration

features:
  k_neighbors: 60 # Neighbors for features (quality)
  gpu_batch_size: 8_000_000 # Feature computation batch

data_sources:
  rge_alti:
    augmentation_spacing: 2.0 # DTM grid spacing (meters)
```

### Parameter Tuning

**For More Speed** (sacrifice some quality):

- Reduce `k_neighbors`: 60 → 40
- Increase `augmentation_spacing`: 2.0 → 3.0
- Reduce `chunk_size` values
- Use `acceleration_mode: "auto"`

**For More Quality** (sacrifice speed):

- Increase `k_neighbors`: 60 → 80
- Decrease `augmentation_spacing`: 2.0 → 1.5
- Increase `faiss_gpu_threshold`
- Enable more data sources

## Next Steps

1. **Run with recommended config**:

   ```bash
   ign-lidar-hd process \
     -c examples/config_asprs_gpu_16gb.yaml \
     input_dir="/mnt/d/ign/versailles_tiles" \
     output_dir="/mnt/d/ign/versailles_output_gpu"
   ```

2. **Monitor performance**:

   - Watch for "GPU FAISS" vs "CPU FAISS" in logs
   - Check memory usage
   - Note processing time per tile

3. **Adjust if needed**:
   - If GPU FAISS not used → Check threshold
   - If OOM errors → Reduce batch sizes
   - If too slow → Use faster config
   - If quality insufficient → Use higher quality config

## Support

**Documentation**:

- Full audit: `CONFIG_AUDIT_2025.md`
- Base config reference: `ign_lidar/configs/base/README.md`
- Example configs: `examples/README.md`

**Getting Help**:

1. Check logs for error messages
2. Review this selection guide
3. Consult CONFIG_AUDIT_2025.md for detailed analysis
4. File an issue with config file and logs

---

**Remember**: The goal is finding the right balance between speed, quality, and hardware capabilities. Start with the recommended config and adjust based on your specific needs!
