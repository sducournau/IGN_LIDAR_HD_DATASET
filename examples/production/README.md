# 🏭 Production Configurations

**Production-ready configurations for real-world deployments.**

These configurations are optimized for **production use**, including external data sources (BD TOPO, DTM, orthophotos), advanced features, and complete workflows. They represent **best practices** for high-quality LiDAR processing.

---

## 📋 Quick Selection Guide

| Config                        | Purpose                 | Features     | Data Sources        | Best For              |
| ----------------------------- | ----------------------- | ------------ | ------------------- | --------------------- |
| **asprs_complete.yaml**       | Complete ASPRS workflow | ~38 features | BD TOPO + DTM + RGB | Production deployment |
| **multi_scale_adaptive.yaml** | Adaptive multi-scale    | ~12 features | Optional            | Noisy datasets        |

---

## 🎯 Choose Your Configuration

### Option 1: asprs_complete.yaml (Recommended)

**⭐ This is the flagship production configuration - use this for real deployments!**

**Perfect for:**

- Production ASPRS classification projects
- High-quality building classification (90-95% accuracy)
- Complete feature extraction
- Projects with external data access (internet or local WFS)

**What you get:**

- ✅ **DTM augmentation** (1m² resolution for accurate ground reference)
- ✅ **Multi-scale computation** (3 scales, variance-weighted aggregation)
- ✅ **Full ASPRS classification** (classes 1-17 + extended)
- ✅ **BD TOPO integration** (buildings, roads, vegetation, water)
- ✅ **Cluster IDs** (building and parcel identification for object tracking)
- ✅ **Spectral features** (RGB, NIR, NDVI)
- ✅ **GPU optimized** (RTX 3090/4080/4090)

**Performance:**

- ~10-15 min per 18M point tile (GPU, 16GB VRAM)
- ~30-45 min per tile (CPU, 16 cores)

**Requirements:**

- GPU: 14GB+ VRAM recommended (16GB optimal)
- RAM: 32GB+
- Network: Internet access for BD TOPO/DTM (first run, then cached)
- Or: Local WFS services (BD TOPO) + local DTM tiles

```bash
ign-lidar-hd process \
  -c examples/production/asprs_complete.yaml \
  input_dir="/path/to/tiles" \
  output_dir="/path/to/output"
```

**Expected Results:**

- ✅ 90-95% point classification rate
- ✅ 50-75% artifact reduction (scan line artifacts)
- ✅ Unique building cluster IDs for all building points
- ✅ Unique parcel cluster IDs for all points within cadastre parcels
- ✅ Accurate height_above_ground features (DTM-based)
- ✅ Spectral enrichment (RGB/NIR) for vegetation analysis

**Output Structure:**

```
output/
├── patches/
│   ├── train_0.h5       # Training patches (24k points, ~38 features)
│   ├── train_1.h5
│   └── ...
├── enriched_tiles/
│   ├── Tile_0706_6300_enriched.laz  # LAZ with all features + classifications
│   └── ...
└── metadata/
    ├── feature_stats.json
    └── processing_log.json
```

**Configuration Highlights:**

```yaml
processor:
  lod_level: "ASPRS"
  processing_mode: "both" # Patches + enriched tiles
  use_gpu: true
  gpu_batch_size: 30_000_000 # 30M points (16GB GPU)

data_sources:
  bd_topo:
    enabled: true # Buildings, roads, etc.
    buildings: true
    roads: true
    vegetation: true
    water: true

  dtm:
    enabled: true # 1m² ground reference
    augmentation_strategy: "intelligent"
    augmentation_spacing: 1.0

  orthophoto:
    enabled: true # RGB + NIR spectral data

features:
  mode: "asprs_classes" # Full ASPRS feature set
  multi_scale_computation: true
  compute_cluster_id: true # Building + parcel IDs
  spectral_features: true
```

**Tuning for Your Hardware:**

```bash
# 12GB VRAM (RTX 3080 Ti)
ign-lidar-hd process -c examples/production/asprs_complete.yaml \
  processor.gpu_batch_size=20000000 \
  processor.vram_limit_gb=10 \
  input_dir="/data/tiles"

# 16GB VRAM (RTX 4080, A5000) [OPTIMAL]
ign-lidar-hd process -c examples/production/asprs_complete.yaml \
  processor.gpu_batch_size=30000000 \
  processor.vram_limit_gb=14 \
  input_dir="/data/tiles"

# 24GB VRAM (RTX 4090, A6000)
ign-lidar-hd process -c examples/production/asprs_complete.yaml \
  processor.gpu_batch_size=50000000 \
  processor.vram_limit_gb=22 \
  input_dir="/data/tiles"

# CPU only (no GPU)
ign-lidar-hd process -c examples/production/asprs_complete.yaml \
  processor.use_gpu=false \
  processor.num_workers=8 \
  input_dir="/data/tiles"
```

**Offline Mode (No Internet):**
If you don't have internet access but have local data sources:

```yaml
# Edit asprs_complete.yaml
data_sources:
  bd_topo:
    enabled: true
    wfs_url: "http://localhost:8080/geoserver/wfs" # Local GeoServer

  dtm:
    enabled: true
    local_directory: "/data/dtm_tiles" # Local DTM files

  orthophoto:
    enabled: true
    local_directory: "/data/orthophotos" # Local RGB/NIR tiles
```

Or disable external data entirely:

```bash
ign-lidar-hd process -c examples/production/asprs_complete.yaml \
  data_sources.bd_topo.enabled=false \
  data_sources.dtm.enabled=false \
  data_sources.orthophoto.enabled=false
```

---

### Option 2: multi_scale_adaptive.yaml

**Perfect for:**

- Datasets with severe scan line artifacts
- Noisy sensor data
- Adaptive processing without external data
- Testing multi-scale effectiveness

**What you get:**

- ✅ **Adaptive multi-scale** (automatically adjusts to artifact severity)
- ✅ **Variance-weighted aggregation** (intelligent feature blending)
- ✅ **LOD2 features** (~12 features for building classification)
- ✅ **No external data** (works offline)

**Performance:**

- ~8-12 min per 18M point tile (GPU)
- ~25-35 min per tile (CPU, 8 cores)

**Artifact Reduction:**

- Standard datasets: 20-40% → 5-10%
- Noisy datasets: 30-50% → 10-15%
- Severe artifacts: 40-60% → 15-20%

```bash
ign-lidar-hd process \
  -c examples/production/multi_scale_adaptive.yaml \
  input_dir="/path/to/tiles" \
  output_dir="/path/to/output"
```

**Adaptive Behavior:**
The configuration automatically adjusts based on detected artifact severity:

- **Low artifacts:** Emphasizes fine-scale features (less smoothing)
- **Medium artifacts:** Balanced weighting across scales
- **High artifacts:** Emphasizes coarse-scale features (more smoothing)

**Tuning Artifact Sensitivity:**

```bash
# More aggressive artifact suppression
ign-lidar-hd process -c examples/production/multi_scale_adaptive.yaml \
  features.variance_penalty_factor=3.0 \
  features.artifact_variance_threshold=0.10

# Less aggressive (preserve more detail)
ign-lidar-hd process -c examples/production/multi_scale_adaptive.yaml \
  features.variance_penalty_factor=1.5 \
  features.artifact_variance_threshold=0.20
```

**Output:**

```
output/
├── patches/
│   ├── train_0.h5       # Training patches (16k points, ~12 features)
│   └── ...
└── metadata/
    ├── feature_stats.json
    └── multi_scale_stats.json  # Artifact analysis
```

---

## 📊 Detailed Comparison

### Feature Sets

| Feature Category   | asprs_complete.yaml                           | multi_scale_adaptive.yaml |
| ------------------ | --------------------------------------------- | ------------------------- |
| **Geometric**      | ✅ Full (normals, curvature, planarity, etc.) | ✅ Full                   |
| **Multi-scale**    | ✅ 3 scales (fine/medium/coarse)              | ✅ 3 scales (adaptive)    |
| **Height-based**   | ✅ DTM-augmented (accurate)                   | ⚠️ Z-based only           |
| **Density**        | ✅ Local + neighborhood                       | ✅ Local + neighborhood   |
| **Spectral**       | ✅ RGB + NIR + NDVI                           | ❌ Not included           |
| **Cluster IDs**    | ✅ Building + parcel IDs                      | ❌ Not included           |
| **BD TOPO**        | ✅ Buildings, roads, vegetation, water        | ❌ Not included           |
| **Total Features** | ~38 features                                  | ~12 features              |

### Processing Modes

| Aspect             | asprs_complete.yaml    | multi_scale_adaptive.yaml |
| ------------------ | ---------------------- | ------------------------- |
| **Output**         | Patches + enriched LAZ | Patches only              |
| **Classification** | ASPRS (17+ classes)    | LOD2 (15 classes)         |
| **Data Sources**   | BD TOPO + DTM + RGB    | None (offline)            |
| **Use Case**       | Production deployment  | Artifact-heavy datasets   |

### Performance Comparison (18M point tile)

| Hardware            | asprs_complete.yaml | multi_scale_adaptive.yaml |
| ------------------- | ------------------- | ------------------------- |
| **RTX 4080 (16GB)** | ~10-15 min          | ~8-12 min                 |
| **RTX 3080 (12GB)** | ~12-18 min          | ~10-14 min                |
| **CPU (16 cores)**  | ~30-45 min          | ~25-35 min                |
| **CPU (8 cores)**   | ~45-60 min          | ~35-50 min                |

---

## 🔧 Common Customizations

### 1. Adjust Processing Mode

```bash
# Only patches (no enriched LAZ)
ign-lidar-hd process -c examples/production/asprs_complete.yaml \
  processor.processing_mode="patches_only"

# Only enriched LAZ (no patches)
ign-lidar-hd process -c examples/production/asprs_complete.yaml \
  processor.processing_mode="enriched_only"

# Both (default)
ign-lidar-hd process -c examples/production/asprs_complete.yaml \
  processor.processing_mode="both"
```

### 2. Adjust Patch Parameters

```bash
# Larger patches (for larger receptive field models)
ign-lidar-hd process -c examples/production/asprs_complete.yaml \
  processor.patch_size=100.0 \
  processor.num_points=32768

# Smaller patches (for memory-constrained training)
ign-lidar-hd process -c examples/production/asprs_complete.yaml \
  processor.patch_size=30.0 \
  processor.num_points=8192
```

### 3. Selective BD TOPO Data

```bash
# Only buildings (skip roads, vegetation)
ign-lidar-hd process -c examples/production/asprs_complete.yaml \
  data_sources.bd_topo.roads=false \
  data_sources.bd_topo.vegetation=false \
  data_sources.bd_topo.water=false

# Only roads
ign-lidar-hd process -c examples/production/asprs_complete.yaml \
  data_sources.bd_topo.buildings=false \
  data_sources.bd_topo.vegetation=false
```

### 4. DTM Augmentation Strategy

```bash
# Minimal augmentation (only gaps)
ign-lidar-hd process -c examples/production/asprs_complete.yaml \
  data_sources.dtm.augmentation_strategy="gaps" \
  data_sources.dtm.augmentation_spacing=3.0

# Aggressive augmentation (maximum coverage)
ign-lidar-hd process -c examples/production/asprs_complete.yaml \
  data_sources.dtm.augmentation_strategy="full" \
  data_sources.dtm.augmentation_spacing=1.0
```

### 5. Multi-Scale Tuning

```bash
# More scales (4 scales for severely noisy data)
ign-lidar-hd process -c examples/production/asprs_complete.yaml \
  features.scales=[{name:fine,k_neighbors:20,search_radius:1.0,weight:0.2},{name:medium,k_neighbors:50,search_radius:2.5,weight:0.3},{name:coarse,k_neighbors:100,search_radius:4.0,weight:0.3},{name:very_coarse,k_neighbors:200,search_radius:6.0,weight:0.2}]

# Fewer scales (2 scales for speed)
ign-lidar-hd process -c examples/production/asprs_complete.yaml \
  features.scales=[{name:fine,k_neighbors:30,search_radius:2.0,weight:0.6},{name:coarse,k_neighbors:80,search_radius:4.0,weight:0.4}]
```

---

## 🆘 Troubleshooting

### Common Issues

**Problem:** DTM fetch fails (502 Bad Gateway, timeout)  
**Solution:**

- System automatically falls back to RGE ALTI
- Check logs for "Falling back to RGE ALTI" message
- Enable cache: `data_sources.dtm.cache_enabled=true` (default)
- Or use local DTM: `data_sources.dtm.local_directory="/data/dtm"`

**Problem:** BD TOPO WFS errors  
**Solution:**

- Check network connectivity: `curl https://data.geopf.fr/wfs`
- Use local GeoServer: `data_sources.bd_topo.wfs_url="http://localhost:8080/geoserver/wfs"`
- Or disable: `data_sources.bd_topo.enabled=false`

**Problem:** GPU out of memory  
**Solution:**

```bash
# Reduce batch size and chunk size
ign-lidar-hd process -c examples/production/asprs_complete.yaml \
  processor.gpu_batch_size=15000000 \
  processor.chunk_size=3000000 \
  processor.vram_limit_gb=8
```

**Problem:** Multi-scale too slow  
**Solution:**

- Reduce to 2 scales (see customization #5 above)
- Or disable: `features.multi_scale_computation=false`

**Problem:** Still many unclassified points  
**Solution:**

- Verify DTM augmentation is working (check logs)
- Increase augmentation: `data_sources.dtm.augmentation_strategy="full"`
- Check BD TOPO data is available
- Ensure `height_above_ground` feature is computed

**Problem:** Over-smoothed features (lost detail)  
**Solution:**

```bash
# Reduce smoothing
ign-lidar-hd process -c examples/production/asprs_complete.yaml \
  features.variance_penalty_factor=1.5 \
  features.artifact_variance_threshold=0.20
```

---

## 📚 Related Documentation

- **[Quickstart Configs](../quickstart/README.md)** - Get started quickly
- **[Advanced Configs](../advanced/README.md)** - Custom scenarios
- **[Configuration Reference](../../docs/docs/configuration/)** - All config options
- **[Multi-Scale Guide](../../docs/multi_scale_user_guide.md)** - Multi-scale details
- **[DTM Integration](../../docs/docs/guides/rge-alti-integration.md)** - DTM augmentation

---

## 💡 Best Practices

### 1. Start with a Pilot Study

```bash
# Process 2-3 representative tiles first
ign-lidar-hd process -c examples/production/asprs_complete.yaml \
  input_dir="/data/pilot_tiles" \
  output_dir="/data/pilot_output"
```

### 2. Monitor First Run

- Watch logs for WFS/DTM fetch success
- Check GPU utilization: `watch -n 1 nvidia-smi`
- Monitor memory: `htop` or Task Manager

### 3. Enable Caching

```yaml
# In your config
data_sources:
  bd_topo:
    cache_enabled: true
    cache_ttl: 86400 # 24 hours

  dtm:
    cache_enabled: true

  orthophoto:
    cache_enabled: true
```

### 4. Batch Processing

```bash
# Process entire directory
ign-lidar-hd process -c examples/production/asprs_complete.yaml \
  input_dir="/data/all_tiles" \
  output_dir="/data/production_output"

# Resume interrupted processing
ign-lidar-hd process -c examples/production/asprs_complete.yaml \
  input_dir="/data/all_tiles" \
  output_dir="/data/production_output" \
  --resume
```

### 5. Quality Checks

After processing, validate:

```python
import h5py
import numpy as np

# Check classification coverage
with h5py.File('output/patches/train_0.h5', 'r') as f:
    classifications = f['labels'][:]
    unclassified_pct = (classifications == 0).sum() / len(classifications) * 100
    print(f"Unclassified: {unclassified_pct:.2f}%")  # Should be <10%

# Check feature completeness
with h5py.File('output/patches/train_0.h5', 'r') as f:
    features = f['points'][:]
    nan_pct = np.isnan(features).sum() / features.size * 100
    print(f"NaN values: {nan_pct:.4f}%")  # Should be <0.1%
```

---

**Need help?** Open an issue on [GitHub](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues) or check the [documentation](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/).

**Version:** 3.2.1  
**Last Updated:** October 25, 2025
