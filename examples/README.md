# 📁 Example Configurations - IGN LiDAR HD v3.2.1

**Welcome to the IGN LiDAR HD configuration library!**

All example configurations are now organized into **three clear tiers** for easy discovery:

---

## 🚀 Quick Start (Choose Your Path)

<table>
<tr>
<th width="33%">🎓 Quickstart</th>
<th width="33%">🏭 Production</th>
<th width="33%">⚙️ Advanced</th>
</tr>
<tr>
<td>

**For:**

- First-time users
- Testing & learning
- Minimal setup

**Configs:**

- Minimal (fastest)
- CPU basic
- GPU basic

**[→ Go to Quickstart](quickstart/)**

</td>
<td>

**For:**

- Production deployments
- Complete workflows
- External data sources

**Configs:**

- ASPRS complete
- Multi-scale adaptive

**[→ Go to Production](production/)**

</td>
<td>

**For:**

- Custom scenarios
- Advanced techniques
- Specialized workflows

**Configs:**

- Multi-scale variants
- Custom templates

**[→ Go to Advanced](advanced/)**

</td>
</tr>
</table>

---

## 📊 Configuration Comparison

| Config                        | Location    | Hardware  | Features | Data Sources        | Speed  | Best For                     |
| ----------------------------- | ----------- | --------- | -------- | ------------------- | ------ | ---------------------------- |
| **00_minimal.yaml**           | quickstart/ | CPU       | ~8       | None                | ⚡⚡⚡ | First exploration            |
| **01_cpu_basic.yaml**         | quickstart/ | CPU       | ~12      | None                | ⚡⚡   | Standard CPU use             |
| **02_gpu_basic.yaml**         | quickstart/ | GPU 12GB+ | ~12      | None                | ⚡     | GPU available                |
| **asprs_complete.yaml** ⭐    | production/ | GPU 16GB+ | ~38      | BD TOPO + DTM + RGB | ⚡     | **Production (recommended)** |
| **multi_scale_adaptive.yaml** | production/ | GPU/CPU   | ~12      | None                | ⚡⚡   | Noisy datasets               |
| **multi_scale_3_scales.yaml** | advanced/   | GPU/CPU   | ~12      | None                | ⚡⚡   | Moderate artifacts           |
| **multi_scale_4_scales.yaml** | advanced/   | GPU/CPU   | ~12      | None                | ⚡⚡⚡ | Severe artifacts             |

**Legend:** ⚡ = Very Fast | ⚡⚡ = Fast | ⚡⚡⚡ = Fastest | Features = approximate feature count

---

## 🎯 Which Configuration Should I Use?

### I'm brand new to this library

→ **[quickstart/00_minimal.yaml](quickstart/00_minimal.yaml)** - Get started in minutes!

### I need production-quality results

→ **[production/asprs_complete.yaml](production/asprs_complete.yaml)** ⭐ - Complete workflow with all features

### I have a noisy dataset with artifacts

→ **[production/multi_scale_adaptive.yaml](production/multi_scale_adaptive.yaml)** - Adaptive artifact suppression

### I don't have a GPU

→ **[quickstart/01_cpu_basic.yaml](quickstart/01_cpu_basic.yaml)** - CPU-optimized processing

### I have a GPU and want speed

→ **[quickstart/02_gpu_basic.yaml](quickstart/02_gpu_basic.yaml)** - GPU-accelerated basics
→ **[production/asprs_complete.yaml](production/asprs_complete.yaml)** - GPU-accelerated complete

### I need to customize heavily

→ **[advanced/](advanced/)** - Templates for customization

---

## 📖 Directory Structure

```
examples/
├── README.md                          # This file
│
├── quickstart/                        # 🎓 Get started quickly
│   ├── README.md                      # Quickstart guide
│   ├── 00_minimal.yaml                # Fastest (~5-10 min/tile)
│   ├── 01_cpu_basic.yaml              # CPU standard (~8-12 min/tile)
│   └── 02_gpu_basic.yaml              # GPU standard (~30-60 sec/tile)
│
├── production/                        # 🏭 Production-ready
│   ├── README.md                      # Production guide
│   ├── asprs_complete.yaml ⭐         # Complete ASPRS workflow
│   └── multi_scale_adaptive.yaml      # Adaptive multi-scale
│
├── advanced/                          # ⚙️ Advanced use cases
│   ├── README.md                      # Advanced guide
│   ├── multi_scale_3_scales.yaml      # 3-scale (standard)
│   └── multi_scale_4_scales.yaml      # 4-scale (aggressive)
│
└── [LEGACY CONFIGS]                   # Old configs (to be removed in v3.3.0)
    ├── config_asprs_production_v6.3.yaml
    ├── config_multi_scale_*.yaml
    └── ...
```

**Note:** Legacy configs in the root examples/ directory are deprecated and will be removed in v3.3.0. Please migrate to the organized structure above.

---

## 🚀 Quick Start Examples

### Example 1: First Time User (Minimal)

```bash
# Fastest way to see results
ign-lidar-hd process \
  -c examples/quickstart/00_minimal.yaml \
  input_dir="/path/to/tiles" \
  output_dir="/path/to/output"
```

**Expected:** ~5-10 min per 18M point tile (CPU), 8 essential features

### Example 2: Production Deployment (Complete)

```bash
# Production-quality with all features
ign-lidar-hd process \
  -c examples/production/asprs_complete.yaml \
  input_dir="/path/to/tiles" \
  output_dir="/path/to/output"
```

**Expected:** ~10-15 min per tile (GPU 16GB), 90-95% classification rate, ~38 features

### Example 3: Noisy Dataset (Adaptive)

```bash
# Adaptive multi-scale for artifact suppression
ign-lidar-hd process \
  -c examples/production/multi_scale_adaptive.yaml \
  input_dir="/path/to/tiles" \
  output_dir="/path/to/output"
```

**Expected:** ~8-12 min per tile (GPU), 50-75% artifact reduction

---

## 🔧 Configuration Override Pattern

All configs support command-line overrides:

```bash
# Override input/output paths
ign-lidar-hd process -c CONFIG.yaml \
  input_dir="/data/tiles" \
  output_dir="/data/output"

# Override hardware settings
ign-lidar-hd process -c CONFIG.yaml \
  processor.use_gpu=true \
  processor.gpu_batch_size=30000000 \
  processor.num_workers=8

# Override data sources
ign-lidar-hd process -c CONFIG.yaml \
  data_sources.bd_topo.enabled=false \
  data_sources.dtm.enabled=true

# Override feature settings
ign-lidar-hd process -c CONFIG.yaml \
  features.multi_scale_computation=true \
  features.compute_cluster_id=false
```

---

## 📚 Additional Resources

### In This Directory

- **[demo\_\*.py](.)** - Python demo scripts for specific features
- **[README\_\*.md](.)** - Feature-specific documentation
  - ASPRS optimization guide
  - Gap detection guide
  - Cluster ID guide
  - Rules framework examples

### Main Documentation

- **[Project README](../README.md)** - Main project overview
- **[Documentation Site](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/)** - Complete documentation
- **[Configuration Reference](../docs/docs/configuration/)** - All configuration options
- **[Feature Documentation](../docs/docs/features/)** - Feature computation details
- **[Multi-Scale Guide](../docs/multi_scale_user_guide.md)** - Multi-scale detailed guide

---

## ⚠️ Migration Notice (v3.2.1 → v3.3.0)

**Legacy configurations in the root examples/ directory are deprecated:**

| Legacy Config (⚠️ Deprecated)           | New Config (✅ Use This)               | Migration                   |
| --------------------------------------- | -------------------------------------- | --------------------------- |
| `config_asprs_production_v6.3.yaml`     | `production/asprs_complete.yaml`       | Same content, new location  |
| `config_multi_scale_minimal.yaml`       | `quickstart/01_cpu_basic.yaml`         | Renamed, same functionality |
| `config_multi_scale_standard.yaml`      | `advanced/multi_scale_3_scales.yaml`   | Moved to advanced/          |
| `config_multi_scale_aggressive.yaml`    | `advanced/multi_scale_4_scales.yaml`   | Moved to advanced/          |
| `config_multi_scale_adaptive_v6.3.yaml` | `production/multi_scale_adaptive.yaml` | Moved to production/        |

**Timeline:**

- **v3.2.1** (current): Both old and new configs available
- **v3.3.0** (Q1 2026): Legacy configs removed, only new structure supported

**Action Required:**
Update your scripts to use the new paths. Command-line overrides remain compatible.

```bash
# OLD (⚠️ will break in v3.3.0)
ign-lidar-hd process -c examples/config_asprs_production_v6.3.yaml

# NEW (✅ use this)
ign-lidar-hd process -c examples/production/asprs_complete.yaml
```

### Documentation & Examples

| File                                                       | Content                   |
| ---------------------------------------------------------- | ------------------------- |
| **[README_ASPRS_OPTIMIZED.md](README_ASPRS_OPTIMIZED.md)** | ASPRS optimization guide  |
| **[README_GAP_DETECTION.md](README_GAP_DETECTION.md)**     | Gap detection guide       |
| **[README_CLUSTER_ID.md](README_CLUSTER_ID.md)**           | Cluster ID features guide |
| **[README_RULES_EXAMPLES.md](README_RULES_EXAMPLES.md)**   | Geometric rules examples  |

### Demo Scripts

| File                                                                                     | Demonstrates                        |
| ---------------------------------------------------------------------------------------- | ----------------------------------- |
| **[demo_gap_detection.py](demo_gap_detection.py)**                                       | Gap detection API                   |
| **[demo_wall_detection.py](demo_wall_detection.py)**                                     | Wall detection & adaptive buffering |
| **[demo_adaptive_building_classification.py](demo_adaptive_building_classification.py)** | Adaptive classification             |
| **[demo_adaptive_polygon_buffering.py](demo_adaptive_polygon_buffering.py)**             | Polygon buffering strategies        |
| **[demo_confidence_scoring.py](demo_confidence_scoring.py)**                             | Confidence score computation        |
| **[demo_custom_geometric_rule.py](demo_custom_geometric_rule.py)**                       | Custom rule creation                |
| **[demo_hierarchical_rules.py](demo_hierarchical_rules.py)**                             | Rule priority system                |
| **[demo_mode_selection.py](demo_mode_selection.py)**                                     | Feature mode selection              |
| **[demo_parcel_classification.py](demo_parcel_classification.py)**                       | Cadastre parcel classification      |

---

## 🚀 Quick Start by Use Case

### 1. ⭐ RECOMMENDED: Production ASPRS Processing (v6.3)

```bash
# Complete production pipeline with all modern features
ign-lidar-hd process \
  -c examples/config_asprs_production_v6.3.yaml \
  input_dir="/path/to/tiles" \
  output_dir="/path/to/output"
```

**Features:**

- ✅ DTM augmentation (1m² resolution) for accurate ground reference
- ✅ Multi-scale artifact suppression (3 scales, variance-weighted)
- ✅ Full ASPRS classification (classes 1-17 + extended)
- ✅ BD TOPO integration (buildings, roads, vegetation, water)
- ✅ Cluster IDs (building and parcel identification for object tracking)
- ✅ Spectral features (RGB, NIR, NDVI)
- ✅ GPU optimized (RTX 3090/4080)

**Expected Results:**

- 90-95% point classification rate
- 5-10% artifact rate (vs 20-40% without multi-scale)
- Unique building and parcel cluster IDs for all classified points
- ~10-15 min per 18M point tile (GPU)

**Requirements:**

- GPU: 14GB+ VRAM recommended
- RAM: 32GB+
- Network: For DTM/orthophoto download (first run)

### 2. Testing & Development (Minimal Multi-Scale)

```bash
# Quick testing with 2-scale computation
ign-lidar-hd process \
  -c examples/config_multi_scale_minimal.yaml \
  input_dir="/path/to/tiles" \
  output_dir="/path/to/output"
```

**Use when:** Testing new datasets, limited resources  
**Performance:** ~2x single-scale (faster than production)

### 3. Maximum Artifact Suppression

```bash
# 4-scale computation for severely noisy data
ign-lidar-hd process \
  -c examples/config_multi_scale_aggressive.yaml \
  input_dir="/path/to/tiles" \
  output_dir="/path/to/output"
```

**Use when:** Severe scan line artifacts, noisy sensors  
**Performance:** ~4x single-scale (slower but cleanest)

---

## 📊 Configuration Comparison

### Main Production Configs (v6.3)

| Feature                  | config_asprs_production_v6.3.yaml | config_multi_scale_standard.yaml |
| ------------------------ | --------------------------------- | -------------------------------- |
| **Purpose**              | Complete production pipeline      | Multi-scale only                 |
| **DTM augmentation**     | ✅ 1m² resolution                 | ❌ Not included                  |
| **Multi-scale**          | ✅ 3 scales (fine/medium/coarse)  | ✅ 3 scales                      |
| **BD TOPO**              | ✅ Full integration               | ❌ Not included                  |
| **Cluster IDs**          | ✅ Building + parcel IDs          | ❌ Not included                  |
| **Spectral (RGB/NIR)**   | ✅ Included                       | ❌ Not included                  |
| **GPU optimization**     | ✅ RTX 4080 optimized             | ⚠️ Basic GPU support             |
| **Processing mode**      | patches_only                      | patches_only                     |
| **Classification rate**  | 90-95%                            | 85-90%                           |
| **Artifact suppression** | 50-75% reduction                  | 50-75% reduction                 |
| **Use case**             | **Production (RECOMMENDED)**      | Multi-scale testing              |

**Recommendation:** Use `config_asprs_production_v6.3.yaml` for all production work.

### Multi-Scale Comparison

| Feature                | Minimal (2 scales) | Standard (3 scales) | Aggressive (4 scales) |
| ---------------------- | ------------------ | ------------------- | --------------------- |
| **Artifact reduction** | 30-40% → 10-15%    | 20-40% → 5-10%      | 30-50% → 2-5%         |
| **Performance**        | ~2x slower         | ~3x slower          | ~4x slower            |
| **Memory overhead**    | Low                | Moderate            | High                  |
| **Use case**           | Testing            | Production          | Noisy datasets        |

---

## 🔧 Parameter Tuning Cheat Sheet

### DTM Augmentation (v6.3)

```yaml
# Conservative (minimal synthetic points)
augmentation_strategy: "gaps"               # Only fill obvious gaps
augmentation_spacing: 3.0                   # 3m grid spacing
min_distance_to_existing: 1.0               # 1m from existing ground

# Balanced (recommended)
augmentation_strategy: "intelligent"        # Priority areas
augmentation_spacing: 1.0                   # 1m grid (1m² resolution)
min_distance_to_existing: 0.5               # 0.5m from existing

# Aggressive (maximum coverage)
augmentation_strategy: "full"               # Add points everywhere
augmentation_spacing: 1.0                   # 1m grid
min_distance_to_existing: 0.5               # 0.5m from existing
```

### Multi-Scale Artifact Suppression

```yaml
# Conservative (less aggressive smoothing)
variance_penalty_factor: 1.5                # Mild penalty
artifact_variance_threshold: 0.20           # Higher threshold

# Balanced (recommended)
variance_penalty_factor: 2.0                # Standard penalty
artifact_variance_threshold: 0.15           # Standard threshold

# Aggressive (maximum artifact suppression)
variance_penalty_factor: 3.0                # Strong penalty
artifact_variance_threshold: 0.10           # Lower threshold (more sensitive)
```

### GPU Optimization

```yaml
# 16GB VRAM (RTX 4080, A5000)
gpu_batch_size: 30_000_000                  # 30M points
vram_limit_gb: 14                           # Conservative limit
chunk_size: 5_000_000                       # 5M chunks

# 24GB VRAM (RTX 4090, A6000)
gpu_batch_size: 50_000_000                  # 50M points
vram_limit_gb: 22                           # Conservative limit
chunk_size: 10_000_000                      # 10M chunks

# 12GB VRAM (RTX 3080 Ti, RTX 3090)
gpu_batch_size: 20_000_000                  # 20M points
vram_limit_gb: 10                           # Conservative limit
chunk_size: 4_000_000                       # 4M chunks
```

---

## 📚 Related Documentation

### Root Documentation

- **[multi_scale_user_guide.md](../docs/multi_scale_user_guide.md)** - Multi-scale feature computation guide
- **[rge-alti-integration.md](../docs/docs/guides/rge-alti-integration.md)** - DTM augmentation guide
- **[BUILDING_CLASSIFICATION_ANALYSIS.md](../docs/BUILDING_CLASSIFICATION_ANALYSIS.md)** - Building classification analysis
- **[ROOF_OVERHANG_DETECTION.md](../docs/ROOF_OVERHANG_DETECTION.md)** - Roof overhang detection
- **[FACADE_BASED_DETECTION_AUDIT.md](../docs/FACADE_BASED_DETECTION_AUDIT.md)** - Facade-based detection
- **[MULTI_SCALE_v6.2_PHASE4_COMPLETE.md](../MULTI_SCALE_v6.2_PHASE4_COMPLETE.md)** - Multi-scale implementation
- **[UNCLASSIFIED_POINTS_FIX_v6.3.md](../UNCLASSIFIED_POINTS_FIX_v6.3.md)** - Unclassified points fix guide

### Main Documentation

- [README.md](../README.md) - Main project README
- [docs/](../docs/) - Full documentation site

---

## 🆘 Troubleshooting

### Common Issues (v6.3)

**Problem:** DTM fetch fails (502 Bad Gateway)  
**Solution:** System automatically falls back to RGE ALTI. Check logs for fallback message. Enable cache to avoid re-fetching.

**Problem:** Multi-scale too slow  
**Solution:**

- Use `config_multi_scale_minimal.yaml` (2 scales instead of 3)
- Or disable: `multi_scale_computation: false` in your config
- Expected: 3x slower with 3 scales

**Problem:** GPU out of memory  
**Solution:**

- Reduce `gpu_batch_size` (try 30M → 20M → 15M)
- Increase `chunk_size` (break into smaller chunks)
- Disable multi-scale for first run to identify issue

**Problem:** Still many unclassified points  
**Solution:**

- Check DTM augmentation is working (look for "Augmented with X synthetic ground points" in logs)
- Verify BD TOPO data is available
- Check `height_above_ground` feature is computed
- Try increasing `augmentation_priority` areas

**Problem:** Too many artifacts still visible  
**Solution:**

- Increase `variance_penalty_factor` from 2.0 to 3.0 or 4.0
- Use `config_multi_scale_aggressive.yaml` (4 scales)
- Lower `artifact_variance_threshold` to 0.10

**Problem:** Over-smoothed features (lost detail)  
**Solution:**

- Decrease `variance_penalty_factor` from 2.0 to 1.5
- Use `config_multi_scale_minimal.yaml` (less smoothing)
- Increase weight on fine scale

**See:** [multi_scale_user_guide.md](../docs/multi_scale_user_guide.md) for full troubleshooting

---

## 📝 Version History

- **v6.3.0** (Oct 2025) - DTM augmentation (1m²) + Multi-scale production config ⭐
- **v6.2.0** (Oct 2025) - Multi-scale feature computation (artifact suppression)
- **v6.0.0** (Oct 2025) - Enhanced ASPRS rules, roof overhang detection
- **v5.5.0** (Oct 2025) - Facade-based detection, aggressive buildings
- **v3.3.3** (Oct 2025) - Gap detection feature
- **v5.0.0** (Aug 2025) - Major refactor, Hydra configs

---

## 🤝 Contributing

When adding new configurations:

1. Use descriptive names: `config_{purpose}_v{version}.yaml`
2. Include comprehensive comments (see `config_asprs_production_v6.3.yaml` as template)
3. Document in this index with comparison table
4. Add usage example with expected results
5. Create demo script if introducing new API
6. Update related documentation
7. Follow the project's coding standards (see `.github/copilot-instructions.md`)

---

**Last Updated:** October 25, 2025  
**Version:** 6.3.0  
**Maintainer:** IGN LiDAR HD Development Team
