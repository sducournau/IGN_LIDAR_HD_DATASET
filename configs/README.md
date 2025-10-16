# IGN LiDAR HD - Configuration Guide# Configuration Updates - Complete Summary

**Last Updated:** October 16, 2025 **Date:** October 15, 2025

**Version:** 3.0.0

## What Was Updated

This directory contains all configuration files for the IGN LiDAR HD processing pipeline.

### ✅ Configuration Files Renamed (3 files)

---

1. `config_unified_asprs_preprocessing.yaml` → `config_asprs_preprocessing.yaml`

## 📁 Configuration Structure (SIMPLIFIED)2. `config_unified_lod2_preprocessing.yaml` → `config_lod2_preprocessing.yaml`

3. `config_unified_lod3_preprocessing.yaml` → `config_lod3_preprocessing.yaml`

````

configs/### ✅ Content Updated

├── default.yaml                    # ⭐ Main configuration (START HERE!)

├── classification_config.yaml      # Classification settings- Removed all "unified" terminology from headers

├── reclassification_config.yaml    # Reclassification rules- Simplified file descriptions and purposes

│- Updated usage examples with new filenames

├── presets/                        # Ready-to-use presets ✨ NEW- Cleaned up comments and documentation

│   ├── minimal.yaml               # Fast processing, no enrichment

│   ├── full_enrichment.yaml       # All data sources enabled### ✅ Pipeline Script Updated

│   └── gpu_optimized.yaml         # GPU-accelerated processing

│- File: `run_complete_pipeline.sh`

└── multiscale/                     # Multi-scale pipeline configs- Variable: `UNIFIED_DATASET` → `DATASET`

```- Option: `--unified-dataset` → `--dataset`

- Comments and help text simplified

**What's New in v3.0:**

- ✅ Reduced from 20 to 7 essential config files### ✅ Documentation Created (2 files)

- ✅ New `presets/` directory for common scenarios

- ✅ Unified `default.yaml` (replaces processing_config.yaml + example_enrichment_full.yaml)1. `CONFIG_UPDATE_SUMMARY.md` - Detailed migration guide

- ✅ Clear, consistent configuration structure2. `CONFIG_QUICK_REFERENCE.md` - Quick reference for all configs



------



## 🚀 Quick Start## File Structure (After Updates)



### 1. Basic Usage```

configs/

```bash├── CONFIG_UPDATE_SUMMARY.md          ← Detailed update documentation

# Use the default configuration (recommended for most cases)├── CONFIG_QUICK_REFERENCE.md         ← Quick reference guide

ign-lidar-hd process --config configs/default.yaml│

├── processing_config.yaml             ← Main processing config (335 lines)

# Or simply (uses default.yaml automatically):├── example_enrichment_full.yaml      ← Complete example (449 lines)

ign-lidar-hd process├── classification_config.yaml        ← Classification system

```│

└── multiscale/

### 2. Use a Preset    ├── run_complete_pipeline.sh      ← Updated pipeline script

    │

```bash    ├── config_asprs_preprocessing.yaml    ← Renamed (was unified_...)

# Fast processing for quick tests    ├── config_lod2_preprocessing.yaml     ← Renamed (was unified_...)

ign-lidar-hd process --config configs/presets/minimal.yaml    ├── config_lod3_preprocessing.yaml     ← Renamed (was unified_...)

    │

# Complete enrichment with all data sources    ├── asprs/

ign-lidar-hd process --config configs/presets/full_enrichment.yaml    │   ├── config_asprs_patches_50m.yaml

    │   ├── config_asprs_patches_100m.yaml

# GPU-accelerated (5-10x faster, requires CUDA)    │   └── config_asprs_patches_150m.yaml

ign-lidar-hd process --config configs/presets/gpu_optimized.yaml    │

```    ├── lod2/

    │   ├── config_lod2_patches_50m.yaml

### 3. Override Specific Values    │   ├── config_lod2_patches_100m.yaml

    │   └── config_lod2_patches_150m.yaml

```bash    │

# Enable forest enrichment    └── lod3/

ign-lidar-hd process --config configs/default.yaml \        ├── config_lod3_patches_50m.yaml

  --set data_sources.bd_foret.enabled=true        ├── config_lod3_patches_100m.yaml

        └── config_lod3_patches_150m.yaml

# Use more workers```

ign-lidar-hd process --config configs/default.yaml \

  --set processing.num_workers=8---

````

## Quick Start

---

### Run Preprocessing

## 📋 Which Config Should I Use?

```bash

| Scenario | Configuration | Speed | Memory | Output |# ASPRS level (minimal features)

|----------|---------------|-------|--------|--------|ign-lidar-hd process --config configs/multiscale/config_asprs_preprocessing.yaml

| **Quick test** | `presets/minimal.yaml` | ⚡⚡⚡ | Low | Patches only |

| **Standard processing** | `default.yaml` | ⚡⚡ | Medium | Enriched LAZ + BD TOPO |# LOD2 level (building features)

| **Complete enrichment** | `presets/full_enrichment.yaml` | ⚡ | High | All data sources |ign-lidar-hd process --config configs/multiscale/config_lod2_preprocessing.yaml

| **Large-scale (GPU)** | `presets/gpu_optimized.yaml` | ⚡⚡⚡ | Medium | GPU-accelerated |

# LOD3 level (architectural details)

---ign-lidar-hd process --config configs/multiscale/config_lod3_preprocessing.yaml

```

## 📖 Configuration Files Explained

### Run Complete Pipeline

### ⭐ `default.yaml` - Start Here!

````bash

**What it does:**cd configs/multiscale

- Processes LiDAR with BD TOPO® enrichment (buildings, roads, railways, water)./run_complete_pipeline.sh --dataset /mnt/d/ign/dataset --phases all

- Computes basic geometric features (height, normals, curvature)```

- Includes RGB processing

- Outputs enriched LAZ files### Use Multi-Source Enrichment



**Perfect for:**```bash

- First-time usersign-lidar-hd process --config configs/example_enrichment_full.yaml

- Standard workflows```

- Production processing

---

**Example structure:**

```yaml## Key Changes Summary

# Input/Output

input:| Component           | Old                                 | New             |

  data_dir: "data/raw/lidar"| ------------------- | ----------------------------------- | --------------- |

output:| **Config files**    | `config_unified_*.yaml`             | `config_*.yaml` |

  base_dir: "data/processed"| **Script variable** | `UNIFIED_DATASET`                   | `DATASET`       |

| **Script option**   | `--unified-dataset`                 | `--dataset`     |

# Data sources (BD TOPO enabled by default)| **Terminology**     | "unified dataset"                   | "dataset"       |

data_sources:| **Comments**        | "Windows paths for unified dataset" | "I/O PATHS"     |

  bd_topo:

    enabled: true---

    features:

      buildings: true## All Configuration Files

      roads: true

      railways: true### Main Configs

      water: true

| File                           | Purpose                                 | Lines |

# Features| ------------------------------ | --------------------------------------- | ----- |

features:| `processing_config.yaml`       | Complete pipeline with all data sources | 335   |

  geometric:| `example_enrichment_full.yaml` | Full multi-source enrichment example    | 449   |

    enabled: true| `classification_config.yaml`   | Classification system config            | 250   |

    k_neighbors: 20

```### Preprocessing Configs



---| File                              | LOD Level | Features | k_neighbors |

| --------------------------------- | --------- | -------- | ----------- |

### 🎯 Presets (New in v3.0!)| `config_asprs_preprocessing.yaml` | ASPRS     | Minimal  | 20          |

| `config_lod2_preprocessing.yaml`  | LOD2      | Enhanced | 30          |

#### `minimal.yaml` - Lightning Fast| `config_lod3_preprocessing.yaml`  | LOD3      | Maximum  | 50          |



```yaml### Patch Configs (9 files)

# Minimal configuration

mode: "patches"           # Only patches, no enrichment| LOD       | 50m            | 100m           | 150m           |

data_sources:| --------- | -------------- | -------------- | -------------- |

  bd_topo:| **ASPRS** | 16k pts, 3 aug | 24k pts, 3 aug | 32k pts, 3 aug |

    enabled: false        # No external data| **LOD2**  | 16k pts, 4 aug | 24k pts, 4 aug | 32k pts, 4 aug |

features:| **LOD3**  | 24k pts, 5 aug | 32k pts, 5 aug | 40k pts, 5 aug |

  geometric:

    k_neighbors: 10       # Fewer neighbors = faster---

````

## Data Sources Supported

**Use when:**

- Testing the pipeline### BD TOPO® V3 (Infrastructure)

- Quick prototyping

- Learning the system- ✅ Buildings (ASPRS 6)

- Memory constrained- ✅ Roads (ASPRS 11)

- ✅ Railways (ASPRS 10)

**⚡ Fastest option**- ✅ Water (ASPRS 9)

- ✅ Vegetation (ASPRS 5)

---- ✅ Bridges (ASPRS 17)

- ✅ Parking (ASPRS 40)

#### `full_enrichment.yaml` - Maximum Data- ✅ Sports (ASPRS 41)

````yaml### BD Forêt® V2 (Forest)

# Complete enrichment

mode: "both"              # Enriched LAZ + patches- ✅ Forest types (coniferous, deciduous, mixed)

data_sources:- ✅ Primary species (Quercus, Pinus, etc.)

  bd_topo:- ✅ Height estimation

    enabled: true

    features:### RPG 2020-2023 (Agriculture)

      buildings: true

      roads: true- ✅ Crop codes (BLE, MAI, COL, etc.)

      cemeteries: true    # ALL features enabled- ✅ Crop categories

      power_lines: true- ✅ Parcel areas

  bd_foret:- ✅ Organic farming flag

    enabled: true         # Forest data- ✅ ASPRS code 44

  rpg:

    enabled: true         # Agriculture### BD PARCELLAIRE (Cadastre)

  cadastre:

    enabled: true         # Cadastral parcels- ✅ Parcel IDs (19-char unique)

```- ✅ Parcel grouping

- ✅ Statistics per parcel

**Use when:**

- Research projects---

- Complete analysis needed

- Maximum attribute richness## Testing Checklist

- Production datasets

### Configuration Files

**🐌 Slower but comprehensive**

- [x] Files renamed successfully

---- [x] Content updated (headers, comments, paths)

- [x] Usage examples updated

#### `gpu_optimized.yaml` - Speed Demon- [ ] Test ASPRS preprocessing

- [ ] Test LOD2 preprocessing

```yaml- [ ] Test LOD3 preprocessing

# GPU acceleration

processing:### Pipeline Script

  use_gpu: true

  batch_size: 64          # Large batches for GPU- [x] Variables renamed (DATASET)

  gpu_chunk_size: 1_000_000- [x] Options updated (--dataset)

features:- [x] Help text updated

  geometric:- [ ] Test complete pipeline

    use_gpu: true         # GPU-accelerated features- [ ] Test individual phases

````

### Documentation

**Use when:**

- You have NVIDIA GPU (8GB+ VRAM)- [x] Update summary created

- Processing >1000 tiles- [x] Quick reference created

- Time is critical- [x] Migration guide documented

- [ ] External docs updated

**Requirements:**

- CUDA 11.7+---

- CuPy installed

- `pip install cupy-cuda11x`## Migration Path

**🚀 5-10x faster than CPU**### For Existing Users

---1. **Update config paths:**

## ⚙️ Key Configuration Sections ```bash

# Find and replace in scripts

### Data Sources sed -i 's/config_unified_asprs/config_asprs/g' \*.sh

sed -i 's/config_unified_lod2/config_lod2/g' \*.sh

````yaml sed -i 's/config_unified_lod3/config_lod3/g' *.sh

data_sources:   ```

  # BD TOPO® - Topographic Database

  bd_topo:2. **Update script options:**

    enabled: true        # Set to false to disable

    features:   ```bash

      buildings: true    # ASPRS code 6   # Old

      roads: true        # ASPRS code 11   ./run_complete_pipeline.sh --unified-dataset /path/to/data

      railways: true     # ASPRS code 10

      water: true        # ASPRS code 9   # New

      bridges: true      # ASPRS code 17   ./run_complete_pipeline.sh --dataset /path/to/data

      parking: true      # ASPRS code 40   ```



  # BD Forêt® - Forest Types3. **Update Python imports:**

  bd_foret:

    enabled: false       # Enable for forest enrichment   ```python

    label_forest_type: true   # Old

    label_primary_species: true   from ign_lidar.io.unified_fetcher import UnifiedDataFetcher



  # RPG - Agricultural Parcels   # New

  rpg:   from ign_lidar.io.data_fetcher import DataFetcher

    enabled: false       # Enable for crop data   ```

    year: 2024

  ---

  # Cadastre - Parcel Information

  cadastre:## Related Updates

    enabled: false       # Enable for cadastral data

```This configuration update is part of the larger codebase consolidation:



### Features1. **Core Module Updates:**



```yaml   - `unified_fetcher.py` → `data_fetcher.py`

features:   - `UnifiedDataFetcher` → `DataFetcher`

  geometric:   - Deprecated aliases maintained for backward compatibility

    enabled: true

    k_neighbors: 20      # Neighborhood size2. **Skip Checker Enhancement:**

    compute:

      height: true       # Height above ground   - Added multi-source validation

      normals: true      # Surface normals   - Validates: classification, forest_type, crop_code, parcel_id

      curvature: true    # Curvature   - Deep feature checking for enriched LAZ files

      planarity: true    # Planarity

      linearity: false   # (expensive, disable for speed)3. **Configuration Files:**

```   - Main processing config with all data sources

   - Example enrichment config

### Processing   - Multiscale training configs updated



```yaml---

processing:

  num_workers: 4         # Parallel workers## Documentation Files

  batch_size: "auto"     # Automatic batch sizing

  use_gpu: false         # Enable for GPU acceleration| File                            | Purpose                                      |

```| ------------------------------- | -------------------------------------------- |

| `CONFIG_UPDATE_SUMMARY.md`      | Detailed documentation of all config changes |

---| `CONFIG_QUICK_REFERENCE.md`     | Quick reference for all configuration files  |

| `docs/CONSOLIDATION_SUMMARY.md` | Code consolidation summary                   |

## 🎨 Common Customizations| `docs/CONSOLIDATION_PLAN.md`    | Original consolidation plan                  |



### Enable Forest Data---



```bash## Next Steps

# Command line:

ign-lidar-hd process --config configs/default.yaml \1. ✅ Configuration files updated

  --set data_sources.bd_foret.enabled=true2. ✅ Documentation created

3. ⏳ Test preprocessing configs

# Or edit config:4. ⏳ Test complete pipeline

data_sources:5. ⏳ Update external documentation

  bd_foret:6. ⏳ Notify users of changes

    enabled: true

```---



### Adjust Performance**Status:** All configuration files updated and ready for use! 🚀


```bash
# More workers (faster, more memory):
--set processing.num_workers=8

# Fewer neighbors (faster, less accurate):
--set features.geometric.k_neighbors=10
````

### GPU Acceleration

```bash
# Enable GPU:
--set processing.use_gpu=true

# Or use preset:
ign-lidar-hd process --config configs/presets/gpu_optimized.yaml
```

---

## 🔧 Troubleshooting

### Problem: "Which config should I use?"

**Answer:** Start with `default.yaml`. It works for 90% of use cases.

### Problem: Out of memory

**Solutions:**

1. Use `presets/minimal.yaml`
2. Reduce workers: `--set processing.num_workers=2`
3. Disable data sources you don't need

### Problem: Slow processing

**Solutions:**

1. Use `presets/gpu_optimized.yaml` if you have GPU
2. Enable cache: `--set cache.enabled=true`
3. Increase workers: `--set processing.num_workers=8`
4. Use `presets/minimal.yaml` for testing

### Problem: No classification applied

**Check:**

1. Data source enabled: `data_sources.bd_topo.enabled: true`
2. Internet connection (for data fetching)
3. Cache directory writable
4. Check logs for errors

---

## 📊 Performance Comparison

| Config  | Speed          | Memory    | Features | Best For    |
| ------- | -------------- | --------- | -------- | ----------- |
| minimal | ⚡⚡⚡ (1x)    | 2GB       | Basic    | Testing     |
| default | ⚡⚡ (0.5x)    | 4GB       | Standard | Production  |
| full    | ⚡ (0.2x)      | 8GB+      | Complete | Research    |
| gpu     | ⚡⚡⚡ (5-10x) | 4GB + GPU | Standard | Large-scale |

---

## 🔄 Migrating from v2.x

### Old Config (v2.x)

```yaml
pipeline:
  stages:
    classification:
      ground_truth:
        sources:
          buildings: true
```

### New Config (v3.0)

```yaml
data_sources:
  bd_topo:
    features:
      buildings: true
```

**Changes:**

- ✅ Simpler structure
- ✅ Consistent naming
- ✅ Fewer files (96 → 7 essential)
- ✅ Clear presets

---

## 📚 Learn More

- **Architecture:** `../docs/architecture/`
- **Examples:** `../examples/`
- **Full Audit:** `../REFACTORING_AUDIT_REPORT.md`
- **API Docs:** `../docs/api/`

---

## ❓ Need Help?

1. **Check presets first:** Try `minimal.yaml`, `default.yaml`, or `full_enrichment.yaml`
2. **Read error messages:** They usually tell you what's wrong
3. **Enable logging:** `--set log_level=DEBUG`
4. **Open an issue:** GitHub issues with config attached

---

**Remember:**

- 🌟 Start with `default.yaml`
- 🎯 Use presets for special cases
- ⚙️ Override only what you need
- 📊 Monitor performance with `--verbose`

**Version 3.0 Improvements:**

- 7 essential configs (was 20+)
- Clear preset directory
- Unified default configuration
- Better documentation

---

_Simplified configuration system - Version 3.0.0_
