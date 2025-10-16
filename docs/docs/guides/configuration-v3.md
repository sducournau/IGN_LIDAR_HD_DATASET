---
sidebar_position: 7
title: Configuration System v3.0
---

# Configuration System v3.0

Simplified, streamlined configuration for IGN LiDAR HD processing.

---

## 🎯 What's New in v3.0

The v3.0 configuration system has been completely redesigned for simplicity and clarity:

### Key Improvements

✅ **Flatter Structure** - Reduced from 3-4 nesting levels to 2 levels  
✅ **Feature Modes** - Easy feature set selection (`asprs_classes`, `minimal`, `lod2`, `lod3`, `full`)  
✅ **45% Fewer Parameters** - Removed rarely-used options  
✅ **Clearer Naming** - `bd_topo_enabled` instead of nested `bd_topo.enabled`  
✅ **Better Defaults** - ASPRS classification optimized out of the box  
✅ **Preset Configs** - Ready-to-use configurations for common scenarios

---

## 📊 Feature Modes

The biggest change in v3.0 is the introduction of **feature modes** - predefined feature sets optimized for different use cases.

### Available Modes

| Mode                   | Features | File Size  | Speed     | Use Case                       |
| ---------------------- | -------- | ---------- | --------- | ------------------------------ |
| **`asprs_classes`** ⭐ | ~15      | Small      | Fast      | ASPRS classification (default) |
| `minimal`              | ~8       | Tiny       | Very Fast | Quick testing                  |
| `lod2`                 | ~17      | Medium     | Medium    | Building detection             |
| `lod3`                 | ~43      | Large      | Slow      | Architectural modeling         |
| `full`                 | ~45      | Very Large | Slowest   | Research, all features         |

### ASPRS Classes Mode (Default)

Optimized for ASPRS LAS 1.4 classification with the best balance of accuracy and performance.

**Features included (~15 total):**

- ✅ XYZ coordinates (3)
- ✅ Normal Z (verticality indicator)
- ✅ Planarity (flat surfaces)
- ✅ Sphericity (vegetation)
- ✅ Height above ground
- ✅ Verticality & horizontality
- ✅ Point density
- ✅ RGB + NIR + NDVI (5)

**Benefits:**

- 🎯 **60-70% smaller files** vs full mode
- ⚡ **2.8x faster** feature computation
- 📊 **Optimized for accuracy** in ASPRS classification
- 💾 **Lower memory usage**

**Configuration:**

```yaml
features:
  mode: "asprs_classes"
  k_neighbors: 20
  use_rgb: true
  use_nir: true
  compute_ndvi: true
```

### Minimal Mode

Ultra-fast processing with basic features only.

**Features included (~8 total):**

- XYZ coordinates
- Normal Z
- Planarity
- Height above ground

**Best for:** Quick testing, prototyping, initial exploration

```yaml
features:
  mode: "minimal"
  k_neighbors: 10 # Fewer neighbors = faster
  use_rgb: false
  use_nir: false
  compute_ndvi: false
```

### LOD2 Mode

Essential features for Level of Detail 2 building detection.

**Features included (~17 total):**

- Core geometric features (normals, curvature)
- Shape descriptors (planarity, linearity, sphericity)
- Height features
- Building detection features

**Best for:** Building footprint extraction, urban analysis

```yaml
features:
  mode: "lod2"
  k_neighbors: 20
  use_rgb: true
  use_nir: false
  compute_ndvi: false
```

### LOD3 Mode

Complete feature set for Level of Detail 3 architectural modeling.

**Features included (~43 total):**

- All geometric features
- Full shape descriptors
- Eigenvalue features
- Architectural features (walls, roofs, facades)
- Density and neighborhood features
- Spectral features

**Best for:** Detailed 3D building reconstruction, architectural analysis

```yaml
features:
  mode: "lod3"
  k_neighbors: 20
  use_rgb: true
  use_nir: true
  compute_ndvi: true
  include_extra: true
```

### Full Mode

All available features for research and development.

**Best for:** Research, feature importance studies, maximum accuracy

```yaml
features:
  mode: "full"
  k_neighbors: 20
  use_rgb: true
  use_nir: true
  compute_ndvi: true
  include_extra: true
```

---

## 📁 Configuration Structure

### Simplified v3.0 Structure

```yaml
# ============================================================================
# Input/Output (Required)
# ============================================================================

input_dir: "data/raw/lidar"
output_dir: "data/processed"

# ============================================================================
# Processing Configuration
# ============================================================================

processing:
  lod_level: "ASPRS" # ASPRS, LOD2, LOD3
  mode: "patches_only" # patches_only, both, enriched_only
  architecture: "pointnet++" # Neural network architecture
  num_workers: 4
  use_gpu: false
  patch_size: 150.0
  patch_overlap: 0.1
  num_points: 16384
  augment: false
  num_augmentations: 3

# ============================================================================
# Feature Computation
# ============================================================================

features:
  mode: "asprs_classes" # Feature set selection ⭐ NEW
  k_neighbors: 20
  search_radius: null
  use_rgb: true
  use_nir: true
  compute_ndvi: true
  include_extra: false
  gpu_batch_size: null # Auto-calculated
  use_gpu_chunked: true

# ============================================================================
# Data Source Enrichment (Optional)
# ============================================================================

data_sources:
  # BD TOPO
  bd_topo_enabled: false
  bd_topo_buildings: true
  bd_topo_roads: true
  bd_topo_water: true
  bd_topo_vegetation: true
  bd_topo_cache_dir: "cache/bd_topo"

  # BD Forêt
  bd_foret_enabled: false
  bd_foret_cache_dir: "cache/bd_foret"

  # RPG (Agricultural parcels)
  rpg_enabled: false
  rpg_year: 2024
  rpg_cache_dir: "cache/rpg"

  # Cadastre
  cadastre_enabled: false
  cadastre_cache_dir: "cache/cadastre"

# ============================================================================
# Preprocessing (Optional)
# ============================================================================

preprocess:
  enabled: false
  sor_k: 12
  sor_std: 2.0
  ror_radius: 1.0
  ror_neighbors: 4

# ============================================================================
# Output Configuration
# ============================================================================

output:
  format: "npz" # npz, hdf5, torch, laz
  save_stats: true
  skip_existing: true
  compression: null

# ============================================================================
# Spatial Filtering (Optional)
# ============================================================================

bbox:
  xmin: null
  ymin: null
  xmax: null
  ymax: null

# ============================================================================
# Global Settings
# ============================================================================

verbose: true
log_level: "INFO"
```

---

## 🎨 Preset Configurations

v3.0 includes ready-to-use preset configurations for common scenarios.

### 1. Default Configuration

**File:** `configs/default_v3.yaml`

**Best for:** Most users, balanced performance and accuracy

```bash
ign-lidar-hd process --config-file configs/default_v3.yaml \
    input_dir=data/raw output_dir=data/processed
```

**Features:**

- Feature mode: `asprs_classes`
- LOD level: `ASPRS`
- Spectral features enabled (RGB, NIR, NDVI)
- No data source enrichment by default

### 2. ASPRS Classification Preset

**File:** `configs/presets/asprs_classification.yaml`

**Best for:** ASPRS LAS 1.4 compliant classification

```bash
ign-lidar-hd process --config-file configs/presets/asprs_classification.yaml
```

**Features:**

- Feature mode: `asprs_classes` (lightweight)
- All data sources enabled (BD TOPO, BD Forêt, RPG, Cadastre)
- Optimized for ASPRS codes
- LAZ output format
- **Produces enriched point clouds with ASPRS classification**

### 3. Minimal Processing Preset

**File:** `configs/presets/minimal.yaml`

**Best for:** Quick testing, initial exploration

```bash
ign-lidar-hd process --config-file configs/presets/minimal.yaml
```

**Features:**

- Feature mode: `minimal` (ultra-fast)
- Smaller patches (100m)
- Fewer points (8192)
- No data enrichment
- **Fastest processing time**

### 4. Full Enrichment Preset

**File:** `configs/presets/full_enrichment.yaml`

**Best for:** Production, high-quality datasets

```bash
ign-lidar-hd process --config-file configs/presets/full_enrichment.yaml
```

**Features:**

- Feature mode: `lod3` (comprehensive)
- All data sources enabled
- Augmentation enabled
- Preprocessing enabled
- **Best quality output**

### 5. GPU-Optimized Preset

**File:** `configs/presets/gpu_optimized.yaml`

**Best for:** Systems with NVIDIA GPU (RTX 3080/4080)

```bash
ign-lidar-hd process --config-file configs/presets/gpu_optimized.yaml
```

**Features:**

- GPU acceleration enabled
- Optimized batch sizes
- No compression (speed priority)
- **Fastest processing with GPU**

---

## 🔧 Command Line Overrides

Override any parameter from the command line:

```bash
# Change feature mode
ign-lidar-hd process \
    --config-file configs/default_v3.yaml \
    features.mode=lod3

# Enable GPU
ign-lidar-hd process \
    --config-file configs/default_v3.yaml \
    processing.use_gpu=true

# Enable data sources
ign-lidar-hd process \
    --config-file configs/default_v3.yaml \
    data_sources.bd_topo_enabled=true \
    data_sources.bd_foret_enabled=true

# Multiple overrides
ign-lidar-hd process \
    --config-file configs/default_v3.yaml \
    processing.use_gpu=true \
    processing.num_workers=8 \
    features.mode=lod3 \
    output.format=hdf5
```

---

## 📚 Next Steps

- Learn about [Data Source Enrichment](./data-sources)
- Understand [Feature Modes](./feature-modes-guide) in detail
- Explore [Processing Modes](./processing-modes)
- Check [Migration Guide](./migration-v2-to-v3) from v2.x

---

## ❓ FAQ

### Which feature mode should I use?

- **ASPRS classification?** → Use `asprs_classes` (default)
- **Quick testing?** → Use `minimal`
- **Building detection?** → Use `lod2`
- **Architectural modeling?** → Use `lod3`
- **Research/maximum accuracy?** → Use `full`

### How much space do different modes use?

Based on 1 km² area with 10 pts/m²:

- `minimal`: ~500 MB
- `asprs_classes`: ~800 MB ⭐
- `lod2`: ~1.2 GB
- `lod3`: ~2.5 GB
- `full`: ~3.0 GB

### Can I customize a preset?

Yes! Use overrides:

```bash
ign-lidar-hd process \
    --config-file configs/presets/asprs_classification.yaml \
    processing.num_workers=16 \
    features.k_neighbors=30
```

### How do I migrate from v2.x?

See the [Migration Guide](./migration-v2-to-v3) or use the automatic migration:

```python
from ign_lidar.config.schema_simplified import migrate_config_v2_to_v3

new_config = migrate_config_v2_to_v3(old_config)
```
