# Configuration Guide v3.0

## Overview

The v3.0 configuration system has been simplified and streamlined:

- **Flatter structure** - Reduced nesting levels
- **Clearer naming** - More intuitive parameter names
- **Feature modes** - Predefined feature sets for different use cases
- **Better defaults** - Sensible defaults for most scenarios
- **Preset configs** - Ready-to-use configurations

## Quick Start

### Use Default Configuration

```bash
ign-lidar-hd process \
    --config-file configs/default_v3.yaml \
    input_dir=data/raw \
    output_dir=data/processed
```

### Use a Preset Configuration

```bash
# ASPRS classification (recommended)
ign-lidar-hd process --config-file configs/presets/asprs_classification.yaml

# Minimal processing (fast)
ign-lidar-hd process --config-file configs/presets/minimal.yaml

# Full enrichment (comprehensive)
ign-lidar-hd process --config-file configs/presets/full_enrichment.yaml

# GPU-optimized (fast with GPU)
ign-lidar-hd process --config-file configs/presets/gpu_optimized.yaml
```

## Feature Modes

The v3.0 system introduces **feature modes** to simplify feature selection:

### Available Modes

| Mode            | Features | Use Case               | Output Size       |
| --------------- | -------- | ---------------------- | ----------------- |
| `asprs_classes` | ~15      | ASPRS classification   | **Lightweight** ✓ |
| `minimal`       | ~8       | Quick testing          | Ultra-fast        |
| `lod2`          | ~17      | Building detection     | Medium            |
| `lod3`          | ~43      | Architectural modeling | Large             |
| `full`          | ~45      | Research               | Very large        |

### ASPRS Classes Mode (Default)

Optimized for ASPRS LAS 1.4 classification:

```yaml
features:
  mode: "asprs_classes"
  k_neighbors: 20
  use_rgb: true
  use_nir: true
  compute_ndvi: true
```

**Features included:**

- XYZ coordinates (3)
- Normal Z (verticality)
- Planarity, sphericity
- Height above ground
- Verticality, horizontality
- Point density
- RGB, NIR, NDVI (5)

**Total: ~15 features** - Perfect balance of accuracy and file size!

## Configuration Structure

### Simplified v3.0 Structure

```yaml
# Required paths
input_dir: "data/raw/lidar"
output_dir: "data/processed"

# Processing settings
processing:
  lod_level: "ASPRS" # ASPRS, LOD2, LOD3
  mode: "patches_only"
  num_workers: 4
  use_gpu: false
  patch_size: 150.0
  num_points: 16384

# Feature computation
features:
  mode: "asprs_classes" # Feature set selection
  k_neighbors: 20
  use_rgb: true
  use_nir: true
  compute_ndvi: true

# Data source enrichment
data_sources:
  bd_topo_enabled: true
  bd_topo_buildings: true
  bd_topo_roads: true
  # ... more sources

# Output
output:
  format: "laz"
  save_stats: true
  skip_existing: true
```

## Migration from v2.x

### Key Changes

1. **Flatter structure:**

   - `config.processor.*` → `config.processing.*`
   - `config.data_sources.bd_topo.enabled` → `config.data_sources.bd_topo_enabled`

2. **Feature mode:**

   - NEW: `features.mode` parameter
   - Default: `"asprs_classes"`

3. **Removed parameters:**
   - `features.mode` (old meaning) - replaced with feature set selection
   - `features.sampling_method` - always uses optimal method
   - `features.normalize_xyz` - rarely used
   - `features.normalize_features` - rarely used

### Migration Helper

```python
from ign_lidar.config.schema_simplified import migrate_config_v2_to_v3

# Load old config
old_config = load_yaml("old_config.yaml")

# Migrate to v3.0
new_config = migrate_config_v2_to_v3(old_config)
```

## Preset Configurations

### 1. ASPRS Classification (`asprs_classification.yaml`)

**Best for:** ASPRS LAS 1.4 classification

- Feature mode: `asprs_classes`
- LOD level: `ASPRS`
- Data sources: All enabled
- Output: LAZ format
- File size: **Lightweight** ✓

### 2. Minimal (`minimal.yaml`)

**Best for:** Quick testing, prototyping

- Feature mode: `minimal`
- Smaller patches (100m)
- Fewer points (8192)
- No data enrichment
- **Fastest processing**

### 3. Full Enrichment (`full_enrichment.yaml`)

**Best for:** Production, high-quality datasets

- Feature mode: `lod3`
- All data sources enabled
- Augmentation enabled
- Preprocessing enabled
- **Best quality**

### 4. GPU Optimized (`gpu_optimized.yaml`)

**Best for:** GPU acceleration (RTX 3080/4080)

- GPU enabled
- Optimized batch sizes
- No compression
- **Fastest with GPU**

## Command Line Overrides

Override any parameter from the command line:

```bash
ign-lidar-hd process \
    --config-file configs/default_v3.yaml \
    processing.use_gpu=true \
    processing.num_workers=8 \
    features.mode=lod3 \
    data_sources.bd_topo_enabled=true \
    output.format=hdf5
```

## Environment Variables

Set default paths:

```bash
export IGN_LIDAR_INPUT_DIR="/data/raw/lidar"
export IGN_LIDAR_OUTPUT_DIR="/data/processed"
export IGN_LIDAR_CACHE_DIR="/data/cache"
```

## Troubleshooting

### Feature Mode Not Recognized

```
ValueError: Invalid feature mode: 'lod2_simplified'
```

**Solution:** Use `lod2` instead of `lod2_simplified` in v3.0

### Missing Parameters

```
KeyError: 'processor'
```

**Solution:** Use `processing` instead of `processor` in v3.0

### Too Many Features

**Problem:** Output files are too large

**Solution:** Use `features.mode: "asprs_classes"` or `"minimal"`

## Best Practices

1. **Start with default:** Use `default_v3.yaml` for most cases
2. **Use presets:** Choose a preset that matches your use case
3. **ASPRS mode:** Use `asprs_classes` for balanced performance
4. **Enable GPU:** Set `processing.use_gpu: true` if available
5. **Cache data:** Keep `data_sources.*_cache_dir` enabled
6. **Skip existing:** Keep `output.skip_existing: true` for incremental processing

## See Also

- [Schema Simplified](../../ign_lidar/config/schema_simplified.py) - Python configuration API
- [Feature Modes](../../ign_lidar/features/feature_modes.py) - Feature set definitions
- [ASPRS Classes Reference](../../ASPRS_CLASSES_REFERENCE.md) - ASPRS classification codes

## Questions?

- Check examples: `examples/`
- Read docs: `docs/`
- Open issue: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues
