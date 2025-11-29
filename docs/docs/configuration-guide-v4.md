# Configuration Guide v4.0 🎯

**IGN LiDAR HD Dataset Processing Library - Configuration v4.0**

This comprehensive guide covers the new unified, flat configuration structure introduced in v4.0, designed for maximum clarity and ease of use.

---

## Table of Contents

1. [Overview](#overview)
2. [What's New in v4.0](#whats-new-in-v40)
3. [Configuration Structure](#configuration-structure)
4. [Essential Parameters](#essential-parameters)
5. [Features Configuration](#features-configuration)
6. [Optimizations](#optimizations)
7. [Advanced Configuration](#advanced-configuration)
8. [Preset Configurations](#preset-configurations)
9. [Loading Configurations](#loading-configurations)
10. [Best Practices](#best-practices)
11. [Examples](#examples)
12. [Migration from v3.x](#migration-from-v3x)

---

## Overview

Configuration v4.0 introduces a **unified, flat structure** that eliminates nested complexity while maintaining full functionality. The new design prioritizes:

- **Clarity**: Essential parameters at top level
- **Simplicity**: Flat structure reduces nesting
- **Consistency**: Unified naming conventions
- **Flexibility**: Powerful presets + customization
- **Backward compatibility**: Legacy configs still work with deprecation warnings

---

## What's New in v4.0

### Key Changes

| Change            | v3.x/v5.1                     | v4.0                              |
| ----------------- | ----------------------------- | --------------------------------- |
| **Structure**     | Nested (processor.\*)         | Flat (top-level)                  |
| **Mode**          | `processor.lod_level: "LOD2"` | `mode: lod2`                      |
| **Features**      | `features.feature_set`        | `features.mode`                   |
| **Optimizations** | Scattered in processor        | Dedicated `optimizations` section |
| **NIR**           | `use_infrared`                | `use_nir`                         |
| **Presets**       | Limited                       | 7 comprehensive presets           |

### Benefits

✅ **27% fewer parameters** at top level  
✅ **Clearer intent** with descriptive names  
✅ **Easier to customize** with flat structure  
✅ **Better defaults** based on common use cases  
✅ **Comprehensive presets** for every scenario

---

## Configuration Structure

### Overview

```yaml
# ============================================================================
# v4.0 Configuration Structure
# ============================================================================

# Meta information (optional)
config_version: 4.0.0
config_name: my_config

# ESSENTIAL PARAMETERS (Top-Level, Flat)
input_dir: /path/to/tiles
output_dir: /path/to/output
mode: lod2 # Classification mode (lod2/lod3/asprs)
processing_mode: patches_only

# Hardware
use_gpu: true
num_workers: 0

# Patch settings
patch_size: 50.0
num_points: 16384
patch_overlap: 0.1

# Architecture
architecture: pointnet++

# FEATURES (Nested - Feature Computation)
features:
  mode: standard # Feature set (minimal/standard/full)
  k_neighbors: 30
  search_radius: 2.5
  use_rgb: true
  use_nir: false
  compute_ndvi: false
  multi_scale: false

# OPTIMIZATIONS (Nested - Performance Tuning)
optimizations:
  enabled: true
  async_io: true
  batch_processing: true
  gpu_pooling: true

# ADVANCED (Nested - Complex Configuration)
advanced:
  preprocessing: { ... }
  ground_truth: { ... }
  reclassification: { ... }
```

---

## Essential Parameters

### Meta Information

```yaml
config_version: 4.0.0 # Configuration schema version (required)
config_name: my_config # Descriptive name (optional, default: "default")
```

### Input/Output

```yaml
input_dir: /path/to/laz/tiles # Required - LiDAR tiles directory
output_dir: /path/to/output # Required - Output directory
```

**Supported input formats**: `.laz`, `.las`

### Classification Mode

```yaml
mode: lod2 # Classification level
```

**Options**:

- `lod2`: Building classification (12 features, 15 classes) - **Recommended for most use cases**
- `lod3`: Detailed architectural classification (38 features, 30+ classes)
- `asprs`: ASPRS standard classification (25 features)

### Processing Mode

```yaml
processing_mode: patches_only # Output format
```

**Options**:

- `patches_only`: Generate NPZ training patches only (default)
- `enriched_only`: Generate enriched LAZ tiles only
- `both`: Generate both patches and enriched tiles

### Hardware Configuration

```yaml
use_gpu: true # Enable GPU acceleration (requires CUDA)
num_workers: 0 # CPU workers (0 = auto, MUST be 0 with GPU)
```

⚠️ **Important**: `num_workers` MUST be 0 when `use_gpu: true` (CUDA context issues)

### Patch Settings

```yaml
patch_size: 50.0 # Patch size in meters (e.g., 50×50m)
num_points: 16384 # Points per patch (8192, 16384, 32768, 65536)
patch_overlap: 0.1 # Overlap ratio (0.0-0.5, typically 0.1)
```

**Recommendations**:

- **Training**: `patch_size: 50.0`, `num_points: 16384`
- **Large buildings**: `patch_size: 100.0`, `num_points: 32768`
- **Dense areas**: `patch_size: 30.0`, `num_points: 8192`

### Architecture

```yaml
architecture: pointnet++ # Neural network architecture
```

**Options**:

- `pointnet++`: PointNet++ (default, best for most cases)
- `dgcnn`: Dynamic Graph CNN
- `pointnet`: Original PointNet
- `hybrid`: Hybrid architecture (PointNet++ + handcrafted features)

---

## Features Configuration

The `features` section controls feature computation.

### Feature Mode

```yaml
features:
  mode: standard # Feature set size
```

**Options**:

- `minimal`: ~8 features (ultra-fast, basic geometric)
- `standard`: ~12 features (LOD2, recommended)
- `full`: ~38 features (LOD3, comprehensive)

### Neighborhood Parameters

```yaml
features:
  k_neighbors: 30 # Number of neighbors for local features
  search_radius: 2.5 # Search radius in meters
```

**Recommendations**:

- **Dense urban**: `k_neighbors: 30`, `search_radius: 2.5`
- **Rural/sparse**: `k_neighbors: 20`, `search_radius: 5.0`
- **Very dense**: `k_neighbors: 50`, `search_radius: 2.0`

### Spectral Features (RGB/NIR)

```yaml
features:
  use_rgb: true # Use RGB orthophoto data
  use_nir: true # Use Near-Infrared data
  compute_ndvi: true # Compute NDVI (requires RGB+NIR)
```

**NDVI** (Normalized Difference Vegetation Index):

- Requires both `use_rgb: true` and `use_nir: true`
- Helps distinguish vegetation from buildings
- Formula: `(NIR - Red) / (NIR + Red)`

### Multi-Scale Features

```yaml
features:
  multi_scale: true # Enable multi-scale computation
  scales: [1.0, 2.0, 5.0] # Scales in meters
```

**Use cases**:

- Buildings with multiple levels of detail
- Complex architectural structures
- Multi-resolution analysis

---

## Optimizations

The `optimizations` section enables performance tuning (new in v4.0).

### Basic Optimizations

```yaml
optimizations:
  enabled: true # Enable all optimizations (master switch)
  print_stats: true # Print performance statistics
```

### I/O Optimizations

```yaml
optimizations:
  async_io: true # Asynchronous I/O
  async_workers: 2 # Number of async I/O workers
  tile_cache_size: 3 # LRU cache size for tiles
```

**Benefits**:

- 20-30% faster I/O with `async_io`
- Reduced disk access with caching

### Batch Processing

```yaml
optimizations:
  batch_processing: true # Process patches in batches
  batch_size: 4 # Patches per batch
```

**Recommendations**:

- **GPU (16GB VRAM)**: `batch_size: 4`
- **GPU (8GB VRAM)**: `batch_size: 2`
- **CPU**: `batch_size: 8-16`

### GPU Memory Pooling

```yaml
optimizations:
  gpu_pooling: true # Enable GPU memory pooling
  gpu_pool_max_size_gb: 4.0 # Max pool size in GB
```

**Benefits**:

- Reduces memory allocation overhead
- 10-15% faster GPU operations
- Prevents memory fragmentation

---

## Advanced Configuration

The `advanced` section contains complex, nested configurations.

### Preprocessing

```yaml
advanced:
  preprocessing:
    outlier_removal:
      enabled: true
      method: statistical # statistical, radius, sor
      k_neighbors: 20
      std_ratio: 2.0

    normalization:
      enabled: true
      method: minmax # minmax, standard, robust
```

### Ground Truth (Building Annotations)

```yaml
advanced:
  ground_truth:
    buildings:
      enabled: true
      source: bdtopo # bdtopo, custom
      buffer: 0.5 # Buffer in meters

    rgb:
      enabled: true
      source: orthophoto # orthophoto, custom
      resolution: 0.2 # Resolution in meters
      cache_enabled: true

    nir:
      enabled: true
      source: irc # irc, custom
      cache_enabled: true
```

### Reclassification Rules

```yaml
advanced:
  reclassification:
    enabled: true
    rules:
      - from_class: 6 # Building
        to_class: [60, 61, 62] # Roof, wall, floor
        condition: height > 3.0
```

---

## Preset Configurations

v4.0 includes **7 comprehensive presets** for common use cases.

### Available Presets

| Preset                     | Use Case                     | Features      | GPU         | Description                    |
| -------------------------- | ---------------------------- | ------------- | ----------- | ------------------------------ |
| `minimal_debug`            | Quick testing                | Minimal (8)   | Optional    | Ultra-fast debugging           |
| `fast_preview`             | Preview/prototyping          | Standard (12) | Recommended | Fast preview with good quality |
| `lod2_buildings`           | LOD2 building classification | Standard (12) | Recommended | **Default for training**       |
| `lod3_detailed`            | LOD3 architectural details   | Full (38)     | Required    | Comprehensive features         |
| `asprs_classification_cpu` | ASPRS standard (CPU)         | Full (25)     | No          | CPU-only ASPRS classification  |
| `asprs_classification_gpu` | ASPRS standard (GPU)         | Full (25)     | Required    | GPU-accelerated ASPRS          |
| `high_quality`             | Maximum quality              | Full (38)     | Required    | Best quality, slower           |

### Loading Presets

#### Python API

```python
from ign_lidar import Config

# Load preset
config = Config.preset('lod2_buildings')

# Customize if needed
config.patch_size = 100.0
config.num_points = 32768
```

#### YAML (Inheritance)

```yaml
# my_config.yaml - Inherit from preset
defaults:
  - presets_v4/lod2_buildings

# Override specific parameters
patch_size: 100.0
num_points: 32768
input_dir: /my/custom/path
```

#### CLI

```bash
# Use preset directly
ign-lidar process --preset lod2_buildings --input /data/tiles

# Or specify config that inherits from preset
ign-lidar process --config my_config.yaml
```

---

## Loading Configurations

### Method 1: Python API

```python
from ign_lidar import Config

# From YAML file
config = Config.from_yaml("config.yaml")

# From preset
config = Config.preset("lod2_buildings")

# From dictionary
config_dict = {
    "mode": "lod2",
    "use_gpu": True,
    # ...
}
config = Config.from_dict(config_dict)

# Programmatic
config = Config(
    input_dir="/data/tiles",
    output_dir="/data/output",
    mode="lod2",
    use_gpu=True
)
```

### Method 2: CLI

```bash
# From file
ign-lidar process --config config.yaml

# From preset
ign-lidar process --preset lod2_buildings --input /data --output /out

# Override parameters
ign-lidar process --config config.yaml \
    --override use_gpu=true \
    --override patch_size=100.0
```

### Method 3: Hydra Composition

```yaml
# config.yaml
defaults:
  - presets_v4/lod2_buildings
  - _self_

# Override values
use_gpu: false
num_workers: 4
```

---

## Best Practices

### 1. Start with Presets

✅ **Do**: Start with the closest preset and customize

```yaml
defaults:
  - presets_v4/lod2_buildings

# Only override what you need
patch_size: 100.0
```

❌ **Don't**: Write configs from scratch

```yaml
# Tedious and error-prone
mode: lod2
use_gpu: true
num_workers: 0
# ... 50 more lines
```

### 2. Use Meaningful Names

✅ **Do**: Descriptive config names

```yaml
config_name: paris_residential_lod2_50m
```

❌ **Don't**: Generic names

```yaml
config_name: config1
```

### 3. Comment Your Customizations

✅ **Do**: Explain why you changed defaults

```yaml
patch_size: 150.0 # Large buildings in industrial zone
num_points: 65536 # High density for detailed facades
```

### 4. Validate Before Processing

```python
from ign_lidar import Config

config = Config.from_yaml("config.yaml")

# Validate
config.validate()

# Check GPU availability
if config.use_gpu and not config.gpu_available():
    print("GPU requested but not available!")
    config.use_gpu = False
```

### 5. Use Version Control

✅ **Do**: Track config changes in git

```bash
git add configs/
git commit -m "Update patch size for dense urban areas"
```

### 6. Organize Configs by Use Case

```
configs/
├── training/
│   ├── lod2_50m.yaml
│   ├── lod3_100m.yaml
├── production/
│   ├── batch_processing.yaml
├── experiments/
│   ├── multi_scale_test.yaml
```

---

## Examples

### Example 1: Quick Training Data Generation

```yaml
# training_quick.yaml
defaults:
  - presets_v4/fast_preview

input_dir: /data/paris/tiles
output_dir: /data/paris/training
config_name: paris_quick_training
```

**Usage**:

```bash
ign-lidar process --config training_quick.yaml
```

### Example 2: High-Quality LOD3 with Spectral Features

```yaml
# lod3_spectral.yaml
defaults:
  - presets_v4/lod3_detailed

config_name: lod3_with_spectral

# Enable spectral features
features:
  use_rgb: true
  use_nir: true
  compute_ndvi: true

# Larger patches for LOD3
patch_size: 100.0
num_points: 32768

# Ground truth
advanced:
  ground_truth:
    buildings:
      enabled: true
      source: bdtopo
    rgb:
      enabled: true
      resolution: 0.2
    nir:
      enabled: true
```

### Example 3: CPU-Only Batch Processing

```yaml
# cpu_batch.yaml
defaults:
  - presets_v4/asprs_classification_cpu

config_name: batch_processing_cpu

# CPU configuration
use_gpu: false
num_workers: 8 # Use 8 CPU cores

# Batch settings
optimizations:
  batch_processing: true
  batch_size: 16 # Larger batches for CPU
  async_io: true
```

### Example 4: Custom Multi-Scale Analysis

```yaml
# multi_scale.yaml
config_version: 4.0.0
config_name: multi_scale_analysis

input_dir: /data/complex_buildings
output_dir: /data/output/multi_scale

mode: lod3
processing_mode: patches_only
use_gpu: true
num_workers: 0

patch_size: 150.0
num_points: 65536

features:
  mode: full
  k_neighbors: 50
  search_radius: 3.0
  multi_scale: true
  scales: [1.0, 2.0, 5.0, 10.0] # 4 scales
  use_rgb: true
  use_nir: true
  compute_ndvi: true

optimizations:
  enabled: true
  gpu_pooling: true
  gpu_pool_max_size_gb: 8.0
  batch_processing: true
  batch_size: 2 # Large patches, reduce batch size
```

---

## Migration from v3.x

See the **[Migration Guide v3→v4](migration-guide-v4.md)** for detailed instructions.

### Quick Migration

```bash
# Automatic migration
ign-lidar migrate-config old_config.yaml

# Preview changes
ign-lidar migrate-config old_config.yaml --dry-run

# Batch migrate directory
ign-lidar migrate-config configs/ --batch
```

### Manual Migration Cheat Sheet

| v3.x/v5.1                        | v4.0                      | Notes       |
| -------------------------------- | ------------------------- | ----------- |
| `processor.lod_level: "LOD2"`    | `mode: lod2`              | Lowercase   |
| `features.feature_set: standard` | `features.mode: standard` | Renamed     |
| `features.use_infrared`          | `features.use_nir`        | Renamed     |
| `processor.use_gpu`              | `use_gpu`                 | Flattened   |
| `processor.patch_size`           | `patch_size`              | Flattened   |
| `processor.async_io`             | `optimizations.async_io`  | New section |

---

## Troubleshooting

### Common Issues

#### Issue: Config validation fails

```python
# Check for missing required fields
config = Config.from_yaml("config.yaml")
config.validate()  # Raises ValidationError with details
```

#### Issue: GPU not available

```python
if config.use_gpu and not torch.cuda.is_available():
    print("GPU requested but CUDA not available")
    config.use_gpu = False
```

#### Issue: Out of memory

```yaml
# Reduce batch size
optimizations:
  batch_size: 2 # Was 4

# Or reduce patch size/num_points
patch_size: 30.0 # Was 50.0
num_points: 8192 # Was 16384
```

#### Issue: Features too slow

```yaml
# Use faster feature mode
features:
  mode: minimal    # Was: full

# Reduce neighborhood size
features:
  k_neighbors: 20    # Was: 50
  search_radius: 2.0 # Was: 5.0
```

---

## Performance Tips

### For Maximum Speed

```yaml
defaults:
  - presets_v4/minimal_debug

use_gpu: true
optimizations:
  enabled: true
  async_io: true
  batch_processing: true
  gpu_pooling: true
```

### For Maximum Quality

```yaml
defaults:
  - presets_v4/high_quality

features:
  multi_scale: true
  scales: [1.0, 2.0, 5.0, 10.0]

advanced:
  preprocessing:
    outlier_removal:
      enabled: true
```

### For Large Datasets

```yaml
optimizations:
  tile_cache_size: 10 # Larger cache
  batch_processing: true
  batch_size: 8 # Process multiple patches
  async_io: true
  async_workers: 4 # More I/O workers
```

---

## Further Reading

- **[Migration Guide v3→v4](migration-guide-v4.md)**: Detailed migration instructions
- **[API Reference](api-reference.md)**: Complete API documentation
- **[Architecture Overview](architecture.md)**: System architecture details
- **[Feature Documentation](features/)**: Individual feature descriptions
- **[Preset Configurations](presets/)**: Detailed preset documentation

---

## Support

- **GitHub Issues**: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues
- **Documentation**: https://sducournau.github.io/IGN_LIDAR_HD_DATASET/
- **Examples**: See `examples/` directory in repository

---

**Last Updated**: November 29, 2025  
**Version**: 4.0.0
