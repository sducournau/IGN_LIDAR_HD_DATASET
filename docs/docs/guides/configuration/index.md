# Configuration System Guide

**Version:** 3.1.0 (Transitioning to 4.0.0)  
**Last Updated:** November 28, 2025

---

## 🎯 Quick Start

### Zero-Config (Simplest)

```bash
ign-lidar-hd process input_dir=/data/tiles output_dir=/data/output
```

### Use a Preset (Recommended)

```bash
# ASPRS classification with GPU
ign-lidar-hd process -c presets/asprs_classification_gpu.yaml \
  input_dir=/data/tiles output_dir=/data/output

# Building LOD2 classification
ign-lidar-hd process -c presets/lod2_buildings.yaml \
  input_dir=/data/tiles output_dir=/data/output
```

### Python API

```python
from ign_lidar.config import Config
from ign_lidar import LiDARProcessor

# Use a preset
config = Config.preset('asprs_production')
config.input_dir = '/data/tiles'
config.output_dir = '/data/output'

# Process
processor = LiDARProcessor(config)
processor.process()
```

---

## 📚 Complete Documentation

### Core Guides

- **[Quick Start](./quickstart.md)** - Get started in 5 minutes
- **[Parameter Reference](./reference.md)** - Complete parameter documentation
- **[Presets Catalog](./presets.md)** - Available presets and use cases
- **[Advanced Configuration](./advanced.md)** - Expert-level features

### Migration & Support

- **[Migration Guide v3 → v4](./migration-v3-to-v4.md)** - Upgrade instructions
- **[FAQ](./faq.md)** - Common questions and solutions
- **[Troubleshooting](./troubleshooting.md)** - Resolve configuration issues

---

## 🏗️ Configuration Architecture

IGN LiDAR HD uses a **three-layer configuration system**:

```
┌─────────────────────────────────────┐
│   1. Base Defaults (base.yaml)     │  ← Foundation
├─────────────────────────────────────┤
│   2. Presets (presets/*.yaml)      │  ← Common workflows
├─────────────────────────────────────┤
│   3. Hardware Profiles             │  ← GPU/CPU optimization
│      (hardware/*.yaml)              │
├─────────────────────────────────────┤
│   4. CLI Overrides                 │  ← User customization
│      (command-line arguments)       │  ← Highest priority
└─────────────────────────────────────┘
```

### Layer 1: Base Configuration

**Location:** `ign_lidar/configs/base.yaml`

Provides sensible defaults for all parameters. Most users never need to modify this.

**Key Sections:**

- `mode`: Classification scheme (asprs, lod2, lod3)
- `processing_mode`: Output type (patches_only, both, enriched_only)
- `features`: Feature computation settings
- `data_sources`: BD TOPO, Cadastre, OSM integration
- `ground_truth`: Ground truth configuration
- `optimizations`: Performance optimizations

### Layer 2: Presets

**Location:** `ign_lidar/configs/presets/`

Pre-configured workflows for common use cases:

| Preset                          | Purpose               | GPU            | Speed             | Quality            |
| ------------------------------- | --------------------- | -------------- | ----------------- | ------------------ |
| `asprs_classification_gpu.yaml` | Production ASPRS      | ✅ Yes         | ⚡ Fast           | ⭐⭐⭐ High        |
| `asprs_classification_cpu.yaml` | ASPRS without GPU     | ❌ No          | 🐢 Slower         | ⭐⭐⭐ High        |
| `lod2_buildings.yaml`           | Building LOD2         | ⚖️ Optional    | ⚡ Fast           | ⭐⭐ Medium        |
| `lod3_detailed.yaml`            | Detailed architecture | ✅ Recommended | 🐢 Slow           | ⭐⭐⭐⭐ Very High |
| `fast_preview.yaml`             | Quick testing         | ❌ No          | ⚡⚡ Very Fast    | ⭐ Low             |
| `minimal_debug.yaml`            | Debugging             | ❌ No          | ⚡⚡⚡ Ultra Fast | ⭐ Minimal         |
| `high_quality.yaml`             | Maximum quality       | ✅ Yes         | 🐢🐢 Very Slow    | ⭐⭐⭐⭐⭐ Maximum |

**Usage:**

```bash
ign-lidar-hd process -c presets/PRESET_NAME.yaml \
  input_dir=/data/tiles output_dir=/data/output
```

### Layer 3: Hardware Profiles

**Location:** `ign_lidar/configs/hardware/`

Optimized settings for specific hardware configurations:

| Profile                 | GPU Model  | VRAM  | Batch Size | Workers |
| ----------------------- | ---------- | ----- | ---------- | ------- |
| `gpu_rtx4090_24gb.yaml` | RTX 4090   | 24 GB | 8          | 0       |
| `gpu_rtx4080_16gb.yaml` | RTX 4080   | 16 GB | 6          | 0       |
| `gpu_rtx3080_12gb.yaml` | RTX 3080   | 12 GB | 4          | 0       |
| `cpu_high_end.yaml`     | 32+ cores  | -     | -          | 24      |
| `cpu_standard.yaml`     | 8-16 cores | -     | -          | 8       |

**Usage (compose with preset):**

```bash
ign-lidar-hd process \
  -c presets/asprs_classification_gpu.yaml \
  -c hardware/gpu_rtx4080_16gb.yaml \
  input_dir=/data/tiles output_dir=/data/output
```

### Layer 4: CLI Overrides

**Highest priority** - overrides all other layers:

```bash
ign-lidar-hd process \
  -c presets/lod2_buildings.yaml \
  input_dir=/data/tiles \
  output_dir=/data/output \
  mode=lod3 \                          # Override mode
  use_gpu=true \                       # Enable GPU
  features.k_neighbors=60 \            # High quality features
  processor.patch_size=100.0           # Custom patch size
```

---

## 📖 Configuration Format

### YAML Structure (v3.1/v5.1)

**Current format** (will be simplified in v4.0):

```yaml
# ============================================================================
# Top-level settings
# ============================================================================
input_dir: "/data/tiles" # Required
output_dir: "/data/output" # Required

# ============================================================================
# Processor configuration
# ============================================================================
processor:
  lod_level: "LOD2" # Classification scheme
  processing_mode: "both" # Output type
  use_gpu: true # GPU acceleration
  num_workers: 0 # Parallel workers

  # Patch settings
  patch_size: 150.0 # Meters
  num_points: 16384 # Points per patch
  patch_overlap: 0.1 # 10% overlap

  # Architecture
  architecture: "pointnet++"

# ============================================================================
# Feature configuration
# ============================================================================
features:
  mode: "lod2" # Feature set
  k_neighbors: 30 # Neighbors for local features
  use_rgb: true # Include RGB
  use_nir: true # Include NIR
  compute_ndvi: true # Vegetation index

# ============================================================================
# Data sources
# ============================================================================
data_sources:
  bd_topo:
    enabled: true
    features:
      buildings: true
      roads: true
      water: true
      vegetation: true

# ============================================================================
# Ground truth
# ============================================================================
ground_truth:
  enabled: true
  fuzzy_boundary_enabled: true
  fuzzy_boundary_outer: 2.5
```

### Python API (v3.2, Recommended)

**Simplified approach** using `Config` class:

```python
from ign_lidar.config import Config

# Method 1: Use a preset
config = Config.preset('lod2_buildings')
config.input_dir = '/data/tiles'
config.output_dir = '/data/output'

# Method 2: Manual configuration
config = Config(
    input_dir='/data/tiles',
    output_dir='/data/output',
    mode='lod2',
    use_gpu=True,
    patch_size=150.0,
    num_points=16384
)

# Method 3: Auto-configuration
config = Config.from_environment(
    input_dir='/data/tiles',
    output_dir='/data/output'
)

# Method 4: Load from YAML
config = Config.from_yaml('my_config.yaml')
```

---

## 🔧 Common Configuration Tasks

### 1. Enable GPU Acceleration

```yaml
processor:
  use_gpu: true
  num_workers: 0 # MUST be 0 with GPU
```

Or via CLI:

```bash
ign-lidar-hd process ... use_gpu=true processor.num_workers=0
```

### 2. Change Classification Scheme

```yaml
processor:
  lod_level: "ASPRS" # or "LOD2", "LOD3"
```

### 3. Adjust Feature Quality

```yaml
features:
  mode: "full" # minimal, lod2, full, asprs_classes
  k_neighbors: 60 # Higher = better quality, slower
```

### 4. Enable Spectral Features

```yaml
features:
  use_rgb: true # RGB from orthophotos
  use_nir: true # Near-infrared
  compute_ndvi: true # Vegetation index
```

### 5. Configure Ground Truth

```yaml
ground_truth:
  enabled: true
  fuzzy_boundary_enabled: true
  fuzzy_boundary_outer: 2.5 # Meters

data_sources:
  bd_topo:
    enabled: true
    features:
      buildings: true
```

### 6. Optimize for Speed

```yaml
processor:
  processing_mode: "patches_only" # Skip enriched LAZ
  patch_size: 50.0 # Smaller = faster

features:
  mode: "minimal" # Minimal features
  k_neighbors: 20 # Lower = faster

# Enable Phase 4 optimizations
optimizations:
  enabled: true
  async_io:
    enabled: true
  batch_processing:
    enabled: true
```

---

## ⚠️ Version Compatibility

### Current State (v3.1)

**Three configuration systems coexist:**

1. ✅ **`config.py`** (v3.2 style) - **RECOMMENDED**

   - Python `Config` class
   - Simplified, flat structure
   - Preset system

2. ⚠️ **`schema.py`** (v3.1 style) - **DEPRECATED**

   - Old Hydra dataclasses
   - Nested structure
   - Will be removed in v4.0

3. ✅ **YAML configs** (v5.1 style) - **CURRENT**
   - Modular base + presets
   - Hydra composition

### Upcoming Changes (v4.0)

**Harmonization planned:**

- Remove `schema.py` completely
- Standardize on `Config` class
- Simplify YAML structure
- Single documentation source

**Migration tool available:**

```bash
ign-lidar-hd migrate-config old_config.yaml --output new_config.yaml
```

See **[Migration Guide](./migration-v3-to-v4.md)** for details.

---

## 🆘 Getting Help

### Documentation

- **[FAQ](./faq.md)** - Common questions
- **[Troubleshooting](./troubleshooting.md)** - Resolve issues
- **[Examples](./examples/)** - Real-world configurations

### Commands

```bash
# Show current configuration
ign-lidar-hd show-config

# List available presets
ign-lidar-hd list-presets

# List hardware profiles
ign-lidar-hd list-profiles

# Validate configuration
ign-lidar-hd validate-config config.yaml
```

### Support

- **GitHub Issues:** https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues
- **Documentation:** https://sducournau.github.io/IGN_LIDAR_HD_DATASET/

---

## 📝 Next Steps

1. **New Users:** Start with [Quick Start Guide](./quickstart.md)
2. **Existing Users:** Check [Migration Guide](./migration-v3-to-v4.md) for v4.0 prep
3. **Advanced Users:** Explore [Advanced Configuration](./advanced.md)
4. **Developers:** See [Configuration System Design](../../architecture/configuration.md)

---

**Last Updated:** November 28, 2025  
**Document Version:** 1.0  
**Package Version:** 3.1.0
