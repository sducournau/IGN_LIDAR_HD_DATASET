# IGN LiDAR HD - Configuration Guide V5.5

**Date**: October 31, 2025  
**Version**: 5.5.0 (Consolidated)

> **🇫🇷 Version française**: Voir [README.md](README.md)

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Quick Start](#quick-start)
3. [Overview](#overview)
4. [Configuration Files](#configuration-files)
5. [Configuration Hierarchy](#configuration-hierarchy)
6. [Core Concepts](#core-concepts)
7. [Configuration Sections](#configuration-sections)
8. [Presets & Profiles](#presets--profiles)
9. [Custom Configurations](#custom-configurations)
10. [Performance Comparison](#performance-comparison)
11. [What's New in V5.5](#whats-new-in-v55)
12. [Migration Guide](#migration-guide)
13. [Troubleshooting](#troubleshooting)

---

## Quick Reference

### Choose Your Config

| If you want...        | Use this...                | Example                                          |
| --------------------- | -------------------------- | ------------------------------------------------ |
| **Quick test**        | `fast_preview`             | `--config-name presets/fast_preview`             |
| **Standard workflow** | `asprs_classification_gpu` | `--config-name presets/asprs_classification_gpu` |
| **Best quality**      | `high_quality`             | `--config-name presets/high_quality`             |
| **No GPU**            | `asprs_classification_cpu` | `--config-name presets/asprs_classification_cpu` |
| **Custom hardware**   | Select profile             | `--config-name profiles/gpu_rtx4090`             |

### Configuration Structure

```
configs/
├── base_complete.yaml    # Complete defaults (430 lines)
├── profiles/             # Hardware-specific (6 files)
│   ├── gpu_rtx4090.yaml  # 24GB VRAM
│   ├── gpu_rtx4080.yaml  # 16GB VRAM
│   ├── gpu_rtx3080.yaml  # 12GB VRAM
│   ├── gpu_rtx3060.yaml  # 8GB VRAM
│   ├── cpu_high_end.yaml # 32+ cores
│   └── cpu_standard.yaml # 8-16 cores
└── presets/              # Task-specific (4 files)
    ├── asprs_classification_gpu.yaml
    ├── asprs_classification_cpu.yaml
    ├── fast_preview.yaml
    └── high_quality.yaml
```

---

## Quick Start

### Zero-Config (Simplest)

```bash
ign-lidar-hd process input_dir=/data output_dir=/output
```

### Use a Preset (Recommended)

```bash
# ASPRS classification with GPU
ign-lidar-hd process --config-name presets/asprs_classification_gpu \
  input_dir=/data output_dir=/output

# Fast preview
ign-lidar-hd process --config-name presets/fast_preview \
  input_dir=/data output_dir=/output
```

### Select Hardware Profile

```bash
# RTX 4080 (16GB)
ign-lidar-hd process --config-name profiles/gpu_rtx4080 \
  input_dir=/data output_dir=/output
```

## Overview

The IGN LiDAR HD configuration system uses Hydra for flexible, composable configurations. This guide explains the harmonized V5.0 configuration structure.

## Configuration Hierarchy

```
ign_lidar/configs/
├── config.yaml              # Root configuration (inherits all bases)
├── base/                    # Modular base configurations
│   ├── processor.yaml       # Core processing parameters
│   ├── features.yaml        # Feature computation settings
│   ├── data_sources.yaml    # Data fetching (BD TOPO, etc.)
│   ├── output.yaml          # Output format settings
│   └── monitoring.yaml      # Logging and metrics
├── presets/                 # Task-specific presets
│   ├── minimal.yaml         # Minimal/fast processing
│   ├── asprs_classification_gpu_optimized.yaml
│   ├── enrichment_only.yaml
│   ├── ground_truth_training.yaml
│   └── ...
├── hardware/                # Hardware-optimized configs
│   ├── cpu_only.yaml
│   ├── rtx3080.yaml
│   ├── rtx4080.yaml
│   └── rtx4090.yaml
└── advanced/                # Experimental features
    └── self_supervised_lod2.yaml
```

## Core Concepts

### 1. Base Configuration (`config.yaml`)

The root config composes all base modules:

```yaml
defaults:
  - base/processor
  - base/features
  - base/data_sources
  - base/output
  - base/monitoring
  - _self_

config_version: "5.0.0"
input_dir: null
output_dir: null
```

**All other configs inherit from this base.**

### 2. Configuration Sections

#### `processor.*` - Core Processing

```yaml
processor:
  lod_level: "ASPRS" # LOD2, LOD3, or ASPRS
  processing_mode: "enriched_only" # patches_only, both, enriched_only
  use_gpu: true
  gpu_batch_size: 5_000_000
  num_workers: 1
  output_format: "laz"
```

#### `features.*` - Feature Computation

```yaml
features:
  mode: "asprs_classes" # minimal, lod2, lod3, asprs_classes, full
  k_neighbors: 10
  use_rgb: true
  use_nir: false
  compute_ndvi: false
```

#### `data_sources.*` - External Data

```yaml
data_sources:
  bd_topo:
    enabled: true
    features:
      buildings: true
      roads: true
      water: true
      vegetation: true
```

#### `output.*` - Output Settings

```yaml
output:
  format: "laz"
  save_enriched: true
  save_patches: false
  save_metadata: true
```

#### `monitoring.*` - Logging & Metrics

```yaml
monitoring:
  log_level: "INFO"
  enable_performance_metrics: true
  enable_gpu_monitoring: true
```

## How to Use Configurations

### Method 1: Use a Preset (Recommended)

```bash
# Use a built-in preset
ign-lidar-hd process \
  -c "ign_lidar/configs/presets/asprs_classification_gpu_optimized.yaml" \
  input_dir="/path/to/input" \
  output_dir="/path/to/output"
```

### Method 2: Hardware-Specific Config

```bash
# Auto-optimize for your GPU
ign-lidar-hd process \
  --config-path ign_lidar/configs/hardware \
  --config-name rtx4080 \
  input_dir="/path/to/input" \
  output_dir="/path/to/output"
```

### Method 3: Custom Overrides

```bash
# Use preset + custom overrides
ign-lidar-hd process \
  -c "ign_lidar/configs/presets/minimal.yaml" \
  input_dir="/path/to/input" \
  output_dir="/path/to/output" \
  processor.gpu_batch_size=8000000 \
  features.k_neighbors=20
```

### Method 4: Default Config + Overrides

```bash
# Use default config with minimal overrides
ign-lidar-hd process \
  input_dir="/path/to/input" \
  output_dir="/path/to/output" \
  processor.use_gpu=true
```

## Creating Custom Configurations

### Template for Custom Preset

```yaml
# ============================================================================
# My Custom Configuration
# ============================================================================
# Description of what this config does
# Expected use case and performance
# ============================================================================

# ALWAYS inherit from base config
defaults:
  - ../config
  - _self_

# Override ONLY what you need to change
processor:
  processing_mode: "enriched_only"
  use_gpu: true
  gpu_batch_size: 10_000_000

features:
  mode: "asprs_classes"
  k_neighbors: 15
# Don't repeat parameters that are already in base configs!
```

### Best Practices

1. **Always inherit from `../config`**

   ```yaml
   defaults:
     - ../config
     - _self_
   ```

2. **Only override what's different**

   - Don't copy-paste all parameters
   - Only specify what changes from the base

3. **Use clear comments**

   - Explain WHY you're overriding
   - Document expected performance/behavior

4. **Follow the naming convention**
   - Presets: `{task}_{variant}.yaml` (e.g., `asprs_classification_gpu_optimized.yaml`)
   - Hardware: `{gpu_model}.yaml` (e.g., `rtx4080.yaml`)

## Common Configuration Patterns

### GPU-Accelerated Processing

```yaml
processor:
  use_gpu: true
  use_gpu_chunked: true
  gpu_batch_size: 5_000_000
  gpu_memory_target: 0.85
  gpu_streams: 8
  ground_truth_method: "gpu_chunked"
```

### CPU-Only Processing

```yaml
processor:
  use_gpu: false
  num_workers: 4
  ground_truth_method: "cpu"
```

### Minimal Features for Speed

```yaml
features:
  mode: "minimal"
  k_neighbors: 5
  use_rgb: false
  use_nir: false
```

### Full Features for Quality

```yaml
features:
  mode: "full"
  k_neighbors: 20
  use_rgb: true
  use_nir: true
  compute_ndvi: true
```

### Enriched LAZ Only (No Patches)

```yaml
processor:
  processing_mode: "enriched_only"

output:
  save_enriched: true
  save_patches: false
```

### Training Patches Only

```yaml
processor:
  processing_mode: "patches_only"

output:
  save_enriched: false
  save_patches: true
```

## Configuration Override Priority

Hydra applies configurations in this order (highest to lowest priority):

1. **CLI overrides** (e.g., `processor.use_gpu=true`)
2. **Custom config file** (e.g., via `-c my_config.yaml`)
3. **Preset/hardware config** (if specified)
4. **Base config** (`config.yaml`)
5. **Individual base modules** (processor.yaml, features.yaml, etc.)

## Validation

All configs are validated at runtime:

```python
# Required sections
- processor
- features
- output

# Required processor fields
- lod_level
- processing_mode
- output_format

# Valid processing_mode values
- "patches_only"
- "both"
- "enriched_only"
```

## Performance Comparison

| Profile   | Hardware  | Time/20M pts | Throughput   |
| --------- | --------- | ------------ | ------------ |
| RTX 4090  | 24GB VRAM | 6-10 min     | 120-160M/min |
| RTX 4080  | 16GB VRAM | 8-14 min     | 80-100M/min  |
| RTX 3080  | 12GB VRAM | 12-18 min    | 60-80M/min   |
| CPU (32c) | 64GB RAM  | 45-60 min    | 15-25M/min   |

---

## What's New in V5.5

### V5.5 (October 31, 2025) - Documentation Consolidation

- ✅ **Consolidated guides** - Merged quick reference into main guide
- ✅ **Updated TOC** - Better navigation structure
- ✅ **Cross-references** - Clearer links between documents

### V5.0 (October 17, 2025) - Harmonized Configuration

- ✅ **97% smaller configs** (20 lines vs 650)
- ✅ **Zero-config mode** (works with just paths)
- ✅ **6 hardware profiles** (GPU + CPU)
- ✅ **4 task presets** (common workflows)
- ✅ **No more missing keys** (all required sections included)
- ✅ **Smart defaults** (works out-of-box for 80% of users)

---

## Migration Guide

### From V4.x to V5.0

**Old approach (650 lines):**

```yaml
# All settings repeated...
processor:
  lod_level: "ASPRS"
  use_gpu: true
  gpu_batch_size: 8_000_000
  # ... 600 more lines
```

**New approach (20 lines):**

```yaml
# my_config.yaml
defaults:
  - /base_complete
  - /profiles/gpu_rtx4080

config_name: "my_custom"

# Only override what changes
processor:
  gpu_batch_size: 10_000_000
features:
  k_neighbors: 40
```

### Automatic Migration

```bash
ign-lidar migrate-config old_config.yaml --output new_config.yaml
```

---

## Troubleshooting

### Config Not Found

If Hydra can't find your config:

```bash
# Check search path
python -c "from hydra import compose, initialize_config_dir; print(compose.__doc__)"

# Or specify full path
ign-lidar-hd process --config-path /full/path/to/configs --config-name my_config
```

### Out of Memory

GPU OOM errors:

```yaml
# Reduce batch size
processor:
  gpu_batch_size: 2_000_000  # Default: 8M

# Or use chunked mode
processor:
  use_gpu_chunked: true
```

**Quick fix via command line:**

```bash
# Use smaller profile
--config-name profiles/gpu_rtx3060

# Or reduce batch size
processor.gpu_batch_size=4_000_000
```

### Validation Errors

Config validation failures:

```bash
# Check config
ign-lidar-hd process --cfg job --config-name my_config

# Enable debug logging
export HYDRA_FULL_ERROR=1
ign-lidar-hd process --config-name my_config
```

### Want to See Merged Config?

```bash
ign-lidar-hd process --config-name presets/asprs_classification_gpu \
  input_dir=/data output_dir=/output --cfg job
```

---

## Support

- **Documentation**: https://sducournau.github.io/IGN_LIDAR_HD_DATASET/
- **Issues**: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues
- **Examples**: `examples/` directory

---

**Version**: 5.5.0  
**Last Updated**: October 31, 2025  
**Note**: This guide consolidates previous CONFIGURATION_GUIDE.md content

### Parameters Not Applied

Check override priority:

1. Ensure parameter name is correct (e.g., `processor.processing_mode` not `output.processing_mode`)
2. Check if preset overrides your value
3. Verify section name (`processor` not `processing`)

### Inheritance Issues

```yaml
# ❌ Wrong - missing base inheritance
processor:
  use_gpu: true

# ✅ Correct - inherits base first
defaults:
  - ../config
  - _self_

processor:
  use_gpu: true
```

## Migration from V4.x

### Key Changes in V5.0

1. **Unified section name**: `processing` → `processor`
2. **Unified base**: All configs inherit from `config.yaml` (not `config_v5.yaml`)
3. **Simplified structure**: Removed redundant parameters
4. **Clear hierarchy**: base → hardware/preset → overrides

### Migration Example

**V4.x (Old)**:

```yaml
defaults:
  - ../config_v5 # ❌ Old reference

processing: # ❌ Old section name
  use_gpu: true
  processing_mode: "enriched_only"
```

**V5.0 (New)**:

```yaml
defaults:
  - ../config # ✅ Correct base

processor: # ✅ Correct section
  use_gpu: true
  processing_mode: "enriched_only"
```

## Examples

### Example 1: Quick ASPRS Classification

```bash
ign-lidar-hd process \
  -c "ign_lidar/configs/presets/asprs_classification_gpu_optimized.yaml" \
  input_dir="/mnt/d/ign/tiles" \
  output_dir="/mnt/d/ign/classified"
```

### Example 2: CPU Processing for Compatibility

```bash
ign-lidar-hd process \
  --config-path ign_lidar/configs/hardware \
  --config-name cpu_only \
  input_dir="/data/tiles" \
  output_dir="/data/output"
```

### Example 3: Custom GPU Batch Size

```bash
ign-lidar-hd process \
  -c "ign_lidar/configs/presets/minimal.yaml" \
  input_dir="/data/tiles" \
  output_dir="/data/output" \
  processor.gpu_batch_size=12000000 \
  features.k_neighbors=15
```

### Example 4: Full Pipeline with Ground Truth

```bash
ign-lidar-hd process \
  -c "ign_lidar/configs/presets/ground_truth_training.yaml" \
  input_dir="/data/tiles" \
  output_dir="/data/training" \
  processor.processing_mode="both"
```

## See Also

- [MIGRATION_V5_GUIDE.md](MIGRATION_V5_GUIDE.md) - Detailed migration guide
- [README.md](README.md) - Configuration structure overview
- [CONFIG_HARMONIZATION_PLAN.md](../../CONFIG_HARMONIZATION_PLAN.md) - Technical details

## Support

For issues or questions:

1. Check [Troubleshooting](#troubleshooting) section above
2. Review [Common Patterns](#common-configuration-patterns)
3. See examples in `ign_lidar/configs/presets/`
