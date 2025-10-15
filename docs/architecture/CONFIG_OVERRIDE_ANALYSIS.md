# Configuration Override Analysis - LOD2 Mode Switching to Minimal

## Problem Summary

When using the `classify_enriched_tiles.yaml` experiment configuration with `override /features: lod2`, the LOD2 mode and its associated settings (like `use_rgb`, `use_infrared`, `compute_ndvi`) are being overwritten with default values from the base `lod2.yaml` config file.

## Root Cause: Hydra Configuration Merging Order

### Configuration Loading Hierarchy

1. **Base Config** (`config.yaml`)

   ```yaml
   defaults:
     - processor: default
     - features: full # Default features config
     - ...
   ```

2. **Experiment Config** (`classify_enriched_tiles.yaml`)

   ```yaml
   defaults:
     - override /features: lod2 # Override with lod2.yaml
     - override /processor: gpu
     - override /ground_truth: update_classification

   # Local overrides
   features:
     mode: lod2
     use_rgb: true
     use_infrared: true
     compute_ndvi: true
   ```

3. **LOD2 Base Config** (`features/lod2.yaml`)
   ```yaml
   mode: lod2
   use_rgb: true
   use_infrared: false # ❌ Overwrites experiment config
   compute_ndvi: false # ❌ Overwrites experiment config
   include_extra: false
   ```

### The Problem

**Hydra merges configurations in this order:**

1. Load base `config.yaml` (features: full)
2. Apply experiment defaults: `override /features: lod2` → loads `features/lod2.yaml`
3. Apply experiment-specific overrides from `classify_enriched_tiles.yaml`

**However**, the issue is that when you use `override /features: lod2`, Hydra loads the **entire** `lod2.yaml` file, which contains:

- `use_infrared: false`
- `compute_ndvi: false`

These values from `lod2.yaml` can override your experiment-specific settings depending on merge order.

## Evidence from Code

### Location 1: `orchestrator.py` Line 751

```python
features_cfg = self.config.get('features', {})
compute_ndvi = features_cfg.get('compute_ndvi', processor_cfg.get('compute_ndvi', False))
```

The code checks `features_cfg` first for `compute_ndvi`. If `lod2.yaml` sets it to `false`, that value takes precedence.

### Location 2: `orchestrator.py` Line 143-144

```python
features_cfg = self.config.get('features', {})
self.use_rgb = features_cfg.get('use_rgb', False)
self.use_infrared = features_cfg.get('use_infrared', False)
```

Same issue - base config values from `lod2.yaml` override experiment settings.

### Location 3: `orchestrator.py` Line 595-597

```python
# Derive include_extra from feature mode (not from config)
# Only MINIMAL mode should exclude extra features
include_extra = self.feature_mode != FeatureMode.MINIMAL
```

This correctly derives from mode, but other settings don't follow this pattern.

## Affected Settings

From `lod2.yaml` that override experiment configs:

| Setting                       | lod2.yaml Value | Desired Value | Issue         |
| ----------------------------- | --------------- | ------------- | ------------- |
| `use_infrared`                | `false`         | `true`        | ❌ Overwrites |
| `compute_ndvi`                | `false`         | `true`        | ❌ Overwrites |
| `search_radius`               | `null`          | `1.5`         | ❌ Overwrites |
| `include_architectural_style` | `false`         | `true`        | ❌ Overwrites |
| `style_encoding`              | `constant`      | `constant`    | ✅ OK         |

## Solutions

### Solution 1: Remove `override /features: lod2` ✅ RECOMMENDED

**Change:**

```yaml
# classify_enriched_tiles.yaml
defaults:
  - override /processor: gpu
  # - override /features: lod2  # ❌ REMOVE THIS
  - override /ground_truth: update_classification

features:
  mode: lod2 # ✅ Set mode explicitly
  use_rgb: true
  use_infrared: true
  compute_ndvi: true
  k_neighbors: 20
  search_radius: 1.5
  # ... all other settings
```

**Why this works:**

- No base config is loaded, so no values are overwritten
- All settings are explicitly defined in the experiment config
- Full control over every parameter

### Solution 2: Use Hydra's `++` Override Syntax

Keep the base config but force overrides:

```yaml
defaults:
  - override /features: lod2

features:
  mode: lod2
  ++use_rgb: true # ++ forces override even if base sets it
  ++use_infrared: true
  ++compute_ndvi: true
  ++search_radius: 1.5
```

### Solution 3: Fix Base Config Files ⚠️ NOT RECOMMENDED

Modify `features/lod2.yaml` to enable NIR/NDVI by default:

```yaml
# features/lod2.yaml
mode: lod2
use_rgb: true
use_infrared: true # Change to true
compute_ndvi: true # Change to true
```

**Why not recommended:**

- Changes global default behavior
- Affects all experiments using LOD2
- May break existing workflows

### Solution 4: Create a New Feature Config

Create `features/lod2_enriched.yaml`:

```yaml
# @package _global_

# LOD2 with enriched tile augmentations (RGB, NIR, NDVI)
mode: lod2
k_neighbors: 20
search_radius: 1.5

# Spectral augmentations enabled
use_rgb: true
use_infrared: true
compute_ndvi: true

include_extra: false
```

Then use in experiment:

```yaml
defaults:
  - override /features: lod2_enriched
```

## Current Code Flow Analysis

### Where Values Come From

```
Hydra Config Load
    ↓
config.yaml (defaults: features: full)
    ↓
classify_enriched_tiles.yaml (override /features: lod2)
    ↓
features/lod2.yaml loaded
    ├─ mode: lod2
    ├─ use_infrared: false  ← PROBLEM
    ├─ compute_ndvi: false  ← PROBLEM
    └─ search_radius: null  ← PROBLEM
    ↓
classify_enriched_tiles.yaml (features: section)
    ├─ mode: lod2
    ├─ use_rgb: true
    ├─ use_infrared: true  ← Tries to override but may fail
    ├─ compute_ndvi: true  ← Tries to override but may fail
    └─ search_radius: 1.5  ← Tries to override but may fail
    ↓
orchestrator.py reads config
    ↓
features_cfg.get('compute_ndvi', False)  ← Gets wrong value!
```

### Why Overrides Fail

Hydra's default merge behavior:

1. **Defaults** are processed first (deep merge)
2. **Overrides** in same file may not take precedence if base config is "stronger"
3. Order matters: `_self_` placement determines when experiment overrides apply

The issue is that `override /features: lod2` loads the base config **after** the defaults section is processed, so the experiment's local `features:` section may not override values from `lod2.yaml`.

## Recommended Fix

**Update `classify_enriched_tiles.yaml`:**

```yaml
# @package _global_

# Experiment: Update Classification for Enriched Tiles with Architectural Styles
# Purpose: Update classification of enriched LAZ tiles using ground truth, NDVI, and architectural style detection
# Input: Enriched tiles with RGB/NIR bands
# Output: Updated LAZ tiles with improved classification and architectural style annotations
# Features: LOD2 optimized feature set with architectural style (constant encoding)

defaults:
  - override /processor: gpu
  - override /ground_truth: update_classification
  # REMOVED: - override /features: lod2
  - _self_ # Ensure this experiment's values take precedence

# Input/Output paths
input_dir: /mnt/c/Users/Simon/ign/classified/input
output_dir: /mnt/c/Users/Simon/ign/classified/output

# Processor settings - LOD2 processing with architectural style detection
processor:
  use_gpu: true
  num_workers: 1
  patch_size: null
  num_points: null
  skip_existing: true
  processing_mode: enriched_only
  include_architectural_style: true
  style_encoding: constant

# Ground truth configuration for classification update
ground_truth:
  enabled: true
  update_classification: true
  use_ndvi: true
  fetch_rgb_nir: false
  save_updated_tiles: true
  ndvi_vegetation_threshold: 0.3
  ndvi_low_vegetation_threshold: 0.15
  fetch_buildings: true
  fetch_roads: true
  fetch_water: true
  fetch_vegetation: true
  building_buffer: 0.5
  road_width_fallback: 6.0
  water_buffer: 1.0

# Features configuration - LOD2 optimized with RGB/NIR/NDVI augmentations
# ALL SETTINGS EXPLICIT - NO BASE CONFIG INHERITANCE
features:
  mode: lod2 # LOD2 feature set (~12-13 features)

  # Spectral augmentations - RGB, NIR, and NDVI
  use_rgb: true
  use_infrared: true
  compute_ndvi: true

  # Geometric features
  include_extra: false
  k_neighbors: 20
  search_radius: 1.5

  # PointNet++ settings
  sampling_method: random
  normalize_xyz: false
  normalize_features: false

  # Architectural style features
  include_architectural_style: true
  style_encoding: constant
  style_from_building_features: true

  # GPU optimization
  gpu_batch_size: 2000000
  use_gpu_chunked: true
# ... rest of config
```

## Verification Commands

After applying the fix, verify the configuration:

```bash
# 1. Dry-run to see resolved config
ign-lidar-hd process experiment=classify_enriched_tiles --cfg job

# 2. Check features section specifically
ign-lidar-hd process experiment=classify_enriched_tiles --cfg job | grep -A 20 "features:"

# 3. Run with debug logging
ign-lidar-hd process experiment=classify_enriched_tiles log_level=DEBUG
```

Look for these log messages:

```
🔍 DEBUG: features_cfg = {'mode': 'lod2', 'use_infrared': True, 'compute_ndvi': True, ...}
✓ NIR channel enabled (will use from input LAZ or fetch if needed)
✓ Computed NDVI from RGB and NIR
```

## Summary

- **Root Cause**: Hydra's `override /features: lod2` loads base config that overwrites experiment settings
- **Best Fix**: Remove the override and define all settings explicitly in experiment config
- **Alternative**: Use `++setting: value` syntax to force overrides
- **Verification**: Check resolved config with `--cfg job` before running

This ensures your experiment has full control over all configuration values without base config interference.
