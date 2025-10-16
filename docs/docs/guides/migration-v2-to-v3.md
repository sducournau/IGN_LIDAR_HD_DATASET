---
sidebar_position: 10
title: Migration Guide v2.x to v3.0
---

# Migration Guide: v2.x to v3.0

Complete guide for migrating from v2.x to the simplified v3.0 configuration system.

---

## 🎯 Overview

v3.0 introduces a simplified configuration system with:

- **Flatter structure** (2 levels instead of 3-4)
- **Feature modes** for easy feature selection
- **Clearer parameter names**
- **45% fewer parameters**
- **Better defaults**

**Good news:** v3.0 is backward compatible with deprecation warnings for 6 months.

---

## 🔄 Key Changes Summary

### Structure Changes

| v2.x                                  | v3.0                           | Change                    |
| ------------------------------------- | ------------------------------ | ------------------------- |
| `config.processor.*`                  | `processing.*`                 | Renamed                   |
| `config.features.mode`                | `features.mode`                | New meaning (feature set) |
| `config.data_sources.bd_topo.enabled` | `data_sources.bd_topo_enabled` | Flattened                 |
| `config.output.processing_mode`       | `processing.mode`              | Moved                     |
| `config.features.use_infrared`        | `features.use_nir`             | Renamed                   |

### Removed Parameters

These parameters were removed (now auto-optimized or rarely used):

- ❌ `features.sampling_method` - Always uses optimal method
- ❌ `features.normalize_xyz` - Rarely used
- ❌ `features.normalize_features` - Rarely used
- ❌ `processor.prefetch_factor` - Auto-optimized
- ❌ `processor.pin_memory` - Auto-detected
- ❌ `processor.batch_size` - Always "auto"

### New Features

- ✅ `features.mode` - Feature set selection (`asprs_classes`, `minimal`, `lod2`, `lod3`, `full`)
- ✅ Preset configurations (`asprs_classification.yaml`, `minimal.yaml`, etc.)
- ✅ Automatic migration helper

---

## 📝 Configuration Migration

### Automatic Migration

Use the built-in migration helper:

```python
from ign_lidar.config.schema_simplified import migrate_config_v2_to_v3

# Load old config
with open("old_config.yaml") as f:
    old_config = yaml.safe_load(f)

# Migrate
new_config = migrate_config_v2_to_v3(old_config)

# Save
with open("new_config_v3.yaml", "w") as f:
    yaml.dump(dict(new_config), f)
```

### Manual Migration

#### Example 1: Basic Configuration

**v2.x:**

```yaml
config:
  processor:
    lod_level: "LOD2"
    processing_mode: "patches_only"
    use_gpu: false
    num_workers: 4
    patch_size: 150.0
    patch_overlap: 0.1
    num_points: 16384

  features:
    mode: "full" # Old meaning: enable all feature flags
    k_neighbors: 20
    use_rgb: true
    use_infrared: true
    compute_ndvi: true

  output:
    format: "npz"
    save_stats: true
```

**v3.0:**

```yaml
processing:
  lod_level: "LOD2"
  mode: "patches_only"
  use_gpu: false
  num_workers: 4
  patch_size: 150.0
  patch_overlap: 0.1
  num_points: 16384

features:
  mode: "asprs_classes" # New meaning: feature set selection
  k_neighbors: 20
  use_rgb: true
  use_nir: true # Renamed from use_infrared
  compute_ndvi: true

output:
  format: "npz"
  save_stats: true
```

#### Example 2: Data Sources

**v2.x:**

```yaml
config:
  data_sources:
    bd_topo:
      enabled: true
      features:
        buildings: true
        roads: true
        water: true
      cache_dir: "cache/bd_topo"

    bd_foret:
      enabled: true
      cache_dir: "cache/bd_foret"
```

**v3.0:**

```yaml
data_sources:
  # Flattened structure
  bd_topo_enabled: true
  bd_topo_buildings: true
  bd_topo_roads: true
  bd_topo_water: true
  bd_topo_cache_dir: "cache/bd_topo"

  bd_foret_enabled: true
  bd_foret_cache_dir: "cache/bd_foret"
```

#### Example 3: GPU Configuration

**v2.x:**

```yaml
config:
  processor:
    use_gpu: true
    gpu_batch_size: 1000000
    use_gpu_chunked: true
    prefetch_factor: 2 # Removed in v3.0
    pin_memory: true # Removed in v3.0
```

**v3.0:**

```yaml
processing:
  use_gpu: true

features:
  gpu_batch_size: null # Auto-calculated (recommended)
  use_gpu_chunked: true
# prefetch_factor and pin_memory removed (auto-optimized)
```

---

## 🚀 Code Migration

### Importing Configuration

**v2.x:**

```python
from ign_lidar.config.schema import IGNLiDARConfig

config = IGNLiDARConfig(
    input_dir="data/raw",
    output_dir="data/processed",
    processor=ProcessorConfig(lod_level="LOD2"),
    features=FeaturesConfig(mode="full")
)
```

**v3.0:**

```python
from ign_lidar.config.schema_simplified import IGNLiDARConfig

config = IGNLiDARConfig(
    input_dir="data/raw",
    output_dir="data/processed",
    processing=ProcessingConfig(lod_level="LOD2"),
    features=FeatureConfig(mode="asprs_classes")
)
```

### Accessing Parameters

**v2.x:**

```python
lod_level = config.processor.lod_level
use_rgb = config.features.use_rgb
bd_topo_enabled = config.data_sources.bd_topo.enabled
```

**v3.0:**

```python
lod_level = config.processing.lod_level
use_rgb = config.features.use_rgb
bd_topo_enabled = config.data_sources.bd_topo_enabled
```

### Using Feature Modes

**v2.x (manual feature selection):**

```python
features = FeaturesConfig(
    mode="full",  # Meant: enable all flags
    k_neighbors=20,
    use_rgb=True,
    use_infrared=True,
    compute_ndvi=True,
    include_extra=True,
    # ... 15 more parameters
)
```

**v3.0 (feature mode selection):**

```python
features = FeatureConfig(
    mode="asprs_classes",  # Selects optimized feature set
    k_neighbors=20,
    use_rgb=True,
    use_nir=True,
    compute_ndvi=True
)
```

---

## 📋 Migration Checklist

### Step 1: Update Dependencies

```bash
pip install --upgrade ign-lidar-hd
```

### Step 2: Backup Old Configs

```bash
cp -r configs/ configs_v2_backup/
```

### Step 3: Choose Migration Strategy

**Option A: Use New Presets (Easiest)**

```bash
# Use ASPRS preset
ign-lidar-hd process --config-file configs/presets/asprs_classification.yaml

# Or default v3
ign-lidar-hd process --config-file configs/default_v3.yaml
```

**Option B: Automatic Migration**

```python
python migrate_config.py old_config.yaml new_config.yaml
```

**Option C: Manual Update**

Edit your config files following the examples above.

### Step 4: Test New Configuration

```bash
# Test on a small area first
ign-lidar-hd process \
    --config-file new_config_v3.yaml \
    bbox.xmin=650000 bbox.ymin=6860000 \
    bbox.xmax=651000 bbox.ymax=6861000
```

### Step 5: Update Scripts

Update any Python scripts that use the old configuration API.

### Step 6: Verify Results

```bash
# Compare outputs
python verify_migration.py \
    output_v2/ \
    output_v3/ \
    --tolerance 0.01
```

---

## ⚠️ Breaking Changes

### Immediate (v3.0)

✅ **None** - v3.0 is backward compatible with deprecation warnings

### Future (v4.0, ~6 months)

❌ `ign_lidar.config.loader` module will be removed  
❌ `ign_lidar.config.schema` (old) will be removed  
❌ Old configuration format will not be supported

---

## 🎯 Feature Mode Migration

### Understanding the Change

**v2.x `features.mode`:**

- Meant: "enable all feature computation flags"
- Values: `minimal`, `full`, `custom`
- Confusing: didn't actually select features

**v3.0 `features.mode`:**

- Means: "select predefined feature set"
- Values: `asprs_classes`, `minimal`, `lod2`, `lod3`, `full`, `custom`
- Clear: directly selects which features to compute

### Migration Table

| v2.x mode | v3.0 equivalent | Notes                        |
| --------- | --------------- | ---------------------------- |
| `minimal` | `minimal`       | Same                         |
| `full`    | `asprs_classes` | Recommended (better balance) |
| `full`    | `full`          | If you need all features     |
| `custom`  | `custom`        | Same concept                 |

### Example Migration

**v2.x:**

```yaml
features:
  mode: full # Enable all feature flags
  # Then manually disable unwanted features
  use_rgb: true
  use_infrared: false
  compute_ndvi: false
  include_extra: false
```

**v3.0:**

```yaml
features:
  mode: asprs_classes # Select optimized feature set
  # Flags now control spectral features only
  use_rgb: true
  use_nir: false
  compute_ndvi: false
```

---

## 📊 Performance Impact

### File Size Changes

With `asprs_classes` mode (vs v2.x `full` mode):

- **-67%** smaller patch files
- **-60%** disk space usage
- **Same classification accuracy** (with ground truth)

### Processing Speed

With `asprs_classes` mode (vs v2.x `full` mode):

- **2.8x faster** feature computation
- **2.5x less memory** usage
- **Same or better** classification quality

---

## 💡 Best Practices

### 1. Start with Defaults

```bash
# Use default v3 config
ign-lidar-hd process --config-file configs/default_v3.yaml
```

### 2. Use Presets

```bash
# For ASPRS classification
ign-lidar-hd process --config-file configs/presets/asprs_classification.yaml

# For quick testing
ign-lidar-hd process --config-file configs/presets/minimal.yaml
```

### 3. Override Only What's Needed

```bash
ign-lidar-hd process \
    --config-file configs/default_v3.yaml \
    processing.num_workers=16 \
    data_sources.bd_topo_enabled=true
```

### 4. Use Feature Modes

Don't manually configure 20+ feature flags. Use modes:

```yaml
# ✅ Good
features:
  mode: asprs_classes

# ❌ Avoid
features:
  use_feature_1: true
  use_feature_2: false
  # ... 18 more flags
```

---

## 🆘 Troubleshooting

### Deprecation Warnings

**Warning:**

```
DeprecationWarning: load_config_from_yaml() is deprecated
```

**Solution:**
Migrate to Hydra/OmegaConf or use v3.0 config files.

### Import Errors

**Error:**

```python
ImportError: cannot import name 'ProcessorConfig'
```

**Solution:**

```python
# Old
from ign_lidar.config.schema import ProcessorConfig

# New
from ign_lidar.config.schema_simplified import ProcessingConfig
```

### Parameter Not Found

**Error:**

```
KeyError: 'processor'
```

**Solution:**
Use `processing` instead of `processor` in v3.0.

### Feature Mode Not Recognized

**Error:**

```
ValueError: Invalid feature mode: 'lod2_simplified'
```

**Solution:**
Use `lod2` instead of `lod2_simplified` in v3.0.

---

## 📚 Resources

- [Configuration System v3.0](./configuration-v3)
- [Feature Modes Guide](./feature-modes-guide)
- [Data Sources](./data-sources)
- [v3.0 Release Notes](../release-notes/v3.0.0)

---

## ❓ FAQ

### Do I have to migrate immediately?

No. v2.x configuration works in v3.0 with deprecation warnings. You have 6 months to migrate before v4.0.

### Will my old configs still work?

Yes, with deprecation warnings. The automatic migration helper ensures compatibility.

### What if I have custom features?

Use `mode: custom` and specify your feature list:

```yaml
features:
  mode: custom
  custom_features:
    - xyz
    - normal_z
    - planarity
    - my_custom_feature
```

### Can I keep using v2.x?

Yes, but v3.0 is recommended for:

- Smaller files (67% reduction)
- Faster processing (2.8x speedup)
- Simpler configuration

### How do I get help?

1. Check this migration guide
2. Read [v3.0 documentation](./configuration-v3)
3. Open an issue on GitHub
4. Join our community discussions
