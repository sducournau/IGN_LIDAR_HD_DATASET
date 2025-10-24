# Migration Guide: Configuration v5.4 → v5.5

**Complete guide for upgrading to the simplified v5.5 configuration system**

---

## 📋 Table of Contents

- [Overview](#overview)
- [What's Changed](#whats-changed)
- [Migration Strategies](#migration-strategies)
- [Step-by-Step Examples](#step-by-step-examples)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

---

## Overview

### What is v5.5?

Version 5.5 introduces a **revolutionary 3-tier configuration architecture** that:

- ✅ Reduces config size by **97%** (430 lines → 15 lines)
- ✅ Adds **zero-config mode** with intelligent defaults
- ✅ Provides **hardware profiles** for automatic optimization
- ✅ Includes **task presets** for common workflows
- ✅ Adds **validation** to catch errors early

### Is Migration Required?

**No!** v5.5 is **100% backward compatible**. Your v5.4 configs continue to work unchanged.

**When to migrate:**

- ✅ Starting new projects
- ✅ Simplifying existing configs
- ✅ Sharing configs with team
- ✅ Using hardware profiles
- ✅ Needing validation

**When to keep v5.4 configs:**

- ✅ Working configs in production
- ✅ No time for changes now
- ✅ Very custom settings

---

## What's Changed

### Configuration Architecture

**v5.4 (Monolithic):**

```
my_config.yaml (430 lines)
├─ Every setting specified manually
├─ No validation
└─ Hard to maintain
```

**v5.5 (3-Tier Composition):**

```
base_complete.yaml (430 lines - you never edit)
    ↓
hardware/gpu_rtx4080.yaml (30 lines - hardware settings)
    ↓
task/asprs_classification.yaml (40 lines - task settings)
    ↓
my_config.yaml (15 lines - only project specifics!)
```

### New CLI Commands

```bash
# v5.4: No discovery tools
# (Had to browse files manually)

# v5.5: Built-in discovery
ign-lidar-hd list-profiles        # List hardware profiles
ign-lidar-hd list-presets         # List task presets
ign-lidar-hd show-config my_config # Show resolved config
ign-lidar-hd validate-config my_config.yaml # Validate before running
```

### Configuration Validation

**v5.4:** Errors discovered at runtime (after hours of processing)

**v5.5:** Errors caught immediately with helpful suggestions:

```bash
$ ign-lidar-hd validate-config my_config.yaml

✗ Validation failed: my_config.yaml

Errors found:
  [processor.lod_level] Invalid value: 'LOD4'
    → Valid values: LOD2, LOD3
    → Did you mean 'LOD3'?

  [features.k_neighbors] Value 200 out of range
    → Valid range: 5-50
    → Recommended: 10 for speed, 30 for accuracy
```

---

## Migration Strategies

### Strategy 1: No Changes (Keep v5.4)

**Best for:** Production configs that work fine

```bash
# Your v5.4 config still works!
ign-lidar-hd process --config-path . --config-name my_old_config_v5.4
```

**Pros:**

- ✅ Zero effort
- ✅ No risk
- ✅ Works today

**Cons:**

- ❌ Miss v5.5 features (validation, profiles, zero-config)
- ❌ Large config files
- ❌ Hard to maintain

---

### Strategy 2: Add Validation Only

**Best for:** Keep existing config, add error checking

```bash
# Add validation to existing v5.4 config
ign-lidar-hd validate-config my_config_v5.4.yaml

# If valid, continue using as-is
ign-lidar-hd process --config-path . --config-name my_config_v5.4
```

**Pros:**

- ✅ Minimal changes
- ✅ Catch errors early
- ✅ Keep familiar config

**Cons:**

- ❌ Still large config files
- ❌ Miss profiles/presets

---

### Strategy 3: Use Hardware Profile

**Best for:** Want automatic hardware optimization

**Migration:**

```yaml
# 1. Old v5.4 config (my_config_v5.4.yaml)
processor:
  use_gpu: true
  use_gpu_chunked: true
  gpu_batch_size: 5000000
  gpu_batch_size_feature_computer: 5000000
  max_workers: 8
  # ... all other settings manually specified ...

# 2. New v5.5 config (my_config_v5.5.yaml)
defaults:
  - hardware/gpu_rtx4080 # Hardware settings inherited automatically!
  - _self_

input_dir: /data/tiles
output_dir: /data/output
# All processor settings inherited from profile!
```

**Pros:**

- ✅ Automatic hardware optimization
- ✅ 90% smaller config
- ✅ Easy to change hardware (just switch profile)

**Cons:**

- ❌ Need to choose matching profile

---

### Strategy 4: Use Task Preset

**Best for:** Common workflows (ASPRS, LOD2, LOD3)

**Migration:**

```yaml
# 1. Old v5.4 config (my_config_v5.4.yaml)
features:
  mode: lod2
  k_neighbors: 10
  compute_normals: true
  compute_curvature: true
  # ... 50+ feature settings ...
data_sources:
  bd_topo:
    enabled: true
    sources:
      - buildings
      - roads
  # ... 100+ data source settings ...

# 2. New v5.5 config (my_config_v5.5.yaml)
defaults:
  - task/lod2_buildings # All feature and data source settings inherited!
  - _self_

input_dir: /data/tiles
output_dir: /data/output
# Everything else inherited from preset!
```

**Pros:**

- ✅ 95% smaller config
- ✅ Tested settings
- ✅ Clear intent

**Cons:**

- ❌ Less customization (but can override)

---

### Strategy 5: Full v5.5 (Hardware + Task)

**Best for:** New projects, maximum simplification

**Migration:**

```yaml
# 1. Old v5.4 config (430 lines)
input_dir: /data/tiles
output_dir: /data/output
preprocess:
  buffer_size: 50.0
  normalize_intensity: true
  handle_overlap: true
processor:
  use_gpu: true
  use_gpu_chunked: true
  gpu_batch_size: 5000000
  lod_level: LOD2
  num_neighbors: 30
  # ... 400+ more lines ...

# 2. New v5.5 config (15 lines!)
defaults:
  - hardware/gpu_rtx4080
  - task/asprs_classification
  - _self_

input_dir: /data/tiles
output_dir: /data/output

# That's it! 97% size reduction
```

**Pros:**

- ✅ Maximum simplification (97% reduction)
- ✅ All v5.5 features
- ✅ Easy to maintain
- ✅ Easy to share

**Cons:**

- ❌ Need to validate new config
- ❌ Team needs v5.5 knowledge

---

## Step-by-Step Examples

### Example 1: Versailles ASPRS Classification

**Scenario:** Large tile processing with full ASPRS classification

#### v5.4 Config (430 lines)

```yaml
# config_versailles_asprs_v5.4.yaml
input_dir: /data/versailles/tiles
output_dir: /data/versailles/results

preprocess:
  buffer_size: 50.0
  normalize_intensity: true
  handle_overlap: true
  remove_duplicates: true

processor:
  use_gpu: true
  use_gpu_chunked: true
  gpu_batch_size: 5000000
  gpu_batch_size_feature_computer: 5000000
  lod_level: LOD2
  num_neighbors: 30
  search_radius: 3.0
  # ... 400+ more lines ...
```

#### v5.5 Migration (15 lines)

```yaml
# config_versailles_asprs_v5.5.yaml
defaults:
  - hardware/gpu_rtx4080 # GPU optimization
  - task/asprs_classification # ASPRS settings
  - _self_

input_dir: /data/versailles/tiles
output_dir: /data/versailles/results

# Optional: Override specific tiles
processor:
  tile_list:
    - Semis_2154_0696_6862_LA93_IGN69
    - Semis_2154_0696_6861_LA93_IGN69
```

**Result:** 97% size reduction, same functionality, easier to maintain!

---

### Example 2: Quick Building Detection (LOD2)

**Scenario:** Fast building classification, no ground truth needed

#### v5.4 Config (280 lines)

```yaml
# config_buildings_lod2_v5.4.yaml
input_dir: /data/buildings
output_dir: /data/buildings/results

processor:
  use_gpu: true
  gpu_batch_size: 3000000
  lod_level: LOD2
  num_neighbors: 20

features:
  mode: lod2
  k_neighbors: 10
  compute_normals: true
  compute_curvature: true
  compute_planarity: true
  # ... 250+ more lines ...
```

#### v5.5 Migration (10 lines)

```yaml
# config_buildings_lod2_v5.5.yaml
defaults:
  - hardware/gpu_rtx3080 # Mid-range GPU
  - task/lod2_buildings # Fast building detection
  - _self_

input_dir: /data/buildings
output_dir: /data/buildings/results
```

**Result:** 96% size reduction, optimized for RTX 3080!

---

### Example 3: CPU-Only Processing

**Scenario:** Server without GPU, high RAM

#### v5.4 Config (350 lines)

```yaml
# config_cpu_processing_v5.4.yaml
input_dir: /data/tiles
output_dir: /data/results

processor:
  use_gpu: false
  cpu_batch_size: 2000000
  max_workers: 8
  lod_level: LOD3
  # ... 330+ more lines ...
```

#### v5.5 Migration (12 lines)

```yaml
# config_cpu_processing_v5.5.yaml
defaults:
  - hardware/cpu_high # 64GB RAM, 8 workers
  - task/lod3_architecture # Detailed features
  - _self_

input_dir: /data/tiles
output_dir: /data/results
```

**Result:** 97% size reduction, optimized for CPU!

---

### Example 4: Architectural Style Analysis

**Scenario:** Detailed architectural feature extraction for style detection

#### v5.4 Config (450 lines)

```yaml
# config_architecture_v5.4.yaml
input_dir: /data/versailles
output_dir: /data/versailles/styles

processor:
  use_gpu: true
  gpu_batch_size: 5000000
  lod_level: LOD3

features:
  mode: lod3
  k_neighbors: 30
  # All architectural features enabled
  # ... 430+ more lines ...
```

#### v5.5 Migration (12 lines)

```yaml
# config_architecture_v5.5.yaml
defaults:
  - hardware/gpu_rtx4080
  - task/lod3_architecture
  - _self_

input_dir: /data/versailles
output_dir: /data/versailles/styles

processor:
  analyze_architectural_styles: true # Enable style detection
```

**Result:** 97% size reduction, clear intent!

---

### Example 5: Custom Overrides

**Scenario:** Use preset but override specific settings

#### v5.5 Config with Overrides

```yaml
# config_custom_overrides_v5.5.yaml
defaults:
  - hardware/gpu_rtx4080
  - task/asprs_classification
  - _self_

input_dir: /data/tiles
output_dir: /data/results

# Override specific settings from preset
features:
  k_neighbors: 20 # Change from default 10

processor:
  num_neighbors: 50 # Change from default 30

data_sources:
  bd_topo:
    buildings:
      buffer_meters: 1.0 # Change from default 0.5

# Everything else inherited from presets!
```

**Key Points:**

- ✅ Start with preset (80% of settings)
- ✅ Override only what you need (20%)
- ✅ Still 90% smaller than v5.4 monolithic config
- ✅ Clear what's customized vs standard

---

## Troubleshooting

### Problem: "Config file not found"

**Symptom:**

```bash
$ ign-lidar-hd process --config-name my_config
Error: Could not find 'my_config.yaml'
```

**Solutions:**

1. **Check config location:**

   ```bash
   # Default search paths:
   # 1. ign_lidar/configs/
   # 2. Current directory
   # 3. Specified config path

   # If config in current directory:
   ign-lidar-hd process --config-path . --config-name my_config

   # If config in examples/:
   ign-lidar-hd process --config-path examples --config-name my_config
   ```

2. **Check filename:**

   ```bash
   # Must be .yaml extension
   ls my_config.yaml  # ✓ Correct
   ls my_config.yml   # ✗ Wrong extension
   ```

---

### Problem: "Unknown profile/preset"

**Symptom:**

```yaml
defaults:
  - hardware/gpu_rtx5080 # ✗ Doesn't exist!
```

**Solutions:**

1. **List available options:**

   ```bash
   ign-lidar-hd list-profiles  # Show available hardware profiles
   ign-lidar-hd list-presets   # Show available task presets
   ```

2. **Use correct name:**

   ```yaml
   defaults:
     - hardware/gpu_rtx4080 # ✓ Exists
   ```

---

### Problem: "Validation errors"

**Symptom:**

```bash
$ ign-lidar-hd validate-config my_config.yaml
✗ Validation failed
```

**Solutions:**

1. **Check error messages:**

   ```bash
   # Validation shows exactly what's wrong
   ✗ [processor.lod_level] Invalid value: 'LOD4'
     → Valid values: LOD2, LOD3
     → Did you mean 'LOD3'?
   ```

2. **Fix errors and revalidate:**

   ```yaml
   # Before
   processor:
     lod_level: LOD4  # ✗ Invalid

   # After
   processor:
     lod_level: LOD3  # ✓ Valid
   ```

3. **Run validation before processing:**

   ```bash
   # Always validate first!
   ign-lidar-hd validate-config my_config.yaml
   # If validation passes:
   ign-lidar-hd process --config-name my_config
   ```

---

### Problem: "Settings not being applied"

**Symptom:**

```yaml
# my_config.yaml
processor:
  gpu_batch_size: 10000000
# But logs show: "Using batch size: 5000000"
```

**Cause:** Override order in Hydra (defaults vs _self_)

**Solution:**

```yaml
# ✗ Wrong order - your settings overridden by defaults
defaults:
  - _self_                  # Your settings first
  - hardware/gpu_rtx4080    # Profile overwrites your settings

# ✓ Correct order - your settings override defaults
defaults:
  - hardware/gpu_rtx4080    # Profile settings first
  - _self_                  # Your settings override profile

processor:
  gpu_batch_size: 10000000  # This now takes precedence!
```

**Rule:** Always put `_self_` **last** in defaults list!

---

### Problem: "Cannot override nested setting"

**Symptom:**

```yaml
# Trying to override one data source setting
data_sources:
  bd_topo:
    buildings:
      buffer_meters: 2.0
# But this replaces entire data_sources section!
```

**Solution:** Use full path in override or use Hydra syntax:

```bash
# CLI override for nested setting
ign-lidar-hd process \
  --config-name my_config \
  data_sources.bd_topo.buildings.buffer_meters=2.0
```

---

## FAQ

### Q: Do I need to migrate immediately?

**A:** No. v5.4 configs work fine in v5.5. Migrate when convenient.

---

### Q: Can I mix v5.4 and v5.5 configs?

**A:** Yes! v5.5 reads both formats. You can have:

- `old_config_v5.4.yaml` (monolithic)
- `new_config_v5.5.yaml` (3-tier composition)

Both work with `ign-lidar-hd process`.

---

### Q: How do I know which hardware profile to use?

**A:** Use `ign-lidar-hd list-profiles`:

```bash
$ ign-lidar-hd list-profiles

Available Hardware Profiles:

  gpu_rtx4080    RTX 4080 (16GB VRAM, 5M batch size)
                 Best for: Large tiles, maximum performance

  gpu_rtx3080    RTX 3080 (10GB VRAM, 3M batch size)
                 Best for: Medium tiles, balanced performance

  cpu_high       High-end CPU (64GB RAM, 8 workers)
                 Best for: Servers without GPU

  cpu_standard   Standard CPU (32GB RAM, 4 workers)
                 Best for: Workstations without GPU
```

Choose based on your hardware!

---

### Q: What if my hardware isn't listed?

**A:** Pick closest match and override specific settings:

```yaml
defaults:
  - hardware/gpu_rtx4080 # Closest match
  - _self_

processor:
  gpu_batch_size: 8000000 # Override for your 20GB VRAM GPU
```

Or create custom profile (see [Configuration Guide](CONFIG_GUIDE.md)).

---

### Q: Can I use multiple task presets?

**A:** No, Hydra only allows one preset per layer. But you can combine manually:

```yaml
defaults:
  - hardware/gpu_rtx4080
  - task/lod3_architecture # Primary preset
  - _self_

# Add settings from other presets manually
data_sources:
  bd_topo:
    # Copy settings from asprs_classification preset
    enabled: true
    sources: [buildings, roads, vegetation]
```

---

### Q: How do I validate before running?

**A:** Always use `validate-config`:

```bash
# Validate config file
ign-lidar-hd validate-config my_config.yaml

# If validation passes:
✓ Configuration validated successfully
  - Processor settings: OK
  - Feature configuration: OK
  - Data sources: OK
  - Output settings: OK

# Then safe to run:
ign-lidar-hd process --config-name my_config
```

---

### Q: What if validation fails in production?

**A:** v5.5 validation is **pre-flight only**. It doesn't affect runtime.

If you skip validation and have errors:

- **v5.4 behavior:** Crash during processing (after hours)
- **v5.5 behavior:** Same (but validation could have caught it early!)

**Best practice:** Always validate in CI/CD:

```bash
# In CI/CD pipeline
ign-lidar-hd validate-config configs/*.yaml || exit 1
```

---

### Q: Can I contribute new profiles/presets?

**A:** Yes! See [Contributing Guide](../CONTRIBUTING.md) for:

- Hardware profile template
- Task preset template
- Testing checklist
- PR submission

---

## Summary

### Quick Decision Guide

**Choose your migration strategy:**

```
                    Is config working?
                           │
                    ┌──────┴───────┐
                   Yes             No
                    │               │
            Need new features?   Fix with validation
                    │               │
              ┌─────┴────┐         └──→ Strategy 2
             Yes         No
              │           │
        New project?   Strategy 1
              │        (keep v5.4)
        ┌─────┴────┐
       Yes         No
        │           │
    Strategy 5   Strategy 3 or 4
  (full v5.5)  (gradual upgrade)
```

### Migration Checklist

Before migration:

- [ ] Backup existing v5.4 config
- [ ] Read this migration guide
- [ ] Check available profiles: `ign-lidar-hd list-profiles`
- [ ] Check available presets: `ign-lidar-hd list-presets`

During migration:

- [ ] Choose migration strategy (1-5)
- [ ] Create new v5.5 config
- [ ] Validate new config: `ign-lidar-hd validate-config`
- [ ] Test with small dataset
- [ ] Compare results with v5.4

After migration:

- [ ] Update documentation
- [ ] Train team on v5.5
- [ ] Update CI/CD to use validation
- [ ] Share simplified configs

### Need Help?

- 📖 [Configuration Guide](CONFIG_GUIDE.md) - Complete v5.5 documentation
- 📊 [Configuration Status](../CONFIG_SIMPLIFICATION_FINAL_STATUS.md) - Implementation details
- 🐛 [Report Issues](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
- 💡 [Feature Requests](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)

---

**Happy migrating! 🚀**
