# Deprecated Configuration Files

**Version**: 5.1.0  
**Date**: October 17, 2025  
**Deprecation Status**: Replaced by preset-based configuration system

---

## Overview

The following configuration files and patterns have been **deprecated** in V5.1.0 and replaced with the new preset-based configuration system.

**Migration Path**: See [MIGRATION_GUIDE_V5.1.md](docs/guides/MIGRATION_GUIDE_V5.1.md)

---

## Deprecated Example Configs

The following example configuration files have been **replaced** with V5.1 preset-based versions:

| Deprecated File (V5.0)                                    | Replacement (V5.1)                                            | Lines Saved     | Status      |
| --------------------------------------------------------- | ------------------------------------------------------------- | --------------- | ----------- |
| `examples/config_versailles_lod2.yaml` (188 lines)        | `examples/config_versailles_lod2_v5.1.yaml` (43 lines)        | **-145 (-77%)** | ✅ Migrated |
| `examples/config_versailles_lod3.yaml` (189 lines)        | `examples/config_versailles_lod3_v5.1.yaml` (45 lines)        | **-144 (-76%)** | ✅ Migrated |
| `examples/config_versailles_asprs.yaml` (183 lines)       | `examples/config_versailles_asprs_v5.1.yaml` (49 lines)       | **-134 (-73%)** | ✅ Migrated |
| `examples/config_architectural_analysis.yaml` (174 lines) | `examples/config_architectural_analysis_v5.1.yaml` (63 lines) | **-111 (-64%)** | ✅ Migrated |
| `examples/config_architectural_training.yaml` (180 lines) | `examples/config_architectural_training_v5.1.yaml` (61 lines) | **-119 (-66%)** | ✅ Migrated |

**Total Reduction**: **-653 lines (-71%)** across all examples

---

## Deprecated Configuration Patterns

### 1. Hydra Defaults Pattern (V5.0)

**Deprecated**:

```yaml
defaults:
  - ../ign_lidar/configs/config
  - _self_

input_dir: /data/tiles
output_dir: /data/output

processor:
  lod_level: LOD2
  # ... 170+ lines of settings
```

**Replacement** (V5.1):

```yaml
preset: lod2

input_dir: /data/tiles
output_dir: /data/output
# All other settings inherited from preset!
```

**Why Deprecated**:

- ❌ Requires 170+ lines of duplicated settings
- ❌ Hard to see what's custom vs default
- ❌ Difficult to maintain across multiple configs
- ❌ No clear indication of use case

**Benefits of New Approach**:

- ✅ Only 15-50 lines per config (71% reduction)
- ✅ Clear inheritance from preset
- ✅ Easy to see custom overrides
- ✅ Self-documenting (preset name indicates use case)

---

### 2. Full Config Specification (V5.0)

**Deprecated**: Specifying every single setting in your config file

```yaml
processor:
  lod_level: LOD2
  architecture: hybrid
  use_gpu: true
  num_workers: 4
  patch_size: 150.0
  patch_overlap: 0.1
  num_points: 32768
  augment: false
  num_augmentations: 0
  pin_memory: true
  gpu_batch_size: 1_000_000
  use_optimized_ground_truth: true
  reclassification:
    enabled: true
    acceleration_mode: "auto"
    chunk_size: 5_000_000
    show_progress: true
    use_geometric_rules: true

features:
  mode: full
  k_neighbors: 30
  search_radius: 1.5
  # ... 100+ more lines
```

**Replacement** (V5.1): Specify only overrides

```yaml
preset: lod2

processor:
  num_workers: 8 # Only override what's different
```

**Why Deprecated**:

- ❌ Massive duplication across config files
- ❌ 80%+ of settings are identical to defaults
- ❌ Hard to update when defaults change
- ❌ Obscures what's actually custom

**Benefits of New Approach**:

- ✅ Minimal override pattern (DRY principle)
- ✅ Automatic updates when presets improve
- ✅ Clear focus on custom settings
- ✅ Easier to review and maintain

---

### 3. Feature Mode Ambiguity (V5.0)

**Deprecated**: Unclear feature mode specification

```yaml
features:
  mode: full
  use_rgb: true
  use_nir: true
  compute_ndvi: true
  include_architectural_style: false
  # ... conflicting settings
```

**Replacement** (V5.1): Clear preset selection

```yaml
preset: lod2 # Clearly indicates LOD2 feature set

# Override only if needed
features:
  k_neighbors: 50 # More neighbors for smoother features
```

**Why Deprecated**:

- ❌ Ambiguous what "mode: full" actually means
- ❌ Possible conflicts between mode and individual settings
- ❌ Not clear which features are enabled

**Benefits of New Approach**:

- ✅ Preset name clearly indicates feature set
- ✅ No ambiguity or conflicts
- ✅ Self-documenting

---

## Deprecation Timeline

### V5.1.0 (October 2025) - Current

**Status**: Deprecated files still work but show warnings

**Action**:

- ✅ New preset-based system released
- ✅ All example configs migrated
- ✅ Documentation updated
- ⚠️ Old configs show deprecation warnings in logs

**User Action**: Migrate to preset-based configs using [MIGRATION_GUIDE_V5.1.md](docs/guides/MIGRATION_GUIDE_V5.1.md)

---

### V5.2.0 (Planned: November 2025)

**Status**: Deprecation warnings become more prominent

**Action**:

- ⚠️ Deprecation warnings shown at startup
- ⚠️ CLI warns when using old config patterns
- 📝 Automatic migration tool released

**User Action**: Complete migration before V6.0

---

### V6.0.0 (Planned: December 2025)

**Status**: Old config patterns no longer supported

**Action**:

- ❌ Old Hydra defaults pattern removed
- ❌ Full config specification pattern discouraged
- ✅ Only preset-based configs supported
- 🔧 Migration tool available for automated conversion

**User Action**: Must use preset-based configs

---

## Migration Instructions

### Quick Migration (5 minutes)

1. **Identify your use case**:

   - Building modeling → `preset: lod2`
   - Detailed architecture → `preset: lod3`
   - Standard classification → `preset: asprs`
   - Quick preview → `preset: minimal`
   - Maximum detail → `preset: full`

2. **Create new config**:

   ```yaml
   preset: [chosen_preset]

   input_dir: [your_input_path]
   output_dir: [your_output_path]
   # Add only custom overrides
   ```

3. **Test**:
   ```bash
   ign-lidar-hd process --config your_new_config_v5.1.yaml
   ```

### Detailed Migration

See [MIGRATION_GUIDE_V5.1.md](docs/guides/MIGRATION_GUIDE_V5.1.md) for:

- Step-by-step migration process
- Before/after examples
- Common patterns
- Validation checklist
- Troubleshooting

---

## Still Using Old Configs?

### Warning Signs

You're using deprecated patterns if your config:

- ✓ Uses `defaults: - ../ign_lidar/configs/config`
- ✓ Has 150+ lines of settings
- ✓ Duplicates settings across multiple files
- ✓ Doesn't use `preset:` key

### How to Migrate

```bash
# View available presets
ign-lidar-hd presets

# Check preset details
ign-lidar-hd presets --preset lod2

# Read migration guide
cat docs/guides/MIGRATION_GUIDE_V5.1.md
```

---

## Support

### Need Help?

- **Quick Start**: [CONFIG_GUIDE.md](docs/guides/CONFIG_GUIDE.md)
- **Migration Guide**: [MIGRATION_GUIDE_V5.1.md](docs/guides/MIGRATION_GUIDE_V5.1.md)
- **Examples**: `examples/config_*_v5.1.yaml`
- **Issues**: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues

### Automatic Migration (Coming in V5.2.0)

```bash
# Planned for V5.2.0
ign-lidar-hd migrate-config \
  --input old_config.yaml \
  --output new_config_v5.1.yaml
```

---

## Git History

Old configuration files are preserved in git history:

```bash
# View old configs
git log --all -- examples/config_versailles_lod2.yaml

# Checkout old config if needed (not recommended)
git show HEAD~1:examples/config_versailles_lod2.yaml > old_config.yaml
```

**Note**: Old configs are **not recommended** for new projects. Use preset-based configs instead.

---

## Benefits of Migration

### Before (V5.0)

❌ **914 lines** across 5 example configs  
❌ **80%+ duplication**  
❌ Hard to see what's custom vs default  
❌ Difficult to maintain across multiple configs  
❌ Easy to miss updates to defaults

### After (V5.1)

✅ **261 lines** across 5 example configs (**-71%**)  
✅ **Zero duplication** (DRY principle)  
✅ Only custom overrides visible  
✅ Automatic updates from preset changes  
✅ Easy preset switching (`preset: lod2` → `preset: lod3`)  
✅ Self-documenting (preset name indicates use case)

---

## Summary

- **5 example configs** deprecated and replaced with V5.1 versions
- **-653 lines** saved (-71% reduction)
- **New preset system** dramatically simplifies configuration
- **Migration guide** available for smooth transition
- **Old configs** preserved in git history but not recommended
- **Timeline**: V5.1 (warnings) → V5.2 (migration tool) → V6.0 (removal)

**Action Required**: Migrate to preset-based configs using [MIGRATION_GUIDE_V5.1.md](docs/guides/MIGRATION_GUIDE_V5.1.md)

---

**Last Updated**: October 17, 2025  
**Version**: 5.1.0  
**Status**: Active Deprecation (warnings enabled)
