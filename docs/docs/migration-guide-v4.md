# Migration Guide v3.x → v4.0 🔄

**Comprehensive guide for migrating IGN LiDAR HD configurations from v3.x/v5.1 to v4.0**

---

## Table of Contents

1. [Overview](#overview)
2. [Why Migrate?](#why-migrate)
3. [Breaking Changes](#breaking-changes)
4. [Migration Methods](#migration-methods)
5. [Automatic Migration (Recommended)](#automatic-migration-recommended)
6. [Manual Migration](#manual-migration)
7. [Validation](#validation)
8. [Common Migration Scenarios](#common-migration-scenarios)
9. [Troubleshooting](#troubleshooting)
10. [FAQ](#faq)

---

## Overview

Configuration v4.0 introduces a **unified, flat structure** that simplifies configuration while maintaining backward compatibility. This guide helps you migrate existing v3.x or v5.1 configurations to v4.0.

### Version Timeline

```
v3.0 (2023) → v3.1 (2024) → v3.2 (2024) → v5.1 (2025) → v4.0 (2025)
                                                            ↑
                                                     You are here
```

### Migration Complexity

| From Version | Difficulty | Time Required | Changes              |
| ------------ | ---------- | ------------- | -------------------- |
| v5.1         | Easy       | 5 minutes     | Structure flattening |
| v3.2         | Easy       | 5 minutes     | Parameter renaming   |
| v3.1         | Moderate   | 10 minutes    | Structure + naming   |
| v3.0         | Moderate   | 15 minutes    | Legacy format        |

---

## Why Migrate?

### Benefits of v4.0

✅ **Simpler structure**: 27% fewer nested levels  
✅ **Clearer naming**: Intuitive parameter names  
✅ **Better presets**: 7 comprehensive presets  
✅ **Improved performance**: Dedicated optimizations section  
✅ **Future-proof**: Active development and support

### Backward Compatibility

⚠️ **v3.x configs still work** with deprecation warnings through v4.x  
⚠️ **v3.x support will be removed in v5.0** (Q3 2026)

**Migration is optional now, but recommended for:**

- New projects
- Configs you actively maintain
- Taking advantage of new features

---

## Breaking Changes

### Structural Changes

#### 1. Flattened Top-Level Parameters

**Before (v3.x/v5.1)**:

```yaml
processor:
  lod_level: "LOD2"
  use_gpu: true
  num_workers: 0
  patch_size: 50.0
```

**After (v4.0)**:

```yaml
mode: lod2 # Flattened + lowercase
use_gpu: true # Flattened
num_workers: 0 # Flattened
patch_size: 50.0 # Flattened
```

#### 2. Feature Configuration Rename

**Before (v3.2/v5.1)**:

```yaml
features:
  feature_set: standard # Old name
```

**After (v4.0)**:

```yaml
features:
  mode: standard # New name (consistent with top-level mode)
```

#### 3. Infrared → NIR

**Before**:

```yaml
features:
  use_infrared: true
```

**After**:

```yaml
features:
  use_nir: true # More standard terminology
```

#### 4. New Optimizations Section

**Before (scattered)**:

```yaml
processor:
  async_io: true
  batch_processing: true
  gpu_pooling: true
```

**After (organized)**:

```yaml
optimizations:
  enabled: true
  async_io: true
  batch_processing: true
  gpu_pooling: true
```

### Removed Parameters

These parameters are deprecated (still work with warnings):

- `processor.lod_level` → use `mode`
- `features.feature_set` → use `features.mode`
- `features.use_infrared` → use `features.use_nir`

---

## Migration Methods

### Method 1: Automatic (Recommended) ⚡

Use the built-in migration tool:

```bash
# Single file
ign-lidar migrate-config old_config.yaml

# Preview changes (dry-run)
ign-lidar migrate-config old_config.yaml --dry-run

# Custom output
ign-lidar migrate-config old_config.yaml -o new_config.yaml

# Batch migrate directory
ign-lidar migrate-config configs/ --batch

# Verbose output
ign-lidar migrate-config old_config.yaml -v
```

### Method 2: Python API 🐍

```python
from ign_lidar.config.migration import ConfigMigrator

migrator = ConfigMigrator(backup=True)

# Migrate file
result = migrator.migrate_file(
    "old_config.yaml",
    output_path="new_config.yaml",
    overwrite=True
)

if result.success:
    print(f"✓ Migrated: {result.old_version} → {result.new_version}")
    print(f"Changes: {len(result.changes)}")
else:
    print(f"✗ Failed: {result.error}")

# Migrate dictionary
old_config = {...}
new_config, warnings = migrator.migrate_dict(old_config)
```

### Method 3: Manual 📝

Edit the YAML file manually (see [Manual Migration](#manual-migration))

---

## Automatic Migration (Recommended)

### CLI Tool

The `migrate-config` command handles all transformations automatically.

#### Basic Usage

```bash
# Migrate single file
ign-lidar migrate-config config_v3.yaml
# Output: config_v3_v4.0.yaml
```

#### Preview Changes (Dry-Run)

```bash
ign-lidar migrate-config config_v3.yaml --dry-run
```

**Output**:

```
📖 Loading configuration from: config_v3.yaml
🔄 Migrating from v5.1 → v4.0.0

======================================================================
📊 Migration Summary
======================================================================
  Version:           5.1 → 4.0.0
  Parameters before: 70
  Parameters after:  51
  Changes:           5 transformations
======================================================================

📋 Detailed Changes:
   • Migrated from v5.1 to v4.0.0
   • Flattened configuration structure
   • Renamed: lod_level → mode
   • Renamed: features.feature_set → features.mode
   • Added optimizations section

📄 New Configuration:
----------------------------------------------------------------------
config_version: 4.0.0
mode: lod2
use_gpu: true
...
----------------------------------------------------------------------

💡 Dry-run complete. Use without --dry-run to save changes.
```

#### Batch Migration

```bash
# Migrate all YAML files in directory
ign-lidar migrate-config configs/ --batch

# Output to specific directory
ign-lidar migrate-config configs/ --batch -o configs_v4/

# Force overwrite existing files
ign-lidar migrate-config configs/ --batch --force
```

**Output**:

```
📂 Found 15 YAML file(s) in: configs/

======================================================================
🔄 Batch Migration Progress
======================================================================

[1/15] config1.yaml
  🔄 v5.1 → v4.0.0
  ✅ Saved: config1.yaml

[2/15] config2.yaml
  ✅ Already v4.0

...

======================================================================
📊 Batch Migration Summary
======================================================================
  Total files:           15
  Successfully migrated: 12
  Already v4.0:          3
  Failed:                0
======================================================================

✅ Migrated files saved to: configs/migrated_v4.0/
```

### Python API

```python
from ign_lidar.config.migration import ConfigMigrator

# Initialize migrator
migrator = ConfigMigrator(backup=True)  # Creates backups

# Migrate file
result = migrator.migrate_file(
    "config_v3.yaml",
    output_path="config_v4.yaml",
    overwrite=True
)

# Check result
if result.success:
    if result.migrated:
        print(f"✓ Migrated: {result.old_version} → {result.new_version}")
        print(f"Changes: {len(result.changes)}")

        # Show changes
        for change in result.changes:
            print(f"  • {change}")

        # Show warnings
        if result.warnings:
            print("Warnings:")
            for warning in result.warnings:
                print(f"  ⚠️ {warning}")
    else:
        print("Already v4.0 - no migration needed")
else:
    print(f"✗ Migration failed: {result.error}")

# Access configs
original = result.original_config
migrated = result.migrated_config
```

### Detect Version

```python
from ign_lidar.config.migration import ConfigMigrator
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

migrator = ConfigMigrator()
version = migrator.detect_version(config)

print(f"Detected version: {version}")
# Output: "5.1", "3.2", "3.1", "4.0", or None
```

---

## Manual Migration

If you prefer manual migration, follow these steps:

### Step 1: Add Version Metadata

```yaml
# Add to top of file
config_version: 4.0.0
config_name: my_migrated_config
```

### Step 2: Flatten Processor Section

**Before**:

```yaml
processor:
  lod_level: "LOD2"
  use_gpu: true
  num_workers: 0
  patch_size: 50.0
  num_points: 16384
  patch_overlap: 0.1
  architecture: "pointnet++"
  processing_mode: "patches_only"
```

**After**:

```yaml
mode: lod2 # Lowercase lod_level
use_gpu: true # Flattened
num_workers: 0 # Flattened
patch_size: 50.0 # Flattened
num_points: 16384 # Flattened
patch_overlap: 0.1 # Flattened
architecture: pointnet++ # Flattened
processing_mode: patches_only # Flattened
```

### Step 3: Rename Feature Parameters

**Before**:

```yaml
features:
  feature_set: standard # Old name
  use_infrared: true # Old name
  multi_scale_computation: true # Old name
```

**After**:

```yaml
features:
  mode: standard # Renamed
  use_nir: true # Renamed
  multi_scale: true # Renamed
```

### Step 4: Create Optimizations Section

Extract optimization parameters from processor:

**Before (in processor)**:

```yaml
processor:
  async_io: true
  async_workers: 2
  tile_cache_size: 3
  batch_processing: true
  batch_size: 4
  gpu_pooling: true
  gpu_pool_max_size_gb: 4.0
```

**After (new section)**:

```yaml
optimizations:
  enabled: true
  async_io: true
  async_workers: 2
  tile_cache_size: 3
  batch_processing: true
  batch_size: 4
  gpu_pooling: true
  gpu_pool_max_size_gb: 4.0
  print_stats: true
```

### Step 5: LOD Level Mapping

| v3.x/v5.1 | v4.0    |
| --------- | ------- |
| `"LOD2"`  | `lod2`  |
| `"LOD3"`  | `lod3`  |
| `"ASPRS"` | `asprs` |

### Step 6: Feature Mode Mapping

| v3.x            | v4.0       |
| --------------- | ---------- |
| `minimal`       | `minimal`  |
| `lod2`          | `standard` |
| `lod3`          | `full`     |
| `asprs_classes` | `full`     |
| `full`          | `full`     |
| `custom`        | `standard` |

### Complete Example

**Before (v5.1)**:

```yaml
input_dir: /data/tiles
output_dir: /data/output

processor:
  lod_level: "LOD2"
  processing_mode: "patches_only"
  use_gpu: true
  num_workers: 0
  patch_size: 50.0
  num_points: 16384
  architecture: "pointnet++"
  async_io: true
  batch_processing: true

features:
  mode: lod2
  k_neighbors: 30
  search_radius: 2.5
  use_rgb: true
  use_infrared: true
  compute_ndvi: true
```

**After (v4.0)**:

```yaml
config_version: 4.0.0
config_name: migrated_config

input_dir: /data/tiles
output_dir: /data/output

mode: lod2
processing_mode: patches_only
use_gpu: true
num_workers: 0
patch_size: 50.0
num_points: 16384
architecture: pointnet++

features:
  mode: standard
  k_neighbors: 30
  search_radius: 2.5
  use_rgb: true
  use_nir: true
  compute_ndvi: true
  multi_scale: false

optimizations:
  enabled: true
  async_io: true
  batch_processing: true
  batch_size: 4
  gpu_pooling: true
```

---

## Validation

After migration, validate your configuration:

### Method 1: Python API

```python
from ign_lidar import Config

# Load and validate
config = Config.from_yaml("config_v4.yaml")

# Automatic validation on load
print("✓ Configuration valid!")

# Manual validation
config.validate()

# Check specific fields
print(f"Mode: {config.mode}")
print(f"GPU: {config.use_gpu}")
print(f"Features: {config.features.mode}")
```

### Method 2: CLI

```bash
# Validate configuration
ign-lidar validate-config config_v4.yaml

# Dry-run processing (validates without processing)
ign-lidar process --config config_v4.yaml --dry-run
```

### Common Validation Errors

#### Missing Required Fields

```
ValidationError: Missing required field: input_dir
```

**Fix**: Add required fields:

```yaml
input_dir: /path/to/tiles
output_dir: /path/to/output
```

#### Invalid Mode

```
ValidationError: Invalid mode 'LOD2'. Must be one of: lod2, lod3, asprs
```

**Fix**: Use lowercase:

```yaml
mode: lod2 # Not "LOD2"
```

#### GPU + Workers Conflict

```
ValidationError: num_workers must be 0 when use_gpu is True
```

**Fix**:

```yaml
use_gpu: true
num_workers: 0 # Must be 0 with GPU
```

---

## Common Migration Scenarios

### Scenario 1: LOD2 Training Dataset

**Before (v5.1)**:

```yaml
processor:
  lod_level: "LOD2"
  use_gpu: true
  patch_size: 50.0

features:
  mode: lod2
  use_rgb: true
```

**After (v4.0)** - Use preset:

```yaml
defaults:
  - presets_v4/lod2_buildings

input_dir: /data/tiles
output_dir: /data/output
```

### Scenario 2: LOD3 with Spectral Features

**Before**:

```yaml
processor:
  lod_level: "LOD3"
  use_gpu: true

features:
  mode: lod3
  use_rgb: true
  use_infrared: true
  compute_ndvi: true
```

**After** - Use preset + customize:

```yaml
defaults:
  - presets_v4/lod3_detailed

features:
  use_rgb: true
  use_nir: true
  compute_ndvi: true
```

### Scenario 3: CPU-Only Processing

**Before**:

```yaml
processor:
  lod_level: "ASPRS"
  use_gpu: false
  num_workers: 8

features:
  mode: asprs_classes
```

**After** - Use CPU preset:

```yaml
defaults:
  - presets_v4/asprs_classification_cpu

num_workers: 8 # Override default
```

### Scenario 4: Custom Multi-Scale

**Before**:

```yaml
processor:
  lod_level: "LOD3"
  patch_size: 100.0

features:
  mode: lod3
  multi_scale_computation: true
  scales: [1.0, 2.0, 5.0]
```

**After**:

```yaml
defaults:
  - presets_v4/lod3_detailed

patch_size: 100.0

features:
  multi_scale: true
  scales: [1.0, 2.0, 5.0]
```

---

## Troubleshooting

### Issue: Migration tool not found

```bash
ign-lidar: command not found: migrate-config
```

**Solution**: Update to v4.0+

```bash
pip install --upgrade ign-lidar-hd
# or
git pull && pip install -e .
```

### Issue: "Already v4.0" but config looks old

**Cause**: Config has `config_version: 4.0.0` but old structure

**Solution**: Remove `config_version` line and re-migrate:

```bash
# Remove version line
sed -i '/config_version/d' config.yaml

# Migrate again
ign-lidar migrate-config config.yaml --force
```

### Issue: Migration changes values unexpectedly

**Cause**: Feature mode mapping (lod2 → standard)

**Explanation**: This is intentional - v4.0 uses more descriptive names:

- `lod2` → `standard` (12 features)
- `lod3` → `full` (38 features)

The actual features computed are the same.

### Issue: Optimizations section empty

**Cause**: Old config had no optimization parameters

**Solution**: Optimizations are added with safe defaults. Customize if needed:

```yaml
optimizations:
  enabled: true
  async_io: true
  batch_processing: true
```

### Issue: Backup files piling up

**Cause**: Migrator creates backups by default

**Solution**:

```python
# Disable backups
migrator = ConfigMigrator(backup=False)
```

Or clean up:

```bash
# Remove backup files
rm *_backup_*.yaml
```

---

## FAQ

### Q: Do I need to migrate immediately?

**A**: No. v3.x configs work with deprecation warnings through v4.x. But migration is recommended for:

- Active projects
- Configs you maintain
- New features

### Q: Will migration break my existing processing?

**A**: No. The migration preserves all functionality. Generated outputs are identical.

### Q: Can I use presets with my old configs?

**A**: Yes, but migrate first for best experience:

```yaml
defaults:
  - presets_v4/lod2_buildings
# Then add your customizations
```

### Q: What if migration fails?

**A**:

1. Check error message for details
2. Try manual migration
3. Report issue: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues

### Q: Can I migrate back to v3.x?

**A**: Use the backup files created during migration:

```bash
ls *_backup_*.yaml
cp config_backup_20251129_120000.yaml config.yaml
```

### Q: How do I migrate programmatically created configs?

**A**:

```python
from ign_lidar.config.migration import ConfigMigrator

# Your old config dict
old_config = {
    "processor": {"lod_level": "LOD2", ...},
    "features": {...}
}

# Migrate
migrator = ConfigMigrator()
new_config, warnings = migrator.migrate_dict(old_config)

# Use new config
from ign_lidar import Config
config = Config.from_dict(new_config)
```

### Q: Are there performance differences between v3.x and v4.0 configs?

**A**: v4.0 has better performance with:

- Dedicated optimizations section
- Improved GPU memory pooling
- Better default values

But migrated configs have same performance as originals.

### Q: Can I mix v3.x and v4.0 configs?

**A**: Not recommended. Migrate all configs to v4.0 for consistency.

---

## Summary Checklist

Before you start:

- [ ] Read [What's New in v4.0](#whats-new-in-v40)
- [ ] Check [Breaking Changes](#breaking-changes)
- [ ] Backup your configs

Migration:

- [ ] Run automatic migration or edit manually
- [ ] Review changes (dry-run)
- [ ] Validate migrated configs
- [ ] Test with sample data

After migration:

- [ ] Update documentation/scripts
- [ ] Delete backup files (optional)
- [ ] Consider using presets for new configs

---

## Additional Resources

- **[Configuration Guide v4.0](configuration-guide-v4.md)**: Comprehensive config documentation
- **[API Reference](api-reference.md)**: Python API documentation
- **[Preset Configurations](presets/)**: Detailed preset docs
- **[Examples](../../examples/)**: Example configurations

---

## Support

Need help with migration?

- **GitHub Issues**: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues
- **Documentation**: https://sducournau.github.io/IGN_LIDAR_HD_DATASET/
- **Examples**: See `examples/` directory

---

**Last Updated**: November 29, 2025  
**Version**: 4.0.0
