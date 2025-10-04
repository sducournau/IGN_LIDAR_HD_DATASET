# Augmentation Default Enabled - Documentation Update

## Summary

Updated the codebase and documentation to reflect that **data augmentation is now ENABLED BY DEFAULT** in the `ign-lidar-hd enrich` command.

## Changes Made

### 1. CLI Implementation (`ign_lidar/cli.py`)

**Added dual flags for better UX:**

```python
# Enable augmentation (default)
--augment (default=True)

# Disable augmentation
--no-augment
```

This allows users to:

- Run without flags → augmentation enabled (default)
- Use `--no-augment` → explicitly disable augmentation
- Use `--augment --num-augmentations N` → customize augmentation count

### 2. README.md Updates

**Added information about default behavior:**

1. **Command examples section:**

   - Added note that augmentation is enabled by default with 3 versions
   - Added example showing how to disable with `--no-augment`

2. **v1.6.0 release note:**

   - Updated to emphasize augmentation is enabled by default
   - Added details: "1 original + 3 augmented versions"
   - Mentioned `--num-augmentations` and `--no-augment` options

3. **Features section:**
   - Updated "Improved augmentation" bullet to mention "enabled by default"

### 3. Docusaurus Documentation Updates

**Updated key documentation files:**

#### `website/docs/intro.md`

- Added **(enabled by default)** to enhanced augmentation feature

#### `website/docs/guides/quick-start.md`

- Added prominent info box explaining augmentation is enabled by default
- Shows that 4 files are created per tile (1 original + 3 augmented)
- Explains how to disable (`--no-augment`) or customize (`--num-augmentations N`)
- Updated patch command example to note that augmented tiles already exist

#### `website/docs/release-notes/v1.6.0.md`

- Added note that augmentation is "ENABLED BY DEFAULT"
- Explains automatic creation of 1 original + 3 augmented versions

### 4. AUGMENTATION_IMPLEMENTATION.md

**Updated implementation documentation:**

- Added prominent notice at top: "ENABLED BY DEFAULT"
- Updated usage examples to show default behavior first
- Clarified that default creates 4 files (1 original + 3 augmented)
- Reorganized examples: default → explicit → disable

## User-Facing Impact

### Before This Update

Users might not realize augmentation is enabled and be surprised by:

- 4× the number of output files
- 4× the processing time
- 4× the storage requirements

### After This Update

Users are clearly informed that:

- ✅ Augmentation is **enabled by default**
- ✅ Each tile creates **4 versions** (1 original + 3 augmented)
- ✅ How to **disable** augmentation: `--no-augment`
- ✅ How to **customize** count: `--num-augmentations N`

## Default Behavior

When running:

```bash
ign-lidar-hd enrich --input-dir raw/ --output enriched/ --mode building
```

**Output:**

```
enriched/
  tile_0001.laz          # Original
  tile_0001_aug1.laz     # Augmented version 1
  tile_0001_aug2.laz     # Augmented version 2
  tile_0001_aug3.laz     # Augmented version 3
```

## How to Control Augmentation

### Default (3 augmented versions)

```bash
ign-lidar-hd enrich --input-dir raw/ --output enriched/
```

### Disable augmentation

```bash
ign-lidar-hd enrich --input-dir raw/ --output enriched/ --no-augment
```

### Custom number of augmentations

```bash
ign-lidar-hd enrich --input-dir raw/ --output enriched/ --num-augmentations 5
# Creates: 1 original + 5 augmented = 6 files per tile
```

## Benefits of Default-Enabled Augmentation

1. **Better ML training** - More diverse training data out of the box
2. **Feature consistency** - Features computed on augmented geometry
3. **Convenience** - No need to remember to add `--augment` flag
4. **Best practice** - Encourages data augmentation usage

## Documentation Locations

All documentation now consistently mentions default behavior:

- ✅ `README.md` - Main repository documentation
- ✅ `AUGMENTATION_IMPLEMENTATION.md` - Technical implementation details
- ✅ `website/docs/intro.md` - Docusaurus introduction
- ✅ `website/docs/guides/quick-start.md` - Quick start guide
- ✅ `website/docs/release-notes/v1.6.0.md` - Release notes

---

**Status:** ✅ Complete  
**Date:** October 4, 2025
