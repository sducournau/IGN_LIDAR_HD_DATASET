# Version Update Summary: v2.2.0 → v2.2.1

**Date:** October 10, 2025

## Overview

Version 2.2.1 is a **critical patch release** that fixes a spatial consistency bug in data augmentation. This release ensures that augmented patches correctly represent the same geographical regions as their original patches.

## Version Updates Applied

### Core Library Files

1. **pyproject.toml**

   - Updated version: `2.2.0` → `2.2.1`
   - Location: Line 7

2. **ign_lidar/**init**.py**
   - Updated `__version__`: `2.1.1` → `2.2.1`
   - Updated docstring to describe v2.2.1 changes
   - Location: Lines 6-16

### Documentation Files

3. **README.md**

   - Updated version badge: `2.2.0` → `2.2.1`
   - Added "What's New" section highlighting the critical augmentation fix
   - Added warning about regenerating datasets with augmentation
   - Location: Lines 13, 92-103

4. **docs/docs/intro.md** (Docusaurus)

   - Updated version: `2.2.0` → `2.2.1`
   - Rewrote "Latest Release" section to highlight v2.2.1 changes
   - Added warning callout about dataset regeneration
   - Updated "What's New" section with augmentation fix details
   - Location: Lines 9, 19-42, 73-99

5. **docs/package.json** (Docusaurus)

   - Updated version: `2.2.0` → `2.2.1`
   - Location: Line 3

6. **docs/docs/release-notes/v2.2.1.md** (NEW)
   - Created comprehensive release notes for v2.2.1
   - Explains the problem, solution, and migration path
   - Includes verification instructions

### Build Configuration Files

7. **conda-recipe/meta.yaml**
   - Updated version: `2.1.1` → `2.2.1`
   - Location: Line 2

### Changelog

8. **CHANGELOG.md**
   - Added new section for v2.2.1
   - Documented the critical augmentation fix
   - Listed all added, fixed, and changed items
   - Location: Lines 9-35

## Key Changes in v2.2.1

### Fixed

- **Critical Augmentation Bug**: Augmented patches now represent the same spatial regions as originals
- **Label Alignment**: Proper dropout mask handling for label consistency

### Added

- **Enhanced Augmentation Function**: `return_mask` parameter in `augment_raw_points()`
- **Patch Metadata**: `_version` and `_patch_idx` tracking
- **Verification Tool**: `scripts/verify_augmentation_fix.py`
- **Documentation**: `AUGMENTATION_FIX.md` with full explanation

### Changed

- **Pipeline Restructure**: Patch-level augmentation instead of tile-level
- Extracts patches once, then augments each individually

## Action Required

Users with augmented datasets from v2.2.0 or earlier should **regenerate their datasets** to ensure spatial consistency.

## Files Modified

Total: **9 files**

- Core library: 2 files
- Documentation: 5 files
- Build config: 1 file
- Changelog: 1 file
- New files: 1 file (release notes)

## Verification Commands

After updating, users can verify their installation:

```bash
# Check version
python -c "import ign_lidar; print(ign_lidar.__version__)"
# Expected output: 2.2.1

# Verify augmentation fix
python scripts/verify_augmentation_fix.py /path/to/patches
```

## Distribution Channels

Ready for distribution on:

- ✅ PyPI (via `pip install ign-lidar-hd`)
- ✅ Conda (via conda-forge)
- ✅ GitHub (source code)
- ✅ Documentation site (via Docusaurus)

## Next Steps

1. Commit changes to git
2. Create git tag: `v2.2.1`
3. Build and upload to PyPI
4. Update conda-forge recipe
5. Build and deploy documentation
6. Announce release

## Related Documents

- `AUGMENTATION_FIX.md` - Technical explanation of the fix
- `CHANGELOG.md` - Full project changelog
- `docs/docs/release-notes/v2.2.1.md` - User-facing release notes
- `scripts/verify_augmentation_fix.py` - Verification tool
