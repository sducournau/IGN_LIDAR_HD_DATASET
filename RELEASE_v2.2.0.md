# Release v2.2.0 - Multi-Format Output Support

## 📅 Release Date

October 10, 2025

## 🎯 Release Type

**Minor Version** - New features with backward compatibility

## 📦 Version Numbers Updated

- **pyproject.toml**: 2.1.2 → 2.2.0
- **README.md**: 2.1.2 → 2.2.0
- **website/package.json**: 0.0.0 → 2.2.0
- **CHANGELOG.md**: Added v2.2.0 entry

## 🚀 Key Features

### 1. Multi-Format Output

Save patches in multiple formats simultaneously:

```yaml
output:
  format: hdf5,laz # Both HDF5 and LAZ
```

### 2. Complete Format Support

- ✅ NPZ (NumPy compressed)
- ✅ HDF5 (with gzip compression) - **FIXED**
- ✅ PyTorch (.pt tensors) - **NEW**
- ✅ LAZ (point clouds) - **NEW**

### 3. New Tools & Documentation

- `scripts/convert_hdf5_to_laz.py` - HDF5 to LAZ converter
- `MULTI_FORMAT_OUTPUT_IMPLEMENTATION.md` - Technical docs
- `MULTI_FORMAT_QUICK_START.md` - Quick reference
- `OUTPUT_FORMAT_ANALYSIS.md` - Format analysis

## 🐛 Critical Fixes

### HDF5 Output Bug (Critical)

- **Issue**: HDF5 format was configured but no files were generated
- **Root Cause**: Saving logic only implemented NPZ format
- **Fix**: Added complete HDF5 saving with h5py
- **Impact**: Users can now use HDF5 format as documented

## 📝 Files Modified

### Core Code

1. **ign_lidar/core/processor.py** (~200 lines changed)

   - Added h5py import
   - Added torch import (optional)
   - Added format validation
   - Added `_save_patch_as_laz()` method
   - Refactored patch saving for multi-format
   - Added HDF5 saving logic
   - Added PyTorch saving logic
   - Added LAZ saving logic

2. **ign_lidar/io/formatters/**init**.py**

   - Added HybridFormatter import

3. **ign_lidar/io/formatters/hybrid_formatter.py** (NEW)
   - Complete hybrid formatter implementation
   - KNN graph building
   - Voxelization support
   - Comprehensive feature handling

### Configuration

4. **ign_lidar/configs/experiment/config_lod3_training.yaml**

   - Changed format from `hdf5` to `hdf5,laz`

5. **ign_lidar/configs/output/multi.yaml** (NEW)
   - Multi-format configuration preset

### Scripts

6. **scripts/convert_hdf5_to_laz.py** (NEW)

   - HDF5 to LAZ conversion tool
   - Batch processing support
   - Inspection mode

7. **scripts/CONVERT_HDF5_TO_LAZ.md** (NEW)
   - Complete documentation for converter

### Documentation

8. **MULTI_FORMAT_OUTPUT_IMPLEMENTATION.md** (NEW)
9. **MULTI_FORMAT_QUICK_START.md** (NEW)
10. **OUTPUT_FORMAT_ANALYSIS.md** (NEW)
11. **OUTPUT_FORMAT_COMPLETE_FIX.md** (NEW)
12. **HDF5_BUG_FIX.md** (NEW)

### Version Files

13. **pyproject.toml** - Version updated
14. **README.md** - Version badge updated
15. **website/package.json** - Version updated
16. **CHANGELOG.md** - Release entry added

## 💾 Git Operations

### Staging

```bash
git add pyproject.toml
git add README.md
git add CHANGELOG.md
git add website/package.json
git add ign_lidar/core/processor.py
git add ign_lidar/io/formatters/__init__.py
git add ign_lidar/io/formatters/hybrid_formatter.py
git add ign_lidar/configs/experiment/config_lod3_training.yaml
git add ign_lidar/configs/output/multi.yaml
git add scripts/convert_hdf5_to_laz.py
git add scripts/CONVERT_HDF5_TO_LAZ.md
git add MULTI_FORMAT_OUTPUT_IMPLEMENTATION.md
git add MULTI_FORMAT_QUICK_START.md
git add OUTPUT_FORMAT_ANALYSIS.md
git add OUTPUT_FORMAT_COMPLETE_FIX.md
git add HDF5_BUG_FIX.md
```

### Commit

```bash
git commit -m "Release v2.2.0: Multi-Format Output Support

Features:
- Multi-format output (hdf5,laz,npz,pytorch)
- LAZ patch export for visualization
- HybridFormatter for ensemble models
- HDF5 to LAZ conversion tool

Fixes:
- Critical HDF5 output bug
- PyTorch format implementation
- Format validation

Documentation:
- Complete multi-format guides
- HDF5 converter documentation
- Format analysis and recommendations

BREAKING: None (backward compatible)
"
```

### Tag

```bash
git tag -a v2.2.0 -m "Release v2.2.0: Multi-Format Output Support

Major new features:
- Multi-format output support (save in multiple formats simultaneously)
- LAZ patch format for visualization in CloudCompare/QGIS
- HybridFormatter for ensemble/hybrid models
- HDF5 to LAZ conversion tool

Critical fixes:
- HDF5 format now properly saves files
- PyTorch format fully implemented

Full changelog: CHANGELOG.md
"
```

### Push

```bash
git push origin main
git push origin v2.2.0
```

## ✅ Pre-Release Checklist

- [x] Version numbers updated in all files
- [x] CHANGELOG.md updated with detailed changes
- [x] README.md version badge updated
- [x] New features documented
- [x] Critical bugs fixed
- [x] New files added to repository
- [x] Configuration files updated
- [x] Scripts and tools included
- [x] Comprehensive documentation created

## 🔄 Post-Release Tasks

### 1. PyPI Release

```bash
# Build package
python -m build

# Upload to PyPI
python -m twine upload dist/ign-lidar-hd-2.2.0*
```

### 2. GitHub Release

- Create release from tag v2.2.0
- Copy CHANGELOG entry as release notes
- Attach wheel and sdist files

### 3. Documentation Deployment

```bash
cd website
npm run build
npm run deploy
```

### 4. Announce Release

- Update repository README
- Update documentation homepage
- Notify users of critical HDF5 fix

## 📊 Impact Assessment

### Breaking Changes

**None** - Fully backward compatible

### Migration Guide

**Not required** - Existing code continues to work

### Recommended Updates

Users currently using `format: hdf5` should:

1. Reinstall package: `pip install -e .`
2. Rerun processing (old HDF5 files were not generated)
3. Consider using `format: hdf5,laz` for visualization support

## 🎉 Release Summary

This release represents a significant enhancement to output flexibility with multi-format support while fixing a critical bug in HDF5 output. The addition of LAZ patch format enables seamless integration with popular visualization tools, and the new HybridFormatter supports advanced ensemble modeling approaches.

**Recommended for all users** - Especially those affected by the HDF5 output bug.

---

**Release Manager**: GitHub Copilot
**Release Date**: October 10, 2025
**Version**: 2.2.0
