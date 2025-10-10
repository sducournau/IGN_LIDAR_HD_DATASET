# Version 2.2.1 Release Checklist

## ✅ Completed Tasks

### 1. Version Number Updates

- [x] **pyproject.toml** - Updated from 2.2.0 to 2.2.1
- [x] **ign_lidar/**init**.py** - Updated from 2.1.1 to 2.2.1
- [x] **conda-recipe/meta.yaml** - Updated from 2.1.1 to 2.2.1
- [x] **docs/package.json** - Updated from 2.2.0 to 2.2.1

### 2. Documentation Updates

- [x] **README.md** - Updated version badge and "What's New" section
- [x] **docs/docs/intro.md** - Updated main documentation page with v2.2.1 highlights
- [x] **docs/docs/release-notes/v2.2.1.md** - Created comprehensive release notes
- [x] **CHANGELOG.md** - Added v2.2.1 entry with all changes
- [x] **AUGMENTATION_FIX.md** - Technical documentation of the fix (already created)
- [x] **VERSION_UPDATE_v2.2.1.md** - Summary of version update process

### 3. Code Changes (Already Completed)

- [x] Fixed augmentation pipeline in `ign_lidar/core/processor.py`
- [x] Enhanced `augment_raw_points()` in `ign_lidar/preprocessing/utils.py`
- [x] Created verification tool `scripts/verify_augmentation_fix.py`

### 4. Testing & Validation

- [x] Fixed linting errors in processor.py
- [x] Verified imports work correctly
- [x] Created verification tool for users

## 📋 Next Steps for Release

### Step 1: Verify Installation

```bash
cd /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET
pip install -e .

# Verify version
python -c "import ign_lidar; print(ign_lidar.__version__)"
# Expected: 2.2.1
```

### Step 2: Test the Fix

```bash
# Run verification on test patches (if available)
python scripts/verify_augmentation_fix.py /path/to/test/patches --max-patches 3
```

### Step 3: Git Commit & Tag

```bash
# Stage all changes
git add CHANGELOG.md README.md pyproject.toml ign_lidar/__init__.py
git add conda-recipe/meta.yaml docs/package.json docs/docs/
git add VERSION_UPDATE_v2.2.1.md

# Commit
git commit -m "Release v2.2.1: Critical augmentation spatial consistency fix

- Fixed augmentation pipeline to maintain spatial consistency
- Patches now extracted once, then augmented individually
- Added return_mask parameter for proper label alignment
- Added verification tool and comprehensive documentation
- Updated all version strings and documentation

BREAKING: Datasets with augmentation from v2.2.0 should be regenerated"

# Create tag
git tag -a v2.2.1 -m "Version 2.2.1: Critical augmentation fix

Critical bug fix ensuring augmented patches represent the same
geographical regions as their original patches.

Key changes:
- Fixed spatial consistency in augmentation pipeline
- Enhanced augment_raw_points() with return_mask parameter
- Added patch metadata tracking (_version, _patch_idx)
- New verification tool: scripts/verify_augmentation_fix.py
- Comprehensive documentation in AUGMENTATION_FIX.md

Action Required: Regenerate datasets with augmentation for spatial consistency."

# Push to GitHub
git push origin main
git push origin v2.2.1
```

### Step 4: Build and Publish to PyPI

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build package
python -m build

# Check package
twine check dist/*

# Upload to PyPI
twine upload dist/*

# Or use the existing script
./upload_to_pypi.sh
```

### Step 5: Update Conda-Forge (if applicable)

```bash
# Update conda-forge feedstock
# The meta.yaml is already updated to 2.2.1
# Follow conda-forge contribution guidelines
```

### Step 6: Deploy Documentation

```bash
cd docs

# Build documentation
npm run build

# Deploy to GitHub Pages
npm run deploy

# Or use the existing script
cd ..
./deploy-docs.sh
```

### Step 7: Create GitHub Release

1. Go to https://github.com/sducournau/IGN_LIDAR_HD_DATASET/releases
2. Click "Draft a new release"
3. Choose tag: v2.2.1
4. Release title: "v2.2.1 - Critical Augmentation Fix"
5. Description: Copy from `docs/docs/release-notes/v2.2.1.md`
6. Attach built wheels from `dist/` folder
7. Publish release

### Step 8: Announce Release

- [ ] Update README badges (if needed)
- [ ] Post announcement on relevant forums/communities
- [ ] Update documentation site
- [ ] Notify users who reported the issue

## 📊 Summary of Changes

**Files Modified:** 8

- Core library: 2 files (pyproject.toml, ign_lidar/**init**.py)
- Documentation: 4 files (README.md, intro.md, release notes)
- Build config: 2 files (meta.yaml, package.json)

**Files Created:** 3

- docs/docs/release-notes/v2.2.1.md
- VERSION_UPDATE_v2.2.1.md
- AUGMENTATION_FIX.md (created earlier)

**Lines Changed:**

- +95 lines added
- -35 lines removed
- Net change: +60 lines

## 🎯 Key Messages for Users

1. **Critical Fix**: Augmented patches now represent correct spatial regions
2. **Action Required**: Regenerate datasets with augmentation
3. **Verification Tool**: Use `scripts/verify_augmentation_fix.py` to check
4. **Documentation**: Read `AUGMENTATION_FIX.md` for details

## 📝 Version History Context

- **v2.2.1** (Current) - Critical augmentation fix
- **v2.2.0** (Oct 10, 2025) - Multi-format output, LAZ patches
- **v2.1.1** (Previous) - Bug fixes for planarity and boundaries
- **v2.0.0** - Major architecture overhaul with Hydra

## ✨ What Makes This Release Critical

This is a **patch release** with **critical importance** because:

1. **Data Integrity**: Previous augmented datasets had spatial inconsistencies
2. **Model Training**: Could have trained on misaligned data
3. **Easy Fix**: Simple upgrade + regeneration solves the issue
4. **Verification**: Tool included to validate the fix

## 🔗 Quick Links

- GitHub: https://github.com/sducournau/IGN_LIDAR_HD_DATASET
- PyPI: https://pypi.org/project/ign-lidar-hd/
- Docs: https://sducournau.github.io/IGN_LIDAR_HD_DATASET/
- Issues: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues
