# Release v3.3.3 - Completion Checklist

## ✅ Completed Steps

### 1. Version Update

- [x] Updated version in `pyproject.toml` to 3.3.3
- [x] Updated version in `ign_lidar/__init__.py` to 3.3.3
- [x] Updated version in `docs/package.json` to 3.3.3
- [x] Updated CHANGELOG.md with release notes

### 2. Git Tag Creation

- [x] Created annotated git tag `v3.3.3`
- [x] Pushed tag to GitHub remote
- [x] Verified tag exists on GitHub

### 3. Documentation

- [x] Created comprehensive release notes (`RELEASE_NOTES_v3.3.3.md`)
- [x] Updated CHANGELOG.md with all changes
- [x] Documented migration paths for users

## 🔄 Next Steps

### 1. Create GitHub Release (REQUIRED)

**Option A: Via GitHub Web UI (Recommended)**

1. Go to: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/releases/new?tag=v3.3.3
2. The tag `v3.3.3` should already be selected
3. Set the title: `Release v3.3.3 - Enhanced DTM Integration, Memory Optimization & Code Simplification`
4. Copy the content from `RELEASE_NOTES_v3.3.3.md` into the description
5. Check "Set as the latest release"
6. Click "Publish release"

**Option B: Via GitHub CLI (if installed)**

```bash
gh release create v3.3.3 \
  --title "Release v3.3.3 - Enhanced DTM Integration, Memory Optimization & Code Simplification" \
  --notes-file RELEASE_NOTES_v3.3.3.md \
  --latest
```

### 2. Upload to PyPI (REQUIRED for pip installation)

**Prerequisites:**

- Ensure you have PyPI credentials configured
- Activate appropriate Python environment (base conda or ign_gpu)

**Upload Command:**

```bash
# Interactive upload script (recommended)
bash scripts/upload_to_pypi.sh
```

**Manual Upload (alternative):**

```bash
# Clean old builds
rm -rf build/ dist/ ign_lidar_hd.egg-info/

# Build package
python -m build

# Validate package
twine check dist/*

# Upload to PyPI (with skip-existing to avoid errors)
twine upload --skip-existing dist/*
```

### 3. Verify Release

After uploading to PyPI, verify the installation:

```bash
# Create test environment
conda create -n test_v3.3.3 python=3.9 -y
conda activate test_v3.3.3

# Install from PyPI
pip install ign-lidar-hd==3.3.3

# Verify version
python -c "import ign_lidar; print(ign_lidar.__version__)"
# Should print: 3.3.3

# Test basic functionality
ign-lidar-hd --help
```

### 4. Announce Release (Optional)

After verifying the release works:

- [ ] Post announcement on project discussions/community channels
- [ ] Update documentation website if needed
- [ ] Send notifications to key users/contributors

## 📋 Release Summary

### Version: 3.3.3

### Release Date: October 28, 2025

### Type: Feature Release

### Key Features:

- RTM spatial indexing (10× faster DTM lookup)
- Intelligent DTM nodata interpolation
- Multi-scale chunked processing with automatic memory optimization
- Memory-optimized configuration for 28-32GB RAM systems
- Enhanced facade detection (+30-40% improvement)
- Building cluster ID features
- Simplified naming convention (no breaking changes)
- Configuration system documentation

### Breaking Changes:

**NONE** - Complete backward compatibility maintained

### Migration Required:

**NO** - All changes are backward compatible with deprecation warnings

## 🔍 Quality Checks

Before announcing the release, verify:

- [ ] GitHub release is published and visible
- [ ] PyPI package is available: https://pypi.org/project/ign-lidar-hd/3.3.3/
- [ ] Fresh installation works: `pip install ign-lidar-hd==3.3.3`
- [ ] Documentation is updated and accessible
- [ ] CHANGELOG.md reflects all changes
- [ ] All tests pass in CI/CD (if configured)

## 📞 Support

If any issues arise during the release process:

1. Check the error messages carefully
2. Verify all prerequisites are installed
3. Ensure you have proper credentials for PyPI
4. Review the upload logs for specific errors

## 🎉 Celebration

Once all steps are complete, you can celebrate the successful release of v3.3.3! 🚀

---

**Created:** October 28, 2025  
**Status:** Ready for GitHub Release and PyPI Upload
