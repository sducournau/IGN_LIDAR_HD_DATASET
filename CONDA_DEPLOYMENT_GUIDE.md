# Conda-Forge Deployment Guide for IGN LiDAR HD

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Recipe Files](#recipe-files)
4. [Local Testing](#local-testing)
5. [Submission Process](#submission-process)
6. [Maintenance](#maintenance)

## Overview

This guide walks through deploying `ign-lidar-hd` to conda-forge with GPU support.

### Package Information

- **Name**: ign-lidar-hd
- **Version**: 1.7.4
- **License**: MIT
- **Homepage**: https://github.com/sducournau/IGN_LIDAR_HD_DATASET
- **PyPI**: https://pypi.org/project/ign-lidar-hd/

### Deployment Strategy

**Approach**: Single package with optional GPU dependencies

- Base package: CPU-only (smaller, works everywhere)
- GPU support: Install cupy/cuml separately
- Clear documentation on both paths

## Prerequisites

### 1. Accounts

- [x] GitHub account: sducournau
- [ ] Anaconda.org account: Sign up at https://anaconda.org/
- [ ] Fork: https://github.com/conda-forge/staged-recipes

### 2. Install Tools

```bash
# Activate base conda environment
conda activate base

# Install required tools
conda install -n base conda-build conda-verify anaconda-client
conda install -n base -c conda-forge grayskull conda-smithy
```

### 3. Create Anaconda.org Account

1. Go to: https://anaconda.org/
2. Click "Sign Up"
3. Verify your email
4. Login with: `anaconda login`

## Recipe Files

### Directory Structure

```
IGN_LIDAR_HD_DATASET/
├── conda/
│   ├── meta.yaml              # Main recipe file
│   ├── build.sh               # Linux/macOS build
│   ├── bld.bat                # Windows build
│   └── conda_build_config.yaml # Build configuration
└── ...
```

### File Templates

See the actual recipe files that will be created in the next step.

## Local Testing

### Build the Package

```bash
cd /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET

# Build for your platform
conda build conda/

# Build for specific Python version
conda build conda/ --python=3.12

# Build for all supported Python versions
for PY_VER in 3.8 3.9 3.10 3.11 3.12; do
    conda build conda/ --python=$PY_VER
done
```

### Test Installation

```bash
# Create test environment
conda create -n test_ign_lidar python=3.12

# Install from local build
conda install -n test_ign_lidar --use-local ign-lidar-hd

# Activate and test
conda activate test_ign_lidar
ign-lidar-hd --help
python -c "import ign_lidar; print(ign_lidar.__version__)"

# Test with GPU (if available)
conda install -n test_ign_lidar -c rapidsai -c conda-forge cuml cupy
python -c "from ign_lidar.features_gpu_chunked import GPU_AVAILABLE, CUML_AVAILABLE; print(f'GPU: {GPU_AVAILABLE}, cuML: {CUML_AVAILABLE}')"

# Cleanup
conda deactivate
conda env remove -n test_ign_lidar
```

## Submission Process

### 1. Fork staged-recipes

```bash
# Go to: https://github.com/conda-forge/staged-recipes
# Click "Fork"

# Clone your fork
cd ~/projects
git clone https://github.com/sducournau/staged-recipes.git
cd staged-recipes

# Create feature branch
git checkout -b ign-lidar-hd
```

### 2. Add Recipe

```bash
# Create recipe directory
mkdir -p recipes/ign-lidar-hd

# Copy recipe files
cp /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET/conda/meta.yaml recipes/ign-lidar-hd/
cp /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET/conda/build.sh recipes/ign-lidar-hd/
cp /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET/conda/bld.bat recipes/ign-lidar-hd/
```

### 3. Validate Recipe

```bash
# Run linter
conda-smithy recipe-lint recipes/ign-lidar-hd

# If validation script exists
python .ci_support/validate_recipe.py recipes/ign-lidar-hd
```

### 4. Commit and Push

```bash
git add recipes/ign-lidar-hd
git commit -m "Add ign-lidar-hd: LiDAR processing library with GPU support

- Version: 1.7.4
- License: MIT
- Python: 3.8+
- Optional GPU acceleration via CuPy/RAPIDS cuML
- Cross-platform: Linux, macOS, Windows"

git push origin ign-lidar-hd
```

### 5. Create Pull Request

**Title**: Add ign-lidar-hd: LiDAR processing library with GPU support

**Description**:

````markdown
## Package Information

- **Name**: ign-lidar-hd
- **Version**: 1.7.4
- **License**: MIT
- **Homepage**: https://github.com/sducournau/IGN_LIDAR_HD_DATASET
- **PyPI**: https://pypi.org/project/ign-lidar-hd/
- **Documentation**: https://sducournau.github.io/IGN_LIDAR_HD_DATASET/

## Description

IGN LiDAR HD is a comprehensive Python library for processing IGN (Institut National de l'Information Géographique et Forestière) LiDAR HD data into machine learning-ready datasets for Building Level of Detail (LOD) classification tasks.

## Key Features

- Process LiDAR point clouds to structured datasets
- RGB/Infrared augmentation from IGN orthophotos
- Geometric feature computation
- Optional GPU acceleration (12-20x speedup with RAPIDS cuML)
- CLI and Python API
- Pipeline configuration support

## GPU Support

The package includes optional GPU acceleration:

- Base installation: CPU-only (works everywhere)
- GPU support: `conda install cupy cuml` (requires NVIDIA GPU)
- Performance: 12-20x faster with GPU

## Testing

All tests pass locally:

```bash
conda build conda/ --python=3.12
# Tests included in recipe
```
````

## Checklist

- [x] License file included
- [x] Compatible with Python 3.8+
- [x] Tests included in recipe
- [x] Documentation available
- [x] PyPI package exists
- [x] GPU dependencies optional
- [x] Cross-platform support

## Maintainer

I (@sducournau) will maintain this feedstock.

````

### 6. Monitor PR

- CI/CD builds will run automatically
- Review bots will comment with suggestions
- conda-forge team will review (usually 2-7 days)
- Address any requested changes

## Maintenance

### Update Package Version

When releasing a new version (e.g., 1.7.5):

```bash
# Clone your feedstock (after initial acceptance)
git clone https://github.com/conda-forge/ign-lidar-hd-feedstock.git
cd ign-lidar-hd-feedstock

# Create branch
git checkout -b update_1.7.5

# Update recipe/meta.yaml
# - Change version number
# - Update sha256 hash from PyPI

# Commit and push
git add recipe/meta.yaml
git commit -m "Update to v1.7.5"
git push origin update_1.7.5

# Create PR on feedstock repo
````

### Automated Updates

Conda-forge has a bot (regro-cf-autotick-bot) that automatically:

- Detects new PyPI releases
- Creates PRs to update the feedstock
- You just need to review and merge

## Timeline

| Phase                  | Duration  | Notes          |
| ---------------------- | --------- | -------------- |
| Setup accounts & tools | 1-2 hours | One-time       |
| Create recipe files    | 2-3 hours | One-time       |
| Local testing          | 1-2 hours | Per version    |
| Submit PR              | 1 hour    | Per submission |
| Review & approval      | 2-7 days  | Async waiting  |
| Post-deployment        | 1 hour    | One-time       |

**Total Active Time**: 6-9 hours  
**Total Calendar Time**: 1-2 weeks

## Resources

### Official Documentation

- Conda-forge docs: https://conda-forge.org/docs/
- Contributing guide: https://conda-forge.org/docs/maintainer/adding_pkgs.html
- Recipe format: https://docs.conda.io/projects/conda-build/en/latest/resources/define-metadata.html

### Tools

- grayskull: Auto-generate recipes from PyPI
- conda-smithy: Manage feedstock repositories
- conda-build: Build conda packages

### Community

- Gitter chat: https://gitter.im/conda-forge/conda-forge.github.io
- GitHub discussions: https://github.com/conda-forge/conda-forge.github.io/discussions

## Next Steps

1. **Review this guide** - Understand the process
2. **Create Anaconda.org account** - Required for publishing
3. **Install build tools** - conda-build, grayskull, etc.
4. **Generate recipe files** - Use templates (next document)
5. **Test locally** - Ensure everything builds
6. **Submit to conda-forge** - Create PR
7. **Respond to feedback** - Address reviewer comments
8. **Celebrate!** - Package available to everyone

---

**Ready to create the recipe files?**

Let me know and I'll generate the complete `meta.yaml`, `build.sh`, and `bld.bat` files customized for your package!
