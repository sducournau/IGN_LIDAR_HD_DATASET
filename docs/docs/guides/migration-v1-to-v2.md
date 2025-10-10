---
sidebar_position: 10
title: Migration Guide v1.x → v2.0
---

# Migration Guide: v1.x → v2.0

Complete guide to upgrading from IGN LiDAR HD v1.x to v2.0+.

---

## 🎯 Quick Summary

**Good News:** Your v1.x code will mostly work! The legacy CLI is still supported for backward compatibility.

**Changes Required:**

1. ✅ **Import paths** - Module reorganization requires import updates
2. ✅ **CLI (optional)** - New Hydra CLI is available (legacy CLI still works!)
3. ✅ **Configuration** - New Hydra-based config system (optional)

**Time to Migrate:** 15-30 minutes for most projects

---

## 📊 What Changed?

### Architecture

| Aspect            | v1.x                   | v2.0                                         |
| ----------------- | ---------------------- | -------------------------------------------- |
| **Structure**     | Flat                   | Modular (core, features, preprocessing, io)  |
| **CLI**           | Legacy only            | Hydra CLI + Legacy (both supported)          |
| **Pipeline**      | Multi-step             | Unified single-step                          |
| **Configuration** | Command-line args      | Hydra configs + command-line                 |
| **Features**      | Per-tile               | Boundary-aware option                        |
| **Output**        | Patches only           | Patches, Enriched LAZ, or Both               |

---

## 🚀 Migration Strategies

### Strategy 1: Keep Using Legacy CLI (Easiest)

**Best for:** Quick upgrade, minimal changes

```bash
# Your v1.x commands still work!
ign-lidar-hd enrich --input-dir data/ --output output/ --use-gpu
ign-lidar-hd patch --input-dir enriched/ --output patches/
```

**Required Changes:**

- Update import paths in Python code (if any)
- Install v2.0: `pip install --upgrade ign-lidar-hd`

**Time:** 5-10 minutes

---

### Strategy 2: Gradual Migration (Recommended)

**Best for:** Learning v2.0 features while maintaining workflow

1. **Week 1:** Upgrade and use legacy CLI
2. **Week 2:** Try Hydra CLI for new projects
3. **Week 3:** Update existing scripts to Hydra CLI
4. **Week 4:** Adopt new features (boundary-aware, stitching)

**Time:** 4 weeks

---

### Strategy 3: Full Migration (Advanced)

**Best for:** Taking full advantage of v2.0 features

- Migrate to Hydra CLI
- Update all import paths
- Use unified pipeline
- Enable boundary-aware features
- Implement tile stitching

**Time:** 1-2 days for full project migration

---

## 📝 CLI Migration

### Command Comparison

#### Download (No Changes)

```bash
# v1.x (still works)
ign-lidar-hd download --bbox "xmin,ymin,xmax,ymax" --output data/

# v2.0 Hydra CLI (alternative)
ign-lidar-hd download bbox="xmin,ymin,xmax,ymax" output_dir=data/
```

#### Enrich → Process

**v1.x (Legacy - Still Works):**

```bash
ign-lidar-hd enrich \
  --input-dir data/raw/ \
  --output output/enriched/ \
  --use-rgb \
  --compute-ndvi \
  --use-gpu \
  --num-workers 4
```

**v2.0 (Hydra CLI - Recommended):**

```bash
ign-lidar-hd process \
  input_dir=data/raw/ \
  output_dir=output/ \
  preset=balanced \
  processor=gpu \
  features.use_rgb=true \
  features.compute_ndvi=true \
  num_workers=4
```

#### Patch (Now Part of Process)

**v1.x (Two Steps):**

```bash
# Step 1: Enrich
ign-lidar-hd enrich --input-dir data/ --output enriched/

# Step 2: Patch
ign-lidar-hd patch --input-dir enriched/ --output patches/ --patch-size 50
```

**v2.0 (Single Step):**

```bash
# One command does both!
ign-lidar-hd process \
  input_dir=data/ \
  output_dir=output/ \
  dataset.patch_size=50
```

### Complete Mapping Table

| v1.x Command          | v1.x Args                                  | v2.0 Hydra CLI                                            |
| --------------------- | ------------------------------------------ | --------------------------------------------------------- |
| `enrich`              | `--input-dir data/`                        | `input_dir=data/`                                         |
| `enrich`              | `--output out/`                            | `output_dir=out/`                                         |
| `enrich`              | `--use-gpu`                                | `processor=gpu`                                           |
| `enrich`              | `--use-rgb`                                | `features.use_rgb=true`                                   |
| `enrich`              | `--compute-ndvi`                           | `features.compute_ndvi=true`                              |
| `enrich`              | `--num-workers 4`                          | `num_workers=4`                                           |
| `enrich`              | `--preprocess`                             | `preprocess=standard`                                     |
| `enrich`              | `--auto-params`                            | `features.auto_params=true`                               |
| `patch`               | `--patch-size 50`                          | `dataset.patch_size=50`                                   |
| `patch`               | `--points-per-patch 4096`                  | `dataset.points_per_patch=4096`                           |
| `verify`              | `--input-dir data/`                        | `ign-lidar-hd verify input_dir=data/`                     |
| _None_                | _N/A_                                      | `preset=balanced` (NEW - fast/balanced/quality/ultra)     |
| _None_                | _N/A_                                      | `features.boundary_aware=true` (NEW)                      |
| _None_                | _N/A_                                      | `stitching=full` (NEW)                                    |
| _None_                | _N/A_                                      | `output=enriched_only` (NEW - v2.0.1)                     |

---

## 🐍 Python API Migration

### Import Path Changes

**v1.x:**

```python
from ign_lidar import LiDARProcessor
from ign_lidar import FeatureComputer
from ign_lidar import read_laz_file
```

**v2.0:**

```python
from ign_lidar.core import LiDARProcessor
from ign_lidar.features import FeatureComputer
from ign_lidar.io import read_laz_file
```

### Complete Import Mapping

| v1.x Import              | v2.0 Import                             |
| ------------------------ | --------------------------------------- |
| `from ign_lidar import`  | `from ign_lidar.core import`            |
| `LiDARProcessor`         | `from ign_lidar.core import`            |
| `FeatureComputer`        | `from ign_lidar.features import`        |
| `read_laz_file`          | `from ign_lidar.io import`              |
| `write_laz_file`         | `from ign_lidar.io import`              |
| `remove_outliers`        | `from ign_lidar.preprocessing import`   |
| `normalize_ground`       | `from ign_lidar.preprocessing import`   |
| `IGNLidarDataset`        | `from ign_lidar.datasets import`        |

### Code Examples

#### Example 1: Basic Processing

**v1.x:**

```python
from ign_lidar import LiDARProcessor

processor = LiDARProcessor(
    input_dir="data/",
    output_dir="output/",
    use_gpu=True
)

processor.enrich_tiles()
processor.create_patches()
```

**v2.0:**

```python
from ign_lidar.core import LiDARProcessor

processor = LiDARProcessor(
    input_dir="data/",
    output_dir="output/",
    preset="balanced",
    use_gpu=True
)

processor.run()  # Single call!
```

#### Example 2: Feature Computation

**v1.x:**

```python
from ign_lidar import FeatureComputer

computer = FeatureComputer(use_rgb=True)
features = computer.compute(points, colors)
```

**v2.0:**

```python
from ign_lidar.features import FeatureComputer

computer = FeatureComputer(
    use_rgb=True,
    boundary_aware=False  # NEW option
)
features = computer.compute(points, colors)
```

#### Example 3: Reading LAZ Files

**v1.x:**

```python
from ign_lidar import read_laz_file

points, colors = read_laz_file("tile.laz")
```

**v2.0:**

```python
from ign_lidar.io import read_laz_file

# Same interface, but can now read enriched LAZ with features
points, colors, features = read_laz_file("enriched_tile.laz")

# For backward compatibility (no features)
points, colors = read_laz_file("tile.laz")[:2]
```

#### Example 4: Preprocessing

**v1.x:**

```python
from ign_lidar import remove_outliers, normalize_ground

points = remove_outliers(points)
points = normalize_ground(points)
```

**v2.0:**

```python
from ign_lidar.preprocessing import remove_outliers, normalize_ground

# Same interface!
points = remove_outliers(points, method="statistical")
points = normalize_ground(points, max_distance=5.0)
```

---

## ⚙️ Configuration Migration

### v1.x: Command-Line Arguments

```bash
ign-lidar-hd enrich \
  --input-dir data/ \
  --output output/ \
  --use-rgb \
  --compute-ndvi \
  --patch-size 50 \
  --use-gpu \
  --num-workers 4
```

### v2.0 Option 1: Hydra CLI with Overrides

```bash
ign-lidar-hd process \
  input_dir=data/ \
  output_dir=output/ \
  preset=balanced \
  features.use_rgb=true \
  features.compute_ndvi=true \
  dataset.patch_size=50 \
  processor=gpu \
  num_workers=4
```

### v2.0 Option 2: Configuration File

**config.yaml:**

```yaml
defaults:
  - base_config
  - preset: balanced
  - processor: gpu
  - _self_

input_dir: "data/"
output_dir: "output/"

features:
  use_rgb: true
  compute_ndvi: true

dataset:
  patch_size: 50

num_workers: 4
```

**Command:**

```bash
ign-lidar-hd process --config-name config
```

---

## 🆕 New Features to Adopt

### 1. Presets (Recommended)

Instead of specifying all parameters:

```bash
# v2.0 - Use presets!
ign-lidar-hd process input_dir=data/ preset=balanced

# Available presets:
# - fast: Quick processing (5-10 min/tile)
# - balanced: Standard quality (15-20 min/tile) ⭐
# - quality: High quality (30-45 min/tile)
# - ultra: Maximum features (60+ min/tile)
```

### 2. Boundary-Aware Features (NEW)

Eliminate edge artifacts:

```bash
ign-lidar-hd process \
  input_dir=tiles/ \
  output_dir=output/ \
  features.boundary_aware=true \
  features.boundary_buffer=10.0
```

### 3. Tile Stitching (NEW)

Multi-tile workflows:

```bash
ign-lidar-hd process \
  input_dir=tiles/ \
  output_dir=output/ \
  stitching=full \
  features.boundary_aware=true
```

### 4. Enriched LAZ Only Mode (NEW in v2.0.1)

Skip patch generation:

```bash
ign-lidar-hd process \
  input_dir=data/ \
  output_dir=output/ \
  output=enriched_only
```

### 5. Multi-Architecture Support (NEW)

Generate patches for different architectures:

```bash
ign-lidar-hd process \
  input_dir=data/ \
  output_dir=output/ \
  dataset.architecture=octree  # or pointnet++, transformer, sparse_conv
```

---

## 🔧 Automated Migration Script

We provide a migration script to update your Python code:

```bash
# Download migration script
wget https://raw.githubusercontent.com/sducournau/IGN_LIDAR_HD_DATASET/main/scripts/migrate_to_v2.py

# Run on your project
python migrate_to_v2.py /path/to/your/project/

# Preview changes without applying
python migrate_to_v2.py /path/to/your/project/ --dry-run
```

**What it does:**

- ✅ Updates import paths
- ✅ Converts old API calls to new API
- ✅ Adds type hints
- ✅ Creates backup files

---

## 🐛 Common Migration Issues

### Issue 1: Import Errors

**Error:**

```python
ImportError: cannot import name 'LiDARProcessor' from 'ign_lidar'
```

**Fix:**

```python
# Old
from ign_lidar import LiDARProcessor

# New
from ign_lidar.core import LiDARProcessor
```

### Issue 2: Legacy CLI Not Working

**Error:**

```bash
$ ign-lidar-hd enrich --help
Error: No such command 'enrich'
```

**Fix:**

This shouldn't happen! Legacy CLI is supported. Try:

```bash
# Reinstall
pip install --force-reinstall ign-lidar-hd

# Verify installation
ign-lidar-hd --version
```

### Issue 3: Configuration Not Found

**Error:**

```bash
Error: Config 'my_config' not found
```

**Fix:**

Ensure config files are in the correct location:

```
your_project/
├── configs/            # Should be here
│   └── my_config.yaml
└── run.py
```

Or specify full path:

```bash
ign-lidar-hd process --config-path /full/path/to/configs --config-name my_config
```

### Issue 4: GPU Not Working

**Error:**

```
RuntimeError: CUDA not available
```

**Fix:**

```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 📚 Learning Resources

### Must-Read Guides

1. **[Hydra CLI Guide](/guides/hydra-cli)** - Learn the new CLI system
2. **[Configuration System](/guides/configuration-system)** - Master Hydra configs
3. **[Boundary-Aware Features](/features/boundary-aware)** - Eliminate edge artifacts
4. **[Complete Workflow](/guides/complete-workflow)** - End-to-end examples

### Quick Start

- **[Quick Start Guide](/guides/quick-start)** - Get running in 5 minutes
- **[API Reference](/api/core-module)** - Complete API documentation

---

## ❓ FAQ

### Q: Do I need to migrate immediately?

**A:** No! The legacy CLI is fully supported. Migrate when you're ready.

### Q: Will my v1.x scripts break?

**A:** CLI scripts work as-is. Python scripts need import path updates only.

### Q: Can I mix legacy and Hydra CLI?

**A:** Yes! Use whichever is appropriate for each task.

### Q: What if I find bugs?

**A:** Report them on [GitHub Issues](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues). We're committed to backward compatibility.

### Q: How do I rollback to v1.x?

**A:** 

```bash
pip install ign-lidar-hd==1.7.6
```

### Q: Where are the old docs?

**A:** V1.x documentation is archived at `/docs/v1/` on the website.

### Q: Is there a migration support channel?

**A:** Yes! Ask questions in [GitHub Discussions](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/discussions).

---

## ✅ Migration Checklist

### Python Projects

- [ ] Update `requirements.txt`: `ign-lidar-hd>=2.0.1`
- [ ] Run migration script: `python migrate_to_v2.py .`
- [ ] Update imports in all `.py` files
- [ ] Test all workflows
- [ ] Update documentation/README
- [ ] Commit changes

### Shell Scripts

- [ ] Option A: Keep using legacy CLI (no changes!)
- [ ] Option B: Update to Hydra CLI
  - [ ] Replace `--arg value` with `arg=value`
  - [ ] Update command names (enrich → process)
  - [ ] Add presets where appropriate
- [ ] Test all scripts
- [ ] Update documentation

### Documentation

- [ ] Update code examples
- [ ] Add v2.0 features to README
- [ ] Update installation instructions
- [ ] Link to migration guide

---

## 🎉 Success Stories

### Case Study 1: Urban Mapping Project

**Before (v1.x):**

```bash
# 3-step process, 45 min/tile
ign-lidar-hd download --bbox "..." --output raw/
ign-lidar-hd enrich --input-dir raw/ --output enriched/ --use-gpu
ign-lidar-hd patch --input-dir enriched/ --output patches/
```

**After (v2.0):**

```bash
# 1-step process, 20 min/tile with boundary-aware
ign-lidar-hd process \
  input_dir=raw/ \
  output_dir=output/ \
  preset=balanced \
  processor=gpu \
  features.boundary_aware=true
```

**Results:**

- ⚡ 55% faster
- ✅ No edge artifacts
- 🎯 Better ML accuracy

### Case Study 2: Large-Scale Classification

**Migration Time:** 2 hours  
**Benefits:**

- Unified pipeline
- Boundary-aware features improved accuracy by 8%
- Tile stitching enabled seamless large-area processing

---

## 📞 Support

**Need Help?**

- 📖 **Documentation:** [Read the Docs](https://ign-lidar-hd.readthedocs.io)
- 💬 **Discussions:** [GitHub Discussions](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/discussions)
- 🐛 **Issues:** [GitHub Issues](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
- 📧 **Email:** simon.ducournau@gmail.com

---

## 🚀 Next Steps

1. **Read** this guide (you're here! ✅)
2. **Upgrade**: `pip install --upgrade ign-lidar-hd`
3. **Test** with legacy CLI (should just work)
4. **Try** Hydra CLI on a small dataset
5. **Explore** new features (boundary-aware, stitching)
6. **Migrate** gradually to v2.0 patterns

**Welcome to v2.0!** 🎉
