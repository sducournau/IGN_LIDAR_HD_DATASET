---
sidebar_position: 3
title: Processing Modes
---

# Processing Modes Guide

Version 2.3.0 introduces three explicit processing modes that replace the old confusing boolean flags (`save_enriched_laz`, `only_enriched_laz`). This makes workflows clearer and more intuitive.

## 🎯 Overview

The three processing modes are:

1. **`patches_only`** (default) - Generate ML training patches
2. **`both`** - Generate both patches and enriched LAZ files
3. **`enriched_only`** - Generate only enriched LAZ files for GIS workflows

## 📋 Mode Comparison

| Mode            | Patches | Enriched LAZ | Use Case             | Speed   |
| --------------- | ------- | ------------ | -------------------- | ------- |
| `patches_only`  | ✅      | ❌           | ML training          | Fast    |
| `both`          | ✅      | ✅           | Research/Full output | Slowest |
| `enriched_only` | ❌      | ✅           | GIS analysis         | Fastest |

---

## 🚀 Mode 1: Patches Only

**Default mode** for creating ML training datasets.

### What It Does

- Extracts fixed-size patches (e.g., 16K or 32K points)
- Computes features for each patch
- Applies data augmentation (optional)
- Saves patches in NPZ/HDF5/PyTorch/LAZ format
- **Does NOT** save enriched full tiles

### When to Use

- Training deep learning models (PointNet++, transformers, etc.)
- Creating benchmark datasets
- Running experiments with different patch sizes
- You don't need full tile visualization

### CLI Usage

```bash
# Explicit mode (recommended)
ign-lidar-hd process \
  input_dir=data/raw \
  output_dir=data/patches \
  processor.processing_mode=patches_only

# Default behavior (mode can be omitted)
ign-lidar-hd process \
  input_dir=data/raw \
  output_dir=data/patches
```

### With Example Config

```bash
ign-lidar-hd process \
  --config-file examples/config_training_dataset.yaml \
  input_dir=data/raw \
  output_dir=data/patches
```

### Python API

```python
from ign_lidar.core.processor import LiDARTileProcessor
from ign_lidar.config.schemas import ProcessorConfig, OutputConfig

processor = LiDARTileProcessor(
    processor_config=ProcessorConfig(),
    output_config=OutputConfig(processing_mode="patches_only")
)

processor.process_tile("tile.laz", "output/")
```

### Output Structure

```
output/
├── tile_patch_0001.npz
├── tile_patch_0002.npz
├── tile_patch_0003.npz
├── tile_patch_0001_aug_0.npz  # If augmentation enabled
├── tile_patch_0001_aug_1.npz
└── ...
```

### Example

```bash
# Complete training dataset workflow
ign-lidar-hd process \
  input_dir=data/raw_tiles \
  output_dir=data/training \
  processor.processing_mode=patches_only \
  processor.num_points=16384 \
  processor.augment=true \
  processor.num_augmentations=5 \
  features.use_rgb=true \
  features.compute_ndvi=true
```

**Result:** ML-ready patches with 5x augmentation, RGB, and NDVI features.

---

## 🔄 Mode 2: Both

Generate **both** patches and enriched LAZ files.

### What It Does

- Extracts patches (as in `patches_only`)
- **ALSO** saves the complete tile as enriched LAZ
- Enriched LAZ contains all computed features as extra dimensions
- Useful when you need both ML training and GIS visualization

### When to Use

- Research projects requiring multiple output formats
- When you need to visualize patches in QGIS/CloudCompare
- Comparing patch-based vs full-tile analysis
- Creating comprehensive datasets for publication

### CLI Usage

```bash
ign-lidar-hd process \
  input_dir=data/raw \
  output_dir=data/both \
  processor.processing_mode=both
```

### With Example Config

```bash
ign-lidar-hd process \
  --config-file examples/config_complete.yaml \
  input_dir=data/raw \
  output_dir=data/both
```

### Python API

```python
from ign_lidar.core.processor import LiDARTileProcessor
from ign_lidar.config.schemas import ProcessorConfig, OutputConfig

processor = LiDARTileProcessor(
    processor_config=ProcessorConfig(),
    output_config=OutputConfig(processing_mode="both")
)

processor.process_tile("tile.laz", "output/")
```

### Output Structure

```
output/
├── tile_patch_0001.npz
├── tile_patch_0002.npz
├── tile_patch_0003.npz
├── tile_enriched.laz          # <-- Full tile with features
└── ...
```

### Example

```bash
# Complete research workflow
ign-lidar-hd process \
  input_dir=data/raw_tiles \
  output_dir=data/research \
  processor.processing_mode=both \
  processor.num_points=32768 \
  processor.augment=true \
  processor.num_augmentations=3 \
  features.use_rgb=true \
  features.compute_ndvi=true \
  stitching.enabled=true
```

**Result:**

- ML patches for training
- Enriched LAZ for QGIS visualization
- Boundary-consistent features

---

## 🗺️ Mode 3: Enriched Only

Generate **only** enriched LAZ files (no patches).

### What It Does

- Computes features for the entire tile
- Saves enriched LAZ with features as extra dimensions
- **Does NOT** extract patches
- Fastest mode (no patch sampling/extraction overhead)

### When to Use

- GIS analysis in QGIS/CloudCompare/ArcGIS
- Feature engineering exploration
- Visualization and quality checking
- When you don't need ML patches
- Quick processing for large tile sets

### CLI Usage

```bash
ign-lidar-hd process \
  input_dir=data/raw \
  output_dir=data/enriched \
  processor.processing_mode=enriched_only
```

### With Example Configs

```bash
# Fast enrichment (minimal features, no preprocessing)
ign-lidar-hd process \
  --config-file examples/config_quick_enrich.yaml \
  input_dir=data/raw \
  output_dir=data/enriched

# GPU-accelerated enrichment (full features)
ign-lidar-hd process \
  --config-file examples/config_gpu_processing.yaml \
  input_dir=data/raw \
  output_dir=data/enriched
```

### Python API

```python
from ign_lidar.core.processor import LiDARTileProcessor
from ign_lidar.config.schemas import ProcessorConfig, OutputConfig

processor = LiDARTileProcessor(
    processor_config=ProcessorConfig(use_gpu=True),
    output_config=OutputConfig(processing_mode="enriched_only")
)

processor.process_tile("tile.laz", "output/")
```

### Output Structure

```
output/
└── tile_enriched.laz
```

### Example: Fast GIS Enrichment

```bash
# Minimal features for quick visualization
ign-lidar-hd process \
  input_dir=data/raw_tiles \
  output_dir=data/qgis \
  processor.processing_mode=enriched_only \
  features.k_neighbors=10 \
  features.use_rgb=false \
  preprocess.enabled=false \
  stitching.enabled=false
```

**Result:** LAZ files with basic geometric features, ready for QGIS.

### Example: GPU-Accelerated Full Features

```bash
# Full features with GPU acceleration
ign-lidar-hd process \
  input_dir=data/raw_tiles \
  output_dir=data/enriched \
  processor.processing_mode=enriched_only \
  processor.use_gpu=true \
  features.k_neighbors=20 \
  features.use_rgb=true \
  features.compute_ndvi=true \
  preprocess.enabled=true \
  stitching.enabled=true
```

**Result:** High-quality enriched LAZ with all features for advanced GIS analysis.

---

## 🔄 Migration from Old API

The new processing modes replace the old confusing boolean flags.

### Old API (Still Works, Deprecated)

```python
# Old way - confusing!
processor = LiDARTileProcessor(
    save_enriched_laz=True,
    only_enriched_laz=True  # What does this combination mean?
)
```

```bash
# Old CLI - unclear
ign-lidar-hd process \
  input_dir=data/ \
  output.save_enriched_laz=true \
  output.only_enriched_laz=true
```

### New API (Recommended)

```python
# New way - clear and explicit!
processor = LiDARTileProcessor(
    processing_mode="enriched_only"  # Crystal clear!
)
```

```bash
# New CLI - intuitive
ign-lidar-hd process \
  input_dir=data/ \
  processor.processing_mode=enriched_only
```

### Migration Table

| Old Flags                                          | New Mode        |
| -------------------------------------------------- | --------------- |
| `save_enriched_laz=False, only_enriched_laz=False` | `patches_only`  |
| `save_enriched_laz=True, only_enriched_laz=False`  | `both`          |
| `save_enriched_laz=True, only_enriched_laz=True`   | `enriched_only` |

### Deprecation Warnings

If you use the old flags, you'll see helpful warnings:

```
⚠️ DeprecationWarning: Parameter 'save_enriched_laz' is deprecated.
   Use 'processing_mode="both"' or 'processing_mode="enriched_only"' instead.

⚠️ DeprecationWarning: Parameter 'only_enriched_laz' is deprecated.
   Use 'processing_mode="enriched_only"' instead.
```

The old flags will be removed in **v3.0.0**.

---

## 💡 Common Patterns

### Pattern 1: GPU Training Dataset

```bash
ign-lidar-hd process \
  input_dir=data/raw \
  output_dir=data/training \
  processor.processing_mode=patches_only \
  processor.use_gpu=true \
  processor.num_points=32768 \
  processor.augment=true \
  processor.num_augmentations=5 \
  features.sampling_method=fps
```

### Pattern 2: Quick GIS Check

```bash
ign-lidar-hd process \
  --config-file examples/config_quick_enrich.yaml \
  input_dir=data/raw \
  output_dir=data/qgis
```

### Pattern 3: Complete Research Dataset

```bash
ign-lidar-hd process \
  --config-file examples/config_complete.yaml \
  input_dir=data/raw \
  output_dir=data/research \
  processor.augment=true \
  processor.num_augmentations=3
```

### Pattern 4: GPU-Accelerated GIS

```bash
ign-lidar-hd process \
  --config-file examples/config_gpu_processing.yaml \
  input_dir=data/raw \
  output_dir=data/enriched \
  features.use_rgb=true \
  features.compute_ndvi=true
```

---

## 🔍 Verification

### Check Processing Mode

Preview your configuration before running:

```bash
ign-lidar-hd process \
  --config-file examples/config_training_dataset.yaml \
  --show-config \
  input_dir=data/raw \
  output_dir=data/patches
```

Look for:

```yaml
output:
  processing_mode: patches_only  # <-- Verify this matches your intent
  format: npz
  ...
```

### Verify Output

After processing:

```bash
# patches_only: Only patches exist
ls output/*.npz

# both: Both patches and enriched LAZ exist
ls output/*.npz
ls output/*_enriched.laz

# enriched_only: Only enriched LAZ exists
ls output/*_enriched.laz
```

---

## 📊 Performance Comparison

Approximate processing times for a 17M point tile:

| Mode            | Time (CPU) | Time (GPU) | Output Size |
| --------------- | ---------- | ---------- | ----------- |
| `patches_only`  | 3-4 min    | 1-2 min    | ~500 MB     |
| `both`          | 4-5 min    | 2-3 min    | ~1.5 GB     |
| `enriched_only` | 2-3 min    | 30-60 sec  | ~1 GB       |

**Note:** `enriched_only` is fastest because it skips patch sampling/extraction.

---

## 🔗 Related Documentation

- [Example Configuration Files](/examples/config-files) - Production-ready YAML configs
- [Configuration System](/guides/configuration-system) - Complete config reference
- [Hydra CLI Guide](/guides/hydra-cli) - Advanced CLI usage
- [GPU Acceleration](/guides/gpu-acceleration) - GPU setup and optimization

---

## 🎓 Best Practices

1. **Use `patches_only` by default** - Fastest for ML workflows
2. **Use config files** - More readable than long CLI commands
3. **Preview with `--show-config`** - Verify settings before processing
4. **Use `enriched_only` for GIS** - Skip unnecessary patch extraction
5. **Use `both` sparingly** - Only when you truly need both outputs
6. **Migrate to new API** - Old flags will be removed in v3.0

---

## 🆘 Troubleshooting

### Issue: "Unknown processing mode"

**Cause:** Typo in mode name

**Solution:** Use exactly: `patches_only`, `both`, or `enriched_only`

```bash
# ❌ Wrong
processor.processing_mode=patch_only  # Missing 'es'

# ✅ Correct
processor.processing_mode=patches_only
```

### Issue: "Both patches and LAZ files created when I wanted only patches"

**Cause:** Using old `save_enriched_laz=true` flag

**Solution:** Use new `processing_mode` parameter

```bash
# ❌ Old way (deprecated)
output.save_enriched_laz=false

# ✅ New way
processor.processing_mode=patches_only
```

### Issue: "No patches created"

**Cause:** Using `enriched_only` mode

**Solution:** Switch to `patches_only` or `both`

```bash
# Change from:
processor.processing_mode=enriched_only

# To:
processor.processing_mode=patches_only
```

---

## 📝 Summary

| Goal                    | Mode            | Config File               |
| ----------------------- | --------------- | ------------------------- |
| ML training             | `patches_only`  | `config_training_dataset` |
| GIS analysis            | `enriched_only` | `config_quick_enrich`     |
| GIS with GPU            | `enriched_only` | `config_gpu_processing`   |
| Research (both outputs) | `both`          | `config_complete`         |

The new processing modes make it clear **what** you're generating, **when** to use each mode, and **how** to configure them. Say goodbye to confusing boolean flags! 🎉
