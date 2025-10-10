---
sidebar_position: 13
title: Multi-Format Output
---

# Multi-Format Output Support

Save patches in multiple formats simultaneously with a single processing run.

:::tip New in v2.2.0
Multi-format output is a new feature in version 2.2.0 that allows you to generate patches in multiple formats at once.
:::

---

## 🎯 Overview

The multi-format output feature enables you to save your processed patches in multiple formats with a single command. This is particularly useful when you need:

- **Training data** in one format (e.g., HDF5)
- **Visualization** in another format (e.g., LAZ)
- **Experimentation** with different formats without reprocessing

---

## 📦 Supported Formats

| Format      | Extension | Description                 | Use Case                    |
| ----------- | --------- | --------------------------- | --------------------------- |
| **NPZ**     | `.npz`    | NumPy compressed archives   | Default, fast I/O           |
| **HDF5**    | `.h5`     | Hierarchical data with gzip | Large datasets, compression |
| **PyTorch** | `.pt`     | PyTorch tensor files        | Direct PyTorch training     |
| **LAZ**     | `.laz`    | LAZ point clouds            | Visualization, QA           |

---

## 🚀 Quick Start

### Single Format

```bash
# NPZ (default)
ign-lidar-hd process \
  input_dir=data/tiles \
  output_dir=data/patches \
  output.format=npz

# HDF5
ign-lidar-hd process \
  input_dir=data/tiles \
  output_dir=data/patches \
  output.format=hdf5

# PyTorch (requires torch installed)
ign-lidar-hd process \
  input_dir=data/tiles \
  output_dir=data/patches \
  output.format=torch

# LAZ for visualization
ign-lidar-hd process \
  input_dir=data/tiles \
  output_dir=data/patches \
  output.format=laz
```

### Multi-Format

```bash
# HDF5 + LAZ (training + visualization)
ign-lidar-hd process \
  input_dir=data/tiles \
  output_dir=data/patches \
  output.format=hdf5,laz

# NPZ + PyTorch
ign-lidar-hd process \
  input_dir=data/tiles \
  output_dir=data/patches \
  output.format=npz,torch

# All formats (experimentation)
ign-lidar-hd process \
  input_dir=data/tiles \
  output_dir=data/patches \
  output.format=npz,hdf5,torch,laz
```

---

## 📊 Format Comparison

### NPZ (NumPy Compressed)

**Pros:**

- ✅ Fast I/O
- ✅ No external dependencies
- ✅ Good compression
- ✅ Simple format

**Cons:**

- ❌ Not hierarchical
- ❌ Limited metadata support

**Best for:** General-purpose training, quick experiments

### HDF5 (Hierarchical Data Format)

**Pros:**

- ✅ Hierarchical structure
- ✅ Excellent compression (gzip)
- ✅ Rich metadata support
- ✅ Industry standard

**Cons:**

- ❌ Slightly slower I/O
- ❌ Requires h5py

**Best for:** Large datasets, organized data, long-term storage

### PyTorch Tensors

**Pros:**

- ✅ Native PyTorch format
- ✅ Direct loading in training
- ✅ GPU-ready
- ✅ No conversion needed

**Cons:**

- ❌ Requires PyTorch
- ❌ Larger file size
- ❌ PyTorch-specific

**Best for:** PyTorch-based training pipelines

### LAZ (Point Cloud)

**Pros:**

- ✅ Visualization in CloudCompare, QGIS
- ✅ Industry-standard format
- ✅ Includes features as extra dimensions
- ✅ Good compression
- ✅ Widely supported

**Cons:**

- ❌ Not optimized for training
- ❌ Limited to point cloud structure

**Best for:** Visualization, quality assurance, sharing data

---

## 🔧 Configuration

### YAML Configuration

```yaml
output:
  format: hdf5,laz # Comma-separated formats
  save_enriched_laz: true
  save_metadata: true
  compression: 6 # gzip level for HDF5
```

### Command Line Override

```bash
ign-lidar-hd process \
  --config-name my_config \
  output.format=npz,laz
```

---

## 💡 Use Cases

### 1. Training + Visualization

```bash
# Save HDF5 for training and LAZ for visualization
ign-lidar-hd process \
  input_dir=data/tiles \
  output_dir=data/patches \
  output.format=hdf5,laz \
  features.add_rgb=true
```

**Result:**

- `patch_0001.h5` - For training
- `patch_0001.laz` - Open in CloudCompare

### 2. Framework Comparison

```bash
# Save both NPZ and PyTorch for framework comparison
ign-lidar-hd process \
  input_dir=data/tiles \
  output_dir=data/patches \
  output.format=npz,torch
```

**Result:**

- `patch_0001.npz` - For TensorFlow, scikit-learn
- `patch_0001.pt` - For PyTorch

### 3. Experimentation

```bash
# Save all formats to test which works best
ign-lidar-hd process \
  input_dir=data/tiles \
  output_dir=data/patches \
  output.format=npz,hdf5,torch,laz
```

**Result:** All formats available for testing

### 4. Production Pipeline

```yaml
# Production config with HDF5 + LAZ
output:
  format: hdf5,laz
  save_enriched_laz: true
  save_metadata: true

features:
  add_rgb: true
  add_infrared: true
```

---

## 🔄 Converting Between Formats

### HDF5 to LAZ Conversion

Convert existing HDF5 patches to LAZ for visualization:

```bash
# Convert single file
python scripts/convert_hdf5_to_laz.py patch_0001.h5

# Convert directory
python scripts/convert_hdf5_to_laz.py patches/ patches_laz/

# Inspect HDF5 structure
python scripts/convert_hdf5_to_laz.py --inspect patch_0001.h5
```

### Python API

```python
import numpy as np
import h5py
import torch

# Load NPZ
data_npz = np.load('patch_0001.npz')
points = data_npz['points']
features = data_npz['features']

# Load HDF5
with h5py.File('patch_0001.h5', 'r') as f:
    points = f['points'][:]
    features = f['features'][:]

# Load PyTorch
data_pt = torch.load('patch_0001.pt')
points = data_pt['points']
features = data_pt['features']
```

---

## 📈 Performance Impact

### Single Format

- **Overhead**: None compared to previous versions
- **Memory**: Same as before
- **Speed**: Same as before

### Multi-Format

- **Overhead**: ~10-20% slower (one-time saving overhead)
- **Memory**: Minimal increase (incremental saving)
- **Speed**: One processing run for multiple outputs

**Example timings (1000 patches):**

- Single format: 10 minutes
- Two formats: 11 minutes (+10%)
- Four formats: 12 minutes (+20%)

---

## 🐛 Troubleshooting

### PyTorch Format Not Available

**Error:** `PyTorch format requires torch to be installed`

**Solution:**

```bash
pip install torch
```

### HDF5 Files Not Generated (v2.1.x or earlier)

**Problem:** Critical bug in versions before v2.2.0

**Solution:**

```bash
# Upgrade to v2.2.0+
pip install --upgrade ign-lidar-hd

# Reprocess with HDF5 format
ign-lidar-hd process \
  input_dir=data/tiles \
  output_dir=data/patches \
  output.format=hdf5
```

### Invalid Format String

**Error:** `Unsupported output format: 'xyz'`

**Solution:** Use only supported formats: `npz`, `hdf5`, `torch`, `pytorch`, `laz`

---

## 📚 Related Documentation

- [Output Configuration Reference](/examples/config-reference#output-configuration)
- [HDF5 to LAZ Conversion Guide](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/blob/main/scripts/CONVERT_HDF5_TO_LAZ.md)
- [Release Notes v2.2.0](/release-notes/v2.2.0)

---

## 💡 Best Practices

### 1. Choose Formats Based on Workflow

```yaml
# Development: Use LAZ for visualization
output.format: laz

# Training: Use HDF5 for compression
output.format: hdf5

# Production: Use both
output.format: hdf5,laz
```

### 2. Avoid Unnecessary Formats

Don't save all formats unless needed - it increases processing time and disk usage.

### 3. Use Compression for HDF5

```yaml
output:
  format: hdf5
  compression: 6 # gzip level (1-9)
```

### 4. Test with Small Dataset First

```bash
# Test with 1-2 tiles first
ign-lidar-hd process \
  input_dir=data/test_tiles \
  output_dir=data/test_patches \
  output.format=npz,hdf5,laz
```

---

## 🎯 Summary

Multi-format output provides:

- ✅ **Flexibility** - Multiple formats from one run
- ✅ **Efficiency** - No reprocessing needed
- ✅ **Compatibility** - Support for all major formats
- ✅ **Convenience** - One command, multiple outputs

**Recommended combinations:**

- **Training:** `hdf5` or `npz`
- **Training + QA:** `hdf5,laz`
- **Experimentation:** `npz,torch,laz`
- **Production:** `hdf5,laz`
