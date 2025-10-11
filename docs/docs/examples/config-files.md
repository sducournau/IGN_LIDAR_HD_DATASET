---
sidebar_position: 2
title: Example Configuration Files
---

# Example Configuration Files

Version 2.3.0 includes four production-ready YAML configuration files for common workflows. These configs provide a simple way to get started without manually specifying all parameters.

## 📁 Location

All example configs are located in the `examples/` directory:

```
examples/
├── README.md
├── config_gpu_processing.yaml
├── config_training_dataset.yaml
├── config_quick_enrich.yaml
└── config_complete.yaml
```

## 🚀 Usage

### Basic Usage

```bash
ign-lidar-hd process \
  --config-file examples/config_training_dataset.yaml \
  input_dir=/path/to/raw \
  output_dir=/path/to/output
```

### Preview Configuration

See what the config does before running:

```bash
ign-lidar-hd process \
  --config-file examples/config_training_dataset.yaml \
  --show-config \
  input_dir=/path/to/raw \
  output_dir=/path/to/output
```

### Override Config Values

CLI arguments override config file values:

```bash
ign-lidar-hd process \
  --config-file examples/config_training_dataset.yaml \
  input_dir=/path/to/raw \
  output_dir=/path/to/output \
  processor.num_points=65536 \
  features.k_neighbors=50
```

## 📝 Available Configurations

### 1. `config_gpu_processing.yaml` - GPU-Accelerated LAZ Enrichment

**Use Case:** Fast feature enrichment with GPU acceleration for GIS analysis

**Key Features:**

- ✅ GPU acceleration enabled
- ✅ Enriched LAZ output only (no patches)
- ✅ Full feature computation (RGB, NIR, NDVI)
- ✅ Preprocessing and stitching enabled

**Configuration:**

```yaml
defaults:
  - override /processor: gpu
  - override /features: full
  - override /preprocess: default
  - override /stitching: enhanced

processor:
  lod_level: LOD2
  architecture: default
  num_points: 16384
  use_gpu: true
  augment: false

features:
  k_neighbors: 20
  use_rgb: true
  compute_ndvi: true

output:
  format: npz
  processing_mode: enriched_only
```

**Usage:**

```bash
ign-lidar-hd process \
  --config-file examples/config_gpu_processing.yaml \
  input_dir=/path/to/raw \
  output_dir=/path/to/enriched
```

**Output:** `tile_enriched.laz` files with computed features

---

### 2. `config_training_dataset.yaml` - ML Training Dataset

**Use Case:** Create ML-ready patches with augmentation for training deep learning models

**Key Features:**

- ✅ Data augmentation (5x per patch)
- ✅ Preprocessing enabled (outlier removal)
- ✅ Tile stitching for boundary consistency
- ✅ Normalized features for deep learning
- ✅ Patches only (no enriched LAZ)

**Configuration:**

```yaml
defaults:
  - override /processor: default
  - override /features: full
  - override /preprocess: default
  - override /stitching: enhanced

processor:
  lod_level: LOD2
  architecture: default
  num_points: 16384
  sampling_method: random
  augment: true
  num_augmentations: 5
  use_gpu: false

features:
  k_neighbors: 20
  use_rgb: true
  compute_ndvi: true
  normalize_xyz: true

output:
  format: npz
  processing_mode: patches_only
```

**Usage:**

```bash
ign-lidar-hd process \
  --config-file examples/config_training_dataset.yaml \
  input_dir=/path/to/raw \
  output_dir=/path/to/training
```

**Output:**

- `tile_patch_0001.npz`, `tile_patch_0001_aug_0.npz`, ...
- Each patch has 5 augmented versions

---

### 3. `config_quick_enrich.yaml` - Quick LAZ Enrichment

**Use Case:** Fastest option for adding features to LAZ files (GIS workflows)

**Key Features:**

- ✅ Minimal features (fastest processing)
- ✅ No preprocessing or stitching
- ✅ Enriched LAZ output only
- ✅ CPU-based (no GPU required)

**Configuration:**

```yaml
defaults:
  - override /processor: cpu_fast
  - override /features: minimal

processor:
  lod_level: LOD2
  architecture: default
  num_points: 16384
  use_gpu: false
  augment: false

features:
  k_neighbors: 10
  use_rgb: false
  compute_ndvi: false

preprocess:
  enabled: false

stitching:
  enabled: false

output:
  format: npz
  processing_mode: enriched_only
```

**Usage:**

```bash
ign-lidar-hd process \
  --config-file examples/config_quick_enrich.yaml \
  input_dir=/path/to/raw \
  output_dir=/path/to/enriched
```

**Output:** `tile_enriched.laz` files with basic geometric features

---

### 4. `config_complete.yaml` - Complete Workflow

**Use Case:** Research projects needing both ML patches AND enriched LAZ files

**Key Features:**

- ✅ Both patches and enriched LAZ output
- ✅ Full preprocessing and stitching
- ✅ RGB, NIR, and NDVI computation
- ✅ Data augmentation enabled

**Configuration:**

```yaml
defaults:
  - override /processor: default
  - override /features: full
  - override /preprocess: default
  - override /stitching: enhanced

processor:
  lod_level: LOD2
  architecture: default
  num_points: 16384
  sampling_method: random
  augment: true
  num_augmentations: 3
  use_gpu: false

features:
  k_neighbors: 20
  use_rgb: true
  compute_ndvi: true
  normalize_xyz: true

output:
  format: npz
  processing_mode: both
```

**Usage:**

```bash
# Option 1: Edit paths in the file, then:
ign-lidar-hd process --config-file examples/config_complete.yaml

# Option 2: Override paths from command line:
ign-lidar-hd process \
  --config-file examples/config_complete.yaml \
  input_dir=/path/to/raw \
  output_dir=/path/to/complete
```

**Output:**

- ML patches: `tile_patch_0001.npz`, `tile_patch_0001_aug_0.npz`, ...
- GIS files: `tile_enriched.laz`

---

## 🔧 Configuration Precedence

The system follows a clear precedence order for configuration values:

1. **Package Defaults** (lowest priority)
   - Built-in defaults from `ign_lidar/configs/`
2. **Custom Config File** (medium priority)
   - Values from your `--config-file`
3. **CLI Overrides** (highest priority)
   - Parameters passed directly on command line

### Example

```bash
# config_training_dataset.yaml has num_points=16384
# This command overrides it to 32768
ign-lidar-hd process \
  --config-file examples/config_training_dataset.yaml \
  input_dir=/path/to/raw \
  output_dir=/path/to/output \
  processor.num_points=32768  # <-- Overrides config file
```

---

## 📊 Comparison Table

| Config                    | Processing Mode | GPU | Augment | RGB/NDVI | Preprocess | Stitching | Speed     | Use Case             |
| ------------------------- | --------------- | --- | ------- | -------- | ---------- | --------- | --------- | -------------------- |
| `config_gpu_processing`   | enriched_only   | ✅  | ❌      | ✅       | ✅         | ✅        | Very Fast | GIS analysis         |
| `config_training_dataset` | patches_only    | ❌  | ✅ (5x) | ✅       | ✅         | ✅        | Medium    | ML training          |
| `config_quick_enrich`     | enriched_only   | ❌  | ❌      | ❌       | ❌         | ❌        | Fastest   | Quick GIS enrichment |
| `config_complete`         | both            | ❌  | ✅ (3x) | ✅       | ✅         | ✅        | Slow      | Research/Full output |

---

## 💡 Tips

### When to Use Each Config

**Use `config_gpu_processing.yaml` when:**

- You have NVIDIA GPU with CUDA support
- You need enriched LAZ files for QGIS/CloudCompare
- Speed is critical
- You don't need ML patches

**Use `config_training_dataset.yaml` when:**

- Training PointNet++, transformers, or other models
- You need data augmentation
- You want boundary-consistent features
- ML is your primary goal

**Use `config_quick_enrich.yaml` when:**

- You just need basic features added to LAZ
- Speed is the top priority
- You're on a laptop or limited hardware
- No ML training needed

**Use `config_complete.yaml` when:**

- You need both ML patches AND enriched LAZ
- Running a research project
- Want to compare ML vs GIS workflows
- Disk space is not a concern

### Customizing Configs

Create your own config based on the examples:

```yaml
# my_custom_config.yaml
defaults:
  - override /processor: gpu
  - override /features: full

processor:
  lod_level: LOD3
  num_points: 32768
  use_gpu: true
  augment: true
  num_augmentations: 10

features:
  k_neighbors: 30
  use_rgb: true
  compute_ndvi: true

output:
  format: "npz,torch" # Multi-format output
  processing_mode: patches_only

input_dir: /mnt/data/lidar_tiles
output_dir: /mnt/data/training_dataset
```

Then use it:

```bash
ign-lidar-hd process --config-file my_custom_config.yaml
```

---

## 🔗 Related Documentation

- [Processing Modes Guide](/guides/processing-modes) - Deep dive into processing modes
- [Hydra CLI Guide](/guides/hydra-cli) - Learn about Hydra configuration system
- [Configuration System](/guides/configuration-system) - Complete config reference
- [GPU Acceleration](/guides/gpu-acceleration) - GPU setup and optimization

---

## 📖 Full Example: Training Workflow

Here's a complete workflow using the example configs:

```bash
# Step 1: Download tiles
ign-lidar-hd download \
  --position 650000 6860000 \
  --radius 5000 \
  data/raw_tiles

# Step 2: Create training dataset with augmentation
ign-lidar-hd process \
  --config-file examples/config_training_dataset.yaml \
  input_dir=data/raw_tiles \
  output_dir=data/training

# Step 3: Verify dataset
ign-lidar-hd verify data/training

# Step 4: Preview statistics
python -c "
import numpy as np
from pathlib import Path

patches = list(Path('data/training').glob('*.npz'))
print(f'Total patches: {len(patches)}')

sample = np.load(patches[0])
print(f'Points per patch: {sample[\"points\"].shape[0]}')
print(f'Features: {sample[\"points\"].shape[1]}')
print(f'Available keys: {list(sample.keys())}')
"
```

This produces a production-ready ML training dataset with:

- ✅ Consistent patch sizes
- ✅ Boundary-aware features
- ✅ 5x augmentation per patch
- ✅ RGB and NDVI features
- ✅ Normalized coordinates
