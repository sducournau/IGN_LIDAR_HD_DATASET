# Example Configuration Files

This directory contains example configuration files for common use cases with IGN LiDAR HD.

---

## 📝 Available Examples

### 1. `config_gpu_processing.yaml` - GPU-Accelerated Processing

Fast processing with GPU acceleration for enriched LAZ creation.

**Use case:** GIS analysis with GPU hardware

**Features:**

- GPU acceleration enabled
- Enriched LAZ output only (no patches)
- Full feature computation including RGB and NDVI

**Usage:**

```bash
ign-lidar-hd process \
  --config-file examples/config_gpu_processing.yaml \
  input_dir=/path/to/raw \
  output_dir=/path/to/enriched
```

---

### 2. `config_training_dataset.yaml` - ML Training Dataset

Create ML-ready patches with augmentation for training deep learning models.

**Use case:** Training PointNet++, transformers, etc.

**Features:**

- Data augmentation (5x per patch)
- Preprocessing enabled (outlier removal)
- Tile stitching for boundary consistency
- Normalized features for deep learning

**Usage:**

```bash
ign-lidar-hd process \
  --config-file examples/config_training_dataset.yaml \
  input_dir=/path/to/raw \
  output_dir=/path/to/training
```

---

### 3. `config_quick_enrich.yaml` - Quick LAZ Enrichment

Fastest option for adding features to LAZ files (GIS workflows).

**Use case:** Quick feature enrichment for QGIS/CloudCompare

**Features:**

- Minimal features (fastest)
- No preprocessing or stitching
- Enriched LAZ output only

**Usage:**

```bash
ign-lidar-hd process \
  --config-file examples/config_quick_enrich.yaml \
  input_dir=/path/to/raw \
  output_dir=/path/to/enriched
```

---

### 4. `config_complete.yaml` - Complete Workflow

Create both ML patches AND enriched LAZ files.

**Use case:** Research projects needing both outputs

**Features:**

- Both patches and enriched LAZ
- Full preprocessing and stitching
- RGB, NIR, and NDVI computation

**Usage:**

```bash
# Edit paths in the file first, then:
ign-lidar-hd process --config-file examples/config_complete.yaml

# Or override paths:
ign-lidar-hd process \
  --config-file examples/config_complete.yaml \
  input_dir=/path/to/raw \
  output_dir=/path/to/output
```

---

## 🛠️ Customizing Configurations

### Preview Before Running

Always preview your config before processing:

```bash
ign-lidar-hd process \
  --config-file examples/config_training_dataset.yaml \
  --show-config
```

### Override Any Parameter

CLI overrides have the highest priority:

```bash
ign-lidar-hd process \
  --config-file examples/config_training_dataset.yaml \
  processor.use_gpu=true \
  processor.num_workers=8 \
  output.processing_mode=both
```

### Create Your Own

Copy an example and modify it:

```bash
cp examples/config_training_dataset.yaml my_project_config.yaml
vim my_project_config.yaml

# Then use it:
ign-lidar-hd process --config-file my_project_config.yaml
```

---

## 📊 Configuration Precedence

Settings are applied in this order (highest priority last):

1. **Package defaults** - Built-in defaults from `ign_lidar/configs/`
2. **Custom config file** - Your file specified with `--config-file`
3. **CLI overrides** - Command-line `key=value` arguments

Example:

```bash
# Custom file sets num_workers=4, override to 8
ign-lidar-hd process \
  -c examples/config_training_dataset.yaml \
  processor.num_workers=8  # This wins!
```

---

## 🎯 Quick Reference

| Config File                    | Mode          | GPU | Augment | Best For    |
| ------------------------------ | ------------- | --- | ------- | ----------- |
| `config_gpu_processing.yaml`   | enriched_only | ✅  | ❌      | GIS + GPU   |
| `config_training_dataset.yaml` | patches_only  | ❌  | ✅      | ML training |
| `config_quick_enrich.yaml`     | enriched_only | ❌  | ❌      | Fast GIS    |
| `config_complete.yaml`         | both          | ❌  | ❌      | Everything  |

---

## 💡 Tips

1. **Start with an example** - Copy and modify rather than creating from scratch
2. **Preview first** - Always use `--show-config` before processing
3. **Use overrides** - Keep configs generic, set paths via CLI
4. **Version control** - Commit your custom configs to track experiments

---

## 🔗 See Also

- [Processing Modes Usage Guide](../PROCESSING_MODES_USAGE.md)
- [Phase 1 Complete Summary](../PHASE1_COMPLETE.md)
- [Main Documentation](../docs/)

---

_Example configs for IGN LiDAR HD v2.3.0_
