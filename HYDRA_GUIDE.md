# IGN LiDAR HD v2.0 - Hydra Configuration Guide

## 🎯 Overview

IGN LiDAR HD v2.0 introduces **Hydra** for powerful, composable configuration management. This modernizes the CLI interface and makes experiment workflows much easier.

## 🚀 Quick Start

### Installation

```bash
# Install with Hydra support
pip install ign-lidar-hd==2.0.0-alpha

# Or from source
cd IGN_LIDAR_HD_DATASET
pip install -e .
```

### Basic Usage

```bash
# Process with defaults (CPU, full features)
python -m ign_lidar.cli.hydra_main \
    input_dir=data/raw \
    output_dir=data/patches

# GPU processing
python -m ign_lidar.cli.hydra_main \
    processor=gpu \
    input_dir=data/raw \
    output_dir=data/patches

# Use experiment preset
python -m ign_lidar.cli.hydra_main \
    experiment=pointnet_training \
    input_dir=data/raw \
    output_dir=data/patches_pointnet
```

## 📁 Configuration Structure

```
configs/
├── config.yaml                 # Root configuration
├── processor/                  # Processor configs
│   ├── default.yaml           # CPU (default)
│   ├── gpu.yaml               # GPU acceleration
│   ├── cpu_fast.yaml          # Speed optimized
│   └── memory_constrained.yaml
├── features/                   # Feature configs
│   ├── minimal.yaml
│   ├── full.yaml              # All features (default)
│   ├── pointnet.yaml          # PointNet++ optimized
│   ├── buildings.yaml
│   └── vegetation.yaml
├── preprocess/                 # Preprocessing configs
│   ├── default.yaml
│   ├── aggressive.yaml
│   └── disabled.yaml
├── stitching/                  # Tile stitching configs
│   ├── enabled.yaml
│   └── disabled.yaml          # (default)
├── output/                     # Output format configs
│   ├── default.yaml           # NPZ format
│   ├── hdf5.yaml
│   └── torch.yaml
└── experiment/                 # Experiment presets
    ├── buildings_lod2.yaml
    ├── buildings_lod3.yaml
    ├── vegetation_ndvi.yaml
    ├── pointnet_training.yaml
    ├── semantic_sota.yaml
    └── fast.yaml
```

## 🎨 Experiment Presets

### Available Presets

| Preset              | Description                  | Use Case                     |
| ------------------- | ---------------------------- | ---------------------------- |
| `buildings_lod2`    | LOD2 building classification | Fast building classification |
| `buildings_lod3`    | LOD3 building classification | Detailed building parts      |
| `vegetation_ndvi`   | Vegetation with NDVI         | Vegetation segmentation      |
| `pointnet_training` | PointNet++ optimized         | Neural network training      |
| `semantic_sota`     | Full features SOTA           | Best quality segmentation    |
| `fast`              | Minimal features, fast       | Quick testing/prototyping    |

### Using Presets

```bash
# Buildings LOD2
python -m ign_lidar.cli.hydra_main \
    experiment=buildings_lod2 \
    input_dir=data/raw \
    output_dir=data/patches_buildings

# Vegetation with NDVI
python -m ign_lidar.cli.hydra_main \
    experiment=vegetation_ndvi \
    input_dir=data/raw \
    output_dir=data/patches_vegetation

# PointNet++ training dataset
python -m ign_lidar.cli.hydra_main \
    experiment=pointnet_training \
    input_dir=data/raw \
    output_dir=data/patches_pointnet
```

## ⚙️ Configuration Composition

### Override Individual Parameters

```bash
# Change number of points
python -m ign_lidar.cli.hydra_main \
    processor.num_points=32768 \
    input_dir=data/raw \
    output_dir=data/patches

# Change multiple parameters
python -m ign_lidar.cli.hydra_main \
    processor.num_points=32768 \
    processor.patch_size=200.0 \
    features.k_neighbors=30 \
    input_dir=data/raw \
    output_dir=data/patches
```

### Mix Configs and Overrides

```bash
# GPU + PointNet features + custom points
python -m ign_lidar.cli.hydra_main \
    processor=gpu \
    features=pointnet \
    processor.num_points=32768 \
    input_dir=data/raw \
    output_dir=data/patches

# Experiment preset + overrides
python -m ign_lidar.cli.hydra_main \
    experiment=buildings_lod2 \
    processor.use_gpu=true \
    processor.num_workers=8 \
    input_dir=data/raw \
    output_dir=data/patches
```

## 🔄 Multi-Run (Parameter Sweeps)

Hydra supports multi-run mode for parameter sweeps:

```bash
# Sweep over different num_points
python -m ign_lidar.cli.hydra_main -m \
    processor.num_points=4096,8192,16384,32768 \
    input_dir=data/raw \
    output_dir=data/patches

# Sweep over processors and features
python -m ign_lidar.cli.hydra_main -m \
    processor=default,gpu \
    features=minimal,full \
    input_dir=data/raw \
    output_dir=data/patches

# Complex sweep
python -m ign_lidar.cli.hydra_main -m \
    experiment=buildings_lod2,vegetation_ndvi \
    processor.num_points=8192,16384 \
    input_dir=data/raw \
    output_dir=data/patches
```

## 📄 Custom Configuration Files

### Create Custom Config

```yaml
# my_config.yaml
defaults:
  - config
  - override processor: gpu
  - override features: pointnet

processor:
  num_points: 32768
  num_workers: 16

features:
  use_rgb: true
  use_infrared: true
```

### Use Custom Config

```bash
python -m ign_lidar.cli.hydra_main \
    --config-name my_config \
    input_dir=data/raw \
    output_dir=data/patches
```

## 🔍 Configuration Inspection

### View Current Configuration

```bash
# Print configuration without processing
python -m ign_lidar.cli.hydra_main \
    experiment=pointnet_training \
    --cfg job
```

### List Available Presets

```bash
python -m ign_lidar.cli.hydra_main info
```

## 📊 Output & Logging

### Output Directory Structure

```
outputs/
└── 2025-10-07/
    └── 14-30-00/
        ├── .hydra/
        │   ├── config.yaml      # Complete resolved config
        │   ├── hydra.yaml
        │   └── overrides.yaml   # CLI overrides used
        └── main.log
```

Each run creates a timestamped directory with:

- Complete configuration snapshot
- Processing logs
- Overrides applied

### Multi-run Output

```
multirun/
└── 2025-10-07/
    └── 14-30-00/
        ├── 0/  # First run
        ├── 1/  # Second run
        └── 2/  # Third run
```

## 🆚 Migration from v1.7.7

### Old CLI (v1.7.7)

```bash
ign-lidar-hd process \
    --input-dir data/raw \
    --output data/patches \
    --num-points 16384 \
    --use-gpu \
    --add-rgb \
    --k-neighbors 20
```

### New CLI (v2.0)

```bash
python -m ign_lidar.cli.hydra_main \
    processor=gpu \
    processor.num_points=16384 \
    features.use_rgb=true \
    features.k_neighbors=20 \
    input_dir=data/raw \
    output_dir=data/patches
```

### Or Use Preset

```bash
python -m ign_lidar.cli.hydra_main \
    experiment=pointnet_training \
    input_dir=data/raw \
    output_dir=data/patches
```

## 🎓 Advanced Usage

### Nested Config Override

```bash
# Override nested values
python -m ign_lidar.cli.hydra_main \
    processor.lod_level=LOD3 \
    features.mode=full \
    features.include_extra=true \
    preprocess.enabled=true \
    preprocess.sor_k=20 \
    preprocess.sor_std=1.5 \
    input_dir=data/raw \
    output_dir=data/patches
```

### Bounding Box Filtering

```bash
python -m ign_lidar.cli.hydra_main \
    bbox.xmin=650000 \
    bbox.ymin=6850000 \
    bbox.xmax=660000 \
    bbox.ymax=6860000 \
    input_dir=data/raw \
    output_dir=data/patches
```

### Enable Tile Stitching

```bash
python -m ign_lidar.cli.hydra_main \
    stitching=enabled \
    stitching.buffer_size=15.0 \
    input_dir=data/raw \
    output_dir=data/patches
```

## 🐛 Troubleshooting

### Configuration Not Found

```
Error: Could not find config file
```

**Solution**: Make sure you're running from the project root or specify config path:

```bash
python -m ign_lidar.cli.hydra_main \
    --config-path /path/to/configs \
    --config-name config \
    input_dir=data/raw \
    output_dir=data/patches
```

### Missing Required Parameters

```
Error: Missing mandatory value: input_dir
```

**Solution**: Provide required parameters:

```bash
python -m ign_lidar.cli.hydra_main \
    input_dir=data/raw \
    output_dir=data/patches
```

### Invalid Override

```
Error: Could not override 'processor.invalid_param'
```

**Solution**: Check parameter name and structure. Use `--cfg job` to see available options.

## 📚 Resources

- **Hydra Documentation**: https://hydra.cc/
- **OmegaConf Documentation**: https://omegaconf.readthedocs.io/
- **Project Repository**: https://github.com/sducournau/IGN_LIDAR_HD_DATASET

## 🔧 Tips & Best Practices

1. **Start with presets**: Use experiment presets as starting points
2. **Incremental overrides**: Override only what you need
3. **Use multi-run**: Test multiple configurations efficiently
4. **Check outputs**: Review `.hydra/config.yaml` to verify settings
5. **Create custom presets**: Save common configurations as new presets
6. **Version control**: Track config files in git for reproducibility

---

**Version**: 2.0.0-alpha  
**Date**: October 7, 2025  
**Maintainer**: @sducournau
