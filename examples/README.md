# Configuration Examples

**Last Updated:** October 15, 2025

This directory contains example configurations and helper scripts for the IGN LiDAR HD processing library.

## 🚀 Quick Start

All configurations use Hydra. Run commands from the repository root:

```bash
ign-lidar-hd process experiment=EXPERIMENT_NAME input_dir=... output_dir=...
```

**See available experiments:**

```bash
ign-lidar-hd info
```

## 📂 Directory Contents

### ✅ Active Example Configs
- `config_architectural_analysis.yaml` - Architectural style analysis example
- `config_architectural_training.yaml` - Training with architectural features

### 📜 Python Examples  
- `example_architectural_styles.py` - Architectural style detection API demo
- `test_ground_truth_module.py` - Ground truth integration example

### 🔧 Utilities
- `merge_multiscale_dataset.py` - Merge multi-scale patches
- `run_multiscale_training.sh` - Multi-scale training workflow

### 📚 Documentation
- `ARCHITECTURAL_CONFIG_REFERENCE.md`
- `ARCHITECTURAL_STYLES_README.md`  
- `MULTISCALE_QUICK_REFERENCE.md`
- `MULTI_SCALE_TRAINING_STRATEGY.md`

### 🗄️ Archive
- `archive/` - Legacy configs (see [archive/README.md](archive/README.md))
- `archive/OLD_README.md` - Previous version of this file

## 🎯 Common Experiments

### Building Classification

```bash
# LOD2 (simple)
ign-lidar-hd process experiment=buildings_lod2 input_dir=data/raw output_dir=data/lod2

# LOD3 (detailed)
ign-lidar-hd process experiment=buildings_lod3 input_dir=data/raw output_dir=data/lod3
```

### Multi-Scale Datasets

```bash
# 50m patches (fine details)
ign-lidar-hd process experiment=dataset_50m input_dir=data/raw output_dir=data/50m

# 100m patches (balanced)
ign-lidar-hd process experiment=dataset_100m input_dir=data/raw output_dir=data/100m

# 150m patches (full context)  
ign-lidar-hd process experiment=dataset_150m input_dir=data/raw output_dir=data/150m
```

### Special Cases

```bash
# Fast testing
ign-lidar-hd process experiment=fast input_dir=data/test output_dir=data/output

# With ground truth
ign-lidar-hd process experiment=ground_truth_patches input_dir=data/enriched output_dir=data/gt

# Classify enriched tiles
ign-lidar-hd process experiment=classify_enriched_tiles input_dir=data/enriched output_dir=data/classified
```

## 🎨 Overriding Parameters

```bash
# Change patch size
ign-lidar-hd process experiment=buildings_lod3 processor.patch_size=75 input_dir=... output_dir=...

# Enable GPU
ign-lidar-hd process experiment=buildings_lod2 processor.use_gpu=true input_dir=... output_dir=...

# Multiple overrides
ign-lidar-hd process experiment=buildings_lod3 \
  processor.patch_size=80 \
  processor.num_points=32768 \
  features.k_neighbors=25 \
  input_dir=... output_dir=...
```

## 🔗 See Also

- **Full docs:** https://sducournau.github.io/IGN_LIDAR_HD_DATASET/
- **Config reference:** ../ign_lidar/configs/README.md
- **Archived configs:** [archive/README.md](archive/README.md)
- **Consolidation plan:** [../CONFIG_CONSOLIDATION_PLAN.md](../CONFIG_CONSOLIDATION_PLAN.md)

---

**📖 For detailed examples and full documentation, see the comprehensive guide in ign_lidar/configs/README.md**
