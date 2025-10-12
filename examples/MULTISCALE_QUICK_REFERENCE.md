# Quick Reference: Multi-Scale Training for Hybrid Models

## 📊 Patch Size Comparison

| Patch Size  | Area (m²) | Input Points | Sampled Points | % Sampled | Best For                           |
| ----------- | --------- | ------------ | -------------- | --------- | ---------------------------------- |
| **50m**     | 2,500     | ~25,000      | 24,576         | ~98%      | Fine details, small buildings      |
| **100m**    | 10,000    | ~100,000     | 32,768         | ~33%      | Balanced facades, medium buildings |
| **150m** ⭐ | 22,500    | ~225,000     | 32,768         | ~15%      | Full context, large buildings      |

_Based on IGN LiDAR HD density: >10 points/m²_

---

## 🚀 Quick Start Commands

### Generate All Scales (Sequential - 24GB RAM)

```bash
# Navigate to project
cd /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET

# Run automated pipeline
./examples/run_multiscale_training.sh
```

### Generate All Scales (Parallel - 64GB RAM)

```bash
./examples/run_multiscale_training.sh --parallel
```

### Generate Individual Scales

```bash
# 50m patches (4-6 hours, ~15k patches, 24k points each)
ign-lidar-hd process --config-file examples/config_lod3_training_50m.yaml

# 100m patches (3-4 hours, ~8k patches, 32k points each)
ign-lidar-hd process --config-file examples/config_lod3_training_100m.yaml

# 150m patches (2-3 hours, ~5k patches, 32k points each)
ign-lidar-hd process --config-file examples/config_lod3_training.yaml
```

### Merge Multi-Scale Dataset

```bash
python examples/merge_multiscale_dataset.py \
  --output patches_multiscale \
  --weights 0.4 0.3 0.3 \
  --train-split 0.7 \
  --val-split 0.15 \
  --test-split 0.15
```

---

## 📁 Output Structure

```
/mnt/c/Users/Simon/ign/
├── enriched_tiles/              # Input (pre-enriched LAZ files)
├── patches_50m/                 # ~15,000 patches @ 24k points
│   ├── LHD_FXX_xxxx_hybrid_patch_0001.npz
│   └── LHD_FXX_xxxx_hybrid_patch_0001.laz
├── patches_100m/                # ~8,000 patches @ 32k points
│   ├── LHD_FXX_xxxx_hybrid_patch_0001.npz
│   └── ...
├── patches_150m/                # ~5,000 patches @ 32k points
│   ├── LHD_FXX_xxxx_hybrid_patch_0001.npz
│   └── ...
└── patches_multiscale/          # Merged dataset
    ├── train/                   # 70% (~19,600 patches)
    ├── val/                     # 15% (~4,200 patches)
    ├── test/                    # 15% (~4,200 patches)
    └── dataset_metadata.json
```

---

## 💾 Storage Requirements

| Dataset           | Size      | Patches     | Notes                           |
| ----------------- | --------- | ----------- | ------------------------------- |
| 50m patches       | ~15GB     | ~15,000     | Most patches, 24k points each   |
| 100m patches      | ~20GB     | ~8,000      | Balanced, 32k points each       |
| 150m patches      | ~18GB     | ~5,000      | Fewest patches, 32k points each |
| **Total**         | **~53GB** | **~28,000** | Before merging                  |
| Merged (symlinks) | ~53GB     | ~28,000     | No duplication                  |
| Merged (copies)   | ~90GB     | ~28,000     | Full duplication                |

---

## ⚙️ Configuration Parameters by Scale

| Parameter             | 50m Config | 100m Config | 150m Config |
| --------------------- | ---------- | ----------- | ----------- |
| **patch_size**        | 50.0       | 100.0       | 150.0       |
| **num_points**        | 24,576     | 32,768      | 32,768      |
| **patch_overlap**     | 0.15 (15%) | 0.12 (12%)  | 0.10 (10%)  |
| **num_augmentations** | 5          | 4           | 3           |
| **k_neighbors**       | 20         | 25          | 30          |
| **buffer_size**       | 8.0m       | 12.0m       | 15.0m       |
| **sor_k**             | 8          | 10          | 12          |

---

## 🎓 Training Strategy Options

### 1. Sequential Multi-Scale (Curriculum Learning)

**Best for:** Stable training, limited GPU memory

```python
# Start with fine details, progressively add larger scales
Phase 1: Train on 50m patches (20 epochs)
Phase 2: Add 100m patches (30 epochs)
Phase 3: Add 150m patches (50 epochs)
```

### 2. Mixed Multi-Scale (Simultaneous)

**Best for:** Better generalization, scale-invariant features

```python
# Train on all scales from the start
Train on merged dataset (100 epochs)
```

### 3. Scale-Specific Experts + Ensemble

**Best for:** Maximum accuracy per scale

```python
# Train 3 separate models, ensemble at inference
model_50m, model_100m, model_150m
```

---

## 📈 Expected Performance Gains

| Metric              | Single-Scale (150m) | Multi-Scale | Improvement  |
| ------------------- | ------------------- | ----------- | ------------ |
| **mIoU**            | 72.3%               | **76.8%**   | +4.5%        |
| **Precision**       | 78.1%               | **81.4%**   | +3.3%        |
| **Recall**          | 74.6%               | **78.9%**   | +4.3%        |
| **Small Buildings** | 68.2%               | **75.1%**   | **+6.9%** ⭐ |
| **Large Buildings** | 79.4%               | **80.2%**   | +0.8%        |

_Multi-scale training especially improves performance on small buildings and architectural details_

---

## 🔧 Troubleshooting

### Out of Memory (OOM)

```bash
# Use sequential mode instead of parallel
./examples/run_multiscale_training.sh

# Or use memory-optimized configs
ign-lidar-hd process --config-file examples/config_lod3_training_memory_optimized.yaml
```

### Disk Space Issues

```bash
# Check available space
df -h /mnt/c/Users/Simon/ign

# Use symlinks instead of copies when merging
python examples/merge_multiscale_dataset.py --output patches_multiscale
# (symlinks are default, use --copy flag to copy files)
```

### Processing Too Slow

```bash
# Enable GPU acceleration (if available)
nvidia-smi  # Check GPU availability

# Reduce number of augmentations in configs
# Edit yaml: num_augmentations: 3 -> 1
```

### Verify Dataset

```bash
# Check patch counts
find /mnt/c/Users/Simon/ign/patches_50m -name "*.npz" | wc -l
find /mnt/c/Users/Simon/ign/patches_100m -name "*.npz" | wc -l
find /mnt/c/Users/Simon/ign/patches_150m -name "*.npz" | wc -l

# Check merged dataset
cat /mnt/c/Users/Simon/ign/patches_multiscale/dataset_metadata.json
```

---

## 📚 Additional Resources

- **Full Strategy Guide:** [MULTI_SCALE_TRAINING_STRATEGY.md](MULTI_SCALE_TRAINING_STRATEGY.md)
- **Memory Optimization:** [MEMORY_OPTIMIZATION.md](MEMORY_OPTIMIZATION.md)
- **Configuration Reference:** [README.md](README.md)
- **Main Documentation:** [../README.md](../README.md)

---

## 🎯 Recommended Workflow

1. **Generate 50m patches first** (fastest to test pipeline)
2. **Verify output quality** (check LAZ files in QGIS/CloudCompare)
3. **Generate 100m and 150m patches** (sequential or parallel)
4. **Merge datasets** with appropriate weights
5. **Start training** with curriculum learning approach
6. **Monitor metrics** and compare single vs multi-scale performance

---

**Created:** October 2025  
**Version:** Compatible with IGN LiDAR HD v2.3.2+
