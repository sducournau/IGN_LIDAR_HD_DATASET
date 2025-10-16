# Multi-Scale Training Configuration - Quick Reference

## 🚀 Quick Start

### 1. Run Complete Pipeline

```bash
cd /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET

# Make script executable
chmod +x scripts/run_complete_pipeline.sh

# Run all phases
./scripts/run_complete_pipeline.sh \
    --unified-dataset /mnt/c/Users/Simon/ign/unified_dataset \
    --output-base /mnt/c/Users/Simon/ign \
    --phases all \
    --gpu
```

### 2. Run Specific Phases

```bash
# Only preprocessing (Phase 2)
./scripts/run_complete_pipeline.sh --phases 2

# Only patch generation (Phase 3)
./scripts/run_complete_pipeline.sh --phases 3

# Preprocessing + patches (Phases 2,3)
./scripts/run_complete_pipeline.sh --phases 2,3

# Training only (Phase 5)
./scripts/run_complete_pipeline.sh --phases 5
```

### 3. Manual Step-by-Step

#### Phase 1: Tile Selection

```bash
python scripts/analyze_unified_dataset.py \
    --input /mnt/c/Users/Simon/ign/unified_dataset \
    --output /mnt/c/Users/Simon/ign/analysis_report.json

python scripts/select_optimal_tiles.py \
    --input /mnt/c/Users/Simon/ign/unified_dataset \
    --output /mnt/c/Users/Simon/ign/selected_tiles \
    --asprs-count 100 --lod2-count 80 --lod3-count 60
```

#### Phase 2: Preprocessing

```bash
# ASPRS
ign-lidar-hd process --config configs/multiscale/config_unified_asprs_preprocessing.yaml

# LOD2
ign-lidar-hd process --config configs/multiscale/config_unified_lod2_preprocessing.yaml

# LOD3
ign-lidar-hd process --config configs/multiscale/config_unified_lod3_preprocessing.yaml
```

#### Phase 3: Patch Generation

```bash
# ASPRS - All scales
ign-lidar-hd process --config configs/multiscale/asprs/config_asprs_patches_50m.yaml
ign-lidar-hd process --config configs/multiscale/asprs/config_asprs_patches_100m.yaml
ign-lidar-hd process --config configs/multiscale/asprs/config_asprs_patches_150m.yaml

# LOD2 - All scales
ign-lidar-hd process --config configs/multiscale/lod2/config_lod2_patches_50m.yaml
ign-lidar-hd process --config configs/multiscale/lod2/config_lod2_patches_100m.yaml
ign-lidar-hd process --config configs/multiscale/lod2/config_lod2_patches_150m.yaml

# LOD3 - All scales
ign-lidar-hd process --config configs/multiscale/lod3/config_lod3_patches_50m.yaml
ign-lidar-hd process --config configs/multiscale/lod3/config_lod3_patches_100m.yaml
ign-lidar-hd process --config configs/multiscale/lod3/config_lod3_patches_150m.yaml
```

#### Phase 4: Dataset Merging

```bash
# ASPRS
python examples/merge_multiscale_dataset.py \
    --input-dirs \
        /mnt/c/Users/Simon/ign/patches/asprs/50m \
        /mnt/c/Users/Simon/ign/patches/asprs/100m \
        /mnt/c/Users/Simon/ign/patches/asprs/150m \
    --output /mnt/c/Users/Simon/ign/merged_datasets/asprs_multiscale \
    --strategy balanced

# LOD2
python examples/merge_multiscale_dataset.py \
    --input-dirs \
        /mnt/c/Users/Simon/ign/patches/lod2/50m \
        /mnt/c/Users/Simon/ign/patches/lod2/100m \
        /mnt/c/Users/Simon/ign/patches/lod2/150m \
    --output /mnt/c/Users/Simon/ign/merged_datasets/lod2_multiscale \
    --strategy weighted --weights 0.3 0.4 0.3

# LOD3
python examples/merge_multiscale_dataset.py \
    --input-dirs \
        /mnt/c/Users/Simon/ign/patches/lod3/50m \
        /mnt/c/Users/Simon/ign/patches/lod3/100m \
        /mnt/c/Users/Simon/ign/patches/lod3/150m \
    --output /mnt/c/Users/Simon/ign/merged_datasets/lod3_multiscale \
    --strategy adaptive --oversample-rare
```

#### Phase 5: Training

```bash
# ASPRS - PointNet++
python -m ign_lidar.core.train \
    --config configs/training/asprs/pointnet++_asprs.yaml \
    --data /mnt/c/Users/Simon/ign/merged_datasets/asprs_multiscale \
    --output /mnt/c/Users/Simon/ign/models/asprs/pointnet++ \
    --epochs 100 --batch-size 32 --lr 0.001

# LOD2 - Point Transformer (with ASPRS pretraining)
python -m ign_lidar.core.train \
    --config configs/training/lod2/point_transformer_lod2.yaml \
    --data /mnt/c/Users/Simon/ign/merged_datasets/lod2_multiscale \
    --output /mnt/c/Users/Simon/ign/models/lod2/point_transformer \
    --pretrained /mnt/c/Users/Simon/ign/models/asprs/point_transformer/best_model.pth \
    --epochs 200 --batch-size 12 --lr 0.0003

# LOD3 - Point Transformer (with LOD2 pretraining)
python -m ign_lidar.core.train \
    --config configs/training/lod3/point_transformer_lod3.yaml \
    --data /mnt/c/Users/Simon/ign/merged_datasets/lod3_multiscale \
    --output /mnt/c/Users/Simon/ign/models/lod3/point_transformer \
    --pretrained /mnt/c/Users/Simon/ign/models/lod2/point_transformer/best_model.pth \
    --epochs 250 --batch-size 8 --lr 0.0002 --focal-loss
```

---

## 📁 Configuration Files

### Preprocessing Configurations

- `configs/multiscale/config_unified_asprs_preprocessing.yaml` - ASPRS tile enrichment
- `configs/multiscale/config_unified_lod2_preprocessing.yaml` - LOD2 tile enrichment
- `configs/multiscale/config_unified_lod3_preprocessing.yaml` - LOD3 tile enrichment

### ASPRS Patch Configurations

- `configs/multiscale/asprs/config_asprs_patches_50m.yaml` - 50m patches, 16k points
- `configs/multiscale/asprs/config_asprs_patches_100m.yaml` - 100m patches, 24k points
- `configs/multiscale/asprs/config_asprs_patches_150m.yaml` - 150m patches, 32k points

### LOD2 Patch Configurations

- `configs/multiscale/lod2/config_lod2_patches_50m.yaml` - 50m patches
- `configs/multiscale/lod2/config_lod2_patches_100m.yaml` - 100m patches
- `configs/multiscale/lod2/config_lod2_patches_150m.yaml` - 150m patches

### LOD3 Patch Configurations

- `configs/multiscale/lod3/config_lod3_patches_50m.yaml` - 50m patches, 24k points
- `configs/multiscale/lod3/config_lod3_patches_100m.yaml` - 100m patches, 32k points
- `configs/multiscale/lod3/config_lod3_patches_150m.yaml` - 150m patches, 40k points

---

## 🎯 Key Parameters by Scale

### 50m Patches

- **Best for**: Local details, small objects, fine architectural features
- **Points per patch**: 16k-24k
- **Overlap**: 15-20%
- **Buffer size**: 8m
- **Use cases**: Windows, doors, small buildings, detailed ornaments

### 100m Patches

- **Best for**: Medium-sized buildings, balanced scenes
- **Points per patch**: 24k-32k
- **Overlap**: 10-18%
- **Buffer size**: 12m
- **Use cases**: Residential buildings, facade patterns, urban blocks

### 150m Patches

- **Best for**: Large buildings, urban context, complex scenes
- **Points per patch**: 32k-40k
- **Overlap**: 10-15%
- **Buffer size**: 15m
- **Use cases**: Large commercial buildings, industrial complexes, urban layout

---

## 🔧 Important Settings

### Features

All configurations use **full features**:

- RGB colors
- NIR (near-infrared)
- NDVI (vegetation index)
- Geometric features (normals, curvature, planarity, etc.)
- k_neighbors: 30 (ASPRS), 40 (LOD2), 50 (LOD3)

### Augmentations

- **ASPRS/LOD2**: 3 augmentations (rotation, flip, jitter)
- **LOD3**: 5 augmentations (+ scale, color_jitter)

### Tile Stitching

- Enabled for all configurations
- Auto-detect and cache neighbors
- Buffer zones to prevent edge artifacts

### Ground Truth

- IGN BD TOPO® integration
- NDVI-based preclassification
- Building height thresholds
- Automatic class assignment

---

## 📊 Expected Outputs

### After Preprocessing (Phase 2)

```
C:\Users\Simon\ign\preprocessed\
├── asprs\enriched_tiles\*.laz (with RGB, NIR, NDVI, features)
├── lod2\enriched_tiles\*.laz
└── lod3\enriched_tiles\*.laz
```

### After Patch Generation (Phase 3)

```
C:\Users\Simon\ign\patches\
├── asprs\
│   ├── 50m\{train,val,test}\*.npz + *.laz
│   ├── 100m\{train,val,test}\*.npz + *.laz
│   └── 150m\{train,val,test}\*.npz + *.laz
├── lod2\...
└── lod3\...
```

### After Merging (Phase 4)

```
C:\Users\Simon\ign\merged_datasets\
├── asprs_multiscale\{train,val,test}\*.npz
├── lod2_multiscale\{train,val,test}\*.npz
└── lod3_multiscale\{train,val,test}\*.npz
```

### After Training (Phase 5)

```
C:\Users\Simon\ign\models\
├── asprs\{pointnet++,point_transformer,intelligent_index}\
├── lod2\{pointnet++,point_transformer,intelligent_index}\
└── lod3\{pointnet++,point_transformer,intelligent_index}\
```

---

## ⚙️ Customization

### Adjust Tile Counts

Edit in `run_complete_pipeline.sh`:

```bash
--asprs-count 100  # Number of ASPRS tiles
--lod2-count 80    # Number of LOD2 tiles
--lod3-count 60    # Number of LOD3 tiles
```

### Change Patch Sizes

Edit in individual config files:

```yaml
processor:
  patch_size: 75.0 # Change from 50/100/150 to custom size
  num_points: 20000 # Adjust point count
```

### Modify Augmentations

```yaml
processor:
  num_augmentations: 5
  augmentation_types:
    - rotation
    - flip
    - jitter
    - scale
    - color_jitter
    - noise # Add more types
```

### Adjust Train/Val/Test Split

```yaml
output:
  split_data: true
  train_split: 0.70 # 70% training
  val_split: 0.15 # 15% validation
  test_split: 0.15 # 15% testing
```

---

## 🐛 Troubleshooting

### Out of Memory

- Reduce `num_workers` in configs
- Reduce `batch_size` in training
- Process scales sequentially (remove `--parallel-patches`)
- Reduce `num_points` per patch

### Slow Processing

- Increase `num_workers` (if RAM allows)
- Enable GPU: `--gpu` flag
- Use `--skip-existing` to avoid reprocessing
- Increase `gpu_batch_size` in features config

### Poor Classification Results

- Increase `num_augmentations`
- Use larger `patch_overlap`
- Increase `num_points` per patch
- Add more training epochs
- Use focal loss for class imbalance

---

## 📞 Support

See full documentation: `MULTISCALE_TRAINING_PLAN.md`

For issues, check logs: `/mnt/c/Users/Simon/ign/logs/`
