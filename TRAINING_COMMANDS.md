# IGN LiDAR HD - Training Patch Generation Commands

This document provides optimized commands for generating training patches from IGN LiDAR HD tiles for different use cases.

---

## 🎯 LOD2 Classification - Architecture Agnostic (RECOMMENDED)

**Use case:** Train models for building classification with maximum flexibility. Supports PointNet++, transformers, sparse convolutions, and hybrid architectures.

```bash
ign-lidar-hd process \
  input_dir="/mnt/c/Users/Simon/ign/raw_tiles/urban_dense" \
  output_dir="/mnt/c/Users/Simon/ign/training_patches_lod2_hybrid" \
  processor.lod_level=LOD2 \
  processor.architecture=hybrid \
  processor.use_gpu=true \
  processor.num_workers=4 \
  processor.num_points=16384 \
  processor.patch_size=150.0 \
  processor.patch_overlap=0.15 \
  processor.augment=true \
  processor.num_augmentations=5 \
  features=full \
  features.mode=full \
  features.k_neighbors=20 \
  features.include_extra=true \
  features.use_rgb=true \
  features.use_infrared=true \
  features.compute_ndvi=true \
  features.sampling_method=fps \
  features.normalize_xyz=true \
  features.normalize_features=true \
  preprocess=aggressive \
  preprocess.enabled=true \
  stitching=enabled \
  stitching.enabled=true \
  stitching.buffer_size=15.0 \
  stitching.auto_detect_neighbors=true \
  stitching.cache_enabled=true \
  output.format=all \
  output.save_enriched_laz=true \
  output.save_stats=true \
  output.save_metadata=true \
  log_level=INFO
```

**Configuration highlights:**

- ✓ LOD2 building classification focus
- ✓ 16,384 points per patch (standard for most models)
- ✓ 5x geometric augmentation
- ✓ RGB + NIR + NDVI features
- ✓ Farthest Point Sampling (FPS)
- ✓ Tile stitching with 15m buffer
- ✓ All output formats (torch, npz, hdf5, laz)
- ✓ Aggressive preprocessing

---

## 🚀 High-Performance GPU Processing

**Use case:** Maximum speed with GPU acceleration for large datasets.

```bash
ign-lidar-hd process \
  input_dir="/mnt/c/Users/Simon/ign/raw_tiles/urban_dense" \
  output_dir="/mnt/c/Users/Simon/ign/training_patches_lod2_gpu" \
  processor=gpu \
  processor.lod_level=LOD2 \
  processor.num_points=16384 \
  processor.augment=true \
  processor.num_augmentations=5 \
  processor.batch_size=32 \
  processor.pin_memory=true \
  features=full \
  features.use_rgb=true \
  features.use_infrared=true \
  features.compute_ndvi=true \
  preprocess=aggressive \
  stitching=enabled \
  output.format=torch \
  log_level=INFO
```

---

## 🎓 PointNet++ Specific Training

**Use case:** Optimized specifically for PointNet++ architecture.

```bash
ign-lidar-hd process \
  input_dir="/mnt/c/Users/Simon/ign/raw_tiles/urban_dense" \
  output_dir="/mnt/c/Users/Simon/ign/training_patches_pointnet" \
  experiment=pointnet_training \
  processor.lod_level=LOD2 \
  processor.augment=true \
  processor.num_augmentations=5 \
  features.use_rgb=true \
  features.use_infrared=true \
  features.compute_ndvi=true \
  stitching.enabled=true \
  output.format=torch
```

---

## 🌳 Vegetation Segmentation

**Use case:** Focus on vegetation classification with NDVI.

```bash
ign-lidar-hd process \
  input_dir="/mnt/c/Users/Simon/ign/raw_tiles/urban_dense" \
  output_dir="/mnt/c/Users/Simon/ign/training_patches_vegetation" \
  processor.num_points=16384 \
  processor.augment=true \
  processor.num_augmentations=5 \
  features=full \
  features.use_rgb=true \
  features.use_infrared=true \
  features.compute_ndvi=true \
  features.include_extra=true \
  preprocess=aggressive \
  stitching=enabled \
  output.format=all
```

---

## 🏗️ LOD3 Classification (Detailed)

**Use case:** Higher detail building classification with more points.

```bash
ign-lidar-hd process \
  input_dir="/mnt/c/Users/Simon/ign/raw_tiles/urban_dense" \
  output_dir="/mnt/c/Users/Simon/ign/training_patches_lod3" \
  processor.lod_level=LOD3 \
  processor.architecture=hybrid \
  processor.num_points=32768 \
  processor.augment=true \
  processor.num_augmentations=5 \
  features=full \
  features.k_neighbors=30 \
  features.use_rgb=true \
  features.use_infrared=true \
  features.compute_ndvi=true \
  features.sampling_method=fps \
  features.normalize_xyz=true \
  features.normalize_features=true \
  preprocess=aggressive \
  stitching=enabled \
  stitching.buffer_size=20.0 \
  output.format=all
```

---

## ⚡ Fast Prototyping (Minimal Features)

**Use case:** Quick iteration during development.

```bash
ign-lidar-hd process \
  input_dir="/mnt/c/Users/Simon/ign/raw_tiles/urban_dense" \
  output_dir="/mnt/c/Users/Simon/ign/training_patches_fast" \
  processor.num_points=8192 \
  processor.augment=false \
  features=minimal \
  features.use_rgb=false \
  preprocess=disabled \
  stitching=disabled \
  output.format=npz
```

---

## 🎯 Memory-Constrained Processing

**Use case:** Lower memory usage for systems with limited RAM/VRAM.

```bash
ign-lidar-hd process \
  input_dir="/mnt/c/Users/Simon/ign/raw_tiles/urban_dense" \
  output_dir="/mnt/c/Users/Simon/ign/training_patches_lowmem" \
  processor.lod_level=LOD2 \
  processor.num_points=8192 \
  processor.num_workers=2 \
  processor.augment=true \
  processor.num_augmentations=3 \
  features=full \
  features.use_rgb=true \
  features.use_infrared=true \
  preprocess=aggressive \
  stitching=enabled \
  stitching.cache_enabled=false \
  output.format=npz \
  output.save_enriched_laz=false
```

---

## 📊 Configuration Parameter Guide

### Processor Options

- `processor.lod_level`: `LOD2` or `LOD3` (classification detail level)
- `processor.use_gpu`: `true` or `false` (GPU acceleration)
- `processor.num_workers`: Number of parallel workers (1-N)
- `processor.num_points`: Points per patch (4096, 8192, 16384, 32768)
- `processor.patch_size`: Patch size in meters (50-200m typical)
- `processor.patch_overlap`: Overlap ratio (0.0-0.3)
- `processor.augment`: Enable augmentation (`true`/`false`)
- `processor.num_augmentations`: Number of augmentations (1-10)

### Features Options

- `features.mode`: `minimal`, `full`, or `custom`
- `features.k_neighbors`: Neighbors for geometric features (10-30)
- `features.use_rgb`: Include RGB from orthophotos
- `features.use_infrared`: Include NIR from IRC
- `features.compute_ndvi`: Compute vegetation index
- `features.sampling_method`: `random`, `fps`, or `grid`
- `features.normalize_xyz`: Normalize coordinates
- `features.normalize_features`: Standardize features

### Preprocessing Options

- `preprocess`: `disabled`, `default`, or `aggressive`
- `preprocess.enabled`: Enable preprocessing
- `preprocess.sor_k`: SOR neighbors (8-16)
- `preprocess.sor_std`: SOR std threshold (1.0-3.0)

### Stitching Options

- `stitching`: `disabled` or `enabled`
- `stitching.enabled`: Enable tile stitching
- `stitching.buffer_size`: Buffer zone in meters (10-20m)
- `stitching.auto_detect_neighbors`: Auto-detect tiles

### Output Options

- `output.format`: `npz`, `hdf5`, `torch`, `laz`, or `all`
- `output.save_enriched_laz`: Save enriched LAZ files
- `output.save_stats`: Save statistics
- `output.save_metadata`: Save patch metadata

---

## 🔧 Augmentation Details

When `processor.augment=true`, the following augmentations are applied:

1. **Random rotation** around Z-axis (0-360°)
2. **Random scaling** (0.8-1.2x)
3. **Random jittering** (small noise)
4. **Random flipping** (X and Y axes)
5. **RGB/NIR augmentation** (if enabled):
   - Brightness adjustment
   - Contrast adjustment
   - Saturation adjustment (RGB only)

---

## 📁 Output Structure

```
output_dir/
├── patches/
│   ├── patch_0000.npz         # NumPy format
│   ├── patch_0000.pt          # PyTorch format
│   ├── patch_0000.h5          # HDF5 format
│   └── ...
├── enriched_laz/
│   ├── tile_0000.laz          # Enriched LAZ with features
│   └── ...
├── stats/
│   ├── processing_stats.json
│   └── feature_stats.json
└── metadata/
    ├── patch_metadata.json
    └── tile_index.json
```

---

## 🎬 Quick Start

1. **Run the recommended command** (copy-paste from top)
2. **Or use the script**:
   ```bash
   bash generate_training_patches_lod2.sh
   ```
3. **Monitor progress** in the terminal
4. **Check outputs** in the specified output directory

---

## 🐛 Troubleshooting

### Out of Memory Error

- Reduce `processor.num_points` (try 8192)
- Reduce `processor.num_workers`
- Disable `stitching.cache_enabled`
- Use `output.format=npz` instead of `all`

### Slow Processing

- Enable `processor.use_gpu=true`
- Increase `processor.num_workers`
- Disable `output.save_enriched_laz`
- Use `preprocess=default` instead of `aggressive`

### Missing RGB/NIR

- Ensure orthophoto tiles are available in input directory
- Check IRC tiles for infrared data
- Set `features.use_rgb=false` if not available

---

## 📚 Additional Resources

- Full documentation: `website/docs/`
- Configuration schema: `ign_lidar/config/schema.py`
- Example configs: `ign_lidar/configs/`
- Dataset usage: `ign_lidar/datasets/multi_arch_dataset.py`
