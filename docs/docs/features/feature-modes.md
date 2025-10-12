---
sidebar_position: 3
title: Feature Modes
description: Understanding LOD2 and LOD3 feature modes for optimal ML training
keywords: [feature modes, LOD2, LOD3, machine learning, training]
---

Choose the right feature set for your machine learning application. IGN LiDAR HD offers predefined feature modes optimized for different use cases.

---

## 🎯 Feature Modes Overview

| Mode        | Features | Speed    | Use Case                        |
| ----------- | -------- | -------- | ------------------------------- |
| **minimal** | ~8       | ⚡⚡⚡⚡ | Quick prototyping               |
| **lod2**    | ~11      | ⚡⚡⚡   | Basic building classification   |
| **lod3**    | ~35      | ⚡⚡     | Detailed architectural modeling |
| **full**    | ~40      | ⚡       | Research, complete analysis     |
| **custom**  | Variable | Variable | User-defined selection          |

---

## 📊 Feature Set Details

### LOD2 Mode (11 features)

**Essential features for basic building classification:**

```yaml
features:
  mode: lod2
  k_neighbors: 20
```

**Feature List:**

- **Coordinates (3)**: `x`, `y`, `z`
- **Normals (1)**: `normal_z` (verticality)
- **Shape (2)**: `planarity`, `linearity`
- **Height (1)**: `height_above_ground`
- **Building (1)**: `verticality`
- **Spectral (4)**: `red`, `green`, `blue`, `ndvi`

**Performance:**

- Processing: ~15s per 1M points (CPU)
- Training: Fast convergence
- Memory: ~200 MB per 1M points

**Best For:**

- Building vs. non-building classification
- LOD2 semantic segmentation
- Baseline model development
- Fast iteration cycles

---

### LOD3 Mode (35 features)

**Complete feature set for detailed architectural modeling:**

```yaml
features:
  mode: lod3
  k_neighbors: 30
  include_extra: true
```

**Additional Features (beyond LOD2):**

**Normals (2 more):**

- `normal_x`, `normal_y` - complete normal vectors

**Advanced Shape (5):**

- `sphericity`, `anisotropy`, `roughness`, `omnivariance`
- `curvature`, `change_curvature`

**Eigenvalues (5):**

- `eigenvalue_1`, `eigenvalue_2`, `eigenvalue_3`
- `sum_eigenvalues`, `eigenentropy`

**Height (1 more):**

- `vertical_std` - vertical variation

**Building Scores (2 more):**

- `wall_score`, `roof_score` - direct classification hints

**Density (4):**

- `density`, `num_points_2m`
- `neighborhood_extent`, `height_extent_ratio`

**Architectural (4):**

- `edge_strength`, `corner_likelihood`
- `overhang_indicator`, `surface_roughness`

**Spectral (1 more):**

- `nir` - near-infrared channel

**Performance:**

- Processing: ~45s per 1M points (CPU)
- Training: Slower but more detailed
- Memory: ~600 MB per 1M points

**Best For:**

- LOD3 architectural modeling
- Fine structure detection (edges, corners, overhangs)
- Detailed building classification
- Research applications

---

## 🚀 Quick Start Examples

### Example 1: Fast LOD2 Training

```bash
ign-lidar-hd process \
  --config-file examples/config_lod2_simplified_features.yaml \
  input_dir=data/raw \
  output_dir=data/patches
```

**Configuration:**

```yaml
processor:
  lod_level: LOD2
  num_points: 16384

features:
  mode: lod2
  k_neighbors: 20
  use_rgb: true
  compute_ndvi: true
```

**Expected Output:**

- 11 features per point
- Fast training convergence
- Good baseline accuracy

---

### Example 2: Detailed LOD3 Modeling

```bash
ign-lidar-hd process \
  --config-file examples/config_lod3_full_features.yaml \
  input_dir=data/raw \
  output_dir=data/patches
```

**Configuration:**

```yaml
processor:
  lod_level: LOD3
  num_points: 32768

features:
  mode: lod3
  k_neighbors: 30
  include_extra: true
  use_rgb: true
  use_infrared: true
  compute_ndvi: true
```

**Expected Output:**

- 35 features per point
- Detailed architectural structures
- Best accuracy for LOD3

---

### Example 3: Multi-Scale Hybrid

```bash
ign-lidar-hd process \
  --config-file examples/config_multiscale_hybrid.yaml \
  input_dir=data/raw \
  output_dir=data/patches
```

**Configuration:**

```yaml
processor:
  patch_configs:
    - size: 50.0
      feature_mode: lod3 # Fine details
      num_points: 24000
    - size: 100.0
      feature_mode: lod3 # Medium context
      num_points: 32000
    - size: 150.0
      feature_mode: lod2 # Coarse generalization
      num_points: 32000
```

**Strategy:**

- Small patches (50m): LOD3 for fine details
- Large patches (150m): LOD2 for context
- Adaptive feature complexity

---

## 🎓 Best Practices

### Choosing the Right Mode

**Use LOD2 when:**

- ✅ Building basic classification models
- ✅ Need fast training cycles
- ✅ Limited computational resources
- ✅ Prototyping new architectures

**Use LOD3 when:**

- ✅ Need detailed architectural features
- ✅ Detecting edges, corners, overhangs
- ✅ LOD3 building modeling
- ✅ Maximum accuracy is priority

**Use Custom when:**

- ✅ Specific feature requirements
- ✅ Domain knowledge guides selection
- ✅ Optimizing for specific architecture

---

### Feature Selection Strategy

**Start Simple:**

1. Begin with LOD2 (11 features)
2. Train baseline model
3. Evaluate performance

**Add Complexity:**

1. Upgrade to LOD3 if needed
2. Monitor overfitting on validation set
3. Compare accuracy improvement

**Optimize:**

1. Remove features with low importance
2. Custom mode with essential features only
3. Balance accuracy vs. training time

---

### Performance Tuning

**For Faster Processing:**

```yaml
features:
  mode: lod2 # Fewer features
  k_neighbors: 20 # Lower k
  use_gpu: true # GPU acceleration
```

**For Better Accuracy:**

```yaml
features:
  mode: lod3 # More features
  k_neighbors: 30 # Higher k
  include_extra: true # All enhanced features
  use_radius: true # Better neighborhoods
```

**For Memory Constraints:**

```yaml
features:
  mode: lod2 # Smaller feature set
  use_gpu_chunked: true
  gpu_batch_size: 500000
```

---

## 📈 Feature Importance Analysis

### Critical Features (Present in both modes)

1. **planarity** - Distinguishes flat surfaces (walls, roofs)
2. **height_above_ground** - Separates ground from buildings
3. **verticality** - Identifies vertical surfaces (walls)
4. **normal_z** - Direct orientation indicator

### LOD3-Specific High-Value Features

1. **edge_strength** - Building edges and corners
2. **wall_score** - Direct wall classification
3. **roof_score** - Direct roof classification
4. **eigenvalue_1** - Dominant structural direction
5. **corner_likelihood** - Junction detection

### Spectral Features (Both modes)

1. **ndvi** - Vegetation vs. building separation
2. **rgb** - Color-based classification
3. **nir** - Vegetation reflectance (LOD3 only)

---

## 🔧 Python API

### Using Feature Modes

```python
from ign_lidar import LiDARProcessor
from ign_lidar.features import get_feature_config

# Get feature configuration
config = get_feature_config("lod3", k_neighbors=30)
print(f"Features: {config.num_features}")
print(f"Names: {config.feature_names}")

# Process with LOD3
processor = LiDARProcessor(
    lod_level="LOD3",
    use_gpu=True
)
patches = processor.process_tile("input.laz", "output/")
```

### Custom Feature Selection

```python
from ign_lidar.features import get_feature_config

# Define custom features
custom_features = {
    'xyz',
    'normal_z',
    'planarity',
    'linearity',
    'height_above_ground',
    'wall_score',
    'roof_score',
    'density'
}

# Create custom configuration
config = get_feature_config(
    mode="custom",
    custom_features=custom_features,
    k_neighbors=25
)

# Use in processor
processor = LiDARProcessor(
    lod_level="LOD3",
    custom_features=custom_features
)
```

---

## 📊 Benchmark Results

### Processing Speed (1M points, CPU)

| Mode | Time | Speedup vs. LOD3 |
| ---- | ---- | ---------------- |
| LOD2 | 15s  | 3x faster        |
| LOD3 | 45s  | baseline         |

### Memory Usage (1M points)

| Mode | RAM    | GPU VRAM |
| ---- | ------ | -------- |
| LOD2 | 200 MB | 150 MB   |
| LOD3 | 600 MB | 400 MB   |

### Training Performance

**Dataset:** 100K patches, PointNet++ architecture

| Mode | Epochs | Val Accuracy | Inference Time |
| ---- | ------ | ------------ | -------------- |
| LOD2 | 50     | 87.3%        | 12ms/patch     |
| LOD3 | 80     | 92.1%        | 18ms/patch     |

**Conclusion:** LOD3 provides +4.8% accuracy at 1.5x training time cost.

---

## 🐛 Troubleshooting

### Issue: Out of memory with LOD3

**Solution:**

```yaml
features:
  mode: lod2 # Use simpler mode
  # Or enable chunking:
  use_gpu_chunked: true
  gpu_batch_size: 500000
```

### Issue: Training overfits with LOD3

**Solution:**

- Increase regularization (dropout, weight decay)
- Add more data augmentation
- Consider LOD2 for better generalization

### Issue: Too slow processing

**Solution:**

```yaml
processor:
  use_gpu: true # Enable GPU
features:
  mode: lod2 # Fewer features
  k_neighbors: 20 # Lower k
```

---

## 📚 Related Documentation

- [Geometric Features Reference](/features/geometric-features)
- [LOD3 Classification Guide](/features/lod3-classification)
- [GPU Acceleration](/gpu/acceleration)
- [Configuration Reference](/api/configuration)

---

## 🔗 Example Configurations

All example configs available in `examples/` directory:

- `config_lod2_simplified_features.yaml`
- `config_lod3_full_features.yaml`
- `config_multiscale_hybrid.yaml`
- `config_training_dataset.yaml`

---

**Choose wisely. Train efficiently. Build better models.** 🚀
