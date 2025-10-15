---
sidebar_position: 3
title: Feature Modes
description: Understanding LOD2 and LOD3 feature modes for optimal ML training
keywords: [feature modes, LOD2, LOD3, machine learning, training]
---

Choose the right feature set for your machine learning application. IGN LiDAR HD offers predefined feature modes optimized for different use cases.

---

## 🎯 Feature Modes Overview

| Mode        | Features | Speed      | Use Case                        | Valid Mode |
| ----------- | -------- | ---------- | ------------------------------- | ---------- |
| **minimal** | 4        | ⚡⚡⚡⚡⚡ | Quick updates, classification   | ✅ Yes     |
| **lod2**    | ~12      | ⚡⚡⚡⚡   | Basic building classification   | ✅ Yes     |
| **lod3**    | ~37      | ⚡⚡       | Detailed architectural modeling | ✅ Yes     |
| **full**    | ~37+     | ⚡         | Research, complete analysis     | ✅ Yes     |
| **custom**  | Variable | Variable   | User-defined selection          | ✅ Yes     |

:::warning Invalid Mode Values
**Only use the mode values listed above!** Invalid values like `core` will cause the system to default to `lod3` mode (37 features), which may be slower than intended.
:::

:::info v2.4.3 Feature Export Fix
**All computed features are now saved to disk!** Previous versions (< 2.4.3) only exported 12 features even when computing 35+. Regenerate datasets for complete feature sets.
:::

---

## 📊 Feature Set Details

### Minimal Mode (4 features)

**Ultra-fast processing with essential features only:**

```yaml
features:
  mode: minimal
  k_neighbors: 10
```

**Feature List:**

- **Normals (1)**: `normal_z` - verticality indicator
- **Shape (1)**: `planarity` - main shape descriptor
- **Height (1)**: `height_above_ground` - essential for building detection
- **Density (1)**: `density` - local point density

**Performance:**

- Processing: ~5s per 1M points (CPU)
- Training: Extremely fast
- Memory: ~50 MB per 1M points

**Best For:**

- Classification updates only
- Quick tile processing
- When you don't need detailed features
- Rapid prototyping

---

### LOD2 Mode (12 features)

**Essential features for basic building classification:**

```yaml
features:
  mode: lod2
  k_neighbors: 20
```

**Feature List:**

- **Coordinates (3)**: `xyz` - point coordinates
- **Normals (1)**: `normal_z` - verticality indicator
- **Shape (2)**: `planarity`, `linearity`
- **Height (1)**: `height_above_ground`
- **Building (1)**: `verticality`
- **Radiometric (5)**: RGB (3), NDVI (1)

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

### LOD3 Mode (37 features)

**Complete feature set for detailed architectural modeling:**

```yaml
features:
  mode: lod3
  k_neighbors: 30
```

**Complete Feature List:**

**Coordinates (3):**

- `xyz` - X, Y, Z point coordinates

**Normals (3):**

- `normal_x`, `normal_y`, `normal_z` - complete normal vectors

**Shape Descriptors (6):**

- `planarity`, `linearity`, `sphericity`
- `anisotropy`, `roughness`, `omnivariance`

**Curvature (2):**

- `curvature`, `change_curvature`

**Eigenvalues (5):**

- `eigenvalue_1`, `eigenvalue_2`, `eigenvalue_3`
- `sum_eigenvalues`, `eigenentropy`

**Height Features (2):**

- `height_above_ground`, `vertical_std`

**Building Scores (3):**

- `verticality`, `wall_score`, `roof_score`

**Density Features (4):**

- `density`, `num_points_2m`
- `neighborhood_extent`, `height_extent_ratio`

**Architectural Features (4):**

- `edge_strength`, `corner_likelihood`
- `overhang_indicator`, `surface_roughness`

**Radiometric (5):**

- RGB (3): `red`, `green`, `blue`
- Infrared (2): `nir`, `ndvi`

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

### Full Mode (37+ features)

**Complete feature set for research and analysis (same as LOD3 plus any additional features):**

```yaml
features:
  mode: full
```

**All Features:**

Same as LOD3 mode (37 features), with all available features computed. This mode ensures you get every feature the system can compute, including any future additions.

**Output Format:**

- **NPZ/HDF5/PyTorch**: Full feature matrix with all computed features
- **LAZ**: All features as extra dimensions
- **Metadata**: `feature_names` list, `num_features` count

**Performance:**

- Processing: ~50s per 1M points (CPU)
- Training: Complete geometric description
- Memory: ~600 MB per 1M points
- File Size: ~3-4x larger than minimal mode

**Best For:**

- Research and feature analysis
- Maximum information extraction
- Feature importance studies
- Complete geometric characterization
- GIS visualization (all features in LAZ)
- When you want future features automatically included

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
  k_neighbors: 20
  include_extra: true
  compute_ndvi: true
```

**Expected Output:**

- 38 features per point
- Detailed architectural structures
- Best accuracy for LOD3

---

### Example 3: Complete Feature Set (Full Mode)

```bash
ign-lidar-hd process \
  --config-file examples/config_complete.yaml \
  input_dir=data/raw \
  output_dir=data/patches
```

**Configuration (v2.4.2+):**

```yaml
processor:
  lod_level: LOD3
  num_points: 32768

features:
  mode: full
  k_neighbors: 30
  include_extra: true
  compute_all: true
  use_rgb: true
  use_infrared: true
  compute_ndvi: true

output:
  formats: ["npz", "laz"] # LAZ for GIS visualization
  include_metadata: true
```

**Expected Output:**

- 43+ features per point (all computed features)
- Complete geometric characterization
- LAZ files with all features as extra dimensions
- Metadata with feature names and counts

**Verification:**

```python
import numpy as np

# Load and check
data = np.load('patches/patch_001.npz')
meta = data['metadata'].item()

print(f"Features: {meta['num_features']}")
print(f"Names: {meta['feature_names']}")
# Expected: 43+ features with full list
```

---

### Example 4: Multi-Scale Hybrid

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

**Use LOD2 (12 features) when:**

- ✅ Building basic classification models
- ✅ Need fast training cycles
- ✅ Limited computational resources
- ✅ Prototyping new architectures
- ✅ Building vs. non-building classification

**Use LOD3 (38 features) when:**

- ✅ Need detailed architectural features
- ✅ Detecting edges, corners, overhangs
- ✅ LOD3 building modeling
- ✅ Maximum accuracy is priority
- ✅ Fine structure detection

**Use Full (43+ features) when:**

- ✅ Research and feature analysis
- ✅ Need all computed features
- ✅ Feature importance studies
- ✅ Maximum information extraction
- ✅ GIS visualization with LAZ export

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

### Critical Features (Present in all modes)

1. **planarity** - Distinguishes flat surfaces (walls, roofs)
2. **height_above_ground** - Separates ground from buildings
3. **verticality** - Identifies vertical surfaces (walls)
4. **normals** - Direct orientation indicators

### LOD3+ High-Value Features

1. **edge_strength** - Building edges and corners
2. **wall_score** - Direct wall classification
3. **roof_score** - Direct roof classification
4. **eigenvalue_1** - Dominant structural direction
5. **corner_likelihood** - Junction detection

### Full Mode Additional Features

1. **horizontality** - Horizontal surface identification
2. **local_roughness** - Fine-scale surface variation
3. **z_from_ground/median** - Multiple height references
4. **distance_to_center** - Radial position information

### Radiometric Features (Optional in all modes)

1. **ndvi** - Vegetation vs. building separation
2. **rgb** - Color-based classification
3. **nir** - Vegetation reflectance

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

| Mode | Features | Time | Speedup vs. Full |
| ---- | -------- | ---- | ---------------- |
| LOD2 | 12       | 15s  | 3.3x faster      |
| LOD3 | 38       | 45s  | 1.1x faster      |
| Full | 43+      | 50s  | baseline         |

### Memory Usage (1M points)

| Mode | Features | RAM    | GPU VRAM |
| ---- | -------- | ------ | -------- |
| LOD2 | 12       | 200 MB | 150 MB   |
| LOD3 | 38       | 600 MB | 400 MB   |
| Full | 43+      | 700 MB | 450 MB   |

### File Sizes (per patch, 16K points)

| Mode | Features | NPZ Size | LAZ Size |
| ---- | -------- | -------- | -------- |
| LOD2 | 12       | ~250 KB  | ~180 KB  |
| LOD3 | 38       | ~650 KB  | ~420 KB  |
| Full | 43+      | ~750 KB  | ~480 KB  |

### Training Performance

**Dataset:** 100K patches, PointNet++ architecture

| Mode | Features | Epochs | Val Accuracy | Inference Time |
| ---- | -------- | ------ | ------------ | -------------- |
| LOD2 | 12       | 50     | 87.3%        | 12ms/patch     |
| LOD3 | 38       | 80     | 92.1%        | 18ms/patch     |
| Full | 43+      | 90     | 93.5%        | 20ms/patch     |

**Conclusion:** LOD3 provides +4.8% accuracy over LOD2. Full mode provides additional +1.4% for research applications.

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

- `config_lod2_simplified_features.yaml` - 12 features, fast training
- `config_lod3_full_features.yaml` - 38 features, detailed modeling
- `config_complete.yaml` - 43+ features, complete feature export
- `config_multiscale_hybrid.yaml` - Multi-scale adaptive features
- `config_training_dataset.yaml` - Production training configs

---

**Choose wisely. Train efficiently. Build better models.** 🚀
