# Feature Computation System - Quick Reference

## 🚀 Quick Start

### 1. Choose Your Processing Mode

```yaml
features:
  mode: lod3 # Options: minimal, lod2, lod3, full, custom
  k_neighbors: 30
  include_extra: true
```

### 2. Run Processing

```bash
# LOD3 with full features (35 features)
ign-lidar-hd process --config-file examples/config_lod3_full_features.yaml

# LOD2 with simplified features (11 features)
ign-lidar-hd process --config-file examples/config_lod2_simplified_features.yaml

# Multi-scale hybrid training
ign-lidar-hd process --config-file examples/config_multiscale_hybrid.yaml
```

## 📊 Feature Modes Comparison

| Mode        | Features | Speed    | Use Case                        |
| ----------- | -------- | -------- | ------------------------------- |
| **minimal** | ~8       | ⚡⚡⚡⚡ | Quick prototyping               |
| **lod2**    | ~11      | ⚡⚡⚡   | Basic building classification   |
| **lod3**    | ~35      | ⚡⚡     | Detailed architectural modeling |
| **full**    | ~40      | ⚡       | Research, complete analysis     |

## 🎯 Feature Sets by Mode

### LOD2 (11 features) - Essential

```
✓ xyz (3)                    - Coordinates
✓ normal_z                   - Verticality
✓ planarity                  - Flat surface detection
✓ linearity                  - Edge detection
✓ height_above_ground        - Height feature
✓ verticality                - Wall detection
✓ red, green, blue (3)       - RGB colors
✓ ndvi                       - Vegetation index
```

### LOD3 (35 features) - Complete

```
✓ All LOD2 features
✓ normal_x, normal_y         - Complete normals
✓ curvature, change_curvature - Surface curvature
✓ sphericity, anisotropy, roughness, omnivariance - Shape descriptors
✓ eigenvalue_1, eigenvalue_2, eigenvalue_3 - Eigenvalues
✓ sum_eigenvalues, eigenentropy - Eigenvalue statistics
✓ vertical_std               - Height variation
✓ wall_score, roof_score     - Building scores
✓ density, num_points_2m     - Density features
✓ neighborhood_extent, height_extent_ratio - Neighborhood stats
✓ edge_strength, corner_likelihood - Architectural features
✓ overhang_indicator, surface_roughness - Advanced architectural
✓ nir                        - Near-infrared
```

## 🔧 Python API

### Basic Usage

```python
from ign_lidar.features import compute_features_by_mode, get_feature_config

# Get feature configuration
config = get_feature_config("lod3", k_neighbors=30)
print(f"Mode: {config.mode}")
print(f"Features: {config.num_features}")
print(f"Feature names: {config.feature_names}")

# Compute features
normals, curvature, height, features = compute_features_by_mode(
    points=points,              # [N, 3] coordinates
    classification=classes,     # [N] ASPRS codes
    mode="lod3",               # Processing mode
    k=30,                      # Number of neighbors
    auto_k=True,               # Auto-estimate k
    use_radius=True            # Use radius search (recommended)
)

# Access computed features
planarity = features['planarity']          # [N] values
wall_score = features['wall_score']        # [N] values
eigenvalues = features['eigenvalue_1']     # [N] values
```

### Advanced: Custom Feature Set

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
    'density'
}

# Create custom configuration
config = get_feature_config(
    mode="custom",
    custom_features=custom_features,
    k_neighbors=20
)

# Use in computation
normals, curvature, height, features = compute_features_by_mode(
    points, classification, mode="custom", k=20
)
```

## 📈 Performance Guidelines

### Memory Usage (per 1M points)

| Mode | CPU RAM | GPU VRAM |
| ---- | ------- | -------- |
| LOD2 | ~200 MB | ~150 MB  |
| LOD3 | ~600 MB | ~400 MB  |

### Processing Speed (1M points)

| Mode | CPU  | GPU |
| ---- | ---- | --- |
| LOD2 | ~15s | ~3s |
| LOD3 | ~45s | ~8s |

### Optimization Tips

1. **Use GPU**: 5-10x faster

   ```yaml
   processor:
     use_gpu: true
   features:
     use_gpu_chunked: true
     gpu_batch_size: 1000000
   ```

2. **Enable Chunking**: For large files (>20M points)

   ```yaml
   features:
     use_gpu_chunked: true
   ```

3. **Adjust k_neighbors**: Lower for speed, higher for accuracy

   ```yaml
   features:
     k_neighbors: 20  # Fast
     k_neighbors: 30  # Accurate
   ```

4. **Use Radius Search**: Avoids scan line artifacts
   ```yaml
   features:
     use_radius: true # Auto-estimated
   ```

## 🎨 Example Configurations

### Configuration 1: Fast LOD2 Training

```yaml
processor:
  lod_level: LOD2
  num_points: 16384
  augment: true

features:
  mode: lod2
  k_neighbors: 20
  use_rgb: true
  compute_ndvi: true
  sampling_method: random
```

**Output**: 11 features, fast training, good baseline

### Configuration 2: Detailed LOD3 Modeling

```yaml
processor:
  lod_level: LOD3
  num_points: 32768
  augment: true

features:
  mode: lod3
  k_neighbors: 30
  include_extra: true
  use_rgb: true
  use_infrared: true
  compute_ndvi: true
  sampling_method: fps
```

**Output**: 35 features, detailed structures, best accuracy

### Configuration 3: Multi-Scale Hybrid

```yaml
processor:
  patch_configs:
    - size: 50.0
      feature_mode: lod3 # Fine details
      num_points: 24000
    - size: 150.0
      feature_mode: lod2 # Coarse context
      num_points: 32000
```

**Output**: Adaptive feature set per scale

## 🔍 Feature Descriptions

### Most Important for Building Detection

1. **planarity** - Distinguishes flat surfaces (roofs, walls)
2. **height_above_ground** - Separates ground from buildings
3. **verticality** - Identifies vertical surfaces (walls)
4. **wall_score** - Direct wall classification hint
5. **normal_z** - Surface orientation indicator

### Key Architectural Features

1. **edge_strength** - Detects building edges/corners
2. **corner_likelihood** - Identifies corners and junctions
3. **curvature** - Captures curved structures
4. **overhang_indicator** - Detects balconies, overhangs

### Spectral Features

1. **ndvi** - Vegetation index (high = plants, low = buildings)
2. **rgb** - Color information
3. **nir** - Near-infrared (vegetation reflectance)

## 🚨 Troubleshooting

### Problem: NaN values in features

**Solution**: Increase `k_neighbors` or enable preprocessing

```yaml
features:
  k_neighbors: 30 # Increase from 20
preprocess:
  enabled: true
  sor_k: 12
```

### Problem: Scan line artifacts

**Solution**: Use radius-based search

```yaml
features:
  use_radius: true
```

### Problem: GPU out of memory

**Solution**: Enable chunking or reduce batch size

```yaml
features:
  use_gpu_chunked: true
  gpu_batch_size: 500000 # Reduce from 1M
```

### Problem: Training overfits

**Solution**: Reduce to LOD2 or increase augmentation

```yaml
features:
  mode: lod2 # Fewer features
processor:
  augment: true
  num_augmentations: 5 # More augmentation
```

## 📚 Additional Resources

- **Full Documentation**: [FEATURE_MODES_DOCUMENTATION.md](../docs/FEATURE_MODES_DOCUMENTATION.md)
- **Example Configs**: [examples/](../examples/)
- **API Reference**: [ign_lidar/features/](../ign_lidar/features/)

## 🎓 Best Practices

1. **Start with LOD2**: Get baseline results quickly
2. **Add features incrementally**: Test impact of each feature group
3. **Use validation set**: Monitor overfitting with LOD3 features
4. **Enable augmentation**: Especially important with many features
5. **Monitor memory**: Use chunking for large datasets

---

**Quick Links**:

- 📖 [Complete Documentation](../docs/FEATURE_MODES_DOCUMENTATION.md)
- ⚙️ [Configuration Examples](../examples/)
- 🐍 [Python API](../ign_lidar/features/)
- 🚀 [Getting Started Guide](../README.md)
