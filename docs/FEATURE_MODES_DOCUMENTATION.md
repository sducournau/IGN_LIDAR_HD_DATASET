# Complete Feature Set Documentation - LOD2/LOD3 Training

## 📊 Overview

This document describes the complete geometric feature set available for LiDAR point cloud processing, optimized for LOD2 and LOD3 building classification and architectural modeling.

**Total Features Available**: ~40 features across 7 categories

## 🎯 Processing Modes

### 1. MINIMAL Mode (~8 features)

**Use Case**: Ultra-fast processing, quick prototyping
**Features**:

- `normal_z`: Verticality from surface normal
- `planarity`: Flat surface detection
- `height_above_ground`: Height feature
- `density`: Local point density

### 2. LOD2_SIMPLIFIED Mode (~11 features)

**Use Case**: Basic building classification, roof/wall detection
**Features**:

- Coordinates: `xyz` (3 features)
- Geometric: `normal_z`, `planarity`, `linearity`
- Building: `height_above_ground`, `verticality`
- Spectral: `red`, `green`, `blue`, `ndvi` (4 features)

**Performance**: Fast training, good generalization
**Recommended for**: LOD2 building extraction, semantic segmentation

### 3. LOD3_FULL Mode (~35 features)

**Use Case**: Detailed architectural modeling, fine structure detection
**Complete Feature List** (35 features total):

#### Core Geometric (9 features)

- `normal_x`, `normal_y`, `normal_z`: Surface orientation vectors
- `curvature`: Local surface curvature (MAD-based, outlier-robust)
- `change_curvature`: Rate of curvature change (eigenvalue variance)
- `planarity`: Planar structure indicator [0,1]
- `linearity`: Linear structure indicator [0,1]
- `sphericity`: 3D structure indicator [0,1]
- `roughness`: Surface texture measure [0,1]

#### Advanced Shape Descriptors (2 features)

- `anisotropy`: Directional variation [0,1]
- `omnivariance`: 3D dispersion measure (geometric mean of eigenvalues)

#### Eigenvalue Features (5 features)

- `eigenvalue_1`: Largest eigenvalue (λ₀) - dominant direction
- `eigenvalue_2`: Medium eigenvalue (λ₁) - secondary direction
- `eigenvalue_3`: Smallest eigenvalue (λ₂) - surface flatness
- `sum_eigenvalues`: Total eigenvalue sum (Σλ) - local scale
- `eigenentropy`: Shannon entropy of eigenvalues - structure complexity

#### Height Features (2 features)

- `height_above_ground`: Elevation above ground level (meters)
- `vertical_std`: Z-coordinate standard deviation in neighborhood

#### Building-Specific Scores (3 features)

- `verticality`: Vertical surface score [0,1] (1 = vertical wall)
- `wall_score`: Wall likelihood (planarity × verticality)
- `roof_score`: Roof likelihood (planarity × horizontality)

#### Density & Neighborhood (4 features)

- `density`: Local point density (points per unit volume)
- `num_points_2m`: Number of points within 2m radius
- `neighborhood_extent`: Maximum distance to k-th neighbor
- `height_extent_ratio`: Vertical std / spatial extent ratio

#### Architectural Features (4 features)

- `edge_strength`: Edge detection (high eigenvalue variance)
- `corner_likelihood`: Corner probability (all eigenvalues similar)
- `overhang_indicator`: Overhang/protrusion detection (normal consistency)
- `surface_roughness`: Fine-scale surface texture

#### Spectral Features (5 features)

- `red`, `green`, `blue`: RGB color channels [0-255]
- `nir`: Near-infrared channel [0-255]
- `ndvi`: Normalized Difference Vegetation Index [-1,1]
  - Formula: (NIR - R) / (NIR + R)
  - High values indicate vegetation

**Performance**: Slower training, captures fine details
**Recommended for**: LOD3 architectural modeling, detailed structure analysis

### 4. FULL Mode (~40 features)

All available features including experimental ones. Use with caution.

### 5. CUSTOM Mode

User-defined feature selection. Specify features explicitly in configuration.

## 📐 Feature Computation Methods

### Eigenvalue-Based Features

All shape descriptors are computed from the eigenvalue decomposition of the local covariance matrix:

```
Covariance Matrix: C = (1/(k-1)) * Σ(xᵢ - μ)ᵀ(xᵢ - μ)
Eigenvalues: λ₀ ≥ λ₁ ≥ λ₂ (sorted descending)
```

**Standard Formulas** (Weinmann et al., 2015):

- **Linearity**: (λ₀ - λ₁) / λ₀ - captures 1D structures (edges, cables)
- **Planarity**: (λ₁ - λ₂) / λ₀ - captures 2D structures (walls, roofs)
- **Sphericity**: λ₂ / λ₀ - captures 3D structures (vegetation, noise)
- **Anisotropy**: (λ₀ - λ₂) / λ₀ - general directionality measure
- **Roughness**: λ₂ / Σλ - surface roughness (smooth vs rough)
- **Omnivariance**: (λ₀ × λ₁ × λ₂)^(1/3) - geometric mean
- **Eigenentropy**: -Σ(pᵢ log pᵢ) where pᵢ = λᵢ/Σλ

**Property**: Linearity + Planarity + Sphericity ≈ 1.0

### Radius-Based vs K-NN Search

**IMPORTANT**: Use radius-based search to avoid LiDAR scan line artifacts!

- **K-NN Search**: Fixed number of neighbors (k=20, k=30)

  - ❌ Can create dashed line patterns in geometric features
  - ❌ Biased by scan pattern
  - ✅ Consistent number of neighbors

- **Radius Search**: Fixed spatial distance (r=0.5m to 2.0m)
  - ✅ Captures true surface geometry
  - ✅ Avoids scan artifacts
  - ✅ Scale-consistent across point clouds
  - ⚠️ Variable neighbor count

**Recommended**: Use `use_radius: true` in configuration (auto-estimated if not specified)

## 🔄 Data Augmentation Considerations

### Augmentation-Safe Features (CAN be augmented)

These features are rotation/translation invariant or transform correctly:

- `xyz` - transformed by augmentation
- `normal_x`, `normal_y`, `normal_z` - rotated with points
- `height_above_ground` - relative height preserved
- `vertical_std`, `neighborhood_extent` - local properties
- `wall_score`, `roof_score`, `verticality` - rotation-invariant
- `red`, `green`, `blue`, `nir`, `ndvi` - colors unchanged

### Augmentation-Invariant Features (should NOT be augmented)

These represent absolute geometric properties:

- `eigenvalue_1`, `eigenvalue_2`, `eigenvalue_3` - magnitude-dependent
- `sum_eigenvalues`, `eigenentropy` - scale-dependent
- `planarity`, `linearity`, `sphericity` - eigenvalue ratios
- `anisotropy`, `roughness`, `omnivariance` - shape measures
- `curvature`, `density` - local properties

**Best Practice**: Compute geometric features AFTER augmentation to ensure consistency.

## 🎯 Recommended Configurations

### For LOD3 Architectural Modeling (Hybrid PointNet++/Transformer)

```yaml
features:
  mode: lod3
  k_neighbors: 30
  include_extra: true
  use_rgb: true
  use_infrared: true
  compute_ndvi: true
  sampling_method: fps
  normalize_xyz: true
  normalize_features: true
```

**Expected**: ~35 features
**Training Time**: ~3x slower than LOD2
**Accuracy**: Best for detailed structures

### For LOD2 Building Classification (Standard PointNet++)

```yaml
features:
  mode: lod2
  k_neighbors: 20
  include_extra: false
  use_rgb: true
  use_infrared: false
  compute_ndvi: true
  sampling_method: random
  normalize_xyz: true
  normalize_features: true
```

**Expected**: ~11 features
**Training Time**: Fast
**Accuracy**: Good for basic classification

### For Multi-Scale Training (Intelligent Feature Scaling)

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

**Strategy**: More features for small patches (details), fewer for large patches (context)

## 📈 Feature Importance (Based on Literature)

### Critical for Building Detection

1. **planarity** - Distinguishes flat surfaces (walls, roofs)
2. **height_above_ground** - Separates ground/vegetation from buildings
3. **verticality** - Identifies walls vs roofs
4. **normal_z** - Direct orientation indicator

### Important for Architectural Details

1. **edge_strength** - Detects building edges and corners
2. **curvature** - Identifies curved structures (balconies, domes)
3. **wall_score** / **roof_score** - Direct classification hints
4. **linearity** - Detects edges and cables

### Useful for Context

1. **density** - Distinguishes dense urban from sparse areas
2. **neighborhood_extent** - Captures local scale
3. **ndvi** - Separates vegetation from buildings
4. **eigenentropy** - Measures structural complexity

## 🚀 Performance Tips

1. **Start with LOD2**: Faster iteration, good baseline
2. **Add features incrementally**: Test impact of each feature group
3. **Use GPU acceleration**: Enable `use_gpu: true` for large datasets
4. **Enable chunking**: Use `use_gpu_chunked: true` for >20M points
5. **Monitor memory**: Reduce `k_neighbors` if OOM errors occur
6. **Use FPS sampling**: Better point distribution than random for LOD3

## 📚 References

- **Weinmann et al. (2015)**: "Semantic point cloud interpretation based on optimal neighborhoods, relevant features and efficient classifiers"
- **Demantké et al. (2011)**: "Dimensionality based scale selection in 3D lidar point clouds"
- **West et al. (2004)**: "Context-driven automated target detection in 3D data"

## 🔧 Troubleshooting

### Issue: Features contain NaN values

**Solution**: Check for degenerate points (collinear neighbors). Increase `k_neighbors` or enable preprocessing.

### Issue: Scan line artifacts in planarity/linearity

**Solution**: Use radius-based search instead of k-NN: `use_radius: true`

### Issue: Training overfits with LOD3 features

**Solution**: Reduce feature set to LOD2, increase augmentation, or add regularization

### Issue: GPU out of memory

**Solution**: Enable chunking (`use_gpu_chunked: true`) or reduce `gpu_batch_size`

## 📝 Usage Examples

### Python API

```python
from ign_lidar.features import get_feature_config, compute_features_by_mode

# LOD3 configuration
config = get_feature_config("lod3", k_neighbors=30)
print(config.feature_names)  # List all features
print(config.num_features)   # 35

# Compute features
normals, curvature, height, features = compute_features_by_mode(
    points, classification, mode="lod3", k=30
)

# Access specific features
planarity = features['planarity']
wall_score = features['wall_score']
eigenvalues = features['eigenvalue_1']  # etc.
```

### CLI Usage

```bash
# LOD3 full features
ign-lidar-hd process --config-file config_lod3_full_features.yaml

# LOD2 simplified features
ign-lidar-hd process --config-file config_lod2_simplified_features.yaml

# Multi-scale hybrid
ign-lidar-hd process --config-file config_multiscale_hybrid.yaml
```

## 🎓 Feature Selection Guidelines

**Rule of Thumb**:

- **LOD2**: 10-15 features (essential geometric + spectral)
- **LOD3**: 30-40 features (complete geometric + architectural)
- **Production**: Start with LOD2, upgrade to LOD3 if needed

**Testing Strategy**:

1. Baseline: LOD2 with 11 features
2. Add eigenvalues: +5 features (test improvement)
3. Add architectural: +4 features (test improvement)
4. Add density: +4 features (test improvement)
5. Keep only features that improve validation accuracy

---

**Created**: October 2025
**Version**: 1.0
**Author**: IGN LiDAR HD Dataset Project
