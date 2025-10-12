---
sidebar_position: 2
title: Geometric Features
description: Complete reference for geometric features in IGN LiDAR HD
keywords: [features, geometric, eigenvalues, validation, LOD2, LOD3]
---

Complete reference for geometric features computed from LiDAR point clouds. All features in v2.4.0+ are **guaranteed to be within valid ranges** for optimal ML stability.

---

## 🎯 Feature Validation (v2.4.0+)

### Guaranteed Ranges

All geometric features are validated and clamped to prevent numerical artifacts:

- **Shape Descriptors**: [0, 1] - linearity, planarity, sphericity, etc.
- **Eigenvalue Features**: [0, ∞) with negative values clamped to 0
- **Density Features**: [0, 1] - normalized and capped at 1000 pts/m³
- **Normal Components**: [-1, 1] - unit vectors
- **Curvature**: [0, 1] - MAD-based, outlier-robust

:::tip Production Ready
v2.4.0+ guarantees zero NaN/Inf values and eliminates out-of-range warnings!
:::

---

## 📊 Feature Categories

### Core Geometric Features

#### Surface Normals (3 features)

Computed via eigenvalue decomposition of local covariance matrix.

- **`normal_x`**: X component of surface normal
- **`normal_y`**: Y component of surface normal
- **`normal_z`**: Z component of surface normal (verticality indicator)

**Range**: [-1, 1] each (unit vector)  
**Use Case**: Surface orientation, wall/roof detection

#### Curvature (1 feature)

- **`curvature`**: Principal curvature based on smallest eigenvalue

**Formula**: `λ₂ / (λ₀ + λ₁ + λ₂)`  
**Range**: [0, 1] (validated)  
**Use Case**: Edge detection, curved structures

---

### Shape Descriptors (Weinmann et al., 2015)

All computed from sorted eigenvalues λ₀ ≥ λ₁ ≥ λ₂:

#### Linearity

- **Formula**: `(λ₀ - λ₁) / λ₀`
- **Range**: [0, 1] (validated)
- **Interpretation**: 1.0 = perfect line (edges, cables)
- **Use Case**: Edge detection, linear structures

#### Planarity

- **Formula**: `(λ₁ - λ₂) / λ₀`
- **Range**: [0, 1] (validated)
- **Interpretation**: 1.0 = perfect plane (walls, roofs)
- **Use Case**: Building surface detection, **most important feature**

#### Sphericity

- **Formula**: `λ₂ / λ₀`
- **Range**: [0, 1] (validated)
- **Interpretation**: 1.0 = isotropic (vegetation, noise)
- **Use Case**: Vegetation vs. building separation

#### Anisotropy

- **Formula**: `(λ₀ - λ₂) / λ₀`
- **Range**: [0, 1] (validated)
- **Interpretation**: Directional variation
- **Use Case**: General shape characterization

#### Roughness

- **Formula**: `λ₂ / (λ₀ + λ₁ + λ₂)`
- **Range**: [0, 1] (validated)
- **Interpretation**: Surface texture (smooth vs. rough)
- **Use Case**: Material classification

#### Omnivariance

- **Formula**: `(λ₀ × λ₁ × λ₂)^(1/3) / λ₀`
- **Range**: [0, 1] (validated)
- **Interpretation**: 3D dispersion (geometric mean)
- **Use Case**: Volume-based features

:::info Validation Notes
v2.4.0+ ensures: **Linearity + Planarity + Sphericity ≈ 1.0** through eigenvalue clamping.
:::

---

### Eigenvalue Features (5 features)

Direct eigenvalue access for advanced analysis:

- **`eigenvalue_1`**: Largest eigenvalue (λ₀) - dominant direction
- **`eigenvalue_2`**: Medium eigenvalue (λ₁) - secondary direction
- **`eigenvalue_3`**: Smallest eigenvalue (λ₂) - surface flatness

**Validation**: `np.maximum(eigenvalues, 0.0)` - prevents negative values

#### Eigenvalue Statistics

- **`sum_eigenvalues`**: Σλ = λ₀ + λ₁ + λ₂

  - **Range**: [0, ∞)
  - **Use Case**: Local scale, neighborhood size

- **`eigenentropy`**: Shannon entropy of eigenvalues

  - **Formula**: `-Σ(pᵢ log pᵢ) / log(3)` where `pᵢ = λᵢ / Σλ`
  - **Range**: [0, 1] (normalized)
  - **Use Case**: Structural complexity

- **`change_curvature`**: Variance of eigenvalues
  - **Formula**: Variance(λ₀, λ₁, λ₂)
  - **Range**: [0, ∞)
  - **Use Case**: Surface change detection

---

### Height Features (2 features)

#### Height Above Ground

- **`height_above_ground`**: Elevation above ground level (meters)
- **Range**: [0, max_z] (absolute)
- **Use Case**: **Critical for building detection**

#### Vertical Variation

- **`vertical_std`**: Standard deviation of Z coordinates in neighborhood
- **Range**: [0, ∞) (meters)
- **Use Case**: Vertical complexity, terrain variation

---

### Building-Specific Scores (3 features)

Derived features optimized for architectural modeling:

#### Verticality

- **Formula**: `1.0 - abs(normal_z)`
- **Range**: [0, 1] (validated)
- **Interpretation**: 1.0 = perfectly vertical surface
- **Use Case**: **Wall detection** (critical)

#### Wall Score

- **Formula**: `planarity × verticality`
- **Range**: [0, 1] (validated)
- **Interpretation**: High = likely wall surface
- **Use Case**: Direct wall classification hint

#### Roof Score

- **Formula**: `planarity × (1 - verticality)` (horizontality)
- **Range**: [0, 1] (validated)
- **Interpretation**: High = likely roof surface
- **Use Case**: Direct roof classification hint

:::tip LOD3 Models
Wall/roof scores dramatically improve LOD3 architectural classification!
:::

---

### Density & Neighborhood (4 features)

Local point cloud density characteristics:

#### Density

- **Formula**: `min(num_neighbors / volume, 1000.0) / 1000.0`
- **Range**: [0, 1] (validated, capped at 1000 pts/m³)
- **Use Case**: Dense urban vs. sparse areas

#### Number of Points

- **`num_points_2m`**: Count of points within 2m radius
- **Range**: [0, ∞) (count)
- **Use Case**: Local density proxy

#### Neighborhood Extent

- **`neighborhood_extent`**: Maximum distance to k-th neighbor
- **Range**: [0, ∞) (meters)
- **Use Case**: Local scale indicator

#### Height Extent Ratio

- **Formula**: `vertical_std / neighborhood_extent`
- **Range**: [0, 1] (validated)
- **Use Case**: Vertical vs. horizontal extent comparison

---

### Architectural Features (4 features)

Advanced features for fine-scale structure detection:

#### Edge Strength

- **Formula**: Based on eigenvalue variance
- **Range**: [0, 1] (normalized)
- **Use Case**: Building edge/corner detection

#### Corner Likelihood

- **Formula**: Similarity of all three eigenvalues
- **Range**: [0, 1] (normalized)
- **Use Case**: Corner and junction detection

#### Overhang Indicator

- **Formula**: Based on normal consistency in vertical neighborhood
- **Range**: [0, 1] (normalized)
- **Use Case**: Balcony, overhang, protrusion detection

#### Surface Roughness

- **Formula**: Fine-scale surface texture measure
- **Range**: [0, 1] (normalized)
- **Use Case**: Material classification, weathering detection

---

## 🔧 Computation Methods

### Eigenvalue Decomposition

All shape descriptors derived from covariance matrix:

```python
# Covariance matrix computation
C = (1/(k-1)) * Σ(xᵢ - μ)ᵀ(xᵢ - μ)

# Eigenvalue decomposition
eigenvalues, eigenvectors = np.linalg.eigh(C)
λ₀, λ₁, λ₂ = sorted_eigenvalues(eigenvalues)  # descending

# v2.4.0+ validation
λ₀, λ₁, λ₂ = np.maximum([λ₀, λ₁, λ₂], 0.0)  # Clamp negatives
```

### Search Methods

#### Radius-Based Search (Recommended)

```yaml
features:
  use_radius: true # Auto-estimated from point cloud
  # Avoids scan line artifacts
```

**Advantages:**

- ✅ True geometric neighborhoods
- ✅ Scale-consistent
- ✅ No scan pattern bias

#### K-Nearest Neighbor Search

```yaml
features:
  use_radius: false
  k_neighbors: 30 # Fixed number
```

**Advantages:**

- ✅ Consistent neighbor count
- ⚠️ May introduce scan line artifacts

:::warning Scan Line Artifacts
K-NN can create dashed line patterns in features. Use radius-based search for best results!
:::

---

## 📈 Feature Importance

### Critical for Building Detection (⭐⭐⭐⭐⭐)

1. **planarity** - Distinguishes flat surfaces
2. **height_above_ground** - Separates buildings from ground
3. **verticality** - Identifies walls vs. roofs
4. **normal_z** - Direct orientation indicator

### Important for Details (⭐⭐⭐⭐)

1. **edge_strength** - Building edges and corners
2. **curvature** - Curved structures (balconies, domes)
3. **wall_score/roof_score** - Direct classification hints
4. **linearity** - Edges and cables

### Useful for Context (⭐⭐⭐)

1. **density** - Urban density patterns
2. **neighborhood_extent** - Local scale
3. **eigenentropy** - Structural complexity

---

## 🎓 Best Practices

### For LOD2 (Basic Building Classification)

```yaml
features:
  mode: lod2 # Essential 11 features
  k_neighbors: 20
  use_radius: true
```

**Includes**: XYZ, normal_z, planarity, linearity, height, verticality, RGB, NDVI

### For LOD3 (Detailed Architectural Modeling)

```yaml
features:
  mode: lod3 # Complete 35 features
  k_neighbors: 30
  include_extra: true
  use_radius: true
```

**Adds**: All eigenvalues, architectural features, density features, building scores

### Feature Normalization

```python
# v2.4.0+ features are already validated!
# No need for custom normalization

from ign_lidar import LiDARProcessor

processor = LiDARProcessor(lod_level="LOD3")
patches = processor.process_tile("input.laz", "output/")

# All features guaranteed in valid ranges ✅
```

---

## 🐛 Troubleshooting

### Issue: NaN values in features

**Solution**: Upgrade to v2.4.0+ for automatic validation

```bash
pip install --upgrade ign-lidar-hd
```

### Issue: Out-of-range warnings

**Solution**: v2.4.0+ eliminates these warnings through validation

### Issue: Scan line artifacts

**Solution**: Use radius-based search

```yaml
features:
  use_radius: true
```

### Issue: Training instability

**Solution**: v2.4.0+ improves convergence through feature validation

---

## 📚 References

- **Weinmann et al. (2015)**: "Semantic point cloud interpretation based on optimal neighborhoods, relevant features and efficient classifiers"
- **Demantké et al. (2011)**: "Dimensionality based scale selection in 3D lidar point clouds"
- **West et al. (2004)**: "Context-driven automated target detection in 3D data"

---

## 🔗 Related Documentation

- [Feature Modes Documentation](/features/feature-modes)
- [LOD3 Classification Guide](/features/lod3-classification)
- [Boundary-Aware Processing](/features/boundary-aware)
- [GPU Acceleration](/gpu/acceleration)

---

**Validated. Robust. Production-Ready.** ✅
