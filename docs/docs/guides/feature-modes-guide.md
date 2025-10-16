---
sidebar_position: 9
title: Feature Modes Guide
---

# Feature Modes Guide

Deep dive into feature computation modes for optimal performance and accuracy.

---

## 🎯 Understanding Feature Modes

Feature modes are **predefined feature sets** that balance accuracy, performance, and file size for different use cases.

### Why Feature Modes?

**v2.x Problem:**

- Users had to manually enable/disable 20+ feature flags
- Unclear which features to use for each task
- Trial and error to find optimal settings

**v3.0 Solution:**

- Select one `mode` parameter
- Optimized feature combinations
- Clear guidance for each use case

---

## 📊 Feature Mode Comparison

### Performance Matrix

| Mode            | Features | Computation Time | Memory | File Size  | Accuracy  |
| --------------- | -------- | ---------------- | ------ | ---------- | --------- |
| `minimal`       | 8        | 1x (fastest)     | Low    | Tiny       | Basic     |
| `asprs_classes` | 15       | 2x               | Medium | Small      | High ⭐   |
| `lod2`          | 17       | 3x               | Medium | Medium     | High      |
| `lod3`          | 43       | 8x               | High   | Large      | Very High |
| `full`          | 45       | 10x              | High   | Very Large | Maximum   |

### Use Case Recommendations

| Task                 | Recommended Mode | Why                                    |
| -------------------- | ---------------- | -------------------------------------- |
| ASPRS Classification | `asprs_classes`  | Optimized for ASPRS codes, lightweight |
| Quick Testing        | `minimal`        | Fastest processing                     |
| Building Detection   | `lod2`           | Essential building features            |
| 3D Reconstruction    | `lod3`           | Architectural details                  |
| Research/ML          | `full`           | All features for experimentation       |
| Production (General) | `asprs_classes`  | Best balance                           |

---

## 🎯 ASPRS Classes Mode (Recommended)

### Overview

The default and recommended mode for most users.

**Key Benefits:**

- ✅ Optimized for ASPRS LAS 1.4 classification
- ✅ 60-70% smaller files vs full mode
- ✅ 2.8x faster than full mode
- ✅ High accuracy (85-95% with ground truth)
- ✅ Works perfectly with BD TOPO/cadastre enrichment

### Included Features (15 total)

#### Coordinate Features (3)

- `xyz` - Point coordinates

#### Surface Orientation (1)

- `normal_z` - Z component of normal (verticality indicator)

#### Shape Descriptors (2)

- `planarity` - Flat surface detection (roads, roofs, ground)
- `sphericity` - Vegetation detection

#### Height Features (1)

- `height_above_ground` - Essential for multi-class separation

#### Building Detection (2)

- `verticality` - Wall vs ground distinction
- `horizontality` - Ground and flat roof detection

#### Spatial Features (1)

- `density` - Point density (varies by class)

#### Spectral Features (5)

- `red`, `green`, `blue` - RGB for visual classification
- `nir` - Near-infrared for vegetation
- `ndvi` - Normalized Difference Vegetation Index

### Configuration

```yaml
features:
  mode: "asprs_classes"
  k_neighbors: 20
  search_radius: null
  use_rgb: true
  use_nir: true
  compute_ndvi: true
  include_extra: false
```

### Best Practices

**Do:**

- ✅ Enable RGB, NIR, NDVI for best results
- ✅ Use with BD TOPO enrichment for 85-95% accuracy
- ✅ Use k_neighbors=20 (optimal)
- ✅ Enable caching for data sources

**Don't:**

- ❌ Disable spectral features (reduces accuracy)
- ❌ Use without ground truth (limits accuracy to 70-80%)
- ❌ Use for detailed architectural modeling (use lod3)

### Expected Results

**File sizes (1 km², 10 pts/m²):**

- NPZ format: ~800 MB
- HDF5 format: ~650 MB
- LAZ format: ~450 MB (compressed)

**Processing time (1 km² tile):**

- CPU: ~3-5 minutes
- GPU: ~1-2 minutes

---

## ⚡ Minimal Mode

### Overview

Ultra-fast processing with only essential features.

**Best for:**

- Quick testing and prototyping
- Initial data exploration
- System validation
- Low-resource environments

### Included Features (8 total)

- `xyz` - Coordinates (3)
- `normal_z` - Verticality (1)
- `planarity` - Flat surfaces (1)
- `sphericity` - Vegetation (1)
- `height_above_ground` - Height (1)
- `density` - Point density (1)

### Configuration

```yaml
features:
  mode: "minimal"
  k_neighbors: 10 # Fewer neighbors = faster
  search_radius: null
  use_rgb: false
  use_nir: false
  compute_ndvi: false
  include_extra: false
```

### Use Cases

✅ System testing  
✅ Data quality checks  
✅ Quick visualization  
✅ Algorithm prototyping

❌ Production use  
❌ High-accuracy classification  
❌ Detailed analysis

---

## 🏢 LOD2 Mode

### Overview

Essential features for Level of Detail 2 building classification.

**Best for:**

- Building footprint extraction
- Urban analysis
- 2.5D city models
- Roof type classification

### Included Features (17 total)

**Geometric (9):**

- Normals (nx, ny, nz)
- Curvature, change_curvature
- Planarity, linearity, sphericity, roughness

**Height (2):**

- height_above_ground
- vertical_std

**Building (3):**

- verticality
- wall_likelihood
- roof_likelihood

**Spatial (1):**

- density

**Spectral (3, optional):**

- RGB (if use_rgb=true)

### Configuration

```yaml
features:
  mode: "lod2"
  k_neighbors: 20
  use_rgb: true # Recommended
  use_nir: false
  compute_ndvi: false
  include_extra: false
```

### Use Cases

✅ Building detection  
✅ Roof classification  
✅ Urban planning  
✅ 2.5D modeling

❌ Fine architectural details  
❌ Window/door detection  
❌ Facade analysis

---

## 🏛️ LOD3 Mode

### Overview

Complete feature set for Level of Detail 3 architectural modeling.

**Best for:**

- 3D building reconstruction
- Architectural analysis
- Facade segmentation
- Heritage documentation

### Included Features (43 total)

**All LOD2 features, plus:**

**Eigenvalues (5):**

- eigenvalue_1, eigenvalue_2, eigenvalue_3
- sum_eigenvalues
- eigenentropy

**Shape Descriptors (3 more):**

- anisotropy
- omnivariance
- roughness (enhanced)

**Architectural (8):**

- facade_score
- flat_roof_score
- sloped_roof_score
- steep_roof_score
- opening_likelihood
- edge_strength
- corner_likelihood
- overhang_indicator

**Density (3 more):**

- num_points_2m
- neighborhood_extent
- height_extent_ratio

**Spectral (5):**

- RGB, NIR, NDVI (all recommended)

### Configuration

```yaml
features:
  mode: "lod3"
  k_neighbors: 20
  search_radius: 2.0 # Use radius for better quality
  use_rgb: true
  use_nir: true
  compute_ndvi: true
  include_extra: true
```

### Use Cases

✅ 3D reconstruction  
✅ Facade analysis  
✅ Architectural styles  
✅ Heritage documentation  
✅ BIM integration

⚠️ Slower processing  
⚠️ Larger files  
⚠️ More memory

---

## 🔬 Full Mode

### Overview

All available features for research and maximum flexibility.

**Best for:**

- Research projects
- Feature importance studies
- Algorithm development
- Maximum accuracy requirements

### Included Features (45 total)

All LOD3 features plus experimental/research features.

### Configuration

```yaml
features:
  mode: "full"
  k_neighbors: 20
  search_radius: 2.0
  use_rgb: true
  use_nir: true
  compute_ndvi: true
  include_extra: true
```

### Use Cases

✅ Research  
✅ Feature engineering  
✅ Algorithm comparison  
✅ Maximum accuracy

⚠️ Slowest mode  
⚠️ Largest files  
⚠️ Most memory

---

## 🔄 Switching Between Modes

### Easy Mode Switching

```bash
# Default: ASPRS classes
ign-lidar-hd process --config-file configs/default_v3.yaml

# Override to minimal
ign-lidar-hd process \
    --config-file configs/default_v3.yaml \
    features.mode=minimal

# Override to LOD3
ign-lidar-hd process \
    --config-file configs/default_v3.yaml \
    features.mode=lod3 \
    features.include_extra=true
```

### Comparing Modes

Process the same data with different modes:

```bash
# ASPRS classes
ign-lidar-hd process \
    input_dir=data/raw \
    output_dir=data/asprs \
    features.mode=asprs_classes

# LOD3
ign-lidar-hd process \
    input_dir=data/raw \
    output_dir=data/lod3 \
    features.mode=lod3

# Compare results
python compare_modes.py data/asprs data/lod3
```

---

## 📊 Performance Benchmarks

### Processing Time (1 km² tile, 10 pts/m²)

| Mode          | CPU Time | GPU Time | Speedup |
| ------------- | -------- | -------- | ------- |
| minimal       | 1.5 min  | 0.5 min  | 3.0x    |
| asprs_classes | 3.0 min  | 1.0 min  | 3.0x    |
| lod2          | 5.0 min  | 1.5 min  | 3.3x    |
| lod3          | 12.0 min | 4.0 min  | 3.0x    |
| full          | 15.0 min | 5.0 min  | 3.0x    |

### File Size (1 km² tile, NPZ format)

| Mode          | File Size | vs Full | Compression |
| ------------- | --------- | ------- | ----------- |
| minimal       | 500 MB    | -83%    | High        |
| asprs_classes | 800 MB    | -73%    | Good ⭐     |
| lod2          | 1.2 GB    | -60%    | Medium      |
| lod3          | 2.5 GB    | -17%    | Low         |
| full          | 3.0 GB    | 100%    | None        |

### Memory Usage

| Mode          | RAM Usage | GPU VRAM | Recommended  |
| ------------- | --------- | -------- | ------------ |
| minimal       | 2 GB      | 1 GB     | 4 GB system  |
| asprs_classes | 4 GB      | 2 GB     | 8 GB system  |
| lod2          | 6 GB      | 3 GB     | 16 GB system |
| lod3          | 12 GB     | 6 GB     | 32 GB system |
| full          | 16 GB     | 8 GB     | 64 GB system |

---

## 🎯 Decision Tree

```
Start here
    |
    ├─ Need ASPRS classification?
    |   └─ Yes → Use asprs_classes ⭐
    |
    ├─ Quick testing?
    |   └─ Yes → Use minimal
    |
    ├─ Building detection only?
    |   └─ Yes → Use lod2
    |
    ├─ 3D reconstruction?
    |   └─ Yes → Use lod3
    |
    └─ Research/everything?
        └─ Yes → Use full
```

---

## 📚 See Also

- [Configuration System v3.0](./configuration-v3)
- [Data Source Enrichment](./data-sources)
- [Performance Optimization](./performance)
- [Processing Modes](./processing-modes)

---

## ❓ FAQ

### Can I customize a feature mode?

Not directly, but you can:

1. Use `custom` mode with specific features
2. Override individual feature flags
3. Create your own preset config

### Which mode for production?

**asprs_classes** - Best balance of speed, size, and accuracy.

### Can I change modes later?

Yes! Reprocess with a different mode. The point cloud is unchanged, only features differ.

### How do I know which features are used?

Check the mode definition in `ign_lidar/features/feature_modes.py` or run:

```python
from ign_lidar.features.feature_modes import get_feature_config

config = get_feature_config("asprs_classes")
print(config.features)
```
