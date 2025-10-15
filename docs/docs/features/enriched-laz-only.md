---
sidebar_position: 13
title: Enriched LAZ Only Mode
---

# Enriched LAZ Only Mode

Generate feature-enriched LAZ files without creating patches - perfect for visualization and exploration workflows.

:::info Classification Schema
Enriched LAZ files use the **LOD2/LOD3 classification taxonomy**, not ASPRS standard codes. Buildings are classified as Class 0 (`wall`), ground as Class 9, etc. See [Classification Taxonomy](../reference/classification-taxonomy.md) for complete details.
:::

---

## 🎯 What is Enriched LAZ Only Mode?

**Enriched LAZ Only Mode** computes geometric features and stores them directly in LAZ files, skipping patch generation entirely. This is ideal when you need enriched point clouds for visualization, analysis, or custom processing pipelines.

### Key Benefits

- ⚡ **3-5x faster** - Skip patch generation
- 💾 **50-70% less disk space** - No duplicate patch data
- 🔍 **Inspectable** - View features in CloudCompare, QGIS, etc.
- 🔄 **Flexible** - Generate patches later if needed

---

## 🚀 Quick Start

### Basic Usage

```bash
# Generate enriched LAZ files only
ign-lidar-hd process \
  input_dir=data/raw/ \
  output_dir=output/ \
  output=enriched_only
```

### With Full Features

```bash
# Compute all features
ign-lidar-hd process \
  input_dir=data/raw/ \
  output_dir=output/ \
  output=enriched_only \
  features=full \
  features.use_rgb=true \
  features.compute_ndvi=true
```

---

## 📁 Output Structure

### Enriched Only Mode

```text
output/
├── enriched/
│   ├── tile_1234_5678.laz    ← With computed features
│   ├── tile_1234_5679.laz
│   └── ...
└── metadata.json              ← Processing statistics
```

### Comparison with Other Modes

| Mode            | Enriched LAZ | Patches | Use Case                  |
| --------------- | ------------ | ------- | ------------------------- |
| `patches`       | ❌           | ✅      | ML training (default)     |
| `both`          | ✅           | ✅      | Complete workflow         |
| `enriched_only` | ✅           | ❌      | Visualization/exploration |

---

## ⚙️ Configuration

### Output Mode Selection

```bash
# Patches only (default)
output=patches

# Both enriched LAZ and patches
output=both

# Enriched LAZ only (new in v2.0.1)
output=enriched_only
```

### Feature Selection

```bash
# Minimal features (fast)
ign-lidar-hd process \
  input_dir=data/ \
  output_dir=output/ \
  output=enriched_only \
  features=minimal

# Full features (comprehensive)
ign-lidar-hd process \
  input_dir=data/ \
  output_dir=output/ \
  output=enriched_only \
  features=full

# Custom features
ign-lidar-hd process \
  input_dir=data/ \
  output_dir=output/ \
  output=enriched_only \
  features.compute_linearity=true \
  features.compute_planarity=true \
  features.compute_sphericity=true \
  features.compute_normals=true
```

---

## 🎯 Use Cases

### 1. Data Exploration

Explore features before committing to full ML pipeline:

```bash
# Quick exploration
ign-lidar-hd process \
  input_dir=data/sample/ \
  output_dir=output/exploration/ \
  output=enriched_only \
  features=full

# View in CloudCompare
cloudcompare output/exploration/enriched/*.laz
```

### 2. Visualization Workflows

Create enriched data for presentations or analysis:

```bash
# Generate for visualization
ign-lidar-hd process \
  input_dir=data/buildings/ \
  output_dir=output/viz/ \
  output=enriched_only \
  features=full \
  features.use_rgb=true \
  features.compute_ndvi=true \
  target_class=building

# Export to QGIS
ign-lidar-qgis \
  input_dir=output/viz/enriched/ \
  output_file=buildings.gpkg
```

### 3. Custom Processing

Generate enriched LAZ for custom downstream workflows:

```bash
# Generate enriched data
ign-lidar-hd process \
  input_dir=data/raw/ \
  output_dir=output/enriched/ \
  output=enriched_only \
  features=full

# Process with custom script
python custom_analysis.py \
  --input output/enriched/enriched/ \
  --output custom_results/
```

### 4. Two-Stage Workflow

Generate enriched LAZ first, decide on patches later:

```bash
# Stage 1: Generate enriched LAZ
ign-lidar-hd process \
  input_dir=data/raw/ \
  output_dir=output/stage1/ \
  output=enriched_only \
  features=full

# Analyze, then decide...

# Stage 2: Generate patches from enriched LAZ
ign-lidar-hd process \
  input_dir=output/stage1/enriched/ \
  output_dir=output/stage2/ \
  output=patches
```

---

## 🔧 Advanced Usage

### Python API

```python
from ign_lidar.core import LiDARProcessor

# Configure for enriched only
processor = LiDARProcessor(
    output_mode='enriched_only',
    features='full'
)

# Process
results = processor.process(
    input_dir='data/raw/',
    output_dir='output/'
)

print(f"Generated {results.num_tiles} enriched LAZ files")
print(f"Total points processed: {results.total_points:,}")
```

### Read Enriched LAZ

```python
from ign_lidar.io import read_laz_file

# Read enriched LAZ
points, colors, features = read_laz_file(
    'output/enriched/tile_1234_5678.laz'
)

print("Available features:")
for name, values in features.items():
    print(f"  {name}: {values.shape}")

# Example output:
# Available features:
#   linearity: (1234567,)
#   planarity: (1234567,)
#   sphericity: (1234567,)
#   omnivariance: (1234567,)
#   anisotropy: (1234567,)
#   eigenentropy: (1234567,)
#   change_curvature: (1234567,)
#   normal_x: (1234567,)
#   normal_y: (1234567,)
#   normal_z: (1234567,)
```

### Feature Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from ign_lidar.io import read_laz_file

# Load enriched LAZ
points, colors, features = read_laz_file('output/enriched/tile.laz')

# Extract specific feature
linearity = features['linearity']

# Visualize
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    points[:, 0],
    points[:, 1],
    points[:, 2],
    c=linearity,
    cmap='viridis',
    s=1
)

plt.colorbar(scatter, label='Linearity')
ax.set_title('Linearity Feature Visualization')
plt.show()
```

---

## 📊 Performance Comparison

### Processing Time

| Mode          | Time per Tile | Relative Speed |
| ------------- | ------------- | -------------- |
| Patches Only  | 60s           | 1.0x           |
| Both          | 90s           | 0.67x          |
| Enriched Only | 20s           | 3.0x ⚡        |

### Disk Usage

| Mode          | Storage per Tile | Relative Size |
| ------------- | ---------------- | ------------- |
| Patches Only  | 150 MB           | 1.0x          |
| Both          | 250 MB           | 1.67x         |
| Enriched Only | 80 MB            | 0.53x 💾      |

### Memory Usage

| Mode          | Peak Memory | GPU Memory |
| ------------- | ----------- | ---------- |
| Patches Only  | 4 GB        | 2 GB       |
| Both          | 6 GB        | 3 GB       |
| Enriched Only | 2 GB        | 1 GB       |

---

## 🎨 Feature Storage in LAZ

### Standard LAZ Fields

```text
Standard point format:
- X, Y, Z (coordinates)
- Intensity
- Return Number
- Classification
- RGB (if available)
- Infrared (if available)
```

### Extra Dimensions for Features

Computed features stored as extra dimensions:

```text
Extra dimensions (32-bit float):
- linearity
- planarity
- sphericity
- omnivariance
- anisotropy
- eigenentropy
- sum_eigenvalues
- change_curvature
- normal_x, normal_y, normal_z
- verticality
- (+ custom features)
```

### Accessing in Other Software

**CloudCompare:**

```bash
cloudcompare output/enriched/tile.laz
# Features appear in "Scalar Fields" menu
# Can visualize, filter, classify based on features
```

**QGIS:**

```bash
# Convert to vector format with features
ign-lidar-qgis \
  input_dir=output/enriched/ \
  output_file=enriched.gpkg \
  include_features=true
```

**Python (laspy):**

```python
import laspy

# Read with laspy
las = laspy.read('output/enriched/tile.laz')

# Access extra dimensions
linearity = las.linearity
planarity = las.planarity
normals = np.column_stack([las.normal_x, las.normal_y, las.normal_z])
```

---

## ✅ Best Practices

### When to Use Enriched Only

**Use enriched_only when:**

- Exploring new datasets
- Creating visualizations
- Need enriched data for custom workflows
- Patches not needed (yet)
- Storage or time constrained

**Use patches or both when:**

- Training ML models
- Need standardized patch format
- Using built-in dataset classes
- Following ML pipeline

### Optimization Tips

```bash
# Fast exploration (minimal features)
output=enriched_only \
features=minimal \
preprocess=none

# Quality exploration (full features)
output=enriched_only \
features=full \
preprocess=aggressive

# With boundary-aware (best quality)
output=enriched_only \
features=full \
features.boundary_aware=true \
stitching.enabled=true
```

### Feature Selection Strategy

```bash
# Start minimal for speed
features=minimal

# Add features as needed
features.compute_linearity=true
features.compute_planarity=true

# Or go full if unsure
features=full
```

---

## 🐛 Troubleshooting

### Features Not Visible

```bash
# Ensure features computed
features=full

# Check extra dimensions
pdal info output/enriched/tile.laz | grep "extra"
```

### Large File Sizes

```bash
# Reduce feature precision
features.precision=float32  # vs float64

# Or select fewer features
features=minimal
```

### Slow Processing

```bash
# Disable expensive features
features.compute_curvature=false
features.compute_normals=false

# Or use CPU only
processor=cpu
```

---

## 🎓 Complete Example

### Building Exploration Workflow

```bash
# 1. Generate enriched LAZ with all features
ign-lidar-hd process \
  input_dir=data/paris_buildings/ \
  output_dir=output/exploration/ \
  output=enriched_only \
  features=full \
  features.use_rgb=true \
  target_class=building \
  preprocess=aggressive

# 2. View in CloudCompare
cloudcompare output/exploration/enriched/*.laz

# 3. Analyze features in Python
python analyze_features.py \
  --input output/exploration/enriched/ \
  --output analysis/

# 4. If satisfied, generate patches for training
ign-lidar-hd process \
  input_dir=output/exploration/enriched/ \
  output_dir=output/training/ \
  output=patches \
  architecture=pointnet++

# 5. Train model
python train.py \
  --data output/training/ \
  --epochs 100
```

---

## 📚 Related Topics

- [Feature Computation](/api/features) - Available features
- [Output Formats](/guides/output-formats) - Output options
- [QGIS Integration](/guides/qgis-integration) - Visualization workflows

---

**Next Steps:**

- Try [Boundary-Aware Processing](/features/boundary-aware)
- Explore [Tile Stitching](/features/tile-stitching)
- Read [Visualization Guide](/guides/visualization)
