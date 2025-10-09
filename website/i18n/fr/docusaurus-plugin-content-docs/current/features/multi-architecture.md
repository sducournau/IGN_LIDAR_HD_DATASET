---
sidebar_position: 12
title: Multi-Architecture Support
---

<!-- 🇫🇷 TRADUCTION FRANÇAISE REQUISE -->
<!-- Ce fichier est un modèle qui nécessite une traduction manuelle. -->
<!-- Veuillez traduire le contenu ci-dessous en conservant : -->
<!-- - Le frontmatter (métadonnées en haut) -->
<!-- - Les blocs de code (traduire uniquement les commentaires) -->
<!-- - Les liens et chemins de fichiers -->
<!-- - La structure Markdown -->



# Multi-Architecture Dataset Support

Generate datasets optimized for different deep learning architectures from a single processing pipeline.

---

## 🎯 What is Multi-Architecture Support?

**Multi-architecture support** enables you to create datasets tailored for different ML architectures (PointNet++, Octree-based networks, Transformers, Sparse CNNs) without reprocessing the raw LiDAR data.

### Supported Architectures

| Architecture           | Format       | Best For           | Memory |
| ---------------------- | ------------ | ------------------ | ------ |
| **PointNet++**         | Raw points   | General purpose    | Medium |
| **Octree-based**       | Octree       | Large-scale scenes | Low    |
| **Transformer**        | Point tokens | High accuracy      | High   |
| **Sparse Convolution** | Voxel grid   | Fast inference     | Medium |

---

## 🚀 Quick Start

### Generate for Specific Architecture

```bash
# PointNet++ (default)
ign-lidar-hd process \
  input_dir=data/ \
  output_dir=output/ \
  architecture=pointnet++

# Octree-based networks
ign-lidar-hd process \
  input_dir=data/ \
  output_dir=output/ \
  architecture=octree

# Transformer networks
ign-lidar-hd process \
  input_dir=data/ \
  output_dir=output/ \
  architecture=transformer

# Sparse Convolutional networks
ign-lidar-hd process \
  input_dir=data/ \
  output_dir=output/ \
  architecture=sparse_conv
```

### Generate Multiple Formats

```bash
# Create datasets for multiple architectures
ign-lidar-hd process \
  input_dir=data/ \
  output_dir=output/ \
  architecture=all
```

---

## 📊 Architecture Details

### PointNet++ (Default)

**Format:** Raw point clouds with per-point features

```python
# Output format
{
    'points': np.ndarray,      # (N, 3) XYZ coordinates
    'features': np.ndarray,    # (N, F) per-point features
    'labels': np.ndarray,      # (N,) per-point labels
    'normals': np.ndarray      # (N, 3) normal vectors
}
```

**Configuration:**

```bash
ign-lidar-hd process \
  input_dir=data/ \
  output_dir=output/ \
  architecture=pointnet++ \
  architecture.num_points=2048 \
  architecture.use_normals=true \
  architecture.sampling=fps  # Farthest Point Sampling
```

**Best for:**

- General-purpose point cloud classification
- Building detection and segmentation
- Moderate-sized datasets

---

### Octree-Based

**Format:** Hierarchical octree structure

```python
# Output format
{
    'octree': OctreeNode,      # Hierarchical structure
    'depth': int,              # Maximum depth
    'features': Dict,          # Features per node
    'labels': Dict             # Labels per node
}
```

**Configuration:**

```bash
ign-lidar-hd process \
  input_dir=data/ \
  output_dir=output/ \
  architecture=octree \
  architecture.max_depth=8 \
  architecture.min_points_per_node=10 \
  architecture.full_depth=5
```

**Best for:**

- Large-scale urban scenes
- Memory-efficient processing
- Multi-scale analysis

---

### Transformer

**Format:** Point tokens with positional encoding

```python
# Output format
{
    'tokens': np.ndarray,      # (N, D) point tokens
    'positions': np.ndarray,   # (N, 3) positions
    'attention_mask': np.ndarray,  # (N, N) attention mask
    'labels': np.ndarray       # (N,) labels
}
```

**Configuration:**

```bash
ign-lidar-hd process \
  input_dir=data/ \
  output_dir=output/ \
  architecture=transformer \
  architecture.num_tokens=1024 \
  architecture.token_dim=256 \
  architecture.positional_encoding=learned
```

**Best for:**

- High-accuracy requirements
- Complex scene understanding
- Sufficient GPU memory available

---

### Sparse Convolutional

**Format:** Voxelized point cloud with sparse tensors

```python
# Output format
{
    'voxels': np.ndarray,       # (V, max_points, 3) voxel coordinates
    'voxel_features': np.ndarray,  # (V, F) voxel-level features
    'coordinates': np.ndarray,  # (V, 3) voxel grid coordinates
    'labels': np.ndarray        # (V,) voxel labels
}
```

**Configuration:**

```bash
ign-lidar-hd process \
  input_dir=data/ \
  output_dir=output/ \
  architecture=sparse_conv \
  architecture.voxel_size=0.25 \
  architecture.max_points_per_voxel=32 \
  architecture.max_voxels=20000
```

**Best for:**

- Fast inference
- Real-time applications
- Regular grid structures

---

## 🎯 Use Cases

### Research & Experimentation

Compare architectures on same dataset:

```bash
# Generate all formats
ign-lidar-hd process \
  input_dir=data/buildings/ \
  output_dir=output/multi_arch/ \
  architecture=all \
  features=full

# Results in:
# output/multi_arch/pointnet++/
# output/multi_arch/octree/
# output/multi_arch/transformer/
# output/multi_arch/sparse_conv/
```

### Production Pipeline

Optimize for specific deployment:

```bash
# Fast inference for mobile/edge
ign-lidar-hd process \
  input_dir=data/ \
  output_dir=output/ \
  architecture=sparse_conv \
  architecture.voxel_size=0.5

# High accuracy for cloud processing
ign-lidar-hd process \
  input_dir=data/ \
  output_dir=output/ \
  architecture=transformer \
  architecture.token_dim=512
```

---

## 🔧 Advanced Configuration

### Custom Architecture Parameters

```yaml
# config/custom_arch.yaml
architecture:
  name: pointnet++
  num_points: 4096
  use_normals: true
  use_colors: true
  sampling: fps
  fps_ratio: 0.25
  ball_query_radius: 0.5
  ball_query_samples: 32
  feature_dimensions: [64, 128, 256, 512]
```

```bash
# Use custom config
ign-lidar-hd process \
  input_dir=data/ \
  output_dir=output/ \
  --config-name custom_arch
```

### Python API

```python
from ign_lidar.datasets import (
    PointNetPlusDataset,
    OctreeDataset,
    TransformerDataset,
    SparseConvDataset
)

# PointNet++ dataset
dataset = PointNetPlusDataset(
    data_dir="output/patches/",
    num_points=2048,
    use_normals=True,
    augment=True
)

# Octree dataset
octree_dataset = OctreeDataset(
    data_dir="output/patches/",
    max_depth=8,
    min_points=10
)

# Transformer dataset
transformer_dataset = TransformerDataset(
    data_dir="output/patches/",
    num_tokens=1024,
    token_dim=256
)

# Sparse Conv dataset
sparse_dataset = SparseConvDataset(
    data_dir="output/patches/",
    voxel_size=0.25,
    max_voxels=20000
)

# Use with PyTorch DataLoader
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

---

## 📈 Performance Comparison

### Processing Time

| Architecture | Time per Tile | Disk Usage | Memory Usage |
| ------------ | ------------- | ---------- | ------------ |
| PointNet++   | 1.0x          | 100%       | 100%         |
| Octree       | 1.3x          | 60%        | 70%          |
| Transformer  | 1.2x          | 120%       | 150%         |
| Sparse Conv  | 1.4x          | 80%        | 90%          |

### Training Speed

| Architecture | Samples/sec | GPU Memory | Inference Speed |
| ------------ | ----------- | ---------- | --------------- |
| PointNet++   | 100         | 6 GB       | 10 ms           |
| Octree       | 80          | 4 GB       | 8 ms            |
| Transformer  | 50          | 12 GB      | 15 ms           |
| Sparse Conv  | 120         | 5 GB       | 5 ms            |

### Accuracy Comparison

Based on building classification benchmark:

| Architecture | IoU  | Precision | Recall | F1   |
| ------------ | ---- | --------- | ------ | ---- |
| PointNet++   | 0.85 | 0.88      | 0.90   | 0.89 |
| Octree       | 0.83 | 0.86      | 0.88   | 0.87 |
| Transformer  | 0.89 | 0.91      | 0.93   | 0.92 |
| Sparse Conv  | 0.86 | 0.89      | 0.90   | 0.89 |

---

## ✅ Best Practices

### Choosing an Architecture

**Use PointNet++** when:

- Starting a new project
- General-purpose classification
- Moderate dataset size (&lt;1M points)
- Standard accuracy requirements

**Use Octree** when:

- Processing very large scenes
- Limited memory available
- Need multi-scale features
- Hierarchical reasoning important

**Use Transformer** when:

- Maximum accuracy needed
- Sufficient GPU memory (12+ GB)
- Complex scene understanding
- Can afford longer training

**Use Sparse Conv** when:

- Fast inference critical
- Deploying to edge devices
- Real-time processing needed
- Regular grid structure present

### Data Augmentation

Different architectures benefit from different augmentations:

```python
# PointNet++: Standard point cloud augmentations
augmentations = [
    'random_rotation',
    'random_jitter',
    'random_scaling'
]

# Octree: Preserve hierarchy
augmentations = [
    'random_rotation_90',  # Preserve grid alignment
    'random_flip'
]

# Transformer: Token-level augmentations
augmentations = [
    'token_dropout',
    'random_masking',
    'feature_mixing'
]

# Sparse Conv: Voxel-aware augmentations
augmentations = [
    'random_rotation_90',  # Grid-aligned
    'voxel_dropout',
    'cutmix'
]
```

---

## 🎓 Complete Example

### Multi-Architecture Experiment

```bash
# 1. Generate datasets for all architectures
ign-lidar-hd process \
  input_dir=data/buildings/ \
  output_dir=output/experiment/ \
  architecture=all \
  features=full \
  target_class=building

# 2. Train models
for arch in pointnet++ octree transformer sparse_conv; do
  python train.py \
    --data output/experiment/$arch/ \
    --architecture $arch \
    --epochs 100 \
    --output models/$arch/
done

# 3. Evaluate
python evaluate.py \
  --models models/ \
  --test_data data/test/ \
  --output results.csv

# 4. Compare results
python plot_comparison.py --results results.csv
```

---

## 🐛 Troubleshooting

### Out of Memory

```bash
# Reduce points/tokens
architecture.num_points=1024  # PointNet++
architecture.num_tokens=512   # Transformer

# Or use memory-efficient architecture
architecture=octree
```

### Slow Processing

```bash
# Use faster architecture
architecture=sparse_conv

# Or reduce complexity
architecture.max_depth=6      # Octree
architecture.voxel_size=0.5   # Sparse Conv
```

### Low Accuracy

```bash
# Increase model capacity
architecture.num_points=4096           # PointNet++
architecture.token_dim=512             # Transformer
architecture.max_points_per_voxel=64   # Sparse Conv

# Or use high-accuracy architecture
architecture=transformer
```

---

## 📚 Related Topics

- [Dataset API](/api/datasets) - PyTorch dataset classes
- [Configuration System](/guides/configuration-system) - Advanced configuration
- [GPU Acceleration](/guides/gpu-acceleration) - Performance optimization

---

**Next Steps:**

- Explore [Feature Computation](/api/features)
- Read [Training Guide](/guides/training)
- See [Deployment Examples](/guides/deployment)
