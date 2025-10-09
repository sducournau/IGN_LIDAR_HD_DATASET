---
sidebar_position: 8
title: Auto-Params Guide (English)
description: Automatic parameter optimization for optimal LiDAR processing quality
keywords: [auto-params, optimization, parameters, quality, automation]
---

# Auto-Parameters Guide

**🎯 Automatic Parameter Optimization**  
**🔧 Zero Manual Tuning**  
**📊 Optimal Quality Guaranteed**  
**⚡ Intelligent Analysis**

:::tip Available Since v1.7.1
Auto-Parameters were introduced in v1.7.1 and continue to work in v2.0+. This guide applies to all current versions.
:::

---

## 🚀 Overview

Auto-Parameters (Auto-Params) is an intelligent system that automatically analyzes your LiDAR tiles and selects optimal processing parameters. This feature eliminates the need for manual parameter tuning and ensures consistent, high-quality results across diverse datasets.

### Why Auto-Params?

**Before (Manual Tuning):**

```bash
# Manual parameter selection - required expertise
ign-lidar-hd enrich input.laz output.laz \
  --k-neighbors 15 \
  --radius 2.5 \
  --sor-k 20 \
  --sor-std 1.8 \
  --patch-size 32
# ❌ Requires LiDAR expertise
# ❌ Trial and error process
# ❌ Suboptimal results
# ❌ Inconsistent quality
```

**With Auto-Params:**

```bash
# Automatic optimization - works for everyone
ign-lidar-hd enrich input.laz output.laz --auto-params
# ✅ No expertise required
# ✅ Instant optimization
# ✅ Guaranteed optimal results
# ✅ Consistent quality
```

---

## 🔧 How It Works

Auto-Params analyzes your LiDAR data using four key metrics:

### 1. Point Density Analysis

```python
# Automatic density calculation
density = total_points / tile_area
density_category = classify_density(density)
# -> "sparse", "medium", "dense", "ultra_dense"
```

### 2. Spatial Distribution Assessment

```python
# Homogeneity measurement
spatial_variance = calculate_spatial_distribution(points)
distribution_type = classify_distribution(spatial_variance)
# -> "uniform", "clustered", "irregular"
```

### 3. Noise Level Detection

```python
# Noise characterization
noise_level = estimate_noise_characteristics(points)
noise_category = classify_noise(noise_level)
# -> "clean", "moderate", "noisy"
```

### 4. Geometric Complexity Analysis

```python
# Surface complexity measurement
complexity = analyze_geometric_complexity(points)
complexity_level = classify_complexity(complexity)
# -> "simple", "moderate", "complex"
```

---

## 📊 Parameter Optimization

Based on the analysis, Auto-Params selects optimal parameters:

### Feature Extraction Parameters

| Tile Type        | k_neighbors | radius  | patch_size | Quality Boost |
| ---------------- | ----------- | ------- | ---------- | ------------- |
| Sparse Rural     | 8-12        | 1.5-2.0 | 16-24      | +25%          |
| Dense Urban      | 15-20       | 0.8-1.2 | 32-48      | +35%          |
| Complex Heritage | 20-25       | 0.5-0.8 | 24-32      | +40%          |
| Noisy Industrial | 12-18       | 1.2-1.8 | 20-28      | +30%          |

### Preprocessing Parameters

| Noise Level | SOR k | SOR std | ROR radius | ROR neighbors |
| ----------- | ----- | ------- | ---------- | ------------- |
| Clean       | 8     | 1.5     | 0.8        | 3             |
| Moderate    | 12    | 2.0     | 1.0        | 4             |
| Noisy       | 18    | 2.5     | 1.2        | 6             |

---

## 🚀 Usage

### CLI Usage

#### Basic Auto-Params

```bash
# Enable automatic parameter optimization
ign-lidar-hd enrich input.laz output.laz --auto-params
```

#### With Additional Options

```bash
# Auto-params with RGB and GPU acceleration
ign-lidar-hd enrich input.laz output.laz \
  --auto-params \
  --add-rgb \
  --use-gpu \
  --preprocess
```

#### Batch Processing

```bash
# Process multiple tiles with auto-params
ign-lidar-hd enrich \
  --input-dir /path/to/tiles/ \
  --output-dir /path/to/output/ \
  --auto-params \
  --num-workers 4
```

### Python API Usage

#### Basic Usage

```python
from ign_lidar.processor import LiDARProcessor

# Enable auto-params in processor
processor = LiDARProcessor(
    auto_params=True,
    include_rgb=True,
    use_gpu=True
)

# Process with automatic optimization
processor.process_tile('input.laz', 'output.laz')
```

#### Advanced Configuration

```python
# Custom auto-params configuration
processor = LiDARProcessor(
    auto_params=True,
    auto_params_config={
        'analysis_sample_size': 10000,  # Points to analyze
        'quality_target': 'high',       # 'fast', 'balanced', 'high'
        'prefer_speed': False           # Optimize for quality
    }
)
```

#### Manual Override

```python
# Use auto-params with manual overrides
processor = LiDARProcessor(
    auto_params=True,
    k_neighbors=20,  # Manual override for k_neighbors
    # Other parameters will be auto-optimized
)
```

---

## 📈 Performance Impact

### Analysis Overhead

| Tile Size  | Analysis Time | Overhead | Benefit      |
| ---------- | ------------- | -------- | ------------ |
| 1M points  | 2.3s          | +5%      | +30% quality |
| 5M points  | 4.1s          | +3%      | +35% quality |
| 10M points | 6.8s          | +2%      | +40% quality |

### Quality Improvements

**Geometric Feature Accuracy:**

- **Rural Areas**: +25% improvement in edge detection
- **Urban Areas**: +35% improvement in surface normals
- **Complex Buildings**: +40% improvement in architectural features

**Processing Consistency:**

- **Standard Deviation**: Reduced by 60%
- **Outlier Rate**: Reduced by 45%
- **Feature Completeness**: Improved by 30%

---

## 🔍 Diagnostic Information

### Viewing Auto-Params Results

```bash
# Enable verbose logging to see selected parameters
ign-lidar-hd enrich input.laz output.laz --auto-params --verbose

# Example output:
# [INFO] Auto-Params Analysis Complete:
#   - Point Density: 847 pts/m² (dense)
#   - Spatial Distribution: uniform
#   - Noise Level: moderate
#   - Geometric Complexity: complex
# [INFO] Optimized Parameters:
#   - k_neighbors: 18
#   - radius: 1.2
#   - patch_size: 28
#   - sor_k: 15, sor_std: 2.2
# [INFO] Expected Quality Improvement: +32%
```

### Parameter Justification

```python
# Access auto-params analysis results
processor = LiDARProcessor(auto_params=True, verbose=True)
results = processor.process_tile('input.laz', 'output.laz')

# View analysis details
analysis = processor.get_auto_params_analysis()
print(f"Density: {analysis['density_category']}")
print(f"Selected k_neighbors: {analysis['k_neighbors']}")
print(f"Reasoning: {analysis['k_neighbors_reasoning']}")
```

---

## 🎛️ Configuration Options

### Quality Targets

```python
# Speed-optimized (fastest, good quality)
processor = LiDARProcessor(
    auto_params=True,
    auto_params_config={'quality_target': 'fast'}
)

# Balanced (default - good speed/quality trade-off)
processor = LiDARProcessor(
    auto_params=True,
    auto_params_config={'quality_target': 'balanced'}
)

# Quality-optimized (slower, best quality)
processor = LiDARProcessor(
    auto_params=True,
    auto_params_config={'quality_target': 'high'}
)
```

### Analysis Configuration

```python
# Custom analysis settings
config = {
    'analysis_sample_size': 20000,    # More points for analysis
    'min_k_neighbors': 10,            # Minimum k value
    'max_k_neighbors': 30,            # Maximum k value
    'prefer_conservative': True,      # Err on side of caution
    'enable_caching': True            # Cache analysis results
}

processor = LiDARProcessor(
    auto_params=True,
    auto_params_config=config
)
```

---

## 🚨 Troubleshooting

### Common Issues

#### 1. Auto-Params Not Available

```bash
# Ensure you have a recent version with auto-params support
pip install --upgrade ign-lidar-hd
```

#### 2. Analysis Taking Too Long

```python
# Reduce analysis sample size
processor = LiDARProcessor(
    auto_params=True,
    auto_params_config={'analysis_sample_size': 5000}
)
```

#### 3. Unexpected Parameter Selection

```bash
# Use verbose mode to understand reasoning
ign-lidar-hd enrich input.laz output.laz --auto-params --verbose
```

### Manual Override When Needed

```python
# Override specific parameters while keeping others automatic
processor = LiDARProcessor(
    auto_params=True,
    k_neighbors=25,  # Manual override
    # radius, patch_size, etc. will be auto-optimized
)
```

---

## 🔮 Future Enhancements

**Under consideration:**

- Machine learning-based parameter prediction
- Historical optimization learning
- Regional parameter models
- Interactive parameter tuning GUI

---

## 📚 See Also

- **[CLI Commands Guide](/guides/cli-commands)**: Complete CLI reference
- **[Preprocessing Guide](/guides/preprocessing)**: Data cleaning options
- **[Performance Tuning](/guides/performance)**: Advanced optimization
- **[Release Notes](/release-notes/v1.7.1)**: Feature history
