# ⚙️ Advanced Configurations

**Custom and specialized configurations for advanced use cases.**

These configurations demonstrate **advanced techniques** and **custom workflows** for specific scenarios. Use these as **templates** for your own customizations.

---

## 📋 Available Configurations

| Config                        | Scales                             | Purpose                      | Artifact Reduction | Speed      |
| ----------------------------- | ---------------------------------- | ---------------------------- | ------------------ | ---------- |
| **multi_scale_3_scales.yaml** | 3 (fine/medium/coarse)             | Standard multi-scale         | 50-75%             | ~3x slower |
| **multi_scale_4_scales.yaml** | 4 (fine/medium/coarse/very_coarse) | Maximum artifact suppression | 70-85%             | ~4x slower |

---

## 🎯 Configurations

### multi_scale_3_scales.yaml

**Standard 3-scale multi-scale computation.**

**Use when:**

- Moderate scan line artifacts
- Balanced performance vs quality
- Production use with artifact issues

**What you get:**

- ✅ 3 scale levels (fine, medium, coarse)
- ✅ Variance-weighted aggregation
- ✅ Balanced artifact suppression
- ✅ Reasonable performance overhead

**Scales:**

```yaml
scales:
  - name: fine
    k_neighbors: 20
    search_radius: 1.0
    weight: 0.3 # Lower weight (more artifacts)

  - name: medium
    k_neighbors: 50
    search_radius: 2.5
    weight: 0.4 # Balanced

  - name: coarse
    k_neighbors: 100
    search_radius: 4.0
    weight: 0.3 # Contextual features
```

**Performance:**

- ~3x slower than single-scale
- ~8-12 min per 18M point tile (GPU)
- ~25-35 min per tile (CPU, 8 cores)

**Artifact Reduction:**

- Input: 20-40% artifact rate
- Output: 5-10% artifact rate
- **Reduction: 50-75%**

```bash
ign-lidar-hd process \
  -c examples/advanced/multi_scale_3_scales.yaml \
  input_dir="/path/to/tiles" \
  output_dir="/path/to/output"
```

---

### multi_scale_4_scales.yaml

**Aggressive 4-scale multi-scale computation for severely noisy datasets.**

**Use when:**

- Severe scan line artifacts
- Noisy sensor data
- Maximum quality needed
- Processing time less critical

**What you get:**

- ✅ 4 scale levels (fine, medium, coarse, very_coarse)
- ✅ Maximum artifact suppression
- ✅ Cleanest possible features
- ✅ Higher computational cost

**Scales:**

```yaml
scales:
  - name: fine
    k_neighbors: 15
    search_radius: 0.5
    weight: 0.2 # Lowest weight (highest artifacts)

  - name: medium
    k_neighbors: 40
    search_radius: 2.0
    weight: 0.3

  - name: coarse
    k_neighbors: 80
    search_radius: 3.5
    weight: 0.3

  - name: very_coarse
    k_neighbors: 150
    search_radius: 5.0
    weight: 0.2 # Smoothest features
```

**Performance:**

- ~4x slower than single-scale
- ~12-18 min per 18M point tile (GPU)
- ~40-55 min per tile (CPU, 8 cores)

**Artifact Reduction:**

- Input: 30-50% artifact rate
- Output: 5-8% artifact rate
- **Reduction: 70-85%**

```bash
ign-lidar-hd process \
  -c examples/advanced/multi_scale_4_scales.yaml \
  input_dir="/path/to/tiles" \
  output_dir="/path/to/output"
```

---

## 🔧 Customization Guide

### Creating Your Own Multi-Scale Configuration

**Step 1: Choose Number of Scales**

| Scales | Use Case              | Performance | Artifact Reduction |
| ------ | --------------------- | ----------- | ------------------ |
| 2      | Testing, speed needed | ~2x slower  | 30-50%             |
| 3      | Standard production   | ~3x slower  | 50-75%             |
| 4      | Severe artifacts      | ~4x slower  | 70-85%             |
| 5+     | Extreme cases only    | ~5x+ slower | 80-90%             |

**Step 2: Design Scale Parameters**

Follow this pattern:

```yaml
scales:
  # Fine scale: captures details but has artifacts
  - name: fine
    k_neighbors: <small, e.g., 15-30>
    search_radius: <small, e.g., 0.5-2.0>
    weight: <low, e.g., 0.2-0.3>

  # Medium scale: balanced
  - name: medium
    k_neighbors: <medium, e.g., 40-60>
    search_radius: <medium, e.g., 2.0-3.0>
    weight: <higher, e.g., 0.3-0.4>

  # Coarse scale: smooth but loses detail
  - name: coarse
    k_neighbors: <large, e.g., 80-120>
    search_radius: <large, e.g., 3.5-5.0>
    weight: <medium, e.g., 0.3-0.4>
```

**Guidelines:**

- **k_neighbors:** Scale by ~2-3x between levels
- **search_radius:** Scale by ~2x between levels
- **weights:** Sum should equal 1.0, emphasize medium/coarse scales

**Step 3: Tune Aggregation**

```yaml
features:
  aggregation_method: "variance_weighted" # or "simple_average" for speed
  variance_penalty_factor: 2.0 # 1.5-3.0 (higher = more smoothing)
  artifact_variance_threshold: 0.15 # 0.10-0.20 (lower = more sensitive)
```

**Step 4: Test and Iterate**

```bash
# Process 1-2 test tiles
ign-lidar-hd process -c your_config.yaml \
  input_dir="/path/to/test_tiles" \
  output_dir="/path/to/test_output"

# Analyze results
# - Check artifact rate in logs
# - Visually inspect features
# - Adjust weights/thresholds
```

---

## 📊 Scale Design Patterns

### Pattern 1: Detail Preservation

**Goal:** Minimize smoothing while reducing artifacts

```yaml
scales:
  - name: fine
    k_neighbors: 30
    search_radius: 2.0
    weight: 0.5 # Emphasize fine details

  - name: coarse
    k_neighbors: 80
    search_radius: 4.0
    weight: 0.5

variance_penalty_factor: 1.5 # Gentle smoothing
```

### Pattern 2: Maximum Smoothing

**Goal:** Maximum artifact reduction (may lose some detail)

```yaml
scales:
  - name: fine
    k_neighbors: 20
    search_radius: 1.0
    weight: 0.2 # Low weight on fine scale

  - name: medium
    k_neighbors: 60
    search_radius: 3.0
    weight: 0.3

  - name: coarse
    k_neighbors: 120
    search_radius: 5.0
    weight: 0.5 # Emphasize smooth scale

variance_penalty_factor: 3.0 # Strong smoothing
```

### Pattern 3: Adaptive (Balanced)

**Goal:** Let algorithm adjust based on detected artifacts

```yaml
scales:
  - name: fine
    k_neighbors: 20
    search_radius: 1.0
    weight: 0.3

  - name: medium
    k_neighbors: 50
    search_radius: 2.5
    weight: 0.4

  - name: coarse
    k_neighbors: 100
    search_radius: 4.0
    weight: 0.3

aggregation_method: "variance_weighted" # Adaptive
variance_penalty_factor: 2.0 # Moderate
```

---

## 🧪 Experimental Configurations

### Custom Feature Subsets

Create a config with only specific features:

```yaml
features:
  mode: "custom"

  # Specify exactly which features
  feature_list:
    - "normals"
    - "curvature"
    - "planarity"
    - "height_above_ground"

  # Disable everything else
  compute_density: false
  compute_cluster_id: false
  spectral_features: false
```

### Hybrid Multi-Scale

Combine multi-scale with custom features:

```yaml
features:
  mode: "custom"
  multi_scale_computation: true

  # Only apply multi-scale to selected features
  multi_scale_features:
    - "normals"
    - "planarity"
    - "sphericity"

  # Compute these at single scale
  single_scale_features:
    - "height"
    - "density_local"
```

### GPU Memory Optimization

For large tiles with limited GPU memory:

```yaml
processor:
  use_gpu: true
  gpu_batch_size: 15_000_000 # Reduce batch size
  chunk_size: 2_000_000 # Smaller chunks
  vram_limit_gb: 8 # Conservative limit

features:
  multi_scale_computation: true
  scales:
    # Use smaller neighborhoods to reduce memory
    - name: fine
      k_neighbors: 15 # Reduced from 30
      search_radius: 1.5
      weight: 0.5

    - name: coarse
      k_neighbors: 50 # Reduced from 100
      search_radius: 3.0
      weight: 0.5
```

---

## 🔬 Analysis Tools

### Visualize Scale Contributions

After processing, analyze which scales contributed most:

```python
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load multi-scale statistics
with h5py.File('output/patches/train_0.h5', 'r') as f:
    if 'multi_scale_stats' in f:
        stats = f['multi_scale_stats'][:]

        # Plot scale weights used
        scale_names = ['fine', 'medium', 'coarse']
        avg_weights = stats.mean(axis=0)

        plt.bar(scale_names, avg_weights)
        plt.title('Average Scale Weights Used')
        plt.ylabel('Weight')
        plt.show()
```

### Artifact Detection

Check artifact rates in your data:

```python
import h5py
import numpy as np

with h5py.File('output/patches/train_0.h5', 'r') as f:
    features = f['points'][:]

    # Check variance in normals (indicator of artifacts)
    normals = features[:, :3]  # First 3 features are normals
    normal_variance = np.var(normals, axis=0).mean()

    print(f"Normal variance: {normal_variance:.6f}")
    # Lower is better (less noisy)
```

---

## 📚 Related Documentation

- **[Quickstart Configs](../quickstart/README.md)** - Get started
- **[Production Configs](../production/README.md)** - Production use
- **[Multi-Scale User Guide](../../docs/multi_scale_user_guide.md)** - Complete multi-scale documentation
- **[Configuration Reference](../../docs/docs/configuration/)** - All config options
- **[Feature Computation](../../docs/docs/features/)** - Feature details

---

## 💡 Tips for Advanced Users

### 1. Benchmark Your Configuration

```bash
# Time a single tile
time ign-lidar-hd process -c your_config.yaml \
  input_dir="/path/to/one_tile" \
  output_dir="/path/to/output"
```

### 2. Profile Memory Usage

```bash
# Monitor during processing
watch -n 1 'nvidia-smi; free -h'  # Linux
# Or use Task Manager / Resource Monitor on Windows
```

### 3. A/B Test Scale Configurations

```bash
# Process same tiles with different configs
ign-lidar-hd process -c config_3_scales.yaml input_dir="/data/tiles" output_dir="/output/3scales"
ign-lidar-hd process -c config_4_scales.yaml input_dir="/data/tiles" output_dir="/output/4scales"

# Compare artifact rates and quality
```

### 4. Document Your Custom Configs

When creating custom configs, add comprehensive comments:

```yaml
# Custom Configuration for [Purpose]
# Author: [Your Name]
# Date: [Date]
# Use Case: [Specific scenario]
# Expected Performance: [Time estimates]
# Notes: [Any special considerations]

processor:
  # ... your settings with comments ...
```

### 5. Version Control Your Configs

Store custom configs in version control:

```bash
git add examples/custom/my_config.yaml
git commit -m "Add custom config for [use case]"
```

---

**Need help?** Open an issue on [GitHub](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues) or check the [documentation](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/).

**Version:** 3.2.1  
**Last Updated:** October 25, 2025
