# Example Configurations - Quick Reference Guide

**Updated:** October 13, 2025  
**Version:** 2.4.4+

This guide helps you choose the right configuration for your use case.

---

## 🎯 Quick Decision Matrix

| **Use Case**              | **Config File**                              | **GPU** | **Output**               | **Features**    | **Memory** |
| ------------------------- | -------------------------------------------- | ------- | ------------------------ | --------------- | ---------- |
| **First time user**       | `config_complete.yaml`                       | ❌      | Patches + Enriched LAZ   | Full (32+)      | Medium     |
| **Production training**   | `config_lod3_training.yaml`                  | ✅      | Patches only             | Full (32+)      | High       |
| **GPU acceleration**      | `config_gpu_processing.yaml`                 | ✅      | Enriched LAZ only        | Full + RGB/NDVI | Medium     |
| **Low memory (8GB)**      | `config_lod3_training_memory_optimized.yaml` | ❌      | Patches only             | Simplified (12) | Low        |
| **Quick test**            | `config_quick_enrich.yaml`                   | ❌      | Enriched LAZ only        | Essential (10)  | Low        |
| **Multi-scale training**  | `config_multiscale_hybrid.yaml`              | ✅      | Patches (multiple sizes) | Full (32+)      | Very High  |
| **LOD2 (simple)**         | `config_lod2_simplified_features.yaml`       | ❌      | Patches only             | Simplified (12) | Low        |
| **LOD3 (detailed)**       | `config_lod3_full_features.yaml`             | ✅      | Patches only             | Full (34)       | High       |
| **Sequential processing** | `config_lod3_training_sequential.yaml`       | ✅      | Patches only             | Full (32+)      | Medium     |
| **50m patches**           | `config_lod3_training_50m.yaml`              | ✅      | Patches only             | Full (32+)      | Medium     |
| **100m patches**          | `config_lod3_training_100m.yaml`             | ✅      | Patches only             | Full (32+)      | Medium     |
| **150m patches**          | `config_lod3_training_150m.yaml`             | ✅      | Patches only             | Full (32+)      | High       |

---

## 📋 Detailed Configuration Guide

### 1. 🚀 **Quick Start** - `config_complete.yaml`

**Best for:** First-time users, testing the pipeline

**What it does:**

- Processes LiDAR tiles with sensible defaults
- Generates both enriched LAZ files and ML-ready patches
- Computes 32 geometric features
- Uses CPU processing (no GPU required)

**Usage:**

```bash
ign-lidar-hd process \
  --config-file examples/config_complete.yaml \
  --input-dir data/tiles/ \
  --output-dir output/
```

**Key settings:**

- Patch size: 50m
- Points per patch: 32,768
- Features: Full set (normals, curvature, planarity, etc.)
- Processing mode: `complete` (both outputs)

---

### 2. 🏭 **Production Training** - `config_lod3_training.yaml`

**Best for:** Training LOD3 building classification models

**What it does:**

- Optimized for ML training pipelines
- GPU-accelerated feature computation (6-20x faster)
- Outputs patches only (skips enriched LAZ for speed)
- Full feature set with boundary-aware processing

**Usage:**

```bash
ign-lidar-hd process \
  --config-file examples/config_lod3_training.yaml \
  --input-dir data/tiles/ \
  --output-dir patches/
```

**Key settings:**

- Patch size: 100m (balanced context)
- Points per patch: 32,768
- Features: 32+ (all geometric)
- GPU: Enabled with chunking
- Processing mode: `patches_only`

---

### 3. ⚡ **GPU Processing** - `config_gpu_processing.yaml`

**Best for:** Fast enrichment for GIS analysis, QGIS visualization

**What it does:**

- GPU-accelerated feature computation
- Outputs enriched LAZ files with features as extra dimensions
- No patch extraction (GIS-focused)
- Includes RGB and NDVI features

**Usage:**

```bash
ign-lidar-hd process \
  --config-file examples/config_gpu_processing.yaml \
  --input-dir data/tiles/ \
  --output-dir enriched/
```

**Key settings:**

- GPU: Enabled
- Features: All + RGB + infrared + NDVI
- Processing mode: `enriched_only`
- Output: LAZ with extra dimensions

---

### 4. 💾 **Memory Optimized** - `config_lod3_training_memory_optimized.yaml`

**Best for:** Systems with 8GB RAM or less

**What it does:**

- Reduced feature set (12 essential features)
- Smaller patches and batch sizes
- Sequential processing (1 tile at a time)
- No GPU (saves VRAM)

**Usage:**

```bash
ign-lidar-hd process \
  --config-file examples/config_lod3_training_memory_optimized.yaml \
  --input-dir data/tiles/ \
  --output-dir patches/
```

**Key settings:**

- Patch size: 50m
- Points per patch: 16,384 (reduced)
- Features: Simplified (12 features)
- Max tiles: 1 concurrent
- Processing mode: `sequential`

---

### 5. 🧪 **Quick Test** - `config_quick_enrich.yaml`

**Best for:** Testing pipeline, validating data

**What it does:**

- Minimal feature computation (10 essential)
- Fast enrichment only (no patches)
- Small k-neighbors for speed
- Good for data exploration

**Usage:**

```bash
ign-lidar-hd process \
  --config-file examples/config_quick_enrich.yaml \
  --input-dir data/tiles/ \
  --output-dir test/
```

**Key settings:**

- Features: Essential only (normals, curvature, height)
- k-neighbors: 10 (fast)
- Processing mode: `enriched_only`
- GPU: Optional (auto-detect)

---

### 6. 🎨 **Multi-Scale Training** - `config_multiscale_hybrid.yaml`

**Best for:** Advanced training with multiple patch sizes

**What it does:**

- Generates patches at 50m, 100m, and 150m scales
- Enables hybrid architecture training
- Best accuracy for complex scenes
- Requires significant memory and storage

**Usage:**

```bash
# Generate all scales
./examples/run_multiscale_training.sh

# Or manually
ign-lidar-hd process --config-file examples/config_lod3_training_50m.yaml
ign-lidar-hd process --config-file examples/config_lod3_training_100m.yaml
ign-lidar-hd process --config-file examples/config_lod3_training_150m.yaml

# Merge datasets
python examples/merge_multiscale_dataset.py --output patches_multiscale/
```

**Key settings:**

- Multiple patch sizes: 50m, 100m, 150m
- Full features (32+)
- GPU required
- Processing mode: `patches_only`

📚 **See [MULTI_SCALE_TRAINING_STRATEGY.md](MULTI_SCALE_TRAINING_STRATEGY.md) for complete guide**

---

### 7-9. 📐 **Specific Patch Sizes**

#### 7. **50m Patches** - `config_lod3_training_50m.yaml`

**Best for:** Fine architectural details, small buildings

- Patch size: 50m × 50m
- Points per patch: 24,576
- Best for: Individual buildings, detailed facades
- Training time: Fastest

#### 8. **100m Patches** - `config_lod3_training_100m.yaml`

**Best for:** Balanced context, most use cases

- Patch size: 100m × 100m
- Points per patch: 32,768
- Best for: General purpose, urban areas
- Training time: Medium

#### 9. **150m Patches** - `config_lod3_training_150m.yaml`

**Best for:** Full building context, large structures

- Patch size: 150m × 150m
- Points per patch: 32,768
- Best for: Large buildings, campus layouts
- Training time: Slower (more context)

---

### 10. 🔄 **Sequential Processing** - `config_lod3_training_sequential.yaml`

**Best for:** Stable processing, avoiding memory issues

**What it does:**

- Processes one tile at a time
- Prevents memory accumulation
- More stable for large datasets
- Slightly slower but more reliable

**Usage:**

```bash
ign-lidar-hd process \
  --config-file examples/config_lod3_training_sequential.yaml \
  --input-dir data/tiles/ \
  --output-dir patches/
```

**Key settings:**

- Max tiles: 1 concurrent
- Memory: Conservative settings
- Processing: Sequential mode
- Output: Patches only

---

## 🛠️ Customizing Configurations

### Override Parameters via CLI

You can override any config parameter without editing files:

```bash
# Enable GPU
ign-lidar-hd process \
  --config-file examples/config_complete.yaml \
  processor.use_gpu=true

# Change patch size
ign-lidar-hd process \
  --config-file examples/config_lod3_training.yaml \
  processor.patch_size=75.0 \
  processor.num_points=24576

# Adjust k-neighbors
ign-lidar-hd process \
  --config-file examples/config_complete.yaml \
  features.k_neighbors=30
```

### Create Custom Config

Start from a base and modify:

```yaml
# my_custom_config.yaml
defaults:
  - config_lod3_training # Inherit from this

processor:
  patch_size: 80.0 # Override specific values
  num_points: 28000
  use_gpu: true

features:
  k_neighbors: 25
  mode: "full" # full, simplified, essential, custom
```

---

## 📊 Feature Modes Explained

### Full Mode (32+ features)

- All geometric features computed
- Best accuracy for complex classification
- Requires: ~5-10 minutes per tile (CPU), ~1-2 minutes (GPU)

**Features include:**

- Normals (3D)
- Curvature
- Eigenvalues (λ1, λ2, λ3)
- Planarity, Linearity, Sphericity
- Anisotropy, Omnivariance
- Surface variation, verticality
- And more...

### Simplified Mode (12 features)

- Essential features for basic classification
- Faster computation
- Good for LOD2 or simple scenes

**Features include:**

- Normals (3D)
- Curvature
- Planarity, Linearity, Sphericity
- Eigenvalue ratios
- Height

### Essential Mode (10 features)

- Minimum for quick testing
- Very fast computation
- Good for data validation

**Features include:**

- Normals (3D)
- Curvature
- Height
- Basic geometric properties

---

## 💡 Tips & Best Practices

### 1. **Start Small**

```bash
# Test on a single tile first
ign-lidar-hd process \
  --config-file examples/config_complete.yaml \
  --input-dir data/tiles/ \
  --max-tiles 1
```

### 2. **Enable Debug Logging**

```bash
# See detailed feature flow
ign-lidar-hd process \
  --config-file examples/config_complete.yaml \
  --log-level DEBUG
```

### 3. **Check GPU Availability**

```bash
# Verify CUDA setup
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

### 4. **Monitor Memory Usage**

```bash
# Watch memory while processing
watch -n 1 free -h
```

### 5. **Validate Output**

```bash
# Check features in patches
python scripts/check_features.py patches/

# Analyze NPZ files
python scripts/analyze_npz_detailed.py patches/patch_0001.npz
```

---

## 🔧 Troubleshooting

### "Out of memory" errors

→ Use `config_lod3_training_memory_optimized.yaml`  
→ Reduce `processor.num_points` or `processor.patch_size`  
→ Enable sequential mode: `processor.max_concurrent_tiles=1`

### "CUDA out of memory"

→ Enable chunking: `processor.use_chunked_processing=true`  
→ Reduce chunk size: `processor.chunk_size=50000`  
→ Fall back to CPU: `processor.use_gpu=false`

### "Only 12 features saved" (Feature loss bug)

→ Enable debug logging: `--log-level DEBUG`  
→ Check logs for `[FEATURE_FLOW]` markers  
→ See FEATURE_LOSS_ROOT_CAUSE.md for details

### "RGB/NIR not found"

→ Ensure RGB tiles are in correct directory  
→ Set `rgb_cache_dir` in config  
→ Or disable: `features.use_rgb=false`

---

## 📚 Additional Resources

- **[MULTI_SCALE_TRAINING_STRATEGY.md](MULTI_SCALE_TRAINING_STRATEGY.md)** - Advanced multi-scale training guide
- **[MULTISCALE_QUICK_REFERENCE.md](MULTISCALE_QUICK_REFERENCE.md)** - Quick reference for multi-scale configs
- **[FEATURE_LOSS_ROOT_CAUSE.md](../FEATURE_LOSS_ROOT_CAUSE.md)** - Feature loss bug analysis
- **[Main README](../README.md)** - Package documentation

---

## 🆕 What's New in v2.4.4+

- ✅ **Enhanced debug logging** for feature tracking
- ✅ **Unified logging configuration** (`logging_config.py`)
- ✅ **Better error messages** with actionable suggestions
- ✅ **Feature flow tracking** to diagnose feature loss bug
- 🔄 **Modular architecture** (ongoing refactor)

---

## 📋 Complete File List

```
examples/
├── README.md                                      # This file
├── config_complete.yaml                           # Complete pipeline (recommended start)
├── config_enriched_only.yaml                      # Enrichment only (no patches)
├── config_gpu_processing.yaml                     # GPU-accelerated enrichment
├── config_lod2_simplified_features.yaml           # Simplified features (LOD2)
├── config_lod3_full_features.yaml                 # All features (LOD3)
├── config_lod3_training.yaml                      # Production training
├── config_lod3_training_50m.yaml                  # 50m patches
├── config_lod3_training_100m.yaml                 # 100m patches
├── config_lod3_training_150m.yaml                 # 150m patches
├── config_lod3_training_memory_optimized.yaml     # Low memory
├── config_lod3_training_sequential.yaml           # Sequential processing
├── config_multiscale_hybrid.yaml                  # Multi-scale training
├── config_quick_enrich.yaml                       # Quick test
├── config_training_dataset.yaml                   # Training dataset generation
├── semantic_sota.yaml                             # State-of-the-art semantic seg
├── run_multiscale_training.sh                     # Automated multi-scale pipeline
└── merge_multiscale_dataset.py                    # Merge multi-scale datasets
```

---

**Questions?** Open an issue on GitHub or check the main documentation.

**Version:** 2.4.4+  
**Last Updated:** October 13, 2025
