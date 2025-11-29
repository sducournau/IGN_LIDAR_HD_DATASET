# IGN LiDAR HD v4.0 Configuration Presets

**Configuration System Version:** 4.0.0  
**Package Version:** 4.0.0+  
**Date:** November 2025

---

## 📋 Overview

This directory contains **v4.0 unified configuration presets** for IGN LiDAR HD processing. These presets use the new flat configuration structure introduced in v4.0 for improved clarity and consistency.

### What's New in v4.0?

- **Flat structure** for essential parameters (no more nested `processor.*`)
- **Consistent naming** across Python/YAML/CLI (`mode` instead of `lod_level`)
- **Optimizations section** for Phase 4 performance features
- **Simplified hierarchy** - advanced options nested, common options at top level

---

## 🎯 Available Presets

### 1. **minimal_debug.yaml** ⚡

**Speed:** FASTEST (2-3× faster than standard)

**Use Cases:**

- Quick testing and debugging
- Development iteration
- CI/CD integration
- Initial data exploration

**Features:** ~8 minimal features (height, normals, planarity)  
**Hardware:** Any (CPU optimized)  
**Performance:** ~2-3 min per 20M point tile

```bash
ign-lidar-hd process --config-name=presets_v4/minimal_debug \
  input_dir=/data/tiles output_dir=/data/output
```

---

### 2. **fast_preview.yaml** 🚀

**Speed:** FAST (optimized for quick results)

**Use Cases:**

- Quick quality checks
- Testing new datasets
- Rapid prototyping
- Preview before full processing

**Features:** ~8 minimal features, basic BD TOPO  
**Hardware:** Any (CPU with 8+ cores recommended)  
**Performance:** ~2-3 min per 20M point tile (GPU), ~5-8 min (CPU)

```bash
ign-lidar-hd process --config-name=presets_v4/fast_preview \
  input_dir=/data/tiles output_dir=/data/output
```

---

### 3. **lod2_buildings.yaml** 🏢

**Speed:** FAST (efficient for building tasks)

**Use Cases:**

- Building facade detection
- Roof plane extraction
- Urban modeling
- Building footprint refinement

**Classification:** LOD2 classes (walls, roofs, chimneys, etc.)  
**Features:** ~20 standard features  
**Hardware:** GPU recommended  
**Performance:** ~5-10 min per 20M point tile

```bash
ign-lidar-hd process --config-name=presets_v4/lod2_buildings \
  input_dir=/data/tiles output_dir=/data/output
```

---

### 4. **lod3_detailed.yaml** 🏛️

**Speed:** MEDIUM (high detail)

**Use Cases:**

- Detailed architectural analysis
- Heritage documentation
- High-fidelity city models
- Research applications

**Classification:** LOD3 classes (15+ detailed architectural features)  
**Features:** ~45 full features + multi-scale  
**Hardware:** GPU strongly recommended (16GB+ VRAM)  
**Performance:** ~15-20 min per 20M point tile

```bash
ign-lidar-hd process --config-name=presets_v4/lod3_detailed \
  input_dir=/data/tiles output_dir=/data/output
```

---

### 5. **asprs_classification_cpu.yaml** 💻

**Speed:** MEDIUM-SLOW (CPU optimized)

**Use Cases:**

- ASPRS classification on CPU-only systems
- Standard point cloud classification
- Systems without GPU

**Classification:** ASPRS LAS 1.4 standard classes  
**Features:** ~20 standard features  
**Hardware:** CPU (8+ cores, 32GB+ RAM recommended)  
**Performance:** ~45-60 min per 20M point tile

```bash
ign-lidar-hd process --config-name=presets_v4/asprs_classification_cpu \
  input_dir=/data/tiles output_dir=/data/output
```

---

### 6. **asprs_classification_gpu.yaml** 🎮

**Speed:** FAST (GPU accelerated)

**Use Cases:**

- Production ASPRS classification
- Large-scale processing
- GPU-equipped systems

**Classification:** ASPRS LAS 1.4 standard classes  
**Features:** ~45 full features + spectral  
**Hardware:** GPU required (12GB+ VRAM recommended)  
**Performance:** ~30-60s per 20M point tile

```bash
ign-lidar-hd process --config-name=presets_v4/asprs_classification_gpu \
  input_dir=/data/tiles output_dir=/data/output
```

---

### 7. **high_quality.yaml** 🌟

**Speed:** MEDIUM (maximum quality)

**Use Cases:**

- Production datasets
- Research publications
- Heritage documentation
- Official deliverables

**Classification:** LOD3 with all features  
**Features:** ALL ~45 features + multi-scale + spectral  
**Hardware:** High-end GPU required (16GB+ VRAM)  
**Performance:** ~15-25 min per 20M point tile

```bash
ign-lidar-hd process --config-name=presets_v4/high_quality \
  input_dir=/data/tiles output_dir=/data/output
```

---

## 🔧 Customization

### Override Parameters

All presets can be customized via CLI overrides:

```bash
# Override GPU usage
ign-lidar-hd process --config-name=presets_v4/lod2_buildings \
  use_gpu=false num_workers=8 \
  input_dir=/data/tiles output_dir=/data/output

# Override feature mode
ign-lidar-hd process --config-name=presets_v4/lod2_buildings \
  features.mode=full features.k_neighbors=60 \
  input_dir=/data/tiles output_dir=/data/output

# Enable optimizations
ign-lidar-hd process --config-name=presets_v4/minimal_debug \
  optimizations.enabled=true optimizations.batch_size=8 \
  input_dir=/data/tiles output_dir=/data/output
```

### Create Custom Preset

Copy and modify any preset:

```bash
cp presets_v4/lod2_buildings.yaml presets_v4/my_custom.yaml
# Edit my_custom.yaml...
ign-lidar-hd process --config-name=presets_v4/my_custom \
  input_dir=/data/tiles output_dir=/data/output
```

---

## 📊 Preset Comparison

| Preset             | Speed  | Features | GPU         | Use Case              |
| ------------------ | ------ | -------- | ----------- | --------------------- |
| **minimal_debug**  | ⚡⚡⚡ | ~8       | Optional    | Testing/Debug         |
| **fast_preview**   | ⚡⚡   | ~8       | Optional    | Quick preview         |
| **lod2_buildings** | ⚡     | ~20      | Recommended | Building modeling     |
| **asprs_cpu**      | 🐢     | ~20      | No          | CPU systems           |
| **asprs_gpu**      | ⚡     | ~45      | Required    | Production ASPRS      |
| **lod3_detailed**  | 🏃     | ~45      | Required    | Detailed architecture |
| **high_quality**   | 🏃     | ~45+     | Required    | Maximum quality       |

---

## 🔄 Migration from v3.x/v5.1

### Automatic Migration

Use the migration tool:

```bash
ign-lidar-hd migrate-config old_config.yaml new_config_v4.yaml
```

### Key Changes

1. **Flat structure**: `processor.lod_level` → `mode`
2. **Lowercase modes**: `LOD2` → `lod2`
3. **Feature parameter**: `features.feature_set` → `features.mode`
4. **New optimizations section**: Phase 4 optimizations consolidated

### Example Migration

**Old (v5.1):**

```yaml
processor:
  lod_level: "LOD2"
  use_gpu: true
  processing_mode: "enriched_only"

features:
  mode: "lod2"
```

**New (v4.0):**

```yaml
mode: lod2
use_gpu: true
processing_mode: enriched_only

features:
  mode: standard
```

---

## 📚 Documentation

- **Configuration Guide**: https://sducournau.github.io/IGN_LIDAR_HD_DATASET/configuration/
- **Migration Guide**: https://sducournau.github.io/IGN_LIDAR_HD_DATASET/migration-v3-to-v4/
- **API Reference**: https://sducournau.github.io/IGN_LIDAR_HD_DATASET/api/config/

---

## 🆘 Support

- **GitHub Issues**: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues
- **Discussions**: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/discussions
- **Documentation**: https://sducournau.github.io/IGN_LIDAR_HD_DATASET/

---

## ✅ Validation

Test a preset without processing:

```bash
# Validate configuration
ign-lidar-hd validate-config --config-name=presets_v4/lod2_buildings \
  input_dir=/data/tiles output_dir=/data/output

# Show effective configuration
ign-lidar-hd show-config --config-name=presets_v4/lod2_buildings
```

---

**Last Updated:** November 2025  
**Config Version:** 4.0.0  
**Package Version:** 4.0.0+
