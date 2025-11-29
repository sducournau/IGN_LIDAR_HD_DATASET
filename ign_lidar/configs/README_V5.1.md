# IGN LiDAR HD - Configuration Guide V5.1

**Version**: 5.1.0 (Consolidated)  
**Date**: November 2025  
**Package Version**: 3.1.0  
**Compatibility**: FeatureOrchestrator V5

---

## 🎯 Overview

This directory contains the **consolidated V5.1 configuration system** for IGN LiDAR HD Dataset. Version 5.1 represents a major cleanup and harmonization effort that:

- ✅ **Simplified structure**: Reduced from 40+ to 27 configuration files
- ✅ **Eliminated duplicates**: One file per purpose
- ✅ **Clear naming**: Descriptive, consistent names
- ✅ **Modular composition**: Base configs + presets + hardware profiles
- ✅ **Production-ready**: All configs tested and documented

---

## 📁 Directory Structure

```text
configs/
├── base.yaml                # 🎯 Complete base configuration (430+ lines)
├── base/                    # 📦 Modular base components (6 files)
│   ├── README.md            #     Base components documentation
│   ├── processor.yaml       #     Core processing parameters
│   ├── features.yaml        #     Feature computation settings
│   ├── data_sources.yaml    #     BD TOPO, cadastre, OSM
│   ├── ground_truth.yaml    #     Ground truth configuration
│   ├── output.yaml          #     Output formats and options
│   └── monitoring.yaml      #     Logging and performance monitoring
├── presets/                 # 🚀 Ready-to-use presets (7 files)
│   ├── asprs_classification_cpu.yaml  # ASPRS with CPU
│   ├── asprs_classification_gpu.yaml  # ASPRS with GPU (recommended)
│   ├── lod2_buildings.yaml            # Building LOD2 classification
│   ├── lod3_detailed.yaml             # Detailed architectural LOD3
│   ├── fast_preview.yaml              # Quick testing/preview
│   ├── minimal_debug.yaml             # Minimal for debugging
│   └── high_quality.yaml              # Maximum quality output
├── advanced/                # 🔬 Specialized configurations (5 files)
│   ├── asprs_classification_gpu_optimized.yaml  # Production ASPRS
│   ├── heritage_lod3.yaml                       # Architectural heritage
│   ├── building_detection.yaml                  # Building detection
│   ├── vegetation_ndvi.yaml                     # Vegetation analysis
│   └── self_supervised.yaml                     # Self-supervised learning
├── hardware/                # ⚡ Hardware-specific profiles (5 files)
│   ├── cpu_standard.yaml           # Standard CPU (8-16 cores)
│   ├── cpu_high_end.yaml           # High-end CPU (32+ cores)
│   ├── gpu_rtx3080_12gb.yaml       # NVIDIA RTX 3080
│   ├── gpu_rtx4080_16gb.yaml       # NVIDIA RTX 4080 (recommended)
│   └── gpu_rtx4090_24gb.yaml       # NVIDIA RTX 4090 (maximum performance)
├── archive/                 # 📚 Historical documentation
│   ├── MIGRATION_V4_TO_V5.md
│   ├── CONFIG_GUIDE_V5.5_DEPRECATED.md
│   └── CONFIGURATION_GUIDE_DEPRECATED.md
└── README.md                # 📖 This guide
```

---

## 🚀 Quick Start

### Zero-Config (Simplest)

```bash
ign-lidar-hd process input_dir=/data/tiles output_dir=/data/output
```

### Use a Preset (Recommended)

```bash
# ASPRS classification with GPU (recommended)
ign-lidar-hd process -c presets/asprs_classification_gpu.yaml \
  input_dir=/data/tiles output_dir=/data/output

# LOD2 building classification
ign-lidar-hd process -c presets/lod2_buildings.yaml \
  input_dir=/data/tiles output_dir=/data/output

# Fast preview (minimal features)
ign-lidar-hd process -c presets/fast_preview.yaml \
  input_dir=/data/tiles output_dir=/data/output
```

### Compose with Hardware Profile

```bash
# ASPRS classification optimized for RTX 4080
ign-lidar-hd process -c presets/asprs_classification_gpu.yaml \
  -c hardware/gpu_rtx4080_16gb.yaml \
  input_dir=/data/tiles output_dir=/data/output

# High-end CPU fallback
ign-lidar-hd process -c presets/asprs_classification_cpu.yaml \
  -c hardware/cpu_high_end.yaml \
  input_dir=/data/tiles output_dir=/data/output
```

### Override Specific Parameters

```bash
# Adjust GPU memory target
ign-lidar-hd process -c presets/asprs_classification_gpu.yaml \
  processor.gpu_memory_target=0.95 \
  processor.gpu_batch_size=10_000_000 \
  input_dir=/data/tiles output_dir=/data/output

# Enable cadastre (slower but more accurate)
ign-lidar-hd process -c presets/asprs_classification_gpu.yaml \
  data_sources.cadastre.enabled=true \
  input_dir=/data/tiles output_dir=/data/output
```

---

## 🎯 Choosing a Configuration

| Use Case                    | Preset                          | Hardware Profile        | Example            |
| --------------------------- | ------------------------------- | ----------------------- | ------------------ |
| **Production ASPRS (GPU)**  | `asprs_classification_gpu.yaml` | `gpu_rtx4080_16gb.yaml` | Standard workflow  |
| **Production ASPRS (CPU)**  | `asprs_classification_cpu.yaml` | `cpu_high_end.yaml`     | No GPU available   |
| **Building classification** | `lod2_buildings.yaml`           | `gpu_rtx4080_16gb.yaml` | LOD2 buildings     |
| **Detailed architecture**   | `lod3_detailed.yaml`            | `gpu_rtx4090_24gb.yaml` | LOD3 architectural |
| **Quick testing**           | `fast_preview.yaml`             | -                       | Rapid iteration    |
| **Debugging**               | `minimal_debug.yaml`            | -                       | Troubleshooting    |
| **Maximum quality**         | `high_quality.yaml`             | `gpu_rtx4090_24gb.yaml` | Best results       |
| **Heritage buildings**      | `heritage_lod3.yaml`            | `gpu_rtx4080_16gb.yaml` | Cultural heritage  |
| **Vegetation analysis**     | `vegetation_ndvi.yaml`          | `gpu_rtx4080_16gb.yaml` | NDVI vegetation    |

---

## 🔧 Configuration Schema V5.1

### Base Configuration Structure

The `base.yaml` file provides complete defaults. All presets inherit from this base.

```yaml
# Configuration metadata
config_version: "5.1.0"
config_name: "base"

# Core processor settings
processor:
  lod_level: "ASPRS" # ASPRS | LOD2 | LOD3
  processing_mode: "enriched_only" # patches_only | both | enriched_only
  use_gpu: true # GPU acceleration
  gpu_batch_size: 8_000_000 # Points per GPU batch
  gpu_memory_target: 0.85 # VRAM utilization target
  num_workers: 1 # Parallel workers (1 for GPU)

# Feature computation
features:
  mode: "asprs_classes" # minimal | asprs_classes | lod2 | lod3 | full
  k_neighbors: 20 # Neighbors for features
  search_radius: 1.0 # Search radius in meters
  use_rgb: true # RGB features from orthophotos
  use_nir: false # Near-infrared features
  compute_ndvi: false # NDVI vegetation index

# Data sources
data_sources:
  bd_topo:
    enabled: true # Enable BD TOPO integration
    buildings: true # Buildings → ASPRS Class 6
    roads: true # Roads → ASPRS Class 11
    water: true # Water → ASPRS Class 9
  cadastre:
    enabled: false # Cadastre (slow but accurate)
  osm:
    enabled: false # OpenStreetMap fallback

# Output configuration
output:
  format: "laz" # laz | las | npz
  compression: "standard" # standard | maximum | none
  save_stats: true # Save processing statistics
  save_metadata: true # Save tile metadata

# Monitoring
monitoring:
  log_level: "INFO" # DEBUG | INFO | WARNING | ERROR
  enable_profiling: false # Performance profiling
  enable_gpu_metrics: true # GPU utilization metrics
```

---

## 📊 What's New in V5.1

### Changes from V5.0

1. **Reorganized Structure**: Eliminated 13 duplicate/redundant files
2. **Consistent Naming**: All files follow clear naming conventions
3. **Consolidated Documentation**: Single README.md instead of 4 overlapping guides
4. **Simplified Presets**: 7 core presets instead of 12 duplicates
5. **Hardware Profiles**: Clear naming with VRAM info (gpu_rtx4080_16gb.yaml)

### Files Removed

- ❌ `config.yaml` (redundant with base.yaml)
- ❌ `presets/asprs.yaml`, `asprs_cpu.yaml`, `asprs_rtx4080.yaml` (consolidated)
- ❌ `advanced/gpu_optimized.yaml` (functionality in base + hardware profiles)
- ❌ `advanced/enrichment_only.yaml` (use processing_mode parameter)
- ❌ Multiple duplicate advanced configs

### Files Renamed

- `lod2.yaml` → `lod2_buildings.yaml` (clearer purpose)
- `lod3.yaml` → `lod3_detailed.yaml` (clearer purpose)
- `minimal.yaml` → `minimal_debug.yaml` (clearer purpose)
- `cpu_only.yaml` → `cpu_standard.yaml` (better naming)
- `rtx4080.yaml` → `gpu_rtx4080_16gb.yaml` (includes VRAM info)

---

## 🔄 Migration from V5.0

### Configuration Loading

**V5.0 (multiple overlapping configs):**

```bash
# Confusing - which config to use?
ign-lidar-hd process --config-name config
ign-lidar-hd process --config-name base
ign-lidar-hd process --config-name asprs
ign-lidar-hd process --config-name asprs_classification_gpu
```

**V5.1 (clear, consistent):**

```bash
# Clear preset name
ign-lidar-hd process -c presets/asprs_classification_gpu.yaml
# Or relative from configs/
ign-lidar-hd process --config-name presets/asprs_classification_gpu
```

### File Mappings

| V5.0                          | V5.1                                    | Notes               |
| ----------------------------- | --------------------------------------- | ------------------- |
| `config.yaml`                 | `base.yaml`                             | Single base config  |
| `presets/asprs.yaml`          | `presets/asprs_classification_gpu.yaml` | Consolidated        |
| `presets/lod2.yaml`           | `presets/lod2_buildings.yaml`           | Renamed for clarity |
| `hardware/rtx4080.yaml`       | `hardware/gpu_rtx4080_16gb.yaml`        | Added VRAM info     |
| `advanced/gpu_optimized.yaml` | Use base.yaml + hardware profile        | Removed redundancy  |

---

## ⚡ Performance Optimization

### GPU Optimization (RTX 4080 Example)

```yaml
# Use preset + hardware profile
defaults:
  - presets/asprs_classification_gpu
  - hardware/gpu_rtx4080_16gb

# Fine-tune if needed
processor:
  gpu_batch_size: 10_000_000 # Adjust for your data
  gpu_memory_target: 0.90 # Push to 90% VRAM
  gpu_streams: 8 # Optimal for Ada Lovelace
```

**Expected Performance:**

- GPU Utilization: >85%
- Processing Time: 30-60s per 20M point tile
- Ground Truth: 10-100× faster than CPU
- Memory Efficiency: Adaptive chunking

### CPU Fallback

The system automatically falls back to CPU if:

- GPU is not available
- GPU runs out of memory
- CUDA errors occur

```yaml
processor:
  ground_truth_method: "auto" # Auto-fallback to CPU if GPU OOM
  reclassification_mode: "auto" # Auto-fallback
```

---

## 🛠️ Development & Debugging

### Debug Configuration

```bash
# Quick test with detailed logs
ign-lidar-hd process -c presets/minimal_debug.yaml \
  monitoring.log_level=DEBUG \
  monitoring.enable_profiling=true \
  input_dir=/data/test output_dir=/data/debug
```

### GPU Monitoring

```bash
# Monitor GPU during processing (separate terminal)
watch -n 1 nvidia-smi

# Or use the provided script
./scripts/gpu_monitor.sh 300  # Monitor for 5 minutes
```

### Creating Custom Presets

```yaml
# my_custom_preset.yaml
defaults:
  - base # Inherit all defaults
  - presets/asprs_classification_gpu # Start from existing preset
  - _self_ # Allow local overrides

config_name: "my_custom"
config_description: "My custom ASPRS configuration"

# Override specific parameters
processor:
  gpu_batch_size: 12_000_000 # Adjusted for my setup

data_sources:
  cadastre:
    enabled: true # Enable cadastre for my use case

features:
  k_neighbors: 40 # More neighbors for better features
```

---

## 📚 Additional Resources

- **Migration Guide**: `archive/MIGRATION_V4_TO_V5.md`
- **Base Components**: `base/README.md`
- **Examples**: `/examples/` directory (config templates)
- **Validation Scripts**: `scripts/validate_gpu_acceleration.sh`
- **Documentation**: https://sducournau.github.io/IGN_LIDAR_HD_DATASET/

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: Configuration file not found

```bash
# Solution: Use relative path from configs/ or absolute path
ign-lidar-hd process -c presets/asprs_classification_gpu.yaml
# OR
ign-lidar-hd process --config-path /full/path/to/ign_lidar/configs \
  --config-name presets/asprs_classification_gpu
```

**Issue**: GPU out of memory

```bash
# Solution: Reduce batch size or enable chunked processing
ign-lidar-hd process -c presets/asprs_classification_gpu.yaml \
  processor.gpu_batch_size=5_000_000 \
  processor.use_gpu_chunked=true
```

**Issue**: Slow processing with CPU

```bash
# Solution: Use GPU preset or increase CPU workers
ign-lidar-hd process -c presets/asprs_classification_cpu.yaml \
  -c hardware/cpu_high_end.yaml \
  processor.num_workers=16
```

---

## 📝 Configuration Version History

- **V5.1** (Nov 2025): Consolidated and reorganized (this version)
- **V5.0** (Oct 2025): Simplified, integrated optimizations
- **V4.0** (Aug 2025): Modular base configs
- **V3.2** (Jun 2025): New Config class
- **V3.1** (May 2025): ProcessorConfig + FeaturesConfig (deprecated)

---

**Questions or Issues?**

- GitHub Issues: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues
- Documentation: https://sducournau.github.io/IGN_LIDAR_HD_DATASET/
- Email: simon.ducournau@protonmail.com
