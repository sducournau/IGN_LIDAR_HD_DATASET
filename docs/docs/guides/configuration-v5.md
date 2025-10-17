# Configuration V5 System

**Version**: 5.0.0  
**Status**: ✅ Current  
**Last Updated**: October 17, 2025

---

## 🎯 Overview

The **V5 Configuration System** represents a **60% reduction in complexity** compared to V4, simplifying configuration management while maintaining full flexibility and power.

### Key Improvements

- **5 Base Configs** (down from 14 in V4)
- **80 Parameters** (down from 200+ in V4)
- **Simplified Structure**: Logical grouping by function
- **Better Defaults**: Sensible defaults for most use cases
- **Clearer Overrides**: Intuitive parameter hierarchy
- **Faster Processing**: Reduced configuration overhead

---

## 📂 Configuration Structure

### V5 Architecture

```yaml
# Any configuration file
defaults:
  - base/processor # Core processing settings
  - base/features # Feature computation
  - base/data_sources # BD TOPO, RPG, Cadastre
  - base/output # Output formats and options
  - base/monitoring # Logging and metrics
  - _self_ # Current file overrides

# Override any defaults here
processor:
  batch_size: 32

features:
  compute_normals: true
```

### Directory Structure

```
ign_lidar/configs/
├── base/                          # Base configurations
│   ├── processor.yaml             # Core processing
│   ├── features.yaml              # Feature computation
│   ├── data_sources.yaml          # Data sources (BD TOPO, etc.)
│   ├── output.yaml                # Output settings
│   └── monitoring.yaml            # Logging & metrics
│
├── presets/                       # Pre-configured presets
│   ├── asprs_classification_gpu_optimized.yaml
│   ├── lod2_classification.yaml
│   ├── lod3_classification.yaml
│   └── fast_prototyping.yaml
│
└── hardware/                      # Hardware-specific profiles
    ├── rtx_4080_super.yaml
    ├── rtx_3080.yaml
    └── cpu_optimized.yaml
```

---

## 🔧 Base Configurations

### 1. Processor Configuration (`base/processor.yaml`)

**Purpose**: Core processing settings for tile handling and computation

```yaml
# ign_lidar/configs/base/processor.yaml
processor:
  # Batch processing
  batch_size: 16 # Tiles per batch
  num_workers: 4 # Parallel workers

  # Memory management
  max_memory_mb: 8192 # Max memory per process
  chunk_size: 1_000_000 # Points per chunk

  # Performance
  use_gpu: false # GPU acceleration
  gpu_device: 0 # GPU device ID

  # Quality
  validate_outputs: true # Validate processed tiles
  skip_existing: true # Skip already processed tiles
```

**Key Parameters**:

| Parameter          | Default | Description                        |
| ------------------ | ------- | ---------------------------------- |
| `batch_size`       | 16      | Number of tiles processed together |
| `num_workers`      | 4       | Parallel processing workers        |
| `max_memory_mb`    | 8192    | Maximum memory per process (MB)    |
| `chunk_size`       | 1M      | Points processed per chunk         |
| `use_gpu`          | false   | Enable GPU acceleration            |
| `gpu_device`       | 0       | CUDA device ID (0-N)               |
| `validate_outputs` | true    | Validate output files              |
| `skip_existing`    | true    | Skip already processed tiles       |

### 2. Features Configuration (`base/features.yaml`)

**Purpose**: Feature computation settings (normals, curvature, RGB, etc.)

```yaml
# ign_lidar/configs/base/features.yaml
features:
  # Geometric features
  compute_normals: true
  compute_curvature: false
  compute_roughness: false

  # Neighborhood
  k_neighbors: 50 # Points for normal estimation
  search_radius: 0.5 # Meters for feature computation

  # RGB augmentation
  rgb_augmentation:
    enabled: false
    method: "orthophoto" # "orthophoto" or "satellite"
    cache_dir: null # Auto-set to input_dir/cache

  # Infrared
  infrared:
    enabled: false
    source: "orthohr" # IGN OrthoHR source

  # NDVI vegetation refinement
  ndvi:
    enabled: false
    threshold: 0.3 # NDVI threshold for vegetation
```

**Key Parameters**:

| Parameter                  | Default | Description                       |
| -------------------------- | ------- | --------------------------------- |
| `compute_normals`          | true    | Compute normal vectors            |
| `compute_curvature`        | false   | Compute curvature (CPU intensive) |
| `k_neighbors`              | 50      | K-nearest neighbors for normals   |
| `search_radius`            | 0.5     | Search radius in meters           |
| `rgb_augmentation.enabled` | false   | Enable RGB from orthophotos       |
| `infrared.enabled`         | false   | Enable infrared channel           |
| `ndvi.enabled`             | false   | Enable NDVI refinement            |

### 3. Data Sources Configuration (`base/data_sources.yaml`)

**Purpose**: External data sources (BD TOPO®, RPG, Cadastre)

```yaml
# ign_lidar/configs/base/data_sources.yaml
data_sources:
  # BD TOPO (IGN topographic database)
  bd_topo:
    enabled: false
    features:
      buildings: true # ASPRS Class 6
      roads: true # ASPRS Class 11
      water: true # ASPRS Class 9
      vegetation: true # ASPRS Class 3/4/5

    # WFS Configuration
    wfs_url: "https://data.geopf.fr/wfs"
    max_features: 10000
    timeout: 30

    # Cache configuration
    cache_enabled: true
    cache_dir: null # Auto: {input_dir}/cache/ground_truth
    use_global_cache: false

  # RPG (Agricultural land registry)
  rpg:
    enabled: false
    wfs_url: "https://data.geopf.fr/wfs"

  # Cadastre (Land parcels)
  cadastre:
    enabled: false
    api_url: "https://cadastre.data.gouv.fr/bundler/cadastre-etalab"
```

**Key Parameters**:

| Parameter               | Default | Description                   |
| ----------------------- | ------- | ----------------------------- |
| `bd_topo.enabled`       | false   | Enable BD TOPO ground truth   |
| `bd_topo.features.*`    | true    | Enable specific feature types |
| `bd_topo.cache_enabled` | true    | Cache WFS responses           |
| `bd_topo.cache_dir`     | null    | Cache location (auto-detect)  |
| `rpg.enabled`           | false   | Enable RPG agricultural data  |
| `cadastre.enabled`      | false   | Enable cadastre parcels       |

### 4. Output Configuration (`base/output.yaml`)

**Purpose**: Output formats, compression, and validation

```yaml
# ign_lidar/configs/base/output.yaml
output:
  # Format preferences
  formats:
    laz: true # LAZ compressed format
    las: false # Uncompressed LAS
    parquet: false # Apache Parquet

  # Compression
  compression:
    laz_backend: "laszip" # "laszip" or "lazrs"
    compression_level: 7 # 0-9 (higher = smaller, slower)

  # LAZ fields
  extra_dims:
    - name: "Curvature"
      type: "float32"
    - name: "Roughness"
      type: "float32"

  # Validation
  validate_format: true # Validate output format
  validate_crs: true # Validate coordinate system

  # File naming
  output_suffix: "_enriched" # Suffix for output files
  preserve_structure: true # Preserve input directory structure
```

**Key Parameters**:

| Parameter                 | Default      | Description             |
| ------------------------- | ------------ | ----------------------- |
| `formats.laz`             | true         | Output LAZ compressed   |
| `formats.las`             | false        | Output uncompressed LAS |
| `formats.parquet`         | false        | Output Apache Parquet   |
| `compression.laz_backend` | "laszip"     | LAZ compression backend |
| `compression_level`       | 7            | Compression level (0-9) |
| `validate_format`         | true         | Validate output files   |
| `output_suffix`           | "\_enriched" | Output filename suffix  |

### 5. Monitoring Configuration (`base/monitoring.yaml`)

**Purpose**: Logging, metrics, and progress tracking

```yaml
# ign_lidar/configs/base/monitoring.yaml
monitoring:
  # Logging
  log_level: "INFO" # DEBUG, INFO, WARNING, ERROR
  log_file: null # Log to file (null = stdout only)

  # Progress
  show_progress: true # Show progress bars
  progress_interval: 1.0 # Update interval (seconds)

  # Metrics
  metrics:
    enabled: true
    track_memory: true # Track memory usage
    track_timing: true # Track processing time
    track_gpu: false # Track GPU metrics

  # Reporting
  summary_report: true # Print summary at end
  detailed_report: false # Print detailed per-tile report
```

**Key Parameters**:

| Parameter              | Default | Description                   |
| ---------------------- | ------- | ----------------------------- |
| `log_level`            | "INFO"  | Logging level                 |
| `log_file`             | null    | Log file path (null = stdout) |
| `show_progress`        | true    | Show progress bars            |
| `metrics.enabled`      | true    | Track performance metrics     |
| `metrics.track_memory` | true    | Track memory usage            |
| `metrics.track_gpu`    | false   | Track GPU metrics             |
| `summary_report`       | true    | Print processing summary      |

---

## 🎨 Preset Configurations

### 1. ASPRS Classification (GPU Optimized)

**File**: `presets/asprs_classification_gpu_optimized.yaml`

**Use Case**: ASPRS LAS 1.4 classification with GPU acceleration

```yaml
defaults:
  - base/processor
  - base/features
  - base/data_sources
  - base/output
  - base/monitoring
  - _self_

# GPU-optimized processing
processor:
  batch_size: 32 # Larger batches for GPU
  use_gpu: true
  gpu_device: 0
  chunk_size: 2_000_000 # Larger chunks for GPU

# Enhanced features
features:
  compute_normals: true
  compute_curvature: true
  k_neighbors: 50

  # RGB from orthophotos
  rgb_augmentation:
    enabled: true
    method: "orthophoto"
    resolution: 0.2

# Ground truth classification
data_sources:
  bd_topo:
    enabled: true
    features:
      buildings: true # ASPRS Class 6
      roads: true # ASPRS Class 11
      water: true # ASPRS Class 9
      vegetation: true # ASPRS Class 3/4/5
    cache_enabled: true

# Output settings
output:
  formats:
    laz: true
  compression_level: 7
  validate_format: true

# Monitoring
monitoring:
  log_level: "INFO"
  metrics:
    enabled: true
    track_gpu: true
```

**Performance** (RTX 4080 Super):

- **Throughput**: ~15 tiles/hour
- **Memory**: ~6 GB GPU, ~12 GB RAM
- **Features**: Normals, Curvature, RGB, ASPRS Classes

### 2. LOD2 Classification

**File**: `presets/lod2_classification.yaml`

**Use Case**: Building-focused LOD2 classification (15 classes)

```yaml
defaults:
  - base/processor
  - base/features
  - base/data_sources
  - base/output
  - base/monitoring
  - _self_

# Classification mode
classification:
  mode: "lod2" # LOD2 taxonomy (15 classes)
  source: "asprs" # Map from ASPRS

# Standard processing
processor:
  batch_size: 16
  use_gpu: false

# Basic features
features:
  compute_normals: true
  compute_curvature: false
  k_neighbors: 50

# Ground truth
data_sources:
  bd_topo:
    enabled: true
    features:
      buildings: true # Primary focus for LOD2

# Output
output:
  formats:
    laz: true
  output_suffix: "_lod2"
```

**LOD2 Classes** (15 total):

- Building components (walls, roofs, floors)
- Context (ground, vegetation, water)
- Transport infrastructure

### 3. LOD3 Classification

**File**: `presets/lod3_classification.yaml`

**Use Case**: Detailed architectural LOD3 classification (30 classes)

```yaml
defaults:
  - base/processor
  - base/features
  - base/data_sources
  - base/output
  - base/monitoring
  - _self_

# Classification mode
classification:
  mode: "lod3" # LOD3 taxonomy (30 classes)
  source: "asprs" # Map from ASPRS

# Enhanced processing
processor:
  batch_size: 8 # Smaller batches for detail
  use_gpu: false

# Full features
features:
  compute_normals: true
  compute_curvature: true # Important for architectural details
  compute_roughness: true
  k_neighbors: 50

# Ground truth
data_sources:
  bd_topo:
    enabled: true
    features:
      buildings: true # Primary focus
      roads: true
      vegetation: true

# Output
output:
  formats:
    laz: true
  output_suffix: "_lod3"
  extra_dims:
    - name: "Curvature"
      type: "float32"
    - name: "Roughness"
      type: "float32"
```

**LOD3 Classes** (30 total):

- Detailed building elements (windows, doors, balconies)
- Roof details (chimneys, dormers, gutters)
- Architectural features

### 4. Fast Prototyping

**File**: `presets/fast_prototyping.yaml`

**Use Case**: Quick testing and development

```yaml
defaults:
  - base/processor
  - base/features
  - base/data_sources
  - base/output
  - base/monitoring
  - _self_

# Minimal processing
processor:
  batch_size: 8
  use_gpu: false
  skip_existing: false # Always reprocess

# Minimal features
features:
  compute_normals: false # Skip expensive features
  compute_curvature: false
  rgb_augmentation:
    enabled: false

# No ground truth
data_sources:
  bd_topo:
    enabled: false

# Fast output
output:
  formats:
    laz: true
  compression_level: 1 # Fast compression
  validate_format: false # Skip validation

# Verbose monitoring
monitoring:
  log_level: "DEBUG"
  show_progress: true
  detailed_report: true
```

**Performance**: ~30 tiles/hour (minimal features)

---

## 🖥️ Hardware Profiles

### RTX 4080 Super Profile

**File**: `hardware/rtx_4080_super.yaml`

```yaml
# Optimized for RTX 4080 Super
# - 16 GB VRAM
# - 9728 CUDA cores
# - 320W TDP

processor:
  batch_size: 32
  use_gpu: true
  gpu_device: 0
  chunk_size: 2_000_000
  max_memory_mb: 12288 # 12 GB RAM

features:
  k_neighbors: 50
  rgb_augmentation:
    enabled: true
    batch_size: 16 # GPU batch for RGB

monitoring:
  metrics:
    track_gpu: true
```

**Expected Performance**:

- **Throughput**: 15-20 tiles/hour (with full features)
- **GPU Memory**: ~6 GB VRAM used
- **System Memory**: ~12 GB RAM used

### RTX 3080 Profile

**File**: `hardware/rtx_3080.yaml`

```yaml
# Optimized for RTX 3080
# - 10 GB VRAM
# - 8704 CUDA cores

processor:
  batch_size: 16 # Smaller batch (less VRAM)
  use_gpu: true
  gpu_device: 0
  chunk_size: 1_500_000
  max_memory_mb: 10240 # 10 GB RAM

features:
  k_neighbors: 50
  rgb_augmentation:
    enabled: true
    batch_size: 8 # Smaller GPU batch

monitoring:
  metrics:
    track_gpu: true
```

**Expected Performance**:

- **Throughput**: 10-15 tiles/hour
- **GPU Memory**: ~5 GB VRAM used
- **System Memory**: ~10 GB RAM used

### CPU Optimized Profile

**File**: `hardware/cpu_optimized.yaml`

```yaml
# Optimized for CPU-only processing
# - Multi-core utilization
# - Memory-efficient

processor:
  batch_size: 4 # Smaller batches
  use_gpu: false
  num_workers: 8 # More workers for CPU
  chunk_size: 500_000 # Smaller chunks
  max_memory_mb: 8192

features:
  k_neighbors: 30 # Fewer neighbors (faster)
  compute_curvature: false # Skip expensive features
  rgb_augmentation:
    enabled: false # RGB is slow on CPU

monitoring:
  metrics:
    track_gpu: false
```

**Expected Performance**:

- **Throughput**: 2-5 tiles/hour (basic features)
- **CPU Usage**: 80-100% (multi-core)
- **Memory**: ~8 GB RAM used

---

## 📖 Usage Examples

### Example 1: Use Preset Configuration

```bash
# Use ASPRS GPU preset
ign-lidar-hd process \
  --config-name asprs_classification_gpu_optimized \
  input_dir=data/tiles/ \
  output_dir=output/
```

### Example 2: Preset + Hardware Profile

```bash
# Combine preset + hardware profile
ign-lidar-hd process \
  --config-name asprs_classification_gpu_optimized \
  --config-path hardware/rtx_4080_super \
  input_dir=data/tiles/ \
  output_dir=output/
```

### Example 3: Preset + Overrides

```bash
# Override specific parameters
ign-lidar-hd process \
  --config-name lod2_classification \
  input_dir=data/tiles/ \
  output_dir=output/ \
  processor.batch_size=32 \
  features.compute_curvature=true
```

### Example 4: Custom Configuration

```yaml
# custom_config.yaml
defaults:
  - base/processor
  - base/features
  - base/data_sources
  - base/output
  - base/monitoring
  - _self_

# Custom overrides
processor:
  batch_size: 24
  use_gpu: true

features:
  compute_normals: true
  compute_curvature: true
  rgb_augmentation:
    enabled: true
    method: "orthophoto"
    resolution: 0.2

data_sources:
  bd_topo:
    enabled: true
    features:
      buildings: true
      roads: true
      water: true
```

```bash
# Use custom config
ign-lidar-hd process \
  --config-path . \
  --config-name custom_config \
  input_dir=data/tiles/ \
  output_dir=output/
```

---

## 🔄 Configuration Override Hierarchy

### Priority Order (Highest to Lowest)

1. **CLI Arguments**: `processor.batch_size=32`
2. **Current Config (`_self_`)**: Parameters in your config file
3. **Preset Configs**: Parameters from presets
4. **Base Configs**: Default parameters from `base/`

### Example Override Flow

```yaml
# base/processor.yaml (Priority 4 - Lowest)
processor:
  batch_size: 16
  use_gpu: false

# presets/asprs_classification_gpu_optimized.yaml (Priority 3)
defaults:
  - base/processor
  - _self_

processor:
  batch_size: 32         # Overrides base
  use_gpu: true          # Overrides base

# custom_config.yaml (Priority 2)
defaults:
  - presets/asprs_classification_gpu_optimized
  - _self_

processor:
  batch_size: 64         # Overrides preset AND base

# CLI argument (Priority 1 - Highest)
processor.batch_size=128  # Overrides everything
```

**Final Value**: `batch_size = 128` (from CLI)

---

## 💡 Best Practices

### 1. Start with Presets

Use presets as a starting point, override only what you need:

```bash
# Good: Use preset + minimal overrides
ign-lidar-hd process \
  --config-name asprs_classification_gpu_optimized \
  input_dir=data/ \
  processor.batch_size=64

# Avoid: Writing entire config from scratch
```

### 2. Use Hardware Profiles

Match your hardware capabilities:

```yaml
# For RTX 4080 Super
defaults:
  - hardware/rtx_4080_super
  - presets/asprs_classification_gpu_optimized
  - _self_

# For CPU only
defaults:
  - hardware/cpu_optimized
  - presets/lod2_classification
  - _self_
```

### 3. Keep Overrides Minimal

Only override what you need to change:

```yaml
# Good: Minimal overrides
defaults:
  - presets/asprs_classification_gpu_optimized
  - _self_

processor:
  batch_size: 64 # Only override batch_size

# Avoid: Repeating all default values
```

### 4. Use Descriptive Config Names

```yaml
# Good naming
config_versailles_asprs_gpu.yaml
config_paris_lod3_cpu.yaml
config_lyon_prototyping.yaml

# Avoid generic names
config1.yaml
my_config.yaml
test.yaml
```

### 5. Document Your Configs

```yaml
# config_versailles_asprs.yaml
# Purpose: ASPRS classification for Versailles dataset
# Hardware: RTX 4080 Super
# Features: Normals, Curvature, RGB, Ground Truth
# Expected: ~15 tiles/hour

defaults:
  - presets/asprs_classification_gpu_optimized
  - hardware/rtx_4080_super
  - _self_

# Versailles-specific overrides
processor:
  batch_size: 32 # Optimal for this dataset
```

---

## 🔍 Configuration Validation

### Check Effective Configuration

```bash
# Show final merged configuration
ign-lidar-hd process \
  --config-name asprs_classification_gpu_optimized \
  --cfg job
```

### Validate Configuration

```python
from ign_lidar.core.config import load_config
from omegaconf import OmegaConf

# Load configuration
cfg = load_config("asprs_classification_gpu_optimized")

# Print configuration
print(OmegaConf.to_yaml(cfg))

# Validate structure
from ign_lidar.core.config import validate_config
validate_config(cfg)  # Raises error if invalid
```

---

## 📊 Comparison: V4 vs V5

| Aspect                | V4        | V5            | Improvement          |
| --------------------- | --------- | ------------- | -------------------- |
| **Base Configs**      | 14 files  | 5 files       | **64% reduction**    |
| **Total Parameters**  | 200+      | 80            | **60% reduction**    |
| **Config Complexity** | High      | Low           | **Simplified**       |
| **Override Clarity**  | Unclear   | Clear         | **Better hierarchy** |
| **Preset System**     | Limited   | Comprehensive | **More presets**     |
| **Hardware Profiles** | None      | Yes           | **New feature**      |
| **Documentation**     | Scattered | Consolidated  | **Better docs**      |

---

## 🚀 Next Steps

1. **Choose a Preset**: Start with a preset matching your use case
2. **Select Hardware Profile**: Match your GPU/CPU configuration
3. **Override as Needed**: Customize only what you need
4. **Validate Config**: Check merged configuration
5. **Run Processing**: Execute with your configuration

---

## 📚 Related Documentation

- [Migration V4 → V5 Guide](./migration-v4-to-v5.md) - Migrate from V4 to V5
- [Processing Modes](./processing-modes.md) - ASPRS, LOD2, LOD3 modes
- [Feature Modes Guide](./feature-modes-guide.md) - Feature computation
- [GPU Acceleration](./gpu-acceleration.md) - GPU setup and optimization
- [Configuration Examples](../reference/config-examples.md) - More examples

---

**Last Updated**: October 17, 2025  
**Version**: 5.0.0  
**Status**: ✅ Current
