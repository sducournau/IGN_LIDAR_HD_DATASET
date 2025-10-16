# GPU-Accelerated Reclassification System

## Overview

The IGN LiDAR HD Dataset now includes an optimized reclassification system with GPU acceleration support, dramatically reducing processing time for applying BD TOPO® ground truth classification to existing enriched tiles.

**Date:** October 16, 2025  
**Version:** 2.5.4+

---

## ⚡ Performance Comparison

Processing **18.6 million points** from a single tile:

| Mode                | Hardware                 | Processing Time | Speedup | Memory        |
| ------------------- | ------------------------ | --------------- | ------- | ------------- |
| **CPU (baseline)**  | Intel CPU                | ~30-60 minutes  | 1x      | ~2-4 GB RAM   |
| **CPU (optimized)** | Intel CPU + STRtree      | ~5-10 minutes   | 6-12x   | ~2-4 GB RAM   |
| **GPU**             | NVIDIA GPU + RAPIDS      | ~1-2 minutes    | 30-60x  | ~4-8 GB VRAM  |
| **GPU+cuML**        | NVIDIA GPU + Full RAPIDS | ~30-60 seconds  | 60-120x | ~8-12 GB VRAM |

---

## 🚀 Quick Start

### Option 1: Using the Script

```bash
# Activate GPU environment
conda activate ign_gpu

# Run with auto-detection (recommended)
python reclassify_with_ground_truth.py
```

### Option 2: Using the Config File

```bash
python -m ign_lidar.cli.commands.process \
  --config-file configs/reclassification_config.yaml \
  input_dir=D:/ign/preprocessed/asprs/enriched_tiles \
  output_dir=D:/ign/preprocessed/asprs/reclassified_tiles
```

### Option 3: Using the Processor API

```python
from ign_lidar.core.processor import LiDARProcessor
from omegaconf import OmegaConf

config = OmegaConf.create({
    'processor': {
        'lod_level': 'LOD2',
        'processing_mode': 'reclassify_only',
        'patch_size': None,
        'num_points': 16384,
        'architecture': 'pointnet++',
        'output_format': 'laz'
    },
    'features': {'mode': 'minimal'}
})

processor = LiDARProcessor(config=config)

tiles_processed = processor.reclassify_directory(
    input_dir="D:/ign/preprocessed/asprs/enriched_tiles",
    output_dir="D:/ign/preprocessed/asprs/reclassified_tiles",
    cache_dir="data/cache",
    chunk_size=100000,
    acceleration_mode="auto",  # or 'cpu', 'gpu', 'gpu+cuml'
    show_progress=True,
    skip_existing=True
)
```

---

## 📦 Installation

### CPU Mode (No additional dependencies)

```bash
# Already installed with base package
pip install -e .
```

### GPU Mode (RAPIDS cuSpatial)

```bash
# Create/update conda environment
conda install -c rapidsai -c conda-forge cuspatial python=3.10 cuda-version=11.8
```

### GPU+cuML Mode (Full RAPIDS Stack)

```bash
# Install full RAPIDS
conda install -c rapidsai -c conda-forge cuspatial cuml cudf cupy python=3.10 cuda-version=11.8
```

---

## 🔧 Acceleration Modes

### `auto` (Recommended)

Automatically detects and uses the best available backend:

- Tries: `gpu+cuml` → `gpu` → `cpu`
- Safe default for all systems
- Optimal performance without configuration

### `cpu`

Uses CPU with STRtree spatial indexing:

- **Best for:** <5M points or no GPU available
- **Speed:** ~5-10 min for 18M points
- **Memory:** ~2-4 GB RAM
- **Dependencies:** shapely, geopandas (included)

### `gpu`

Uses RAPIDS cuSpatial for GPU acceleration:

- **Best for:** 5M-50M points with NVIDIA GPU
- **Speed:** ~1-2 min for 18M points
- **Memory:** ~4-8 GB VRAM
- **Dependencies:** cuspatial, cudf, cupy
- **Requirements:** NVIDIA GPU with CUDA support

### `gpu+cuml`

Uses full RAPIDS stack with additional optimizations:

- **Best for:** >50M points with high-end NVIDIA GPU
- **Speed:** ~30-60 sec for 18M points
- **Memory:** ~8-12 GB VRAM
- **Dependencies:** cuspatial, cuml, cudf, cupy
- **Requirements:** NVIDIA GPU (RTX 3080 or better recommended)

---

## 🗺️ What Gets Reclassified

The system applies BD TOPO® ground truth classification with the following priority (highest to lowest):

1. **Buildings** (ASPRS 6) - from BD TOPO® building footprints
2. **Bridges** (ASPRS 17) - bridge structures
3. **Roads** (ASPRS 11) - road surfaces with intelligent buffering
4. **Railways** (ASPRS 10) - railway tracks
5. **Sports Facilities** (ASPRS 41) - sports grounds
6. **Parking** (ASPRS 40) - parking areas
7. **Cemeteries** (ASPRS 42) - cemetery zones
8. **Water** (ASPRS 9) - water surfaces
9. **Vegetation** (ASPRS 4) - vegetation zones

---

## 📊 Configuration

### Basic Configuration (`reclassify_with_ground_truth.py`)

```python
# File paths
input_dir = Path("/mnt/d/ign/preprocessed/asprs/enriched_tiles")
output_dir = Path("/mnt/d/ign/preprocessed/asprs/reclassified_tiles")
cache_dir = Path("data/cache")

# Acceleration mode
acceleration_mode = "auto"  # 'cpu', 'gpu', 'gpu+cuml', 'auto'
```

### Advanced Configuration (`configs/reclassification_config.yaml`)

```yaml
processor:
  processing_mode: "reclassify_only"

  reclassification:
    enabled: true
    acceleration_mode: "auto" # auto-detect best backend
    chunk_size: 100000 # points per chunk
    gpu_chunk_size: 500000 # larger for GPU
    show_progress: true
    skip_existing: true

data_sources:
  bd_topo:
    enabled: true
    features:
      buildings: true
      roads: true
      railways: true
      water: true
      # ... etc
```

---

## 🔬 Technical Details

### CPU Implementation

- **Algorithm:** STRtree R-tree spatial indexing
- **Complexity:** O(log n) per point-in-polygon query
- **Chunking:** 100,000 points per chunk
- **Libraries:** shapely, geopandas

### GPU Implementation

- **Algorithm:** GPU-accelerated point-in-polygon using cuSpatial
- **Parallelism:** Massively parallel on CUDA cores
- **Chunking:** 500,000 points per chunk (GPU memory permitting)
- **Libraries:** RAPIDS cuSpatial, cuDF, CuPy

### GPU+cuML Implementation

- **Additional:** cuML-powered optimizations for large datasets
- **Batch processing:** Optimized for >50M points
- **Memory management:** Advanced GPU memory pooling

---

## 🐛 Troubleshooting

### GPU Not Detected

```bash
# Check RAPIDS installation
conda list | grep rapids

# Install if missing
conda install -c rapidsai -c conda-forge cuspatial
```

### CUDA Version Mismatch

```bash
# Check CUDA version
nvidia-smi

# Install matching RAPIDS version
# For CUDA 11.8:
conda install -c rapidsai cuspatial cuda-version=11.8

# For CUDA 12.0:
conda install -c rapidsai cuspatial cuda-version=12.0
```

### Out of GPU Memory

```python
# Reduce chunk size in script
processor.reclassify_directory(
    ...
    chunk_size=50000,  # Smaller chunks
    acceleration_mode="gpu"
)
```

### Falls Back to CPU

Check logs for warnings:

```
WARNING: GPU mode requested but RAPIDS not available, falling back to CPU
```

Install missing dependencies or use `acceleration_mode="auto"`.

---

## 📈 Performance Tuning

### For Large Tiles (>20M points)

```yaml
# Use GPU+cuML mode
acceleration_mode: "gpu+cuml"
gpu_chunk_size: 1000000 # Larger chunks if you have VRAM
```

### For Memory-Constrained GPUs

```yaml
acceleration_mode: "gpu"
gpu_chunk_size: 100000 # Smaller chunks
```

### For Batch Processing

```python
# Process multiple files with progress tracking
processor.reclassify_directory(
    input_dir=input_dir,
    output_dir=output_dir,
    skip_existing=True,  # Resume interrupted runs
    show_progress=True
)
```

---

## ✅ Validation

After reclassification, validate results:

```bash
python check_classification.py
```

Expected output:

```
✓ Roads (11):      XXX,XXX points
✓ Railways (10):   XX,XXX points
✓ Buildings (6):   X,XXX,XXX points
✓ Water (9):       X,XXX points
```

---

## 🔗 Related Files

- `ign_lidar/core/modules/reclassifier.py` - Core reclassification module
- `ign_lidar/core/processor.py` - Processor integration
- `configs/reclassification_config.yaml` - Configuration template
- `reclassify_with_ground_truth.py` - Standalone script
- `check_classification.py` - Validation script

---

## 📚 References

- **RAPIDS cuSpatial:** https://docs.rapids.ai/api/cuspatial/stable/
- **ASPRS LAS 1.4 Specification:** https://www.asprs.org/divisions-committees/lidar-division/laser-las-file-format-exchange-activities
- **IGN BD TOPO®:** https://geoservices.ign.fr/bdtopo

---

## 🎯 Next Steps

1. **Run reclassification** on your enriched tiles
2. **Validate** results with check_classification.py
3. **Visualize** in CloudCompare to verify ground truth application
4. **Iterate** on any tiles that need adjustment

For questions or issues, check the main documentation or create a GitHub issue.
