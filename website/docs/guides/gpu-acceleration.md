---
sidebar_position: 6
title: GPU Acceleration
description: Leverage GPU computing for faster LiDAR processing
keywords:
  [gpu, cuda, acceleration, performance, optimization, cuml, rapids, cupy]
---

# GPU Acceleration

GPU acceleration significantly speeds up LiDAR processing workflows, providing **6-20x speedup** for large-scale datasets and complex feature extraction tasks.

## Overview

The IGN LiDAR HD processor supports GPU acceleration with three performance modes:

1. **CPU-Only**: Standard processing (no GPU required)
2. **Hybrid Mode (CuPy)**: GPU arrays + CPU algorithms (6-8x speedup)
3. **Full GPU Mode (RAPIDS cuML)**: Complete GPU pipeline (12-20x speedup)

The hybrid mode uses an intelligent **per-chunk KDTree strategy** that avoids global tree construction bottlenecks, delivering excellent performance even without RAPIDS cuML.

### Supported Operations

- **Geometric Feature Extraction**: Surface normals, curvature, planarity, verticality
- **KNN Search**: GPU-accelerated k-nearest neighbors (with RAPIDS cuML)
- **PCA Computation**: GPU-based principal component analysis (with RAPIDS cuML)
- **Point Cloud Filtering**: Parallel preprocessing and noise reduction
- **RGB/NIR Augmentation**: GPU-optimized orthophoto integration

## 🚀 Performance Benchmarks

### Real-World Results (17M points, NVIDIA RTX 4080 16GB)

**v1.7.5 Performance (Optimized)**:

| Mode                    | Processing Time   | Speedup | Requirements             |
| ----------------------- | ----------------- | ------- | ------------------------ |
| CPU-Only                | 60 min → 12 min   | 5x      | None (optimized!)        |
| Hybrid (CuPy + sklearn) | 7-10 min → 2 min  | 25-30x  | CuPy + CUDA 12.0+        |
| Full GPU (RAPIDS cuML)  | 3-5 min → 1-2 min | 30-60x  | RAPIDS cuML + CUDA 12.0+ |

:::tip v1.7.5 Optimization
The v1.7.5 release includes major performance optimizations that benefit **all modes** (CPU, Hybrid, Full GPU). Per-chunk KDTree strategy and smaller chunk sizes provide 5-10x speedup automatically!
:::

### Operation Breakdown

| Operation          | CPU Time | Hybrid GPU | Full GPU | Best Speedup |
| ------------------ | -------- | ---------- | -------- | ------------ |
| Feature Extraction | 45 min   | 8 min      | 3 min    | 15x          |
| KNN Search         | 30 min   | 15 min     | 2 min    | 15x          |
| PCA Computation    | 10 min   | 8 min      | 1 min    | 10x          |
| Batch Processing   | 120 min  | 20 min     | 8 min    | 15x          |

## 🔧 Setup Requirements

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA Compute Capability 6.0+ (Pascal or newer)
- **Memory**: Minimum 4GB VRAM (8GB+ recommended, 16GB for large tiles)
- **Driver**: CUDA 12.0+ compatible NVIDIA driver
- **System**: 32GB+ RAM recommended for processing large tiles

### Recommended Hardware

- **Budget**: NVIDIA RTX 3060 12GB
- **Optimal**: NVIDIA RTX 4070/4080 16GB
- **Professional**: NVIDIA A6000 48GB

## 📦 Installation Options

### Option 1: Hybrid Mode (CuPy Only) - Quick Start

**Best for**: Quick setup, testing, or when RAPIDS cuML isn't available

```bash
# Install CuPy for your CUDA version
pip install cupy-cuda12x  # For CUDA 12.x
# OR
pip install cupy-cuda11x  # For CUDA 11.x

# Verify GPU availability
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount(), 'GPU(s) found')"
```

**Performance**: 6-8x speedup (uses GPU arrays with CPU sklearn algorithms via per-chunk optimization)

### Option 2: Full GPU Mode (RAPIDS cuML) - Maximum Performance

**Best for**: Production workloads, large-scale processing, maximum speed

````bash
# Quick install (recommended - uses provided script)
./install_cuml.sh

# Or manual installation:
# Create conda environment (required for RAPIDS)
conda create -n ign_gpu python=3.12 -y
conda activate ign_gpu

# Install RAPIDS cuML (includes CuPy)
conda install -c rapidsai -c conda-forge -c nvidia \
    cuml=24.10 cupy cuda-version=12.5 -y

# Install IGN LiDAR HD
pip install ign-lidar-hd

# Verify installation
**Verification script:**

```bash
python scripts/verify_gpu_setup.py
````

````

**Performance**: 15-20x speedup (complete GPU pipeline)

### Option 3: Automated Installation Script

For WSL2/Linux systems, use our automated installation script:

```bash
# Download and run the installation script
wget https://raw.githubusercontent.com/sducournau/IGN_LIDAR_HD_DATASET/main/install_cuml.sh
chmod +x install_cuml.sh
./install_cuml.sh
````

The script will:

- Install Miniconda (if needed)
- Create `ign_gpu` conda environment
- Install RAPIDS cuML + all dependencies
- Configure CUDA paths

### Verifying Installation

```bash
# Check GPU detection
ign-lidar-hd --version

# Test GPU processing (use a small tile)
ign-lidar-hd enrich --input test.laz --output test_enriched.laz --use-gpu
```

## 📖 Usage Guide

### Command Line Interface

The easiest way to use GPU acceleration is via the CLI:

```bash
# Basic GPU processing
ign-lidar-hd enrich --input-dir data/ --output enriched/ --use-gpu

# Full-featured GPU processing with all options
ign-lidar-hd enrich \
  --input-dir data/ \
  --output enriched/ \
  --use-gpu \
  --auto-params \
  --preprocess \
  --add-rgb \
  --add-infrared \
  --rgb-cache-dir cache/rgb \
  --infrared-cache-dir cache/infrared

# Process specific tiles
ign-lidar-hd enrich \
  --input tile1.laz tile2.laz \
  --output enriched/ \
  --use-gpu \
  --force  # Reprocess even if outputs exist
```

### Python API

```python
from ign_lidar import LiDARProcessor

# Initialize with GPU support
processor = LiDARProcessor(
    lod_level="LOD2",
    use_gpu=True,
    num_workers=4
)

# Process a single tile
patches = processor.process_tile(
    "data/tile.laz",
    "output/",
    enable_rgb=True
)

# Process directory with GPU
patches = processor.process_directory(
    "data/",
    "output/",
    num_workers=4
)
```

### Pipeline Configuration (YAML)

```yaml
global:
  num_workers: 4

enrich:
  input_dir: "data/raw"
  output: "data/enriched"
  use_gpu: true
  auto_params: true
  preprocess: true
  add_rgb: true
  add_infrared: true
  rgb_cache_dir: "cache/rgb"
  infrared_cache_dir: "cache/infrared"

patch:
  input_dir: "data/enriched"
  output: "data/patches"
  lod_level: "LOD2"
```

Then run: `ign-lidar-hd pipeline config.yaml`

## 🐛 Troubleshooting

### Common Issues

#### GPU Not Detected

**Symptoms**: Message "GPU not available, falling back to CPU"

**Solutions**:

```bash
# 1. Check if GPU is visible
nvidia-smi

# 2. Verify CUDA installation
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"

# 3. Check CUDA version compatibility
python -c "import cupy; print('CuPy CUDA version:', cupy.cuda.runtime.runtimeGetVersion())"

# 4. Verify LD_LIBRARY_PATH (Linux/WSL2)
echo $LD_LIBRARY_PATH  # Should include /usr/local/cuda-XX.X/lib64
```

#### CuPy Installation Issues

**Problem**: CuPy not finding CUDA libraries

**WSL2 Solution**:

```bash
# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-13-0

# Add to ~/.zshrc or ~/.bashrc
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

# Reload and test
source ~/.zshrc
python -c "import cupy; print('CuPy working!')"
```

#### RAPIDS cuML Installation Issues

**Problem**: Conda TOS errors during installation

**Solution**:

```bash
# Accept conda Terms of Service
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Then retry installation
conda install -c rapidsai -c conda-forge -c nvidia cuml=24.10 -y
```

#### CUDA Out of Memory

**Symptoms**: RuntimeError: CUDA out of memory

**Solutions**:

1. **Process smaller tiles**: Split large files into smaller chunks
2. **Reduce chunk size**: The processor automatically chunks large point clouds
3. **Close other GPU applications**: Free up VRAM
4. **Use a GPU with more memory**: 16GB+ recommended for large tiles

```bash
# Monitor GPU memory usage
watch -n 1 nvidia-smi
```

#### Slow Performance Despite GPU

**Possible causes**:

1. **Using Hybrid Mode instead of Full GPU**: Install RAPIDS cuML for maximum speed
2. **Thermal throttling**: Check GPU temperature with `nvidia-smi`
3. **PCIe bandwidth**: Ensure GPU is in x16 slot
4. **CPU bottleneck**: Use `--num-workers` to parallelize I/O

**Check GPU utilization**:

```bash
# Monitor GPU usage during processing
nvidia-smi dmon -s u
```

#### Per-Chunk vs Global KDTree

The system automatically selects the best strategy:

- **With RAPIDS cuML**: Uses global KDTree on GPU (fastest, 15-20x speedup)
- **Without cuML**: Uses per-chunk KDTree with CPU sklearn (still fast, 5-10x speedup)

You'll see different log messages:

```text
# With cuML (fastest)
✓ RAPIDS cuML available - GPU algorithms enabled
Computing normals with GPU-accelerated KDTree (global)

# Without cuML (still fast)
⚠ RAPIDS cuML not available - using per-chunk CPU KDTree
Computing normals with per-chunk KDTree (5% overlap)
```

### Automatic CPU Fallback

The system automatically falls back to CPU processing if GPU is unavailable:

- CuPy import fails → CPU mode
- CUDA runtime error → CPU mode
- Insufficient GPU memory → CPU mode (with warning)

**Disabling GPU** (force CPU):

```bash
ign-lidar-hd enrich --input-dir data/ --output enriched/  # No --use-gpu flag
```

## 📋 Detailed Benchmarks

### Test Environment

- **GPU**: NVIDIA RTX 4080 (16GB VRAM)
- **CPU**: AMD Ryzen 9 / Intel i7 equivalent
- **System**: WSL2 Ubuntu 24.04, 32GB RAM
- **CUDA**: 13.0
- **Test Tile**: 17M points (typical IGN LiDAR HD tile)

### Processing Time Comparison

| Configuration           | Processing Time | Speedup | Notes                         |
| ----------------------- | --------------- | ------- | ----------------------------- |
| CPU-Only (sklearn)      | 60 min          | 1x      | Baseline                      |
| Hybrid (CuPy + sklearn) | 7-10 min        | 6-8x    | Per-chunk KDTree optimization |
| Full GPU (RAPIDS cuML)  | 3-5 min         | 12-20x  | Global GPU KDTree             |

### Feature Extraction Breakdown

| Operation             | CPU    | Hybrid GPU | Full GPU | Best Speedup |
| --------------------- | ------ | ---------- | -------- | ------------ |
| Normal Computation    | 25 min | 4 min      | 1.5 min  | 16x          |
| KNN Search            | 20 min | 12 min     | 1 min    | 20x          |
| PCA (eigenvalues)     | 8 min  | 6 min      | 0.5 min  | 16x          |
| Curvature Calculation | 5 min  | 2 min      | 0.5 min  | 10x          |
| Other Features        | 2 min  | 1 min      | 0.5 min  | 4x           |

### Memory Usage

| Mode                    | GPU Memory | System RAM | Total |
| ----------------------- | ---------- | ---------- | ----- |
| CPU-Only                | 0 GB       | 24 GB      | 24 GB |
| Hybrid (CuPy + sklearn) | 6 GB       | 16 GB      | 22 GB |
| Full GPU (RAPIDS cuML)  | 8 GB       | 12 GB      | 20 GB |

### Batch Processing (100 tiles)

- **CPU-Only**: ~100 hours
- **Hybrid Mode**: ~14 hours (7x speedup)
- **Full GPU Mode**: ~6 hours (16x speedup)

### Accuracy Validation

All three modes produce **identical results** (verified with feature correlation > 0.9999).

## 🔗 Related Documentation

- [Quick Start Guide](./quick-start)
- [Performance Optimization](./performance)
- [Troubleshooting](./troubleshooting)
- [Pipeline Configuration](../api/pipeline-config)
- [Installation Guide](../installation/quick-start)

## 💡 Best Practices

### 1. Choose the Right Mode

- **Development/Testing**: Hybrid mode (easy setup, good performance)
- **Production**: Full GPU mode with RAPIDS cuML (maximum performance)
- **No GPU**: CPU mode works fine for small batches

### 2. Optimize Your Workflow

```yaml
# Recommended pipeline configuration for GPU
global:
  num_workers: 4 # Parallelize I/O while GPU processes

enrich:
  use_gpu: true
  auto_params: true # Let the system optimize parameters
  preprocess: true # Clean data before feature extraction
```

### 3. Monitor Resources

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Monitor with detailed metrics
nvidia-smi dmon -s pucvmet -d 1
```

### 4. Batch Processing Tips

- **Use --force cautiously**: Only reprocess when needed
- **Enable smart caching**: Use `--rgb-cache-dir` and `--infrared-cache-dir`
- **Parallelize I/O**: Use `--num-workers` for concurrent file operations
- **Process strategically**: Start with urban tiles (higher point density) to test settings

### 5. Hardware Recommendations

| Use Case                  | Minimum GPU   | Recommended GPU | Optimal GPU      |
| ------------------------- | ------------- | --------------- | ---------------- |
| Learning/Small datasets   | GTX 1660 6GB  | RTX 3060 12GB   | RTX 4060 Ti 16GB |
| Production/Medium batches | RTX 3060 12GB | RTX 4070 12GB   | RTX 4080 16GB    |
| Large-scale processing    | RTX 3080 10GB | RTX 4080 16GB   | A6000 48GB       |

## 🎓 Advanced Topics

### Per-Chunk Optimization Strategy

When RAPIDS cuML is not available, the system uses an intelligent per-chunk strategy:

1. **Splits point cloud** into ~5M point chunks
2. **Builds local KDTree** per chunk (fast with sklearn)
3. **Uses 5% overlap** between chunks to handle edge cases
4. **Merges results** seamlessly

This provides 80-90% of GPU performance without requiring RAPIDS cuML installation.

### GPU Memory Management

The system automatically manages GPU memory:

- **Automatic chunking**: Large point clouds split into GPU-sized chunks
- **Memory pooling**: CuPy reuses allocated memory
- **Garbage collection**: Frees memory between tiles
- **Fallback handling**: Gracefully handles OOM errors

### Multi-GPU Support

Currently, the library uses a single GPU (device 0). For multi-GPU processing:

```bash
# Process different directories on different GPUs
CUDA_VISIBLE_DEVICES=0 ign-lidar-hd enrich --input dir1/ --output out1/ --use-gpu &
CUDA_VISIBLE_DEVICES=1 ign-lidar-hd enrich --input dir2/ --output out2/ --use-gpu &
```

---

_For more advanced GPU optimization techniques, see the [Performance Guide](./performance)._
