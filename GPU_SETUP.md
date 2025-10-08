# GPU Acceleration Setup Guide

This guide helps you set up GPU acceleration for IGN LiDAR HD processing. GPU acceleration can provide **6-20x speedup** over CPU-only processing.

## ⚡ Performance Comparison

| Configuration                     | Processing Time (17M points) | Speedup       |
| --------------------------------- | ---------------------------- | ------------- |
| **CPU Only**                      | ~60 minutes                  | 1x (baseline) |
| **CuPy (Basic GPU)**              | ~7-10 minutes                | 6-8x          |
| **CuPy + RAPIDS cuML (Full GPU)** | ~3-5 minutes                 | 12-20x        |

## 🔍 Understanding the Warnings

If you see these warnings, it means GPU libraries are not installed (CPU fallback is active):

```
⚠ CuPy non disponible - fallback CPU
⚠ RAPIDS cuML non disponible - fallback sklearn
```

The library automatically falls back to CPU processing using NumPy and scikit-learn. To enable GPU acceleration, install the optional GPU dependencies below.

## 📋 Prerequisites

Before installing GPU dependencies, ensure you have:

1. **NVIDIA GPU** with Compute Capability 6.0+ (recommended 4GB+ VRAM)
2. **CUDA Toolkit** installed (11.x or 12.x)
3. **NVIDIA Driver** (compatible with your CUDA version)

### Check Your CUDA Version

```bash
nvidia-smi
```

Look for "CUDA Version" in the output (e.g., `CUDA Version: 12.2`).

## 🚀 Installation Options

### Option 1: Basic GPU Acceleration (CuPy Only)

**Performance:** 6-8x speedup  
**Complexity:** Easy  
**Recommended for:** Quick setup, limited VRAM

#### Using pip (Choose based on your CUDA version):

```bash
# For CUDA 11.x
pip install cupy-cuda11x>=12.0.0

# For CUDA 12.x
pip install cupy-cuda12x>=12.0.0
```

#### Using conda:

```bash
conda install -c conda-forge cupy
```

#### Verify Installation:

```bash
python -c "import cupy as cp; print(f'✅ CuPy OK - GPU: {cp.cuda.Device(0).compute_capability}')"
```

### Option 2: Full GPU Acceleration (CuPy + RAPIDS cuML)

**Performance:** 12-20x speedup  
**Complexity:** Moderate  
**Recommended for:** Maximum performance, sufficient VRAM (8GB+)

#### Automated Installation (Recommended):

We provide a helper script that sets up everything:

```bash
./install_cuml.sh
```

This script will:

- Create a conda environment `ign_gpu` with Python 3.12
- Install RAPIDS cuML 24.10 + CuPy
- Install IGN LiDAR HD and all dependencies
- Verify GPU setup

#### Manual Installation with conda:

```bash
# Create environment
conda create -n ign_gpu python=3.12 -y
conda activate ign_gpu

# Install RAPIDS cuML (for CUDA 12.x)
conda install -c rapidsai -c conda-forge -c nvidia \
    cuml=24.10 python=3.12 cuda-version=12.5

# Install CuPy
conda install -c conda-forge cupy

# Install IGN LiDAR HD
pip install ign-lidar-hd
```

#### Manual Installation with pip:

```bash
# Install CuPy first
pip install cupy-cuda12x>=12.0.0  # or cupy-cuda11x for CUDA 11.x

# Install cuML (may require additional configuration)
pip install cuml-cu12>=23.10.0  # or cuml-cu11 for CUDA 11.x
```

⚠️ **Note:** RAPIDS cuML installation via pip can be complex. Conda is strongly recommended for cuML.

#### Verify Installation:

```bash
# Test CuPy
python -c "import cupy as cp; print(f'✅ CuPy OK - GPU: {cp.cuda.Device(0).compute_capability}')"

# Test RAPIDS cuML
python -c "from cuml.neighbors import NearestNeighbors; print('✅ RAPIDS cuML OK')"
```

## 📦 Installing via requirements files

### Basic Installation (CPU only):

```bash
pip install -r requirements.txt
```

### GPU Installation:

```bash
pip install -r requirements_gpu.txt
```

**Note:** The `requirements_gpu.txt` file includes commented instructions. You'll need to uncomment the appropriate lines for your CUDA version.

## 🎯 Using with conda environment.yml

The `conda-recipe/environment.yml` file includes commented GPU dependencies:

```bash
# CPU only (default)
conda env create -f conda-recipe/environment.yml

# Edit environment.yml to uncomment GPU dependencies, then:
conda env create -f conda-recipe/environment.yml
```

## 🔧 Integration with pyproject.toml

If you're installing from source or using pip with extras:

```bash
# Basic installation (CPU only)
pip install .

# With optional features
pip install .[all]  # All features except GPU

# GPU dependencies must be installed separately:
pip install cupy-cuda12x  # Choose your CUDA version
pip install .[gpu-full]   # Attempts to install cuML (may fail without conda)
```

## 🐛 Troubleshooting

### CuPy Installation Issues

**Problem:** CuPy fails to install or import

**Solutions:**

1. Verify CUDA is installed: `nvidia-smi`
2. Ensure you're using the correct CuPy version for your CUDA:
   - CUDA 11.x → `cupy-cuda11x`
   - CUDA 12.x → `cupy-cuda12x`
3. Try conda installation: `conda install -c conda-forge cupy`

### RAPIDS cuML Installation Issues

**Problem:** cuML fails to install via pip

**Solutions:**

1. Use conda instead (recommended): See Option 2 above
2. Check CUDA compatibility: RAPIDS requires CUDA 11.4+
3. Ensure sufficient VRAM: Minimum 4GB, recommended 8GB+
4. Use the automated script: `./install_cuml.sh`

### Runtime Errors

**Problem:** "CUDA out of memory" errors

**Solutions:**

1. Reduce batch size in processing config
2. Process smaller tiles
3. Use CuPy-only mode instead of full RAPIDS
4. Free GPU memory: `nvidia-smi` to check usage

**Problem:** GPU libraries installed but still seeing warnings

**Solutions:**

1. Verify installation: Run verification commands above
2. Restart Python kernel/terminal after installation
3. Check environment: `conda list cupy` or `pip list | grep cupy`
4. Ensure correct environment is activated

## 📚 Additional Resources

- [CuPy Documentation](https://docs.cupy.dev/)
- [RAPIDS cuML Documentation](https://docs.rapids.ai/api/cuml/stable/)
- [CUDA Toolkit Download](https://developer.nvidia.com/cuda-downloads)
- [IGN LiDAR HD GPU Performance Report](./PERFORMANCE.md) (if available)

## 💡 Recommendations

- **For most users:** Start with Option 1 (CuPy only) for simplicity
- **For maximum performance:** Use Option 2 (Full GPU) with the automated script
- **For production:** Use conda environments for better reproducibility
- **For development:** Consider keeping separate CPU and GPU environments

## 📝 Configuration Files Summary

| File                           | Purpose                | GPU Dependencies                         |
| ------------------------------ | ---------------------- | ---------------------------------------- |
| `requirements.txt`             | Basic pip installation | Commented, manual uncomment needed       |
| `requirements_gpu.txt`         | GPU-specific pip deps  | Detailed instructions included           |
| `pyproject.toml`               | Package metadata       | Optional extras: `[gpu]`, `[gpu-full]`   |
| `conda-recipe/environment.yml` | Conda environment      | Commented, manual uncomment needed       |
| `conda-recipe/meta.yaml`       | Conda package build    | Uses `run_constrained` for optional deps |
| `install_cuml.sh`              | Automated GPU setup    | Full RAPIDS installation script          |

## ✅ Quick Start

### Just want it to work? Run this:

```bash
# For automatic full GPU setup
./install_cuml.sh

# Then activate and use
conda activate ign_gpu
ign-lidar-hd process --config your_config.yaml
```

No more warnings! 🎉
