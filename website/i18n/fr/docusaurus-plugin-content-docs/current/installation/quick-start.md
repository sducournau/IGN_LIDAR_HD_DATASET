---
sidebar_position: 1
title: "Installation" Guide
description: Complete installation guide for IGN LiDAR HD Processing Library
keywords: [installation, pip, setup, gpu, cuda]
---

<!-- 🇫🇷 TRADUCTION FRANÇAISE REQUISE -->
<!-- Ce fichier est un modèle qui nécessite une traduction manuelle. -->
<!-- Veuillez traduire le contenu ci-dessous en conservant : -->
<!-- - Le frontmatter (métadonnées en haut) -->
<!-- - Les blocs de code (traduire uniquement les commentaires) -->
<!-- - Les liens et chemins de fichiers -->
<!-- - La structure Markdown -->



# Installation Guide

Complete installation guide for IGN LiDAR HD Processing Library. Get up and running in minutes with our step-by-step instructions.

## 📋 Requirements

- **Python 3.8+** (Python 3.9-3.11 recommended)
- **pip** package manager
- **Operating System:** Windows, Linux, or macOS

:::tip Check Python Version

```bash
python --version  # Should show Python 3.8 or higher
```

:::

## 🚀 Standard Installation

### Via PyPI (Recommended)

```bash
pip install ign-lidar-hd
```

### Verify Installation

```bash
# Check version
ign-lidar-hd --version

# Test CLI
ign-lidar-hd --help
```

### Installation Options

```bash
# Standard installation (CPU only)
pip install ign-lidar-hd

# With RGB augmentation support
pip install ign-lidar-hd[rgb]

# With all features (excluding GPU)
pip install ign-lidar-hd[all]
```

## ⚡ GPU Acceleration (Optional)

**Performance Boost:** 5-10x faster feature computation

### Prerequisites

1. **NVIDIA GPU** with CUDA support
2. **CUDA Toolkit 11.0+** installed
3. **GPU Memory:** 4GB+ recommended

Verify GPU setup:

```bash
nvidia-smi  # Should display GPU information
```

### Install GPU Support

```bash
# Install base package first
pip install ign-lidar-hd

# Then add CuPy for your CUDA version
pip install cupy-cuda11x  # For CUDA 11.x
# OR
pip install cupy-cuda12x  # For CUDA 12.x
```

### Advanced GPU (RAPIDS cuML)

For maximum performance:

```bash
# Using conda (recommended for RAPIDS)
conda create -n ign-lidar python=3.10
conda activate ign-lidar
pip install ign-lidar-hd
conda install -c rapidsai -c conda-forge -c nvidia cuml
```

## 🔧 Development Installation

### From Source

```bash
git clone https://github.com/sducournau/IGN_LIDAR_HD_DATASET.git
cd IGN_LIDAR_HD_DATASET
pip install -e .
```

### With Development Dependencies

```bash
pip install -e .[dev,test,docs]
```

## 🐍 Virtual Environments

### Using venv (Built-in)

```bash
python -m venv ign-lidar-env
source ign-lidar-env/bin/activate  # Linux/macOS
# or
ign-lidar-env\Scripts\activate     # Windows
pip install ign-lidar-hd
```

### Using conda

```bash
conda create -n ign-lidar python=3.10
conda activate ign-lidar
pip install ign-lidar-hd
```

## ✅ Verify Installation

### Basic Verification

```python
# Test Python imports
import ign_lidar
print(f"IGN LiDAR HD version: {ign_lidar.__version__}")

# Test main classes
from ign_lidar import LiDARProcessor, IGNLiDARDownloader
print("✓ Installation successful!")
```

### GPU Verification

```python
# Check GPU availability
from ign_lidar.features_gpu import GPU_AVAILABLE, CUML_AVAILABLE

print(f"GPU (CuPy) available: {GPU_AVAILABLE}")
print(f"RAPIDS cuML available: {CUML_AVAILABLE}")

if GPU_AVAILABLE:
    print("✓ GPU acceleration enabled!")
else:
    print("⚠️  GPU not detected - using CPU")
```

## 🔧 Troubleshooting

### Command Not Found

```bash
# If ign-lidar-hd command is not found, try:
python -m ign_lidar.cli --help
```

### Import Errors

```bash
# Reinstall in development mode
pip install -e .

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### GPU Issues

```bash
# Test CUDA availability
python -c "import cupy; print('CUDA works!')"

# Check CUDA version
nvcc --version
```

## 🚀 Next Steps

Now that you're installed:

1. 📖 Follow the [Quick Start Guide](../guides/quick-start)
2. 🖥️ Try [Basic Usage Examples](../guides/basic-usage)
3. ⚡ Configure [GPU acceleration](../gpu/overview) (if available)
4. 📋 Explore [Pipeline Configuration](../features/pipeline-configuration)

## 💡 Need Help?

- 📚 Read the [Complete Documentation](/)
- 🐛 Report issues on [GitHub Issues](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
- 💬 Browse [Examples](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/tree/main/examples)
