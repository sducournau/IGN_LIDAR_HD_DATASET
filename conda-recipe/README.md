# Conda Installation Guide

This guide provides instructions for installing and packaging IGN LiDAR HD using conda.

## 🚀 Quick Installation

### Option 1: From Environment File (Recommended)

```bash
# Create environment from file
conda env create -f conda-recipe/environment.yml

# Activate environment
conda activate ign-lidar-hd

# Install the package in development mode
pip install -e .
```

### Option 2: Manual Environment Creation

```bash
# Create new environment
conda create -n ign-lidar-hd python=3.10 -y

# Activate environment
conda activate ign-lidar-hd

# Install dependencies from conda-forge
conda install -c conda-forge numpy scikit-learn tqdm click pyyaml psutil requests pillow hydra-core omegaconf h5py

# Install PyPI-only dependencies
pip install laspy>=2.3.0 lazrs>=0.5.0

# Install the package
pip install -e .
```

### Option 3: With GPU Acceleration (Recommended for Performance)

For GPU acceleration with CuPy and RAPIDS cuML, use the automated installation script:

```bash
# Automated full GPU setup
./install_cuml.sh
```

Or manually install GPU dependencies:

```bash
# Create environment with GPU support
conda create -n ign-lidar-gpu python=3.12 -y
conda activate ign-lidar-gpu

# Install base dependencies
conda install -c conda-forge numpy scikit-learn tqdm click pyyaml psutil requests pillow hydra-core omegaconf h5py

# Install GPU dependencies
conda install -c conda-forge cupy  # Basic GPU acceleration (6-8x speedup)

# Install RAPIDS cuML for full GPU acceleration (12-20x speedup)
conda install -c rapidsai -c conda-forge -c nvidia cuml=24.10 python=3.12 cuda-version=12.5

# Install PyPI-only dependencies
pip install laspy>=2.3.0 lazrs>=0.5.0

# Install the package
pip install -e .
```

For detailed GPU setup instructions, see [GPU_SETUP.md](../GPU_SETUP.md).

## Building Conda Package

### Prerequisites

Install conda-build if not already installed:

```bash
conda install conda-build
```

### Build Package

```bash
# Run the build script
./conda-recipe/build.sh

# Or build manually
conda-build conda-recipe/ --output-folder dist/conda
```

### Install Built Package

```bash
# Install from local build
conda install -c file://$(pwd)/dist/conda ign-lidar-hd

# Or create new environment with the package
conda create -n test-env -c file://$(pwd)/dist/conda ign-lidar-hd
```

## 🧪 Testing Installation

After installation, test the package:

```bash
# Test imports
python -c "import ign_lidar; print('✓ Package imported successfully')"

# Test CLI commands
ign-lidar-hd --help

# Verify version
python -c "import ign_lidar; print(f'Version: {ign_lidar.__version__}')"
```

## 📦 Package Structure

The conda package includes:

- **Main CLI**: `ign-lidar-hd` - Complete processing pipeline with Hydra configuration
- **Python API**: Full programmatic access to all functionality
- **Example Configs**: YAML templates in `examples/` directory

## 📚 Documentation

For complete documentation, visit:

- [Installation Guide](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/installation/quick-start)
- [GPU Setup Guide](../GPU_SETUP.md)
- [Configuration Reference](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/api/configuration)

## 🔗 See Also

- [Main README](../README.md)
- [Example Configurations](../examples/README.md)
- [Full Documentation](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/)

---

_Conda installation guide for IGN LiDAR HD v2.4.2_
