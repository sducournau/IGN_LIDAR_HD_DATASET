# IGN LiDAR HD Conda Package Information

## Package Summary

**Name**: ign-lidar-hd  
**Version**: 2.0.0-alpha  
**License**: MIT  
**Platform**: noarch (pure Python)

## Installation Methods

### Method 1: From Environment File (Recommended)

```bash
# Create environment from file
conda env create -f conda-recipe/environment.yml

# Activate environment
conda activate ign-lidar-hd

# Install the package in development mode
pip install -e .
```

### Method 2: Manual Installation

```bash
# Create environment with dependencies
conda create -n ign-lidar-hd python=3.10 numpy scikit-learn tqdm click pyyaml psutil requests pillow hydra-core omegaconf -c conda-forge

# Activate environment
conda activate ign-lidar-hd

# Install PyPI-only dependencies
pip install laspy>=2.3.0 lazrs>=0.5.0

# Install the package
pip install -e .
```

### Method 3: Build and Install Conda Package

```bash
# Build conda package
conda-build conda-recipe/

# Install from local build
conda install -c file://$(pwd)/dist/conda ign-lidar-hd
```

## CLI Commands

After installation, these commands are available:

- **`ign-lidar-hd`** - Main processing CLI with Hydra configuration support
- **`ign-lidar-qgis`** - QGIS converter for LAZ files

## Dependencies

### Runtime Dependencies (conda-forge)

- python ≥3.8
- numpy ≥1.21.0
- scikit-learn ≥1.0.0
- tqdm ≥4.60.0
- click ≥8.0.0
- pyyaml ≥6.0
- psutil ≥5.8.0
- requests ≥2.25.0
- pillow ≥9.0.0
- hydra-core ≥1.3.0
- omegaconf ≥2.3.0

### PyPI Dependencies

- laspy ≥2.3.0 (LAZ/LAS file support)
- lazrs ≥0.5.0 (LAZ compression backend)

### Optional Dependencies

- torch (for neural network datasets)
- cupy (for GPU acceleration)
- h5py (for HDF5 output format)

## Package Structure

```
ign-lidar-hd/
├── ign_lidar/              # Main package
│   ├── cli/                # Command-line interface
│   ├── core/               # Core processing
│   ├── preprocessing/      # Data preprocessing
│   ├── features/          # Feature extraction
│   ├── io/                # Input/output utilities
│   └── datasets/          # Dataset classes
├── configs/               # Hydra configuration files
├── data/                  # Sample data
├── conda-recipe/          # Conda packaging files
│   ├── meta.yaml         # Conda recipe
│   ├── environment.yml   # Environment specification
│   ├── build.sh          # Build script
│   └── README.md         # Installation guide
└── tests/                # Test suite
```

## Testing

Run the installation test:

```bash
python test_installation.py
```

Expected output:

```
============================================================
IGN LiDAR HD Package Installation Test
============================================================
✓ Main package imported successfully
✓ LiDARProcessor imported successfully
✓ Preprocessing functions imported successfully
✓ Downloader imported successfully
✓ Package version: 2.0.0-alpha
✓ Found 3 sample LAZ files
✓ CLI main function accessible
✓ QGIS converter CLI accessible
============================================================
✅ ALL TESTS PASSED - Installation is working correctly!
============================================================
```

## Usage Examples

### Basic Processing

```bash
# Modern CLI (recommended)
ign-lidar-hd process \
    input_dir=data/sample_laz \
    output_dir=data/output \
    processor.patch_size=50.0 \
    processor.num_points=1000
```

### QGIS Conversion

```bash
conda activate ign-lidar-hd
ign-lidar-qgis input.laz output.laz
```

### Advanced Configuration

```bash
ign-lidar-hd process \
    experiment=buildings_lod2 \
    input_dir=data/input \
    output_dir=data/output \
    processor.use_gpu=false \
    features.mode=full
```

## Build Information

- **Build System**: setuptools with pyproject.toml
- **Entry Points**: Defined in pyproject.toml
- **Testing**: pytest with custom installation test
- **Configuration**: Hydra-based hierarchical configuration

## Platform Support

- **Operating Systems**: Linux, macOS, Windows
- **Python Versions**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Architectures**: x86_64, aarch64 (through noarch build)
