---
sidebar_position: 1
---

# Quick Start Installation

## Requirements

- Python 3.8 or higher
- pip package manager

## Install from PyPI

```bash
pip install ign-lidar-hd
```

## Verify Installation

```bash
ign-lidar-hd --version
```

## Alternative Installation Methods

### From Source

```bash
git clone https://github.com/sducournau/IGN_LIDAR_HD_DATASET.git
cd IGN_LIDAR_HD_DATASET
pip install -e .
```

### With Development Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

## Optional: GPU Support

For GPU-accelerated feature computation, install CuPy separately after installing the base package:

```bash
# Install base package
pip install ign-lidar-hd

# Then install CuPy for your CUDA version
pip install cupy-cuda11x  # For CUDA 11.x
# OR
pip install cupy-cuda12x  # For CUDA 12.x
```

Or using the GPU requirements file:

```bash
pip install -r requirements_gpu.txt
```

**GPU Requirements:**

- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.0 or higher
- CuPy package matching your CUDA version (`cupy-cuda11x` or `cupy-cuda12x`)

## Environment Setup

### Using conda (recommended)

```bash
conda create -n ign-lidar python=3.9
conda activate ign-lidar
pip install ign-lidar-hd
```

### Using venv

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
pip install ign-lidar-hd
```

## Test Installation

Test that everything works:

```bash
# Check CLI access
ign-lidar-hd --help

# Or use the installed command
ign-lidar-hd --help
```

You should see the available commands:

- `download` - Download IGN LiDAR tiles
- `enrich` - Add building features to LAZ files
- `process` - Extract patches from enriched tiles

## Next Steps

- Try the [Basic Usage Guide](../guides/basic-usage.md)
- Explore [CLI Commands](../guides/cli-commands.md)
- Learn about [Smart Skip Features](../features/smart-skip.md)

## Troubleshooting

### Command Not Found

If `ign-lidar-hd` command is not found:

```bash
# Use Python module syntax instead
ign-lidar-hd --help
```

### Import Errors

If you get import errors:

```bash
# Reinstall in development mode
pip install -e .

# Or check your Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### Missing Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
pip list  # Verify installation
```
