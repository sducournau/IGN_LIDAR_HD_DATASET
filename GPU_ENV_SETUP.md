# IGN LiDAR HD - GPU Environment Setup

## Summary

This document summarizes the GPU environment configuration for the IGN LiDAR HD package with full PyTorch, RAPIDS cuML, and CuPy support.

## Environment: `ign_gpu`

The `ign_gpu` conda environment includes:

### Core Dependencies

- **Python**: 3.12
- **NumPy**, **SciPy**, **scikit-learn**
- **Hydra** for configuration management
- **laspy** for LiDAR file handling

### GPU Acceleration Stack

1. **CuPy** (13.6.0): GPU-accelerated NumPy
2. **RAPIDS cuML** (24.10): GPU-accelerated machine learning
3. **PyTorch** (2.5.1) with CUDA 12.1 support
   - torchvision
   - torchaudio

### CUDA Configuration

- **CUDA Version**: 12.5 (compatible with CUDA 13.0 runtime)
- **PyTorch CUDA**: 12.1

## Files Updated

### 1. `conda-recipe/environment_gpu.yml`

Complete conda environment specification with all GPU dependencies including PyTorch.

### 2. `install_cuml.sh`

Updated installation script that:

- Creates/updates `ign_gpu` environment (not `ign_lidar`)
- Installs RAPIDS cuML with CuPy
- Installs PyTorch with CUDA support
- Installs IGN LiDAR HD package
- Verifies all components including GPU detection
- Fixed import path: `ign_lidar.features.features_gpu` (v2.0 architecture)

### 3. `verify_gpu_installation.py`

Standalone verification script that checks:

- CuPy installation and GPU detection
- RAPIDS cuML version
- PyTorch with CUDA support
- IGN LiDAR HD GPU integration
- Compute capability and GPU specs

## Key Changes

### Import Path Migration

✅ **Fixed**: Changed from `ign_lidar.features_gpu` to `ign_lidar.features.features_gpu`

This reflects the v2.0 architecture reorganization where feature modules are now in the `features/` subdirectory.

### PyTorch Integration

✅ **Added**: Full PyTorch support with CUDA for:

- Deep learning-based point cloud processing
- PyTorch datasets (IGNLiDARMultiArchDataset)
- Neural network-based feature extraction

### Dependencies Added to Installation

- PyTorch (2.5.1) with CUDA 12.1
- torchvision, torchaudio
- pillow, h5py, hydra-core, omegaconf

## Installation Instructions

### Quick Start

```bash
# Run the installation script
./install_cuml.sh

# Or manually install PyTorch in existing environment
conda install -n ign_gpu -c pytorch -c nvidia pytorch pytorch-cuda=12.1 torchvision torchaudio -y
```

### From Environment File

```bash
# Create environment from file
conda env create -f conda-recipe/environment_gpu.yml

# Activate environment
conda activate ign_gpu

# Install IGN LiDAR HD in development mode
pip install -e .
```

## Verification

### Using the Script

```bash
conda activate ign_gpu
python verify_gpu_installation.py
```

### Expected Output

```
🔍 Verifying GPU installation...

   Python: 3.12.x
   ✓ CuPy: 13.6.0
   ✓ GPU: NVIDIA GeForce RTX 4080 SUPER (16.0 GB)
     - Compute Capability: 8.9
   ✓ RAPIDS cuML: 24.10.00
   ✓ PyTorch: 2.5.1
     - CUDA available: NVIDIA GeForce RTX 4080 SUPER
     - CUDA version: 12.1
   ✓ IGN LiDAR HD: 2.0.0
     - GPU_AVAILABLE: True
     - CUML_AVAILABLE: True

   🎉 All components verified successfully!
```

## Performance Expectations

### Processing 17M Points (1 tile)

- **CPU-only**: 60 minutes
- **CuPy only**: 7-10 minutes (6-8x speedup)
- **Full GPU** (CuPy + cuML): 3-5 minutes (12-20x speedup)
- **With PyTorch**: Additional deep learning capabilities

### Batch Processing (100 tiles)

- **CPU-only**: 100 hours
- **Full GPU**: 6 hours ← **Saves 94 hours!**

## Troubleshooting

### PyTorch Dataset Warning

If you see:

```
⚠️ Dataset classes not available (missing dependencies): No module named 'torch'
```

**Solution**: PyTorch is now installed in `ign_gpu` environment. The warning should disappear.

### CUDA Warnings

FutureWarnings about deprecated CUDA modules are suppressed in the verification script but are harmless.

### GPU Not Detected

1. Check NVIDIA drivers: `nvidia-smi`
2. Verify CUDA installation
3. Ensure you're in the `ign_gpu` environment: `conda activate ign_gpu`

## Next Steps

1. **Test the installation**: Run `python verify_gpu_installation.py`
2. **Process data**: Use `ign-lidar-hd enrich --use-gpu` for GPU-accelerated processing
3. **Benchmark**: Run performance tests to verify speedup
4. **Dataset loading**: Use PyTorch datasets with `IGNLiDARMultiArchDataset`

## Environment Management

### Activate Environment

```bash
conda activate ign_gpu
```

### Update Packages

```bash
conda update -n ign_gpu --all
```

### Export Environment

```bash
conda env export -n ign_gpu > environment_gpu_snapshot.yml
```

### Remove Environment

```bash
conda env remove -n ign_gpu
```

## References

- RAPIDS cuML: https://docs.rapids.ai/api/cuml/stable/
- CuPy Documentation: https://docs.cupy.dev/
- PyTorch CUDA: https://pytorch.org/get-started/locally/
- IGN LiDAR HD Docs: https://sducournau.github.io/IGN_LIDAR_HD_DATASET/
