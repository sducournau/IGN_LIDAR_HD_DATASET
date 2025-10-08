# GPU Dependencies - Quick Reference Card

## ⚠️ You're seeing these warnings?

```
⚠ CuPy non disponible - fallback CPU
⚠ RAPIDS cuML non disponible - fallback sklearn
```

**Don't worry!** Your application is working correctly. These warnings simply inform you that GPU acceleration is available but not currently installed.

## 🚀 Quick Fix

### Option 1: Quick Automated Installation (Recommended)

```bash
./install_cuml.sh
```

This script will set everything up automatically.

### Option 2: Manual Installation - Basic GPU (6-8x faster)

```bash
# Choose based on your CUDA version:
pip install cupy-cuda11x>=12.0.0  # For CUDA 11.x
# OR
pip install cupy-cuda12x>=12.0.0  # For CUDA 12.x
```

### Option 3: Manual Installation - Full GPU (12-20x faster)

```bash
# Using conda (recommended)
conda install -c conda-forge cupy
conda install -c rapidsai -c conda-forge -c nvidia cuml=24.10
```

## 📖 More Information

For detailed installation instructions, troubleshooting, and performance benchmarks, see:

- **[GPU_SETUP.md](GPU_SETUP.md)** - Comprehensive setup guide

## 💡 Do I Need GPU Acceleration?

| Your Situation                 | Recommendation                           |
| ------------------------------ | ---------------------------------------- |
| Small datasets (<1M points)    | CPU is fine, no GPU needed               |
| Medium datasets (1-10M points) | CuPy gives good speedup (Option 2)       |
| Large datasets (>10M points)   | Full GPU strongly recommended (Option 3) |
| No NVIDIA GPU                  | Stick with CPU, it works great!          |
| Production/batch processing    | Definitely use GPU for best performance  |

## ✅ Verify Installation

After installing, verify it works:

```bash
# Test CuPy
python -c "import cupy as cp; print('✅ CuPy installed successfully')"

# Test RAPIDS cuML
python -c "from cuml.neighbors import NearestNeighbors; print('✅ RAPIDS cuML installed successfully')"
```

## 🎯 Summary

- **Warnings are informational** - your code still works
- **GPU is optional** - install only if you want faster processing
- **Easy to add later** - you can add GPU support anytime
- **Automatic fallback** - CPU processing works without any changes

---

**TL;DR:** Run `./install_cuml.sh` for fastest setup, or ignore the warnings if CPU performance is acceptable for your use case.
