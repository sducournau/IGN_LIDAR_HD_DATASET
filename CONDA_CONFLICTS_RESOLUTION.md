# Conda Environment Conflicts Resolution

**Date:** October 8, 2025  
**Environment:** `ign_gpu`  
**Status:** ✅ Resolved

## Issues Encountered

### 1. Corrupted PyTorch Package

Multiple `CondaVerificationError` messages indicating corrupted PyTorch package cache:

```
CondaVerificationError: The package for pytorch located at
/home/simon/miniconda3/pkgs/pytorch-2.5.1-py3.12_cuda12.1_cudnn9.1.0_0
appears to be corrupted. Missing .pyc files in various subdirectories.
```

### 2. ClobberError Conflicts

Multiple package path collisions:

#### a. CUDA NVTX Headers

- **Conflict:** `nvidia::cuda-nvtx-12.1.105` vs `rapidsai::librmm-24.10.00`
- **Files:** NVTX header files in `include/nvtx3/`

#### b. CUDA cuSPARSE Libraries

- **Conflict:** `nvidia::libcusparse-12.0.2.55` vs `nvidia::libcusparse-dev-12.0.2.55`
- **Files:** Shared library files `lib/libcusparse.so.*`

#### c. JPEG Libraries

- **Conflict:** `defaults::jpeg-9f` vs `pytorch::libjpeg-turbo-2.0.0`
- **Files:** JPEG binaries and headers (`bin/cjpeg`, `include/jpeglib.h`, etc.)

#### d. Fortran Libraries

- **Conflict:** `defaults::libgfortran-ng-8.2.0` vs `conda-forge::libgfortran5-15.2.0`
- **Files:** Fortran runtime libraries (`lib/libgfortran.so*`)

#### e. OpenMP Libraries

- **Conflict:** `defaults::intel-openmp-2023.0.0` vs `defaults::llvm-openmp-14.0.6`
- **Files:** OpenMP runtime libraries (`lib/libiomp5.so`, `lib/libomptarget.so`)

## Root Causes

1. **Mixed Channels:** Packages from multiple channels (`defaults`, `conda-forge`, `pytorch`, `nvidia`, `rapidsai`) with overlapping dependencies
2. **Priority Issues:** No explicit channel priorities causing conda to select incompatible package combinations
3. **Cache Corruption:** Incomplete or corrupted package extraction in conda cache
4. **Dev vs Runtime:** Development packages (e.g., `libcusparse-dev`) conflicting with runtime packages

## Solution Applied

### Step 1: Clean Corrupted Cache

```bash
rm -rf $HOME/miniconda3/pkgs/pytorch-2.5.1-py3.12_cuda12.1_cudnn9.1.0_0
conda clean --all -y
```

### Step 2: Remove Conflicting Packages

Forcefully removed all conflicting packages:

```bash
conda remove --force -y \
    pytorch pytorch-cuda torchvision torchaudio \
    intel-openmp llvm-openmp \
    jpeg libjpeg-turbo \
    cuda-nvtx libcusparse libcusparse-dev \
    libgfortran-ng
```

### Step 3: Install with Channel Pinning

Reinstalled packages with explicit channel specifications:

```bash
# Base packages from conda-forge
conda install -c conda-forge -y \
    jpeg \
    llvm-openmp \
    libgfortran-ng

# PyTorch from pytorch channel
conda install -c pytorch -c nvidia -y \
    pytorch::pytorch \
    pytorch::torchvision \
    pytorch::torchaudio \
    pytorch-cuda=12.1

# RAPIDS from rapidsai channel
conda install -c rapidsai -c conda-forge -c nvidia -y \
    cuml=24.10 \
    cupy \
    cuda-version=12.5
```

### Step 4: Updated environment.yml

Modified `conda-recipe/environment_gpu.yml` to prevent future conflicts:

- Added explicit channel prefixes (e.g., `conda-forge::numpy`)
- Reordered channels: `pytorch`, `rapidsai`, `conda-forge`, `nvidia`
- Removed `defaults` channel from explicit specs
- Added consistent OpenMP library (`llvm-openmp` instead of `intel-openmp`)

## Verification

After fixes, all components verified successfully:

```
✓ Python: 3.12.7
✓ CuPy: 13.6.0
✓ GPU: NVIDIA GeForce RTX 4080 SUPER (16.0 GB, Compute 8.9)
✓ RAPIDS cuML: 24.10.00
✓ PyTorch: 2.5.1 (CUDA 12.1, cuDNN 90100)
✓ IGN LiDAR HD: 2.0.0
  - GPU_AVAILABLE: True
  - CUML_AVAILABLE: True
```

## Prevention

### Best Practices for Conda Environments

1. **Use Channel Pinning:**

   ```yaml
   dependencies:
     - conda-forge::numpy>=1.21.0
     - pytorch::pytorch>=2.0.0
   ```

2. **Strict Channel Priority:**

   ```bash
   conda config --set channel_priority strict
   ```

3. **Single Source for Similar Packages:**

   - Use `llvm-openmp` from conda-forge (not `intel-openmp`)
   - Use `jpeg` from conda-forge (not `libjpeg-turbo` from pytorch)

4. **Avoid Development Packages Unless Needed:**

   - Don't install both `libcusparse` and `libcusparse-dev` unless compiling CUDA code

5. **Regular Cache Cleaning:**

   ```bash
   conda clean --all -y  # Monthly or after failed installs
   ```

6. **Recreate Environment from Scratch:**
   When conflicts are complex:
   ```bash
   conda env remove -n ign_gpu -y
   conda env create -f conda-recipe/environment_gpu.yml
   ```

## Files Created/Modified

1. **Created:** `fix_conda_conflicts.sh` - Automated resolution script
2. **Modified:** `conda-recipe/environment_gpu.yml` - Added channel pinning
3. **Created:** This documentation

## Commands for Quick Recovery

If issues recur:

```bash
# Quick fix
./fix_conda_conflicts.sh

# Or nuclear option (full rebuild)
conda env remove -n ign_gpu -y
conda env create -f conda-recipe/environment_gpu.yml
cd /path/to/project
pip install -e .
```

## References

- [Conda Channel Priority](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-channels.html#channel-priority)
- [Managing Environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
- [RAPIDS Installation Guide](https://docs.rapids.ai/install)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)

## Status

✅ **Environment is now fully functional and ready for GPU-accelerated processing.**

Expected performance:

- CPU-only: 60 minutes for 17M points
- Hybrid GPU: 7-10 minutes (6-8x speedup)
- **Full GPU (current): 3-5 minutes (12-20x speedup)** 🎉
