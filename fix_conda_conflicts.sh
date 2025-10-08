#!/bin/bash
# Fix dependency conflicts in ign_gpu conda environment
# Addresses: corrupted PyTorch, ClobberErrors, and package conflicts

set -e  # Exit on error

echo "=============================================="
echo "Fixing ign_gpu Conda Environment"
echo "=============================================="
echo ""

# Activate environment
eval "$(conda shell.bash hook)"
conda activate ign_gpu

echo "🧹 Step 1: Cleaning corrupted PyTorch package cache..."
# Remove corrupted PyTorch package from cache
if [ -d "$HOME/miniconda3/pkgs/pytorch-2.5.1-py3.12_cuda12.1_cudnn9.1.0_0" ]; then
    rm -rf "$HOME/miniconda3/pkgs/pytorch-2.5.1-py3.12_cuda12.1_cudnn9.1.0_0"
    echo "   ✓ Removed corrupted PyTorch cache"
else
    echo "   ℹ️  PyTorch cache not found (may already be cleaned)"
fi

echo ""
echo "🧹 Step 2: Cleaning conda cache..."
conda clean --all -y

echo ""
echo "🔧 Step 3: Removing conflicting packages..."
# Remove packages that cause ClobberErrors
conda remove --force -y \
    pytorch \
    pytorch-cuda \
    torchvision \
    torchaudio \
    intel-openmp \
    llvm-openmp \
    jpeg \
    libjpeg-turbo \
    cuda-nvtx \
    libcusparse \
    libcusparse-dev \
    libgfortran-ng \
    2>/dev/null || echo "   ℹ️  Some packages not found (may not be installed)"

echo ""
echo "🔧 Step 4: Updating base packages from conda-forge..."
# Update core packages to use consistent channel (conda-forge)
conda install -c conda-forge -y \
    jpeg \
    llvm-openmp \
    libgfortran-ng

echo ""
echo "🔧 Step 5: Installing PyTorch with proper channel priority..."
# Install PyTorch from pytorch channel with CUDA support
# Use pytorch channel explicitly to avoid conflicts
conda install -c pytorch -c nvidia -y \
    pytorch::pytorch \
    pytorch::torchvision \
    pytorch::torchaudio \
    pytorch-cuda=12.1

echo ""
echo "🔧 Step 6: Ensuring RAPIDS cuML compatibility..."
# Ensure cuML and related packages are from rapidsai channel
conda install -c rapidsai -c conda-forge -c nvidia -y \
    cuml=24.10 \
    cupy \
    cuda-version=12.5

echo ""
echo "🔍 Step 7: Verifying installation..."
python << 'EOF'
import sys
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

errors = []

print("\n   Checking installations:")

# Check CuPy
try:
    import cupy as cp
    print(f"   ✓ CuPy: {cp.__version__}")
    device = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    gpu_name = props['name'].decode()
    print(f"   ✓ GPU: {gpu_name}")
except Exception as e:
    print(f"   ✗ CuPy error: {e}")
    errors.append("CuPy")

# Check cuML
try:
    import cuml
    print(f"   ✓ RAPIDS cuML: {cuml.__version__}")
except Exception as e:
    print(f"   ✗ RAPIDS cuML error: {e}")
    errors.append("cuML")

# Check PyTorch
try:
    import torch
    print(f"   ✓ PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"   ✓ PyTorch CUDA: {torch.version.cuda}")
    else:
        print(f"   ⚠️  PyTorch CUDA not available")
        errors.append("PyTorch CUDA")
except Exception as e:
    print(f"   ✗ PyTorch error: {e}")
    errors.append("PyTorch")

# Check IGN LiDAR
try:
    from ign_lidar.features.features_gpu import GPU_AVAILABLE, CUML_AVAILABLE
    print(f"   ✓ IGN LiDAR GPU: {GPU_AVAILABLE}")
    print(f"   ✓ IGN LiDAR cuML: {CUML_AVAILABLE}")
    if not (GPU_AVAILABLE and CUML_AVAILABLE):
        errors.append("IGN LiDAR GPU support")
except Exception as e:
    print(f"   ✗ IGN LiDAR error: {e}")
    errors.append("IGN LiDAR")

if errors:
    print(f"\n   ❌ Errors found in: {', '.join(errors)}")
    sys.exit(1)
else:
    print("\n   ✅ All checks passed!")
    sys.exit(0)
EOF

VERIFY_STATUS=$?

echo ""
if [ $VERIFY_STATUS -eq 0 ]; then
    echo "✅ Environment fixed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Test with: python -c 'import torch; print(torch.cuda.is_available())'"
    echo "2. Run verification: python verify_gpu_installation.py"
    echo ""
else
    echo "❌ Verification failed. Trying alternative fix..."
    echo ""
    echo "🔧 Alternative: Recreating environment from scratch..."
    
    # Deactivate first
    conda deactivate
    
    # Remove and recreate
    conda env remove -n ign_gpu -y
    
    echo "   Creating fresh environment..."
    conda env create -f conda-recipe/environment_gpu.yml
    
    echo ""
    echo "   Please run './install_cuml.sh' to complete the setup"
    exit 1
fi

echo "=============================================="
