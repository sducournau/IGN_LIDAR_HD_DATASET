#!/bin/bash
# Install RAPIDS cuML for Full GPU Acceleration
# This will give you 12-20x speedup vs CPU (vs 6-8x with CuPy only)
#
# Requirements:
#   - NVIDIA GPU with Compute Capability 6.0+ (4GB+ VRAM recommended)
#   - CUDA 12.0+ driver
#   - Linux or WSL2
#
# Performance expectations:
#   - CPU-only: 60 min for 17M points
#   - Hybrid (CuPy only): 7-10 min (6-8x speedup)
#   - Full GPU (this script): 3-5 min (12-20x speedup)

set -e  # Exit on error

echo "=============================================="
echo "RAPIDS cuML Installation for IGN LiDAR HD"
echo "=============================================="
echo ""
echo "📋 This script will:"
echo "   1. Create conda environment 'ign_gpu' with Python 3.12"
echo "   2. Install RAPIDS cuML 24.10 + CuPy"
echo "   3. Install IGN LiDAR HD and dependencies"
echo "   4. Verify GPU setup"
echo ""
echo "⏱️  Expected time: 10-20 minutes"
echo "💾 Required space: ~5GB"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Installation cancelled."
    exit 1
fi
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "📦 Conda not found. Installing Miniconda..."
    
    # Download Miniconda
    cd ~
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    
    # Install Miniconda
    bash miniconda.sh -b -p $HOME/miniconda3
    
    # Initialize conda
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init zsh
    
    echo "✅ Miniconda installed!"
    echo "⚠️  Please restart your terminal and run this script again"
    exit 0
else
    echo "✅ Conda found: $(conda --version)"
fi

echo ""
echo "🔧 Checking for existing 'ign_gpu' environment..."

# Check if environment exists
if conda env list | grep -q "ign_gpu"; then
    echo "   ⚠️  Environment 'ign_gpu' already exists"
    read -p "   Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   Removing existing environment..."
        conda env remove -n ign_gpu -y
        echo "   Creating fresh environment with Python 3.12..."
        conda create -n ign_gpu python=3.12 -y
    else
        echo "   Using existing environment (will update packages)..."
    fi
else
    echo "   Creating new environment with Python 3.12..."
    conda create -n ign_gpu python=3.12 -y
fi

# Activate environment
eval "$(conda shell.bash hook)"
conda activate ign_gpu

echo ""
echo "📦 Installing RAPIDS cuML and dependencies..."
echo "   This may take 5-10 minutes (downloading ~2GB)..."
echo ""

# Check CUDA version
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "   Detected CUDA Version: $CUDA_VERSION"
else
    echo "   ⚠️  Warning: nvidia-smi not found. Proceeding anyway..."
fi

# Install RAPIDS cuML (for CUDA 12.x, compatible with 13.0)
# Note: cuda-version=12.5 is compatible with CUDA 13.0 runtime
echo "   Installing RAPIDS cuML 24.10 with CUDA 12.5 compatibility..."
conda install -c rapidsai -c conda-forge -c nvidia \
    cuml=24.10 \
    cupy \
    cuda-version=12.5 \
    -y

echo ""
echo "📦 Installing project dependencies..."
echo ""

# Install other required packages
pip install laspy[lazrs] \
    numpy \
    scikit-learn \
    scipy \
    requests \
    tqdm \
    pyyaml

echo ""
echo "📦 Installing IGN LiDAR HD package..."
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Install the package in editable mode
if [ -f "$SCRIPT_DIR/pyproject.toml" ] || [ -f "$SCRIPT_DIR/setup.py" ]; then
    pip install -e "$SCRIPT_DIR"
else
    echo "   ⚠️  Cannot find package. Installing from PyPI..."
    pip install ign-lidar-hd
fi

echo ""
echo "🔍 Verifying installation..."
echo ""

# Verify installation
python << 'EOF'
import sys
print("   Python:", sys.version.split()[0])

try:
    import cupy as cp
    print(f"   ✓ CuPy: {cp.__version__}")
    device = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    gpu_name = props['name'].decode()
    total_mem = props['totalGlobalMem'] / (1024**3)
    print(f"   ✓ GPU: {gpu_name} ({total_mem:.1f} GB)")
except Exception as e:
    print(f"   ✗ CuPy error: {e}")
    sys.exit(1)

try:
    import cuml
    print(f"   ✓ RAPIDS cuML: {cuml.__version__}")
except Exception as e:
    print(f"   ✗ RAPIDS cuML error: {e}")
    sys.exit(1)

try:
    from ign_lidar.features_gpu import GPU_AVAILABLE, CUML_AVAILABLE
    print(f"   ✓ IGN LiDAR HD GPU integration")
    print(f"     - GPU_AVAILABLE: {GPU_AVAILABLE}")
    print(f"     - CUML_AVAILABLE: {CUML_AVAILABLE}")
    if not (GPU_AVAILABLE and CUML_AVAILABLE):
        print("   ⚠️  Warning: GPU or cuML not properly detected")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ IGN LiDAR HD error: {e}")
    sys.exit(1)

print("\n   🎉 All components verified successfully!")
EOF

VERIFY_STATUS=$?

if [ $VERIFY_STATUS -ne 0 ]; then
    echo ""
    echo "❌ Verification failed!"
    echo "   Please check the errors above and try:"
    echo "   1. Ensure NVIDIA drivers are installed: nvidia-smi"
    echo "   2. Check Python version: python --version (should be 3.12.x)"
    echo "   3. Reinstall cuML: conda install -c rapidsai cuml=24.10 -y"
    echo ""
    exit 1
fi

echo ""
echo "✅ Installation complete and verified!"
echo ""
echo "=============================================="
echo "🚀 Quick Start Guide"
echo "=============================================="
echo ""
echo "1️⃣  Activate the environment:"
echo "   conda activate ign_gpu"
echo ""
echo "2️⃣  Test GPU setup:"
echo "   python scripts/verify_gpu_setup.py"
echo ""
echo "3️⃣  Run a quick benchmark (1M points):"
echo "   python scripts/benchmarks/profile_gpu_bottlenecks.py --points 1000000"
echo ""
echo "4️⃣  Process your data:"
echo "   ign-lidar-hd enrich --input file.laz --output dir --use-gpu"
echo ""
echo "   You should see:"
echo "   ✓ RAPIDS cuML available - GPU algorithms enabled"
echo "   🚀 GPU chunked mode enabled (full acceleration)"
echo ""
echo "=============================================="
echo "📊 Performance Expectations"
echo "=============================================="
echo ""
echo "For a typical tile (17M points):"
echo "   • CPU-only:    60 minutes"
echo "   • Hybrid GPU:   7-10 minutes (6-8x speedup)"
echo "   • Full GPU:     3-5 minutes (12-20x speedup) ← YOU ARE HERE! 🎉"
echo ""
echo "Batch processing (100 tiles):"
echo "   • CPU-only:    100 hours"
echo "   • Hybrid GPU:   14 hours"
echo "   • Full GPU:      6 hours ← Saves 94 hours!"
echo ""
echo "=============================================="
echo "📚 Documentation & Support"
echo "=============================================="
echo ""
echo "• Installation Guide:    RAPIDS_INSTALLATION_GUIDE.md"
echo "• Performance Analysis:  GPU_PERFORMANCE_ANALYSIS.md"
echo "• Quick Start:          QUICK_START_GPU.md"
echo "• Troubleshooting:      https://sducournau.github.io/IGN_LIDAR_HD_DATASET/guides/gpu-acceleration"
echo ""
echo "Need help? Check 'python scripts/verify_gpu_setup.py' output for diagnostics"
echo ""
echo "=============================================="
