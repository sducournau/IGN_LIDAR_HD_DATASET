#!/bin/bash
# GPU Processing Wrapper Script
# Ensures ign_gpu conda environment is activated

set -e

echo "========================================="
echo "🎮 RTX 4080 Super GPU Processing"
echo "========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ conda not found!"
    echo "   Please install Anaconda/Miniconda first"
    exit 1
fi

# Check if ign_gpu environment exists
if ! conda env list | grep -q "ign_gpu"; then
    echo "❌ ign_gpu environment not found!"
    echo ""
    echo "Creating ign_gpu environment..."
    conda create -n ign_gpu python=3.10 -y
    
    echo ""
    echo "Installing dependencies..."
    eval "$(conda shell.bash hook)"
    conda activate ign_gpu
    pip install cupy-cuda12x
    pip install -e .
    
    echo ""
    echo "✅ Environment created successfully"
else
    echo "✅ Found ign_gpu environment"
fi

echo ""
echo "Activating ign_gpu environment..."
eval "$(conda shell.bash hook)"
conda activate ign_gpu

echo ""
echo "Testing GPU detection..."
python test_gpu.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ GPU test failed!"
    echo "   Please check your CUDA installation and CuPy"
    exit 1
fi

echo ""
echo "========================================="
echo "🚀 Starting GPU-optimized processing"
echo "========================================="
echo ""

# Run the GPU-optimized script with any passed arguments
python process_asprs_with_cadastre_gpu.py "$@"
