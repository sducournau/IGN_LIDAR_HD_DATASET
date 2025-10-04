#!/bin/bash
# Install RAPIDS cuML for Full GPU Acceleration
# This will give you 15-20x speedup vs CPU

set -e  # Exit on error

echo "=============================================="
echo "RAPIDS cuML Installation for IGN LiDAR HD"
echo "=============================================="
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
echo "🔧 Creating new conda environment: ign_gpu"
echo "   This will have Python 3.12 + RAPIDS cuML + CuPy"
echo ""

# Create conda environment with RAPIDS
conda create -n ign_gpu python=3.12 -y

# Activate environment
eval "$(conda shell.bash hook)"
conda activate ign_gpu

echo ""
echo "📦 Installing RAPIDS cuML and dependencies..."
echo "   This may take 5-10 minutes..."
echo ""

# Install RAPIDS cuML (for CUDA 12.x, compatible with 13.0)
conda install -c rapidsai -c conda-forge -c nvidia \
    cuml=24.10 \
    cupy \
    cudatoolkit=12.5 \
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

# Install the package in editable mode
cd /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET
pip install -e .

echo ""
echo "✅ Installation complete!"
echo ""
echo "=============================================="
echo "🚀 How to use:"
echo "=============================================="
echo ""
echo "1. Activate the environment:"
echo "   conda activate ign_gpu"
echo ""
echo "2. Set CUDA paths (add to ~/.zshrc if not already there):"
echo "   export PATH=/usr/local/cuda-13.0/bin:\$PATH"
echo "   export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:\$LD_LIBRARY_PATH"
echo ""
echo "3. Run your enrichment:"
echo "   ign-lidar-hd enrich --input file.laz --output dir --use-gpu"
echo ""
echo "4. You should see:"
echo "   ✓ RAPIDS cuML available - GPU algorithms enabled"
echo "   🚀 GPU chunked mode enabled (full acceleration)"
echo ""
echo "=============================================="
echo "Expected performance: 3-5 minutes for 17M points!"
echo "=============================================="
