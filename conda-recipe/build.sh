#!/bin/bash

# Conda build script for IGN LiDAR HD package
# This script builds the conda package from the recipe

set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Building IGN LiDAR HD conda package..."
echo "Project directory: $PROJECT_DIR"
echo "Recipe directory: $SCRIPT_DIR"

# Check if conda-build is installed
if ! command -v conda-build &> /dev/null; then
    echo "conda-build not found. Installing..."
    conda install -y conda-build
fi

# Build the package
echo "Building conda package..."
cd "$PROJECT_DIR"
conda-build conda-recipe/ --output-folder dist/conda

echo "✅ Conda package built successfully!"
echo "Package location: dist/conda/"

# Display the built package
echo "Built packages:"
find dist/conda/ -name "*.tar.bz2" -o -name "*.conda" | head -10