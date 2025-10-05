#!/bin/bash

# IGN LiDAR HD - PyPI Upload Script
# This script helps automate the PyPI upload process

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}IGN LiDAR HD - PyPI Upload Helper${NC}"
echo "=================================="

# Check if we're in a conda environment or virtual environment
if [[ "$CONDA_DEFAULT_ENV" != "" ]]; then
    echo -e "${GREEN}Using conda environment: $CONDA_DEFAULT_ENV${NC}"
elif [[ "$VIRTUAL_ENV" != "" ]]; then
    echo -e "${GREEN}Using virtual environment: $VIRTUAL_ENV${NC}"
else
    echo -e "${YELLOW}No environment detected. Attempting to activate...${NC}"
    
    # Try to activate conda environment first (ign_gpu is the project's conda env)
    if command -v conda &> /dev/null; then
        # Check if ign_gpu conda environment exists
        if conda env list | grep -q "^ign_gpu "; then
            echo -e "${YELLOW}Activating conda environment 'ign_gpu'...${NC}"
            eval "$(conda shell.bash hook)"
            conda activate ign_gpu
        elif conda env list | grep -q "^base "; then
            echo -e "${YELLOW}Activating conda base environment...${NC}"
            eval "$(conda shell.bash hook)"
            conda activate base
        fi
    # Fall back to venv if conda not available
    elif [ -d ".venv" ]; then
        echo -e "${YELLOW}Activating virtual environment (.venv)...${NC}"
        source .venv/bin/activate
    elif [ -d "venv" ]; then
        echo -e "${YELLOW}Activating virtual environment (venv)...${NC}"
        source venv/bin/activate
    else
        echo -e "${RED}Warning: No Python environment found. Proceeding with system Python.${NC}"
        read -p "Continue anyway? (y/N): " continue_anyway
        if [[ $continue_anyway != [yY] && $continue_anyway != [yY][eE][sS] ]]; then
            echo -e "${YELLOW}Exiting.${NC}"
            exit 1
        fi
    fi
fi

# Check if required tools are installed
echo ""
echo -e "${YELLOW}Checking required tools...${NC}"

if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python is not installed or not in PATH${NC}"
    exit 1
fi

if ! python -c "import build" 2>/dev/null; then
    echo -e "${YELLOW}Installing 'build' package...${NC}"
    pip install build
fi

if ! python -c "import twine" 2>/dev/null; then
    echo -e "${YELLOW}Installing 'twine' package...${NC}"
    pip install twine
fi

echo -e "${GREEN}All required tools are available.${NC}"

# Ask for upload type
echo ""
echo "Select upload destination:"
echo "1) TestPyPI (recommended for testing)"
echo "2) Production PyPI"
read -p "Enter your choice (1 or 2): " choice

case $choice in
    1)
        REPO="testpypi"
        REPO_URL="--repository testpypi"
        TEST_URL="--index-url https://test.pypi.org/simple/"
        echo -e "${YELLOW}Selected: TestPyPI${NC}"
        ;;
    2)
        REPO="pypi"
        REPO_URL=""
        TEST_URL=""
        echo -e "${YELLOW}Selected: Production PyPI${NC}"
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

# Clean previous builds
echo -e "${YELLOW}Cleaning previous builds...${NC}"
rm -rf build/ dist/ ign_lidar_hd.egg-info/

# Build package
echo -e "${YELLOW}Building package...${NC}"
python -m build

# Validate package
echo -e "${YELLOW}Validating package...${NC}"
twine check dist/*

if [ $? -ne 0 ]; then
    echo -e "${RED}Package validation failed. Please fix errors and try again.${NC}"
    exit 1
fi

echo -e "${GREEN}Package validation passed!${NC}"

# Show files to be uploaded
echo ""
echo "Files to be uploaded:"
ls -la dist/

# Confirm upload
echo ""
read -p "Do you want to proceed with upload to $REPO? (y/N): " confirm

if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
    echo -e "${YELLOW}Upload cancelled.${NC}"
    exit 0
fi

# Upload to PyPI
echo -e "${YELLOW}Uploading to $REPO...${NC}"
twine upload $REPO_URL dist/*

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Upload successful!${NC}"
    
    if [ "$REPO" = "testpypi" ]; then
        echo ""
        echo "To test the installation, run:"
        echo "pip install $TEST_URL ign-lidar-hd"
    else
        echo ""
        echo "Package is now available on PyPI:"
        echo "pip install ign-lidar-hd"
        echo ""
        echo "Don't forget to:"
        echo "1. Create a GitHub release (v1.1.0)"
        echo "2. Update documentation with installation instructions"
        echo "3. Test the installation in a fresh environment"
    fi
else
    echo -e "${RED}Upload failed. Please check the error messages above.${NC}"
    exit 1
fi