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

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source .venv/bin/activate
fi

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