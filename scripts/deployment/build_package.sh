#!/bin/bash
# Quick Package Builder for IGN LiDAR HD
# This script rebuilds the package with updated metadata

set -e

echo "🔧 IGN LiDAR HD - Package Builder"
echo "================================"

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ ign_lidar_hd.egg-info/

# Build the package
echo "🏗️ Building package..."
python -m build

# Validate the package
echo "✅ Validating package..."
twine check dist/*

echo ""
echo "🎉 Package build completed successfully!"
echo ""
echo "📦 Distribution files created:"
ls -la dist/
echo ""
echo "🚀 Ready for upload to PyPI!"
echo ""
echo "Next steps:"
echo "1. Test upload: twine upload --repository testpypi dist/*"
echo "2. Production upload: twine upload dist/*"
echo ""