#!/bin/bash

# Deploy Docusaurus to GitHub Pages
# This script builds and deploys the documentation site

set -e

echo "🚀 Deploying IGN LiDAR HD Documentation to GitHub Pages..."

# Navigate to website directory
cd "$(dirname "$0")/website"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Build the site
echo "🔧 Building documentation site..."
npm run build

# Deploy to GitHub Pages
echo "📤 Deploying to GitHub Pages..."
npm run deploy

echo "✅ Deployment completed successfully!"
echo "📚 Documentation is available at: https://sducournau.github.io/IGN_LIDAR_HD_DATASET/"