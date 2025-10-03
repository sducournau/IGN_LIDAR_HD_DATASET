# 🎯 Release v1.1.0: QGIS Compatibility & Geometric Features

**🎉 Package now available on PyPI: `pip install ign-lidar-hd`**

This major release fixes critical issues with QGIS compatibility and geometric feature calculation, eliminating scan line artifacts and ensuring enriched LAZ files can be visualized in QGIS.

## 📦 Installation

```bash
pip install ign-lidar-hd
```

## 🆕 What's New

### 🔧 QGIS Compatibility

- **New QGIS conversion script** - Convert LAZ files for QGIS visualization
- **Format compatibility** - LAZ 1.4 → LAZ 1.2 conversion preserving key features
- **File size reduction** - ~73% smaller files (192 MB → 51 MB typical)

### 🎯 Enhanced Geometric Features

- **Radius-based calculations** - Eliminates scan line artifacts in geometric features
- **Auto-adaptive radius** - Automatically calculates optimal search radius (0.75-1.5m)
- **Fixed formulas** - Corrected normalization for linearity, planarity, sphericity

### 🛠️ Technical Improvements

- **LAZ compression fixes** - Proper compression for all output files
- **Backend compatibility** - Fixed laspy backend issues with latest versions
- **Diagnostic tools** - New validation and testing scripts

## 🔄 Migration from Previous Versions

1. **Update the package**:

   ```bash
   pip install --upgrade ign-lidar-hd
   ```

2. **Re-enrich existing files** (recommended to fix scan artifacts):

   ```bash
   ign-lidar enrich your_file.laz
   ```

3. **Convert for QGIS visualization**:
   ```bash
   python scripts/validation/simplify_for_qgis.py enriched_file.laz
   ```

## 📊 Key Features

- 🏗️ **14 core modules** - Complete LiDAR processing toolkit
- 🎯 **Building LOD classification** - ML-ready datasets for architectural analysis
- ⚡ **GPU/CPU processing** - Optimized performance for large datasets
- 🗺️ **QGIS integration** - Seamless visualization workflow
- 📈 **Smart data management** - Efficient tile downloading and processing

## 🐛 Bug Fixes

- Fixed geometric feature scan line artifacts
- Resolved QGIS file reading compatibility issues
- Fixed LAZ compression problems
- Corrected laspy backend detection errors
- Improved geometric formula normalization

## 📚 Documentation

- Complete QGIS troubleshooting guide
- Radius-based features technical documentation
- Migration guide for existing users
- Enhanced example scripts and workflows

## 🔗 Links

- **PyPI Package**: https://pypi.org/project/ign-lidar-hd/
- **Documentation**: See `docs/` directory
- **Examples**: See `examples/` directory
- **Issue Reports**: GitHub Issues

## 🙏 Acknowledgments

Built for processing IGN (Institut National de l'Information Géographique et Forestière) LiDAR HD data. Special thanks to the Python geospatial community for tools like laspy, scikit-learn, and numpy that make this work possible.
