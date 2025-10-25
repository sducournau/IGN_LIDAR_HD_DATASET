# Project Overview: IGN LiDAR HD Processing Library

## Purpose
A Python library for processing French IGN LiDAR HD data into machine learning-ready datasets with building Level of Detail (LOD) classification support.

## Version & Type
- **Version:** 3.0.0
- **Language:** Python 3.8+
- **Type:** Data Processing & ML Pipeline Library
- **License:** MIT

## Key Capabilities
- **GPU Acceleration:** 16× faster processing with optimized batching
- **Optimized Pipeline:** 8× overall speedup (80min → 10min per large tile)
- **Smart Ground Truth:** 10× faster classification with auto-method selection
- **Multi-modal Data:** Geometry + RGB + Infrared (NDVI-ready)
- **Building Classification:** LOD2/LOD3 schemas with 15-30+ classes
- **Flexible Output:** NPZ, HDF5, PyTorch, LAZ formats
- **YAML Configuration:** Reproducible workflows with example configs
- **Rules Framework:** Extensible rule-based classification system (v3.2.0)

## Core Technologies
- **LiDAR Processing:** laspy, lazrs, NumPy, SciPy
- **ML/Scientific:** scikit-learn, NumPy, PyTorch (optional)
- **GPU Acceleration:** CuPy, RAPIDS cuML, FAISS (optional)
- **Configuration:** Hydra, OmegaConf
- **Geospatial:** Shapely, GeoPandas, Rasterio, Rtree
- **Testing:** pytest

## LOD Classification
- **LOD2:** Simplified building classification (12 features, 15 classes)
- **LOD3:** Detailed architectural classification (38 features, 30+ classes)
- **ASPRS:** American Society for Photogrammetry standard codes

## Links
- **Repository:** https://github.com/sducournau/IGN_LIDAR_HD_DATASET
- **Documentation:** https://sducournau.github.io/IGN_LIDAR_HD_DATASET/
- **PyPI:** https://pypi.org/project/ign-lidar-hd/
