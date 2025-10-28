# Release v3.3.3 - Enhanced DTM Integration, Memory Optimization & Code Simplification

**Release Date:** October 28, 2025

## 🎯 Highlights

This release brings significant improvements in DTM processing, memory management, and code quality:

- **10× faster DTM lookup** with RTM spatial indexing
- **Intelligent gap filling** for missing DTM values
- **Automatic memory optimization** prevents OOM crashes on large tiles
- **40-50% faster processing** with memory-optimized configuration
- **+30-40% facade detection** improvement
- **Simplified API** with cleaner naming conventions

## ✨ New Features

### RTM Spatial Indexing

- **10× faster DTM file lookup** using rtree spatial index
- Efficient sub-second DTM file discovery for bounding boxes
- Automatic fallback to sequential search if rtree unavailable

### DTM Nodata Interpolation

- **Intelligent gap filling** for missing DTM values
- Nearest-neighbor interpolation using scipy KDTree (up to 10m radius)
- Accurate elevation estimation for complex terrain
- Graceful handling of urban areas and data gaps

### Multi-Scale Chunked Processing

- **Automatic memory optimization** based on available RAM (psutil)
- Auto-chunking when estimated memory >50% of available
- 2M-5M point chunks prevent out-of-memory crashes
- Seamless processing of 18M+ point tiles

### Memory-Optimized Configuration

- **NEW `asprs_memory_optimized.yaml`** for 28-32GB RAM systems
- Single-scale computation (40-50% faster than asprs_complete)
- 2m DTM grid spacing (75% fewer synthetic points)
- Peak memory: 20-24GB (vs 30-35GB in asprs_complete)
- 92-95% classification rate, 8-12 min per tile
- 5-7% artifact rate (vs 2-5% in asprs_complete)

### Enhanced Facade Detection

- **asprs_complete.yaml v6.3.2 optimizations**
- Adaptive buffers: 0.7m-7.5m range (increased from 0.6m-6.0m)
- Wall verticality threshold: 0.55 (lowered from 0.60)
- Ultra-fine gap detection: 60 sectors at 6° resolution
- Enhanced 3D bounding boxes: 6m overhang detection
- **Result**: +30-40% facade point capture, +25% verticality detection

### Building Cluster IDs

- Assign unique IDs to buildings from BD TOPO polygons
- Cadastral parcel cluster IDs for property-level analysis
- Enable building-level statistics and change detection
- Complete guide in `docs/docs/features/CLUSTER_ID_FEATURES_GUIDE.md`

### Configuration System Documentation

- **NEW `ign_lidar/config/README.md`**
- Complete configuration architecture documentation
- Python schema vs YAML configs explanation
- Migration guide from v2.x to v3.x
- Development guidelines for adding new options

## 🔄 Changes

### Simplified Naming Convention

Major refactoring for cleaner, more intuitive API:

- `UnifiedClassifier` → `Classifier` (removed redundant "Unified" prefix)
- `EnhancedBuildingClassifier` → `BuildingClassifier` (removed "Enhanced" prefix)
- `OptimizedReclassifier` → `Reclassifier` (removed "Optimized" prefix)

**Rationale:** Eliminates marketing-style prefixes, follows principle that current implementation should have the simple name.

**Impact:** Zero breaking changes - old names still work via backward compatibility layer.

**Migration:** Automatic via import redirects with deprecation warnings.

### DTM Augmentation

Enhanced validation parameters:

- Search radius: 12m (increased from 10m)
- Min neighbors: 4 (increased from 3)
- Min distance to existing: 0.4m (reduced from 0.5m for denser coverage)
- Max elevation difference: 6.0m (increased from 5.0m for complex terrain)

### Multi-Scale Processing

Memory-aware chunking strategy:

- Estimates 64 bytes per point per scale
- Auto-enables chunking if total memory >50% available RAM
- Target chunk size: ~20% of available memory
- Clamps to 100K-N range for reasonable performance

## 🚀 Performance Improvements

| Improvement             | Speedup                                        |
| ----------------------- | ---------------------------------------------- |
| DTM Lookup              | 10× faster                                     |
| Memory-Optimized Config | 40-50% faster (8-12 min vs 12-18 min per tile) |
| Facade Detection        | +30-40% point capture                          |
| Memory Safety           | Automatic chunking prevents OOM crashes        |

## 🐛 Bug Fixes

- **DTM Nodata Handling**: Fixed interpolation using nearest-neighbor search
- **Multi-Scale Memory**: Added psutil fallback for systems without memory detection
- **Version Consistency**: Aligned all version references to 3.3.3 across package files

## 🗑️ Removed

- **Deprecated Module**: Removed `ign_lidar/optimization/gpu_dataframe_ops.py` (relocated to `io/` in v3.1.0)
- **Obsolete Documentation**: Cleaned up 6 milestone tracking files:
  - `MULTI_SCALE_v6.2_PHASE5.1_ADAPTIVE_COMPLETE.md`
  - `MULTI_SCALE_v6.2_PHASE4_COMPLETE.md`
  - `MULTI_SCALE_IMPLEMENTATION_STATUS.md`
  - `HARMONIZATION_ACTION_PLAN.md`
  - `HARMONIZATION_SUMMARY.md`
  - `CODEBASE_AUDIT_2025-10-25.md`

## 📚 Documentation

- Updated `README.md` to v3.3.3 with gap detection and spatial indexing
- Updated `docs/docusaurus.config.ts` tagline
- Updated `docs/package.json` to v3.3.3
- Updated `docs/docs/intro.md` with complete v3.3.3 release notes
- New cluster ID features guide (400+ lines)
- New configuration system architecture documentation

## 🔧 Migration Guide

### For Users

**No action required!** All changes are backward compatible. Old naming conventions still work with deprecation warnings.

**Recommended updates:**

```python
# Old (still works)
from ign_lidar.core.classification import UnifiedClassifier
from ign_lidar.core.classification import EnhancedBuildingClassifier
from ign_lidar.core.classification import OptimizedReclassifier

# New (recommended)
from ign_lidar.core.classification import Classifier
from ign_lidar.core.classification import BuildingClassifier
from ign_lidar.core.classification import Reclassifier
```

### For Memory-Constrained Systems

If you have 28-32GB RAM (instead of the recommended 64GB+), use the new optimized config:

```yaml
# Use memory-optimized configuration
config: examples/config_asprs_memory_optimized.yaml
```

### For Best Facade Detection

Update to the enhanced v6.3.2 configuration:

```yaml
# Use asprs_complete.yaml v6.3.2
config: examples/config_asprs_complete.yaml
```

## 📦 Installation

### PyPI (Recommended)

```bash
pip install ign-lidar-hd==3.3.3
```

### From Source

```bash
git clone https://github.com/sducournau/IGN_LIDAR_HD_DATASET.git
cd IGN_LIDAR_HD_DATASET
git checkout v3.3.3
pip install -e .
```

### GPU Support (Optional)

```bash
# For CUDA 12.x
pip install cupy-cuda12x

# For CUDA 11.x
pip install cupy-cuda11x

# For RAPIDS cuML (requires conda)
conda install -c rapidsai -c conda-forge -c nvidia cuml cuspatial cudf
```

## 🧪 Testing

All tests pass:

```bash
pytest tests/ -v
# 340+ tests passed
```

## 🔗 Links

- **Documentation**: https://sducournau.github.io/IGN_LIDAR_HD_DATASET/
- **GitHub Repository**: https://github.com/sducournau/IGN_LIDAR_HD_DATASET
- **PyPI Package**: https://pypi.org/project/ign-lidar-hd/
- **Issues**: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues

## 👥 Contributors

- @sducournau - Lead Developer

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Full Changelog**: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/compare/v3.2.1...v3.3.3
