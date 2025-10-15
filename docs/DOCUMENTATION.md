# IGN LiDAR HD - Documentation Index

Complete documentation guide for the IGN LiDAR HD Processing Library.

---

## 📚 Getting Started

- **[README.md](README.md)** - Main project overview and quick start
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and release notes
- **[TESTING.md](TESTING.md)** - Testing guide and test suite documentation
- **[LICENSE](LICENSE)** - MIT License

---

## 🌐 Online Documentation

### Main Documentation Site

**[https://sducournau.github.io/IGN_LIDAR_HD_DATASET/](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/)**

The full documentation site includes:

- Installation guides
- API reference
- Tutorials and examples
- Feature documentation
- Configuration schemas

---

## 📖 User Guides

All user guides are located in **`docs/guides/`**:

### Classification Guides

- **[ASPRS_CLASSIFICATION_GUIDE.md](docs/guides/ASPRS_CLASSIFICATION_GUIDE.md)** - Complete ASPRS classification standard guide
- **[ASPRS_IMPLEMENTATION_SUMMARY.md](docs/guides/ASPRS_IMPLEMENTATION_SUMMARY.md)** - Implementation details for ASPRS standards
- **[ASPRS_QUICK_REFERENCE.md](docs/guides/ASPRS_QUICK_REFERENCE.md)** - Quick reference for ASPRS classes
- **[BUILDING_CLASSIFICATION_IMPROVEMENTS.md](docs/guides/BUILDING_CLASSIFICATION_IMPROVEMENTS.md)** - Building classification improvements and best practices
- **[BUILDING_CLASSIFICATION_QUICK_REFERENCE.md](docs/guides/BUILDING_CLASSIFICATION_QUICK_REFERENCE.md)** - Quick reference for building classes
- **[VEGETATION_CLASSIFICATION_GUIDE.md](docs/guides/VEGETATION_CLASSIFICATION_GUIDE.md)** - Vegetation classification guide

---

## 🏗️ Architecture & Design

All architecture documentation is located in **`docs/architecture/`**:

- **[ARCHITECTURAL_STYLES_IMPLEMENTATION.md](docs/architecture/ARCHITECTURAL_STYLES_IMPLEMENTATION.md)** - Architectural styles and patterns
- **[CONFIG_CONSOLIDATION_PLAN.md](docs/architecture/CONFIG_CONSOLIDATION_PLAN.md)** - Configuration consolidation strategy
- **[CONFIG_CONSOLIDATION_SUMMARY.md](docs/architecture/CONFIG_CONSOLIDATION_SUMMARY.md)** - Configuration system overview
- **[CONFIG_OVERRIDE_ANALYSIS.md](docs/architecture/CONFIG_OVERRIDE_ANALYSIS.md)** - Configuration override system analysis

---

## 📋 Reference Documentation

All reference documentation is located in **`docs/references/`**:

- **[PHASE2_BASE_CONFIGS_COMPLETE.md](docs/references/PHASE2_BASE_CONFIGS_COMPLETE.md)** - Phase 2 base configurations

---

## 💻 Examples & Configuration

### Examples Directory (`examples/`)

Configuration templates and example scripts:

#### Configuration Files

- **[config_versailles_asprs.yaml](examples/config_versailles_asprs.yaml)** - Versailles ASPRS classification config
- **[config_versailles_lod2.yaml](examples/config_versailles_lod2.yaml)** - Versailles LOD2 configuration
- **[config_versailles_lod3.yaml](examples/config_versailles_lod3.yaml)** - Versailles LOD3 configuration
- **[config_architectural_analysis.yaml](examples/config_architectural_analysis.yaml)** - Architectural analysis configuration
- **[config_architectural_training.yaml](examples/config_architectural_training.yaml)** - Architectural training configuration

#### Documentation

- **[ARCHITECTURAL_CONFIG_REFERENCE.md](examples/ARCHITECTURAL_CONFIG_REFERENCE.md)** - Architectural configuration reference
- **[ARCHITECTURAL_STYLES_README.md](examples/ARCHITECTURAL_STYLES_README.md)** - Architectural styles guide
- **[MULTISCALE_QUICK_REFERENCE.md](examples/MULTISCALE_QUICK_REFERENCE.md)** - Multi-scale processing reference
- **[MULTI_SCALE_TRAINING_STRATEGY.md](examples/MULTI_SCALE_TRAINING_STRATEGY.md)** - Multi-scale training strategy
- **[README.md](examples/README.md)** - Examples overview

#### Scripts

- **[example_architectural_styles.py](examples/example_architectural_styles.py)** - Architectural styles example
- **[merge_multiscale_dataset.py](examples/merge_multiscale_dataset.py)** - Dataset merging script
- **[run_multiscale_training.sh](examples/run_multiscale_training.sh)** - Multi-scale training script
- **[test_ground_truth_module.py](examples/test_ground_truth_module.py)** - Ground truth testing

---

## 🔧 Additional Documentation

### In `docs/` Directory

- **[ROAD_SEGMENTATION_IMPROVEMENTS.md](docs/ROAD_SEGMENTATION_IMPROVEMENTS.md)** - Road segmentation improvements
- **[TRAINING_PLAN_LOD2_SELF_SUPERVISED.md](docs/TRAINING_PLAN_LOD2_SELF_SUPERVISED.md)** - Self-supervised training plan

---

## 📦 Package & Development

### Package Configuration

- **[pyproject.toml](pyproject.toml)** - Python package configuration
- **[MANIFEST.in](MANIFEST.in)** - Package manifest
- **[requirements.txt](requirements.txt)** - Python dependencies
- **[requirements_gpu.txt](requirements_gpu.txt)** - GPU dependencies
- **[pytest.ini](pytest.ini)** - Test configuration

### Conda Recipe

See **[conda-recipe/](conda-recipe/)** directory for conda package details:

- **[README.md](conda-recipe/README.md)** - Conda packaging guide
- **[PACKAGE_INFO.md](conda-recipe/PACKAGE_INFO.md)** - Package information

---

## 🎯 Quick Navigation

### By Task

| Task                | Documentation                                                                             |
| ------------------- | ----------------------------------------------------------------------------------------- |
| **Getting Started** | [README.md](README.md)                                                                    |
| **Installation**    | [Online Docs](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/installation/quick-start) |
| **Classification**  | [docs/guides/](docs/guides/)                                                              |
| **Configuration**   | [examples/](examples/), [docs/architecture/](docs/architecture/)                          |
| **Architecture**    | [docs/architecture/](docs/architecture/)                                                  |
| **Testing**         | [TESTING.md](TESTING.md)                                                                  |
| **Development**     | [README.md](README.md#-development)                                                       |

### By User Type

| User Type          | Recommended Reading                                                                        |
| ------------------ | ------------------------------------------------------------------------------------------ |
| **New User**       | [README.md](README.md) → [Online Docs](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/) |
| **Data Scientist** | [Classification Guides](docs/guides/) → [Examples](examples/)                              |
| **Developer**      | [Architecture Docs](docs/architecture/) → [TESTING.md](TESTING.md)                         |
| **Contributor**    | [README.md](README.md#-development) → [TESTING.md](TESTING.md)                             |

---

## 🔍 Search & Navigate

- **Source Code**: See `ign_lidar/` directory
- **Test Suite**: See `tests/` directory
- **Scripts**: See `scripts/` directory
- **Data**: See `data/` directory

---

## 📞 Support

- 🐛 [Report Issues](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
- 💡 [Feature Requests](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
- 📖 [Online Documentation](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/)

---

**Last Updated**: October 2025  
**Version**: 2.5.1
