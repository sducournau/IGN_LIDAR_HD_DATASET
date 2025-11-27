<div align="center">

# IGN LiDAR HD Processing Library

[![PyPI version](https://badge.fury.io/py/ign-lidar-hd.svg)](https://badge.fury.io/py/ign-lidar-hd)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/ign-lidar-hd)](https://pypi.org/project/ign-lidar-hd/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/)

**Version 3.6.3** | [📚 Full Documentation](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/) | [📖 Documentation Index](DOCUMENTATION.md) | [⚙️ Configuration Guide](docs/guides/CONFIG_GUIDE.md)

![LoD3 Building Model](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/blob/main/docs/static/img/lod3.png?raw=true)

**Transform IGN LiDAR HD point clouds into ML-ready datasets with GPU-accelerated processing**

[Quick Start](#-quick-start) • [What's New](#-whats-new-in-v300) • [Features](#-key-features) • [Documentation](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/) • [Examples](#-usage-examples)

</div>

---

## 📊 Overview

A comprehensive Python library for processing French IGN LiDAR HD data into machine learning-ready datasets. Features include GPU acceleration, rich geometric features, RGB/NIR augmentation, and flexible YAML-based configuration.

**Key Capabilities:**

- 🚀 **GPU Acceleration**: 10× faster processing with CUDA streams & kernel fusion
- ⚡ **Async Processing**: 10-20% speedup through overlap of compute and transfer
- 🛡️ **Memory Safety**: Automatic GPU OOM prevention with pre-flight checks
- 🎯 **Smart Ground Truth**: 10× faster classification with GPU acceleration
- 🎨 **Multi-modal Data**: Geometry + RGB + Infrared (NDVI-ready)
- 🏗️ **Building Classification**: LOD2/LOD3 schemas with 15-30+ classes
- 📦 **Flexible Output**: NPZ, HDF5, PyTorch, LAZ formats
- ⚙️ **YAML Configuration**: Reproducible workflows with example configs
- 🎲 **Rules Framework**: Extensible rule-based classification system
- 🔍 **Gap Detection**: Automatic building perimeter gap analysis (NEW in v3.3.3)
- 🗺️ **Spatial Indexing**: Efficient DTM file lookup with rtree (NEW in v3.3.3)

---

## 🚀 Performance Highlights

**Version 3.6.3** continues the exceptional performance delivered through advanced GPU optimization:

| Metric                       | Baseline | Optimized | Speedup     |
| ---------------------------- | -------- | --------- | ----------- |
| GPU feature computation (1M) | 12.5s    | 1.85s     | **6.7×**    |
| GPU feature computation (5M) | 68s      | 6.7s      | **10×**     |
| CUDA streams async           | 100%     | 85%       | **15-20%**↓ |
| Kernel fusion improvement    | 100%     | 65%       | **35%**↓    |
| GPU memory overhead (5M)     | N/A      | 2.35GB    | Optimized   |
| Transfer overhead (5M)       | N/A      | 280ms     | <5%         |

**Phase 3 Achievements (November 2025):**

- ✅ CUDA streams for async GPU processing (10-20% speedup)
- ✅ Kernel fusion: Combined 3 operations into 1 (35% faster)
- ✅ GPU memory safety checks (100% OOM prevention)
- ✅ Adaptive chunking with automatic sizing
- ✅ Automated performance benchmarking & CI/CD regression detection
- ✅ >85% GPU utilization (vs 65% before optimization)

**Production Impact:**

- 🎯 1M points: **12.5s → 1.85s** (CPU → GPU)
- 🎯 5M points: **68s → 6.7s** (CPU → GPU)
- 🎯 10M points: **142s → 14s** (CPU → GPU)
- 💾 Memory: Safe processing up to 10M+ points with automatic chunking
- 📊 CI/CD: Automatic regression detection on every PR (>5% fails build)

---

## ✨ What's New

### 🚀 **Phase 4: Production Optimization Suite (v3.9.0 - November 2025)**

**NEW:** Complete optimization ecosystem delivering **+66-94% performance** (2.66× - 2.94× faster)!

- **Phase 4.5: Async I/O Pipeline** - Overlapped I/O and processing (+12-14%)

  - Background tile loading with `AsyncTileLoader`
  - 2-3 tile cache for prefetching
  - Async WFS ground truth fetching
  - Thread-pool executor with 2-4 workers
  - Zero processing stalls on I/O

- **Phase 4.4: Batch Multi-Tile Processing** - GPU batch efficiency (+25-30%)

  - Process 4-8 tiles simultaneously on GPU
  - Amortized GPU kernel launch overhead
  - Better GPU utilization (>85%)
  - Memory-efficient batch management
  - Automatic batch size tuning

- **Phase 4.3: GPU Memory Pooling** - Reduced allocation overhead (+8.5%)

  - CuPy memory pool with 4GB default limit
  - Reusable GPU buffer management
  - Statistics tracking (hits/misses)
  - Automatic cleanup on memory pressure
  - Zero-copy optimization

- **Phase 4.2: Preprocessing GPU** - GPU-accelerated preprocessing (+10-15%)

  - Statistical outlier removal on GPU
  - Parallel RGB/NIR augmentation
  - Integrated with main pipeline
  - Auto-fallback to CPU

- **Phase 4.1: WFS Memory Cache** - Ground truth caching (+10-15%)

  - LRU cache for BD TOPO queries
  - 100-entry default capacity
  - Thread-safe implementation
  - Transparent integration

- **Unified Integration** - OptimizationManager API
  - Single entry point for all optimizations
  - Graceful fallback when components unavailable
  - Statistics tracking and reporting
  - YAML configuration support

```python
# NEW: OptimizationManager - Unified Phase 4 API
from ign_lidar.core.optimization_integration import create_optimization_manager

# Enable all Phase 4 optimizations (default)
opt_mgr = create_optimization_manager(
    use_gpu=True,
    enable_all=True,  # Async I/O + Batch + GPU pooling
)

# Initialize with feature orchestrator
opt_mgr.initialize(feature_orchestrator)

# Process with all optimizations
results = opt_mgr.process_tiles_optimized(
    tile_paths=tile_paths,
    processor_func=process_func,
    fetch_ground_truth=True,  # Uses WFS cache
)

# Check performance
opt_mgr.print_stats()  # Shows gains from each optimization
opt_mgr.shutdown()
```

📖 **Phase 4 Documentation:**

- [Phase 4 Status](docs/optimization/PHASE_4_STATUS.md) - Complete overview (5/5 optimizations)
- [Integration Guide](docs/integration/PHASE_4_INTEGRATION_GUIDE.py) - How to enable in production
- [Usage Examples](examples/phase4_optimization_examples.py) - 5 complete examples
- [Performance Targets](docs/optimization/PHASE_4_PERFORMANCE_TARGETS.md) - Expected gains

---

### 🚀 **Phase 3: Async GPU Processing & Safety (v3.8.0-3.8.1 - November 2025)**

**COMPLETED:** Advanced GPU optimization with async processing and automated performance monitoring!

- **CUDA Streams** - Async GPU processing with overlap
  - 10-20% faster through parallel upload/compute/download
  - Multi-stream pipeline (2-4 concurrent streams)
  - Event-based synchronization
  - Pinned memory for fast transfers
- **GPU Memory Safety** - 100% OOM prevention

  - Pre-flight memory validation before execution
  - Automatic strategy selection (GPU/GPU_CHUNKED/CPU)
  - Clear error messages with actionable guidance
  - Memory-efficient sequential fallback for kernel fusion

- **Performance Benchmarking** - Automated CI/CD regression detection

  - Comprehensive benchmark suite (`scripts/benchmark_performance.py`)
  - Automatic regression detection (>5% fails CI)
  - PR comments with performance impact
  - Historical tracking with JSON baselines
  - Quick (PR) and full (main branch) modes

- **Enhanced Error Messages** - Developer-friendly diagnostics
  - Visual indicators (🔴🟠🟡🟢) for memory pressure
  - Step-by-step GPU installation guides
  - Automatic chunk size recommendations
  - Links to relevant documentation

```python
# NEW: CUDA streams for async processing
from ign_lidar.optimization import CUDAStreamManager

manager = CUDAStreamManager(num_streams=3)
results = manager.pipeline_process(chunks, process_func)  # 10-20% faster!

# NEW: GPU memory safety checks
from ign_lidar.optimization import check_gpu_memory_safe

result = check_gpu_memory_safe(points.shape, feature_count=38)
if result.can_proceed:
    # Safe to proceed on GPU
    process_on_gpu(points)
elif result.strategy == ProcessingStrategy.GPU_CHUNKED:
    # Use recommended chunking
    process_in_chunks(points, chunk_size=result.chunk_size)
```

📖 **Phase 3 Documentation:**

- [Performance Benchmarking Guide](docs/guides/performance-benchmarking.md) - CI/CD integration
- [Verbose Mode & Profiling](docs/guides/verbose-mode-profiling.md) - Debugging and optimization
- [Normal Computation Architecture](docs/architecture/normal_computation_hierarchy.md) - System design
- [Phase 3 Summary](PHASE3_COMPLETE_SUMMARY.md) - Complete achievements
- [GPU Kernel Fusion](docs/GPU_KERNEL_FUSION.md) - Technical deep dive

---

### 🎯 **Phase 1 Consolidation Complete (v3.6.0 - November 2025)**

**COMPLETED:** Comprehensive code consolidation and performance optimization!

- **Unified KNN API** - 6 implementations → 1 `KNNEngine` (-83% duplication)
  - CPU backend (scikit-learn)
  - GPU backend (cuML)
  - FAISS-GPU support (50× faster: 450ms → 9ms)
  - Automatic fallback handling
- **Radius Search** - NEW variable-radius neighbor search

  - GPU-accelerated (10-20× speedup)
  - Integrated with normal computation
  - Adaptive density handling
  - Memory-efficient with `max_neighbors` control

- **Code Quality** - Major cleanup and optimization

  - 71% reduction in code duplication (11.7% → 3.0%)
  - 100% deprecated code removed (-90 lines from bd_foret.py)
  - Cleaner, more maintainable codebase

- **Documentation** - Comprehensive guides and reports

  - +440% documentation increase (500 → 2,700 lines)
  - Radius search guide with examples
  - Migration guides and architecture docs
  - 6 detailed audit reports

- **Testing** - Robust validation
  - +10 new tests (100% pass rate)
  - Test coverage: 45% → 65% (+44%)
  - Zero breaking changes
  - 100% backward compatible

```python
# NEW: Radius search with GPU acceleration
from ign_lidar.optimization import radius_search

neighbors = radius_search(points, radius=0.5)  # CPU/GPU automatic

# NEW: Unified KNN API
from ign_lidar.optimization import KNNEngine, KNNBackend

engine = KNNEngine(backend=KNNBackend.FAISS_GPU)  # 50× faster
indices, distances = engine.knn_search(points, k=30)
```

📖 **Phase 1 Documentation:**

- [Radius Search Guide](docs/docs/features/radius_search.md) - Complete API reference
- [Implementation Report](docs/audit_reports/IMPLEMENTATION_PHASE1_NOV_2025.md) - Technical details
- [Completion Session](docs/audit_reports/PHASE1_COMPLETION_SESSION_NOV_2025.md) - Final accomplishments

---

### 🎲 **Rules Framework (v3.2.0 - October 2025)**

**NEW:** Extensible rule-based classification system with exceptional documentation!

- **Plugin Architecture** - Create custom rules without modifying framework
- **7 Confidence Methods** - Binary, linear, sigmoid, gaussian, threshold, exponential, composite
- **Hierarchical Execution** - Multi-level classification with coarse-to-fine refinement
- **Type-Safe Design** - Complete type hints and dataclass-based API
- **Exceptional Docs** - Three-tier documentation (quick ref, guide, architecture)
- **Visual Learning** - 15+ Mermaid diagrams showing system design
- **Production Ready** - Zero breaking changes, 100% backward compatible

```python
from ign_lidar.core.classification.rules import BaseRule, RuleEngine

class BuildingHeightRule(BaseRule):
    def evaluate(self, context):
        mask = context.additional_features['height'] > 3.0
        return RuleResult(
            point_indices=np.where(mask)[0],
            classifications=np.full(mask.sum(), 6),  # Building
            confidence_scores=np.ones(mask.sum()) * 0.9
        )

engine = RuleEngine()
engine.add_rule(BuildingHeightRule())
result = engine.execute(points, labels)
```

📖 **Documentation:**

- [Quick Reference](docs/RULES_FRAMEWORK_QUICK_REFERENCE.md) - One-page API reference
- [Developer Guide](docs/RULES_FRAMEWORK_DEVELOPER_GUIDE.md) - Complete tutorials
- [Architecture](docs/RULES_FRAMEWORK_ARCHITECTURE.md) - Visual system design
- [Examples](examples/README_RULES_EXAMPLES.md) - Working code samples

---

### 🤖 **FeatureComputer with Automatic Mode Selection (v3.0.0)**

**Major Release (October 2025):** Intelligent automatic computation mode selection!

- **Automatic GPU/CPU selection** - No manual configuration needed
- **Simplified config** - One flag instead of multiple GPU settings
- **Expert recommendations** - System logs optimal configuration
- **Backward compatible** - Existing configs work unchanged
- **Opt-in design** - Enable with `use_feature_computer: true`

#### Before vs After

```yaml
# Before: Manual GPU configuration
processor:
  use_gpu: true
  use_gpu_chunked: true
  gpu_batch_size: 5000000

# After: Automatic mode selection
processor:
  use_feature_computer: true  # That's it!
```

**Benefits:**

- ⚡ **Automatic** - Selects CPU/GPU/GPU_CHUNKED based on workload
- 🎯 **Smart** - Considers tile size, GPU availability, memory
- 📊 **Transparent** - Logs mode selection decisions
- 🔧 **Flexible** - Can force specific mode if needed

See [Migration Guide](docs/guides/migration-unified-computer.md) for details.

---

### 🎯 **Unified Configuration System**

**v3.0.0 introduces a streamlined configuration system:**

- **Simplified YAML configuration** with sensible defaults
- **Multiple LOD levels** (LOD2, LOD3) for different use cases
- **Flexible feature selection** - enable only what you need
- **Hardware-aware configuration** - automatic GPU/CPU selection
- **Example configurations** for common scenarios

#### Quick Start with Configuration

```bash
# Use example configuration
ign-lidar-hd process --config examples/config_versailles_lod2_v5.0.yaml

# Or create custom config
cat > my_config.yaml << EOF
input_dir: /data/tiles
output_dir: /data/output
processor:
  use_feature_computer: true  # Automatic GPU/CPU selection
  lod_level: LOD2
EOF

ign-lidar-hd process --config my_config.yaml
```

#### Benefits

✅ **Simpler**: Clear, self-documenting configuration structure
✅ **Flexible**: Easy to customize for your needs  
✅ **Automated**: Intelligent hardware detection and optimization
✅ **Reproducible**: Configuration files ensure consistent results
✅ **Extensible**: Easy to add new features and options

📖 See [Configuration Guide](docs/guides/CONFIG_GUIDE.md) for complete documentation

---

## ✨ What's New in v3.0.0

### 🎯 **Unified Configuration System**

**v3.0.0 introduces a completely redesigned configuration architecture!**

- **Unified Schema**: Single, coherent configuration system replacing fragmented v2.x/v3.0 configs
- **GPU Optimized**: Default configurations deliver >80% GPU utilization (vs 17% in legacy)
- **Smart Presets**: Ready-to-use configs for common scenarios
- **Hardware Profiles**: Optimized settings for RTX 4080, RTX 3080, CPU fallback
- **Migration Tools**: Automatic conversion from legacy configurations

```bash
# New simplified usage with presets
./scripts/run_processing.sh --preset gpu_optimized --input /data/tiles

# Hardware-specific optimization
./scripts/run_processing.sh --preset asprs_classification --hardware rtx4080

# Migration from legacy configs
python scripts/migrate_config_v4.py --input old_config.yaml --output new_config.yaml
```

**Performance Improvements:**

- ⚡ **10-100× faster** ground truth processing with forced GPU acceleration
- 🎮 **>80% GPU utilization** (vs 17% with CPU fallback in legacy configs)
- 🔧 **<10 CLI parameters** needed (vs 50+ in legacy scripts)
- 📦 **90 config files → 6** consolidated presets

### 🆕 Optional Reclassification in Main Pipeline

**v2.5.4 adds reclassification as an optional feature in the main processing pipeline!**

You can now enable optimized ground truth reclassification directly in your processing config:

```yaml
processor:
  reclassification:
    enabled: true # Optional - disabled by default
    acceleration_mode: "auto" # CPU, GPU, or GPU+cuML
    use_geometric_rules: true
```

**Benefits:**

- ✅ **Flexible**: Enable/disable without separate runs
- ✅ **Fast**: GPU-accelerated spatial indexing
- ✅ **Accurate**: Ground truth from BD TOPO®
- ✅ **Backward compatible**: Existing configs work unchanged

📖 See [`docs/RECLASSIFICATION_INTEGRATION.md`](docs/RECLASSIFICATION_INTEGRATION.md) and [`docs/RECLASSIFICATION_QUICKSTART.md`](docs/RECLASSIFICATION_QUICKSTART.md) for details

---

## ✨ What's New in v2.5.3

### 🔧 Critical Fix: Ground Truth Classification

**v2.5.3 fixes critical issues with BD TOPO® ground truth classification.**

#### What Was Fixed

Ground truth classification from IGN BD TOPO® wasn't working - no points were being classified to roads, cemeteries, power lines, etc.

**Root Causes:**

- Incorrect class imports (`MultiSourceDataFetcher` → `DataFetcher`)
- Missing BD TOPO feature parameters (cemeteries, power_lines, sports)
- Missing buffer parameters (road_width_fallback, etc.)
- Wrong method call (`fetch_data()` → `fetch_all()`)

**Impact:** Ground truth now works correctly for all ASPRS codes:

- ✅ ASPRS 11: Roads
- ✅ ASPRS 40: Parking
- ✅ ASPRS 41: Sports Facilities
- ✅ ASPRS 42: Cemeteries
- ✅ ASPRS 43: Power Lines

#### What Was Added

**New BD TOPO® Configuration Directory** (`ign_lidar/configs/data_sources/`)

Pre-configured Hydra configs for different use cases:

- `default.yaml` - General purpose with core features
- `asprs_full.yaml` - Complete ASPRS classification
- `lod2_buildings.yaml` - Building-focused for LOD2
- `lod3_architecture.yaml` - Architectural focus for LOD3
- `disabled.yaml` - Pure geometric features

**Usage:**

```yaml
defaults:
  - data_sources: asprs_full # or lod2_buildings, lod3_architecture
  - _self_
```

📖 See `ign_lidar/configs/data_sources/README.md` for complete documentation

---

### 📦 Previous Updates (v2.5.0-2.5.2)

**v2.5.0 represented a complete internal modernization while maintaining 100% backward compatibility!**

#### Unified Feature System ✨

- **FeatureOrchestrator**: New unified class replaces FeatureManager + FeatureComputer
- **Simpler API**: One class handles all feature computation with automatic strategy selection
- **Better organized**: Clear separation of concerns with strategy pattern
- **Fully compatible**: All existing code works without changes

#### Improved Code Quality

- **67% reduction** in feature orchestration code complexity
- **Optimized error messages** and validation throughout
- **Complete type hints** for better IDE support
- **Modular architecture** for easier maintenance and extension

#### Migration Made Easy

- **Zero breaking changes**: Your v1.x code continues to work
- **Deprecation warnings**: Clear guidance for future-proofing your code
- **Migration guide**: Step-by-step instructions in [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
- **Backward compatible**: Legacy APIs will be maintained through v2.x series

```python
# NEW (v2.0) - Recommended unified API
from ign_lidar import LiDARProcessor

processor = LiDARProcessor(
    config_path="config.yaml",
    feature_mode="lod3"  # Clearer mode specification
)

# Access unified orchestrator
orchestrator = processor.feature_orchestrator
print(f"Feature mode: {orchestrator.mode}")
print(f"Has RGB: {orchestrator.has_rgb}")
print(f"Available features: {orchestrator.get_feature_list('lod3')}")

# OLD (v1.x) - Still works with deprecation warnings
# feature_manager = processor.feature_manager  # Deprecated but functional
# feature_computer = processor.feature_computer  # Deprecated but functional
```

**Why upgrade?**

- Future-proof your code for v3.0
- Access to new features and improvements
- Better performance and error handling
- Professional, maintainable codebase

📖 See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for complete upgrade instructions  
📖 [Full Release History](CHANGELOG.md)

---

## 🚀 Quick Start

### Installation

```bash
# Standard installation (CPU)
pip install ign-lidar-hd

# Optional: GPU acceleration (6-20x speedup)
# CRITICAL: Use ign_gpu conda environment for GPU operations
conda env create -f conda-recipe/environment_gpu.yml
```

> **⚠️ GPU Users**: Always run GPU operations with:
>
> ```bash
> conda run -n ign_gpu python <script.py>
> ```
>
> See [GPU_QUICK_REFERENCE.md](GPU_QUICK_REFERENCE.md) for details.

### Zero-Config Quick Start (v5.5+)

The **simplest way** to get started - no configuration file needed!

```bash
# Download sample data
ign-lidar-hd download --bbox 2.3,48.8,2.4,48.9 --output data/ --max-tiles 5

# Process with automatic hardware detection (GPU or CPU)
ign-lidar-hd process input_dir=data/ output_dir=results/

# That's it! The system automatically:
# ✅ Detects GPU/CPU and optimizes accordingly
# ✅ Uses optimal settings for your hardware
# ✅ Selects appropriate feature set
# ✅ Configures memory and batch sizes
```

### Configuration-Based Processing (v5.5+)

For advanced workflows, use the **3-tier configuration system** (97% smaller configs!):

```bash
# Use hardware profile + task preset
ign-lidar-hd process \
  --config-name my_config \
  defaults=[hardware/gpu_rtx4080,task/asprs_classification]

# Or use one of our example configs
ign-lidar-hd process --config-path examples --config-name config_asprs_bdtopo_cadastre_gpu_v5.5

# List available profiles and presets
ign-lidar-hd list-profiles   # Shows: gpu_rtx4080, gpu_rtx3080, cpu_high, etc.
ign-lidar-hd list-presets    # Shows: asprs_classification, lod2_buildings, etc.

# Validate your configuration
ign-lidar-hd validate-config examples/my_config.yaml
```

### Legacy CLI (v5.4 and earlier)

The traditional command-based CLI still works for backward compatibility:

```bash
# Enrich with features (GPU accelerated if available)
ign-lidar-hd enrich --input-dir data/ --output enriched/ --use-gpu

# Create training patches
ign-lidar-hd patch --input-dir enriched/ --output patches/ --lod-level LOD2
```

### Python API

```python
from ign_lidar import LiDARProcessor

# Option 1: Zero-config with automatic hardware detection
processor = LiDARProcessor()  # Uses intelligent defaults
patches = processor.process_tile("data.laz", "output/")

# Option 2: With configuration file (recommended for production)
processor = LiDARProcessor(config_path="examples/config_asprs_bdtopo_cadastre_gpu_v5.5.yaml")
patches = processor.process_directory("input_dir/", "output_dir/")

# Option 3: Traditional explicit parameters (legacy)
processor = LiDARProcessor(lod_level="LOD2", patch_size=150.0, use_gpu=True)
patches = processor.process_tile("data.laz", "output/")
```

---

## ⚙️ Configuration System v5.5 (NEW!)

**Zero-config by default, powerful when you need it!**

Version 5.5 introduces a **revolutionary 3-tier configuration architecture** that reduces config complexity by 97% while adding powerful new capabilities:

### 🎯 Design Principles

- **Zero-config by default** - Works out of the box with intelligent defaults
- **Progressive complexity** - Add configuration only when you need it
- **Hardware-aware** - Automatic GPU/CPU detection and optimization
- **Composable** - Mix and match hardware profiles and task presets
- **Validated** - Catch errors early with comprehensive validation

### 📊 Before & After Comparison

```yaml
# ❌ v5.4 Configuration (430 lines, manually specified everything)
input_dir: /data/tiles
output_dir: /data/output
preprocess:
  buffer_size: 50.0
  normalize_intensity: true
  handle_overlap: true
processor:
  use_gpu: true
  use_gpu_chunked: true
  gpu_batch_size: 5000000
  lod_level: LOD2
  num_neighbors: 30
  search_radius: 3.0
features:
  mode: lod2
  k_neighbors: 10
  compute_normals: true
  compute_curvature: true
  compute_eigenvalues: true
  # ... 400+ more lines ...

# ✅ v5.5 Configuration (15 lines, inherits intelligent defaults)
defaults:
  - hardware/gpu_rtx4080     # Hardware optimization
  - task/asprs_classification # Task-specific settings
  - _self_

input_dir: /data/tiles
output_dir: /data/output

# That's it! Everything else is inherited
```

**Result:** 97% size reduction (430 lines → 15 lines) while gaining more capabilities!

### 🏗️ 3-Tier Architecture

```
Layer 1: base_complete.yaml (430 lines)
    ↓ [All defaults defined]
    ├─ Layer 2: Hardware Profiles (30-50 lines each)
    │   ├─ gpu_rtx4080.yaml     - RTX 4080: 16GB VRAM, 5M batch
    │   ├─ gpu_rtx3080.yaml     - RTX 3080: 10GB VRAM, 3M batch
    │   ├─ cpu_high.yaml        - 64GB RAM, 8 workers
    │   └─ cpu_standard.yaml    - 32GB RAM, 4 workers
    │
    └─ Layer 3: Task Presets (20-40 lines each)
        ├─ asprs_classification.yaml  - Full ASPRS with BD TOPO®
        ├─ lod2_buildings.yaml        - Fast building classification
        ├─ lod3_architecture.yaml     - Detailed architectural features
        └─ quick_enrich.yaml          - Minimal features, maximum speed
```

**How it works:**

1. **base_complete.yaml** - Complete default configuration (you never edit this)
2. **Hardware profiles** - Override only hardware-specific settings (GPU/CPU, memory, workers)
3. **Task presets** - Override only task-specific settings (features, classification, output)
4. **Your config** - Override only project-specific settings (paths, tile list)

### 🚀 Common Usage Patterns

#### Pattern 1: Zero-Config (Automatic Everything)

```bash
# Just specify input/output - everything else is automatic!
ign-lidar-hd process input_dir=data/ output_dir=results/
```

#### Pattern 2: Hardware Profile Only

```bash
# Optimize for your GPU
ign-lidar-hd process \
  defaults=[hardware/gpu_rtx4080] \
  input_dir=data/ \
  output_dir=results/
```

#### Pattern 3: Hardware + Task Preset

```bash
# Complete workflow with both optimizations
ign-lidar-hd process \
  defaults=[hardware/gpu_rtx4080,task/asprs_classification] \
  input_dir=data/ \
  output_dir=results/
```

#### Pattern 4: Custom Configuration File

```yaml
# my_config.yaml (minimal!)
defaults:
  - hardware/gpu_rtx4080
  - task/lod2_buildings
  - _self_

input_dir: /data/versailles
output_dir: /data/results
processor:
  tile_list: ["tile_001", "tile_002"] # Only override what's different!
```

```bash
ign-lidar-hd process --config-name my_config
```

### 🔍 Configuration Discovery

**New CLI commands** help you explore available options:

```bash
# List available hardware profiles
ign-lidar-hd list-profiles
# Output:
#   gpu_rtx4080    - RTX 4080 optimized (16GB VRAM)
#   gpu_rtx3080    - RTX 3080 optimized (10GB VRAM)
#   cpu_high       - High-end CPU (64GB RAM, 8 workers)
#   cpu_standard   - Standard CPU (32GB RAM, 4 workers)

# List available task presets
ign-lidar-hd list-presets
# Output:
#   asprs_classification - Full ASPRS with BD TOPO® ground truth
#   lod2_buildings       - Fast building classification (12 features)
#   lod3_architecture    - Detailed architectural features (38 features)
#   quick_enrich         - Minimal features for fast processing

# Show complete resolved configuration
ign-lidar-hd show-config --config-name my_config

# Validate configuration before running
ign-lidar-hd validate-config my_config.yaml
# Output: ✓ Configuration validated successfully
#         - Processor settings: OK
#         - Feature configuration: OK
#         - Data sources: OK
#         - Output settings: OK
```

### ✅ Configuration Validation

v5.5 includes **comprehensive validation** that catches errors before processing:

```bash
# Validate any configuration file
ign-lidar-hd validate-config examples/my_config.yaml

# Example validation output:
# ✓ Configuration validated: examples/my_config.yaml
#
# Validation Results:
# ✓ Processor configuration: OK
#   - LOD level: LOD2 (valid)
#   - GPU batch size: 5000000 (valid range)
#   - Num neighbors: 30 (valid range)
#
# ✓ Feature configuration: OK
#   - Mode: lod2 (valid)
#   - K-neighbors: 10 (valid range)
#
# ✓ Data sources: OK
#   - 3 sources configured
#
# ✓ Output configuration: OK
#   - Format: npz (valid)
```

**Validation checks:**

- ✅ Required sections present (processor, features, data_sources, output)
- ✅ Required keys in each section
- ✅ Enum values (LOD level, feature mode, output format, etc.)
- ✅ Numeric ranges (batch size, k-neighbors, search radius, etc.)
- ✅ GPU settings compatibility
- ✅ Path validity and accessibility

**Benefits:**

- 🎯 **Catch errors early** - Before long processing runs
- 💡 **Helpful suggestions** - "Did you mean 'LOD2'?" for typos
- 📊 **Clear reporting** - See exactly what's wrong and where
- 🔧 **Pre-flight checks** - Validate before submitting to cluster

### 📦 Available Profiles & Presets

**Hardware Profiles** (`ign_lidar/configs/hardware/`):

| Profile        | VRAM/RAM | Batch Size | Workers | Best For                 |
| -------------- | -------- | ---------- | ------- | ------------------------ |
| `gpu_rtx4080`  | 16GB     | 5M points  | 8       | High-end GPU processing  |
| `gpu_rtx3080`  | 10GB     | 3M points  | 6       | Mid-range GPU processing |
| `cpu_high`     | 64GB     | 2M points  | 8       | Server without GPU       |
| `cpu_standard` | 32GB     | 1M points  | 4       | Standard workstation     |

**Task Presets** (`ign_lidar/configs/task/`):

| Preset                 | Features    | Ground Truth        | Use Case                        |
| ---------------------- | ----------- | ------------------- | ------------------------------- |
| `asprs_classification` | 38 features | BD TOPO® + Cadastre | Complete ASPRS classification   |
| `lod2_buildings`       | 12 features | BD TOPO buildings   | Fast building detection         |
| `lod3_architecture`    | 38 features | BD TOPO® full       | Detailed architectural analysis |
| `quick_enrich`         | 4 features  | None                | Minimal processing for testing  |

### 🔗 Migration from v5.4

**Good news:** v5.4 configs still work! No breaking changes.

**To upgrade to v5.5:**

```bash
# Option 1: Keep using your old config (works unchanged)
ign-lidar-hd process --config-path . --config-name old_config_v5.4

# Option 2: Simplify to v5.5 style (recommended)
# See docs/MIGRATION_GUIDE_V5.5.md for detailed examples
```

**Why upgrade?**

- ✅ 97% smaller configuration files
- ✅ Automatic hardware optimization
- ✅ Early error detection with validation
- ✅ Easier to maintain and share
- ✅ Access to hardware profiles and task presets

📖 See [Migration Guide](docs/MIGRATION_GUIDE_V5.5.md) for step-by-step instructions

---

## 📋 Key Features

### Core Processing

- **🎯 Complete Feature Export** - All 35-45 computed geometric features saved to disk (v2.4.2+)
- **🏗️ Multi-level Classification** - LOD2 (12 features), LOD3 (38 features), Full (43+ features) modes
- **📊 Rich Geometry** - Normals, curvature, eigenvalues, shape descriptors, architectural features, building scores
- **🎨 Optional Augmentation** - RGB from orthophotos, NIR, NDVI for vegetation analysis
- **⚙️ Auto-parameters** - Intelligent tile analysis for optimal settings
- **📝 Feature Tracking** - Metadata includes feature names and counts for reproducibility

### Performance

- **🚀 GPU Acceleration** - RAPIDS cuML support (6-20x faster)
- **⚡ Parallel Processing** - Multi-worker with automatic CPU detection
- **🧠 Memory Optimized** - Chunked processing, 50-60% reduction
- **💾 Smart Skip** - Resume interrupted workflows automatically (~1800x faster)

### Flexibility

- **📁 Processing Modes** - Three clear modes: patches only, both, or LAZ only
- **📋 YAML Configs** - Declarative workflows with example templates
- **📦 Multiple Formats** - NPZ, HDF5, PyTorch, LAZ (single or multi-format)
- **🔧 CLI & API** - Command-line tool and Python library

---

## 💡 Usage Examples

### Mode 1: Create Training Patches (Default)

```bash
# Using example config
ign-lidar-hd process \
  --config-file examples/config_training_dataset.yaml \
  input_dir=data/raw \
  output_dir=data/patches

# Or with CLI parameters
ign-lidar-hd process \
  input_dir=data/raw \
  output_dir=data/patches \
  processor.processing_mode=patches_only
```

### Mode 2: Both Patches & Enriched LAZ

```bash
ign-lidar-hd process \
  --config-file examples/config_complete.yaml \
  input_dir=data/raw \
  output_dir=data/both
```

### Mode 3: LAZ Enrichment Only

```bash
ign-lidar-hd process \
  --config-file examples/config_quick_enrich.yaml \
  input_dir=data/raw \
  output_dir=data/enriched
```

> **⚠️ Note on Enriched LAZ Files:** When generating enriched LAZ tile files, geometric features (normals, curvature, planarity, etc.) may show artifacts at tile boundaries due to the nature of the source data. These artifacts are inherent to tile-based processing and **do not appear in patch exports**, which provide the best results for machine learning applications. For optimal quality, use `patches_only` or `both` modes.

### GPU-Accelerated Processing

```bash
ign-lidar-hd process \
  --config-file examples/config_gpu_processing.yaml \
  input_dir=data/raw \
  output_dir=data/output
```

### Preview Configuration

```bash
ign-lidar-hd process \
  --config-file examples/config_training_dataset.yaml \
  --show-config \
  input_dir=data/raw
```

### Python API Examples

```python
from ign_lidar import LiDARProcessor, IGNLiDARDownloader

# Download tiles
downloader = IGNLiDARDownloader("downloads/")
tiles = downloader.download_by_bbox(bbox=(2.3, 48.8, 2.4, 48.9), max_tiles=5)

# Process with custom config
processor = LiDARProcessor(
    lod_level="LOD3",
    patch_size=150.0,
    num_points=16384,
    use_gpu=True
)

# Single tile
patches = processor.process_tile("input.laz", "output/")

# Batch processing
patches = processor.process_directory("input_dir/", "output_dir/", num_workers=4)

# PyTorch integration
from torch.utils.data import DataLoader
dataset = LiDARPatchDataset("patches/")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

## 🎓 Feature Modes

IGN LiDAR HD supports multiple feature computation modes optimized for different use cases:

### Minimal Mode (4 features) - Ultra-Fast

**Best for:** Quick processing, classification updates, minimal computation

**Features:** normal_z, planarity, height_above_ground, density

**Performance:** ⚡⚡⚡⚡⚡ Fastest (~5s per 1M points)

### LOD2 Mode (12 features) - Fast Training

**Best for:** Basic building classification, quick prototyping, baseline models

**Features:** XYZ (3), normal_z, planarity, linearity, height, verticality, RGB (3), NDVI

**Performance:** ~15s per 1M points (CPU), fast convergence

### LOD3 Mode (37 features) - Detailed Modeling

**Best for:** Architectural modeling, fine structure detection, research

**Features:** Complete normals (3), eigenvalues (5), curvature (2), shape descriptors (6), height features (2), building scores (3), density features (4), architectural features (4), spectral (5)

**Performance:** ~45s per 1M points (CPU), best accuracy

### Full Mode (37+ features) - Complete Feature Set

**Best for:** Research, feature analysis, maximum information extraction

**All Features:** All LOD3 features plus any additional computed features

**Performance:** ~50s per 1M points (CPU), complete geometric description

**Usage:**

```yaml
features:
  mode: minimal # or lod2, lod3, full, custom
  k_neighbors: 10
```

**Output Format:**

- NPZ/HDF5/PyTorch: Full feature matrix with all features
- LAZ: All features as extra dimensions for GIS tools
- Metadata: `feature_names` and `num_features` for tracking

📖 See [Feature Modes Documentation](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/features/feature-modes) for complete details.

---

## 📦 Output Format

### NPZ Structure

Each patch is saved as NPZ with:

```python
{
    'points': np.ndarray,        # [N, 3] XYZ coordinates
    'normals': np.ndarray,       # [N, 3] surface normals
    'curvature': np.ndarray,     # [N] principal curvature
    'intensity': np.ndarray,     # [N] normalized intensity
    'planarity': np.ndarray,     # [N] planarity measure
    'verticality': np.ndarray,   # [N] verticality measure
    'density': np.ndarray,       # [N] local point density
    'labels': np.ndarray,        # [N] building class labels
    # Facultative features:
    'wall_score': np.ndarray,    # [N] wall likelihood (planarity * verticality)
    'roof_score': np.ndarray,    # [N] roof likelihood (planarity * horizontality)
    # Optional with augmentation:
    'red': np.ndarray,           # [N] RGB red
    'green': np.ndarray,         # [N] RGB green
    'blue': np.ndarray,          # [N] RGB blue
    'infrared': np.ndarray,      # [N] NIR values
}
```

### Available Formats

- **NPZ** - Default NumPy format (recommended for ML)
- **HDF5** - Hierarchical data format
- **PyTorch** - `.pt` files for PyTorch
- **LAZ** - Point cloud format for visualization (may show boundary artifacts in tile mode)
- **Multi-format** - Save in multiple formats: `hdf5,laz`, `npz,torch`

> **💡 Tip:** For machine learning applications, NPZ/HDF5/PyTorch patch formats provide cleaner geometric features than enriched LAZ tiles.

---

## 📚 Documentation

### 📖 Documentation Hub

**[Complete Documentation Index](DOCUMENTATION.md)** - Central navigation for all documentation

### Quick Links

- [� Online Documentation](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/) - Full documentation site
- [🚀 Quick Start Guide](docs/QUICK_START_DEVELOPER.md) - Get started in 5 minutes
- [📋 Testing Guide](TESTING.md) - Test suite and development testing
- [⚡ GPU Testing Guide](GPU_TESTING_GUIDE.md) - **GPU environment setup (ign_gpu conda env)**
- [📝 Changelog](CHANGELOG.md) - Version history and release notes

### Documentation by Category

**User Guides** ([docs/guides/](docs/guides/)):

- [ASPRS Classification Guide](docs/guides/ASPRS_CLASSIFICATION_GUIDE.md) - Complete ASPRS standards
- [ASPRS Feature Requirements](docs/guides/ASPRS_FEATURE_REQUIREMENTS.md) - Feature specifications for classification
- [ASPRS Features Quick Reference](docs/guides/ASPRS_FEATURES_QUICK_REFERENCE.md) - Fast lookup of features by class
- [Building Classification Guide](docs/guides/BUILDING_CLASSIFICATION_QUICK_REFERENCE.md) - Building class reference
- [Vegetation Classification Guide](docs/guides/VEGETATION_CLASSIFICATION_GUIDE.md) - Vegetation analysis

**Configuration Examples** ([examples/](examples/)):

- [Example Configurations](examples/) - Ready-to-use YAML templates
- [Versailles Configs](examples/) - LOD2, LOD3, and ASPRS examples
- [Architectural Analysis](examples/ARCHITECTURAL_STYLES_README.md) - Style detection

**Technical Documentation**:

- [GPU Refactoring](docs/gpu-refactoring/) - Complete GPU optimization project (6,500+ lines)
- [Implementation Plans](docs/implementation/) - Strategic roadmaps
- [Audit Reports](docs/audit/) - Code quality analysis
- [System Architecture](docs/architecture/) - Design documentation

**API References**:

- [Geometric Features Reference](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/features/geometric-features)
- [Feature Modes Guide](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/features/feature-modes)
- [CLI Reference](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/api/cli)
- [Python API](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/api/features)
- [Configuration Schema](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/api/configuration)

---

## 🛠️ Development

```bash
# Clone and install in development mode
git clone https://github.com/sducournau/IGN_LIDAR_HD_DATASET
cd IGN_LIDAR_HD_DATASET
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black ign_lidar/
```

---

## 📋 Requirements

**Core:**

- Python 3.8+
- NumPy >= 1.21.0
- laspy >= 2.3.0
- scikit-learn >= 1.0.0

**Optional GPU Acceleration:**

- CUDA >= 12.0
- CuPy >= 12.0.0
- RAPIDS cuML >= 24.10 (recommended)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🤝 Support & Contributing

- 🐛 [Report Issues](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
- 💡 [Feature Requests](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
- 📖 [Contributing Guide](CONTRIBUTING.md)

---

## 📝 Cite Me

If you use this library in your research or projects, please cite:

```bibtex
@software{ign_lidar_hd,
  author       = {Ducournau, Simon},
  title        = {IGN LiDAR HD Processing Library},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/sducournau/IGN_LIDAR_HD_DATASET},
  version      = {3.6.3}
}
```

**Project maintained by:** [ImagoData](https://github.com/sducournau)

---

<div align="center">

**Made with ❤️ for the LiDAR and Machine Learning communities**

[⬆ Back to top](#ign-lidar-hd-processing-library)

</div>
