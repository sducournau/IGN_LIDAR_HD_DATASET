# Configuration System Visual Architecture

**Version:** 3.1.0 → 4.0.0 Transition  
**Last Updated:** November 28, 2025

---

## 📐 System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    IGN LiDAR HD Configuration System                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌────────────────┐          ┌────────────────┐         ┌────────────────┐
│  Python Config │          │  YAML Configs  │         │   CLI Layer    │
│   (config.py)  │◄────────►│  (configs/)    │◄────────┤  (Hydra + Click)│
└────────────────┘          └────────────────┘         └────────────────┘
        │                           │                           │
        │                           │                           │
        ▼                           ▼                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Unified Configuration                          │
│                    (OmegaConf DictConfig)                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Validation Layer                            │
│              (ConfigValidator, type checking)                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Processing Pipeline                           │
│            (LiDARProcessor, FeatureOrchestrator)                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Directory Structure

### Current State (v3.1)

```
IGN_LIDAR_HD_DATASET/
│
├── ign_lidar/
│   ├── config/                          # Python configuration modules
│   │   ├── __init__.py
│   │   ├── config.py                    # ✅ NEW Config class (v3.2+)
│   │   ├── schema.py                    # ⚠️  DEPRECATED (v3.1)
│   │   ├── schema_simplified.py         # ⚠️  DEPRECATED (v3.1)
│   │   ├── building_config.py           # ✅ Building-specific config
│   │   ├── preset_loader.py             # ✅ Preset loading logic
│   │   ├── validator.py                 # ✅ Config validation
│   │   └── README.md                    # ⚠️  Needs update
│   │
│   └── configs/                         # YAML configuration files
│       ├── base.yaml                    # 🎯 Base defaults (436 lines)
│       ├── base/                        # 📦 Modular base components
│       │   ├── processor.yaml
│       │   ├── features.yaml
│       │   ├── data_sources.yaml
│       │   ├── ground_truth.yaml
│       │   ├── output.yaml
│       │   └── monitoring.yaml
│       │
│       ├── presets/                     # 🚀 Ready-to-use presets (7 files)
│       │   ├── asprs_classification_gpu.yaml
│       │   ├── asprs_classification_cpu.yaml
│       │   ├── lod2_buildings.yaml
│       │   ├── lod3_detailed.yaml
│       │   ├── fast_preview.yaml
│       │   ├── minimal_debug.yaml
│       │   └── high_quality.yaml
│       │
│       ├── hardware/                    # ⚡ Hardware profiles (5 files)
│       │   ├── gpu_rtx4090_24gb.yaml
│       │   ├── gpu_rtx4080_16gb.yaml
│       │   ├── gpu_rtx3080_12gb.yaml
│       │   ├── cpu_high_end.yaml
│       │   └── cpu_standard.yaml
│       │
│       ├── advanced/                    # 🔬 Specialized configs (5 files)
│       │   ├── asprs_classification_gpu_optimized.yaml
│       │   ├── heritage_lod3.yaml
│       │   ├── building_detection.yaml
│       │   ├── vegetation_ndvi.yaml
│       │   └── self_supervised.yaml
│       │
│       ├── archive/                     # 📚 Historical docs
│       └── README.md                    # ✅ V5.1 guide
│
└── examples/
    ├── TEMPLATE_v3.2.yaml               # Template (v3.2 style)
    ├── config_training_fast_50m_v3.2.yaml
    ├── config_asprs_production.yaml
    └── config_multi_scale_adaptive.yaml
```

### Proposed Structure (v4.0)

```diff
IGN_LIDAR_HD_DATASET/
│
├── ign_lidar/
│   ├── config/
│   │   ├── __init__.py
│   │   ├── config.py                    # ✅ SINGLE Config class
-  │   │   ├── schema.py                    # ❌ REMOVED
-  │   │   ├── schema_simplified.py         # ❌ REMOVED
│   │   ├── building_config.py           # ✅ Keep (specialized)
│   │   ├── preset_loader.py             # ✅ Keep
│   │   ├── validator.py                 # ✅ Keep
+  │   │   ├── migration.py                 # 🆕 Config migration
-  │   │   └── README.md                    # ❌ Move to docs/
│   │
│   └── configs/
-      │   ├── base.yaml                    # Simplified structure ✏️
+      │   ├── base.yaml                    # v4.0 flat structure
│       ├── base/                        # Keep modular components
│       ├── presets/                     # ✏️  Updated to v4.0
│       ├── hardware/                    # Keep as-is
│       ├── advanced/                    # Keep as-is
│       └── archive/
+          │   └── README_v3.1.md            # Archived
+          │   └── README_v5.1.md            # Archived
│
├── examples/
+   ├── TEMPLATE_v4.0.yaml               # 🆕 v4.0 template
-   ├── TEMPLATE_v3.2.yaml               # ❌ Archive
    └── ...
│
└── docs/docs/
+   └── guides/
+       └── configuration/               # 🆕 Unified documentation
+           ├── index.md
+           ├── quickstart.md
+           ├── reference.md
+           ├── presets.md
+           ├── advanced.md
+           └── migration-v3-to-v4.md
```

---

## 🔄 Configuration Flow Diagrams

### Current Flow (v3.1 - Multiple Paths)

```
User Input
    │
    ├─── Python API ──────────┐
    │       │                 │
    │       ├─ Config ─────────────────┐
    │       │  (v3.2 style)            │
    │       │                          │
    │       └─ IGNLiDARConfig ──────────┤ (deprecated)
    │          (v3.1 schema.py)        │
    │                                  │
    ├─── YAML File ───────────────────────┤
    │       │                          │
    │       ├─ v3.2 flat style         │
    │       ├─ v5.1 nested style       │
    │       └─ v3.1 legacy             │
    │                                  │
    └─── CLI Arguments ───────────────────┤
            (Hydra overrides)          │
                                       │
                                       ▼
                            ┌───────────────────┐
                            │  _migrate_config  │
                            │   (scattered)     │
                            └───────────────────┘
                                       │
                                       ▼
                            ┌───────────────────┐
                            │   Validation      │
                            │  (inconsistent)   │
                            └───────────────────┘
                                       │
                                       ▼
                            ┌───────────────────┐
                            │  ProcessorCore    │
                            └───────────────────┘
```

**Problems:**

- 🔴 3 different config formats
- 🔴 Multiple conversion paths
- 🔴 Inconsistent validation
- 🔴 Confusing for users

### Proposed Flow (v4.0 - Unified Path)

```
User Input
    │
    ├─── Python API ──────────┐
    │       │                 │
    │       └─ Config ─────────────────┐ (single class)
    │          (v4.0 unified)          │
    │                                  │
    ├─── YAML File ───────────────────────┤
    │       │                          │
    │       └─ v4.0 standard format    │
    │                                  │
    └─── CLI Arguments ───────────────────┤
            (Hydra overrides)          │
                                       │
                                       ▼
                            ┌───────────────────┐
                            │  Config.from_*()  │
                            │  (unified loader) │
                            └───────────────────┘
                                       │
                                       ▼
                            ┌───────────────────┐
                            │   Validation      │
                            │   (type-safe)     │
                            └───────────────────┘
                                       │
                                       ▼
                            ┌───────────────────┐
                            │  ProcessorCore    │
                            └───────────────────┘
```

**Benefits:**

- ✅ Single config format
- ✅ Unified loading
- ✅ Type-safe validation
- ✅ Clear for users

---

## 🏗️ Class Hierarchy

### Current (v3.1)

```
┌──────────────────────────────────────────────────────────┐
│                    DEPRECATED (v3.1)                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  @dataclass IGNLiDARConfig                               │
│  ├── processor: ProcessorConfig                          │
│  │   ├── lod_level: str                                  │
│  │   ├── use_gpu: bool                                   │
│  │   ├── num_workers: int                                │
│  │   └── ... (20+ fields)                                │
│  │                                                        │
│  ├── features: FeaturesConfig                            │
│  │   ├── mode: str                                       │
│  │   ├── k_neighbors: int                                │
│  │   ├── multi_scale_computation: bool                   │
│  │   └── ... (30+ fields)                                │
│  │                                                        │
│  ├── preprocess: PreprocessConfig                        │
│  ├── stitching: StitchingConfig                          │
│  ├── output: OutputConfig                                │
│  └── bbox: BBoxConfig                                    │
│                                                          │
│  Total: 118 parameters (deeply nested)                   │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                   CURRENT (v3.2+)                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  @dataclass Config                                       │
│  ├── input_dir: str                                      │
│  ├── output_dir: str                                     │
│  ├── mode: str                    # flat structure       │
│  ├── processing_mode: str                                │
│  ├── use_gpu: bool                                       │
│  ├── num_workers: int                                    │
│  ├── patch_size: float                                   │
│  ├── num_points: int                                     │
│  ├── patch_overlap: float                                │
│  ├── architecture: str                                   │
│  │                                                        │
│  ├── features: FeatureConfig      # nested (simplified)  │
│  │   ├── feature_set: str                                │
│  │   ├── k_neighbors: int                                │
│  │   ├── use_rgb: bool                                   │
│  │   ├── use_nir: bool                                   │
│  │   ├── compute_ndvi: bool                              │
│  │   ├── multi_scale: bool                               │
│  │   └── scales: List[str]                               │
│  │                                                        │
│  └── advanced: Optional[AdvancedConfig]  # for experts   │
│                                                          │
│  Total: 15 top-level + 7 feature params (simple!)       │
└──────────────────────────────────────────────────────────┘
```

### Proposed (v4.0 - Harmonized)

```
┌──────────────────────────────────────────────────────────┐
│                     UNIFIED (v4.0)                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  @dataclass Config                                       │
│  ├── [REQUIRED]                                          │
│  │   ├── input_dir: str                                  │
│  │   └── output_dir: str                                 │
│  │                                                        │
│  ├── [CORE]                                              │
│  │   ├── mode: Literal["asprs", "lod2", "lod3"]         │
│  │   ├── processing_mode: str                            │
│  │   ├── use_gpu: bool                                   │
│  │   └── num_workers: int                                │
│  │                                                        │
│  ├── [PATCHES]                                           │
│  │   ├── patch_size: float                               │
│  │   ├── num_points: int                                 │
│  │   ├── patch_overlap: float                            │
│  │   └── architecture: str                               │
│  │                                                        │
│  ├── [FEATURES] (nested)                                 │
│  │   features: FeatureConfig                             │
│  │   ├── mode: Literal["minimal", "standard", "full"]   │
│  │   ├── k_neighbors: int                                │
│  │   ├── use_rgb: bool                                   │
│  │   ├── use_nir: bool                                   │
│  │   └── compute_ndvi: bool                              │
│  │                                                        │
│  ├── [OPTIMIZATIONS] (nested) 🆕                         │
│  │   optimizations: OptimizationsConfig                  │
│  │   ├── enabled: bool                                   │
│  │   ├── async_io: Dict                                  │
│  │   ├── batch_processing: Dict                          │
│  │   └── gpu_pooling: Dict                               │
│  │                                                        │
│  └── [ADVANCED] (optional)                               │
│      advanced: Optional[AdvancedConfig]                  │
│      ├── preprocessing: Dict                             │
│      ├── ground_truth: Dict                              │
│      ├── classification: Dict                            │
│      └── performance: Dict                               │
│                                                          │
│  Methods:                                                │
│  ├── .preset(name) -> Config                             │
│  ├── .from_yaml(path) -> Config                          │
│  ├── .from_environment() -> Config                       │
│  ├── .from_legacy_schema(old) -> Config  🆕              │
│  └── .validate() -> List[str]                            │
│                                                          │
│  Total: 15 top-level + organized subsections             │
└──────────────────────────────────────────────────────────┘
```

---

## 📊 Configuration Composition

### Hydra Composition (v5.1 YAML)

```yaml
# presets/asprs_classification_gpu.yaml

defaults:
  - ../base # ← Layer 1: Base defaults
  - _self_ # ← Layer 2: This preset

# Preset overrides
processor:
  lod_level: "ASPRS"
  use_gpu: true

features:
  mode: "asprs_classes"
  k_neighbors: 60
```

**Composition order:**

1. `configs/base.yaml` (foundation)
2. `configs/presets/asprs_classification_gpu.yaml` (overrides)
3. `configs/hardware/gpu_rtx4080_16gb.yaml` (if specified)
4. CLI arguments (highest priority)

**Example:**

```bash
ign-lidar-hd process \
  -c presets/asprs_classification_gpu.yaml \    # Preset
  -c hardware/gpu_rtx4080_16gb.yaml \           # Hardware profile
  input_dir=/data/tiles \                       # CLI override
  features.k_neighbors=80                       # CLI override
```

**Effective configuration:**

```
base.yaml
  + asprs_classification_gpu.yaml
  + gpu_rtx4080_16gb.yaml
  + {input_dir=/data/tiles, features.k_neighbors=80}
  = Final config
```

---

## 🎨 Parameter Naming Standards (v4.0)

### Standardization Rules

| Concept               | v3.1 (Old)        | v5.1 (YAML)                 | v4.0 (New)            | Rationale          |
| --------------------- | ----------------- | --------------------------- | --------------------- | ------------------ |
| Classification scheme | `lod_level`       | `processor.lod_level`       | **`mode`**            | Simpler, top-level |
| Feature set           | `features.mode`   | `features.mode`             | **`features.mode`**   | Keep as-is         |
| Output type           | `processing_mode` | `processor.processing_mode` | **`processing_mode`** | Top-level, clear   |

### Naming Conventions

- **Top-level:** Short, clear names (`mode`, `use_gpu`, `num_workers`)
- **Nested:** Context-specific (`features.mode`, `optimizations.enabled`)
- **Boolean flags:** Prefix with `use_`, `enable_`, `compute_`
- **Sizes:** Suffix with units (`patch_size` = meters, `_gb` = gigabytes)

---

## 📈 Migration Path Visualization

### Timeline

```
2024 Q4          2025 Q1          2025 Q2          2025 Q3
   │                │                │                │
   │  v3.1         │  v3.2          │  v3.9          │  v4.0
   │  3 configs    │  Config class  │  Deprecation   │  Unified
   │               │  introduced    │  warnings      │
   ▼               ▼                ▼                ▼
schema.py     config.py        migration.py      Single config
+ config      + schema.py      + warnings        system
+ YAML v5.1   + YAML v5.1      + tool

PARALLEL ─────────────────────► TRANSITION ─────► HARMONIZED
```

### User Migration Journey

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: User with v3.1 config                              │
│  ├── old_config.yaml (v3.1 nested structure)                │
│  └── Works with v3.1, v3.2, v3.9                            │
└─────────────────────────────────────────────────────────────┘
                    │
                    │ Install v3.9
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: See deprecation warnings                           │
│  ⚠️  "config.schema is deprecated, migrate to v4.0"         │
│  ⚠️  "Run: ign-lidar-hd migrate-config old_config.yaml"    │
└─────────────────────────────────────────────────────────────┘
                    │
                    │ Run migration tool
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Automated migration                                │
│  $ ign-lidar-hd migrate-config old_config.yaml              │
│  ✓ Detected: v3.1                                           │
│  ✓ Migrated to: v4.0                                        │
│  ✓ Saved: old_config.yaml.v4.yaml                          │
│  ✓ Validated: no errors                                     │
└─────────────────────────────────────────────────────────────┘
                    │
                    │ Test with v3.9
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Test migrated config (still v3.9)                  │
│  $ ign-lidar-hd process -c old_config.yaml.v4.yaml ...      │
│  ✓ Works with v3.9 (backward compatible)                    │
└─────────────────────────────────────────────────────────────┘
                    │
                    │ Upgrade to v4.0
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 5: Use v4.0 with migrated config                      │
│  $ pip install ign-lidar-hd==4.0.0                          │
│  $ ign-lidar-hd process -c old_config.yaml.v4.yaml ...      │
│  ✓ Clean, unified configuration system                      │
└─────────────────────────────────────────────────────────────┘
```

---

**Document Version:** 1.0  
**Last Updated:** November 28, 2025  
**Package Version:** 3.1.0 (Transitioning to 4.0.0)
