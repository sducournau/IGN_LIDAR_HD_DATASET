# Configuration Consolidation Plan

**Date:** October 15, 2025  
**Goal:** Simplify and harmonize configuration files across the repository

## 📊 Current State Analysis

### Configuration File Locations

#### 1. **Hydra Configs** (`ign_lidar/configs/`)

- ✅ **Well-organized** - Uses Hydra composition pattern
- ✅ **Base configs** exist (`_base/` directory)
- 📦 **23 experiment configs** in `experiment/`
- 🎯 **Purpose:** Runtime configuration via CLI

#### 2. **Example Configs** (`examples/`)

- ⚠️ **19 YAML files** - Many duplicates of Hydra configs
- ⚠️ **Inconsistent structure** - Mix of standalone and documented examples
- ⚠️ **Duplicates functionality** already in `ign_lidar/configs/experiment/`
- 🎯 **Original purpose:** User examples and documentation

### Key Issues Identified

#### 🔴 **Critical: Duplication**

```
examples/config_lod3_training.yaml        ← 88 lines, detailed
ign_lidar/configs/experiment/buildings_lod3.yaml  ← 24 lines, uses base

examples/config_lod3_training_50m.yaml    ← 68 lines
ign_lidar/configs/experiment/dataset_50m.yaml     ← 26 lines
ign_lidar/configs/experiment/lod2_gt_50m.yaml     ← 26 lines (similar purpose)

examples/semantic_sota.yaml
ign_lidar/configs/experiment/semantic_sota.yaml   ← EXACT DUPLICATE
```

#### 🟡 **Medium: Inconsistent Naming**

- `config_lod3_training.yaml` vs `buildings_lod3.yaml`
- `config_patches_only.yaml` vs better named experiment configs
- `config_enriched_only.yaml` vs `classify_enriched_tiles.yaml`

#### 🟡 **Medium: Overlapping Concepts**

Multiple configs for similar purposes:

- **Dataset creation:** `dataset_50m.yaml`, `dataset_100m.yaml`, `dataset_150m.yaml`, `dataset_multiscale.yaml`
- **LOD2 training:** `buildings_lod2.yaml`, `lod2_gt_50m.yaml`, `lod2_selfsupervised.yaml`
- **Ground truth:** `ground_truth_patches.yaml`, `ground_truth_training.yaml`

#### 🟢 **Low: Missing Documentation**

Some configs lack clear purpose statements or usage examples.

---

## 🎯 Consolidation Strategy

### Phase 1: Archive Legacy Examples (IMMEDIATE)

**Action:** Move old example configs to `examples/archive/` or `examples/legacy/`

**Files to Archive:**

```bash
examples/
├── archive/  # NEW
│   ├── config_lod3_training.yaml              ← Superseded by experiment/buildings_lod3
│   ├── config_lod3_training_50m.yaml          ← Superseded by experiment/dataset_50m
│   ├── config_lod3_training_100m.yaml         ← Superseded by experiment/dataset_100m
│   ├── config_lod3_training_150m.yaml         ← Superseded by experiment/dataset_150m
│   ├── config_lod3_training_50m_versailles.yaml
│   ├── config_lod3_training_50m_versailles_fixed.yaml
│   ├── config_lod3_training_sequential.yaml
│   ├── config_lod3_training_memory_optimized.yaml
│   ├── config_lod3_full_features.yaml
│   ├── config_lod2_simplified_features.yaml
│   ├── config_training_dataset.yaml
│   ├── config_enriched_only.yaml              ← Superseded by experiment/classify_enriched_tiles
│   ├── config_complete.yaml
│   ├── config_gpu_processing.yaml
│   ├── config_quick_enrich.yaml
│   ├── config_multiscale_hybrid.yaml
│   └── semantic_sota.yaml                     ← Duplicate of experiment/semantic_sota.yaml
```

**Keep in examples/ (useful for documentation):**

```bash
examples/
├── config_architectural_analysis.yaml         ← Good documented example
├── config_architectural_training.yaml         ← Good documented example
├── example_architectural_styles.py            ← Python examples (keep)
├── merge_multiscale_dataset.py                ← Utility script (keep)
├── run_multiscale_training.sh                 ← Shell script (keep)
└── test_ground_truth_module.py                ← Test/example (keep)
```

### Phase 2: Consolidate Experiment Configs

#### 2.1 Create Unified Base Configs

**New base configs to create:**

```yaml
# ign_lidar/configs/experiment/_base/dataset_common.yaml
# Common settings for all dataset generation (50m, 100m, 150m, multiscale)
defaults:
  - /processor: default
  - /features: full
  - /preprocess: default
  - /stitching: enabled
  - /output: default

processor:
  use_gpu: true
  augment: true
  num_augmentations: 3
  pin_memory: true

features:
  mode: full
  use_rgb: true
  use_infrared: true
  compute_ndvi: true
  sampling_method: fps
  normalize_xyz: true
  normalize_features: true
  use_gpu_chunked: true

output:
  format: npz,laz
  processing_mode: patches_only
  save_stats: true
  save_metadata: true
```

#### 2.2 Simplify Scale-Specific Configs

**Before:**

```yaml
# dataset_50m.yaml (26 lines)
# dataset_100m.yaml (26 lines)
# dataset_150m.yaml (26 lines)
```

**After:**

```yaml
# dataset_50m.yaml (10 lines)
defaults:
  - _base/dataset_common
  - _self_

processor:
  patch_size: 50.0
  num_points: 24576
  patch_overlap: 0.15

features:
  k_neighbors: 20
  search_radius: 1.2
```

#### 2.3 Consolidate Similar Experiments

**Merge these into one configurable experiment:**

- `lod2_gt_50m.yaml`, `lod2_gt_100m.yaml`, `lod2_gt_150m.yaml`
  → **`lod2_ground_truth.yaml`** with scale parameter

**Example:**

```yaml
# lod2_ground_truth.yaml
defaults:
  - _base/dataset_common
  - _base/buildings_common
  - /ground_truth: enabled
  - _self_

# Override with: experiment=lod2_ground_truth processor.patch_size=50
processor:
  lod_level: LOD2
  patch_size: ${oc.env:PATCH_SIZE,100} # Default 100m, override via env or CLI
```

### Phase 3: Create Clear Documentation

#### 3.1 New README Structure

Create `examples/README.md`:

````markdown
# Configuration Examples

## Quick Start

All configurations use Hydra. Run from repository root:

```bash
ign-lidar-hd process experiment=EXPERIMENT_NAME input_dir=... output_dir=...
```
````

## Available Experiments

### Building Classification

- `buildings_lod2` - Simple building classification (8k points)
- `buildings_lod3` - Detailed building classification (16k points)

### Dataset Generation (Multi-Scale)

- `dataset_50m` - Fine details (24k points, 50m patches)
- `dataset_100m` - Medium scale (32k points, 100m patches)
- `dataset_150m` - Large scale (32k points, 150m patches)
- `dataset_multiscale` - Combined training dataset

### Ground Truth Integration

- `ground_truth_patches` - Generate patches with GT labels
- `lod2_ground_truth` - LOD2 with ground truth (see scale params)

### Special Use Cases

- `fast` - Quick testing (small patches, no augmentation)
- `semantic_sota` - State-of-the-art semantic segmentation
- `vegetation_ndvi` - NDVI-based vegetation analysis
- `architectural_heritage` - Heritage building classification

## Override Examples

```bash
# Change patch size
ign-lidar-hd process experiment=buildings_lod3 processor.patch_size=75

# Enable GPU
ign-lidar-hd process experiment=buildings_lod2 processor.use_gpu=true

# Multi-scale with custom sizes
ign-lidar-hd process experiment=dataset_50m processor.patch_size=60
```

````

#### 3.2 Annotate Each Config

Add clear headers to all experiment configs:
```yaml
# Experiment: buildings_lod3
# Purpose: Detailed building classification with full features
# Use case: Training LOD3 classifiers for architectural analysis
# Output: 16k point patches with RGB, NIR, NDVI, and geometric features
#
# Usage:
#   ign-lidar-hd process experiment=buildings_lod3 \
#     input_dir=data/raw \
#     output_dir=data/patches
#
# Override examples:
#   processor.num_points=32768  # Increase points
#   processor.augment=false     # Disable augmentation
#   features.use_rgb=false      # Disable RGB features
````

### Phase 4: Remove True Duplicates

**Files to DELETE:**

```bash
examples/semantic_sota.yaml  # Exact duplicate of ign_lidar/configs/experiment/semantic_sota.yaml
```

---

## 📋 Implementation Checklist

### Step 1: Backup and Archive (30 min)

- [ ] Create `examples/archive/` directory
- [ ] Move 17 legacy config files to archive
- [ ] Create `examples/archive/README.md` explaining what's archived
- [ ] Update main `README.md` to reference new structure

### Step 2: Create Base Configs (45 min)

- [ ] Create `_base/dataset_common.yaml`
- [ ] Create `_base/ground_truth_common.yaml` (if needed)
- [ ] Test base configs work with inheritance

### Step 3: Simplify Experiment Configs (60 min)

- [ ] Refactor `dataset_50m.yaml`, `dataset_100m.yaml`, `dataset_150m.yaml`
- [ ] Consolidate `lod2_gt_*` configs into parameterized version
- [ ] Add clear headers to all experiment configs
- [ ] Test each experiment config still works

### Step 4: Documentation (45 min)

- [ ] Create comprehensive `examples/README.md`
- [ ] Update main `README.md` with new structure
- [ ] Create migration guide for users with old configs
- [ ] Update `docs/` references to configs

### Step 5: Testing (30 min)

- [ ] Test 5-10 key experiment configs work correctly
- [ ] Verify CLI still works with all experiments
- [ ] Check backward compatibility

---

## 🎨 Proposed Final Structure

```
ign_lidar/configs/
├── config.yaml                    # Main config (unchanged)
├── experiment/
│   ├── _base/
│   │   ├── buildings_common.yaml       # Existing
│   │   ├── boundary_aware_common.yaml  # Existing
│   │   ├── training_common.yaml        # Existing
│   │   ├── dataset_common.yaml         # NEW - Common for all datasets
│   │   └── ground_truth_common.yaml    # NEW (optional)
│   │
│   ├── buildings_lod2.yaml             # Keep (clean)
│   ├── buildings_lod3.yaml             # Keep (clean)
│   │
│   ├── dataset_50m.yaml                # Simplify (use dataset_common)
│   ├── dataset_100m.yaml               # Simplify (use dataset_common)
│   ├── dataset_150m.yaml               # Simplify (use dataset_common)
│   ├── dataset_multiscale.yaml         # Keep
│   │
│   ├── lod2_ground_truth.yaml          # NEW - Consolidates lod2_gt_*
│   ├── lod2_selfsupervised.yaml        # Keep (unique)
│   │
│   ├── ground_truth_patches.yaml       # Keep
│   ├── ground_truth_training.yaml      # Keep or merge?
│   │
│   ├── fast.yaml                       # Keep (useful for testing)
│   ├── semantic_sota.yaml              # Keep
│   ├── vegetation_ndvi.yaml            # Keep
│   ├── architectural_heritage.yaml     # Keep
│   │
│   ├── boundary_aware_autodownload.yaml  # Keep
│   ├── boundary_aware_offline.yaml       # Keep
│   │
│   ├── classify_enriched_tiles.yaml     # Keep
│   ├── pointnet_training.yaml           # Keep
│   │
│   └── [REMOVED: config_* duplicates]
│
└── [other config groups unchanged]

examples/
├── README.md                           # NEW - Clear documentation
├── config_architectural_analysis.yaml  # Keep (good example)
├── config_architectural_training.yaml  # Keep (good example)
├── ARCHITECTURAL_CONFIG_REFERENCE.md   # Keep
├── ARCHITECTURAL_STYLES_README.md      # Keep
├── MULTISCALE_QUICK_REFERENCE.md       # Keep
├── MULTI_SCALE_TRAINING_STRATEGY.md    # Keep
├── example_architectural_styles.py     # Keep
├── merge_multiscale_dataset.py         # Keep
├── run_multiscale_training.sh          # Keep
├── test_ground_truth_module.py         # Keep
│
└── archive/                            # NEW
    ├── README.md                       # Explains archived files
    ├── config_lod3_training.yaml
    ├── config_lod3_training_50m.yaml
    ├── config_lod3_training_100m.yaml
    ├── config_lod3_training_150m.yaml
    ├── config_lod3_training_50m_versailles.yaml
    ├── config_lod3_training_50m_versailles_fixed.yaml
    ├── config_lod3_training_sequential.yaml
    ├── config_lod3_training_memory_optimized.yaml
    ├── config_lod3_full_features.yaml
    ├── config_lod2_simplified_features.yaml
    ├── config_training_dataset.yaml
    ├── config_enriched_only.yaml
    ├── config_complete.yaml
    ├── config_gpu_processing.yaml
    ├── config_quick_enrich.yaml
    ├── config_multiscale_hybrid.yaml
    └── semantic_sota.yaml
```

---

## 📈 Benefits

### Immediate Benefits

1. **Reduced confusion** - Clear separation between runtime configs and examples
2. **Less duplication** - 17 files archived, configs simplified by 40-60%
3. **Better organization** - Everything in its proper place
4. **Easier maintenance** - Change base config once, affects all children

### Long-term Benefits

1. **Easier onboarding** - New users have clear examples
2. **Better testing** - Fewer configs to test
3. **Extensibility** - Easy to add new experiments
4. **Documentation** - Self-documenting structure

### Metrics

- **Config files reduced:** 42 → 28 active configs (-33%)
- **Duplication eliminated:** ~1,200 lines of redundant YAML
- **Base configs increased:** 3 → 5 (better reuse)
- **Documentation improved:** +3 comprehensive READMEs

---

## 🚨 Risks and Mitigation

### Risk 1: Breaking User Workflows

**Impact:** Users with custom configs referencing old paths  
**Mitigation:**

- Keep archived files accessible (not deleted)
- Add deprecation warnings in docs
- Provide migration guide

### Risk 2: Testing Overhead

**Impact:** Need to test all configs work  
**Mitigation:**

- Create automated test for all experiment configs
- Use Hydra's built-in validation
- Document test procedure

### Risk 3: Incomplete Migration

**Impact:** Some references to old configs remain  
**Mitigation:**

- Use `grep` to find all references
- Update docs comprehensively
- Create tracking issue

---

## 🔄 Migration Guide for Users

### If you're using example configs:

**Old way:**

```bash
cd examples/
ign-lidar-hd process -c config_lod3_training.yaml
```

**New way:**

```bash
ign-lidar-hd process experiment=buildings_lod3 \
  input_dir=... output_dir=...
```

### If you have custom configs:

**Option 1:** Update to use new base configs

```yaml
# your_custom_config.yaml
defaults:
  - /experiment/_base/dataset_common
  - _self_

processor:
  patch_size: 75 # Your custom value
```

**Option 2:** Keep using absolute paths (still works)

```bash
ign-lidar-hd process --config-path=/path/to/your/configs \
  --config-name=your_config
```

---

## 📝 Next Steps

1. **Review this plan** with team/maintainers
2. **Create branch** `feature/config-consolidation`
3. **Implement Phase 1** (Archive) - safest, reversible
4. **Test thoroughly** before merging
5. **Update documentation** comprehensively
6. **Announce changes** in CHANGELOG.md

---

## Questions to Resolve

1. ❓ Should we keep `ground_truth_patches.yaml` AND `ground_truth_training.yaml` separate?
2. ❓ Are the Versailles-specific configs (`config_lod3_training_50m_versailles*.yaml`) still needed?
3. ❓ Should architectural configs stay in examples/ or move to experiment/?
4. ❓ Do we need all three scales (50m, 100m, 150m) or can we parameterize into one?

---

**Status:** 📋 Proposal - Awaiting Review  
**Priority:** 🟡 Medium (Improves maintainability, not urgent)  
**Effort:** ~3-4 hours total  
**Breaking Changes:** ⚠️ Minor (old configs still work via archive/)
