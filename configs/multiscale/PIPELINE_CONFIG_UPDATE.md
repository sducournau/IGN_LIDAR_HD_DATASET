# Pipeline Configuration Update Summary

**Date:** October 15, 2025

## Overview

Updated the multi-scale preprocessing pipeline configurations to optimize feature extraction and output formats for each LOD level.

## Changes Made

### 1. ASPRS Configuration (`config_unified_asprs_preprocessing.yaml`)

**Purpose:** Base classification with minimal features, enriched LAZ output only

#### Key Changes:

- ✅ **Features:** Core minimal features only
  - `k_neighbors: 20` (reduced from 30)
  - `include_extra: false` (minimal features)
  - Core geometric: normals only (no planarity, no curvature)
- ✅ **Spectral Features:**
  - RGB: Enabled with augmentations
    - Brightness range: [0.8, 1.2]
    - Contrast range: [0.8, 1.2]
    - Saturation range: [0.8, 1.2]
  - Infrared: Enabled with augmentations
    - Intensity range: [0.9, 1.1]
  - NDVI: Computed
- ✅ **Output:**

  - Format: LAZ (enriched tiles only)
  - No patches generated
  - `patch_size: null`
  - `augment: false`

- ✅ **Defaults:**
  - Changed from `features: full` to `features: minimal`

---

### 2. LOD2 Configuration (`config_unified_lod2_preprocessing.yaml`)

**Purpose:** Building classification with full features, patches only

#### Key Changes:

- ✅ **Processor:**

  - **Patches enabled** (changed from null)
  - `patch_size: 100.0` meters
  - `patch_overlap: 20.0` meters
  - `num_points: 16384` points per patch
  - `augment: true`
  - `num_augmentations: 5`

- ✅ **Features:**

  - Full features retained
  - `k_neighbors: 40` (for building details)
  - Building-specific features enabled:
    - `compute_roof_angles: true`
    - `compute_wall_normals: true`
    - `compute_height_relative: true`

- ✅ **Output:**
  - Format: **NPZ + LAZ** (training patches in both formats)
  - Processing mode: **patches_only** (changed from enriched_only)
  - `save_laz_patches: true` (patches saved as LAZ files)
  - No enriched tile LAZ output
- ✅ **Paths:**
  - Output directory: `/mnt/c/Users/Simon/ign/preprocessed/lod2/patches`
  - (changed from `enriched_tiles`)

---

### 3. LOD3 Configuration (`config_unified_lod3_preprocessing.yaml`)

**Purpose:** Detailed architectural classification with full features, patches only

#### Status:

- ✅ Already configured correctly for patches only
- ✅ Full features with high resolution
  - `k_neighbors: 50` (fine details)
  - `search_radius: 0.5` (local features)
  - Enhanced architectural features enabled
- ✅ **Processor:**

  - `patch_size: 50.0` meters (smaller for details)
  - `patch_overlap: 10.0` meters
  - `num_points: 16384`
  - `augment: true`
  - `num_augmentations: 4`

- ✅ **Output:**
  - Format: **NPZ + LAZ** (training patches in both formats)
  - Processing mode: patches_only
  - `save_laz_patches: true` (patches saved as LAZ files)
  - No enriched tile LAZ output

---

## Configuration Summary Table

| LOD Level | Features | RGB/IR Aug | NDVI | Output Format | Patches | Enriched LAZ |
| --------- | -------- | ---------- | ---- | ------------- | ------- | ------------ |
| **ASPRS** | Minimal  | ✅ Yes     | ✅   | LAZ           | ❌ No   | ✅ Yes       |
| **LOD2**  | Full     | ✅ Yes     | ✅   | NPZ + LAZ     | ✅ Yes  | ❌ No        |
| **LOD3**  | Full     | ✅ Yes     | ✅   | NPZ + LAZ     | ✅ Yes  | ❌ No        |

---

## Pipeline Flow

```
ASPRS (Base Classification)
├── Input: Selected tiles from unified_dataset
├── Features: Minimal core + RGB/IR/NDVI with augmentations
├── Output: Enriched LAZ tiles only
└── Purpose: Pre-classified tiles for further processing

LOD2 (Building Classification)
├── Input: Selected tiles from unified_dataset
├── Features: Full (k=40) + building-specific
├── Output: Training patches (NPZ) only
├── Patch size: 100m x 100m (20m overlap)
└── Purpose: Training data for building structure models

LOD3 (Architectural Details)
├── Input: Selected tiles from unified_dataset
├── Features: Full (k=50) + architectural-specific
├── Output: Training patches (NPZ) only
├── Patch size: 50m x 50m (10m overlap)
└── Purpose: Training data for detailed architectural models
```

---

## Usage

### ASPRS Preprocessing (Enriched LAZ)

```bash
ign-lidar-hd process \
    --config-file configs/multiscale/config_unified_asprs_preprocessing.yaml
```

**Output:** `/mnt/c/Users/Simon/ign/preprocessed/asprs/enriched_tiles/*.laz`

### LOD2 Preprocessing (Patches)

```bash
ign-lidar-hd process \
    --config-file configs/multiscale/config_unified_lod2_preprocessing.yaml
```

**Output:**

- `/mnt/c/Users/Simon/ign/preprocessed/lod2/patches/*.npz` (training data)
- `/mnt/c/Users/Simon/ign/preprocessed/lod2/patches/*.laz` (visualization)

### LOD3 Preprocessing (Patches)

```bash
ign-lidar-hd process \
    --config-file configs/multiscale/config_unified_lod3_preprocessing.yaml
```

**Output:**

- `/mnt/c/Users/Simon/ign/preprocessed/lod3/patches/*.npz` (training data)
- `/mnt/c/Users/Simon/ign/preprocessed/lod3/patches/*.laz` (visualization)

---

## Benefits

### ASPRS Configuration

1. **Faster processing** - Minimal features reduce computation time
2. **RGB/IR augmentations** - Better spectral feature learning
3. **Enriched LAZ output** - Can be used for visualization and further analysis
4. **No patches** - Saves disk space, patches generated later if needed

### LOD2/LOD3 Configurations

1. **Direct patch generation** - Ready for training immediately
2. **Full features** - Maximum information for complex classification
3. **Dual format output** - NPZ for training, LAZ for visualization/inspection
4. **Scale-appropriate patches** - 100m for LOD2, 50m for LOD3
5. **LAZ patches** - Enables visual inspection and quality control of training data

---

## Next Steps

1. **Run ASPRS preprocessing** to generate enriched LAZ tiles
2. **Run LOD2 preprocessing** to generate building patches
3. **Run LOD3 preprocessing** to generate architectural detail patches
4. **Merge multi-scale datasets** for each LOD level
5. **Train models** using the generated patches

---

## Notes

- All configurations use GPU acceleration when available
- Tile stitching enabled for boundary continuity
- Ground truth from IGN BD TOPO® integrated
- Cache directories configured for neighbors and ground truth
- Statistical outlier removal optimized per LOD level
