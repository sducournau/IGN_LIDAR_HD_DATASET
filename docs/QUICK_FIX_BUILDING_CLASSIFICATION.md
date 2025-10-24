# 🚀 Quick Fix Guide - Building Classification Issue (CORRECTED)

**Problem CORRECTED:** High unclassified rate (~30-40% white points). Buildings ARE detected (light green) but coverage incomplete.

**Solution:** Use V2 pre-configured file with more aggressive thresholds.

**Solution Time:** 10-20 minutes reprocessing + validation

---

## ⚡ FASTEST Solution: Use V2 Pre-Fixed Config

### Step 1: Use V2 Configuration (Already Applied!)

The file `examples/config_asprs_bdtopo_cadastre_cpu_fixed.yaml` already has **ALL V2 fixes** applied:

✅ 12 parameters made more aggressive to reduce unclassified rate

### Step 2: Reprocess (10-20 minutes)

```bash
cd /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET

# Reprocess your tile with V2 fixes
ign-lidar-hd process \
  -c examples/config_asprs_bdtopo_cadastre_cpu_fixed.yaml \
  input_dir="path/to/your/tile" \
  output_dir="output/v2_fixed"
```

### Step 3: Validate Results (2 minutes)

```bash
# Run diagnostic
python scripts/diagnose_classification.py output/v2_fixed/tile_enriched.laz

# Create visualization
python scripts/visualize_classification.py output/v2_fixed/tile_enriched.laz v2_result.png
```

---

## 📋 V2 Fixes Applied in config_asprs_bdtopo_cadastre_cpu_fixed.yaml

### Section: adaptive_building_classification

| Parameter                        | Original | V2 Value | Impact      |
| -------------------------------- | -------- | -------- | ----------- |
| `min_classification_confidence`  | 0.55     | **0.40** | 🔴 Critical |
| `expansion_confidence_threshold` | 0.65     | **0.50** | 🔴 Critical |
| `rejection_confidence_threshold` | 0.45     | **0.35** | 🟡 High     |
| `roof_planarity_min`             | 0.70     | **0.60** | 🟡 High     |
| `roof_curvature_max`             | 0.10     | **0.20** | 🟡 High     |
| `wall_verticality_min`           | 0.60     | **0.55** | 🟢 Medium   |
| `min_cluster_size`               | 8        | **5**    | 🟢 Medium   |

### Section: reclassification

| Parameter                  | Original | V2 Value | Impact      |
| -------------------------- | -------- | -------- | ----------- |
| `min_confidence`           | 0.75     | **0.50** | 🔴 Critical |
| `spatial_cluster_eps`      | 0.4      | **0.5**  | 🟡 High     |
| `min_cluster_size`         | 8        | **5**    | 🟡 High     |
| `building_buffer_distance` | 3.5      | **5.0**  | 🟡 High     |
| `verticality_threshold`    | 0.65     | **0.55** | 🟢 Medium   |

**Total: 12 parameters optimized to reduce unclassified rate**

### Change 1: Line ~150 - Building Fusion

Find:

```yaml
building_fusion:
  enable_translation: true
  max_translation: 6.0
  enable_scaling: true
  max_scale_factor: 1.8
  enable_rotation: false # ← CHANGE THIS
```

Replace with:

```yaml
building_fusion:
  enable_translation: true
  max_translation: 12.0 # ⬆️ Increased from 6.0
  enable_scaling: true
  max_scale_factor: 2.5 # ⬆️ Increased from 1.8
  enable_rotation: true # ⬆️ ENABLED
  max_rotation: 45.0
```

### Change 2: Line ~200 - Adaptive Building Classification

Find:

```yaml
adaptive_building_classification:
  enabled: true
  signature:
    min_height: 1.2
    typical_height_range: [2.0, 60.0]
    wall_verticality_min: 0.60
    wall_planarity_max: 0.65
    roof_planarity_min: 0.70 # ← CHANGE THIS
    roof_curvature_max: 0.10 # ← CHANGE THIS
    roof_normal_z_range: [0.25, 1.0] # ← CHANGE THIS
```

Replace with:

```yaml
adaptive_building_classification:
  enabled: true
  signature:
    min_height: 1.2
    typical_height_range: [2.0, 60.0]
    wall_verticality_min: 0.60
    wall_planarity_max: 0.65
    roof_planarity_min: 0.65 # ⬇️ Reduced from 0.70
    roof_curvature_max: 0.15 # ⬆️ Increased from 0.10
    roof_normal_z_range: [0.15, 1.0] # ⬇️ Lowered from [0.25, 1.0]
```

### Change 3: Line ~210 - Fuzzy Boundaries

Find:

```yaml
fuzzy_boundary_inner: 0.0
fuzzy_boundary_outer: 2.5 # ← CHANGE THIS
fuzzy_decay_function: "gaussian"
enable_adaptive_expansion: true
max_expansion_distance: 3.5 # ← CHANGE THIS
```

Replace with:

```yaml
fuzzy_boundary_inner: 0.0
fuzzy_boundary_outer: 5.0 # ⬆️ Increased from 2.5
fuzzy_decay_function: "gaussian"
enable_adaptive_expansion: true
max_expansion_distance: 6.0 # ⬆️ Increased from 3.5
```

### Change 4: Line ~225 - Confidence Thresholds

Find:

```yaml
expansion_confidence_threshold: 0.65 # ← CHANGE THIS
enable_intelligent_rejection: true
rejection_confidence_threshold: 0.45
enable_spatial_clustering: true
spatial_radius: 2.5
min_neighbor_ratio: 0.25
min_classification_confidence: 0.55 # ← CHANGE THIS
```

Replace with:

```yaml
expansion_confidence_threshold: 0.55 # ⬇️ Reduced from 0.65
enable_intelligent_rejection: true
rejection_confidence_threshold: 0.45
enable_spatial_clustering: true
spatial_radius: 2.5
min_neighbor_ratio: 0.25
min_classification_confidence: 0.45 # ⬇️ Reduced from 0.55
```

### Change 5: Line ~235 - Feature Weights

Find:

```yaml
feature_weights:
  height: 0.22 # ← CHANGE THIS
  geometry: 0.35 # ← CHANGE THIS
  spectral: 0.18 # ← CHANGE THIS
  spatial: 0.15
  ground_truth: 0.10 # ← CHANGE THIS
```

Replace with:

```yaml
feature_weights:
  height: 0.25 # ⬆️ Increased from 0.22
  geometry: 0.30 # ⬇️ Reduced from 0.35
  spectral: 0.15 # ⬇️ Reduced from 0.18
  spatial: 0.15
  ground_truth: 0.15 # ⬆️ Increased from 0.10
```

### Change 6: Line ~300 - RGE ALTI DTM

Find:

```yaml
rge_alti:
  enabled: true
  use_wcs: true
  # ...
  augment_ground_points: true
  augmentation_spacing: 3.0 # ← CHANGE THIS
  augmentation_areas:
    - "vegetation"
    - "gaps"
    # ← ADD "buildings" HERE
```

Replace with:

```yaml
rge_alti:
  enabled: true
  use_wcs: true
  # ...
  augment_ground_points: true
  augmentation_spacing: 2.0 # ⬇️ Reduced from 3.0
  augmentation_areas:
    - "vegetation"
    - "gaps"
    - "buildings" # ⬆️ ADDED
```

### Change 7: Line ~310 - DTM Spacing

Find:

```yaml
min_spacing_to_existing: 2.0 # ← CHANGE THIS
```

Replace with:

```yaml
min_spacing_to_existing: 1.5 # ⬇️ Reduced from 2.0
```

---

## Step 3: Save and Reprocess (10-20 minutes)

```bash
# Reprocess your problematic tile
ign-lidar-hd process \
  -c examples/config_asprs_bdtopo_cadastre_cpu_optimized.yaml \
  input_dir="path/to/your/tile" \
  output_dir="output/fixed"
```

---

## Step 4: Validate Results (2 minutes)

```bash
# Run diagnostic
python scripts/diagnose_classification.py output/fixed/tile_enriched.laz

# Create visualization
python scripts/visualize_classification.py output/fixed/tile_enriched.laz fixed_result.png
```

---

## Expected Results

### Before Fixes:

- 🔴 Building classification: <5% of points
- 🔴 Many elevated points unclassified
- 🔴 Buildings appear pink/magenta in visualization

### After Fixes:

- ✅ Building classification: >10-15% of points
- ✅ Most elevated points classified correctly
- ✅ Buildings appear **RED** in visualization

---

## If Still Not Working

1. Run diagnostic script:

   ```bash
   python scripts/diagnose_classification.py output/fixed/tile_enriched.laz > diagnostic_report.txt
   ```

2. Share the diagnostic report for further analysis

3. Check if ground truth data (BD TOPO) is available for your area:
   ```bash
   # Check logs for messages like:
   # "No buildings found in BD TOPO for this area"
   # "Failed to fetch BD TOPO data"
   ```

---

## Summary of Changes

| Setting                          | Before                   | After                    | Reason                        |
| -------------------------------- | ------------------------ | ------------------------ | ----------------------------- |
| `max_translation`                | 6.0m                     | 12.0m                    | Handle polygon misalignment   |
| `enable_rotation`                | false                    | true                     | Allow rotated polygons        |
| `max_scale_factor`               | 1.8                      | 2.5                      | Allow larger size differences |
| `fuzzy_boundary_outer`           | 2.5m                     | 5.0m                     | Wider fuzzy zone              |
| `max_expansion_distance`         | 3.5m                     | 6.0m                     | Allow more expansion          |
| `min_classification_confidence`  | 0.55                     | 0.45                     | Less strict threshold         |
| `expansion_confidence_threshold` | 0.65                     | 0.55                     | Allow more expansion          |
| `roof_planarity_min`             | 0.70                     | 0.65                     | Accept complex roofs          |
| `roof_curvature_max`             | 0.10                     | 0.15                     | Accept curved roofs           |
| `roof_normal_z_range`            | [0.25,1.0]               | [0.15,1.0]               | Include steep roofs           |
| `augmentation_spacing`           | 3.0m                     | 2.0m                     | Denser DTM grid               |
| `augmentation_areas`             | veg, gaps                | +buildings               | Better building HAG           |
| `min_spacing_to_existing`        | 2.0m                     | 1.5m                     | More DTM points               |
| Feature weights                  | 0.22/0.35/0.18/0.15/0.10 | 0.25/0.30/0.15/0.15/0.15 | Better balance                |

---

**Total time:** ~15-30 minutes (5 min config + 10-20 min processing + 2 min validation)

**Expected improvement:** Building detection rate should increase from ~30-50% to >80%
