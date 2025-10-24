# 🔍 Classification Quality Audit Report - October 2025

**Date:** October 24, 2025  
**Analyst:** GitHub Copilot  
**Status:** 🔴 **CRITICAL ISSUES IDENTIFIED**  
**Scope:** Comprehensive audit of building classification quality

---

## 🚨 Executive Summary

### Critical Finding: Poor Building Classification

**Visual Evidence:** Submitted image shows:

- ❌ Buildings (light gray/white areas) are **NOT** being classified as Class 6 (Building)
- ❌ Many building roofs appear classified as **unclassified** or other classes
- ❌ Building points showing as pink/magenta (likely Class 1 - Unclassified or vegetation)
- ✅ Vegetation (green) appears correctly classified
- ✅ Ground (beige/tan) appears correctly classified

**Impact:** 🔴 **SEVERE**

- Building detection rate appears **<50%** based on visual inspection
- Ground truth polygons not effectively guiding classification
- ASPRS Class 6 (Building) underrepresented in output

---

## 📊 Classification Quality Analysis

### Expected vs. Observed Behavior

| Feature                   | Expected             | Observed         | Status     |
| ------------------------- | -------------------- | ---------------- | ---------- |
| Building roofs (flat)     | Class 6              | Class 1 or 3/4/5 | ❌ FAIL    |
| Building walls (vertical) | Class 6              | Class 1          | ❌ FAIL    |
| Building polygons         | Guide classification | Not effective    | ❌ FAIL    |
| Vegetation                | Class 3/4/5          | Class 3/4/5      | ✅ PASS    |
| Ground                    | Class 2              | Class 2          | ✅ PASS    |
| Roads                     | Class 11             | Unknown          | ⚠️ UNCLEAR |

### Severity Assessment

```
Building Classification: 🔴 CRITICAL FAILURE
├── Detection Rate: ~30-50% (should be >90%)
├── False Negatives: HIGH (buildings missed)
├── False Positives: LOW (non-buildings classified)
└── Impact: Pipeline unusable for building applications
```

---

## 🔬 Root Cause Analysis

### 1. Ground Truth Polygon Issues ⚠️ **HIGH PROBABILITY**

**Problem:** BD TOPO® polygons may be:

- ❌ **Misaligned** - Offset from actual building footprints (common issue)
- ❌ **Outdated** - Buildings changed since polygon creation
- ❌ **Incomplete** - Missing extensions, annexes, or building parts
- ❌ **Oversized** - Include surrounding terrain
- ❌ **Undersized** - Miss building edges/walls

**Evidence from Code:**

```python
# From adaptive.py - fuzzy_boundary_outer default is 2.5m
fuzzy_boundary_outer: float = 2.5  # May be insufficient for severe misalignment
```

**Config Settings:**

```yaml
# From config_asprs_bdtopo_cadastre_cpu_optimized.yaml
building_fusion:
  enable_translation: true
  max_translation: 6.0 # May need increase
  enable_scaling: true
  max_scale_factor: 1.8
  enable_rotation: false # DISABLED - Could help alignment!
```

**Recommended Fix:**

```yaml
building_fusion:
  max_translation: 12.0 # Increase from 6.0m
  max_scale_factor: 2.2 # Increase from 1.8
  enable_rotation: true # ENABLE for misaligned polygons
  max_rotation: 45.0 # Allow significant rotation

adaptive_building_classification:
  fuzzy_boundary_outer: 5.0 # Increase from 2.5m
  max_expansion_distance: 6.0 # Increase from 3.5m
```

---

### 2. Feature Computation Issues ⚠️ **MEDIUM PROBABILITY**

**Problem:** Features may not be computed correctly or are missing critical data

**Potential Issues:**

#### A. Height Above Ground (HAG) - **CRITICAL**

```python
# Config shows RGE ALTI is enabled but may not be working
rge_alti:
  enabled: true
  augment_ground_points: true
  augmentation_spacing: 3.0  # Sparse grid - may miss elevation changes
```

**Symptoms:**

- If HAG is incorrect, buildings appear at ground level
- Minimum height threshold not met → points rejected

**Diagnostic Check:**

```python
# Check if HAG is being computed
if 'height_above_ground' not in features:
    print("❌ HAG missing - buildings will fail height check")
if np.mean(features['height_above_ground'][building_points]) < 2.5:
    print("❌ Buildings below minimum height threshold")
```

**Recommended Fix:**

```yaml
features:
  height_method: "hybrid" # ✅ Already set
  use_rge_alti_for_height: true # ✅ Already set
  compute_height_above_ground: true # ✅ Already set

rge_alti:
  augmentation_spacing: 2.0 # REDUCE from 3.0m for denser DTM
  augmentation_areas:
    - "vegetation"
    - "gaps"
    - "buildings" # ADD THIS - critical for building HAG
```

---

#### B. Geometric Features - **MEDIUM**

**Required Features:**

- ✅ `planarity` - Roof detection (should be high >0.75)
- ✅ `verticality` - Wall detection (should be high >0.65)
- ✅ `curvature` - Roof quality (should be low <0.08)
- ✅ `normals` - Surface orientation

**Potential Issues:**

```yaml
features:
  k_neighbors: 50 # CPU config uses 50 (vs 55-60 in GPU)
  # Too low? May cause noisy geometric features
```

**Recommended Fix:**

```yaml
features:
  k_neighbors: 60 # Increase for smoother features
  search_radius: 2.0 # Increase from 1.5m
```

---

#### C. NDVI/Spectral Features - **LOW-MEDIUM**

**Current Thresholds:**

```yaml
adaptive_building_classification:
  signature:
    ndvi_max: 0.28 # Buildings should have NDVI < 0.28
    ndvi_wall_max: 0.22 # Walls even lower
```

**Potential Issue:**

- If NDVI not computed correctly → buildings rejected as vegetation
- If orthophotos unavailable → NDVI missing

**Diagnostic Check:**

```python
if 'ndvi' not in features:
    print("⚠️ NDVI missing - spectral filtering disabled")
building_ndvi = features['ndvi'][building_mask]
if np.mean(building_ndvi) > 0.28:
    print(f"⚠️ Buildings have high NDVI ({np.mean(building_ndvi):.2f}) - vegetation confusion")
```

---

### 3. Classification Logic Issues ⚠️ **MEDIUM PROBABILITY**

**Problem:** Classification thresholds too strict or confidence scoring wrong

**Current Thresholds:**

```yaml
adaptive_building_classification:
  min_classification_confidence: 0.55 # May be too high
  expansion_confidence_threshold: 0.65 # Too strict?
  rejection_confidence_threshold: 0.45

  signature:
    min_height: 1.2 # Good
    wall_verticality_min: 0.60 # Good
    roof_planarity_min: 0.70 # May be too high
    roof_curvature_max: 0.10 # May be too strict
```

**Recommended Fix:**

```yaml
adaptive_building_classification:
  min_classification_confidence: 0.45 # REDUCE from 0.55
  expansion_confidence_threshold: 0.55 # REDUCE from 0.65

  signature:
    roof_planarity_min: 0.65 # REDUCE from 0.70
    roof_curvature_max: 0.15 # INCREASE from 0.10 (allow more complex roofs)
    roof_normal_z_range: [0.15, 1.0] # EXPAND from [0.25, 1.0] for inclined roofs
```

---

### 4. Feature Weight Balance ⚠️ **LOW-MEDIUM PROBABILITY**

**Current Weights:**

```yaml
feature_weights:
  height: 0.22 # 22%
  geometry: 0.35 # 35% - HIGHEST
  spectral: 0.18 # 18%
  spatial: 0.15 # 15%
  ground_truth: 0.10 # 10% - TOO LOW?
```

**Analysis:**

- Ground truth weight only 10% → Polygons not influential enough
- Geometry weight 35% → May be too strict if features noisy

**Recommended Fix:**

```yaml
feature_weights:
  height: 0.25 # INCREASE - critical discriminator
  geometry: 0.30 # REDUCE - too dominant
  spectral: 0.15 # REDUCE slightly
  spatial: 0.15 # Keep
  ground_truth: 0.15 # INCREASE - trust polygons more
```

---

### 5. Processing Pipeline Order ⚠️ **LOW PROBABILITY**

**Potential Issue:** Classification may run before proper feature enrichment

**Check Pipeline Order:**

```python
# Expected order:
1. Load point cloud
2. Compute features (normals, HAG, planarity, verticality, NDVI)
3. Fetch ground truth polygons (BD TOPO, Cadastre, OSM)
4. Run building fusion (optimize polygon fit)
5. Run adaptive classification (with fuzzy boundaries)
6. Run reclassification refinement
7. Save enriched point cloud
```

**Diagnostic:** Check processor logs for order of operations

---

## 🎯 Recommended Fixes (Priority Order)

### 🔴 **CRITICAL - Implement Immediately**

#### 1. Increase Polygon Adaptation Tolerance

```yaml
# In config_asprs_bdtopo_cadastre_*.yaml
building_fusion:
  enable_translation: true
  max_translation: 12.0 # ⬆️ from 6.0m
  enable_scaling: true
  max_scale_factor: 2.5 # ⬆️ from 1.8
  enable_rotation: true # ⬆️ ENABLE (was false)
  max_rotation: 45.0 # ⬆️ Allow significant misalignment

adaptive_building_classification:
  fuzzy_boundary_outer: 5.0 # ⬆️ from 2.5m
  max_expansion_distance: 6.0 # ⬆️ from 3.5m
```

**Impact:** ✅ Handle severely misaligned polygons  
**Effort:** 5 minutes (config change)  
**Risk:** Low (increases tolerance, can't hurt)

---

#### 2. Improve DTM Ground Truth for Better HAG

```yaml
rge_alti:
  enabled: true
  augment_ground_points: true
  augmentation_spacing: 2.0 # ⬇️ from 3.0m (denser grid)
  augmentation_areas:
    - "vegetation"
    - "gaps"
    - "buildings" # ⬆️ ADD THIS
  min_spacing_to_existing: 1.5 # ⬇️ from 2.0m
```

**Impact:** ✅ Correct height above ground computation  
**Effort:** 5 minutes (config change)  
**Risk:** Low (more accurate HAG)

---

#### 3. Relax Classification Thresholds

```yaml
adaptive_building_classification:
  min_classification_confidence: 0.45 # ⬇️ from 0.55
  expansion_confidence_threshold: 0.55 # ⬇️ from 0.65

  signature:
    roof_planarity_min: 0.65 # ⬇️ from 0.70
    roof_curvature_max: 0.15 # ⬆️ from 0.10
    roof_normal_z_range: [0.15, 1.0] # ⬆️ from [0.25, 1.0]
```

**Impact:** ✅ Allow more points to be classified as buildings  
**Effort:** 5 minutes (config change)  
**Risk:** Low-Medium (may increase false positives slightly)

---

### 🟡 **HIGH PRIORITY - Implement Soon**

#### 4. Rebalance Feature Weights

```yaml
feature_weights:
  height: 0.25 # ⬆️ from 0.22
  geometry: 0.30 # ⬇️ from 0.35
  spectral: 0.15 # ⬇️ from 0.18
  spatial: 0.15 # Keep
  ground_truth: 0.15 # ⬆️ from 0.10
```

**Impact:** ✅ Trust ground truth more, reduce geometry dominance  
**Effort:** 5 minutes (config change)  
**Risk:** Low (better balance)

---

#### 5. Increase k_neighbors for Better Features

```yaml
features:
  k_neighbors: 60 # ⬆️ from 50 (CPU) / 55 (GPU)
  search_radius: 2.0 # ⬆️ from 1.5m
  neighbor_query_batch_size: 2_000_000 # Keep for CPU
```

**Impact:** ✅ Smoother, more reliable geometric features  
**Effort:** 5 minutes (config change)  
**Risk:** Low (slight memory increase)

---

### 🟢 **MEDIUM PRIORITY - Diagnostic & Validation**

#### 6. Add Diagnostic Logging

Create a validation script to check feature quality:

```python
#!/usr/bin/env python3
"""
Diagnostic script to validate classification features
"""
import numpy as np
import laspy

def diagnose_classification_features(las_path: str):
    """Check if features are computed correctly."""

    print("🔍 Building Classification Diagnostic")
    print("=" * 60)

    # Load point cloud
    las = laspy.read(las_path)
    points = np.vstack([las.x, las.y, las.z]).T
    labels = las.classification

    print(f"\n📊 Point Cloud Statistics:")
    print(f"  Total points: {len(points):,}")
    print(f"  Classification distribution:")
    for cls in [1, 2, 3, 4, 5, 6, 9, 11]:
        count = np.sum(labels == cls)
        pct = count / len(labels) * 100
        name = {1: "Unclassified", 2: "Ground", 3: "Low Veg", 4: "Med Veg",
                5: "High Veg", 6: "Building", 9: "Water", 11: "Road"}[cls]
        print(f"    Class {cls:2d} ({name:12s}): {count:8,} ({pct:5.2f}%)")

    # Check if extra dimensions exist
    print(f"\n📝 Extra Dimensions:")
    extra_dims = [dim.name for dim in las.point_format.extra_dimensions]

    critical_features = [
        'height_above_ground', 'planarity', 'verticality',
        'curvature', 'ndvi', 'BuildingConfidence'
    ]

    for feature in critical_features:
        if feature in extra_dims:
            data = las[feature]
            print(f"  ✅ {feature:25s}: "
                  f"min={np.min(data):.3f}, "
                  f"mean={np.mean(data):.3f}, "
                  f"max={np.max(data):.3f}")
        else:
            print(f"  ❌ {feature:25s}: MISSING")

    # Analyze building points
    building_mask = labels == 6
    if np.sum(building_mask) > 0:
        print(f"\n🏢 Building Points Analysis ({np.sum(building_mask):,} points):")

        if 'height_above_ground' in extra_dims:
            hag = las['height_above_ground'][building_mask]
            print(f"  Height above ground: "
                  f"min={np.min(hag):.2f}m, "
                  f"mean={np.mean(hag):.2f}m, "
                  f"max={np.max(hag):.2f}m")

            if np.mean(hag) < 2.5:
                print(f"  ⚠️  WARNING: Mean building height < 2.5m")

        if 'planarity' in extra_dims:
            plan = las['planarity'][building_mask]
            print(f"  Planarity (roofs):   "
                  f"min={np.min(plan):.3f}, "
                  f"mean={np.mean(plan):.3f}, "
                  f"max={np.max(plan):.3f}")

            if np.mean(plan) < 0.65:
                print(f"  ⚠️  WARNING: Low planarity - roof detection may fail")

        if 'verticality' in extra_dims:
            vert = las['verticality'][building_mask]
            print(f"  Verticality (walls): "
                  f"min={np.min(vert):.3f}, "
                  f"mean={np.mean(vert):.3f}, "
                  f"max={np.max(vert):.3f}")

        if 'BuildingConfidence' in extra_dims:
            conf = las['BuildingConfidence'][building_mask]
            print(f"  Classification confidence: "
                  f"min={np.min(conf):.3f}, "
                  f"mean={np.mean(conf):.3f}, "
                  f"max={np.max(conf):.3f}")

            low_conf = np.sum(conf < 0.5) / len(conf) * 100
            print(f"    {low_conf:.1f}% have confidence < 0.5")
    else:
        print(f"\n❌ NO BUILDING POINTS FOUND")
        print(f"   This is the PRIMARY ISSUE")

    # Check unclassified points that might be buildings
    unclass_mask = labels == 1
    if np.sum(unclass_mask) > 0 and 'height_above_ground' in extra_dims:
        unclass_hag = las['height_above_ground'][unclass_mask]
        elevated = np.sum(unclass_hag > 2.5)

        print(f"\n⚠️  Unclassified Points Analysis:")
        print(f"  Total unclassified: {np.sum(unclass_mask):,}")
        print(f"  Elevated (>2.5m): {elevated:,} ({elevated/np.sum(unclass_mask)*100:.1f}%)")

        if elevated > 1000:
            print(f"  🔴 CRITICAL: Many elevated points unclassified - likely missed buildings")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python diagnose_classification.py <las_file>")
        sys.exit(1)

    diagnose_classification_features(sys.argv[1])
```

**Usage:**

```bash
python diagnose_classification.py output/tile_enriched.laz
```

**Impact:** ✅ Identify specific failure modes  
**Effort:** 30 minutes (create and run script)  
**Risk:** None (diagnostic only)

---

#### 7. Visual Validation Tool

Create a script to visualize classification quality:

```python
#!/usr/bin/env python3
"""
Visual validation of building classification
"""
import numpy as np
import laspy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_classification(las_path: str, output_path: str = None):
    """Create visualization of classification results."""

    las = laspy.read(las_path)
    points = np.vstack([las.x, las.y, las.z]).T
    labels = las.classification

    # Sample for visualization (max 500k points)
    if len(points) > 500000:
        idx = np.random.choice(len(points), 500000, replace=False)
        points = points[idx]
        labels = labels[idx]

    # Define colors
    colors = {
        1: 'gray',       # Unclassified
        2: 'brown',      # Ground
        3: 'lightgreen', # Low vegetation
        4: 'green',      # Medium vegetation
        5: 'darkgreen',  # High vegetation
        6: 'red',        # Building (highlight in red)
        9: 'blue',       # Water
        11: 'black',     # Road
    }

    point_colors = [colors.get(label, 'gray') for label in labels]

    # Create figure
    fig = plt.figure(figsize=(15, 10))

    # 3D view
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                c=point_colors, s=0.1, alpha=0.6)
    ax1.set_title('3D Classification View')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')

    # Top-down view
    ax2 = fig.add_subplot(122)
    ax2.scatter(points[:, 0], points[:, 1],
                c=point_colors, s=0.1, alpha=0.6)
    ax2.set_title('Top-Down Classification View')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_aspect('equal')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=f'Class {cls}')
                      for cls, color in colors.items()]
    ax2.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"✅ Saved visualization to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python visualize_classification.py <las_file> [output.png]")
        sys.exit(1)

    output = sys.argv[2] if len(sys.argv) > 2 else None
    visualize_classification(sys.argv[1], output)
```

---

## 📋 Testing & Validation Protocol

### Step 1: Apply Critical Fixes

1. Create modified config file:

```bash
cp examples/config_asprs_bdtopo_cadastre_cpu_optimized.yaml \
   examples/config_asprs_bdtopo_cadastre_cpu_fixed.yaml
```

2. Apply all CRITICAL fixes from section above

3. Test on problematic tile:

```bash
ign-lidar-hd process \
  -c examples/config_asprs_bdtopo_cadastre_cpu_fixed.yaml \
  input_dir="path/to/problematic/tile" \
  output_dir="output/test_fixed"
```

---

### Step 2: Run Diagnostic

```bash
python diagnose_classification.py output/test_fixed/tile_enriched.laz
```

**Expected Output:**

```
✅ BuildingConfidence present
✅ Mean building HAG: 5-8m (reasonable)
✅ Mean building planarity: >0.65
✅ Building classification: >60% of points
⚠️  Unclassified elevated points: <20%
```

---

### Step 3: Visual Comparison

```bash
# Before fixes
python visualize_classification.py output/original/tile_enriched.laz before.png

# After fixes
python visualize_classification.py output/test_fixed/tile_enriched.laz after.png
```

Compare building coverage (red points) in both images.

---

### Step 4: Quantitative Validation

```python
import laspy
import numpy as np

# Load both versions
las_before = laspy.read("output/original/tile_enriched.laz")
las_after = laspy.read("output/test_fixed/tile_enriched.laz")

# Compare building classification rates
buildings_before = np.sum(las_before.classification == 6)
buildings_after = np.sum(las_after.classification == 6)

print(f"Building points before: {buildings_before:,} "
      f"({buildings_before/len(las_before.classification)*100:.2f}%)")
print(f"Building points after:  {buildings_after:,} "
      f"({buildings_after/len(las_after.classification)*100:.2f}%)")
print(f"Improvement: {(buildings_after - buildings_before):,} points "
      f"(+{(buildings_after/buildings_before - 1)*100:.1f}%)")
```

**Success Criteria:**

- ✅ Building classification rate increases by >50%
- ✅ Building classification rate reaches >10% of total points
- ✅ Visual inspection shows roofs properly classified

---

## 🔧 Long-Term Improvements

### 1. Implement Building-Specific KDTree Search ⚠️ Code Change Required

**Problem:** Current neighbor search may include too much non-building context

**Solution:**

```python
# In adaptive.py, modify feature computation
def compute_features_building_context(points, building_polygons):
    """Compute features considering only building-relevant neighbors."""

    # For points near buildings, use only nearby building points as neighbors
    # This prevents vegetation from affecting building point features
    pass
```

---

### 2. Add Polygon Quality Pre-Filtering

**Problem:** Low-quality polygons poison the classification

**Solution:**

```yaml
building_fusion:
  min_quality_score: 0.60 # ⬆️ from 0.50 - reject poor polygons

quality_checks:
  reject_if_centroid_offset_exceeds: 15.0 # meters
  reject_if_coverage_below: 0.15 # 15% of polygon must have points
```

---

### 3. Implement Multi-Pass Classification

**Current:** Single-pass classification  
**Proposed:** Multi-pass with refinement

```python
# Pass 1: High-confidence buildings only (strict thresholds)
# Pass 2: Expand to medium-confidence using spatial coherence
# Pass 3: Final refinement based on clustering
```

---

## 📈 Success Metrics

### Target Improvements

| Metric                    | Current | Target | Status |
| ------------------------- | ------- | ------ | ------ |
| Building detection rate   | ~30-50% | >85%   | ❌     |
| False negative rate       | ~50-70% | <15%   | ❌     |
| Visual quality (roofs)    | Poor    | Good   | ❌     |
| Classification confidence | Unknown | >0.60  | ⚠️     |
| Polygon utilization       | Low     | High   | ❌     |

### Validation Criteria

✅ **PASS** if:

- Building classification rate >80% of ground truth polygons
- Visual inspection shows most roofs properly classified
- Mean building confidence >0.60
- <20% of elevated points remain unclassified

❌ **FAIL** if:

- Building classification rate <60%
- Large building sections remain unclassified
- Mean building confidence <0.50

---

## 🎯 Immediate Action Items

### For User (Next 30 Minutes)

1. ✅ Apply CRITICAL fixes to config file (5 min)
2. ✅ Run test processing on problematic tile (10-15 min)
3. ✅ Create diagnostic script and run (5 min)
4. ✅ Visual comparison before/after (5 min)
5. ✅ Report results

### For Development Team

1. 📝 Create test suite for building classification quality
2. 📝 Add automated validation in CI/CD
3. 📝 Implement polygon quality pre-filtering
4. 📝 Add building-context-aware feature computation
5. 📝 Create comprehensive troubleshooting guide

---

## 📚 Related Documentation

- [Building Classification Guide](guides/BUILDING_CLASSIFICATION_QUICK_REFERENCE.md)
- [Adaptive Classification System](features/adaptive-classification.md)
- [Building Fusion Documentation](features/building-analysis.md)
- [Classification Analysis Report](CLASSIFICATION_ANALYSIS_REPORT_2025.md)
- [ASPRS Classification Guide](guides/ASPRS_CLASSIFICATION_GUIDE.md)

---

## 🔖 Conclusion

**Current State:** 🔴 **Building classification is critically underperforming**

**Root Cause:** Likely combination of:

1. 🔴 Misaligned ground truth polygons (HIGH IMPACT)
2. ⚠️ Incorrect height above ground (MEDIUM IMPACT)
3. ⚠️ Too strict classification thresholds (MEDIUM IMPACT)

**Recommended Action:**

1. Apply CRITICAL configuration fixes immediately
2. Test and validate on problematic tile
3. If still poor, run diagnostic script to identify specific failure mode
4. Implement code-level fixes as needed

**Expected Outcome:**

- Building classification rate should improve from ~30-50% to >80%
- Most building roofs should appear in red (Class 6) in visualization
- Unclassified (pink/magenta) areas should significantly reduce

---

**Report Status:** ✅ **COMPLETE**  
**Next Steps:** Implement recommended fixes and validate
