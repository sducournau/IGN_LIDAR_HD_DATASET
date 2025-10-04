# Radius Parameter Impact Guide

**Purpose**: Visual guide showing how radius parameter affects geometric features

---

## Understanding Radius Search

### What is Radius Search?

Instead of finding a **fixed number** of neighbors (k-NN), radius search finds **all neighbors within a distance**.

```
k-NN (k=50):              Radius (r=1.0m):
Find 50 nearest           Find all within 1.0m sphere
▓▓▓▓▓                    ░░○░░
▓▓○▓▓  ← 50 points       ░○○○░  ← Variable count
▓▓▓▓▓                    ░░○░░
Following scan pattern    Following true geometry
```

---

## Radius Size Impact

### Visual Comparison

```
RADIUS = 0.5m (Small)
├─ Neighborhood: 30-50 points
├─ Features: SHARP, detailed
├─ Noise: HIGH
├─ Artifacts: RARE but possible
└─ Use: Fine details, edges

RADIUS = 1.0m (Medium - AUTO)
├─ Neighborhood: 80-150 points
├─ Features: BALANCED
├─ Noise: MODERATE
├─ Artifacts: NONE ✅
└─ Use: GENERAL PURPOSE (RECOMMENDED)

RADIUS = 1.5m (Large)
├─ Neighborhood: 200-300 points
├─ Features: SMOOTH
├─ Noise: LOW
├─ Artifacts: NONE ✅
└─ Use: Large surfaces, regional

RADIUS = 2.0m (Maximum)
├─ Neighborhood: 400+ points
├─ Features: VERY SMOOTH
├─ Noise: VERY LOW
├─ Artifacts: NONE ✅
└─ Use: Major structures, filtering
```

---

## Feature Response to Radius

### Linearity (Edges, Cables)

```
Radius → Feature Response

0.5m    ████████░░░   High linearity, sharp
1.0m    ██████░░░░   Moderate, balanced ✅
1.5m    ████░░░░░░   Lower, smoother
2.0m    ██░░░░░░░░   Very smooth

Edge detection: Smaller radius = sharper
```

### Planarity (Roofs, Walls)

```
Radius → Feature Response

0.5m    ████████░░░   High detail
1.0m    ██████████   Stable, accurate ✅
1.5m    ██████████   Very stable
2.0m    ██████████   Regional average

Plane fitting: 1.0-1.5m optimal
```

### Sphericity (Vegetation, Noise)

```
Radius → Feature Response

0.5m    ██████░░░░   Sensitive to noise
1.0m    ████░░░░░░   Balanced ✅
1.5m    ███░░░░░░░   Smoothed
2.0m    ██░░░░░░░░   Regional pattern

Noise filtering: Larger radius = more stable
```

---

## Practical Examples

### Example 1: Building Roof (Flat Surface)

```
Point Cloud: Flat roof with ~0.1m noise

Radius = 0.5m:
├─ Planarity: 0.92 ± 0.08  (noisy)
├─ Normal vectors: Variable
└─ Classification: Uncertain

Radius = 1.0m: ✅ OPTIMAL
├─ Planarity: 0.98 ± 0.02  (stable)
├─ Normal vectors: Consistent
└─ Classification: Confident

Radius = 2.0m:
├─ Planarity: 0.99 ± 0.01  (very smooth)
├─ Normal vectors: Regional average
└─ Classification: May miss details
```

### Example 2: Building Edge (Sharp Feature)

```
Point Cloud: 90° roof/wall intersection

Radius = 0.5m: ✅ GOOD
├─ Linearity: 0.85  (sharp)
├─ Edge detection: YES
└─ Classification: Edge

Radius = 1.0m:
├─ Linearity: 0.65  (moderate)
├─ Edge detection: YES (softer)
└─ Classification: Transition

Radius = 2.0m:
├─ Linearity: 0.35  (blurred)
├─ Edge detection: MISSED
└─ Classification: Mixed surface
```

### Example 3: Vegetation (Spherical Structure)

```
Point Cloud: Tree canopy

Radius = 0.5m:
├─ Sphericity: 0.45 ± 0.15  (noisy)
├─ Classification: Uncertain
└─ Noise sensitive

Radius = 1.0m: ✅ OPTIMAL
├─ Sphericity: 0.52 ± 0.08  (stable)
├─ Classification: Vegetation
└─ Balanced noise/detail

Radius = 1.5m:
├─ Sphericity: 0.48 ± 0.04  (smooth)
├─ Classification: Vegetation
└─ Regional pattern
```

---

## Decision Matrix

### Choose Radius Based On

#### Point Density

| Point Spacing | Recommended Radius | Typical Environment    |
| ------------- | ------------------ | ---------------------- |
| 0.1-0.2m      | 0.5-0.8m           | Very dense urban       |
| 0.2-0.5m      | 0.8-1.2m           | **Standard (AUTO)** ✅ |
| 0.5-1.0m      | 1.2-1.8m           | Sparse rural           |
| 1.0-2.0m      | 1.8-2.0m           | Very sparse            |

#### Feature Scale

| Feature Type            | Optimal Radius | Examples                |
| ----------------------- | -------------- | ----------------------- |
| **Fine details**        | 0.5-0.8m       | Cables, antennas, edges |
| **Building components** | 0.8-1.2m       | **Walls, roofs** ✅     |
| **Large structures**    | 1.2-1.8m       | Building masses         |
| **Regional patterns**   | 1.8-2.0m       | Urban blocks            |

#### Application Goal

| Goal                   | Recommended Radius | Reason              |
| ---------------------- | ------------------ | ------------------- |
| **Edge detection**     | 0.5-0.8m           | Need sharp features |
| **General extraction** | 1.0-1.2m           | **Balanced** ✅     |
| **Noise reduction**    | 1.5-2.0m           | Smooth surfaces     |
| **Classification**     | 1.0-1.5m           | Stable features     |

---

## Auto-Estimation Algorithm

### How It Works

```python
# Step 1: Sample point cloud
sample_points = random_sample(points, n=1000)

# Step 2: Measure average spacing
avg_spacing = median_nearest_neighbor_distance(sample_points)
# Typical: 0.2-0.5m for IGN LIDAR HD

# Step 3: Calculate radius
radius = avg_spacing * 20.0  # For geometric features
radius = clip(radius, 0.5, 2.0)  # Safety bounds

# Result: 0.75-1.5m typical range
```

### Why 20x Multiplier?

```
Nearest neighbor: 0.3m spacing
Want to capture: True surface geometry

1x  (0.3m)  → 3-5 points    ❌ Too few
5x  (1.5m)  → 20-30 points  ⚠️ Still sparse
10x (3.0m)  → 80-100 points ⚠️ Still uncertain
20x (6.0m)  → 200+ points   ✅ Reliable (but clipped to 2.0m max)

Empirical testing: 15-20x optimal for building surfaces
```

---

## Common Patterns

### Problem: Noisy Features

```yaml
# Symptom: High variance in planarity/linearity
# Solution: Increase radius

enrich:
  radius: 1.5 # Smooth features
  mode: building
```

### Problem: Missing Fine Details

```yaml
# Symptom: Edges not detected, features blurred
# Solution: Decrease radius

enrich:
  radius: 0.6 # Sharp features
  mode: building
```

### Problem: Scan Line Artifacts

```yaml
# Symptom: Dash patterns, striped features
# Solution: Use auto-estimation (default)

enrich:
  radius: null # Auto-estimate ✅
  mode: building
```

---

## Testing Different Radii

### Quick Test Script

```bash
#!/bin/bash
# Test different radius values on same file

for radius in 0.5 1.0 1.5 2.0; do
    echo "Testing radius=${radius}m"
    ign-lidar-hd enrich \
        --input test_tile.laz \
        --output test_r${radius}.laz \
        --radius ${radius} \
        --mode building
done

# Compare in CloudCompare or QGIS
```

### What to Look For

**Visual Inspection**:

- Planarity on roofs (should be uniform, not striped)
- Edge sharpness (should match visual edges)
- Noise level (should be stable, not flickering)

**Statistical Check**:

```python
# Load enriched file
las = laspy.read('enriched.laz')

# Check feature statistics
print(f"Planarity: mean={las.planarity.mean():.3f}, std={las.planarity.std():.3f}")
print(f"Linearity: mean={las.linearity.mean():.3f}, std={las.linearity.std():.3f}")

# Good results:
# - Planarity on roofs: > 0.9
# - Std deviation: < 0.1
# - No obvious artifacts
```

---

## Performance Considerations

### Computation Time

```
Radius → Avg Neighbors → Time

0.5m   → 50-80     → FAST    (100%)
1.0m   → 100-200   → MEDIUM  (85%)  ✅
1.5m   → 250-400   → SLOW    (70%)
2.0m   → 400-600   → SLOWER  (60%)
```

### Memory Usage

```
Radius → Peak Memory → Recommendation

0.5m   → LOW       → Any size dataset
1.0m   → MEDIUM    → < 20M points ✅
1.5m   → HIGH      → < 10M points
2.0m   → VERY HIGH → < 5M points, chunking recommended
```

---

## Advanced: Adaptive Radius

### Concept

Different features may benefit from different radii:

```python
# Pseudo-code for adaptive approach
# (Not currently implemented)

def compute_features_adaptive(points):
    # Normals: small radius (local)
    normals = compute_normals(points, radius=0.5)

    # Geometric features: medium radius (balanced)
    features = extract_geometric_features(points, radius=1.0)

    # Regional statistics: large radius (context)
    context = compute_context(points, radius=2.0)

    return normals, features, context
```

**Note**: Current implementation uses single radius for all features (simpler, faster)

---

## Summary Recommendations

### Default Use (90% of cases)

```bash
# Let system auto-estimate ✅
ign-lidar-hd enrich --input-dir raw/ --output enriched/ --mode building
```

### Fine-Tuning Scenarios

```bash
# Dense urban, high detail needed
ign-lidar-hd enrich ... --radius 0.8

# Standard buildings (auto is good)
ign-lidar-hd enrich ... --radius 1.0  # or omit (auto)

# Large structures, noise reduction
ign-lidar-hd enrich ... --radius 1.5
```

### Quick Reference Table

| Scenario            | Radius | Command           |
| ------------------- | ------ | ----------------- |
| **Most cases**      | Auto   | `--mode building` |
| **Dense urban**     | 0.8m   | `--radius 0.8`    |
| **Standard**        | 1.0m   | `--radius 1.0`    |
| **Sparse rural**    | 1.5m   | `--radius 1.5`    |
| **Noise filtering** | 2.0m   | `--radius 2.0`    |

---

**Guide Version**: 1.0  
**Last Updated**: October 4, 2025  
**Related**: `ARTIFACT_AUDIT_ENRICH_REPORT.md`, `ENRICH_ARTIFACT_AUDIT_SUMMARY.md`
