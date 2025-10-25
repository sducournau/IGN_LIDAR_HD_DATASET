# Facade Detection Improvements - Configuration Update

## Issue Identified

From the classification visualization, building facades are partially classified correctly (green) but significant portions remain unclassified (blue). This indicates that the facade detection parameters are too conservative.

## Root Causes

1. **Verticality thresholds too strict** - Missing semi-vertical or textured facades
2. **Insufficient buffer distances** - Not capturing facades outside building footprints
3. **Gap detection not aggressive enough** - Missing sparse or broken facades
4. **Confidence thresholds too high** - Rejecting uncertain but valid facade points

## Configuration Changes Applied

### 1. Building Footprint Detection (Ground Truth)

#### Wall/Facade Detection Parameters

```yaml
# BEFORE → AFTER
wall_verticality_threshold: 0.50 → 0.40 # Catch more semi-vertical facades
missing_wall_threshold: 0.12 → 0.08 # More aggressive expansion
vertical_buffer: 1.2 → 1.5 # Larger vertical tolerance
horizontal_buffer_ground: 2.5 → 3.5 # Larger ground buffer
horizontal_buffer_upper: 3.5 → 4.5 # Larger upper buffer for balconies
```

**Impact**: Expands building detection zones to capture facades that are:

- Not perfectly vertical (textured walls, balconies)
- Outside the strict building footprint polygon
- At ground level (low walls, annexes)

#### Advanced Facade Detection

```yaml
# BEFORE → AFTER
overhang_max_distance: 9.0 → 12.0 # Detect larger overhangs
roof_horizontal_expansion: 5.0 → 6.5 # Expand from roof edges
alpha_value: 3.5 → 4.5 # More aggressive shape detection
```

**Impact**: Better handles complex building geometries with:

- Large overhangs and balconies
- Recessed facades
- Irregular footprints

#### Wall-Specific Detection

```yaml
# BEFORE → AFTER
min_wall_height: 1.5 → 1.2 # Catch lower walls
max_wall_thickness: 1.5 → 2.0 # Thicker wall tolerance
wall_normal_tolerance: 0.30 → 0.40 # More variation allowed
```

**Impact**: Captures walls that are:

- Lower height (annexes, low sections)
- Thicker (compound walls, textured)
- Less uniform (irregular surfaces)

#### Gap Detection (Critical for Missing Facades)

```yaml
# BEFORE → AFTER
gap_detection_resolution: 96 → 120 # Higher resolution scan
gap_detection_band_width: 4.0 → 5.5 # Search further from polygon
gap_min_points_per_sector: 8 → 5 # More sensitive to sparse facades
gap_significant_threshold: 0.08 → 0.05 # More aggressive detection
```

**Impact**: Finds missing facades through:

- Higher resolution angular scanning (120 sectors vs 96)
- Wider search band (5.5m vs 4.0m)
- Lower point density requirements
- More sensitive gap detection

### 2. Classification Thresholds

#### Building Classification Thresholds

```yaml
# BEFORE → AFTER
min_verticality: 0.45 → 0.35 # Accept more angles as facades
min_verticality_strict: 0.60 → 0.50 # Less strict mode
min_wall_verticality: 0.65 → 0.55 # Wall-specific threshold
max_wall_planarity: 0.50 → 0.60 # Allow rougher surfaces
min_wall_points: 50 → 35 # Fewer points required
wall_to_roof_ratio: 0.25 → 0.20 # Adjust wall/roof balance
```

**Impact**: Classifies points as building when:

- Verticality as low as 0.35 (was 0.45) - catches angled facades
- Fewer points required (35 vs 50) - handles sparse data
- Less planar surfaces accepted - textured walls

#### Facade-Specific Thresholds

```yaml
# BEFORE → AFTER
min_facade_height: 1.5 → 1.2 # Lower facades detected
min_facade_verticality: 0.50 → 0.40 # More permissive angle
facade_thickness_max: 1.2 → 1.5 # Thicker facades allowed
facade_continuity_min: 0.5 → 0.4 # Accept broken facades
max_ndvi: 0.20 → 0.22 # Slightly relaxed vegetation filter
building_buffer_expansion: 5 → 6 # Larger polygon expansion
```

**Impact**: Dedicated facade detection that:

- Accepts lower, less vertical facades
- Handles thicker wall structures
- Works with broken/discontinuous facades
- Slightly more permissive with vegetation (may include some climbing plants)

### 3. Reclassification Rules

#### Confidence Thresholds

```yaml
# BEFORE → AFTER
min_classification_confidence: 0.35 → 0.25 # Accept more uncertain points
rejection_confidence_threshold: 0.25 → 0.30 # Less aggressive rejection
```

**Impact**:

- Keeps facade points that are less certain but geometrically valid
- Reduces false negatives (unclassified facades)
- May increase some false positives (needs monitoring)

#### Vertical Point Reclassification

```yaml
# BEFORE → AFTER
min_verticality: 0.30 → 0.25 # Even more permissive
max_distance_to_building: 3.5 → 4.5 # Search further from polygons
min_height: 1.5 → 1.2 # Lower facades included
max_ndvi: 0.20 → 0.22 # Slightly relaxed
```

**Impact**: Reclassifies unclassified vertical points as buildings when:

- Within 4.5m of building polygon (was 3.5m)
- Verticality ≥ 0.25 (very permissive)
- Height ≥ 1.2m (includes low walls)

## Expected Results

### Improvements

1. **More complete facades** - Blue (unclassified) areas should turn green (building)
2. **Better coverage** - Facades outside footprints should be captured
3. **Low walls included** - Ground-level annexes and low sections detected
4. **Textured surfaces** - Rough or irregular facades classified correctly

### Potential Side Effects (Monitor)

1. **Slightly more false positives** - Some vegetation near buildings might be included
2. **More edge points** - Building edges may be less sharp
3. **Noise near buildings** - Some scattered points near facades may be classified as building

### Mitigation Strategies

If too many false positives appear:

1. **Reduce NDVI tolerance**: Lower `max_ndvi` from 0.22 back to 0.20
2. **Increase confidence**: Raise `min_classification_confidence` from 0.25 to 0.28
3. **Tighten verticality**: Increase `min_verticality` from 0.35 to 0.38
4. **Reduce search distance**: Lower `max_distance_to_building` from 4.5m to 4.0m

## Testing Recommendations

### Visual Validation

1. **Check facade completeness**: Are blue facades now green?
2. **Inspect boundaries**: Are building edges clean or noisy?
3. **Vegetation filter**: Any green vegetation classified as building?
4. **Low structures**: Are low walls and annexes captured?

### Quantitative Metrics

```bash
# Check classification distribution
ign-lidar-hd process \
  -c examples/production/asprs_memory_optimized.yaml \
  input_dir="/data/tiles" \
  output_dir="/data/output_improved" \
  monitoring.compute_statistics=true

# Compare with previous run
# Look for:
# - Increased building class percentage
# - Decreased unclassified percentage
# - Similar vegetation/road percentages
```

### A/B Testing

Process the same tile with both configurations and compare:

1. Visual inspection side-by-side
2. Class distribution statistics
3. Point counts per class
4. Processing time differences (minimal expected)

## Rollback Instructions

If results are worse, revert these specific values:

```yaml
# Ground truth - buildings
wall_verticality_threshold: 0.50
missing_wall_threshold: 0.12
horizontal_buffer_ground: 2.5
horizontal_buffer_upper: 3.5
gap_detection_band_width: 4.0

# Classification - buildings
min_verticality: 0.45
min_wall_points: 50
facade_continuity_min: 0.5

# Reclassification
min_classification_confidence: 0.35
min_verticality: 0.30 (reclassify_vertical_to_building)
```

## Next Steps

1. **Run test processing** on sample tile
2. **Visual inspection** of results
3. **Compare statistics** with previous run
4. **Fine-tune if needed** based on results
5. **Document final parameters** once satisfied

## Configuration File

Updated configuration: `examples/production/asprs_memory_optimized.yaml`

## Summary of Key Changes

- **10 parameter groups updated** across ground truth, classification, and reclassification
- **Focus: Facade detection aggressiveness** - more permissive thresholds throughout
- **Trade-off: Completeness vs precision** - prioritizing facade coverage over strict filtering
- **Monitoring required** - check for false positives with vegetation

---

**Author**: GitHub Copilot  
**Date**: 2025-10-26  
**Version**: 6.3.3-cpu facade enhancement v2
