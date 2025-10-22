# Threshold Module Migration Guide

**Version:** 3.1.0  
**Date:** October 22, 2025  
**Status:** ⚠️ Deprecation Active

---

## Overview

As of v3.1.0, the threshold configuration system has been consolidated into a single unified module: `ign_lidar.core.classification.thresholds`. This deprecates the following modules:

- ❌ `classification_thresholds.py` (will be removed in v4.0.0)
- ❌ `optimized_thresholds.py` (will be removed in v4.0.0)

**All functionality remains available** through backward compatibility wrappers, but you should migrate to the new module to avoid future breakage.

---

## Why This Change?

### Problems with Old System:

1. **Duplication:** Three files defining similar thresholds
2. **Confusion:** Unclear which module to use
3. **Inconsistency:** Risk of different thresholds in different modules
4. **Maintenance:** Had to update thresholds in multiple places

### Benefits of New System:

1. ✅ **Single source of truth:** All thresholds in `thresholds.py`
2. ✅ **Better organization:** Grouped by category (NDVI, Geometric, Height, etc.)
3. ✅ **Context awareness:** Adaptive thresholds for different modes and seasons
4. ✅ **Easier maintenance:** Update once, apply everywhere

---

##Quick Migration

### Basic Usage

**Old Code:**

```python
from ign_lidar.core.classification.classification_thresholds import ClassificationThresholds

# Access thresholds
road_height_max = ClassificationThresholds.ROAD_HEIGHT_MAX
building_height_min = ClassificationThresholds.BUILDING_HEIGHT_MIN
```

**New Code:**

```python
from ign_lidar.core.classification.thresholds import get_thresholds

# Get threshold configuration
thresholds = get_thresholds()

# Access thresholds
road_height_max = thresholds.height.road_height_max
building_height_min = thresholds.building.height_min
```

---

### Mode-Specific Thresholds

**Old Code:**

```python
from ign_lidar.core.classification.classification_thresholds import ClassificationThresholds

# Get LOD2 building thresholds
building_thresholds = ClassificationThresholds.get_building_thresholds('lod2')
wall_verticality = building_thresholds['wall_verticality_min']
```

**New Code:**

```python
from ign_lidar.core.classification.thresholds import get_thresholds

# Get LOD2 thresholds
thresholds = get_thresholds(mode='lod2')

# Access building thresholds
wall_verticality = thresholds.building.wall_verticality_min_lod2
# Or use the helper method
building_dict = thresholds.building.get_for_mode('lod2')
wall_verticality = building_dict['wall_verticality_min']
```

---

### NDVI and Geometric Thresholds

**Old Code:**

```python
from ign_lidar.core.classification.optimized_thresholds import NDVIThresholds, GeometricThresholds

ndvi = NDVIThresholds()
geom = GeometricThresholds()

veg_min = ndvi.vegetation_min
planarity_min = geom.planarity_ground_min
```

**New Code:**

```python
from ign_lidar.core.classification.thresholds import get_thresholds

thresholds = get_thresholds()

veg_min = thresholds.ndvi.vegetation_min
planarity_min = thresholds.geometric.planarity_ground_min
```

---

### Context-Adaptive Thresholds

**Old Code:**

```python
from ign_lidar.core.classification.optimized_thresholds import NDVIThresholds

ndvi = NDVIThresholds()
adjusted = ndvi.get_context_adjusted(season='summer', urban_context=True)
```

**New Code (Same API):**

```python
from ign_lidar.core.classification.thresholds import get_thresholds

# Get thresholds already adjusted for context
thresholds = get_thresholds(season='summer', urban_context=True)
veg_min = thresholds.ndvi.vegetation_min  # Already adjusted

# Or adjust manually
thresholds = get_thresholds()
adjusted_ndvi = thresholds.ndvi.get_context_adjusted(season='summer', urban_context=True)
```

---

## Complete Migration Table

| Old Module                  | Old Import                                           | New Import                                      |
| --------------------------- | ---------------------------------------------------- | ----------------------------------------------- |
| `classification_thresholds` | `ClassificationThresholds`                           | `ThresholdConfig` or `get_thresholds()`         |
| `classification_thresholds` | `ClassificationThresholds.ROAD_HEIGHT_MAX`           | `thresholds.height.road_height_max`             |
| `classification_thresholds` | `ClassificationThresholds.get_building_thresholds()` | `thresholds.building.get_for_mode()`            |
| `optimized_thresholds`      | `NDVIThresholds`                                     | `NDVIThresholds` (same name, new location)      |
| `optimized_thresholds`      | `GeometricThresholds`                                | `GeometricThresholds` (same name, new location) |
| `optimized_thresholds`      | `HeightThresholds`                                   | `HeightThresholds` (same name, new location)    |
| `optimized_thresholds`      | `ClassificationThresholds`                           | `ThresholdConfig`                               |

---

## Detailed Examples

### Example 1: Transport Detection

**Before:**

```python
from ign_lidar.core.classification.classification_thresholds import ClassificationThresholds

class TransportDetector:
    def __init__(self, strict_mode=False):
        if strict_mode:
            self.road_height_max = ClassificationThresholds.ROAD_HEIGHT_MAX_STRICT
            self.road_planarity_min = ClassificationThresholds.ROAD_PLANARITY_MIN_STRICT
        else:
            self.road_height_max = ClassificationThresholds.ROAD_HEIGHT_MAX
            self.road_planarity_min = ClassificationThresholds.ROAD_PLANARITY_MIN

    def classify_point(self, height, planarity):
        if height < self.road_height_max and planarity > self.road_planarity_min:
            return "road"
        return "unclassified"
```

**After:**

```python
from ign_lidar.core.classification.thresholds import get_thresholds

class TransportDetector:
    def __init__(self, strict_mode=False):
        # Get appropriate thresholds
        self.thresholds = get_thresholds(strict=strict_mode)

    def classify_point(self, height, planarity):
        if (height < self.thresholds.height.road_height_max and
            planarity > self.thresholds.transport.road_planarity_min):
            return "road"
        return "unclassified"
```

---

### Example 2: Building Detection

**Before:**

```python
from ign_lidar.core.classification.classification_thresholds import ClassificationThresholds

class BuildingDetector:
    def __init__(self, mode='asprs'):
        self.thresholds = ClassificationThresholds.get_building_thresholds(mode)

    def is_wall(self, verticality, planarity):
        return (verticality > self.thresholds['wall_verticality_min'] and
                planarity > self.thresholds['wall_planarity_min'])
```

**After:**

```python
from ign_lidar.core.classification.thresholds import get_thresholds

class BuildingDetector:
    def __init__(self, mode='asprs'):
        self.config = get_thresholds(mode=mode)
        self.thresholds = self.config.building.get_for_mode(mode)

    def is_wall(self, verticality, planarity):
        return (verticality > self.thresholds['wall_verticality_min'] and
                planarity > self.thresholds['wall_planarity_min'])
```

---

### Example 3: Vegetation Classification

**Before:**

```python
from ign_lidar.core.classification.optimized_thresholds import (
    NDVIThresholds, GeometricThresholds, HeightThresholds
)

ndvi_thresh = NDVIThresholds()
geom_thresh = GeometricThresholds()
height_thresh = HeightThresholds()

def classify_vegetation(ndvi, height, planarity):
    if ndvi < ndvi_thresh.vegetation_min:
        return "non_vegetation"

    if height < height_thresh.low_vegetation_max:
        return "low_vegetation"
    elif height > height_thresh.high_vegetation_min:
        return "high_vegetation"

    return "medium_vegetation"
```

**After:**

```python
from ign_lidar.core.classification.thresholds import get_thresholds

# Get all thresholds at once
thresholds = get_thresholds()

def classify_vegetation(ndvi, height, planarity):
    if ndvi < thresholds.ndvi.vegetation_min:
        return "non_vegetation"

    if height < thresholds.height.low_veg_height_max:
        return "low_vegetation"
    elif height > thresholds.height.high_veg_height_min:
        return "high_vegetation"

    return "medium_vegetation"
```

---

## New Features in Unified Module

### 1. Comprehensive Threshold Categories

The new module organizes thresholds by category:

```python
from ign_lidar.core.classification.thresholds import get_thresholds

thresholds = get_thresholds()

# NDVI thresholds
thresholds.ndvi.vegetation_min
thresholds.ndvi.building_max
thresholds.ndvi.forest_min

# Geometric thresholds
thresholds.geometric.planarity_ground_min
thresholds.geometric.verticality_wall_min
thresholds.geometric.curvature_vegetation_min

# Height thresholds
thresholds.height.ground_max
thresholds.height.building_min
thresholds.height.high_veg_height_min

# Transport-specific
thresholds.transport.road_planarity_min
thresholds.transport.rail_buffer_multiplier

# Building-specific
thresholds.building.height_min
thresholds.building.wall_score_min

# Additional categories
thresholds.water.planarity_min
thresholds.bridge.height_min
thresholds.vehicle.height_range
```

### 2. Mode-Aware Configuration

```python
# ASPRS mode (lenient)
asprs_config = get_thresholds(mode='asprs')

# LOD2 mode (stricter)
lod2_config = get_thresholds(mode='lod2')

# LOD3 mode (strictest)
lod3_config = get_thresholds(mode='lod3')

# Strict mode (urban areas)
strict_config = get_thresholds(mode='asprs', strict=True)
```

### 3. Context-Adaptive Thresholds

```python
# Summer urban context
summer_urban = get_thresholds(season='summer', urban_context=True)

# Winter rural context
winter_rural = get_thresholds(season='winter', urban_context=False)
```

### 4. Validation and Export

```python
thresholds = get_thresholds()

# Validate threshold consistency
warnings = thresholds.validate()
if warnings:
    for key, msg in warnings.items():
        print(f"Warning: {msg}")

# Export to dictionary
config_dict = thresholds.get_all()

# Print summary
from ign_lidar.core.classification.thresholds import print_threshold_summary
print_threshold_summary(thresholds)
```

---

## Backward Compatibility

### How It Works

The old modules (`classification_thresholds.py` and `optimized_thresholds.py`) are now **thin wrappers** that:

1. Import classes from the new `thresholds.py` module
2. Issue deprecation warnings when imported
3. Re-export everything with the same names

This means **your existing code will continue to work**, but you'll see warnings like:

```
DeprecationWarning: classification_thresholds.py is deprecated as of v3.1.0
and will be removed in v4.0.0. Please use 'from ign_lidar.core.classification.thresholds
import ThresholdConfig' instead. See docs/THRESHOLD_MIGRATION_GUIDE.md for migration guide.
```

### Suppressing Warnings (Temporary)

If you need to suppress warnings temporarily:

```python
import warnings

# Suppress only deprecation warnings from threshold modules
warnings.filterwarnings('ignore', category=DeprecationWarning, module='.*classification.*')

# Your code here
from ign_lidar.core.classification.classification_thresholds import ClassificationThresholds
```

**Note:** This is only a temporary solution. You should migrate to avoid breakage in v4.0.0.

---

## Migration Checklist

### For Application Code:

- [ ] Search for imports from `classification_thresholds`
- [ ] Search for imports from `optimized_thresholds`
- [ ] Replace with imports from `thresholds`
- [ ] Update threshold access patterns (see examples above)
- [ ] Test that behavior is unchanged
- [ ] Remove deprecation warning suppressions

### For Library/Module Code:

- [ ] Update all internal imports
- [ ] Update configuration classes to use new threshold structure
- [ ] Update tests to use new module
- [ ] Update documentation and examples
- [ ] Add migration notes to CHANGELOG

---

## Timeline

| Version    | Status      | Details                                       |
| ---------- | ----------- | --------------------------------------------- |
| **v3.1.0** | ✅ Current  | Deprecation warnings active, old modules work |
| **v3.2.x** | 🔄 Planned  | Continued support, louder warnings            |
| **v3.3.x** | 🔄 Planned  | Final version with old modules                |
| **v4.0.0** | ⚠️ Breaking | Old modules removed completely                |

---

## Testing Your Migration

### Unit Tests

```python
import pytest
from ign_lidar.core.classification.thresholds import get_thresholds

def test_threshold_access():
    """Test new threshold access pattern."""
    thresholds = get_thresholds()

    # Test NDVI thresholds
    assert 0 < thresholds.ndvi.vegetation_min < 1
    assert thresholds.ndvi.building_max < thresholds.ndvi.vegetation_min

    # Test geometric thresholds
    assert 0 < thresholds.geometric.planarity_ground_min <= 1

    # Test height thresholds
    assert thresholds.height.building_min > thresholds.height.ground_max

def test_mode_specific():
    """Test mode-specific thresholds."""
    asprs = get_thresholds(mode='asprs')
    lod2 = get_thresholds(mode='lod2')
    lod3 = get_thresholds(mode='lod3')

    # LOD3 should be strictest
    assert (lod3.building.wall_verticality_min_lod3 >
            lod2.building.wall_verticality_min_lod2 >
            asprs.building.wall_verticality_min_asprs)
```

### Integration Tests

Run your existing integration tests after migration to ensure behavior is unchanged.

---

## Getting Help

### Documentation:

- Main plan: `docs/CLASSIFICATION_CONSOLIDATION_PLAN.md`
- API reference: Generated from docstrings in `thresholds.py`
- Examples: `examples/` directory

### Common Issues:

**Issue:** `AttributeError: 'ThresholdConfig' object has no attribute 'ROAD_HEIGHT_MAX'`  
**Solution:** Use `thresholds.height.road_height_max` instead of `ROAD_HEIGHT_MAX` constant

**Issue:** Deprecation warnings flooding logs  
**Solution:** Migrate your code or temporarily suppress warnings (see above)

**Issue:** Different threshold values after migration  
**Solution:** This should not happen. If it does, please file a bug report.

---

## Questions?

If you have questions or encounter issues during migration:

1. Check this guide
2. Check the docstrings in `thresholds.py`
3. Check existing examples in `examples/` directory
4. File an issue on GitHub

---

**Last Updated:** October 22, 2025  
**Next Review:** v3.2.0 release
