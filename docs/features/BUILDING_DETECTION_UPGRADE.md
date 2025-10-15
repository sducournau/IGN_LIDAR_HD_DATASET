# Multi-Mode Building Detection - Upgrade Guide

**Date**: October 15, 2025  
**Version**: 2.0  
**Status**: ✅ Completed

## 📋 Overview

This document describes the upgraded building detection system that supports different detection modes optimized for specific use cases: ASPRS classification, LOD2 building reconstruction, and LOD3 detailed architectural modeling.

## 🎯 Key Features

### Three Detection Modes

1. **ASPRS Mode** - General building detection
   - Output: Single building class (ASPRS code 6)
   - Use case: Standard LiDAR classification workflows
   - Detection: Walls, roofs, structured elements, edges
2. **LOD2 Mode** - Building element detection
   - Output: Walls (0), Flat roofs (1), Gable roofs (2), Hip roofs (3)
   - Use case: LOD2 building reconstruction training
   - Detection: Separates walls from roofs, classifies roof types
3. **LOD3 Mode** - Detailed architectural detection
   - Output: All LOD2 classes + Windows (13), Doors (14), Balconies (15), Chimneys (18), Dormers (20)
   - Use case: LOD3 detailed building modeling
   - Detection: Architectural details using intensity and spatial analysis

### Five Detection Strategies

All modes use a combination of complementary strategies:

1. **Ground Truth Detection** - Highest priority
   - Uses building polygons from BD TOPO® or other sources
   - Overrides geometric detections when enabled
2. **Wall Detection** - Verticality-based
   - Identifies vertical planar surfaces
   - Uses verticality, planarity, and wall scores
3. **Roof Detection** - Horizontality-based
   - Identifies horizontal planar surfaces
   - Separates flat vs sloped roofs (LOD2+)
4. **Structure Detection** - Anisotropy-based
   - Identifies organized directional structures
   - Distinguishes buildings from random vegetation
5. **Edge Detection** - Linearity-based
   - Detects building corners and boundaries
   - Uses linear feature analysis

## 📁 New Files Created

### Core Module

```
ign_lidar/core/modules/building_detection.py
```

New building detection module with:

- `BuildingDetectionMode` enum (ASPRS, LOD2, LOD3)
- `BuildingDetectionConfig` class with mode-specific thresholds
- `BuildingDetector` class with mode-aware detection
- `detect_buildings_multi_mode()` convenience function

### Tests

```
tests/test_building_detection_modes.py
```

Comprehensive test suite covering:

- Configuration for each mode
- Detection strategies
- Ground truth handling
- Edge cases and error handling

## 🔧 Modified Files

### 1. `ign_lidar/core/modules/classification_refinement.py`

**Changes:**

- Added import of new building detection module
- Updated `refine_building_classification()` to support `mode` parameter
- Integration with new detection system while maintaining backward compatibility
- Enhanced logging with mode information

**New parameters:**

```python
def refine_building_classification(
    ...,
    curvature: Optional[np.ndarray] = None,  # For LOD3
    intensity: Optional[np.ndarray] = None,  # For LOD3
    points: Optional[np.ndarray] = None,     # For LOD3
    mode: str = 'lod2',                      # Detection mode
    ...
)
```

### 2. `ign_lidar/core/modules/advanced_classification.py`

**Changes:**

- Added import of building detection module
- Added `building_detection_mode` parameter to `AdvancedClassifier.__init__()`
- Updated `_classify_by_geometry()` to use mode-aware detection
- Fallback to legacy detection if mode-aware detection fails

**New parameter:**

```python
def __init__(
    self,
    ...,
    building_detection_mode: str = 'asprs'  # Detection mode
)
```

## 💻 Usage Examples

### Example 1: ASPRS Mode (Simple Building Detection)

```python
from ign_lidar.core.modules.building_detection import detect_buildings_multi_mode
import numpy as np

# Prepare features
features = {
    'height': height_array,
    'planarity': planarity_array,
    'verticality': verticality_array,
    'normals': normals_array,
    'linearity': linearity_array,
    'anisotropy': anisotropy_array
}

# Detect buildings in ASPRS mode
refined_labels, stats = detect_buildings_multi_mode(
    labels=initial_labels,
    features=features,
    mode='asprs',
    ground_truth_mask=building_polygons_mask
)

# Result: Buildings classified as ASPRS code 6
print(f"Total buildings detected: {stats['total']}")
print(f"  - From ground truth: {stats['ground_truth']}")
print(f"  - Walls detected: {stats['walls']}")
print(f"  - Roofs detected: {stats['roofs']}")
```

### Example 2: LOD2 Mode (Building Elements)

```python
from ign_lidar.core.modules.building_detection import (
    BuildingDetector,
    BuildingDetectionConfig,
    BuildingDetectionMode
)

# Create LOD2 configuration
config = BuildingDetectionConfig(mode=BuildingDetectionMode.LOD2)

# Customize thresholds if needed
config.wall_verticality_min = 0.75  # Stricter wall detection
config.roof_planarity_min = 0.75    # Stricter roof detection

# Create detector
detector = BuildingDetector(config=config)

# Detect building elements
refined_labels, stats = detector.detect_buildings(
    labels=labels,
    height=height,
    planarity=planarity,
    verticality=verticality,
    normals=normals,
    linearity=linearity,
    anisotropy=anisotropy,
    wall_score=wall_score,
    roof_score=roof_score,
    ground_truth_mask=building_mask
)

# Results: Buildings separated into elements
print(f"Total building points: {stats['total_building']}")
print(f"  - Walls (class 0): {stats['walls']}")
print(f"  - Flat roofs (class 1): {stats['flat_roofs']}")
print(f"  - Sloped roofs (class 2/3): {stats['sloped_roofs']}")
```

### Example 3: LOD3 Mode (Detailed Architecture)

```python
from ign_lidar.core.modules.building_detection import detect_buildings_multi_mode

# Prepare all features (including LOD3-specific ones)
features = {
    'height': height_array,
    'planarity': planarity_array,
    'verticality': verticality_array,
    'normals': normals_array,
    'linearity': linearity_array,
    'anisotropy': anisotropy_array,
    'curvature': curvature_array,      # For detail detection
    'intensity': intensity_array,      # For opening detection
    'points': xyz_coordinates          # For spatial analysis
}

# Detect detailed architecture
refined_labels, stats = detect_buildings_multi_mode(
    labels=initial_labels,
    features=features,
    mode='lod3',
    ground_truth_mask=building_polygons_mask
)

# Results: Detailed architectural elements
print(f"Total building points: {stats['total_building']}")
print(f"  - Walls: {stats['walls']}")
print(f"  - Roofs: {stats['flat_roofs'] + stats['sloped_roofs']}")
print(f"  - Windows: {stats['windows']}")
print(f"  - Doors: {stats['doors']}")
print(f"  - Balconies: {stats['balconies']}")
print(f"  - Chimneys: {stats['chimneys']}")
print(f"  - Dormers: {stats['dormers']}")
```

### Example 4: Integration with Classification Refinement

```python
from ign_lidar.core.modules.classification_refinement import refine_classification

# Prepare features
features = {
    'ndvi': ndvi_array,
    'height': height_array,
    'planarity': planarity_array,
    'verticality': verticality_array,
    'normals': normals_array,
    'linearity': linearity_array,
    'anisotropy': anisotropy_array,
    'curvature': curvature_array,
    'intensity': intensity_array,
    'points': xyz_coordinates
}

# Refine classification with LOD2 building detection
refined_labels, stats = refine_classification(
    labels=initial_labels,
    features=features,
    ground_truth_data={'building_mask': building_polygons},
    lod_level='LOD2',  # Determines building detection mode
    config=None
)

# Or with LOD3 detection
refined_labels, stats = refine_classification(
    labels=initial_labels,
    features=features,
    ground_truth_data={'building_mask': building_polygons},
    lod_level='LOD3',  # Uses LOD3 mode for detailed detection
    config=None
)
```

### Example 5: Integration with AdvancedClassifier

```python
from ign_lidar.core.modules.advanced_classification import AdvancedClassifier

# Create classifier with LOD2 building detection
classifier = AdvancedClassifier(
    use_ground_truth=True,
    use_ndvi=True,
    use_geometric=True,
    building_detection_mode='lod2'  # Use LOD2 mode
)

# Classify points
labels = classifier.classify_points(
    points=xyz_array,
    ground_truth_features=gdf_dict,
    ndvi=ndvi_array,
    height=height_array,
    normals=normals_array,
    planarity=planarity_array,
    curvature=curvature_array,
    intensity=intensity_array
)

# Or use LOD3 mode for detailed detection
classifier_lod3 = AdvancedClassifier(
    building_detection_mode='lod3'
)
```

## 🔍 Detection Mode Comparison

| Feature               | ASPRS Mode   | LOD2 Mode       | LOD3 Mode              |
| --------------------- | ------------ | --------------- | ---------------------- |
| **Output Classes**    | 1 (Building) | 4 (Wall, Roofs) | 9 (Elements + Details) |
| **Wall Detection**    | ✅ Yes       | ✅ Yes          | ✅ Yes                 |
| **Roof Detection**    | ✅ Yes       | ✅ Typed        | ✅ Typed               |
| **Roof Types**        | ❌ No        | ✅ Yes          | ✅ Yes                 |
| **Window Detection**  | ❌ No        | ❌ No           | ✅ Yes                 |
| **Door Detection**    | ❌ No        | ❌ No           | ✅ Yes                 |
| **Balcony Detection** | ❌ No        | ❌ No           | ✅ Yes                 |
| **Detail Detection**  | ❌ No        | ❌ No           | ✅ Yes                 |
| **Uses Intensity**    | ❌ No        | ❌ No           | ✅ Yes                 |
| **Uses Curvature**    | ❌ No        | ❌ No           | ✅ Yes                 |
| **Spatial Analysis**  | ❌ No        | ❌ No           | ✅ Yes                 |
| **Speed**             | ⚡⚡⚡ Fast  | ⚡⚡ Medium     | ⚡ Slower              |
| **Accuracy**          | Good         | Better          | Best                   |

## ⚙️ Configuration Options

### BuildingDetectionConfig Parameters

#### Common (All Modes)

```python
config.min_height = 2.5              # Minimum building height (m)
config.max_height = 200.0            # Maximum building height (m)
config.min_planarity = 0.5           # Minimum planarity threshold
config.use_ground_truth = True       # Use ground truth polygons
config.ground_truth_priority = True  # Ground truth overrides detection
```

#### ASPRS Mode Specific

```python
config.wall_verticality_min = 0.65   # Wall detection threshold
config.wall_planarity_min = 0.5      # Wall flatness threshold
config.roof_horizontality_min = 0.80 # Roof detection threshold
config.anisotropy_min = 0.45         # Structure detection threshold
config.linearity_edge_min = 0.35     # Edge detection threshold
```

#### LOD2 Mode Specific

```python
config.wall_verticality_min = 0.70   # Stricter wall detection
config.roof_planarity_min = 0.70     # Stricter roof detection
config.wall_score_min = 0.35         # Combined wall metric
config.roof_score_min = 0.50         # Combined roof metric
config.detect_flat_roofs = True      # Enable flat roof detection
config.detect_sloped_roofs = True    # Enable sloped roof detection
config.separate_walls_roofs = True   # Separate walls from roofs
```

#### LOD3 Mode Specific

```python
config.wall_verticality_min = 0.75   # Very strict wall detection
config.detect_windows = True         # Enable window detection
config.detect_doors = True           # Enable door detection
config.detect_balconies = True       # Enable balcony detection
config.detect_chimneys = True        # Enable chimney detection
config.detect_dormers = True         # Enable dormer detection
config.opening_intensity_threshold = 0.25  # Low intensity for glass
config.opening_depth_threshold = 0.15      # Recessed opening depth
config.balcony_linearity_min = 0.35        # Balcony edge detection
config.chimney_height_min = 1.5            # Minimum chimney height
```

## 📊 Detection Statistics

Each mode returns different statistics:

### ASPRS Mode Stats

```python
{
    'ground_truth': 15420,  # Points from ground truth
    'walls': 8930,          # Wall points detected
    'roofs': 6210,          # Roof points detected
    'structured': 340,      # Structured element points
    'edges': 180,           # Edge points detected
    'total': 31080          # Total building points
}
```

### LOD2 Mode Stats

```python
{
    'ground_truth': 15420,   # Points from ground truth
    'walls': 18650,          # Wall points (class 0)
    'flat_roofs': 4230,      # Flat roof points (class 1)
    'sloped_roofs': 5890,    # Sloped roof points (class 2/3)
    'structured': 280,       # Additional structured points
    'edges': 160,            # Edge points
    'total_building': 29210  # Total building element points
}
```

### LOD3 Mode Stats

```python
{
    'ground_truth': 15420,   # Points from ground truth
    'walls': 16840,          # Wall points (class 0)
    'flat_roofs': 4120,      # Flat roof points (class 1)
    'sloped_roofs': 5670,    # Sloped roof points (class 2/3)
    'windows': 1240,         # Window points (class 13)
    'doors': 320,            # Door points (class 14)
    'balconies': 180,        # Balcony points (class 15)
    'chimneys': 45,          # Chimney points (class 18)
    'dormers': 92,           # Dormer points (class 20)
    'total_building': 28507  # Total architectural points
}
```

## 🧪 Testing

Run the test suite:

```bash
# All building detection tests
pytest tests/test_building_detection_modes.py -v

# Specific test classes
pytest tests/test_building_detection_modes.py::TestBuildingDetector -v
pytest tests/test_building_detection_modes.py::TestDetectionStrategies -v

# With coverage
pytest tests/test_building_detection_modes.py --cov=ign_lidar.core.modules.building_detection
```

## 🔄 Migration Guide

### For Existing Code Using Simple Building Detection

**Before:**

```python
from ign_lidar.core.modules.classification_refinement import refine_building_classification

refined, n_changed = refine_building_classification(
    labels=labels,
    height=height,
    planarity=planarity,
    verticality=verticality,
    ground_truth_mask=building_mask
)
```

**After (Backward Compatible):**

```python
from ign_lidar.core.modules.classification_refinement import refine_building_classification

# Still works! Defaults to LOD2 mode
refined, n_changed = refine_building_classification(
    labels=labels,
    height=height,
    planarity=planarity,
    verticality=verticality,
    ground_truth_mask=building_mask,
    mode='lod2'  # Optional: specify mode explicitly
)
```

### For New Projects

Use the new building detection module directly:

```python
from ign_lidar.core.modules.building_detection import detect_buildings_multi_mode

# Choose the mode that fits your use case
refined, stats = detect_buildings_multi_mode(
    labels=labels,
    features=features_dict,
    mode='asprs',  # or 'lod2' or 'lod3'
    ground_truth_mask=building_mask
)
```

## 🎨 Class Mapping

### ASPRS Mode Output

| Class    | Name                | Code |
| -------- | ------------------- | ---- |
| Building | All building points | 6    |

### LOD2 Mode Output

| Class        | Name                       | Code |
| ------------ | -------------------------- | ---- |
| Wall         | Vertical building surfaces | 0    |
| Roof (Flat)  | Horizontal roof surfaces   | 1    |
| Roof (Gable) | Sloped gable roofs         | 2    |
| Roof (Hip)   | Sloped hip roofs           | 3    |

### LOD3 Mode Output

| Class        | Name                       | Code |
| ------------ | -------------------------- | ---- |
| Wall         | Vertical building surfaces | 0    |
| Roof (Flat)  | Horizontal roof surfaces   | 1    |
| Roof (Gable) | Sloped gable roofs         | 2    |
| Roof (Hip)   | Sloped hip roofs           | 3    |
| Window       | Glass openings in facades  | 13   |
| Door         | Entry/exit openings        | 14   |
| Balcony      | Protruding platforms       | 15   |
| Chimney      | Vertical roof structures   | 18   |
| Dormer       | Roof protrusions           | 20   |

## 🚀 Performance Considerations

### Speed vs Accuracy Trade-off

- **ASPRS Mode**: Fastest, good for large datasets
- **LOD2 Mode**: Balanced speed and detail
- **LOD3 Mode**: Slowest, highest detail (use for focused areas)

### Memory Usage

- All modes: Similar memory footprint
- LOD3 requires additional features (intensity, curvature)

### Recommendations

1. Use **ASPRS mode** for:

   - Initial data exploration
   - Large-scale processing
   - Standard LiDAR workflows

2. Use **LOD2 mode** for:

   - Building reconstruction training
   - Urban modeling
   - Roof type analysis

3. Use **LOD3 mode** for:
   - Detailed architectural modeling
   - Heritage building documentation
   - Window/door extraction
   - High-detail urban areas

## 📚 References

### Related Documentation

- `BUILDING_CLASSIFICATION_IMPROVEMENTS.md` - Previous building detection enhancements
- `CLASSIFICATION_GRAMMAR_COMPLETE_SUMMARY.md` - Grammar-based building parsing
- `CLASSIFICATION_REFERENCE.md` - General classification reference

### Code References

- `ign_lidar/core/modules/building_detection.py` - Main module
- `ign_lidar/core/modules/classification_refinement.py` - Integration
- `ign_lidar/core/modules/advanced_classification.py` - Advanced classifier integration

## ✅ Summary

The building detection system has been successfully upgraded with:

1. ✅ **Three detection modes** (ASPRS, LOD2, LOD3)
2. ✅ **Five detection strategies** (ground truth, walls, roofs, structure, edges)
3. ✅ **Mode-specific configurations** with optimized thresholds
4. ✅ **Backward compatibility** with existing code
5. ✅ **Comprehensive tests** for all modes
6. ✅ **Integration** with existing classification modules
7. ✅ **Detailed documentation** with examples

The system is now ready for production use across different building detection scenarios!

---

**Author**: GitHub Copilot  
**Date**: October 15, 2025  
**Status**: ✅ Complete
