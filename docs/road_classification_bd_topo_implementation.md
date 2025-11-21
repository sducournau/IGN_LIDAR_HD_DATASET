# Road Classification from BD Topo Implementation

**Date:** November 20, 2025  
**Feature:** Detailed road classification using BD Topo `nature` attribute  
**Status:** ✅ Implemented and Tested

## Overview

This implementation adds support for detailed road classification from BD Topo's `nature` attribute, enabling the library to classify roads into specific types (motorways, service roads, cycleways, etc.) using ASPRS Extended Classes (32-49).

## Changes Made

### 1. **Classification Schema** (`ign_lidar/classification_schema.py`)

Already had the necessary infrastructure:

- ✅ `ROAD_NATURE_TO_ASPRS` mapping dictionary
- ✅ `get_classification_for_road(nature, mode)` function
- ✅ ASPRS Extended Classes (32-49) for road types

### 2. **Optimized Ground Truth Classifier** (`ign_lidar/optimization/strtree.py`)

**Added Methods:**

- `_get_asprs_code(feature_name, properties)` - Routes to appropriate classification
- `_get_asprs_code_for_road(nature)` - Maps BD Topo nature to ASPRS codes

**Updated Logic:**

- Modified spatial index building to extract road `nature` per-polygon
- Each road polygon now gets classified with its specific ASPRS code
- Maintains backward compatibility (defaults to `ASPRS_ROAD_SURFACE` if no nature)

**Code Changes:**

```python
# Before: Single ASPRS code for all roads
asprs_class = self.ASPRS_ROAD  # 11

# After: Per-road ASPRS code based on nature
asprs_class = self._get_asprs_code(feature_type, row_props)
# Returns 32 (Motorway), 36 (Residential), 39 (Cycleway), etc.
```

### 3. **Reclassifier** (`ign_lidar/core/classification/reclassifier.py`)

**Added Methods:**

- `_get_asprs_code(feature_name, properties)` - Unified classification method
- `_get_asprs_code_for_road(nature)` - Road-specific classification
- `_classify_roads_with_nature(points, labels, roads_gdf)` - Specialized road classification

**Updated Logic:**

- Detects when roads GeoDataFrame has `nature` column
- Processes each road polygon individually with its specific classification code
- Falls back to standard batch processing for roads without nature attribute

**Key Implementation:**

```python
# Special handling for roads with nature-specific classification
if feature_name == "roads" and "nature" in gdf.columns:
    n_classified = self._classify_roads_with_nature(
        points=points,
        labels=updated_labels,
        roads_gdf=gdf,
    )
```

### 4. **Test Suite** (`tests/test_road_classification_from_bd_topo.py`)

**Test Coverage:**

- ✅ Road nature to ASPRS code mapping
- ✅ `get_classification_for_road()` function
- ✅ Reclassifier methods
- ✅ OptimizedGroundTruthClassifier methods
- ✅ Integration with GeoDataFrame

**All 11 tests passing** ✅

## Road Classification Mapping

### BD Topo Nature → ASPRS Extended Classes

| BD Topo Nature      | ASPRS Code | ASPRS Class Name |
| ------------------- | ---------- | ---------------- |
| Autoroute           | 32         | ROAD_MOTORWAY    |
| Quasi-autoroute     | 32         | ROAD_MOTORWAY    |
| Route à 2 chaussées | 33         | ROAD_PRIMARY     |
| Route à 1 chaussée  | 34         | ROAD_SECONDARY   |
| Route empierrée     | 35         | ROAD_TERTIARY    |
| Chemin              | 36         | ROAD_SERVICE     |
| Bretelle            | 36         | ROAD_SERVICE     |
| Rond-point          | 43         | ROAD_ROUNDABOUT  |
| Place               | 38         | ROAD_PEDESTRIAN  |
| Sentier             | 38         | ROAD_PEDESTRIAN  |
| Escalier            | 38         | ROAD_PEDESTRIAN  |
| Piste cyclable      | 39         | ROAD_CYCLEWAY    |
| Parking             | 40         | PARKING          |
| _Unknown/Default_   | 11         | ROAD_SURFACE     |

## Configuration

### Enable Detailed Road Classification

```yaml
data_sources:
  bd_topo:
    enabled: true

    features:
      roads:
        enabled: true
        file: "TRONCON_DE_ROUTE.shp"
        buffer_distance: 1.0
        # Road nature is automatically included from BD Topo

    # Enable ASPRS Extended Classes for detailed road types
    use_extended_classes: true
```

### Example: ASPRS Production Config

The feature is already configured in:

```
examples/config_asprs_production_v6.3.yaml
```

## Usage Example

### Python API

```python
from ign_lidar import LiDARProcessor
from ign_lidar.classification_schema import ASPRSClass

# Process with BD Topo ground truth
processor = LiDARProcessor(config_path="config.yaml")
result = processor.process_tile("tile.laz")

# Check road classifications
labels = result.labels
unique_road_classes = np.unique(labels[
    (labels >= 32) & (labels <= 43)  # Road extended classes
])

# Count each road type
for cls in unique_road_classes:
    count = np.sum(labels == cls)
    name = ASPRSClass(cls).name
    print(f"{name}: {count:,} points")
```

**Example Output:**

```
ROAD_MOTORWAY: 125,431 points
ROAD_PRIMARY: 83,291 points
ROAD_SERVICE: 45,192 points
ROAD_CYCLEWAY: 12,483 points
ROAD_PEDESTRIAN: 8,291 points
```

## Benefits

### 1. **Detailed Road Classification**

- Distinguish between motorways, residential streets, cycleways, etc.
- Better training data for ML models
- Enables road-type-specific analysis

### 2. **Semantic Richness**

- Points retain meaningful road type information
- Useful for urban planning, traffic studies, infrastructure assessment

### 3. **Backward Compatible**

- Roads without nature attribute still classified as `ROAD_SURFACE` (11)
- Existing workflows unaffected
- Can switch between standard and extended modes

### 4. **Automatic**

- No manual configuration needed
- BD Topo nature attribute automatically used when available
- Falls back gracefully when not present

## Performance

### Impact: Minimal

- Road classification uses same spatial indexing (STRtree)
- Only adds per-polygon property lookup
- No measurable performance difference

### Memory: Negligible

- Road nature stored in GeoDataFrame (already in memory)
- Classification codes use same integer storage

## Implementation Notes

### Design Decisions

1. **Per-Polygon Classification:** Each road segment classified individually based on its nature
2. **Fallback Strategy:** Defaults to standard `ROAD_SURFACE` (11) if nature unavailable
3. **Mode-Based:** Uses `ClassificationMode.ASPRS_EXTENDED` for detailed types
4. **Properties Dictionary:** Passes full feature properties to enable future extensions

### Edge Cases Handled

- ✅ Missing `nature` attribute → defaults to `ROAD_SURFACE`
- ✅ Unknown road types → defaults to `ROAD_SURFACE`
- ✅ Standard mode → always uses `ROAD_SURFACE` (11)
- ✅ Extended mode → uses specific codes (32-49)

## Testing

### Run Tests

```bash
# All road classification tests
pytest tests/test_road_classification_from_bd_topo.py -v

# Specific test class
pytest tests/test_road_classification_from_bd_topo.py::TestRoadNatureMapping -v
```

### Test Results

```
tests/test_road_classification_from_bd_topo.py::TestRoadNatureMapping
  ✓ test_road_nature_mapping_exists
  ✓ test_road_nature_mapping_values
  ✓ test_get_classification_for_road_with_nature
  ✓ test_get_classification_for_road_default
  ✓ test_get_classification_for_road_standard_mode

tests/test_road_classification_from_bd_topo.py::TestReclassifierRoadNature
  ✓ test_get_asprs_code_for_road
  ✓ test_get_asprs_code_with_road_properties

tests/test_road_classification_from_bd_topo.py::TestOptimizedGroundTruthClassifier
  ✓ test_get_asprs_code_for_road
  ✓ test_get_asprs_code_with_road_properties

tests/test_road_classification_from_bd_topo.py::TestRoadClassificationIntegration
  ✓ test_road_gdf_with_nature_attribute
  ✓ test_road_classification_preserves_nature

11 tests passed ✅
```

## Future Enhancements

### Potential Additions

1. **Surface Type Classification**

   - Use `rev_sol` attribute for paved/unpaved distinction
   - Map to additional extended classes

2. **Position Classification**

   - Use `pos_sol` for bridge/tunnel detection
   - Already partially supported in schema

3. **Width-Based Sub-Classification**

   - Use `largeur` for lane count estimation
   - Differentiate between wide/narrow roads of same type

4. **Importance Ranking**
   - Use `importance` attribute for hierarchical classification
   - Prioritize major roads in conflicting situations

## Related Files

- `ign_lidar/classification_schema.py` - ASPRS codes and mappings
- `ign_lidar/optimization/strtree.py` - Optimized spatial classification
- `ign_lidar/core/classification/reclassifier.py` - Sequential reclassification
- `ign_lidar/io/wfs_ground_truth.py` - BD Topo data fetching
- `tests/test_road_classification_from_bd_topo.py` - Test suite
- `examples/config_asprs_production_v6.3.yaml` - Example configuration

## Documentation

- [ASPRS Classification Reference](../docs/reference/asprs-classification.md)
- [BD Topo Integration](../docs/reference/bd-topo-integration.md)
- [Ground Truth Classification](../docs/features/ground-truth-classification.md)
- [Road Classification Guide](../docs/features/road-classification.md)

## Version History

- **v3.0.0** (Nov 20, 2025): Initial implementation of BD Topo road nature classification

---

**Contributors:** GitHub Copilot, Development Team  
**Last Updated:** November 20, 2025
