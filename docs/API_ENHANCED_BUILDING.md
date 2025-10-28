# Enhanced Building Classification API Reference

**Version:** 3.4.0  
**Module:** `ign_lidar.core.classification.building`

---

## Overview

This document provides API documentation for the Enhanced Building Classification system (Phase 2.4), which adds advanced architectural feature detection to the existing facade-based building classification.

---

## Configuration Classes

### `EnhancedBuildingConfig`

**Module:** `ign_lidar.config.enhanced_building`

Configuration dataclass for enhanced LOD3 building classification.

#### Class Definition

```python
@dataclass
class EnhancedBuildingConfig:
    """Configuration for enhanced LOD3 building classification."""

    # Feature toggles
    enable_roof_detection: bool = True
    enable_chimney_detection: bool = True
    enable_balcony_detection: bool = True

    # Roof detection parameters
    roof_flat_threshold: float = 15.0
    roof_dbscan_eps: float = 0.5
    roof_dbscan_min_samples: int = 30

    # Chimney detection parameters
    chimney_min_height_above_roof: float = 1.0
    chimney_max_height_above_roof: float = 10.0
    chimney_min_points: int = 20
    chimney_dbscan_eps: float = 0.3

    # Balcony detection parameters
    balcony_min_distance_from_facade: float = 0.5
    balcony_max_distance_from_facade: float = 3.0
    balcony_min_points: int = 25
    balcony_height_tolerance: float = 0.3
    balcony_dbscan_eps: float = 0.4
    balcony_confidence_threshold: float = 0.5
```

#### Parameters

##### Feature Toggles

- **`enable_roof_detection`** (`bool`, default: `True`)

  - Enable roof type classification (flat, gabled, hipped, complex)

- **`enable_chimney_detection`** (`bool`, default: `True`)

  - Enable chimney and superstructure detection

- **`enable_balcony_detection`** (`bool`, default: `True`)
  - Enable balcony and overhang detection

##### Roof Detection Parameters

- **`roof_flat_threshold`** (`float`, default: `15.0`)

  - Maximum angle (degrees) for flat roof classification
  - Range: 5.0-30.0
  - Lower = stricter flat definition

- **`roof_dbscan_eps`** (`float`, default: `0.5`)

  - DBSCAN clustering distance for roof segmentation (meters)
  - Range: 0.3-1.0
  - Higher = larger clusters

- **`roof_dbscan_min_samples`** (`int`, default: `30`)
  - Minimum points per roof plane cluster
  - Range: 10-100
  - Lower = detect smaller planes

##### Chimney Detection Parameters

- **`chimney_min_height_above_roof`** (`float`, default: `1.0`)

  - Minimum chimney height above roof (meters)
  - Range: 0.5-3.0
  - Lower = detect smaller chimneys

- **`chimney_max_height_above_roof`** (`float`, default: `10.0`)

  - Maximum chimney height above roof (meters)
  - Range: 5.0-30.0
  - Higher = allow industrial stacks

- **`chimney_min_points`** (`int`, default: `20`)

  - Minimum points for chimney cluster
  - Range: 10-50
  - Lower = detect smaller features

- **`chimney_dbscan_eps`** (`float`, default: `0.3`)
  - DBSCAN clustering distance for chimneys (meters)
  - Range: 0.2-0.5
  - Controls chimney grouping

##### Balcony Detection Parameters

- **`balcony_min_distance_from_facade`** (`float`, default: `0.5`)

  - Minimum balcony protrusion from facade (meters)
  - Range: 0.3-1.0
  - Lower = detect smaller balconies

- **`balcony_max_distance_from_facade`** (`float`, default: `3.0`)

  - Maximum balcony protrusion from facade (meters)
  - Range: 2.0-5.0
  - Higher = allow large terraces

- **`balcony_min_points`** (`int`, default: `25`)

  - Minimum points for balcony cluster
  - Range: 15-50
  - Lower = detect smaller balconies

- **`balcony_height_tolerance`** (`float`, default: `0.3`)

  - Vertical tolerance for balcony floor plane (meters)
  - Range: 0.2-0.5
  - Controls floor flatness requirement

- **`balcony_dbscan_eps`** (`float`, default: `0.4`)

  - DBSCAN clustering distance for balconies (meters)
  - Range: 0.3-0.6
  - Controls balcony grouping

- **`balcony_confidence_threshold`** (`float`, default: `0.5`)
  - Minimum confidence score for balcony detection
  - Range: 0.3-0.8
  - Higher = fewer false positives

#### Class Methods

##### `preset_residential()`

```python
@classmethod
def preset_residential(cls) -> 'EnhancedBuildingConfig':
    """Create preset for residential buildings."""
```

Returns configuration optimized for standard residential buildings.

**Use for:**

- Single-family homes
- Multi-family residential
- Suburban areas

**Returns:**

- `EnhancedBuildingConfig` with default residential settings

##### `preset_urban_high_density()`

```python
@classmethod
def preset_urban_high_density(cls) -> 'EnhancedBuildingConfig':
    """Create preset for urban high-density buildings."""
```

Returns configuration optimized for dense urban areas with smaller architectural features.

**Use for:**

- Urban apartments
- Dense city centers
- Multi-story buildings

**Returns:**

- `EnhancedBuildingConfig` with stricter thresholds for small features

**Key differences:**

- `roof_flat_threshold`: 10.0 (more flat roofs)
- `chimney_min_height_above_roof`: 0.5 (smaller chimneys)
- `chimney_min_points`: 15
- `balcony_min_distance_from_facade`: 0.3 (smaller balconies)
- `balcony_min_points`: 20

##### `preset_industrial()`

```python
@classmethod
def preset_industrial(cls) -> 'EnhancedBuildingConfig':
    """Create preset for industrial buildings."""
```

Returns configuration optimized for industrial buildings.

**Use for:**

- Warehouses
- Factories
- Industrial complexes

**Returns:**

- `EnhancedBuildingConfig` with large feature detection, no balconies

**Key differences:**

- `enable_balcony_detection`: False
- `roof_flat_threshold`: 20.0 (mostly flat)
- `chimney_min_height_above_roof`: 2.0 (large stacks)
- `chimney_min_points`: 40
- `chimney_max_height_above_roof`: 20.0

##### `preset_historic()`

```python
@classmethod
def preset_historic(cls) -> 'EnhancedBuildingConfig':
    """Create preset for historic buildings."""
```

Returns configuration optimized for historic districts with architectural details.

**Use for:**

- Historic city centers
- Protected buildings
- Architecturally significant areas

**Returns:**

- `EnhancedBuildingConfig` optimized for detecting architectural details

**Key differences:**

- `roof_flat_threshold`: 25.0 (complex roofs)
- `roof_dbscan_min_samples`: 40 (more detail)
- `chimney_min_height_above_roof`: 0.8 (smaller features)
- `balcony_min_distance_from_facade`: 0.3 (ornate balconies)
- `balcony_confidence_threshold`: 0.4 (more lenient)

##### `to_dict()`

```python
def to_dict(self) -> Dict[str, Any]:
    """Convert configuration to dictionary."""
```

Serializes configuration to dictionary for YAML/JSON export.

**Returns:**

- `dict` with all configuration parameters

**Example:**

```python
config = EnhancedBuildingConfig.preset_residential()
config_dict = config.to_dict()
# Use in YAML export
```

##### `from_dict()`

```python
@classmethod
def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnhancedBuildingConfig':
    """Create configuration from dictionary."""
```

Deserializes configuration from dictionary.

**Parameters:**

- `config_dict` (`dict`): Dictionary with configuration parameters

**Returns:**

- `EnhancedBuildingConfig` instance

**Raises:**

- `ValueError`: If invalid parameters provided

**Example:**

```python
config_dict = {
    'enable_roof_detection': True,
    'roof_flat_threshold': 12.0,
}
config = EnhancedBuildingConfig.from_dict(config_dict)
```

---

## Classifier Classes

### `EnhancedBuildingClassifier`

**Module:** `ign_lidar.core.classification.building.enhanced_classifier`

Main classifier for enhanced LOD3 building features.

#### Class Definition

```python
class EnhancedBuildingClassifier:
    """Enhanced building classifier with architectural feature detection."""

    def __init__(self, config: Optional[EnhancedClassifierConfig] = None):
        """Initialize enhanced building classifier."""
```

#### Parameters

- **`config`** (`EnhancedClassifierConfig`, optional)
  - Configuration for enhanced classification
  - If None, uses default configuration

#### Methods

##### `classify()`

```python
def classify(
    self,
    building_points: np.ndarray,
    building_labels: np.ndarray,
    building_features: Dict[str, np.ndarray],
    building_polygon: Polygon,
    building_id: int,
) -> EnhancedClassificationResult:
    """
    Classify building with enhanced architectural features.

    Args:
        building_points: Point cloud [N, 3] with XYZ coordinates
        building_labels: Initial facade-based labels [N]
        building_features: Dictionary with feature arrays
        building_polygon: Building footprint polygon
        building_id: Unique building identifier

    Returns:
        EnhancedClassificationResult with refined labels and metadata

    Raises:
        ValueError: If inputs are invalid
    """
```

Main classification method that processes a single building.

**Required Features:**

- `normals`: Point normals [N, 3]
- `verticality`: Verticality scores [N]
- `curvature`: Curvature values [N]
- `planarity`: Planarity scores [N] (optional but recommended)

**Returns:**

- `EnhancedClassificationResult` with:
  - Refined point labels
  - Detection metadata
  - Performance statistics

**Example:**

```python
classifier = EnhancedBuildingClassifier()

result = classifier.classify(
    building_points=points,
    building_labels=labels,
    building_features={
        'normals': normals,
        'verticality': verticality,
        'curvature': curvature,
    },
    building_polygon=polygon,
    building_id=42,
)

# Access results
refined_labels = result.labels
num_chimneys = result.metadata['num_chimneys']
roof_type = result.metadata['roof_type']
```

---

### `EnhancedClassificationResult`

**Module:** `ign_lidar.core.classification.building.enhanced_classifier`

Result container for enhanced building classification.

#### Attributes

- **`labels`** (`np.ndarray`)

  - Refined point cloud labels [N]
  - Type: uint8

- **`metadata`** (`Dict[str, Any]`)

  - Classification metadata and statistics
  - Keys:
    - `roof_type`: Detected roof type (flat, gabled, hipped, complex)
    - `num_chimneys`: Number of chimneys detected
    - `num_balconies`: Number of balconies detected
    - `chimney_points`: Total points classified as chimney
    - `balcony_points`: Total points classified as balcony
    - `processing_time_ms`: Processing time in milliseconds

- **`success`** (`bool`)

  - Whether classification succeeded
  - False if critical errors occurred

- **`error_message`** (`Optional[str]`)
  - Error message if classification failed
  - None if successful

#### Example

```python
result = classifier.classify(...)

if result.success:
    print(f"Roof type: {result.metadata['roof_type']}")
    print(f"Chimneys: {result.metadata['num_chimneys']}")
    print(f"Balconies: {result.metadata['num_balconies']}")

    # Use refined labels
    points_with_labels = np.column_stack([points, result.labels])
else:
    print(f"Classification failed: {result.error_message}")
```

---

## Integration Classes

### `BuildingFacadeClassifier`

**Module:** `ign_lidar.core.classification.building.facade_processor`

Main facade-based building classifier with enhanced LOD3 integration.

#### Constructor Parameters (Enhanced)

```python
def __init__(
    self,
    # ... existing parameters ...
    enable_enhanced_lod3: bool = False,
    enhanced_building_config: Optional[Dict[str, Any]] = None,
):
```

##### New Parameters

- **`enable_enhanced_lod3`** (`bool`, default: `False`)

  - Enable enhanced LOD3 architectural feature detection
  - Adds roof, chimney, and balcony classification

- **`enhanced_building_config`** (`Dict[str, Any]`, optional)
  - Configuration dictionary for enhanced features
  - If None, uses default configuration
  - Keys should match `EnhancedBuildingConfig` parameters

#### Usage Example

```python
from ign_lidar.core.classification.building import BuildingFacadeClassifier

# Create classifier with enhanced features
classifier = BuildingFacadeClassifier(
    enable_enhanced_lod3=True,
    enhanced_building_config={
        'enable_roof_detection': True,
        'enable_chimney_detection': True,
        'enable_balcony_detection': True,
        'roof_flat_threshold': 15.0,
        'chimney_min_height_above_roof': 1.0,
        'balcony_min_distance_from_facade': 0.5,
    }
)

# Use in classification pipeline
labels, stats = classifier.classify_single_building(
    building_id=1,
    polygon=building_polygon,
    points=point_cloud,
    heights=heights,
    labels=initial_labels,
    normals=normals,
    verticality=verticality,
    curvature=curvature,
)

# Check enhanced statistics
if stats.get('enhanced_lod3_enabled'):
    print(f"Roof type: {stats['roof_type_enhanced']}")
    print(f"Chimneys: {stats['num_chimneys']}")
    print(f"Balconies: {stats['num_balconies']}")
```

---

## Utility Functions

### `classify_building_enhanced()`

**Module:** `ign_lidar.core.classification.building`

Convenience function for enhanced building classification.

```python
def classify_building_enhanced(
    building_points: np.ndarray,
    building_labels: np.ndarray,
    building_features: Dict[str, np.ndarray],
    building_polygon: Polygon,
    building_id: int,
    config: Optional[EnhancedBuildingConfig] = None,
) -> EnhancedClassificationResult:
    """
    Classify building with enhanced features (convenience function).

    Args:
        building_points: Point cloud [N, 3]
        building_labels: Initial labels [N]
        building_features: Feature dictionary
        building_polygon: Building footprint
        building_id: Building identifier
        config: Optional configuration

    Returns:
        EnhancedClassificationResult
    """
```

Wrapper function that creates classifier and performs classification in one call.

**Example:**

```python
from ign_lidar.core.classification.building import classify_building_enhanced
from ign_lidar.config import EnhancedBuildingConfig

# Quick classification with preset
config = EnhancedBuildingConfig.preset_urban_high_density()

result = classify_building_enhanced(
    building_points=points,
    building_labels=labels,
    building_features=features,
    building_polygon=polygon,
    building_id=42,
    config=config,
)
```

---

## Type Definitions

### `EnhancedClassifierConfig`

Internal configuration type (alias for `EnhancedBuildingConfig`).

---

## Constants

### Classification Labels

```python
# Roof types (new in v3.4.0)
ROOF_FLAT = 63
ROOF_GABLED = 64
ROOF_HIPPED = 65
ROOF_COMPLEX = 66
ROOF_RIDGE = 67
ROOF_EDGE = 68

# Chimneys (reusing codes)
CHIMNEY = 68  # or 69

# Balconies (pending allocation)
BALCONY = 70  # (provisional)
```

---

## Error Handling

### Exceptions

#### `ValueError`

Raised when invalid parameters provided:

- Invalid array shapes
- Missing required features
- Out-of-range parameter values

**Example:**

```python
try:
    result = classifier.classify(
        building_points=invalid_points,  # Wrong shape
        # ...
    )
except ValueError as e:
    print(f"Invalid input: {e}")
```

#### `ImportError`

Raised if Phase 2 modules not available:

- `EnhancedBuildingClassifier` not found
- Enhanced classification dependencies missing

**Example:**

```python
try:
    from ign_lidar.core.classification.building import EnhancedBuildingClassifier
except ImportError:
    print("Enhanced building classification not available")
    print("Please upgrade to version >= 3.4.0")
```

### Graceful Degradation

The system gracefully handles missing enhanced features:

```python
# If enhanced classifier not available, facade classification continues
classifier = BuildingFacadeClassifier(
    enable_enhanced_lod3=True,  # Requested
)

# If import fails, classifier logs warning and continues without enhanced features
# No exception raised - backwards compatible
```

---

## Performance Considerations

### Computational Complexity

- **Roof detection:** O(N log N) - DBSCAN clustering
- **Chimney detection:** O(N) - Height filtering + clustering
- **Balcony detection:** O(N log N) - Distance computation + clustering

### Memory Usage

- **Roof detection:** ~O(2N) - Original points + labels
- **Chimney detection:** ~O(N) - Candidate points
- **Balcony detection:** ~O(N) - Distance arrays

### Optimization Tips

1. **GPU acceleration:** Enable `use_gpu` in main config
2. **Batch processing:** Process multiple buildings in parallel (CPU mode)
3. **Selective features:** Disable unused detectors
4. **Threshold tuning:** Increase `min_points` for faster processing

---

## Migration Guide

### From v3.3.x to v3.4.x

**Old code (v3.3.x):**

```python
classifier = BuildingFacadeClassifier()
labels, stats = classifier.classify_single_building(...)
```

**New code (v3.4.x) - Basic:**

```python
# No changes required - fully backward compatible
classifier = BuildingFacadeClassifier()
labels, stats = classifier.classify_single_building(...)
```

**New code (v3.4.x) - With enhanced features:**

```python
# Enable enhanced LOD3 classification
classifier = BuildingFacadeClassifier(
    enable_enhanced_lod3=True,
    enhanced_building_config={
        'enable_roof_detection': True,
        'enable_chimney_detection': True,
    }
)
labels, stats = classifier.classify_single_building(...)

# Check for new statistics
if 'roof_type_enhanced' in stats:
    print(f"Roof type: {stats['roof_type_enhanced']}")
```

**Configuration changes:**

```yaml
# Old (v3.3.x) - still works
processor:
  lod_level: LOD3

# New (v3.4.x) - with enhanced features
processor:
  lod_level: LOD3

advanced:
  classification:
    enhanced_building:
      enable_roof_detection: true
      enable_chimney_detection: true
      enable_balcony_detection: true
```

---

## Testing

### Unit Tests

```python
import pytest
from ign_lidar.config import EnhancedBuildingConfig

def test_config_creation():
    """Test configuration creation."""
    config = EnhancedBuildingConfig()
    assert config.enable_roof_detection == True
    assert config.roof_flat_threshold == 15.0

def test_preset_residential():
    """Test residential preset."""
    config = EnhancedBuildingConfig.preset_residential()
    assert config.roof_flat_threshold == 15.0

def test_serialization():
    """Test to_dict/from_dict."""
    config1 = EnhancedBuildingConfig()
    config_dict = config1.to_dict()
    config2 = EnhancedBuildingConfig.from_dict(config_dict)
    assert config1 == config2
```

### Integration Tests

See `tests/test_enhanced_building_integration.py` for comprehensive integration tests.

---

## See Also

- [Enhanced Building Classification User Guide](ENHANCED_BUILDING_CLASSIFICATION_GUIDE.md)
- [Configuration Guide](CONFIG_GUIDE.md)
- [Phase 2.4 Integration Summary](PHASE_2.4_PIPELINE_INTEGRATION_SUMMARY.md)
- [Main API Documentation](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/)

---

**Last Updated:** October 2025  
**Version:** 3.4.0
