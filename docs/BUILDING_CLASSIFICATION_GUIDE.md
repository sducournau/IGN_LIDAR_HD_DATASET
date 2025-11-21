# Building Classification (LOD3) - User Guide

**Version:** 3.4.0 (Phase 2.4)  
**Status:** Production Ready  
**Date:** October 2025

---

## Overview

The Building Classification system provides advanced architectural feature detection for buildings, enabling detailed LOD3 (Level of Detail 3) classification. This includes:

- **Roof Type Detection** - Classify roofs as flat, gabled, hipped, or complex
- **Chimney Detection** - Identify chimneys, antennas, and superstructures
- **Balcony Detection** - Detect balconies, overhangs, and canopies

These features complement the existing facade-based building classification to provide comprehensive building analysis.

---

## Quick Start

### 1. Enable in Configuration

Add to your YAML configuration:

```yaml
advanced:
  classification:
    building:
      enable_roof_detection: true
      enable_chimney_detection: true
      enable_balcony_detection: true
```

### 2. Run Processing

```bash
ign-lidar process --config my_config.yaml
```

That's it! The LOD3 features will be automatically detected and classified.

---

## Configuration

### Basic Configuration

```yaml
advanced:
  classification:
    building:
      # Feature toggles
      enable_roof_detection: true
      enable_chimney_detection: true
      enable_balcony_detection: true

      # Basic parameters (use defaults for most cases)
      roof_flat_threshold: 15.0
      chimney_min_height_above_roof: 1.0
      balcony_min_distance_from_facade: 0.5
```

### Building Type Presets

#### Residential Buildings (Default)

Best for standard residential areas:

```yaml
advanced:
  classification:
    building:
      # Use defaults - optimized for residential
      enable_roof_detection: true
      enable_chimney_detection: true
      enable_balcony_detection: true
```

#### Urban High-Density

For urban areas with smaller architectural features:

```yaml
advanced:
  classification:
    building:
      enable_roof_detection: true
      enable_chimney_detection: true
      enable_balcony_detection: true

      # Stricter thresholds for smaller features
      roof_flat_threshold: 10.0 # More flat roofs
      chimney_min_height_above_roof: 0.5 # Smaller chimneys
      chimney_min_points: 15
      balcony_min_distance_from_facade: 0.3 # Smaller balconies
      balcony_min_points: 20
```

#### Industrial Buildings

For industrial areas (large chimneys, no balconies):

```yaml
advanced:
  classification:
    building:
      enable_roof_detection: true
      enable_chimney_detection: true
      enable_balcony_detection: false # No balconies

      # Large features only
      roof_flat_threshold: 20.0 # Mostly flat
      chimney_min_height_above_roof: 2.0 # Large chimneys
      chimney_min_points: 40
      chimney_max_height_above_roof: 20.0 # Industrial stacks
```

#### Historic Buildings

For historic districts with architectural details:

```yaml
advanced:
  classification:
    building:
      enable_roof_detection: true
      enable_chimney_detection: true
      enable_balcony_detection: true

      # Sensitive to details
      roof_flat_threshold: 25.0 # Complex roofs
      roof_dbscan_min_samples: 40 # More detail
      chimney_min_height_above_roof: 0.8 # Smaller features
      balcony_min_distance_from_facade: 0.3 # Ornate balconies
      balcony_confidence_threshold: 0.4 # More lenient
```

---

## Python API

### Method 1: Using Config

```python
from ign_lidar import LiDARProcessor
from ign_lidar.config import Config

# Load configuration
config = Config.from_yaml('my_config.yaml')

# Or create programmatically
config = Config.preset('lod3_buildings')
config.input_dir = '/data/tiles'
config.output_dir = '/data/output'

# Process
processor = LiDARProcessor(config)
processor.process_directory()
```

### Method 2: Using BuildingConfig

```python
from ign_lidar import LiDARProcessor
from ign_lidar.config import Config, BuildingConfig

# Create base config
config = Config(
    input_dir='/data/tiles',
    output_dir='/data/output',
    mode='lod3',
    processing_mode='both',
)

# Add enhanced building config
enhanced_config = BuildingConfig.preset_residential()

# Or customize
enhanced_config = BuildingConfig(
    enable_roof_detection=True,
    enable_chimney_detection=True,
    enable_balcony_detection=False,  # Disable balconies
    roof_flat_threshold=12.0,
)

# Attach to main config
config.advanced.classification = {
    'enhanced_building': enhanced_config.to_dict()
}

# Process
processor = LiDARProcessor(config)
processor.process_directory()
```

### Method 3: Direct Classifier Usage (Advanced)

```python
from ign_lidar.core.classification.building import (
    BuildingFacadeClassifier,
    BuildingClassifierConfig,
)

# Create facade classifier with LOD3 features
classifier = BuildingFacadeClassifier(
    enable_enhanced_lod3=True,
    enhanced_building_config={
        'enable_roof_detection': True,
        'enable_chimney_detection': True,
        'enable_balcony_detection': True,
        'roof_flat_threshold': 15.0,
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

# Check results
print(f"Roof type: {stats['roof_type_enhanced']}")
print(f"Chimneys detected: {stats['num_chimneys']}")
print(f"Balconies detected: {stats['num_balconies']}")
```

---

## Output

### Point Cloud Classifications

The building classifier adds new class labels:

| Class | Name         | Description                     |
| ----- | ------------ | ------------------------------- |
| 63    | Roof Flat    | Flat roof surfaces              |
| 64    | Roof Gabled  | Gabled/pitched roofs (2 planes) |
| 65    | Roof Hipped  | Hipped roofs (3-4 planes)       |
| 66    | Roof Complex | Complex roofs (5+ planes)       |
| 67    | Roof Ridge   | Ridge lines                     |
| 68    | Roof Edge    | Roof edges/eaves                |
| 69    | Dormer       | Dormer windows                  |
| 68-69 | Chimney      | Chimneys (reusing codes)        |
| 70-72 | Balcony      | Balconies (pending allocation)  |

### Statistics

Per-building statistics include:

```python
{
    'enhanced_lod3_enabled': True,
    'roof_type_enhanced': 'gabled',  # flat, gabled, hipped, complex
    'num_chimneys': 2,
    'num_balconies': 4,
    'chimney_points': 150,
    'balcony_points': 320,
    # ... other statistics ...
}
```

### LAZ Extra Dimensions

Add these to your config for enriched LAZ output:

```yaml
output:
  laz_extra_dims:
    - name: roof_type
      type: uint8
      description: "Roof type (0=unknown, 1=flat, 2=gabled, 3=hipped, 4=complex)"
    - name: is_chimney
      type: uint8
      description: "Chimney flag (0=no, 1=yes)"
    - name: is_balcony
      type: uint8
      description: "Balcony flag (0=no, 1=yes)"
    - name: balcony_confidence
      type: float32
      description: "Balcony detection confidence score"
```

---

## Parameter Tuning

### Roof Detection

**`roof_flat_threshold`** - Maximum angle for flat roof classification (degrees)

- **Lower** = Stricter flat roof definition
  - 10.0: Very flat roofs only (many multi-story buildings)
  - 15.0: Standard (default for residential)
  - 20.0: Moderately pitched still considered flat
- **Use lower** for urban areas with flat roofs
- **Use higher** for areas with varied roof types

**`roof_dbscan_min_samples`** - Minimum points per roof plane

- **Lower** = Detect smaller roof features
  - 20: Fine detail detection
  - 30: Standard (default)
  - 50: Large planes only
- **Use lower** for complex roofs
- **Use higher** for simple roofs

### Chimney Detection

**`chimney_min_height_above_roof`** - Minimum chimney height (meters)

- **Lower** = Detect smaller features (more false positives)
  - 0.5: Small chimneys, vents (urban)
  - 1.0: Standard chimneys (default)
  - 1.5: Tall chimneys only
  - 2.0: Industrial stacks
- **Use lower** in urban areas
- **Use higher** in industrial areas

**`chimney_min_points`** - Minimum points for chimney

- **Lower** = Detect smaller features
  - 15: Small features (urban)
  - 20: Standard (default)
  - 30: Robust detection
  - 40: Large features only (industrial)

### Balcony Detection

**`balcony_min_distance_from_facade`** - Minimum protrusion (meters)

- **Lower** = Detect smaller balconies (more false positives)
  - 0.3: Small balconies (urban/historic)
  - 0.5: Standard balconies (default)
  - 0.8: Large balconies/terraces only
- **Use lower** in dense urban areas
- **Use higher** for large residential buildings

**`balcony_confidence_threshold`** - Minimum confidence score

- **Lower** = More detections (more false positives)
  - 0.4: Lenient (historic buildings)
  - 0.5: Standard (default)
  - 0.6: Conservative
  - 0.7: Strict (high confidence only)

---

## Performance

### Computational Overhead

- **Basic classification:** 100ms per building (baseline)
- **+ Roof detection:** +50-100ms (~10-15%)
- **+ Chimney detection:** +50-100ms (~10-15%)
- **+ Balcony detection:** +50-150ms (~10-20%)
- **Total (all enabled):** ~300-600ms per building

**Optimization tips:**

- Enable GPU acceleration (`use_gpu: true`)
- Disable unused detectors
- Increase `min_points` thresholds
- Process in batches

### Memory Usage

- Small buildings (<500 points): <10MB
- Medium buildings (500-2000 points): 10-50MB
- Large buildings (>2000 points): 50-150MB

### Scalability

- Linear scaling with number of buildings
- 1000 buildings: ~10 minutes (all features enabled)

---

## Troubleshooting

### No Features Detected

**Problem:** Building classifier enabled but no architectural features detected

**Solutions:**

1. Lower detection thresholds:

   ```yaml
   chimney_min_height_above_roof: 0.5 # Was 1.0
   balcony_min_distance_from_facade: 0.3 # Was 0.5
   ```

2. Reduce minimum points:

   ```yaml
   chimney_min_points: 15 # Was 20
   balcony_min_points: 20 # Was 25
   ```

3. Check feature quality (ensure normals, verticality computed)

### Too Many False Positives

**Problem:** Detecting features that aren't real

**Solutions:**

1. Increase confidence threshold:

   ```yaml
   balcony_confidence_threshold: 0.6 # Was 0.5
   ```

2. Increase minimum points:

   ```yaml
   chimney_min_points: 30 # Was 20
   balcony_min_points: 30 # Was 25
   ```

3. Stricter height/distance thresholds:
   ```yaml
   chimney_min_height_above_roof: 1.5 # Was 1.0
   balcony_min_distance_from_facade: 0.6 # Was 0.5
   ```

### Slow Performance

**Problem:** Processing takes too long

**Solutions:**

1. Enable GPU:

   ```yaml
   processor:
     use_gpu: true
   ```

2. Disable unused detectors:

   ```yaml
   building:
     enable_balcony_detection: false # If not needed
   ```

3. Increase thresholds (fewer detections = faster):
   ```yaml
   chimney_min_points: 40
   balcony_min_points: 40
   ```

### Import Errors

**Problem:** `BuildingClassifier not found`

**Solution:** Ensure Phase 2 modules are installed:

```bash
# Check version
ign-lidar --version  # Should be >= 3.4.0

# Reinstall if needed
pip install --upgrade ign-lidar-hd
```

---

## Examples

### Complete Configuration File

See `examples/production/config_lod3_enhanced_buildings.yaml` for a complete, documented configuration file.

### Python Script

```python
#!/usr/bin/env python3
"""
Example: Process LiDAR tiles with enhanced building classification.
"""

from pathlib import Path
from ign_lidar import LiDARProcessor
from ign_lidar.config import Config, BuildingConfig

# Setup
input_dir = Path('/data/lidar_tiles')
output_dir = Path('/data/output_lod3')

# Create configuration
config = Config(
    input_dir=str(input_dir),
    output_dir=str(output_dir),
    mode='lod3',
    processing_mode='both',
    use_gpu=True,
)

# Add enhanced building detection
enhanced_config = BuildingConfig.preset_residential()
config.advanced.classification = {
    'enhanced_building': enhanced_config.to_dict()
}

# Process
print(f"Processing tiles from {input_dir}")
print(f"Output will be saved to {output_dir}")
print(f"Enhanced LOD3 features: ENABLED")

processor = LiDARProcessor(config)
results = processor.process_directory()

print(f"\nProcessing complete!")
print(f"Tiles processed: {len(results)}")
```

---

## See Also

- [Configuration Guide](CONFIG_GUIDE.md)
- [Feature Computation Guide](docs/docs/features/README.md)
- [API Documentation](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/)
- [Phase 2.4 Integration Summary](PHASE_2.4_PIPELINE_INTEGRATION_SUMMARY.md)

---

## Support

For issues or questions:

- GitHub Issues: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues
- Documentation: https://sducournau.github.io/IGN_LIDAR_HD_DATASET/

---

**Last Updated:** October 2025  
**Version:** 3.4.0
