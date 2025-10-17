# ASPRS Classification Example

**Tutorial**: Complete ASPRS LAS 1.4 classification workflow  
**Level**: Intermediate  
**Time**: ~45 minutes  
**Version**: 5.0.0

---

## 🎯 Overview

This tutorial demonstrates how to apply ASPRS LAS 1.4 classification to IGN LiDAR HD data, using ground truth from BD TOPO® and geometric features.

### What You'll Learn

- ✅ Understand ASPRS LAS 1.4 classification system
- ✅ Apply standard ASPRS classes (0-31)
- ✅ Use BD TOPO® extended classes (32-255)
- ✅ Handle Class 67 (non-standard IGN class)
- ✅ Validate classification results
- ✅ Export for use in other software

### Prerequisites

- IGN LiDAR HD tiles
- Understanding of ASPRS classification system
- Basic Python knowledge

---

## 📚 ASPRS Classification System

### Standard Classes (0-31)

| Class | Name                     | Description               | Source      |
| ----- | ------------------------ | ------------------------- | ----------- |
| 0     | Never Classified         | Created, never classified | Default     |
| 1     | Unclassified             | Not assigned to a class   | Processing  |
| 2     | Ground                   | Terrain surface           | DEM/DTM     |
| 3     | Low Vegetation           | Height < 0.5m             | NDVI/Height |
| 4     | Medium Vegetation        | Height 0.5-2m             | NDVI/Height |
| 5     | High Vegetation          | Height > 2m               | NDVI/Height |
| 6     | Building                 | Permanent structures      | BD TOPO     |
| 7     | Low Point (Noise)        | Low outliers              | Filter      |
| 8     | Reserved                 | -                         | -           |
| 9     | Water                    | Water bodies              | BD TOPO     |
| 10    | Rail                     | Railway tracks            | BD TOPO     |
| 11    | Road Surface             | Road pavement             | BD TOPO     |
| 12    | Reserved                 | -                         | -           |
| 13    | Wire - Guard             | Shield wire               | Detection   |
| 14    | Wire - Conductor         | Transmission lines        | Detection   |
| 15    | Transmission Tower       | Power line towers         | Detection   |
| 16    | Wire-Structure Connector | Insulators                | Detection   |
| 17    | Bridge Deck              | Bridge surfaces           | BD TOPO     |
| 18    | High Noise               | High outliers             | Filter      |

### BD TOPO® Extended Classes (32-255)

France-specific extensions from IGN BD TOPO®:

| Class | Name                   | BD TOPO® Source                 |
| ----- | ---------------------- | ------------------------------- |
| 32    | Building - Residential | BATI_INDIFFERENCIE              |
| 33    | Building - Industrial  | BATI_INDUSTRIEL                 |
| 34    | Building - Commercial  | BATI_REMARQUABLE                |
| 40    | Road - Highway         | ROUTE_PRIMAIRE                  |
| 41    | Road - Primary         | ROUTE_SECONDAIRE                |
| 42    | Road - Secondary       | CHEMIN                          |
| 50    | Vegetation - Forest    | ZONE_VEGETATION.NATURE="Forêt"  |
| 51    | Vegetation - Orchard   | ZONE_VEGETATION.NATURE="Verger" |
| 60    | Water - River          | COURS_D_EAU                     |
| 61    | Water - Lake           | PLAN_D_EAU                      |

---

## 🔧 Configuration

### Basic ASPRS Configuration

Create `config_asprs.yaml`:

```yaml
# config_asprs.yaml
defaults:
  - base/processor
  - base/features
  - base/data_sources
  - base/output
  - base/monitoring
  - _self_

# Classification mode
classification:
  mode: "asprs" # ASPRS LAS 1.4 classification
  standard_only: false # Include BD TOPO extended classes

  # Class 67 handling (non-standard IGN class)
  handle_class_67: true
  remap_class_67_to: 6 # Remap to Building (Class 6)

# Standard processing
processor:
  batch_size: 16
  use_gpu: false

# Geometric features for classification
features:
  compute_normals: true
  compute_curvature: true
  k_neighbors: 50

  # Height-based vegetation classification
  height_classification:
    enabled: true
    low_vegetation_max: 0.5 # < 0.5m = Class 3
    medium_vegetation_max: 2.0 # 0.5-2m = Class 4
    high_vegetation_min: 2.0 # > 2m = Class 5

# Ground truth from BD TOPO
data_sources:
  bd_topo:
    enabled: true
    features:
      buildings: true # Class 6 (+ extended 32-34)
      roads: true # Class 11 (+ extended 40-42)
      water: true # Class 9 (+ extended 60-61)
      vegetation: true # Class 3/4/5 (+ extended 50-51)
      railways: true # Class 10
      bridges: true # Class 17

    # Extended classification
    use_extended_classes: true

    cache_enabled: true

# Output with ASPRS classes
output:
  formats:
    laz: true
  output_suffix: "_asprs"

  # Validate ASPRS compliance
  validate_asprs: true

monitoring:
  log_level: "INFO"
  show_progress: true
```

---

## 🚀 Basic Workflow

### Step 1: Download Sample Data

```bash
# Create project
mkdir -p ~/asprs_tutorial
cd ~/asprs_tutorial

# Download tiles (Versailles area)
ign-lidar-hd download \
  --department 78 \
  --tile-range 650 651 6860 6861 \
  --output data/input/
```

### Step 2: Apply ASPRS Classification

```bash
# Process with ASPRS classification
ign-lidar-hd process \
  --config-name config_asprs \
  input_dir=data/input/ \
  output_dir=data/output/

# Output files with ASPRS classes:
# data/output/
# ├── tile_0650_6860_asprs.laz
# └── tile_0651_6860_asprs.laz
```

### Step 3: Verify ASPRS Classes

```python
import laspy
import numpy as np

def verify_asprs_classification(laz_path):
    """Verify ASPRS classification compliance."""
    las = laspy.read(laz_path)

    print(f"File: {laz_path.name}")
    print(f"Total points: {len(las.points):,}\n")

    # Get classification distribution
    classes, counts = np.unique(las.classification, return_counts=True)

    # ASPRS class names
    asprs_names = {
        0: "Never Classified",
        1: "Unclassified",
        2: "Ground",
        3: "Low Vegetation",
        4: "Medium Vegetation",
        5: "High Vegetation",
        6: "Building",
        7: "Low Point (Noise)",
        9: "Water",
        10: "Rail",
        11: "Road Surface",
        17: "Bridge Deck",
        32: "Building - Residential",
        33: "Building - Industrial",
        40: "Road - Highway",
        41: "Road - Primary",
        50: "Vegetation - Forest",
        60: "Water - River",
        61: "Water - Lake",
        67: "Building (IGN Legacy)"
    }

    print("ASPRS Classification Distribution:")
    print("="*70)

    for cls, count in zip(classes, counts):
        pct = count / len(las.points) * 100
        name = asprs_names.get(cls, f"Unknown Class {cls}")

        # Validate class is in valid ASPRS range
        valid = cls in asprs_names or (32 <= cls <= 255)
        marker = "✅" if valid else "⚠️ "

        print(f"{marker} Class {cls:3d}: {name:30s} "
              f"{count:10,} ({pct:5.2f}%)")

    print("="*70)

    # Check for invalid classes
    invalid_classes = [c for c in classes if c > 255]
    if invalid_classes:
        print(f"\n⚠️  Invalid classes found: {invalid_classes}")
    else:
        print(f"\n✅ All classes are valid ASPRS classes")

    return classes, counts

# Verify classification
verify_asprs_classification("data/output/tile_0650_6860_asprs.laz")
```

---

## ⚡ GPU-Accelerated ASPRS Classification

### GPU Configuration

```yaml
# config_asprs_gpu.yaml
defaults:
  - presets/asprs_classification_gpu_optimized
  - _self_

# Override for your data
processor:
  batch_size: 32
  gpu_device: 0

# Enhanced features with GPU
features:
  compute_normals: true
  compute_curvature: true
  compute_roughness: true # Additional feature

  # RGB from orthophotos (GPU-accelerated)
  rgb_augmentation:
    enabled: true
    method: "orthophoto"
    resolution: 0.2
    use_gpu: true

# Extended ASPRS classification
data_sources:
  bd_topo:
    enabled: true
    features:
      buildings: true
      roads: true
      water: true
      vegetation: true
      railways: true
      bridges: true
    use_extended_classes: true

output:
  formats:
    laz: true
  extra_dims:
    - name: "Curvature"
      type: "float32"
    - name: "Roughness"
      type: "float32"
  output_suffix: "_asprs_gpu"

monitoring:
  metrics:
    track_gpu: true
```

### Process with GPU

```bash
# GPU-accelerated ASPRS classification
ign-lidar-hd process \
  --config-name config_asprs_gpu \
  input_dir=data/input/ \
  output_dir=data/output_gpu/

# Performance:
# CPU: ~5 tiles/hour
# GPU (RTX 4080): ~15-20 tiles/hour
```

---

## 🎨 Advanced Classification

### Custom ASPRS Rules

```python
from ign_lidar.core.asprs_classifier import ASPRSClassifier
import numpy as np

class CustomASPRSClassifier(ASPRSClassifier):
    """Custom ASPRS classifier with additional rules."""

    def classify_buildings(self, points, bd_topo_buildings):
        """Enhanced building classification."""

        # Start with ground truth
        classification = super().classify_buildings(points, bd_topo_buildings)

        # Additional rule: Use height and planarity
        if hasattr(points, 'curvature'):
            # Flat surfaces above 3m likely buildings
            ground_height = points['z'].min()
            height_above_ground = points['z'] - ground_height

            is_flat = points['curvature'] < 0.1
            is_elevated = height_above_ground > 3.0

            likely_building = is_flat & is_elevated

            # Only apply if not already classified
            unclassified = classification == 1
            classification[unclassified & likely_building] = 6

        return classification

    def classify_vegetation(self, points, height_thresholds=None):
        """Enhanced vegetation classification using NDVI."""

        if height_thresholds is None:
            height_thresholds = {
                'low': 0.5,
                'medium': 2.0
            }

        classification = np.ones(len(points), dtype=np.uint8)

        # Use NDVI if available
        if hasattr(points, 'ndvi'):
            is_vegetation = points['ndvi'] > 0.3

            # Height-based sub-classification
            ground_height = points['z'].min()
            height = points['z'] - ground_height

            # Class 3: Low vegetation
            classification[is_vegetation & (height < height_thresholds['low'])] = 3

            # Class 4: Medium vegetation
            classification[is_vegetation &
                         (height >= height_thresholds['low']) &
                         (height < height_thresholds['medium'])] = 4

            # Class 5: High vegetation
            classification[is_vegetation & (height >= height_thresholds['medium'])] = 5

            # Extended classes for forest
            if hasattr(points, 'tree_density'):
                is_forest = points['tree_density'] > 0.8
                classification[is_vegetation & is_forest] = 50  # Forest (extended)

        return classification

# Use custom classifier
classifier = CustomASPRSClassifier()

# Process tile
from pathlib import Path
tile_path = Path("data/input/tile_0650_6860.laz")
classified_points = classifier.classify_tile(tile_path)

# Save result
output_path = Path("data/output/tile_0650_6860_custom_asprs.laz")
classifier.save_classified(classified_points, output_path)

print(f"✅ Custom ASPRS classification complete")
```

---

## 🔧 Python API Examples

### Example 1: Basic ASPRS Classification

```python
from ign_lidar.core.asprs_classifier import ASPRSClassifier
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
from pathlib import Path
import laspy

# Initialize classifier
classifier = ASPRSClassifier(
    use_extended_classes=True,
    handle_class_67=True
)

# Fetch ground truth
fetcher = IGNGroundTruthFetcher(cache_dir=None, verbose=True)
tile_path = Path("data/input/tile_0650_6860.laz")

ground_truth = fetcher.fetch_for_tile(
    tile_path,
    feature_types=["buildings", "roads", "water", "vegetation"]
)

# Read tile
las = laspy.read(tile_path)

# Apply ASPRS classification
classified = classifier.classify(
    points=las.points,
    ground_truth=ground_truth
)

# Update classification
las.classification = classified

# Save
output_path = Path("data/output/tile_0650_6860_asprs.laz")
las.write(output_path)

print(f"✅ ASPRS classification applied")
print(f"   Classes used: {len(np.unique(classified))}")
```

### Example 2: Class 67 Handling

```python
from ign_lidar.core.asprs_classifier import ASPRSClassifier
import numpy as np

def handle_class_67(las_file):
    """Handle non-standard Class 67 from IGN."""

    las = laspy.read(las_file)

    # Check for Class 67
    has_class_67 = 67 in las.classification

    if has_class_67:
        count_67 = np.sum(las.classification == 67)
        print(f"Found {count_67:,} points with Class 67")

        # Option 1: Remap to Building (Class 6)
        las.classification[las.classification == 67] = 6
        print("  Remapped to Class 6 (Building)")

        # Option 2: Keep as extended class
        # (No change needed if using extended classes)

        # Save corrected file
        output_path = las_file.parent / f"{las_file.stem}_fixed.laz"
        las.write(output_path)

        return output_path
    else:
        print("No Class 67 found")
        return las_file

# Process file
fixed_file = handle_class_67(Path("data/input/tile_0650_6860.laz"))
print(f"✅ Class 67 handled: {fixed_file}")
```

### Example 3: ASPRS Validation

```python
from ign_lidar.core.asprs_classifier import ASPRSValidator
import laspy

def validate_asprs_compliance(laz_path):
    """Validate ASPRS LAS 1.4 compliance."""

    las = laspy.read(laz_path)

    validator = ASPRSValidator()

    # Run validation
    results = validator.validate(las)

    print(f"ASPRS Validation Results:")
    print(f"="*50)

    # Check version
    print(f"LAS Version: {las.header.version}")
    if las.header.version >= (1, 4):
        print("  ✅ LAS 1.4 or higher")
    else:
        print("  ⚠️  LAS version < 1.4")

    # Check point format
    print(f"Point Format: {las.header.point_format.id}")
    if las.header.point_format.id >= 6:
        print("  ✅ LAS 1.4 point format")
    else:
        print("  ⚠️  Legacy point format")

    # Check classifications
    classes = np.unique(las.classification)
    print(f"\nClasses present: {len(classes)}")

    # Validate class ranges
    invalid_standard = [c for c in classes if c > 31 and c < 64]
    invalid_extended = [c for c in classes if c > 255]

    if invalid_standard:
        print(f"  ⚠️  Invalid standard classes: {invalid_standard}")
    if invalid_extended:
        print(f"  ⚠️  Invalid extended classes: {invalid_extended}")
    if not invalid_standard and not invalid_extended:
        print(f"  ✅ All classes valid")

    # Check for required fields
    required_fields = ['x', 'y', 'z', 'classification', 'return_number']
    missing_fields = [f for f in required_fields if not hasattr(las, f)]

    if missing_fields:
        print(f"  ⚠️  Missing required fields: {missing_fields}")
    else:
        print(f"  ✅ All required fields present")

    print("="*50)

    return results

# Validate file
validate_asprs_compliance("data/output/tile_0650_6860_asprs.laz")
```

---

## 📊 Quality Control

### Classification Statistics

```python
import laspy
import numpy as np
import pandas as pd

def classification_statistics(laz_path):
    """Generate detailed classification statistics."""

    las = laspy.read(laz_path)

    # Classification distribution
    classes, counts = np.unique(las.classification, return_counts=True)

    # ASPRS class info
    asprs_info = {
        2: {"name": "Ground", "expected_pct": 40.0},
        3: {"name": "Low Vegetation", "expected_pct": 5.0},
        4: {"name": "Medium Vegetation", "expected_pct": 5.0},
        5: {"name": "High Vegetation", "expected_pct": 10.0},
        6: {"name": "Building", "expected_pct": 30.0},
        9: {"name": "Water", "expected_pct": 2.0},
        11: {"name": "Road", "expected_pct": 5.0},
    }

    # Build statistics table
    stats = []
    for cls, count in zip(classes, counts):
        pct = count / len(las.points) * 100

        info = asprs_info.get(cls, {"name": f"Class {cls}", "expected_pct": None})

        row = {
            "Class": cls,
            "Name": info["name"],
            "Count": count,
            "Percentage": pct,
            "Expected (%)": info["expected_pct"],
            "Deviation": pct - info["expected_pct"] if info["expected_pct"] else None
        }
        stats.append(row)

    # Create DataFrame
    df = pd.DataFrame(stats)

    print(f"\nClassification Statistics:")
    print(f"="*80)
    print(df.to_string(index=False))
    print(f"="*80)

    # Quality metrics
    unclassified_pct = (np.sum(las.classification == 1) / len(las.points)) * 100
    print(f"\nQuality Metrics:")
    print(f"  Unclassified: {unclassified_pct:.2f}%")

    if unclassified_pct < 5:
        print("  ✅ Low unclassified rate (good)")
    elif unclassified_pct < 15:
        print("  ⚠️  Moderate unclassified rate")
    else:
        print("  ❌ High unclassified rate (needs improvement)")

    return df

# Generate statistics
stats = classification_statistics("data/output/tile_0650_6860_asprs.laz")
```

---

## 🌍 Export for Other Software

### Export for QGIS

```python
import laspy
from pathlib import Path

def export_for_qgis(laz_path, output_dir):
    """Export ASPRS-classified data for QGIS."""

    las = laspy.read(laz_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure LAS 1.4 format
    if las.header.version < (1, 4):
        print("Converting to LAS 1.4...")
        # Conversion handled automatically by laspy

    # Save as LAZ (compressed)
    output_laz = output_dir / f"{laz_path.stem}_qgis.laz"
    las.write(output_laz)

    # Also save style file for QGIS
    qml_content = generate_qgis_style()
    qml_path = output_dir / f"{laz_path.stem}_qgis.qml"
    with open(qml_path, 'w') as f:
        f.write(qml_content)

    print(f"✅ Exported for QGIS:")
    print(f"   Data: {output_laz}")
    print(f"   Style: {qml_path}")

    return output_laz, qml_path

def generate_qgis_style():
    """Generate QGIS style (QML) for ASPRS classes."""

    # Simplified QGIS style with ASPRS colors
    qml = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.0">
  <renderer-v2 type="pointCloudClassifiedRenderer">
    <categories>
      <category value="2" label="Ground" color="#CD853F"/>
      <category value="3" label="Low Vegetation" color="#90EE90"/>
      <category value="4" label="Medium Vegetation" color="#228B22"/>
      <category value="5" label="High Vegetation" color="#006400"/>
      <category value="6" label="Building" color="#FF0000"/>
      <category value="9" label="Water" color="#0000FF"/>
      <category value="11" label="Road" color="#808080"/>
    </categories>
  </renderer-v2>
</qgis>"""

    return qml

# Export
export_for_qgis(
    "data/output/tile_0650_6860_asprs.laz",
    "data/export/qgis/"
)
```

### Export for CloudCompare

```python
def export_for_cloudcompare(laz_path, output_dir):
    """Export for CloudCompare with ASCII classification."""

    las = laspy.read(laz_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CloudCompare can read LAZ directly
    output_laz = output_dir / f"{laz_path.stem}_cc.laz"
    las.write(output_laz)

    # Also create ASCII file with classification
    ascii_path = output_dir / f"{laz_path.stem}_cc.txt"

    with open(ascii_path, 'w') as f:
        # Header
        f.write("//X Y Z Classification R G B\n")

        # Data
        for i in range(len(las.points)):
            x, y, z = las.x[i], las.y[i], las.z[i]
            c = las.classification[i]
            r = las.red[i] if hasattr(las, 'red') else 128
            g = las.green[i] if hasattr(las, 'green') else 128
            b = las.blue[i] if hasattr(las, 'blue') else 128

            f.write(f"{x:.3f} {y:.3f} {z:.3f} {c} {r} {g} {b}\n")

    print(f"✅ Exported for CloudCompare:")
    print(f"   LAZ: {output_laz}")
    print(f"   ASCII: {ascii_path}")

    return output_laz, ascii_path
```

---

## 🐛 Troubleshooting

### Issue 1: Many Unclassified Points

**Symptom**: High percentage of Class 1 (Unclassified)

**Solutions**:

1. Check ground truth coverage
2. Adjust classification buffers
3. Add fallback classification

```yaml
classification:
  # Increase buffer for ground truth matching
  buffer_distance: 2.0 # Default: 1.0

  # Fallback classification
  fallback_rules:
    enabled: true
    use_height: true # Use height for vegetation
    use_geometry: true # Use geometric features
```

### Issue 2: Class 67 Errors

**Symptom**: Invalid class 67 in output

**Solution**:

```yaml
classification:
  handle_class_67: true
  remap_class_67_to: 6 # Remap to Building
```

### Issue 3: Missing Extended Classes

**Symptom**: Only standard classes (0-31) present

**Solution**:

```yaml
classification:
  standard_only: false # Enable extended classes

data_sources:
  bd_topo:
    use_extended_classes: true
```

---

## 📚 Related Documentation

- [ASPRS Classification Reference](../reference/asprs-classification.md)
- [Ground Truth Classification](../features/ground-truth-classification.md)
- [BD TOPO Integration](../reference/bd-topo-integration.md)
- [Classification Workflow](../reference/classification-workflow.md)

---

## 🎯 Summary

You've learned how to:

- ✅ Apply ASPRS LAS 1.4 classification
- ✅ Use standard and extended ASPRS classes
- ✅ Handle Class 67 (IGN legacy)
- ✅ Validate ASPRS compliance
- ✅ Export for QGIS and CloudCompare

**Next Steps**:

- Try [LOD2 Classification Example](./lod2-classification-example.md)
- Explore [LOD3 Classification Example](./lod3-classification-example.md)
- Learn about [Classification Workflows](../reference/classification-workflow.md)

---

**Tutorial Version**: 1.0  
**Last Updated**: October 17, 2025  
**Tested With**: IGN LiDAR HD Dataset v5.0.0
