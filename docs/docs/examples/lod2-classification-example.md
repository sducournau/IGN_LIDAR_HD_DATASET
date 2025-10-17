# LOD2 Classification Example

**Tutorial**: Building-focused LOD2 classification  
**Level**: Advanced  
**Time**: ~45 minutes  
**Version**: 5.0.0

---

## 🎯 Overview

This tutorial demonstrates LOD2 (Level of Detail 2) classification for building-focused 3D modeling, using a simplified 15-class taxonomy ideal for urban analysis and BIM applications.

### What You'll Learn

- ✅ Understand LOD2 classification system (15 classes)
- ✅ Apply building-centric classification
- ✅ Map from ASPRS to LOD2
- ✅ Extract building components
- ✅ Export for BIM/CAD software

### Prerequisites

- ASPRS-classified LiDAR data
- Understanding of LOD levels
- Basic 3D modeling knowledge

---

## 📚 LOD2 Classification System

### LOD2 Taxonomy (15 Classes)

| Class | Name              | Description           | Use Case              |
| ----- | ----------------- | --------------------- | --------------------- |
| 0     | Unclassified      | Not yet classified    | Default               |
| 1     | Ground            | Terrain surface       | Base terrain          |
| 2     | Wall              | Building walls        | Building envelope     |
| 3     | Roof - Flat       | Flat roofs            | Modern buildings      |
| 4     | Roof - Gable      | Gable roofs           | Traditional houses    |
| 5     | Roof - Hip        | Hip roofs             | Complex buildings     |
| 6     | Floor             | Building floors/slabs | Multi-story buildings |
| 7     | Vegetation - Low  | < 2m vegetation       | Shrubs, grass         |
| 8     | Vegetation - Tree | Trees (> 2m)          | Urban forestry        |
| 9     | Water             | Water bodies          | Hydrology             |
| 10    | Road              | Road surfaces         | Infrastructure        |
| 11    | Rail              | Railway tracks        | Transport             |
| 12    | Bridge            | Bridge structures     | Infrastructure        |
| 13    | Power Line        | Transmission lines    | Utilities             |
| 14    | Other             | Everything else       | Miscellaneous         |

### LOD2 vs ASPRS Mapping

```python
ASPRS_TO_LOD2 = {
    2: 1,   # Ground → Ground
    6: 2,   # Building → Wall (default)
    3: 7,   # Low Vegetation → Vegetation Low
    4: 7,   # Medium Vegetation → Vegetation Low
    5: 8,   # High Vegetation → Vegetation Tree
    9: 9,   # Water → Water
    11: 10, # Road → Road
    10: 11, # Rail → Rail
    17: 12, # Bridge → Bridge
}
```

---

## 🔧 Configuration

### Basic LOD2 Configuration

Create `config_lod2.yaml`:

```yaml
# config_lod2.yaml
defaults:
  - base/processor
  - base/features
  - base/data_sources
  - base/output
  - base/monitoring
  - _self_

# LOD2 classification mode
classification:
  mode: "lod2" # LOD2 taxonomy
  source: "asprs" # Map from ASPRS classes

  # Building component extraction
  building_components:
    enabled: true
    extract_walls: true
    extract_roofs: true
    extract_floors: true

    # Roof type detection
    roof_detection:
      enabled: true
      methods: ["geometry", "planarity", "slope"]
      min_roof_area: 10.0 # m²

# Enhanced features for LOD2
processor:
  batch_size: 16
  use_gpu: false

features:
  compute_normals: true
  compute_curvature: true # Important for roof detection
  compute_planarity: true # For flat surfaces
  k_neighbors: 50

  # Building-specific features
  building_features:
    enabled: true
    detect_facades: true
    detect_roof_type: true
    detect_floors: true

# Ground truth for building extraction
data_sources:
  bd_topo:
    enabled: true
    features:
      buildings: true # Primary focus
      roads: true
      vegetation: true
      water: true

    # Building details from BD TOPO
    building_details:
      fetch_height: true
      fetch_type: true # Residential, industrial, etc.

# Output with LOD2 classes
output:
  formats:
    laz: true
  output_suffix: "_lod2"

  extra_dims:
    - name: "RoofType" # 0=flat, 1=gable, 2=hip
      type: "uint8"
    - name: "FloorLevel" # Floor number
      type: "uint8"

monitoring:
  log_level: "INFO"
  show_progress: true
```

---

## 🚀 Basic Workflow

### Step 1: Prepare ASPRS-Classified Data

```bash
# Start with ASPRS-classified tiles
# (See ASPRS Classification Example)

mkdir -p ~/lod2_tutorial
cd ~/lod2_tutorial

# Use ASPRS-classified data as input
cp ../asprs_tutorial/data/output/*.laz data/input/
```

### Step 2: Apply LOD2 Classification

```bash
# Process with LOD2 classification
ign-lidar-hd process \
  --config-name config_lod2 \
  input_dir=data/input/ \
  output_dir=data/output/

# Output files with LOD2 classes:
# data/output/
# ├── tile_0650_6860_lod2.laz
# └── tile_0651_6860_lod2.laz
```

### Step 3: Verify LOD2 Classes

```python
import laspy
import numpy as np

def verify_lod2_classification(laz_path):
    """Verify LOD2 classification."""

    las = laspy.read(laz_path)

    # LOD2 class names
    lod2_names = {
        0: "Unclassified",
        1: "Ground",
        2: "Wall",
        3: "Roof - Flat",
        4: "Roof - Gable",
        5: "Roof - Hip",
        6: "Floor",
        7: "Vegetation - Low",
        8: "Vegetation - Tree",
        9: "Water",
        10: "Road",
        11: "Rail",
        12: "Bridge",
        13: "Power Line",
        14: "Other"
    }

    classes, counts = np.unique(las.classification, return_counts=True)

    print(f"LOD2 Classification Distribution:")
    print("="*70)

    for cls, count in zip(classes, counts):
        pct = count / len(las.points) * 100
        name = lod2_names.get(cls, f"Unknown ({cls})")

        print(f"Class {cls:2d}: {name:20s} {count:10,} ({pct:5.2f}%)")

    print("="*70)

    # Building component analysis
    building_classes = [2, 3, 4, 5, 6]  # Wall, roofs, floor
    building_points = sum(counts[np.isin(classes, building_classes)])
    building_pct = building_points / len(las.points) * 100

    print(f"\nBuilding Analysis:")
    print(f"  Total building points: {building_points:,} ({building_pct:.2f}%)")

    # Roof type distribution
    roof_classes = [3, 4, 5]
    roof_counts = counts[np.isin(classes, roof_classes)]
    if len(roof_counts) > 0:
        print(f"  Roof types detected: {len(roof_counts)}")
        for rc, roof_cls in zip(roof_counts, [3, 4, 5]):
            if roof_cls in classes:
                print(f"    {lod2_names[roof_cls]}: {rc:,} points")

    return classes, counts

# Verify
verify_lod2_classification("data/output/tile_0650_6860_lod2.laz")
```

---

## 🏗️ Building Component Extraction

### Extract Building Components

```python
from ign_lidar.classification.lod2_classifier import LOD2Classifier
from ign_lidar.features.building_extractor import BuildingExtractor
import laspy
import numpy as np

def extract_building_components(laz_path):
    """Extract and analyze building components."""

    las = laspy.read(laz_path)

    # Initialize extractor
    extractor = BuildingExtractor()

    # Extract buildings
    buildings = extractor.extract_buildings(las)

    print(f"Extracted {len(buildings)} buildings\n")

    for i, building in enumerate(buildings[:5]):  # First 5 buildings
        print(f"Building {i+1}:")
        print(f"  Total points: {building['total_points']:,}")

        # Component breakdown
        components = building['components']
        print(f"  Components:")
        print(f"    Walls: {components.get('wall_points', 0):,} points")
        print(f"    Roof: {components.get('roof_points', 0):,} points")
        print(f"    Floor: {components.get('floor_points', 0):,} points")

        # Roof analysis
        if 'roof_type' in building:
            roof_types = {0: "Flat", 1: "Gable", 2: "Hip"}
            print(f"  Roof type: {roof_types.get(building['roof_type'], 'Unknown')}")
            print(f"  Roof area: {building.get('roof_area', 0):.2f} m²")

        # Building dimensions
        print(f"  Dimensions:")
        print(f"    Width: {building.get('width', 0):.2f} m")
        print(f"    Length: {building.get('length', 0):.2f} m")
        print(f"    Height: {building.get('height', 0):.2f} m")

        # Floor detection
        if 'floor_levels' in building:
            print(f"  Floors: {len(building['floor_levels'])}")

        print()

    return buildings

# Extract components
buildings = extract_building_components("data/output/tile_0650_6860_lod2.laz")
```

### Roof Type Detection

```python
def detect_roof_types(las):
    """Detect roof types using geometry and planarity."""

    # Extract roof points (classes 3, 4, 5)
    roof_mask = np.isin(las.classification, [3, 4, 5])
    roof_points = las.points[roof_mask]

    if len(roof_points) == 0:
        return {}

    # Analyze roof geometry
    from sklearn.cluster import DBSCAN
    from scipy.spatial import ConvexHull

    # Cluster roof points by building
    clustering = DBSCAN(eps=1.0, min_samples=50).fit(
        np.column_stack([roof_points['x'], roof_points['y']])
    )

    roof_types = {}

    for cluster_id in np.unique(clustering.labels_):
        if cluster_id == -1:
            continue

        cluster_mask = clustering.labels_ == cluster_id
        cluster_points = roof_points[cluster_mask]

        # Detect roof type
        if hasattr(cluster_points, 'curvature'):
            # Flat roof: low curvature
            if np.mean(cluster_points['curvature']) < 0.05:
                roof_type = 3  # Flat

            # Gable roof: two main planes
            elif detect_gable_pattern(cluster_points):
                roof_type = 4  # Gable

            # Hip roof: multiple planes
            elif detect_hip_pattern(cluster_points):
                roof_type = 5  # Hip

            else:
                roof_type = 3  # Default to flat

            roof_types[cluster_id] = {
                'type': roof_type,
                'points': len(cluster_points),
                'area': calculate_roof_area(cluster_points)
            }

    return roof_types

def detect_gable_pattern(points):
    """Detect gable roof pattern (two main planes)."""

    # Compute normals
    from sklearn.decomposition import PCA

    # PCA on normals to find dominant directions
    if hasattr(points, 'normal_x'):
        normals = np.column_stack([
            points['normal_x'],
            points['normal_y'],
            points['normal_z']
        ])

        pca = PCA(n_components=2)
        pca.fit(normals)

        # Gable roofs have 2 dominant normal directions
        explained_var = pca.explained_variance_ratio_

        # Strong first two components = likely gable
        return explained_var[0] > 0.4 and explained_var[1] > 0.3

    return False

def detect_hip_pattern(points):
    """Detect hip roof pattern (4+ planes)."""

    # Hip roofs have 4 or more distinct planes
    # Use plane segmentation

    from sklearn.cluster import KMeans

    if hasattr(points, 'normal_x'):
        normals = np.column_stack([
            points['normal_x'],
            points['normal_y'],
            points['normal_z']
        ])

        # Cluster normals to find planes
        kmeans = KMeans(n_clusters=4, random_state=42)
        kmeans.fit(normals)

        # Check if 4 clusters are well-separated
        inertia = kmeans.inertia_

        # Lower inertia = better separation = likely hip roof
        return inertia < 0.3

    return False

def calculate_roof_area(points):
    """Calculate roof surface area."""

    from scipy.spatial import ConvexHull

    # 2D convex hull for footprint area
    coords_2d = np.column_stack([points['x'], points['y']])

    try:
        hull = ConvexHull(coords_2d)
        return hull.volume  # In 2D, volume = area
    except:
        return 0.0

# Detect roof types
las = laspy.read("data/output/tile_0650_6860_lod2.laz")
roof_types = detect_roof_types(las)

print(f"Detected {len(roof_types)} roofs:")
for roof_id, info in roof_types.items():
    type_names = {3: "Flat", 4: "Gable", 5: "Hip"}
    print(f"  Roof {roof_id}: {type_names[info['type']]} "
          f"({info['points']:,} points, {info['area']:.2f} m²)")
```

---

## ⚡ GPU-Accelerated LOD2

### GPU Configuration

```yaml
# config_lod2_gpu.yaml
defaults:
  - base/processor
  - base/features
  - base/data_sources
  - base/output
  - base/monitoring
  - _self_

classification:
  mode: "lod2"
  source: "asprs"

  building_components:
    enabled: true
    use_gpu: true # GPU-accelerated extraction

processor:
  batch_size: 32 # Larger batches for GPU
  use_gpu: true
  gpu_device: 0

features:
  compute_normals: true
  compute_curvature: true
  compute_planarity: true
  k_neighbors: 50

  # GPU-accelerated features
  use_gpu_features: true

  building_features:
    enabled: true
    use_gpu: true # GPU roof detection

data_sources:
  bd_topo:
    enabled: true
    features:
      buildings: true

output:
  formats:
    laz: true
  extra_dims:
    - name: "RoofType"
      type: "uint8"
    - name: "FloorLevel"
      type: "uint8"
  output_suffix: "_lod2_gpu"

monitoring:
  metrics:
    track_gpu: true
```

### Process with GPU

```bash
# GPU-accelerated LOD2
ign-lidar-hd process \
  --config-name config_lod2_gpu \
  input_dir=data/input/ \
  output_dir=data/output_gpu/

# Performance:
# CPU: ~3 tiles/hour
# GPU (RTX 4080): ~12 tiles/hour
```

---

## 📊 Export for BIM/CAD

### Export to CityGML (LOD2)

```python
from ign_lidar.export.citygml_exporter import CityGMLExporter
from pathlib import Path

def export_to_citygml(laz_path, output_path):
    """Export LOD2 classification to CityGML format."""

    las = laspy.read(laz_path)

    # Initialize exporter
    exporter = CityGMLExporter(lod_level=2)

    # Extract buildings
    from ign_lidar.features.building_extractor import BuildingExtractor
    extractor = BuildingExtractor()
    buildings = extractor.extract_buildings(las)

    # Convert to CityGML
    citygml = exporter.export_buildings(
        buildings,
        output_path=output_path,
        include_textures=False,    # LOD2 doesn't include textures
        include_geometry=True
    )

    print(f"✅ Exported {len(buildings)} buildings to CityGML LOD2")
    print(f"   Output: {output_path}")

    return citygml

# Export
output_path = Path("data/export/buildings_lod2.gml")
export_to_citygml(
    "data/output/tile_0650_6860_lod2.laz",
    output_path
)
```

### Export to IFC (BIM)

```python
from ign_lidar.export.ifc_exporter import IFCExporter

def export_to_ifc(laz_path, output_path):
    """Export LOD2 to IFC format for BIM software."""

    las = laspy.read(laz_path)

    # Extract buildings
    from ign_lidar.features.building_extractor import BuildingExtractor
    extractor = BuildingExtractor()
    buildings = extractor.extract_buildings(las)

    # Initialize IFC exporter
    exporter = IFCExporter()

    # Create IFC file
    ifc_file = exporter.create_ifc(
        buildings=buildings,
        lod_level=2,
        coordinate_system="EPSG:2154"  # Lambert 93
    )

    # Write to file
    exporter.write(ifc_file, output_path)

    print(f"✅ Exported to IFC:")
    print(f"   Buildings: {len(buildings)}")
    print(f"   Output: {output_path}")

    return ifc_file

# Export to IFC
output_ifc = Path("data/export/buildings_lod2.ifc")
export_to_ifc(
    "data/output/tile_0650_6860_lod2.laz",
    output_ifc
)
```

---

## 🔍 Quality Control

### Building Validation

```python
def validate_building_extraction(buildings):
    """Validate extracted building components."""

    print("Building Validation Report")
    print("="*60)

    valid_buildings = 0
    issues = []

    for i, building in enumerate(buildings):
        # Check minimum points
        if building['total_points'] < 100:
            issues.append(f"Building {i}: Too few points ({building['total_points']})")
            continue

        # Check components
        components = building['components']

        # Must have at least walls or roof
        if components.get('wall_points', 0) == 0 and components.get('roof_points', 0) == 0:
            issues.append(f"Building {i}: No walls or roof detected")
            continue

        # Check dimensions
        if building.get('height', 0) < 2.5:
            issues.append(f"Building {i}: Suspiciously low ({building.get('height', 0):.1f}m)")
            continue

        # Check roof area
        if building.get('roof_area', 0) < 20:
            issues.append(f"Building {i}: Very small roof area ({building.get('roof_area', 0):.1f}m²)")
            continue

        valid_buildings += 1

    print(f"Total buildings: {len(buildings)}")
    print(f"Valid buildings: {valid_buildings}")
    print(f"Issues found: {len(issues)}")

    if issues:
        print("\nIssues:")
        for issue in issues[:10]:  # First 10 issues
            print(f"  - {issue}")

    print("="*60)

    return valid_buildings, issues

# Validate
valid, issues = validate_building_extraction(buildings)
```

---

## 🐛 Troubleshooting

### Issue 1: No Roof Types Detected

**Symptom**: All roofs classified as flat (Class 3)

**Solutions**:

```yaml
features:
  compute_curvature: true # Required for roof type detection
  compute_planarity: true # Required for flat detection

classification:
  building_components:
    roof_detection:
      enabled: true
      min_roof_area: 5.0 # Lower threshold
      methods: ["geometry", "planarity", "slope"]
```

### Issue 2: Buildings Not Extracted

**Symptom**: No building components found

**Solutions**:

1. Check ASPRS input has Class 6 (Building)
2. Verify BD TOPO ground truth
3. Lower extraction thresholds

```yaml
classification:
  building_components:
    min_building_points: 50 # Lower threshold
    min_building_area: 10.0 # Lower area threshold
```

### Issue 3: Floor Detection Fails

**Symptom**: No floors detected in multi-story buildings

**Solutions**:

```yaml
features:
  building_features:
    detect_floors: true
    floor_height: 3.0 # Typical floor height
    min_floor_points: 100 # Minimum points per floor
```

---

## 📚 Related Documentation

- [LOD Classification Reference](../reference/lod-classification.md)
- [Classification Workflow](../reference/classification-workflow.md)
- [Building Component Extraction](../features/building-extraction.md)
- [ASPRS to LOD2 Mapping](../reference/asprs-lod-mapping.md)

---

## 🎯 Summary

You've learned how to:

- ✅ Apply LOD2 classification (15 classes)
- ✅ Extract building components (walls, roofs, floors)
- ✅ Detect roof types (flat, gable, hip)
- ✅ Export to CityGML and IFC
- ✅ Validate building extraction

**Next Steps**:

- Try [LOD3 Classification Example](./lod3-classification-example.md)
- Explore [Building Component Extraction](../features/building-extraction.md)
- Learn about [CityGML Export](../guides/citygml-export.md)

---

**Tutorial Version**: 1.0  
**Last Updated**: October 17, 2025  
**Tested With**: IGN LiDAR HD Dataset v5.0.0
