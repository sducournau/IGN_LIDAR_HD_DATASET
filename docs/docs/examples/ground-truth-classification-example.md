# Ground Truth Classification Example

**Tutorial**: Complete workflow for ground truth classification using IGN BD TOPO®  
**Level**: Intermediate  
**Time**: ~30 minutes  
**Version**: 5.0.0

---

## 🎯 Overview

This tutorial demonstrates how to use IGN BD TOPO® ground truth data to automatically classify LiDAR point clouds with ASPRS classes.

### What You'll Learn

- ✅ Fetch ground truth data from IGN WFS services
- ✅ Apply ASPRS classification to point clouds
- ✅ Use NDVI refinement for vegetation
- ✅ Optimize with GPU acceleration
- ✅ Cache ground truth data for reuse

### Prerequisites

- IGN LiDAR HD tiles downloaded
- Internet connection (for WFS fetching)
- GPU (optional, for acceleration)

---

## 📥 Setup

### 1. Prepare Your Data

```bash
# Create project directory
mkdir -p ~/lidar_tutorial
cd ~/lidar_tutorial

# Download sample tiles (Versailles area)
ign-lidar-hd download \
  --department 78 \
  --tile-range 650 651 6860 6861 \
  --output data/input/

# Directory structure:
# data/
# ├── input/
# │   ├── tile_0650_6860.laz
# │   └── tile_0651_6860.laz
# └── output/
```

### 2. Create Configuration

Create `config_ground_truth.yaml`:

```yaml
# config_ground_truth.yaml
defaults:
  - base/processor
  - base/features
  - base/data_sources
  - base/output
  - base/monitoring
  - _self_

# Standard processing
processor:
  batch_size: 16
  use_gpu: false
  skip_existing: true

# Basic features
features:
  compute_normals: true
  k_neighbors: 50

# Ground truth classification
data_sources:
  bd_topo:
    enabled: true
    features:
      buildings: true # ASPRS Class 6
      roads: true # ASPRS Class 11
      water: true # ASPRS Class 9
      vegetation: true # ASPRS Class 3/4/5

    # WFS service
    wfs_url: "https://data.geopf.fr/wfs"
    max_features: 10000
    timeout: 30

    # Cache configuration
    cache_enabled: true
    cache_dir: null # Auto: data/input/cache/ground_truth

# Output settings
output:
  formats:
    laz: true
  output_suffix: "_classified"
  validate_format: true

# Monitoring
monitoring:
  log_level: "INFO"
  show_progress: true
```

---

## 🚀 Basic Usage

### Step 1: Fetch Ground Truth Data

```bash
# Process tiles with ground truth
ign-lidar-hd process \
  --config-name config_ground_truth \
  input_dir=data/input/ \
  output_dir=data/output/
```

**What happens**:

1. **WFS Fetching**: System fetches BD TOPO® features for tile extent
2. **Caching**: Features cached to `data/input/cache/ground_truth/`
3. **Classification**: Points classified based on ground truth
4. **Output**: Enriched LAZ files with ASPRS classes

### Step 2: Check Results

```bash
# List output files
ls -lh data/output/

# Expected output:
# tile_0650_6860_classified.laz
# tile_0651_6860_classified.laz

# Check cache
ls -lh data/input/cache/ground_truth/

# Expected cache files:
# buildings_650000_6860000_651000_6861000.geojson
# roads_650000_6860000_651000_6861000.geojson
# water_650000_6860000_651000_6861000.geojson
# vegetation_650000_6860000_651000_6861000.geojson
```

### Step 3: Verify Classification

```python
import laspy
import numpy as np

# Read classified file
las = laspy.read("data/output/tile_0650_6860_classified.laz")

# Check classification distribution
classes, counts = np.unique(las.classification, return_counts=True)

print("Classification Distribution:")
for cls, count in zip(classes, counts):
    pct = count / len(las.points) * 100
    print(f"  Class {cls:2d}: {count:10,} points ({pct:5.2f}%)")

# Expected output:
# Class  1:    500,000 points (10.00%) - Unclassified
# Class  2:  2,000,000 points (40.00%) - Ground
# Class  3:    250,000 points ( 5.00%) - Low Vegetation
# Class  4:    300,000 points ( 6.00%) - Medium Vegetation
# Class  5:    200,000 points ( 4.00%) - High Vegetation
# Class  6:  1,500,000 points (30.00%) - Building
# Class  9:    150,000 points ( 3.00%) - Water
# Class 11:    100,000 points ( 2.00%) - Road
```

---

## ⚡ GPU Acceleration

### Configuration for GPU

```yaml
# config_ground_truth_gpu.yaml
defaults:
  - base/processor
  - base/features
  - base/data_sources
  - base/output
  - base/monitoring
  - _self_

# GPU-optimized processing
processor:
  batch_size: 32 # Larger batches for GPU
  use_gpu: true
  gpu_device: 0
  chunk_size: 2_000_000 # Larger chunks

# GPU-accelerated features
features:
  compute_normals: true
  compute_curvature: true # GPU-accelerated
  k_neighbors: 50

# Ground truth (same as before)
data_sources:
  bd_topo:
    enabled: true
    features:
      buildings: true
      roads: true
      water: true
      vegetation: true
    cache_enabled: true

output:
  formats:
    laz: true
  output_suffix: "_classified_gpu"

monitoring:
  log_level: "INFO"
  metrics:
    enabled: true
    track_gpu: true # Track GPU usage
```

### Run with GPU

```bash
# Process with GPU acceleration
ign-lidar-hd process \
  --config-name config_ground_truth_gpu \
  input_dir=data/input/ \
  output_dir=data/output_gpu/

# Performance comparison:
# CPU: ~5 tiles/hour
# GPU: ~15 tiles/hour (RTX 4080 Super)
```

---

## 🌿 NDVI Vegetation Refinement

### Add NDVI Refinement

```yaml
# config_ground_truth_ndvi.yaml
defaults:
  - base/processor
  - base/features
  - base/data_sources
  - base/output
  - base/monitoring
  - _self_

processor:
  batch_size: 16
  use_gpu: false

features:
  compute_normals: true

  # NDVI refinement for vegetation
  ndvi:
    enabled: true
    threshold: 0.3 # NDVI > 0.3 = vegetation
    source: "orthohr" # IGN OrthoHR with infrared
    resolution: 0.2 # 20cm resolution

# Ground truth with vegetation
data_sources:
  bd_topo:
    enabled: true
    features:
      buildings: true
      roads: true
      water: true
      vegetation: true # Will be refined by NDVI

output:
  formats:
    laz: true
  output_suffix: "_classified_ndvi"
```

### Python API Example

```python
from ign_lidar.core.processor import LiDARProcessor
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
from pathlib import Path

# Initialize processor
processor = LiDARProcessor(
    input_dir=Path("data/input"),
    output_dir=Path("data/output"),
    use_gpu=False
)

# Configure ground truth fetcher
ground_truth = IGNGroundTruthFetcher(
    cache_dir=None,  # Auto-detect
    verbose=True
)

# Fetch ground truth for tile
tile_path = Path("data/input/tile_0650_6860.laz")
features = ground_truth.fetch_for_tile(
    tile_path,
    feature_types=["buildings", "roads", "water", "vegetation"]
)

print(f"Fetched {len(features)} ground truth features")
print(f"  Buildings: {len(features.get('buildings', []))}")
print(f"  Roads: {len(features.get('roads', []))}")
print(f"  Water: {len(features.get('water', []))}")
print(f"  Vegetation: {len(features.get('vegetation', []))}")

# Process tile with ground truth
processor.process_tile(
    tile_path,
    ground_truth_features=features,
    apply_ndvi_refinement=True
)

print(f"✅ Tile processed and classified")
```

---

## 🎨 Custom Classification Rules

### Advanced Configuration

```yaml
# config_ground_truth_custom.yaml
defaults:
  - base/processor
  - base/features
  - base/data_sources
  - base/output
  - base/monitoring
  - _self_

processor:
  batch_size: 16

features:
  compute_normals: true
  compute_curvature: true

  # RGB for additional context
  rgb_augmentation:
    enabled: true
    method: "orthophoto"
    resolution: 0.2

# Selective ground truth
data_sources:
  bd_topo:
    enabled: true
    features:
      buildings: true
      roads: true
      water: false # Don't classify water
      vegetation: false # Use NDVI instead

    # WFS filtering
    wfs_filter:
      buildings:
        min_area: 50 # Only buildings > 50 m²
        min_height: 3 # Only buildings > 3m tall
      roads:
        types: ["highway", "primary", "secondary"]

    cache_enabled: true

output:
  formats:
    laz: true
  extra_dims:
    - name: "GroundTruthSource"
      type: "uint8" # Track classification source
  output_suffix: "_custom_classified"
```

### Python API for Custom Rules

```python
from ign_lidar.core.processor import LiDARProcessor
from ign_lidar.features.ground_truth_classifier import GroundTruthClassifier
import numpy as np

# Custom classification function
def custom_classify(points, features):
    """Custom classification with additional rules."""

    # Initialize with unclassified
    classification = np.ones(len(points), dtype=np.uint8)

    # Classify buildings (Class 6)
    if "buildings" in features:
        for building in features["buildings"]:
            # Only large buildings
            if building["properties"].get("area", 0) > 100:
                mask = point_in_polygon(points, building["geometry"])
                classification[mask] = 6

    # Classify roads (Class 11)
    if "roads" in features:
        for road in features["roads"]:
            # Only major roads
            if road["properties"].get("importance") > 3:
                mask = point_near_line(points, road["geometry"], buffer=2.0)
                classification[mask] = 11

    # Use height for vegetation classification
    # Class 3: Low vegetation (< 2m)
    # Class 4: Medium vegetation (2-5m)
    # Class 5: High vegetation (> 5m)
    ground_height = points["z"].min()
    height_above_ground = points["z"] - ground_height

    vegetation_mask = classification == 1  # Unclassified
    classification[vegetation_mask & (height_above_ground < 2)] = 3
    classification[vegetation_mask & (height_above_ground >= 2) & (height_above_ground < 5)] = 4
    classification[vegetation_mask & (height_above_ground >= 5)] = 5

    return classification

# Apply custom classification
processor = LiDARProcessor(
    input_dir="data/input",
    output_dir="data/output",
    classification_function=custom_classify
)

processor.process()
```

---

## 📊 Performance Optimization

### Cache Management

```python
from pathlib import Path
import json

# Check cache size
cache_dir = Path("data/input/cache/ground_truth")
total_size = sum(f.stat().st_size for f in cache_dir.glob("*.geojson"))
print(f"Cache size: {total_size / 1024 / 1024:.2f} MB")

# Validate cache
for cache_file in cache_dir.glob("*.geojson"):
    with open(cache_file) as f:
        data = json.load(f)
    print(f"{cache_file.name}: {len(data['features'])} features")

# Clear old cache (if needed)
# cache_dir.rmdir()  # Will be recreated automatically
```

### Batch Processing

```bash
# Process multiple areas efficiently
for dept in 75 78 92 93 94; do
  echo "Processing department $dept..."

  ign-lidar-hd process \
    --config-name config_ground_truth_gpu \
    input_dir=data/dept_${dept}/ \
    output_dir=output/dept_${dept}/ \
    data_sources.bd_topo.cache_dir=cache/dept_${dept}/
done
```

---

## 🔍 Validation

### Check Classification Quality

```python
import laspy
import numpy as np
from pathlib import Path

def validate_classification(laz_path):
    """Validate classification results."""
    las = laspy.read(laz_path)

    # Check for unclassified points
    unclassified = np.sum(las.classification == 1)
    total = len(las.points)
    classified_pct = (1 - unclassified / total) * 100

    print(f"File: {laz_path.name}")
    print(f"  Total points: {total:,}")
    print(f"  Classified: {classified_pct:.2f}%")

    # Check class distribution
    classes, counts = np.unique(las.classification, return_counts=True)
    print(f"  Classes present: {len(classes)}")

    # Validate expected classes
    expected_classes = {2, 3, 4, 5, 6, 9, 11}
    present_classes = set(classes)

    if expected_classes.issubset(present_classes):
        print("  ✅ All expected classes present")
    else:
        missing = expected_classes - present_classes
        print(f"  ⚠️  Missing classes: {missing}")

    return classified_pct

# Validate all output files
output_dir = Path("data/output")
for laz_file in output_dir.glob("*.laz"):
    validate_classification(laz_file)
    print()
```

---

## 🎓 Best Practices

### 1. Cache Strategy

```yaml
# Production: Use global cache for large datasets
data_sources:
  bd_topo:
    cache_enabled: true
    cache_dir: "/mnt/shared/cache/ground_truth"
    use_global_cache: true

# Development: Use local cache per project
data_sources:
  bd_topo:
    cache_enabled: true
    cache_dir: null  # Auto: project/cache/ground_truth
```

### 2. Progressive Enhancement

```bash
# Step 1: Basic ground truth
ign-lidar-hd process \
  --config-name config_ground_truth \
  input_dir=data/ \
  output_dir=output_v1/

# Step 2: Add NDVI refinement
ign-lidar-hd process \
  --config-name config_ground_truth_ndvi \
  input_dir=data/ \
  output_dir=output_v2/

# Step 3: Add GPU acceleration
ign-lidar-hd process \
  --config-name config_ground_truth_gpu \
  input_dir=data/ \
  output_dir=output_v3/
```

### 3. Error Handling

```python
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
from ign_lidar.core.exceptions import WFSError

fetcher = IGNGroundTruthFetcher(verbose=True)

try:
    features = fetcher.fetch_for_tile(
        tile_path,
        feature_types=["buildings", "roads"]
    )
except WFSError as e:
    print(f"❌ WFS error: {e}")
    print("Using cached data or skipping ground truth...")
    features = fetcher.load_from_cache(tile_path)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
```

---

## 🐛 Troubleshooting

### Issue 1: WFS Timeout

**Error**: `WFS request timeout after 30s`

**Solution**:

```yaml
data_sources:
  bd_topo:
    timeout: 60 # Increase timeout
    max_features: 5000 # Reduce features per request
```

### Issue 2: Cache Not Used

**Error**: Fetching data on every run despite cache

**Solution**:

```python
# Check cache directory exists and is writable
from pathlib import Path

cache_dir = Path("data/input/cache/ground_truth")
cache_dir.mkdir(parents=True, exist_ok=True)

# Verify cache files
print(f"Cache directory: {cache_dir}")
print(f"Cache files: {list(cache_dir.glob('*.geojson'))}")
```

### Issue 3: Missing Classifications

**Error**: Many points remain unclassified

**Solution**:

1. Check ground truth coverage
2. Adjust classification buffer
3. Add fallback classification

```python
# Increase classification buffer
ground_truth_classifier = GroundTruthClassifier(
    buffer_distance=2.0  # Increase from 1.0 to 2.0
)
```

---

## 📚 Related Documentation

- [Ground Truth Classification](../features/ground-truth-classification.md)
- [ASPRS Classification Reference](../reference/asprs-classification.md)
- [BD TOPO Integration](../reference/bd-topo-integration.md)
- [Configuration V5](../guides/configuration-v5.md)

---

## 🎯 Summary

You've learned how to:

- ✅ Configure ground truth classification with BD TOPO®
- ✅ Fetch and cache WFS data
- ✅ Apply ASPRS classification to LiDAR tiles
- ✅ Use NDVI refinement for vegetation
- ✅ Optimize with GPU acceleration
- ✅ Validate classification results

**Next Steps**:

- Try [Tile Stitching Example](./tile-stitching-example.md)
- Explore [LOD2 Classification](./lod2-classification-example.md)
- Learn about [ASPRS Classification](./asprs-classification-example.md)

---

**Tutorial Version**: 1.0  
**Last Updated**: October 17, 2025  
**Tested With**: IGN LiDAR HD Dataset v5.0.0
