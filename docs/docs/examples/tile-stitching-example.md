# Tile Stitching Workflow Example

**Tutorial**: Complete workflow for seamless tile stitching  
**Level**: Intermediate  
**Time**: ~30 minutes  
**Version**: 5.0.0

---

## 🎯 Overview

This tutorial demonstrates how to process multiple LiDAR tiles and stitch them together seamlessly, eliminating boundary artifacts and ensuring continuity across tile edges.

### What You'll Learn

- ✅ Configure boundary-aware processing
- ✅ Process multi-tile datasets
- ✅ Stitch tiles seamlessly
- ✅ Handle edge effects and artifacts
- ✅ Optimize performance for large datasets

### Prerequisites

- Multiple IGN LiDAR HD tiles (at least 4 tiles in a 2x2 grid)
- Basic understanding of LiDAR processing
- ~10 GB disk space for examples

---

## 📥 Setup

### 1. Download Tile Grid

```bash
# Create project directory
mkdir -p ~/tile_stitching_tutorial
cd ~/tile_stitching_tutorial

# Download 2x2 tile grid (Versailles area)
ign-lidar-hd download \
  --department 78 \
  --tile-range 650 652 6860 6862 \
  --output data/input/

# Directory structure:
# data/input/
# ├── tile_0650_6860.laz  (SW corner)
# ├── tile_0651_6860.laz  (SE corner)
# ├── tile_0650_6861.laz  (NW corner)
# └── tile_0651_6861.laz  (NE corner)
```

### 2. Verify Tile Continuity

```python
import laspy
from pathlib import Path

def check_tile_bounds(tile_path):
    """Check tile boundaries."""
    las = laspy.read(tile_path)

    print(f"\n{tile_path.name}:")
    print(f"  X: [{las.x.min():.2f}, {las.x.max():.2f}]")
    print(f"  Y: [{las.y.min():.2f}, {las.y.max():.2f}]")
    print(f"  Points: {len(las.points):,}")

    return {
        "name": tile_path.name,
        "x_min": las.x.min(),
        "x_max": las.x.max(),
        "y_min": las.y.min(),
        "y_max": las.y.max(),
        "points": len(las.points)
    }

# Check all tiles
input_dir = Path("data/input")
tiles = sorted(input_dir.glob("*.laz"))

bounds = [check_tile_bounds(tile) for tile in tiles]

# Check for gaps
print("\n" + "="*50)
print("Tile Adjacency Check:")
# Should show continuous coverage
```

---

## 🔧 Basic Stitching Configuration

### Configuration File

Create `config_tile_stitching.yaml`:

```yaml
# config_tile_stitching.yaml
defaults:
  - base/processor
  - base/features
  - base/data_sources
  - base/output
  - base/monitoring
  - _self_

# Multi-tile processing
processor:
  batch_size: 4 # Process tiles in batches
  num_workers: 4 # Parallel workers
  use_gpu: false

# Boundary-aware features
features:
  compute_normals: true
  k_neighbors: 50

  # Boundary awareness
  boundary_aware:
    enabled: true
    overlap_width: 10.0 # 10m overlap for seamless stitching
    edge_buffer: 5.0 # 5m buffer at edges

  # Cross-tile neighbor search
  cross_tile_neighbors: true

# Tile stitching
stitching:
  enabled: true
  method: "seamless" # "seamless" or "simple"

  # Overlap handling
  overlap_strategy: "average" # "average", "max_quality", or "first"

  # Edge refinement
  edge_refinement:
    enabled: true
    smooth_transitions: true
    blend_width: 2.0 # 2m blend zone at edges

# Output
output:
  formats:
    laz: true
  output_suffix: "_stitched"

  # Preserve tile structure
  preserve_structure: false # Output single stitched file

monitoring:
  log_level: "INFO"
  show_progress: true
```

---

## 🚀 Basic Stitching Workflow

### Step 1: Process Tiles with Boundary Awareness

```bash
# Process all tiles with boundary awareness
ign-lidar-hd process \
  --config-name config_tile_stitching \
  input_dir=data/input/ \
  output_dir=data/processed/

# Output:
# data/processed/
# ├── tile_0650_6860_stitched.laz
# ├── tile_0651_6860_stitched.laz
# ├── tile_0650_6861_stitched.laz
# └── tile_0651_6861_stitched.laz
```

### Step 2: Stitch Tiles Together

```bash
# Stitch all processed tiles
ign-lidar-hd stitch \
  input_dir=data/processed/ \
  output_file=data/output/versailles_stitched.laz \
  --seamless \
  --edge-buffer 5.0

# Output: Single seamless file
# data/output/versailles_stitched.laz
```

### Step 3: Verify Stitching Quality

```python
import laspy
import numpy as np

# Read stitched file
stitched = laspy.read("data/output/versailles_stitched.laz")

print(f"Stitched point cloud:")
print(f"  Total points: {len(stitched.points):,}")
print(f"  X range: [{stitched.x.min():.2f}, {stitched.x.max():.2f}]")
print(f"  Y range: [{stitched.y.min():.2f}, {stitched.y.max():.2f}]")
print(f"  Z range: [{stitched.z.min():.2f}, {stitched.z.max():.2f}]")

# Check for duplicate points at boundaries
from scipy.spatial import cKDTree

# Build KD-tree
tree = cKDTree(np.column_stack([stitched.x, stitched.y, stitched.z]))

# Find points closer than 0.01m (likely duplicates)
duplicates = tree.query_pairs(r=0.01)

print(f"\nQuality check:")
print(f"  Potential duplicates: {len(duplicates)}")
print(f"  Duplicate rate: {len(duplicates)/len(stitched.points)*100:.4f}%")

# Should be very low (<0.01%) for good stitching
```

---

## ⚡ Advanced Stitching

### GPU-Accelerated Processing

```yaml
# config_tile_stitching_gpu.yaml
defaults:
  - base/processor
  - base/features
  - base/data_sources
  - base/output
  - base/monitoring
  - _self_

# GPU processing
processor:
  batch_size: 8 # Larger batches for GPU
  use_gpu: true
  gpu_device: 0
  chunk_size: 2_000_000

# GPU-accelerated features
features:
  compute_normals: true
  compute_curvature: true # GPU-accelerated
  k_neighbors: 50

  boundary_aware:
    enabled: true
    overlap_width: 10.0
    use_gpu: true # GPU-accelerated overlap processing

stitching:
  enabled: true
  method: "seamless"

  # GPU-accelerated stitching
  use_gpu: true

  overlap_strategy: "max_quality" # Keep highest quality points

  edge_refinement:
    enabled: true
    smooth_transitions: true
    blend_width: 2.0
    use_gpu: true # GPU-accelerated blending

output:
  formats:
    laz: true
  compression_level: 7
  output_suffix: "_stitched_gpu"

monitoring:
  metrics:
    track_gpu: true
```

### Process with GPU

```bash
# GPU-accelerated processing and stitching
ign-lidar-hd process \
  --config-name config_tile_stitching_gpu \
  input_dir=data/input/ \
  output_dir=data/processed_gpu/

# Performance:
# CPU: ~2 tiles/hour
# GPU: ~10 tiles/hour (RTX 4080 Super)
```

---

## 🎨 Advanced Stitching Strategies

### 1. Quality-Based Stitching

```yaml
# config_quality_stitching.yaml
stitching:
  enabled: true
  method: "seamless"

  # Use point quality for overlap resolution
  overlap_strategy: "max_quality"

  quality_metrics:
    - "return_number" # Prefer first returns
    - "point_density" # Prefer denser areas
    - "normal_confidence" # Prefer confident normals
    - "edge_distance" # Prefer points away from edges

  # Weighted combination
  quality_weights:
    return_number: 0.3
    point_density: 0.2
    normal_confidence: 0.3
    edge_distance: 0.2
```

### 2. Feature-Preserving Stitching

```yaml
# config_feature_preserving_stitching.yaml
stitching:
  enabled: true
  method: "seamless"

  # Preserve features across boundaries
  feature_preservation:
    enabled: true
    preserve_buildings: true
    preserve_roads: true
    preserve_vegetation: true

  # Smart edge handling
  edge_refinement:
    enabled: true
    smooth_transitions: true
    preserve_sharp_features: true # Keep building edges sharp
    blend_width: 2.0
    feature_aware_blending: true # Different blend for features
```

### 3. Large Dataset Stitching

```yaml
# config_large_dataset_stitching.yaml
processor:
  batch_size: 16
  num_workers: 8
  max_memory_mb: 16384 # 16 GB RAM

stitching:
  enabled: true
  method: "seamless"

  # Memory-efficient stitching
  streaming_mode: true
  chunk_size: 10_000_000 # Process 10M points at a time

  # Temporary storage
  temp_dir: "/tmp/stitching"
  cleanup_temp: true # Remove temp files after stitching

  # Progressive output
  progressive_output: true # Output partial results
```

---

## 🔧 Python API Examples

### Example 1: Basic Stitching

```python
from ign_lidar.preprocessing.tile_stitcher import TileStitcher
from pathlib import Path

# Initialize stitcher
stitcher = TileStitcher(
    overlap_width=10.0,
    edge_buffer=5.0,
    seamless=True
)

# Load tiles
input_dir = Path("data/processed")
tiles = list(input_dir.glob("*.laz"))

print(f"Loading {len(tiles)} tiles...")

# Stitch tiles
stitched_cloud = stitcher.stitch_tiles(
    tiles,
    output_path=Path("data/output/stitched.laz"),
    overlap_strategy="average"
)

print(f"✅ Stitched {len(tiles)} tiles")
print(f"   Total points: {len(stitched_cloud.points):,}")
```

### Example 2: Progressive Stitching

```python
from ign_lidar.preprocessing.tile_stitcher import TileStitcher
import laspy

# Initialize stitcher with streaming
stitcher = TileStitcher(
    overlap_width=10.0,
    streaming_mode=True,
    chunk_size=10_000_000
)

# Progressive stitching
output_path = Path("data/output/large_stitched.laz")

with laspy.open(output_path, mode='w') as writer:
    for i, tile_path in enumerate(tiles):
        print(f"Processing tile {i+1}/{len(tiles)}: {tile_path.name}")

        # Stitch tile incrementally
        chunk = stitcher.stitch_tile_incremental(
            tile_path,
            previous_tiles=tiles[:i]  # Consider previous tiles for overlap
        )

        # Write chunk
        writer.write_points(chunk)

        # Progress
        progress = (i + 1) / len(tiles) * 100
        print(f"  Progress: {progress:.1f}%")

print("✅ Progressive stitching complete")
```

### Example 3: Quality-Based Stitching

```python
from ign_lidar.preprocessing.tile_stitcher import TileStitcher
import numpy as np

def calculate_point_quality(points):
    """Calculate quality score for each point."""

    # Prefer first returns
    return_quality = (points.return_number == 1).astype(float) * 0.3

    # Prefer points away from edges
    x_center = (points.x.min() + points.x.max()) / 2
    y_center = (points.y.min() + points.y.max()) / 2
    distance_from_center = np.sqrt(
        (points.x - x_center)**2 + (points.y - y_center)**2
    )
    max_distance = np.max(distance_from_center)
    edge_quality = (1 - distance_from_center / max_distance) * 0.3

    # Prefer points with high intensity (better signal)
    intensity_quality = (points.intensity / 65535) * 0.2

    # Prefer points with confident normals (if available)
    if hasattr(points, 'normal_confidence'):
        normal_quality = points.normal_confidence * 0.2
    else:
        normal_quality = np.ones(len(points)) * 0.2

    # Total quality score
    quality = return_quality + edge_quality + intensity_quality + normal_quality

    return quality

# Stitch with quality-based overlap resolution
stitcher = TileStitcher(
    overlap_width=10.0,
    overlap_strategy="custom",
    quality_function=calculate_point_quality
)

stitched = stitcher.stitch_tiles(
    tiles,
    output_path=Path("data/output/quality_stitched.laz")
)

print("✅ Quality-based stitching complete")
```

---

## 📊 Performance Optimization

### Batch Processing Strategy

```python
from ign_lidar.core.processor import LiDARProcessor
from pathlib import Path

def process_large_dataset(input_dir, output_dir, batch_size=16):
    """Process large dataset in batches with stitching."""

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all tiles
    tiles = sorted(input_path.glob("*.laz"))
    total_tiles = len(tiles)

    print(f"Processing {total_tiles} tiles in batches of {batch_size}")

    # Process in batches
    for i in range(0, total_tiles, batch_size):
        batch = tiles[i:i+batch_size]
        batch_num = i // batch_size + 1

        print(f"\nBatch {batch_num}/{(total_tiles + batch_size - 1) // batch_size}")
        print(f"  Tiles: {[t.name for t in batch]}")

        # Process batch
        processor = LiDARProcessor(
            input_dir=input_path,
            output_dir=output_path / f"batch_{batch_num}",
            boundary_aware=True,
            overlap_width=10.0
        )

        processor.process_batch(batch)

        # Stitch batch
        from ign_lidar.preprocessing.tile_stitcher import TileStitcher
        stitcher = TileStitcher(overlap_width=10.0)

        stitched_output = output_path / f"batch_{batch_num}_stitched.laz"
        stitcher.stitch_tiles(
            [output_path / f"batch_{batch_num}" / t.name for t in batch],
            output_path=stitched_output
        )

        print(f"  ✅ Batch {batch_num} complete: {stitched_output.name}")

# Process entire dataset
process_large_dataset(
    "data/input",
    "data/output",
    batch_size=16
)
```

### Memory-Efficient Stitching

```python
from ign_lidar.preprocessing.tile_stitcher import TileStitcher
import psutil

def memory_efficient_stitch(tiles, output_path, max_memory_gb=8):
    """Stitch tiles with memory constraints."""

    max_memory_bytes = max_memory_gb * 1024**3

    stitcher = TileStitcher(
        overlap_width=10.0,
        streaming_mode=True
    )

    # Estimate memory per tile
    sample_tile = laspy.read(tiles[0])
    bytes_per_point = sample_tile.points.array.itemsize
    points_per_tile = len(sample_tile.points)
    memory_per_tile = bytes_per_point * points_per_tile

    # Calculate chunk size
    tiles_per_chunk = max(1, int(max_memory_bytes / memory_per_tile))

    print(f"Memory-efficient stitching:")
    print(f"  Max memory: {max_memory_gb} GB")
    print(f"  Tiles per chunk: {tiles_per_chunk}")

    # Process in chunks
    for i in range(0, len(tiles), tiles_per_chunk):
        chunk_tiles = tiles[i:i+tiles_per_chunk]

        print(f"\nProcessing chunk {i//tiles_per_chunk + 1}")
        print(f"  Memory usage: {psutil.virtual_memory().percent:.1f}%")

        # Stitch chunk
        chunk_output = output_path.parent / f"chunk_{i//tiles_per_chunk}.laz"
        stitcher.stitch_tiles(chunk_tiles, output_path=chunk_output)

        print(f"  ✅ Chunk saved: {chunk_output.name}")

    # Final merge
    print("\nMerging chunks...")
    chunk_files = sorted(output_path.parent.glob("chunk_*.laz"))
    stitcher.stitch_tiles(chunk_files, output_path=output_path)

    # Cleanup
    for chunk in chunk_files:
        chunk.unlink()

    print("✅ Memory-efficient stitching complete")
```

---

## 🔍 Quality Validation

### Boundary Artifact Detection

```python
import laspy
import numpy as np
from scipy.spatial import cKDTree

def detect_boundary_artifacts(laz_path, tile_size=1000):
    """Detect artifacts at tile boundaries."""

    las = laspy.read(laz_path)

    # Expected tile boundaries (multiples of tile_size)
    x_boundaries = np.arange(
        np.floor(las.x.min() / tile_size) * tile_size,
        np.ceil(las.x.max() / tile_size) * tile_size,
        tile_size
    )
    y_boundaries = np.arange(
        np.floor(las.y.min() / tile_size) * tile_size,
        np.ceil(las.y.max() / tile_size) * tile_size,
        tile_size
    )

    print(f"Checking for boundary artifacts...")
    print(f"  X boundaries: {len(x_boundaries)}")
    print(f"  Y boundaries: {len(y_boundaries)}")

    # Check density near boundaries
    boundary_width = 10.0  # 10m on each side

    artifacts = []

    for x_bound in x_boundaries:
        # Points near this boundary
        near_boundary = np.abs(las.x - x_bound) < boundary_width

        if np.sum(near_boundary) > 0:
            # Calculate point density
            boundary_points = las.points[near_boundary]
            density = len(boundary_points) / (boundary_width * 2 * las.y.ptp())

            # Check for density anomalies
            overall_density = len(las.points) / (las.x.ptp() * las.y.ptp())
            density_ratio = density / overall_density

            if density_ratio > 1.5 or density_ratio < 0.5:
                artifacts.append({
                    "type": "X boundary",
                    "position": x_bound,
                    "density_ratio": density_ratio,
                    "severity": "high" if abs(density_ratio - 1) > 0.5 else "medium"
                })

    for y_bound in y_boundaries:
        # Points near this boundary
        near_boundary = np.abs(las.y - y_bound) < boundary_width

        if np.sum(near_boundary) > 0:
            # Calculate point density
            boundary_points = las.points[near_boundary]
            density = len(boundary_points) / (boundary_width * 2 * las.x.ptp())

            overall_density = len(las.points) / (las.x.ptp() * las.y.ptp())
            density_ratio = density / overall_density

            if density_ratio > 1.5 or density_ratio < 0.5:
                artifacts.append({
                    "type": "Y boundary",
                    "position": y_bound,
                    "density_ratio": density_ratio,
                    "severity": "high" if abs(density_ratio - 1) > 0.5 else "medium"
                })

    print(f"\nArtifacts found: {len(artifacts)}")
    for artifact in artifacts:
        print(f"  {artifact['type']} at {artifact['position']:.2f}m:")
        print(f"    Density ratio: {artifact['density_ratio']:.2f}")
        print(f"    Severity: {artifact['severity']}")

    return artifacts

# Check stitched file
artifacts = detect_boundary_artifacts("data/output/versailles_stitched.laz")

if len(artifacts) == 0:
    print("\n✅ No boundary artifacts detected")
else:
    print(f"\n⚠️  {len(artifacts)} boundary artifacts detected")
```

---

## 🐛 Troubleshooting

### Issue 1: Duplicate Points at Boundaries

**Symptom**: Doubled point density at tile boundaries

**Solution**:

```yaml
stitching:
  overlap_strategy: "first" # Keep points from first tile only

  # Or use deduplication
  deduplicate:
    enabled: true
    tolerance: 0.01 # 1cm tolerance
```

### Issue 2: Visible Seams

**Symptom**: Visible discontinuities at tile boundaries

**Solution**:

```yaml
stitching:
  edge_refinement:
    enabled: true
    smooth_transitions: true
    blend_width: 5.0 # Increase blend width

  # Recalculate normals at boundaries
  recalculate_boundary_normals: true
```

### Issue 3: Memory Issues with Large Datasets

**Symptom**: Out of memory errors during stitching

**Solution**:

```yaml
stitching:
  streaming_mode: true
  chunk_size: 5_000_000 # Reduce chunk size
  temp_dir: "/tmp" # Use fast temp storage
```

---

## 📚 Related Documentation

- [Tile Stitching Feature](../features/tile-stitching.md)
- [Boundary-Aware Processing](../features/boundary-aware.md)
- [Configuration V5](../guides/configuration-v5.md)
- [Performance Optimization](../guides/performance.md)

---

## 🎯 Summary

You've learned how to:

- ✅ Configure boundary-aware tile processing
- ✅ Stitch multiple tiles seamlessly
- ✅ Handle edge effects and artifacts
- ✅ Optimize performance for large datasets
- ✅ Validate stitching quality

**Next Steps**:

- Try [ASPRS Classification Example](./asprs-classification-example.md)
- Explore [LOD2 Classification](./lod2-classification-example.md)
- Learn about [GPU Acceleration](../guides/gpu-acceleration.md)

---

**Tutorial Version**: 1.0  
**Last Updated**: October 17, 2025  
**Tested With**: IGN LiDAR HD Dataset v5.0.0
