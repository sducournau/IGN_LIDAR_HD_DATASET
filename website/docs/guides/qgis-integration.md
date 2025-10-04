---
sidebar_position: 3
title: QGIS Integration
description: Guide for using enriched LAZ files in QGIS for visualization and analysis
keywords: [qgis, visualization, laz, point cloud, gis]
---

# QGIS Integration Guide

Learn how to visualize and analyze enriched LiDAR files in QGIS with geometric features for building component analysis.

## Overview

Enriched LAZ files from this library are fully compatible with QGIS and include 30+ geometric features perfect for:

- Building component visualization
- Surface analysis (walls, roofs, ground)
- Edge detection and linearity analysis
- Density and roughness mapping

## Prerequisites

### QGIS Requirements

- **QGIS 3.10+** with point cloud support
- **LAZ/LAS reader plugin** (usually included)

### Verify Installation

```bash
# Check if QGIS can read point clouds
# Open QGIS and look for: Layer > Add Layer > Add Point Cloud Layer
```

## Step 1: Enrich LAZ Files

First, create enriched LAZ files with geometric features:

```bash
# Enrich tiles with all features
ign-lidar-hd enrich \
  --input-dir /path/to/raw_tiles/ \
  --output /path/to/enriched_tiles/ \
  --mode full \
  --num-workers 4
```

This adds 30+ attributes to each point including:

- **Surface properties**: planarity, sphericity, linearity
- **Geometric features**: normals, curvature, roughness
- **Building analysis**: verticality, wall_score, roof_score
- **Height metrics**: height_above_ground, vertical_std

## Step 2: Load in QGIS

### Import Point Cloud

1. **Open QGIS**
2. **Menu**: Layer → Add Layer → Add Point Cloud Layer
3. **Browse**: Select your enriched LAZ file
4. **Click Add**: Point cloud appears in the map

### Initial Display

The point cloud will initially display with default elevation coloring. You'll see your LiDAR data as colored points.

## Step 3: Visualize Features

### Access Symbology

1. **Right-click** on the layer → Properties
2. **Symbology tab**
3. **Rendered by**: Attribute
4. **Attribute**: Select feature to visualize

### Available Features

#### Core Geometric Features

| Feature      | Description            | Range  | Use Case                             |
| ------------ | ---------------------- | ------ | ------------------------------------ |
| `planarity`  | Surface flatness score | 0-1    | Detect walls, roofs, ground          |
| `linearity`  | Edge/line score        | 0-1    | Find building edges, cables          |
| `sphericity` | 3D spherical score     | 0-1    | Identify vegetation, rounded objects |
| `curvature`  | Surface curvature      | varies | Edge detection, surface analysis     |
| `roughness`  | Surface texture        | varies | Material classification              |

#### Building-Specific Features

| Feature               | Description                | Range  | Use Case                 |
| --------------------- | -------------------------- | ------ | ------------------------ |
| `verticality`         | Vertical orientation score | 0-1    | Wall detection           |
| `wall_score`          | Wall probability           | 0-1    | Building facade analysis |
| `roof_score`          | Roof probability           | 0-1    | Roof surface detection   |
| `height_above_ground` | Normalized height          | meters | Multi-story analysis     |

#### Normal Vectors

| Feature    | Description                   | Range   | Use Case            |
| ---------- | ----------------------------- | ------- | ------------------- |
| `normal_x` | X component of surface normal | -1 to 1 | Surface orientation |
| `normal_y` | Y component of surface normal | -1 to 1 | Surface orientation |
| `normal_z` | Z component of surface normal | -1 to 1 | Surface orientation |

## Step 4: Recommended Visualizations

### Building Wall Detection

**Goal**: Highlight building walls

**Settings**:

- **Attribute**: `wall_score`
- **Color Ramp**: RdYlGn (Red-Yellow-Green)
- **Min**: 0.0, **Max**: 1.0
- **Filter**: `wall_score > 0.7` (optional)

**Interpretation**:

- Red (0.0-0.3): Non-wall points (ground, vegetation)
- Yellow (0.4-0.6): Possible walls
- Green (0.7-1.0): High confidence walls

### Roof Surface Analysis

**Goal**: Identify flat roof surfaces

**Settings**:

- **Attribute**: `planarity`
- **Color Ramp**: Viridis
- **Min**: 0.0, **Max**: 1.0
- **Filter**: `roof_score > 0.5 AND verticality < 0.3`

**Interpretation**:

- Dark blue (0.0-0.3): Irregular surfaces
- Green-yellow (0.7-1.0): Flat surfaces (roof candidates)

### Building Edge Detection

**Goal**: Find building corners and edges

**Settings**:

- **Attribute**: `linearity`
- **Color Ramp**: Hot (Black to White)
- **Min**: 0.0, **Max**: 1.0
- **Filter**: `linearity > 0.6`

**Interpretation**:

- Black (0.0-0.3): Flat surfaces
- White (0.7-1.0): Linear features (edges, corners)

### Height-based Analysis

**Goal**: Visualize building stories

**Settings**:

- **Attribute**: `height_above_ground`
- **Color Ramp**: Turbo or Plasma
- **Min**: 0, **Max**: 30 (adjust to your data)
- **Filter**: `wall_score > 0.5`

**Interpretation**:

- Blue (0-3m): Ground floor
- Green (3-6m): First floor
- Yellow-Red (6m+): Upper floors

## Advanced Filtering

### Multi-attribute Filtering

Use QGIS expressions for complex filtering:

```sql
-- High-confidence building walls above ground floor
"wall_score" > 0.8 AND "height_above_ground" > 3 AND "height_above_ground" < 20

-- Flat roof surfaces
"planarity" > 0.7 AND "roof_score" > 0.6 AND "verticality" < 0.2

-- Building edges at upper levels
"linearity" > 0.6 AND "height_above_ground" > 3

-- Vegetation points (high sphericity, low planarity)
"sphericity" > 0.5 AND "planarity" < 0.4
```

### Expression Builder

1. **Right-click layer** → Filter
2. **Expression builder** opens
3. **Enter expression** using attribute names
4. **Test** and **Apply**

## Visualization Examples

### Example 1: Building Component Classification

**Steps**:

1. Load enriched LAZ file
2. Create **3 separate layers** (duplicate the layer)
3. **Layer 1**: Walls (`wall_score > 0.7`, red color)
4. **Layer 2**: Roofs (`roof_score > 0.7 AND verticality < 0.3`, blue color)
5. **Layer 3**: Ground (`height_above_ground < 1`, green color)

**Result**: Color-coded building components

### Example 2: Surface Quality Analysis

**Steps**:

1. Visualize by `roughness` attribute
2. Use discrete classes: Smooth (0-0.1), Medium (0.1-0.3), Rough (0.3+)
3. Apply to identify material types

### Example 3: Architectural Detail Detection

**Steps**:

1. Filter: `linearity > 0.6 AND height_above_ground > 2`
2. Color by `height_above_ground`
3. Result: Building edges and architectural details colored by height

## Troubleshooting

### Issue: LAZ File Won't Open

**Symptoms**: "Cannot read file" error

**Solutions**:

1. Check QGIS version (needs 3.10+)
2. Verify LAZ support: Plugins → Manage Plugins → Search "LAZ"
3. Try converting to LAS format if needed

### Issue: No Custom Attributes Visible

**Symptoms**: Only X, Y, Z, Intensity available

**Verification**:

```bash
# Check if enrichment worked
python -c "
import laspy
las = laspy.read('enriched_file.laz')
print('Extra dimensions:', las.point_format.extra_dimension_names)
"
```

**Solutions**:

1. Re-run enrichment process
2. Ensure enrichment completed successfully
3. Check file wasn't overwritten

### Issue: Visualization Looks Noisy

**Symptoms**: Point cloud shows scan line artifacts

**Cause**: Using old fixed k-neighbors instead of adaptive radius

**Solution**: Update to latest version with radius-based feature computation

### Issue: Performance Problems

**Symptoms**: QGIS slow with large files

**Solutions**:

1. **Limit point display**: Set maximum points in layer properties
2. **Use Level of Detail**: Enable in rendering settings
3. **Clip to area**: Use spatial filter for region of interest
4. **Simplify**: Process smaller tiles or downsample

## Data Export

### Export Filtered Points

1. **Right-click layer** → Export → Save Features As
2. **Format**: LAS/LAZ
3. **Filter**: Use current filter or create new one
4. **Save**: Filtered point cloud as new file

### Export to Other Formats

- **CSV**: For spreadsheet analysis
- **Shapefile**: For vector GIS analysis (centroids)
- **GeoJSON**: For web mapping

## Performance Tips

### Large Dataset Handling

```bash
# Process smaller tiles for better QGIS performance
ign-lidar-hd process \
  --input large_tile.laz \
  --output patches/ \
  --patch-size 20.0  # Smaller patches

# Or clip large files geographically
# Use QGIS Clip tool or command line utilities
```

### Display Optimization

- **Point budget**: Limit to 1M points for smooth interaction
- **Level of detail**: Enable automatic point reduction
- **Attribute caching**: QGIS will cache attribute statistics

## Integration with Other Tools

### Combine with Vector Data

- **Building footprints**: Overlay building polygons
- **Road networks**: Add context with street data
- **Property boundaries**: Combine with cadastral data

### Export Results

- **Screenshots**: High-resolution map exports
- **Analysis results**: Statistical summaries
- **Filtered datasets**: Export classified points

## See Also

- [Basic Usage Guide](basic-usage.md) - Creating enriched LAZ files
- [CLI Commands](cli-commands.md) - Enrichment command reference
- [QGIS Troubleshooting](qgis-troubleshooting.md) - Common issues and solutions
- [Memory Optimization](../reference/memory-optimization.md) - Performance tuning
