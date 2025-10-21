---
sidebar_position: 2
title: Building Analysis & Fusion
description: Multi-source building polygon fusion and cluster enrichment
tags: [buildings, fusion, clustering, quality-scoring, metadata]
---

# 🏢 Building Analysis & Fusion

:::tip Version & Status
**Version:** 5.1  
**Date:** October 2025  
**Status:** Production Ready ✅
:::

## Overview

The **Building Analysis & Fusion** system provides comprehensive tools for working with building data in LiDAR point clouds. It combines two powerful capabilities:

1. **Multi-Source Fusion** - Intelligently merge building polygons from multiple data sources (BD TOPO®, Cadastre, OpenStreetMap)
2. **Cluster Enrichment** - Add rich spatial metadata by organizing points into building clusters with architectural properties

### Key Capabilities

✅ **Multi-source integration** - BD TOPO®, Cadastre, OpenStreetMap  
✅ **Quality scoring** - Automatic evaluation of polygon accuracy (coverage, geometry, completeness)  
✅ **Intelligent fusion** - Best selection or weighted merge strategies  
✅ **Adaptive adjustment** - Translation, scaling, rotation, buffering  
✅ **Conflict resolution** - Merge overlapping buildings, remove duplicates  
✅ **Cluster enrichment** - 18 metadata fields per point (parcel ID, building type, facade ID, etc.)  
✅ **Hierarchical relationships** - Link buildings to parcels for spatial coherence

---

## Multi-Source Building Fusion

### Data Sources

#### 1. BD TOPO® (Primary Source)

**Characteristics:**

- Official IGN source
- High geometric precision
- Regular updates
- Comprehensive coverage

**Typical Quality:**

- Average score: 0.75-0.85
- Coverage: 60-80% of building points
- Centroid accuracy: ±2m

**Configuration:**

```yaml
data_sources:
  bd_topo:
    enabled: true
    features:
      buildings: true
```

#### 2. Cadastre (Secondary Source)

**Characteristics:**

- Cadastral parcels (legal boundaries)
- Very precise geometry
- May differ from actual buildings
- Complements BD TOPO where missing

**Typical Quality:**

- Average score: 0.60-0.70
- Coverage: 50-70% of building points
- Frequent offset (parcel ≠ building)

**Configuration:**

```yaml
data_sources:
  cadastre:
    enabled: true
    use_as_building_proxy: true
```

#### 3. OpenStreetMap (Tertiary Source)

**Characteristics:**

- Community-contributed data
- Variable quality by area
- Often up-to-date in urban zones
- Complements official sources

**Typical Quality:**

- Average score: 0.50-0.75
- Coverage: 40-70% of building points
- Good quality in urban areas
- Variable quality in rural areas

**Configuration:**

```yaml
data_sources:
  osm:
    enabled: true
    overpass_url: "https://overpass-api.de/api/interpreter"
    timeout: 180

    building_tags:
      - "building=yes"
      - "building=house"
      - "building=residential"
      - "building=apartments"

    # Quality filters
    min_building_area: 10.0 # m²
    max_building_area: 10000.0 # m²

    cache_enabled: true
    cache_ttl_days: 30
```

### Quality Scoring System

Each polygon is evaluated using multiple criteria to determine its fitness:

#### 1. Coverage Score (40% weight)

**Definition:** Percentage of building points inside the polygon

```python
coverage_ratio = points_inside_polygon / total_building_points
```

**Criteria:**

- Points inside polygon
- Points near polygon (1m buffer)
- Coverage ratio (0-1)

**Example:**

```
BD TOPO:  850/1000 points = 0.85
Cadastre: 720/1000 points = 0.72
OSM:      780/1000 points = 0.78
→ BD TOPO has best coverage
```

#### 2. Geometric Fit Score (30% weight)

**Centroid Offset:**

```python
centroid_offset = distance(polygon_centroid, point_cloud_centroid)
penalty = exp(-offset / 2.0)
```

**Area Ratio:**

```python
area_ratio = polygon_area / point_cloud_area
penalty = 1.0 - abs(1.0 - area_ratio)
```

**Shape Similarity:**

- IoU (Intersection over Union)
- Contour matching

**Example:**

```
BD TOPO:
  - Offset: 1.2m → penalty = 0.88
  - Area ratio: 1.1 → penalty = 0.90
  - IoU: 0.75
  → Geometric score: (0.88 + 0.90 + 0.75) / 3 = 0.84

Cadastre:
  - Offset: 3.5m → penalty = 0.55
  - Area ratio: 0.8 → penalty = 0.80
  - IoU: 0.60
  → Geometric score: 0.65
```

#### 3. Completeness Score (30% weight)

**Wall Coverage:**

```python
wall_coverage = wall_points_inside / total_wall_points
# Walls detected by verticality >= 0.7
```

**Roof Coverage:**

```python
roof_coverage = roof_points_inside / total_roof_points
# Roofs detected by verticality < 0.7
```

**Example:**

```
BD TOPO:
  - Walls: 320/350 = 0.91
  - Roofs: 530/650 = 0.82
  → Completeness score: 0.87

OSM:
  - Walls: 280/350 = 0.80
  - Roofs: 500/650 = 0.77
  → Completeness score: 0.78
```

#### Overall Quality Score

```python
quality_score = (
    0.4 * coverage_score +
    0.3 * geometric_score +
    0.3 * completeness_score
)
```

**Interpretation:**

| Score Range | Quality Level           |
| ----------- | ----------------------- |
| > 0.80      | Excellent               |
| 0.60-0.80   | Good                    |
| 0.50-0.60   | Acceptable              |
| < 0.50      | Insufficient (rejected) |

### Fusion Strategies

#### 1. Best Selection Mode (Recommended)

**Principle:** Select the polygon with the highest quality score

**Algorithm:**

1. Sort by priority (BD TOPO > Cadastre > OSM)
2. Select best score if above quality threshold
3. Switch to lower priority source only if difference > 0.15

**Configuration:**

```yaml
building_fusion:
  fusion_mode: "best"
  source_priority:
    - "bd_topo"
    - "cadastre"
    - "osm"
  min_quality_score: 0.5
  quality_difference_threshold: 0.15
```

**Example:**

```
BD TOPO:  0.78
Cadastre: 0.85 (better +0.07)
OSM:      0.62

Result: BD TOPO (difference < 0.15, priority respected)

But if:
BD TOPO:  0.65
Cadastre: 0.85 (better +0.20)

Result: Cadastre (difference > 0.15, switch accepted)
```

#### 2. Weighted Merge Mode

**Principle:** Merge multiple polygons weighted by quality

**Algorithm:**

1. Filter sources with score > 0.5
2. Weighted geometric union
3. Simplify resulting polygon

**Configuration:**

```yaml
building_fusion:
  fusion_mode: "weighted_merge"
  enable_multi_source_fusion: true
```

**Result:**

- Larger polygon (union)
- Captures more points
- May include non-building areas

**Use Case:** Areas with conflicting data

#### 3. Consensus Mode (Conservative)

**Principle:** Intersection of good-quality polygons

**Algorithm:**

1. Filter sources with score > 0.5
2. Geometric intersection
3. If intersection too small, fallback to weighted_merge

**Configuration:**

```yaml
building_fusion:
  fusion_mode: "consensus"
```

**Result:**

- Smaller polygon (intersection)
- Very high confidence
- May miss extensions

**Use Case:** Critical applications requiring high precision

### Adaptive Polygon Adjustment

The system automatically optimizes building polygons through a series of transformations:

#### 1. Translation (Movement)

**Principle:** Move polygon to match point cloud centroid

**Algorithm:**

1. Compute point cloud 2D centroid
2. Compute polygon centroid
3. Move if offset > 0.5m and < max_translation

**Configuration:**

```yaml
building_fusion:
  enable_translation: true
  max_translation: 5.0 # meters
```

**Example:**

```
Polygon centroid: (100.0, 200.0)
Point centroid:   (102.5, 201.8)
Offset: 2.9m

→ Translation applied: (+2.5m, +1.8m)
```

**Result:**

- Better coverage: +10-20%
- Perfect centroid alignment
- Corrects GPS/projection offsets

#### 2. Scaling (Size Adjustment)

**Principle:** Adjust size to match point cloud extent

**Algorithm:**

1. Compute point extent (width × height)
2. Compute polygon extent
3. `scale_factor = point_extent / polygon_extent`
4. Clamp between `1/max_scale` and `max_scale`

**Configuration:**

```yaml
building_fusion:
  enable_scaling: true
  max_scale_factor: 1.5 # 1.5x max expansion/contraction
```

**Example:**

```
Polygon: 20m × 15m
Points:  24m × 18m

Scale X: 24/20 = 1.20
Scale Y: 18/15 = 1.20
Average scale: 1.20

→ Scaling applied: 1.20x (20% expansion)
```

**Result:**

- Captures peripheral walls
- Better match with reality
- Corrects undersized polygons

#### 3. Rotation (Alignment)

**Principle:** Align with principal axes of point cloud

**Algorithm:**

1. PCA on points (principal axes)
2. Compute rotation angle
3. Apply if |angle| > 1° and < max_rotation

**Configuration:**

```yaml
building_fusion:
  enable_rotation: false # Disabled by default (expensive)
  max_rotation: 15.0 # degrees
```

:::warning Performance Impact
Rotation is very CPU-intensive (PCA computation). Little benefit for regular buildings. Recommended: Disable unless specific needs.
:::

#### 4. Adaptive Buffering

**Principle:** Variable buffer based on wall detection

**Algorithm:**

1. Detect walls (verticality >= 0.7)
2. Compute wall ratio
3. Adaptive buffer: `buffer = min_buffer + (max_buffer - min_buffer) × wall_ratio`

**Configuration:**

```yaml
building_fusion:
  enable_buffering: true
  adaptive_buffer_range: [0.3, 1.0] # min/max meters
```

**Example:**

```
Total points: 1000
Wall points (verticality >= 0.7): 350
Wall ratio: 0.35

Buffer = 0.3 + (1.0 - 0.3) × 0.35 = 0.55m

→ Buffer applied: 0.55m
```

**Result:**

- Small buffer (0.3m) for flat roofs
- Large buffer (0.8-1.0m) for many walls
- Optimal wall point capture

### Conflict Resolution

#### 1. Overlap Detection

**Method: IoU (Intersection over Union)**

```python
intersection = polygon1.intersection(polygon2).area
union = polygon1.union(polygon2).area
iou = intersection / union

if iou >= overlap_threshold:  # 0.3 by default
    # Conflict detected
```

**Configuration:**

```yaml
building_fusion:
  overlap_threshold: 0.3 # IoU 30%
```

#### 2. Merging Nearby Buildings

**Criteria:**

- Distance < 2m
- IoU >= 0.3
- Maximum 2 buildings to merge

**Algorithm:**

1. Detect nearby buildings
2. Geometric union
3. Polygon simplification
4. Point accumulation

**Configuration:**

```yaml
building_fusion:
  merge_nearby_buildings: true
  merge_distance_threshold: 2.0 # meters
```

**Example:**

```
Building A: 850 points, 200m² polygon
Building B: 320 points, 80m² polygon
Distance: 1.5m
IoU: 0.15 (no significant overlap)

→ Merge applied:
  - Polygon: union(A, B) = 275m²
  - Points: 1170 points
  - Source: FUSED
```

#### 3. Duplicate Removal

**Criteria:**

- IoU > 0.7 (significant overlap)
- Keep building with more points

**Algorithm:**

1. Sort by point count
2. Remove heavily overlapped buildings

---

## Cluster Enrichment

### Overview

The **Cluster Enrichment** feature adds rich spatial metadata to LiDAR point clouds by organizing points into **parcel clusters** and **building clusters**. This provides:

- **Parcel Clustering** - Group points by cadastral parcels with land use metadata
- **Building Clustering** - Identify individual buildings with architectural properties
- **Hierarchical Relationships** - Link buildings to parcels for spatial coherence
- **Rich Metadata** - 18 new attributes per point for advanced analysis

### New Point Cloud Fields

#### Parcel Cluster Fields (9 fields)

| Field Name                | Type    | Description                     | Example Values                      |
| ------------------------- | ------- | ------------------------------- | ----------------------------------- |
| `ParcelID`                | str     | Unique cadastral parcel ID      | "75056000AB0123"                    |
| `ParcelType`              | uint8   | Land use type (encoded)         | 1=forest, 2=agriculture, 3=building |
| `ParcelConfidence`        | float32 | Classification confidence       | 0.0-1.0                             |
| `ClusterDensity`          | float32 | Point density (pts/m²)          | 0.5-50.0                            |
| `DistanceToClusterCenter` | float32 | Distance to parcel centroid (m) | 0.0-100.0                           |
| `IsClusterBoundary`       | uint8   | 1 if on parcel boundary         | 0 or 1                              |
| `NeighborClusters`        | uint8   | Number of neighboring parcels   | 0-255                               |
| `ClusterSize`             | uint32  | Total points in parcel          | 100-1000000                         |
| `ClusterHeight`           | float32 | Mean height of parcel (m)       | 0.0-50.0                            |

#### Building Cluster Fields (9 fields)

| Field Name                 | Type    | Description                     | Example Values                            |
| -------------------------- | ------- | ------------------------------- | ----------------------------------------- |
| `BuildingClusterID`        | int32   | Unique building cluster ID      | 0, 1, 2, ...                              |
| `BuildingType`             | uint8   | Building type (encoded)         | 1=residential, 2=multi-story, 3=high-rise |
| `BuildingConfidence`       | float32 | Classification confidence       | 0.0-1.0                                   |
| `BuildingHeight`           | float32 | Building height (m)             | 2.0-100.0                                 |
| `IsBuildingBoundary`       | uint8   | 1 if on building edge           | 0 or 1                                    |
| `FacadeID`                 | int16   | Facade/wall identifier          | 0, 1, 2, ...                              |
| `IsWall`                   | uint8   | 1 if vertical wall              | 0 or 1                                    |
| `IsRoof`                   | uint8   | 1 if horizontal roof            | 0 or 1                                    |
| `DistanceToBuildingCenter` | float32 | Distance to building center (m) | 0.0-50.0                                  |

### Field Encodings

**Parcel Type:**

```python
0 = Unknown
1 = Forest
2 = Agriculture
3 = Building
4 = Road
5 = Water
6 = Vegetation
7 = Mixed
```

**Building Type:**

```python
0 = Unknown
1 = Residential
2 = Multi-story
3 = High-rise
4 = Commercial
5 = Industrial
6 = Large
```

### Quick Start

#### Basic Usage

```python
from ign_lidar.core.classification.cluster_enrichment import enrich_with_clusters

# Enrich point cloud with cluster metadata
cluster_fields = enrich_with_clusters(
    points=points,              # XYZ coordinates
    labels=labels,              # ASPRS classification
    features=features,          # Geometric features
    ground_truth_features={
        'cadastre': cadastre_gdf,
        'buildings': buildings_gdf
    }
)

# Access new fields
parcel_ids = cluster_fields['ParcelID']
building_ids = cluster_fields['BuildingClusterID']
parcel_types = cluster_fields['ParcelType']
```

#### Advanced Configuration

```python
from ign_lidar.core.classification.cluster_enrichment import (
    ClusterEnricher, ClusterEnrichmentConfig
)

# Configure enrichment
config = ClusterEnrichmentConfig()
config.enable_parcel_enrichment = True
config.enable_building_enrichment = True
config.building_clustering.detect_facades = True

# Create enricher
enricher = ClusterEnricher(config)

# Get all cluster fields
all_fields = enricher.get_all_cluster_fields(
    points=points,
    labels=labels,
    features=features,
    ground_truth_features=ground_truth_features
)

# Export statistics
enricher.export_cluster_statistics(
    output_path=Path("cluster_stats.csv"),
    format="csv"
)
```

---

## Complete Integration Pipeline

### Full Example

```python
from ign_lidar.core.classification.building_fusion import (
    BuildingFusion, BuildingSource
)
from ign_lidar.core.classification.cluster_enrichment import ClusterEnricher

# Step 1: Load point cloud
points, colors = load_point_cloud("tile.laz")
normals = compute_normals(points, k_neighbors=30)
verticality = compute_verticality(normals)

# Step 2: Load building sources
building_sources = {
    BuildingSource.BD_TOPO: load_bd_topo(bbox),
    BuildingSource.CADASTRE: load_cadastre(bbox),
    BuildingSource.OSM: load_osm(bbox)
}

# Step 3: Create fusion system
fusion = BuildingFusion(
    source_priority=[
        BuildingSource.BD_TOPO,
        BuildingSource.CADASTRE,
        BuildingSource.OSM
    ],
    fusion_mode="best",

    # Adaptive adjustment
    enable_translation=True,
    enable_scaling=True,
    enable_rotation=False,
    enable_buffering=True,

    max_translation=5.0,
    max_scale_factor=1.5,
    adaptive_buffer_range=(0.3, 1.0),

    # Conflict resolution
    merge_nearby_buildings=True,
    overlap_threshold=0.3
)

# Step 4: Fuse buildings
fused_buildings, stats = fusion.fuse_building_sources(
    points=points,
    building_sources=building_sources,
    normals=normals,
    verticality=verticality
)

# Step 5: Enrich with cluster metadata
enricher = ClusterEnricher()
cluster_fields = enricher.get_all_cluster_fields(
    points=points,
    labels=labels,
    features=features,
    ground_truth_features={
        'buildings': fused_buildings,
        'cadastre': building_sources[BuildingSource.CADASTRE]
    }
)

# Step 6: Analyze results
print(f"Fused buildings: {len(fused_buildings)}")
print(f"Total points: {sum(b.n_points for b in fused_buildings):,}")

print("\nSources used:")
for source, count in stats['sources_used'].items():
    print(f"  {source}: {count} buildings")

print("\nAdaptations:")
print(f"  Translated: {stats['adaptations']['translated']}")
print(f"  Scaled: {stats['adaptations']['scaled']}")
print(f"  Rotated: {stats['adaptations']['rotated']}")
print(f"  Buffered: {stats['adaptations']['buffered']}")

# Step 7: Export enriched point cloud
save_enriched_las("output_enriched.laz", points, labels, cluster_fields)
save_fusion_report("fusion_report.json", stats)
```

---

## Configuration Reference

### Complete YAML Configuration

```yaml
# Building Fusion Configuration
building_fusion:
  # Fusion strategy
  fusion_mode: "best" # Options: "best", "weighted_merge", "consensus"

  # Source priority
  source_priority:
    - "bd_topo"
    - "cadastre"
    - "osm"

  # Quality thresholds
  min_quality_score: 0.5
  quality_difference_threshold: 0.15

  # Adaptive adjustment
  enable_translation: true
  max_translation: 5.0 # meters

  enable_scaling: true
  max_scale_factor: 1.5 # 1.5x max

  enable_rotation: false # Expensive, usually disabled
  max_rotation: 15.0 # degrees

  enable_buffering: true
  adaptive_buffer_range: [0.3, 1.0] # min/max meters

  # Conflict resolution
  merge_nearby_buildings: true
  overlap_threshold: 0.3 # IoU
  merge_distance_threshold: 2.0 # meters

  # Multi-source fusion
  enable_multi_source_fusion: true

# Data Sources
data_sources:
  bd_topo:
    enabled: true
    features:
      buildings: true

  cadastre:
    enabled: true
    use_as_building_proxy: true

  osm:
    enabled: true
    overpass_url: "https://overpass-api.de/api/interpreter"
    timeout: 180
    building_tags:
      - "building=yes"
      - "building=house"
      - "building=residential"
      - "building=apartments"
    min_building_area: 10.0
    max_building_area: 10000.0
    cache_enabled: true
    cache_ttl_days: 30

# Cluster Enrichment Configuration
cluster_enrichment:
  # Enable/disable modules
  enable_parcel_enrichment: true
  enable_building_enrichment: true
  enable_hierarchical_clustering: true

  # Parcel clustering
  parcel_clustering:
    min_parcel_points: 20
    min_parcel_area: 10.0 # m²
    boundary_distance_threshold: 2.0 # m
    neighbor_search_radius: 5.0 # m
    min_confidence_threshold: 0.5
    include_parcel_metadata: true
    include_statistical_features: true

  # Building clustering
  building_clustering:
    min_building_points: 15
    min_building_area: 8.0 # m²
    max_building_area: 15000.0 # m²
    clustering_method: "dbscan" # or "footprint", "hierarchical"
    spatial_eps: 1.5 # m - DBSCAN epsilon
    detect_facades: true
    detect_roof_sections: true
    min_facade_points: 20
    boundary_percentile: 90.0
    fusion_enabled: true
    source_priority: ["bd_topo", "cadastre", "osm"]

  # Performance
  use_spatial_index: true
  parallel_processing: false
  max_workers: 4

  # Output
  export_cluster_statistics: true
  export_format: "csv,geojson"
  verbose: true
```

---

## Use Cases

### 1. Urban Planning

**Building Density Analysis:**

```python
# Find high-density residential parcels
residential_parcels = cluster_fields['ParcelType'] == 1
high_density = cluster_fields['ClusterDensity'] > 20.0
candidates = residential_parcels & high_density
```

**Multi-Source Comparison:**

```python
# Compare building detection across sources
urban_zone_buildings = fusion.compare_sources(
    bbox=urban_zone,
    metrics=['coverage', 'geometric_fit', 'completeness']
)
```

### 2. Building Inventory

**Count Buildings Per Parcel:**

```python
for parcel_id in np.unique(cluster_fields['ParcelID']):
    parcel_mask = cluster_fields['ParcelID'] == parcel_id
    building_ids = np.unique(cluster_fields['BuildingClusterID'][parcel_mask])
    n_buildings = len(building_ids[building_ids > 0])
    print(f"Parcel {parcel_id}: {n_buildings} buildings")
```

**Building Type Classification:**

```python
# Classify by height
building_heights = cluster_fields['BuildingHeight']
residential = (building_heights >= 2.5) & (building_heights < 10.0)
multi_story = (building_heights >= 10.0) & (building_heights < 25.0)
high_rise = building_heights >= 25.0
```

### 3. Quality Control

**Validate Building-Parcel Alignment:**

```python
building_mask = labels == 6  # ASPRS Building
inside_parcel = cluster_fields['ParcelID'][building_mask] != ''
coverage = inside_parcel.sum() / len(inside_parcel) * 100
print(f"Building-parcel coverage: {coverage:.1f}%")
```

**Detect Polygon Misalignment:**

```python
# Find buildings with poor quality scores
poor_quality = [b for b in fused_buildings if b.quality_score < 0.6]
print(f"Buildings needing review: {len(poor_quality)}")
```

### 4. Facade Analysis

**Extract Individual Walls:**

```python
walls = cluster_fields['IsWall'] == 1
facades = cluster_fields['FacadeID'][walls]
unique_facades = np.unique(facades[facades > 0])
print(f"Detected {len(unique_facades)} facades")
```

**Wall Height Statistics:**

```python
for facade_id in unique_facades:
    facade_mask = cluster_fields['FacadeID'] == facade_id
    facade_points = points[facade_mask]
    height_range = facade_points[:, 2].max() - facade_points[:, 2].min()
    print(f"Facade {facade_id}: {height_range:.1f}m height")
```

### 5. Historical Analysis

**Prioritize Recent OSM Data:**

```yaml
# For historical zones with updated OSM
source_priority:
  - "osm" # Most recent
  - "bd_topo"
  - "cadastre"
min_quality_score: 0.4 # More permissive
```

---

## Performance & Metrics

### Processing Performance

**Per km² tile:**

| Operation            | Time         | Memory      |
| -------------------- | ------------ | ----------- |
| Multi-source loading | 30-60 sec    | ~50 MB      |
| Quality scoring      | 1-2 min      | ~30 MB      |
| Polygon adaptation   | 1-2 min      | ~40 MB      |
| Conflict resolution  | 30 sec       | ~20 MB      |
| Parcel clustering    | 2-5 sec      | ~30 MB      |
| Building clustering  | 3-8 sec      | ~40 MB      |
| Metadata computation | 2-5 sec      | ~30 MB      |
| **Total**            | **8-12 min** | **~240 MB** |

### Expected Results

**Source Distribution (typical urban area):**

```
Input:
  BD TOPO:  250 buildings
  Cadastre: 380 parcels
  OSM:      180 buildings

Fused Output: 265 buildings

Source Breakdown:
  BD TOPO:  185 (70%)
  Cadastre:  50 (19%)
  OSM:       25 (9%)
  FUSED:      5 (2%)
```

**Average Quality Scores:**

| Source   | Average | Min  | Max  |
| -------- | ------- | ---- | ---- |
| BD TOPO  | 0.78    | 0.55 | 0.92 |
| Cadastre | 0.66    | 0.42 | 0.85 |
| OSM      | 0.61    | 0.35 | 0.88 |

**Adaptations Applied (per 100 buildings):**

| Adaptation  | Count | Average Improvement |
| ----------- | ----- | ------------------- |
| Translation | 72    | +15% coverage       |
| Scaling     | 58    | +10% coverage       |
| Rotation    | 12    | +5% coverage        |
| Buffering   | 85    | +20% wall points    |

---

## API Reference

### `BuildingFusion` Class

```python
class BuildingFusion:
    def __init__(
        self,
        source_priority: List[BuildingSource],
        fusion_mode: str = "best",
        enable_translation: bool = True,
        enable_scaling: bool = True,
        enable_rotation: bool = False,
        enable_buffering: bool = True,
        max_translation: float = 5.0,
        max_scale_factor: float = 1.5,
        adaptive_buffer_range: Tuple[float, float] = (0.3, 1.0),
        merge_nearby_buildings: bool = True,
        overlap_threshold: float = 0.3
    )

    def fuse_building_sources(
        self,
        points: np.ndarray,
        building_sources: Dict[BuildingSource, gpd.GeoDataFrame],
        normals: np.ndarray,
        verticality: np.ndarray
    ) -> Tuple[List[Building], Dict[str, Any]]
```

### `ClusterEnricher` Class

```python
class ClusterEnricher:
    def __init__(
        self,
        config: Optional[ClusterEnrichmentConfig] = None
    )

    def enrich_parcel_clusters(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        features: Dict[str, np.ndarray],
        cadastre_gdf: gpd.GeoDataFrame,
        bd_foret_gdf: Optional[gpd.GeoDataFrame] = None,
        rpg_gdf: Optional[gpd.GeoDataFrame] = None
    ) -> Dict[str, np.ndarray]

    def enrich_building_clusters(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        features: Dict[str, np.ndarray],
        building_gdf: Optional[gpd.GeoDataFrame] = None,
        parcel_fields: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]

    def get_all_cluster_fields(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        features: Dict[str, np.ndarray],
        ground_truth_features: Dict[str, gpd.GeoDataFrame]
    ) -> Dict[str, np.ndarray]

    def export_cluster_statistics(
        self,
        output_path: Path,
        format: str = "csv"
    ) -> None
```

### Convenience Functions

```python
def enrich_with_clusters(
    points: np.ndarray,
    labels: np.ndarray,
    features: Dict[str, np.ndarray],
    ground_truth_features: Dict[str, gpd.GeoDataFrame],
    config: Optional[ClusterEnrichmentConfig] = None
) -> Dict[str, np.ndarray]
    """Quick enrichment with default configuration."""
```

---

## Zone-Specific Configuration

### Dense Urban Areas

```yaml
building_fusion:
  max_translation: 3.0 # Frequent GPS offsets
  adaptive_buffer_range: [0.3, 0.8] # Moderate buffer
  merge_nearby_buildings: true
  min_quality_score: 0.5
```

### Rural/Dispersed Areas

```yaml
building_fusion:
  max_translation: 5.0 # More flexibility
  adaptive_buffer_range: [0.5, 1.2] # Larger buffer
  merge_nearby_buildings: false
  min_quality_score: 0.6 # Stricter
```

### Historical Districts

```yaml
source_priority:
  - "osm" # Often more up-to-date
  - "bd_topo"
  - "cadastre"
min_quality_score: 0.4 # More permissive
```

---

## Related Documentation

- **[Adaptive Classification](./adaptive-classification.md)** - Feature-driven classification system
- **[RGE ALTI Integration](../guides/rge-alti-integration.md)** - DTM integration for height computation
- **[Plane Detection Guide](../guides/plane-detection.md)** - Roof and facade plane detection

---

## Summary

### Key Capabilities Delivered

✅ **Multi-source fusion** - BD TOPO®, Cadastre, OSM with quality scoring  
✅ **Intelligent selection** - Best, weighted merge, consensus strategies  
✅ **Adaptive adjustment** - Translation, scaling, rotation, buffering  
✅ **Quality metrics** - Coverage (40%), geometry (30%), completeness (30%)  
✅ **Conflict resolution** - Merge nearby, remove duplicates  
✅ **Cluster enrichment** - 18 metadata fields per point  
✅ **Hierarchical relationships** - Buildings linked to parcels  
✅ **Facade detection** - Individual wall identification

### Performance Impact

- **Processing time:** 8-12 minutes per km² tile
- **Memory overhead:** ~240 MB additional
- **Accuracy improvements:** +15-25% coverage, +10-20% wall capture
- **Quality scoring:** Automatic polygon evaluation (0.0-1.0)

### Production Readiness

The building analysis and fusion system is **production ready** and has been tested on:

- Dense urban areas (multi-story buildings, complex geometry)
- Suburban zones (mixed residential, commercial)
- Rural areas (isolated structures, agricultural land)
- Historical districts (OSM priority, updated data)

**Ready to use** with comprehensive configuration options! 🚀
