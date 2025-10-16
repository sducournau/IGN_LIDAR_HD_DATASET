# Data Sources Configuration

**Directory:** `ign_lidar/configs/data_sources/`  
**Date:** October 16, 2025  
**Purpose:** Configuration files for IGN multi-source data integration

---

## Overview

This directory contains configuration files for integrating multiple French national geographic databases into the LiDAR processing pipeline:

- **BD TOPO® V3** - Infrastructure (buildings, roads, railways, water, etc.)
- **BD Forêt® V2** - Forest types and species
- **RPG 2024** - Agricultural parcels and crop types
- **BD PARCELLAIRE** - Cadastral parcels

---

## Available Configurations

### 1. `default.yaml` (Recommended)

**Use Case:** General purpose with core features

**Enabled Features:**

- ✅ BD TOPO® core features (buildings, roads, railways, water, vegetation, bridges, parking, sports, cemeteries, power_lines)
- ❌ BD Forêt® (disabled by default)
- ❌ RPG (disabled by default)
- ❌ Cadastre (disabled by default)

**When to use:** Standard processing with infrastructure ground truth

---

### 2. `asprs_full.yaml`

**Use Case:** Complete ASPRS classification with all infrastructure codes

**Enabled Features:**

- ✅ BD TOPO® - ALL features
- ✅ BD Forêt® - Forest classification
- ✅ RPG - Agricultural classification
- ❌ Cadastre (optional)

**When to use:**

- Complete topographic mapping
- Urban planning
- Infrastructure inventory
- Need all ASPRS classification codes (11, 40, 41, 42, 43, etc.)

**Output Classes:**

- ASPRS 6: Buildings
- ASPRS 9: Water
- ASPRS 10: Railways
- ASPRS 11: Roads
- ASPRS 17: Bridges
- ASPRS 40: Parking
- ASPRS 41: Sports facilities
- ASPRS 42: Cemeteries
- ASPRS 43: Power lines

---

### 3. `lod2_buildings.yaml`

**Use Case:** Building-focused classification for LOD2 reconstruction

**Enabled Features:**

- ✅ BD TOPO® core (buildings, roads, railways, water, vegetation, bridges, parking)
- ❌ Extended features (sports, cemeteries, power_lines) - not essential
- ❌ BD Forêt® (not relevant)
- ❌ RPG (not relevant)
- ✅ Cadastre - for building-parcel association

**When to use:**

- Building geometry reconstruction
- Wall/roof detection
- BIM generation
- 3D city models

**Focus:** Wall and roof classification, transport as ground class

---

### 4. `lod3_architecture.yaml`

**Use Case:** Detailed architectural classification for LOD3 components

**Enabled Features:**

- ✅ BD TOPO® core (buildings, roads, railways, water, vegetation, bridges, parking)
- ❌ Extended features (sports, cemeteries, power_lines) - not essential
- ❌ BD Forêt® (not relevant)
- ❌ RPG (not relevant)
- ✅ Cadastre - for building-parcel linking

**When to use:**

- Detailed building models
- Window/door detection
- Architectural analysis
- Heritage documentation

**Focus:** Architectural component detection (windows, doors, balconies, chimneys, etc.)

---

### 5. `disabled.yaml`

**Use Case:** Pure geometric classification without ground truth

**Enabled Features:**

- ❌ All data sources disabled

**When to use:**

- Testing geometric features only
- No internet connection
- No need for ground truth classification

---

## Usage

### In Your Configuration File

```yaml
defaults:
  - data_sources: default # or asprs_full, lod2_buildings, lod3_architecture, disabled
  - _self_

# Override specific features if needed
data_sources:
  bd_topo:
    features:
      power_lines: false # Disable if not needed
```

### With CLI

```bash
# Use default data sources
ign-lidar-hd process --config-file myconfig.yaml

# Override data sources
ign-lidar-hd process --config-file myconfig.yaml \
    data_sources=asprs_full

# Override specific feature
ign-lidar-hd process --config-file myconfig.yaml \
    data_sources.bd_topo.features.power_lines=false
```

---

## Configuration Parameters

### BD TOPO® Parameters

```yaml
bd_topo:
  enabled: true
  cache_enabled: true

  features:
    buildings: true # ASPRS 6
    roads: true # ASPRS 11
    railways: true # ASPRS 10
    water: true # ASPRS 9
    vegetation: true # ASPRS 3-5
    bridges: true # ASPRS 17
    parking: true # ASPRS 40
    sports: true # ASPRS 41
    cemeteries: true # ASPRS 42
    power_lines: true # ASPRS 43

  parameters:
    # Buffer widths (meters)
    road_width_fallback: 4.0 # Default road width
    road_buffer_tolerance: 0.5 # Additional buffer for roads
    railway_width_fallback: 3.5 # Default railway width
    railway_buffer_tolerance: 0.6 # Additional buffer for railways
    power_line_buffer: 2.0 # Power line corridor width

    # Height filtering (meters) - exclude bridges/overpasses
    road_height_max: 1.5 # Maximum height for ground-level roads
    road_height_min: -0.3 # Minimum height (depressions)
    rail_height_max: 1.2 # Maximum height for ground-level railways
    rail_height_min: -0.2 # Minimum height

    # Geometric filtering
    road_planarity_min: 0.6 # Minimum planarity for road surfaces
    rail_planarity_min: 0.5 # Minimum planarity for rail surfaces

    # Intensity filtering (0-1 normalized)
    enable_intensity_filter: true
    road_intensity_min: 0.15 # Minimum intensity (asphalt/concrete)
    road_intensity_max: 0.7 # Maximum intensity
    rail_intensity_min: 0.1 # Minimum intensity
    rail_intensity_max: 0.8 # Maximum intensity
```

---

## Feature Selection Strategy

### By Use Case

| Use Case                           | Config            | Buildings | Roads | Extended\* | Forest | Agriculture |
| ---------------------------------- | ----------------- | --------- | ----- | ---------- | ------ | ----------- |
| **Topographic Mapping**            | asprs_full        | ✅        | ✅    | ✅         | ✅     | ✅          |
| **Building Reconstruction (LOD2)** | lod2_buildings    | ✅        | ✅    | ❌         | ❌     | ❌          |
| **Architectural Detail (LOD3)**    | lod3_architecture | ✅        | ✅    | ❌         | ❌     | ❌          |
| **General Processing**             | default           | ✅        | ✅    | ✅         | ❌     | ❌          |
| **Geometric Only**                 | disabled          | ❌        | ❌    | ❌         | ❌     | ❌          |

\*Extended = sports, cemeteries, power_lines

---

## Performance Considerations

### Caching

- All WFS queries are cached by default
- Cache location: `cache_dir` from root config or per-source override
- First run: Slower (fetches from WFS)
- Subsequent runs: Fast (loads from cache)

### Processing Time Impact

| Configuration  | WFS Queries | Processing Time | Use When                |
| -------------- | ----------- | --------------- | ----------------------- |
| disabled       | 0           | Fastest         | Testing, no GT needed   |
| lod2_buildings | ~5-7        | Fast            | Building focus          |
| default        | ~10         | Medium          | General use             |
| asprs_full     | ~15+        | Slower          | Complete classification |

### Optimization Tips

1. **Enable only needed features** - Disable unused extended features
2. **Use caching** - Keep `cache_enabled: true`
3. **Choose appropriate config** - Don't use asprs_full if you only need buildings

---

## Examples

### Example 1: ASPRS Preprocessing

```yaml
defaults:
  - data_sources: asprs_full
  - _self_

cache_dir: /path/to/cache
```

### Example 2: LOD2 Buildings Only

```yaml
defaults:
  - data_sources: lod2_buildings
  - _self_

# Optional: override cadastre
data_sources:
  cadastre:
    enabled: false # Disable if not needed
```

### Example 3: Custom Configuration

```yaml
defaults:
  - data_sources: default
  - _self_

# Enable forest classification
data_sources:
  bd_foret:
    enabled: true

  # Disable extended features
  bd_topo:
    features:
      sports: false
      cemeteries: false
      power_lines: false
```

---

## Troubleshooting

### Issue: No points classified to roads/cemeteries/etc.

**Solution:** Check that:

1. `bd_topo.enabled: true`
2. Specific feature is enabled (e.g., `bd_topo.features.roads: true`)
3. Parameters are correctly configured (buffer widths, etc.)

### Issue: Slow processing

**Solution:**

1. Use appropriate config for your use case (don't use asprs_full for building focus)
2. Disable unused features
3. Check cache is enabled and writable

### Issue: Missing WFS data

**Solution:**

1. Check internet connection
2. Verify bounding box is within France
3. Check WFS service is available: https://data.geopf.fr/wfs

---

## See Also

- **Main Config README:** `ign_lidar/configs/README.md`
- **Processor Config:** `ign_lidar/configs/processor/`
- **Ground Truth Config:** `ign_lidar/configs/ground_truth/`
- **WFS Fetcher:** `ign_lidar/io/wfs_ground_truth.py`
- **Data Fetcher:** `ign_lidar/io/data_fetcher.py`

---

## Version History

- **v2.5.1 (2025-10-16):** Initial data_sources configuration directory
  - Added default.yaml
  - Added asprs_full.yaml
  - Added lod2_buildings.yaml
  - Added lod3_architecture.yaml
  - Added disabled.yaml
