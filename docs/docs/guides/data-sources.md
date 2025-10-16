---
sidebar_position: 8
title: Data Source Enrichment
---

# Data Source Enrichment

Enrich LiDAR point clouds with authoritative geographic data from IGN and government databases.

---

## 🎯 Overview

The IGN LiDAR HD package can enrich point clouds with ground truth data from multiple authoritative sources:

- **BD TOPO®** - IGN topographic database (buildings, roads, water, etc.)
- **BD Forêt®** - IGN forest database (tree species, forest types)
- **RPG** - Agricultural parcel register (crop types, land use)
- **Cadastre** - Land parcel boundaries and ownership

This enrichment significantly improves classification accuracy, especially for ASPRS classification.

---

## 🗺️ BD TOPO® (IGN Topographic Database)

France's reference topographic database, maintained by IGN (Institut National de l'Information Géographique et Forestière).

### Available Features

| Feature     | ASPRS Code | Description                        |
| ----------- | ---------- | ---------------------------------- |
| Buildings   | 6          | Building footprints and heights    |
| Roads       | 11         | Road networks with classifications |
| Railways    | 10         | Railway tracks and stations        |
| Water       | 9          | Rivers, lakes, water bodies        |
| Vegetation  | 5          | Parks, forests, green spaces       |
| Bridges     | 17         | Bridge structures                  |
| Parking     | 40         | Parking lots and areas             |
| Sports      | 41         | Sports facilities                  |
| Cemeteries  | 42         | Cemetery boundaries                |
| Power Lines | 43         | Electrical transmission lines      |

### Configuration

```yaml
data_sources:
  # Enable BD TOPO
  bd_topo_enabled: true

  # Select specific features
  bd_topo_buildings: true # ASPRS code 6
  bd_topo_roads: true # ASPRS code 11
  bd_topo_water: true # ASPRS code 9
  bd_topo_vegetation: true # ASPRS code 5

  # Cache settings
  bd_topo_cache_dir: "cache/bd_topo"
```

### Usage Example

```bash
ign-lidar-hd process \
    --config-file configs/default_v3.yaml \
    data_sources.bd_topo_enabled=true \
    data_sources.bd_topo_buildings=true \
    data_sources.bd_topo_roads=true
```

### Data Quality

- **Accuracy:** ± 1-5 meters planimetric, ± 1-3 meters altimetric
- **Update frequency:** Annual to quarterly (varies by region)
- **Coverage:** Complete coverage of metropolitan France
- **Source:** Aerial photography, satellite imagery, field surveys
- **Version:** BD TOPO® V3 (latest as of October 2025)

### What BD TOPO Adds

**For Buildings:**

- Precise footprint geometry
- Building height information
- Building type classification
- Construction date (when available)

**For Roads:**

- Road centerlines and widths
- Road classification (highway, street, path)
- Surface type information
- Traffic direction

**For Water:**

- Water body boundaries
- Flow direction for rivers
- Water body type (lake, river, canal)
- Seasonal vs permanent

---

## 🌲 BD Forêt® (IGN Forest Database)

Detailed forest inventory database with tree species and forest characteristics.

### Available Information

- **Tree species:** Oak, beech, pine, spruce, etc.
- **Forest type:** Deciduous, coniferous, mixed
- **Density:** Open, medium, dense
- **Age class:** Young, mature, old-growth
- **Management:** Production, conservation, recreation

### Configuration

```yaml
data_sources:
  bd_foret_enabled: true
  bd_foret_cache_dir: "cache/bd_foret"
```

### Usage Example

```bash
ign-lidar-hd process \
    --config-file configs/default_v3.yaml \
    data_sources.bd_foret_enabled=true
```

### What BD Forêt Adds

- **Vegetation classification:** Distinguishes tree species
- **Forest structure:** Height classes, density
- **Ecological zones:** Protected areas, biodiversity hotspots
- **Forest management:** Production vs conservation areas

### Data Quality

- **Accuracy:** Forest boundary ± 10 meters, species identification ~90%
- **Update frequency:** Every 5-10 years
- **Coverage:** Forest areas in metropolitan France
- **Source:** Aerial photography, field surveys, satellite imagery
- **Version:** BD Forêt® V2 (latest as of October 2025)

---

## 🌾 RPG (Registre Parcellaire Graphique)

Agricultural parcel register maintained by the French Ministry of Agriculture.

### Available Information

- **Crop types:** 28+ crop categories
- **Land use:** Agricultural, grassland, fallow
- **Parcel boundaries:** Field-level precision
- **Cultivation practices:** Organic, conventional
- **CAP subsidies:** EU Common Agricultural Policy data

### Configuration

```yaml
data_sources:
  rpg_enabled: true
  rpg_year: 2024 # Use 2024 data
  rpg_cache_dir: "cache/rpg"
```

### Usage Example

```bash
ign-lidar-hd process \
    --config-file configs/default_v3.yaml \
    data_sources.rpg_enabled=true \
    data_sources.rpg_year=2024
```

### Crop Categories (Examples)

- Cereals (wheat, barley, corn)
- Oilseeds (rapeseed, sunflower)
- Protein crops (soybeans, peas)
- Root crops (potatoes, beets)
- Forage crops (alfalfa, grass)
- Permanent crops (vineyards, orchards)

### What RPG Adds

- **Agricultural land classification:** Precise crop type identification
- **Seasonal context:** Current year's cultivation
- **Field boundaries:** Parcel-level geometry
- **Land use patterns:** Crop rotation analysis

### Data Quality

- **Accuracy:** Parcel boundary ± 5 meters, crop type ~95%
- **Update frequency:** Annual
- **Coverage:** Agricultural areas in metropolitan France
- **Source:** Farmer declarations, satellite verification
- **Available years:** 2020-2024

---

## 🏘️ Cadastre (Land Parcel Registry)

Official land parcel boundaries and ownership information.

### Available Information

- **Parcel boundaries:** Legal property limits
- **Parcel identifiers:** Unique cadastral codes
- **Land use type:** Urban, agricultural, forest
- **Building footprints:** Building positions within parcels
- **Section codes:** Cadastral section identifiers

### Configuration

```yaml
data_sources:
  cadastre_enabled: true
  cadastre_cache_dir: "cache/cadastre"
```

### Usage Example

```bash
ign-lidar-hd process \
    --config-file configs/default_v3.yaml \
    data_sources.cadastre_enabled=true
```

### What Cadastre Adds

- **Property boundaries:** Precise parcel limits
- **Building context:** Buildings within land parcels
- **Urban/rural distinction:** Land use classification
- **Parcel grouping:** Group points by property

### Data Quality

- **Accuracy:** Parcel boundary ± 0.5-2 meters (urban areas)
- **Update frequency:** Continuous updates
- **Coverage:** Complete coverage of France
- **Source:** Official land registry, surveyor measurements
- **Maintained by:** Direction Générale des Finances Publiques (DGFiP)

---

## 🎯 Combining Data Sources

### Recommended Combinations

#### For ASPRS Classification (Best Accuracy)

```yaml
data_sources:
  bd_topo_enabled: true
  bd_topo_buildings: true
  bd_topo_roads: true
  bd_topo_water: true
  bd_topo_vegetation: true
  bd_foret_enabled: true
  rpg_enabled: true
  rpg_year: 2024
  cadastre_enabled: true
```

**Benefits:**

- Maximum classification accuracy
- Complete ground truth coverage
- Cross-validation between sources

#### For Urban Analysis

```yaml
data_sources:
  bd_topo_enabled: true
  bd_topo_buildings: true
  bd_topo_roads: true
  cadastre_enabled: true
```

**Benefits:**

- Building and infrastructure focus
- Property boundary context
- Road network integration

#### For Forest/Rural Analysis

```yaml
data_sources:
  bd_foret_enabled: true
  rpg_enabled: true
  rpg_year: 2024
```

**Benefits:**

- Vegetation and agriculture focus
- Tree species identification
- Crop type mapping

---

## 📊 Performance Impact

### Processing Time

| Data Sources       | Relative Time | Cache Benefit               |
| ------------------ | ------------- | --------------------------- |
| None               | 1.0x          | -                           |
| BD TOPO only       | 1.2x          | Yes (3x faster after cache) |
| BD TOPO + BD Forêt | 1.4x          | Yes (4x faster after cache) |
| All sources        | 1.8x          | Yes (5x faster after cache) |

### File Size Impact

Data source enrichment does **not** significantly increase file size:

- Point cloud size: No change (same points)
- Classification field: Already present in LAZ
- Metadata: Minimal increase (<1%)

### Caching System

All data sources use intelligent caching:

```yaml
# Cache configuration
data_sources:
  bd_topo_cache_dir: "cache/bd_topo"
  bd_foret_cache_dir: "cache/bd_foret"
  rpg_cache_dir: "cache/rpg"
  cadastre_cache_dir: "cache/cadastre"
```

**Cache benefits:**

- **First run:** Downloads and caches data
- **Subsequent runs:** Uses cached data (5x faster)
- **Automatic updates:** Checks for new data versions
- **Disk space:** ~100-500 MB per region

---

## 🔍 How It Works

### Data Fetching Process

1. **Query:** System identifies tile bounding box
2. **WFS Request:** Queries IGN/government Web Feature Service
3. **Filter:** Retrieves only relevant features
4. **Cache:** Stores data locally for reuse
5. **Intersect:** Identifies points within feature geometries
6. **Classify:** Assigns ASPRS codes based on feature type

### Spatial Operations

```python
# Pseudo-code for enrichment
for tile in tiles:
    # Fetch relevant features
    buildings = bd_topo.get_buildings(tile.bbox)
    roads = bd_topo.get_roads(tile.bbox)

    # Classify points
    for point in tile.points:
        if point.intersects(buildings):
            point.classification = 6  # Building
        elif point.intersects(roads):
            point.classification = 11  # Road
```

### Priority Rules

When a point intersects multiple features:

1. **Buildings** (code 6) - Highest priority
2. **Water** (code 9) - High priority
3. **Roads** (code 11) - Medium priority
4. **Vegetation** (code 5) - Lower priority
5. **Ground** (code 2) - Default

---

## 🚀 Quick Start Examples

### Example 1: Basic Enrichment

```bash
ign-lidar-hd process \
    input_dir=data/raw \
    output_dir=data/enriched \
    data_sources.bd_topo_enabled=true
```

### Example 2: Full ASPRS Classification

```bash
ign-lidar-hd process \
    --config-file configs/presets/asprs_classification.yaml \
    input_dir=data/raw \
    output_dir=data/asprs
```

### Example 3: Custom Feature Selection

```bash
ign-lidar-hd process \
    input_dir=data/raw \
    output_dir=data/custom \
    data_sources.bd_topo_enabled=true \
    data_sources.bd_topo_buildings=true \
    data_sources.bd_topo_roads=true \
    data_sources.bd_topo_water=false \
    data_sources.bd_foret_enabled=false
```

---

## 📚 See Also

- [Configuration System v3.0](./configuration-v3)
- [Feature Modes Guide](./feature-modes-guide)
- [ASPRS Classification Reference](../reference/classification-taxonomy)
- [Processing Modes](./processing-modes)

---

## ❓ FAQ

### Do I need all data sources?

No. Enable only what you need:

- **ASPRS classification:** BD TOPO + BD Forêt recommended
- **Building analysis:** BD TOPO buildings + Cadastre
- **Forest analysis:** BD Forêt only
- **Agriculture:** RPG only

### How much does caching help?

First run: Downloads data (slower)  
Subsequent runs: Uses cache (5x faster)

### Can I use my own data sources?

Yes! See the [Custom Data Sources](../api/custom-data-sources) guide.

### What if WFS is unavailable?

The system falls back to geometry-based classification using LiDAR features only.

### How accurate is the classification?

With data sources: 85-95% accuracy  
Without data sources: 70-80% accuracy

The improvement is significant, especially for infrastructure classes.
