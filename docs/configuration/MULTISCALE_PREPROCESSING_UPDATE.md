# Multiscale & Preprocessing Configuration Updates

**Date**: October 15, 2025  
**Update Type**: Data Sources Integration & Latest Dataset Versions  
**Status**: ✅ Complete

---

## Overview

Updated all multiscale preprocessing configuration files to include complete `data_sources` sections with the latest IGN dataset versions. Each LOD level now has appropriate dataset enablement based on its specific focus.

---

## Files Updated

### 1. ASPRS Preprocessing (`config_asprs_preprocessing.yaml`)

**Purpose**: General-purpose ASPRS classification (ground, vegetation, buildings, water, roads, etc.)

**Data Sources Configuration**:

```yaml
data_sources:
  bd_topo:
    enabled: true
    features:
      buildings: true
      roads: true
      railways: true
      water: true
      vegetation: true
      bridges: true
      parking: true
      sports: true
      cemeteries: false
      power_lines: false

  bd_foret:
    enabled: true
    year: V2 (latest)

  rpg:
    enabled: true
    year: 2024

  cadastre:
    enabled: false # Optional for ASPRS
```

**Rationale**:

- All infrastructure features needed for complete ASPRS classification
- Forest types useful for vegetation refinement
- Agriculture data useful for complete land cover
- Cadastre optional (not critical for ASPRS)

---

### 2. LOD2 Preprocessing (`config_lod2_preprocessing.yaml`)

**Purpose**: Building-focused classification (walls, roofs, building components)

**Data Sources Configuration**:

```yaml
data_sources:
  bd_topo:
    enabled: true
    features:
      buildings: true # ⭐ PRIMARY FOCUS
      roads: true # Context
      railways: true # Context
      water: true # Context
      vegetation: true # Context
      bridges: true # Context
      parking: true # Context
      sports: false # Less relevant
      cemeteries: false # Less relevant
      power_lines: false # Less relevant

  bd_foret:
    enabled: false # Less relevant for building detection

  rpg:
    enabled: false # Less relevant for building detection
    year: 2024

  cadastre:
    enabled: true # ⭐ USEFUL for building-parcel association
    group_by_parcel: true
```

**Rationale**:

- Buildings are primary focus for LOD2
- Infrastructure provides context
- Forest/agriculture less relevant for building detection
- Cadastre very useful for associating buildings with parcels

---

### 3. LOD3 Preprocessing (`config_lod3_preprocessing.yaml`)

**Purpose**: Detailed architectural classification (windows, doors, balconies, architectural details)

**Data Sources Configuration**:

```yaml
data_sources:
  bd_topo:
    enabled: true
    features:
      buildings: true # ⭐⭐ MAXIMUM FOCUS
      roads: true # Minimal context
      railways: true # Minimal context
      water: true # Minimal context
      vegetation: true # Minimal context
      bridges: true # Context
      parking: true # Context
      sports: false # Not relevant
      cemeteries: false # Not relevant
      power_lines: false # Not relevant

  bd_foret:
    enabled: false # Not relevant for architectural details

  rpg:
    enabled: false # Not relevant for architectural details
    year: 2024

  cadastre:
    enabled: true # ⭐⭐ ESSENTIAL for building analysis
    group_by_parcel: true
    compute_statistics: true
```

**Rationale**:

- Buildings are almost exclusive focus for LOD3
- Only basic infrastructure context needed
- Forest/agriculture irrelevant for architectural details
- Cadastre essential for detailed building-parcel analysis

---

## Key Changes Summary

### Added to All Files

1. **Complete `data_sources` section** with all four IGN datasets:

   - BD TOPO® V3
   - BD Forêt® V2
   - RPG 2024
   - Cadastre (BD PARCELLAIRE)

2. **Proper cache directory configuration** for each source

3. **Export attributes settings** where relevant

### Dataset Version Updates

| Dataset       | Version | Access                              | Configuration             |
| ------------- | ------- | ----------------------------------- | ------------------------- |
| **BD TOPO®**  | V3      | WFS `BDTOPO_V3:*`                   | Feature selection per LOD |
| **BD Forêt®** | V2      | WFS `BDFORET_V2:formation_vegetale` | Enabled for ASPRS only    |
| **RPG**       | 2024    | WFS `RPG.2024:parcelles_graphiques` | Enabled for ASPRS only    |
| **Cadastre**  | Current | API Géoportail                      | Enabled for LOD2/LOD3     |

---

## LOD-Specific Configurations

### ASPRS (General Classification)

**Focus**: Complete land cover classification

**Enabled Sources**:

- ✅ BD TOPO V3 (all features)
- ✅ BD Forêt V2 (forest types)
- ✅ RPG 2024 (agriculture)
- ❌ Cadastre (optional)

**Use Cases**:

- General terrain classification
- Vegetation/forest mapping
- Agricultural land classification
- Infrastructure mapping
- Urban/rural classification

---

### LOD2 (Building Structures)

**Focus**: Building geometry (walls, roofs, basic components)

**Enabled Sources**:

- ✅ BD TOPO V3 (buildings + context)
- ❌ BD Forêt V2 (not relevant)
- ❌ RPG 2024 (not relevant)
- ✅ Cadastre (building-parcel linking)

**Use Cases**:

- Building footprint detection
- Wall/roof classification
- Roof type detection (flat, gable, hip)
- Building height estimation
- Building-parcel association

---

### LOD3 (Architectural Details)

**Focus**: Fine architectural details (windows, doors, balconies, etc.)

**Enabled Sources**:

- ✅ BD TOPO V3 (buildings primary)
- ❌ BD Forêt V2 (not relevant)
- ❌ RPG 2024 (not relevant)
- ✅ Cadastre (essential for analysis)

**Use Cases**:

- Window detection
- Door detection
- Balcony identification
- Chimney detection
- Dormer detection
- Architectural style classification
- Detailed building facade analysis

---

## Parameters Added

### All Preprocessing Configs

```yaml
data_sources:
  bd_topo:
    parameters:
      road_width_fallback: 4.0 # Default road width (m)
      railway_width_fallback: 3.5 # Default railway width (m)
      power_line_buffer: 2.0 # Buffer for power lines (m)
```

### Forest-Enabled Configs (ASPRS)

```yaml
bd_foret:
  label_forest_type: true # Coniferous, deciduous, mixed
  label_primary_species: true # Oak, Pine, Beech, etc.
  estimate_height: true # Height by forest type
  export_attributes: true # Export to LAZ
```

### Agriculture-Enabled Configs (ASPRS)

```yaml
rpg:
  year: 2024
  apply_asprs_classification: true # ASPRS code 44
  label_crop_code: true # BLE, MAI, COL, etc.
  label_crop_category: true # Cereals, oleaginous, etc.
  export_attributes: true # Export to LAZ
```

### Cadastre-Enabled Configs (LOD2, LOD3)

```yaml
cadastre:
  group_by_parcel: true # Group points by parcel
  label_parcel_id: true # Add parcel ID to points
  compute_statistics: true # LOD3 only
```

---

## Cache Directory Structure

```
cache/
├── ground_truth/          # BD TOPO® V3 data
│   ├── buildings/
│   ├── roads/
│   ├── railways/
│   └── ...
├── bd_foret/              # BD Forêt® V2 data
│   └── formations/
├── rpg/                   # RPG agricultural data
│   ├── 2024/              # Year-specific
│   │   └── parcelles/
│   └── 2023/              # Previous years
└── cadastre/              # Cadastral parcels
    └── parcelles/
```

---

## Impact on Pipeline

### Processing Flow

1. **Tile Loading** → Input LAZ/LAS files
2. **Feature Computation** → Geometric & spectral features
3. **Data Fetching** → Query enabled data sources (WFS/API)
4. **Classification** → Apply ground truth & rules
5. **Enrichment** → Add attributes from external sources
6. **Output** → Enriched LAZ with all features

### Performance Considerations

**ASPRS** (all sources enabled):

- More WFS queries required
- Longer processing time
- Richer output data
- Recommended: Enable caching

**LOD2/LOD3** (selective sources):

- Fewer WFS queries
- Faster processing
- Focused on buildings
- Cadastre adds minimal overhead

---

## Testing

### Test ASPRS Preprocessing

```bash
cd configs/multiscale
ign-lidar-hd process --config config_asprs_preprocessing.yaml \
  input_dir=/mnt/d/ign/selected_tiles/asprs/tiles \
  output_dir=/mnt/d/ign/test_output/asprs
```

**Expected Output**:

- Enriched LAZ with ASPRS classification
- Forest types in extra dimensions
- Crop types in extra dimensions
- Complete infrastructure classification

### Test LOD2 Preprocessing

```bash
ign-lidar-hd process --config config_lod2_preprocessing.yaml \
  input_dir=/mnt/d/ign/selected_tiles/lod2/tiles \
  output_dir=/mnt/d/ign/test_output/lod2
```

**Expected Output**:

- Enriched LAZ focused on buildings
- Wall/roof classification
- Parcel IDs for buildings
- Building geometry features

### Test LOD3 Preprocessing

```bash
ign-lidar-hd process --config config_lod3_preprocessing.yaml \
  input_dir=/mnt/d/ign/selected_tiles/lod3/tiles \
  output_dir=/mnt/d/ign/test_output/lod3
```

**Expected Output**:

- Enriched LAZ with architectural details
- Window/door detection features
- Detailed parcel statistics
- Maximum geometric feature density

---

## Validation Checklist

- [x] All configs have `data_sources` section
- [x] RPG year set to 2024 where enabled
- [x] BD TOPO V3 features appropriate for each LOD
- [x] BD Forêt V2 enabled only for ASPRS
- [x] RPG 2024 enabled only for ASPRS
- [x] Cadastre enabled for LOD2 and LOD3
- [x] Cache directories configured
- [x] Export attributes set correctly
- [x] Parameters include fallback values
- [x] Headers updated with version info

---

## Migration Notes

### For Existing Pipelines

**No breaking changes**:

- Existing configs without `data_sources` section will use defaults
- Ground truth section still works independently
- Cache structure unchanged (separate by source/year)

**To enable new features**:

- Add `data_sources` section from updated configs
- Set `enabled: true` for desired sources
- Configure cache directories
- Run preprocessing

---

## Future Enhancements

### Planned for Q4 2025

1. **Dynamic source selection**:

   - Auto-enable sources based on tile content
   - Adaptive feature fetching

2. **Source priority configuration**:

   - Define which source wins in overlaps
   - Custom priority orders per LOD

3. **Advanced caching**:
   - Cross-tile caching optimization
   - Predictive prefetching

---

## Related Documentation

- `CONFIG_DATASET_VERSIONS_UPDATE.md` - Main version update doc
- `BD_TOPO_RPG_INTEGRATION.md` - Integration details
- `RAILWAYS_AND_FOREST_INTEGRATION.md` - Forest & railway features
- `configs/README.md` - Configuration guide

---

**Update completed**: October 15, 2025 ✅

All multiscale preprocessing configs now have complete data source integration with the latest IGN dataset versions, properly configured for each LOD level's specific focus.
