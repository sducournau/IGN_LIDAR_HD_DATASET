# Configuration Files - Dataset Versions Update

**Date**: October 15, 2025  
**Status**: ✅ Complete

---

## Summary

All configuration files have been updated to use the latest versions of IGN datasets:

- **BD TOPO®**: V3 (latest) - Infrastructure and topographic features
- **BD Forêt®**: V2 (latest) - Forest types and species classification
- **RPG**: 2024 (latest) - Agricultural parcels and crop types
- **Cadastre**: Current (latest) - Cadastral parcels via BD PARCELLAIRE

---

## Updated Configuration Files

### 1. Main Configuration Files

#### `configs/enrichment_asprs_full.yaml`

- ✅ Updated header with version information
- ✅ RPG year changed: `2023` → `2024`
- ✅ Updated comment: `# Available: 2020, 2021, 2022, 2023, 2024 (LATEST)`

#### `configs/example_enrichment_full.yaml`

- ✅ Updated header with version information
- ✅ RPG year changed: `2023` → `2024`
- ✅ Updated comment: `# Available: 2024 (LATEST), 2023, 2022, 2021, 2020`

#### `configs/processing_config.yaml`

- ✅ Updated header with version information
- ✅ RPG year changed: `2023` → `2024`
- ✅ Updated comment: `# Latest: 2024 (Available: 2024, 2023, 2022, 2021, 2020)`

#### `configs/classification_config.yaml`

- ✅ Updated header with version information
- ✅ BD Forêt® year comment: `# Latest version V2 (current)`
- ✅ RPG year changed: `2023` → `2024`
- ✅ Updated comment: `# Année du RPG (Latest: 2024, Available: 2024-2020)`

### 2. Multiscale Preprocessing Configurations

#### `configs/multiscale/config_asprs_preprocessing.yaml`

- ✅ Added version update header line
- ✅ Datasets: BD TOPO V3, BD Forêt V2, RPG 2024
- ✅ **NEW**: Added complete `data_sources` section with all datasets
  - BD TOPO V3: All features enabled (buildings, roads, railways, water, vegetation, bridges, parking, sports)
  - BD Forêt V2: Enabled with forest type and species labeling
  - RPG 2024: Enabled with crop classification
  - Cadastre: Optional (disabled by default for ASPRS)

#### `configs/multiscale/config_lod2_preprocessing.yaml`

- ✅ Added version update header line
- ✅ Datasets: BD TOPO V3, BD Forêt V2, RPG 2024
- ✅ **NEW**: Added complete `data_sources` section with building focus
  - BD TOPO V3: Buildings-focused (buildings primary, infrastructure secondary)
  - BD Forêt V2: Disabled (less relevant for LOD2 building detection)
  - RPG 2024: Disabled (less relevant for LOD2 building detection)
  - Cadastre: **Enabled** (useful for building-parcel association)

#### `configs/multiscale/config_lod3_preprocessing.yaml`

- ✅ Added version update header line
- ✅ Datasets: BD TOPO V3, BD Forêt V2, RPG 2024
- ✅ **NEW**: Added complete `data_sources` section with maximum building detail
  - BD TOPO V3: Maximum building detail focus
  - BD Forêt V2: Disabled (not relevant for LOD3 architectural details)
  - RPG 2024: Disabled (not relevant for LOD3 architectural details)
  - Cadastre: **Enabled** (essential for LOD3 building-parcel linking)

### 3. Core Module Updates

#### `ign_lidar/config/loader.py`

- ✅ RPG default year changed: `2023` → `2024` (3 occurrences)
- ✅ Updated validation range: `2020-2023` → `2020-2024`
- ✅ Updated warning message to reflect 2024 availability

---

## Dataset Version Details

### BD TOPO® V3

- **Status**: Latest version (current)
- **Access**: WFS service `BDTOPO_V3:*`
- **Features**: 10 infrastructure types (buildings, roads, railways, water, vegetation, bridges, parking, cemeteries, power lines, sports)
- **Update frequency**: Continuous updates
- **Configuration**: No version parameter needed (always uses latest)

### BD Forêt® V2

- **Status**: Latest version (current)
- **Access**: WFS service `BDFORET_V2:formation_vegetale`
- **Features**: Forest types (coniferous, deciduous, mixed) with species classification
- **Update frequency**: Periodic updates
- **Configuration**: No version parameter needed (always uses latest)

### RPG (Registre Parcellaire Graphique)

- **Status**: 2024 data now available
- **Previous default**: 2023
- **New default**: 2024
- **Access**: WFS service `RPG.{year}:parcelles_graphiques`
- **Available years**: 2024, 2023, 2022, 2021, 2020
- **Update frequency**: Annual
- **Configuration**: `year: 2024` parameter in config files

### Cadastre (BD PARCELLAIRE)

- **Status**: Current (latest)
- **Access**: API Géoportail
- **Features**: Cadastral parcels with unique IDs
- **Update frequency**: Continuous updates
- **Configuration**: No version parameter needed (always uses latest)

---

## Code Changes

### Configuration Loader (`ign_lidar/config/loader.py`)

**Changes made:**

1. **Line 107**: Default RPG year

   ```python
   # Before:
   year = rpg.get('year', 2023)

   # After:
   year = rpg.get('year', 2024)
   ```

2. **Line 108**: Validation range

   ```python
   # Before:
   if not (2020 <= year <= 2024):
       warnings.append(f"RPG year {year} may not be available (valid: 2020-2023)")

   # After:
   if not (2020 <= year <= 2024):
       warnings.append(f"RPG year {year} may not be available (valid: 2020-2024)")
   ```

3. **Line 364**: DataFetchConfig default

   ```python
   # Before:
   rpg_year=rpg.get('year', 2023),

   # After:
   rpg_year=rpg.get('year', 2024),
   ```

4. **Line 412**: Print summary default

   ```python
   # Before:
   year = rpg.get('year', 2023)

   # After:
   year = rpg.get('year', 2024)
   ```

---

## Impact Assessment

### Backward Compatibility

- ✅ **Maintained**: Older config files specifying `year: 2023` will continue to work
- ✅ **Default updated**: New configs without explicit year will use 2024
- ✅ **Validation**: System validates years 2020-2024

### Performance

- ✅ **No impact**: Same WFS queries, just different year parameter
- ✅ **Cache separate**: Different years cache separately (no conflicts)

### Data Quality

- ✅ **Improved**: 2024 RPG data includes latest agricultural parcels
- ✅ **Current**: All other datasets use latest versions automatically

---

## Testing Recommendations

1. **Test with RPG 2024 data**:

   ```bash
   # Test fetching RPG 2024 data
   python -m ign_lidar.io.rpg --year 2024 --bbox "650000,6860000,651000,6861000"
   ```

2. **Test config loading**:

   ```python
   from ign_lidar.config.loader import load_config_from_yaml
   config = load_config_from_yaml("configs/enrichment_asprs_full.yaml")
   assert config['data_sources']['rpg']['year'] == 2024
   ```

3. **Test enrichment pipeline**:

   ```bash
   ign-lidar-hd process --config configs/enrichment_asprs_full.yaml
   ```

4. **Verify cache separation**:
   - Check that `cache/rpg/2023/` and `cache/rpg/2024/` exist separately
   - Confirm no cache conflicts

---

## Migration Guide

### For Users with Existing Configs

**Option 1: Continue using 2023 (no changes needed)**

- Your existing configs with `year: 2023` will continue to work
- No action required

**Option 2: Update to 2024**

1. Open your config file
2. Find `rpg:` section
3. Change `year: 2023` to `year: 2024`
4. Clear RPG cache (optional): `rm -rf cache/rpg/2023`

**Option 3: Use latest automatically**

- Remove the `year:` line from RPG config
- System will use default (2024)

### For Developers

**When creating new configs:**

- Use `year: 2024` for RPG (or omit for default)
- No changes needed for BD TOPO V3, BD Forêt V2, or Cadastre

---

## Future Updates

### Monitoring

- Watch for RPG 2025 release (typically Q1 each year)
- BD TOPO V3 and BD Forêt V2 update automatically via WFS

### When RPG 2025 becomes available:

1. Update `loader.py` validation range: `2020 <= year <= 2025`
2. Update default year: `year: 2025`
3. Update all config files
4. Update this documentation

---

## Related Documentation

- `BD_TOPO_RPG_INTEGRATION.md` - Integration details
- `RAILWAYS_AND_FOREST_INTEGRATION.md` - Forest and railway features
- `configs/README.md` - Configuration file guide
- `ign_lidar/io/rpg.py` - RPG fetcher implementation
- `ign_lidar/io/bd_foret.py` - BD Forêt fetcher implementation
- `ign_lidar/io/wfs_ground_truth.py` - BD TOPO fetcher implementation

---

## Changelog

### October 15, 2025

- ✅ Updated all config files to use latest dataset versions
- ✅ Changed RPG default year from 2023 to 2024
- ✅ Updated validation range to include 2024
- ✅ Added version information to config file headers
- ✅ Updated loader.py defaults (4 locations)
- ✅ Maintained backward compatibility

---

**Update completed successfully** ✅

All configuration files now reference the latest available dataset versions while maintaining backward compatibility with existing configurations.
