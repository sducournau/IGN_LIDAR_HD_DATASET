# Quick Reference: Roads & Railways Enrichment Fix

## Problem

Roads and railways are not appearing in enriched LAZ files in `D:\ign\preprocessed\asprs\enriched_tiles\enriched`.

## Root Cause

Configuration files were not properly enabling roads and railways features with correct parameters.

## Solution

### 1. Updated Configuration Files

All configuration files have been updated to include:

#### Key Configuration Points:

```yaml
data_sources:
  bd_topo:
    enabled: true
    features:
      roads: true # ⚠️ MUST BE TRUE
      railways: true # ⚠️ MUST BE TRUE

    parameters:
      road_width_fallback: 4.0 # Default road width (m)
      railway_width_fallback: 3.5 # Default railway width (m)
```

### 2. Updated Files

1. **`configs/processing_config.yaml`** - Main processing configuration
2. **`configs/example_enrichment_full.yaml`** - Full enrichment example
3. **`configs/classification_config.yaml`** - Classification settings
4. **`configs/enrichment_asprs_full.yaml`** - NEW: Complete ASPRS enrichment config

### 3. How to Use

#### Option A: Use the new comprehensive config (RECOMMENDED)

```bash
python -m ign_lidar.cli.commands.process \
  --config-file configs/enrichment_asprs_full.yaml \
  input_dir=D:/ign/raw \
  output_dir=D:/ign/preprocessed/asprs
```

#### Option B: Use Python script

```python
from pathlib import Path
from ign_lidar.core.processor import LiDARProcessor
from omegaconf import OmegaConf

# Load configuration
config = OmegaConf.load("configs/enrichment_asprs_full.yaml")

# Override paths if needed
config.input_dir = "D:/ign/raw"
config.output_dir = "D:/ign/preprocessed/asprs"

# Create processor
processor = LiDARProcessor(config=config)

# Process tiles
processor.process_directory(
    input_dir=Path(config.input_dir),
    output_dir=Path(config.output_dir),
    num_workers=4,
    skip_existing=False  # Reprocess to fix missing features
)
```

### 4. What Gets Enriched

After processing with the updated configuration, your enriched LAZ files will include:

#### ASPRS Classification Codes:

- **Code 6**: Buildings (from BD TOPO® buildings)
- **Code 9**: Water (from BD TOPO® water surfaces)
- **Code 10**: Railways ✅ **NOW INCLUDED**
- **Code 11**: Roads ✅ **NOW INCLUDED**
- **Code 17**: Bridges (from BD TOPO® bridges)
- **Code 40**: Parking areas
- **Code 41**: Sports facilities
- **Code 44**: Agriculture (from RPG)

#### Extra Dimensions (optional):

- **Forest attributes**: forest_type, primary_species, estimated_height (from BD Forêt®)
- **Crop attributes**: crop_code, crop_category, crop_name, parcel_area, is_organic (from RPG)
- **Cadastre attributes**: parcel_id, parcel_area, commune_code (from BD PARCELLAIRE)

### 5. Verify Results

After processing, check your enriched LAZ files:

```python
import laspy

# Load enriched file
las = laspy.read("D:/ign/preprocessed/asprs/enriched_tiles/enriched/tile_name.laz")

# Check classification distribution
import numpy as np
unique, counts = np.unique(las.classification, return_counts=True)

print("Classification distribution:")
for code, count in zip(unique, counts):
    print(f"  Code {code}: {count:,} points")

# Look for roads (11) and railways (10)
roads = np.sum(las.classification == 11)
railways = np.sum(las.classification == 10)

print(f"\nRoads (code 11): {roads:,} points")
print(f"Railways (code 10): {railways:,} points")
```

### 6. Troubleshooting

If roads/railways still don't appear:

1. **Check BD TOPO® data availability**:

   - Roads and railways must exist in the BD TOPO® WFS service for your area
   - The fetcher queries: `BDTOPO_V3:troncon_de_route` (roads) and `BDTOPO_V3:troncon_de_voie_ferree` (railways)

2. **Check cache**:

   ```bash
   # Clear cache to force re-fetch
   rm -rf D:/ign/cache/ground_truth/*
   ```

3. **Enable debug logging**:

   ```yaml
   log_level: "DEBUG"
   verbose: true
   ```

4. **Check bounding box**:

   - Ensure your tiles overlap with road/railway infrastructure
   - Urban areas typically have more roads
   - Rural areas may have fewer features

5. **Verify WFS service**:
   - Check if IGN WFS service is accessible
   - Test URL: https://data.geopf.fr/wfs

### 7. Configuration Priority

Classification priority (from lowest to highest):

```
vegetation → water → parking → sports → agriculture →
power_lines → railways → roads → bridges → buildings
```

Higher priority features will overwrite lower priority ones in overlapping areas.

## Key Changes Summary

1. ✅ Added `parameters` section with width fallbacks
2. ✅ Enabled `roads: true` and `railways: true` in all configs
3. ✅ Added proper cache directory configuration
4. ✅ Set correct classification priority order
5. ✅ Created comprehensive `enrichment_asprs_full.yaml` config
6. ✅ Added export_attributes for BD Forêt®, RPG, and Cadastre

## Next Steps

1. Clear existing cache (optional, if you want fresh data):

   ```bash
   rm -rf D:/ign/cache/ground_truth/*
   ```

2. Run processing with updated configuration:

   ```bash
   python -m ign_lidar.cli.commands.process \
     --config-file configs/enrichment_asprs_full.yaml \
     input_dir=D:/ign/raw \
     output_dir=D:/ign/preprocessed/asprs
   ```

3. Verify enriched LAZ files contain roads (code 11) and railways (code 10)

4. If needed, adjust `road_width_fallback` and `railway_width_fallback` parameters based on your data

## Documentation

For more information, see:

- `BD_TOPO_RPG_INTEGRATION.md` - BD TOPO® features and RPG integration
- `RAILWAYS_FOREST_INTEGRATION_SUMMARY.md` - Railways and BD Forêt® details
- `configs/README.md` - Configuration guide
- `CLASSIFICATION_REFERENCE.md` - Complete classification system reference
