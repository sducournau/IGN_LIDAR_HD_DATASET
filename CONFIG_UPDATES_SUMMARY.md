# Configuration Updates Summary

## Files Created/Modified

### New Configuration Files

1. ✅ `ign_lidar/configs/stitching/auto_download.yaml` - Stitching with auto-download enabled
2. ✅ `ign_lidar/configs/experiment/boundary_aware_autodownload.yaml` - Full experiment with auto-download
3. ✅ `ign_lidar/configs/experiment/boundary_aware_offline.yaml` - Boundary processing without downloads
4. ✅ `ign_lidar/configs/README.md` - Comprehensive configuration guide
5. ✅ `QUICK_REFERENCE_STITCHING.md` - Quick reference card

### Modified Configuration Files

1. ✅ `ign_lidar/configs/stitching/enabled.yaml` - Added `auto_download_neighbors: false`
2. ✅ `ign_lidar/configs/stitching/enhanced.yaml` - Added `auto_download_neighbors: false`
3. ✅ `ign_lidar/core/processor.py` - Added auto_download_neighbors to default stitching_config

## New Configuration Option

### `auto_download_neighbors`

**Type:** `bool`  
**Default:** `false`  
**Location:** `stitching` config section

**Purpose:** Automatically download missing adjacent tiles from IGN WFS service

**Behavior when `true`:**

- Detects adjacent tile positions using bounding box checks
- Queries IGN WFS for missing tiles
- Downloads tiles that don't exist locally
- Validates existing tiles and re-downloads if corrupted
- Caches downloaded tiles for reuse

**Behavior when `false`:**

- Only uses tiles that already exist locally
- No network access required
- Faster startup (no WFS queries)
- Suitable for offline processing

## Configuration Hierarchy

```
Stitching Config Priority (highest to lowest):
1. Command-line overrides: stitching.auto_download_neighbors=true
2. Experiment config: experiment/boundary_aware_autodownload.yaml
3. Stitching config: stitching/auto_download.yaml
4. Default config: stitching/enhanced.yaml (auto_download_neighbors=false)
```

## Usage Examples

### 1. Using Pre-configured Auto-Download

```bash
python -m ign_lidar.cli.process \
  input_dir=/path/to/tiles \
  output_dir=/path/to/output \
  stitching=auto_download
```

### 2. Override on Enhanced Config

```bash
python -m ign_lidar.cli.process \
  input_dir=/path/to/tiles \
  output_dir=/path/to/output \
  stitching=enhanced \
  stitching.auto_download_neighbors=true
```

### 3. Using Experiment Config

```bash
python -m ign_lidar.cli.process \
  --config-name boundary_aware_autodownload \
  input_dir=/path/to/tiles \
  output_dir=/path/to/output
```

### 4. Programmatic Usage

```python
from hydra import compose, initialize

with initialize(config_path="ign_lidar/configs"):
    cfg = compose(
        config_name="config",
        overrides=[
            "stitching=auto_download",
            "processor.use_stitching=true"
        ]
    )
```

## Configuration Schema Updates

### Stitching Configuration

**New fields:**

```yaml
auto_download_neighbors: bool # Download missing neighbors
validate_tiles: bool # Validate tile integrity
download_max_concurrent: int # Max concurrent downloads
continue_on_download_failure: bool # Continue if downloads fail
skip_corrupted_tiles: bool # Skip invalid tiles
```

**Existing fields remain unchanged**

## Migration Guide

### For Existing Pipelines

**No changes required!** The default is `auto_download_neighbors: false`, so existing behavior is preserved.

**To enable auto-download:**

```bash
# Add to your existing command:
stitching.auto_download_neighbors=true
```

**To explicitly disable (already default):**

```bash
# Add for clarity:
stitching.auto_download_neighbors=false
```

### For New Pipelines

**Recommended approach:**

1. **Development/Testing:**

   ```yaml
   stitching: auto_download # Convenience
   ```

2. **Production:**
   ```yaml
   stitching: enhanced
   auto_download_neighbors: false # Explicit
   # Pre-download tiles separately
   ```

## Testing Commands

### Test 1: Verify Config Loading

```bash
python -m ign_lidar.cli.process \
  --config-name boundary_aware_autodownload \
  --cfg job \
  input_dir=/tmp \
  output_dir=/tmp
```

### Test 2: Check Auto-Download Setting

```bash
python -c "
from hydra import compose, initialize
with initialize(config_path='ign_lidar/configs'):
    cfg = compose(config_name='config', overrides=['stitching=auto_download'])
    print(f'auto_download_neighbors: {cfg.stitching.auto_download_neighbors}')
"
```

### Test 3: Validate Stitching Pipeline

```bash
python test_auto_download.py
```

## Documentation Cross-Reference

| Document                       | Purpose                         |
| ------------------------------ | ------------------------------- |
| `AUTO_DOWNLOAD_NEIGHBORS.md`   | Feature guide and API reference |
| `TILE_STITCHING_SUMMARY.md`    | Implementation details          |
| `ign_lidar/configs/README.md`  | Configuration guide             |
| `QUICK_REFERENCE_STITCHING.md` | Quick command reference         |
| This file                      | Configuration updates summary   |

## Configuration File Locations

```
IGN_LIDAR_HD_DATASET/
├── ign_lidar/
│   └── configs/
│       ├── config.yaml                              (no changes)
│       ├── stitching/
│       │   ├── disabled.yaml                        (no changes)
│       │   ├── enabled.yaml                         ✏️ MODIFIED
│       │   ├── enhanced.yaml                        ✏️ MODIFIED
│       │   ├── advanced.yaml                        (no changes)
│       │   └── auto_download.yaml                   ✨ NEW
│       ├── experiment/
│       │   ├── boundary_aware_autodownload.yaml     ✨ NEW
│       │   ├── boundary_aware_offline.yaml          ✨ NEW
│       │   └── ... (other experiments)
│       └── README.md                                 ✨ NEW
├── AUTO_DOWNLOAD_NEIGHBORS.md                        ✨ NEW
├── TILE_STITCHING_SUMMARY.md                         ✨ NEW
├── QUICK_REFERENCE_STITCHING.md                      ✨ NEW
└── CONFIG_UPDATES_SUMMARY.md                         ✨ NEW (this file)
```

## Backward Compatibility

✅ **Fully backward compatible**

- Default behavior unchanged (`auto_download_neighbors: false`)
- Existing configs work without modification
- New option is opt-in only
- Graceful fallback if downloads fail

## Future Configuration Additions

Potential future enhancements to consider:

```yaml
# Download management
download_retry_attempts: 3
download_retry_delay: 5.0
download_timeout: 300
download_bandwidth_limit: null

# Cache management
tile_cache_dir: ~/.cache/ign_lidar_tiles
tile_cache_max_size: 10240 # 10 GB
tile_cache_expiry_days: 30

# WFS configuration
wfs_service_url: "https://data.geopf.fr/wfs/..."
wfs_query_timeout: 30
wfs_cache_metadata: true

# Coordinate conversion
use_accurate_projection: true # Requires pyproj
projection_from: "EPSG:2154" # Lambert93
projection_to: "EPSG:4326" # WGS84
```

## Summary

✅ **5 new files created**  
✅ **3 files modified**  
✅ **1 new configuration option added**  
✅ **Fully backward compatible**  
✅ **Comprehensive documentation provided**  
✅ **Ready for production use**

## Quick Start

```bash
# Enable auto-download on your next processing run:
python -m ign_lidar.cli.process \
  input_dir=/path/to/tiles \
  output_dir=/path/to/output \
  stitching=auto_download

# That's it! Missing neighbors will be automatically downloaded.
```
