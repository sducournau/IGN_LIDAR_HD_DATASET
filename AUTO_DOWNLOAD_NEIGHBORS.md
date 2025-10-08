# Auto-Download Adjacent Tiles Feature

## Overview

The tile stitcher now supports **automatic downloading of missing adjacent tiles** to ensure seamless boundary-aware feature computation. This eliminates edge artifacts by automatically fetching neighboring tiles from the IGN WFS service when they're needed but not present locally.

## Key Features

### 1. **Intelligent Neighbor Detection**

- Uses **bounding box adjacency checks** (primary method)
- Falls back to **filename pattern matching** if bbox check fails
- Supports multiple IGN LiDAR HD naming conventions

### 2. **Automatic Download**

- Identifies missing adjacent tiles (N, S, E, W, NE, NW, SE, SW)
- Queries IGN WFS service for available tiles
- Downloads only tiles that are spatially adjacent
- Skips tiles that already exist locally

### 3. **Graceful Fallback**

- Continues processing even if downloads fail
- Falls back to standard (non-stitched) processing
- No pipeline interruption

## Configuration

### Python API

```python
from ign_lidar.core.tile_stitcher import TileStitcher

# Method 1: Direct TileStitcher usage
config = {
    'buffer_size': 15.0,                    # Buffer zone width in meters
    'auto_detect_neighbors': True,          # Auto-detect adjacent tiles
    'auto_download_neighbors': True,        # Download missing neighbors
    'cache_enabled': True                   # Cache loaded tiles
}

stitcher = TileStitcher(config=config)
tile_data = stitcher.load_tile_with_neighbors(
    tile_path=Path("your_tile.laz"),
    auto_detect_neighbors=True
)
```

```python
# Method 2: Through LiDARProcessor
from ign_lidar.core.processor import LiDARProcessor

processor = LiDARProcessor(
    use_stitching=True,
    buffer_size=15.0,
    stitching_config={
        'auto_detect_neighbors': True,
        'auto_download_neighbors': True
    }
)
```

### Hydra Configuration

Add to your config YAML file:

```yaml
processor:
  use_stitching: true
  buffer_size: 15.0
  stitching_config:
    auto_detect_neighbors: true
    auto_download_neighbors: true # Enable auto-download
    cache_enabled: true
    parallel_loading: false # Optional: parallel neighbor loading
```

### Command Line (via Hydra overrides)

```bash
python -m ign_lidar.cli.process \
  input_dir=/path/to/tiles \
  output_dir=/path/to/output \
  processor.use_stitching=true \
  processor.stitching_config.auto_download_neighbors=true
```

## How It Works

### Step 1: Detect Neighbors

```
Core Tile: 0891_6248
Expected Neighbors:
  - North:     0891_6249
  - South:     0891_6247  ✓ (exists)
  - East:      0892_6248  ✓ (exists)
  - West:      0890_6248
  - Northeast: 0892_6249
  - Northwest: 0890_6249
  - Southeast: 0892_6247  ✓ (exists)
  - Southwest: 0890_6247
```

### Step 2: Identify Missing

```
Missing: North, West, NE, NW, SW (5 tiles)
```

### Step 3: Query WFS

```
Querying IGN WFS service for tiles in bounding box...
Found 3 matching tiles in WFS
```

### Step 4: Download

```
Downloading north neighbor: LHD_FXX_0891_6249_PTS_C_LAMB93_IGN69.laz
✓ Downloaded (245 MB)
Downloading west neighbor: LHD_FXX_0890_6248_PTS_C_LAMB93_IGN69.laz
✓ Downloaded (198 MB)
...
```

### Step 5: Extract Buffers

```
Extracting 15m buffer zones from 6 neighbors...
✓ Buffer extraction complete: 892,456 buffer points
```

## Neighbor Detection Methods

### Primary: Bounding Box Adjacency

The primary method loads tile bounds and checks spatial adjacency:

```python
# Core tile bounds: (891000, 6247000, 892000, 6248000)
# Neighbor tile bounds: (892000, 6247000, 893000, 6248000)
# Shares vertical edge at x=892000 → Adjacent!
```

**Advantages:**

- Works with any filename pattern
- Spatially accurate
- Handles irregular tile grids

### Fallback: Filename Pattern Matching

Supports two IGN LiDAR HD patterns:

1. **Simple pattern**: `LIDAR_HD_<XXXX>_<YYYY>.laz`
2. **Full pattern**: `LHD_FXX_<XXXX>_<YYYY>_PTS_C_LAMB93_IGN69.laz`

```python
# Core: LHD_FXX_0891_6248_PTS_C_LAMB93_IGN69.laz
# East neighbor: LHD_FXX_0892_6248_PTS_C_LAMB93_IGN69.laz
```

## Performance Considerations

### Download Time

- Each tile: ~100-300 MB (1-3 minutes at 1 MB/s)
- 8 neighbors: ~15-25 minutes worst case
- Parallel downloads: 2 concurrent by default

### Storage

- Each tile: ~100-300 MB
- 8 neighbors: ~1-2 GB additional storage
- Tiles reused across multiple core tiles

### Memory

- Buffer points cached during processing
- Cache cleared after each tile (configurable)
- Typical memory overhead: ~500 MB per tile with neighbors

## Best Practices

### 1. **Use for Boundary-Critical Applications**

Enable auto-download when edge quality matters:

- Building detection near tile boundaries
- Road network extraction
- Continuous terrain analysis

### 2. **Pre-download for Large Datasets**

For processing many tiles in an area:

```python
# Pre-download region
from ign_lidar.downloader import IGNLiDARDownloader

downloader = IGNLiDARDownloader(output_dir="tiles/")
downloader.download_region(
    bbox=(890000, 6247000, 893000, 6250000)
)

# Then process without auto-download
processor = LiDARProcessor(
    use_stitching=True,
    stitching_config={'auto_download_neighbors': False}
)
```

### 3. **Monitor Download Progress**

```python
import logging
logging.basicConfig(level=logging.INFO)

# Detailed logs will show:
# - Neighbor detection progress
# - WFS query results
# - Download status for each tile
```

### 4. **Disable for Offline Processing**

```yaml
processor:
  stitching_config:
    auto_detect_neighbors: true
    auto_download_neighbors: false # Offline mode
```

## Troubleshooting

### No Neighbors Downloaded

**Possible causes:**

1. Tiles not available in IGN WFS service
2. Network connectivity issues
3. WFS service temporarily unavailable
4. Tiles outside IGN coverage area

**Solution:** Check logs for specific error messages

### Download Fails

```
Failed to download missing neighbors: HTTPError 503
```

**Solution:**

- Retry later (service may be busy)
- Pre-download tiles manually
- Disable auto-download and process without stitching

### Wrong Tiles Downloaded

**Cause:** Coordinate conversion approximation

**Solution:** The code uses rough Lambert93↔WGS84 conversion for WFS queries. For production, install `pyproj` for accurate conversion:

```bash
pip install pyproj
```

Then update `_lambert93_to_wgs84_bbox()` method to use pyproj.

## Example Use Cases

### Use Case 1: Urban Building Detection

```python
# Process city center with seamless boundaries
processor = LiDARProcessor(
    use_stitching=True,
    buffer_size=20.0,  # Larger buffer for buildings
    stitching_config={
        'auto_download_neighbors': True
    }
)
```

### Use Case 2: Road Network Extraction

```python
# Continuous road features across tiles
config = {
    'buffer_size': 15.0,
    'auto_download_neighbors': True,
    'compute_boundary_features': True
}
stitcher = TileStitcher(config=config)
```

### Use Case 3: Terrain Analysis

```python
# Smooth elevation gradients at boundaries
processor.process_tile(
    laz_file=tile_path,
    use_stitching=True
)
```

## API Reference

### TileStitcher Configuration Options

| Option                    | Type  | Default | Description                 |
| ------------------------- | ----- | ------- | --------------------------- |
| `buffer_size`             | float | 15.0    | Buffer zone width in meters |
| `auto_detect_neighbors`   | bool  | True    | Auto-detect adjacent tiles  |
| `auto_download_neighbors` | bool  | False   | Download missing neighbors  |
| `cache_enabled`           | bool  | True    | Cache loaded tiles          |
| `parallel_loading`        | bool  | False   | Load neighbors in parallel  |

### Methods

#### `_detect_neighbor_tiles(tile_path)`

Auto-detect adjacent tiles using bbox or pattern matching.

#### `_identify_missing_neighbors(tile_path, found_neighbors)`

Identify which adjacent positions lack tiles.

#### `_download_missing_neighbors(missing_neighbors, output_dir)`

Download missing tiles from IGN WFS service.

## Testing

Run the test script to verify functionality:

```bash
python test_auto_download.py
```

Expected output:

```
[1] Checking current neighbors...
    ✓ Currently have 3 neighbors
[2] Identifying missing neighbors...
    ✓ Found 5 missing neighbors
[3] Testing with auto-download enabled...
    ✓ Stitcher initialized
[4] Loading tile with auto-download...
    Downloading north neighbor...
    ✓ Downloaded (245 MB)
    ...
✓✓ SUCCESS: Buffer points extracted!
```

## Future Enhancements

1. **Accurate coordinate conversion** using pyproj
2. **Resume partial downloads** for large tiles
3. **Parallel downloads** (configurable concurrency)
4. **Cache WFS metadata** to reduce API calls
5. **Progress bars** for download monitoring
6. **Checksum verification** for downloaded tiles

## Related Documentation

- [Tile Stitching Guide](TILE_STITCHING.md)
- [Boundary-Aware Features](BOUNDARY_FEATURES.md)
- [IGN Downloader API](DOWNLOADER_API.md)
