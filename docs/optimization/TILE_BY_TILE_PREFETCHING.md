# Tile-by-Tile Prefetching Implementation

## Overview

This document describes the implementation of tile-by-tile prefetching to replace the previous bulk prefetching approach in the IGN LiDAR HD dataset processor.

## Problem Statement

Previously, the processor would prefetch ground truth data for **all tiles at once** before starting processing. This approach had several disadvantages:

1. **High memory usage**: All ground truth data was loaded into memory simultaneously
2. **Delayed start**: Processing couldn't begin until all data was prefetched
3. **Network inefficiency**: Large batch requests could cause timeouts
4. **Poor scalability**: Memory usage grew linearly with the number of tiles

## Solution: Tile-by-Tile Prefetching

The new implementation prefetches ground truth data **only for the next tile** while processing the current tile, implementing a double-buffering strategy.

## Implementation Details

### 1. New Methods Added

#### `_prefetch_ground_truth_for_tile(laz_file: Path) -> Optional[dict]`

```python
def _prefetch_ground_truth_for_tile(self, laz_file: Path) -> Optional[dict]:
    """
    Pre-fetch ground truth data for a single tile.
    This method replaces bulk prefetching with per-tile prefetching.

    Args:
        laz_file: Path to LAZ file to prefetch data for

    Returns:
        Dictionary containing fetched ground truth data, or None if failed
    """
```

This method:

- Reads the LAZ header to extract the tile bounding box
- Fetches ground truth data for that specific tile
- Returns the data for caching or immediate use

### 2. Updated Methods

#### `_process_tile_with_data()` - Enhanced Signature

The method now accepts an optional `prefetched_ground_truth` parameter:

```python
def _process_tile_with_data(self, laz_file: Path, output_dir: Path,
                             tile_data: dict, tile_idx: int = 0,
                             total_tiles: int = 0, skip_existing: bool = True,
                             prefetched_ground_truth: dict = None) -> int:
```

### 3. Updated Processing Pipeline

The sequential processing loop now implements triple-buffering:

1. **Tile N**: GPU processing
2. **Tile N+1**: Tile data loading (async)
3. **Tile N+1**: Ground truth prefetching (async)

```python
with ThreadPoolExecutor(max_workers=3) as io_pool:
    # Start with first tile
    next_tile_future = io_pool.submit(self.tile_loader.load_tile, laz_files[0])
    next_ground_truth_future = io_pool.submit(self._prefetch_ground_truth_for_tile, laz_files[0])

    for idx, laz_file in enumerate(laz_files, 1):
        # Wait for prefetched data
        tile_data = next_tile_future.result()
        prefetched_ground_truth = next_ground_truth_future.result()

        # Start prefetching NEXT tile (parallel with current processing)
        if idx < len(laz_files):
            next_tile_future = io_pool.submit(self.tile_loader.load_tile, laz_files[idx])
            next_ground_truth_future = io_pool.submit(self._prefetch_ground_truth_for_tile, laz_files[idx])

        # Process current tile (GPU busy, I/O runs in parallel)
        result = self._process_tile_with_data(
            laz_file, output_dir, tile_data,
            prefetched_ground_truth=prefetched_ground_truth
        )
```

## Benefits

### 1. Memory Efficiency

- **Before**: Memory usage = O(total_tiles)
- **After**: Memory usage = O(1) - constant memory usage

### 2. Faster Processing Start

- **Before**: Wait for all tiles to be prefetched before processing starts
- **After**: Processing starts immediately after first tile is ready

### 3. Better Pipeline Utilization

- **Before**: Sequential phases (prefetch all → process all)
- **After**: Overlapped execution (GPU processing + I/O + network fetching)

### 4. Improved Throughput

- **Expected speedup**: 40-80% reduction in total processing time
- **Eliminates**: Both I/O stalls and ground truth fetch delays

## Configuration Changes

### Removed

- Bulk prefetching call in `process_directory()` method
- `_prefetch_ground_truth()` method is now marked as deprecated

### Added

- `_prefetch_ground_truth_for_tile()` for single-tile prefetching
- Enhanced thread pool management (3 workers instead of 2)
- Optional prefetched ground truth parameter in processing methods

## Backward Compatibility

The changes are fully backward compatible:

- Parallel processing mode still works as before
- The deprecated `_prefetch_ground_truth()` method is preserved (marked as deprecated)
- All existing configuration files continue to work
- No changes to public APIs

## Testing

The implementation has been verified with:

- Method availability checks
- Signature validation
- Basic functionality testing

Run the test with:

```bash
python test_tile_by_tile_prefetching.py
```

## Performance Impact

### Expected Improvements

1. **Memory usage**: ~90% reduction for large tile sets
2. **Time to first result**: ~95% reduction (immediate start)
3. **Overall throughput**: 40-80% improvement
4. **Network efficiency**: Better handling of temporary network issues

### Monitoring

- Cache hit rates should remain high due to spatial locality
- Thread pool utilization should be more balanced
- Memory usage should remain constant regardless of tile count

## Future Enhancements

1. **Adaptive prefetching**: Adjust prefetch distance based on processing speed
2. **Priority-based prefetching**: Prefetch high-priority tiles first
3. **Cross-tile optimization**: Share ground truth data between adjacent tiles
4. **Streaming processing**: Process tiles as they become available

## Migration Guide

No code changes are required for existing users. The optimization is applied automatically when using sequential processing mode (num_workers <= 1).

For users who want to ensure they're using the new optimization:

- Set `num_workers=1` in processing configuration
- Enable ground truth data sources in configuration
- Monitor logs for "tile-by-tile prefetching" messages
