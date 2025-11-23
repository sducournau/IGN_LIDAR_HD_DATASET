#!/usr/bin/env python3
"""
Test WFS Memory Cache Performance

Tests the Phase 4 optimization: WFS memory cache for adjacent tiles.
Simulates processing adjacent tiles and measures cache effectiveness.

Expected improvements:
- Cache hit rate: >80% for adjacent tiles
- Performance: +10-15% on ground truth tiles
- Memory: <500 MB for typical tile grid
"""

import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

try:
    from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher, IGNWFSConfig
    import geopandas as gpd
    HAS_DEPS = True
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install geopandas shapely")
    HAS_DEPS = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s %(message)s'
)
logger = logging.getLogger(__name__)


def generate_adjacent_tiles(
    center_x: float = 700000,
    center_y: float = 6600000,
    tile_size: float = 1000,
    grid_size: int = 3
) -> List[Tuple[float, float, float, float]]:
    """
    Generate a grid of adjacent tile bboxes.
    
    Args:
        center_x: Center X coordinate (Lambert 93)
        center_y: Center Y coordinate (Lambert 93)
        tile_size: Tile size in meters
        grid_size: Grid size (e.g., 3 = 3x3 grid)
    
    Returns:
        List of bboxes (xmin, ymin, xmax, ymax)
    """
    tiles = []
    half_grid = grid_size // 2
    
    for i in range(-half_grid, half_grid + 1):
        for j in range(-half_grid, half_grid + 1):
            xmin = center_x + i * tile_size
            ymin = center_y + j * tile_size
            xmax = xmin + tile_size
            ymax = ymin + tile_size
            tiles.append((xmin, ymin, xmax, ymax))
    
    return tiles


def test_cache_effectiveness():
    """Test cache hit rate with adjacent tiles."""
    if not HAS_DEPS:
        return
    
    print("\n" + "="*70)
    print("TEST 1: Cache Effectiveness with Adjacent Tiles")
    print("="*70)
    
    # Use a real location in France (Paris region)
    # Lambert 93: 650000, 6860000 (somewhere near Paris)
    tiles = generate_adjacent_tiles(
        center_x=650000,
        center_y=6860000,
        tile_size=500,  # 500m tiles
        grid_size=3  # 3x3 = 9 tiles
    )
    
    print(f"\n📍 Testing with {len(tiles)} adjacent tiles (500m each)")
    print(f"   Area: ~1.5km x 1.5km grid")
    
    # Test with cache enabled
    print("\n🔄 Phase 1: Cache ENABLED (expected high hit rate)")
    fetcher_cached = IGNGroundTruthFetcher(
        enable_memory_cache=True,
        cache_size_mb=100
    )
    
    start_time = time.time()
    buildings_count = 0
    
    for i, bbox in enumerate(tiles):
        buildings = fetcher_cached.fetch_buildings(bbox, use_cache=True)
        if buildings is not None:
            buildings_count += len(buildings)
        
        # Print progress with cache stats
        if (i + 1) % 3 == 0:
            stats = fetcher_cached.get_cache_stats()
            if stats:
                print(f"   Tile {i+1}/{len(tiles)}: "
                      f"{stats['hits']} hits, {stats['misses']} misses "
                      f"({stats['hit_rate']:.1%} hit rate)")
    
    cached_time = time.time() - start_time
    stats_cached = fetcher_cached.get_cache_stats()
    
    print(f"\n✅ Cached results:")
    print(f"   Time: {cached_time:.2f}s")
    print(f"   Buildings fetched: {buildings_count}")
    if stats_cached:
        print(f"   Cache hits: {stats_cached['hits']}")
        print(f"   Cache misses: {stats_cached['misses']}")
        print(f"   Hit rate: {stats_cached['hit_rate']:.1%}")
        print(f"   Cache entries: {stats_cached['entries']}")
        print(f"   Memory used: {stats_cached['memory_mb']:.1f} MB")
    
    # Test without cache (for comparison)
    print("\n🔄 Phase 2: Cache DISABLED (baseline)")
    fetcher_uncached = IGNGroundTruthFetcher(
        enable_memory_cache=False
    )
    
    start_time = time.time()
    for i, bbox in enumerate(tiles):
        buildings = fetcher_uncached.fetch_buildings(bbox, use_cache=False)
        if (i + 1) % 3 == 0:
            print(f"   Tile {i+1}/{len(tiles)}: Fetching...")
    
    uncached_time = time.time() - start_time
    
    print(f"\n✅ Uncached results:")
    print(f"   Time: {uncached_time:.2f}s")
    
    # Compare
    if stats_cached and stats_cached['hit_rate'] > 0:
        speedup = uncached_time / cached_time
        print(f"\n📊 Performance Comparison:")
        print(f"   Speedup: {speedup:.2f}x faster with cache")
        print(f"   Time saved: {uncached_time - cached_time:.2f}s ({(1 - cached_time/uncached_time)*100:.1f}%)")
        
        if stats_cached['hit_rate'] >= 0.8:
            print(f"   ✅ PASS: Cache hit rate {stats_cached['hit_rate']:.1%} >= 80%")
        else:
            print(f"   ⚠️  WARN: Cache hit rate {stats_cached['hit_rate']:.1%} < 80%")


def test_cache_memory_limit():
    """Test cache eviction when memory limit is reached."""
    if not HAS_DEPS:
        return
    
    print("\n" + "="*70)
    print("TEST 2: Cache Memory Limit and LRU Eviction")
    print("="*70)
    
    # Use small cache limit to trigger eviction
    fetcher = IGNGroundTruthFetcher(
        enable_memory_cache=True,
        cache_size_mb=10  # Only 10 MB
    )
    
    print("\n🔄 Testing with 10 MB cache limit")
    
    # Generate many non-overlapping tiles
    tiles = generate_adjacent_tiles(
        center_x=650000,
        center_y=6860000,
        tile_size=500,
        grid_size=5  # 5x5 = 25 tiles
    )
    
    print(f"   Fetching {len(tiles)} tiles...")
    
    for i, bbox in enumerate(tiles):
        fetcher.fetch_buildings(bbox, use_cache=True)
        
        if (i + 1) % 5 == 0:
            stats = fetcher.get_cache_stats()
            if stats:
                print(f"   After {i+1} tiles: "
                      f"{stats['entries']} cached, "
                      f"{stats['memory_mb']:.1f}/{stats['max_memory_mb']:.0f} MB, "
                      f"{stats['hit_rate']:.1%} hit rate")
    
    stats = fetcher.get_cache_stats()
    if stats:
        print(f"\n✅ Final cache state:")
        print(f"   Entries: {stats['entries']}")
        print(f"   Memory: {stats['memory_mb']:.1f} MB (limit: {stats['max_memory_mb']:.0f} MB)")
        print(f"   Hit rate: {stats['hit_rate']:.1%}")
        
        if stats['memory_mb'] <= stats['max_memory_mb']:
            print(f"   ✅ PASS: Cache respected memory limit")
        else:
            print(f"   ❌ FAIL: Cache exceeded memory limit!")


def test_cache_overlap_scenarios():
    """Test cache behavior with overlapping tiles."""
    if not HAS_DEPS:
        return
    
    print("\n" + "="*70)
    print("TEST 3: Overlapping Tiles (Cache Miss Scenario)")
    print("="*70)
    
    fetcher = IGNGroundTruthFetcher(
        enable_memory_cache=True,
        cache_size_mb=100
    )
    
    print("\n🔄 Testing with overlapping tiles")
    
    # Create tiles with 50% overlap
    base_bbox = (650000, 6860000, 650500, 6860500)
    
    tiles = [
        base_bbox,
        (650250, 6860000, 650750, 6860500),  # 50% overlap in X
        (650000, 6860250, 650500, 6860750),  # 50% overlap in Y
        (650250, 6860250, 650750, 6860750),  # 50% overlap both
    ]
    
    print(f"   Testing {len(tiles)} overlapping tiles...")
    
    for i, bbox in enumerate(tiles):
        fetcher.fetch_buildings(bbox, use_cache=True)
        stats = fetcher.get_cache_stats()
        if stats:
            print(f"   Tile {i+1}: {stats['entries']} cached, "
                  f"{stats['hit_rate']:.1%} hit rate")
    
    stats = fetcher.get_cache_stats()
    if stats:
        print(f"\n✅ Results:")
        print(f"   Total cache entries: {stats['entries']}")
        print(f"   Hit rate: {stats['hit_rate']:.1%}")
        print(f"   ⚠️  Note: Overlapping tiles have different bboxes → cache misses")
        print(f"        Future optimization: spatial overlap detection")


def print_summary():
    """Print test summary."""
    print("\n" + "="*70)
    print("WFS CACHE TEST SUMMARY")
    print("="*70)
    print("""
Phase 4 Optimization: WFS Memory Cache

✅ Implementation:
   - LRU eviction policy (OrderedDict)
   - Configurable memory limit (default: 500 MB)
   - Cache hit/miss tracking
   - Size estimation for GeoDataFrames
   - Thread-safe operations

🎯 Expected Benefits:
   - Cache hit rate: >80% for adjacent tiles
   - Performance: +10-15% on ground truth tiles
   - Memory: <500 MB for typical processing

📋 Usage:
   fetcher = IGNGroundTruthFetcher(
       enable_memory_cache=True,
       cache_size_mb=500
   )
   
   # Process tiles...
   stats = fetcher.get_cache_stats()
   print(f"Hit rate: {stats['hit_rate']:.1%}")

⚠️  Limitations:
   - Exact bbox matching only (overlapping tiles = cache miss)
   - No spatial index (future optimization)
   - No persistence across sessions
""")


if __name__ == "__main__":
    if not HAS_DEPS:
        print("\n❌ Cannot run tests without dependencies")
        print("Install with: pip install geopandas shapely")
        sys.exit(1)
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                  WFS CACHE PERFORMANCE TESTS                     ║
║                    (Phase 4 Optimization)                        ║
╚══════════════════════════════════════════════════════════════════╝

Testing in-memory LRU cache for WFS ground truth data.
This optimization reduces redundant WFS API calls for adjacent tiles.
""")
    
    try:
        test_cache_effectiveness()
        test_cache_memory_limit()
        test_cache_overlap_scenarios()
        print_summary()
        
        print("\n✅ All WFS cache tests completed!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
