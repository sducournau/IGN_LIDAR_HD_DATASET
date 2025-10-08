#!/usr/bin/env python3
"""
Test script for auto-downloading missing adjacent tiles
"""
from pathlib import Path
from ign_lidar.core.tile_stitcher import TileStitcher

# Test tile path
tile_path = Path("/mnt/c/Users/Simon/ign/raw_tiles/urban_dense/LHD_FXX_0891_6248_PTS_C_LAMB93_IGN69.laz")

print("=" * 80)
print("TILE STITCHER - AUTO-DOWNLOAD MISSING NEIGHBORS TEST")
print("=" * 80)

# Test 1: Check current neighbors
print(f"\n[1] Checking current neighbors for: {tile_path.name}")
stitcher_no_download = TileStitcher(buffer_size=15.0)

current_neighbors = stitcher_no_download._detect_neighbor_tiles(tile_path)
if current_neighbors:
    print(f"    ✓ Currently have {len(current_neighbors)} neighbors:")
    for n in current_neighbors:
        # Validate each existing neighbor
        is_valid = stitcher_no_download._validate_tile(n)
        status = "✓ valid" if is_valid else "✗ corrupted"
        print(f"      - {n.name} ({status})")
else:
    print(f"    ✗ No neighbors currently available")

# Test 2: Identify missing neighbors
print(f"\n[2] Identifying missing neighbors...")
missing = stitcher_no_download._identify_missing_neighbors(
    tile_path, 
    current_neighbors or []
)

if missing:
    print(f"    ✓ Found {len(missing)} missing neighbors:")
    for m in missing:
        print(f"      - {m['direction']:10s}: Center at ({m['center_x']:.0f}, {m['center_y']:.0f})")
else:
    print(f"    ✓ All neighbors are present!")

# Test 3: Initialize stitcher with auto-download enabled
print(f"\n[3] Testing with auto-download enabled...")
config_with_download = {
    'buffer_size': 15.0,
    'auto_detect_neighbors': True,
    'auto_download_neighbors': True,  # Enable auto-download
    'cache_enabled': True
}

stitcher_with_download = TileStitcher(config=config_with_download)
print(f"    ✓ Stitcher initialized with auto_download_neighbors=True")

# Test 4: Load tile with auto-download (dry run - shows what would be downloaded)
print(f"\n[4] Loading tile with neighbor auto-detection...")
print(f"    (This will attempt to download missing neighbors if they exist in IGN WFS)")

try:
    tile_data = stitcher_with_download.load_tile_with_neighbors(
        tile_path=tile_path,
        auto_detect_neighbors=True
    )
    
    print(f"\n✓ Tile loading successful!")
    print(f"  Core points:   {tile_data['num_core']:>10,}")
    print(f"  Buffer points: {tile_data['num_buffer']:>10,}")
    print(f"  Total points:  {tile_data['num_core'] + tile_data['num_buffer']:>10,}")
    
    if tile_data['num_buffer'] > 0:
        buffer_ratio = tile_data['num_buffer'] / tile_data['num_core'] * 100
        print(f"\n✓✓ SUCCESS: Buffer points extracted!")
        print(f"   Buffer represents {buffer_ratio:.1f}% of core tile")
    else:
        print(f"\n⚠️  No buffer points - either no neighbors available or download failed")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Verify download results
print(f"\n[5] Checking if new neighbors were downloaded...")
final_neighbors = stitcher_with_download._detect_neighbor_tiles(tile_path)
if final_neighbors:
    new_count = len(final_neighbors) - (len(current_neighbors) if current_neighbors else 0)
    print(f"    Final count: {len(final_neighbors)} neighbors")
    if new_count > 0:
        print(f"    ✓✓ Downloaded {new_count} new neighbors!")
    else:
        print(f"    No new neighbors downloaded (may already exist or not available)")
else:
    print(f"    Still no neighbors found")

print("\n" + "=" * 80)
print("CONFIGURATION EXAMPLE")
print("=" * 80)
print("""
To enable auto-download in your processing pipeline:

1. Via Python API:
   
   from ign_lidar.core.processor import LiDARProcessor
   
   processor = LiDARProcessor(
       use_stitching=True,
       buffer_size=15.0,
       stitching_config={
           'auto_detect_neighbors': True,
           'auto_download_neighbors': True,  # Enable auto-download
           'cache_enabled': True
       }
   )

2. Via Hydra Config (add to your config YAML):
   
   processor:
     use_stitching: true
     buffer_size: 15.0
     stitching_config:
       auto_detect_neighbors: true
       auto_download_neighbors: true  # Enable auto-download
       cache_enabled: true

Benefits:
- Automatically downloads missing neighbors for seamless boundary processing
- Reduces edge artifacts in feature computation
- No manual tile management needed
- Falls back gracefully if tiles are not available in IGN WFS
""")
