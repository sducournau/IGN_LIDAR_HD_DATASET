#!/usr/bin/env python3
"""
Test script to verify neighbor tile detection
"""
from pathlib import Path
from ign_lidar.core.tile_stitcher import TileStitcher

# Test tile path
tile_path = Path("/mnt/c/Users/Simon/ign/raw_tiles/urban_dense/LHD_FXX_0891_6248_PTS_C_LAMB93_IGN69.laz")

print("=" * 80)
print("TILE STITCHER - NEIGHBOR DETECTION TEST")
print("=" * 80)

# Initialize stitcher
stitcher = TileStitcher(buffer_size=15.0)

# Test 1: Core tile bounds
print(f"\n[1] Core tile: {tile_path.name}")
try:
    core_bounds = stitcher.get_tile_bounds(tile_path)
    print(f"    Bounds: X=[{core_bounds[0]:.2f}, {core_bounds[2]:.2f}], Y=[{core_bounds[1]:.2f}, {core_bounds[3]:.2f}]")
    print(f"    Size: {core_bounds[2]-core_bounds[0]:.2f}m x {core_bounds[3]-core_bounds[1]:.2f}m")
except Exception as e:
    print(f"    ✗ Error getting bounds: {e}")

# Test 2: Bbox-based neighbor detection
print(f"\n[2] Testing bbox-based neighbor detection...")
try:
    neighbors_bbox = stitcher._detect_neighbors_by_bbox(tile_path)
    if neighbors_bbox:
        print(f"    ✓ Found {len(neighbors_bbox)} neighbors via bbox:")
        for n in neighbors_bbox:
            n_bounds = stitcher.get_tile_bounds(n)
            print(f"      - {n.name}")
            print(f"        Bounds: X=[{n_bounds[0]:.2f}, {n_bounds[2]:.2f}], Y=[{n_bounds[1]:.2f}, {n_bounds[3]:.2f}]")
    else:
        print(f"    ✗ No neighbors found via bbox")
except Exception as e:
    print(f"    ✗ Bbox detection failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Pattern-based neighbor detection
print(f"\n[3] Testing pattern-based neighbor detection...")
try:
    neighbors_pattern = stitcher._detect_neighbors_by_pattern(tile_path)
    if neighbors_pattern:
        print(f"    ✓ Found {len(neighbors_pattern)} neighbors via pattern:")
        for n in neighbors_pattern:
            print(f"      - {n.name}")
    else:
        print(f"    ✗ No neighbors found via pattern")
except Exception as e:
    print(f"    ✗ Pattern detection failed: {e}")

# Test 4: Full auto-detection
print(f"\n[4] Testing full auto-detection (bbox + fallback)...")
neighbors = stitcher._detect_neighbor_tiles(tile_path)
if neighbors:
    print(f"    ✓ Found {len(neighbors)} neighbors:")
    for n in neighbors:
        print(f"      - {n.name}")
else:
    print("    ✗ No neighbors found!")

# Test 5: Full tile loading with neighbors
print("\n" + "=" * 80)
print("FULL TILE LOADING TEST")
print("=" * 80)

try:
    tile_data = stitcher.load_tile_with_neighbors(
        tile_path=tile_path,
        auto_detect_neighbors=True
    )
    
    print(f"\n✓ Tile loading successful!")
    print(f"  Core points:   {tile_data['num_core']:>10,}")
    print(f"  Buffer points: {tile_data['num_buffer']:>10,}")
    print(f"  Total points:  {tile_data['num_core'] + tile_data['num_buffer']:>10,}")
    print(f"  Core bounds: {tile_data['core_bounds']}")
    
    if tile_data['num_buffer'] > 0:
        buffer_ratio = tile_data['num_buffer'] / tile_data['num_core'] * 100
        print(f"\n✓✓ SUCCESS: Buffer points extracted from neighbors!")
        print(f"   Buffer represents {buffer_ratio:.1f}% of core tile size")
    else:
        print("\n✗ WARNING: No buffer points extracted!")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
