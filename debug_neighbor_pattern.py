#!/usr/bin/env python3
"""
Debug script for neighbor detection
"""
from pathlib import Path

tile_path = Path("/mnt/c/Users/Simon/ign/raw_tiles/urban_dense/LHD_FXX_0891_6248_PTS_C_LAMB93_IGN69.laz")

filename = tile_path.stem
print(f"Filename: {filename}")
print(f"Stem: {tile_path.stem}")

parts = filename.split('_')
print(f"Parts: {parts}")
print(f"Length: {len(parts)}")

# Check pattern matching
if len(parts) >= 9 and parts[0] == 'LHD' and parts[1].startswith('F'):
    print("✓ Pattern matched!")
    print(f"  parts[0] = {parts[0]} (expect 'LHD')")
    print(f"  parts[1] = {parts[1]} (expect starts with 'F')")
    print(f"  parts[2] = {parts[2]} (X coord)")
    print(f"  parts[3] = {parts[3]} (Y coord)")
    print(f"  parts[4] = {parts[4]} (expect 'PTS')")
    print(f"  parts[5] = {parts[5]} (C/O flag)")
    
    x_coord = int(parts[2])
    y_coord = int(parts[3])
    dept_code = parts[1]
    class_flag = parts[5]
    
    print(f"\nParsed coordinates: X={x_coord}, Y={y_coord}")
    print(f"Department: {dept_code}, Classification: {class_flag}")
    
    # Test neighbor generation
    print("\nTesting neighbor file names:")
    neighbor_coords = [
        (x_coord, y_coord + 1, "North"),
        (x_coord, y_coord - 1, "South"),
        (x_coord + 1, y_coord, "East"),
        (x_coord - 1, y_coord, "West"),
    ]
    
    tile_dir = tile_path.parent
    for nx, ny, direction in neighbor_coords:
        neighbor_name = f"LHD_{dept_code}_{nx:04d}_{ny:04d}_PTS_{class_flag}_LAMB93_IGN69.laz"
        neighbor_path = tile_dir / neighbor_name
        exists = neighbor_path.exists()
        print(f"  {direction:6s}: {neighbor_name} {'✓ EXISTS' if exists else '✗ NOT FOUND'}")
else:
    print("✗ Pattern did not match!")
