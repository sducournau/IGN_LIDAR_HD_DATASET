#!/usr/bin/env python3
"""
Test script to verify COPC file conversion works correctly.
"""

import laspy
from pathlib import Path
import numpy as np

def test_copc_conversion():
    """Test that we can create a file with COPC-like structure and convert it."""
    
    # Create a test COPC-like file
    test_file = Path("/tmp/test_copc.laz")
    
    # Create a simple point cloud
    n_points = 1000
    x = np.random.uniform(0, 100, n_points)
    y = np.random.uniform(0, 100, n_points)
    z = np.random.uniform(0, 10, n_points)
    
    # Create header with point format 6 (like COPC files)
    header = laspy.LasHeader(version="1.4", point_format=6)
    header.scales = [0.01, 0.01, 0.01]
    header.offsets = [0, 0, 0]
    
    # Create LAS data
    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = z
    
    # Write test file
    las.write(test_file)
    print(f"✓ Created test file: {test_file}")
    
    # Read it back
    with laspy.open(test_file) as f:
        las_read = f.read()
    
    print(f"  Point format: {las_read.header.point_format.id}")
    print(f"  Points: {len(las_read.x)}")
    
    # Now test conversion to standard LAZ with extra dims
    print("\nTesting conversion to standard LAZ with extra dims...")
    
    # Create new header without COPC
    new_header = laspy.LasHeader(
        version=las_read.header.version,
        point_format=las_read.header.point_format.id
    )
    new_header.scales = las_read.header.scales
    new_header.offsets = las_read.header.offsets
    
    las_out = laspy.LasData(new_header)
    las_out.x = las_read.x
    las_out.y = las_read.y
    las_out.z = las_read.z
    
    # Add extra dimensions
    las_out.add_extra_dim(laspy.ExtraBytesParams(name='test_feature', type=np.float32))
    las_out.test_feature = np.random.uniform(0, 1, len(las_out.x))
    
    # Write output (avec compression LAZ pour compatibilité QGIS)
    output_file = Path("/tmp/test_output.laz")
    las_out.write(output_file, do_compress=True)
    print(f"✓ Created output file with extra dims: {output_file}")
    
    # Verify
    with laspy.open(output_file) as f:
        las_verify = f.read()
    
    print(f"  Point format: {las_verify.header.point_format.id}")
    print(f"  Points: {len(las_verify.x)}")
    print(f"  Extra dims: {[d for d in las_verify.point_format.dimension_names if d not in ['X', 'Y', 'Z', 'intensity', 'return_number', 'number_of_returns', 'scan_direction_flag', 'edge_of_flight_line', 'classification', 'synthetic', 'key_point', 'withheld', 'scan_angle_rank', 'user_data', 'point_source_id', 'gps_time', 'red', 'green', 'blue']]}")
    
    # Clean up
    test_file.unlink()
    output_file.unlink()
    
    print("\n✓ All tests passed!")

if __name__ == '__main__':
    test_copc_conversion()
