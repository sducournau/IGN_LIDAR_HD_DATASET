#!/usr/bin/env python3
"""
Quick test to verify enriched LAZ writing with all features (RGB, NIR, NDVI, geometric features).
"""
import numpy as np
import laspy
from pathlib import Path

def test_laz_writing_with_rgb():
    """Test that we can create a LAZ file with RGB and extra dimensions."""
    print("=" * 70)
    print("Testing LAZ Writing with RGB + Extra Dimensions")
    print("=" * 70)
    
    # Create dummy data
    n_points = 1000
    points = np.random.rand(n_points, 3).astype(np.float32) * 100
    intensity = np.random.rand(n_points).astype(np.float32)
    classification = np.random.randint(1, 7, n_points, dtype=np.uint8)
    rgb = np.random.rand(n_points, 3).astype(np.float32)  # 0-1 range
    
    # Simulate computed features
    normals = np.random.rand(n_points, 3).astype(np.float32)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)  # Normalize
    curvature = np.random.rand(n_points).astype(np.float32)
    height = np.random.rand(n_points).astype(np.float32) * 50
    planarity = np.random.rand(n_points).astype(np.float32)
    linearity = np.random.rand(n_points).astype(np.float32)
    sphericity = np.random.rand(n_points).astype(np.float32)
    verticality = np.random.rand(n_points).astype(np.float32)
    nir = np.random.rand(n_points).astype(np.float32)
    
    # Compute NDVI
    red = rgb[:, 0]
    ndvi = (nir - red) / (nir + red + 1e-8)
    
    print(f"\n📊 Test data created: {n_points} points")
    print(f"   RGB range: [{rgb.min():.2f}, {rgb.max():.2f}]")
    print(f"   NIR range: [{nir.min():.2f}, {nir.max():.2f}]")
    print(f"   NDVI range: [{ndvi.min():.2f}, {ndvi.max():.2f}]")
    
    # Test different scenarios
    scenarios = [
        ("Point Format 6 → 7 (with RGB)", 6),
        ("Point Format 0 → 2 (with RGB)", 0),
    ]
    
    for scenario_name, original_format in scenarios:
        print(f"\n{'='*70}")
        print(f"Scenario: {scenario_name}")
        print(f"{'='*70}")
        
        try:
            # Determine target format
            if rgb is not None:
                if original_format in [0, 1]:
                    target_format = 2
                elif original_format in [6]:
                    target_format = 7
                elif original_format in [2, 3, 5, 7, 8, 10]:
                    target_format = original_format
                else:
                    target_format = 7
            else:
                target_format = original_format
            
            print(f"   Original format: {original_format} → Target format: {target_format}")
            
            # Create header
            header = laspy.LasHeader(version="1.4", point_format=target_format)
            header.scales = [0.01, 0.01, 0.01]
            header.offsets = [0, 0, 0]
            
            # Create LAS data
            las = laspy.LasData(header)
            las.x = points[:, 0]
            las.y = points[:, 1]
            las.z = points[:, 2]
            las.intensity = (intensity * 65535.0).astype(np.uint16)
            las.classification = classification
            
            # Add RGB first
            las.red = (rgb[:, 0] * 65535.0).astype(np.uint16)
            las.green = (rgb[:, 1] * 65535.0).astype(np.uint16)
            las.blue = (rgb[:, 2] * 65535.0).astype(np.uint16)
            print(f"   ✓ RGB added to point format {target_format}")
            
            # Add extra dimensions
            las.add_extra_dim(laspy.ExtraBytesParams(name="normal_x", type=np.float32))
            las.add_extra_dim(laspy.ExtraBytesParams(name="normal_y", type=np.float32))
            las.add_extra_dim(laspy.ExtraBytesParams(name="normal_z", type=np.float32))
            las.add_extra_dim(laspy.ExtraBytesParams(name="curvature", type=np.float32))
            las.add_extra_dim(laspy.ExtraBytesParams(name="height", type=np.float32))
            las.add_extra_dim(laspy.ExtraBytesParams(name="planarity", type=np.float32))
            las.add_extra_dim(laspy.ExtraBytesParams(name="linearity", type=np.float32))
            las.add_extra_dim(laspy.ExtraBytesParams(name="sphericity", type=np.float32))
            las.add_extra_dim(laspy.ExtraBytesParams(name="verticality", type=np.float32))
            las.add_extra_dim(laspy.ExtraBytesParams(name="nir", type=np.float32))
            las.add_extra_dim(laspy.ExtraBytesParams(name="ndvi", type=np.float32))
            
            las.normal_x = normals[:, 0]
            las.normal_y = normals[:, 1]
            las.normal_z = normals[:, 2]
            las.curvature = curvature
            las.height = height
            las.planarity = planarity
            las.linearity = linearity
            las.sphericity = sphericity
            las.verticality = verticality
            las.nir = nir
            las.ndvi = ndvi
            
            print(f"   ✓ Added 11 extra dimensions (geometric + NIR + NDVI)")
            
            # Write to file
            output_path = Path(f"/tmp/test_enriched_format_{target_format}.laz")
            las.write(output_path)
            print(f"   ✓ Written to: {output_path}")
            
            # Verify by reading back
            las_read = laspy.read(output_path)
            print(f"\n   📖 Verification:")
            print(f"      Points: {len(las_read.points)}")
            print(f"      RGB: red={las_read.red.min()}-{las_read.red.max()}, "
                  f"green={las_read.green.min()}-{las_read.green.max()}, "
                  f"blue={las_read.blue.min()}-{las_read.blue.max()}")
            print(f"      Extra dims: {las_read.point_format.extra_dimension_names}")
            print(f"      Normal X range: [{las_read.normal_x.min():.3f}, {las_read.normal_x.max():.3f}]")
            print(f"      NIR range: [{las_read.nir.min():.3f}, {las_read.nir.max():.3f}]")
            print(f"      NDVI range: [{las_read.ndvi.min():.3f}, {las_read.ndvi.max():.3f}]")
            print(f"   ✅ SUCCESS")
            
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("All tests completed!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    test_laz_writing_with_rgb()
