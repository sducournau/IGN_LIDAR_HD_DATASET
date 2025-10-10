#!/usr/bin/env python3
"""
Verify that LAZ patches contain all computed features.

This script checks that saved LAZ patches include:
- Geometric features (planarity, linearity, sphericity, etc.)
- Normals (nx, ny, nz)
- Height features
- Radiometric features (NDVI if available)
"""

import sys
from pathlib import Path
import numpy as np

try:
    import laspy
except ImportError:
    print("ERROR: laspy not installed. Run: pip install laspy")
    sys.exit(1)


def verify_laz_features(laz_path: Path) -> dict:
    """
    Verify that a LAZ patch contains all expected computed features.
    
    Args:
        laz_path: Path to LAZ patch file
        
    Returns:
        Dictionary with verification results
    """
    print(f"\n{'='*80}")
    print(f"Verifying LAZ features: {laz_path.name}")
    print(f"{'='*80}\n")
    
    # Read LAZ file
    try:
        las = laspy.read(str(laz_path))
    except Exception as e:
        print(f"❌ ERROR reading LAZ file: {e}")
        return {'success': False, 'error': str(e)}
    
    # Basic info
    print(f"✓ LAZ file loaded successfully")
    print(f"  - Points: {len(las.points):,}")
    print(f"  - Point format: {las.point_format.id}")
    print(f"  - Version: {las.header.version}")
    
    # Check standard fields
    print(f"\n📊 Standard LAS Fields:")
    standard_fields = ['x', 'y', 'z', 'intensity', 'classification', 
                       'red', 'green', 'blue', 'nir', 'return_number']
    for field in standard_fields:
        if hasattr(las, field):
            values = getattr(las, field)
            print(f"  ✓ {field:20s} - {len(values):,} values")
        else:
            print(f"  ⚠ {field:20s} - Not present")
    
    # Check geometric features
    print(f"\n🔧 Geometric Features:")
    geometric_features = ['planarity', 'linearity', 'sphericity', 'anisotropy',
                         'roughness', 'density', 'curvature', 'verticality']
    
    missing_geometric = []
    for feat in geometric_features:
        if hasattr(las, feat):
            values = getattr(las, feat)
            print(f"  ✓ {feat:20s} - min: {values.min():.4f}, max: {values.max():.4f}, mean: {values.mean():.4f}")
        else:
            print(f"  ❌ {feat:20s} - MISSING")
            missing_geometric.append(feat)
    
    # Check normals
    print(f"\n📐 Normals:")
    normals_present = []
    for comp in ['nx', 'ny', 'nz']:
        if hasattr(las, comp):
            values = getattr(las, comp)
            normals_present.append(comp)
            print(f"  ✓ {comp:20s} - min: {values.min():.4f}, max: {values.max():.4f}")
        else:
            print(f"  ❌ {comp:20s} - MISSING")
    
    # Check height features
    print(f"\n📏 Height Features:")
    height_features = ['height', 'z_normalized', 'z_from_ground', 'z_from_median']
    missing_height = []
    for feat in height_features:
        if hasattr(las, feat):
            values = getattr(las, feat)
            print(f"  ✓ {feat:20s} - min: {values.min():.4f}, max: {values.max():.4f}")
        else:
            print(f"  ⚠ {feat:20s} - Not present (optional)")
            missing_height.append(feat)
    
    # Check radiometric features
    print(f"\n🌿 Radiometric Features:")
    if hasattr(las, 'ndvi'):
        ndvi = las.ndvi
        print(f"  ✓ {'ndvi':20s} - min: {ndvi.min():.4f}, max: {ndvi.max():.4f}")
    else:
        print(f"  ⚠ {'ndvi':20s} - Not present (optional, requires NIR)")
    
    # List all available dimensions
    print(f"\n📋 All Available Dimensions:")
    all_dims = las.point_format.dimension_names
    print(f"  Total: {len(all_dims)} dimensions")
    for dim in sorted(all_dims):
        print(f"    - {dim}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY:")
    print(f"{'='*80}")
    
    success = len(missing_geometric) == 0 and len(normals_present) == 3
    
    if success:
        print(f"✅ SUCCESS: All critical features present!")
        print(f"   - {len(geometric_features)} geometric features")
        print(f"   - 3 normal components")
        if missing_height:
            print(f"   ⚠ {len(missing_height)} optional height features missing")
    else:
        print(f"❌ FAILED: Missing critical features!")
        if missing_geometric:
            print(f"   - Missing geometric: {', '.join(missing_geometric)}")
        if len(normals_present) < 3:
            print(f"   - Incomplete normals: only {normals_present}")
    
    return {
        'success': success,
        'points': len(las.points),
        'geometric_features': len(geometric_features) - len(missing_geometric),
        'normals': len(normals_present),
        'missing_geometric': missing_geometric,
        'all_dimensions': all_dims
    }


def main():
    """Main verification routine."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Verify LAZ patches contain all computed features'
    )
    parser.add_argument(
        'laz_files',
        nargs='+',
        type=Path,
        help='LAZ patch file(s) to verify'
    )
    
    args = parser.parse_args()
    
    results = []
    for laz_path in args.laz_files:
        if not laz_path.exists():
            print(f"❌ File not found: {laz_path}")
            continue
        
        result = verify_laz_features(laz_path)
        results.append((laz_path.name, result))
    
    # Overall summary
    if len(results) > 1:
        print(f"\n\n{'='*80}")
        print(f"OVERALL SUMMARY ({len(results)} files)")
        print(f"{'='*80}")
        
        successful = sum(1 for _, r in results if r.get('success', False))
        print(f"✓ Successful: {successful}/{len(results)}")
        print(f"❌ Failed: {len(results) - successful}/{len(results)}")
        
        if successful == len(results):
            print(f"\n🎉 All LAZ patches contain complete features!")
            return 0
        else:
            print(f"\n⚠️  Some LAZ patches are missing features")
            return 1
    
    return 0 if results and results[0][1].get('success', False) else 1


if __name__ == '__main__':
    sys.exit(main())
