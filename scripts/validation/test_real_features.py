#!/usr/bin/env python3
"""
Test geometric features on real LIDAR data

Validates that the formulas work correctly on actual building point clouds.
"""

import numpy as np
import laspy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ign_lidar.features import extract_geometric_features


def test_real_lidar_file(laz_file: Path):
    """Test geometric features on a real LAZ file."""
    
    print("\n" + "="*70)
    print(f"TESTING GEOMETRIC FEATURES ON REAL LIDAR DATA")
    print("="*70 + "\n")
    
    print(f"File: {laz_file.name}\n")
    
    # Read LAZ file
    with laspy.open(laz_file) as f:
        las = f.read()
    
    points = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
    classification = np.array(las.classification, dtype=np.uint8)
    
    print(f"Total points: {len(points):,}\n")
    
    # Get building points (class 6)
    building_mask = classification == 6
    building_points = points[building_mask]
    
    if len(building_points) == 0:
        print("⚠️  No building points found (class 6)")
        return
    
    print(f"Building points: {len(building_points):,}\n")
    
    # Compute features
    print("Computing geometric features...")
    normals = np.zeros_like(building_points)  # Dummy
    features = extract_geometric_features(building_points, normals, k=20)
    
    print("\n" + "-"*70)
    print("FEATURE STATISTICS")
    print("-"*70 + "\n")
    
    # Check sum property
    sum_features = (features['linearity'] + 
                   features['planarity'] + 
                   features['sphericity'])
    
    print("Sum of Linearity + Planarity + Sphericity:")
    print(f"  Mean: {np.mean(sum_features):.6f}  (should be ~1.0)")
    print(f"  Min:  {np.min(sum_features):.6f}")
    print(f"  Max:  {np.max(sum_features):.6f}")
    print()
    
    # Statistics for each feature
    for name in ['linearity', 'planarity', 'sphericity', 
                 'anisotropy', 'roughness']:
        values = features[name]
        print(f"{name.capitalize():12s}:")
        print(f"  Mean: {np.mean(values):.4f}")
        print(f"  Std:  {np.std(values):.4f}")
        print(f"  Min:  {np.min(values):.4f}")
        print(f"  Max:  {np.max(values):.4f}")
        
        # Show percentiles for interesting features
        if name in ['linearity', 'planarity']:
            p90 = np.percentile(values, 90)
            print(f"  90th percentile: {p90:.4f}")
        print()
    
    # Analyze dominant feature types
    print("-"*70)
    print("DOMINANT FEATURE TYPES")
    print("-"*70 + "\n")
    
    lin = features['linearity']
    pla = features['planarity']
    sph = features['sphericity']
    
    # Classify points by dominant feature
    linear_mask = (lin > pla) & (lin > sph)
    planar_mask = (pla > lin) & (pla > sph)
    spherical_mask = (sph > lin) & (sph > pla)
    
    n_linear = np.sum(linear_mask)
    n_planar = np.sum(planar_mask)
    n_spherical = np.sum(spherical_mask)
    
    total = len(building_points)
    
    print(f"Linear-dominant (edges):     {n_linear:6d} "
          f"({100*n_linear/total:5.1f}%)")
    print(f"Planar-dominant (surfaces):  {n_planar:6d} "
          f"({100*n_planar/total:5.1f}%)")
    print(f"Spherical-dominant (noise):  {n_spherical:6d} "
          f"({100*n_spherical/total:5.1f}%)")
    print()
    
    # For buildings, we expect mostly planar points (roofs, walls)
    print("✓ For buildings, planar points should dominate")
    print(f"  Expected: >60% planar")
    print(f"  Actual:   {100*n_planar/total:.1f}% planar")
    
    if n_planar / total > 0.6:
        print("  ✅ GOOD - Building geometry detected correctly\n")
    elif n_planar / total > 0.4:
        print("  ⚠️  MARGINAL - Some planar features detected\n")
    else:
        print("  ❌ POOR - Too few planar features\n")
    
    # High planarity points (roofs/walls)
    high_planar = np.sum(pla > 0.7)
    print(f"High planarity points (>0.7): {high_planar:,} "
          f"({100*high_planar/total:.1f}%)")
    
    # High linearity points (edges)
    high_linear = np.sum(lin > 0.7)
    print(f"High linearity points (>0.7): {high_linear:,} "
          f"({100*high_linear/total:.1f}%)")
    print()


if __name__ == "__main__":
    # Find a LAZ file to test
    if len(sys.argv) > 1:
        laz_file = Path(sys.argv[1])
    else:
        # Try to find a file in raw_tiles
        raw_tiles = Path("/mnt/c/Users/Simon/ign/raw_tiles")
        if raw_tiles.exists():
            laz_files = list(raw_tiles.rglob("*.laz"))
            if laz_files:
                laz_file = laz_files[0]
                print(f"Using: {laz_file}")
            else:
                print("No LAZ files found in /mnt/c/Users/Simon/ign/raw_tiles")
                sys.exit(1)
        else:
            print("Usage: python test_real_features.py <laz_file>")
            sys.exit(1)
    
    if not laz_file.exists():
        print(f"Error: File not found: {laz_file}")
        sys.exit(1)
    
    test_real_lidar_file(laz_file)
