#!/usr/bin/env python3
"""
Check features in enriched LAZ file (V3 - Extended ASPRS features)
"""
import laspy
import sys
from pathlib import Path

def check_laz_features(laz_path):
    """
    Analyzes LAZ file dimensions and prints feature summary
    """
    print(f"📂 Analyzing: {laz_path}")
    print("=" * 80)
    
    with laspy.open(laz_path) as laz:
        las = laz.read()
        
        print(f"📊 Point count: {len(las.points):,}")
        print(f"📐 Point format: {las.point_format}")
        print()
        
        # Standard LAS dimensions
        print("🔷 Standard LAS dimensions:")
        standard_dims = ['X', 'Y', 'Z', 'intensity', 'return_number', 'classification', 
                         'red', 'green', 'blue', 'nir']
        for dim in standard_dims:
            if dim in las.point_format.dimension_names:
                print(f"  ✅ {dim}")
        print()
        
        # Extra dimensions (features)
        extra_dims = las.point_format.extra_dimension_names
        print(f"✨ Extra dimensions (features): {len(extra_dims)}")
        
        if not extra_dims:
            print("  ❌ NO EXTRA FEATURES FOUND!")
            return
        
        # Expected V3 features (19 total)
        expected_features = {
            'normal_x', 'normal_y', 'normal_z',  # Full normals
            'curvature',                          # NEW in V3
            'planarity', 'sphericity', 'verticality', 'horizontality',
            'height', 'height_above_ground',      # Both heights
            'density',
            'ndvi',
            # Extra dims from config (if implemented)
            'BuildingConfidence', 'IsWall', 'IsRoof', 
            'DistanceToPolygon', 'AdaptiveExpanded', 'IntelligentRejected'
        }
        
        # Sort features into categories
        geometric_features = []
        height_features = []
        spectral_features = []
        building_features = []
        other_features = []
        
        for dim in extra_dims:
            if 'normal' in dim.lower() or dim in {'curvature', 'planarity', 'sphericity', 
                                                    'verticality', 'horizontality', 'density'}:
                geometric_features.append(dim)
            elif 'height' in dim.lower():
                height_features.append(dim)
            elif dim in {'ndvi', 'red', 'green', 'blue', 'nir'}:
                spectral_features.append(dim)
            elif 'building' in dim.lower() or 'wall' in dim.lower() or 'roof' in dim.lower():
                building_features.append(dim)
            else:
                other_features.append(dim)
        
        # Print categorized features
        print()
        if geometric_features:
            print(f"  🔷 Geometric features ({len(geometric_features)}):")
            for feat in sorted(geometric_features):
                status = "✅ V3" if feat in {'normal_x', 'normal_y', 'curvature'} else "✅"
                print(f"    {status} {feat}")
        
        if height_features:
            print(f"\n  📏 Height features ({len(height_features)}):")
            for feat in sorted(height_features):
                print(f"    ✅ {feat}")
        
        if spectral_features:
            print(f"\n  🌈 Spectral features ({len(spectral_features)}):")
            for feat in sorted(spectral_features):
                print(f"    ✅ {feat}")
        
        if building_features:
            print(f"\n  🏗️  Building features ({len(building_features)}):")
            for feat in sorted(building_features):
                print(f"    ✅ {feat}")
        
        if other_features:
            print(f"\n  ⚡ Other features ({len(other_features)}):")
            for feat in sorted(other_features):
                print(f"    ✅ {feat}")
        
        # Check for missing expected features
        print()
        missing = expected_features - set(extra_dims)
        if missing:
            print(f"⚠️  Missing expected features ({len(missing)}):")
            for feat in sorted(missing):
                print(f"    ❌ {feat}")
        else:
            print("✅ All expected V3 features present!")
        
        print()
        print("=" * 80)
        print(f"📊 SUMMARY: {len(extra_dims)} extra features saved")
        print(f"   Expected: {len(expected_features)} features")
        print(f"   Success rate: {len(extra_dims)/len(expected_features)*100:.1f}%")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        laz_path = sys.argv[1]
    else:
        # Default: find enriched LAZ in V3 output
        v3_dir = Path("/mnt/d/ign/versailles_output_v3")
        enriched_files = list(v3_dir.glob("*enriched.laz"))
        if not enriched_files:
            print("❌ No enriched LAZ file found in V3 output!")
            print(f"   Searched: {v3_dir}")
            sys.exit(1)
        laz_path = enriched_files[0]
    
    check_laz_features(laz_path)
