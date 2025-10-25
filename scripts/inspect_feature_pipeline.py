#!/usr/bin/env python3
"""
Quick Feature Pipeline Inspector

Shows what happens to features at each stage:
1. Feature mode definition
2. Computed features
3. Filtered features  
4. Saved features

Usage:
    python scripts/inspect_feature_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.features.feature_modes import (
    ASPRS_FEATURES, 
    LOD2_FEATURES, 
    LOD3_FEATURES,
    FeatureMode
)

def print_header(title):
    """Print section header."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")

def print_feature_set(name, features):
    """Print a set of features."""
    print(f"\n{name}: {len(features)} features")
    print("-" * 80)
    
    # Group by category
    geometric = []
    height = []
    spectral = []
    density = []
    other = []
    
    for feat in sorted(features):
        if feat in ['normal_x', 'normal_y', 'normal_z', 'normals', 'planarity', 'sphericity', 
                    'curvature', 'verticality', 'horizontality', 'linearity', 'anisotropy',
                    'omnivariance', 'eigenentropy', 'sum_eigenvalues', 'change_curvature']:
            geometric.append(feat)
        elif feat in ['height', 'height_above_ground', 'z_normalized', 'z_from_ground', 'z_from_median']:
            height.append(feat)
        elif feat in ['red', 'green', 'blue', 'rgb', 'nir', 'ndvi']:
            spectral.append(feat)
        elif feat in ['density', 'num_points_2m', 'neighborhood_extent', 'height_extent_ratio']:
            density.append(feat)
        else:
            other.append(feat)
    
    if geometric:
        print(f"\n  Geometric ({len(geometric)}):")
        for feat in geometric:
            print(f"    • {feat}")
    
    if height:
        print(f"\n  Height ({len(height)}):")
        for feat in height:
            print(f"    • {feat}")
    
    if spectral:
        print(f"\n  Spectral ({len(spectral)}):")
        for feat in spectral:
            print(f"    • {feat}")
    
    if density:
        print(f"\n  Density ({len(density)}):")
        for feat in density:
            print(f"    • {feat}")
    
    if other:
        print(f"\n  Other ({len(other)}):")
        for feat in other:
            print(f"    • {feat}")

def compare_modes():
    """Compare feature sets across modes."""
    print_header("FEATURE MODE COMPARISON")
    
    # Remove 'xyz' as it's coordinates, not a feature array
    asprs = ASPRS_FEATURES - {'xyz'}
    lod2 = LOD2_FEATURES - {'xyz'}
    lod3 = LOD3_FEATURES - {'xyz'}
    
    print_feature_set("ASPRS_CLASSES Mode", asprs)
    print_feature_set("LOD2_SIMPLIFIED Mode", lod2)
    print_feature_set("LOD3_FULL Mode", lod3)
    
    # Find differences
    print_header("MODE DIFFERENCES")
    
    # ASPRS vs LOD2
    asprs_only = asprs - lod2
    lod2_only = lod2 - asprs
    
    if asprs_only:
        print(f"\n📊 ASPRS has, LOD2 doesn't ({len(asprs_only)}):")
        for feat in sorted(asprs_only):
            print(f"   • {feat}")
    
    if lod2_only:
        print(f"\n📊 LOD2 has, ASPRS doesn't ({len(lod2_only)}):")
        for feat in sorted(lod2_only):
            print(f"   • {feat}")
    
    # LOD2 vs LOD3
    lod3_extra = lod3 - lod2
    if lod3_extra:
        print(f"\n📊 LOD3 additional features ({len(lod3_extra)}):")
        for feat in sorted(lod3_extra):
            print(f"   • {feat}")
    
    # Common to all
    common = asprs & lod2 & lod3
    print(f"\n📊 Common to all modes ({len(common)}):")
    for feat in sorted(common):
        print(f"   • {feat}")

def check_implementation():
    """Check which compute modules provide which features."""
    print_header("FEATURE IMPLEMENTATION MAPPING")
    
    print("""
📦 Feature Computation Modules:

1. ign_lidar/features/compute/geometric.py
   Provides: normals, curvature, planarity, sphericity, verticality,
             horizontality, linearity, anisotropy, omnivariance,
             eigenentropy, sum_eigenvalues, change_curvature

2. ign_lidar/features/compute/density.py
   Provides: density, num_points_2m, neighborhood_extent,
             height_extent_ratio

3. ign_lidar/features/compute/architectural.py
   Provides: wall_score, roof_score, facade_score, flat_roof_score,
             sloped_roof_score, steep_roof_score, opening_likelihood,
             edge_strength, corner_likelihood, overhang_indicator,
             surface_roughness

4. ign_lidar/features/orchestrator.py (spectral)
   Provides: red, green, blue, nir, ndvi

5. Height computation (in orchestrator)
   Provides: height, height_above_ground, z_normalized,
             z_from_ground, z_from_median
""")

def check_save_compatibility():
    """Check which features can be saved to LAZ."""
    print_header("LAZ SAVE COMPATIBILITY")
    
    asprs = ASPRS_FEATURES - {'xyz'}
    
    print("""
✅ Save Compatible (1D arrays):
   All ASPRS features except 'normals'
   
⚙️  Special Handling:
   • normals - Split into normal_x, normal_y, normal_z
   • rgb - Saved as standard LAZ field (red, green, blue)
   • nir - Saved as standard LAZ field
   
❌ Not Saved (Multi-dimensional):
   • Any 2D or 3D arrays (except normals)
   • Matrix features
   • Eigenvalue arrays (unless flattened)

📋 ASPRS Features Save Status:
""")
    
    for feat in sorted(asprs):
        if feat == 'normals':
            print(f"   ⚙️  {feat:30s} → normal_x, normal_y, normal_z")
        elif feat in ['rgb', 'red', 'green', 'blue']:
            print(f"   📦 {feat:30s} → Standard LAZ RGB field")
        elif feat == 'nir':
            print(f"   📦 {feat:30s} → Standard LAZ NIR field")
        elif feat == 'xyz':
            print(f"   📦 {feat:30s} → Coordinates (X, Y, Z)")
        else:
            print(f"   ✅ {feat:30s} → Extra dimension")

def suggest_config():
    """Suggest optimal configuration."""
    print_header("RECOMMENDED ASPRS CONFIGURATION")
    
    print("""
features:
  # Mode selection
  mode: asprs_classes
  
  # Neighborhood parameters
  k_neighbors: 20              # 20-50 recommended for ASPRS
  search_radius: 1.0           # 1.0m typical for urban areas
  
  # Core computations (REQUIRED)
  compute_normals: true        # → normal_x, normal_y, normal_z
  compute_curvature: true      # → curvature
  compute_height: true         # → height, height_above_ground
  compute_geometric: true      # → planarity, sphericity, verticality, horizontality
  
  # Density computation (RECOMMENDED)
  compute_density: true        # → density
  
  # Spectral features (OPTIONAL - requires RGB + NIR)
  compute_ndvi: true           # → ndvi (if RGB+NIR available)
  
  # Architectural features (NOT in ASPRS mode - use LOD3 for these)
  compute_architectural: false

output:
  formats:
    - laz                      # Enriched LAZ with features
  
  # Extra dimensions to save
  extra_dims:
    - height
    - planarity
    - verticality
    - curvature
    - ndvi                     # If available
""")

def main():
    """Run all inspections."""
    print("="*80)
    print("IGN LiDAR HD - Feature Pipeline Inspector")
    print("="*80)
    
    compare_modes()
    check_implementation()
    check_save_compatibility()
    suggest_config()
    
    print_header("NEXT STEPS")
    print("""
1. Verify your config matches the recommended settings above
   
2. Run the debug script to test your pipeline:
   python scripts/debug_asprs_features.py [your_file.laz]
   
3. Check output with:
   python scripts/check_laz_features_v3.py output_enriched.laz
   
4. Review detailed analysis in:
   ASPRS_FEATURE_ANALYSIS.md
   DEBUG_ASPRS_FEATURES_QUICKSTART.md
""")
    
    print("="*80)

if __name__ == '__main__':
    main()
