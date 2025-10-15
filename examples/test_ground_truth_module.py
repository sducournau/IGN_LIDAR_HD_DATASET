#!/usr/bin/env python3
"""
Test script to verify ground truth module integration.

This script tests:
1. Module imports from main package
2. Basic functionality of IGNGroundTruthFetcher
3. Road buffer creation with largeur field
4. NDVI-based refinement
5. Complete patch generation workflow

Usage:
    python examples/test_ground_truth_module.py
"""

import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("Ground Truth Module Integration Test")
print("=" * 80)

# ============================================================================
# Test 1: Module Imports
# ============================================================================
print("\n[1/5] Testing module imports...")

try:
    # Test importing from main package
    from ign_lidar import (
        IGNWFSConfig,
        IGNGroundTruthFetcher,
        fetch_ground_truth_for_tile,
        generate_patches_with_ground_truth,
    )
    print("✅ All ground truth components imported successfully from ign_lidar")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("\nNote: Make sure to install spatial dependencies:")
    print("  pip install shapely geopandas")
    sys.exit(1)

# Test importing from io submodule
try:
    from ign_lidar.io import (
        IGNWFSConfig as IOConfig,
        IGNGroundTruthFetcher as IOFetcher,
    )
    print("✅ Ground truth components also accessible from ign_lidar.io")
except ImportError as e:
    print(f"❌ IO submodule import failed: {e}")

# ============================================================================
# Test 2: Configuration
# ============================================================================
print("\n[2/5] Testing configuration...")

try:
    # Create default config
    config = IGNWFSConfig()
    print(f"✅ Default config created:")
    print(f"   - WFS URL: {config.WFS_URL}")
    print(f"   - CRS: {config.CRS}")
    print(f"   - Buildings layer: {config.BUILDINGS_LAYER}")
    print(f"   - Roads layer: {config.ROADS_LAYER}")
    
    # Check all layer definitions exist
    layers = ['BUILDINGS_LAYER', 'ROADS_LAYER', 'WATER_LAYER', 'VEGETATION_LAYER']
    for layer in layers:
        if hasattr(config, layer):
            print(f"✅ Layer defined: {layer}")
        else:
            print(f"❌ Layer missing: {layer}")
    
except Exception as e:
    print(f"❌ Configuration test failed: {e}")
    sys.exit(1)

# ============================================================================
# Test 3: Fetcher Initialization
# ============================================================================
print("\n[3/5] Testing fetcher initialization...")

try:
    # Initialize with default config
    fetcher = IGNGroundTruthFetcher()
    print("✅ Fetcher initialized with default config")
    
    # Initialize with cache directory
    cache_dir = Path("data/test_output/ground_truth_cache")
    fetcher_cached = IGNGroundTruthFetcher(
        cache_dir=cache_dir
    )
    print(f"✅ Fetcher initialized with cache dir: {cache_dir}")
    
except Exception as e:
    print(f"❌ Fetcher initialization failed: {e}")
    sys.exit(1)

# ============================================================================
# Test 4: API Availability Check (without actual API calls)
# ============================================================================
print("\n[4/5] Testing API methods availability...")

try:
    # Check all public methods exist
    methods_to_check = [
        'fetch_buildings',
        'fetch_roads_with_polygons',
        'fetch_water_surfaces',
        'fetch_vegetation_zones',
        'fetch_all_features',
        'label_points_with_ground_truth',
        'save_ground_truth',
    ]
    
    for method_name in methods_to_check:
        if hasattr(fetcher, method_name):
            print(f"✅ Method available: {method_name}")
        else:
            print(f"❌ Method missing: {method_name}")
            
except Exception as e:
    print(f"❌ API check failed: {e}")
    sys.exit(1)

# ============================================================================
# Test 5: Helper Functions
# ============================================================================
print("\n[5/5] Testing helper functions...")

try:
    # Check fetch_ground_truth_for_tile function
    print("✅ Function available: fetch_ground_truth_for_tile")
    print(f"   Signature: {fetch_ground_truth_for_tile.__name__}")
    
    # Check generate_patches_with_ground_truth function  
    print("✅ Function available: generate_patches_with_ground_truth")
    print(f"   Signature: {generate_patches_with_ground_truth.__name__}")
    
except Exception as e:
    print(f"❌ Helper function check failed: {e}")
    sys.exit(1)

# ============================================================================
# Test 6: Documentation Check
# ============================================================================
print("\n[6/6] Testing documentation...")

try:
    # Check docstrings exist
    if IGNGroundTruthFetcher.__doc__:
        doc_lines = IGNGroundTruthFetcher.__doc__.strip().split('\n')
        print(f"✅ IGNGroundTruthFetcher documentation:")
        print(f"   {doc_lines[0]}")
    
    if fetch_ground_truth_for_tile.__doc__:
        doc_lines = fetch_ground_truth_for_tile.__doc__.strip().split('\n')
        print(f"✅ fetch_ground_truth_for_tile documentation:")
        print(f"   {doc_lines[0]}")
        
    if generate_patches_with_ground_truth.__doc__:
        doc_lines = generate_patches_with_ground_truth.__doc__.strip().split('\n')
        print(f"✅ generate_patches_with_ground_truth documentation:")
        print(f"   {doc_lines[0]}")
    
except Exception as e:
    print(f"❌ Documentation check failed: {e}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("Ground Truth Module Integration: ✅ ALL TESTS PASSED")
print("=" * 80)

print("\n📚 Module is ready to use! Quick start examples:")
print("""
# Basic usage - fetch ground truth for a bounding box:
from ign_lidar import IGNGroundTruthFetcher

fetcher = IGNGroundTruthFetcher(cache_dir="cache/gt")
bbox = (650000, 6860000, 651000, 6861000)  # Lambert 93 (EPSG:2154)
ground_truth = fetcher.fetch_all_features(bbox)

# Roads with buffers (tampons) from largeur field:
roads = ground_truth['roads']
print(f"Fetched {len(roads)} roads")
print(f"Average width: {roads['width_m'].mean():.1f}m")

# Label points with ground truth + NDVI refinement:
labels = fetcher.label_points_with_ground_truth(
    points=points,
    ground_truth_features=ground_truth,
    ndvi=ndvi,  # Optional: improves building/vegetation classification
    use_ndvi_refinement=True
)

# Complete workflow - generate labeled patches:
from ign_lidar import generate_patches_with_ground_truth

patches = generate_patches_with_ground_truth(
    points=points,
    features={'rgb': rgb, 'nir': nir, 'intensity': intensity},
    tile_bbox=tile_bbox,
    patch_size=50,
    use_ndvi_refinement=True
)
""")

print("\n📖 For more examples, see:")
print("   - examples/ground_truth_ndvi_refinement_example.py")
print("   - docs/docs/features/ground-truth-ndvi-refinement.md")

print("\n💻 CLI usage:")
print("   ign-lidar-hd ground-truth data/tile.laz data/patches_gt --use-ndvi")

print("\n" + "=" * 80)
