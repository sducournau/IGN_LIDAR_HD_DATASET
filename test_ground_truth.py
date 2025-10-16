"""
Test script to verify BD TOPO® ground truth classification is working.
"""

import logging
import laspy
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Test file
laz_file = Path('/mnt/d/ign/preprocessed/asprs/enriched_tiles/LHD_FXX_0326_6829_PTS_C_LAMB93_IGN69_enriched.laz')

print("\n" + "="*80)
print("BD TOPO® GROUND TRUTH CLASSIFICATION TEST")
print("="*80)

# Read LAZ file
logger.info(f"Reading: {laz_file.name}")
las = laspy.read(str(laz_file))

# Get bbox
bbox = (
    float(las.x.min()),
    float(las.y.min()),
    float(las.x.max()),
    float(las.y.max())
)
logger.info(f"Tile bbox: {bbox}")
logger.info(f"Total points: {len(las.points):,}")

# Check classification distribution
print("\n" + "-"*80)
print("CLASSIFICATION DISTRIBUTION")
print("-"*80)
unique, counts = np.unique(las.classification, return_counts=True)
for cls, count in zip(unique, counts):
    pct = (count / len(las.points)) * 100
    print(f"Class {cls:2d}: {count:10,} points ({pct:5.2f}%)")

# Check for road/rail classes
has_roads = 11 in unique
has_rails = 10 in unique

print("\n" + "-"*80)
print("GROUND TRUTH CLASSIFICATION CHECK")
print("-"*80)
print(f"Class 10 (Rail) present: {'✅ YES' if has_rails else '❌ NO'}")
print(f"Class 11 (Road) present: {'✅ YES' if has_roads else '❌ NO'}")

# Now try to fetch BD TOPO® data directly for this bbox
print("\n" + "-"*80)
print("FETCHING BD TOPO® DATA FOR THIS TILE")
print("-"*80)

try:
    from ign_lidar.io.data_fetcher import DataFetcher, DataFetchConfig
    
    config = DataFetchConfig(
        include_buildings=True,
        include_roads=True,
        include_railways=True,
        include_water=True,
        include_vegetation=True,
        road_width_fallback=4.0,
        railway_width_fallback=3.5
    )
    
    fetcher = DataFetcher(
        cache_dir="/mnt/d/ign/cache",
        config=config
    )
    
    logger.info("Fetching ground truth data...")
    gt_data = fetcher.fetch_all(bbox=bbox, use_cache=False)  # Don't use cache to force fresh fetch
    
    if gt_data and 'ground_truth' in gt_data:
        gt_features = gt_data['ground_truth']
        
        print("\n📍 BD TOPO® Features Found:")
        for feat_type, gdf in gt_features.items():
            if gdf is not None and len(gdf) > 0:
                print(f"  ✅ {feat_type}: {len(gdf)} features")
            else:
                print(f"  ❌ {feat_type}: None")
        
        # Check specifically for roads and railways
        roads = gt_features.get('roads')
        railways = gt_features.get('railways')
        
        print("\n" + "-"*80)
        print("ROAD/RAILWAY DATA ANALYSIS")
        print("-"*80)
        
        if roads is not None and len(roads) > 0:
            print(f"\n✅ ROADS FOUND: {len(roads)} road segments")
            print(f"   Total road length: {roads.geometry.length.sum():.1f} m")
            print(f"   Average road width: {roads['width_m'].mean():.1f} m")
            if 'road_type' in roads.columns:
                print(f"   Road types: {roads['road_type'].unique()}")
        else:
            print("\n❌ NO ROADS FOUND IN BD TOPO®")
        
        if railways is not None and len(railways) > 0:
            print(f"\n✅ RAILWAYS FOUND: {len(railways)} railway segments")
            print(f"   Total railway length: {railways.geometry.length.sum():.1f} m")
        else:
            print("\n❌ NO RAILWAYS FOUND IN BD TOPO®")
            
    else:
        print("\n❌ NO GROUND TRUTH DATA RETURNED")
        
except Exception as e:
    logger.error(f"Failed to fetch ground truth: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
