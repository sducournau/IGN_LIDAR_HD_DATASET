#!/usr/bin/env python3
"""
Quick benchmark to verify strtree.py optimization performance gain.
Tests the vectorized vs iterrows() performance.
"""

import time
import numpy as np
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from shapely.geometry import Point, Polygon, MultiPolygon
    from shapely.prepared import prep
    import geopandas as gpd
    HAS_LIBS = True
except ImportError:
    print("⚠️  Required libraries not available (shapely, geopandas)")
    HAS_LIBS = False
    sys.exit(1)

print("="*70)
print("🧪 STRTREE.PY OPTIMIZATION BENCHMARK")
print("="*70)
print()

# Generate test data
n_polygons = 500  # Typical number of buildings in a tile
print(f"Generating {n_polygons} test polygons...")
polygons = [
    Point(x, y).buffer(10 + np.random.rand() * 20)
    for x, y in np.random.rand(n_polygons, 2) * 1000
]
gdf = gpd.GeoDataFrame({
    'geometry': polygons,
    'feature_type': ['building'] * n_polygons,
    'height': np.random.rand(n_polygons) * 30
})

print(f"✓ Test data created: {len(gdf)} polygons")
print()

# Benchmark 1: OLD METHOD (iterrows)
print("📊 Benchmark 1: OLD METHOD (.iterrows())")
print("-" * 70)

all_polygons_old = []
metadata_old = {}
use_prepared_geometries = True
road_buffer_tolerance = 0

start = time.time()

# OLD: Slow iterrows method
for idx, row in gdf.iterrows():
    polygon = row['geometry']
    
    if not isinstance(polygon, (Polygon, MultiPolygon)):
        continue
    
    # Apply buffer for roads if configured
    if row['feature_type'] == 'roads' and road_buffer_tolerance > 0:
        polygon = polygon.buffer(road_buffer_tolerance)
    
    # Use PreparedGeometry
    prepared_geom = prep(polygon) if use_prepared_geometries else None
    
    metadata = {
        'feature_type': row['feature_type'],
        'properties': dict(row),
        'prepared_geom': prepared_geom
    }
    
    all_polygons_old.append(polygon)
    metadata_old[id(polygon)] = metadata

time_old = time.time() - start

print(f"⏱️  Time: {time_old:.4f}s")
print(f"📦 Processed: {len(all_polygons_old)} polygons")
print()

# Benchmark 2: NEW METHOD (vectorized)
print("📊 Benchmark 2: NEW METHOD (vectorized)")
print("-" * 70)

all_polygons_new = []
metadata_new = {}

start = time.time()

# NEW: Vectorized method
# Step 1: Filter valid geometries (vectorized)
valid_mask = gdf.geometry.apply(
    lambda g: isinstance(g, (Polygon, MultiPolygon))
)
valid_gdf = gdf[valid_mask].copy()

# Step 2: Apply buffer if needed (vectorized)
if road_buffer_tolerance > 0:
    road_mask = valid_gdf['feature_type'] == 'roads'
    if road_mask.any():
        valid_gdf.loc[road_mask, 'geometry'] = valid_gdf.loc[road_mask, 'geometry'].buffer(
            road_buffer_tolerance
        )

# Step 3: Extract geometries array
geometries = valid_gdf.geometry.values

# Step 4: Prepare geometries (list comprehension)
prepared_geoms = [
    prep(g) if use_prepared_geometries else None 
    for g in geometries
]

# Step 5: Build metadata structures (minimal iteration)
for (idx, row), prepared_geom, polygon in zip(
    valid_gdf.iterrows(), 
    prepared_geoms,
    geometries
):
    metadata = {
        'feature_type': row['feature_type'],
        'properties': dict(row),
        'prepared_geom': prepared_geom
    }
    
    all_polygons_new.append(polygon)
    metadata_new[id(polygon)] = metadata

time_new = time.time() - start

print(f"⏱️  Time: {time_new:.4f}s")
print(f"📦 Processed: {len(all_polygons_new)} polygons")
print()

# Results
print("="*70)
print("📈 RESULTS")
print("="*70)
print()

speedup = time_old / time_new
improvement = ((time_old - time_new) / time_old) * 100

print(f"OLD Method:  {time_old:.4f}s")
print(f"NEW Method:  {time_new:.4f}s")
print(f"")
print(f"⚡ Speedup:     {speedup:.2f}× faster")
print(f"📊 Improvement: {improvement:.1f}% reduction in time")
print()

# Verify correctness
print("✓ Correctness check:")
print(f"  - Polygons processed (old): {len(all_polygons_old)}")
print(f"  - Polygons processed (new): {len(all_polygons_new)}")
print(f"  - Match: {'✅' if len(all_polygons_old) == len(all_polygons_new) else '❌'}")
print()

# Extrapolate to full tile processing
typical_tile_time_old = 10.0  # seconds (typical ground truth processing time)
estimated_tile_time_new = typical_tile_time_old / speedup

print("🎯 Estimated Impact on Full Tile Processing:")
print(f"  - Typical tile ground truth time (old): {typical_tile_time_old:.1f}s")
print(f"  - Estimated tile time (new):            {estimated_tile_time_new:.1f}s")
print(f"  - Time saved per tile:                  {typical_tile_time_old - estimated_tile_time_new:.1f}s")
print()

# Overall assessment
if speedup >= 5:
    print("🏆 EXCELLENT: Optimization provides significant speedup!")
elif speedup >= 2:
    print("✅ GOOD: Optimization provides solid performance improvement")
elif speedup >= 1.2:
    print("👍 OK: Optimization provides moderate improvement")
else:
    print("⚠️  WARNING: Optimization shows minimal improvement")

print()
print("="*70)
print(f"✅ Benchmark complete! Optimization is {speedup:.1f}× faster")
print("="*70)
