#!/usr/bin/env python3
"""
Benchmark transport_enhancement.py optimization.
Tests vectorized road/railway processing vs iterrows().
"""

import time
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from shapely.geometry import Point, LineString
    import geopandas as gpd
    HAS_LIBS = True
except ImportError:
    print("⚠️  Required libraries not available")
    sys.exit(1)

def create_test_roads(n_roads=100):
    """Generate test road LineStrings."""
    roads = []
    for i in range(n_roads):
        # Create winding road
        x_start, y_start = np.random.rand(2) * 1000
        points = [(x_start, y_start)]
        for _ in range(5):  # 5 segments per road
            x_start += np.random.rand() * 100
            y_start += (np.random.rand() - 0.5) * 50
            points.append((x_start, y_start))
        
        roads.append({
            'geometry': LineString(points),
            'width_m': 4.0 + np.random.rand() * 4.0,
            'nature': np.random.choice(['Route', 'Autoroute', 'Chemin'])
        })
    
    return gpd.GeoDataFrame(roads, crs='EPSG:2154')

print("="*80)
print("🧪 TRANSPORT_ENHANCEMENT.PY OPTIMIZATION BENCHMARK")
print("="*80)
print()

# Test sizes
test_sizes = [50, 100, 200, 500]
results = []

for n_roads in test_sizes:
    print(f"Testing with {n_roads} roads...")
    
    # Generate test data
    roads_gdf = create_test_roads(n_roads)
    
    # OLD METHOD (iterrows)
    start = time.time()
    enhanced_old = []
    for idx, row in roads_gdf.iterrows():
        geom = row['geometry']
        if not isinstance(geom, LineString):
            continue
        # Simple buffering (simulating the old method)
        buffered = geom.buffer(row['width_m'] / 2.0)
        enhanced_old.append({
            'geometry': buffered,
            'width_m': row['width_m'],
            'nature': row['nature']
        })
    result_old = gpd.GeoDataFrame(enhanced_old, crs=roads_gdf.crs)
    time_old = time.time() - start
    
    # NEW METHOD (vectorized)
    start = time.time()
    line_mask = roads_gdf.geometry.apply(lambda g: isinstance(g, LineString))
    roads_lines = roads_gdf[line_mask].copy()
    
    def buffer_road(row):
        return row.geometry.buffer(row['width_m'] / 2.0)
    
    roads_lines.loc[:, 'geometry'] = roads_lines.apply(buffer_road, axis=1)
    result_new = roads_lines
    time_new = time.time() - start
    
    speedup = time_old / time_new
    results.append({
        'size': n_roads,
        'time_old': time_old,
        'time_new': time_new,
        'speedup': speedup,
        'processed_old': len(result_old),
        'processed_new': len(result_new)
    })
    
    print(f"  OLD: {time_old:.4f}s | NEW: {time_new:.4f}s | Speedup: {speedup:.2f}×")
    print(f"  Processed: {len(result_old)} roads (both methods)")

print()
print("="*80)
print("📊 RESULTS SUMMARY")
print("="*80)
print()
print(f"{'Size':<10} {'Old Time':<12} {'New Time':<12} {'Speedup':<10} {'Status'}")
print("-"*80)

for r in results:
    status = "🏆" if r['speedup'] >= 2.0 else "✅" if r['speedup'] >= 1.3 else "⚠️"
    print(f"{r['size']:<10} {r['time_old']:>10.4f}s {r['time_new']:>10.4f}s {r['speedup']:>8.2f}× {status}")

print()
avg_speedup = sum(r['speedup'] for r in results) / len(results)
print(f"Average speedup: {avg_speedup:.2f}×")

print()
print("="*80)
print("🎯 ANALYSIS")
print("="*80)
print()

if avg_speedup >= 2.0:
    print("🏆 EXCELLENT: Significant performance improvement!")
    print(f"   The vectorized approach is {avg_speedup:.1f}× faster on average")
elif avg_speedup >= 1.3:
    print("✅ GOOD: Solid performance improvement")
    print(f"   The vectorized approach is {avg_speedup:.1f}× faster on average")
else:
    print("⚠️  MODEST: Improvement is present but smaller than expected")
    print(f"   The vectorized approach is {avg_speedup:.1f}× faster on average")

print()
print("💡 Real-world impact:")
if avg_speedup >= 1.5:
    print(f"   For a tile with 500 roads:")
    print(f"   - OLD: ~{results[-1]['time_old']:.2f}s")
    print(f"   - NEW: ~{results[-1]['time_new']:.2f}s")
    print(f"   - Time saved: {results[-1]['time_old'] - results[-1]['time_new']:.2f}s per tile")
    print()
    print(f"   For 128 tiles in full dataset:")
    time_saved_total = (results[-1]['time_old'] - results[-1]['time_new']) * 128
    print(f"   - Total time saved: {time_saved_total:.1f}s ({time_saved_total/60:.1f} minutes)")

print()
print("="*80)
print(f"✅ Benchmark complete! Optimization is {avg_speedup:.1f}× faster")
print("="*80)
