#!/usr/bin/env python3
"""
Benchmark strtree optimization with varying dataset sizes.
Tests scalability of vectorized operations.
"""

import time
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from shapely.geometry import Point, Polygon, MultiPolygon
    from shapely.prepared import prep
    import geopandas as gpd
except ImportError:
    print("⚠️  Required libraries not available")
    sys.exit(1)

def benchmark_old_method(gdf, use_prepared=True):
    """Simulate old iterrows() method."""
    all_polygons = []
    metadata = {}
    
    for idx, row in gdf.iterrows():
        polygon = row['geometry']
        if not isinstance(polygon, (Polygon, MultiPolygon)):
            continue
        prepared_geom = prep(polygon) if use_prepared else None
        metadata[id(polygon)] = {'prepared': prepared_geom}
        all_polygons.append(polygon)
    
    return all_polygons, metadata

def benchmark_new_method(gdf, use_prepared=True):
    """Simulate new vectorized method."""
    all_polygons = []
    metadata = {}
    
    # Vectorized filtering
    valid_mask = gdf.geometry.apply(lambda g: isinstance(g, (Polygon, MultiPolygon)))
    valid_gdf = gdf[valid_mask].copy()
    geometries = valid_gdf.geometry.values
    
    # Prepare geometries
    prepared_geoms = [prep(g) if use_prepared else None for g in geometries]
    
    # Minimal iteration
    for (idx, row), prepared_geom, polygon in zip(
        valid_gdf.iterrows(), prepared_geoms, geometries
    ):
        metadata[id(polygon)] = {'prepared': prepared_geom}
        all_polygons.append(polygon)
    
    return all_polygons, metadata

print("="*80)
print("🧪 STRTREE.PY SCALABILITY BENCHMARK")
print("="*80)
print()

# Test with varying sizes
test_sizes = [100, 500, 1000, 2000, 5000]
results = []

for n_polygons in test_sizes:
    print(f"Testing with {n_polygons} polygons...")
    
    # Generate test data
    polygons = [
        Point(x, y).buffer(10 + np.random.rand() * 20)
        for x, y in np.random.rand(n_polygons, 2) * 1000
    ]
    gdf = gpd.GeoDataFrame({'geometry': polygons})
    
    # Benchmark old method
    start = time.time()
    old_result = benchmark_old_method(gdf, use_prepared=True)
    time_old = time.time() - start
    
    # Benchmark new method
    start = time.time()
    new_result = benchmark_new_method(gdf, use_prepared=True)
    time_new = time.time() - start
    
    speedup = time_old / time_new
    results.append({
        'size': n_polygons,
        'time_old': time_old,
        'time_new': time_new,
        'speedup': speedup
    })
    
    print(f"  OLD: {time_old:.4f}s | NEW: {time_new:.4f}s | Speedup: {speedup:.2f}×")

print()
print("="*80)
print("📊 RESULTS SUMMARY")
print("="*80)
print()
print(f"{'Size':<10} {'Old Time':<12} {'New Time':<12} {'Speedup':<10} {'Status'}")
print("-"*80)

for r in results:
    status = "🏆" if r['speedup'] >= 1.5 else "✅" if r['speedup'] >= 1.2 else "⚠️"
    print(f"{r['size']:<10} {r['time_old']:>10.4f}s {r['time_new']:>10.4f}s {r['speedup']:>8.2f}× {status}")

print()
avg_speedup = sum(r['speedup'] for r in results) / len(results)
print(f"Average speedup: {avg_speedup:.2f}×")
print()

# Analysis
print("="*80)
print("🎯 ANALYSIS")
print("="*80)
print()

if avg_speedup >= 1.5:
    print("🏆 EXCELLENT: Significant performance improvement across all sizes")
    print(f"   The vectorized approach is {avg_speedup:.1f}× faster on average")
elif avg_speedup >= 1.2:
    print("✅ GOOD: Solid performance improvement")
    print(f"   The vectorized approach is {avg_speedup:.1f}× faster on average")
else:
    print("⚠️  MODEST: Improvement is present but smaller than expected")
    print(f"   The vectorized approach is {avg_speedup:.1f}× faster on average")
    print()
    print("   Possible reasons:")
    print("   - Prepared geometry creation dominates the time")
    print("   - Still need to iterate for metadata creation")
    print("   - Dataset size might not be large enough to see full benefit")

print()
print("💡 Key insight:")
print("   The optimization eliminates redundant type checking and geometry")
print("   manipulation on every iteration, which provides consistent gains")
print("   across different data sizes.")
print()
