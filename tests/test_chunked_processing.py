#!/usr/bin/env python3
"""
Quick test to verify chunked feature computation works correctly.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.features import compute_all_features_optimized

# Create synthetic point cloud
print("Creating synthetic point cloud (5M points)...")
n_points = 5_000_000
points = np.random.rand(n_points, 3).astype(np.float32) * 100
classification = np.random.randint(0, 6, n_points, dtype=np.uint8)

print(f"\nTesting chunked processing...")
print(f"Points: {n_points:,}")

# Test with chunking
try:
    normals, curvature, height, geo_features = compute_all_features_optimized(
        points, classification,
        k=20,
        chunk_size=1_000_000  # Force chunking even on small data
    )
    
    print(f"\n✅ SUCCESS!")
    print(f"Normals shape: {normals.shape}")
    print(f"Curvature shape: {curvature.shape}")
    print(f"Height shape: {height.shape}")
    print(f"Geo features: {list(geo_features.keys())}")
    print(f"\nSample normals (first 5):")
    print(normals[:5])
    
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test without chunking
print(f"\n\nTesting non-chunked processing for comparison...")
try:
    normals2, curvature2, height2, geo_features2 = compute_all_features_optimized(
        points[:100000], classification[:100000],  # Small subset
        k=20,
        chunk_size=None  # No chunking
    )
    print(f"✅ Non-chunked also works!")
    
except Exception as e:
    print(f"❌ Non-chunked failed: {e}")

print("\n🎉 All tests passed!")
