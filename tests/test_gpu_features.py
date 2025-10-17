#!/usr/bin/env python3
"""
Test what features are returned by the GPU chunked computer

Updated for Week 2: Uses Strategy Pattern instead of deprecated Factory Pattern
"""

import numpy as np
import sys
import logging

logging.basicConfig(level=logging.INFO)

# Create a simple test point cloud
np.random.seed(42)
N = 50000
points = np.random.randn(N, 3).astype(np.float32) * 10
classification = np.ones(N, dtype=np.uint8) * 6  # Building

print("="*80)
print("TESTING GPU CHUNKED STRATEGY WITH MODE='full'")
print("="*80)
print(f"\nTest data: {N} points")

# Create GPU chunked strategy (Week 2: Strategy Pattern)
from ign_lidar.features.strategies import BaseFeatureStrategy

# Auto-select will choose GPU chunked for this dataset size
strategy = BaseFeatureStrategy.auto_select(
    n_points=N,
    mode='auto',
    k_neighbors=20,
    chunk_size=25000
)

print(f"\nStrategy type: {type(strategy).__name__}")
print(f"Strategy description: {strategy}")

# Compute features with mode='full'
print("\n" + "="*80)
print("Computing features with mode='full'")
print("="*80)

result = strategy.compute(
    points=points,
    classification=classification,
    mode='full'
)

print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\nFeatures returned:")
for key in sorted(result.keys()):
    val = result[key]
    if isinstance(val, np.ndarray):
        if val.ndim == 1:
            print(f"  {key:30s} shape={str(val.shape):15s} dtype={val.dtype}")
        elif val.ndim == 2:
            print(f"  {key:30s} shape={str(val.shape):15s} dtype={val.dtype}")
    else:
        print(f"  {key:30s} type={type(val)}")

print(f"\n{len(result)} total features/arrays returned")

# Check for expected features
expected_missing = [
    'eigenvalue_1', 'eigenvalue_2', 'eigenvalue_3',
    'sum_eigenvalues', 'eigenentropy',
    'change_curvature',
    'wall_score', 'roof_score',
    'edge_strength', 'corner_likelihood', 'overhang_indicator', 'surface_roughness'
]

missing = [f for f in expected_missing if f not in result]
found = [f for f in expected_missing if f in result]

print("\n" + "="*80)
print("EXPECTED ADVANCED FEATURES")
print("="*80)
print(f"\n✅ Found ({len(found)}):")
for f in found:
    print(f"  - {f}")

print(f"\n❌ Missing ({len(missing)}):")
for f in missing:
    print(f"  - {f}")

if missing:
    print("\n⚠️  WARNING: Some advanced features are missing!")
    print("This confirms the issue - GPU chunked computer is not returning all features.")
else:
    print("\n✅ All expected features found!")
    print("The issue might be elsewhere in the pipeline.")
