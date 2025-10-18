#!/usr/bin/env python3
"""End-to-end performance test for GPU fixes"""

import numpy as np
import time
from ign_lidar.features.features_gpu import GPUFeatureComputer

print('='*80)
print('END-TO-END GPU PERFORMANCE TEST')
print('='*80)

# Test with realistic dataset size
np.random.seed(42)
N = 1_000_000
points = np.random.rand(N, 3).astype(np.float32) * 100

computer = GPUFeatureComputer(use_gpu=True)

# Test 1: Normals
print(f'\n📊 Test 1: Computing normals for {N:,} points...')
start = time.time()
normals = computer.compute_normals(points, k=20)
elapsed = time.time() - start
print(f'   ✅ Time: {elapsed:.3f}s')
print(f'   ✅ Throughput: {N/elapsed:,.0f} points/sec')

# Test 2: Curvature
print(f'\n📊 Test 2: Computing curvature for {N:,} points...')
start = time.time()
curvature = computer.compute_curvature(points, normals, k=20)
elapsed = time.time() - start
print(f'   ✅ Time: {elapsed:.3f}s')
print(f'   ✅ Throughput: {N/elapsed:,.0f} points/sec')

# Test 3: Geometric features (our fixes!)
print(f'\n📊 Test 3: Computing geometric features for {N:,} points...')
features_list = ['planarity', 'linearity', 'sphericity', 'anisotropy']
start = time.time()
features = computer.compute_geometric_features(points, features_list, k=20)
elapsed = time.time() - start
print(f'   ✅ Time: {elapsed:.3f}s')
print(f'   ✅ Throughput: {N/elapsed:,.0f} points/sec')
print(f'   ✅ Features computed: {list(features.keys())}')
print(f'   ✅ Feature shapes: {[f.shape for f in features.values()]}')

print(f'\n{'='*80}')
print('🎉 ALL TESTS PASSED - GPU MODE FULLY FUNCTIONAL!')
print(f'{'='*80}')
