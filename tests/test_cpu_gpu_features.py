#!/usr/bin/env python3
"""
Test feature computation for both CPU and GPU to compare results
"""

import numpy as np
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Create a simple test point cloud
np.random.seed(42)
N = 50000
points = np.random.randn(N, 3).astype(np.float32) * 10
classification = np.ones(N, dtype=np.uint8) * 6  # Building

print("="*80)
print("COMPARING CPU vs GPU FEATURE COMPUTATION")
print("="*80)
print(f"\nTest data: {N} points")

# Expected advanced features
expected_features = [
    'planarity', 'linearity', 'sphericity', 'roughness', 'anisotropy', 'omnivariance',
    'curvature', 'change_curvature',
    'eigenvalue_1', 'eigenvalue_2', 'eigenvalue_3', 'sum_eigenvalues', 'eigenentropy',
    'height_above_ground', 'vertical_std',
    'verticality', 'wall_score', 'roof_score',
    'density', 'num_points_2m', 'neighborhood_extent', 'height_extent_ratio',
    'edge_strength', 'corner_likelihood', 'overhang_indicator', 'surface_roughness'
]

def test_computer(use_gpu, use_chunked, name):
    """Test a specific feature computer configuration"""
    print("\n" + "="*80)
    print(f"Testing: {name}")
    print("="*80)
    
    from ign_lidar.features.factory import FeatureComputerFactory
    
    try:
        computer = FeatureComputerFactory.create(
            use_gpu=use_gpu,
            use_chunked=use_chunked,
            k_neighbors=20,
            gpu_batch_size=25000
        )
        
        print(f"Computer type: {type(computer).__name__}")
        print(f"Available: {computer.is_available()}")
        
        if not computer.is_available():
            print("⚠️  Computer not available, skipping test")
            return None
        
        # Compute features with mode='full'
        result = computer.compute_features(
            points=points,
            classification=classification,
            mode='full'
        )
        
        # Analyze results
        print(f"\n✅ Computation successful!")
        print(f"Total features/arrays returned: {len(result)}")
        
        # Check for expected features
        found = [f for f in expected_features if f in result]
        missing = [f for f in expected_features if f not in result]
        
        print(f"\nFeature completeness:")
        print(f"  Expected: {len(expected_features)} features")
        print(f"  Found:    {len(found)} features ({len(found)*100//len(expected_features)}%)")
        print(f"  Missing:  {len(missing)} features")
        
        if missing:
            print(f"\n❌ Missing features:")
            for f in missing[:10]:  # Show first 10
                print(f"  - {f}")
            if len(missing) > 10:
                print(f"  ... and {len(missing)-10} more")
        else:
            print("\n✅ All expected features found!")
        
        # List all features returned
        print(f"\nAll features returned ({len(result)}):")
        for key in sorted(result.keys()):
            val = result[key]
            if isinstance(val, np.ndarray):
                if val.ndim == 1:
                    print(f"  {key:30s} shape={str(val.shape):15s}")
                elif val.ndim == 2:
                    print(f"  {key:30s} shape={str(val.shape):15s}")
        
        return {
            'total': len(result),
            'found': len(found),
            'missing': len(missing),
            'missing_list': missing,
            'all_features': sorted(result.keys())
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

# Test different configurations
results = {}

# 1. CPU
print("\n" + "#"*80)
print("# TEST 1: CPU-based feature computation")
print("#"*80)
results['cpu'] = test_computer(use_gpu=False, use_chunked=False, name="CPU")

# 2. GPU (non-chunked)
print("\n" + "#"*80)
print("# TEST 2: GPU-based feature computation (non-chunked)")
print("#"*80)
results['gpu'] = test_computer(use_gpu=True, use_chunked=False, name="GPU")

# 3. GPU Chunked
print("\n" + "#"*80)
print("# TEST 3: GPU-based feature computation (chunked)")
print("#"*80)
results['gpu_chunked'] = test_computer(use_gpu=True, use_chunked=True, name="GPU Chunked")

# Summary comparison
print("\n" + "="*80)
print("SUMMARY COMPARISON")
print("="*80)

print(f"\n{'Configuration':<20} {'Total Features':<15} {'Found':<10} {'Missing':<10} {'Status':<10}")
print("-"*80)

for name, result in results.items():
    if result is None:
        print(f"{name:<20} {'N/A':<15} {'N/A':<10} {'N/A':<10} {'SKIPPED':<10}")
    else:
        status = "✅ PASS" if result['missing'] == 0 else f"❌ FAIL"
        print(f"{name:<20} {result['total']:<15} {result['found']:<10} {result['missing']:<10} {status:<10}")

# Check if all configurations produce the same features
print("\n" + "="*80)
print("CONSISTENCY CHECK")
print("="*80)

valid_results = {k: v for k, v in results.items() if v is not None}

if len(valid_results) > 1:
    # Compare feature sets
    feature_sets = {name: set(r['all_features']) for name, r in valid_results.items()}
    
    # Find common and unique features
    all_features_union = set()
    for fs in feature_sets.values():
        all_features_union.update(fs)
    
    common_features = set(list(feature_sets.values())[0])
    for fs in feature_sets.values():
        common_features &= fs
    
    print(f"\nTotal unique features across all configurations: {len(all_features_union)}")
    print(f"Features common to all configurations: {len(common_features)}")
    
    # Show differences
    for name, fs in feature_sets.items():
        unique = fs - common_features
        if unique:
            print(f"\n{name} has {len(unique)} unique features:")
            for f in sorted(unique)[:5]:
                print(f"  - {f}")
        
        missing_from_this = all_features_union - fs
        if missing_from_this:
            print(f"\n{name} is missing {len(missing_from_this)} features that others have:")
            for f in sorted(missing_from_this)[:5]:
                print(f"  - {f}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

all_pass = all(r and r['missing'] == 0 for r in results.values() if r is not None)

if all_pass:
    print("\n✅ All configurations compute all expected features correctly!")
    print("The issue is likely in the downstream processing (patch extraction, formatting, or saving).")
else:
    print("\n❌ Some configurations are missing features!")
    print("The issue is in the feature computation itself.")

print("\n" + "="*80)
