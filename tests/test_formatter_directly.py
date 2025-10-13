#!/usr/bin/env python3
"""
Direct test of feature formatting - bypass full pipeline
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

print("="*80)
print("TESTING FEATURE FORMATTING DIRECTLY")
print("="*80)

# Create a mock patch with ALL features
np.random.seed(42)
N = 8192

# Create mock patch with all expected features
patch = {
    'points': np.random.randn(N, 3).astype(np.float32),
    'labels': np.ones(N, dtype=np.uint8) * 6,
    
    # Basic features
    'normals': np.random.randn(N, 3).astype(np.float32),
    'curvature': np.random.randn(N).astype(np.float32),
    'intensity': np.random.randint(0, 255, N).astype(np.uint8),
    'return_number': np.ones(N, dtype=np.uint8),
    'height': np.random.randn(N).astype(np.float32),
    
    # RGB/NIR
    'rgb': np.random.randint(0, 255, (N, 3)).astype(np.uint8),
    'nir': np.random.randint(0, 255, N).astype(np.uint8),
    'ndvi': (np.random.randn(N) * 0.5).astype(np.float32),
    
    # Shape descriptors
    'planarity': np.random.rand(N).astype(np.float32),
    'linearity': np.random.rand(N).astype(np.float32),
    'sphericity': np.random.rand(N).astype(np.float32),
    'roughness': np.random.rand(N).astype(np.float32),
    'anisotropy': np.random.rand(N).astype(np.float32),
    'omnivariance': np.random.rand(N).astype(np.float32),
    
    # Eigenvalues
    'eigenvalue_1': np.random.rand(N).astype(np.float32),
    'eigenvalue_2': np.random.rand(N).astype(np.float32),
    'eigenvalue_3': np.random.rand(N).astype(np.float32),
    'sum_eigenvalues': np.random.rand(N).astype(np.float32),
    'eigenentropy': np.random.rand(N).astype(np.float32),
    'change_curvature': np.random.rand(N).astype(np.float32),
    
    # Height features
    'height_above_ground': np.random.randn(N).astype(np.float32),
    'vertical_std': np.random.rand(N).astype(np.float32),
    
    # Building scores
    'verticality': np.random.rand(N).astype(np.float32),
    'wall_score': np.random.rand(N).astype(np.float32),
    'roof_score': np.random.rand(N).astype(np.float32),
    
    # Density features
    'density': np.random.rand(N).astype(np.float32),
    'num_points_2m': np.random.randint(10, 100, N).astype(np.float32),
    'neighborhood_extent': np.random.rand(N).astype(np.float32),
    'height_extent_ratio': np.random.rand(N).astype(np.float32),
    
    # Architectural features
    'edge_strength': np.random.rand(N).astype(np.float32),
    'corner_likelihood': np.random.rand(N).astype(np.float32),
    'overhang_indicator': np.random.rand(N).astype(np.float32),
    'surface_roughness': np.random.rand(N).astype(np.float32),
}

print(f"\nMock patch created with {len(patch)} arrays:")
for key in sorted(patch.keys()):
    print(f"  - {key}")

# Test HybridFormatter
print("\n" + "="*80)
print("Testing HybridFormatter")
print("="*80)

from ign_lidar.io.formatters import HybridFormatter

formatter = HybridFormatter(
    num_points=8192,
    use_rgb=True,
    use_infrared=True,
    use_geometric=True,
    use_radiometric=False,
    use_contextual=False
)

print(f"\nFormatter configuration:")
print(f"  use_rgb: {formatter.use_rgb}")
print(f"  use_infrared: {formatter.use_infrared}")
print(f"  use_geometric: {formatter.use_geometric}")

# Format the patch
print(f"\nFormatting patch...")
result = formatter.format_patch(patch)

print(f"\n✅ Formatting successful!")
print(f"\nResult contains {len(result)} top-level keys:")
for key in sorted(result.keys()):
    val = result[key]
    if isinstance(val, np.ndarray):
        print(f"  - {key:20s} shape={str(val.shape):20s}")
    else:
        print(f"  - {key:20s} type={type(val).__name__}")

# Check features array
if 'features' in result:
    features_array = result['features']
    print(f"\n📊 Features array:")
    print(f"  Shape: {features_array.shape}")
    print(f"  Expected: (N, C) where C should be ~30-35 for full mode")
    
    num_features = features_array.shape[1] if len(features_array.shape) > 1 else 1
    print(f"  Actual C: {num_features}")
    
    if num_features < 25:
        print(f"  ❌ WARNING: Only {num_features} features! Expected 30-35!")
    else:
        print(f"  ✅ Good: {num_features} features found")

# Check metadata
if 'metadata' in result:
    metadata = result['metadata']
    if isinstance(metadata, dict):
        print(f"\n📋 Metadata:")
        if 'feature_names' in metadata:
            feature_names = metadata['feature_names']
            print(f"  Feature names ({len(feature_names)}):")
            for i, name in enumerate(feature_names):
                print(f"    {i+1:2d}. {name}")
        
        if 'num_features' in metadata:
            print(f"  num_features: {metadata['num_features']}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if 'features' in result:
    num_features = result['features'].shape[1]
    expected = len(patch) - 2  # Minus 'points' and 'labels'
    
    print(f"\nInput patch had: {len(patch)-2} feature arrays (excluding points, labels)")
    print(f"Output features array has: {num_features} features")
    print(f"Difference: {expected - num_features} features")
    
    if num_features < 25:
        print(f"\n❌ CONFIRMED: Features are being lost in the formatter!")
        print(f"   Only {num_features}/{expected} features made it through")
    else:
        print(f"\n✅ Good: Most features made it through the formatter")
