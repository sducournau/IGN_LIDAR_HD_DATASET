#!/usr/bin/env python3
"""
Test the new patch-level augmentation approach to verify correspondence.

This script tests that augmented patches properly correspond to their original
counterparts when augmentation is applied at the patch level instead of tile level.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.preprocessing.utils import extract_patches, augment_patch


def create_test_data():
    """Create synthetic test data representing a tile."""
    # Create a 100x100 meter tile with 10000 points
    np.random.seed(42)
    
    points = np.random.uniform(0, 100, (10000, 3))
    labels = np.random.randint(0, 5, 10000)
    
    # Create simple features
    features = {
        'intensity': np.random.random(10000).astype(np.float32),
        'return_number': np.ones(10000, dtype=np.uint8),
        'normals': np.random.randn(10000, 3).astype(np.float32),
        'curvature': np.random.random(10000).astype(np.float32),
        'height': points[:, 2].copy().astype(np.float32),
    }
    
    # Normalize normals
    features['normals'] /= np.linalg.norm(features['normals'], axis=1, keepdims=True)
    
    return points, features, labels


def test_patch_level_augmentation():
    """Test patch-level augmentation maintains correspondence."""
    print("="*80)
    print("Testing Patch-Level Augmentation Correspondence")
    print("="*80)
    
    # Create test data
    print("\n1. Creating synthetic tile data...")
    points, features, labels = create_test_data()
    print(f"   Created tile with {len(points):,} points")
    
    # Extract patches
    print("\n2. Extracting patches from original tile...")
    patches = extract_patches(
        points=points,
        features=features,
        labels=labels,
        patch_size=20.0,
        overlap=0.1,
        min_points=100,
        target_num_points=512
    )
    print(f"   Extracted {len(patches)} patches")
    
    # Test augmentation on first patch
    print("\n3. Testing augmentation on patch 0...")
    orig_patch = patches[0]
    
    print(f"   Original patch:")
    print(f"     Points shape: {orig_patch['points'].shape}")
    print(f"     Labels shape: {orig_patch['labels'].shape}")
    print(f"     Points range: X=[{orig_patch['points'][:,0].min():.2f}, {orig_patch['points'][:,0].max():.2f}]")
    print(f"     Points range: Y=[{orig_patch['points'][:,1].min():.2f}, {orig_patch['points'][:,1].max():.2f}]")
    print(f"     Points range: Z=[{orig_patch['points'][:,2].min():.2f}, {orig_patch['points'][:,2].max():.2f}]")
    
    # Create augmented version
    aug_patch = augment_patch(orig_patch)
    
    print(f"\n   Augmented patch:")
    print(f"     Points shape: {aug_patch['points'].shape}")
    print(f"     Labels shape: {aug_patch['labels'].shape}")
    print(f"     Points range: X=[{aug_patch['points'][:,0].min():.2f}, {aug_patch['points'][:,0].max():.2f}]")
    print(f"     Points range: Y=[{aug_patch['points'][:,1].min():.2f}, {aug_patch['points'][:,1].max():.2f}]")
    print(f"     Points range: Z=[{aug_patch['points'][:,2].min():.2f}, {aug_patch['points'][:,2].max():.2f}]")
    
    # Check correspondence
    print("\n4. Checking correspondence...")
    
    # Both should have similar number of points (accounting for dropout)
    orig_count = len(orig_patch['points'])
    aug_count = len(aug_patch['points'])
    dropout_ratio = 1 - (aug_count / orig_count)
    print(f"   Point counts: Original={orig_count}, Augmented={aug_count}")
    print(f"   Dropout ratio: {dropout_ratio*100:.1f}%")
    
    if not (0.05 <= dropout_ratio <= 0.15):
        print(f"   ⚠️  WARNING: Dropout ratio outside expected range [5%, 15%]")
        return False
    
    # Check that points are different (augmentation was applied)
    if np.allclose(orig_patch['points'][:aug_count], aug_patch['points'][:aug_count], atol=0.01):
        print(f"   ❌ FAILED: Points are identical - augmentation not applied!")
        return False
    else:
        print(f"   ✓ Points are different - augmentation applied")
    
    # Check that label distribution is similar (same spatial region)
    orig_label_dist = np.bincount(orig_patch['labels'], minlength=10)
    aug_label_dist = np.bincount(aug_patch['labels'], minlength=10)
    
    print(f"\n   Label distribution:")
    for i in range(10):
        if orig_label_dist[i] > 0 or aug_label_dist[i] > 0:
            print(f"     Class {i}: Original={orig_label_dist[i]:4d}, Augmented={aug_label_dist[i]:4d}")
    
    # Calculate label distribution similarity (accounting for dropout)
    orig_label_ratios = orig_label_dist / orig_count
    aug_label_ratios = aug_label_dist / aug_count
    
    # Allow 20% difference in ratios (due to random dropout)
    max_diff = np.abs(orig_label_ratios - aug_label_ratios).max()
    print(f"\n   Max label ratio difference: {max_diff*100:.1f}%")
    
    if max_diff > 0.25:
        print(f"   ⚠️  WARNING: Label distribution changed significantly")
        print(f"   This might indicate patches don't correspond properly")
        return False
    else:
        print(f"   ✓ Label distributions are similar - patches correspond!")
    
    # Check that feature keys match
    orig_keys = set(orig_patch.keys())
    aug_keys = set(aug_patch.keys())
    
    if orig_keys != aug_keys:
        print(f"\n   ❌ FAILED: Feature keys don't match!")
        print(f"   Original: {orig_keys}")
        print(f"   Augmented: {aug_keys}")
        return False
    else:
        print(f"\n   ✓ All features present in both patches")
    
    print("\n" + "="*80)
    print("✓ TEST PASSED: Patch-level augmentation maintains correspondence!")
    print("="*80)
    
    return True


def main():
    """Main test function."""
    try:
        success = test_patch_level_augmentation()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ TEST FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
