"""
Test RGB Augmentation Fix

Verifies that RGB augmentation is applied correctly and consistently
across original and augmented patch versions.
"""

import numpy as np
from pathlib import Path

from ign_lidar.core.modules.patch_extractor import (
    extract_patches,
    create_patch_versions,
    PatchConfig,
    AugmentationConfig
)


def test_patch_metadata_stored():
    """Test that patch extraction stores center and bounds metadata."""
    # Create synthetic point cloud (100m x 100m tile)
    np.random.seed(42)
    n_points = 10000
    points = np.random.rand(n_points, 3) * 100
    features = {'intensity': np.random.rand(n_points)}
    labels = np.random.randint(0, 10, n_points)
    
    # Extract patches
    patches = extract_patches(
        points=points,
        features=features,
        labels=labels,
        patch_size=50.0,
        overlap=0.1,
        min_points=100
    )
    
    # Verify metadata present
    assert len(patches) > 0, "Should extract at least one patch"
    
    for patch in patches:
        assert '_patch_center' in patch, "Patch should have _patch_center metadata"
        assert '_patch_bounds' in patch, "Patch should have _patch_bounds metadata"
        assert patch['_patch_center'].shape == (3,), "Patch center should be 3D"
        assert len(patch['_patch_bounds']) == 4, "Patch bounds should have 4 values"
        
        # Verify bounds are reasonable
        x_start, y_start, x_end, y_end = patch['_patch_bounds']
        assert x_start < x_end, "x_start should be less than x_end"
        assert y_start < y_end, "y_start should be less than y_end"
        
        # Verify center is within bounds
        center_x, center_y = patch['_patch_center'][:2]
        assert x_start <= center_x <= x_end, "Center X should be within bounds"
        assert y_start <= center_y <= y_end, "Center Y should be within bounds"
    
    print(f"✓ Test passed: {len(patches)} patches have correct metadata")


def test_rgb_extracted_with_patches():
    """Test that RGB is extracted along with patches when present in features."""
    # Create synthetic tile with RGB based on spatial location
    np.random.seed(42)
    n_points = 10000
    points = np.random.rand(n_points, 3) * 100
    
    # Create "fake" RGB based on spatial location (for verification)
    rgb = np.zeros((n_points, 3), dtype=np.float32)
    rgb[:, 0] = points[:, 0] / 100.0  # Red = normalized X
    rgb[:, 1] = points[:, 1] / 100.0  # Green = normalized Y
    rgb[:, 2] = 0.5  # Blue = constant
    
    features = {'rgb': rgb, 'intensity': np.random.rand(n_points)}
    labels = np.random.randint(0, 10, n_points)
    
    # Extract patches
    patches = extract_patches(
        points=points,
        features=features,
        labels=labels,
        patch_size=50.0,
        overlap=0.1,
        min_points=100
    )
    
    assert len(patches) > 0, "Should extract at least one patch"
    
    # Verify RGB is present in all patches
    for patch in patches:
        assert 'rgb' in patch, "Patch should have RGB"
        assert patch['rgb'].shape[0] == len(patch['points']), "RGB should match point count"
        assert patch['rgb'].shape[1] == 3, "RGB should have 3 channels"
        
        # Verify RGB is in valid range [0, 1]
        assert patch['rgb'].min() >= 0, "RGB should be >= 0"
        assert patch['rgb'].max() <= 1, "RGB should be <= 1"
    
    print(f"✓ Test passed: {len(patches)} patches have RGB data")


def test_rgb_consistency_across_augmentations():
    """Test that RGB values are consistent across patch versions."""
    # Create synthetic tile with RGB
    np.random.seed(42)
    n_points = 10000
    points = np.random.rand(n_points, 3) * 100
    
    # Create RGB based on spatial location
    rgb = np.zeros((n_points, 3), dtype=np.float32)
    rgb[:, 0] = points[:, 0] / 100.0  # Red = normalized X
    rgb[:, 1] = points[:, 1] / 100.0  # Green = normalized Y
    rgb[:, 2] = 0.5  # Blue = constant
    
    features = {'rgb': rgb, 'intensity': np.random.rand(n_points)}
    labels = np.random.randint(0, 10, n_points)
    
    # Extract base patches
    base_patches = extract_patches(
        points=points,
        features=features,
        labels=labels,
        patch_size=50.0,
        overlap=0.1,
        min_points=100
    )
    
    assert len(base_patches) > 0, "Should extract at least one base patch"
    
    # Create augmented versions
    num_augmentations = 3
    aug_config = AugmentationConfig()
    all_patches = []
    
    for idx, base in enumerate(base_patches):
        # Original
        orig = base.copy()
        orig['_version'] = 'original'
        orig['_patch_idx'] = idx
        all_patches.append(orig)
        
        # Augmented
        for aug_idx in range(num_augmentations):
            from ign_lidar.core.modules.patch_extractor import augment_patch
            aug = augment_patch(base, aug_config)
            aug['_version'] = f'aug_{aug_idx}'
            aug['_patch_idx'] = idx
            all_patches.append(aug)
    
    # Group by patch index
    from collections import defaultdict
    groups = defaultdict(list)
    for patch in all_patches:
        groups[patch['_patch_idx']].append(patch)
    
    # Test each group
    for patch_idx, patch_group in groups.items():
        original = [p for p in patch_group if p['_version'] == 'original'][0]
        augmented = [p for p in patch_group if 'aug' in p['_version']]
        
        # RGB mean should be similar (same spatial region, different dropout)
        orig_rgb_mean = original['rgb'].mean(axis=0)
        
        for aug_patch in augmented:
            aug_rgb_mean = aug_patch['rgb'].mean(axis=0)
            
            # Allow some difference due to dropout (loses some points)
            # but means should be close (within 0.15)
            diff = np.abs(orig_rgb_mean - aug_rgb_mean)
            
            # Check each channel
            for i, channel in enumerate(['Red', 'Green', 'Blue']):
                assert diff[i] < 0.15, (
                    f"RGB {channel} mean difference too large: {diff[i]:.4f} "
                    f"(original: {orig_rgb_mean[i]:.4f}, aug: {aug_rgb_mean[i]:.4f})"
                )
    
    print(f"✓ Test passed: RGB consistent across {len(groups)} patch groups")


def test_metadata_preserved_through_augmentation():
    """Test that metadata is preserved through augmentation."""
    # Create synthetic patch
    np.random.seed(42)
    n_points = 1000
    base_patch = {
        'points': np.random.rand(n_points, 3),
        'labels': np.random.randint(0, 10, n_points),
        'rgb': np.random.rand(n_points, 3),
        '_patch_center': np.array([50.0, 50.0, 0.0]),
        '_patch_bounds': (25.0, 25.0, 75.0, 75.0),
    }
    
    # Create augmented versions
    from ign_lidar.core.modules.patch_extractor import augment_patch
    aug_config = AugmentationConfig()
    
    for i in range(3):
        aug_patch = augment_patch(base_patch, aug_config)
        
        # Verify metadata is preserved
        assert '_patch_center' in aug_patch, "Metadata should be preserved"
        assert '_patch_bounds' in aug_patch, "Metadata should be preserved"
        
        # Verify values match original
        assert np.allclose(aug_patch['_patch_center'], base_patch['_patch_center']), \
            "Patch center should be unchanged"
        assert aug_patch['_patch_bounds'] == base_patch['_patch_bounds'], \
            "Patch bounds should be unchanged"
        
        # Verify points are transformed (different count due to dropout)
        assert aug_patch['points'].shape[0] != base_patch['points'].shape[0] or \
               not np.allclose(aug_patch['points'], base_patch['points']), \
            "Augmented points should be different from original (transformed or dropout)"
        
        # Verify RGB is transformed (same dropout as points)
        assert aug_patch['rgb'].shape[0] == aug_patch['points'].shape[0], \
            "RGB should have same dropout as points"
    
    print("✓ Test passed: Metadata preserved through augmentation")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Testing RGB Augmentation Fix")
    print("="*60 + "\n")
    
    test_patch_metadata_stored()
    print()
    
    test_rgb_extracted_with_patches()
    print()
    
    test_rgb_consistency_across_augmentations()
    print()
    
    test_metadata_preserved_through_augmentation()
    print()
    
    print("="*60)
    print("All tests passed! ✓")
    print("="*60)
