"""
Test Augmentation Verification

Verifies that when augmentation is enabled with num_augmentations > 1,
the augmented versions are properly calculated and different from the original.
"""

import numpy as np
from pathlib import Path
import tempfile
import shutil

from ign_lidar.core.modules.patch_extractor import (
    augment_raw_points,
    augment_patch,
    create_patch_versions,
    AugmentationConfig
)


def test_augment_raw_points_creates_different_versions():
    """Test that augment_raw_points creates different versions each time."""
    print("\n" + "="*70)
    print("TEST: augment_raw_points creates different versions")
    print("="*70)
    
    # Create synthetic data
    np.random.seed(42)
    n_points = 1000
    points = np.random.rand(n_points, 3) * 100
    intensity = np.random.rand(n_points)
    return_number = np.ones(n_points, dtype=np.uint8)
    classification = np.random.randint(0, 10, n_points, dtype=np.uint8)
    rgb = np.random.rand(n_points, 3)
    nir = np.random.rand(n_points)
    ndvi = np.random.rand(n_points)
    
    # Create multiple augmented versions
    num_versions = 5
    augmented_versions = []
    
    for i in range(num_versions):
        aug_points, aug_intensity, aug_return, aug_class, aug_rgb, aug_nir, aug_ndvi = augment_raw_points(
            points=points.copy(),
            intensity=intensity.copy(),
            return_number=return_number.copy(),
            classification=classification.copy(),
            rgb=rgb.copy(),
            nir=nir.copy(),
            ndvi=ndvi.copy(),
            config=AugmentationConfig()
        )
        augmented_versions.append({
            'points': aug_points,
            'intensity': aug_intensity,
            'return_number': aug_return,
            'classification': aug_class,
            'rgb': aug_rgb,
            'nir': aug_nir,
            'ndvi': aug_ndvi,
            'count': len(aug_points)
        })
    
    # Verify all versions are different
    print(f"\n✓ Created {num_versions} augmented versions:")
    for i, version in enumerate(augmented_versions):
        print(f"  Version {i+1}: {version['count']:,} points (dropout from {n_points:,})")
    
    # Check that point counts differ (due to dropout)
    counts = [v['count'] for v in augmented_versions]
    unique_counts = len(set(counts))
    print(f"\n✓ Unique point counts: {unique_counts}/{num_versions}")
    assert unique_counts > 1, "Expected different point counts due to random dropout"
    
    # Check that point coordinates differ
    for i in range(num_versions - 1):
        for j in range(i + 1, num_versions):
            v1 = augmented_versions[i]
            v2 = augmented_versions[j]
            
            # Compare first few points (if both versions have enough)
            compare_n = min(10, v1['count'], v2['count'])
            if compare_n > 0:
                diff = np.abs(v1['points'][:compare_n] - v2['points'][:compare_n]).mean()
                assert diff > 0.01, f"Versions {i+1} and {j+1} are too similar (diff={diff:.6f})"
    
    print(f"✓ All {num_versions} versions are different from each other")
    print("✓ TEST PASSED: augment_raw_points creates unique versions\n")


def test_augment_patch_creates_different_versions():
    """Test that augment_patch creates different versions each time."""
    print("\n" + "="*70)
    print("TEST: augment_patch creates different versions")
    print("="*70)
    
    # Create synthetic patch
    np.random.seed(42)
    n_points = 1000
    base_patch = {
        'points': np.random.rand(n_points, 3),
        'labels': np.random.randint(0, 10, n_points),
        'intensity': np.random.rand(n_points),
        'return_number': np.ones(n_points, dtype=np.uint8),
        'rgb': np.random.rand(n_points, 3),
        'nir': np.random.rand(n_points),
        'ndvi': np.random.rand(n_points),
        '_patch_center': np.array([50.0, 50.0, 0.0]),
        '_patch_bounds': (25.0, 25.0, 75.0, 75.0),
    }
    
    # Create multiple augmented versions
    num_versions = 5
    config = AugmentationConfig()
    augmented_patches = []
    
    for i in range(num_versions):
        aug_patch = augment_patch(base_patch.copy(), config=config)
        augmented_patches.append(aug_patch)
    
    # Verify all versions are different
    print(f"\n✓ Created {num_versions} augmented patch versions:")
    for i, patch in enumerate(augmented_patches):
        print(f"  Version {i+1}: {len(patch['points']):,} points")
    
    # Check that point counts differ (due to dropout)
    counts = [len(p['points']) for p in augmented_patches]
    unique_counts = len(set(counts))
    print(f"\n✓ Unique point counts: {unique_counts}/{num_versions}")
    assert unique_counts > 1, "Expected different point counts due to random dropout"
    
    # Check that coordinates differ between versions
    for i in range(num_versions - 1):
        for j in range(i + 1, num_versions):
            p1 = augmented_patches[i]
            p2 = augmented_patches[j]
            
            # Compare first few points
            compare_n = min(10, len(p1['points']), len(p2['points']))
            if compare_n > 0:
                diff = np.abs(p1['points'][:compare_n] - p2['points'][:compare_n]).mean()
                assert diff > 0.01, f"Patch versions {i+1} and {j+1} are too similar"
    
    # Verify metadata is preserved
    for i, patch in enumerate(augmented_patches):
        assert '_patch_center' in patch, f"Version {i+1} missing _patch_center"
        assert '_patch_bounds' in patch, f"Version {i+1} missing _patch_bounds"
        assert np.allclose(patch['_patch_center'], base_patch['_patch_center']), \
            f"Version {i+1} has modified _patch_center"
    
    print(f"✓ All {num_versions} patch versions are different from each other")
    print("✓ Metadata preserved across all versions")
    print("✓ TEST PASSED: augment_patch creates unique versions\n")


def test_create_patch_versions_with_augmentation():
    """Test create_patch_versions with augmentation enabled."""
    print("\n" + "="*70)
    print("TEST: create_patch_versions with augmentation enabled")
    print("="*70)
    
    # Create synthetic base patches
    np.random.seed(42)
    num_base_patches = 3
    base_patches = []
    
    for i in range(num_base_patches):
        n_points = 1000 + i * 100
        patch = {
            'points': np.random.rand(n_points, 3),
            'labels': np.random.randint(0, 10, n_points),
            'intensity': np.random.rand(n_points),
            'return_number': np.ones(n_points, dtype=np.uint8),
            'rgb': np.random.rand(n_points, 3),
        }
        base_patches.append(patch)
    
    # Create versions with augmentation
    num_augmentations = 3
    config = AugmentationConfig()
    
    all_patches = create_patch_versions(
        base_patches=base_patches,
        num_augmentations=num_augmentations,
        augment_config=config
    )
    
    # Verify total count
    expected_total = num_base_patches * (1 + num_augmentations)  # original + augmented
    print(f"\n✓ Base patches: {num_base_patches}")
    print(f"✓ Augmentations per patch: {num_augmentations}")
    print(f"✓ Total patches created: {len(all_patches)} (expected: {expected_total})")
    assert len(all_patches) == expected_total, \
        f"Expected {expected_total} patches, got {len(all_patches)}"
    
    # Verify versions are labeled correctly
    version_counts = {}
    patch_idx_counts = {}
    
    for patch in all_patches:
        assert '_version' in patch, "Patch missing _version metadata"
        assert '_patch_idx' in patch, "Patch missing _patch_idx metadata"
        
        version = patch['_version']
        patch_idx = patch['_patch_idx']
        
        version_counts[version] = version_counts.get(version, 0) + 1
        patch_idx_counts[patch_idx] = patch_idx_counts.get(patch_idx, 0) + 1
    
    print(f"\n✓ Version distribution:")
    for version, count in sorted(version_counts.items()):
        print(f"  {version}: {count} patches")
    
    # Verify each base patch has correct number of versions
    assert version_counts['original'] == num_base_patches, \
        f"Expected {num_base_patches} original patches"
    
    for aug_idx in range(num_augmentations):
        version_name = f'aug_{aug_idx}'
        assert version_name in version_counts, f"Missing augmentation {version_name}"
        assert version_counts[version_name] == num_base_patches, \
            f"Expected {num_base_patches} patches for {version_name}"
    
    # Verify each patch_idx has correct number of versions
    print(f"\n✓ Patch index distribution:")
    for patch_idx, count in sorted(patch_idx_counts.items()):
        print(f"  Patch {patch_idx}: {count} versions")
        expected = 1 + num_augmentations
        assert count == expected, \
            f"Patch {patch_idx} has {count} versions, expected {expected}"
    
    # Verify augmented versions are different from originals
    originals = [p for p in all_patches if p['_version'] == 'original']
    augmented = [p for p in all_patches if 'aug' in p['_version']]
    
    print(f"\n✓ Comparing augmented vs original versions:")
    for orig in originals:
        patch_idx = orig['_patch_idx']
        aug_versions = [p for p in augmented if p['_patch_idx'] == patch_idx]
        
        print(f"  Patch {patch_idx}:")
        print(f"    Original: {len(orig['points']):,} points")
        
        for aug in aug_versions:
            version = aug['_version']
            count = len(aug['points'])
            print(f"    {version}: {count:,} points")
            
            # Points should be different due to augmentation
            assert count != len(orig['points']) or \
                   not np.allclose(aug['points'], orig['points'][:count]), \
                f"Augmented version {version} is identical to original"
    
    print("✓ TEST PASSED: create_patch_versions generates correct versions\n")


def test_augmentation_with_num_greater_than_one():
    """Test that setting num_augmentations > 1 creates multiple distinct versions."""
    print("\n" + "="*70)
    print("TEST: Augmentation with num_augmentations > 1")
    print("="*70)
    
    # Create synthetic patch
    np.random.seed(42)
    n_points = 500
    base_patch = {
        'points': np.random.rand(n_points, 3) * 50,
        'labels': np.random.randint(0, 5, n_points),
        'intensity': np.random.rand(n_points),
        'rgb': np.random.rand(n_points, 3),
    }
    
    # Test different num_augmentations values
    test_cases = [1, 3, 5, 10]
    
    for num_aug in test_cases:
        config = AugmentationConfig()
        all_patches = create_patch_versions(
            base_patches=[base_patch],
            num_augmentations=num_aug,
            augment_config=config
        )
        
        expected_total = 1 + num_aug  # 1 original + num_aug augmented
        print(f"\n✓ num_augmentations={num_aug}:")
        print(f"  Total patches: {len(all_patches)} (expected: {expected_total})")
        assert len(all_patches) == expected_total
        
        # Count versions
        originals = [p for p in all_patches if p['_version'] == 'original']
        augmented = [p for p in all_patches if 'aug' in p['_version']]
        
        print(f"  Original versions: {len(originals)}")
        print(f"  Augmented versions: {len(augmented)}")
        
        assert len(originals) == 1, "Should have exactly 1 original"
        assert len(augmented) == num_aug, f"Should have exactly {num_aug} augmented versions"
        
        # Verify all augmented versions are unique
        for i in range(len(augmented)):
            for j in range(i + 1, len(augmented)):
                p1 = augmented[i]
                p2 = augmented[j]
                
                # Check point counts differ or coordinates differ
                if len(p1['points']) == len(p2['points']):
                    # Same count, check if coordinates differ
                    diff = np.abs(p1['points'] - p2['points']).mean()
                    assert diff > 0.001, \
                        f"Augmented versions {i} and {j} are too similar (diff={diff:.6f})"
                
        print(f"  ✓ All {num_aug} augmented versions are unique")
    
    print("\n✓ TEST PASSED: Multiple augmentations create distinct versions\n")


def test_augmentation_config_parameters():
    """Test that augmentation config parameters affect the results."""
    print("\n" + "="*70)
    print("TEST: Augmentation config parameters")
    print("="*70)
    
    # Create synthetic data
    np.random.seed(42)
    n_points = 1000
    points = np.random.rand(n_points, 3) * 100
    intensity = np.random.rand(n_points)
    return_number = np.ones(n_points, dtype=np.uint8)
    classification = np.random.randint(0, 10, n_points, dtype=np.uint8)
    
    # Test with different configs
    configs = {
        'default': AugmentationConfig(),
        'no_dropout': AugmentationConfig(dropout_range=(0.0, 0.0)),
        'high_jitter': AugmentationConfig(jitter_sigma=0.5),
        'large_rotation': AugmentationConfig(rotation_range=np.pi),
        'scale_variation': AugmentationConfig(scale_range=(0.5, 1.5)),
    }
    
    results = {}
    
    for name, config in configs.items():
        aug_points, _, _, _, _, _, _ = augment_raw_points(
            points=points.copy(),
            intensity=intensity.copy(),
            return_number=return_number.copy(),
            classification=classification.copy(),
            config=config
        )
        results[name] = {
            'count': len(aug_points),
            'mean': aug_points.mean(axis=0),
            'std': aug_points.std(axis=0),
        }
    
    print("\n✓ Augmentation results with different configs:")
    for name, result in results.items():
        print(f"\n  {name}:")
        print(f"    Points: {result['count']:,} / {n_points:,} (retention: {result['count']/n_points*100:.1f}%)")
        print(f"    Mean: [{result['mean'][0]:.2f}, {result['mean'][1]:.2f}, {result['mean'][2]:.2f}]")
        print(f"    Std:  [{result['std'][0]:.2f}, {result['std'][1]:.2f}, {result['std'][2]:.2f}]")
    
    # Verify no_dropout config keeps all points
    assert results['no_dropout']['count'] == n_points, \
        "no_dropout config should keep all points"
    
    # Verify default config has dropout
    assert results['default']['count'] < n_points, \
        "default config should have some dropout"
    
    print("\n✓ TEST PASSED: Config parameters affect augmentation results\n")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("AUGMENTATION VERIFICATION TEST SUITE")
    print("="*70)
    print("\nThis test suite verifies that:")
    print("1. augment_raw_points creates different versions each time")
    print("2. augment_patch creates different versions each time")
    print("3. create_patch_versions correctly generates all versions")
    print("4. num_augmentations > 1 produces multiple distinct versions")
    print("5. Augmentation config parameters affect the results")
    print("="*70)
    
    try:
        test_augment_raw_points_creates_different_versions()
        test_augment_patch_creates_different_versions()
        test_create_patch_versions_with_augmentation()
        test_augmentation_with_num_greater_than_one()
        test_augmentation_config_parameters()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)
        print("\nSummary:")
        print("✓ Augmentation creates unique versions each time")
        print("✓ Multiple augmentations (num > 1) work correctly")
        print("✓ All versions are properly labeled and tracked")
        print("✓ Configuration parameters are respected")
        print("="*70 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        raise
