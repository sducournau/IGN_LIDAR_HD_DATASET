"""
Simple Augmentation Verification Demo

Quick demonstration that augmentation is working correctly
when augment=true and num_augmentations > 1.
"""

import numpy as np
from ign_lidar.core.modules.patch_extractor import (
    augment_raw_points,
    create_patch_versions,
    AugmentationConfig
)


def demo_augmentation():
    """Demonstrate that augmentation creates unique versions."""
    print("\n" + "="*70)
    print("AUGMENTATION VERIFICATION DEMO")
    print("="*70)
    print("\nScenario: augment=true, num_augmentations=3")
    print("-"*70)
    
    # Create a simple synthetic patch
    np.random.seed(42)
    n_points = 1000
    base_patch = {
        'points': np.random.rand(n_points, 3) * 50,
        'labels': np.random.randint(0, 5, n_points),
        'intensity': np.random.rand(n_points),
        'rgb': np.random.rand(n_points, 3),
    }
    
    # Generate augmented versions (like the processor does)
    num_augmentations = 3
    config = AugmentationConfig()
    
    all_patches = create_patch_versions(
        base_patches=[base_patch],
        num_augmentations=num_augmentations,
        augment_config=config
    )
    
    print(f"\n📊 Results:")
    print(f"   Base patches: 1")
    print(f"   Augmentations per patch: {num_augmentations}")
    print(f"   Total patches generated: {len(all_patches)}")
    print(f"   Expected: 1 original + {num_augmentations} augmented = {1 + num_augmentations}")
    
    # Separate by version type
    original = [p for p in all_patches if p['_version'] == 'original'][0]
    augmented = [p for p in all_patches if 'aug' in p['_version']]
    
    print(f"\n📋 Version Breakdown:")
    print(f"   Original version: 1 patch")
    print(f"   Augmented versions: {len(augmented)} patches")
    
    print(f"\n🔍 Detailed Analysis:")
    print(f"\n   Original patch:")
    print(f"      Points: {len(original['points']):,}")
    print(f"      Mean XYZ: [{original['points'][:, 0].mean():.3f}, "
          f"{original['points'][:, 1].mean():.3f}, {original['points'][:, 2].mean():.3f}]")
    print(f"      Std XYZ:  [{original['points'][:, 0].std():.3f}, "
          f"{original['points'][:, 1].std():.3f}, {original['points'][:, 2].std():.3f}]")
    
    for i, aug in enumerate(augmented):
        version = aug['_version']
        print(f"\n   Augmented patch ({version}):")
        print(f"      Points: {len(aug['points']):,} "
              f"(dropout: {100 * (1 - len(aug['points'])/len(original['points'])):.1f}%)")
        print(f"      Mean XYZ: [{aug['points'][:, 0].mean():.3f}, "
              f"{aug['points'][:, 1].mean():.3f}, {aug['points'][:, 2].mean():.3f}]")
        print(f"      Std XYZ:  [{aug['points'][:, 0].std():.3f}, "
              f"{aug['points'][:, 1].std():.3f}, {aug['points'][:, 2].std():.3f}]")
        
        # Calculate difference from original
        compare_n = min(len(aug['points']), len(original['points']), 100)
        coord_diff = np.abs(aug['points'][:compare_n] - original['points'][:compare_n]).mean()
        print(f"      Coordinate difference from original: {coord_diff:.4f}")
    
    # Verify uniqueness
    print(f"\n✅ Verification:")
    
    # Check that all have different point counts
    counts = [len(aug['points']) for aug in augmented]
    unique_counts = len(set(counts))
    print(f"   • Different point counts: {unique_counts}/{num_augmentations} "
          f"({'✓ PASS' if unique_counts > 1 else '✗ FAIL'})")
    
    # Check that coordinates are different
    all_different = True
    for i in range(len(augmented)):
        for j in range(i + 1, len(augmented)):
            compare_n = min(len(augmented[i]['points']), len(augmented[j]['points']), 10)
            if compare_n > 0:
                diff = np.abs(augmented[i]['points'][:compare_n] - 
                             augmented[j]['points'][:compare_n]).mean()
                if diff < 0.001:
                    all_different = False
                    break
    
    print(f"   • All versions have different coordinates: {'✓ PASS' if all_different else '✗ FAIL'}")
    
    # Check that augmented versions differ from original
    all_different_from_orig = True
    for aug in augmented:
        compare_n = min(len(aug['points']), len(original['points']), 10)
        if compare_n > 0:
            diff = np.abs(aug['points'][:compare_n] - original['points'][:compare_n]).mean()
            if diff < 0.001:
                all_different_from_orig = False
                break
    
    print(f"   • Augmented versions differ from original: {'✓ PASS' if all_different_from_orig else '✗ FAIL'}")
    
    # Check version labeling
    versions_found = set([p['_version'] for p in all_patches])
    expected_versions = {'original', 'aug_0', 'aug_1', 'aug_2'}
    versions_correct = versions_found == expected_versions
    print(f"   • Correct version labels: {'✓ PASS' if versions_correct else '✗ FAIL'}")
    print(f"     Found: {sorted(versions_found)}")
    print(f"     Expected: {sorted(expected_versions)}")
    
    # Final verdict
    all_pass = unique_counts > 1 and all_different and all_different_from_orig and versions_correct
    
    print(f"\n{'='*70}")
    if all_pass:
        print("✅ VERIFICATION PASSED: Augmentation is working correctly!")
        print("   When augment=true and num_augmentations=3:")
        print("   • 1 original version is created")
        print("   • 3 unique augmented versions are created")
        print("   • All versions have different transformations applied")
        print("   • Total: 4 patches (1 original + 3 augmented)")
    else:
        print("❌ VERIFICATION FAILED: Augmentation may not be working correctly")
    print("="*70 + "\n")
    
    return all_pass


if __name__ == '__main__':
    success = demo_augmentation()
    exit(0 if success else 1)
