#!/usr/bin/env python3
"""
Verify that LAZ patches correspond between original and augmented versions.

This script checks if augmented patches actually correspond to their original
counterparts by examining spatial bounds and point distributions.
"""

import sys
from pathlib import Path
import numpy as np
import laspy

def load_laz_patch(path: Path):
    """Load a LAZ patch and return XYZ coordinates."""
    las = laspy.read(str(path))
    points = np.vstack([las.x, las.y, las.z]).T
    return points, las

def analyze_patch_pair(orig_path: Path, aug_path: Path):
    """
    Analyze correspondence between original and augmented patches.
    
    Args:
        orig_path: Path to original patch LAZ
        aug_path: Path to augmented patch LAZ
    """
    print(f"\n{'='*80}")
    print(f"Analyzing patch pair:")
    print(f"  Original: {orig_path.name}")
    print(f"  Augmented: {aug_path.name}")
    print(f"{'='*80}\n")
    
    # Load patches
    orig_points, orig_las = load_laz_patch(orig_path)
    aug_points, aug_las = load_laz_patch(aug_path)
    
    # Check point counts
    print(f"Point counts:")
    print(f"  Original:  {len(orig_points):,}")
    print(f"  Augmented: {len(aug_points):,}")
    print(f"  Difference: {len(orig_points) - len(aug_points):,} points")
    
    # Check spatial bounds
    print(f"\nSpatial bounds (X, Y, Z):")
    print(f"  Original:")
    print(f"    X: [{orig_points[:,0].min():.4f}, {orig_points[:,0].max():.4f}]")
    print(f"    Y: [{orig_points[:,1].min():.4f}, {orig_points[:,1].max():.4f}]")
    print(f"    Z: [{orig_points[:,2].min():.4f}, {orig_points[:,2].max():.4f}]")
    print(f"  Augmented:")
    print(f"    X: [{aug_points[:,0].min():.4f}, {aug_points[:,0].max():.4f}]")
    print(f"    Y: [{aug_points[:,1].min():.4f}, {aug_points[:,1].max():.4f}]")
    print(f"    Z: [{aug_points[:,2].min():.4f}, {aug_points[:,2].max():.4f}]")
    
    # Check if bounds overlap
    orig_bbox = [
        orig_points[:,0].min(), orig_points[:,0].max(),
        orig_points[:,1].min(), orig_points[:,1].max()
    ]
    aug_bbox = [
        aug_points[:,0].min(), aug_points[:,0].max(),
        aug_points[:,1].min(), aug_points[:,1].max()
    ]
    
    # Calculate overlap
    x_overlap = (min(orig_bbox[1], aug_bbox[1]) - max(orig_bbox[0], aug_bbox[0])) / (orig_bbox[1] - orig_bbox[0])
    y_overlap = (min(orig_bbox[3], aug_bbox[3]) - max(orig_bbox[2], aug_bbox[2])) / (orig_bbox[3] - orig_bbox[2])
    
    print(f"\nBounding box overlap:")
    print(f"  X overlap: {x_overlap*100:.1f}%")
    print(f"  Y overlap: {y_overlap*100:.1f}%")
    
    if x_overlap < 0.5 or y_overlap < 0.5:
        print(f"\n⚠️  WARNING: Low spatial overlap!")
        print(f"  The augmented patch appears to cover a DIFFERENT spatial region.")
        print(f"  This indicates that patches are extracted from the WRONG location after augmentation.")
        return False
    
    # Check if points are identical (they shouldn't be due to augmentation)
    if np.allclose(orig_points, aug_points, atol=0.01):
        print(f"\n⚠️  WARNING: Points are identical!")
        print(f"  Augmentation may not have been applied.")
        return False
    
    # Calculate center of mass to check if same "object" is captured
    orig_com = orig_points.mean(axis=0)
    aug_com = aug_points.mean(axis=0)
    com_distance = np.linalg.norm(orig_com - aug_com)
    
    print(f"\nCenter of mass:")
    print(f"  Original:  [{orig_com[0]:.4f}, {orig_com[1]:.4f}, {orig_com[2]:.4f}]")
    print(f"  Augmented: [{aug_com[0]:.4f}, {aug_com[1]:.4f}, {aug_com[2]:.4f}]")
    print(f"  Distance: {com_distance:.4f}")
    
    # For normalized patches (centered at origin), com should be near origin
    orig_com_magnitude = np.linalg.norm(orig_com)
    aug_com_magnitude = np.linalg.norm(aug_com)
    
    print(f"\nCenter of mass magnitude (should be ~0 for normalized patches):")
    print(f"  Original:  {orig_com_magnitude:.4f}")
    print(f"  Augmented: {aug_com_magnitude:.4f}")
    
    # Check classification distribution
    if hasattr(orig_las, 'classification') and hasattr(aug_las, 'classification'):
        orig_classes = np.bincount(orig_las.classification, minlength=32)
        aug_classes = np.bincount(aug_las.classification, minlength=32)
        
        print(f"\nClassification distribution:")
        for i in range(32):
            if orig_classes[i] > 0 or aug_classes[i] > 0:
                print(f"  Class {i:2d}: Original={orig_classes[i]:5d}, Augmented={aug_classes[i]:5d}")
    
    print(f"\n✓ Analysis complete")
    return True


def main():
    """Main analysis function."""
    # Check for urban_dense directory
    urban_dense_dir = Path("/mnt/c/Users/Simon/ign/patch_1st_training/urban_dense")
    
    if not urban_dense_dir.exists():
        print(f"❌ Directory not found: {urban_dense_dir}")
        return 1
    
    # Find all original patches (not augmented)
    original_patches = sorted([
        p for p in urban_dense_dir.glob("*_patch_*.laz")
        if "_aug_" not in p.name
    ])
    
    if not original_patches:
        print(f"❌ No original patches found in {urban_dense_dir}")
        return 1
    
    print(f"Found {len(original_patches)} original patches")
    
    # Analyze first few patch pairs
    num_to_analyze = min(5, len(original_patches))
    
    issues_found = False
    
    for i, orig_path in enumerate(original_patches[:num_to_analyze]):
        # Find corresponding augmented patch
        # Original: ...patch_0000.laz
        # Augmented: ...patch_0000_aug_0.laz
        aug_path = orig_path.parent / f"{orig_path.stem}_aug_0.laz"
        
        if not aug_path.exists():
            print(f"\n⚠️  Augmented patch not found: {aug_path.name}")
            continue
        
        result = analyze_patch_pair(orig_path, aug_path)
        if not result:
            issues_found = True
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    if issues_found:
        print(f"❌ Issues found: Original and augmented patches do NOT correspond!")
        print(f"\n🔍 Root Cause:")
        print(f"   The augmentation is applied to the FULL TILE before patch extraction.")
        print(f"   Rotation changes the spatial distribution, causing patches to be")
        print(f"   extracted from DIFFERENT spatial regions.")
        print(f"\n💡 Solution:")
        print(f"   Extract patches FIRST, then apply augmentation to EACH patch.")
        print(f"   This ensures augmented patches correspond to their original counterparts.")
        return 1
    else:
        print(f"✓ All patches correspond correctly between original and augmented versions")
        return 0


if __name__ == '__main__':
    sys.exit(main())
