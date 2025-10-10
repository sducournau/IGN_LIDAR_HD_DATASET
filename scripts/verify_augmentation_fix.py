"""
Test script to verify augmentation fix - ensures same patches represent same spatial regions
"""

import numpy as np
from pathlib import Path
from collections import Counter
import argparse


def load_patch(filepath):
    """Load patch data from NPZ file"""
    data = np.load(filepath)
    return {
        'points': data['points'],
        'labels': data.get('labels', None),
        'intensity': data.get('intensity', None)
    }


def compute_bbox(points):
    """Compute bounding box of points"""
    return points.min(axis=0), points.max(axis=0)


def compute_centroid(points):
    """Compute centroid of points"""
    return points.mean(axis=0)


def unrotate_patch(points, reference_centroid=None):
    """
    Approximate unrotation by aligning principal axes
    Returns approximate original orientation
    """
    # Center points
    centroid = points.mean(axis=0)
    centered = points - centroid
    
    # Compute covariance matrix (XY only for horizontal rotation)
    cov = np.cov(centered[:, :2].T)
    
    # Get principal axis angle
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    principal_angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    
    # Rotation matrix to align with X-axis
    cos_a, sin_a = np.cos(-principal_angle), np.sin(-principal_angle)
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    
    # Apply rotation
    unrotated = centered @ rotation_matrix.T
    
    # Recenter at reference if provided
    if reference_centroid is not None:
        unrotated += reference_centroid
    else:
        unrotated += centroid
    
    return unrotated


def compare_patches(original_path, augmented_paths):
    """
    Compare original patch with its augmented versions
    
    Returns dict with comparison metrics
    """
    # Load patches
    original = load_patch(original_path)
    augmented = [load_patch(p) for p in augmented_paths]
    
    print(f"\n{'='*70}")
    print(f"Comparing: {original_path.name}")
    print(f"{'='*70}")
    
    results = {
        'original_path': str(original_path),
        'augmented_paths': [str(p) for p in augmented_paths],
        'metrics': []
    }
    
    # Original stats
    orig_points = original['points']
    orig_labels = original['labels']
    orig_centroid = compute_centroid(orig_points)
    orig_bbox_min, orig_bbox_max = compute_bbox(orig_points)
    orig_label_dist = Counter(orig_labels) if orig_labels is not None else {}
    
    print(f"\nOriginal Patch:")
    print(f"  Points: {len(orig_points):,}")
    print(f"  Centroid: ({orig_centroid[0]:.2f}, {orig_centroid[1]:.2f}, {orig_centroid[2]:.2f})")
    print(f"  Bbox: ({orig_bbox_min[0]:.2f}, {orig_bbox_min[1]:.2f}) to "
          f"({orig_bbox_max[0]:.2f}, {orig_bbox_max[1]:.2f})")
    if orig_label_dist:
        print(f"  Label distribution: {dict(orig_label_dist)}")
    
    # Compare with each augmented version
    for idx, aug_patch in enumerate(augmented):
        aug_points = aug_patch['points']
        aug_labels = aug_patch['labels']
        aug_centroid = compute_centroid(aug_points)
        aug_bbox_min, aug_bbox_max = compute_bbox(aug_points)
        aug_label_dist = Counter(aug_labels) if aug_labels is not None else {}
        
        # Unrotate augmented patch to compare spatial location
        aug_unrotated = unrotate_patch(aug_points, reference_centroid=orig_centroid)
        unrot_centroid = compute_centroid(aug_unrotated)
        unrot_bbox_min, unrot_bbox_max = compute_bbox(aug_unrotated)
        
        # Compute metrics
        point_count_ratio = len(aug_points) / len(orig_points)
        centroid_distance = np.linalg.norm(unrot_centroid - orig_centroid)
        
        # Label distribution similarity (Jaccard coefficient)
        if orig_label_dist and aug_label_dist:
            all_classes = set(orig_label_dist.keys()) | set(aug_label_dist.keys())
            orig_total = sum(orig_label_dist.values())
            aug_total = sum(aug_label_dist.values())
            
            # Normalized distributions
            orig_norm = {c: orig_label_dist.get(c, 0) / orig_total for c in all_classes}
            aug_norm = {c: aug_label_dist.get(c, 0) / aug_total for c in all_classes}
            
            # Compute cosine similarity
            dot_product = sum(orig_norm[c] * aug_norm[c] for c in all_classes)
            label_similarity = dot_product  # Already normalized
        else:
            label_similarity = None
        
        print(f"\nAugmented Version {idx}:")
        print(f"  Points: {len(aug_points):,} ({point_count_ratio:.2%} of original)")
        print(f"  Centroid (raw): ({aug_centroid[0]:.2f}, {aug_centroid[1]:.2f}, {aug_centroid[2]:.2f})")
        print(f"  Centroid (unrotated): ({unrot_centroid[0]:.2f}, {unrot_centroid[1]:.2f}, {unrot_centroid[2]:.2f})")
        print(f"  Centroid distance from original: {centroid_distance:.2f}m")
        if aug_label_dist:
            print(f"  Label distribution: {dict(aug_label_dist)}")
            if label_similarity is not None:
                print(f"  Label similarity: {label_similarity:.2%}")
        
        # Assess results
        issues = []
        if point_count_ratio < 0.85 or point_count_ratio > 1.0:
            issues.append(f"Point count unusual: {point_count_ratio:.2%}")
        if centroid_distance > 50:  # More than 50m away is suspicious
            issues.append(f"Centroid too far: {centroid_distance:.2f}m")
        if label_similarity is not None and label_similarity < 0.7:
            issues.append(f"Label distribution too different: {label_similarity:.2%}")
        
        status = "✅ PASS" if not issues else "⚠️  WARN"
        print(f"\n  {status}")
        if issues:
            for issue in issues:
                print(f"    - {issue}")
        
        results['metrics'].append({
            'version': idx,
            'point_count': len(aug_points),
            'point_count_ratio': point_count_ratio,
            'centroid_distance': centroid_distance,
            'label_similarity': label_similarity,
            'status': 'pass' if not issues else 'warn',
            'issues': issues
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Verify augmentation fix: check if augmented patches represent same spatial regions'
    )
    parser.add_argument(
        'patch_dir',
        type=Path,
        help='Directory containing patch files'
    )
    parser.add_argument(
        '--patch-prefix',
        type=str,
        default=None,
        help='Check specific patch prefix (e.g., "urban_dense_patch_0000")'
    )
    parser.add_argument(
        '--max-patches',
        type=int,
        default=5,
        help='Maximum number of patches to check (default: 5)'
    )
    
    args = parser.parse_args()
    
    patch_dir = Path(args.patch_dir)
    if not patch_dir.exists():
        print(f"Error: Directory not found: {patch_dir}")
        return
    
    # Find original patches
    if args.patch_prefix:
        original_patches = list(patch_dir.glob(f"{args.patch_prefix}.npz"))
    else:
        original_patches = [
            p for p in patch_dir.glob("*_patch_*.npz")
            if not any(x in p.stem for x in ['aug_'])
        ]
    
    original_patches = sorted(original_patches)[:args.max_patches]
    
    if not original_patches:
        print(f"No patches found in {patch_dir}")
        return
    
    print(f"\nFound {len(original_patches)} original patches to verify")
    
    # Check each patch
    all_results = []
    for orig_patch in original_patches:
        # Find augmented versions
        stem = orig_patch.stem
        aug_patterns = [
            f"{stem}_aug_*.npz"
        ]
        
        aug_patches = []
        for pattern in aug_patterns:
            aug_patches.extend(patch_dir.glob(pattern))
        
        aug_patches = sorted(aug_patches)
        
        if not aug_patches:
            print(f"\n⚠️  No augmented versions found for {orig_patch.name}")
            continue
        
        # Compare
        results = compare_patches(orig_patch, aug_patches)
        all_results.append(results)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    total_comparisons = sum(len(r['metrics']) for r in all_results)
    passed = sum(
        1 for r in all_results
        for m in r['metrics']
        if m['status'] == 'pass'
    )
    warned = total_comparisons - passed
    
    print(f"Total patches checked: {len(all_results)}")
    print(f"Total augmented versions: {total_comparisons}")
    print(f"✅ Passed: {passed}")
    print(f"⚠️  Warnings: {warned}")
    
    if warned > 0:
        print(f"\n⚠️  Some augmented patches may not represent the same spatial region!")
        print(f"   This could indicate the augmentation fix was not applied.")
    else:
        print(f"\n✅ All checks passed! Augmented patches represent same spatial regions.")


if __name__ == '__main__':
    main()
