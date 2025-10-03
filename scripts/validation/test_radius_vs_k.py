#!/usr/bin/env python3
"""
Test comparison: k-neighbors vs radius-based geometric features

This script demonstrates the difference between:
1. Fixed k-neighbors (captures LIDAR scan patterns → artifacts)
2. Radius-based search (captures true geometry → clean results)

Usage:
    python scripts/validation/test_radius_vs_k.py <file.laz>
"""

import sys
from pathlib import Path
import numpy as np
import laspy
from sklearn.neighbors import KDTree


def test_k_neighbors(points, k=50):
    """Test with fixed k-neighbors (old method, creates artifacts)."""
    print(f"\n{'='*70}")
    print(f"TEST 1: Fixed k-neighbors (k={k})")
    print(f"{'='*70}")
    
    tree = KDTree(points)
    distances, indices = tree.query(points, k=k)
    
    # Sample 1000 points
    sample_size = min(1000, len(points))
    sample_idx = np.random.choice(len(points), sample_size, replace=False)
    
    linearity_list = []
    planarity_list = []
    
    for i in sample_idx:
        neighbors = points[indices[i]]
        
        # Center and compute covariance
        centroid = neighbors.mean(axis=0)
        centered = neighbors - centroid
        cov = (centered.T @ centered) / (k - 1)
        
        # Eigenvalues
        eigenvals = np.linalg.eigvalsh(cov)
        eigenvals = np.sort(eigenvals)[::-1]
        
        λ0, λ1, λ2 = eigenvals[0], eigenvals[1], eigenvals[2]
        sum_λ = λ0 + λ1 + λ2 + 1e-8
        
        linearity = (λ0 - λ1) / sum_λ
        planarity = (λ1 - λ2) / sum_λ
        
        linearity_list.append(linearity)
        planarity_list.append(planarity)
    
    linearity_arr = np.array(linearity_list)
    planarity_arr = np.array(planarity_list)
    
    print(f"\nLinearity:")
    print(f"  Mean:   {linearity_arr.mean():.3f}")
    print(f"  Std:    {linearity_arr.std():.3f}")
    print(f"  >0.7:   {(linearity_arr > 0.7).sum() / len(linearity_arr) * 100:.1f}%")
    print(f"  ^^ High linearity = SCAN LINE ARTIFACTS (bad!)")
    
    print(f"\nPlanarity:")
    print(f"  Mean:   {planarity_arr.mean():.3f}")
    print(f"  Std:    {planarity_arr.std():.3f}")
    print(f"  >0.7:   {(planarity_arr > 0.7).sum() / len(planarity_arr) * 100:.1f}%")
    
    return linearity_arr, planarity_arr


def test_radius_based(points, radius=0.75):
    """Test with radius-based search (new method, clean results)."""
    print(f"\n{'='*70}")
    print(f"TEST 2: Radius-based search (r={radius}m)")
    print(f"{'='*70}")
    
    tree = KDTree(points)
    
    # Sample 1000 points
    sample_size = min(1000, len(points))
    sample_idx = np.random.choice(len(points), sample_size, replace=False)
    
    linearity_list = []
    planarity_list = []
    
    for i in sample_idx:
        # Query neighbors within radius
        neighbor_idx = tree.query_radius(points[i:i+1], r=radius)[0]
        
        if len(neighbor_idx) < 3:
            continue
        
        neighbors = points[neighbor_idx]
        
        # Center and compute covariance
        centroid = neighbors.mean(axis=0)
        centered = neighbors - centroid
        cov = (centered.T @ centered) / (len(neighbors) - 1)
        
        # Eigenvalues
        eigenvals = np.linalg.eigvalsh(cov)
        eigenvals = np.sort(eigenvals)[::-1]
        
        λ0, λ1, λ2 = eigenvals[0], eigenvals[1], eigenvals[2]
        sum_λ = λ0 + λ1 + λ2 + 1e-8
        
        linearity = (λ0 - λ1) / sum_λ
        planarity = (λ1 - λ2) / sum_λ
        
        linearity_list.append(linearity)
        planarity_list.append(planarity)
    
    linearity_arr = np.array(linearity_list)
    planarity_arr = np.array(planarity_list)
    
    print(f"\nLinearity:")
    print(f"  Mean:   {linearity_arr.mean():.3f}")
    print(f"  Std:    {linearity_arr.std():.3f}")
    print(f"  >0.7:   {(linearity_arr > 0.7).sum() / len(linearity_arr) * 100:.1f}%")
    print(f"  ^^ Should be LOW for building surfaces (good!)")
    
    print(f"\nPlanarity:")
    print(f"  Mean:   {planarity_arr.mean():.3f}")
    print(f"  Std:    {planarity_arr.std():.3f}")
    print(f"  >0.7:   {(planarity_arr > 0.7).sum() / len(planarity_arr) * 100:.1f}%")
    print(f"  ^^ Should be HIGH for building surfaces (good!)")
    
    return linearity_arr, planarity_arr


def main(laz_file: Path):
    """Run comparison test."""
    print(f"\n{'='*70}")
    print(f"Testing: {laz_file.name}")
    print(f"{'='*70}")
    
    # Load LAZ
    print("\nLoading LAZ file...")
    las = laspy.read(str(laz_file))
    points = np.vstack([las.x, las.y, las.z]).T
    
    print(f"Points loaded: {len(points):,}")
    
    # Filter building points if classification available
    if hasattr(las, 'classification'):
        building_mask = las.classification == 6
        if building_mask.sum() > 100:
            points = points[building_mask]
            print(f"Building points: {len(points):,}")
    
    # Test 1: k-neighbors (old method)
    lin_k, plan_k = test_k_neighbors(points, k=50)
    
    # Test 2: radius-based (new method)
    lin_r, plan_r = test_radius_based(points, radius=0.75)
    
    # Comparison
    print(f"\n{'='*70}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nLinearity:")
    print(f"  k-neighbors:  {lin_k.mean():.3f} ± {lin_k.std():.3f}")
    print(f"  radius-based: {lin_r.mean():.3f} ± {lin_r.std():.3f}")
    delta_lin = lin_k.mean() - lin_r.mean()
    if delta_lin > 0.3:
        print(f"  ✅ IMPROVEMENT: {delta_lin:.3f} reduction (fewer artifacts!)")
    else:
        print(f"  ⚠️  Small difference: {delta_lin:.3f}")
    
    print(f"\nPlanarity:")
    print(f"  k-neighbors:  {plan_k.mean():.3f} ± {plan_k.std():.3f}")
    print(f"  radius-based: {plan_r.mean():.3f} ± {plan_r.std():.3f}")
    delta_plan = plan_r.mean() - plan_k.mean()
    if delta_plan > 0.2:
        print(f"  ✅ IMPROVEMENT: {delta_plan:.3f} increase (better surfaces!)")
    else:
        print(f"  ⚠️  Small difference: {delta_plan:.3f}")
    
    print(f"\n{'='*70}")
    print(f"CONCLUSION")
    print(f"{'='*70}")
    
    if delta_lin > 0.3 and delta_plan > 0.2:
        print("\n✅ Radius-based search is CLEARLY BETTER!")
        print("   - Reduces false linearity from scan artifacts")
        print("   - Increases planarity on true surfaces")
        print("   - Recommended for building extraction")
    elif delta_lin > 0.1:
        print("\n✅ Radius-based search shows improvement")
        print("   - Some reduction in scan artifacts")
        print("   - May work better on denser point clouds")
    else:
        print("\n⚠️  Results similar between methods")
        print("   - Point cloud may be too sparse")
        print("   - Try adjusting radius parameter")
    
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_radius_vs_k.py <file.laz>")
        sys.exit(1)
    
    laz_file = Path(sys.argv[1])
    if not laz_file.exists():
        print(f"Error: File not found: {laz_file}")
        sys.exit(1)
    
    main(laz_file)
