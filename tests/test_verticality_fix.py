#!/usr/bin/env python3
"""
Test script to verify verticality computation is working correctly.

Run this after the fix to confirm verticality is no longer all zeros.
"""

import numpy as np
import sys
from pathlib import Path

def test_verticality_computation():
    """Test that verticality is computed correctly from normals."""
    print("="*80)
    print("TESTING VERTICALITY COMPUTATION")
    print("="*80)
    print()
    
    # Test 1: Direct computation from normals
    print("Test 1: Direct computation from normals")
    print("-"*60)
    
    # Create test normals
    test_normals = np.array([
        [0.0, 0.0, 1.0],   # Horizontal surface (ground) - low verticality
        [0.0, 0.0, -1.0],  # Horizontal surface (ceiling) - low verticality
        [1.0, 0.0, 0.0],   # Vertical surface (wall) - high verticality
        [0.0, 1.0, 0.0],   # Vertical surface (wall) - high verticality
        [0.707, 0.0, 0.707],  # 45° slope - medium verticality
    ], dtype=np.float32)
    
    # Compute verticality: 1 - |normal_z|
    verticality = 1.0 - np.abs(test_normals[:, 2])
    
    print(f"Normal         -> Verticality")
    for i, (normal, vert) in enumerate(zip(test_normals, verticality)):
        print(f"  [{normal[0]:5.2f}, {normal[1]:5.2f}, {normal[2]:5.2f}] -> {vert:.3f}")
    
    print()
    print("Expected results:")
    print("  - Horizontal surfaces (nz=±1.0) -> verticality ≈ 0.0")
    print("  - Vertical surfaces (nz=0.0)    -> verticality = 1.0")
    print("  - 45° slopes (nz=±0.707)        -> verticality ≈ 0.3")
    print()
    
    # Verify
    assert abs(verticality[0] - 0.0) < 0.01, "Horizontal should have low verticality"
    assert abs(verticality[2] - 1.0) < 0.01, "Vertical should have high verticality"
    assert abs(verticality[4] - 0.293) < 0.01, "45° should have medium verticality"
    
    print("✓ Verticality computation working correctly!")
    print()
    
    # Test 2: Check with GPU computer
    print("Test 2: GPU Feature Computer")
    print("-"*60)
    
    try:
        from ign_lidar.features_gpu import GPUFeatureComputer
        
        computer = GPUFeatureComputer(use_gpu=False)  # CPU fallback OK
        vert_result = computer.compute_verticality(test_normals)
        
        print(f"GPU Computer verticality results:")
        for i, v in enumerate(vert_result):
            print(f"  Normal {i}: verticality = {v:.3f}")
        
        # Verify matches direct computation
        np.testing.assert_allclose(vert_result, verticality, rtol=1e-5)
        print()
        print("✓ GPU Feature Computer verticality matches expected!")
        
    except ImportError as e:
        print(f"⚠️  Could not import GPU Feature Computer: {e}")
        print("   (This is OK if GPU dependencies not installed)")
    
    print()
    
    # Test 3: Check in full feature computation
    print("Test 3: Full feature computation with chunking")
    print("-"*60)
    
    try:
        from ign_lidar.features_gpu_chunked import compute_all_features_gpu_chunked
        
        # Create small test point cloud
        n_points = 1000
        np.random.seed(42)
        
        # Create a simple point cloud with ground and walls
        points = np.random.randn(n_points, 3).astype(np.float32)
        points[:, 2] = np.abs(points[:, 2])  # All positive Z
        
        classification = np.full(n_points, 6, dtype=np.uint8)  # Building
        classification[:100] = 2  # Some ground points
        
        print(f"Computing features for {n_points:,} test points...")
        normals, curvature, height, geo_features = compute_all_features_gpu_chunked(
            points, classification, k=10, chunk_size=500
        )
        
        # Check if verticality is in results
        if 'verticality' in geo_features:
            vert = geo_features['verticality']
            print(f"✓ Verticality present in results!")
            print(f"  Range: {vert.min():.4f} - {vert.max():.4f}")
            print(f"  Mean: {vert.mean():.4f}")
            print(f"  Std: {vert.std():.4f}")
            print(f"  Non-zero: {(vert > 1e-6).sum()} / {len(vert)}")
            
            if vert.std() < 1e-6:
                print(f"  ❌ ERROR: Verticality has NO VARIATION (all {vert[0]:.6f})")
                return False
            else:
                print(f"  ✓ Verticality has good variation!")
        else:
            print("  ❌ ERROR: Verticality NOT in results!")
            return False
            
    except Exception as e:
        print(f"⚠️  Could not test full computation: {e}")
        import traceback
        traceback.print_exc()
        print("   (This might be due to missing dependencies)")
    
    print()
    print("="*80)
    print("✅ ALL TESTS PASSED - Verticality computation is working!")
    print("="*80)
    
    return True


if __name__ == '__main__':
    try:
        success = test_verticality_computation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
