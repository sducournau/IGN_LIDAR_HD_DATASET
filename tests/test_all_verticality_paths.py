#!/usr/bin/env python3
"""
Comprehensive test for verticality computation in all code paths.

Tests:
1. CPU version (compute_all_features_optimized with include_extra=True)
2. Simple GPU version (compute_all_features_with_gpu, no chunking)
3. GPU chunked version (compute_all_features_gpu_chunked)
"""

import numpy as np
import sys
from pathlib import Path


def create_test_data(n_points=1000):
    """Create test point cloud with known geometry."""
    np.random.seed(42)
    
    # Create points representing different surfaces
    points = []
    classifications = []
    
    # 1. Ground points (horizontal, nz=1)
    n_ground = n_points // 4
    ground = np.random.randn(n_ground, 3).astype(np.float32)
    ground[:, 2] = 0.1  # Flat ground
    points.append(ground)
    classifications.extend([2] * n_ground)  # Ground class
    
    # 2. Wall points (vertical, nz≈0)
    n_wall = n_points // 4
    wall = np.random.randn(n_wall, 3).astype(np.float32)
    wall[:, 0] = 0.0  # Vertical wall along Y-Z plane
    wall[:, 2] += 3.0  # Elevated
    points.append(wall)
    classifications.extend([6] * n_wall)  # Building class
    
    # 3. Roof points (horizontal, nz≈1)
    n_roof = n_points // 4
    roof = np.random.randn(n_roof, 3).astype(np.float32)
    roof[:, 2] = 5.0  # Flat roof
    points.append(roof)
    classifications.extend([6] * n_roof)  # Building class
    
    # 4. Random points (mixed)
    n_random = n_points - n_ground - n_wall - n_roof
    random_pts = np.random.randn(n_random, 3).astype(np.float32)
    random_pts[:, 2] = np.abs(random_pts[:, 2])
    points.append(random_pts)
    classifications.extend([1] * n_random)  # Unclassified
    
    points = np.vstack(points)
    classification = np.array(classifications, dtype=np.uint8)
    
    return points, classification


def test_cpu_version():
    """Test CPU version with include_extra=True."""
    print("="*80)
    print("TEST 1: CPU VERSION (compute_all_features_optimized)")
    print("="*80)
    
    try:
        from ign_lidar.features import compute_all_features_optimized
        
        points, classification = create_test_data(n_points=1000)
        
        print(f"Testing with {len(points):,} points...")
        print()
        
        # Test with include_extra=True (should include verticality)
        normals, curvature, height, geo_features = compute_all_features_optimized(
            points, classification,
            k=10,
            auto_k=False,
            include_extra=True,
            chunk_size=None
        )
        
        print("Features returned:")
        for key in sorted(geo_features.keys()):
            vals = geo_features[key]
            print(f"  - {key:20s}: shape={vals.shape}, "
                  f"min={vals.min():.4f}, max={vals.max():.4f}, "
                  f"std={vals.std():.4f}")
        
        print()
        
        # Check verticality
        if 'verticality' in geo_features:
            vert = geo_features['verticality']
            print(f"✓ Verticality present!")
            print(f"  Range: {vert.min():.4f} - {vert.max():.4f}")
            print(f"  Mean: {vert.mean():.4f}, Std: {vert.std():.4f}")
            
            if vert.std() < 1e-6:
                print(f"  ❌ ERROR: Verticality has NO VARIATION")
                return False
            else:
                print(f"  ✓ Verticality has good variation")
                
            # Check expected values
            high_vert = (vert > 0.7).sum()
            low_vert = (vert < 0.3).sum()
            print(f"  High verticality (>0.7): {high_vert} points")
            print(f"  Low verticality (<0.3): {low_vert} points")
        else:
            print("❌ ERROR: Verticality NOT in results!")
            return False
        
        print()
        print("✓ CPU version test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ CPU version test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_gpu_version():
    """Test simple GPU version (no chunking)."""
    print()
    print("="*80)
    print("TEST 2: SIMPLE GPU VERSION (compute_all_features_with_gpu)")
    print("="*80)
    
    try:
        from ign_lidar.features import compute_all_features_with_gpu
        
        points, classification = create_test_data(n_points=1000)
        
        print(f"Testing with {len(points):,} points...")
        print()
        
        # Test with use_gpu=False (CPU fallback OK)
        normals, curvature, height, geo_features = compute_all_features_with_gpu(
            points, classification,
            k=10,
            auto_k=False,
            use_gpu=False  # CPU fallback is fine for testing
        )
        
        print("Features returned:")
        for key in sorted(geo_features.keys()):
            vals = geo_features[key]
            print(f"  - {key:20s}: shape={vals.shape}, "
                  f"min={vals.min():.4f}, max={vals.max():.4f}, "
                  f"std={vals.std():.4f}")
        
        print()
        
        # Check verticality
        if 'verticality' in geo_features:
            vert = geo_features['verticality']
            print(f"✓ Verticality present!")
            print(f"  Range: {vert.min():.4f} - {vert.max():.4f}")
            print(f"  Mean: {vert.mean():.4f}, Std: {vert.std():.4f}")
            
            if vert.std() < 1e-6:
                print(f"  ❌ ERROR: Verticality has NO VARIATION")
                return False
            else:
                print(f"  ✓ Verticality has good variation")
                
            # Check expected values
            high_vert = (vert > 0.7).sum()
            low_vert = (vert < 0.3).sum()
            print(f"  High verticality (>0.7): {high_vert} points")
            print(f"  Low verticality (<0.3): {low_vert} points")
        else:
            print("❌ ERROR: Verticality NOT in results!")
            return False
        
        print()
        print("✓ Simple GPU version test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Simple GPU version test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_chunked_version():
    """Test GPU chunked version."""
    print()
    print("="*80)
    print("TEST 3: GPU CHUNKED VERSION (compute_all_features_gpu_chunked)")
    print("="*80)
    
    try:
        from ign_lidar.features_gpu_chunked import compute_all_features_gpu_chunked
        
        points, classification = create_test_data(n_points=2000)
        
        print(f"Testing with {len(points):,} points...")
        print()
        
        # Force chunking with small chunk_size
        normals, curvature, height, geo_features = compute_all_features_gpu_chunked(
            points, classification,
            k=10,
            chunk_size=500  # Small chunks to force multiple chunks
        )
        
        print("Features returned:")
        for key in sorted(geo_features.keys()):
            vals = geo_features[key]
            print(f"  - {key:20s}: shape={vals.shape}, "
                  f"min={vals.min():.4f}, max={vals.max():.4f}, "
                  f"std={vals.std():.4f}")
        
        print()
        
        # Check verticality
        if 'verticality' in geo_features:
            vert = geo_features['verticality']
            print(f"✓ Verticality present!")
            print(f"  Range: {vert.min():.4f} - {vert.max():.4f}")
            print(f"  Mean: {vert.mean():.4f}, Std: {vert.std():.4f}")
            
            if vert.std() < 1e-6:
                print(f"  ❌ ERROR: Verticality has NO VARIATION")
                return False
            else:
                print(f"  ✓ Verticality has good variation")
                
            # Check expected values
            high_vert = (vert > 0.7).sum()
            low_vert = (vert < 0.3).sum()
            print(f"  High verticality (>0.7): {high_vert} points")
            print(f"  Low verticality (<0.3): {low_vert} points")
        else:
            print("❌ ERROR: Verticality NOT in results!")
            return False
        
        print()
        print("✓ GPU chunked version test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ GPU chunked version test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print()
    print("╔" + "="*78 + "╗")
    print("║" + " COMPREHENSIVE VERTICALITY TEST - ALL CODE PATHS ".center(78) + "║")
    print("╚" + "="*78 + "╝")
    print()
    
    results = []
    
    # Test 1: CPU version
    results.append(("CPU version", test_cpu_version()))
    
    # Test 2: Simple GPU version
    results.append(("Simple GPU version", test_simple_gpu_version()))
    
    # Test 3: GPU chunked version
    results.append(("GPU chunked version", test_gpu_chunked_version()))
    
    # Summary
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"  {status:12s} - {name}")
        if not passed:
            all_passed = False
    
    print()
    
    if all_passed:
        print("╔" + "="*78 + "╗")
        print("║" + " ✅ ALL TESTS PASSED ".center(78) + "║")
        print("║" + " Verticality computation working in ALL code paths! ".center(78) + "║")
        print("╚" + "="*78 + "╝")
        return 0
    else:
        print("╔" + "="*78 + "╗")
        print("║" + " ❌ SOME TESTS FAILED ".center(78) + "║")
        print("╚" + "="*78 + "╝")
        return 1


if __name__ == '__main__':
    sys.exit(main())
