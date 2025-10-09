#!/usr/bin/env python3
"""
Quick test to verify GPU acceleration is working with CuML
"""

import numpy as np
import sys

def test_gpu_availability():
    """Test if GPU libraries are available"""
    print("="*70)
    print("GPU AVAILABILITY TEST")
    print("="*70)
    
    # Test CuPy
    try:
        import cupy as cp
        print(f"\n✅ CuPy {cp.__version__} available")
        print(f"   CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
        print(f"   Device: {cp.cuda.Device(0).compute_capability}")
        
        # Quick GPU test
        x = cp.random.randn(1000, 100)
        y = cp.dot(x, x.T)
        cp.cuda.Stream.null.synchronize()
        print(f"   GPU computation test: ✅ PASSED")
    except ImportError as e:
        print(f"\n❌ CuPy not available: {e}")
        return False
    except Exception as e:
        print(f"\n⚠️  CuPy error: {e}")
        return False
    
    # Test CuML
    try:
        import cuml
        from cuml.neighbors import NearestNeighbors
        print(f"\n✅ CuML {cuml.__version__} available")
        
        # Quick CuML test
        X = cp.random.randn(10000, 10, dtype=cp.float32)
        nn = NearestNeighbors(n_neighbors=10)
        nn.fit(X)
        distances, indices = nn.kneighbors(X[:100])
        print(f"   CuML NearestNeighbors test: ✅ PASSED")
    except ImportError as e:
        print(f"\n❌ CuML not available: {e}")
        return False
    except Exception as e:
        print(f"\n⚠️  CuML error: {e}")
        return False
    
    return True


def test_ign_lidar_gpu():
    """Test IGN LiDAR GPU feature computation"""
    print("\n" + "="*70)
    print("IGN LIDAR GPU FEATURE TEST")
    print("="*70)
    
    try:
        from ign_lidar.features.features import compute_all_features_with_gpu
        print("\n✅ Successfully imported compute_all_features_with_gpu")
        
        # Create test data
        n_points = 50000
        points = np.random.randn(n_points, 3).astype(np.float32)
        points[:, 2] = np.abs(points[:, 2]) * 10  # positive heights
        classification = np.random.randint(1, 7, n_points)
        
        print(f"\n📊 Testing with {n_points:,} points")
        
        # Test GPU computation
        import time
        start = time.time()
        normals, curvature, height, geo_features = compute_all_features_with_gpu(
            points=points,
            classification=classification,
            k=10,
            auto_k=False,
            use_gpu=True
        )
        elapsed = time.time() - start
        
        print(f"\n✅ GPU feature computation successful!")
        print(f"   Time: {elapsed:.3f}s ({n_points/elapsed:.0f} pts/s)")
        print(f"   Normals shape: {normals.shape}")
        print(f"   Curvature shape: {curvature.shape}")
        print(f"   Height shape: {height.shape}")
        print(f"   Geometric features: {list(geo_features.keys())}")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Computation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "🔍 VERIFYING GPU SETUP FOR IGN LIDAR HD" + "\n")
    
    # Test 1: GPU availability
    gpu_ok = test_gpu_availability()
    
    if not gpu_ok:
        print("\n" + "="*70)
        print("❌ GPU libraries not available")
        print("="*70)
        sys.exit(1)
    
    # Test 2: IGN LiDAR GPU features
    features_ok = test_ign_lidar_gpu()
    
    # Summary
    print("\n" + "="*70)
    if gpu_ok and features_ok:
        print("✅ ALL TESTS PASSED - GPU IS WORKING!")
        print("="*70)
        print("\nYou can now run processing with:")
        print("  processor=gpu")
        print("  processor.num_workers=1")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        print("="*70)
        sys.exit(1)


if __name__ == "__main__":
    main()
