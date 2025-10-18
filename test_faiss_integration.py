#!/usr/bin/env python
"""
Quick test to verify FAISS integration works in compute_all_features_chunked
"""
import sys
sys.path.insert(0, '.')

# Test FAISS detection
from ign_lidar.features import features_gpu_chunked

print("=" * 70)
print("FAISS Integration Test")
print("=" * 70)
print(f"✓ FAISS_AVAILABLE: {features_gpu_chunked.FAISS_AVAILABLE}")
print(f"✓ CUML_AVAILABLE: {features_gpu_chunked.CUML_AVAILABLE}")
print(f"✓ GPU_AVAILABLE: {features_gpu_chunked.GPU_AVAILABLE}")

if features_gpu_chunked.FAISS_AVAILABLE:
    import faiss
    print(f"✓ FAISS version: {faiss.__version__}")
    print(f"✓ FAISS GPU count: {faiss.get_num_gpus()}")
    
    # Check if the _build_faiss_index method exists
    from ign_lidar.features.features_gpu_chunked import GPUChunkedFeatureComputer
    computer = GPUChunkedFeatureComputer(use_gpu=True)
    
    if hasattr(computer, '_build_faiss_index'):
        print("✓ _build_faiss_index method exists")
    else:
        print("✗ _build_faiss_index method NOT FOUND!")
        sys.exit(1)
    
    # Quick functional test with small data
    import numpy as np
    print("\n" + "=" * 70)
    print("Functional Test (1000 points)")
    print("=" * 70)
    
    test_points = np.random.randn(1000, 3).astype(np.float32)
    test_classification = np.ones(1000, dtype=np.uint8)
    
    try:
        import time
        start = time.time()
        normals, curvature, height, geo_features = computer.compute_all_features_chunked(
            test_points, test_classification, k=10, mode='asprs_classes'
        )
        elapsed = time.time() - start
        
        print(f"✓ Feature computation successful!")
        print(f"  - Normals shape: {normals.shape}")
        print(f"  - Curvature shape: {curvature.shape}")
        print(f"  - Height shape: {height.shape}")
        print(f"  - Geo features: {list(geo_features.keys())}")
        print(f"  - Time: {elapsed:.3f}s")
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED - FAISS is ready!")
        print("=" * 70)
        
    except Exception as e:
        print(f"✗ Feature computation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print("✗ FAISS not available - install with: conda install -c pytorch faiss-gpu")
    sys.exit(1)
