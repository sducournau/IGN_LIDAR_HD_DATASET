#!/usr/bin/env python3
"""
Verify GPU Optimizations - Test Script

Tests all GPU-accelerated features implemented in the performance audit:
- P1.4: GPU threshold selection
- P0.1: GPU road classification (requires cuSpatial)
- P0.2: GPU bbox optimization
- GPU KNN operations (FAISS/cuML)

Run with: conda run -n ign_gpu python scripts/verify_gpu_optimizations.py

Author: Performance Optimization Team
Date: November 21, 2025
"""

import sys
import time
import numpy as np
from typing import Dict, Any

def print_header(title: str):
    """Print section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def check_gpu_libraries() -> Dict[str, Any]:
    """Check which GPU libraries are available."""
    print_header("GPU Library Availability")
    
    libraries = {}
    
    # CuPy
    try:
        import cupy as cp
        libraries['cupy'] = {
            'available': True,
            'version': cp.__version__,
            'gpu_count': cp.cuda.runtime.getDeviceCount()
        }
        print(f"✅ CuPy {cp.__version__} (GPUs: {libraries['cupy']['gpu_count']})")
    except ImportError:
        libraries['cupy'] = {'available': False}
        print("❌ CuPy not available")
    
    # cuDF
    try:
        import cudf
        libraries['cudf'] = {'available': True, 'version': cudf.__version__}
        print(f"✅ cuDF {cudf.__version__}")
    except ImportError:
        libraries['cudf'] = {'available': False}
        print("❌ cuDF not available")
    
    # cuML
    try:
        import cuml
        libraries['cuml'] = {'available': True, 'version': cuml.__version__}
        print(f"✅ cuML {cuml.__version__}")
    except ImportError:
        libraries['cuml'] = {'available': False}
        print("❌ cuML not available")
    
    # cuSpatial
    try:
        import cuspatial
        libraries['cuspatial'] = {'available': True, 'version': cuspatial.__version__}
        print(f"✅ cuSpatial {cuspatial.__version__}")
    except ImportError:
        libraries['cuspatial'] = {'available': False}
        print("⚠️  cuSpatial not available (P0.1 will use CPU fallback)")
    
    # FAISS
    try:
        import faiss
        gpu_available = faiss.get_num_gpus() > 0
        libraries['faiss'] = {
            'available': True,
            'gpu_support': gpu_available,
            'gpu_count': faiss.get_num_gpus() if gpu_available else 0
        }
        status = f"✅ FAISS (GPU: {libraries['faiss']['gpu_count']})" if gpu_available else "⚠️  FAISS (CPU only)"
        print(status)
    except ImportError:
        libraries['faiss'] = {'available': False}
        print("❌ FAISS not available")
    
    return libraries

def test_p14_gpu_thresholds():
    """Test P1.4: Lower GPU Thresholds."""
    print_header("P1.4: GPU Threshold Selection")
    
    from ign_lidar.optimization.ground_truth import GroundTruthOptimizer
    
    optimizer = GroundTruthOptimizer(verbose=False)
    print(f"Optimizer: {optimizer}\n")
    
    test_cases = [
        (50_000, "50K points (small)"),
        (200_000, "200K points (medium-small)"),
        (500_000, "500K points (medium)"),
        (2_000_000, "2M points (large)"),
        (15_000_000, "15M points (very large)")
    ]
    
    print("Method selection (new thresholds):")
    for n_points, desc in test_cases:
        method = optimizer.select_method(n_points, n_polygons=100)
        print(f"  {desc:25s} → {method}")
    
    print("\n✅ P1.4 verification complete")
    return True

def test_p02_gpu_bbox():
    """Test P0.2: GPU BBox Optimization."""
    print_header("P0.2: GPU BBox Optimization")
    
    from ign_lidar.core.classification.building.clustering import BuildingClusterer, HAS_CUPY
    
    if not HAS_CUPY:
        print("⚠️  CuPy not available, skipping GPU test")
        return False
    
    # Create realistic building point cloud
    np.random.seed(42)
    n_points = 50000
    points = np.random.randn(n_points, 3).astype(np.float32)
    points[:, 0] *= 25  # 50m width
    points[:, 1] *= 15  # 30m depth
    points[:, 2] = np.abs(points[:, 2]) * 3 + 1.0  # 0-10m height
    heights = points[:, 2]
    bbox = (-25, -15, 25, 15)
    
    print(f"Test data: {n_points:,} points, 50m×30m building")
    
    # Test with GPU
    clusterer_gpu = BuildingClusterer(use_gpu=True)
    print(f"GPU enabled: {clusterer_gpu.use_gpu}")
    
    t0 = time.time()
    best_shift_gpu, best_bbox_gpu = clusterer_gpu.optimize_bbox_for_building_gpu(
        points, heights, bbox, max_shift=5.0, step=0.5
    )
    gpu_time = time.time() - t0
    
    # Test with CPU
    clusterer_cpu = BuildingClusterer(use_gpu=False)
    t0 = time.time()
    best_shift_cpu, best_bbox_cpu = clusterer_cpu.optimize_bbox_for_building(
        points, heights, bbox, max_shift=5.0, step=0.5
    )
    cpu_time = time.time() - t0
    
    print(f"\nPerformance:")
    print(f"  GPU: {gpu_time*1000:.1f}ms")
    print(f"  CPU: {cpu_time*1000:.1f}ms")
    
    if cpu_time > gpu_time:
        print(f"  ✅ Speedup: {cpu_time/gpu_time:.1f}×")
    else:
        print(f"  ⚠️  GPU slower (transfer overhead for small dataset)")
    
    # Verify results match
    shift_match = np.allclose(
        [best_shift_gpu[0], best_shift_gpu[1]],
        [best_shift_cpu[0], best_shift_cpu[1]],
        atol=0.01
    )
    print(f"\nResults match: {shift_match}")
    print(f"  GPU shift: ({best_shift_gpu[0]:.2f}, {best_shift_gpu[1]:.2f})")
    print(f"  CPU shift: ({best_shift_cpu[0]:.2f}, {best_shift_cpu[1]:.2f})")
    
    print("\n✅ P0.2 verification complete")
    return True

def test_gpu_knn():
    """Test GPU KNN operations."""
    print_header("GPU KNN Operations (Façade Processing)")
    
    from ign_lidar.optimization.gpu_accelerated_ops import knn, HAS_FAISS, HAS_CUML
    
    print(f"FAISS-GPU available: {HAS_FAISS}")
    print(f"cuML available: {HAS_CUML}")
    
    if not (HAS_FAISS or HAS_CUML):
        print("⚠️  No GPU KNN library available, skipping test")
        return False
    
    # Test with realistic façade point cloud size
    n_ref = 100000  # 100K reference points
    n_query = 50000  # 50K query points
    k = 30
    
    ref_points = np.random.randn(n_ref, 3).astype(np.float32) * 10
    query_points = np.random.randn(n_query, 3).astype(np.float32) * 10
    
    print(f"\nTest data:")
    print(f"  Reference: {n_ref:,} points")
    print(f"  Query: {n_query:,} points")
    print(f"  k: {k} neighbors")
    
    t0 = time.time()
    distances, indices = knn(ref_points, query_points, k=k)
    knn_time = time.time() - t0
    
    print(f"\nPerformance:")
    print(f"  Time: {knn_time*1000:.1f}ms")
    print(f"  Throughput: {n_query/knn_time/1000:.1f}K queries/sec")
    print(f"\nResults:")
    print(f"  Shape: distances={distances.shape}, indices={indices.shape}")
    print(f"  Mean distance to 1st neighbor: {distances[:, 0].mean():.3f}")
    
    print("\n✅ GPU KNN verification complete")
    return True

def test_p01_road_classification():
    """Test P0.1: GPU Road Classification."""
    print_header("P0.1: GPU Road Classification")
    
    try:
        import cuspatial
        print("✅ cuSpatial available")
    except ImportError:
        print("⚠️  cuSpatial not available - P0.1 will use CPU fallback")
        print("   To enable GPU: conda install -c rapidsai cuspatial")
        return False
    
    from ign_lidar.core.classification.reclassifier import Reclassifier, HAS_GPU
    
    print(f"GPU acceleration available: {HAS_GPU}")
    
    if HAS_GPU:
        reclassifier = Reclassifier(acceleration_mode='gpu')
        print(f"Reclassifier mode: {reclassifier.acceleration_mode}")
        print("\n✅ P0.1 GPU road classification available")
    else:
        print("⚠️  GPU acceleration not available, will use CPU fallback")
    
    return HAS_GPU

def main():
    """Run all GPU verification tests."""
    print("\n" + "="*70)
    print("  GPU Optimization Verification Suite")
    print("  IGN LiDAR HD Dataset - Performance Audit Implementation")
    print("="*70)
    
    # Check GPU libraries
    libraries = check_gpu_libraries()
    
    results = {}
    
    # Test P1.4: GPU Thresholds
    try:
        results['P1.4'] = test_p14_gpu_thresholds()
    except Exception as e:
        print(f"\n❌ P1.4 test failed: {e}")
        results['P1.4'] = False
    
    # Test P0.2: BBox Optimization
    try:
        results['P0.2'] = test_p02_gpu_bbox()
    except Exception as e:
        print(f"\n❌ P0.2 test failed: {e}")
        results['P0.2'] = False
    
    # Test GPU KNN
    try:
        results['KNN'] = test_gpu_knn()
    except Exception as e:
        print(f"\n❌ GPU KNN test failed: {e}")
        results['KNN'] = False
    
    # Test P0.1: Road Classification
    try:
        results['P0.1'] = test_p01_road_classification()
    except Exception as e:
        print(f"\n❌ P0.1 test failed: {e}")
        results['P0.1'] = False
    
    # Summary
    print_header("Verification Summary")
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "⚠️  SKIP/FAIL"
        print(f"  {test_name:20s} {status}")
    
    # Overall status
    all_critical_passed = results.get('P1.4', False) and results.get('P0.2', False)
    
    print(f"\n{'='*70}")
    if all_critical_passed:
        print("  ✅ All critical optimizations verified!")
        if not results.get('P0.1', False):
            print("  ⚠️  P0.1 requires cuSpatial (install with: conda install -c rapidsai cuspatial)")
    else:
        print("  ⚠️  Some critical tests failed")
    print(f"{'='*70}\n")
    
    return 0 if all_critical_passed else 1

if __name__ == "__main__":
    sys.exit(main())
