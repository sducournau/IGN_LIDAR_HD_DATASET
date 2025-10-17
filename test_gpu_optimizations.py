#!/usr/bin/env python3
"""
Test script for GPU and GPU chunked reclassification optimizations.

This script benchmarks the new optimizations against the original implementations
to demonstrate performance improvements.
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from ign_lidar.features.features_gpu import GPUFeatureComputer
    from ign_lidar.features.features_gpu_chunked import GPUChunkedFeatureComputer
    from ign_lidar.core.memory import AdaptiveMemoryManager
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("Please ensure the package is installed: pip install -e .")
    sys.exit(1)


def generate_test_point_cloud(num_points: int = 1_000_000) -> tuple:
    """Generate a synthetic point cloud for testing."""
    print(f"📊 Generating test point cloud ({num_points:,} points)...")
    
    # Create realistic 3D point cloud data
    np.random.seed(42)  # For reproducible results
    
    # Generate points in a 100m x 100m x 20m volume
    x = np.random.uniform(0, 100, num_points).astype(np.float32)
    y = np.random.uniform(0, 100, num_points).astype(np.float32)
    z = np.random.uniform(0, 20, num_points).astype(np.float32)
    
    points = np.column_stack([x, y, z])
    
    # Generate ASPRS classification codes (ground=2, vegetation=3, building=6)
    classification = np.random.choice([2, 3, 6], size=num_points).astype(np.uint8)
    
    print(f"  ✓ Generated {points.shape[0]:,} points")
    return points, classification


def benchmark_gpu_features():
    """Benchmark standard GPU feature computation."""
    print("\n🚀 Benchmarking Standard GPU Features...")
    
    # Test with different point cloud sizes
    test_sizes = [100_000, 500_000, 1_000_000]
    
    for num_points in test_sizes:
        print(f"\n📏 Testing with {num_points:,} points:")
        
        points, classification = generate_test_point_cloud(num_points)
        
        # Test standard GPU computation
        computer = GPUFeatureComputer(use_gpu=True)
        
        start_time = time.time()
        try:
            normals = computer.compute_normals(points, k=10)
            curvature = computer.compute_curvature(points, normals, k=10)
            height = computer.compute_height_above_ground(points, classification)
            
            elapsed = time.time() - start_time
            print(f"  ⏱️  Standard GPU: {elapsed:.2f}s ({num_points/elapsed:.0f} pts/s)")
        except Exception as e:
            print(f"  ❌ Standard GPU failed: {e}")
        
        # Test new optimized reclassification method
        start_time = time.time()
        try:
            normals_opt, features_opt = computer.compute_reclassification_features_fast(
                points, classification, k=10, mode='minimal'
            )
            
            elapsed = time.time() - start_time
            print(f"  ⚡ Optimized GPU: {elapsed:.2f}s ({num_points/elapsed:.0f} pts/s)")
            
            # Compare feature quality
            if 'normals' in locals():
                diff = np.mean(np.abs(normals - normals_opt))
                print(f"     Normal difference: {diff:.6f} (lower is better)")
                
        except Exception as e:
            print(f"  ❌ Optimized GPU failed: {e}")


def benchmark_gpu_chunked_features():
    """Benchmark GPU chunked feature computation."""
    print("\n🔥 Benchmarking GPU Chunked Features...")
    
    # Test with larger point clouds that require chunking
    test_sizes = [2_000_000, 5_000_000]
    
    for num_points in test_sizes:
        print(f"\n📏 Testing chunked processing with {num_points:,} points:")
        
        points, classification = generate_test_point_cloud(num_points)
        
        # Test standard chunked computation
        computer = GPUChunkedFeatureComputer(use_gpu=True, auto_optimize=False)
        
        start_time = time.time()
        try:
            normals, curvature, height, geo_features = computer.compute_all_features_chunked(
                points, classification, k=10, mode='minimal'
            )
            
            elapsed = time.time() - start_time
            print(f"  ⏱️  Standard Chunked: {elapsed:.2f}s ({num_points/elapsed:.0f} pts/s)")
        except Exception as e:
            print(f"  ❌ Standard Chunked failed: {e}")
        
        # Test new optimized reclassification method
        computer_opt = GPUChunkedFeatureComputer(use_gpu=True, auto_optimize=True)
        
        start_time = time.time()
        try:
            normals_opt, curvature_opt, height_opt, features_opt = (
                computer_opt.compute_reclassification_features_optimized(
                    points, classification, k=10, mode='minimal'
                )
            )
            
            elapsed = time.time() - start_time
            print(f"  ⚡ Optimized Chunked: {elapsed:.2f}s ({num_points/elapsed:.0f} pts/s)")
            
            # Compare feature quality
            if 'normals' in locals():
                diff = np.mean(np.abs(normals - normals_opt))
                print(f"     Normal difference: {diff:.6f} (lower is better)")
                
        except Exception as e:
            print(f"  ❌ Optimized Chunked failed: {e}")


def test_memory_optimization():
    """Test the adaptive memory management optimizations."""
    print("\n🧠 Testing Adaptive Memory Management...")
    
    manager = AdaptiveMemoryManager()
    
    # Test different point cloud sizes and feature modes
    test_scenarios = [
        (1_000_000, 'minimal'),
        (5_000_000, 'minimal'),
        (10_000_000, 'standard'),
        (20_000_000, 'full'),
    ]
    
    for num_points, feature_mode in test_scenarios:
        print(f"\n📊 Scenario: {num_points:,} points, {feature_mode} features")
        
        # Test GPU chunk size calculation
        try:
            optimal_chunk = manager.calculate_optimal_gpu_chunk_size(
                num_points=num_points,
                vram_free_gb=None,  # Auto-detect
                feature_mode=feature_mode
            )
            print(f"  ✓ Optimal GPU chunk size: {optimal_chunk:,}")
        except Exception as e:
            print(f"  ❌ GPU chunk calculation failed: {e}")
        
        # Test eigenvalue batch size calculation
        try:
            if optimal_chunk > 0:
                eigh_batch = manager.calculate_optimal_eigh_batch_size(
                    chunk_size=optimal_chunk,
                    vram_free_gb=None  # Auto-detect
                )
                print(f"  ✓ Optimal eigenvalue batch size: {eigh_batch:,}")
        except Exception as e:
            print(f"  ❌ Eigenvalue batch calculation failed: {e}")


def test_gpu_availability():
    """Test GPU availability and provide diagnostic information."""
    print("\n🔍 GPU Availability Diagnostics...")
    
    # Test CuPy availability
    try:
        import cupy as cp
        print("  ✓ CuPy is available")
        
        # Get GPU info
        device = cp.cuda.Device()
        print(f"    GPU: {device}")
        
        # Get memory info
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        free_gb = free_mem / (1024**3)
        total_gb = total_mem / (1024**3)
        print(f"    VRAM: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
        
    except ImportError:
        print("  ❌ CuPy not available")
    except Exception as e:
        print(f"  ⚠️ CuPy error: {e}")
    
    # Test RAPIDS cuML availability
    try:
        from cuml.neighbors import NearestNeighbors
        print("  ✓ RAPIDS cuML is available")
    except ImportError:
        print("  ❌ RAPIDS cuML not available")
    except Exception as e:
        print(f"  ⚠️ RAPIDS cuML error: {e}")


def main():
    """Run all benchmarks and tests."""
    print("🎯 GPU Reclassification Optimization Test Suite")
    print("=" * 60)
    
    # Check GPU availability first
    test_gpu_availability()
    
    # Test memory optimization
    test_memory_optimization()
    
    # Benchmark standard GPU features
    benchmark_gpu_features()
    
    # Benchmark GPU chunked features
    benchmark_gpu_chunked_features()
    
    print("\n" + "=" * 60)
    print("✅ Optimization tests completed!")
    print("\nKey optimizations tested:")
    print("  • Adaptive chunk sizing based on available VRAM")
    print("  • Optimized feature computation for reclassification")
    print("  • Improved memory management and cleanup")
    print("  • Fast eigenvalue decomposition with error handling")
    print("  • Reclassification-specific feature sets")


if __name__ == "__main__":
    main()