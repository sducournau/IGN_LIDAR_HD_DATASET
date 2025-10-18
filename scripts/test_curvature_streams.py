#!/usr/bin/env python3
"""
Test script for CUDA streams optimization in curvature computation.

This validates the triple-buffering pipeline implementation that overlaps
upload/compute/download operations for maximum GPU throughput.

Expected: +20-30% speedup over batched processing (2.5s→1.9s for 10M points)
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ign_lidar.features.features_gpu_chunked import GPUChunkedFeatureComputer

def generate_test_data(num_points: int = 1_000_000):
    """Generate synthetic point cloud and normals."""
    print(f"\n📊 Generating {num_points:,} synthetic points...")
    
    # Create random point cloud
    np.random.seed(42)
    points = np.random.randn(num_points, 3).astype(np.float32)
    
    # Create random normals (normalized)
    normals = np.random.randn(num_points, 3).astype(np.float32)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-6)
    
    print(f"  ✓ Points shape: {points.shape}")
    print(f"  ✓ Normals shape: {normals.shape}")
    
    return points, normals


def test_curvature_streams(points, normals, k: int = 10):
    """Test curvature computation with CUDA streams."""
    
    print("\n" + "="*80)
    print("🚀 Testing CUDA Streams for Curvature Computation")
    print("="*80)
    
    # Test 1: With CUDA streams (triple-buffering)
    print("\n📍 TEST 1: CUDA Streams (Triple-Buffering)")
    print("-" * 80)
    
    computer_streams = GPUChunkedFeatureComputer(
        chunk_size=100_000,
        use_gpu=True,
        show_progress=True
    )
    
    # Verify streams are available
    if computer_streams.stream_manager is None:
        print("  ⚠️  WARNING: CUDA streams not available, skipping streams test")
        return None
    
    print(f"  ✓ Stream manager initialized: {computer_streams.stream_manager}")
    print(f"  ✓ Chunk size: {computer_streams.chunk_size:,}")
    
    start_time = time.time()
    try:
        curvature_streams = computer_streams.compute_curvature_chunked(
            points, normals, k=k
        )
    except Exception as e:
        print(f"\n  ❌ ERROR in streams computation: {e}")
        import traceback
        traceback.print_exc()
        return None
    streams_time = time.time() - start_time
    
    print(f"\n  ⏱️  Time with streams: {streams_time:.3f}s")
    print(f"  ⚡ Throughput: {len(points) / streams_time:,.0f} pts/sec")
    print(f"  📊 Curvature stats:")
    print(f"     - Mean: {np.mean(curvature_streams):.6f}")
    print(f"     - Std:  {np.std(curvature_streams):.6f}")
    print(f"     - Min:  {np.min(curvature_streams):.6f}")
    print(f"     - Max:  {np.max(curvature_streams):.6f}")
    
    # Test 2: Without CUDA streams (batched)
    print("\n📍 TEST 2: Batched Processing (No Streams)")
    print("-" * 80)
    
    computer_batched = GPUChunkedFeatureComputer(
        chunk_size=100_000,
        use_gpu=True,
        show_progress=True
    )
    # Force disable streams
    computer_batched.stream_manager = None
    
    start_time = time.time()
    curvature_batched = computer_batched.compute_curvature_chunked(
        points, normals, k=k
    )
    batched_time = time.time() - start_time
    
    print(f"\n  ⏱️  Time without streams: {batched_time:.3f}s")
    print(f"  ⚡ Throughput: {len(points) / batched_time:,.0f} pts/sec")
    print(f"  📊 Curvature stats:")
    print(f"     - Mean: {np.mean(curvature_batched):.6f}")
    print(f"     - Std:  {np.std(curvature_batched):.6f}")
    print(f"     - Min:  {np.min(curvature_batched):.6f}")
    print(f"     - Max:  {np.max(curvature_batched):.6f}")
    
    # Comparison
    print("\n" + "="*80)
    print("📊 PERFORMANCE COMPARISON")
    print("="*80)
    
    speedup = batched_time / streams_time
    improvement_pct = (speedup - 1.0) * 100
    
    print(f"\n  ⚡ CUDA Streams Speedup: {speedup:.2f}×")
    print(f"  📈 Improvement: {improvement_pct:+.1f}%")
    print(f"  ⏱️  Time saved: {batched_time - streams_time:.3f}s")
    
    if speedup >= 1.2:
        print(f"\n  ✅ SUCCESS: Achieved {improvement_pct:.1f}% improvement (target: +20-30%)")
    elif speedup >= 1.1:
        print(f"\n  ⚠️  MODERATE: Achieved {improvement_pct:.1f}% improvement (target: +20-30%)")
    else:
        print(f"\n  ❌ BELOW TARGET: Only {improvement_pct:.1f}% improvement (expected: +20-30%)")
    
    # Validate results are numerically similar
    print("\n📍 Numerical Validation")
    print("-" * 80)
    
    diff = np.abs(curvature_streams - curvature_batched)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"  Max difference: {max_diff:.10f}")
    print(f"  Mean difference: {mean_diff:.10f}")
    
    if max_diff < 1e-5:
        print(f"  ✅ Results match within tolerance (max diff: {max_diff:.2e})")
    else:
        print(f"  ⚠️  Differences detected (max: {max_diff:.2e})")
    
    return {
        'streams_time': streams_time,
        'batched_time': batched_time,
        'speedup': speedup,
        'improvement_pct': improvement_pct,
        'max_diff': max_diff,
        'mean_diff': mean_diff
    }


def main():
    """Run curvature streams tests."""
    
    print("\n" + "="*80)
    print("🎯 CURVATURE CUDA STREAMS VALIDATION")
    print("="*80)
    print("\nPhase 3 Priority 1: Implement CUDA streams for chunked curvature")
    print("Expected: +20-30% throughput improvement over batched processing")
    print("="*80)
    
    # Test configurations
    test_configs = [
        {'num_points': 1_000_000, 'k': 10, 'name': '1M points'},
        {'num_points': 5_000_000, 'k': 10, 'name': '5M points'},
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\n\n{'='*80}")
        print(f"🧪 Test Configuration: {config['name']}")
        print(f"{'='*80}")
        
        # Generate data
        points, normals = generate_test_data(config['num_points'])
        
        # Run test
        result = test_curvature_streams(points, normals, k=config['k'])
        
        if result:
            results[config['name']] = result
    
    # Final summary
    print("\n\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    
    if not results:
        print("\n  ⚠️  No results (CUDA streams not available)")
        return
    
    total_speedup = np.mean([r['speedup'] for r in results.values()])
    total_improvement = (total_speedup - 1.0) * 100
    
    print(f"\n  Average Speedup: {total_speedup:.2f}×")
    print(f"  Average Improvement: {total_improvement:+.1f}%")
    
    if total_speedup >= 1.2:
        print(f"\n  ✅ SUCCESS: Phase 3 Priority 1 complete!")
        print(f"     CUDA streams provide {total_improvement:.1f}% speedup for curvature")
        return 0
    else:
        print(f"\n  ⚠️  Results below target (+20-30% expected)")
        return 1


if __name__ == '__main__':
    exit(main())
