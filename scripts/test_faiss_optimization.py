#!/usr/bin/env python3
"""
Test script for FAISS GPU optimization improvements.

Tests the optimized FAISS parameters on a single tile to validate:
- Reduced nprobe (64 instead of 256)
- Larger batch size (2M instead of 500K)
- IVFFlat for 10-20M point datasets
- Improved time estimates
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ign_lidar.features.gpu_processor import GPUProcessor
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def test_faiss_optimization(num_points=18_000_000):
    """
    Test FAISS optimization with a synthetic dataset.
    
    Args:
        num_points: Number of points to test with (default: 18M, similar to real tiles)
    """
    logger.info("="*80)
    logger.info("FAISS GPU Optimization Test")
    logger.info("="*80)
    logger.info(f"Testing with {num_points:,} points")
    logger.info("")
    
    # Create synthetic point cloud (random points in unit cube)
    logger.info("Generating synthetic point cloud...")
    points = np.random.rand(num_points, 3).astype(np.float32)
    logger.info(f"✓ Generated {len(points):,} points")
    logger.info("")
    
    # Initialize GPU processor
    logger.info("Initializing GPU processor...")
    processor = GPUProcessor(
        use_gpu=True,
        vram_limit_gb=14,
        auto_chunk=True,
        show_progress=True
    )
    logger.info("✓ GPU processor initialized")
    logger.info(f"  VRAM limit: {processor.vram_limit_gb}GB")
    logger.info("")
    
    # Test FAISS k-NN with optimized parameters
    k = 25
    logger.info(f"Testing FAISS k-NN (k={k})...")
    logger.info("")
    
    start_time = time.time()
    
    # Build index
    index = processor._build_faiss_index(points, k)
    build_time = time.time() - start_time
    logger.info("")
    logger.info(f"✓ Index built in {build_time:.1f} seconds")
    logger.info("")
    
    # Query neighbors (batched)
    query_start = time.time()
    
    # Use same batching logic as in production
    if processor.use_gpu:
        batch_size = 2_000_000  # Optimized GPU batch size
    else:
        batch_size = 500_000
    
    num_batches = (num_points + batch_size - 1) // batch_size
    logger.info(f"Querying {num_points:,} points in {num_batches} batches...")
    logger.info(f"  Batch size: {batch_size:,}")
    logger.info("")
    
    all_indices = np.zeros((num_points, k), dtype=np.int64)
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, num_points)
        
        batch_points = points[batch_start:batch_end].astype(np.float32)
        batch_distances, batch_indices = index.search(batch_points, k)
        all_indices[batch_start:batch_end] = batch_indices
        
        if (batch_idx + 1) % 2 == 0 or batch_idx == num_batches - 1:
            elapsed = time.time() - query_start
            progress = (batch_idx + 1) / num_batches * 100
            logger.info(f"  Batch {batch_idx+1}/{num_batches} ({progress:.0f}%) - {elapsed:.1f}s elapsed")
    
    query_time = time.time() - query_start
    total_time = time.time() - start_time
    
    logger.info("")
    logger.info("="*80)
    logger.info("RESULTS")
    logger.info("="*80)
    logger.info(f"Dataset size: {num_points:,} points")
    logger.info(f"k-neighbors: {k}")
    logger.info(f"Batch size: {batch_size:,}")
    logger.info(f"Number of batches: {num_batches}")
    logger.info("")
    logger.info(f"Index build time: {build_time:.1f}s")
    logger.info(f"Query time: {query_time:.1f}s")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info("")
    logger.info(f"Throughput: {num_points/total_time:,.0f} points/sec")
    logger.info(f"Query throughput: {num_points/query_time:,.0f} points/sec")
    logger.info("")
    
    # Expected vs actual
    expected_time = 60  # 60 seconds expected for 18M points
    if total_time <= expected_time:
        logger.info(f"✅ PASS - Completed in {total_time:.1f}s (expected: <{expected_time}s)")
    elif total_time <= expected_time * 1.5:
        logger.info(f"⚠️  ACCEPTABLE - Completed in {total_time:.1f}s (expected: <{expected_time}s)")
    else:
        logger.warning(f"❌ SLOW - Completed in {total_time:.1f}s (expected: <{expected_time}s)")
    
    logger.info("")
    logger.info("="*80)
    
    return {
        'num_points': num_points,
        'k': k,
        'build_time': build_time,
        'query_time': query_time,
        'total_time': total_time,
        'throughput': num_points / total_time
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test FAISS GPU optimization')
    parser.add_argument('--points', type=int, default=18_000_000,
                        help='Number of points to test (default: 18M)')
    
    args = parser.parse_args()
    
    try:
        results = test_faiss_optimization(num_points=args.points)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
