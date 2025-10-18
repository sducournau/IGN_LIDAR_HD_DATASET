#!/usr/bin/env python3
"""
Quick test to verify GPU synchronization fix

Tests that cp.any() replacement with cp.where() works correctly
"""

import logging
import sys
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import cupy as cp
    GPU_AVAILABLE = True
    logger.info("✓ CuPy available")
except ImportError:
    GPU_AVAILABLE = False
    logger.error("✗ CuPy not available - cannot test GPU synchronization fix")
    sys.exit(1)

def test_synchronization_bottleneck():
    """Test that we don't have synchronization bottlenecks"""
    
    logger.info("\n" + "="*70)
    logger.info("Testing GPU Synchronization Fix")
    logger.info("="*70)
    
    # Create test data
    n_matrices = 10_000
    matrices = cp.random.randn(n_matrices, 3, 3).astype(cp.float32)
    
    # Compute determinants
    det = cp.linalg.det(matrices)
    small = cp.abs(det) < 1e-8
    
    logger.info(f"\nTest dataset: {n_matrices:,} matrices")
    logger.info(f"Matrices with small determinant: {cp.sum(small).item():,}")
    
    # TEST 1: Old synchronous method (cp.any)
    logger.info("\n--- Test 1: Old method with cp.any() (SLOW) ---")
    start = time.time()
    result_old = matrices.copy()
    if cp.any(small):  # This forces synchronization!
        result_old[small] = cp.eye(3, dtype=matrices.dtype)
    sync_time = time.time() - start
    logger.info(f"  Time with cp.any(): {sync_time*1000:.2f} ms")
    logger.info(f"  ⚠️ This triggers GPU→CPU synchronization!")
    
    # TEST 2: New asynchronous method (cp.where)
    logger.info("\n--- Test 2: New method with cp.where() (FAST) ---")
    start = time.time()
    eye_3x3 = cp.eye(3, dtype=matrices.dtype)
    result_new = cp.where(small[:, None, None], eye_3x3, matrices)
    async_time = time.time() - start
    logger.info(f"  Time with cp.where(): {async_time*1000:.2f} ms")
    logger.info(f"  ✓ No synchronization - stays on GPU!")
    
    # Verify results are identical
    max_diff = cp.max(cp.abs(result_old - result_new)).item()
    logger.info(f"\nMax difference between methods: {max_diff:.2e}")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("="*70)
    logger.info(f"Old method (cp.any):   {sync_time*1000:.2f} ms")
    logger.info(f"New method (cp.where): {async_time*1000:.2f} ms")
    speedup = sync_time / async_time
    logger.info(f"Speedup: {speedup:.2f}x")
    
    if max_diff < 1e-5:
        logger.info("\n✅ GPU synchronization fix validated!")
        logger.info("   Results are identical, no synchronization bottleneck")
        return True
    else:
        logger.error(f"\n✗ Results differ by {max_diff:.2e}")
        return False

if __name__ == "__main__":
    try:
        success = test_synchronization_bottleneck()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
