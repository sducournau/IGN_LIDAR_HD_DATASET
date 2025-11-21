#!/usr/bin/env python3
"""
Test FAISS memory optimization fixes.

Tests the new dynamic memory allocation and chunked processing.
"""

import numpy as np
import logging
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ign_lidar.features.compute.faiss_knn import (
    knn_search_faiss,
    _get_gpu_memory_info,
    _calculate_safe_temp_memory,
    HAS_FAISS_GPU
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def test_memory_info():
    """Test GPU memory detection."""
    logger.info("Testing GPU memory detection...")
    total_gb, free_gb = _get_gpu_memory_info()
    logger.info(f"  Total GPU memory: {total_gb:.2f} GB")
    logger.info(f"  Free GPU memory: {free_gb:.2f} GB")
    return total_gb, free_gb


def test_temp_memory_calculation():
    """Test temp memory size calculation."""
    logger.info("\nTesting temp memory calculation...")
    
    test_cases = [
        (10_000, 3, 30, "Small dataset"),
        (1_000_000, 3, 30, "Medium dataset (1M points)"),
        (10_000_000, 3, 30, "Large dataset (10M points)"),
        (18_651_688, 3, 30, "Current dataset (18.6M points)"),
    ]
    
    for n_points, n_dims, k, desc in test_cases:
        temp_bytes = _calculate_safe_temp_memory(n_points, n_dims, k)
        temp_gb = temp_bytes / (1024**3)
        logger.info(f"  {desc}: {temp_gb:.2f} GB temp memory")


def test_small_dataset():
    """Test with small dataset (should work)."""
    logger.info("\nTesting small dataset (10K points)...")
    points = np.random.randn(10_000, 3).astype(np.float32)
    
    try:
        distances, indices = knn_search_faiss(points, k=30, use_gpu=True)
        logger.info(f"  ✓ Success! Shape: {distances.shape}")
        logger.info(f"  Average distance: {distances.mean():.3f}")
        return True
    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
        return False


def test_medium_dataset():
    """Test with medium dataset (1M points)."""
    logger.info("\nTesting medium dataset (1M points)...")
    points = np.random.randn(1_000_000, 3).astype(np.float32)
    
    try:
        distances, indices = knn_search_faiss(points, k=30, use_gpu=True)
        logger.info(f"  ✓ Success! Shape: {distances.shape}")
        logger.info(f"  Average distance: {distances.mean():.3f}")
        return True
    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
        return False


def test_large_dataset():
    """Test with large dataset (5M points) - should trigger chunking."""
    logger.info("\nTesting large dataset (5M points - should trigger chunking)...")
    points = np.random.randn(5_000_000, 3).astype(np.float32)
    
    try:
        distances, indices = knn_search_faiss(points, k=30, use_gpu=True)
        logger.info(f"  ✓ Success! Shape: {distances.shape}")
        logger.info(f"  Average distance: {distances.mean():.3f}")
        return True
    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("=" * 70)
    logger.info("FAISS Memory Optimization Test Suite")
    logger.info("=" * 70)
    
    if not HAS_FAISS_GPU:
        logger.error("FAISS-GPU not available! Please install faiss-gpu.")
        return 1
    
    # Test 1: Memory info
    total_gb, free_gb = test_memory_info()
    
    # Test 2: Temp memory calculation
    test_temp_memory_calculation()
    
    # Test 3: Small dataset
    success_small = test_small_dataset()
    
    # Test 4: Medium dataset
    success_medium = test_medium_dataset()
    
    # Test 5: Large dataset (chunking)
    success_large = test_large_dataset()
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Test Summary:")
    logger.info(f"  Small dataset (10K): {'✓ PASS' if success_small else '✗ FAIL'}")
    logger.info(f"  Medium dataset (1M): {'✓ PASS' if success_medium else '✗ FAIL'}")
    logger.info(f"  Large dataset (5M):  {'✓ PASS' if success_large else '✗ FAIL'}")
    logger.info("=" * 70)
    
    # Return 0 if all passed
    if success_small and success_medium and success_large:
        logger.info("✓ All tests passed!")
        return 0
    else:
        logger.error("✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
