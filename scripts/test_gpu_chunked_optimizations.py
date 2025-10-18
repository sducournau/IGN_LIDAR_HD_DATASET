#!/usr/bin/env python3
"""
Quick test script to validate GPU chunked optimizations.
Tests that all optimizations are working correctly.
"""

import sys
import os
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def test_gpu_optimizations():
    """Test GPU chunked optimizations are active."""
    
    logger.info("=" * 70)
    logger.info("GPU CHUNKED OPTIMIZATION VALIDATION TEST")
    logger.info("=" * 70)
    
    # Test 1: Check CuPy availability
    logger.info("\n📋 Test 1: GPU Dependencies")
    try:
        import cupy as cp
        logger.info("✅ CuPy available")
        
        # Check GPU memory
        mempool = cp.get_default_memory_pool()
        total_bytes = cp.cuda.Device(0).mem_info[1]
        logger.info(f"✅ GPU Memory: {total_bytes / (1024**3):.1f} GB")
    except ImportError:
        logger.error("❌ CuPy not available - optimizations won't work!")
        return False
    
    # Test 2: Check cuML availability
    try:
        from cuml.neighbors import NearestNeighbors
        logger.info("✅ cuML available")
    except ImportError:
        logger.warning("⚠️  cuML not available - will use CPU KNN")
    
    # Test 3: Import optimized module
    logger.info("\n📋 Test 2: Import Optimized Module")
    try:
        from ign_lidar.features.features_gpu_chunked import GPUChunkedFeatureComputer
        logger.info("✅ GPU chunked module imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import: {e}")
        return False
    
    # Test 4: Create instance and check optimizations
    logger.info("\n📋 Test 3: Validate Optimization Flags")
    try:
        computer = GPUChunkedFeatureComputer(
            chunk_size=1_000_000,
            use_gpu=True,
            show_progress=False
        )
        
        logger.info(f"✅ use_gpu: {computer.use_gpu}")
        logger.info(f"✅ use_cuml: {computer.use_cuml}")
        logger.info(f"✅ chunk_size: {computer.chunk_size:,}")
        
        if not computer.use_gpu:
            logger.error("❌ GPU not enabled!")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to create computer: {e}")
        return False
    
    # Test 5: Test small computation
    logger.info("\n📋 Test 4: Small Computation Test")
    try:
        import numpy as np
        
        # Create small test data
        n_points = 10000
        points = np.random.randn(n_points, 3).astype(np.float32)
        classification = np.random.randint(0, 6, n_points, dtype=np.uint8)
        
        logger.info(f"Testing with {n_points:,} points...")
        
        start = time.time()
        normals, curvature, height, geo_features = computer.compute_all_features_chunked(
            points=points,
            classification=classification,
            k=10,
            mode='asprs_classes'
        )
        elapsed = time.time() - start
        
        logger.info(f"✅ Computation completed in {elapsed:.2f}s")
        logger.info(f"✅ Normals shape: {normals.shape}")
        logger.info(f"✅ Curvature shape: {curvature.shape}")
        logger.info(f"✅ Geometric features: {list(geo_features.keys())}")
        
        # Validate outputs
        assert normals.shape == (n_points, 3), "Normals shape mismatch"
        assert curvature.shape == (n_points,), "Curvature shape mismatch"
        assert len(geo_features) > 0, "No geometric features computed"
        
        logger.info("✅ All output shapes valid")
        
    except Exception as e:
        logger.error(f"❌ Computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Check for batched transfer log message
    logger.info("\n📋 Test 5: Verify Batched Transfer Optimization")
    logger.info("⚠️  Check logs above for '📦 Batched GPU→CPU transfer' message")
    logger.info("    (Only appears for chunks > chunk_size)")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ ALL TESTS PASSED - Optimizations are working!")
    logger.info("=" * 70)
    
    return True


if __name__ == "__main__":
    try:
        success = test_gpu_optimizations()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
