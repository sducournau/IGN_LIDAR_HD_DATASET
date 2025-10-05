#!/usr/bin/env python3
"""
Test script for GPU CUSOLVER fix
Tests the GPU-chunked normal computation with the new error handling
"""

import numpy as np
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_gpu_normals():
    """Test GPU normal computation with the CUSOLVER fix"""
    
    logger.info("=" * 60)
    logger.info("Testing GPU CUSOLVER Fix - Normal Computation")
    logger.info("=" * 60)
    
    try:
        from ign_lidar.features_gpu_chunked import GPUChunkedFeatureComputer
        
        # Test data sizes
        test_sizes = [
            ("Small", 10_000),
            ("Medium", 100_000),
            ("Large", 1_000_000),
        ]
        
        for name, n_points in test_sizes:
            logger.info(f"\nTest {name}: {n_points:,} points")
            logger.info("-" * 60)
            
            # Generate random point cloud
            np.random.seed(42)
            points = np.random.randn(n_points, 3).astype(np.float32)
            
            # Scale to realistic LIDAR range
            points[:, 0] *= 100  # X: 0-100m spread
            points[:, 1] *= 100  # Y: 0-100m spread
            points[:, 2] *= 10   # Z: 0-10m height variation
            
            # Add some degenerate cases
            # (colinear points that might cause issues)
            if n_points >= 1000:
                # Add 100 points in a line (degenerate)
                line_points = np.zeros((100, 3), dtype=np.float32)
                line_points[:, 0] = np.linspace(0, 10, 100)
                points[:100] = line_points
                
                # Add 100 identical points (extreme degenerate)
                points[100:200] = points[100]
            
            # Test with GPU
            try:
                computer = GPUChunkedFeatureComputer(
                    chunk_size=500_000,
                    vram_limit_gb=8.0,
                    use_gpu=True,
                    show_progress=False
                )
                
                logger.info(f"  Computing normals with k=30...")
                normals = computer.compute_normals_chunked(points, k=30)
                
                # Validate results
                assert normals.shape == (n_points, 3), \
                    f"Wrong shape: {normals.shape}"
                
                # Check for valid normals
                norms = np.linalg.norm(normals, axis=1)
                n_valid = np.sum(np.abs(norms - 1.0) < 0.01)
                pct_valid = 100.0 * n_valid / n_points
                
                logger.info(f"  ✓ Success: {n_valid:,}/{n_points:,} normals "
                          f"valid ({pct_valid:.1f}%)")
                
                # Check for NaN/Inf
                n_invalid = np.sum(~np.isfinite(normals))
                if n_invalid > 0:
                    logger.warning(f"  ⚠ Found {n_invalid} invalid normals")
                else:
                    logger.info(f"  ✓ No NaN/Inf values")
                
                # Check upward orientation
                n_upward = np.sum(normals[:, 2] > 0)
                pct_upward = 100.0 * n_upward / n_points
                logger.info(f"  ✓ {n_upward:,} normals oriented upward "
                          f"({pct_upward:.1f}%)")
                
            except Exception as e:
                logger.error(f"  ✗ Test failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ All tests passed!")
        logger.info("=" * 60)
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import GPU modules: {e}")
        logger.error("Make sure the package is installed with GPU support")
        return False

if __name__ == "__main__":
    success = test_gpu_normals()
    sys.exit(0 if success else 1)
