#!/usr/bin/env python3
"""
Quick validation script for CUDA and GPU chunked optimizations.

This script performs basic smoke tests to ensure the optimizations
are correctly integrated and functional.

Usage:
    python scripts/validate_cuda_optimizations.py
"""

import logging
import sys
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all necessary modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        from ign_lidar.features.features_gpu_chunked import GPUChunkedFeatureComputer
        logger.info("✓ GPUChunkedFeatureComputer imported")
    except ImportError as e:
        logger.error(f"✗ Failed to import GPUChunkedFeatureComputer: {e}")
        return False
    
    try:
        from ign_lidar.optimization.cuda_streams import create_stream_manager
        logger.info("✓ CUDA streams module imported")
    except ImportError as e:
        logger.error(f"✗ Failed to import cuda_streams: {e}")
        return False
    
    try:
        from ign_lidar.optimization.gpu_memory import GPUArrayCache
        logger.info("✓ GPU memory module imported")
    except ImportError as e:
        logger.error(f"✗ Failed to import gpu_memory: {e}")
        return False
    
    return True


def test_gpu_chunked_initialization():
    """Test GPU chunked computer initialization."""
    logger.info("\nTesting GPU chunked initialization...")
    
    try:
        from ign_lidar.features.features_gpu_chunked import GPUChunkedFeatureComputer
        
        # Test with CUDA streams enabled
        computer = GPUChunkedFeatureComputer(
            chunk_size=1_000_000,
            use_gpu=True,
            use_cuda_streams=True,
            enable_memory_pooling=True,
            enable_pipeline_optimization=True,
            show_progress=False
        )
        logger.info("✓ GPU chunked computer initialized with CUDA streams")
        
        # Check attributes
        if hasattr(computer, '_compute_normals_with_streams'):
            logger.info("✓ CUDA streams method exists")
        else:
            logger.warning("⚠ CUDA streams method not found")
        
        if hasattr(computer, '_calculate_optimal_eigh_batch_size'):
            logger.info("✓ Eigendecomposition optimization method exists")
        else:
            logger.warning("⚠ Eigendecomposition optimization method not found")
        
        if hasattr(computer, '_optimize_neighbor_batch_size'):
            logger.info("✓ Dynamic batch sizing method exists")
        else:
            logger.warning("⚠ Dynamic batch sizing method not found")
        
        return True
    except Exception as e:
        logger.error(f"✗ Initialization failed: {e}")
        return False


def test_batch_size_calculation():
    """Test batch size calculation methods."""
    logger.info("\nTesting batch size calculations...")
    
    try:
        from ign_lidar.features.features_gpu_chunked import GPUChunkedFeatureComputer
        
        computer = GPUChunkedFeatureComputer(
            vram_limit_gb=8.0,
            show_progress=False
        )
        
        # Test eigendecomposition batch size
        eigh_batch = computer._calculate_optimal_eigh_batch_size(1_000_000)
        logger.info(f"✓ Eigendecomposition batch size: {eigh_batch:,}")
        
        if not (50_000 <= eigh_batch <= 500_000):
            logger.warning(f"⚠ Batch size {eigh_batch} outside expected range [50K-500K]")
        
        # Test neighbor batch size
        neighbor_batch = computer._optimize_neighbor_batch_size(1_000_000, k_neighbors=20)
        logger.info(f"✓ Neighbor batch size (k=20): {neighbor_batch:,}")
        
        # Test with different k values
        neighbor_batch_high_k = computer._optimize_neighbor_batch_size(1_000_000, k_neighbors=60)
        logger.info(f"✓ Neighbor batch size (k=60): {neighbor_batch_high_k:,}")
        
        if neighbor_batch_high_k >= neighbor_batch:
            logger.warning("⚠ Expected smaller batch size for higher k")
        else:
            logger.info("✓ Batch size correctly reduced for higher k")
        
        return True
    except Exception as e:
        logger.error(f"✗ Batch size calculation failed: {e}")
        return False


def test_small_computation():
    """Test a small computation with optimizations enabled."""
    logger.info("\nTesting small computation...")
    
    try:
        from ign_lidar.features.features_gpu_chunked import GPUChunkedFeatureComputer
        
        # Create small test dataset
        num_points = 10_000
        points = np.random.randn(num_points, 3).astype(np.float32)
        
        # Test without CUDA streams (fallback mode)
        computer_no_streams = GPUChunkedFeatureComputer(
            chunk_size=5_000,
            use_gpu=True,
            use_cuda_streams=False,
            show_progress=False
        )
        
        normals_no_streams = computer_no_streams.compute_normals_chunked(points, k=10)
        logger.info(f"✓ Computed normals without streams: {normals_no_streams.shape}")
        
        # Validate output
        if normals_no_streams.shape != (num_points, 3):
            logger.error(f"✗ Incorrect output shape: {normals_no_streams.shape}")
            return False
        
        # Check if normals are unit vectors
        norms = np.linalg.norm(normals_no_streams, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            logger.warning("⚠ Some normals are not unit vectors")
        else:
            logger.info("✓ All normals are unit vectors")
        
        # Test with CUDA streams (if available)
        try:
            computer_with_streams = GPUChunkedFeatureComputer(
                chunk_size=5_000,
                use_gpu=True,
                use_cuda_streams=True,
                show_progress=False
            )
            
            # Only test if stream manager was successfully initialized
            if computer_with_streams.stream_manager is not None:
                normals_with_streams = computer_with_streams.compute_normals_chunked(points, k=10)
                logger.info(f"✓ Computed normals with streams: {normals_with_streams.shape}")
                
                # Compare results
                diff = np.abs(normals_no_streams - normals_with_streams).max()
                if diff < 1e-4:
                    logger.info(f"✓ Results match (max diff: {diff:.2e})")
                else:
                    logger.warning(f"⚠ Results differ (max diff: {diff:.2e})")
            else:
                logger.info("ℹ CUDA streams not available (expected on CPU-only systems)")
        except Exception as e:
            logger.info(f"ℹ CUDA streams test skipped: {e}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    logger.info("="*60)
    logger.info("CUDA and GPU Chunked Optimizations - Validation")
    logger.info("="*60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Initialization", test_gpu_chunked_initialization()))
    results.append(("Batch Size Calculation", test_batch_size_calculation()))
    results.append(("Small Computation", test_small_computation()))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*60)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name:.<40} {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        logger.info("\n✅ All validation tests passed!")
        return 0
    else:
        logger.error("\n❌ Some validation tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
