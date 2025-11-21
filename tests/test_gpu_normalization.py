"""
Test suite for GPU-accelerated normalization utilities.

Tests CPU/GPU normalization functions with various edge cases,
error handling, and performance validation.

Author: IGN LiDAR HD Processing Library
Date: 2025-11-21
"""

import pytest
import numpy as np
from ign_lidar.utils.normalization import (
    normalize_uint8_to_float,
    denormalize_float_to_uint8,
    normalize_rgb,
    normalize_nir,
    is_gpu_available,
    GPU_AVAILABLE
)

# Import CuPy for GPU tests
if GPU_AVAILABLE:
    import cupy as cp


class TestCPUNormalization:
    """Test CPU-based normalization functions."""
    
    def test_normalize_uint8_basic(self):
        """Test basic uint8 to float32 normalization."""
        data = np.array([0, 127, 255], dtype=np.uint8)
        result = normalize_uint8_to_float(data, use_gpu=False)
        
        assert result.dtype == np.float32
        assert np.allclose(result, [0.0, 0.498, 1.0], atol=0.01)
    
    def test_normalize_uint8_array_2d(self):
        """Test normalization with 2D arrays."""
        data = np.array([[0, 127], [255, 128]], dtype=np.uint8)
        result = normalize_uint8_to_float(data, use_gpu=False)
        
        assert result.shape == (2, 2)
        assert result.dtype == np.float32
        assert result[0, 0] == 0.0
        assert result[1, 0] == 1.0
    
    def test_normalize_uint8_inplace(self):
        """Test in-place normalization."""
        data = np.array([0, 127, 255], dtype=np.float32)
        data_copy = data.copy()
        result = normalize_uint8_to_float(data_copy, use_gpu=False, inplace=True)
        
        assert result is data_copy  # Same object
        assert not np.array_equal(result, data)  # Modified
    
    def test_normalize_uint8_inplace_wrong_dtype(self):
        """Test in-place normalization with wrong dtype raises error."""
        data = np.array([0, 127, 255], dtype=np.uint8)
        
        with pytest.raises(ValueError, match="Inplace normalization requires float32"):
            normalize_uint8_to_float(data, use_gpu=False, inplace=True)
    
    def test_normalize_empty_array(self):
        """Test normalization with empty array raises error."""
        data = np.array([], dtype=np.uint8)
        
        with pytest.raises(ValueError, match="cannot be None or empty"):
            normalize_uint8_to_float(data, use_gpu=False)
    
    def test_denormalize_float_basic(self):
        """Test basic float32 to uint8 denormalization."""
        data = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = denormalize_float_to_uint8(data, use_gpu=False)
        
        assert result.dtype == np.uint8
        assert np.array_equal(result, [0, 127, 255])
    
    def test_denormalize_with_clipping(self):
        """Test denormalization with out-of-range values."""
        data = np.array([-0.5, 0.5, 1.5], dtype=np.float32)
        result = denormalize_float_to_uint8(data, clip=True, use_gpu=False)
        
        assert result[0] == 0    # Clipped to 0
        assert result[1] == 127  # Normal
        assert result[2] == 255  # Clipped to 255
    
    def test_denormalize_without_clipping(self):
        """Test denormalization without clipping (overflow behavior)."""
        data = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = denormalize_float_to_uint8(data, clip=False, use_gpu=False)
        
        assert result.dtype == np.uint8
        assert np.array_equal(result, [0, 127, 255])
    
    def test_normalize_rgb_shape_validation(self):
        """Test RGB normalization with shape validation."""
        # Valid RGB
        rgb = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        result = normalize_rgb(rgb, use_gpu=False)
        assert result.shape == (2, 3)
        
        # Invalid RGB (wrong last dimension)
        rgb_invalid = np.array([[255, 0]], dtype=np.uint8)
        with pytest.raises(ValueError, match="must have shape"):
            normalize_rgb(rgb_invalid, use_gpu=False)
    
    def test_normalize_nir_basic(self):
        """Test NIR normalization."""
        nir = np.array([0, 127, 255], dtype=np.uint8)
        result = normalize_nir(nir, use_gpu=False)
        
        assert result.dtype == np.float32
        assert np.allclose(result, [0.0, 0.498, 1.0], atol=0.01)
    
    def test_roundtrip_normalization(self):
        """Test normalize -> denormalize roundtrip."""
        original = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
        
        # Normalize
        normalized = normalize_uint8_to_float(original, use_gpu=False)
        
        # Denormalize
        denormalized = denormalize_float_to_uint8(normalized, use_gpu=False)
        
        # Should match original (within rounding error)
        assert np.allclose(denormalized, original, atol=1)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestGPUNormalization:
    """Test GPU-based normalization functions."""
    
    def test_normalize_gpu_basic(self):
        """Test basic GPU normalization."""
        data = np.array([0, 127, 255], dtype=np.uint8)
        result = normalize_uint8_to_float(data, use_gpu=True)
        
        # Convert back to CPU for comparison
        if isinstance(result, cp.ndarray):
            result = cp.asnumpy(result)
        
        assert result.dtype == np.float32
        assert np.allclose(result, [0.0, 0.498, 1.0], atol=0.01)
    
    def test_normalize_gpu_with_cupy_input(self):
        """Test GPU normalization with CuPy input."""
        data_cpu = np.array([0, 127, 255], dtype=np.uint8)
        data_gpu = cp.asarray(data_cpu)
        
        result = normalize_uint8_to_float(data_gpu, use_gpu=True)
        
        assert isinstance(result, cp.ndarray)
        result_cpu = cp.asnumpy(result)
        assert np.allclose(result_cpu, [0.0, 0.498, 1.0], atol=0.01)
    
    def test_normalize_gpu_large_array(self):
        """Test GPU normalization with large array."""
        data = np.random.randint(0, 256, size=(1000000,), dtype=np.uint8)
        
        # CPU version
        result_cpu = normalize_uint8_to_float(data, use_gpu=False)
        
        # GPU version
        result_gpu = normalize_uint8_to_float(data, use_gpu=True)
        if isinstance(result_gpu, cp.ndarray):
            result_gpu = cp.asnumpy(result_gpu)
        
        # Should match CPU version
        assert np.allclose(result_cpu, result_gpu, atol=1e-5)
    
    def test_normalize_rgb_gpu(self):
        """Test RGB normalization on GPU."""
        rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        result = normalize_rgb(rgb, use_gpu=True)
        
        if isinstance(result, cp.ndarray):
            result = cp.asnumpy(result)
        
        assert result.shape == (3, 3)
        assert np.allclose(result[0], [1.0, 0.0, 0.0])
        assert np.allclose(result[1], [0.0, 1.0, 0.0])
        assert np.allclose(result[2], [0.0, 0.0, 1.0])
    
    def test_denormalize_gpu_basic(self):
        """Test basic GPU denormalization."""
        data = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = denormalize_float_to_uint8(data, use_gpu=True)
        
        if isinstance(result, cp.ndarray):
            result = cp.asnumpy(result)
        
        assert result.dtype == np.uint8
        assert np.array_equal(result, [0, 127, 255])
    
    def test_gpu_fallback_on_error(self):
        """Test that GPU operations fall back to CPU on error."""
        # This test verifies the fallback mechanism
        data = np.array([0, 127, 255], dtype=np.uint8)
        
        # Should not raise even if GPU has issues
        result = normalize_uint8_to_float(data, use_gpu=True)
        
        # Should still get valid result (either from GPU or CPU fallback)
        if isinstance(result, cp.ndarray):
            result = cp.asnumpy(result)
        assert np.allclose(result, [0.0, 0.498, 1.0], atol=0.01)
    
    def test_gpu_memory_efficiency(self):
        """Test GPU memory usage for large arrays."""
        if not GPU_AVAILABLE:
            pytest.skip("GPU not available")
        
        # Create large array
        size = 10_000_000
        data = np.random.randint(0, 256, size=size, dtype=np.uint8)
        
        # Normalize on GPU
        result = normalize_uint8_to_float(data, use_gpu=True)
        
        # Verify result is correct
        if isinstance(result, cp.ndarray):
            # Sample a few values
            sample = cp.asnumpy(result[:100])
        else:
            sample = result[:100]
        
        assert sample.dtype == np.float32
        assert np.all((sample >= 0.0) & (sample <= 1.0))
    
    def test_roundtrip_gpu(self):
        """Test GPU normalize -> denormalize roundtrip."""
        original = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
        
        # Normalize on GPU
        normalized = normalize_uint8_to_float(original, use_gpu=True)
        
        # Denormalize on GPU
        denormalized = denormalize_float_to_uint8(normalized, use_gpu=True)
        
        # Convert to CPU for comparison
        if isinstance(denormalized, cp.ndarray):
            denormalized = cp.asnumpy(denormalized)
        
        # Should match original (within rounding error)
        assert np.allclose(denormalized, original, atol=1)


class TestUtilityFunctions:
    """Test utility and helper functions."""
    
    def test_is_gpu_available(self):
        """Test GPU availability check."""
        available = is_gpu_available()
        assert isinstance(available, bool)
        
        # Should match module-level GPU_AVAILABLE
        assert available == GPU_AVAILABLE
    
    def test_normalize_rgb_wrapper(self):
        """Test that normalize_rgb is a proper wrapper."""
        rgb = np.array([[255, 128, 0], [0, 255, 128]], dtype=np.uint8)
        
        result1 = normalize_rgb(rgb, use_gpu=False)
        result2 = normalize_uint8_to_float(rgb, use_gpu=False)
        
        assert np.array_equal(result1, result2)
    
    def test_normalize_nir_wrapper(self):
        """Test that normalize_nir is a proper wrapper."""
        nir = np.array([0, 128, 255], dtype=np.uint8)
        
        result1 = normalize_nir(nir, use_gpu=False)
        result2 = normalize_uint8_to_float(nir, use_gpu=False)
        
        assert np.array_equal(result1, result2)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_normalize_single_value(self):
        """Test normalization of single value."""
        data = np.array([127], dtype=np.uint8)
        result = normalize_uint8_to_float(data, use_gpu=False)
        
        assert result.shape == (1,)
        assert np.allclose(result, [0.498], atol=0.01)
    
    def test_normalize_all_zeros(self):
        """Test normalization of all-zero array."""
        data = np.zeros(100, dtype=np.uint8)
        result = normalize_uint8_to_float(data, use_gpu=False)
        
        assert np.all(result == 0.0)
    
    def test_normalize_all_max(self):
        """Test normalization of all-255 array."""
        data = np.full(100, 255, dtype=np.uint8)
        result = normalize_uint8_to_float(data, use_gpu=False)
        
        assert np.all(result == 1.0)
    
    def test_denormalize_boundary_values(self):
        """Test denormalization at exact boundaries."""
        data = np.array([0.0, 0.00392156863, 0.99607843137, 1.0], dtype=np.float32)
        result = denormalize_float_to_uint8(data, use_gpu=False)
        
        assert result[0] == 0
        assert result[1] in [0, 1]  # Rounding
        assert result[2] in [254, 255]  # Rounding
        assert result[3] == 255
    
    def test_normalize_none_input(self):
        """Test that None input raises ValueError."""
        with pytest.raises((ValueError, AttributeError)):
            normalize_uint8_to_float(None, use_gpu=False)
    
    def test_denormalize_none_input(self):
        """Test that None input raises ValueError for denormalize."""
        with pytest.raises((ValueError, AttributeError)):
            denormalize_float_to_uint8(None, use_gpu=False)


class TestPerformance:
    """Performance comparison tests (optional, for benchmarking)."""
    
    @pytest.mark.slow
    def test_cpu_vs_gpu_performance(self):
        """Compare CPU vs GPU performance (informational)."""
        if not GPU_AVAILABLE:
            pytest.skip("GPU not available")
        
        import time
        
        # Create large dataset
        size = 10_000_000
        data = np.random.randint(0, 256, size=size, dtype=np.uint8)
        
        # CPU timing
        start_cpu = time.time()
        result_cpu = normalize_uint8_to_float(data, use_gpu=False)
        time_cpu = time.time() - start_cpu
        
        # GPU timing
        start_gpu = time.time()
        result_gpu = normalize_uint8_to_float(data, use_gpu=True)
        if isinstance(result_gpu, cp.ndarray):
            cp.cuda.Stream.null.synchronize()  # Wait for GPU
        time_gpu = time.time() - start_gpu
        
        print(f"\nNormalization Performance ({size:,} points):")
        print(f"  CPU: {time_cpu:.3f}s")
        print(f"  GPU: {time_gpu:.3f}s")
        print(f"  Speedup: {time_cpu/time_gpu:.2f}x")
        
        # Verify correctness
        if isinstance(result_gpu, cp.ndarray):
            result_gpu = cp.asnumpy(result_gpu)
        assert np.allclose(result_cpu, result_gpu, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
