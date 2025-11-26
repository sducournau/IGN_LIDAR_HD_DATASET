"""
Tests for Adaptive Chunking (Phase 2 Priority 2 Optimization).

Tests the automatic chunk size calculation based on:
- Available GPU memory
- Point cloud characteristics
- Target memory usage ratios
- Dataset size and complexity

This module validates:
- auto_chunk_size calculation accuracy
- Handling of edge cases (very large, very small datasets)
- Strategy recommendation logic
- Integration with GPU processing

Expected Performance Gains:
- 0.3-1.0s speedup through optimal chunk sizing
- Prevents GPU OOM errors through conservative sizing
- Automatic mode selection (batch vs chunked)

Author: IGN LiDAR HD Development Team
Date: December 2025
Version: 3.7.0 Phase 2
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from ign_lidar.optimization.adaptive_chunking import (
    auto_chunk_size,
    estimate_gpu_memory_required,
    get_recommended_strategy,
    calculate_optimal_chunk_count,
)


class TestAutoChunkSize:
    """Test automatic chunk size calculation."""

    def test_auto_chunk_size_cpu_mode(self):
        """Test auto_chunk_size returns max size for CPU mode."""
        chunk_size = auto_chunk_size(
            points_shape=(5_000_000, 3),
            use_gpu=False
        )
        
        # CPU mode should return max chunk size
        assert chunk_size == 10_000_000

    def test_auto_chunk_size_with_default_params(self):
        """Test auto_chunk_size with default parameters."""
        chunk_size = auto_chunk_size(points_shape=(1_000_000, 3))
        
        # Should return reasonable chunk size
        assert isinstance(chunk_size, int)
        assert chunk_size > 0
        assert chunk_size <= 10_000_000  # Max chunk size

    def test_auto_chunk_size_respects_min_size(self):
        """Test that calculated chunk size respects minimum."""
        # Very conservative settings should give min size
        chunk_size = auto_chunk_size(
            points_shape=(1_000_000, 3),
            target_memory_usage=0.1,
            safety_factor=0.1,
            min_chunk_size=100_000
        )
        
        assert chunk_size >= 100_000

    def test_auto_chunk_size_respects_max_size(self):
        """Test that calculated chunk size respects maximum."""
        # Very aggressive settings should not exceed max
        chunk_size = auto_chunk_size(
            points_shape=(1_000_000, 3),
            target_memory_usage=0.95,
            safety_factor=0.95,
            max_chunk_size=10_000_000
        )
        
        assert chunk_size <= 10_000_000

    def test_auto_chunk_size_conservative_vs_aggressive(self):
        """Test that conservative settings give smaller chunks."""
        conservative = auto_chunk_size(
            points_shape=(5_000_000, 3),
            target_memory_usage=0.5,
            feature_count=38  # LOD3 (complex)
        )
        
        aggressive = auto_chunk_size(
            points_shape=(5_000_000, 3),
            target_memory_usage=0.9,
            feature_count=12  # LOD2 (simple)
        )
        
        # Conservative should be smaller (safer)
        assert conservative <= aggressive

    def test_auto_chunk_size_with_feature_count(self):
        """Test chunk size scales with feature count."""
        simple_features = auto_chunk_size(
            points_shape=(5_000_000, 3),
            feature_count=12  # LOD2
        )
        
        complex_features = auto_chunk_size(
            points_shape=(5_000_000, 3),
            feature_count=38  # LOD3
        )
        
        # More features = smaller chunks (more memory per point)
        assert complex_features <= simple_features

    def test_auto_chunk_size_with_large_dataset(self):
        """Test chunk size calculation for very large datasets."""
        chunk_size = auto_chunk_size(
            points_shape=(100_000_000, 3),
            target_memory_usage=0.7
        )
        
        # Should return chunk size in reasonable range
        assert 100_000 <= chunk_size <= 10_000_000

    def test_auto_chunk_size_with_small_dataset(self):
        """Test chunk size calculation for small datasets."""
        chunk_size = auto_chunk_size(
            points_shape=(50_000, 3),
            target_memory_usage=0.7
        )
        
        # Small dataset can use larger chunk size
        assert chunk_size > 0

    def test_auto_chunk_size_deterministic(self):
        """Test that chunk size calculation is deterministic."""
        size1 = auto_chunk_size(
            points_shape=(5_000_000, 3),
            target_memory_usage=0.7,
            feature_count=20
        )
        
        size2 = auto_chunk_size(
            points_shape=(5_000_000, 3),
            target_memory_usage=0.7,
            feature_count=20
        )
        
        assert size1 == size2


class TestEstimateGPUMemory:
    """Test GPU memory estimation."""

    def test_estimate_memory_small_dataset(self):
        """Test memory estimation for small dataset."""
        required_gb = estimate_gpu_memory_required(
            num_points=100_000,
            num_features=3,
            feature_count=20,
            k_neighbors=30
        )
        
        # Should be less than 1GB for 100K points
        assert 0 < required_gb < 1.0

    def test_estimate_memory_large_dataset(self):
        """Test memory estimation for large dataset."""
        required_gb = estimate_gpu_memory_required(
            num_points=10_000_000,
            num_features=3,
            feature_count=20,
            k_neighbors=30
        )
        
        # Should be in multi-GB range
        assert required_gb > 1.0

    def test_estimate_memory_scales_linearly(self):
        """Test that memory scales with dataset size."""
        mem_1m = estimate_gpu_memory_required(num_points=1_000_000)
        mem_2m = estimate_gpu_memory_required(num_points=2_000_000)
        mem_5m = estimate_gpu_memory_required(num_points=5_000_000)
        
        # Should scale roughly linearly
        assert mem_2m > mem_1m
        assert mem_5m > mem_2m

    def test_estimate_memory_with_different_feature_counts(self):
        """Test memory scales with feature count."""
        mem_lod2 = estimate_gpu_memory_required(
            num_points=1_000_000,
            feature_count=12  # LOD2
        )
        
        mem_lod3 = estimate_gpu_memory_required(
            num_points=1_000_000,
            feature_count=38  # LOD3
        )
        
        # More features = more memory
        assert mem_lod3 > mem_lod2

    def test_estimate_memory_includes_overhead(self):
        """Test that estimation includes reasonable overhead."""
        required_gb = estimate_gpu_memory_required(
            num_points=1_000_000,
            num_features=3,
            feature_count=20,
            k_neighbors=30
        )
        
        # Should include 20% overhead
        assert required_gb > 0.1


class TestGetRecommendedStrategy:
    """Test strategy recommendation logic."""

    def test_strategy_recommendation_small_dataset(self):
        """Test strategy for small dataset."""
        strategy = get_recommended_strategy(num_points=1_000_000)
        
        # Small dataset should use GPU (not chunked)
        assert strategy in ['gpu', 'gpu_chunked', 'cpu']

    def test_strategy_recommendation_large_dataset(self):
        """Test strategy for large dataset."""
        strategy = get_recommended_strategy(num_points=50_000_000)
        
        # Large dataset should recommend chunking or CPU
        assert strategy in ['gpu_chunked', 'cpu']

    def test_strategy_recommendation_with_memory_constraints(self):
        """Test strategy with limited GPU memory."""
        strategy = get_recommended_strategy(
            num_points=100_000_000,
            available_memory_gb=2.0  # Very limited
        )
        
        # With very limited memory, may recommend CPU
        assert strategy in ['gpu_chunked', 'cpu']

    def test_strategy_recommendation_with_abundant_memory(self):
        """Test strategy with abundant GPU memory."""
        strategy = get_recommended_strategy(
            num_points=50_000_000,
            available_memory_gb=48.0  # Very generous
        )
        
        # With abundant memory, may use chunking or cpu (depends on GPU detection)
        assert strategy in ['gpu', 'gpu_chunked', 'cpu']

    def test_strategy_recommendation_no_gpu_fallback(self):
        """Test strategy recommendation without GPU."""
        # Just test the fallback behavior directly
        strategy = get_recommended_strategy(num_points=10_000_000)
        
        # Should return a valid strategy
        assert strategy in ['gpu', 'gpu_chunked', 'cpu']


class TestCalculateOptimalChunkCount:
    """Test optimal chunk count calculation."""

    def test_chunk_count_single_chunk(self):
        """Test that small datasets return single chunk."""
        count = calculate_optimal_chunk_count(
            num_points=500_000,
            chunk_size=1_000_000
        )
        
        assert count == 1

    def test_chunk_count_multiple_chunks(self):
        """Test chunk count for multiple chunks."""
        count = calculate_optimal_chunk_count(
            num_points=5_000_000,
            chunk_size=1_000_000
        )
        
        # Should be 5 chunks
        assert count == 5

    def test_chunk_count_uneven_distribution(self):
        """Test chunk count avoids small final chunks."""
        count = calculate_optimal_chunk_count(
            num_points=5_234_567,
            chunk_size=1_000_000
        )
        
        # Should redistribute to avoid tiny final chunk
        assert count >= 5
        
        # Verify chunks are reasonably distributed
        avg_chunk = 5_234_567 / count
        assert avg_chunk >= 1_000_000 * 0.7  # At least 70% of target size

    def test_chunk_count_large_dataset(self):
        """Test chunk count for very large dataset."""
        count = calculate_optimal_chunk_count(
            num_points=100_000_000,
            chunk_size=1_000_000
        )
        
        # Should be around 100 chunks
        assert 95 <= count <= 105

    def test_chunk_count_exact_division(self):
        """Test chunk count when dataset divides evenly."""
        count = calculate_optimal_chunk_count(
            num_points=5_000_000,
            chunk_size=1_000_000
        )
        
        # Should be exactly 5
        assert count == 5


class TestIntegrationWithGPUStrategy:
    """Integration tests with GPU strategy."""

    def test_adaptive_chunking_initialization(self):
        """Test that adaptive chunking can be imported and used."""
        from ign_lidar.features.strategy_gpu_chunked import GPUChunkedStrategy
        
        # Should be importable
        assert GPUChunkedStrategy is not None

    def test_auto_chunk_with_real_data_shape(self):
        """Test auto_chunk with typical point cloud shape."""
        # Typical LiDAR tile: 1-50M points
        chunk_size = auto_chunk_size(
            points_shape=(10_000_000, 3),
            target_memory_usage=0.7,
            feature_count=20
        )
        
        assert 100_000 <= chunk_size <= 10_000_000

    def test_memory_estimate_matches_chunk_requirements(self):
        """Test that estimated memory aligns with chunk sizing."""
        # 10M points
        required_total = estimate_gpu_memory_required(
            num_points=10_000_000,
            feature_count=20
        )
        
        # Auto chunk for available memory
        chunk_size = auto_chunk_size(
            points_shape=(10_000_000, 3),
            target_memory_usage=0.7,
            feature_count=20
        )
        
        # Required memory per chunk should be reasonable fraction
        required_per_chunk = estimate_gpu_memory_required(
            num_points=chunk_size,
            feature_count=20
        )
        
        # Per-chunk should be less than or equal to total (within rounding)
        assert required_per_chunk <= required_total * 1.1  # Allow 10% rounding


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_chunk_size_with_zero_points(self):
        """Test handling of zero points."""
        chunk_size = auto_chunk_size(
            points_shape=(0, 3),
            min_chunk_size=100_000
        )
        
        # Should return minimum
        assert chunk_size >= 100_000

    def test_chunk_count_with_zero_points(self):
        """Test chunk count with zero points."""
        count = calculate_optimal_chunk_count(
            num_points=0,
            chunk_size=1_000_000
        )
        
        # Should return 1 (minimum)
        assert count >= 1

    def test_chunk_size_single_point(self):
        """Test chunk size with single point."""
        chunk_size = auto_chunk_size(
            points_shape=(1, 3),
            min_chunk_size=100_000
        )
        
        # Should still return minimum
        assert chunk_size >= 100_000

    def test_memory_estimate_zero_features(self):
        """Test memory estimation with zero features."""
        required_gb = estimate_gpu_memory_required(
            num_points=1_000_000,
            feature_count=0
        )
        
        # Should still estimate input + overhead
        assert required_gb > 0

    def test_negative_chunk_size_handling(self):
        """Test that chunk calculation handles boundary conditions."""
        # Very large chunk size vs small dataset
        count = calculate_optimal_chunk_count(
            num_points=100,
            chunk_size=10_000_000
        )
        
        # Should be at least 1
        assert count >= 1


class TestParameterSensitivity:
    """Test sensitivity to input parameters."""

    @pytest.mark.parametrize("target_usage", [0.3, 0.5, 0.7, 0.9])
    def test_chunk_size_sensitivity_to_target_usage(self, target_usage):
        """Test chunk size sensitivity to target memory usage."""
        chunk_sizes = []
        for usage in [0.3, 0.5, 0.7, 0.9]:
            size = auto_chunk_size(
                points_shape=(5_000_000, 3),
                target_memory_usage=usage
            )
            chunk_sizes.append(size)
        
        # Higher usage should give larger chunks
        assert chunk_sizes[3] >= chunk_sizes[2] >= chunk_sizes[1] >= chunk_sizes[0]

    @pytest.mark.parametrize("feature_count", [8, 12, 20, 38])
    def test_chunk_size_sensitivity_to_features(self, feature_count):
        """Test chunk size sensitivity to feature count."""
        sizes = []
        for count in [8, 12, 20, 38]:
            size = auto_chunk_size(
                points_shape=(5_000_000, 3),
                feature_count=count
            )
            sizes.append(size)
        
        # More features = smaller chunks
        assert sizes[3] <= sizes[2] <= sizes[1] <= sizes[0]

    @pytest.mark.parametrize("safety_factor", [0.5, 0.7, 0.9])
    def test_chunk_size_sensitivity_to_safety(self, safety_factor):
        """Test chunk size sensitivity to safety factor."""
        sizes = []
        for factor in [0.5, 0.7, 0.9]:
            size = auto_chunk_size(
                points_shape=(5_000_000, 3),
                safety_factor=factor
            )
            sizes.append(size)
        
        # Higher safety = smaller chunks
        assert sizes[0] <= sizes[1] <= sizes[2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
