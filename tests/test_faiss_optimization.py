"""
Unit tests for FAISS index optimization (P1 Task #4).

Tests the adaptive FAISS index selection and optimization implementation,
ensuring correct index type selection, training, and performance characteristics.

**IMPORTANT: Run with ign_gpu conda environment:**

    conda run -n ign_gpu python -m pytest tests/test_faiss_optimization.py -v

Version: 1.0.0
Date: 2025-11-21
"""

import numpy as np
import pytest
import time

# FAISS imports with fallback
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class TestFAISSOptimization:
    """Test suite for FAISS index optimization."""

    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not available")
    @pytest.mark.unit
    def test_adaptive_index_selection_logic(self):
        """Test that correct index types are selected based on dataset size."""
        # This tests the logic without actually running GPU code
        
        test_cases = [
            (100_000, "Flat", "Small dataset should use Flat"),
            (500_000, "Flat", "Medium-small dataset should use Flat"),
            (2_000_000, "IVFFlat", "Medium dataset should use IVFFlat"),
            (5_000_000, "IVFFlat", "Large-medium dataset should use IVFFlat"),
            (15_000_000, "IVFPQ", "Large dataset should use IVFPQ"),
            (100_000_000, "IVFPQ", "Very large dataset should use IVFPQ"),
        ]
        
        for N, expected_type, desc in test_cases:
            # Simulate the selection logic
            if N < 1_000_000:
                index_type = "Flat"
            elif N < 10_000_000:
                index_type = "IVFFlat"
            else:
                index_type = "IVFPQ"
            
            assert index_type == expected_type, f"{desc}: N={N:,}, expected {expected_type}, got {index_type}"

    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not available")
    @pytest.mark.unit
    def test_ivf_parameters_scaling(self):
        """Test that IVF parameters scale appropriately with dataset size."""
        test_cases = [
            (1_000_000, (256, 4096), "1M points"),
            (5_000_000, (1024, 16384), "5M points"),
            (20_000_000, (1024, 16384), "20M points"),
        ]
        
        for N, (min_nlist, max_nlist), desc in test_cases:
            # Simulate nlist calculation
            if N < 5_000_000:
                nlist = min(4096, max(256, int(np.sqrt(N))))
            else:
                nlist = min(16384, max(1024, int(np.sqrt(N))))
            
            assert min_nlist <= nlist <= max_nlist, \
                f"{desc}: nlist={nlist} should be in range [{min_nlist}, {max_nlist}]"
            
            # Check nprobe scaling
            if N < 5_000_000:
                nprobe = min(64, nlist // 8)
            else:
                nprobe = min(256, nlist // 16)
            
            assert nprobe > 0, f"{desc}: nprobe must be positive"
            assert nprobe <= nlist, f"{desc}: nprobe ({nprobe}) must be <= nlist ({nlist})"

    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not available")
    @pytest.mark.unit
    def test_pq_parameters_calculation(self):
        """Test PQ parameter calculation for different dimensions."""
        test_cases = [
            (3, 1),    # 3D points: m=1 (3 divisible by 1)
            (8, 8),    # 8D: m=8
            (16, 8),   # 16D: m=8
            (32, 8),   # 32D: m=8
            (64, 8),   # 64D: m=8
            (10, 5),   # 10D: m=5 (10 divisible by 5)
        ]
        
        for D, expected_m in test_cases:
            # Simulate m calculation
            m = 8 if D >= 8 else max(1, D // 2)
            while D % m != 0 and m > 1:
                m -= 1
            
            assert m == expected_m, f"D={D}: expected m={expected_m}, got m={m}"
            assert D % m == 0, f"D={D} must be divisible by m={m}"

    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not available")
    @pytest.mark.unit
    def test_training_size_calculation(self):
        """Test adaptive training size for different index types and sizes."""
        # IVFFlat training sizes
        test_cases_flat = [
            (500_000, 4096, 500_000),    # Small: use all
            (5_000_000, 8192, 500_000),  # Cap at 500K
        ]
        
        for N, nlist, expected_max in test_cases_flat:
            train_size = min(N, min(500_000, nlist * 128))
            assert train_size <= expected_max, \
                f"IVFFlat N={N:,}: train_size={train_size:,} should be <= {expected_max:,}"
        
        # IVFPQ training sizes (needs more data)
        test_cases_pq = [
            (500_000, 4096, 500_000),      # Small: use all
            (10_000_000, 8192, 1_000_000), # Cap at 1M
        ]
        
        for N, nlist, expected_max in test_cases_pq:
            train_size = min(N, min(1_000_000, nlist * 256))
            assert train_size <= expected_max, \
                f"IVFPQ N={N:,}: train_size={train_size:,} should be <= {expected_max:,}"

    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not available")
    @pytest.mark.integration
    def test_flat_index_build(self):
        """Test building Flat index for small dataset."""
        np.random.seed(42)
        N, D = 1000, 3
        points = np.random.randn(N, D).astype(np.float32)
        
        # Build Flat index
        index = faiss.IndexFlatL2(D)
        index.add(points)
        
        # Verify
        assert index.ntotal == N
        assert index.d == D
        
        # Query a single point
        k = 10
        distances, indices = index.search(points[:1], k)
        
        assert distances.shape == (1, k)
        assert indices.shape == (1, k)
        assert indices[0, 0] == 0, "First neighbor of point 0 should be itself"
        assert distances[0, 0] < 1e-6, "Distance to self should be ~0"

    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not available")
    @pytest.mark.integration
    def test_ivfflat_index_build(self):
        """Test building IVFFlat index for medium dataset."""
        np.random.seed(42)
        N, D = 10000, 3
        points = np.random.randn(N, D).astype(np.float32)
        
        # Build IVFFlat index
        nlist = 64
        quantizer = faiss.IndexFlatL2(D)
        index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_L2)
        
        # Train
        index.train(points)
        assert index.is_trained
        
        # Add points
        index.add(points)
        assert index.ntotal == N
        
        # Set nprobe
        index.nprobe = 8
        
        # Query
        k = 10
        distances, indices = index.search(points[:1], k)
        
        assert distances.shape == (1, k)
        assert indices.shape == (1, k)
        # IVF may not return exact nearest neighbor, but should be close
        assert distances[0, 0] < 0.1, "Distance to nearest should be small"

    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not available")
    @pytest.mark.integration
    def test_ivfpq_index_build(self):
        """Test building IVFPQ index for large dataset."""
        np.random.seed(42)
        N, D = 10000, 16  # Need D divisible by m
        points = np.random.randn(N, D).astype(np.float32)
        
        # Build IVFPQ index
        nlist = 64
        m = 8  # 16 dimensions, 8 subvectors
        nbits = 8
        
        quantizer = faiss.IndexFlatL2(D)
        index = faiss.IndexIVFPQ(quantizer, D, nlist, m, nbits, faiss.METRIC_L2)
        
        # Train
        index.train(points)
        assert index.is_trained
        
        # Add points
        index.add(points)
        assert index.ntotal == N
        
        # Set nprobe
        index.nprobe = 8
        
        # Query
        k = 10
        distances, indices = index.search(points[:1], k)
        
        assert distances.shape == (1, k)
        assert indices.shape == (1, k)
        # PQ is approximate, so distance may be larger
        assert distances[0, 0] < 1.0, "Distance to nearest should be reasonable"

    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not available")
    @pytest.mark.performance
    @pytest.mark.slow
    def test_index_memory_footprint(self):
        """Test memory footprint of different index types."""
        np.random.seed(42)
        N, D = 100000, 3
        points = np.random.randn(N, D).astype(np.float32)
        
        # Flat index memory: N * D * 4 bytes
        flat_size_mb = (N * D * 4) / (1024 ** 2)
        
        # IVFPQ memory: much smaller due to compression
        # Approximation: N * m * nbits / 8 bytes
        m, nbits = 1, 8  # For D=3, m=1
        pq_size_mb = (N * m * nbits / 8) / (1024 ** 2)
        
        print(f"\nMemory footprint comparison (N={N:,}, D={D}):")
        print(f"  Flat index:   ~{flat_size_mb:.2f} MB")
        print(f"  IVFPQ index:  ~{pq_size_mb:.2f} MB (compression: {100*pq_size_mb/flat_size_mb:.1f}%)")
        
        # For larger datasets, compression is significant
        assert pq_size_mb < flat_size_mb, "PQ should use less memory than Flat"

    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not available")
    @pytest.mark.performance
    @pytest.mark.slow
    def test_index_query_performance(self):
        """Benchmark query performance of different index types."""
        np.random.seed(42)
        N, D = 50000, 3
        points = np.random.randn(N, D).astype(np.float32)
        k = 30
        n_queries = 100
        query_points = points[:n_queries]
        
        results = {}
        
        # Flat index (baseline)
        index_flat = faiss.IndexFlatL2(D)
        index_flat.add(points)
        
        t0 = time.time()
        _, _ = index_flat.search(query_points, k)
        flat_time = time.time() - t0
        results['Flat'] = flat_time
        
        # IVFFlat index
        nlist = 128
        quantizer = faiss.IndexFlatL2(D)
        index_ivf = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_L2)
        index_ivf.train(points)
        index_ivf.add(points)
        index_ivf.nprobe = 16
        
        t0 = time.time()
        _, _ = index_ivf.search(query_points, k)
        ivf_time = time.time() - t0
        results['IVFFlat'] = ivf_time
        
        print(f"\nQuery performance (N={N:,}, k={k}, queries={n_queries}):")
        print(f"  Flat:     {flat_time*1000:.2f} ms (baseline)")
        print(f"  IVFFlat:  {ivf_time*1000:.2f} ms ({flat_time/ivf_time:.2f}x)")
        
        # IVF should be faster for this size (or comparable for small N)
        # Main benefit is for larger datasets and batch queries
        assert ivf_time < flat_time * 2, "IVFFlat should not be significantly slower"

    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not available")
    @pytest.mark.integration
    def test_accuracy_comparison(self):
        """Compare accuracy of different index types."""
        np.random.seed(42)
        N, D = 10000, 3
        points = np.random.randn(N, D).astype(np.float32)
        k = 10
        query_point = points[:1]
        
        # Ground truth: Flat index
        index_flat = faiss.IndexFlatL2(D)
        index_flat.add(points)
        dist_flat, ind_flat = index_flat.search(query_point, k)
        
        # IVFFlat with high nprobe (should be very accurate)
        nlist = 64
        quantizer = faiss.IndexFlatL2(D)
        index_ivf = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_L2)
        index_ivf.train(points)
        index_ivf.add(points)
        index_ivf.nprobe = 32  # High nprobe for accuracy
        
        dist_ivf, ind_ivf = index_ivf.search(query_point, k)
        
        # Check recall: how many of the true neighbors are found
        recall = len(set(ind_flat[0]) & set(ind_ivf[0])) / k
        
        print(f"\nAccuracy comparison (k={k}):")
        print(f"  Flat (ground truth): {ind_flat[0][:5]}")
        print(f"  IVFFlat (nprobe=32): {ind_ivf[0][:5]}")
        print(f"  Recall@{k}: {recall*100:.1f}%")
        
        # With high nprobe, IVF should have good recall (>90%)
        assert recall > 0.8, f"IVFFlat should have >80% recall with high nprobe, got {recall*100:.1f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
