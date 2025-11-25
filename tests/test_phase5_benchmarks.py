"""
Performance benchmarking suite for Phase 5 unified managers.

Benchmarks:
- GPU Stream Manager throughput and latency
- Performance Manager overhead
- Config Validator performance
- End-to-end pipeline performance
"""

import pytest
import time
import numpy as np
from ign_lidar.core.gpu_stream_manager import get_stream_manager
from ign_lidar.core.performance_manager import get_performance_manager
from ign_lidar.core.config_validator import get_config_validator


class TestGPUStreamBenchmarks:
    """Benchmark GPU Stream Manager performance."""

    @pytest.mark.benchmark
    @pytest.mark.integration
    def test_async_transfer_throughput(self):
        """Benchmark async transfer throughput."""
        streams = get_stream_manager()

        # Prepare data
        transfers = [
            (np.random.rand(1000, 3), np.zeros((1000, 3)))
            for _ in range(100)
        ]

        start = time.time()

        # Process all transfers
        for src, dst in transfers:
            streams.async_transfer(src, dst)

        streams.wait_all()
        duration = time.time() - start

        throughput = len(transfers) / duration
        print(f"\nAsync Transfer Throughput: {throughput:.1f} transfers/sec")
        assert throughput > 10  # At least 10 transfers per second

    @pytest.mark.benchmark
    @pytest.mark.integration
    def test_batch_transfer_efficiency(self):
        """Benchmark batch transfer efficiency."""
        streams = get_stream_manager()

        batch_sizes = [10, 50, 100, 500]
        results = {}

        for batch_size in batch_sizes:
            batch = [
                (np.random.rand(500, 3), np.zeros((500, 3)))
                for _ in range(batch_size)
            ]

            start = time.time()
            streams.batch_transfers(batch)
            streams.wait_all()
            duration = time.time() - start

            throughput = batch_size / duration
            results[batch_size] = throughput
            print(f"\nBatch size {batch_size}: {throughput:.1f} transfers/sec")

        # Larger batches should be more efficient
        assert results[100] >= results[10] * 0.8

    @pytest.mark.benchmark
    @pytest.mark.integration
    def test_stream_allocation_overhead(self):
        """Benchmark stream allocation overhead."""
        streams = get_stream_manager()

        # Get stream multiple times
        iterations = 1000

        start = time.time()
        for _ in range(iterations):
            stream = streams.get_stream()
        duration = time.time() - start

        overhead_us = (duration * 1_000_000) / iterations
        print(f"\nStream allocation overhead: {overhead_us:.2f} µs per call")
        assert overhead_us < 100  # Should be under 100 microseconds


class TestPerformanceManagerBenchmarks:
    """Benchmark Performance Manager overhead."""

    @pytest.mark.benchmark
    @pytest.mark.integration
    def test_phase_tracking_overhead(self):
        """Benchmark phase tracking overhead."""
        perf = get_performance_manager()
        perf.reset()

        # Track many phases
        iterations = 100

        start = time.time()
        for i in range(iterations):
            perf.start_phase(f"phase_{i}")
            perf.end_phase()
        duration = time.time() - start

        overhead_us = (duration * 1_000_000) / (iterations * 2)
        print(f"\nPhase tracking overhead: {overhead_us:.2f} µs per call")
        assert overhead_us < 200  # Should be under 200 microseconds

    @pytest.mark.benchmark
    @pytest.mark.integration
    def test_metric_recording_overhead(self):
        """Benchmark metric recording overhead."""
        perf = get_performance_manager()
        perf.reset()

        perf.start_phase("metrics_test")

        # Record many metrics
        iterations = 1000

        start = time.time()
        for i in range(iterations):
            perf.record_metric(f"metric_{i % 10}", i * 0.1)
        duration = time.time() - start

        perf.end_phase()

        overhead_us = (duration * 1_000_000) / iterations
        print(f"\nMetric recording overhead: {overhead_us:.2f} µs per call")
        assert overhead_us < 500  # Should be under 500 microseconds

    @pytest.mark.benchmark
    @pytest.mark.integration
    def test_summary_generation_performance(self):
        """Benchmark summary generation performance."""
        perf = get_performance_manager()
        perf.reset()

        # Create complex tracking scenario
        for i in range(10):
            perf.start_phase(f"phase_{i}")
            for j in range(100):
                perf.record_metric(f"metric_{j}", j * 0.1)
            perf.end_phase()

        # Benchmark summary generation
        iterations = 100

        start = time.time()
        for _ in range(iterations):
            summary = perf.get_summary()
        duration = time.time() - start

        time_ms = (duration * 1000) / iterations
        print(f"\nSummary generation time: {time_ms:.2f} ms per call")
        assert time_ms < 50  # Should generate summary in under 50ms


class TestConfigValidatorBenchmarks:
    """Benchmark Config Validator performance."""

    @pytest.mark.benchmark
    @pytest.mark.integration
    def test_simple_validation_performance(self):
        """Benchmark simple validation performance."""
        validator = get_config_validator()
        validator.clear_rules()
        validator.add_lod_validator()

        config = {"lod_level": "LOD2"}

        iterations = 1000

        start = time.time()
        for _ in range(iterations):
            is_valid, _ = validator.validate(config)
        duration = time.time() - start

        time_us = (duration * 1_000_000) / iterations
        print(f"\nSimple validation time: {time_us:.2f} µs per call")
        assert time_us < 1000  # Should be under 1ms

    @pytest.mark.benchmark
    @pytest.mark.integration
    def test_complex_validation_performance(self):
        """Benchmark complex validation performance."""
        validator = get_config_validator()
        validator.clear_rules()

        # Add multiple validators
        validator.add_lod_validator()
        validator.add_gpu_validator()
        validator.add_numeric_range_validator("batch_size", 1, 10000)
        validator.add_numeric_range_validator("learning_rate", 0.0001, 0.1)

        config = {
            "lod_level": "LOD2",
            "gpu_memory_fraction": 0.8,
            "batch_size": 256,
            "learning_rate": 0.001,
        }

        iterations = 1000

        start = time.time()
        for _ in range(iterations):
            is_valid, _ = validator.validate(config)
        duration = time.time() - start

        time_us = (duration * 1_000_000) / iterations
        print(f"\nComplex validation time: {time_us:.2f} µs per call")
        assert time_us < 5000  # Should be under 5ms

    @pytest.mark.benchmark
    @pytest.mark.integration
    def test_rule_addition_performance(self):
        """Benchmark rule addition performance."""
        validator = get_config_validator()

        iterations = 100

        start = time.time()
        for i in range(iterations):
            validator.add_numeric_range_validator(f"param_{i}", 0, 100)
        duration = time.time() - start

        time_us = (duration * 1_000_000) / iterations
        print(f"\nRule addition time: {time_us:.2f} µs per call")
        assert time_us < 10000  # Should be under 10ms


class TestEndToEndBenchmarks:
    """Benchmark end-to-end pipeline performance."""

    @pytest.mark.benchmark
    @pytest.mark.integration
    def test_complete_pipeline_performance(self):
        """Benchmark complete pipeline with all three managers."""
        validator = get_config_validator()
        validator.clear_rules()
        validator.add_lod_validator()
        validator.add_gpu_validator()

        perf = get_performance_manager()
        perf.reset()

        streams = get_stream_manager()

        iterations = 10

        start = time.time()

        for iteration in range(iterations):
            # Validate config
            perf.start_phase("validate")
            config = {
                "lod_level": "LOD2",
                "gpu_memory_fraction": 0.8,
                "batch_size": 256,
            }
            is_valid, _ = validator.validate(config)
            perf.end_phase()

            if not is_valid:
                continue

            # Transfer data
            perf.start_phase("transfer")
            batch = [
                (np.random.rand(256, 3), np.zeros((256, 3)))
                for _ in range(5)
            ]
            streams.batch_transfers(batch)
            streams.wait_all()
            perf.end_phase()

            # Process
            perf.start_phase("process")
            for i in range(len(batch)):
                perf.record_metric("accuracy", 0.85 + i * 0.01)
            perf.end_phase()

        duration = time.time() - start
        time_per_iteration = (duration * 1000) / iterations

        print(f"\nEnd-to-end iteration time: {time_per_iteration:.2f} ms")
        assert time_per_iteration < 1000  # Should be under 1 second per iteration

    @pytest.mark.benchmark
    @pytest.mark.integration
    def test_scalability_with_data_size(self):
        """Benchmark scalability with varying data sizes."""
        streams = get_stream_manager()
        perf = get_performance_manager()
        perf.reset()

        data_sizes = [100, 500, 1000, 5000]
        results = {}

        for size in data_sizes:
            perf.reset()
            perf.start_phase(f"transfer_{size}")

            batch = [
                (np.random.rand(size, 3), np.zeros((size, 3)))
                for _ in range(10)
            ]

            start = time.time()
            streams.batch_transfers(batch)
            streams.wait_all()
            duration = time.time() - start

            perf.end_phase()

            time_ms = duration * 1000
            results[size] = time_ms
            print(f"\nData size {size}: {time_ms:.2f} ms for 10 transfers")

        # Should scale reasonably with data size
        assert results[500] < results[100] * 20  # Roughly linear


class TestMemoryEfficiency:
    """Benchmark memory efficiency of managers."""

    @pytest.mark.benchmark
    @pytest.mark.integration
    def test_manager_memory_footprint(self):
        """Benchmark manager memory footprint."""
        import sys

        # Get stream manager
        streams = get_stream_manager()
        stream_size = sys.getsizeof(streams)

        # Get performance manager
        perf = get_performance_manager()
        perf_size = sys.getsizeof(perf)

        # Get validator
        validator = get_config_validator()
        validator_size = sys.getsizeof(validator)

        total_size = stream_size + perf_size + validator_size

        print(f"\nManager memory footprints:")
        print(f"  GPU Stream Manager: {stream_size / 1024:.2f} KB")
        print(f"  Performance Manager: {perf_size / 1024:.2f} KB")
        print(f"  Config Validator: {validator_size / 1024:.2f} KB")
        print(f"  Total: {total_size / 1024:.2f} KB")

        # Total should be under 1 MB
        assert total_size < 1_000_000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "benchmark"])
