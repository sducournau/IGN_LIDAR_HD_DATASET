"""
Phase 5 Unified Managers - Usage Examples

This module demonstrates usage of the three Phase 5 facade managers:
1. GPUStreamManager - Unified GPU stream management
2. PerformanceManager - Unified performance monitoring
3. ConfigValidator - Unified configuration validation

Topics covered:
1. GPU stream management (high-level and low-level)
2. Performance monitoring workflows
3. Configuration validation patterns
4. Integration patterns
"""

import numpy as np
from omegaconf import OmegaConf
import logging

# Import Phase 5 managers
from ign_lidar.core.gpu_stream_manager import GPUStreamManager, get_stream_manager
from ign_lidar.core.performance_manager import PerformanceManager, get_performance_manager
from ign_lidar.core.config_validator import ConfigValidator, get_config_validator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ============================================================================
# Example 1: GPU Stream Management (High-Level)
# ============================================================================

def example_1_gpu_stream_high_level():
    """
    Simple GPU stream management with automatic handling.

    HIGH-LEVEL API: Recommended for most use cases.

    Output:
        Successful async transfers with automatic synchronization
    """
    print("\n" + "=" * 70)
    print("Example 1: GPU Stream Management (High-Level)")
    print("=" * 70)

    # Get stream manager
    manager = get_stream_manager(pool_size=4)

    # Simulate data transfers
    print("\nPerforming async transfers...")
    for i in range(5):
        src = np.random.rand(1000, 3)
        dst = np.zeros((1000, 3))

        manager.async_transfer(src, dst, size_mb=10)
        print(f"  ✓ Transfer {i+1} initiated")

    # Wait for all to complete
    print("\nWaiting for all transfers...")
    manager.wait_all()
    print("  ✓ All transfers complete")

    # Get stats
    stats = manager.get_performance_stats()
    print(f"\nTransfer stats: {stats['transfer_stats']}")

    return manager


# ============================================================================
# Example 2: GPU Stream Management (Batch Processing)
# ============================================================================

def example_2_gpu_stream_batch():
    """
    Batch GPU transfers with efficient batching.

    HIGH-LEVEL API: Automatic load balancing.

    Output:
        Multiple transfers processed efficiently
    """
    print("\n" + "=" * 70)
    print("Example 2: GPU Stream Batch Processing")
    print("=" * 70)

    manager = get_stream_manager(pool_size=4)

    # Prepare batch of transfers
    print("\nPreparing batch of transfers...")
    transfers = [
        (np.random.rand(1000, 3), np.zeros((1000, 3)))
        for _ in range(10)
    ]
    print(f"  ✓ Prepared {len(transfers)} transfers")

    # Process batch
    print("\nProcessing batch...")
    manager.batch_transfers(transfers)
    print("  ✓ All transfers initiated")

    # Wait and sync
    manager.wait_all()
    print("  ✓ Batch complete")

    return manager


# ============================================================================
# Example 3: GPU Stream Management (Low-Level)
# ============================================================================

def example_3_gpu_stream_low_level():
    """
    Direct stream access for advanced control.

    LOW-LEVEL API: Manual stream management.

    Output:
        Fine-grained stream control
    """
    print("\n" + "=" * 70)
    print("Example 3: GPU Stream Management (Low-Level)")
    print("=" * 70)

    manager = get_stream_manager(pool_size=4)

    # Get specific stream
    print("\nManual stream control...")
    stream = manager.get_stream(0)
    print(f"  ✓ Got stream: {stream}")

    # Perform transfer on specific stream
    src = np.random.rand(1000, 3)
    dst = np.zeros((1000, 3))

    stream.transfer_async(src, dst)
    print("  ✓ Transfer initiated on stream 0")

    # Synchronize
    stream.synchronize()
    print("  ✓ Stream synchronized")

    # Get stats
    stats = stream.get_stats()
    print(f"  Stream stats: {stats}")

    return manager


# ============================================================================
# Example 4: Performance Monitoring (High-Level)
# ============================================================================

def example_4_performance_high_level():
    """
    Automatic performance tracking.

    HIGH-LEVEL API: Simple phase-based timing.

    Output:
        Automatic metrics collection
    """
    print("\n" + "=" * 70)
    print("Example 4: Performance Monitoring (High-Level)")
    print("=" * 70)

    manager = get_performance_manager()
    manager.reset()

    # Track data loading phase
    print("\nTracking phases...")
    manager.start_phase("data_loading")
    import time

    time.sleep(0.05)
    manager.end_phase()
    print("  ✓ Data loading tracked")

    # Track processing phase
    manager.start_phase("feature_computation")
    time.sleep(0.1)
    manager.end_phase()
    print("  ✓ Feature computation tracked")

    # Track classification phase
    manager.start_phase("classification")
    time.sleep(0.02)
    manager.end_phase()
    print("  ✓ Classification tracked")

    # Get summary
    print("\nPerformance Summary:")
    summary = manager.get_summary()
    print(f"  Total time: {summary['total_time']:.3f}s")
    print(f"  Number of phases: {summary['num_phases']}")

    for phase_name, metrics in summary["phases"].items():
        print(f"\n  {phase_name}:")
        print(f"    Duration: {metrics['duration']:.3f}s")
        print(f"    Memory: {metrics['memory_mb']:.1f} MB")

    return manager


# ============================================================================
# Example 5: Performance Monitoring (Custom Metrics)
# ============================================================================

def example_5_performance_custom_metrics():
    """
    Custom metric recording.

    HIGH-LEVEL API: User-defined metrics.

    Output:
        Custom metrics integrated with phases
    """
    print("\n" + "=" * 70)
    print("Example 5: Performance Monitoring (Custom Metrics)")
    print("=" * 70)

    manager = get_performance_manager()
    manager.reset()

    print("\nTracking processing with custom metrics...")

    manager.start_phase("processing")

    # Simulate processing with metrics
    for i in range(5):
        accuracy = 0.90 + i * 0.02
        manager.record_metric("batch_accuracy", accuracy)
        print(f"  Batch {i}: accuracy={accuracy:.2f}")

    manager.end_phase()

    # Get stats
    stats = manager.get_phase_stats("processing")
    print(f"\nPhase stats:")
    print(f"  Duration: {stats['duration']:.3f}s")
    print(f"  Custom metrics: {stats['custom_metrics']}")

    return manager


# ============================================================================
# Example 6: Configuration Validation (High-Level)
# ============================================================================

def example_6_config_validation_high_level():
    """
    Simple configuration validation.

    HIGH-LEVEL API: Easy validation.

    Output:
        Configuration validation results
    """
    print("\n" + "=" * 70)
    print("Example 6: Configuration Validation (High-Level)")
    print("=" * 70)

    validator = get_config_validator()
    validator.clear_rules()
    validator.add_required_field("processor")
    validator.add_field_type("processor", dict)

    # Test valid config
    print("\nValidating configuration...")
    config = {"processor": {"lod_level": "LOD2"}}

    is_valid, errors = validator.validate(config)

    if is_valid:
        print("  ✓ Configuration is valid")
    else:
        print("  ✗ Configuration errors:")
        for error in errors:
            print(f"    {error}")

    # Test invalid config
    print("\nValidating invalid configuration...")
    bad_config = {"other_field": "value"}

    is_valid, errors = bad_config
    if not is_valid:
        print("  ✓ Errors detected as expected")

    return validator


# ============================================================================
# Example 7: Configuration Validation (Advanced Rules)
# ============================================================================

def example_7_config_validation_advanced():
    """
    Advanced validation with custom rules.

    LOW-LEVEL API: Custom validation.

    Output:
        Multi-field validation with custom rules
    """
    print("\n" + "=" * 70)
    print("Example 7: Configuration Validation (Advanced Rules)")
    print("=" * 70)

    validator = get_config_validator()
    validator.clear_rules()

    # Add validators
    print("\nSetting up validators...")
    validator.add_lod_validator()
    validator.add_gpu_validator()
    validator.add_numeric_range_validator("batch_size", 1, 10000)
    validator.add_required_field("lod_level")
    print("  ✓ Validators configured")

    # Validate config
    print("\nValidating processor config...")
    config = {
        "lod_level": "LOD3",
        "gpu_memory_fraction": 0.8,
        "batch_size": 256,
    }

    is_valid, errors = validator.validate(config)

    if is_valid:
        print("  ✓ All validations passed")
    else:
        print("  ✗ Validation errors:")
        for error in errors:
            print(f"    {error}")

    return validator


# ============================================================================
# Example 8: Integration (All Three Managers)
# ============================================================================

def example_8_integration():
    """
    Integrated workflow using all three managers.

    Integration pattern showing how managers work together.

    Output:
        Complete workflow with validation, performance tracking, and GPU
    """
    print("\n" + "=" * 70)
    print("Example 8: Integration - Complete Workflow")
    print("=" * 70)

    # Step 1: Validate configuration
    print("\n1. Validating configuration...")
    config_validator = get_config_validator()
    config_validator.clear_rules()
    config_validator.add_lod_validator()
    config_validator.add_numeric_range_validator("num_points", 1000, 10000000)

    config = {"lod_level": "LOD2", "num_points": 50000}
    is_valid, errors = config_validator.validate(config)

    if not is_valid:
        print("  ✗ Configuration invalid")
        return

    print("  ✓ Configuration valid")

    # Step 2: Initialize GPU stream manager
    print("\n2. Initializing GPU stream manager...")
    stream_manager = get_stream_manager(pool_size=4)
    print(f"  ✓ {stream_manager.get_stream_count()} streams ready")

    # Step 3: Start performance monitoring
    print("\n3. Starting performance monitoring...")
    perf_manager = get_performance_manager()
    perf_manager.reset()
    perf_manager.start_phase("complete_pipeline")

    # Step 4: Simulate data processing
    print("\n4. Processing data...")
    perf_manager.start_phase("data_transfer")

    # Simulate GPU transfers
    for i in range(3):
        src = np.random.rand(10000, 3)
        dst = np.zeros((10000, 3))
        stream_manager.async_transfer(src, dst)

    stream_manager.wait_all()
    perf_manager.end_phase()
    print("  ✓ Data transfer complete")

    # Step 5: Simulate feature computation
    print("\n5. Computing features...")
    perf_manager.start_phase("feature_computation")

    for i in range(5):
        perf_manager.record_metric("accuracy", 0.85 + i * 0.02)

    import time

    time.sleep(0.05)
    perf_manager.end_phase()
    print("  ✓ Feature computation complete")

    # Step 6: End and report
    perf_manager.end_phase("complete_pipeline")

    print("\n6. Performance Summary:")
    summary = perf_manager.get_summary()
    print(f"  Total pipeline time: {summary['total_time']:.3f}s")
    print(f"  Phases executed: {summary['num_phases']}")

    print("\n  Pipeline Summary:")
    for phase_name, metrics in summary["phases"].items():
        print(f"    {phase_name}: {metrics['duration']:.3f}s")

    return config_validator, stream_manager, perf_manager


# ============================================================================
# Main: Run All Examples
# ============================================================================

def run_all_examples():
    """Run all examples in sequence."""
    print("\n" + "=" * 70)
    print("Phase 5 Unified Managers - Complete Usage Examples")
    print("=" * 70)

    examples = [
        ("GPU Stream High-Level", example_1_gpu_stream_high_level),
        ("GPU Stream Batch", example_2_gpu_stream_batch),
        ("GPU Stream Low-Level", example_3_gpu_stream_low_level),
        ("Performance High-Level", example_4_performance_high_level),
        ("Performance Custom Metrics", example_5_performance_custom_metrics),
        ("Config Validation High-Level", example_6_config_validation_high_level),
        ("Config Validation Advanced", example_7_config_validation_advanced),
        ("Integration", example_8_integration),
    ]

    results = {}

    for example_name, example_func in examples:
        try:
            print(f"\n▶ Running: {example_name}")
            result = example_func()
            results[example_name] = result
            print(f"✓ Success: {example_name}")

        except Exception as e:
            print(f"✗ Error in {example_name}: {e}")
            results[example_name] = None

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    successful = sum(1 for r in results.values() if r is not None)
    total = len(results)

    print(f"\nSuccessful examples: {successful}/{total}")
    for name, result in results.items():
        status = "✓" if result is not None else "✗"
        print(f"  {status} {name}")

    print("\nKey takeaways:")
    print("  1. Use GPUStreamManager for automatic async transfers")
    print("  2. Use batch_transfers() for multiple data movements")
    print("  3. Use PerformanceManager for automatic phase timing")
    print("  4. Record custom metrics within phases")
    print("  5. Use ConfigValidator for consistent validation")
    print("  6. Integrate all three for complete pipelines")

    return results


if __name__ == "__main__":
    results = run_all_examples()
