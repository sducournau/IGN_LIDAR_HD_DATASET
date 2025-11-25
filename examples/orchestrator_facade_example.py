"""
Feature Orchestration Service - Usage Examples

This module demonstrates common workflows and best practices for using the
FeatureOrchestrationService facade.

Topics covered:
1. Basic feature computation (LOD2)
2. Advanced mode selection (LOD3 with GPU)
3. Spectral features with RGB data
4. Batch processing workflow
5. Performance monitoring
6. Error handling and fallbacks
"""

import numpy as np
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import logging

# Import the facade
from ign_lidar.features.orchestrator_facade import FeatureOrchestrationService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ============================================================================
# Example 1: Basic Feature Computation (LOD2, Recommended Default)
# ============================================================================

def example_1_basic_computation():
    """
    Compute features using default settings (LOD2, auto GPU detection).

    LOD2 provides 12 essential features for building classification.
    This is the recommended starting point for most use cases.

    Output:
        Dictionary of 12 geometric features
    """
    print("\n" + "=" * 70)
    print("Example 1: Basic Feature Computation (LOD2)")
    print("=" * 70)

    # Load configuration (or create minimal one)
    config = OmegaConf.create({
        "processor": {"lod_level": "LOD2"},
        "features": {"mode": "lod2"}
    })

    # Initialize service
    service = FeatureOrchestrationService(config, verbose=True)

    # Simulate point cloud (real usage would load from LAZ file)
    points = np.random.rand(10000, 3) * 100  # [10k points, XYZ]
    classification = np.random.randint(0, 6, 10000)  # [10k labels]

    # Compute features with defaults (HIGH-LEVEL API)
    features = service.compute_features(points, classification)

    print(f"\nComputed {len(features)} feature types:")
    for feat_name, feat_data in features.items():
        print(f"  - {feat_name:20s}: {feat_data.shape}")

    # Check optimization strategy
    info = service.get_optimization_info()
    print(f"\nOptimization strategy: {info['strategy']}")
    print(f"GPU available: {info['gpu_available']}")

    return features


# ============================================================================
# Example 2: Advanced Mode Selection (LOD3 with GPU)
# ============================================================================

def example_2_advanced_lod3_gpu():
    """
    Compute comprehensive LOD3 features with GPU acceleration.

    LOD3 provides 38 detailed features including advanced geometric descriptors
    and architectural details. GPU acceleration is recommended for >100k points.

    Output:
        Dictionary of 38 LOD3-specific features
    """
    print("\n" + "=" * 70)
    print("Example 2: Advanced LOD3 with GPU Acceleration")
    print("=" * 70)

    config = OmegaConf.create({
        "processor": {"use_gpu": True},
        "features": {"mode": "lod3", "k_neighbors": 50}
    })

    service = FeatureOrchestrationService(config, verbose=True)

    # Larger point cloud for GPU benefit
    num_points = 50000
    points = np.random.rand(num_points, 3) * 200
    classification = np.random.randint(0, 6, num_points)

    print(f"Processing {num_points:,} points with LOD3 features...")

    # Use compute_with_mode for explicit control
    features = service.compute_with_mode(
        points=points,
        classification=classification,
        mode='LOD3',
        use_gpu=True,  # Force GPU
        k_neighbors=50,  # More neighbors for detailed features
        search_radius=5.0  # Larger search radius for context
    )

    print(f"\nComputed {len(features)} LOD3 features:")
    for feat_name, feat_data in list(features.items())[:5]:  # Show first 5
        print(f"  - {feat_name:20s}: {feat_data.shape}")
    if len(features) > 5:
        print(f"  ... and {len(features) - 5} more")

    # Get performance metrics
    metrics = service.get_performance_summary()
    if metrics:
        print(f"\nPerformance metrics: {metrics}")

    return features


# ============================================================================
# Example 3: Spectral Features with RGB Data
# ============================================================================

def example_3_spectral_features_with_rgb():
    """
    Compute spectral features using RGB orthophoto data.

    Spectral features enhance classification accuracy by incorporating
    color information from orthophotos and aerial imagery.

    Output:
        Features including spectral indices (NDVI-like)
    """
    print("\n" + "=" * 70)
    print("Example 3: Spectral Features with RGB Data")
    print("=" * 70)

    config = OmegaConf.create({
        "processor": {"use_gpu": True},
        "features": {"mode": "lod2"}
    })

    service = FeatureOrchestrationService(config, verbose=True)

    # Point cloud
    points = np.random.rand(20000, 3) * 150
    classification = np.random.randint(0, 6, 20000)

    # Simulate RGB orthophoto [height, width, 3]
    # Real usage would load from GeoTIFF
    rgb_ortho = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)

    print(f"Point cloud: {points.shape}")
    print(f"RGB data: {rgb_ortho.shape}")

    # Compute with RGB enhancement
    features = service.compute_with_mode(
        points=points,
        classification=classification,
        mode='LOD2',
        use_rgb=True,  # Enable spectral features
        use_gpu=True
    )

    print(f"\nComputed {len(features)} features with spectral enhancement")

    # Check if spectral features were included
    spectral_features = [f for f in features.keys() if 'spectral' in f.lower()]
    if spectral_features:
        print(f"Spectral features computed: {spectral_features}")

    return features


# ============================================================================
# Example 4: Batch Processing Workflow
# ============================================================================

def example_4_batch_processing():
    """
    Process multiple tiles in batch with cache management.

    This workflow demonstrates:
    - Processing multiple tiles
    - Memory management with cache clearing
    - Consistent feature computation across tiles

    Output:
        List of feature dictionaries for each tile
    """
    print("\n" + "=" * 70)
    print("Example 4: Batch Processing with Cache Management")
    print("=" * 70)

    config = OmegaConf.create({
        "processor": {"use_gpu": True},
        "features": {"mode": "lod2"}
    })

    service = FeatureOrchestrationService(config, verbose=True)

    # Simulate processing 3 tiles
    num_tiles = 3
    points_per_tile = 5000

    all_results = []

    for tile_idx in range(num_tiles):
        print(f"\nProcessing tile {tile_idx + 1}/{num_tiles}...")

        # Simulate loading tile data
        points = np.random.rand(points_per_tile, 3) * 100
        points[:, 2] += tile_idx * 50  # Vary elevation

        classification = np.random.randint(0, 6, points_per_tile)

        # Compute features
        features = service.compute_features(points, classification)
        all_results.append(features)

        print(f"  ✓ Computed {len(features)} features")
        print(f"  ✓ Feature shape: {list(features.values())[0].shape}")

        # Clear cache between tiles to manage memory
        if tile_idx < num_tiles - 1:
            service.clear_cache()
            print(f"  ✓ Cache cleared")

    print(f"\nBatch processing complete: {len(all_results)} tiles processed")

    return all_results


# ============================================================================
# Example 5: Performance Monitoring
# ============================================================================

def example_5_performance_monitoring():
    """
    Monitor computation performance and optimization choices.

    This workflow demonstrates:
    - Checking available computation modes
    - Getting optimization information
    - Monitoring performance metrics

    Output:
        Performance metrics and optimization summary
    """
    print("\n" + "=" * 70)
    print("Example 5: Performance Monitoring")
    print("=" * 70)

    config = OmegaConf.create({
        "processor": {"use_gpu": True},
        "features": {"mode": "lod2"}
    })

    service = FeatureOrchestrationService(config, verbose=False)

    # Display available modes
    modes = service.get_feature_modes()
    print("\nAvailable feature modes:")
    for mode, description in modes.items():
        print(f"  - {mode:10s}: {description}")

    # Get optimization info before computation
    print("\nOptimization configuration:")
    info = service.get_optimization_info()
    for key, value in info.items():
        print(f"  - {key:20s}: {value}")

    # Compute features
    points = np.random.rand(10000, 3)
    classification = np.random.randint(0, 6, 10000)

    features = service.compute_features(points, classification)

    # Get performance metrics
    print("\nPerformance metrics:")
    metrics = service.get_performance_summary()
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  - {key:20s}: {value:.3f}")
            else:
                print(f"  - {key:20s}: {value}")
    else:
        print("  (Not available)")

    return metrics


# ============================================================================
# Example 6: Error Handling and Fallbacks
# ============================================================================

def example_6_error_handling():
    """
    Demonstrate error handling and graceful fallbacks.

    This workflow shows:
    - Handling computation errors
    - Fallback to CPU on GPU failure
    - Validation of inputs

    Output:
        Successful computation despite errors
    """
    print("\n" + "=" * 70)
    print("Example 6: Error Handling and Fallbacks")
    print("=" * 70)

    config = OmegaConf.create({
        "processor": {"use_gpu": True},
        "features": {"mode": "lod2"}
    })

    service = FeatureOrchestrationService(config, verbose=True)

    # Test 1: Invalid input handling
    print("\nTest 1: Handling invalid inputs...")
    try:
        # Missing classification array
        points = np.random.rand(100, 3)
        classification = None

        features = service.compute_features(points, classification)

    except TypeError as e:
        print(f"  ✓ Caught expected error: {type(e).__name__}")

    # Test 2: Try GPU, fallback to CPU
    print("\nTest 2: GPU fallback handling...")
    points = np.random.rand(100, 3)
    classification = np.random.randint(0, 6, 100)

    try:
        # Try with GPU first (may fail if unavailable)
        features = service.compute_with_mode(
            points, classification, mode='LOD2', use_gpu=True
        )
        print("  ✓ GPU computation successful")

    except Exception as e:
        print(f"  ! GPU failed: {type(e).__name__}")
        print("  → Falling back to CPU...")

        # Fallback to CPU
        features = service.compute_with_mode(
            points, classification, mode='LOD2', use_gpu=False
        )
        print("  ✓ CPU computation successful")

    print(f"  ✓ Got {len(features)} features")

    return features


# ============================================================================
# Example 7: Direct Orchestrator Access (Advanced)
# ============================================================================

def example_7_advanced_orchestrator_access():
    """
    Direct access to underlying orchestrator for advanced use cases.

    LOW-LEVEL API: For users who need full control over internal operations.

    Output:
        Results from directly accessing orchestrator
    """
    print("\n" + "=" * 70)
    print("Example 7: Advanced Orchestrator Access (Low-Level API)")
    print("=" * 70)

    config = OmegaConf.create({
        "processor": {"use_gpu": True},
        "features": {"mode": "lod2"}
    })

    service = FeatureOrchestrationService(config, verbose=True)

    # Get direct orchestrator access
    orch = service.get_orchestrator()
    print(f"Got orchestrator: {type(orch).__name__}")

    # Now users can access internal methods if needed
    # This is powerful but requires knowledge of orchestrator internals
    print("\nOrchestrator is available for advanced operations")
    print("  - Access internal methods: orch._compute_geometric_features(...)")
    print("  - Direct configuration: orch.config, orch.parameters")
    print("  - Internal state inspection: orch.cache, orch.performance_data")

    return orch


# ============================================================================
# Main: Run All Examples
# ============================================================================

def run_all_examples():
    """Run all examples in sequence."""
    print("\n" + "=" * 70)
    print("Feature Orchestration Service - Complete Usage Examples")
    print("=" * 70)

    examples = [
        ("Basic Computation", example_1_basic_computation),
        ("Advanced LOD3 + GPU", example_2_advanced_lod3_gpu),
        ("Spectral Features with RGB", example_3_spectral_features_with_rgb),
        ("Batch Processing", example_4_batch_processing),
        ("Performance Monitoring", example_5_performance_monitoring),
        ("Error Handling", example_6_error_handling),
        ("Orchestrator Access", example_7_advanced_orchestrator_access),
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
    print("  1. Use compute_features() for default LOD2 computation")
    print("  2. Use compute_with_mode() for advanced control")
    print("  3. Enable GPU for >100k points")
    print("  4. Include RGB/NIR for spectral features")
    print("  5. Clear cache between tiles for memory management")
    print("  6. Check optimization_info() for hardware capabilities")
    print("  7. Use error handling for production robustness")

    return results


if __name__ == "__main__":
    # Run all examples
    results = run_all_examples()

    # Individual example access:
    # - uncomment to run single example
    # example_1_basic_computation()
    # example_2_advanced_lod3_gpu()
    # example_3_spectral_features_with_rgb()
    # etc.
