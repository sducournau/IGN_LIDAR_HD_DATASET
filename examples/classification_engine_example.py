"""
Example: Using the Unified ClassificationEngine

This example demonstrates the new unified classification interface
introduced in v3.6.0, which consolidates multiple classification engines.

Features:
- Simple interface for basic classification
- Advanced methods for specialized classification with additional data
- Support for Spectral, Geometric, and ASPRS classification modes
"""

import numpy as np
from ign_lidar.core.classification import ClassificationEngine, ClassificationMode


def example_basic_classification():
    """Example: Basic classification with features only."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Classification")
    print("=" * 70)

    # Initialize engine with ASPRS mode (default)
    engine = ClassificationEngine(mode="asprs", use_gpu=False)

    # Create dummy feature array [N_points, N_features]
    n_points = 1000
    n_features = 15  # e.g., normals, curvature, planarity, etc.
    features = np.random.rand(n_points, n_features).astype(np.float32)

    # Classify
    labels = engine.classify(features)

    print(f"✓ Classified {len(labels)} points")
    print(f"  Output shape: {labels.shape}")
    print(f"  Label range: [{labels.min()}, {labels.max()}]")

    # Get confidence scores
    confidence = engine.get_confidence(labels)
    print(f"✓ Confidence scores computed")
    print(f"  Mean confidence: {confidence.mean():.3f}")
    print(f"  Min/Max: [{confidence.min():.3f}, {confidence.max():.3f}]")


def example_switch_modes():
    """Example: Switching between classification modes."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Switching Classification Modes")
    print("=" * 70)

    engine = ClassificationEngine(mode="asprs")
    print(f"Initial mode: {engine.mode}")

    # Get available modes
    available = engine.get_available_modes()
    print(f"Available modes: {available}")

    # Switch modes
    for mode in ["spectral", "geometric", "asprs"]:
        engine.set_mode(mode)
        print(f"✓ Switched to: {engine.strategy.get_name()}")


def example_spectral_classification():
    """Example: Spectral classification with RGB + NIR data."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Spectral Classification")
    print("=" * 70)

    engine = ClassificationEngine(mode="spectral")

    # Dummy data
    n_points = 500
    rgb = np.random.rand(n_points, 3).astype(np.float32)  # RGB [0, 1]
    nir = np.random.rand(n_points).astype(np.float32)      # NIR [0, 1]
    current_labels = np.zeros(n_points, dtype=np.int32)    # Initial labels

    try:
        # Use advanced spectral classification method
        updated_labels, stats = engine.classify_spectral(
            rgb=rgb,
            nir=nir,
            current_labels=current_labels,
            apply_to_unclassified_only=True,
        )

        print(f"✓ Spectral classification completed")
        print(f"  Updated {stats.get('total_reclassified', 0)} points")
        print(f"  Statistics: {stats}")

    except RuntimeError as e:
        print(f"⚠ Spectral classification not available: {e}")
        print("  This is normal if SpectralRulesEngine is not fully configured")


def example_geometric_classification():
    """Example: Geometric classification with points and ground truth."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Geometric Classification")
    print("=" * 70)

    engine = ClassificationEngine(mode="geometric")

    # Dummy data
    n_points = 500
    points = np.random.rand(n_points, 3).astype(np.float32) * 100  # XYZ coordinates
    labels = np.zeros(n_points, dtype=np.int32)
    ground_truth_features = {}  # Would normally have GeoDataFrames

    try:
        # Use advanced geometric classification method
        updated_labels, stats = engine.classify_geometric(
            points=points,
            labels=labels,
            ground_truth_features=ground_truth_features,
            preserve_ground_truth=True,
        )

        print(f"✓ Geometric classification completed")
        print(f"  Rules applied: {len(stats)}")

    except RuntimeError as e:
        print(f"⚠ Geometric classification not available: {e}")
        print("  This is normal if GeometricRulesEngine is not fully configured")


def example_asprs_classification():
    """Example: ASPRS classification with points and features."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: ASPRS Classification")
    print("=" * 70)

    engine = ClassificationEngine(mode="asprs")

    # Dummy data
    n_points = 500
    points = np.random.rand(n_points, 3).astype(np.float32) * 100  # XYZ
    features_dict = {
        "height": np.random.rand(n_points),
        "intensity": np.random.rand(n_points),
    }
    classification = np.zeros(n_points, dtype=np.int32)

    try:
        # Use advanced ASPRS classification method
        updated_labels = engine.classify_asprs(
            points=points,
            features=features_dict,
            classification=classification,
            ground_truth=None,
        )

        print(f"✓ ASPRS classification completed")
        print(f"  Updated {len(updated_labels)} point labels")

    except RuntimeError as e:
        print(f"⚠ ASPRS classification not available: {e}")
        print("  This is normal if ASPRSClassRulesEngine is not fully configured")


def example_engine_info():
    """Example: Getting engine information."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Engine Information")
    print("=" * 70)

    engine = ClassificationEngine(mode="asprs", use_gpu=True)

    print(f"Engine representation: {repr(engine)}")
    print(f"Current mode: {engine.mode}")
    print(f"GPU enabled: {engine.use_gpu}")
    print(f"Strategy: {engine.strategy.get_name()}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("UNIFIED CLASSIFICATION ENGINE - EXAMPLES (v3.6.0)")
    print("=" * 70)

    example_basic_classification()
    example_switch_modes()
    example_spectral_classification()
    example_geometric_classification()
    example_asprs_classification()
    example_engine_info()

    print("\n" + "=" * 70)
    print("✓ All examples completed!")
    print("=" * 70 + "\n")
