"""
Demonstration of the preprocessing module for artifact mitigation.

This example shows how to use the new preprocessing functions to:
1. Remove outliers (SOR and ROR)
2. Homogenize point density (voxel downsampling)
3. Improve geometric feature quality
"""

import numpy as np
import laspy
from pathlib import Path
from ign_lidar.preprocessing import (
    preprocess_point_cloud,
    preprocess_for_features,
    preprocess_for_urban,
    preprocess_for_natural,
    statistical_outlier_removal,
    radius_outlier_removal,
    voxel_downsample
)
from ign_lidar.features import compute_all_features_optimized


def example_1_basic_preprocessing():
    """Example 1: Basic preprocessing with default settings"""
    print("=" * 60)
    print("Example 1: Basic Preprocessing")
    print("=" * 60)
    
    # Simulate a small point cloud with noise
    np.random.seed(42)
    
    # Main building surface
    building = np.random.randn(500, 3) * 0.2
    building[:, 2] += 10.0  # Elevate to 10m
    
    # Add outliers (measurement errors)
    outliers = np.array([
        [0, 0, 20],    # Bird
        [5, 5, -2],    # Ground noise
        [10, 0, 10]    # Isolated point
    ])
    
    points = np.vstack([building, outliers])
    
    print(f"Original points: {len(points)}")
    
    # Apply default preprocessing
    processed, stats = preprocess_point_cloud(points)
    
    print(f"Processed points: {len(processed)}")
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return processed


def example_2_custom_config():
    """Example 2: Custom preprocessing configuration"""
    print("\n" + "=" * 60)
    print("Example 2: Custom Configuration")
    print("=" * 60)
    
    # Generate test data
    np.random.seed(42)
    points = np.random.randn(1000, 3) * 2.0
    
    # Custom configuration for urban scenes
    config = {
        'sor': {
            'enable': True,
            'k': 15,  # More neighbors for denser clouds
            'std_multiplier': 2.5  # More lenient
        },
        'ror': {
            'enable': True,
            'radius': 1.5,  # Larger radius for urban
            'min_neighbors': 5
        },
        'voxel': {
            'enable': True,
            'voxel_size': 0.3,  # 30cm voxels
            'method': 'centroid'
        }
    }
    
    processed, stats = preprocess_point_cloud(points, config)
    
    print(f"Original: {len(points)} points")
    print(f"After SOR: {stats['sor_removed']} removed")
    print(f"After ROR: {stats['ror_removed']} removed")
    print(f"After voxel: {stats['voxel_reduced']} reduced")
    print(f"Final: {len(processed)} points ({stats['reduction_ratio']:.1%} reduction)")
    
    return processed


def example_3_convenience_functions():
    """Example 3: Using convenience functions"""
    print("\n" + "=" * 60)
    print("Example 3: Convenience Functions")
    print("=" * 60)
    
    np.random.seed(42)
    points = np.random.randn(800, 3) * 1.5
    
    print(f"Original points: {len(points)}")
    
    # Standard preprocessing for feature computation
    standard = preprocess_for_features(points, mode='standard')
    print(f"Standard mode: {len(standard)} points")
    
    # Light preprocessing (minimal filtering)
    light = preprocess_for_features(points, mode='light')
    print(f"Light mode: {len(light)} points")
    
    # Aggressive preprocessing (strong filtering + downsampling)
    aggressive = preprocess_for_features(points, mode='aggressive')
    print(f"Aggressive mode: {len(aggressive)} points")
    
    # Urban-specific preprocessing
    urban = preprocess_for_urban(points)
    print(f"Urban preset: {len(urban)} points")
    
    # Natural environment preprocessing
    natural = preprocess_for_natural(points)
    print(f"Natural preset: {len(natural)} points")


def example_4_individual_filters():
    """Example 4: Using individual filter functions"""
    print("\n" + "=" * 60)
    print("Example 4: Individual Filters")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create noisy cloud
    clean = np.random.randn(300, 3) * 0.5
    outliers = np.array([[10, 10, 10], [-8, -8, -8]])
    points = np.vstack([clean, outliers])
    
    print(f"Original: {len(points)} points")
    
    # Step 1: Statistical Outlier Removal
    after_sor, sor_mask = statistical_outlier_removal(
        points, k=12, std_multiplier=2.0
    )
    print(f"After SOR: {len(after_sor)} points ({np.sum(~sor_mask)} removed)")
    
    # Step 2: Radius Outlier Removal
    after_ror, ror_mask = radius_outlier_removal(
        after_sor, radius=1.0, min_neighbors=4
    )
    print(f"After ROR: {len(after_ror)} points ({np.sum(~ror_mask)} removed)")
    
    # Step 3: Voxel Downsampling
    after_voxel, voxel_idx = voxel_downsample(
        after_ror, voxel_size=0.5, method='centroid'
    )
    print(f"After voxel: {len(after_voxel)} points")
    print(f"Total reduction: {1 - len(after_voxel)/len(points):.1%}")


def example_5_with_features():
    """Example 5: Preprocessing before feature computation"""
    print("\n" + "=" * 60)
    print("Example 5: Preprocessing + Feature Extraction")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Simulate building facade
    facade = np.random.randn(1000, 3) * 0.1
    facade[:, 0] = 0  # Make it planar (X=0)
    facade[:, 2] += 5.0  # Elevate
    
    # Add noise
    noise = np.random.randn(50, 3) * 5.0
    points = np.vstack([facade, noise])
    
    print(f"Original points: {len(points)}")
    
    # Preprocess
    processed = preprocess_for_features(points, mode='standard')
    print(f"After preprocessing: {len(processed)}")
    
    # Compute features (requires XYZ columns)
    # Note: In real usage, you'd have a full point cloud with attributes
    # Here we demonstrate the workflow
    
    print("\nFeature computation would happen next...")
    print("Example: features = compute_all_features_optimized(processed, radius=1.0)")


def example_6_laz_file_preprocessing():
    """Example 6: Preprocessing a LAZ file (if available)"""
    print("\n" + "=" * 60)
    print("Example 6: LAZ File Preprocessing")
    print("=" * 60)
    
    # This is a template - adjust path to your actual LAZ file
    laz_path = Path("path/to/your/tile.laz")
    
    if not laz_path.exists():
        print(f"LAZ file not found: {laz_path}")
        print("This example requires a real LAZ file.")
        print("\nTemplate workflow:")
        print("  1. Read LAZ: las = laspy.read(laz_path)")
        print("  2. Extract XYZ: points = np.vstack([las.x, las.y, las.z]).T")
        print("  3. Preprocess: processed = preprocess_for_features(points)")
        print("  4. Compute features: features = compute_all_features_optimized(...)")
        return
    
    # Read LAZ file
    las = laspy.read(laz_path)
    points = np.vstack([las.x, las.y, las.z]).T
    
    print(f"Loaded {len(points):,} points from {laz_path.name}")
    
    # Preprocess
    processed, stats = preprocess_point_cloud(points)
    
    print(f"Processed: {len(processed):,} points")
    print(f"Reduction: {stats['reduction_ratio']:.1%}")
    print(f"Time: {stats['processing_time_ms']:.0f}ms")
    
    # Save preprocessed points
    output_path = laz_path.parent / f"{laz_path.stem}_preprocessed.laz"
    
    # Create new LAS file with preprocessed points
    # (In real usage, you'd also filter other attributes accordingly)
    print(f"\nWould save to: {output_path}")


def example_7_parameter_comparison():
    """Example 7: Compare different parameter settings"""
    print("\n" + "=" * 60)
    print("Example 7: Parameter Comparison")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create test cloud
    points = np.random.randn(1000, 3) * 2.0
    
    # Add some clear outliers
    outliers = np.random.randn(20, 3) * 10.0
    points = np.vstack([points, outliers])
    
    print(f"Test cloud: {len(points)} points (including 20 outliers)")
    
    # Test different SOR settings
    print("\nStatistical Outlier Removal settings:")
    
    for k, std_mult in [(10, 2.0), (15, 2.5), (20, 3.0)]:
        filtered, mask = statistical_outlier_removal(
            points, k=k, std_multiplier=std_mult
        )
        removed = np.sum(~mask)
        print(f"  k={k}, std={std_mult}: {removed} removed, {len(filtered)} kept")
    
    # Test different voxel sizes
    print("\nVoxel Downsampling sizes:")
    
    for voxel_size in [0.5, 1.0, 2.0]:
        downsampled, _ = voxel_downsample(points, voxel_size=voxel_size)
        reduction = 1 - len(downsampled) / len(points)
        print(f"  {voxel_size}m voxels: {len(downsampled)} points ({reduction:.1%} reduction)")


if __name__ == '__main__':
    print("IGN LiDAR HD - Preprocessing Module Demonstration")
    print("=" * 60)
    
    # Run all examples
    example_1_basic_preprocessing()
    example_2_custom_config()
    example_3_convenience_functions()
    example_4_individual_filters()
    example_5_with_features()
    example_6_laz_file_preprocessing()
    example_7_parameter_comparison()
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Integrate preprocessing into your pipeline")
    print("2. Experiment with different parameter settings")
    print("3. Monitor feature quality improvements")
    print("4. See ARTIFACT_MITIGATION_PLAN.md for more details")
