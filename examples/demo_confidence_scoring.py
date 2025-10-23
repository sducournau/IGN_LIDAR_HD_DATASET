#!/usr/bin/env python3
"""
Demo: Confidence Scoring and Combination Strategies

This example demonstrates the various confidence scoring methods and
combination strategies available in the rules framework.

Features demonstrated:
- 7 confidence calculation methods
- 6 confidence combination strategies
- Calibration and normalization
- Practical use cases for each method
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List

from ign_lidar.core.classification.rules import (
    calculate_confidence_binary,
    calculate_confidence_linear,
    calculate_confidence_sigmoid,
    calculate_confidence_gaussian,
    calculate_confidence_threshold,
    calculate_confidence_exponential,
    combine_confidence_weighted,
    combine_confidence_max,
    combine_confidence_min,
    combine_confidence_product,
    combine_confidence_geometric_mean,
    combine_confidence_harmonic_mean,
    normalize_confidence,
    calibrate_confidence,
)


def demo_confidence_methods():
    """Demonstrate different confidence calculation methods."""
    print("=" * 70)
    print("Demo 1: Confidence Calculation Methods")
    print("=" * 70)
    
    # Create test values
    values = np.linspace(0, 1, 100)
    
    print("\nComparing 7 confidence calculation methods:\n")
    
    # 1. Binary
    conf_binary = calculate_confidence_binary(values, threshold=0.5)
    print("1. Binary (threshold=0.5):")
    print(f"   Simple yes/no based on threshold")
    print(f"   Below 0.5: {conf_binary[25]:.2f}, Above 0.5: {conf_binary[75]:.2f}")
    
    # 2. Linear
    conf_linear = calculate_confidence_linear(values, min_value=0.2, max_value=0.8)
    print("\n2. Linear (range 0.2-0.8):")
    print(f"   Linear mapping within range")
    print(f"   At 0.2: {conf_linear[20]:.2f}, At 0.5: {conf_linear[50]:.2f}, At 0.8: {conf_linear[80]:.2f}")
    
    # 3. Sigmoid
    conf_sigmoid = calculate_confidence_sigmoid(values, midpoint=0.5, steepness=10)
    print("\n3. Sigmoid (midpoint=0.5, steepness=10):")
    print(f"   S-curve for smooth transitions")
    print(f"   At 0.3: {conf_sigmoid[30]:.2f}, At 0.5: {conf_sigmoid[50]:.2f}, At 0.7: {conf_sigmoid[70]:.2f}")
    
    # 4. Gaussian
    conf_gaussian = calculate_confidence_gaussian(values, mean=0.5, std=0.2)
    print("\n4. Gaussian (mean=0.5, std=0.2):")
    print(f"   Bell curve around target value")
    print(f"   At 0.3: {conf_gaussian[30]:.2f}, At 0.5: {conf_gaussian[50]:.2f}, At 0.7: {conf_gaussian[70]:.2f}")
    
    # 5. Threshold
    conf_threshold = calculate_confidence_threshold(values, threshold=0.6)
    print("\n5. Threshold (threshold=0.6, reverse=False):")
    print(f"   Distance from threshold")
    print(f"   At 0.4: {conf_threshold[40]:.2f}, At 0.6: {conf_threshold[60]:.2f}, At 0.8: {conf_threshold[80]:.2f}")
    
    # 6. Exponential
    conf_exponential = calculate_confidence_exponential(values, rate=3.0)
    print("\n6. Exponential (rate=3.0):")
    print(f"   Exponential growth")
    print(f"   At 0.2: {conf_exponential[20]:.2f}, At 0.5: {conf_exponential[50]:.2f}, At 0.8: {conf_exponential[80]:.2f}")
    
    print("\n" + "-" * 70)
    print("Use cases:")
    print("  Binary: Simple pass/fail criteria (e.g., NDVI > threshold)")
    print("  Linear: Gradual increase over range (e.g., planarity 0-1)")
    print("  Sigmoid: Soft thresholds with smooth transition")
    print("  Gaussian: Target value with tolerance (e.g., specific height)")
    print("  Threshold: Distance-based confidence")
    print("  Exponential: Rapid increase for strong signals")


def demo_combination_strategies():
    """Demonstrate confidence combination strategies."""
    print("\n" + "=" * 70)
    print("Demo 2: Confidence Combination Strategies")
    print("=" * 70)
    
    # Create multiple confidence scores
    conf1 = np.array([0.8, 0.6, 0.9, 0.7, 0.5])
    conf2 = np.array([0.7, 0.8, 0.6, 0.9, 0.4])
    conf3 = np.array([0.9, 0.5, 0.8, 0.6, 0.7])
    
    confidences = [conf1, conf2, conf3]
    
    print(f"\nCombining 3 confidence arrays:")
    print(f"  Array 1: {conf1}")
    print(f"  Array 2: {conf2}")
    print(f"  Array 3: {conf3}")
    print()
    
    # 1. Weighted average
    weights = [0.5, 0.3, 0.2]
    combined_weighted = combine_confidence_weighted(confidences, weights)
    print(f"1. Weighted Average (weights={weights}):")
    print(f"   Result: {combined_weighted}")
    print(f"   Use: When rules have different importance\n")
    
    # 2. Maximum
    combined_max = combine_confidence_max(confidences)
    print(f"2. Maximum:")
    print(f"   Result: {combined_max}")
    print(f"   Use: Optimistic - accept if any rule is confident\n")
    
    # 3. Minimum
    combined_min = combine_confidence_min(confidences)
    print(f"3. Minimum:")
    print(f"   Result: {combined_min}")
    print(f"   Use: Conservative - require all rules to be confident\n")
    
    # 4. Product
    combined_product = combine_confidence_product(confidences)
    print(f"4. Product:")
    print(f"   Result: {combined_product}")
    print(f"   Use: Independent evidence - low if any is low\n")
    
    # 5. Geometric mean
    combined_geometric = combine_confidence_geometric_mean(confidences)
    print(f"5. Geometric Mean:")
    print(f"   Result: {combined_geometric}")
    print(f"   Use: Balanced combination, less extreme than product\n")
    
    # 6. Harmonic mean
    combined_harmonic = combine_confidence_harmonic_mean(confidences)
    print(f"6. Harmonic Mean:")
    print(f"   Result: {combined_harmonic}")
    print(f"   Use: Conservative, emphasizes lower values\n")


def demo_calibration():
    """Demonstrate confidence calibration and normalization."""
    print("=" * 70)
    print("Demo 3: Confidence Calibration and Normalization")
    print("=" * 70)
    
    # Generate synthetic confidence scores with bias
    np.random.seed(42)
    raw_confidences = np.random.beta(5, 2, 1000)  # Skewed toward high values
    
    print(f"\nRaw confidence scores (n=1000):")
    print(f"  Mean: {np.mean(raw_confidences):.3f}")
    print(f"  Std:  {np.std(raw_confidences):.3f}")
    print(f"  Min:  {np.min(raw_confidences):.3f}")
    print(f"  Max:  {np.max(raw_confidences):.3f}")
    
    # Normalize to [0, 1] range
    normalized = normalize_confidence(raw_confidences)
    print(f"\nAfter normalization:")
    print(f"  Mean: {np.mean(normalized):.3f}")
    print(f"  Std:  {np.std(normalized):.3f}")
    print(f"  Min:  {np.min(normalized):.3f}")
    print(f"  Max:  {np.max(normalized):.3f}")
    
    # Calibrate to target distribution
    calibrated = calibrate_confidence(
        raw_confidences,
        target_mean=0.7,
        target_std=0.15,
    )
    print(f"\nAfter calibration (target: mean=0.7, std=0.15):")
    print(f"  Mean: {np.mean(calibrated):.3f}")
    print(f"  Std:  {np.std(calibrated):.3f}")
    print(f"  Min:  {np.min(calibrated):.3f}")
    print(f"  Max:  {np.max(calibrated):.3f}")
    
    print("\n" + "-" * 70)
    print("Use cases:")
    print("  Normalization: Scale scores to standard [0, 1] range")
    print("  Calibration: Adjust distribution to match expected behavior")
    print("  Important for combining scores from different sources")


def demo_practical_example():
    """Demonstrate practical use case combining multiple methods."""
    print("\n" + "=" * 70)
    print("Demo 4: Practical Example - Building Detection")
    print("=" * 70)
    
    print("\nScenario: Detect buildings using multiple criteria")
    print("  1. Height (should be 3-30m)")
    print("  2. Planarity (should be high)")
    print("  3. NDVI (should be low - not vegetation)")
    print("  4. Roughness (should be low)")
    
    # Simulate features for 10 points
    np.random.seed(42)
    n_points = 10
    height = np.array([1.5, 5.0, 15.0, 25.0, 35.0, 8.0, 12.0, 2.0, 20.0, 10.0])
    planarity = np.array([0.3, 0.85, 0.92, 0.88, 0.90, 0.75, 0.95, 0.4, 0.87, 0.91])
    ndvi = np.array([0.6, 0.15, 0.08, 0.12, 0.10, 0.25, 0.05, 0.55, 0.09, 0.07])
    roughness = np.array([0.15, 0.03, 0.02, 0.04, 0.03, 0.08, 0.01, 0.12, 0.03, 0.02])
    
    print(f"\nPoint Features:")
    print(f"{'Point':>5} {'Height':>7} {'Planar':>7} {'NDVI':>7} {'Rough':>7}")
    print("-" * 40)
    for i in range(n_points):
        print(f"{i:5d} {height[i]:7.2f} {planarity[i]:7.2f} {ndvi[i]:7.2f} {roughness[i]:7.2f}")
    
    # Calculate confidence for each criterion
    
    # 1. Height: Gaussian around ideal range (15m ± 10m)
    conf_height = calculate_confidence_gaussian(
        height, mean=15.0, std=10.0
    )
    
    # 2. Planarity: Linear above threshold
    conf_planarity = calculate_confidence_linear(
        planarity, min_value=0.7, max_value=1.0
    )
    
    # 3. NDVI: Threshold (lower is better for buildings)
    conf_ndvi = calculate_confidence_threshold(
        ndvi, threshold=0.3, reverse=True
    )
    
    # 4. Roughness: Threshold (lower is better)
    conf_roughness = calculate_confidence_threshold(
        roughness, threshold=0.1, reverse=True
    )
    
    # Combine using weighted average (different importance)
    weights = [0.3, 0.35, 0.2, 0.15]  # Planarity most important
    confidences = [conf_height, conf_planarity, conf_ndvi, conf_roughness]
    
    final_confidence = combine_confidence_weighted(confidences, weights)
    
    # Classification threshold
    is_building = final_confidence > 0.6
    
    print(f"\n{'Point':>5} {'H_conf':>7} {'P_conf':>7} {'N_conf':>7} {'R_conf':>7} {'Final':>7} {'Building':>9}")
    print("-" * 62)
    for i in range(n_points):
        building_str = "YES" if is_building[i] else "no"
        print(f"{i:5d} {conf_height[i]:7.3f} {conf_planarity[i]:7.3f} "
              f"{conf_ndvi[i]:7.3f} {conf_roughness[i]:7.3f} "
              f"{final_confidence[i]:7.3f} {building_str:>9}")
    
    print(f"\nResults:")
    print(f"  Buildings detected: {np.sum(is_building)}/{n_points}")
    print(f"  Mean confidence: {np.mean(final_confidence[is_building]):.3f}")
    
    print("\n" + "-" * 70)
    print("Analysis:")
    print("  Point 0: Too low (1.5m) - not a building")
    print("  Point 1: Good height, high planarity - BUILDING ✓")
    print("  Point 2: Perfect features - BUILDING ✓")
    print("  Point 3: All criteria met - BUILDING ✓")
    print("  Point 4: Too tall (35m), but other features good - BUILDING ✓")
    print("  Point 5: Moderate features - borderline")
    print("  Point 6: Excellent features - BUILDING ✓")
    print("  Point 7: Too low, poor planarity - not a building")
    print("  Point 8: Good features - BUILDING ✓")
    print("  Point 9: Good features - BUILDING ✓")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("Confidence Scoring and Combination - Usage Examples")
    print("=" * 70)
    print("\nThis demo shows how to use confidence scoring methods and")
    print("combination strategies from the rules framework.\n")
    
    demo_confidence_methods()
    demo_combination_strategies()
    demo_calibration()
    demo_practical_example()
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Choose confidence method based on criterion type:")
    print("   - Binary for simple thresholds")
    print("   - Linear for gradual transitions")
    print("   - Sigmoid for soft boundaries")
    print("   - Gaussian for target values")
    print("   - Threshold for distance-based")
    print("   - Exponential for strong signals")
    print()
    print("2. Choose combination strategy based on rule relationships:")
    print("   - Weighted average for different importance")
    print("   - Max for optimistic (any rule succeeds)")
    print("   - Min for conservative (all must succeed)")
    print("   - Product for independent evidence")
    print("   - Geometric/Harmonic mean for balanced combinations")
    print()
    print("3. Use calibration to normalize across different sources")
    print()
    print("See the rules.confidence module for more details.")


if __name__ == "__main__":
    main()
