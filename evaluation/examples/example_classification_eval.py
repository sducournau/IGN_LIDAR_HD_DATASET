#!/usr/bin/env python3
"""
Example: Classification Model Evaluation

Demonstrates how to evaluate a trained classification model on test data.

Usage:
    python examples/example_classification_eval.py
"""

import logging
import numpy as np
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.classification_evaluator import ClassificationEvaluator, evaluate_classification

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_synthetic_test_data(n_points=10000):
    """Create synthetic test data for demonstration."""
    logger.info(f"Creating synthetic test data with {n_points} points...")
    
    # Generate random classifications (ASPRS codes)
    # Classes: 2=Ground, 3-5=Vegetation, 6=Building, 11=Road
    classes = [2, 3, 4, 5, 6, 11]
    
    # Ground truth with realistic distribution
    y_true = np.random.choice(
        classes,
        size=n_points,
        p=[0.15, 0.20, 0.15, 0.10, 0.30, 0.10],  # Distribution
    )
    
    # Predictions with ~85% accuracy
    y_pred = y_true.copy()
    noise_mask = np.random.random(n_points) < 0.15  # 15% error rate
    y_pred[noise_mask] = np.random.choice(classes, size=noise_mask.sum())
    
    # Generate point coordinates
    points = np.random.rand(n_points, 3) * 100  # 100m x 100m area
    
    # Generate confidence scores (lower for incorrect predictions)
    confidence = np.ones(n_points) * 0.9
    confidence[y_true != y_pred] = np.random.uniform(0.3, 0.7, size=(y_true != y_pred).sum())
    
    return y_true, y_pred, points, confidence


def example_basic_evaluation():
    """Example 1: Basic classification evaluation."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Classification Evaluation")
    print("=" * 70)
    
    # Create test data
    y_true, y_pred, points, confidence = create_synthetic_test_data(10000)
    
    # Run evaluation
    metrics = evaluate_classification(
        y_true=y_true,
        y_pred=y_pred,
        points=points,
        confidence=confidence,
        output_dir=Path("evaluation_results/example_basic"),
        save_report=True,
    )
    
    # Print summary
    print(f"\n✓ Evaluation complete!")
    print(f"  Overall Accuracy: {metrics.overall_accuracy:.2%}")
    print(f"  Balanced Accuracy: {metrics.balanced_accuracy:.2%}")
    print(f"  F1 Macro: {metrics.f1_macro:.2%}")
    print(f"  F1 Weighted: {metrics.f1_weighted:.2%}")
    print(f"  Kappa: {metrics.kappa_coefficient:.3f}")
    print(f"  Spatial Coherence: {metrics.spatial_coherence_score:.2%}")
    
    print(f"\n✓ Reports saved to: evaluation_results/example_basic/")


def example_multiple_tests():
    """Example 2: Evaluate multiple test scenarios."""
    print("\n" + "=" * 70)
    print("Example 2: Multiple Test Scenarios")
    print("=" * 70)
    
    evaluator = ClassificationEvaluator(
        output_dir=Path("evaluation_results/example_multi"),
        compute_spatial_metrics=True,
    )
    
    # Simulate different test scenarios
    scenarios = {
        "urban_high_density": {"n_points": 50000, "building_ratio": 0.4},
        "urban_low_density": {"n_points": 30000, "building_ratio": 0.2},
        "forest": {"n_points": 40000, "building_ratio": 0.05},
    }
    
    results = []
    
    for scenario_name, params in scenarios.items():
        print(f"\n🔍 Evaluating scenario: {scenario_name}")
        
        # Generate scenario-specific data
        y_true, y_pred, points, confidence = create_synthetic_test_data(params["n_points"])
        
        # Adjust distribution based on scenario
        # (In real use, this would be actual test data)
        
        # Evaluate
        metrics = evaluator.evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            points=points,
            confidence=confidence,
            test_name=scenario_name,
        )
        
        results.append((scenario_name, metrics))
        
        print(f"  ✓ Accuracy: {metrics.overall_accuracy:.2%}")
        print(f"  ✓ F1 Score: {metrics.f1_macro:.2%}")
    
    # Save combined report
    print(f"\n✓ Saving reports...")
    evaluator.save_report(format="json")
    evaluator.save_report(format="html")
    evaluator.save_report(format="csv")
    
    print(f"\n✓ All reports saved to: evaluation_results/example_multi/")
    
    # Compare results
    print(f"\n" + "=" * 70)
    print("Scenario Comparison")
    print("=" * 70)
    print(f"{'Scenario':<25} {'Accuracy':>10} {'F1 Score':>10} {'Kappa':>10}")
    print("-" * 70)
    
    for scenario_name, metrics in results:
        print(f"{scenario_name:<25} {metrics.overall_accuracy:>9.1%} "
              f"{metrics.f1_macro:>9.1%} {metrics.kappa_coefficient:>9.3f}")


def example_advanced_analysis():
    """Example 3: Advanced analysis with per-class breakdown."""
    print("\n" + "=" * 70)
    print("Example 3: Advanced Per-Class Analysis")
    print("=" * 70)
    
    # Create test data
    y_true, y_pred, points, confidence = create_synthetic_test_data(20000)
    
    # Evaluate
    evaluator = ClassificationEvaluator(
        output_dir=Path("evaluation_results/example_advanced"),
    )
    
    metrics = evaluator.evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        points=points,
        confidence=confidence,
        test_name="advanced_analysis",
    )
    
    # Print detailed per-class metrics
    print(f"\n📊 Per-Class Performance:")
    print("-" * 90)
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'IoU':>10} {'Support':>10}")
    print("-" * 90)
    
    for class_name in sorted(metrics.per_class_precision.keys()):
        precision = metrics.per_class_precision[class_name]
        recall = metrics.per_class_recall[class_name]
        f1 = metrics.per_class_f1[class_name]
        iou = metrics.per_class_iou[class_name]
        support = metrics.per_class_support[class_name]
        
        print(f"{class_name:<20} {precision:>9.1%} {recall:>9.1%} "
              f"{f1:>9.1%} {iou:>9.1%} {support:>10,}")
    
    # Most confused pairs
    print(f"\n🔀 Most Confused Class Pairs:")
    print("-" * 70)
    
    for class1, class2, count in metrics.most_confused_pairs[:5]:
        print(f"  {class1} ↔ {class2}: {count} errors")
    
    # Error distribution
    print(f"\n❌ Error Distribution by True Class:")
    print("-" * 70)
    
    for class_name, error_count in sorted(
        metrics.error_distribution.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]:
        error_rate = error_count / metrics.per_class_support.get(class_name, 1) * 100
        print(f"  {class_name}: {error_count} errors ({error_rate:.1f}%)")
    
    print(f"\n✓ Detailed reports saved to: evaluation_results/example_advanced/")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("IGN LiDAR HD - Classification Evaluation Examples")
    print("=" * 70)
    
    # Run examples
    example_basic_evaluation()
    example_multiple_tests()
    example_advanced_analysis()
    
    print("\n" + "=" * 70)
    print("✓ All examples completed successfully!")
    print("=" * 70)
    print("\nCheck evaluation_results/ directory for outputs:")
    print("  - JSON reports (for programmatic analysis)")
    print("  - HTML reports (for visualization)")
    print("  - CSV reports (for spreadsheet analysis)")
    print()


if __name__ == "__main__":
    main()
