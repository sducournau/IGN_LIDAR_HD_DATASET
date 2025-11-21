"""
Classification Model Evaluation Framework

Comprehensive evaluation system for LiDAR point cloud classification models.
Evaluates:
- Classification accuracy (overall, per-class, confusion matrix)
- Feature quality and consistency
- Spatial coherence and boundary detection
- Model performance across different scenarios

Author: IGN LiDAR HD Dataset Team
Date: November 20, 2025
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from collections import defaultdict

# Import existing validation modules
from ign_lidar.core.classification.classification_validation import (
    ClassificationMetrics,
    ClassificationValidator,
)
from ign_lidar.classification_schema import ASPRS_CLASS_NAMES, get_class_name

logger = logging.getLogger(__name__)


# ============================================================================
# Evaluation Metrics
# ============================================================================

@dataclass
class ClassificationEvaluationMetrics:
    """Complete evaluation metrics for classification model."""
    
    # Overall metrics
    overall_accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    kappa_coefficient: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    
    # Per-class metrics
    per_class_precision: Dict[str, float] = None
    per_class_recall: Dict[str, float] = None
    per_class_f1: Dict[str, float] = None
    per_class_iou: Dict[str, float] = None
    per_class_support: Dict[str, int] = None
    
    # Confusion matrix
    confusion_matrix: np.ndarray = None
    class_names: List[str] = None
    
    # Spatial quality
    spatial_coherence_score: float = 0.0
    boundary_accuracy: float = 0.0
    
    # Error analysis
    most_confused_pairs: List[Tuple[str, str, int]] = None
    error_distribution: Dict[str, int] = None
    
    # Processing metrics
    inference_time_ms: float = 0.0
    points_per_second: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding arrays)."""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif value is None:
                result[key] = None
            else:
                result[key] = value
        return result


# ============================================================================
# Classification Evaluator
# ============================================================================

class ClassificationEvaluator:
    """
    Evaluate classification model performance on LiDAR point clouds.
    
    Features:
    - Comprehensive accuracy metrics (overall, per-class, confusion matrix)
    - Spatial quality assessment (coherence, boundary detection)
    - Error analysis and visualization
    - Support for multiple test scenarios
    - Export to JSON, CSV, and HTML reports
    """
    
    def __init__(
        self,
        class_names: Optional[Dict[int, str]] = None,
        output_dir: Optional[Path] = None,
        compute_spatial_metrics: bool = True,
    ):
        """
        Initialize evaluator.
        
        Args:
            class_names: Optional mapping from class ID to name
            output_dir: Directory to save evaluation reports
            compute_spatial_metrics: Whether to compute spatial coherence metrics
        """
        self.class_names = class_names or ASPRS_CLASS_NAMES
        self.output_dir = Path(output_dir) if output_dir else Path("evaluation_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.compute_spatial_metrics = compute_spatial_metrics
        
        # Use existing validator
        self.validator = ClassificationValidator(class_names=self.class_names)
        
        # Results storage
        self.evaluation_results = []
    
    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        points: Optional[np.ndarray] = None,
        confidence: Optional[np.ndarray] = None,
        test_name: str = "test",
        inference_time_ms: Optional[float] = None,
    ) -> ClassificationEvaluationMetrics:
        """
        Evaluate classification predictions against ground truth.
        
        Args:
            y_true: Ground truth labels [N]
            y_pred: Predicted labels [N]
            points: Optional point coordinates [N, 3] for spatial metrics
            confidence: Optional confidence scores [N]
            test_name: Name of the test scenario
            inference_time_ms: Model inference time in milliseconds
            
        Returns:
            ClassificationEvaluationMetrics object
        """
        logger.info(f"Evaluating predictions for '{test_name}'...")
        
        # Validate inputs
        assert len(y_true) == len(y_pred), "Predictions and labels must have same length"
        
        # Compute base metrics using existing validator
        base_metrics = self.validator.compute_metrics(
            predicted=y_pred,
            reference=y_true,
            confidence_scores=confidence,
            points=points if self.compute_spatial_metrics else None,
        )
        
        # Create evaluation metrics
        metrics = ClassificationEvaluationMetrics()
        
        # Overall metrics
        metrics.overall_accuracy = base_metrics.overall_accuracy
        metrics.kappa_coefficient = base_metrics.kappa_coefficient
        metrics.f1_macro = base_metrics.f1_score
        
        # Per-class metrics (convert to class names)
        metrics.per_class_precision = {
            self.class_names.get(k, f"Class_{k}"): v 
            for k, v in base_metrics.per_class_precision.items()
        }
        metrics.per_class_recall = {
            self.class_names.get(k, f"Class_{k}"): v 
            for k, v in base_metrics.per_class_recall.items()
        }
        metrics.per_class_f1 = {
            self.class_names.get(k, f"Class_{k}"): v 
            for k, v in base_metrics.per_class_f1.items()
        }
        
        # Confusion matrix
        metrics.confusion_matrix = base_metrics.confusion_matrix
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        metrics.class_names = [
            self.class_names.get(c, f"Class_{c}") for c in unique_classes
        ]
        
        # Spatial metrics
        if self.compute_spatial_metrics and points is not None:
            metrics.spatial_coherence_score = base_metrics.spatial_coherence_score
            metrics.boundary_accuracy = self._compute_boundary_accuracy(
                y_true, y_pred, points
            )
        
        # Error analysis
        metrics.most_confused_pairs = self._get_confused_pairs(
            base_metrics.confusion_matrix, unique_classes
        )
        metrics.error_distribution = self._compute_error_distribution(y_true, y_pred)
        
        # Performance metrics
        if inference_time_ms is not None:
            metrics.inference_time_ms = inference_time_ms
            metrics.points_per_second = len(y_pred) / (inference_time_ms / 1000.0)
        
        # Compute IoU per class
        metrics.per_class_iou = self._compute_iou_per_class(y_true, y_pred)
        
        # Support (number of samples per class)
        unique, counts = np.unique(y_true, return_counts=True)
        metrics.per_class_support = {
            self.class_names.get(c, f"Class_{c}"): int(count)
            for c, count in zip(unique, counts)
        }
        
        # Balanced accuracy
        metrics.balanced_accuracy = np.mean(list(base_metrics.per_class_recall.values()))
        
        # Weighted F1
        total_support = sum(metrics.per_class_support.values())
        metrics.f1_weighted = sum(
            metrics.per_class_f1[name] * metrics.per_class_support[name]
            for name in metrics.per_class_f1.keys()
        ) / total_support
        
        # Store result
        result_entry = {
            "test_name": test_name,
            "metrics": metrics,
            "n_points": len(y_pred),
        }
        self.evaluation_results.append(result_entry)
        
        logger.info(f"✓ Overall Accuracy: {metrics.overall_accuracy:.2%}")
        logger.info(f"✓ Balanced Accuracy: {metrics.balanced_accuracy:.2%}")
        logger.info(f"✓ F1 Macro: {metrics.f1_macro:.2%}")
        logger.info(f"✓ Kappa: {metrics.kappa_coefficient:.3f}")
        
        return metrics
    
    def _compute_boundary_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        points: np.ndarray,
        boundary_threshold: float = 2.0,
    ) -> float:
        """Compute classification accuracy at class boundaries."""
        try:
            from scipy.spatial import cKDTree
            
            # Find boundary points (points with different class neighbors)
            tree = cKDTree(points)
            boundary_mask = np.zeros(len(points), dtype=bool)
            
            # Sample for efficiency (check 10% of points)
            sample_size = min(len(points) // 10, 10000)
            sample_indices = np.random.choice(len(points), sample_size, replace=False)
            
            for idx in sample_indices:
                neighbors = tree.query_ball_point(points[idx], boundary_threshold)
                neighbor_classes = y_true[neighbors]
                if len(np.unique(neighbor_classes)) > 1:
                    boundary_mask[idx] = True
            
            if boundary_mask.sum() > 0:
                boundary_acc = (y_true[boundary_mask] == y_pred[boundary_mask]).mean()
                return float(boundary_acc)
            
        except Exception as e:
            logger.warning(f"Could not compute boundary accuracy: {e}")
        
        return 0.0
    
    def _get_confused_pairs(
        self,
        confusion_matrix: np.ndarray,
        class_ids: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[str, str, int]]:
        """Get most confused class pairs."""
        confused_pairs = []
        
        for i in range(len(confusion_matrix)):
            for j in range(len(confusion_matrix)):
                if i != j:
                    count = int(confusion_matrix[i, j])
                    if count > 0:
                        class1 = self.class_names.get(class_ids[i], f"Class_{class_ids[i]}")
                        class2 = self.class_names.get(class_ids[j], f"Class_{class_ids[j]}")
                        confused_pairs.append((class1, class2, count))
        
        # Sort by count and return top k
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        return confused_pairs[:top_k]
    
    def _compute_error_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, int]:
        """Compute distribution of errors by true class."""
        error_dist = defaultdict(int)
        
        for true_label, pred_label in zip(y_true, y_pred):
            if true_label != pred_label:
                class_name = self.class_names.get(true_label, f"Class_{true_label}")
                error_dist[class_name] += 1
        
        return dict(error_dist)
    
    def _compute_iou_per_class(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute IoU (Intersection over Union) for each class."""
        iou_dict = {}
        
        for class_id in np.unique(y_true):
            true_mask = y_true == class_id
            pred_mask = y_pred == class_id
            
            intersection = np.logical_and(true_mask, pred_mask).sum()
            union = np.logical_or(true_mask, pred_mask).sum()
            
            iou = intersection / union if union > 0 else 0.0
            class_name = self.class_names.get(class_id, f"Class_{class_id}")
            iou_dict[class_name] = float(iou)
        
        return iou_dict
    
    def evaluate_from_files(
        self,
        prediction_files: List[Path],
        ground_truth_files: List[Path],
        test_names: Optional[List[str]] = None,
    ) -> List[ClassificationEvaluationMetrics]:
        """
        Evaluate predictions from saved files.
        
        Args:
            prediction_files: List of NPZ files with predictions
            ground_truth_files: List of NPZ files with ground truth
            test_names: Optional names for each test
            
        Returns:
            List of evaluation metrics
        """
        assert len(prediction_files) == len(ground_truth_files), \
            "Number of prediction and ground truth files must match"
        
        results = []
        
        for i, (pred_file, gt_file) in enumerate(zip(prediction_files, ground_truth_files)):
            test_name = test_names[i] if test_names else f"test_{i}"
            
            # Load data
            pred_data = np.load(pred_file)
            gt_data = np.load(gt_file)
            
            y_pred = pred_data['labels']
            y_true = gt_data['labels']
            points = gt_data.get('points', None)
            confidence = pred_data.get('confidence', None)
            
            # Evaluate
            metrics = self.evaluate_predictions(
                y_true=y_true,
                y_pred=y_pred,
                points=points,
                confidence=confidence,
                test_name=test_name,
            )
            
            results.append(metrics)
        
        return results
    
    def save_report(
        self,
        output_path: Optional[Path] = None,
        format: str = "json",
    ) -> Path:
        """
        Save evaluation report.
        
        Args:
            output_path: Output file path (auto-generated if None)
            format: Report format ('json', 'csv', or 'html')
            
        Returns:
            Path to saved report
        """
        if output_path is None:
            output_path = self.output_dir / f"evaluation_report.{format}"
        
        if format == "json":
            return self._save_json_report(output_path)
        elif format == "csv":
            return self._save_csv_report(output_path)
        elif format == "html":
            return self._save_html_report(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_json_report(self, output_path: Path) -> Path:
        """Save report as JSON."""
        report = {
            "evaluation_summary": {
                "num_tests": len(self.evaluation_results),
                "timestamp": pd.Timestamp.now().isoformat(),
            },
            "tests": [
                {
                    "name": entry["test_name"],
                    "n_points": entry["n_points"],
                    "metrics": entry["metrics"].to_dict(),
                }
                for entry in self.evaluation_results
            ],
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✓ Saved JSON report to {output_path}")
        return output_path
    
    def _save_csv_report(self, output_path: Path) -> Path:
        """Save report as CSV."""
        rows = []
        
        for entry in self.evaluation_results:
            metrics = entry["metrics"]
            row = {
                "test_name": entry["test_name"],
                "n_points": entry["n_points"],
                "overall_accuracy": metrics.overall_accuracy,
                "balanced_accuracy": metrics.balanced_accuracy,
                "f1_macro": metrics.f1_macro,
                "f1_weighted": metrics.f1_weighted,
                "kappa": metrics.kappa_coefficient,
                "spatial_coherence": metrics.spatial_coherence_score,
                "boundary_accuracy": metrics.boundary_accuracy,
                "inference_time_ms": metrics.inference_time_ms,
                "points_per_second": metrics.points_per_second,
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        logger.info(f"✓ Saved CSV report to {output_path}")
        return output_path
    
    def _save_html_report(self, output_path: Path) -> Path:
        """Save report as HTML."""
        html_content = ["<html><head><title>Classification Evaluation Report</title>"]
        html_content.append("<style>")
        html_content.append("body { font-family: Arial, sans-serif; margin: 20px; }")
        html_content.append("table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
        html_content.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html_content.append("th { background-color: #4CAF50; color: white; }")
        html_content.append("tr:nth-child(even) { background-color: #f2f2f2; }")
        html_content.append(".metric-good { color: green; font-weight: bold; }")
        html_content.append(".metric-ok { color: orange; }")
        html_content.append(".metric-bad { color: red; }")
        html_content.append("</style></head><body>")
        
        html_content.append("<h1>Classification Evaluation Report</h1>")
        html_content.append(f"<p>Generated: {pd.Timestamp.now()}</p>")
        html_content.append(f"<p>Number of tests: {len(self.evaluation_results)}</p>")
        
        for entry in self.evaluation_results:
            metrics = entry["metrics"]
            html_content.append(f"<h2>{entry['test_name']}</h2>")
            html_content.append(f"<p>Points evaluated: {entry['n_points']:,}</p>")
            
            # Overall metrics table
            html_content.append("<h3>Overall Metrics</h3>")
            html_content.append("<table>")
            html_content.append("<tr><th>Metric</th><th>Value</th></tr>")
            
            for metric_name, value in [
                ("Overall Accuracy", metrics.overall_accuracy),
                ("Balanced Accuracy", metrics.balanced_accuracy),
                ("F1 Macro", metrics.f1_macro),
                ("F1 Weighted", metrics.f1_weighted),
                ("Kappa Coefficient", metrics.kappa_coefficient),
            ]:
                css_class = "metric-good" if value > 0.8 else "metric-ok" if value > 0.6 else "metric-bad"
                html_content.append(
                    f"<tr><td>{metric_name}</td>"
                    f"<td class='{css_class}'>{value:.2%}</td></tr>"
                )
            
            html_content.append("</table>")
            
            # Per-class metrics table
            html_content.append("<h3>Per-Class Metrics</h3>")
            html_content.append("<table>")
            html_content.append("<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>IoU</th><th>Support</th></tr>")
            
            for class_name in metrics.per_class_precision.keys():
                precision = metrics.per_class_precision.get(class_name, 0.0)
                recall = metrics.per_class_recall.get(class_name, 0.0)
                f1 = metrics.per_class_f1.get(class_name, 0.0)
                iou = metrics.per_class_iou.get(class_name, 0.0)
                support = metrics.per_class_support.get(class_name, 0)
                
                html_content.append(
                    f"<tr><td>{class_name}</td>"
                    f"<td>{precision:.2%}</td>"
                    f"<td>{recall:.2%}</td>"
                    f"<td>{f1:.2%}</td>"
                    f"<td>{iou:.2%}</td>"
                    f"<td>{support:,}</td></tr>"
                )
            
            html_content.append("</table>")
        
        html_content.append("</body></html>")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(html_content))
        
        logger.info(f"✓ Saved HTML report to {output_path}")
        return output_path


# ============================================================================
# Convenience Functions
# ============================================================================

def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    points: Optional[np.ndarray] = None,
    confidence: Optional[np.ndarray] = None,
    class_names: Optional[Dict[int, str]] = None,
    output_dir: Optional[Path] = None,
    save_report: bool = True,
) -> ClassificationEvaluationMetrics:
    """
    Quick evaluation function.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        points: Optional point coordinates
        confidence: Optional confidence scores
        class_names: Optional class name mapping
        output_dir: Output directory for reports
        save_report: Whether to save evaluation report
        
    Returns:
        ClassificationEvaluationMetrics object
    """
    evaluator = ClassificationEvaluator(
        class_names=class_names,
        output_dir=output_dir,
    )
    
    metrics = evaluator.evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        points=points,
        confidence=confidence,
    )
    
    if save_report:
        evaluator.save_report(format="json")
        evaluator.save_report(format="html")
    
    return metrics
