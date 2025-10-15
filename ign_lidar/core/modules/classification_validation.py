"""
Classification Validation and Quality Metrics

This module provides tools for validating classification results and computing
quality metrics. Includes:
- Confusion matrix computation
- Per-class accuracy metrics
- Spatial coherence analysis
- Error detection and correction
- Quality scoring

Author: IGN LiDAR HD Dataset Team
Date: October 15, 2025
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================================
# Quality Metrics
# ============================================================================

@dataclass
class ClassificationMetrics:
    """Comprehensive classification quality metrics."""
    
    # Basic metrics
    overall_accuracy: float = 0.0
    kappa_coefficient: float = 0.0
    f1_score: float = 0.0
    
    # Per-class metrics
    per_class_accuracy: Dict[int, float] = field(default_factory=dict)
    per_class_precision: Dict[int, float] = field(default_factory=dict)
    per_class_recall: Dict[int, float] = field(default_factory=dict)
    per_class_f1: Dict[int, float] = field(default_factory=dict)
    
    # Confusion matrix
    confusion_matrix: Optional[np.ndarray] = None
    class_names: Optional[Dict[int, str]] = None
    
    # Spatial coherence
    spatial_coherence_score: float = 0.0
    isolated_point_ratio: float = 0.0
    
    # Confidence statistics
    avg_confidence: float = 0.0
    low_confidence_ratio: float = 0.0  # Ratio of points with confidence < 0.5
    
    # Error analysis
    most_confused_pairs: List[Tuple[int, int, int]] = field(default_factory=list)  # (class1, class2, count)
    
    def get_summary(self) -> str:
        """Get human-readable summary of metrics."""
        summary = [
            "=" * 70,
            "Classification Quality Metrics",
            "=" * 70,
            f"Overall Accuracy: {self.overall_accuracy:.2%}",
            f"Kappa Coefficient: {self.kappa_coefficient:.3f}",
            f"F1 Score (macro): {self.f1_score:.2%}",
            "",
            "Per-Class Performance:",
            "-" * 70,
        ]
        
        # Sort classes by accuracy
        if self.per_class_accuracy:
            sorted_classes = sorted(
                self.per_class_accuracy.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            summary.append(f"{'Class':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
            summary.append("-" * 70)
            
            for class_id, accuracy in sorted_classes:
                class_name = self.class_names.get(class_id, f"Class_{class_id}") if self.class_names else str(class_id)
                precision = self.per_class_precision.get(class_id, 0.0)
                recall = self.per_class_recall.get(class_id, 0.0)
                f1 = self.per_class_f1.get(class_id, 0.0)
                
                summary.append(
                    f"{class_name:<20} {accuracy:>9.1%} {precision:>9.1%} {recall:>9.1%} {f1:>9.1%}"
                )
        
        summary.extend([
            "",
            "Spatial Quality:",
            "-" * 70,
            f"Spatial Coherence: {self.spatial_coherence_score:.2%}",
            f"Isolated Points: {self.isolated_point_ratio:.2%}",
        ])
        
        if self.avg_confidence > 0:
            summary.extend([
                "",
                "Confidence Statistics:",
                "-" * 70,
                f"Average Confidence: {self.avg_confidence:.2%}",
                f"Low Confidence Points: {self.low_confidence_ratio:.2%}",
            ])
        
        if self.most_confused_pairs:
            summary.extend([
                "",
                "Most Confused Class Pairs:",
                "-" * 70,
            ])
            for class1, class2, count in self.most_confused_pairs[:5]:
                name1 = self.class_names.get(class1, f"Class_{class1}") if self.class_names else str(class1)
                name2 = self.class_names.get(class2, f"Class_{class2}") if self.class_names else str(class2)
                summary.append(f"  {name1} <-> {name2}: {count:,} errors")
        
        summary.append("=" * 70)
        
        return "\n".join(summary)


class ClassificationValidator:
    """
    Validator for classification results.
    
    Provides tools for:
    - Computing accuracy metrics against ground truth
    - Analyzing spatial coherence
    - Detecting common errors
    - Generating quality reports
    """
    
    def __init__(self, class_names: Optional[Dict[int, str]] = None):
        """
        Initialize validator.
        
        Args:
            class_names: Optional mapping of class IDs to human-readable names
        """
        self.class_names = class_names or {}
    
    def compute_metrics(
        self,
        predicted: np.ndarray,
        reference: np.ndarray,
        confidence_scores: Optional[np.ndarray] = None,
        points: Optional[np.ndarray] = None
    ) -> ClassificationMetrics:
        """
        Compute comprehensive classification metrics.
        
        Args:
            predicted: Predicted class labels [N]
            reference: Reference/ground truth labels [N]
            confidence_scores: Optional confidence scores [N]
            points: Optional point coordinates [N, 3] for spatial analysis
        
        Returns:
            ClassificationMetrics object
        """
        metrics = ClassificationMetrics()
        metrics.class_names = self.class_names
        
        # Basic validation
        assert len(predicted) == len(reference), "Predicted and reference must have same length"
        
        # Overall accuracy
        correct = predicted == reference
        metrics.overall_accuracy = np.mean(correct)
        
        # Confusion matrix
        classes = np.unique(np.concatenate([predicted, reference]))
        n_classes = len(classes)
        confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
        
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for pred, ref in zip(predicted, reference):
            confusion[class_to_idx[ref], class_to_idx[pred]] += 1
        
        metrics.confusion_matrix = confusion
        
        # Per-class metrics
        for i, class_id in enumerate(classes):
            # True positives, false positives, false negatives
            tp = confusion[i, i]
            fp = confusion[:, i].sum() - tp
            fn = confusion[i, :].sum() - tp
            tn = confusion.sum() - tp - fp - fn
            
            # Accuracy (for this class)
            class_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            metrics.per_class_accuracy[int(class_id)] = class_accuracy
            
            # Precision
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            metrics.per_class_precision[int(class_id)] = precision
            
            # Recall
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics.per_class_recall[int(class_id)] = recall
            
            # F1 score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            metrics.per_class_f1[int(class_id)] = f1
        
        # Macro-averaged F1
        if metrics.per_class_f1:
            metrics.f1_score = np.mean(list(metrics.per_class_f1.values()))
        
        # Kappa coefficient
        metrics.kappa_coefficient = self._compute_kappa(confusion)
        
        # Confidence statistics
        if confidence_scores is not None:
            metrics.avg_confidence = float(np.mean(confidence_scores))
            metrics.low_confidence_ratio = float(np.mean(confidence_scores < 0.5))
        
        # Spatial coherence analysis
        if points is not None:
            metrics.spatial_coherence_score = self._compute_spatial_coherence(
                predicted, points
            )
            metrics.isolated_point_ratio = self._compute_isolated_ratio(
                predicted, points
            )
        
        # Most confused class pairs
        metrics.most_confused_pairs = self._find_confused_pairs(confusion, classes)
        
        return metrics
    
    def _compute_kappa(self, confusion: np.ndarray) -> float:
        """Compute Cohen's Kappa coefficient."""
        n = confusion.sum()
        if n == 0:
            return 0.0
        
        # Observed accuracy
        po = np.trace(confusion) / n
        
        # Expected accuracy by chance
        row_sums = confusion.sum(axis=1)
        col_sums = confusion.sum(axis=0)
        pe = np.sum(row_sums * col_sums) / (n * n)
        
        # Kappa
        kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
        return kappa
    
    def _compute_spatial_coherence(
        self,
        labels: np.ndarray,
        points: np.ndarray,
        k_neighbors: int = 10
    ) -> float:
        """
        Compute spatial coherence score.
        
        Measures how consistent labels are with their spatial neighbors.
        Higher scores indicate more spatially coherent classification.
        """
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            logger.warning("scipy not available, skipping spatial coherence")
            return 0.0
        
        # Build KD-tree
        tree = cKDTree(points)
        
        # Query neighbors for each point
        distances, indices = tree.query(points, k=k_neighbors + 1)
        
        # Compute coherence (ratio of neighbors with same label)
        coherence_scores = []
        for i, neighbors in enumerate(indices):
            # Exclude self (first neighbor)
            neighbor_labels = labels[neighbors[1:]]
            same_label = neighbor_labels == labels[i]
            coherence = np.mean(same_label)
            coherence_scores.append(coherence)
        
        return float(np.mean(coherence_scores))
    
    def _compute_isolated_ratio(
        self,
        labels: np.ndarray,
        points: np.ndarray,
        k_neighbors: int = 10,
        threshold: float = 0.3
    ) -> float:
        """
        Compute ratio of isolated points.
        
        A point is considered isolated if fewer than threshold fraction of
        its neighbors share the same label.
        """
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            logger.warning("scipy not available, skipping isolated point detection")
            return 0.0
        
        # Build KD-tree
        tree = cKDTree(points)
        
        # Query neighbors
        distances, indices = tree.query(points, k=k_neighbors + 1)
        
        # Count isolated points
        isolated_count = 0
        for i, neighbors in enumerate(indices):
            neighbor_labels = labels[neighbors[1:]]
            same_label_ratio = np.mean(neighbor_labels == labels[i])
            if same_label_ratio < threshold:
                isolated_count += 1
        
        return isolated_count / len(labels)
    
    def _find_confused_pairs(
        self,
        confusion: np.ndarray,
        classes: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[int, int, int]]:
        """Find the most commonly confused class pairs."""
        confused_pairs = []
        
        n_classes = len(classes)
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                # Count bidirectional confusion
                confusion_count = confusion[i, j] + confusion[j, i]
                if confusion_count > 0:
                    confused_pairs.append((
                        int(classes[i]),
                        int(classes[j]),
                        int(confusion_count)
                    ))
        
        # Sort by confusion count
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return confused_pairs[:top_k]
    
    def detect_errors(
        self,
        labels: np.ndarray,
        features: Dict[str, np.ndarray],
        confidence_scores: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Detect potential classification errors.
        
        Returns masks for different error types:
        - low_confidence: Points with low confidence scores
        - height_mismatch: Height inconsistent with label
        - ndvi_mismatch: NDVI inconsistent with label
        - isolated: Spatially isolated points
        
        Args:
            labels: Classification labels [N]
            features: Dictionary of features (height, ndvi, points, etc.)
            confidence_scores: Optional confidence scores [N]
        
        Returns:
            Dictionary of error masks
        """
        n_points = len(labels)
        errors = {}
        
        # Low confidence points
        if confidence_scores is not None:
            errors['low_confidence'] = confidence_scores < 0.5
        
        # Height mismatches
        if 'height' in features:
            height = features['height']
            height_errors = np.zeros(n_points, dtype=bool)
            
            # Check for common height-label inconsistencies
            # Example: High vegetation should have significant height
            veg_high_mask = labels == 11  # Assuming LOD2 high vegetation
            height_errors[veg_high_mask & (height < 2.0)] = True
            
            # Example: Ground should be low
            ground_mask = labels == 9  # Assuming LOD2 ground
            height_errors[ground_mask & (height > 0.5)] = True
            
            errors['height_mismatch'] = height_errors
        
        # NDVI mismatches
        if 'ndvi' in features:
            ndvi = features['ndvi']
            ndvi_errors = np.zeros(n_points, dtype=bool)
            
            # Vegetation should have high NDVI
            veg_mask = np.isin(labels, [10, 11])  # LOD2 vegetation classes
            ndvi_errors[veg_mask & (ndvi < 0.3)] = True
            
            # Buildings should have low NDVI
            building_mask = labels == 0  # LOD2 wall
            ndvi_errors[building_mask & (ndvi > 0.3)] = True
            
            errors['ndvi_mismatch'] = ndvi_errors
        
        # Spatial isolation
        if 'points' in features:
            points = features['points']
            isolated_mask = self._detect_isolated_points(labels, points)
            errors['isolated'] = isolated_mask
        
        return errors
    
    def _detect_isolated_points(
        self,
        labels: np.ndarray,
        points: np.ndarray,
        k_neighbors: int = 10,
        threshold: float = 0.3
    ) -> np.ndarray:
        """Detect spatially isolated points (same as _compute_isolated_ratio but returns mask)."""
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            logger.warning("scipy not available, skipping isolated point detection")
            return np.zeros(len(labels), dtype=bool)
        
        tree = cKDTree(points)
        distances, indices = tree.query(points, k=k_neighbors + 1)
        
        isolated_mask = np.zeros(len(labels), dtype=bool)
        for i, neighbors in enumerate(indices):
            neighbor_labels = labels[neighbors[1:]]
            same_label_ratio = np.mean(neighbor_labels == labels[i])
            if same_label_ratio < threshold:
                isolated_mask[i] = True
        
        return isolated_mask


# ============================================================================
# Error Correction
# ============================================================================

class ErrorCorrector:
    """
    Automatic error correction for common classification mistakes.
    
    Uses heuristics and spatial consistency to correct obvious errors.
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize error corrector.
        
        Args:
            confidence_threshold: Minimum confidence to keep original label
        """
        self.confidence_threshold = confidence_threshold
    
    def correct_isolated_points(
        self,
        labels: np.ndarray,
        points: np.ndarray,
        confidence_scores: Optional[np.ndarray] = None,
        k_neighbors: int = 10
    ) -> Tuple[np.ndarray, int]:
        """
        Correct isolated points by assigning majority neighbor label.
        
        Args:
            labels: Classification labels [N]
            points: Point coordinates [N, 3]
            confidence_scores: Optional confidence scores [N]
            k_neighbors: Number of neighbors to consider
        
        Returns:
            Tuple of (corrected_labels, num_corrected)
        """
        try:
            from scipy.spatial import cKDTree
            from scipy.stats import mode
        except ImportError:
            logger.warning("scipy not available, skipping correction")
            return labels.copy(), 0
        
        corrected = labels.copy()
        num_corrected = 0
        
        # Build KD-tree
        tree = cKDTree(points)
        distances, indices = tree.query(points, k=k_neighbors + 1)
        
        # Correct each isolated point
        for i, neighbors in enumerate(indices):
            neighbor_labels = labels[neighbors[1:]]
            same_label_ratio = np.mean(neighbor_labels == labels[i])
            
            # Is this point isolated?
            if same_label_ratio < 0.3:
                # Check confidence
                if confidence_scores is None or confidence_scores[i] < self.confidence_threshold:
                    # Assign majority neighbor label
                    majority_label = mode(neighbor_labels, keepdims=True)[0][0]
                    corrected[i] = majority_label
                    num_corrected += 1
        
        logger.info(f"Corrected {num_corrected:,} isolated points")
        return corrected, num_corrected
    
    def correct_height_errors(
        self,
        labels: np.ndarray,
        height: np.ndarray,
        confidence_scores: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Correct height-based errors.
        
        Example corrections:
        - High vegetation with low height -> low vegetation
        - Ground with high elevation -> reclassify
        - Low objects labeled as buildings -> correct
        
        Args:
            labels: Classification labels [N]
            height: Height above ground [N]
            confidence_scores: Optional confidence scores [N]
        
        Returns:
            Tuple of (corrected_labels, num_corrected)
        """
        corrected = labels.copy()
        num_corrected = 0
        
        # LOD2 class IDs (adjust if using different schema)
        GROUND = 9
        VEG_LOW = 10
        VEG_HIGH = 11
        WALL = 0
        
        # High vegetation that's too low -> low vegetation
        mask = (labels == VEG_HIGH) & (height < 2.0)
        if confidence_scores is not None:
            mask &= confidence_scores < self.confidence_threshold
        corrected[mask] = VEG_LOW
        num_corrected += np.sum(mask)
        
        # Low vegetation that's too high -> high vegetation
        mask = (labels == VEG_LOW) & (height > 3.5)
        if confidence_scores is not None:
            mask &= confidence_scores < self.confidence_threshold
        corrected[mask] = VEG_HIGH
        num_corrected += np.sum(mask)
        
        # Ground that's elevated -> low vegetation or other
        mask = (labels == GROUND) & (height > 0.5)
        if confidence_scores is not None:
            mask &= confidence_scores < self.confidence_threshold
        corrected[mask] = VEG_LOW  # Conservative choice
        num_corrected += np.sum(mask)
        
        # Low objects labeled as building -> correct
        mask = (labels == WALL) & (height < 2.0)
        if confidence_scores is not None:
            mask &= confidence_scores < self.confidence_threshold
        corrected[mask] = VEG_LOW  # Or GROUND depending on other features
        num_corrected += np.sum(mask)
        
        logger.info(f"Corrected {num_corrected:,} height-based errors")
        return corrected, num_corrected
    
    def correct_ndvi_errors(
        self,
        labels: np.ndarray,
        ndvi: np.ndarray,
        confidence_scores: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Correct NDVI-based errors.
        
        Example corrections:
        - Vegetation with very low NDVI -> building/ground
        - Building with high NDVI -> vegetation
        
        Args:
            labels: Classification labels [N]
            ndvi: NDVI values [N]
            confidence_scores: Optional confidence scores [N]
        
        Returns:
            Tuple of (corrected_labels, num_corrected)
        """
        corrected = labels.copy()
        num_corrected = 0
        
        # LOD2 class IDs
        VEG_LOW = 10
        VEG_HIGH = 11
        WALL = 0
        
        # Vegetation with very low NDVI -> building
        veg_mask = np.isin(labels, [VEG_LOW, VEG_HIGH])
        mask = veg_mask & (ndvi < 0.2)
        if confidence_scores is not None:
            mask &= confidence_scores < self.confidence_threshold
        corrected[mask] = WALL
        num_corrected += np.sum(mask)
        
        # Building with high NDVI -> vegetation
        mask = (labels == WALL) & (ndvi > 0.5)
        if confidence_scores is not None:
            mask &= confidence_scores < self.confidence_threshold
        corrected[mask] = VEG_LOW
        num_corrected += np.sum(mask)
        
        logger.info(f"Corrected {num_corrected:,} NDVI-based errors")
        return corrected, num_corrected
    
    def apply_all_corrections(
        self,
        labels: np.ndarray,
        features: Dict[str, np.ndarray],
        confidence_scores: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Apply all available corrections.
        
        Args:
            labels: Classification labels [N]
            features: Dictionary of features (height, ndvi, points, etc.)
            confidence_scores: Optional confidence scores [N]
        
        Returns:
            Tuple of (corrected_labels, correction_counts)
        """
        corrected = labels.copy()
        counts = {}
        
        # Height corrections
        if 'height' in features:
            corrected, count = self.correct_height_errors(
                corrected, features['height'], confidence_scores
            )
            counts['height'] = count
        
        # NDVI corrections
        if 'ndvi' in features:
            corrected, count = self.correct_ndvi_errors(
                corrected, features['ndvi'], confidence_scores
            )
            counts['ndvi'] = count
        
        # Spatial corrections
        if 'points' in features:
            corrected, count = self.correct_isolated_points(
                corrected, features['points'], confidence_scores
            )
            counts['isolated'] = count
        
        total_corrected = sum(counts.values())
        logger.info(f"Total corrections applied: {total_corrected:,}")
        logger.info(f"Correction breakdown: {counts}")
        
        return corrected, counts


# ============================================================================
# Convenience Functions
# ============================================================================

def validate_classification(
    predicted: np.ndarray,
    reference: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
    confidence_scores: Optional[np.ndarray] = None,
    points: Optional[np.ndarray] = None,
    print_summary: bool = True
) -> ClassificationMetrics:
    """
    Convenience function to validate classification results.
    
    Args:
        predicted: Predicted labels [N]
        reference: Reference/ground truth labels [N]
        class_names: Optional class name mapping
        confidence_scores: Optional confidence scores [N]
        points: Optional point coordinates [N, 3]
        print_summary: Whether to print summary to console
    
    Returns:
        ClassificationMetrics object
    """
    validator = ClassificationValidator(class_names=class_names)
    metrics = validator.compute_metrics(
        predicted=predicted,
        reference=reference,
        confidence_scores=confidence_scores,
        points=points
    )
    
    if print_summary:
        print(metrics.get_summary())
    
    return metrics


def auto_correct_classification(
    labels: np.ndarray,
    features: Dict[str, np.ndarray],
    confidence_scores: Optional[np.ndarray] = None,
    confidence_threshold: float = 0.5
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Convenience function to automatically correct classification errors.
    
    Args:
        labels: Classification labels [N]
        features: Dictionary of features
        confidence_scores: Optional confidence scores [N]
        confidence_threshold: Minimum confidence to keep original label
    
    Returns:
        Tuple of (corrected_labels, correction_counts)
    """
    corrector = ErrorCorrector(confidence_threshold=confidence_threshold)
    corrected, counts = corrector.apply_all_corrections(
        labels=labels,
        features=features,
        confidence_scores=confidence_scores
    )
    
    return corrected, counts
