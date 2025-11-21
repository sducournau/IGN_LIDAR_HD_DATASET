"""
Base Classifier Interface for IGN LiDAR HD v3.2+

This module defines the interface that all classifiers must follow,
ensuring consistency across the codebase and making it easier for users
to swap between different classification strategies.

Key Features:
- Abstract BaseClassifier class with standard classify() signature
- ClassificationResult dataclass for all return values
- Input validation utilities
- Consistent error handling

Author: IGN LiDAR HD Team
Date: October 25, 2025
Version: 3.2.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import numpy as np

try:
    import geopandas as gpd

    HAS_GEOPANDAS = True
except ImportError:
    gpd = None
    HAS_GEOPANDAS = False


# ============================================================================
# Classification Result
# ============================================================================


@dataclass
class ClassificationResult:
    """
    Result object returned by all classifiers.

    This standardizes the return value across all classification methods,
    making it easier to work with different classifiers and ensuring
    consistent metadata tracking.

    Attributes:
        labels: Classification labels [N] - integer class codes
        confidence: Confidence scores [N], range [0, 1] (optional)
        metadata: Additional information about classification process

    Examples:
        >>> result = classifier.classify(points, features)
        >>> labels = result.labels
        >>> stats = result.get_statistics()
        >>> print(f"Classified {stats['total_points']} points")
    """

    labels: np.ndarray
    confidence: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate result data."""
        if not isinstance(self.labels, np.ndarray):
            raise TypeError(f"labels must be numpy array, got {type(self.labels)}")

        if self.labels.ndim != 1:
            raise ValueError(f"labels must be 1D array, got shape {self.labels.shape}")

        if self.confidence is not None:
            if not isinstance(self.confidence, np.ndarray):
                raise TypeError(
                    f"confidence must be numpy array, got {type(self.confidence)}"
                )
            if self.confidence.shape != self.labels.shape:
                raise ValueError(
                    f"confidence shape {self.confidence.shape} must match "
                    f"labels shape {self.labels.shape}"
                )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get classification statistics.

        Returns:
            Dictionary with statistics:
                - total_points: Total number of points
                - num_classes: Number of unique classes
                - class_distribution: Dict mapping class → count
                - class_percentages: Dict mapping class → percentage
                - avg_confidence: Average confidence (if available)
                - low_confidence_count: Points with confidence < 0.5

        Example:
            >>> stats = result.get_statistics()
            >>> print(f"Total: {stats['total_points']} points")
            >>> print(f"Classes: {stats['num_classes']}")
            >>> for cls, pct in stats['class_percentages'].items():
            ...     print(f"  Class {cls}: {pct:.1f}%")
        """
        unique, counts = np.unique(self.labels, return_counts=True)
        total = len(self.labels)

        stats = {
            "total_points": int(total),
            "num_classes": int(len(unique)),
            "class_distribution": {
                int(cls): int(count) for cls, count in zip(unique, counts)
            },
            "class_percentages": {
                int(cls): float(count) / total * 100
                for cls, count in zip(unique, counts)
            },
        }

        # Add confidence statistics if available
        if self.confidence is not None:
            stats["avg_confidence"] = float(np.mean(self.confidence))
            stats["min_confidence"] = float(np.min(self.confidence))
            stats["max_confidence"] = float(np.max(self.confidence))
            stats["low_confidence_count"] = int(np.sum(self.confidence < 0.5))
            stats["low_confidence_percentage"] = float(
                stats["low_confidence_count"] / total * 100
            )

        # Add metadata
        if self.metadata:
            stats["metadata"] = self.metadata

        return stats

    def filter_by_confidence(self, threshold: float = 0.5) -> "ClassificationResult":
        """
        Filter points by confidence threshold.

        Args:
            threshold: Minimum confidence (0-1)

        Returns:
            New ClassificationResult with only high-confidence points

        Raises:
            ValueError: If confidence scores are not available

        Example:
            >>> high_conf = result.filter_by_confidence(0.7)
            >>> print(f"Kept {len(high_conf.labels)} high-confidence points")
        """
        if self.confidence is None:
            raise ValueError(
                "Cannot filter by confidence - no confidence scores available"
            )

        mask = self.confidence >= threshold

        return ClassificationResult(
            labels=self.labels[mask],
            confidence=self.confidence[mask],
            metadata={
                **self.metadata,
                "filtered_by_confidence": threshold,
                "points_removed": int(np.sum(~mask)),
            },
        )

    def get_class_mask(self, class_id: int) -> np.ndarray:
        """
        Get boolean mask for a specific class.

        Args:
            class_id: Class ID to select

        Returns:
            Boolean mask [N] where True = this class

        Example:
            >>> building_mask = result.get_class_mask(6)  # ASPRS Building
            >>> building_points = points[building_mask]
        """
        return self.labels == class_id


# ============================================================================
# Base Classifier
# ============================================================================


class BaseClassifier(ABC):
    """
    Abstract base class for all classifiers.

    All classifiers in IGN LiDAR HD must inherit from this class and
    implement the classify() method with the standard signature.
    This ensures API consistency and makes it easy to swap classifiers.

    The classify() method must:
    1. Accept points [N, 3], features dict, and optional ground truth
    2. Return ClassificationResult with labels and optional confidence
    3. Validate inputs using validate_inputs()
    4. Handle errors gracefully with clear messages

    Examples:
        >>> class MyClassifier(BaseClassifier):
        ...     def classify(self, points, features, ground_truth=None, **kwargs):
        ...         self.validate_inputs(points, features)
        ...         # ... classification logic ...
        ...         return ClassificationResult(labels=labels, confidence=conf)

        >>> classifier = MyClassifier()
        >>> result = classifier.classify(points, features)
    """

    @abstractmethod
    def classify(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        ground_truth: Optional[Union["gpd.GeoDataFrame", Dict[str, Any]]] = None,
        **kwargs,
    ) -> ClassificationResult:
        """
        Classify point cloud.

        This is the main method that all classifiers must implement.

        Args:
            points: Point cloud [N, 3] with XYZ coordinates
            features: Dictionary mapping feature name → feature array [N]
                     Common features: 'planarity', 'verticality', 'height',
                     'curvature', 'ndvi', etc.
            ground_truth: Optional ground truth data:
                - GeoDataFrame with polygons (BD TOPO, cadastre, etc.)
                - Dictionary with ground truth metadata
            **kwargs: Classifier-specific parameters

        Returns:
            ClassificationResult with:
                - labels: Classification labels [N]
                - confidence: Optional confidence scores [N]
                - metadata: Information about classification process

        Raises:
            ValueError: If input data is invalid
            ProcessingError: If classification fails

        Examples:
            >>> result = classifier.classify(points, features)
            >>> labels = result.labels
            >>> print(result.get_statistics())

            >>> # With ground truth
            >>> result = classifier.classify(
            ...     points, features,
            ...     ground_truth=bd_topo_gdf
            ... )
        """
        pass

    def validate_inputs(
        self, points: np.ndarray, features: Dict[str, np.ndarray]
    ) -> None:
        """
        Validate input data.

        Checks that:
        1. Points is 2D array with shape [N, 3]
        2. All features have length N (matching points)
        3. Features are numeric arrays

        Args:
            points: Point cloud array
            features: Feature dictionary

        Raises:
            ValueError: If inputs are invalid

        Example:
            >>> def classify(self, points, features, **kwargs):
            ...     self.validate_inputs(points, features)
            ...     # ... proceed with classification ...
        """
        # Validate points
        if not isinstance(points, np.ndarray):
            raise ValueError(f"points must be numpy array, got {type(points)}")

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must have shape [N, 3], got {points.shape}")

        n_points = len(points)

        if n_points == 0:
            raise ValueError("points array is empty")

        # Validate features
        if not isinstance(features, dict):
            raise ValueError(f"features must be dictionary, got {type(features)}")

        if not features:
            raise ValueError("features dictionary is empty")

        # Validate each feature
        for name, feat in features.items():
            if not isinstance(feat, np.ndarray):
                raise ValueError(
                    f"Feature '{name}' must be numpy array, got {type(feat)}"
                )

            if len(feat) != n_points:
                raise ValueError(
                    f"Feature '{name}' has {len(feat)} values, "
                    f"expected {n_points} (matching points)"
                )

            if not np.issubdtype(feat.dtype, np.number):
                raise ValueError(
                    f"Feature '{name}' must be numeric, got dtype {feat.dtype}"
                )

            # Check for invalid values
            if np.any(~np.isfinite(feat)):
                n_invalid = np.sum(~np.isfinite(feat))
                raise ValueError(
                    f"Feature '{name}' contains {n_invalid} invalid values "
                    f"(NaN or Inf)"
                )

    def validate_ground_truth(
        self, ground_truth: Union["gpd.GeoDataFrame", Dict[str, Any]]
    ) -> None:
        """
        Validate ground truth data.

        Args:
            ground_truth: Ground truth data (GeoDataFrame or dict)

        Raises:
            ValueError: If ground truth is invalid
            ImportError: If geopandas not available but GeoDataFrame passed
        """
        if ground_truth is None:
            return

        if HAS_GEOPANDAS and isinstance(ground_truth, gpd.GeoDataFrame):
            if len(ground_truth) == 0:
                raise ValueError("Ground truth GeoDataFrame is empty")

            if "geometry" not in ground_truth.columns:
                raise ValueError(
                    "Ground truth GeoDataFrame must have 'geometry' column"
                )

        elif isinstance(ground_truth, dict):
            if not ground_truth:
                raise ValueError("Ground truth dictionary is empty")

        else:
            raise ValueError(
                f"ground_truth must be GeoDataFrame or dict, "
                f"got {type(ground_truth)}"
            )

    def __repr__(self) -> str:
        """String representation of classifier."""
        return f"{self.__class__.__name__}()"


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "BaseClassifier",
    "ClassificationResult",
]
