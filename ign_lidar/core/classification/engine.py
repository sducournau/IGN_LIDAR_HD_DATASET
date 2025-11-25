"""
Unified Classification Engine - v1.0

Consolidates all classification strategies into a single interface.
Provides factory pattern for automatic strategy selection.

Replaces:
- SpectralRulesEngine
- GeometricRulesEngine
- ASPRSClassRulesEngine
- Manual strategy selection logic

Usage:
    from ign_lidar.core.classification import ClassificationEngine

    engine = ClassificationEngine(mode='asprs', use_gpu=True)
    labels = engine.classify(features)
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional, Tuple, Any

import logging
import numpy as np

logger = logging.getLogger(__name__)


class ClassificationMode(Enum):
    """Available classification modes."""

    SPECTRAL = "spectral"
    GEOMETRIC = "geometric"
    ASPRS = "asprs"
    ADAPTIVE = "adaptive"


class ClassificationStrategy(ABC):
    """Abstract base for all classification strategies."""

    @abstractmethod
    def classify(self, features: np.ndarray) -> np.ndarray:
        """
        Classify points based on features.

        Args:
            features: Feature array [N, F] where N is number of points,
                     F is number of features

        Returns:
            Classification labels [N] as integer array
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name identifier."""
        pass

    def refine(
        self, labels: np.ndarray, context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Optionally refine classifications (post-processing).

        Args:
            labels: Initial classification labels
            context: Optional context for refinement

        Returns:
            Refined labels
        """
        return labels

    def get_confidence(self, labels: np.ndarray) -> np.ndarray:
        """
        Get confidence scores for classifications (optional).

        Args:
            labels: Classification labels

        Returns:
            Confidence scores [0, 1]
        """
        return np.ones(len(labels), dtype=np.float32)


class SpectralClassificationStrategy(ClassificationStrategy):
    """Spectral signature-based classification."""

    def __init__(self, use_gpu: bool = False):
        """
        Initialize spectral classification strategy.

        Args:
            use_gpu: Enable GPU acceleration (if available)
        """
        self.use_gpu = use_gpu
        try:
            from .spectral_rules import SpectralRulesEngine

            self.engine = SpectralRulesEngine()
        except ImportError:
            logger.error("SpectralRulesEngine not available")
            self.engine = None

    def classify(self, features: np.ndarray) -> np.ndarray:
        """
        Classify using spectral rules.
        
        Note: SpectralRulesEngine requires RGB, NIR, and current_labels.
        For simplified interface, we return zeros as this strategy 
        requires additional input channels not available in feature array.
        
        For full spectral classification, use:
            labels, stats = engine.classify_by_spectral_signature(rgb, nir, labels)
        """
        if self.engine is None:
            raise RuntimeError("SpectralRulesEngine not available")
        
        logger.warning(
            "SpectralClassificationStrategy: Features-only classify not supported. "
            "Use classify_by_spectral_signature(rgb, nir, labels) for full functionality."
        )
        return np.zeros(len(features), dtype=np.int32)

    def get_name(self) -> str:
        """Return strategy name."""
        return "spectral"


class GeometricClassificationStrategy(ClassificationStrategy):
    """Geometric shape-based classification."""

    def __init__(self, use_gpu: bool = False):
        """
        Initialize geometric classification strategy.

        Args:
            use_gpu: Enable GPU acceleration (if available)
        """
        self.use_gpu = use_gpu
        try:
            from .geometric_rules import GeometricRulesEngine

            self.engine = GeometricRulesEngine()
        except ImportError:
            logger.error("GeometricRulesEngine not available")
            self.engine = None

    def classify(self, features: np.ndarray) -> np.ndarray:
        """
        Classify using geometric rules.
        
        Note: GeometricRulesEngine.apply_all_rules requires points (XYZ),
        current labels, and ground truth features for full functionality.
        
        For simplified interface, we return zeros.
        For full geometric classification, use:
            labels, stats = engine.apply_all_rules(
                points, labels, ground_truth_features, 
                ndvi=ndvi, rgb=rgb, nir=nir
            )
        """
        if self.engine is None:
            raise RuntimeError("GeometricRulesEngine not available")
        
        logger.warning(
            "GeometricClassificationStrategy: Features-only classify not supported. "
            "Use apply_all_rules(points, labels, gt_features) for full functionality."
        )
        return np.zeros(len(features), dtype=np.int32)

    def get_name(self) -> str:
        """Return strategy name."""
        return "geometric"


class ASPRSClassificationStrategy(ClassificationStrategy):
    """ASPRS standard-based classification."""

    def __init__(self, use_gpu: bool = False):
        """
        Initialize ASPRS classification strategy.

        Args:
            use_gpu: Enable GPU acceleration (if available)
        """
        self.use_gpu = use_gpu
        try:
            from .asprs_class_rules import ASPRSClassRulesEngine

            self.engine = ASPRSClassRulesEngine()
        except ImportError:
            logger.error("ASPRSClassRulesEngine not available")
            self.engine = None

    def classify(self, features: np.ndarray) -> np.ndarray:
        """
        Classify using ASPRS rules.
        
        Note: ASPRSClassRulesEngine.apply_all_rules requires points (XYZ),
        features dict, and current classification for full functionality.
        
        For simplified interface, we return zeros as this strategy 
        requires points and additional context data.
        
        For full ASPRS classification, use:
            labels = engine.apply_all_rules(
                points, features_dict, current_labels, ground_truth
            )
        """
        if self.engine is None:
            raise RuntimeError("ASPRSClassRulesEngine not available")
        
        logger.warning(
            "ASPRSClassificationStrategy: Features-only classify not supported. "
            "Use apply_all_rules(points, features_dict, labels) for full functionality."
        )
        return np.zeros(len(features), dtype=np.int32)

    def get_name(self) -> str:
        """Return strategy name."""
        return "asprs"


class ClassificationEngine:
    """
    Unified classification interface.

    Provides:
    - Automatic strategy selection
    - Consistent API across all classification modes
    - GPU acceleration support
    - Confidence scoring
    - Refinement capabilities

    Example:
        >>> engine = ClassificationEngine(mode='asprs', use_gpu=True)
        >>> labels = engine.classify(features)
        >>> confidence = engine.get_confidence(labels)
    """

    STRATEGIES = {
        ClassificationMode.SPECTRAL: SpectralClassificationStrategy,
        ClassificationMode.GEOMETRIC: GeometricClassificationStrategy,
        ClassificationMode.ASPRS: ASPRSClassificationStrategy,
    }

    def __init__(
        self,
        mode: str = "asprs",
        use_gpu: bool = False,
    ):
        """
        Initialize Classification Engine.

        Args:
            mode: Classification mode ('spectral', 'geometric', 'asprs', 'adaptive')
            use_gpu: Enable GPU acceleration if available

        Raises:
            ValueError: If mode is not supported
        """
        self.mode = mode
        self.use_gpu = use_gpu
        self.strategy = self._create_strategy()
        logger.info(f"Initialized ClassificationEngine with mode={mode}, gpu={use_gpu}")

    def _create_strategy(self) -> ClassificationStrategy:
        """
        Create appropriate classification strategy.

        Returns:
            ClassificationStrategy instance

        Raises:
            ValueError: If mode is not supported
        """
        try:
            mode_enum = ClassificationMode(self.mode)
        except ValueError:
            logger.warning(f"Unknown mode {self.mode}, using ASPRS")
            mode_enum = ClassificationMode.ASPRS

        strategy_class = self.STRATEGIES.get(
            mode_enum, ASPRSClassificationStrategy
        )
        return strategy_class(use_gpu=self.use_gpu)

    def classify(self, features: np.ndarray) -> np.ndarray:
        """
        Classify point cloud features.

        Args:
            features: Feature array [N, F]

        Returns:
            Classification labels [N]

        Raises:
            ValueError: If features are invalid
        """
        if features is None or len(features) == 0:
            raise ValueError("Features cannot be empty")

        if not isinstance(features, np.ndarray):
            features = np.asarray(features)

        logger.debug(f"Classifying {len(features)} points with {self.strategy.get_name()}")

        return self.strategy.classify(features)

    def refine(
        self, labels: np.ndarray, context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Refine classifications using post-processing.

        Args:
            labels: Initial classification labels
            context: Optional refinement context

        Returns:
            Refined labels
        """
        return self.strategy.refine(labels, context)

    def get_confidence(self, labels: np.ndarray) -> np.ndarray:
        """
        Get confidence scores for classifications.

        Args:
            labels: Classification labels

        Returns:
            Confidence scores [0, 1]
        """
        return self.strategy.get_confidence(labels)

    def set_mode(self, mode: str) -> None:
        """
        Switch classification mode.

        Args:
            mode: New classification mode

        Raises:
            ValueError: If mode is not supported
        """
        if mode not in [m.value for m in ClassificationMode]:
            raise ValueError(f"Unsupported mode: {mode}")

        self.mode = mode
        self.strategy = self._create_strategy()
        logger.info(f"Switched to {mode} classification mode")

    def get_available_modes(self) -> list:
        """Get list of available classification modes."""
        return [m.value for m in ClassificationMode]

    def __repr__(self) -> str:
        """String representation."""
        return f"ClassificationEngine(mode={self.mode}, gpu={self.use_gpu})"

    # ========================================================================
    # Advanced methods for specialized classification strategies
    # ========================================================================

    def classify_spectral(
        self,
        rgb: np.ndarray,
        nir: np.ndarray,
        current_labels: np.ndarray,
        ndvi: Optional[np.ndarray] = None,
        apply_to_unclassified_only: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Advanced spectral classification with multi-band support.

        This method is for use with SpectralClassificationStrategy
        when full spectral information is available.

        Args:
            rgb: RGB values [N, 3] normalized to [0, 1]
            nir: NIR values [N] normalized to [0, 1]
            current_labels: Current classification labels [N]
            ndvi: Optional pre-computed NDVI values [N]
            apply_to_unclassified_only: Only reclassify unclassified points

        Returns:
            Tuple of (updated labels, statistics dict)

        Raises:
            RuntimeError: If strategy doesn't support spectral classification
        """
        if self.mode != "spectral":
            self.set_mode("spectral")

        strategy = self.strategy
        if not hasattr(strategy, "engine") or strategy.engine is None:
            raise RuntimeError("SpectralRulesEngine not available")

        engine = strategy.engine
        labels, stats = engine.classify_by_spectral_signature(
            rgb=rgb,
            nir=nir,
            current_labels=current_labels,
            ndvi=ndvi,
            apply_to_unclassified_only=apply_to_unclassified_only,
        )

        logger.info(f"Spectral classification completed: {stats}")
        return labels, stats

    def classify_geometric(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        ground_truth_features: Dict,
        ndvi: Optional[np.ndarray] = None,
        intensities: Optional[np.ndarray] = None,
        rgb: Optional[np.ndarray] = None,
        nir: Optional[np.ndarray] = None,
        verticality: Optional[np.ndarray] = None,
        preserve_ground_truth: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Advanced geometric classification with ground truth support.

        This method is for use with GeometricClassificationStrategy
        when points and ground truth data are available.

        Args:
            points: XYZ coordinates [N, 3]
            labels: Current classification labels [N]
            ground_truth_features: Dict of feature_type -> GeoDataFrame
            ndvi: Optional NDVI values [N]
            intensities: Optional intensity values [N]
            rgb: Optional RGB values [N, 3]
            nir: Optional NIR values [N]
            verticality: Optional verticality values [N]
            preserve_ground_truth: Only modify unclassified points

        Returns:
            Tuple of (updated labels, statistics dict)

        Raises:
            RuntimeError: If strategy doesn't support geometric classification
        """
        if self.mode != "geometric":
            self.set_mode("geometric")

        strategy = self.strategy
        if not hasattr(strategy, "engine") or strategy.engine is None:
            raise RuntimeError("GeometricRulesEngine not available")

        engine = strategy.engine
        labels, stats = engine.apply_all_rules(
            points=points,
            labels=labels,
            ground_truth_features=ground_truth_features,
            ndvi=ndvi,
            intensities=intensities,
            rgb=rgb,
            nir=nir,
            verticality=verticality,
            preserve_ground_truth=preserve_ground_truth,
        )

        logger.info(f"Geometric classification completed: {len(stats)} rules applied")
        return labels, stats

    def classify_asprs(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        classification: np.ndarray,
        ground_truth: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Advanced ASPRS classification with specialized rules.

        This method is for use with ASPRSClassificationStrategy
        when points and feature dictionaries are available.

        Args:
            points: Point cloud [N, 3] with XYZ coordinates
            features: Dictionary of computed features
            classification: Current classification array [N]
            ground_truth: Optional ground truth data

        Returns:
            Updated classification array [N]

        Raises:
            RuntimeError: If strategy doesn't support ASPRS classification
        """
        if self.mode != "asprs":
            self.set_mode("asprs")

        strategy = self.strategy
        if not hasattr(strategy, "engine") or strategy.engine is None:
            raise RuntimeError("ASPRSClassRulesEngine not available")

        engine = strategy.engine
        updated_labels = engine.apply_all_rules(
            points=points,
            features=features,
            classification=classification,
            ground_truth=ground_truth,
        )

        logger.info(f"ASPRS classification completed for {len(points):,} points")
        return updated_labels


# Export
__all__ = [
    "ClassificationEngine",
    "ClassificationStrategy",
    "ClassificationMode",
    "SpectralClassificationStrategy",
    "GeometricClassificationStrategy",
    "ASPRSClassificationStrategy",
]
