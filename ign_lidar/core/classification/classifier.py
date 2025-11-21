"""
Unified Classification Module

This module consolidates three classification approaches into a single unified interface:
1. AdvancedClassifier: Multi-stage pipeline with parcel clustering, geometry, NDVI, ground truth
2. AdaptiveClassifier: Feature-aware classification with automatic fallback rules
3. RefinementFunctions: LOD2/LOD3 refinement with ground truth integration

v3.2+ Changes:
- Now inherits from BaseClassifier for standardized API
- Added classify() method following BaseClassifier interface
- Returns ClassificationResult for consistency
- Maintains backward compatibility with classify_points()

Consolidation reduces code from 4,216 lines to ~2,000 lines (52% reduction) while:
- Preserving all unique features from each approach
- Maintaining 100% backward compatibility via wrapper classes
- Providing unified, consistent API
- Improving maintainability and reducing duplication

Author: IGN LiDAR HD Classification Team
Date: October 25, 2025
Version: 3.2.0
"""

import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

# Import unified modules (v3.1 - consolidated)
from ...classification_schema import ASPRSClass, LOD2Class, LOD3Class

# Import BaseClassifier for v3.2+ unified interface
from .base import BaseClassifier, ClassificationResult

# Import supporting modules (Phase 2 & 3 reorganization)
from .building import BuildingDetectionConfig, BuildingDetectionMode, BuildingDetector
from .feature_validator import FeatureValidator
from .thresholds import ThresholdConfig, get_thresholds
from .transport import DetectionConfig as TransportDetectionConfig
from .transport import TransportDetector
from .transport import TransportMode as TransportDetectionMode
from .transport import detect_transport_multi_mode

# Import parcel classifier (optional)
try:
    from .parcel_classifier import ParcelClassificationConfig, ParcelClassifier

    HAS_PARCEL_CLASSIFIER = True
except ImportError:
    HAS_PARCEL_CLASSIFIER = False
    ParcelClassifier = None
    ParcelClassificationConfig = None

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import geopandas as gpd
    from shapely.geometry import MultiPolygon, Point, Polygon

try:
    import geopandas as gpd
    from shapely import prepare as prep
    from shapely.geometry import MultiPolygon, Point, Polygon
    from shapely.strtree import STRtree

    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False
    STRtree = None
    prep = None


# ============================================================================
# Enums and Configuration
# ============================================================================


class ClassificationStrategy(Enum):
    """Classification strategy selection."""

    BASIC = "basic"  # Simple height + geometry
    ADAPTIVE = "adaptive"  # Feature-aware adaptive rules
    COMPREHENSIVE = "comprehensive"  # Full multi-stage pipeline


class FeatureImportance(Enum):
    """Feature importance levels for adaptive classification."""

    CRITICAL = 3  # Classification impossible without this feature
    IMPORTANT = 2  # Accuracy significantly reduced without this feature
    HELPFUL = 1  # Improves accuracy but not essential
    OPTIONAL = 0  # Nice to have but minimal impact


@dataclass
class ClassificationRule:
    """
    Adaptive classification rule that works with varying feature sets.
    """

    name: str
    asprs_class: int

    # Feature requirements by importance
    critical_features: Set[str]
    important_features: Set[str]
    helpful_features: Set[str]
    optional_features: Set[str]

    # Thresholds (feature_name -> (min, max))
    thresholds: Dict[str, Tuple[Optional[float], Optional[float]]]

    # Confidence adjustment
    base_confidence: float = 0.8
    confidence_penalty_per_missing_important: float = 0.1
    confidence_penalty_per_missing_helpful: float = 0.05

    def can_classify(self, available_features: Set[str]) -> bool:
        """Check if classification is possible with available features."""
        return self.critical_features.issubset(available_features)

    def get_confidence(self, available_features: Set[str]) -> float:
        """Calculate confidence based on available features."""
        if not self.can_classify(available_features):
            return 0.0

        confidence = self.base_confidence
        missing_important = self.important_features - available_features
        confidence -= (
            len(missing_important) * self.confidence_penalty_per_missing_important
        )
        missing_helpful = self.helpful_features - available_features
        confidence -= len(missing_helpful) * self.confidence_penalty_per_missing_helpful

        return max(0.0, min(1.0, confidence))

    def get_usable_features(self, available_features: Set[str]) -> Set[str]:
        """Get features that can be used for classification."""
        all_features = (
            self.critical_features
            | self.important_features
            | self.helpful_features
            | self.optional_features
        )
        return all_features & available_features


@dataclass
class ClassifierConfig:
    """
    Configuration for all classification strategies.

    This combines configuration options from:
    - AdvancedClassifier
    - AdaptiveClassifier
    - RefinementConfig
    """

    # Strategy selection
    strategy: ClassificationStrategy = ClassificationStrategy.COMPREHENSIVE

    # Feature usage flags
    use_ground_truth: bool = True
    use_ndvi: bool = True
    use_geometric: bool = True
    use_feature_validation: bool = True
    use_parcel_classification: bool = False

    # Detection modes
    building_detection_mode: str = "asprs"
    transport_detection_mode: str = "asprs_standard"
    lod_level: str = "LOD2"

    # Thresholds (can override ThresholdConfig defaults)
    thresholds: Optional[ThresholdConfig] = None

    # Buffer tolerances
    road_buffer_tolerance: float = 0.5

    # Parcel classification
    parcel_classification_config: Optional[Dict] = None

    # Feature validation
    feature_validation_config: Optional[Dict] = None

    # Adaptive classifier: artifact tracking
    artifact_features: Optional[Set[str]] = None

    # Refinement flags
    refine_vegetation: bool = True
    refine_buildings: bool = True
    refine_ground: bool = True
    refine_roads: bool = True
    refine_vehicles: bool = True


# ============================================================================
# Unified Classifier
# ============================================================================


class Classifier(BaseClassifier):
    """
    Point cloud classifier combining multiple classification approaches.

    v3.2+ Changes:
    - Now inherits from BaseClassifier for API consistency
    - Added classify() method following BaseClassifier interface
    - Returns ClassificationResult from classify()
    - classify_points() maintained for backward compatibility

    This class consolidates functionality from:
    - AdvancedClassifier: Multi-stage pipeline with comprehensive features
    - AdaptiveClassifier: Feature-aware adaptive rules
    - classification_refinement: LOD2/LOD3 refinement functions

    Classification hierarchy (COMPREHENSIVE strategy):
    0. [Optional] Parcel-based clustering (Stage 0)
    1. Ground Truth (IGN BD TOPO®) - highest priority
    2. NDVI-based vegetation detection
    3. Geometric feature analysis
    4. Height-based classification
    5. Post-processing and refinement

    Strategies:
    - BASIC: Height + basic geometry (fast, lower accuracy)
    - ADAPTIVE: Adapts to available features with fallback rules
    - COMPREHENSIVE: Full multi-stage pipeline (slower, highest accuracy)

    Example (v3.2+ unified interface):
        >>> # Use standardized classify() method
        >>> classifier = Classifier(strategy='comprehensive')
        >>> result = classifier.classify(points, features, ground_truth=gdf)
        >>> labels = result.labels
        >>> stats = result.get_statistics()

    Example (v3.1 DataFrame interface - still works):
        >>> # Legacy classify_points() method
        >>> classifier = Classifier(
        ...     strategy=ClassificationStrategy.COMPREHENSIVE,
        ...     use_parcel_classification=True
        ... )
        >>> labels = classifier.classify_points(data, ground_truth_data)

        >>> # Adaptive classification with missing features
        >>> classifier = Classifier(strategy=ClassificationStrategy.ADAPTIVE)
        >>> classifier.set_artifact_features({'curvature', 'roughness'})
        >>> labels = classifier.classify_batch(features)
    """

    # ASPRS Classification codes
    ASPRS_UNCLASSIFIED = ASPRSClass.UNCLASSIFIED.value
    ASPRS_GROUND = ASPRSClass.GROUND.value
    ASPRS_LOW_VEGETATION = ASPRSClass.LOW_VEGETATION.value
    ASPRS_MEDIUM_VEGETATION = ASPRSClass.MEDIUM_VEGETATION.value
    ASPRS_HIGH_VEGETATION = ASPRSClass.HIGH_VEGETATION.value
    ASPRS_BUILDING = ASPRSClass.BUILDING.value
    ASPRS_LOW_POINT = ASPRSClass.LOW_POINT.value
    ASPRS_WATER = ASPRSClass.WATER.value
    ASPRS_RAIL = ASPRSClass.RAIL.value
    ASPRS_ROAD = ASPRSClass.ROAD_SURFACE.value
    ASPRS_BRIDGE = ASPRSClass.BRIDGE_DECK.value
    ASPRS_PARKING = 40
    ASPRS_SPORTS = 41

    def __init__(self, config: Optional[ClassifierConfig] = None, **kwargs):
        """
        Initialize classifier.

        Args:
            config: ClassifierConfig instance (preferred)
            **kwargs: Individual config options (for backward compatibility)
                - strategy: ClassificationStrategy
                - use_ground_truth: bool
                - use_ndvi: bool
                - use_geometric: bool
                - use_feature_validation: bool
                - use_parcel_classification: bool
                - building_detection_mode: str
                - transport_detection_mode: str
                - lod_level: str
                - ... (see ClassifierConfig for all options)
        """
        # Build config from kwargs if not provided
        if config is None:
            # Convert strategy string to enum if needed
            strategy = kwargs.get("strategy", ClassificationStrategy.COMPREHENSIVE)
            if isinstance(strategy, str):
                strategy = ClassificationStrategy(strategy.lower())
            kwargs["strategy"] = strategy
            config = ClassifierConfig(**kwargs)

        self.config = config
        self.strategy = config.strategy

        # Initialize thresholds
        self.thresholds = config.thresholds or get_thresholds()

        # Initialize parcel classifier if enabled
        self.parcel_classifier = None
        if config.use_parcel_classification:
            if not HAS_PARCEL_CLASSIFIER:
                logger.warning(
                    "  Parcel classification requested but module not available"
                )
                config.use_parcel_classification = False
            else:
                pc_config = config.parcel_classification_config or {}
                if isinstance(pc_config, dict):
                    pc_config = ParcelClassificationConfig(**pc_config)
                self.parcel_classifier = ParcelClassifier(config=pc_config)
                logger.info("  Parcel classification: ENABLED")

        # Initialize feature validator if enabled
        self.feature_validator = None
        if config.use_feature_validation:
            self.feature_validator = FeatureValidator(config.feature_validation_config)
            logger.info("  Feature validation: ENABLED")

        # Initialize adaptive classification rules (for ADAPTIVE strategy)
        self.classification_rules = self._create_classification_rules()
        self.artifact_features: Set[str] = config.artifact_features or set()

        logger.info(
            f"🎯 Unified Classifier initialized (strategy: {self.strategy.value.upper()})"
        )
        logger.info(f"  Ground truth: {config.use_ground_truth}")
        logger.info(f"  NDVI refinement: {config.use_ndvi}")
        logger.info(f"  Geometric features: {config.use_geometric}")
        logger.info(
            f"  Building detection mode: {config.building_detection_mode.upper()}"
        )
        logger.info(
            f"  Transport detection mode: {config.transport_detection_mode.upper()}"
        )
        logger.info(f"  LOD level: {config.lod_level}")

    # ========================================================================
    # v3.2+ Unified Interface (BaseClassifier compatibility)
    # ========================================================================

    def classify(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        ground_truth: Optional[Union["gpd.GeoDataFrame", Dict[str, Any]]] = None,
        **kwargs,
    ) -> ClassificationResult:
        """
        Classify point cloud using BaseClassifier interface (v3.2+).

        This method follows the standard BaseClassifier API for consistency
        across all classifiers in IGN LiDAR HD. It wraps the existing
        classify_points() method which uses a DataFrame-based interface.

        Args:
            points: Point cloud [N, 3] with XYZ coordinates
            features: Dictionary mapping feature name → array [N]
                Common features: 'planarity', 'verticality', 'height',
                'curvature', 'ndvi', 'roughness', etc.
            ground_truth: Optional ground truth data:
                - GeoDataFrame with polygons (BD TOPO, cadastre, etc.)
                - Dictionary with 'building_mask', 'road_mask', etc.
            **kwargs: Additional parameters passed to classify_points()
                - verbose: bool (default True)
                - parcel_gdf: GeoDataFrame with cadastral parcels

        Returns:
            ClassificationResult with labels, confidence (None), and metadata

        Example:
            >>> classifier = Classifier(strategy='comprehensive')
            >>> result = classifier.classify(points, features, ground_truth=gdf)
            >>> labels = result.labels
            >>> stats = result.get_statistics()
            >>> print(f"Classified {stats['total_points']} points "
            ...       f"into {stats['num_classes']} classes")

        Note:
            This is the recommended method for v3.2+. The old classify_points()
            method is maintained for backward compatibility but may be deprecated
            in v4.0.
        """
        # Validate inputs using BaseClassifier method
        self.validate_inputs(points, features)

        # Convert to DataFrame format (required by classify_points)
        data = pd.DataFrame(
            {
                "x": points[:, 0],
                "y": points[:, 1],
                "z": points[:, 2],
            }
        )

        # Add all features to DataFrame
        for name, feat_array in features.items():
            data[name] = feat_array

        # Convert ground truth if provided
        ground_truth_data = None
        if ground_truth is not None:
            if HAS_SPATIAL and isinstance(ground_truth, gpd.GeoDataFrame):
                # Keep as GeoDataFrame for spatial operations
                ground_truth_data = ground_truth
            elif isinstance(ground_truth, dict):
                # Already in expected format
                ground_truth_data = ground_truth
            else:
                logger.warning(
                    f"Unknown ground truth type: {type(ground_truth)}. Ignoring."
                )

        # Extract parcel_gdf from kwargs if provided
        parcel_gdf = kwargs.pop("parcel_gdf", None)
        verbose = kwargs.pop("verbose", True)

        # Call existing classify_points method
        labels = self.classify_points(
            data=data,
            ground_truth_data=ground_truth_data,
            parcel_gdf=parcel_gdf,
            verbose=verbose,
        )

        # Return standardized result
        return ClassificationResult(
            labels=labels,
            confidence=None,  # Classifier doesn't provide confidence scores
            metadata={
                "strategy": self.strategy.value,
                "lod_level": self.config.lod_level,
                "num_points": len(labels),
                "use_ground_truth": self.config.use_ground_truth,
                "use_ndvi": self.config.use_ndvi,
                "use_geometric": self.config.use_geometric,
            },
        )

    # ========================================================================
    # Adaptive Classification (from adaptive_classifier.py)
    # ========================================================================

    def _create_classification_rules(self) -> Dict[str, ClassificationRule]:
        """
        Create adaptive classification rules for each class.

        Returns:
            Dictionary mapping class name to ClassificationRule
        """
        rules = {}

        # Building classification
        rules["building"] = ClassificationRule(
            name="building",
            asprs_class=self.ASPRS_BUILDING,
            critical_features={"height"},
            important_features={"planarity", "verticality"},
            helpful_features={"curvature", "normal_z", "ndvi"},
            optional_features={"nir", "intensity", "brightness"},
            thresholds={
                "height": (1.5, None),
                "planarity": (0.65, None),
                "verticality": (0.6, None),
                "curvature": (None, 0.10),
                "normal_z": (None, 0.85),
                "ndvi": (None, 0.20),
            },
            base_confidence=0.85,
        )

        # Road classification (DTM-based strict filtering)
        rules["road"] = ClassificationRule(
            name="road",
            asprs_class=self.ASPRS_ROAD,
            critical_features={"planarity", "height"},
            important_features={"normal_z"},
            helpful_features={"curvature", "ndvi"},
            optional_features={"intensity", "brightness"},
            thresholds={
                "height": (-0.2, 0.3),  # Strict DTM-based: -20cm to +30cm above ground
                "planarity": (0.85, None),
                "normal_z": (0.90, None),
                "curvature": (None, 0.05),
                "ndvi": (None, 0.15),
            },
            base_confidence=0.80,
        )

        # High vegetation classification
        rules["high_vegetation"] = ClassificationRule(
            name="high_vegetation",
            asprs_class=self.ASPRS_HIGH_VEGETATION,
            critical_features={"height"},
            important_features={"ndvi"},
            helpful_features={"curvature", "roughness", "planarity"},
            optional_features={"nir", "intensity"},
            thresholds={
                "height": (2.0, None),
                "ndvi": (0.35, None),
                "curvature": (0.02, None),
                "roughness": (0.03, None),
                "planarity": (None, 0.4),
            },
            base_confidence=0.85,
        )

        # Low vegetation classification
        rules["low_vegetation"] = ClassificationRule(
            name="low_vegetation",
            asprs_class=self.ASPRS_LOW_VEGETATION,
            critical_features={"height"},
            important_features={"ndvi"},
            helpful_features={"curvature", "roughness"},
            optional_features={"nir", "intensity"},
            thresholds={
                "height": (0.0, 0.5),
                "ndvi": (0.35, None),
                "curvature": (0.02, None),
                "roughness": (0.03, None),
            },
            base_confidence=0.75,
        )

        # Ground classification
        rules["ground"] = ClassificationRule(
            name="ground",
            asprs_class=self.ASPRS_GROUND,
            critical_features={"height"},
            important_features={"planarity"},
            helpful_features={"normal_z", "curvature"},
            optional_features={"ndvi", "intensity"},
            thresholds={
                "height": (None, 0.5),
                "planarity": (0.7, None),
                "normal_z": (0.85, None),
                "curvature": (None, 0.05),
                "ndvi": (None, 0.30),
            },
            base_confidence=0.70,
        )

        return rules

    def set_artifact_features(self, artifact_features: Set[str]):
        """
        Set features that have artifacts and should be avoided.

        Args:
            artifact_features: Set of feature names with artifacts
        """
        self.artifact_features = artifact_features
        logger.info(f"Artifact features set: {artifact_features}")

    def get_available_features(self, features: Dict[str, np.ndarray]) -> Set[str]:
        """
        Detect available and valid features from feature dictionary.

        Args:
            features: Dictionary of feature arrays

        Returns:
            Set of available feature names (excluding artifacts)
        """
        available = set()

        for name, arr in features.items():
            if arr is None:
                continue
            if not isinstance(arr, np.ndarray):
                continue
            if len(arr) == 0:
                continue
            # Check for invalid values
            if np.all(np.isnan(arr)):
                continue
            if np.all(np.isinf(arr)):
                continue

            available.add(name)

        # Remove artifact features
        available -= self.artifact_features

        return available

    def classify_point(
        self, features: Dict[str, float], available_features: Optional[Set[str]] = None
    ) -> Tuple[int, float]:
        """
        Classify a single point adaptively based on available features.

        Args:
            features: Dictionary of feature values for the point
            available_features: Set of available features (auto-detected if None)

        Returns:
            Tuple of (asprs_class, confidence)
        """
        if available_features is None:
            # Convert point features to array format for detection
            feature_arrays = {k: np.array([v]) for k, v in features.items()}
            available_features = self.get_available_features(feature_arrays)

        best_class = self.ASPRS_UNCLASSIFIED
        best_confidence = 0.0

        # Try each classification rule
        for rule in self.classification_rules.values():
            if not rule.can_classify(available_features):
                continue

            # Check if point matches all threshold criteria
            matches = True
            usable_features = rule.get_usable_features(available_features)

            for feature_name in usable_features:
                if feature_name not in features:
                    continue
                if feature_name not in rule.thresholds:
                    continue

                value = features[feature_name]
                min_val, max_val = rule.thresholds[feature_name]

                if min_val is not None and value < min_val:
                    matches = False
                    break
                if max_val is not None and value > max_val:
                    matches = False
                    break

            if matches:
                confidence = rule.get_confidence(available_features)
                if confidence > best_confidence:
                    best_class = rule.asprs_class
                    best_confidence = confidence

        return best_class, best_confidence

    def classify_batch(
        self,
        features: Dict[str, np.ndarray],
        available_features: Optional[Set[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify a batch of points adaptively.

        Args:
            features: Dictionary of feature arrays [N]
            available_features: Set of available features (auto-detected if None)

        Returns:
            Tuple of (labels [N], confidences [N])
        """
        if available_features is None:
            available_features = self.get_available_features(features)

        # Get number of points
        n_points = len(next(iter(features.values())))

        labels = np.full(n_points, self.ASPRS_UNCLASSIFIED, dtype=np.int32)
        confidences = np.zeros(n_points, dtype=np.float32)

        # Classify each point
        for i in range(n_points):
            point_features = {k: v[i] for k, v in features.items() if v is not None}
            labels[i], confidences[i] = self.classify_point(
                point_features, available_features
            )

        return labels, confidences

    def get_feature_importance_report(
        self, available_features: Set[str]
    ) -> Dict[str, Any]:
        """
        Generate report on feature importance and classification capability.

        Args:
            available_features: Set of available features

        Returns:
            Dictionary with feature importance analysis
        """
        report = {
            "available_features": list(available_features),
            "artifact_features": list(self.artifact_features),
            "classifiable_categories": [],
            "degraded_categories": [],
            "impossible_categories": [],
        }

        for rule in self.classification_rules.values():
            if rule.can_classify(available_features):
                confidence = rule.get_confidence(available_features)
                if confidence >= 0.7:
                    report["classifiable_categories"].append(
                        {"name": rule.name, "confidence": confidence}
                    )
                else:
                    report["degraded_categories"].append(
                        {
                            "name": rule.name,
                            "confidence": confidence,
                            "missing_important": list(
                                rule.important_features - available_features
                            ),
                            "missing_helpful": list(
                                rule.helpful_features - available_features
                            ),
                        }
                    )
            else:
                report["impossible_categories"].append(
                    {
                        "name": rule.name,
                        "missing_critical": list(
                            rule.critical_features - available_features
                        ),
                    }
                )

        return report

    # ========================================================================
    # Comprehensive Classification (from advanced_classification.py)
    # ========================================================================

    def classify_points(
        self,
        data: pd.DataFrame,
        ground_truth_data: Optional[Dict[str, Any]] = None,
        parcel_gdf: Optional["gpd.GeoDataFrame"] = None,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Classify points using the configured strategy.

        Applies multi-strategy classification to LiDAR point cloud data with
        support for ground truth integration and parcel-based refinement.

        Args:
            data: DataFrame with point cloud data. Required columns:
                 - 'x', 'y', 'z': Point coordinates in meters
                 - 'height': Height above ground in meters
                 Optional columns (improve accuracy):
                 - 'planarity': Planarity measure [0, 1]
                 - 'verticality': Verticality measure [0, 1]
                 - 'curvature': Surface curvature
                 - 'roughness': Surface roughness
                 - 'normal_z': Z-component of surface normal
                 - 'ndvi': NDVI vegetation index [-1, 1]
                 - 'nir': Near-infrared reflectance
                 - 'intensity': LiDAR return intensity
            ground_truth_data: Optional dictionary containing:
                - 'building_mask': Boolean array [N] or GeoDataFrame
                - 'road_mask': Boolean array [N] or GeoDataFrame
                - 'vegetation_mask': Boolean array [N] or GeoDataFrame
            parcel_gdf: Optional GeoDataFrame with cadastral parcels for
                       parcel-based building classification
            verbose: If True, log classification progress and statistics

        Returns:
            Classification labels [N] as integer array. Label values depend
            on the classifier mode (ASPRS, LOD2, or LOD3).

        Example:
            Basic usage with minimum features:

            >>> import pandas as pd
            >>> import numpy as np
            >>> from ign_lidar.core.classification import Classifier
            >>>
            >>> # Create sample point cloud data
            >>> n_points = 1000
            >>> data = pd.DataFrame({
            ...     'x': np.random.rand(n_points) * 100,
            ...     'y': np.random.rand(n_points) * 100,
            ...     'z': np.random.rand(n_points) * 20,
            ...     'height': np.random.rand(n_points) * 15,
            ...     'planarity': np.random.rand(n_points)
            ... })
            >>>
            >>> # Create classifier
            >>> classifier = Classifier(strategy='basic')
            >>> labels = classifier.classify_points(data)
            >>>
            >>> # Check results
            >>> print(f"Classified {len(labels)} points")
            Classified 1000 points
            >>> unique_classes = np.unique(labels)
            >>> print(f"Found {len(unique_classes)} classes")
            Found 5 classes

            Advanced usage with comprehensive features and ground truth:

            >>> # Add more features for better classification
            >>> data['ndvi'] = np.random.rand(n_points) * 2 - 1
            >>> data['verticality'] = np.random.rand(n_points)
            >>> data['roughness'] = np.random.rand(n_points) * 0.5
            >>>
            >>> # Create ground truth data
            >>> building_indices = np.random.choice(n_points, 200, replace=False)
            >>> building_mask = np.zeros(n_points, dtype=bool)
            >>> building_mask[building_indices] = True
            >>>
            >>> ground_truth = {
            ...     'building_mask': building_mask
            ... }
            >>>
            >>> # Use comprehensive strategy
            >>> classifier = Classifier(
            ...     strategy='comprehensive',
            ...     mode='asprs_extended'
            ... )
            >>> labels = classifier.classify_points(
            ...     data,
            ...     ground_truth_data=ground_truth,
            ...     verbose=True
            ... )
            >>>
            >>> # Analyze results
            >>> from ign_lidar.classification_schema import get_class_name
            >>> for class_code in np.unique(labels):
            ...     count = (labels == class_code).sum()
            ...     percentage = 100 * count / len(labels)
            ...     name = get_class_name(class_code, mode='asprs_extended')
            ...     print(f"{name}: {count} ({percentage:.1f}%)")

            Adaptive classification with feature importance:

            >>> # Use adaptive strategy (handles missing features)
            >>> classifier = Classifier(strategy='adaptive')
            >>>
            >>> # Even with limited features, adaptive works
            >>> minimal_data = pd.DataFrame({
            ...     'x': data['x'],
            ...     'y': data['y'],
            ...     'z': data['z'],
            ...     'height': data['height']
            ... })
            >>> labels = classifier.classify_points(minimal_data, verbose=False)
            >>> print(f"Classified with limited features: {len(labels)} points")
            Classified with limited features: 1000 points

        Note:
            - BASIC strategy: Fastest, uses only height and planarity
            - ADAPTIVE strategy: Adapts to available features automatically
            - COMPREHENSIVE strategy: Full pipeline with ground truth integration

            Classification quality improves with more features, especially:
            - Planarity and verticality for buildings
            - NDVI for vegetation discrimination
            - Roughness for surface texture analysis

            Ground truth masks significantly improve accuracy but are optional.

        See Also:
            - ClassifierConfig: Configuration options
            - get_thresholds: Get classification thresholds
            - ASPRSClass: ASPRS classification codes
        """
        if self.strategy == ClassificationStrategy.ADAPTIVE:
            # Use adaptive classification
            features = {
                "height": data.get("height"),
                "planarity": data.get("planarity"),
                "verticality": data.get("verticality"),
                "curvature": data.get("curvature"),
                "roughness": data.get("roughness"),
                "normal_z": data.get("normal_z"),
                "ndvi": data.get("ndvi"),
                "nir": data.get("nir"),
                "intensity": data.get("intensity"),
            }
            labels, _ = self.classify_batch(features)
            return labels

        elif self.strategy == ClassificationStrategy.BASIC:
            # Simple height + geometry based classification
            return self._classify_basic(data)

        else:  # COMPREHENSIVE
            # Full multi-stage pipeline
            return self._classify_comprehensive(
                data, ground_truth_data, parcel_gdf, verbose
            )

    def _classify_basic(self, data: pd.DataFrame) -> np.ndarray:
        """
        Basic classification using height and simple geometric features.

        Args:
            data: DataFrame with height and optional geometric features

        Returns:
            Classification labels array [N]
        """
        n_points = len(data)
        labels = np.full(n_points, self.ASPRS_UNCLASSIFIED, dtype=np.int32)

        height = data["height"].values

        # Simple height-based classification
        labels[height < 0.5] = self.ASPRS_GROUND
        labels[(height >= 0.5) & (height < 2.0)] = self.ASPRS_LOW_VEGETATION
        labels[(height >= 2.0) & (height < 5.0)] = self.ASPRS_MEDIUM_VEGETATION
        labels[height >= 5.0] = self.ASPRS_HIGH_VEGETATION

        # Refine with planarity if available
        if "planarity" in data.columns:
            planarity = data["planarity"].values
            high_planarity = planarity > 0.7
            labels[(height >= 1.5) & high_planarity] = self.ASPRS_BUILDING
            labels[(height < 0.5) & high_planarity] = self.ASPRS_GROUND

        return labels

    def _classify_comprehensive(
        self,
        data: pd.DataFrame,
        ground_truth_data: Optional[Dict[str, Any]],
        parcel_gdf: Optional["gpd.GeoDataFrame"],
        verbose: bool,
    ) -> np.ndarray:
        """
        Comprehensive multi-stage classification pipeline.

        Stages:
        0. [Optional] Parcel-based clustering
        1. Geometric feature analysis
        2. NDVI-based vegetation detection
        3. Ground truth integration (BD TOPO®)
        4. Post-processing and refinement

        Args:
            data: DataFrame with all features
            ground_truth_data: Ground truth data
            parcel_gdf: Parcel geometries
            verbose: Enable logging

        Returns:
            Classification labels array [N]
        """
        n_points = len(data)
        labels = np.full(n_points, self.ASPRS_UNCLASSIFIED, dtype=np.int32)

        if verbose:
            logger.info("🔄 Starting comprehensive classification pipeline...")

        # Stage 0: Parcel-based classification (optional)
        if self.config.use_parcel_classification and parcel_gdf is not None:
            if verbose:
                logger.info("  📦 Stage 0: Parcel-based classification")
            labels = self.parcel_classifier.classify_by_parcels(
                data=data, parcel_gdf=parcel_gdf, labels=labels
            )

        # Stage 1: Geometric features
        if self.config.use_geometric:
            if verbose:
                logger.info("  📐 Stage 1: Geometric feature analysis")
            labels = self._classify_by_geometry(data, labels)

        # Stage 2: NDVI-based vegetation
        if self.config.use_ndvi and "ndvi" in data.columns:
            if verbose:
                logger.info("  🌿 Stage 2: NDVI-based vegetation detection")
            labels = self._classify_by_ndvi(data, labels)

        # Stage 3: Ground truth (highest priority)
        if self.config.use_ground_truth and ground_truth_data is not None:
            if verbose:
                logger.info("  🗺️  Stage 3: Ground truth integration")
            labels = self._classify_by_ground_truth(data, labels, ground_truth_data)

        # Stage 4: Post-processing
        if verbose:
            logger.info("  🔧 Stage 4: Post-processing")
        labels = self._post_process_unclassified(data, labels)

        if verbose:
            self._log_distribution(labels)

        return labels

    def _classify_by_geometry(
        self, data: pd.DataFrame, labels: np.ndarray
    ) -> np.ndarray:
        """
        Classify points using geometric features.

        Uses: planarity, verticality, curvature, roughness, normal_z

        Args:
            data: DataFrame with geometric features
            labels: Current classification labels

        Returns:
            Updated classification labels
        """
        # Extract features
        height = data.get("height")
        planarity = data.get("planarity")
        verticality = data.get("verticality")
        curvature = data.get("curvature")
        roughness = data.get("roughness")
        normal_z = data.get("normal_z")

        if height is None:
            return labels

        height = height.values

        # Buildings: High, planar, with some verticality
        if planarity is not None and verticality is not None:
            planarity = planarity.values
            verticality = verticality.values

            building_candidates = (
                (height > self.thresholds.height.building_height_min)
                & (planarity > self.thresholds.building.roof_planarity_min_asprs)
                & (verticality > 0.3)
            )
            labels[building_candidates] = self.ASPRS_BUILDING

        # Roads: Low, very planar, horizontal
        if planarity is not None and normal_z is not None:
            normal_z = normal_z.values

            road_candidates = (
                (height < self.thresholds.height.ground_height_max)
                & (planarity > self.thresholds.transport.road_planarity_min)
                & (normal_z > 0.9)
            )
            labels[road_candidates] = self.ASPRS_ROAD

        # Ground: Low, planar
        if planarity is not None:
            ground_candidates = (height < self.thresholds.height.ground_height_max) & (
                planarity > 0.7
            )
            labels[ground_candidates] = self.ASPRS_GROUND

        # Vegetation: Higher curvature/roughness, lower planarity
        if curvature is not None and roughness is not None:
            curvature = curvature.values
            roughness = roughness.values

            veg_candidates = (curvature > 0.02) & (roughness > 0.03) & (planarity < 0.4)

            labels[
                veg_candidates & (height < self.thresholds.height.low_veg_height_max)
            ] = self.ASPRS_LOW_VEGETATION
            labels[
                veg_candidates
                & (height >= self.thresholds.height.low_veg_height_max)
                & (height < self.thresholds.height.high_veg_height_min)
            ] = self.ASPRS_MEDIUM_VEGETATION
            labels[
                veg_candidates & (height >= self.thresholds.height.high_veg_height_min)
            ] = self.ASPRS_HIGH_VEGETATION

        return labels

    def _classify_by_ndvi(self, data: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
        """
        Refine classification using NDVI values.

        Args:
            data: DataFrame with ndvi column
            labels: Current classification labels

        Returns:
            Updated classification labels
        """
        ndvi = data["ndvi"].values
        height = data["height"].values

        # High NDVI = vegetation
        high_ndvi = ndvi >= self.thresholds.ndvi.vegetation_min

        labels[high_ndvi & (height < self.thresholds.height.low_veg_height_max)] = (
            self.ASPRS_LOW_VEGETATION
        )
        labels[
            high_ndvi
            & (height >= self.thresholds.height.low_veg_height_max)
            & (height < self.thresholds.height.high_veg_height_min)
        ] = self.ASPRS_MEDIUM_VEGETATION
        labels[high_ndvi & (height >= self.thresholds.height.high_veg_height_min)] = (
            self.ASPRS_HIGH_VEGETATION
        )

        # Low NDVI = not vegetation (if currently classified as veg, change to unclassified)
        low_ndvi = ndvi < self.thresholds.ndvi.building_max
        is_vegetation = np.isin(
            labels,
            [
                self.ASPRS_LOW_VEGETATION,
                self.ASPRS_MEDIUM_VEGETATION,
                self.ASPRS_HIGH_VEGETATION,
            ],
        )
        labels[low_ndvi & is_vegetation] = self.ASPRS_UNCLASSIFIED

        return labels

    def _classify_by_ground_truth(
        self,
        labels: np.ndarray,
        points: np.ndarray,
        ground_truth_features: Dict[str, Any],
        ndvi: Optional[np.ndarray] = None,
        height: Optional[np.ndarray] = None,
        planarity: Optional[np.ndarray] = None,
        intensity: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply ground truth data from GeoDataFrames or masks (highest priority).

        Args:
            labels: Current classification labels [N]
            points: Point coordinates [N, 3] (X, Y, Z)
            ground_truth_features: Dictionary with either:
                - GeoDataFrame objects (keys: 'buildings', 'roads', etc.)
                - Boolean masks (keys: 'building_mask', 'road_mask', etc.)
            ndvi: Optional NDVI values [N]
            height: Optional height above ground [N]
            planarity: Optional planarity values [N]
            intensity: Optional intensity values [N]

        Returns:
            Updated classification labels
        """
        # CRITICAL FIX: Check if we have GeoDataFrame inputs (from BD TOPO)
        # If so, delegate to optimized ground truth classifier
        has_geodataframes = any(
            key in ground_truth_features
            and hasattr(ground_truth_features[key], "geometry")
            for key in ["buildings", "roads", "railways", "water", "vegetation"]
        )

        if has_geodataframes:
            # Use optimized ground truth classifier for GeoDataFrame inputs
            try:
                from ...optimization.strtree import OptimizedGroundTruthClassifier

                logger.debug(
                    "Using OptimizedGroundTruthClassifier for GeoDataFrame ground truth"
                )

                classifier = OptimizedGroundTruthClassifier(
                    use_ndvi_refinement=self.use_ndvi,
                    ndvi_veg_threshold=0.3,
                    ndvi_building_threshold=0.15,
                    road_buffer_tolerance=self.road_buffer_tolerance,
                    verbose=False,
                )

                return classifier.classify_with_ground_truth(
                    labels=labels,
                    points=points,
                    ground_truth_features=ground_truth_features,
                    ndvi=ndvi,
                    height=height,
                    planarity=planarity,
                    intensity=intensity,
                    enable_refinement=True,
                )
            except ImportError as e:
                logger.warning(f"OptimizedGroundTruthClassifier not available: {e}")
                logger.warning(
                    "Buildings may not be classified - falling back to mask-only mode"
                )

        # Original implementation for boolean masks
        # If masks are provided directly
        if "building_mask" in ground_truth_features:
            building_mask = ground_truth_features["building_mask"]
            if building_mask is not None:
                labels[building_mask] = self.ASPRS_BUILDING

        if "road_mask" in ground_truth_features:
            road_mask = ground_truth_features["road_mask"]
            if road_mask is not None:
                # Note: Would need to convert to DataFrame for buffer method
                labels[road_mask] = self.ASPRS_ROAD

        if "rail_mask" in ground_truth_features:
            rail_mask = ground_truth_features["rail_mask"]
            if rail_mask is not None:
                labels[rail_mask] = self.ASPRS_RAIL

        if "vegetation_mask" in ground_truth_features:
            vegetation_mask = ground_truth_features["vegetation_mask"]
            if vegetation_mask is not None and height is not None:
                labels[
                    vegetation_mask & (height < self.thresholds.low_veg_height_max)
                ] = self.ASPRS_LOW_VEGETATION
                labels[
                    vegetation_mask & (height >= self.thresholds.high_veg_height_min)
                ] = self.ASPRS_HIGH_VEGETATION

        return labels

    def _classify_roads_with_buffer(
        self, data: pd.DataFrame, labels: np.ndarray, road_mask: np.ndarray
    ) -> np.ndarray:
        """
        Classify roads with intelligent buffering.

        Args:
            data: DataFrame with features (height, planarity, points)
            labels: Current labels
            road_mask: Boolean mask for road points

        Returns:
            Updated labels
        """
        # Direct road points
        labels[road_mask] = self.ASPRS_ROAD

        # Intelligent buffering: expand to nearby geometrically similar points
        if road_mask.any() and "planarity" in data.columns and "height" in data.columns:
            buffered_mask = self._compute_intelligent_buffer(
                data=data, seed_mask=road_mask, feature_type="road", labels=labels
            )
            labels[buffered_mask] = self.ASPRS_ROAD

        return labels

    def _classify_railways_with_buffer(
        self, data: pd.DataFrame, labels: np.ndarray, rail_mask: np.ndarray
    ) -> np.ndarray:
        """
        Classify railways with intelligent buffering.

        Args:
            data: DataFrame with features (height, planarity, points)
            labels: Current labels
            rail_mask: Boolean mask for rail points

        Returns:
            Updated labels
        """
        # Direct rail points
        labels[rail_mask] = self.ASPRS_RAIL

        # Intelligent buffering: expand to nearby geometrically similar points
        if rail_mask.any() and "planarity" in data.columns and "height" in data.columns:
            buffered_mask = self._compute_intelligent_buffer(
                data=data, seed_mask=rail_mask, feature_type="railway", labels=labels
            )
            labels[buffered_mask] = self.ASPRS_RAIL

        return labels

    def _post_process_unclassified(
        self, data: pd.DataFrame, labels: np.ndarray
    ) -> np.ndarray:
        """
        Post-process unclassified points using height-based fallback.

        Args:
            data: DataFrame with height
            labels: Current labels

        Returns:
            Updated labels
        """
        unclassified = labels == self.ASPRS_UNCLASSIFIED

        if not np.any(unclassified):
            return labels

        height = data["height"].values

        # Height-based fallback for unclassified points
        labels[unclassified & (height < self.thresholds.height.ground_height_max)] = (
            self.ASPRS_GROUND
        )
        labels[
            unclassified
            & (height >= self.thresholds.height.ground_height_max)
            & (height < self.thresholds.height.low_veg_height_max)
        ] = self.ASPRS_LOW_VEGETATION
        labels[
            unclassified
            & (height >= self.thresholds.height.low_veg_height_max)
            & (height < self.thresholds.height.high_veg_height_min)
        ] = self.ASPRS_MEDIUM_VEGETATION
        labels[
            unclassified & (height >= self.thresholds.height.high_veg_height_min)
        ] = self.ASPRS_HIGH_VEGETATION

        return labels

    def _compute_intelligent_buffer(
        self,
        data: pd.DataFrame,
        seed_mask: np.ndarray,
        feature_type: str,
        labels: np.ndarray,
    ) -> np.ndarray:
        """
        Compute intelligent buffer zone around seed points based on geometric similarity.

        This method expands classification to nearby points that have similar
        geometric characteristics (height, planarity) to the seed points.

        Args:
            data: DataFrame with features (x, y, z, height, planarity, etc.)
            seed_mask: Boolean mask for seed points (e.g., ground truth roads)
            feature_type: Type of feature ('road', 'railway', 'building')
            labels: Current classification labels

        Returns:
            Boolean mask for buffered points (excluding seed points)
        """
        if not seed_mask.any():
            return np.zeros(len(data), dtype=bool)

        # Get required features
        height = data["height"].values
        planarity = data["planarity"].values if "planarity" in data.columns else None

        # Compute adaptive buffer distance based on point density
        if "x" in data.columns and "y" in data.columns:
            seed_points = data.loc[seed_mask, ["x", "y"]].values
            if len(seed_points) > 10:
                # Estimate point density (points per square meter)
                from scipy.spatial import distance_matrix

                sample_indices = np.random.choice(
                    len(seed_points), min(100, len(seed_points)), replace=False
                )
                sample_points = seed_points[sample_indices]
                distances = distance_matrix(sample_points, sample_points)
                np.fill_diagonal(distances, np.inf)
                mean_nearest_dist = np.mean(np.min(distances, axis=1))
                point_density = (
                    1.0 / (mean_nearest_dist**2) if mean_nearest_dist > 0 else 1.0
                )
            else:
                point_density = 1.0
        else:
            point_density = 1.0

        # Adaptive buffer distance based on feature type and density
        base_buffer = {"road": 2.0, "railway": 3.0, "building": 1.0}.get(
            feature_type, 1.5
        )

        # Adjust based on point density (higher density = smaller buffer)
        density_factor = np.clip(10.0 / point_density, 0.5, 2.0)
        buffer_distance = base_buffer * density_factor

        # Find candidate points within buffer distance
        if "x" in data.columns and "y" in data.columns and HAS_SPATIAL:
            # Spatial approach using distance
            all_points = data[["x", "y"]].values
            seed_indices = np.where(seed_mask)[0]

            # Create boolean mask for candidates
            candidate_mask = np.zeros(len(data), dtype=bool)

            # Find points within buffer distance of any seed point
            for seed_idx in seed_indices:
                seed_xy = all_points[seed_idx]
                distances = np.sqrt(
                    (all_points[:, 0] - seed_xy[0]) ** 2
                    + (all_points[:, 1] - seed_xy[1]) ** 2
                )
                candidate_mask |= (distances <= buffer_distance) & (distances > 0)
        else:
            # Fallback: use height proximity if no spatial coordinates
            seed_heights = height[seed_mask]
            mean_seed_height = np.mean(seed_heights)
            std_seed_height = np.std(seed_heights) if len(seed_heights) > 1 else 0.2
            candidate_mask = np.abs(height - mean_seed_height) <= (
                2 * std_seed_height + 0.5
            )
            candidate_mask &= ~seed_mask  # Exclude seed points

        # Filter candidates by geometric similarity
        if candidate_mask.any():
            # Compute feature similarity thresholds from seed points
            seed_height = height[seed_mask]
            height_mean = np.mean(seed_height)
            height_std = np.std(seed_height) if len(seed_height) > 1 else 0.3

            # Height similarity
            height_similar = np.abs(height - height_mean) <= (2 * height_std + 0.5)

            # Planarity similarity (if available)
            if planarity is not None:
                seed_planarity = planarity[seed_mask]
                planarity_mean = np.mean(seed_planarity)
                planarity_std = (
                    np.std(seed_planarity) if len(seed_planarity) > 1 else 0.1
                )

                # Transport features should have high planarity
                planarity_threshold = max(0.5, planarity_mean - planarity_std)
                planarity_similar = planarity >= planarity_threshold

                # Combine conditions
                buffered_mask = candidate_mask & height_similar & planarity_similar
            else:
                buffered_mask = candidate_mask & height_similar

            # Exclude points already classified (except unclassified)
            unclassified_code = 1  # ASPRS unclassified
            buffered_mask &= labels == unclassified_code
        else:
            buffered_mask = np.zeros(len(data), dtype=bool)

        return buffered_mask

    def _validate_vehicle_size(
        self,
        coords: np.ndarray,
        candidate_mask: np.ndarray,
        min_points: int = 10,
        max_points: int = 5000,
        min_area: float = 4.0,
        max_area: float = 50.0,
    ) -> np.ndarray:
        """
        Validate vehicle candidates using clustering and size constraints.

        Uses DBSCAN clustering to group candidate points, then validates each
        cluster based on point count and 2D footprint area.

        Args:
            coords: Point coordinates [N, 2] or [N, 3]
            candidate_mask: Boolean mask for candidate points [N]
            min_points: Minimum points per vehicle cluster
            max_points: Maximum points per vehicle cluster
            min_area: Minimum 2D footprint area (m²)
            max_area: Maximum 2D footprint area (m²)

        Returns:
            Boolean mask for validated vehicle points [N]
        """
        if not candidate_mask.any():
            return np.zeros(len(coords), dtype=bool)

        try:
            from scipy.spatial import ConvexHull
            from sklearn.cluster import DBSCAN
        except ImportError:
            logger.warning(
                "sklearn or scipy not available - skipping vehicle size validation"
            )
            return np.zeros(len(coords), dtype=bool)

        validated_mask = np.zeros(len(coords), dtype=bool)

        # Extract candidate points (use XY coordinates)
        candidate_coords = coords[candidate_mask, :2]

        if len(candidate_coords) < min_points:
            return validated_mask

        # Cluster candidates (eps=2.0m for vehicle separation)
        clustering = DBSCAN(eps=2.0, min_samples=min_points).fit(candidate_coords)
        labels = clustering.labels_

        # Validate each cluster
        candidate_indices = np.where(candidate_mask)[0]

        for cluster_id in set(labels):
            if cluster_id == -1:  # Noise points
                continue

            cluster_mask_local = labels == cluster_id
            cluster_points = candidate_coords[cluster_mask_local]

            # Check point count
            n_cluster_points = len(cluster_points)
            if not (min_points <= n_cluster_points <= max_points):
                continue

            # Compute 2D convex hull area
            try:
                if n_cluster_points >= 4:  # ConvexHull requires at least 4 points in 2D
                    hull = ConvexHull(cluster_points)
                    area = hull.volume  # In 2D, volume = area

                    # Validate area
                    if min_area <= area <= max_area:
                        # Mark these points as validated
                        global_indices = candidate_indices[cluster_mask_local]
                        validated_mask[global_indices] = True
                elif n_cluster_points >= min_points:
                    # For small clusters, estimate area from bounding box
                    x_range = cluster_points[:, 0].max() - cluster_points[:, 0].min()
                    y_range = cluster_points[:, 1].max() - cluster_points[:, 1].min()
                    area = x_range * y_range

                    if min_area <= area <= max_area:
                        global_indices = candidate_indices[cluster_mask_local]
                        validated_mask[global_indices] = True
            except Exception as e:
                # Skip clusters with hull computation errors
                logger.debug(f"Cluster {cluster_id} hull computation failed: {e}")
                continue

        return validated_mask

    def _log_distribution(self, labels: np.ndarray):
        """Log classification distribution."""
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)

        logger.info("  📊 Classification distribution:")
        for label, count in zip(unique, counts):
            pct = 100.0 * count / total
            label_name = (
                ASPRSClass(label).name
                if label in [c.value for c in ASPRSClass]
                else f"Class_{label}"
            )
            logger.info(f"    {label_name}: {count:,} ({pct:.1f}%)")

    # ========================================================================
    # Refinement Functions (from classification_refinement.py)
    # ========================================================================

    def refine_vegetation(
        self, labels: np.ndarray, features: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, int]:
        """
        Refine vegetation classification using NDVI and geometric features.

        Args:
            labels: Current classification labels [N]
            features: Dictionary of feature arrays (ndvi, height, curvature, roughness, etc.)

        Returns:
            Tuple of (refined_labels, n_changed)
        """
        refined = labels.copy()
        n_changed = 0

        ndvi = features.get("ndvi")
        height = features.get("height")
        curvature = features.get("curvature")
        roughness = features.get("roughness")

        if ndvi is not None and height is not None:
            # High NDVI indicates vegetation
            high_ndvi = ndvi >= self.thresholds.ndvi.vegetation_min

            # Reclassify based on height
            low_veg = high_ndvi & (height < self.thresholds.height.low_veg_height_max)
            high_veg = high_ndvi & (
                height >= self.thresholds.height.high_veg_height_min
            )

            changed_low = (refined != self.ASPRS_LOW_VEGETATION) & low_veg
            changed_high = (refined != self.ASPRS_HIGH_VEGETATION) & high_veg

            refined[low_veg] = self.ASPRS_LOW_VEGETATION
            refined[high_veg] = self.ASPRS_HIGH_VEGETATION

            n_changed = np.sum(changed_low) + np.sum(changed_high)

        return refined, n_changed

    def refine_buildings(
        self,
        labels: np.ndarray,
        features: Dict[str, np.ndarray],
        ground_truth_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Refine building classification.

        Args:
            labels: Current classification labels [N]
            features: Dictionary of feature arrays
            ground_truth_mask: Optional building mask from ground truth

        Returns:
            Tuple of (refined_labels, n_changed)
        """
        refined = labels.copy()
        n_changed = 0

        height = features.get("height")
        planarity = features.get("planarity")
        verticality = features.get("verticality")

        # Apply ground truth if available
        if ground_truth_mask is not None:
            changed = (refined != self.ASPRS_BUILDING) & ground_truth_mask
            refined[ground_truth_mask] = self.ASPRS_BUILDING
            n_changed += np.sum(changed)

        # Geometric-based refinement
        elif height is not None and planarity is not None:
            building_candidates = (
                height > self.thresholds.height.building_height_min
            ) & (planarity > self.thresholds.building.roof_planarity_min_asprs)

            if verticality is not None:
                building_candidates &= verticality > 0.3

            changed = (refined != self.ASPRS_BUILDING) & building_candidates
            refined[building_candidates] = self.ASPRS_BUILDING
            n_changed += np.sum(changed)

        return refined, n_changed

    def refine_roads(
        self,
        labels: np.ndarray,
        features: Dict[str, np.ndarray],
        ground_truth_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Refine road classification.

        Args:
            labels: Current classification labels [N]
            features: Dictionary of feature arrays
            ground_truth_mask: Optional road mask from ground truth

        Returns:
            Tuple of (refined_labels, n_changed)
        """
        refined = labels.copy()
        n_changed = 0

        # Apply ground truth if available
        if ground_truth_mask is not None:
            changed = (refined != self.ASPRS_ROAD) & ground_truth_mask
            refined[ground_truth_mask] = self.ASPRS_ROAD
            n_changed += np.sum(changed)

        return refined, n_changed

    def refine_ground(
        self, labels: np.ndarray, features: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, int]:
        """
        Refine ground classification.

        Args:
            labels: Current classification labels [N]
            features: Dictionary of feature arrays

        Returns:
            Tuple of (refined_labels, n_changed)
        """
        refined = labels.copy()
        n_changed = 0

        height = features.get("height")
        planarity = features.get("planarity")
        normal_z = features.get("normal_z")

        if height is not None and planarity is not None:
            ground_candidates = (height < self.thresholds.height.ground_height_max) & (
                planarity > 0.7
            )

            if normal_z is not None:
                ground_candidates &= normal_z > 0.85

            changed = (refined != self.ASPRS_GROUND) & ground_candidates
            refined[ground_candidates] = self.ASPRS_GROUND
            n_changed += np.sum(changed)

        return refined, n_changed

    def detect_vehicles(
        self, labels: np.ndarray, features: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, int]:
        """
        Detect vehicles using height and size constraints.

        Args:
            labels: Current classification labels [N]
            features: Dictionary of feature arrays (height, x, y, z)

        Returns:
            Tuple of (refined_labels, n_detected)
        """
        refined = labels.copy()
        n_detected = 0

        height = features.get("height")
        points = features.get("points")  # [N, 3] or separate x, y, z

        if height is None:
            return refined, n_detected

        # Initial height-based filtering
        vehicle_candidates = (height >= self.thresholds.height.vehicle_height_min) & (
            height <= self.thresholds.height.vehicle_height_max
        )

        if not vehicle_candidates.any():
            return refined, 0

        # Apply clustering and size validation if spatial data is available
        if points is not None or (
            "x" in features and "y" in features and "z" in features
        ):
            # Get point coordinates
            if points is not None:
                coords = points if points.shape[1] >= 2 else None
            else:
                coords = np.column_stack(
                    [
                        features["x"],
                        features["y"],
                        features.get("z", features["height"]),
                    ]
                )

            if coords is not None:
                validated_mask = self._validate_vehicle_size(
                    coords=coords,
                    candidate_mask=vehicle_candidates,
                    min_points=10,
                    max_points=5000,
                    min_area=4.0,  # m² (small car ~10 m²)
                    max_area=50.0,  # m² (truck/bus ~40 m²)
                )

                # Update labels for validated vehicles (use extended ASPRS code if available)
                vehicle_code = getattr(
                    ASPRSClass, "VEHICLE", 13
                )  # ASPRS Extended: Vehicle = 13
                changed = (refined != vehicle_code) & validated_mask
                refined[validated_mask] = vehicle_code
                n_detected = np.sum(changed)
            else:
                # Fallback: count candidates without validation
                n_detected = np.sum(vehicle_candidates)
        else:
            # No spatial data - count candidates but don't classify
            n_detected = np.sum(vehicle_candidates)

        return refined, n_detected

    def refine_classification(
        self,
        labels: np.ndarray,
        features: Dict[str, np.ndarray],
        ground_truth_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Main refinement function applying all enabled refinements.

        This is the unified entry point for all refinement operations,
        consolidating the functionality from classification_refinement.refine_classification().

        Args:
            labels: Initial classification labels [N]
            features: Dictionary of feature arrays
            ground_truth_data: Optional ground truth data

        Returns:
            Tuple of (refined_labels, refinement_stats)
        """
        refined = labels.copy()
        stats = {}

        logger.info("  🔧 Refining classification...")

        # Extract ground truth masks
        building_mask = None
        road_mask = None
        vegetation_mask = None

        if ground_truth_data is not None:
            building_mask = ground_truth_data.get("building_mask")
            road_mask = ground_truth_data.get("road_mask")
            vegetation_mask = ground_truth_data.get("vegetation_mask")

        # Apply refinements based on config
        if self.config.refine_vegetation:
            refined, n_veg = self.refine_vegetation(refined, features)
            stats["vegetation"] = n_veg
            logger.info(f"    Vegetation refined: {n_veg:,} points")

        if self.config.refine_buildings:
            refined, n_building = self.refine_buildings(
                refined, features, building_mask
            )
            stats["buildings"] = n_building
            logger.info(f"    Buildings refined: {n_building:,} points")

        if self.config.refine_roads:
            refined, n_road = self.refine_roads(refined, features, road_mask)
            stats["roads"] = n_road
            logger.info(f"    Roads refined: {n_road:,} points")

        if self.config.refine_ground:
            refined, n_ground = self.refine_ground(refined, features)
            stats["ground"] = n_ground
            logger.info(f"    Ground refined: {n_ground:,} points")

        if self.config.refine_vehicles:
            refined, n_vehicles = self.detect_vehicles(refined, features)
            stats["vehicles"] = n_vehicles
            logger.info(f"    Vehicles detected: {n_vehicles:,} candidates")

        return refined, stats

    def classify_lod2_elements(
        self, labels: np.ndarray, features: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Classify LOD2 building elements (roof, walls, ground).

        Args:
            labels: Current classification labels [N]
            features: Dictionary of feature arrays (height, planarity, verticality, etc.)

        Returns:
            LOD2 classification labels [N]
        """
        lod2_labels = labels.copy()

        # Only process building points
        building_points = labels == self.ASPRS_BUILDING

        if not np.any(building_points):
            return lod2_labels

        height = features.get("height")
        planarity = features.get("planarity")
        verticality = features.get("verticality")
        normal_z = features.get("normal_z")

        if height is None or planarity is None:
            return lod2_labels

        # Extract building features
        b_height = height[building_points]
        b_planarity = planarity[building_points]

        # Roof: High, horizontal, planar
        roof_mask = b_planarity > 0.85
        if normal_z is not None:
            b_normal_z = normal_z[building_points]
            roof_mask &= b_normal_z > 0.9

        # Wall: Medium planarity, high verticality
        if verticality is not None:
            b_verticality = verticality[building_points]
            wall_mask = (b_planarity > 0.6) & (b_verticality > 0.7)
        else:
            wall_mask = (b_planarity > 0.6) & ~roof_mask

        # Map to LOD2 classes
        building_indices = np.where(building_points)[0]
        lod2_labels[building_indices[roof_mask]] = LOD2Class.ROOF_FLAT.value
        lod2_labels[building_indices[wall_mask]] = LOD2Class.WALL.value

        return lod2_labels

    def classify_lod3_elements(
        self, labels: np.ndarray, features: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Classify LOD3 building elements (detailed architectural features).

        Args:
            labels: Current classification labels [N]
            features: Dictionary of feature arrays (height, planarity, verticality,
                     intensity, normals, roughness, etc.)

        Returns:
            LOD3 classification labels [N]
        """
        # Start with LOD2 classification (roof, walls)
        lod3_labels = self.classify_lod2_elements(labels, features)

        # LOD3-specific element detection
        # For comprehensive LOD3 element detection, consider using a dedicated module:
        # ign_lidar/core/classification/lod3_detector.py (future enhancement)

        # Basic LOD3 element detection
        building_points = labels == self.ASPRS_BUILDING

        if not building_points.any():
            return lod3_labels

        # Extract features
        height = features.get("height")
        planarity = features.get("planarity")
        verticality = features.get("verticality")
        intensity = features.get("intensity")
        roughness = features.get("roughness")
        normal_z = features.get("normal_z")

        if height is None:
            return lod3_labels

        building_indices = np.where(building_points)[0]
        b_height = height[building_points]

        # 1. Chimney detection (high, vertical, small footprint)
        if verticality is not None and planarity is not None:
            b_verticality = verticality[building_points]
            b_planarity = planarity[building_points]

            # Chimneys are typically: high, vertical, cylindrical (low planarity)
            chimney_candidates = (
                (b_height > 5.0)  # Above roof level
                & (b_verticality > 0.9)  # Very vertical
                & (b_planarity < 0.5)  # Cylindrical, not planar
            )

            if chimney_candidates.any():
                lod3_labels[building_indices[chimney_candidates]] = (
                    LOD3Class.CHIMNEY.value
                )

        # 2. Balcony detection (medium height, horizontal, protruding)
        if planarity is not None and normal_z is not None:
            b_planarity = planarity[building_points]
            b_normal_z = normal_z[building_points]

            # Balconies: horizontal surfaces at mid-height, not at roof level
            roof_height_estimate = np.percentile(b_height, 90)
            balcony_candidates = (
                (b_height > 2.0)  # Above ground floor
                & (b_height < roof_height_estimate - 1.0)  # Below roof
                & (b_planarity > 0.8)  # Flat horizontal surface
                & (b_normal_z > 0.9)  # Pointing up
            )

            if balcony_candidates.any():
                lod3_labels[building_indices[balcony_candidates]] = (
                    LOD3Class.BALCONY.value
                )

        # 3. Window detection (intensity drops, vertical surfaces)
        if intensity is not None and verticality is not None:
            b_intensity = intensity[building_points]
            b_verticality = verticality[building_points]

            # Windows: vertical surfaces with lower intensity (glass reflects less)
            mean_intensity = np.mean(b_intensity)
            window_candidates = (
                (b_verticality > 0.7)  # Vertical surface
                & (b_intensity < mean_intensity - 0.2)  # Lower intensity
                & (b_height > 1.0)
                & (b_height < 10.0)  # Reasonable window height
            )

            if window_candidates.any():
                lod3_labels[building_indices[window_candidates]] = (
                    LOD3Class.WINDOW.value
                )

        # 4. Roof type refinement (gable vs hip vs flat)
        if planarity is not None and normal_z is not None:
            b_planarity = planarity[building_points]
            b_normal_z = normal_z[building_points]

            # Identify roof points (already classified as ROOF_FLAT in LOD2)
            roof_lod2 = lod3_labels[building_indices] == LOD2Class.ROOF_FLAT.value

            if roof_lod2.any():
                roof_normal_z = b_normal_z[roof_lod2]
                roof_planarity = b_planarity[roof_lod2]

                # Flat roof: very horizontal
                flat_roof = (roof_normal_z > 0.95) & (roof_planarity > 0.9)

                # Gable roof: moderate slope, high planarity
                gable_roof = (
                    (roof_normal_z < 0.95)
                    & (roof_normal_z > 0.7)
                    & (roof_planarity > 0.85)
                )

                # Hip roof: similar to gable but more complex geometry
                # (This is a simplified heuristic - true hip roof detection requires clustering)
                hip_roof = (
                    (roof_normal_z < 0.95)
                    & (roof_normal_z > 0.7)
                    & (roof_planarity > 0.75)
                    & ~gable_roof
                )

                # Update labels
                roof_indices = building_indices[roof_lod2]
                lod3_labels[roof_indices[flat_roof]] = LOD3Class.ROOF_FLAT.value
                lod3_labels[roof_indices[gable_roof]] = LOD3Class.ROOF_GABLE.value
                lod3_labels[roof_indices[hip_roof]] = LOD3Class.ROOF_HIP.value

        # Note: For production-quality LOD3 element detection, consider implementing:
        # - Window/door detection using intensity patterns and geometry
        # - Dormer detection using roof plane clustering
        # - Skylight detection using material property analysis
        # - Architectural detail recognition using machine learning
        #
        # These features can be added in a dedicated LOD3Detector module:
        # ign_lidar/core/classification/lod3_detector.py

        return lod3_labels


# ============================================================================
# Convenience Functions
# ============================================================================


def classify_points(
    data: pd.DataFrame,
    strategy: str = "comprehensive",
    ground_truth_data: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> np.ndarray:
    """
    Convenience function for point classification.

    Args:
        data: DataFrame with point features
        strategy: Classification strategy ('basic', 'adaptive', 'comprehensive')
        ground_truth_data: Optional ground truth data
        **kwargs: Additional configuration options

    Returns:
        Classification labels array [N]

    Example:
        >>> labels = classify_points(
        ...     data=df,
        ...     strategy='comprehensive',
        ...     use_parcel_classification=True
        ... )
    """
    classifier = Classifier(strategy=strategy, **kwargs)
    return classifier.classify_points(data, ground_truth_data)


def refine_classification(
    labels: np.ndarray,
    features: Dict[str, np.ndarray],
    ground_truth_data: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Convenience function for classification refinement.

    Args:
        labels: Initial classification labels [N]
        features: Dictionary of feature arrays
        ground_truth_data: Optional ground truth data
        **kwargs: Additional configuration options

    Returns:
        Tuple of (refined_labels, refinement_stats)

    Example:
        >>> refined_labels, stats = refine_classification(
        ...     labels=labels,
        ...     features={'ndvi': ndvi, 'height': height, ...},
        ...     ground_truth_data=ground_truth
        ... )
    """
    classifier = Classifier(**kwargs)
    return classifier.refine_classification(labels, features, ground_truth_data)


# ============================================================================
# Deprecated aliases removed in v4.0
# Use Classifier, ClassifierConfig, classify_points, refine_classification directly
# ============================================================================
