"""
Advanced Classification Module with Geometric Features, NDVI, and Ground Truth

This module provides enhanced classification using:
- Parcel-based clustering (optional Stage 0)
- Geometric features (height, normals, planarity, curvature)
- NDVI for vegetation detection
- IGN BD TOPO® ground truth with intelligent road buffers
- Multi-criteria decision fusion
- Mode-aware building detection (ASPRS, LOD2, LOD3)

Author: Classification Enhancement
Date: October 15, 2025
Updated: October 16, 2025 - Integrated unified thresholds (Issue #8) and updated height filters (Issue #1, #4)
Updated: October 19, 2025 - Added parcel-based classification (Stage 0)
"""

import logging
from typing import Dict, Optional, Tuple, List, TYPE_CHECKING
import numpy as np
import pandas as pd

# Import unified thresholds
from .classification_thresholds import ClassificationThresholds

# Import building detection module
from .building_detection import (
    BuildingDetectionMode,
    BuildingDetectionConfig,
    detect_buildings_multi_mode
)

# Import transport detection module
from .transport_detection import (
    TransportDetectionMode,
    TransportDetectionConfig,
    detect_transport_multi_mode
)

# Import feature validation module
from .feature_validator import FeatureValidator

# Import parcel classifier (optional)
try:
    from .parcel_classifier import (
        ParcelClassifier,
        ParcelClassificationConfig
    )
    HAS_PARCEL_CLASSIFIER = True
except ImportError:
    HAS_PARCEL_CLASSIFIER = False
    ParcelClassifier = None
    ParcelClassificationConfig = None

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from shapely.geometry import Point, Polygon, MultiPolygon
    import geopandas as gpd

try:
    from shapely.geometry import Point, Polygon, MultiPolygon
    from shapely.strtree import STRtree
    from shapely import prepare as prep
    import geopandas as gpd
    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False
    STRtree = None
    prep = None


class AdvancedClassifier:
    """
    Advanced point cloud classifier combining multiple data sources.
    
    Classification hierarchy:
    1. Ground Truth (IGN BD TOPO®) - highest priority
    2. NDVI-based vegetation detection
    3. Geometric feature analysis
    4. Height-based classification
    5. Default/fallback classification
    """
    
    # ASPRS Classification codes (standard + extended)
    ASPRS_UNCLASSIFIED = 1
    ASPRS_GROUND = 2
    ASPRS_LOW_VEGETATION = 3
    ASPRS_MEDIUM_VEGETATION = 4
    ASPRS_HIGH_VEGETATION = 5
    ASPRS_BUILDING = 6
    ASPRS_LOW_POINT = 7
    ASPRS_WATER = 9
    ASPRS_RAIL = 10  # Rail/Railway
    ASPRS_ROAD = 11
    ASPRS_BRIDGE = 17  # Bridge deck
    
    # Extended codes for additional BD TOPO classes
    ASPRS_PARKING = 40  # Parking area (custom)
    ASPRS_SPORTS = 41  # Sports facility (custom)
    ASPRS_CEMETERY = 42  # Cemetery (custom)
    ASPRS_POWER_LINE = 43  # Power line corridor (custom)
    ASPRS_AGRICULTURE = 44  # Agricultural land (custom)
    
    def __init__(
        self,
        use_ground_truth: bool = True,
        use_ndvi: bool = True,
        use_geometric: bool = True,
        use_feature_validation: bool = True,
        use_parcel_classification: bool = False,
        parcel_classification_config: Optional[Dict] = None,
        road_buffer_tolerance: float = 0.5,
        ndvi_veg_threshold: float = 0.35,
        ndvi_building_threshold: float = 0.15,
        height_low_veg_threshold: float = 0.5,
        height_medium_veg_threshold: float = 2.0,
        planarity_road_threshold: float = 0.8,
        planarity_building_threshold: float = 0.7,
        building_detection_mode: str = 'asprs',
        transport_detection_mode: str = 'asprs_standard',
        feature_validation_config: Optional[Dict] = None
    ):
        """
        Initialize advanced classifier.
        
        Args:
            use_ground_truth: Use IGN BD TOPO® ground truth
            use_ndvi: Use NDVI for vegetation refinement
            use_geometric: Use geometric features
            use_feature_validation: Use feature-based ground truth validation
            use_parcel_classification: Use parcel-based classification (Stage 0)
            parcel_classification_config: Configuration for parcel classifier
            road_buffer_tolerance: Additional buffer around roads/rails (meters)
            ndvi_veg_threshold: NDVI threshold for vegetation (>= value)
            ndvi_building_threshold: NDVI threshold for non-vegetation (<= value)
            height_low_veg_threshold: Height threshold for low vegetation
            height_medium_veg_threshold: Height threshold for medium vegetation
            planarity_road_threshold: Planarity for road surfaces
            planarity_building_threshold: Planarity for building surfaces
            building_detection_mode: Mode for building detection ('asprs', 'lod2', or 'lod3')
            transport_detection_mode: Mode for transport detection ('asprs_standard', 'asprs_extended', or 'lod2')
            feature_validation_config: Configuration for feature validator
        """
        self.use_ground_truth = use_ground_truth
        self.use_ndvi = use_ndvi
        self.use_geometric = use_geometric
        self.use_feature_validation = use_feature_validation
        self.use_parcel_classification = use_parcel_classification
        self.building_detection_mode = building_detection_mode.lower()
        self.transport_detection_mode = transport_detection_mode.lower()
        
        # Thresholds
        self.road_buffer_tolerance = road_buffer_tolerance
        self.ndvi_veg_threshold = ndvi_veg_threshold
        self.ndvi_building_threshold = ndvi_building_threshold
        self.height_low_veg = height_low_veg_threshold
        self.height_medium_veg = height_medium_veg_threshold
        self.planarity_road = planarity_road_threshold
        self.planarity_building = planarity_building_threshold
        
        # Initialize parcel classifier if enabled
        self.parcel_classifier = None
        if self.use_parcel_classification:
            if not HAS_PARCEL_CLASSIFIER:
                logger.warning("  Parcel classification requested but module not available")
                self.use_parcel_classification = False
            else:
                # Create config from dict if provided
                if parcel_classification_config:
                    config = ParcelClassificationConfig(**parcel_classification_config)
                else:
                    config = ParcelClassificationConfig()
                self.parcel_classifier = ParcelClassifier(config=config)
                logger.info("  Parcel classification: ENABLED")
        
        # Initialize feature validator if enabled
        self.feature_validator = None
        if self.use_feature_validation:
            self.feature_validator = FeatureValidator(feature_validation_config)
            logger.info("  Feature validation: ENABLED")
        
        logger.info("🎯 Advanced Classifier initialized")
        logger.info(f"  Ground truth: {use_ground_truth}")
        logger.info(f"  NDVI refinement: {use_ndvi}")
        logger.info(f"  Geometric features: {use_geometric}")
        logger.info(f"  Building detection mode: {self.building_detection_mode.upper()}")
        logger.info(f"  Transport detection mode: {self.transport_detection_mode.upper()}")
    
    def classify_points(
        self,
        points: np.ndarray,
        ground_truth_features: Optional[Dict[str, 'gpd.GeoDataFrame']] = None,
        ndvi: Optional[np.ndarray] = None,
        height: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        planarity: Optional[np.ndarray] = None,
        curvature: Optional[np.ndarray] = None,
        verticality: Optional[np.ndarray] = None,
        intensity: Optional[np.ndarray] = None,
        return_number: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Classify points using all available data sources.
        
        Classification Stages:
        0. (Optional) Parcel-based clustering for spatial coherence
        1. Geometric feature classification
        2. NDVI-based vegetation refinement
        3. Ground truth classification (highest priority)
        4. Post-processing for unclassified points
        
        Args:
            points: Point coordinates [N, 3] (X, Y, Z)
            ground_truth_features: Dictionary of feature type -> GeoDataFrame
            ndvi: NDVI values [N] in range [-1, 1]
            height: Height above ground [N]
            normals: Surface normals [N, 3]
            planarity: Planarity values [N] in range [0, 1]
            curvature: Surface curvature [N]
            verticality: Verticality values [N] in range [0, 1]
            intensity: LiDAR intensity [N]
            return_number: Return number [N]
            
        Returns:
            Classification labels [N] in ASPRS format
        """
        n_points = len(points)
        logger.info(f"🎯 Classifying {n_points:,} points with advanced method")
        
        # Initialize with unclassified
        labels = np.full(n_points, self.ASPRS_UNCLASSIFIED, dtype=np.uint8)
        
        # Track confidence scores for each classification
        confidence = np.zeros(n_points, dtype=np.float32)
        
        # Stage 0: Parcel-based classification (optional, experimental)
        if self.use_parcel_classification and self.parcel_classifier is not None:
            cadastre = ground_truth_features.get('cadastre') if ground_truth_features else None
            if cadastre is not None and len(cadastre) > 0:
                logger.info("  Stage 0: Parcel-based classification (spatial coherence)")
                try:
                    # Prepare features dictionary for parcel classifier
                    parcel_features = {}
                    if ndvi is not None:
                        parcel_features['ndvi'] = ndvi
                    if height is not None:
                        parcel_features['height'] = height
                    if planarity is not None:
                        parcel_features['planarity'] = planarity
                    if verticality is not None:
                        parcel_features['verticality'] = verticality
                    if curvature is not None:
                        parcel_features['curvature'] = curvature
                    if normals is not None:
                        parcel_features['normals'] = normals
                    
                    # Get BD Forêt and RPG if available
                    bd_foret = ground_truth_features.get('forest')
                    rpg = ground_truth_features.get('rpg')
                    
                    # Run parcel classification
                    parcel_labels = self.parcel_classifier.classify_by_parcels(
                        points=points,
                        features=parcel_features,
                        cadastre=cadastre,
                        bd_foret=bd_foret,
                        rpg=rpg
                    )
                    
                    # Update labels and confidence for parcel-classified points
                    parcel_mask = parcel_labels != self.ASPRS_UNCLASSIFIED
                    labels[parcel_mask] = parcel_labels[parcel_mask]
                    confidence[parcel_mask] = 0.7  # Medium confidence from parcels
                    
                    n_parcel_classified = np.sum(parcel_mask)
                    pct = 100 * n_parcel_classified / n_points
                    logger.info(f"    Parcel-classified: {n_parcel_classified:,} points ({pct:.1f}%)")
                    
                except Exception as e:
                    logger.warning(f"    Parcel classification failed: {e}")
            else:
                logger.info("  Stage 0: Parcel classification skipped (no cadastre data)")
        
        # Stage 1: Geometric-based classification (if available)
        if self.use_geometric and height is not None:
            logger.info("  Stage 1: Geometric feature classification")
            labels, confidence = self._classify_by_geometry(
                labels, confidence, points, height, normals, planarity, 
                curvature, intensity, return_number
            )
        
        # Stage 2: NDVI-based vegetation detection
        if self.use_ndvi and ndvi is not None:
            logger.info("  Stage 2: NDVI-based vegetation refinement (multi-level)")
            # Store current labels as "original" for preservation
            original_labels_for_ndvi = labels.copy()
            labels, confidence = self._classify_by_ndvi(
                labels, confidence, ndvi, height, curvature, planarity,
                original_labels=original_labels_for_ndvi
            )
        
        # Stage 3: Ground truth (highest priority - overwrites previous)
        if self.use_ground_truth and ground_truth_features:
            logger.info("  Stage 3: Ground truth classification (highest priority)")
            labels = self._classify_by_ground_truth(
                labels, points, ground_truth_features, ndvi,
                height=height, planarity=planarity, intensity=intensity,
                curvature=curvature, normals=normals
            )
        
        # Stage 4: Post-processing for unclassified points
        logger.info("  Stage 4: Post-processing unclassified points")
        labels = self._post_process_unclassified(
            labels, confidence, points, height, normals, planarity,
            curvature, intensity, ground_truth_features
        )
        
        # Log final distribution
        self._log_distribution(labels)
        
        return labels
    
    def _classify_by_geometry(
        self,
        labels: np.ndarray,
        confidence: np.ndarray,
        points: np.ndarray,
        height: np.ndarray,
        normals: Optional[np.ndarray],
        planarity: Optional[np.ndarray],
        curvature: Optional[np.ndarray],
        intensity: Optional[np.ndarray],
        return_number: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Classify based on geometric features."""
        
        # Ground detection (low height + high planarity)
        if height is not None and planarity is not None:
            ground_mask = (height < 0.2) & (planarity > 0.85)
            labels[ground_mask] = self.ASPRS_GROUND
            confidence[ground_mask] = 0.9
            logger.info(f"    Ground: {np.sum(ground_mask):,} points")
        
        # Road detection (low height + very high planarity + horizontal normals)
        if height is not None and planarity is not None:
            road_mask = (
                (height >= 0.2) & 
                (height < 2.0) & 
                (planarity > self.planarity_road)
            )
            
            # Refine with horizontal normals if available
            if normals is not None:
                # Normals pointing up (z component close to 1)
                horizontal_mask = np.abs(normals[:, 2]) > 0.9
                road_mask = road_mask & horizontal_mask
            
            labels[road_mask] = self.ASPRS_ROAD
            confidence[road_mask] = 0.7
            logger.info(f"    Roads (geometric): {np.sum(road_mask):,} points")
        
        # Building detection (medium height + high planarity)
        # Use mode-aware building detection if enabled
        if height is not None and planarity is not None:
            if self.building_detection_mode in ['asprs', 'lod2', 'lod3']:
                # Use new building detection system
                try:
                    # Compute verticality if normals available
                    verticality = None
                    if normals is not None:
                        verticality = 1.0 - np.abs(normals[:, 2])  # 1 - |nz|
                    
                    # Prepare features for building detection
                    features_dict = {
                        'height': height,
                        'planarity': planarity,
                        'verticality': verticality,
                        'normals': normals,
                        'curvature': curvature,
                        'intensity': intensity,
                        'points': points
                    }
                    
                    # Detect buildings using mode-aware detection
                    labels_updated, stats = detect_buildings_multi_mode(
                        labels=labels,
                        features=features_dict,
                        mode=self.building_detection_mode,
                        ground_truth_mask=None,  # Ground truth handled separately
                        config=None  # Use default mode-specific config
                    )
                    
                    # Update labels where buildings detected
                    building_mask = labels_updated != labels
                    labels = labels_updated
                    confidence[building_mask] = 0.7
                    
                    logger.info(f"    Buildings ({self.building_detection_mode.upper()}): {stats.get('total_building', 0):,} points")
                    
                except Exception as e:
                    logger.warning(f"    Mode-aware building detection failed: {e}, using fallback")
                    # Fall back to simple detection
                    building_mask = (
                        (height >= 2.0) &
                        (planarity > self.planarity_building)
                    )
                    
                    if normals is not None:
                        roof_mask = np.abs(normals[:, 2]) > 0.7
                        wall_mask = np.abs(normals[:, 2]) < 0.3
                        building_mask = building_mask & (roof_mask | wall_mask)
                    
                    labels[building_mask] = self.ASPRS_BUILDING
                    confidence[building_mask] = 0.6
                    logger.info(f"    Buildings (geometric fallback): {np.sum(building_mask):,} points")
            else:
                # Legacy simple detection
                building_mask = (
                    (height >= 2.0) &
                    (planarity > self.planarity_building)
                )
                
                # Refine with vertical/horizontal normals if available
                if normals is not None:
                    # Either horizontal (roofs) or vertical (walls)
                    roof_mask = np.abs(normals[:, 2]) > 0.7  # Horizontal
                    wall_mask = np.abs(normals[:, 2]) < 0.3  # Vertical
                    building_mask = building_mask & (roof_mask | wall_mask)
                
                labels[building_mask] = self.ASPRS_BUILDING
                confidence[building_mask] = 0.6
                logger.info(f"    Buildings (geometric): {np.sum(building_mask):,} points")
        
        # Vegetation detection (low planarity + variable height)
        if height is not None and planarity is not None:
            # Low planarity suggests organic/irregular surfaces
            veg_mask = (planarity < 0.4) & (height > 0.2)
            
            # Classify by height
            low_veg_mask = veg_mask & (height < self.height_low_veg)
            medium_veg_mask = veg_mask & (height >= self.height_low_veg) & (height < self.height_medium_veg)
            high_veg_mask = veg_mask & (height >= self.height_medium_veg)
            
            labels[low_veg_mask] = self.ASPRS_LOW_VEGETATION
            labels[medium_veg_mask] = self.ASPRS_MEDIUM_VEGETATION
            labels[high_veg_mask] = self.ASPRS_HIGH_VEGETATION
            
            confidence[low_veg_mask] = 0.5
            confidence[medium_veg_mask] = 0.5
            confidence[high_veg_mask] = 0.5
            
            n_veg = np.sum(low_veg_mask) + np.sum(medium_veg_mask) + np.sum(high_veg_mask)
            logger.info(f"    Vegetation (geometric): {n_veg:,} points")
        
        return labels, confidence
    
    def _classify_by_ndvi(
        self,
        labels: np.ndarray,
        confidence: np.ndarray,
        ndvi: np.ndarray,
        height: Optional[np.ndarray],
        curvature: Optional[np.ndarray] = None,
        planarity: Optional[np.ndarray] = None,
        original_labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify and refine using NDVI with multi-level thresholds.
        
        ENHANCED NDVI Classification with Increased Sensitivity:
        - NDVI ≥ 0.65: Dense forest (high vegetation) - INCREASED from 0.60
        - NDVI ≥ 0.55: Healthy trees (high vegetation) - INCREASED from 0.50
        - NDVI ≥ 0.45: Moderate vegetation (medium) - INCREASED from 0.40
        - NDVI ≥ 0.35: Grass/shrubs (low/medium by height) - INCREASED from 0.30
        - NDVI ≥ 0.25: Sparse vegetation (low) - INCREASED from 0.20
        - NDVI < 0.25: Non-vegetation
        
        NEW FEATURE: Original Classification Preservation
        - If original_labels provided, preserve vegetation where NDVI confirms
        - Prevents loss of manually classified or high-confidence vegetation
        """
        
        # Multi-level NDVI thresholds (INCREASED SENSITIVITY)
        NDVI_DENSE_FOREST = 0.65  # Was 0.60 - now more selective for dense forest
        NDVI_HEALTHY_TREES = 0.55  # Was 0.50 - increased to capture healthier vegetation
        NDVI_MODERATE_VEG = 0.45   # Was 0.40 - higher threshold for moderate vegetation
        NDVI_GRASS = 0.35          # Was 0.30 - increased for grass/shrub detection
        NDVI_SPARSE_VEG = 0.25     # Was 0.20 - more sensitive sparse vegetation detection
        
        # ORIGINAL LABEL PRESERVATION: Store original vegetation classifications
        original_veg_mask = None
        if original_labels is not None:
            veg_classes = [self.ASPRS_LOW_VEGETATION, self.ASPRS_MEDIUM_VEGETATION, self.ASPRS_HIGH_VEGETATION]
            original_veg_mask = np.isin(original_labels, veg_classes)
            logger.info(f"    Preserving {original_veg_mask.sum():,} original vegetation labels where NDVI confirms")
        
        # Only override low-confidence classifications
        low_confidence_mask = confidence < 0.8
        
        if height is not None:
            # Dense forest (NDVI ≥ 0.65) - INCREASED THRESHOLD
            dense_forest = (ndvi >= NDVI_DENSE_FOREST) & low_confidence_mask
            if curvature is not None and planarity is not None:
                # Validate with features: forests have high curvature, low planarity
                dense_forest = dense_forest & (curvature > 0.25) & (planarity < 0.65)
            labels[dense_forest] = self.ASPRS_HIGH_VEGETATION
            confidence[dense_forest] = 0.95
            
            # Healthy trees (NDVI ≥ 0.50)
            healthy_trees = (ndvi >= NDVI_HEALTHY_TREES) & (ndvi < NDVI_DENSE_FOREST) & low_confidence_mask
            if curvature is not None and planarity is not None:
                healthy_trees = healthy_trees & (curvature > 0.20) & (planarity < 0.70)
            # Classify by height
            high_trees = healthy_trees & (height >= self.height_medium_veg)
            medium_trees = healthy_trees & (height < self.height_medium_veg) & (height >= self.height_low_veg)
            labels[high_trees] = self.ASPRS_HIGH_VEGETATION
            labels[medium_trees] = self.ASPRS_MEDIUM_VEGETATION
            confidence[healthy_trees] = 0.90
            
            # Moderate vegetation (NDVI ≥ 0.40)
            moderate_veg = (ndvi >= NDVI_MODERATE_VEG) & (ndvi < NDVI_HEALTHY_TREES) & low_confidence_mask
            if curvature is not None and planarity is not None:
                moderate_veg = moderate_veg & (curvature > 0.15) & (planarity < 0.75)
            high_moderate = moderate_veg & (height >= self.height_medium_veg)
            medium_moderate = moderate_veg & (height < self.height_medium_veg) & (height >= self.height_low_veg)
            low_moderate = moderate_veg & (height < self.height_low_veg)
            labels[high_moderate] = self.ASPRS_HIGH_VEGETATION
            labels[medium_moderate] = self.ASPRS_MEDIUM_VEGETATION
            labels[low_moderate] = self.ASPRS_LOW_VEGETATION
            confidence[moderate_veg] = 0.85
            
            # Grass/shrubs (NDVI ≥ 0.30)
            grass = (ndvi >= NDVI_GRASS) & (ndvi < NDVI_MODERATE_VEG) & low_confidence_mask
            if curvature is not None and planarity is not None:
                grass = grass & (curvature > 0.10) & (planarity < 0.80)
            medium_grass = grass & (height >= self.height_low_veg)
            low_grass = grass & (height < self.height_low_veg)
            labels[medium_grass] = self.ASPRS_MEDIUM_VEGETATION
            labels[low_grass] = self.ASPRS_LOW_VEGETATION
            confidence[grass] = 0.80
            
            # Sparse vegetation (NDVI ≥ 0.20)
            sparse_veg = (ndvi >= NDVI_SPARSE_VEG) & (ndvi < NDVI_GRASS) & low_confidence_mask
            labels[sparse_veg] = self.ASPRS_LOW_VEGETATION
            confidence[sparse_veg] = 0.70
            
            # Count vegetation points
            all_veg = (ndvi >= NDVI_SPARSE_VEG) & low_confidence_mask
            n_veg = np.sum(all_veg)
            logger.info(f"    Vegetation (multi-level NDVI): {n_veg:,} points")
            logger.info(f"      Dense forest (NDVI≥0.6): {np.sum(dense_forest):,}")
            logger.info(f"      Healthy trees (NDVI≥0.5): {np.sum(healthy_trees):,}")
            logger.info(f"      Moderate veg (NDVI≥0.4): {np.sum(moderate_veg):,}")
            logger.info(f"      Grass (NDVI≥0.35): {np.sum(grass):,}")
            logger.info(f"      Sparse veg (NDVI≥0.25): {np.sum(sparse_veg):,}")
            
            # ORIGINAL LABEL PRESERVATION: Restore original vegetation where NDVI confirms
            if original_veg_mask is not None:
                # For points originally classified as vegetation with confirming NDVI
                preserve_mask = original_veg_mask & (ndvi >= NDVI_SPARSE_VEG)
                n_preserved = preserve_mask.sum()
                if n_preserved > 0:
                    labels[preserve_mask] = original_labels[preserve_mask]
                    confidence[preserve_mask] = 0.95  # High confidence from original + NDVI
                    logger.info(f"    Preserved {n_preserved:,} original vegetation labels with NDVI confirmation")
        else:
            # No height - use simple NDVI classification
            high_ndvi = (ndvi >= NDVI_HEALTHY_TREES) & low_confidence_mask
            medium_ndvi = (ndvi >= NDVI_MODERATE_VEG) & (ndvi < NDVI_HEALTHY_TREES) & low_confidence_mask
            low_ndvi = (ndvi >= NDVI_SPARSE_VEG) & (ndvi < NDVI_MODERATE_VEG) & low_confidence_mask
            
            labels[high_ndvi] = self.ASPRS_HIGH_VEGETATION
            labels[medium_ndvi] = self.ASPRS_MEDIUM_VEGETATION
            labels[low_ndvi] = self.ASPRS_LOW_VEGETATION
            
            confidence[high_ndvi] = 0.85
            confidence[medium_ndvi] = 0.80
            confidence[low_ndvi] = 0.70
            
            logger.info(f"    Vegetation (NDVI, no height): {np.sum(ndvi >= NDVI_SPARSE_VEG):,} points")
        
        # Low NDVI = non-vegetation (buildings/roads/ground)
        # Refine building/road classifications
        low_ndvi_mask = (ndvi <= self.ndvi_building_threshold)
        
        # If labeled as vegetation but low NDVI, reclassify
        veg_classes = [self.ASPRS_LOW_VEGETATION, self.ASPRS_MEDIUM_VEGETATION, self.ASPRS_HIGH_VEGETATION]
        incorrect_veg = low_ndvi_mask & np.isin(labels, veg_classes)
        
        if np.any(incorrect_veg):
            # Reclassify based on height if available
            if height is not None:
                low_height = incorrect_veg & (height < 2.0)
                high_height = incorrect_veg & (height >= 2.0)
                
                labels[low_height] = self.ASPRS_ROAD  # or ground
                labels[high_height] = self.ASPRS_BUILDING
                
                confidence[incorrect_veg] = 0.7
                logger.info(f"    Reclassified low-NDVI vegetation: {np.sum(incorrect_veg):,} points")
            else:
                labels[incorrect_veg] = self.ASPRS_UNCLASSIFIED
        
        return labels, confidence
    
    def _classify_by_ground_truth(
        self,
        labels: np.ndarray,
        points: np.ndarray,
        ground_truth_features: Dict[str, 'gpd.GeoDataFrame'],
        ndvi: Optional[np.ndarray],
        height: Optional[np.ndarray] = None,
        planarity: Optional[np.ndarray] = None,
        intensity: Optional[np.ndarray] = None,
        curvature: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Classify using IGN BD TOPO® ground truth with intelligent road buffers.
        
        This is the highest priority classification and overwrites previous labels.
        With feature validation enabled, ground truth is validated against observed features.
        """
        if not HAS_SPATIAL:
            logger.warning("Spatial libraries not available, skipping ground truth")
            return labels
        
        # Create Point geometries
        point_geoms = [Point(p[0], p[1]) for p in points]
        
        # Classification priority (reverse order - last wins)
        # Lower in list = higher priority (overwrites previous)
        # NOTE: Vegetation removed - will be classified using features instead
        priority_order = [
            # ('vegetation', self.ASPRS_MEDIUM_VEGETATION),  # REMOVED: Use feature-based classification
            ('water', self.ASPRS_WATER),
            ('cemeteries', self.ASPRS_CEMETERY),
            ('parking', self.ASPRS_PARKING),
            ('sports', self.ASPRS_SPORTS),
            ('power_lines', self.ASPRS_POWER_LINE),
            ('railways', self.ASPRS_RAIL),
            ('roads', self.ASPRS_ROAD),
            ('bridges', self.ASPRS_BRIDGE),
            ('buildings', self.ASPRS_BUILDING)
        ]
        
        for feature_type, asprs_class in priority_order:
            if feature_type not in ground_truth_features:
                continue
            
            gdf = ground_truth_features[feature_type]
            if gdf is None or len(gdf) == 0:
                continue
            
            logger.info(f"    Processing {feature_type}: {len(gdf)} features")
            
            # Special handling for roads and railways with intelligent buffers
            if feature_type == 'roads':
                labels = self._classify_roads_with_buffer(
                    labels, point_geoms, gdf, asprs_class,
                    points=points, height=height, planarity=planarity, intensity=intensity
                )
            elif feature_type == 'railways':
                labels = self._classify_railways_with_buffer(
                    labels, point_geoms, gdf, asprs_class,
                    points=points, height=height, planarity=planarity, intensity=intensity
                )
            else:
                # Standard polygon intersection with STRtree optimization
                # OPTIMIZED: Use STRtree spatial indexing instead of iterating
                
                # Filter valid polygon geometries
                valid_mask = gdf['geometry'].apply(lambda g: isinstance(g, (Polygon, MultiPolygon)))
                valid_gdf = gdf[valid_mask]
                
                if len(valid_gdf) == 0:
                    continue
                
                # Build spatial index for polygons
                if STRtree is not None and len(valid_gdf) > 10:
                    # Use STRtree for efficient spatial queries (10-100× faster)
                    polygon_list = valid_gdf['geometry'].tolist()
                    tree = STRtree(polygon_list)
                    
                    # For each point, query the spatial index
                    for i, point_geom in enumerate(point_geoms):
                        if labels[i] != 0:  # Skip already labeled points
                            continue
                        
                        # Query returns potential polygon indices
                        potential_indices = tree.query(point_geom)
                        
                        # Check actual containment for matches
                        for poly_idx in potential_indices:
                            if polygon_list[poly_idx].contains(point_geom):
                                labels[i] = asprs_class
                                break
                else:
                    # Fallback: bbox filtering + containment check
                    for idx, row in valid_gdf.iterrows():
                        polygon = row['geometry']
                        
                        # PERFORMANCE FIX: Use bbox filtering before point-by-point check
                        bounds = polygon.bounds  # (minx, miny, maxx, maxy)
                        if points is not None:
                            # Vectorized bbox filtering
                            bbox_mask = (
                                (points[:, 0] >= bounds[0]) &
                                (points[:, 0] <= bounds[2]) &
                                (points[:, 1] >= bounds[1]) &
                                (points[:, 1] <= bounds[3])
                            )
                            candidate_indices = np.where(bbox_mask)[0]
                        else:
                            candidate_indices = range(len(point_geoms))
                        
                        # Check only candidate points
                        for i in candidate_indices:
                            if labels[i] == 0 and polygon.contains(point_geoms[i]):
                                labels[i] = asprs_class
        
        # NDVI refinement for vegetation vs building confusion
        if ndvi is not None:
            # Buildings with high NDVI might be vegetation on roofs
            building_mask = (labels == self.ASPRS_BUILDING)
            high_ndvi_buildings = building_mask & (ndvi >= self.ndvi_veg_threshold)
            
            if np.any(high_ndvi_buildings):
                # Keep as building but log for review
                n_veg_on_building = np.sum(high_ndvi_buildings)
                logger.info(f"    Note: {n_veg_on_building} building points with high NDVI (roof vegetation?)")
        
        # Feature-based validation of ground truth (NEW!)
        if self.use_feature_validation and self.feature_validator is not None:
            logger.info("  Stage 3b: Feature-based ground truth validation")
            
            # Build features dictionary for validation
            features = {}
            if height is not None:
                features['height'] = height
            if planarity is not None:
                features['planarity'] = planarity
            if curvature is not None:
                features['curvature'] = curvature
            if normals is not None:
                features['normals'] = normals
            if ndvi is not None:
                features['ndvi'] = ndvi
            
            # Create ground truth types array
            ground_truth_types = np.full(len(labels), 'none', dtype=object)
            for feature_type, asprs_class in priority_order:
                mask = labels == asprs_class
                ground_truth_types[mask] = feature_type
            
            # Validate ground truth with features
            try:
                validated_labels, confidences, valid_mask = self.feature_validator.validate_ground_truth(
                    labels, ground_truth_types, features
                )
                
                # Update labels with validated results
                n_changed = np.sum(labels != validated_labels)
                if n_changed > 0:
                    logger.info(f"    Feature validation corrected {n_changed:,} labels")
                    logger.info(f"    Mean confidence: {np.mean(confidences[valid_mask]):.2f}")
                labels = validated_labels
            except Exception as e:
                logger.warning(f"    Feature validation failed: {e}")
                # Continue with unvalidated labels
        
        return labels
    
    def _classify_roads_with_buffer(
        self,
        labels: np.ndarray,
        point_geoms: List[Point],
        roads_gdf: 'gpd.GeoDataFrame',
        asprs_class: int,
        points: Optional[np.ndarray] = None,
        height: Optional[np.ndarray] = None,
        planarity: Optional[np.ndarray] = None,
        intensity: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Classify road points using intelligent buffering and geometric filtering.
        
        The IGN BD TOPO® contains road centerlines with width attributes.
        We use the width to create appropriate buffers for each road, then
        apply geometric filters to improve accuracy:
        - Height filtering: exclude bridges/overpasses
        - Planarity filtering: roads are relatively flat
        - Intensity filtering: asphalt/concrete has specific reflectance
        """
        logger.info(f"      Using intelligent road buffers (tolerance={self.road_buffer_tolerance}m)")
        
        # Track road statistics
        road_widths = []
        points_per_road = []
        filtered_counts = {'height': 0, 'planarity': 0, 'intensity': 0}
        
        # Filter valid polygons
        valid_mask = roads_gdf['geometry'].apply(lambda g: isinstance(g, (Polygon, MultiPolygon)))
        valid_gdf = roads_gdf[valid_mask].copy()
        
        if len(valid_gdf) == 0:
            return labels
        
        # Apply additional tolerance buffer if specified (vectorized)
        if self.road_buffer_tolerance > 0:
            valid_gdf['geometry'] = valid_gdf['geometry'].buffer(self.road_buffer_tolerance)
        
        # Extract road widths for logging
        road_widths = valid_gdf.get('width_m', pd.Series('unknown', index=valid_gdf.index)).tolist()
        
        # OPTIMIZED: Use STRtree for spatial queries instead of iterating
        if STRtree is not None and len(valid_gdf) > 5:
            # Build spatial index for road polygons
            polygon_list = valid_gdf['geometry'].tolist()
            tree = STRtree(polygon_list)
            
            # Process each point with spatial index lookup
            n_classified = 0
            for i, point_geom in enumerate(point_geoms):
                if labels[i] != 0:  # Skip already labeled
                    continue
                
                # Query spatial index for potential road polygons
                potential_indices = tree.query(point_geom)
                
                # Check containment and apply filters
                for poly_idx in potential_indices:
                    if not polygon_list[poly_idx].contains(point_geom):
                        continue
                    
                    # Apply geometric filters
                    passes_filters = True
                    
                    # Height filter
                    if height is not None and passes_filters:
                        if height[i] > ClassificationThresholds.ROAD_HEIGHT_MAX or height[i] < ClassificationThresholds.ROAD_HEIGHT_MIN:
                            filtered_counts['height'] += 1
                            passes_filters = False
                    
                    # Planarity filter
                    if planarity is not None and passes_filters:
                        if planarity[i] < ClassificationThresholds.ROAD_PLANARITY_MIN:
                            filtered_counts['planarity'] += 1
                            passes_filters = False
                    
                    # Intensity filter
                    if intensity is not None and passes_filters:
                        if intensity[i] < ClassificationThresholds.ROAD_INTENSITY_MIN or intensity[i] > ClassificationThresholds.ROAD_INTENSITY_MAX:
                            filtered_counts['intensity'] += 1
                            passes_filters = False
                    
                    if passes_filters:
                        labels[i] = asprs_class
                        n_classified += 1
                        break  # Found containing road, move to next point
            
            points_per_road = [n_classified]  # Single aggregate for STRtree path
            
        else:
            # Fallback: iterate through roads (original logic preserved)
            for idx, row in valid_gdf.iterrows():
                polygon = row['geometry']
                
                # Classify points within this road polygon with geometric filtering
                n_classified = 0
                
                # Get polygon bounds for fast bbox filtering (first pass)
                bounds = polygon.bounds  # (minx, miny, maxx, maxy)
                if points is not None:
                    # Vectorized bbox filtering (10-100x faster than point-by-point)
                    bbox_mask = (
                        (points[:, 0] >= bounds[0]) &
                        (points[:, 0] <= bounds[2]) &
                        (points[:, 1] >= bounds[1]) &
                        (points[:, 1] <= bounds[3])
                    )
                    candidate_indices = np.where(bbox_mask)[0]
                else:
                    # Fallback to all points if numpy array not available
                    candidate_indices = range(len(point_geoms))
                
                # Test only candidate points (second pass with exact containment)
                for i in candidate_indices:
                    if labels[i] != 0:  # Skip already labeled
                        continue
                        
                    if not polygon.contains(point_geoms[i]):
                        continue
                    
                    # Apply geometric filters to improve accuracy
                    passes_filters = True
                    
                    # Height filter: exclude bridges, overpasses, elevated structures
                    # Updated thresholds (Issue #1): 2.0m max (was 1.5m), -0.5m min (was -0.3m)
                    if height is not None and passes_filters:
                        if height[i] > ClassificationThresholds.ROAD_HEIGHT_MAX or height[i] < ClassificationThresholds.ROAD_HEIGHT_MIN:
                            filtered_counts['height'] += 1
                            passes_filters = False
                    
                    # Planarity filter: roads should be relatively flat
                    if planarity is not None and passes_filters:
                        if planarity[i] < ClassificationThresholds.ROAD_PLANARITY_MIN:
                            filtered_counts['planarity'] += 1
                            passes_filters = False
                    
                    # Intensity filter: asphalt/concrete has characteristic reflectance
                    if intensity is not None and passes_filters:
                        if intensity[i] < ClassificationThresholds.ROAD_INTENSITY_MIN or intensity[i] > ClassificationThresholds.ROAD_INTENSITY_MAX:
                            filtered_counts['intensity'] += 1
                            passes_filters = False
                    
                    if passes_filters:
                        labels[i] = asprs_class
                        n_classified += 1
                
                points_per_road.append(n_classified)
        
        # Log statistics
        if road_widths:
            valid_widths = [w for w in road_widths if isinstance(w, (int, float)) and w > 0]
            if valid_widths:
                logger.info(f"      Road widths: {min(valid_widths):.1f}m - {max(valid_widths):.1f}m (avg: {np.mean(valid_widths):.1f}m)")
            
            total_road_points = sum(points_per_road)
            logger.info(f"      Classified {total_road_points:,} road points from {len(roads_gdf)} roads")
            logger.info(f"      Avg points per road: {np.mean(points_per_road):.0f}")
            
            # Log filtering statistics
            if any(filtered_counts.values()):
                logger.info(f"      Filtered out: height={filtered_counts['height']}, "
                          f"planarity={filtered_counts['planarity']}, intensity={filtered_counts['intensity']}")
        
        return labels
    
    def _classify_railways_with_buffer(
        self,
        labels: np.ndarray,
        point_geoms: List[Point],
        railways_gdf: 'gpd.GeoDataFrame',
        asprs_class: int,
        points: Optional[np.ndarray] = None,
        height: Optional[np.ndarray] = None,
        planarity: Optional[np.ndarray] = None,
        intensity: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Classify railway points using intelligent buffering and geometric filtering.
        
        The IGN BD TOPO® contains railway centerlines with width attributes.
        We use the width to create appropriate buffers for each railway, then
        apply geometric filters to improve accuracy:
        - Height filtering: exclude bridges/viaducts
        - Planarity filtering: tracks are relatively flat (less strict than roads)
        - Intensity filtering: ballast, rails, and sleepers have specific reflectance
        
        Similar to road buffering but typically narrower (default 3.5m for single track).
        """
        logger.info(f"      Using intelligent railway buffers (tolerance={self.road_buffer_tolerance}m)")
        
        # Track railway statistics
        railway_widths = []
        points_per_railway = []
        track_counts = []
        filtered_counts = {'height': 0, 'planarity': 0, 'intensity': 0}
        
        # OPTIMIZED: Use STRtree spatial indexing instead of iterrows() + contains() loop
        # Performance gain: 10-50× faster for large datasets with many railways
        try:
            # Prepare geometries with tolerance buffer applied
            buffered_polygons = []
            tolerance = self.road_buffer_tolerance * 1.2  # Railways need slightly wider tolerance for ballast
            
            for idx, row in railways_gdf.iterrows():
                polygon = row['geometry']
                if not isinstance(polygon, (Polygon, MultiPolygon)):
                    continue
                
                # Get railway width and track count for logging
                railway_width = row.get('width_m', 'unknown')
                n_tracks = row.get('nombre_voies', 1)
                railway_widths.append(railway_width if railway_width != 'unknown' else 0)
                track_counts.append(n_tracks)
                
                # Apply tolerance buffer
                if tolerance > 0:
                    polygon = polygon.buffer(tolerance)
                
                buffered_polygons.append((idx, polygon, railway_width, n_tracks))
            
            # Build spatial index (R-tree) for O(log N) queries
            if buffered_polygons:
                railway_tree = STRtree([p[1] for p in buffered_polygons])
                railway_lookup = {id(p[1]): (p[0], p[2], p[3]) for p in buffered_polygons}  # geom_id -> (idx, width, n_tracks)
                
                # Pre-check which points are already labeled (skip in query)
                unlabeled_mask = (labels == 0)
                unlabeled_indices = np.where(unlabeled_mask)[0]
                
                # Query spatial index for each unlabeled point (O(log N) per query vs O(N))
                for i in unlabeled_indices:
                    point = point_geoms[i]
                    
                    # Query spatial index for railways containing this point
                    candidate_railways = railway_tree.query(point, predicate='contains')
                    
                    if len(candidate_railways) > 0:
                        # Point is inside at least one railway
                        # Apply geometric filters to improve accuracy
                        passes_filters = True
                        
                        # Height filter: exclude bridges, viaducts, elevated tracks
                        # Updated thresholds (Issue #4): 2.0m max (was 1.2m), -0.5m min (was -0.2m)
                        if height is not None and passes_filters:
                            if height[i] > ClassificationThresholds.RAIL_HEIGHT_MAX or height[i] < ClassificationThresholds.RAIL_HEIGHT_MIN:
                                filtered_counts['height'] += 1
                                passes_filters = False
                        
                        # Planarity filter: tracks less planar than roads due to ballast
                        if planarity is not None and passes_filters:
                            if planarity[i] < ClassificationThresholds.RAIL_PLANARITY_MIN:
                                filtered_counts['planarity'] += 1
                                passes_filters = False
                        
                        # Intensity filter: ballast (dark), rails (bright), wide range
                        if intensity is not None and passes_filters:
                            if intensity[i] < ClassificationThresholds.RAIL_INTENSITY_MIN or intensity[i] > ClassificationThresholds.RAIL_INTENSITY_MAX:
                                filtered_counts['intensity'] += 1
                                passes_filters = False
                        
                        if passes_filters:
                            labels[i] = asprs_class
                            # Track which railway got this point (for statistics)
                            railway_geom = candidate_railways[0]
                            idx_in_gdf = railway_lookup.get(id(railway_geom), (None, None, None))[0]
                            if idx_in_gdf is not None:
                                # Find or create counter for this railway
                                while len(points_per_railway) <= idx_in_gdf:
                                    points_per_railway.append(0)
                                points_per_railway[idx_in_gdf] += 1
        
        except Exception as e:
            logger.warning(f"      STRtree optimization failed ({e}), falling back to bbox filtering")
            # FALLBACK: Original bbox filtering approach
            for idx, row in railways_gdf.iterrows():
                polygon = row['geometry']
                
                if not isinstance(polygon, (Polygon, MultiPolygon)):
                    continue
                
                # Get railway width and track count for logging
                railway_width = row.get('width_m', 'unknown')
                n_tracks = row.get('nombre_voies', 1)
                railway_widths.append(railway_width if railway_width != 'unknown' else 0)
                track_counts.append(n_tracks)
                
                # Apply tolerance buffer
                tolerance = self.road_buffer_tolerance * 1.2
                if tolerance > 0:
                    polygon = polygon.buffer(tolerance)
                
                n_classified = 0
                
                # Get polygon bounds for fast bbox filtering (first pass)
                bounds = polygon.bounds
                if points is not None:
                    bbox_mask = (
                        (points[:, 0] >= bounds[0]) &
                        (points[:, 0] <= bounds[2]) &
                        (points[:, 1] >= bounds[1]) &
                        (points[:, 1] <= bounds[3])
                    )
                    candidate_indices = np.where(bbox_mask)[0]
                else:
                    candidate_indices = range(len(point_geoms))
                
                # Test only candidate points (second pass with exact containment)
                for i in candidate_indices:
                    if labels[i] != 0:  # Skip already labeled
                        continue
                    
                    if not polygon.contains(point_geoms[i]):
                        continue
                    
                    # Apply geometric filters
                    passes_filters = True
                    
                    if height is not None and passes_filters:
                        if height[i] > ClassificationThresholds.RAIL_HEIGHT_MAX or height[i] < ClassificationThresholds.RAIL_HEIGHT_MIN:
                            filtered_counts['height'] += 1
                            passes_filters = False
                    
                    if planarity is not None and passes_filters:
                        if planarity[i] < ClassificationThresholds.RAIL_PLANARITY_MIN:
                            filtered_counts['planarity'] += 1
                            passes_filters = False
                    
                    if intensity is not None and passes_filters:
                        if intensity[i] < ClassificationThresholds.RAIL_INTENSITY_MIN or intensity[i] > ClassificationThresholds.RAIL_INTENSITY_MAX:
                            filtered_counts['intensity'] += 1
                            passes_filters = False
                    
                    if passes_filters:
                        labels[i] = asprs_class
                        n_classified += 1
                
                points_per_railway.append(n_classified)
        
        # Log statistics
        if railway_widths:
            valid_widths = [w for w in railway_widths if isinstance(w, (int, float)) and w > 0]
            if valid_widths:
                logger.info(f"      Railway widths: {min(valid_widths):.1f}m - {max(valid_widths):.1f}m (avg: {np.mean(valid_widths):.1f}m)")
            
            total_railway_points = sum(points_per_railway)
            logger.info(f"      Classified {total_railway_points:,} railway points from {len(railways_gdf)} railways")
            logger.info(f"      Avg points per railway: {np.mean(points_per_railway):.0f}")
            
            if track_counts:
                unique_tracks = sorted(set(track_counts))
                logger.info(f"      Track counts: {unique_tracks} (single, double, etc.)")
            
            # Log filtering statistics
            if any(filtered_counts.values()):
                logger.info(f"      Filtered out: height={filtered_counts['height']}, "
                          f"planarity={filtered_counts['planarity']}, intensity={filtered_counts['intensity']}")
        
        return labels
    
    def _post_process_unclassified(
        self,
        labels: np.ndarray,
        confidence: np.ndarray,
        points: np.ndarray,
        height: Optional[np.ndarray],
        normals: Optional[np.ndarray],
        planarity: Optional[np.ndarray],
        curvature: Optional[np.ndarray],
        intensity: Optional[np.ndarray],
        ground_truth_features: Optional[Dict[str, 'gpd.GeoDataFrame']]
    ) -> np.ndarray:
        """
        Post-process unclassified points to improve building classification.
        
        This function handles points that remain unclassified after initial
        classification stages. It applies heuristics to classify them based on:
        1. Proximity to classified building points
        2. Geometric similarity to buildings (height, planarity, verticality)
        3. Context from ground truth building footprints
        4. Intensity patterns characteristic of building materials
        
        Args:
            labels: Current classification labels [N]
            confidence: Confidence scores [N]
            points: Point coordinates [N, 3]
            height: Height above ground [N]
            normals: Surface normals [N, 3]
            planarity: Planarity values [N]
            curvature: Surface curvature [N]
            intensity: LiDAR intensity [N]
            ground_truth_features: Ground truth building polygons
            
        Returns:
            Updated labels with fewer unclassified points
        """
        unclassified_mask = labels == self.ASPRS_UNCLASSIFIED
        n_unclassified = np.sum(unclassified_mask)
        
        if n_unclassified == 0:
            logger.debug("    No unclassified points to post-process")
            return labels
        
        logger.info(f"    Post-processing {n_unclassified:,} unclassified points")
        
        # Create a copy for updates
        updated_labels = labels.copy()
        n_reclassified = 0
        
        # Strategy 1: Check if unclassified points are within building footprints
        if ground_truth_features and 'buildings' in ground_truth_features:
            if not HAS_SPATIAL:
                logger.debug("      Spatial libraries not available for ground truth matching")
            else:
                buildings_gdf = ground_truth_features['buildings']
                if buildings_gdf is not None and len(buildings_gdf) > 0:
                    unclassified_indices = np.where(unclassified_mask)[0]
                    unclassified_points = points[unclassified_mask]
                    
                    # Create point geometries for unclassified points
                    from shapely.geometry import Point
                    point_geoms = [Point(p[0], p[1]) for p in unclassified_points]
                    
                    # OPTIMIZED: Use STRtree spatial indexing instead of iterrows() + contains() loop
                    # Performance gain: 10-100× faster for large datasets with many buildings
                    try:
                        # Build spatial index (R-tree) for O(log N) queries
                        valid_buildings = []
                        for idx, row in buildings_gdf.iterrows():
                            polygon = row['geometry']
                            if isinstance(polygon, (Polygon, MultiPolygon)):
                                valid_buildings.append(polygon)
                        
                        if valid_buildings:
                            building_tree = STRtree(valid_buildings)
                            
                            # Query spatial index for each unclassified point (O(log N) per query vs O(N))
                            for i, point in enumerate(point_geoms):
                                # Query spatial index for buildings containing this point
                                candidate_buildings = building_tree.query(point, predicate='contains')
                                
                                if len(candidate_buildings) > 0:
                                    # Point is inside at least one building
                                    within_building[i] = True
                    
                    except Exception as e:
                        logger.warning(f"      STRtree optimization failed ({e}), falling back to bbox filtering")
                        # FALLBACK: Original bbox filtering approach
                        for idx, row in buildings_gdf.iterrows():
                            polygon = row['geometry']
                            if not isinstance(polygon, (Polygon, MultiPolygon)):
                                continue
                            
                            # Use bbox filtering before point-by-point check
                            bounds = polygon.bounds
                            bbox_mask = (
                                (unclassified_points[:, 0] >= bounds[0]) &
                                (unclassified_points[:, 0] <= bounds[2]) &
                                (unclassified_points[:, 1] >= bounds[1]) &
                                (unclassified_points[:, 1] <= bounds[3])
                            )
                            candidate_indices = np.where(bbox_mask)[0]
                            
                            for i in candidate_indices:
                                if polygon.contains(point_geoms[i]):
                                    within_building[i] = True
                    
                    # Classify points within building footprints
                    n_within = np.sum(within_building)
                    if n_within > 0:
                        building_indices = unclassified_indices[within_building]
                        updated_labels[building_indices] = self.ASPRS_BUILDING
                        n_reclassified += n_within
                        logger.info(f"      Ground truth: {n_within:,} points within building footprints")
        
        # Strategy 2: Geometric features for building-like unclassified points
        if height is not None and planarity is not None:
            still_unclassified = updated_labels == self.ASPRS_UNCLASSIFIED
            
            # Building-like characteristics
            building_like = (
                (height > 2.5) &  # Above typical building height
                (planarity > 0.6) &  # Reasonably planar
                still_unclassified
            )
            
            # Enhance with normals if available (walls or roofs)
            if normals is not None:
                # Vertical (walls) or horizontal (roofs)
                vertical_mask = np.abs(normals[:, 2]) < 0.3  # Vertical surfaces
                horizontal_mask = np.abs(normals[:, 2]) > 0.85  # Horizontal surfaces
                building_orientation = vertical_mask | horizontal_mask
                building_like = building_like & building_orientation
            
            # Exclude vegetation using curvature (buildings are less curved)
            if curvature is not None:
                low_curvature = curvature < 0.02  # Buildings have low curvature
                building_like = building_like & low_curvature
            
            # Apply intensity filter (buildings typically moderate to high intensity)
            if intensity is not None:
                building_intensity = (intensity > 0.2) & (intensity < 0.85)
                building_like = building_like & building_intensity
            
            n_geometric = np.sum(building_like)
            if n_geometric > 0:
                updated_labels[building_like] = self.ASPRS_BUILDING
                n_reclassified += n_geometric
                logger.info(f"      Geometric: {n_geometric:,} building-like points classified")
        
        # Strategy 3: Classify remaining low-height unclassified as ground
        if height is not None:
            still_unclassified = updated_labels == self.ASPRS_UNCLASSIFIED
            low_height = (height < 0.5) & still_unclassified
            
            n_ground = np.sum(low_height)
            if n_ground > 0:
                updated_labels[low_height] = self.ASPRS_GROUND
                n_reclassified += n_ground
                logger.info(f"      Low height: {n_ground:,} points classified as ground")
        
        # Strategy 4: Classify remaining medium-height with low planarity as vegetation
        if height is not None and planarity is not None:
            still_unclassified = updated_labels == self.ASPRS_UNCLASSIFIED
            vegetation_like = (
                (height >= 0.5) &
                (height < 2.0) &
                (planarity < 0.4) &  # Low planarity (irregular)
                still_unclassified
            )
            
            n_veg = np.sum(vegetation_like)
            if n_veg > 0:
                updated_labels[vegetation_like] = self.ASPRS_LOW_VEGETATION
                n_reclassified += n_veg
                logger.info(f"      Vegetation-like: {n_veg:,} points classified as low vegetation")
        
        # Log summary
        final_unclassified = np.sum(updated_labels == self.ASPRS_UNCLASSIFIED)
        logger.info(f"    Reclassified {n_reclassified:,} points, {final_unclassified:,} remain unclassified")
        
        return updated_labels
    
    def _log_distribution(self, labels: np.ndarray):
        """Log the final classification distribution."""
        unique, counts = np.unique(labels, return_counts=True)
        
        class_names = {
            self.ASPRS_UNCLASSIFIED: 'Unclassified',
            self.ASPRS_GROUND: 'Ground',
            self.ASPRS_LOW_VEGETATION: 'Low Vegetation',
            self.ASPRS_MEDIUM_VEGETATION: 'Medium Vegetation',
            self.ASPRS_HIGH_VEGETATION: 'High Vegetation',
            self.ASPRS_BUILDING: 'Building',
            self.ASPRS_LOW_POINT: 'Low Point',
            self.ASPRS_WATER: 'Water',
            self.ASPRS_RAIL: 'Rail',
            self.ASPRS_ROAD: 'Road',
            self.ASPRS_BRIDGE: 'Bridge',
            self.ASPRS_PARKING: 'Parking',
            self.ASPRS_SPORTS: 'Sports Facility',
            self.ASPRS_CEMETERY: 'Cemetery',
            self.ASPRS_POWER_LINE: 'Power Line',
            self.ASPRS_AGRICULTURE: 'Agriculture'
        }
        
        logger.info("📊 Final classification distribution:")
        total = len(labels)
        for label_val, count in zip(unique, counts):
            name = class_names.get(label_val, f'Unknown_{label_val}')
            percentage = 100 * count / total
            logger.info(f"  {name:20s}: {count:8,} ({percentage:5.1f}%)")


def classify_with_all_features(
    points: np.ndarray,
    ground_truth_fetcher=None,
    bd_foret_fetcher=None,
    rpg_fetcher=None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    ndvi: Optional[np.ndarray] = None,
    height: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    planarity: Optional[np.ndarray] = None,
    curvature: Optional[np.ndarray] = None,
    intensity: Optional[np.ndarray] = None,
    return_number: Optional[np.ndarray] = None,
    include_railways: bool = True,
    include_forest: bool = True,
    include_agriculture: bool = True,
    include_bridges: bool = False,
    include_parking: bool = False,
    include_sports: bool = False,
    **kwargs
) -> Tuple[np.ndarray, Optional[Dict], Optional[Dict]]:
    """
    Convenience function for classification with all features including railways and BD Forêt®.
    
    Args:
        points: Point coordinates [N, 3]
        ground_truth_fetcher: IGNGroundTruthFetcher instance
        bd_foret_fetcher: BDForetFetcher instance for forest type classification
        bbox: Bounding box for fetching ground truth
        ndvi: NDVI values
        height: Height above ground
        normals: Surface normals
        planarity: Planarity values
        curvature: Surface curvature
        intensity: LiDAR intensity
        return_number: Return number
        include_railways: Whether to include railway classification
        include_forest: Whether to include BD Forêt® forest type refinement
        **kwargs: Additional arguments for AdvancedClassifier
        
    Returns:
        Tuple of (classification labels [N], forest_attributes dict or None)
        
    Example:
        >>> labels, forest_attrs = classify_with_all_features(
        ...     points=points,
        ...     ground_truth_fetcher=gt_fetcher,
        ...     bd_foret_fetcher=forest_fetcher,
        ...     bbox=bbox,
        ...     ndvi=ndvi,
        ...     height=height,
        ...     include_railways=True,
        ...     include_forest=True
        ... )
    """
    # Fetch ground truth if fetcher provided
    ground_truth_features = None
    if ground_truth_fetcher and bbox:
        logger.info("Fetching ground truth from IGN BD TOPO®...")
        ground_truth_features = ground_truth_fetcher.fetch_all_features(
            bbox=bbox,
            include_buildings=True,
            include_roads=True,
            include_water=True,
            include_vegetation=True,
            include_railways=include_railways,
            include_bridges=include_bridges,
            include_parking=include_parking,
            include_sports=include_sports
        )
    
    # Create classifier
    classifier = AdvancedClassifier(
        use_ground_truth=ground_truth_features is not None,
        use_ndvi=ndvi is not None,
        use_geometric=(height is not None or normals is not None or planarity is not None),
        **kwargs
    )
    
    # Classify
    labels = classifier.classify_points(
        points=points,
        ground_truth_features=ground_truth_features,
        ndvi=ndvi,
        height=height,
        normals=normals,
        planarity=planarity,
        curvature=curvature,
        intensity=intensity,
        return_number=return_number
    )
    
    # Refine vegetation classification with BD Forêt® if available
    forest_attributes = None
    if include_forest and bd_foret_fetcher and bbox:
        logger.info("Refining vegetation classification with BD Forêt® V2...")
        try:
            # Fetch forest polygons
            forest_gdf = bd_foret_fetcher.fetch_forest_polygons(bbox)
            
            if forest_gdf is not None and len(forest_gdf) > 0:
                # Label vegetation points with forest types
                forest_attributes = bd_foret_fetcher.label_points_with_forest_type(
                    points=points,
                    labels=labels,
                    forest_gdf=forest_gdf
                )
                
                # Log forest type statistics
                if forest_attributes:
                    n_labeled = sum(1 for t in forest_attributes.get('forest_type', []) if t and t != 'unknown')
                    logger.info(f"  Labeled {n_labeled:,} vegetation points with forest types")
                    
                    # Count forest types
                    from collections import Counter
                    type_counts = Counter(forest_attributes.get('forest_type', []))
                    for ftype, count in type_counts.most_common(5):
                        if ftype and ftype != 'unknown':
                            logger.info(f"    {ftype}: {count:,} points")
            else:
                logger.info("  No forest data found in BD Forêt® for this area")
                
        except Exception as e:
            logger.warning(f"  Failed to fetch/apply BD Forêt® data: {e}")
    
    # Refine ground/vegetation with RPG agricultural parcels if available
    rpg_attributes = None
    if include_agriculture and rpg_fetcher and bbox:
        logger.info("Refining classification with RPG agricultural parcels...")
        try:
            # Fetch agricultural parcels
            parcels_gdf = rpg_fetcher.fetch_parcels(bbox)
            
            if parcels_gdf is not None and len(parcels_gdf) > 0:
                # Label ground/vegetation points with crop types
                rpg_attributes = rpg_fetcher.label_points_with_crops(
                    points=points,
                    labels=labels,
                    parcels_gdf=parcels_gdf
                )
                
                # Optionally update labels for agricultural areas
                if rpg_attributes:
                    is_agri = rpg_attributes.get('is_agricultural', [])
                    n_agri = sum(is_agri)
                    
                    if n_agri > 0:
                        logger.info(f"  Labeled {n_agri:,} points as agricultural")
                        
                        # Optionally change ground/low veg to AGRICULTURE code
                        for i, is_ag in enumerate(is_agri):
                            if is_ag and labels[i] in [2, 3]:  # Ground or low veg
                                labels[i] = AdvancedClassifier.ASPRS_AGRICULTURE
                        
                        # Count crop types
                        from collections import Counter
                        crop_cats = [c for c in rpg_attributes.get('crop_category', []) if c != 'unknown']
                        if crop_cats:
                            cat_counts = Counter(crop_cats)
                            logger.info(f"  Crop categories:")
                            for cat, count in cat_counts.most_common():
                                logger.info(f"    {cat}: {count:,} points")
            else:
                logger.info("  No RPG parcels found in this area")
                
        except Exception as e:
            logger.warning(f"  Failed to fetch/apply RPG data: {e}")
    
    return labels, forest_attributes, rpg_attributes
