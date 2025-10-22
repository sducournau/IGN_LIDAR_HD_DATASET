"""
Hierarchical Multi-Level Classification System

This module implements a hierarchical classification system that operates at multiple
levels of detail:
    - ASPRS Standard (baseline classification)
    - LOD2 (building-focused with 15 classes)
    - LOD3 (detailed building elements with 30 classes)

The system uses intelligent mapping rules, confidence scores, and progressive refinement
to ensure accurate classification at each level.

Author: IGN LiDAR HD Dataset Team
Date: October 15, 2025
"""

import logging
from typing import Dict, Optional, Tuple, List, Any
from enum import IntEnum
import numpy as np
from dataclasses import dataclass

from ign_lidar.classification_schema import (
    LOD2_CLASSES, LOD3_CLASSES, ASPRS_TO_LOD2, ASPRS_TO_LOD3, ASPRSClass
)

logger = logging.getLogger(__name__)


# ============================================================================
# Classification Levels
# ============================================================================

class ClassificationLevel(IntEnum):
    """Supported classification levels."""
    ASPRS = 1      # ASPRS Standard classification (22 basic classes)
    LOD2 = 2       # Level of Detail 2 (15 building-focused classes)
    LOD3 = 3       # Level of Detail 3 (30 detailed building classes)


@dataclass
class ClassificationResult:
    """Result of hierarchical classification with metadata."""
    labels: np.ndarray                    # Final classification labels [N]
    level: ClassificationLevel            # Classification level applied
    confidence_scores: Optional[np.ndarray] = None  # Confidence per point [N]
    feature_importance: Optional[Dict[str, float]] = None  # Feature contributions
    num_refined: int = 0                  # Number of points refined
    hierarchy_path: Optional[List[str]] = None  # Classification hierarchy trace
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get classification statistics."""
        unique, counts = np.unique(self.labels, return_counts=True)
        total = len(self.labels)
        
        stats = {
            'total_points': total,
            'num_classes': len(unique),
            'class_distribution': dict(zip(unique.tolist(), counts.tolist())),
            'class_percentages': {int(c): float(cnt) / total * 100 
                                 for c, cnt in zip(unique, counts)},
            'num_refined': self.num_refined,
            'level': self.level.name
        }
        
        if self.confidence_scores is not None:
            stats['avg_confidence'] = float(np.mean(self.confidence_scores))
            stats['min_confidence'] = float(np.min(self.confidence_scores))
            stats['low_confidence_points'] = int(np.sum(self.confidence_scores < 0.5))
        
        return stats


# ============================================================================
# Hierarchical Classifier
# ============================================================================

class HierarchicalClassifier:
    """
    Multi-level hierarchical classifier for LiDAR point clouds.
    
    This classifier operates at three levels:
    1. ASPRS Standard - Base classification from LAS files
    2. LOD2 - Building-focused classification (15 classes)
    3. LOD3 - Detailed architectural elements (30 classes)
    
    Features:
    - Intelligent mapping between levels
    - Confidence scoring at each level
    - Progressive refinement using multiple data sources
    - Feature importance tracking
    """
    
    def __init__(
        self,
        target_level: ClassificationLevel = ClassificationLevel.LOD2,
        use_confidence_scores: bool = True,
        refinement_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the hierarchical classifier.
        
        Args:
            target_level: Target classification level (ASPRS, LOD2, or LOD3)
            use_confidence_scores: Whether to compute confidence scores
            refinement_config: Configuration for classification refinement
        """
        self.target_level = target_level
        self.use_confidence_scores = use_confidence_scores
        self.refinement_config = refinement_config or {}
        
        # Initialize mappings
        self._initialize_mappings()
        
        logger.info(f"Initialized HierarchicalClassifier (target level: {target_level.name})")
    
    def _initialize_mappings(self):
        """Initialize classification mapping dictionaries."""
        # ASPRS to LOD2 mapping (already defined in classes.py)
        self.asprs_to_lod2 = ASPRS_TO_LOD2
        
        # ASPRS to LOD3 mapping (already defined in classes.py)
        self.asprs_to_lod3 = ASPRS_TO_LOD3 if hasattr('ASPRS_TO_LOD3', '__module__') else {}
        
        # LOD2 to LOD3 mapping (for progressive refinement)
        self.lod2_to_lod3 = {
            # Wall mapping
            0: 0,   # wall -> wall_plain (requires refinement)
            
            # Roof types (preserve)
            1: 3,   # roof_flat -> roof_flat
            2: 4,   # roof_gable -> roof_gable
            3: 5,   # roof_hip -> roof_hip
            
            # Roof details (preserve)
            4: 8,   # chimney -> chimney
            5: 9,   # dormer -> dormer_gable (requires refinement)
            
            # Facades
            6: 16,  # balcony -> balcony
            7: 18,  # overhang -> overhang
            
            # Foundation
            8: 21,  # foundation -> foundation
            
            # Context (preserve)
            9: 23,  # ground -> ground
            10: 24, # vegetation_low -> vegetation_low
            11: 25, # vegetation_high -> vegetation_high
            12: 26, # water -> water
            13: 27, # vehicle -> vehicle
            14: 29, # other -> other
        }
        
        # Reverse mappings for confidence computation
        self.lod2_to_asprs = {v: k for k, v in self.asprs_to_lod2.items()}
        self.lod3_to_lod2 = {v: k for k, v in self.lod2_to_lod3.items()}
    
    def classify(
        self,
        asprs_labels: np.ndarray,
        features: Optional[Dict[str, np.ndarray]] = None,
        ground_truth: Optional[Dict[str, Any]] = None,
        track_hierarchy: bool = False
    ) -> ClassificationResult:
        """
        Perform hierarchical classification to target level.
        
        Args:
            asprs_labels: Initial ASPRS classification labels [N]
            features: Optional dictionary of geometric features for refinement
                - 'height': Height above ground [N]
                - 'ndvi': NDVI values [N]
                - 'normals': Surface normals [N, 3]
                - 'planarity': Planarity scores [N]
                - 'curvature': Curvature values [N]
                - 'intensity': LiDAR intensity [N]
            ground_truth: Optional ground truth data for refinement
            track_hierarchy: Whether to track classification hierarchy path
        
        Returns:
            ClassificationResult with labels and metadata
        """
        n_points = len(asprs_labels)
        hierarchy_path = [] if track_hierarchy else None
        
        # Initialize confidence scores
        confidence_scores = np.ones(n_points) if self.use_confidence_scores else None
        
        # Step 1: Map from ASPRS to target level
        if self.target_level == ClassificationLevel.ASPRS:
            # No mapping needed
            final_labels = asprs_labels.copy()
            if hierarchy_path is not None:
                hierarchy_path.append("ASPRS (baseline)")
        
        elif self.target_level == ClassificationLevel.LOD2:
            # Map ASPRS -> LOD2
            final_labels = self._map_asprs_to_lod2(asprs_labels)
            if hierarchy_path is not None:
                hierarchy_path.append("ASPRS -> LOD2")
            
            # Compute initial confidence based on ASPRS quality
            if confidence_scores is not None:
                confidence_scores = self._compute_mapping_confidence(
                    asprs_labels, final_labels, level='LOD2'
                )
        
        elif self.target_level == ClassificationLevel.LOD3:
            # Progressive mapping: ASPRS -> LOD2 -> LOD3
            lod2_labels = self._map_asprs_to_lod2(asprs_labels)
            if hierarchy_path is not None:
                hierarchy_path.append("ASPRS -> LOD2")
            
            final_labels = self._map_lod2_to_lod3(lod2_labels)
            if hierarchy_path is not None:
                hierarchy_path.append("LOD2 -> LOD3")
            
            # Compute confidence for two-stage mapping
            if confidence_scores is not None:
                conf_asprs_lod2 = self._compute_mapping_confidence(
                    asprs_labels, lod2_labels, level='LOD2'
                )
                conf_lod2_lod3 = self._compute_mapping_confidence(
                    lod2_labels, final_labels, level='LOD3'
                )
                # Combined confidence (multiplicative)
                confidence_scores = conf_asprs_lod2 * conf_lod2_lod3
        
        else:
            raise ValueError(f"Unsupported classification level: {self.target_level}")
        
        # Step 2: Refine classification using additional features
        num_refined = 0
        if features is not None:
            final_labels, refined_count, feature_importance = self._refine_classification(
                final_labels, features, ground_truth, confidence_scores
            )
            num_refined = refined_count
            if hierarchy_path is not None:
                hierarchy_path.append(f"Refined ({refined_count} points)")
        else:
            feature_importance = None
        
        # Create result
        result = ClassificationResult(
            labels=final_labels,
            level=self.target_level,
            confidence_scores=confidence_scores,
            feature_importance=feature_importance,
            num_refined=num_refined,
            hierarchy_path=hierarchy_path
        )
        
        # Log statistics
        stats = result.get_statistics()
        logger.info(f"Classification complete: {stats['total_points']:,} points, "
                   f"{stats['num_classes']} classes, {num_refined:,} refined")
        if confidence_scores is not None:
            logger.info(f"Average confidence: {stats['avg_confidence']:.2%}")
        
        return result
    
    def _map_asprs_to_lod2(self, asprs_labels: np.ndarray) -> np.ndarray:
        """Map ASPRS labels to LOD2 classes."""
        lod2_labels = np.zeros_like(asprs_labels)
        
        for asprs_class, lod2_class in self.asprs_to_lod2.items():
            mask = asprs_labels == asprs_class
            lod2_labels[mask] = lod2_class
        
        # Unmapped classes -> 'other' (14)
        unmapped_mask = ~np.isin(asprs_labels, list(self.asprs_to_lod2.keys()))
        lod2_labels[unmapped_mask] = LOD2_CLASSES['other']
        
        return lod2_labels
    
    def _map_lod2_to_lod3(self, lod2_labels: np.ndarray) -> np.ndarray:
        """Map LOD2 labels to LOD3 classes."""
        lod3_labels = np.zeros_like(lod2_labels)
        
        for lod2_class, lod3_class in self.lod2_to_lod3.items():
            mask = lod2_labels == lod2_class
            lod3_labels[mask] = lod3_class
        
        # Unmapped classes -> 'other' (29)
        unmapped_mask = ~np.isin(lod2_labels, list(self.lod2_to_lod3.keys()))
        lod3_labels[unmapped_mask] = LOD3_CLASSES['other']
        
        return lod3_labels
    
    def _compute_mapping_confidence(
        self,
        source_labels: np.ndarray,
        target_labels: np.ndarray,
        level: str
    ) -> np.ndarray:
        """
        Compute confidence scores for label mapping.
        
        Confidence is based on:
        - Unambiguous mappings get high confidence (0.9)
        - Ambiguous mappings (many-to-one) get medium confidence (0.6)
        - Default/fallback mappings get low confidence (0.4)
        """
        n_points = len(source_labels)
        confidence = np.ones(n_points)
        
        # Determine mapping dictionary
        if level == 'LOD2':
            mapping = self.asprs_to_lod2
            default_class = LOD2_CLASSES['other']
        elif level == 'LOD3':
            mapping = self.lod2_to_lod3
            default_class = LOD3_CLASSES['other']
        else:
            return confidence
        
        # Analyze mapping ambiguity
        target_to_sources = {}
        for src, tgt in mapping.items():
            if tgt not in target_to_sources:
                target_to_sources[tgt] = []
            target_to_sources[tgt].append(src)
        
        # Assign confidence based on mapping type
        for i in range(n_points):
            src = source_labels[i]
            tgt = target_labels[i]
            
            if tgt == default_class:
                # Fallback mapping
                confidence[i] = 0.4
            elif src in mapping:
                expected_tgt = mapping[src]
                if tgt == expected_tgt:
                    # Direct mapping
                    num_sources = len(target_to_sources.get(tgt, []))
                    if num_sources == 1:
                        # Unambiguous (1-to-1)
                        confidence[i] = 0.9
                    else:
                        # Ambiguous (many-to-1)
                        confidence[i] = 0.6
                else:
                    # Unexpected mapping (error?)
                    confidence[i] = 0.3
            else:
                # Source not in mapping
                confidence[i] = 0.5
        
        return confidence
    
    def _refine_classification(
        self,
        labels: np.ndarray,
        features: Dict[str, np.ndarray],
        ground_truth: Optional[Dict[str, Any]],
        confidence_scores: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, int, Dict[str, float]]:
        """
        Refine classification using geometric features and ground truth.
        
        Returns:
            Tuple of (refined_labels, num_refined, feature_importance)
        """
        refined = labels.copy()
        initial_labels = labels.copy()
        feature_importance = {}
        
        # Track refinements by feature source
        refined_by_height = 0
        refined_by_ndvi = 0
        refined_by_geometry = 0
        refined_by_ground_truth = 0
        
        # 1. Height-based refinement
        if 'height' in features:
            height = features['height']
            refined, count = self._refine_by_height(refined, height)
            refined_by_height = count
            feature_importance['height'] = count / len(labels)
        
        # 2. NDVI-based refinement (vegetation)
        if 'ndvi' in features:
            ndvi = features['ndvi']
            refined, count = self._refine_by_ndvi(
                refined, ndvi, features.get('height')
            )
            refined_by_ndvi = count
            feature_importance['ndvi'] = count / len(labels)
        
        # 3. Geometry-based refinement (planarity, normals, curvature)
        geom_features = {}
        for key in ['planarity', 'normals', 'curvature', 'intensity']:
            if key in features:
                geom_features[key] = features[key]
        
        if geom_features:
            refined, count = self._refine_by_geometry(
                refined, geom_features, features.get('height')
            )
            refined_by_geometry = count
            feature_importance['geometry'] = count / len(labels)
        
        # 4. Ground truth refinement (highest priority)
        if ground_truth is not None:
            refined, count = self._refine_by_ground_truth(
                refined, ground_truth, features
            )
            refined_by_ground_truth = count
            feature_importance['ground_truth'] = count / len(labels)
        
        # Total refined points
        num_refined = np.sum(refined != initial_labels)
        
        # Update confidence scores for refined points
        if confidence_scores is not None:
            changed_mask = refined != initial_labels
            # Boost confidence for ground truth refinements
            if refined_by_ground_truth > 0:
                confidence_scores[changed_mask] = np.maximum(
                    confidence_scores[changed_mask], 0.85
                )
        
        logger.debug(f"Refinement summary: height={refined_by_height}, "
                    f"ndvi={refined_by_ndvi}, geometry={refined_by_geometry}, "
                    f"ground_truth={refined_by_ground_truth}")
        
        return refined, num_refined, feature_importance
    
    def _refine_by_height(
        self,
        labels: np.ndarray,
        height: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """Refine classification using height information."""
        refined = labels.copy()
        changed = 0
        
        # Get class IDs for current level
        if self.target_level == ClassificationLevel.LOD2:
            ground_id = LOD2_CLASSES['ground']
            veg_low_id = LOD2_CLASSES['vegetation_low']
            veg_high_id = LOD2_CLASSES['vegetation_high']
        elif self.target_level == ClassificationLevel.LOD3:
            ground_id = LOD3_CLASSES['ground']
            veg_low_id = LOD3_CLASSES['vegetation_low']
            veg_high_id = LOD3_CLASSES['vegetation_high']
        else:
            return refined, 0
        
        # Height-based rules
        # Very low points -> ground
        very_low_mask = (height < 0.2) & (labels != ground_id)
        refined[very_low_mask] = ground_id
        changed += np.sum(very_low_mask)
        
        # Low points classified as high veg -> low veg
        low_height_high_veg = (height < 2.0) & (labels == veg_high_id)
        refined[low_height_high_veg] = veg_low_id
        changed += np.sum(low_height_high_veg)
        
        # High points classified as low veg -> high veg
        high_height_low_veg = (height > 3.0) & (labels == veg_low_id)
        refined[high_height_low_veg] = veg_high_id
        changed += np.sum(high_height_low_veg)
        
        return refined, changed
    
    def _refine_by_ndvi(
        self,
        labels: np.ndarray,
        ndvi: np.ndarray,
        height: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, int]:
        """Refine classification using NDVI values."""
        refined = labels.copy()
        changed = 0
        
        # Get class IDs
        if self.target_level == ClassificationLevel.LOD2:
            veg_low_id = LOD2_CLASSES['vegetation_low']
            veg_high_id = LOD2_CLASSES['vegetation_high']
            building_id = LOD2_CLASSES['wall']
        elif self.target_level == ClassificationLevel.LOD3:
            veg_low_id = LOD3_CLASSES['vegetation_low']
            veg_high_id = LOD3_CLASSES['vegetation_high']
            building_id = LOD3_CLASSES['wall_plain']
        else:
            return refined, 0
        
        # High NDVI -> vegetation
        high_ndvi_mask = (ndvi > 0.4) & ~np.isin(labels, [veg_low_id, veg_high_id])
        if height is not None:
            # Distinguish low/high vegetation by height
            refined[high_ndvi_mask & (height < 2.0)] = veg_low_id
            refined[high_ndvi_mask & (height >= 2.0)] = veg_high_id
        else:
            refined[high_ndvi_mask] = veg_low_id  # Default to low veg
        changed += np.sum(high_ndvi_mask)
        
        # Low NDVI vegetation -> building (likely misclassification)
        low_ndvi_veg = (ndvi < 0.2) & np.isin(labels, [veg_low_id, veg_high_id])
        refined[low_ndvi_veg] = building_id
        changed += np.sum(low_ndvi_veg)
        
        return refined, changed
    
    def _refine_by_geometry(
        self,
        labels: np.ndarray,
        geom_features: Dict[str, np.ndarray],
        height: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, int]:
        """Refine classification using geometric features."""
        refined = labels.copy()
        changed = 0
        
        # Get class IDs
        if self.target_level == ClassificationLevel.LOD2:
            ground_id = LOD2_CLASSES['ground']
            roof_flat_id = LOD2_CLASSES['roof_flat']
            wall_id = LOD2_CLASSES['wall']
        elif self.target_level == ClassificationLevel.LOD3:
            ground_id = LOD3_CLASSES['ground']
            roof_flat_id = LOD3_CLASSES['roof_flat']
            wall_id = LOD3_CLASSES['wall_plain']
        else:
            return refined, 0
        
        planarity = geom_features.get('planarity')
        normals = geom_features.get('normals')
        
        # High planarity + horizontal + elevated -> roof
        if planarity is not None and normals is not None and height is not None:
            # Horizontal surfaces (normal points up)
            horizontal_mask = normals[:, 2] > 0.9  # Z component > 0.9
            
            # High planarity + elevated + horizontal -> roof
            roof_mask = (planarity > 0.8) & horizontal_mask & (height > 3.0)
            refined[roof_mask] = roof_flat_id
            changed += np.sum(roof_mask)
            
            # High planarity + low height + horizontal -> ground
            ground_mask = (planarity > 0.85) & horizontal_mask & (height < 0.5)
            refined[ground_mask] = ground_id
            changed += np.sum(ground_mask)
            
            # Vertical surfaces + elevated -> wall
            vertical_mask = normals[:, 2] < 0.3  # Not horizontal
            wall_mask = (planarity > 0.5) & vertical_mask & (height > 2.0)
            refined[wall_mask] = wall_id
            changed += np.sum(wall_mask)
        
        return refined, changed
    
    def _refine_by_ground_truth(
        self,
        labels: np.ndarray,
        ground_truth: Dict[str, Any],
        features: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, int]:
        """Refine classification using ground truth data (highest priority)."""
        # This is a placeholder - implement based on ground truth format
        # Should use point-in-polygon tests and spatial joins
        refined = labels.copy()
        changed = 0
        
        # Ground truth refinement would go here
        # Example: Check if points are within building polygons
        # Example: Check if points are on road centerlines with buffer
        
        logger.debug("Ground truth refinement not yet implemented in hierarchical classifier")
        
        return refined, changed


# ============================================================================
# Helper Functions
# ============================================================================

def classify_hierarchical(
    asprs_labels: np.ndarray,
    target_level: str = 'LOD2',
    features: Optional[Dict[str, np.ndarray]] = None,
    ground_truth: Optional[Dict[str, Any]] = None,
    use_confidence: bool = True,
    track_hierarchy: bool = False
) -> ClassificationResult:
    """
    Convenience function for hierarchical classification.
    
    Args:
        asprs_labels: Initial ASPRS classification [N]
        target_level: Target classification level ('ASPRS', 'LOD2', or 'LOD3')
        features: Optional geometric features for refinement
        ground_truth: Optional ground truth data for refinement
        use_confidence: Whether to compute confidence scores
        track_hierarchy: Whether to track classification hierarchy
    
    Returns:
        ClassificationResult with labels and metadata
    """
    # Parse level
    level_map = {
        'ASPRS': ClassificationLevel.ASPRS,
        'LOD2': ClassificationLevel.LOD2,
        'LOD3': ClassificationLevel.LOD3
    }
    level = level_map.get(target_level.upper())
    if level is None:
        raise ValueError(f"Invalid target level: {target_level}. "
                        f"Must be one of {list(level_map.keys())}")
    
    # Create classifier and run
    classifier = HierarchicalClassifier(
        target_level=level,
        use_confidence_scores=use_confidence
    )
    
    result = classifier.classify(
        asprs_labels=asprs_labels,
        features=features,
        ground_truth=ground_truth,
        track_hierarchy=track_hierarchy
    )
    
    return result


def get_class_name(class_id: int, level: ClassificationLevel) -> str:
    """Get human-readable class name for a class ID at given level."""
    if level == ClassificationLevel.ASPRS:
        try:
            return ASPRSClass(class_id).name
        except ValueError:
            return f"ASPRS_{class_id}"
    
    elif level == ClassificationLevel.LOD2:
        for name, cid in LOD2_CLASSES.items():
            if cid == class_id:
                return name
        return f"LOD2_{class_id}"
    
    elif level == ClassificationLevel.LOD3:
        for name, cid in LOD3_CLASSES.items():
            if cid == class_id:
                return name
        return f"LOD3_{class_id}"
    
    return f"unknown_{class_id}"
