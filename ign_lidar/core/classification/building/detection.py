"""
Building Detection Module with Multi-Mode Support

This module provides enhanced building detection capabilities for different
classification modes: ASPRS, LOD2, and LOD3. Each mode uses tailored detection
strategies and thresholds optimized for their specific use cases.

Modes:
- ASPRS: General building detection for ASPRS classification (code 6)
- LOD2: Building element detection for LOD2 training (walls, roofs)
- LOD3: Detailed architectural element detection (windows, doors, balconies)

Author: Building Detection Enhancement
Date: October 15, 2025
"""

import logging
from ..constants import ASPRSClass
from typing import Optional, Dict, Any, Tuple
from ..constants import ASPRSClass
from enum import Enum
from ..constants import ASPRSClass
import numpy as np
from ..constants import ASPRSClass

logger = logging.getLogger(__name__)


# ============================================================================
# Building Detection Modes
# ============================================================================


class BuildingDetectionMode(str, Enum):
    """Building detection modes with different strategies and outputs."""

    ASPRS = "asprs"  # General building detection (ASPRS class 6)
    LOD2 = "lod2"  # LOD2 building elements (walls, roofs)
    LOD3 = "lod3"  # LOD3 detailed elements (windows, doors, etc.)


# ============================================================================
# Configuration Classes
# ============================================================================


class BuildingDetectionConfig:
    """Configuration for building detection with mode-specific thresholds."""

    def __init__(self, mode: BuildingDetectionMode = BuildingDetectionMode.ASPRS):
        """
        Initialize building detection configuration.

        Args:
            mode: Detection mode (ASPRS, LOD2, or LOD3)
        """
        self.mode = mode

        # === Common Thresholds (used by all modes) ===
        # v6.1: Lowered min_height to catch low walls and annexes
        self.min_height = 1.5  # Minimum building height (meters)
        self.max_height = 200.0  # Maximum building height (meters)
        self.min_planarity = 0.4  # Lower for rough/textured surfaces

        # === Mode-specific Thresholds ===
        if mode == BuildingDetectionMode.ASPRS:
            # ASPRS: Simple binary detection (building vs non-building)
            # ENHANCED: Near-vertical wall detection with buffer extension
            # v6.1: More aggressive facade detection
            self.wall_verticality_min = 0.50  # Further lowered to catch rougher facades
            self.wall_planarity_min = 0.35  # Lower for textured/windowed facades
            self.wall_buffer_distance = 0.3  # NEW: Extend to wall boundaries (meters)
            self.roof_horizontality_min = 0.80
            self.roof_planarity_min = 0.65
            self.anisotropy_min = 0.40  # Lower to catch more structured surfaces
            self.linearity_edge_min = 0.30  # Lower for more edge detection
            self.use_ground_truth = True
            self.ground_truth_priority = True
            self.use_polygon_buffer = True  # NEW: Use building polygon buffers

        elif mode == BuildingDetectionMode.LOD2:
            # LOD2: Separate walls, roofs, other building elements
            # ENHANCED: Improved wall detection with near-vertical tolerance
            self.wall_verticality_min = 0.65  # Lowered from 0.70 for near-vertical
            self.wall_planarity_min = 0.50  # Lowered from 0.55 for textured walls
            self.wall_score_min = 0.30  # Lowered from 0.35 (planarity × verticality)
            self.wall_buffer_distance = 0.4  # NEW: Extend to wall boundaries
            self.roof_horizontality_min = 0.85
            self.roof_planarity_min = 0.70
            self.roof_score_min = 0.50  # planarity × horizontality
            self.anisotropy_min = 0.50
            self.linearity_edge_min = 0.40
            self.use_ground_truth = True
            self.ground_truth_priority = True
            self.use_polygon_buffer = True  # NEW: Use building polygon buffers
            # LOD2-specific
            self.detect_flat_roofs = True
            self.detect_sloped_roofs = True
            self.separate_walls_roofs = True

        elif mode == BuildingDetectionMode.LOD3:
            # LOD3: Detailed architectural elements
            self.wall_verticality_min = 0.75
            self.wall_planarity_min = 0.60
            self.wall_score_min = 0.40
            self.roof_horizontality_min = 0.85
            self.roof_planarity_min = 0.75
            self.roof_score_min = 0.55
            self.anisotropy_min = 0.55
            self.linearity_edge_min = 0.45
            self.use_ground_truth = True
            self.ground_truth_priority = True
            # LOD3-specific
            self.detect_windows = True
            self.detect_doors = True
            self.detect_balconies = True
            self.detect_chimneys = True
            self.detect_dormers = True
            self.opening_intensity_threshold = 0.25  # Low intensity for openings
            self.opening_depth_threshold = 0.15  # Recessed openings
            self.balcony_linearity_min = 0.35
            self.chimney_height_min = 1.5
            self.dormer_detection_enabled = True

        # === Detection Strategies (flags) ===
        self.use_wall_detection = True
        self.use_roof_detection = True
        self.use_anisotropy_detection = True
        self.use_edge_detection = True
        self.use_combined_score = True


# ============================================================================
# Building Detection Strategies
# ============================================================================


class BuildingDetector:
    """
    Advanced building detector supporting multiple detection modes.

    Implements different strategies optimized for:
    - ASPRS: General purpose building classification
    - LOD2: Building reconstruction training (walls, roofs)
    - LOD3: Detailed architectural modeling (windows, doors, etc.)
    """

    def __init__(self, config: Optional[BuildingDetectionConfig] = None):
        """
        Initialize building detector.

        Args:
            config: Detection configuration (defaults to ASPRS mode)
        """
        self.config = config or BuildingDetectionConfig(
            mode=BuildingDetectionMode.ASPRS
        )
        logger.info(
            f"🏢 Building Detector initialized in {self.config.mode.upper()} mode"
        )

    def detect_buildings(
        self,
        labels: np.ndarray,
        height: np.ndarray,
        planarity: np.ndarray,
        verticality: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        linearity: Optional[np.ndarray] = None,
        anisotropy: Optional[np.ndarray] = None,
        curvature: Optional[np.ndarray] = None,
        intensity: Optional[np.ndarray] = None,
        wall_score: Optional[np.ndarray] = None,
        roof_score: Optional[np.ndarray] = None,
        ground_truth_mask: Optional[np.ndarray] = None,
        points: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Detect buildings using mode-appropriate strategies.

        Applies multi-strategy building detection optimized for different
        classification modes: ASPRS (general), LOD2 (training), LOD3 (detailed).

        Args:
            labels: Current classification labels [N]. Points already classified
                   as buildings will be preserved or refined.
            height: Height above ground [N] in meters. Used for height-based
                   filtering (min 2.5m, max 200m by default).
            planarity: Planarity values [N] in range [0, 1]. Indicates how
                      flat/planar a surface is (1.0 = perfectly planar).
            verticality: Verticality values [N] in range [0, 1]. Indicates
                        vertical orientation (1.0 = perfectly vertical).
                        Required for wall detection.
            normals: Surface normals [N, 3]. Unit vectors indicating surface
                    orientation. Z-component used for horizontality.
            linearity: Linearity values [N] in range [0, 1]. Indicates
                      linear/edge-like features (building edges).
            anisotropy: Anisotropy values [N] in range [0, 1]. Indicates
                       structured/organized point distribution.
            curvature: Surface curvature [N]. Used for LOD3 element detection
                      (chimneys, dormers have high curvature).
            intensity: LiDAR return intensity [N], normalized to [0, 1].
                      Different materials have different intensities.
            wall_score: Pre-computed wall likelihood [N] = planarity × verticality.
                       Optional optimization to avoid recomputation.
            roof_score: Pre-computed roof likelihood [N] = planarity × horizontality.
                       Optional optimization to avoid recomputation.
            ground_truth_mask: Boolean mask [N] indicating known building points
                              from external data (BD TOPO, cadastre). Highest
                              confidence source.
            points: Point coordinates [N, 3] in meters. Required for LOD3
                   spatial analysis and clustering.

        Returns:
            Tuple of (refined_labels, detection_stats) where:
            - refined_labels: Updated classification labels [N]
            - detection_stats: Dictionary with detection statistics:
              - 'ground_truth': Points from ground truth data
              - 'walls': Points detected as walls
              - 'roofs': Points detected as roofs
              - 'structured': Points from structure detection
              - 'edges': Points from edge detection
              - 'total': Total building points detected

        Example:
            Basic ASPRS building detection with minimal features:

            >>> import numpy as np
            >>> from ign_lidar.core.classification.building import (
            ...     BuildingDetector, BuildingDetectionMode
            ... )
            >>>
            >>> # Create detector
            >>> detector = BuildingDetector()  # Default: ASPRS mode
            >>>
            >>> # Minimal required features
            >>> n = 1000
            >>> labels = np.ones(n, dtype=int)  # All unclassified initially
            >>> height = np.random.rand(n) * 15  # Heights 0-15m
            >>> planarity = np.random.rand(n)
            >>>
            >>> # Detect buildings
            >>> refined_labels, stats = detector.detect_buildings(
            ...     labels, height, planarity
            ... )
            >>>
            >>> print(f"Detected {stats['total']} building points")
            >>> print(f"Wall points: {stats['walls']}")
            >>> print(f"Roof points: {stats['roofs']}")

            Advanced LOD2 detection with ground truth:

            >>> from ign_lidar.core.classification.building import (
            ...     BuildingDetector, BuildingDetectionConfig,
            ...     BuildingDetectionMode
            ... )
            >>>
            >>> # Configure for LOD2 mode
            >>> config = BuildingDetectionConfig(mode=BuildingDetectionMode.LOD2)
            >>> detector = BuildingDetector(config=config)
            >>>
            >>> # Full feature set
            >>> height = np.random.rand(1000) * 20
            >>> planarity = np.random.rand(1000)
            >>> verticality = np.random.rand(1000)
            >>> normals = np.random.randn(1000, 3)
            >>> normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
            >>>
            >>> # Ground truth from cadastre
            >>> building_indices = np.random.choice(1000, 300, replace=False)
            >>> ground_truth = np.zeros(1000, dtype=bool)
            >>> ground_truth[building_indices] = True
            >>>
            >>> # Detect with all features
            >>> labels = np.ones(1000, dtype=int)
            >>> refined, stats = detector.detect_buildings(
            ...     labels=labels,
            ...     height=height,
            ...     planarity=planarity,
            ...     verticality=verticality,
            ...     normals=normals,
            ...     ground_truth_mask=ground_truth
            ... )
            >>>
            >>> print(f"Ground truth: {stats['ground_truth']} points")
            >>> print(f"Geometric detection: {stats['walls'] + stats['roofs']} points")
            >>> print(f"Total buildings: {stats['total']} points")

            LOD3 detection with spatial clustering:

            >>> # Configure for LOD3 (detailed elements)
            >>> config = BuildingDetectionConfig(mode=BuildingDetectionMode.LOD3)
            >>> detector = BuildingDetector(config=config)
            >>>
            >>> # Add spatial coordinates for clustering
            >>> points = np.random.rand(1000, 3) * 100
            >>> intensity = np.random.rand(1000)
            >>> curvature = np.random.rand(1000) * 0.5
            >>>
            >>> refined, stats = detector.detect_buildings(
            ...     labels=labels,
            ...     height=height,
            ...     planarity=planarity,
            ...     verticality=verticality,
            ...     normals=normals,
            ...     intensity=intensity,
            ...     curvature=curvature,
            ...     points=points
            ... )
            >>>
            >>> # LOD3 provides element-level classification
            >>> # (walls, roofs, windows, doors, balconies, etc.)

        Note:
            Detection strategies by mode:

            **ASPRS Mode:**
            - Binary classification (building vs non-building)
            - Uses: ground truth → walls → roofs → structure → edges
            - Output: All building points get ASPRS code 6

            **LOD2 Mode:**
            - Element classification (walls, roofs, other)
            - Uses: ground truth → geometric elements
            - Output: Separate codes for walls, roofs
            - Purpose: Training data for LOD2 reconstruction

            **LOD3 Mode:**
            - Detailed element detection
            - Uses: ground truth → geometry → clustering → intensity
            - Output: Detailed elements (windows, doors, balconies, etc.)
            - Purpose: Architectural detail modeling

            **Ground Truth Priority:**
            When ground_truth_mask is provided and ground_truth_priority=True,
            those points are classified with highest confidence regardless of
            geometric features.

            **Feature Importance:**
            - Height + planarity: Minimum required
            - Verticality: Essential for wall detection
            - Normals: Improves roof detection accuracy
            - Linearity/anisotropy: Catches building edges/structure
            - Intensity: LOD3 material discrimination
            - Points: LOD3 spatial clustering

        See Also:
            - BuildingDetectionConfig: Configuration options
            - BuildingDetectionMode: Available detection modes
            - AdaptiveBuildingClassifier: Alternative adaptive approach
        """
        if self.config.mode == BuildingDetectionMode.ASPRS:
            return self._detect_asprs(
                labels,
                height,
                planarity,
                verticality,
                normals,
                linearity,
                anisotropy,
                wall_score,
                roof_score,
                ground_truth_mask,
            )
        elif self.config.mode == BuildingDetectionMode.LOD2:
            return self._detect_lod2(
                labels,
                height,
                planarity,
                verticality,
                normals,
                linearity,
                anisotropy,
                wall_score,
                roof_score,
                ground_truth_mask,
            )
        elif self.config.mode == BuildingDetectionMode.LOD3:
            return self._detect_lod3(
                labels,
                height,
                planarity,
                verticality,
                normals,
                linearity,
                anisotropy,
                curvature,
                intensity,
                wall_score,
                roof_score,
                ground_truth_mask,
                points,
            )
        else:
            raise ValueError(f"Unsupported detection mode: {self.config.mode}")

    def _detect_asprs(
        self,
        labels: np.ndarray,
        height: np.ndarray,
        planarity: np.ndarray,
        verticality: Optional[np.ndarray],
        normals: Optional[np.ndarray],
        linearity: Optional[np.ndarray],
        anisotropy: Optional[np.ndarray],
        wall_score: Optional[np.ndarray],
        roof_score: Optional[np.ndarray],
        ground_truth_mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        ASPRS mode: Simple building detection (all building points → class 6).

        Detection strategies:
        1. Ground truth (if available)
        2. Wall detection (vertical planar surfaces)
        3. Roof detection (horizontal planar surfaces)
        4. Structure detection (anisotropy)
        5. Edge detection (linear features)
        """
        refined = labels.copy()
        stats = {
            "ground_truth": 0,
            "walls": 0,
            "roofs": 0,
            "structured": 0,
            "edges": 0,
            "total": 0,
        }

        

        # Strategy 1: Ground truth (highest priority)
        if ground_truth_mask is not None and self.config.use_ground_truth:
            if self.config.ground_truth_priority:
                building_points = ground_truth_mask & (refined != int(ASPRSClass.BUILDING))
                refined[building_points] = int(ASPRSClass.BUILDING)
                stats["ground_truth"] = building_points.sum()

        # Strategy 2: Wall detection
        if self.config.use_wall_detection and verticality is not None:
            wall_candidates = (
                (height > self.config.min_height)
                & (height < self.config.max_height)
                & (verticality > self.config.wall_verticality_min)
                & (planarity > self.config.wall_planarity_min)
                & (refined != int(ASPRSClass.BUILDING))
            )

            if wall_score is not None:
                wall_candidates = wall_candidates & (
                    wall_score > self.config.wall_score_min
                )

            refined[wall_candidates] = int(ASPRSClass.BUILDING)
            stats["walls"] = wall_candidates.sum()

        # Strategy 3: Roof detection
        if self.config.use_roof_detection and normals is not None:
            horizontality = np.abs(normals[:, 2])

            roof_candidates = (
                (height > self.config.min_height)
                & (height < self.config.max_height)
                & (horizontality > self.config.roof_horizontality_min)
                & (planarity > self.config.roof_planarity_min)
                & (refined != int(ASPRSClass.BUILDING))
            )

            if roof_score is not None:
                roof_candidates = roof_candidates & (
                    roof_score > self.config.roof_score_min
                )

            refined[roof_candidates] = int(ASPRSClass.BUILDING)
            stats["roofs"] = roof_candidates.sum()

        # Strategy 4: Structure detection (anisotropy)
        if self.config.use_anisotropy_detection and anisotropy is not None:
            structured_candidates = (
                (height > self.config.min_height)
                & (height < self.config.max_height)
                & (anisotropy > self.config.anisotropy_min)
                & (planarity > 0.3)
                & (refined != int(ASPRSClass.BUILDING))
            )

            refined[structured_candidates] = int(ASPRSClass.BUILDING)
            stats["structured"] = structured_candidates.sum()

        # Strategy 5: Edge detection
        if self.config.use_edge_detection and linearity is not None:
            edge_candidates = (
                (height > self.config.min_height)
                & (height < self.config.max_height)
                & (linearity > self.config.linearity_edge_min)
                & (refined != int(ASPRSClass.BUILDING))
            )

            if verticality is not None:
                # Vertical or horizontal edges only
                edge_candidates = edge_candidates & (
                    (verticality > 0.7) | (verticality < 0.3)
                )

            refined[edge_candidates] = int(ASPRSClass.BUILDING)
            stats["edges"] = edge_candidates.sum()

        # Strategy 6: Aggressive facade detection for rough/textured walls
        # NEW v6.1: Catch semi-vertical surfaces (missing facades)
        if verticality is not None and normals is not None:
            # More lenient facade detection for rough walls with windows/texture
            facade_candidates = (
                (height > 0.8)  # Lower height for low walls/annexes
                & (height < self.config.max_height)
                & (verticality > 0.40)  # Semi-vertical OK
                & (planarity > 0.25)  # Very low for rough/windowed facades
                & (refined != int(ASPRSClass.BUILDING))
            )

            # Additional check: prefer structured surfaces
            if anisotropy is not None:
                # Accept if either fairly vertical OR shows structure
                structured = anisotropy > 0.35
                facade_candidates = facade_candidates & (
                    (verticality > 0.45) | structured
                )

            if facade_candidates.any():
                refined[facade_candidates] = int(ASPRSClass.BUILDING)
                stats["rough_facades"] = facade_candidates.sum()

        # Strategy 7: Handle unclassified building-like points
        # This catches points that might be on building edges or have partial geometric features
        unclassified_mask = (refined == 1) | (refined == 0)  # ASPRS unclassified codes

        if unclassified_mask.any():
            # Check for building-like characteristics among unclassified
            building_like = (
                (height > self.config.min_height)
                & (height < self.config.max_height)
                & (planarity > 0.5)  # Some planarity
                & unclassified_mask
            )

            # Additional validation with available features
            if normals is not None:
                # Must have strong vertical or horizontal orientation
                horizontality = np.abs(normals[:, 2])
                building_orientation = (horizontality > 0.75) | (horizontality < 0.35)
                building_like = building_like & building_orientation

            if anisotropy is not None:
                # Must show some structural organization
                building_like = building_like & (anisotropy > 0.35)

            # Apply with slightly lower confidence
            if building_like.any():
                refined[building_like] = int(ASPRSClass.BUILDING)
                stats["unclassified_recovery"] = building_like.sum()

        stats["total"] = (refined == int(ASPRSClass.BUILDING)).sum()

        return refined, stats

    def _detect_lod2(
        self,
        labels: np.ndarray,
        height: np.ndarray,
        planarity: np.ndarray,
        verticality: Optional[np.ndarray],
        normals: Optional[np.ndarray],
        linearity: Optional[np.ndarray],
        anisotropy: Optional[np.ndarray],
        wall_score: Optional[np.ndarray],
        roof_score: Optional[np.ndarray],
        ground_truth_mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        LOD2 mode: Detect building elements (walls, roofs, other).

        LOD2 Classes:
        - 0: Wall
        - 1: Roof (flat)
        - 2: Roof (gable)
        - 3: Roof (hip)
        - 9: Ground
        - 14: Other

        Detection strategies prioritize separating walls from roofs.
        """
        refined = labels.copy()
        stats = {
            "ground_truth": 0,
            "walls": 0,
            "flat_roofs": 0,
            "sloped_roofs": 0,
            "structured": 0,
            "edges": 0,
            "total_building": 0,
        }

        LOD2_WALL = 0
        LOD2_ROOF_FLAT = 1
        LOD2_ROOF_GABLE = 2
        LOD2_ROOF_HIP = 3

        # Strategy 1: Ground truth
        if ground_truth_mask is not None and self.config.use_ground_truth:
            if self.config.ground_truth_priority:
                # Ground truth gives general building mask, classify as walls initially
                building_points = ground_truth_mask & (refined != LOD2_WALL)
                refined[building_points] = LOD2_WALL
                stats["ground_truth"] = building_points.sum()

        # Strategy 2: Wall detection (vertical planar surfaces)
        if self.config.use_wall_detection and verticality is not None:
            wall_candidates = (
                (height > self.config.min_height)
                & (height < self.config.max_height)
                & (verticality > self.config.wall_verticality_min)
                & (planarity > self.config.wall_planarity_min)
            )

            if wall_score is not None:
                wall_candidates = wall_candidates & (
                    wall_score > self.config.wall_score_min
                )

            # Only update if not already classified as building
            wall_updates = wall_candidates & (refined != LOD2_WALL)
            refined[wall_updates] = LOD2_WALL
            stats["walls"] = wall_updates.sum()

        # Strategy 3: Roof detection (horizontal planar surfaces)
        if self.config.use_roof_detection and normals is not None:
            horizontality = np.abs(normals[:, 2])

            # Base roof candidates
            roof_candidates = (
                (height > self.config.min_height)
                & (height < self.config.max_height)
                & (horizontality > self.config.roof_horizontality_min)
                & (planarity > self.config.roof_planarity_min)
            )

            if roof_score is not None:
                roof_candidates = roof_candidates & (
                    roof_score > self.config.roof_score_min
                )

            # Classify roof type based on horizontality
            # Flat roofs: very horizontal
            flat_roof_mask = roof_candidates & (horizontality > 0.95)
            refined[flat_roof_mask] = LOD2_ROOF_FLAT
            stats["flat_roofs"] = flat_roof_mask.sum()

            # Sloped roofs: moderately horizontal
            sloped_roof_mask = (
                roof_candidates & (horizontality <= 0.95) & (horizontality > 0.85)
            )
            # Default to gable for sloped roofs (can be refined with geometry analysis)
            refined[sloped_roof_mask] = LOD2_ROOF_GABLE
            stats["sloped_roofs"] = sloped_roof_mask.sum()

        # Strategy 4: Structure detection for ambiguous cases
        if self.config.use_anisotropy_detection and anisotropy is not None:
            structured_candidates = (
                (height > self.config.min_height)
                & (height < self.config.max_height)
                & (anisotropy > self.config.anisotropy_min)
                & (planarity > 0.3)
                & (refined != LOD2_WALL)
                & (refined != LOD2_ROOF_FLAT)
                & (refined != LOD2_ROOF_GABLE)
                & (refined != LOD2_ROOF_HIP)
            )

            refined[structured_candidates] = LOD2_WALL
            stats["structured"] = structured_candidates.sum()

        # Strategy 5: Edge detection (corners, boundaries)
        if self.config.use_edge_detection and linearity is not None:
            edge_candidates = (
                (height > self.config.min_height)
                & (height < self.config.max_height)
                & (linearity > self.config.linearity_edge_min)
                & (refined != LOD2_WALL)
                & (refined != LOD2_ROOF_FLAT)
                & (refined != LOD2_ROOF_GABLE)
                & (refined != LOD2_ROOF_HIP)
            )

            # Classify edges as walls by default
            refined[edge_candidates] = LOD2_WALL
            stats["edges"] = edge_candidates.sum()

        # Count total building points
        building_classes = [LOD2_WALL, LOD2_ROOF_FLAT, LOD2_ROOF_GABLE, LOD2_ROOF_HIP]
        stats["total_building"] = np.isin(refined, building_classes).sum()

        return refined, stats

    def _detect_lod3(
        self,
        labels: np.ndarray,
        height: np.ndarray,
        planarity: np.ndarray,
        verticality: Optional[np.ndarray],
        normals: Optional[np.ndarray],
        linearity: Optional[np.ndarray],
        anisotropy: Optional[np.ndarray],
        curvature: Optional[np.ndarray],
        intensity: Optional[np.ndarray],
        wall_score: Optional[np.ndarray],
        roof_score: Optional[np.ndarray],
        ground_truth_mask: Optional[np.ndarray],
        points: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        LOD3 mode: Detect detailed architectural elements.

        LOD3 Classes:
        - 0: Wall
        - 1: Roof (flat)
        - 2: Roof (gable)
        - 3: Roof (hip)
        - 13: Window
        - 14: Door
        - 15: Balcony
        - 18: Chimney
        - 20: Dormer

        Advanced detection for openings, projections, and details.
        """
        refined = labels.copy()
        stats = {
            "ground_truth": 0,
            "walls": 0,
            "flat_roofs": 0,
            "sloped_roofs": 0,
            "windows": 0,
            "doors": 0,
            "balconies": 0,
            "chimneys": 0,
            "dormers": 0,
            "total_building": 0,
        }

        LOD3_WALL = 0
        LOD3_ROOF_FLAT = 1
        LOD3_ROOF_GABLE = 2
        LOD3_ROOF_HIP = 3
        LOD3_WINDOW = 13
        LOD3_DOOR = 14
        LOD3_BALCONY = 15
        LOD3_CHIMNEY = 18
        LOD3_DORMER = 20

        # First pass: Detect base building elements (walls and roofs)
        # Similar to LOD2 but with stricter thresholds

        # Ground truth
        if ground_truth_mask is not None and self.config.use_ground_truth:
            if self.config.ground_truth_priority:
                building_points = ground_truth_mask & (refined != LOD3_WALL)
                refined[building_points] = LOD3_WALL
                stats["ground_truth"] = building_points.sum()

        # Wall detection
        if verticality is not None:
            wall_candidates = (
                (height > self.config.min_height)
                & (height < self.config.max_height)
                & (verticality > self.config.wall_verticality_min)
                & (planarity > self.config.wall_planarity_min)
            )

            if wall_score is not None:
                wall_candidates = wall_candidates & (
                    wall_score > self.config.wall_score_min
                )

            wall_updates = wall_candidates & (refined != LOD3_WALL)
            refined[wall_updates] = LOD3_WALL
            stats["walls"] = wall_updates.sum()

        # Roof detection
        if normals is not None:
            horizontality = np.abs(normals[:, 2])

            roof_candidates = (
                (height > self.config.min_height)
                & (height < self.config.max_height)
                & (horizontality > self.config.roof_horizontality_min)
                & (planarity > self.config.roof_planarity_min)
            )

            if roof_score is not None:
                roof_candidates = roof_candidates & (
                    roof_score > self.config.roof_score_min
                )

            # Flat roofs
            flat_roof_mask = roof_candidates & (horizontality > 0.95)
            refined[flat_roof_mask] = LOD3_ROOF_FLAT
            stats["flat_roofs"] = flat_roof_mask.sum()

            # Sloped roofs
            sloped_roof_mask = (
                roof_candidates & (horizontality <= 0.95) & (horizontality > 0.85)
            )
            refined[sloped_roof_mask] = LOD3_ROOF_GABLE
            stats["sloped_roofs"] = sloped_roof_mask.sum()

        # Second pass: Detect detailed architectural elements

        # Window detection (vertical surfaces with low intensity, openings)
        if (
            self.config.detect_windows
            and intensity is not None
            and verticality is not None
        ):
            window_candidates = (
                (height > 1.0)  # Above ground floor
                & (height < self.config.max_height - 1.0)  # Not on roof
                & (verticality > 0.6)  # On vertical surfaces
                & (
                    intensity < self.config.opening_intensity_threshold
                )  # Low reflection (glass)
                & (planarity > 0.4)  # Some planarity (flat glass)
                & (refined == LOD3_WALL)  # Currently classified as wall
            )

            refined[window_candidates] = LOD3_WINDOW
            stats["windows"] = window_candidates.sum()

        # Door detection (lower height, vertical surface with opening characteristics)
        if (
            self.config.detect_doors
            and intensity is not None
            and verticality is not None
        ):
            door_candidates = (
                (height > 0.2)
                & (height < 3.0)  # Ground level
                & (verticality > 0.65)
                & (intensity < 0.4)  # Darker than walls
                & (planarity > 0.5)
                & (refined == LOD3_WALL)
            )

            # Additional constraint: avoid classifying windows as doors
            if LOD3_WINDOW in refined:
                door_candidates = door_candidates & (refined != LOD3_WINDOW)

            refined[door_candidates] = LOD3_DOOR
            stats["doors"] = door_candidates.sum()

        # Balcony detection (horizontal projections from facades)
        if (
            self.config.detect_balconies
            and linearity is not None
            and normals is not None
        ):
            horizontality = np.abs(normals[:, 2])

            balcony_candidates = (
                (height > 2.0)  # Above ground
                & (height < self.config.max_height - 2.0)
                & (horizontality > 0.8)  # Horizontal surface
                & (linearity > self.config.balcony_linearity_min)  # Linear edge
                & (planarity > 0.6)
            )

            # Check if near wall points (spatial analysis)
            if balcony_candidates.any():
                refined[balcony_candidates] = LOD3_BALCONY
                stats["balconies"] = balcony_candidates.sum()

        # Chimney detection (vertical structures on roofs)
        if self.config.detect_chimneys and verticality is not None:
            # Chimneys: small vertical structures above roof level
            roof_classes = [LOD3_ROOF_FLAT, LOD3_ROOF_GABLE, LOD3_ROOF_HIP]

            chimney_candidates = (
                (height > self.config.min_height + self.config.chimney_height_min)
                & (verticality > 0.7)
                & (planarity > 0.5)
                & (anisotropy > 0.4)
                if anisotropy is not None
                else True
            )

            # Must be above roof height (heuristic)
            if chimney_candidates.any():
                refined[chimney_candidates] = LOD3_CHIMNEY
                stats["chimneys"] = chimney_candidates.sum()

        # Dormer detection (roof protrusions)
        if self.config.detect_dormers and self.config.dormer_detection_enabled:
            # Dormers: combination of vertical and sloped surfaces above main roof
            dormer_candidates = (
                (height > self.config.min_height + 1.0)
                & (verticality > 0.4)
                & (verticality < 0.8)  # Intermediate angle
                & (planarity > 0.5)
                & (anisotropy > 0.4)
                if anisotropy is not None
                else True
            )

            if dormer_candidates.any():
                refined[dormer_candidates] = LOD3_DORMER
                stats["dormers"] = dormer_candidates.sum()

        # Count total building points
        building_classes = [
            LOD3_WALL,
            LOD3_ROOF_FLAT,
            LOD3_ROOF_GABLE,
            LOD3_ROOF_HIP,
            LOD3_WINDOW,
            LOD3_DOOR,
            LOD3_BALCONY,
            LOD3_CHIMNEY,
            LOD3_DORMER,
        ]
        stats["total_building"] = np.isin(refined, building_classes).sum()

        return refined, stats


# ============================================================================
# Convenience Functions
# ============================================================================


def detect_buildings_multi_mode(
    labels: np.ndarray,
    features: Dict[str, np.ndarray],
    mode: str = "asprs",
    ground_truth_mask: Optional[np.ndarray] = None,
    config: Optional[BuildingDetectionConfig] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Convenience function for building detection with automatic mode selection.

    Args:
        labels: Current classification labels [N]
        features: Dictionary of computed features
        mode: Detection mode ('asprs', 'lod2', or 'lod3')
        ground_truth_mask: Optional ground truth building mask
        config: Optional custom configuration

    Returns:
        Tuple of (refined_labels, detection_stats)
    """
    # Parse mode
    if isinstance(mode, str):
        mode = mode.lower()
        if mode not in ["asprs", "lod2", "lod3"]:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'asprs', 'lod2', or 'lod3'"
            )
        mode_enum = BuildingDetectionMode(mode)
    else:
        mode_enum = mode

    # Create config if not provided
    if config is None:
        config = BuildingDetectionConfig(mode=mode_enum)

    # Create detector
    detector = BuildingDetector(config=config)

    # Extract features
    height = features.get("height")
    planarity = features.get("planarity")
    verticality = features.get("verticality")
    normals = features.get("normals")
    linearity = features.get("linearity")
    anisotropy = features.get("anisotropy")
    curvature = features.get("curvature")
    intensity = features.get("intensity")
    wall_score = features.get("wall_score")
    roof_score = features.get("roof_score")
    points = features.get("points")

    # Validate required features
    if height is None or planarity is None:
        raise ValueError(
            "Height and planarity features are required for building detection"
        )

    # Run detection
    return detector.detect_buildings(
        labels=labels,
        height=height,
        planarity=planarity,
        verticality=verticality,
        normals=normals,
        linearity=linearity,
        anisotropy=anisotropy,
        curvature=curvature,
        intensity=intensity,
        wall_score=wall_score,
        roof_score=roof_score,
        ground_truth_mask=ground_truth_mask,
        points=points,
    )


# ============================================================================
# Export
# ============================================================================

__all__ = [
    "BuildingDetectionMode",
    "BuildingDetectionConfig",
    "BuildingDetector",
    "detect_buildings_multi_mode",
]
