"""
Per-Facade Building Detection & Classification

Module avancé pour le traitement individuel de bâtiments avec analyse détaillée
de chaque façade de la bounding box. Permet une classification adaptative et
précise point par point.

Fonctionnalités:
1. ✅ Décomposition automatique de chaque bâtiment BD TOPO en 4 façades (N/S/E/W)
2. ✅ Traitement indépendant de chaque façade (buffer, classification)
3. ✅ Détection adaptative de points de mur par façade
4. ✅ Gestion des occlusions et gaps par façade
5. ✅ Scoring de qualité et confiance par façade
6. ✅ Fusion intelligente des résultats multi-façades

Architecture:
- FacadeSegment: Représente une façade individuelle avec ses paramètres
- FacadeProcessor: Traite chaque façade indépendamment
- BuildingFacadeClassifier: Orchestrateur pour l'ensemble du bâtiment

Author: Facade-Based Building Classification v5.5
Date: October 2025
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np

from . import utils

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from shapely.geometry import LineString, Point, Polygon

try:
    from shapely.affinity import translate
    from shapely.geometry import LineString, Point, Polygon, box
    from shapely.ops import unary_union

    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


class FacadeOrientation(str, Enum):
    """Orientation des façades."""

    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


@dataclass
class FacadeSegment:
    """
    Représente une façade individuelle d'un bâtiment.

    Une façade est définie par:
    - Une arête de la bounding box (LineString)
    - Une orientation (N/S/E/W)
    - Des paramètres de traitement adaptatifs
    - Ajustements géométriques indépendants
    """

    edge_line: "LineString"
    orientation: FacadeOrientation
    building_id: int

    # Paramètres géométriques
    length: float = 0.0
    centroid: Optional[np.ndarray] = None
    normal_vector: Optional[np.ndarray] = None  # Vecteur normal vers l'extérieur

    # 🆕 Ajustements géométriques par façade
    translation_offset: float = (
        0.0  # Offset perpendiculaire à la façade (+ = vers extérieur)
    )
    lateral_expansion: Tuple[float, float] = (
        0.0,
        0.0,
    )  # Extension (gauche, droite) le long de la façade
    adjusted_edge_line: Optional["LineString"] = None  # Arête ajustée après adaptation

    # 🆕 NEW: Rotation adaptation (v3.0.3)
    rotation_angle: float = 0.0  # Angle de rotation appliqué (radians)
    rotation_confidence: float = 0.0  # Confiance de la rotation (0-1)
    is_rotated: bool = False  # Indique si une rotation a été appliquée

    # 🆕 NEW: Scaling adaptation (v3.0.3)
    scale_factor: float = 1.0  # Facteur d'échelle appliqué
    scale_confidence: float = 0.0  # Confiance du scaling (0-1)
    is_scaled: bool = False  # Indique si un scaling a été appliqué

    # Paramètres de traitement adaptatifs
    buffer_distance: float = 2.0  # Buffer pour capturer les points
    search_radius: float = 3.0  # Rayon de recherche de voisins
    verticality_threshold: float = (
        0.55  # ✅ IMPROVED: Abaissé de 0.70→0.55 pour capturer plus de façades
    )

    # Points assignés à cette façade
    point_indices: Optional[np.ndarray] = None
    n_points: int = 0
    n_wall_points: int = 0

    # Statistiques de qualité
    coverage_ratio: float = 0.0  # Couverture de la façade (0-1)
    avg_verticality: float = 0.0  # Verticalité moyenne
    point_density: float = 0.0  # Points par m²
    confidence_score: float = 0.0  # Score de confiance (0-1)

    # Gaps détectés
    has_gaps: bool = False
    gap_segments: List[Tuple[float, float]] = field(default_factory=list)
    gap_ratio: float = 0.0

    # 🆕 Centroïde des points détectés
    point_cloud_centroid: Optional[np.ndarray] = None
    centroid_offset_distance: float = 0.0  # Distance entre edge et points

    # Flags
    needs_adaptive_buffer: bool = False
    needs_translation: bool = False
    needs_lateral_expansion: bool = False
    has_occlusion: bool = False
    is_processed: bool = False
    is_adapted: bool = False


class FacadeProcessor:
    """
    Processeur pour une façade individuelle.

    Responsabilités:
    1. Identifier les points appartenant à la façade
    2. Classifier les points (mur/non-mur)
    3. Détecter gaps et occlusions
    4. Calculer scores de qualité et confiance
    5. Recommander ajustements de paramètres
    """

    def __init__(
        self,
        facade: FacadeSegment,
        points: np.ndarray,
        heights: np.ndarray,
        normals: Optional[np.ndarray] = None,
        verticality: Optional[np.ndarray] = None,
        # Paramètres
        min_point_density: float = 50.0,
        gap_detection_resolution: float = 0.5,
        adaptive_buffer_range: Tuple[float, float] = (0.5, 8.0),
    ):
        """
        Initialiser le processeur de façade.

        Args:
            facade: Segment de façade à traiter
            points: Nuage de points complet [N, 3]
            heights: Hauteurs au-dessus du sol [N]
            normals: Normales [N, 3]
            verticality: Verticalité [N]
            min_point_density: Densité minimale attendue (pts/m²)
            gap_detection_resolution: Résolution pour détecter gaps (m)
            adaptive_buffer_range: (min, max) pour buffer adaptatif
        """
        self.facade = facade
        self.points = points
        self.heights = heights
        self.normals = normals
        self.verticality = verticality

        self.min_point_density = min_point_density
        self.gap_detection_resolution = gap_detection_resolution
        self.adaptive_buffer_min, self.adaptive_buffer_max = adaptive_buffer_range

        # État interne
        self._candidate_mask = None
        self._wall_mask = None

    def process(self, building_height: float = 10.0) -> FacadeSegment:
        """
        Traiter la façade complètement.

        Args:
            building_height: Hauteur du bâtiment (pour calculs)

        Returns:
            Façade mise à jour avec résultats
        """
        logger.debug(
            f"Processing {self.facade.orientation.value} facade (length={self.facade.length:.1f}m)"
        )

        # 1. Identifier les points candidats
        self._identify_candidate_points()

        if self.facade.n_points == 0:
            logger.debug(
                f"  No points found for {self.facade.orientation.value} facade"
            )
            self.facade.is_processed = True
            return self.facade

        # 2. Classifier les points (mur vs non-mur)
        self._classify_wall_points()

        # 3. Calculer statistiques de couverture
        self._compute_coverage_statistics(building_height)

        # 4. Détecter gaps et occlusions
        self._detect_gaps()

        # 5. Calculer score de confiance
        self._compute_confidence_score()

        # 6. Recommander ajustements si nécessaire
        self._recommend_adjustments()

        self.facade.is_processed = True

        logger.debug(
            f"  {self.facade.orientation.value}: "
            f"points={self.facade.n_points}, "
            f"walls={self.facade.n_wall_points}, "
            f"coverage={self.facade.coverage_ratio:.1%}, "
            f"confidence={self.facade.confidence_score:.2f}"
        )

        return self.facade

    def _identify_candidate_points(self):
        """
        Identifier les points proches de cette façade avec bounding box optimisée.

        Améliorations:
        - Bounding box orientée (OBB) au lieu de AABB pour mieux capturer façades obliques
        - Buffer adaptatif selon densité de points
        - Pré-filtrage spatial plus efficace
        """
        # Créer zone de recherche (buffer autour de l'arête)
        search_zone = self.facade.edge_line.buffer(self.facade.buffer_distance)

        # 🔥 AMÉLIORATION 1: Bounding box orientée (OBB) pour façades obliques
        # Calculer l'orientation de la façade
        coords = list(self.facade.edge_line.coords)
        if len(coords) >= 2:
            dx = coords[1][0] - coords[0][0]
            dy = coords[1][1] - coords[0][1]
            facade_angle = np.arctan2(dy, dx)

            # Créer OBB orientée selon la façade
            facade_center = np.array(self.facade.edge_line.centroid.coords[0])

            # Rotation inverse pour aligner avec axes
            cos_a = np.cos(-facade_angle)
            sin_a = np.sin(-facade_angle)

            # Transformer points dans repère de la façade
            points_centered = self.points[:, :2] - facade_center
            points_rotated = np.column_stack(
                [
                    points_centered[:, 0] * cos_a - points_centered[:, 1] * sin_a,
                    points_centered[:, 0] * sin_a + points_centered[:, 1] * cos_a,
                ]
            )

            # Calculer limites OBB dans repère façade
            half_length = self.facade.length / 2 + self.facade.buffer_distance
            half_width = self.facade.buffer_distance

            # Masque OBB: points dans rectangle orienté
            obb_mask = (np.abs(points_rotated[:, 0]) <= half_length) & (
                np.abs(points_rotated[:, 1]) <= half_width
            )
        else:
            # Fallback: bounding box standard
            bounds = search_zone.bounds
            obb_mask = (
                (self.points[:, 0] >= bounds[0])
                & (self.points[:, 0] <= bounds[2])
                & (self.points[:, 1] >= bounds[1])
                & (self.points[:, 1] <= bounds[3])
            )

        # 🔥 AMÉLIORATION 2: Filtrage raffiné par polygone pour candidats OBB
        if not HAS_SHAPELY:
            self._candidate_mask = obb_mask
        else:
            from shapely.vectorized import contains

            refined_mask = obb_mask.copy()
            if np.any(obb_mask):
                refined_mask[obb_mask] = contains(
                    search_zone, self.points[obb_mask, 0], self.points[obb_mask, 1]
                )
            self._candidate_mask = refined_mask

        # Stocker indices
        self.facade.point_indices = np.where(self._candidate_mask)[0]
        self.facade.n_points = len(self.facade.point_indices)

    def _classify_wall_points(self):
        """Classifier les points candidats (mur vs non-mur) avec critères assouplis."""
        if self.verticality is None:
            # Si pas de verticalité, tous les points sont considérés comme murs
            self._wall_mask = np.ones(self.facade.n_points, dtype=bool)
            self.facade.n_wall_points = self.facade.n_points
            return

        # Verticalité des points candidats
        candidate_verticality = self.verticality[self._candidate_mask]
        candidate_points = self.points[self._candidate_mask]
        candidate_heights = self.heights[self._candidate_mask]

        # 🔥 AMÉLIORATION 1: Seuil de verticalité assoupli
        # Au lieu d'un seuil strict, utiliser un seuil progressif
        strict_threshold = self.facade.verticality_threshold
        relaxed_threshold = max(
            0.5, strict_threshold - 0.2
        )  # -0.2 par rapport au seuil

        # Points avec haute verticalité (haute confiance)
        high_confidence_mask = candidate_verticality >= strict_threshold

        # Points avec verticalité modérée (confiance moyenne)
        medium_confidence_mask = (candidate_verticality >= relaxed_threshold) & (
            ~high_confidence_mask
        )

        # 🔥 AMÉLIORATION 2: Propagation spatiale depuis zones de haute confiance
        # Si un point est proche d'un point de haute confiance, l'inclure aussi
        if np.sum(high_confidence_mask) > 10:  # Au moins 10 points de haute confiance
            from scipy.spatial import cKDTree

            high_conf_points = candidate_points[high_confidence_mask]
            tree = cKDTree(high_conf_points[:, :2])  # XY seulement

            # Pour chaque point de confiance moyenne, vérifier proximité
            medium_conf_points = candidate_points[medium_confidence_mask]
            if len(medium_conf_points) > 0:
                distances, _ = tree.query(medium_conf_points[:, :2], k=1)

                # Si à moins de 1.5m d'un point de haute confiance, inclure
                proximity_mask = distances < 1.5

                # Mettre à jour le masque de confiance moyenne
                medium_indices = np.where(medium_confidence_mask)[0]
                promoted_indices = medium_indices[proximity_mask]
                high_confidence_mask[promoted_indices] = True
                medium_confidence_mask[promoted_indices] = False

        # 🔥 AMÉLIORATION 3: Densité locale
        # Points à faible verticalité mais dans zone dense → probablement mur
        if len(candidate_points) > 20:
            from scipy.spatial import cKDTree

            tree = cKDTree(candidate_points[:, :2])

            # Compter voisins dans rayon 1m
            neighbors_count = np.array(
                [len(tree.query_ball_point(pt[:2], r=1.0)) for pt in candidate_points]
            )

            # Points à densité élevée (>30 voisins) même si faible verticalité
            dense_mask = (
                (neighbors_count > 30)
                & (candidate_verticality >= 0.4)  # Verticalité minimale
                & (~high_confidence_mask)
                & (~medium_confidence_mask)
            )

            high_confidence_mask |= dense_mask

        # Combiner tous les masques
        self._wall_mask = high_confidence_mask | medium_confidence_mask
        self.facade.n_wall_points = self._wall_mask.sum()

        # Verticalité moyenne
        self.facade.avg_verticality = candidate_verticality.mean()

    def _compute_coverage_statistics(self, building_height: float):
        """Calculer statistiques de couverture."""
        # Surface de la façade
        facade_area = self.facade.length * building_height

        if facade_area <= 0:
            return

        # Densité de points
        self.facade.point_density = self.facade.n_points / facade_area

        # Points attendus
        expected_points = facade_area * self.min_point_density

        # Ratio de couverture
        if expected_points > 0:
            self.facade.coverage_ratio = min(
                1.0, self.facade.n_points / expected_points
            )

    def _detect_gaps(self):
        """Détecter les gaps le long de la façade."""
        if self.facade.n_points == 0:
            self.facade.has_gaps = True
            self.facade.gap_segments = [(0.0, self.facade.length)]
            self.facade.gap_ratio = 1.0
            return

        # Projeter les points sur la ligne de la façade
        candidate_points_2d = self.points[self._candidate_mask][:, :2]

        projected_distances = []
        for point in candidate_points_2d:
            pt = Point(point)
            dist = self.facade.edge_line.project(pt)
            projected_distances.append(dist)

        if not projected_distances:
            return

        projected_distances = np.array(sorted(projected_distances))

        # Détecter gaps
        gaps = []
        gap_threshold = self.gap_detection_resolution

        # Gap au début
        if projected_distances[0] > gap_threshold:
            gaps.append((0.0, projected_distances[0]))

        # Gaps intermédiaires
        for i in range(len(projected_distances) - 1):
            gap_size = projected_distances[i + 1] - projected_distances[i]
            if gap_size > gap_threshold:
                gaps.append((projected_distances[i], projected_distances[i + 1]))

        # Gap à la fin
        if self.facade.length - projected_distances[-1] > gap_threshold:
            gaps.append((projected_distances[-1], self.facade.length))

        if gaps:
            self.facade.has_gaps = True
            self.facade.gap_segments = gaps
            total_gap_length = sum(end - start for start, end in gaps)
            self.facade.gap_ratio = total_gap_length / self.facade.length

    def _compute_confidence_score(self):
        """
        Calculer le score de confiance de la classification.

        Facteurs:
        - Couverture (40%)
        - Verticalité moyenne (30%)
        - Absence de gaps (20%)
        - Densité de points (10%)
        """
        # Couverture
        coverage_score = self.facade.coverage_ratio

        # Verticalité
        verticality_score = self.facade.avg_verticality

        # Gaps (inversé)
        gap_score = 1.0 - self.facade.gap_ratio

        # Densité
        density_ratio = min(1.0, self.facade.point_density / self.min_point_density)

        # Score pondéré
        confidence = (
            0.4 * coverage_score
            + 0.3 * verticality_score
            + 0.2 * gap_score
            + 0.1 * density_ratio
        )

        self.facade.confidence_score = np.clip(confidence, 0.0, 1.0)

    def _recommend_adjustments(self):
        """Recommander des ajustements si nécessaire."""
        # Buffer adaptatif si couverture faible
        if self.facade.coverage_ratio < 0.6:
            self.facade.needs_adaptive_buffer = True

            # Calculer nouveau buffer recommandé
            if self.facade.coverage_ratio < 0.3:
                # Très faible → buffer max
                recommended_buffer = self.adaptive_buffer_max
            else:
                # Faible à moyen → interpoler
                t = (0.6 - self.facade.coverage_ratio) / 0.3  # 0 à 1
                recommended_buffer = self.facade.buffer_distance + t * (
                    self.adaptive_buffer_max - self.facade.buffer_distance
                )

            self.facade.buffer_distance = min(
                recommended_buffer, self.adaptive_buffer_max
            )

        # 🆕 Calculer centroïde des points et offset par rapport à l'arête
        if self.facade.n_points > 0:
            candidate_points_2d = self.points[self._candidate_mask][:, :2]
            self.facade.point_cloud_centroid = candidate_points_2d.mean(axis=0)

            # Distance du centroïde des points à l'arête
            from shapely.geometry import Point

            pt = Point(self.facade.point_cloud_centroid)
            closest_point_on_edge = self.facade.edge_line.interpolate(
                self.facade.edge_line.project(pt)
            )

            offset_vector = self.facade.point_cloud_centroid - np.array(
                closest_point_on_edge.coords[0]
            )
            self.facade.centroid_offset_distance = np.linalg.norm(offset_vector)

            # 🆕 Recommander translation si offset significatif
            if self.facade.centroid_offset_distance > 1.0:  # > 1m
                self.facade.needs_translation = True
                logger.debug(
                    f"  {self.facade.orientation.value}: Point cloud offset by {self.facade.centroid_offset_distance:.2f}m"
                )

        # 🆕 Recommander expansion latérale si gaps aux extrémités
        if self.facade.has_gaps:
            gaps_at_start = any(start < 1.0 for start, end in self.facade.gap_segments)
            gaps_at_end = any(
                end > self.facade.length - 1.0
                for start, end in self.facade.gap_segments
            )

            if gaps_at_start or gaps_at_end:
                self.facade.needs_lateral_expansion = True
                logger.debug(
                    f"  {self.facade.orientation.value}: Gaps at extremities detected"
                )

        # Détecter occlusion si gaps significatifs
        if self.facade.has_gaps and self.facade.gap_ratio > 0.3:
            self.facade.has_occlusion = True

    def _rotate_points_2d(
        self, points: np.ndarray, angle: float, center: np.ndarray
    ) -> np.ndarray:
        """
        Rotate 2D points around a center point.

        Args:
            points: Points to rotate [N, 2]
            angle: Rotation angle in radians (counter-clockwise)
            center: Center of rotation [2]

        Returns:
            Rotated points [N, 2]
        """
        # Translation to origin
        points_centered = points - center

        # Rotation matrix
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        # Apply rotation
        points_rotated = points_centered @ rotation_matrix.T

        # Translation back
        return points_rotated + center

    def _compute_alignment_score(
        self, points: np.ndarray, facade_line: "LineString"
    ) -> float:
        """
        Compute alignment score measuring how well points align with facade line.

        The score is based on the distance of points from the facade line.
        Lower distances = higher score.

        Args:
            points: Points to evaluate [N, 2]
            facade_line: Reference facade line

        Returns:
            Alignment score (0-1), higher is better
        """
        if len(points) == 0:
            return 0.0

        # Calculate distance from each point to facade line
        from shapely.geometry import Point

        distances = []
        for pt in points:
            point_geom = Point(pt)
            dist = facade_line.distance(point_geom)
            distances.append(dist)

        distances = np.array(distances)

        # Score based on median distance (robust to outliers)
        median_dist = np.median(distances)

        # Lower distance = higher score
        # Use exponential decay: score = exp(-distance / scale)
        scale = 1.0  # 1 meter scale
        score = np.exp(-median_dist / scale)

        return float(score)

    def _apply_rotation_to_line(
        self, line: "LineString", angle: float, center: np.ndarray
    ) -> "LineString":
        """
        Apply rotation to a LineString around a center point.

        Args:
            line: LineString to rotate
            angle: Rotation angle in radians
            center: Center of rotation [2]

        Returns:
            Rotated LineString
        """
        from shapely.geometry import LineString

        # Get line coordinates
        coords = np.array(line.coords)

        # Rotate coordinates
        rotated_coords = self._rotate_points_2d(coords, angle, center)

        # Create new LineString
        return LineString(rotated_coords)

    def _detect_optimal_rotation(
        self,
        candidate_points: np.ndarray,
        current_angle: float,
        max_rotation_degrees: float,
    ) -> Tuple[float, float]:
        """
        Detect optimal rotation angle for facade alignment.

        Tests multiple rotation angles and scores alignment quality based on
        point-to-line distances.

        Args:
            candidate_points: Points near facade [N, 3]
            current_angle: Current facade angle (radians)
            max_rotation_degrees: Maximum rotation allowed (degrees)

        Returns:
            Tuple of (optimal_rotation_angle, confidence_score)
                - optimal_rotation_angle: Best rotation angle in radians
                - confidence_score: Confidence of rotation (0-1)
        """
        if len(candidate_points) < 10:
            # Not enough points for reliable rotation detection
            return 0.0, 0.0

        # Get facade center
        facade_center = np.array(self.facade.centroid[:2])

        # Extract 2D coordinates
        points_2d = candidate_points[:, :2]

        # Test rotations from -max to +max degrees
        max_rot_rad = np.radians(max_rotation_degrees)
        test_angles = np.linspace(-max_rot_rad, max_rot_rad, 31)  # Test 31 angles

        best_angle = 0.0
        best_score = 0.0

        for delta_angle in test_angles:
            # Total angle
            total_angle = current_angle + delta_angle

            # Create rotated facade line
            rotated_line = self._apply_rotation_to_line(
                self.facade.edge_line, delta_angle, facade_center
            )

            # Score alignment
            score = self._compute_alignment_score(points_2d, rotated_line)

            if score > best_score:
                best_score = score
                best_angle = delta_angle

        # Confidence based on score improvement over no rotation
        baseline_score = self._compute_alignment_score(points_2d, self.facade.edge_line)
        if baseline_score > 0:
            confidence = min(1.0, (best_score - baseline_score) / baseline_score + 0.5)
        else:
            confidence = best_score

        return best_angle, float(confidence)

    def _project_points_on_facade_direction(
        self, candidate_points: np.ndarray
    ) -> np.ndarray:
        """
        Project points onto the facade direction vector to measure extent.

        Args:
            candidate_points: Points to project [N, 3]

        Returns:
            Projected distances along facade direction [N]
        """
        if len(candidate_points) == 0:
            return np.array([])

        # Get facade direction vector
        coords = np.array(self.facade.edge_line.coords)
        start_point = coords[0]
        end_point = coords[-1]

        direction_vector = end_point - start_point
        direction_length = np.linalg.norm(direction_vector)

        if direction_length == 0:
            return np.zeros(len(candidate_points))

        direction_unit = direction_vector / direction_length

        # Project points onto facade direction
        points_2d = candidate_points[:, :2]
        relative_positions = points_2d - start_point

        # Dot product gives projection onto direction
        projected_distances = relative_positions @ direction_unit

        return projected_distances

    def _apply_scaling_to_line(
        self, line: "LineString", scale_factor: float, center: np.ndarray
    ) -> "LineString":
        """
        Apply scaling to a LineString around a center point.

        Args:
            line: LineString to scale
            scale_factor: Scaling factor (>1 expands, <1 shrinks)
            center: Center of scaling [2]

        Returns:
            Scaled LineString
        """
        from shapely.geometry import LineString

        # Get line coordinates
        coords = np.array(line.coords)

        # Scale relative to center
        coords_centered = coords - center
        coords_scaled = coords_centered * scale_factor
        coords_final = coords_scaled + center

        # Create new LineString
        return LineString(coords_final)

    def _detect_optimal_scale(
        self,
        candidate_points: np.ndarray,
        current_length: float,
        max_scale_factor: float,
    ) -> Tuple[float, float]:
        """
        Detect optimal scaling factor for facade length.

        Analyzes point distribution along facade direction to determine
        if facade should be scaled up or down.

        Args:
            candidate_points: Points near facade [N, 3]
            current_length: Current facade length (m)
            max_scale_factor: Maximum scaling (e.g., 1.5 = 150%)

        Returns:
            Tuple of (scale_factor, confidence_score)
                - scale_factor: Optimal scaling factor
                - confidence_score: Confidence of scaling (0-1)
        """
        if len(candidate_points) < 10:
            # Not enough points for reliable scaling detection
            return 1.0, 0.0

        # Project points onto facade direction
        projected_distances = self._project_points_on_facade_direction(candidate_points)

        if len(projected_distances) == 0:
            return 1.0, 0.0

        # Find actual extent of points along facade (use percentiles to be robust)
        min_dist = np.percentile(projected_distances, 5)  # 5th percentile
        max_dist = np.percentile(projected_distances, 95)  # 95th percentile
        actual_length = max_dist - min_dist

        if actual_length <= 0 or current_length <= 0:
            return 1.0, 0.0

        # Calculate scale factor
        scale_factor = actual_length / current_length

        # Clamp to valid range [1/max_scale_factor, max_scale_factor]
        min_scale = 1.0 / max_scale_factor
        scale_factor = np.clip(scale_factor, min_scale, max_scale_factor)

        # Confidence based on point density
        density = len(candidate_points) / actual_length
        expected_density = self.min_point_density

        if expected_density > 0:
            density_ratio = min(1.0, density / expected_density)
            confidence = float(density_ratio)
        else:
            confidence = 0.5

        return float(scale_factor), confidence

    def adapt_facade_geometry(
        self,
        max_translation: float = 3.0,
        max_lateral_expansion: float = 2.0,
        max_rotation_degrees: float = 15.0,  # 🆕 NEW: Maximum rotation (v3.0.3)
        enable_scaling: bool = True,  # 🆕 NEW: Enable scaling (v3.0.3)
        max_scale_factor: float = 1.5,  # 🆕 NEW: Maximum scale factor (v3.0.3)
    ) -> "LineString":
        """
        🆕 Adapter la géométrie de la façade basée sur les points détectés.

        Cette méthode ajuste l'arête de la façade pour mieux correspondre
        à la distribution réelle des points du nuage.

        Améliorations v3.0.3:
        - Rotation adaptative (±15°)
        - Scaling adaptatif (0.5-1.5x)

        Args:
            max_translation: Translation maximale perpendiculaire (m)
            max_lateral_expansion: Extension latérale maximale (m)
            max_rotation_degrees: Rotation maximale autorisée (degrés)
            enable_scaling: Activer le scaling adaptatif
            max_scale_factor: Facteur d'échelle maximal (1.5 = 150%)

        Returns:
            LineString ajustée
        """
        if not self.facade.is_processed:
            logger.warning("Facade must be processed before adaptation")
            return self.facade.edge_line

        adjusted_line = self.facade.edge_line

        # 🆕 0. Rotation adaptative si activée
        if max_rotation_degrees > 0 and self.facade.point_indices is not None:
            candidate_points = self.points[self.facade.point_indices]

            # Calculate current facade angle
            coords = np.array(self.facade.edge_line.coords)
            dx = coords[-1][0] - coords[0][0]
            dy = coords[-1][1] - coords[0][1]
            current_angle = np.arctan2(dy, dx)

            rotation_angle, rotation_confidence = self._detect_optimal_rotation(
                candidate_points=candidate_points,
                current_angle=current_angle,
                max_rotation_degrees=max_rotation_degrees,
            )

            # Only rotate if angle is significant (>2 degrees)
            if abs(np.degrees(rotation_angle)) > 2.0:
                facade_center = np.array(self.facade.centroid[:2])
                adjusted_line = self._apply_rotation_to_line(
                    adjusted_line, rotation_angle, facade_center
                )

                self.facade.rotation_angle = rotation_angle
                self.facade.rotation_confidence = rotation_confidence
                self.facade.is_rotated = True

                logger.debug(
                    f"  {self.facade.orientation.value}: Rotated by "
                    f"{np.degrees(rotation_angle):.2f}°, "
                    f"confidence={rotation_confidence:.2f}"
                )

        # 🆕 0b. Scaling adaptatif si activé
        if enable_scaling and self.facade.point_indices is not None:
            candidate_points = self.points[self.facade.point_indices]
            current_length = self.facade.length

            scale_factor, scale_confidence = self._detect_optimal_scale(
                candidate_points=candidate_points,
                current_length=current_length,
                max_scale_factor=max_scale_factor,
            )

            # Only scale if factor is significant (>10% difference from 1.0)
            if abs(scale_factor - 1.0) > 0.1:
                facade_center = np.array(self.facade.centroid[:2])
                adjusted_line = self._apply_scaling_to_line(
                    adjusted_line, scale_factor, facade_center
                )

                self.facade.scale_factor = scale_factor
                self.facade.scale_confidence = scale_confidence
                self.facade.is_scaled = True

                logger.debug(
                    f"  {self.facade.orientation.value}: Scaled by "
                    f"{scale_factor:.2f}x (length: {current_length:.2f}m → "
                    f"{current_length * scale_factor:.2f}m), "
                    f"confidence={scale_confidence:.2f}"
                )

        # 1. Translation perpendiculaire si nécessaire
        if (
            self.facade.needs_translation
            and self.facade.point_cloud_centroid is not None
        ):
            # Calculer vecteur de translation vers le centroïde des points
            from shapely.geometry import Point

            pt = Point(self.facade.point_cloud_centroid)
            closest_point_on_edge = adjusted_line.interpolate(adjusted_line.project(pt))

            translation_vector = self.facade.point_cloud_centroid - np.array(
                closest_point_on_edge.coords[0]
            )

            # Limiter la translation
            translation_distance = min(
                np.linalg.norm(translation_vector), max_translation
            )

            if translation_distance > 0.1:  # Si > 10cm
                # Normaliser et scaler
                translation_vector = (
                    translation_vector
                    / np.linalg.norm(translation_vector)
                    * translation_distance
                )

                # Appliquer translation
                from shapely.affinity import translate as shapely_translate

                adjusted_line = shapely_translate(
                    adjusted_line,
                    xoff=translation_vector[0],
                    yoff=translation_vector[1],
                )

                self.facade.translation_offset = translation_distance
                logger.debug(
                    f"  {self.facade.orientation.value}: Translated by {translation_distance:.2f}m"
                )

        # 2. Extension latérale aux extrémités si gaps détectés
        if self.facade.needs_lateral_expansion and self.facade.gap_segments:
            coords = list(adjusted_line.coords)
            start_point = np.array(coords[0])
            end_point = np.array(coords[-1])

            # Vecteur directeur de la façade
            direction_vector = end_point - start_point
            direction_length = np.linalg.norm(direction_vector)

            if direction_length > 0:
                direction_unit = direction_vector / direction_length

                # Extension au début (gauche)
                expand_start = 0.0
                for start, end in self.facade.gap_segments:
                    if start < 1.0:  # Gap au début
                        expand_start = max(
                            expand_start, min(end, max_lateral_expansion)
                        )

                # Extension à la fin (droite)
                expand_end = 0.0
                for start, end in self.facade.gap_segments:
                    if end > self.facade.length - 1.0:  # Gap à la fin
                        expand_end = max(
                            expand_end,
                            min(self.facade.length - start, max_lateral_expansion),
                        )

                if expand_start > 0 or expand_end > 0:
                    # Nouveau point de départ (étendu vers l'arrière)
                    new_start = start_point - direction_unit * expand_start
                    # Nouveau point de fin (étendu vers l'avant)
                    new_end = end_point + direction_unit * expand_end

                    from shapely.geometry import LineString

                    adjusted_line = LineString([new_start, new_end])

                    self.facade.lateral_expansion = (expand_start, expand_end)
                    logger.debug(
                        f"  {self.facade.orientation.value}: Extended laterally by "
                        f"({expand_start:.2f}m, {expand_end:.2f}m)"
                    )

        # Stocker l'arête ajustée
        self.facade.adjusted_edge_line = adjusted_line
        self.facade.is_adapted = True

        return adjusted_line


class BuildingFacadeClassifier:
    """
    Classificateur de bâtiment basé sur l'analyse par façade.

    Processus:
    1. Décomposer le bâtiment en 4 façades (N/S/E/W)
    2. Traiter chaque façade indépendamment
    3. 🆕 Adapter géométrie de chaque façade
    4. 🆕 Reconstruire polygone avec façades adaptées
    5. Fusionner les résultats
    6. Appliquer classification finale point par point
    """

    def __init__(
        self,
        # Paramètres de traitement
        initial_buffer: float = 3.0,  # 🔥 INCREASED: 2.5→3.0m pour capturer plus de points en bordure
        verticality_threshold: float = 0.55,  # ✅ IMPROVED: Abaissé de 0.70→0.55 pour détecter plus de façades
        min_point_density: float = 35.0,  # 🔥 LOWERED: 40→35 pts/m² pour accepter zones moins denses
        gap_detection_resolution: float = 0.5,
        adaptive_buffer_range: Tuple[float, float] = (
            0.5,
            12.0,
        ),  # 🔥 EXPANDED: Max augmenté 10→12m pour meilleure couverture
        # 🆕 Paramètres d'adaptation géométrique
        enable_facade_adaptation: bool = True,
        max_translation: float = 5.0,  # 🔥 INCREASED: 4→5m pour ajustements plus larges
        max_lateral_expansion: float = 4.0,  # 🔥 INCREASED: 3→4m pour extensions latérales plus larges
        max_rotation_degrees: float = 15.0,  # 🆕 NEW: Rotation maximale pour alignement façades
        enable_scaling: bool = True,  # 🆕 NEW: Activer mise à l'échelle adaptative
        max_scale_factor: float = 1.5,  # 🆕 NEW: Facteur d'échelle maximal (150%)
        # 🆕 Paramètres de détection d'arêtes
        enable_edge_detection: bool = True,  # 🆕 NEW: Détecter points d'arêtes/bordures
        edge_detection_threshold: float = 0.3,  # 🆕 NEW: Seuil pour détecter arêtes (curvature)
        edge_expansion_radius: float = 0.5,  # 🆕 NEW: Rayon d'expansion autour arêtes
        # 🆕 Paramètres de filtrage avec is_ground
        use_ground_filter: bool = True,  # 🆕 NEW: Utiliser feature is_ground pour filtrer
        ground_height_tolerance: float = 2.0,  # 🆕 NEW: Tolérance hauteur par rapport au sol (m)
        # 🆕 v3.1 Paramètres de classification de toit
        enable_roof_classification: bool = False,  # 🆕 v3.1: Activer classification LOD3 des toits
        roof_flat_threshold: float = 15.0,  # 🆕 v3.1: Angle max pour toit plat (degrés)
        roof_pitched_threshold: float = 20.0,  # 🆕 v3.1: Angle min pour toit en pente (degrés)
        # 🆕 v3.4 Paramètres de classification avancée LOD3
        enable_enhanced_lod3: bool = False,  # 🆕 v3.4: Activer classification LOD3 complète (Phase 2.4)
        enhanced_building_config: Optional[
            Dict[str, Any]
        ] = None,  # 🆕 v3.4: Configuration EnhancedBuildingClassifier
        # Paramètres de classification
        building_class: int = 6,  # ASPRS building
        wall_subclass: Optional[int] = None,
        min_confidence: float = 0.20,  # 🔥 LOWERED: 0.25→0.20 pour capturer plus de façades
    ):
        """
        Initialiser le classificateur par façade avec améliorations avancées.

        Args:
            initial_buffer: Buffer initial pour chaque façade (3.0m élargi)
            verticality_threshold: Seuil de verticalité pour murs
                (0.55 = IMPROVED: abaissé de 0.70 pour capturer plus de façades)
            min_point_density: Densité minimale attendue (35 pts/m², abaissé pour zones moins denses)
            gap_detection_resolution: Résolution pour détecter gaps
            adaptive_buffer_range: (min, max) pour buffers adaptatifs (max 12m)
            enable_facade_adaptation: Activer adaptation géométrique
            max_translation: Translation maximale perpendiculaire (5m augmenté)
            max_lateral_expansion: Extension latérale maximale (4m augmenté)
            max_rotation_degrees: Rotation maximale pour aligner façades (15°)
            enable_scaling: Activer mise à l'échelle adaptative des façades
            max_scale_factor: Facteur d'échelle maximal (1.5 = 150%)
            enable_edge_detection: Activer détection de points d'arêtes/bordures
            edge_detection_threshold: Seuil de courbure pour détecter arêtes
            edge_expansion_radius: Rayon d'expansion autour des arêtes
            use_ground_filter: Utiliser is_ground pour filtrer points
            ground_height_tolerance: Tolérance hauteur au-dessus du sol
            enable_roof_classification: Activer classification toits (v3.1)
            roof_flat_threshold: Angle max toit plat en degrés (v3.1)
            roof_pitched_threshold: Angle min toit pente en degrés (v3.1)
            building_class: Code de classification pour bâtiments
            wall_subclass: Code optionnel pour sous-classe "mur"
            min_confidence: Confiance minimale (0.20 abaissé)
        """
        self.initial_buffer = initial_buffer
        self.verticality_threshold = verticality_threshold
        self.min_point_density = min_point_density
        self.gap_detection_resolution = gap_detection_resolution
        self.adaptive_buffer_range = adaptive_buffer_range

        self.enable_facade_adaptation = enable_facade_adaptation
        self.max_translation = max_translation
        self.max_lateral_expansion = max_lateral_expansion
        self.max_rotation_degrees = max_rotation_degrees
        self.enable_scaling = enable_scaling
        self.max_scale_factor = max_scale_factor

        self.enable_edge_detection = enable_edge_detection
        self.edge_detection_threshold = edge_detection_threshold
        self.edge_expansion_radius = edge_expansion_radius

        self.use_ground_filter = use_ground_filter
        self.ground_height_tolerance = ground_height_tolerance

        # v3.1 Roof classification
        self.enable_roof_classification = enable_roof_classification
        self.roof_flat_threshold = roof_flat_threshold
        self.roof_pitched_threshold = roof_pitched_threshold

        # Initialize roof classifier if enabled
        self.roof_classifier = None
        if self.enable_roof_classification:
            try:
                from ign_lidar.core.classification.building.roof_classifier import (
                    RoofTypeClassifier,
                )

                self.roof_classifier = RoofTypeClassifier(
                    flat_threshold=roof_flat_threshold,
                    pitched_threshold=roof_pitched_threshold,
                    verticality_threshold=verticality_threshold,
                )
                logger.info("Roof classifier enabled (v3.1)")
            except ImportError as e:
                logger.warning(f"Roof classifier unavailable: {e}")
                self.enable_roof_classification = False

        # v3.4 Enhanced LOD3 classification (Phase 2.4 integration)
        self.enable_enhanced_lod3 = enable_enhanced_lod3
        self.enhanced_building_config = enhanced_building_config

        # Initialize EnhancedBuildingClassifier if enabled
        self.enhanced_classifier = None
        if self.enable_enhanced_lod3:
            try:
                from ign_lidar.core.classification.building import (
                    EnhancedBuildingClassifier,
                    EnhancedClassifierConfig,
                )

                # Build config from provided dict or use defaults
                if enhanced_building_config:
                    classifier_config = EnhancedClassifierConfig(
                        **enhanced_building_config
                    )
                else:
                    classifier_config = EnhancedClassifierConfig()

                self.enhanced_classifier = EnhancedBuildingClassifier(classifier_config)
                logger.info("Enhanced LOD3 classifier enabled (v3.4 - Phase 2.4)")
                logger.info(
                    f"  - Roof detection: " f"{classifier_config.enable_roof_detection}"
                )
                logger.info(
                    f"  - Chimney detection: "
                    f"{classifier_config.enable_chimney_detection}"
                )
                logger.info(
                    f"  - Balcony detection: "
                    f"{classifier_config.enable_balcony_detection}"
                )
            except ImportError as e:
                logger.warning(f"Enhanced LOD3 classifier unavailable: {e}")
                self.enable_enhanced_lod3 = False

        self.building_class = building_class
        self.wall_subclass = wall_subclass
        self.min_confidence = min_confidence

        # Statistiques
        self.n_buildings_processed = 0
        self.n_points_classified = 0
        self.n_facades_adapted = 0

    def classify_buildings(
        self,
        buildings_gdf,
        points: np.ndarray,
        heights: np.ndarray,
        labels: np.ndarray,
        normals: Optional[np.ndarray] = None,
        verticality: Optional[np.ndarray] = None,
        planarity: Optional[np.ndarray] = None,  # 🆕 v3.1: For roof classification
        curvature: Optional[np.ndarray] = None,  # 🆕 NEW: Pour détection d'arêtes
        is_ground: Optional[np.ndarray] = None,  # 🆕 NEW: Feature is_ground
        building_classes: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Classifier les points pour tous les bâtiments avec améliorations avancées.

        Args:
            buildings_gdf: GeoDataFrame avec polygones BD TOPO
            points: Nuage de points [N, 3]
            heights: Hauteurs au-dessus du sol [N]
            labels: Labels actuels [N]
            normals: Normales [N, 3]
            verticality: Verticalité [N]
            planarity: Planarité [N] pour classification toit (v3.1)
            curvature: Courbure [N] pour détection d'arêtes
            is_ground: Feature binaire is_ground [N] (0=non-sol, 1=sol)
            building_classes: Classes existantes considérées comme bâtiments

        Returns:
            (labels_updated, statistics)
        """
        logger.info(
            f"🏢 Classifying {len(buildings_gdf)} buildings using facade-based approach (IMPROVED)"
        )

        if self.use_ground_filter and is_ground is not None:
            logger.info("  ✓ Using is_ground feature for filtering")
        if self.enable_edge_detection and curvature is not None:
            logger.info(
                f"  ✓ Edge detection enabled (threshold={self.edge_detection_threshold})"
            )
        if self.enable_roof_classification:
            logger.info("  ✓ Roof classification enabled (v3.1)")

        labels_updated = labels.copy()
        building_classes = building_classes or [6]  # ASPRS building

        # Statistiques
        stats = {
            "buildings_processed": 0,
            "points_classified": 0,
            "facades_processed": 0,
            "avg_confidence": 0.0,
            "low_confidence_buildings": 0,
            "edge_points_total": 0,
            "ground_points_filtered_total": 0,
            "roofs_classified": 0,  # 🆕 v3.1
            "roof_types": {},  # 🆕 v3.1: Count by roof type
        }

        confidences = []

        # Traiter chaque bâtiment
        for idx, row in buildings_gdf.iterrows():
            polygon = row["geometry"]

            if not isinstance(polygon, Polygon):
                continue

            # Classifier ce bâtiment avec nouvelles features
            building_labels, building_stats = self.classify_single_building(
                building_id=idx,
                polygon=polygon,
                points=points,
                heights=heights,
                labels=labels_updated,
                normals=normals,
                verticality=verticality,
                curvature=curvature,
                is_ground=is_ground,
            )

            # Mettre à jour labels
            labels_updated = building_labels

            # Accumuler stats
            stats["buildings_processed"] += 1
            stats["points_classified"] += building_stats["points_classified"]
            stats["facades_processed"] += building_stats["facades_processed"]
            stats["edge_points_total"] += building_stats.get("edge_points_detected", 0)
            stats["ground_points_filtered_total"] += building_stats.get(
                "ground_points_filtered", 0
            )

            # 🆕 v3.1: Track roof classification stats
            if "roof_type" in building_stats:
                stats["roofs_classified"] += 1
                roof_type = building_stats["roof_type"]
                stats["roof_types"][roof_type] = (
                    stats["roof_types"].get(roof_type, 0) + 1
                )

            if building_stats["avg_confidence"] > 0:
                confidences.append(building_stats["avg_confidence"])

            if building_stats["avg_confidence"] < self.min_confidence:
                stats["low_confidence_buildings"] += 1

            if (idx + 1) % 100 == 0:
                logger.info(f"  Processed {idx + 1}/{len(buildings_gdf)} buildings...")

        # Statistiques globales
        if confidences:
            stats["avg_confidence"] = np.mean(confidences)

        logger.info(
            f"✅ Classified {stats['points_classified']:,} points across "
            f"{stats['buildings_processed']} buildings (avg confidence: {stats['avg_confidence']:.2f})"
        )

        if stats["edge_points_total"] > 0:
            logger.info(f"  ✓ Edge points detected: {stats['edge_points_total']:,}")
        if stats["ground_points_filtered_total"] > 0:
            logger.info(
                f"  ✓ Ground points filtered: {stats['ground_points_filtered_total']:,}"
            )
        if stats["roofs_classified"] > 0:  # 🆕 v3.1
            logger.info(f"  ✓ Roofs classified: {stats['roofs_classified']}")
            for roof_type, count in stats["roof_types"].items():
                logger.info(f"    - {roof_type}: {count}")

        self.n_buildings_processed += stats["buildings_processed"]
        self.n_points_classified += stats["points_classified"]

        return labels_updated, stats

    def classify_single_building(
        self,
        building_id: int,
        polygon: "Polygon",
        points: np.ndarray,
        heights: np.ndarray,
        labels: np.ndarray,
        normals: Optional[np.ndarray] = None,
        verticality: Optional[np.ndarray] = None,
        curvature: Optional[np.ndarray] = None,  # 🆕 NEW: Pour détection d'arêtes
        is_ground: Optional[np.ndarray] = None,  # 🆕 NEW: Feature is_ground
    ) -> Tuple[np.ndarray, Dict]:
        """
        Classifier les points d'un bâtiment individuel avec améliorations avancées.

        Args:
            building_id: ID du bâtiment
            polygon: Polygone BD TOPO
            points: Nuage de points complet [N, 3]
            heights: Hauteurs au-dessus du sol [N]
            labels: Labels actuels [N]
            normals: Normales [N, 3]
            verticality: Verticalité [N]
            curvature: Courbure [N] pour détection d'arêtes
            is_ground: Feature binaire is_ground [N] (0=non-sol, 1=sol)

        Returns:
            (labels_updated, statistics)
        """
        labels_updated = labels.copy()

        stats = {
            "points_classified": 0,
            "facades_processed": 0,
            "avg_confidence": 0.0,
            "edge_points_detected": 0,
            "ground_points_filtered": 0,
            "facades_rotated": 0,  # 🆕 NEW: v3.0.3
            "avg_rotation_angle": 0.0,  # 🆕 NEW: v3.0.3
            "facades_scaled": 0,  # 🆕 NEW: v3.0.3
            "avg_scale_factor": 1.0,  # 🆕 NEW: v3.0.3
        }

        # 1. Décomposer en 4 façades
        facades = self._decompose_into_facades(building_id, polygon)

        if not facades:
            return labels_updated, stats

        # 2. Extraire points du bâtiment avec buffer élargi
        building_points, building_mask = self._extract_building_points(
            polygon, points, buffer=self.adaptive_buffer_range[1]
        )
        if len(building_points) == 0:
            return labels_updated, stats

        # 🐛 2.5 Validate and sanitize features (BUGFIX v3.0.4)
        from ign_lidar.core.classification.feature_validator import (
            validate_features_for_classification,
        )

        # Prepare features dict
        features_dict = {}
        if normals is not None:
            features_dict["normals"] = normals
        if verticality is not None:
            features_dict["verticality"] = verticality
        if curvature is not None:
            features_dict["curvature"] = curvature

        # Determine required features
        required_features = []
        if normals is not None:
            required_features.append("normals")
        if verticality is not None:
            required_features.append("verticality")
        if self.enable_edge_detection and curvature is not None:
            required_features.append("curvature")

        # Validate and sanitize
        if features_dict:
            is_valid, sanitized_features, validation_issues = (
                validate_features_for_classification(
                    features=features_dict,
                    required_features=required_features,
                    point_mask=building_mask,
                    clip_sigma=5.0,
                )
            )

            if validation_issues:
                logger.debug(
                    f"Building {building_id}: Feature validation - {len(validation_issues)} issues detected"
                )
                for issue in validation_issues[:3]:  # Log first 3 issues
                    logger.debug(f"  - {issue}")

            # Use sanitized versions
            if "normals" in sanitized_features:
                normals = sanitized_features["normals"]
            if "verticality" in sanitized_features:
                verticality = sanitized_features["verticality"]
            if "curvature" in sanitized_features:
                curvature = sanitized_features["curvature"]

            stats["features_validated"] = True
            stats["features_sanitized"] = len(validation_issues) > 0
        else:
            stats["features_validated"] = False

        # 🆕 3. Filtrer points au sol si is_ground disponible
        valid_mask = building_mask.copy()
        if self.use_ground_filter and is_ground is not None:
            ground_mask = is_ground[building_mask] == 1

            # Exclure points au sol qui sont trop bas
            low_ground_mask = ground_mask & (
                heights[building_mask] < self.ground_height_tolerance
            )

            # Créer masque des points valides (non-sol ou sol élevé)
            valid_building_mask = ~low_ground_mask

            stats["ground_points_filtered"] = np.sum(low_ground_mask)

            # Mettre à jour le masque global
            valid_indices = np.where(building_mask)[0][valid_building_mask]
            valid_mask = np.zeros_like(building_mask)
            valid_mask[valid_indices] = True

        # 🐛 BUGFIX v3.0.4: Always use valid_mask consistently for height calculation
        if np.any(valid_mask):
            building_height = heights[valid_mask].max()
            building_points_clean = points[valid_mask]
            building_heights_clean = heights[valid_mask]
        else:
            # Fallback if no valid points after filtering
            logger.warning(
                f"Building {building_id}: No valid points after ground filtering, using all building points"
            )
            valid_mask = building_mask.copy()
            building_height = heights[building_mask].max()
            building_points_clean = building_points
            building_heights_clean = heights[building_mask]

        # 🆕 4. Détecter points d'arêtes/bordures si activé (SECURED v3.0.4)
        edge_points_mask = np.zeros(len(points), dtype=bool)
        if self.enable_edge_detection and curvature is not None:
            # 🐛 BUGFIX: Validate curvature first (prevent NaN/Inf artifacts)
            curvature_clean = curvature.copy()
            invalid_curvature = ~np.isfinite(curvature)

            if invalid_curvature.any():
                n_invalid = invalid_curvature.sum()
                logger.debug(
                    f"Building {building_id}: {n_invalid} invalid curvature values, setting to 0"
                )
                curvature_clean[invalid_curvature] = 0.0  # Safe default

            # Only consider VALID building points (not ground-filtered)
            valid_edge_candidates_mask = valid_mask & (
                curvature_clean > self.edge_detection_threshold
            )

            # Filter by verticality if available
            if verticality is not None:
                verticality_clean = verticality.copy()
                invalid_verticality = ~np.isfinite(verticality)
                if invalid_verticality.any():
                    verticality_clean[invalid_verticality] = 0.0

                valid_edge_candidates_mask &= verticality_clean > 0.3  # Vertical edges

            edge_points_mask[valid_edge_candidates_mask] = True
            stats["edge_points_detected"] = np.sum(valid_edge_candidates_mask)

        # 5. Traiter chaque façade
        all_classified_indices = set()
        confidences = []

        for facade in facades:
            processor = FacadeProcessor(
                facade=facade,
                points=points,
                heights=heights,
                normals=normals,
                verticality=verticality,
                min_point_density=self.min_point_density,
                gap_detection_resolution=self.gap_detection_resolution,
                adaptive_buffer_range=self.adaptive_buffer_range,
            )

            # Traiter la façade
            processed_facade = processor.process(building_height=building_height)

            # 🔥 Adapter géométrie de la façade si activé
            if self.enable_facade_adaptation:
                if (
                    processed_facade.needs_translation
                    or processed_facade.needs_lateral_expansion
                ):

                    adapted_line = processor.adapt_facade_geometry(
                        max_translation=self.max_translation,
                        max_lateral_expansion=self.max_lateral_expansion,
                        max_rotation_degrees=self.max_rotation_degrees,  # 🆕 v3.0.3
                        enable_scaling=self.enable_scaling,  # 🆕 v3.0.3
                        max_scale_factor=self.max_scale_factor,  # 🆕 v3.0.3
                    )

                    if processed_facade.is_adapted:
                        self.n_facades_adapted += 1
                        logger.debug(
                            f"  Adapted {processed_facade.orientation.value} facade: "
                            f"translation={processed_facade.translation_offset:.2f}m, "
                            f"expansion=({processed_facade.lateral_expansion[0]:.2f}m, "
                            f"{processed_facade.lateral_expansion[1]:.2f}m)"
                        )

                    # 🆕 Track rotation statistics (v3.0.3)
                    if processed_facade.is_rotated:
                        stats["facades_rotated"] += 1
                        stats["avg_rotation_angle"] += abs(
                            processed_facade.rotation_angle
                        )

                    # 🆕 Track scaling statistics (v3.0.3)
                    if processed_facade.is_scaled:
                        stats["facades_scaled"] += 1
                        stats["avg_scale_factor"] += processed_facade.scale_factor

            # Classifier les points si confiance suffisante
            if processed_facade.confidence_score >= self.min_confidence:
                if processed_facade.point_indices is not None:
                    # Classifier comme bâtiment
                    wall_indices = processed_facade.point_indices[processor._wall_mask]

                    # 🆕 Filtrer points au sol si activé
                    if self.use_ground_filter and is_ground is not None:
                        non_ground_mask = is_ground[wall_indices] == 0
                        wall_indices = wall_indices[non_ground_mask]

                    labels_updated[wall_indices] = self.building_class

                    # Ajouter à l'ensemble des indices classifiés
                    all_classified_indices.update(wall_indices)

                    confidences.append(processed_facade.confidence_score)

            stats["facades_processed"] += 1

        # 🆕 6. Ajouter points d'arêtes détectés dans zone proximité (SECURED v3.0.4)
        if self.enable_edge_detection and np.any(edge_points_mask):
            # Points d'arêtes près des façades classifiées
            if all_classified_indices:
                from scipy.spatial import cKDTree

                classified_points = points[list(all_classified_indices)]
                edge_candidates_indices = np.where(edge_points_mask)[0]
                edge_candidates_points = points[edge_candidates_indices]

                if len(classified_points) > 0 and len(edge_candidates_points) > 0:
                    tree = cKDTree(classified_points)
                    distances, _ = tree.query(edge_candidates_points, k=1)

                    # Ajouter arêtes proches (< edge_expansion_radius)
                    nearby_edges = distances < self.edge_expansion_radius
                    nearby_edge_indices = edge_candidates_indices[nearby_edges]

                    # 🐛 BUGFIX v3.0.4: Additional spatial validation
                    # Only expand within building polygon area
                    if (
                        HAS_SHAPELY
                        and polygon is not None
                        and len(nearby_edge_indices) > 0
                    ):
                        nearby_points = points[nearby_edge_indices]
                        buffered_polygon = polygon.buffer(
                            self.edge_expansion_radius * 1.5
                        )

                        from shapely.vectorized import contains

                        inside_polygon = contains(
                            buffered_polygon,
                            nearby_points[:, 0],
                            nearby_points[:, 1],
                        )

                        # Only keep edges inside building area
                        nearby_edge_indices = nearby_edge_indices[inside_polygon]

                    if len(nearby_edge_indices) > 0:
                        labels_updated[nearby_edge_indices] = self.building_class
                        all_classified_indices.update(nearby_edge_indices)
                        stats["edge_points_expanded"] = len(nearby_edge_indices)

        # 🔥 7. Reconstruire polygone adapté (si au moins une façade adaptée)
        adapted_facades = [f for f in facades if f.is_adapted]
        if adapted_facades:
            adapted_polygon = self._reconstruct_polygon_from_facades(facades)
            stats["adapted_polygon"] = adapted_polygon

        # 🆕 8. Classification des toits (v3.1, SECURED v3.0.4)
        if self.enable_roof_classification and self.roof_classifier:
            try:
                # 🐛 BUGFIX v3.0.4: Use VALID building mask (ground-filtered)
                # and validate features before roof classification
                roof_building_mask = valid_mask.copy()

                # Extract and validate roof features
                roof_normals = (
                    normals[roof_building_mask] if normals is not None else None
                )
                roof_verticality = (
                    verticality[roof_building_mask] if verticality is not None else None
                )
                roof_curvature = (
                    curvature[roof_building_mask] if curvature is not None else None
                )

                # Validate features are clean (no NaN/Inf)
                features_valid = True

                if roof_normals is not None:
                    if not np.all(np.isfinite(roof_normals)):
                        n_invalid = (~np.isfinite(roof_normals)).sum()
                        logger.warning(
                            f"Building {building_id}: {n_invalid} invalid normals in roof classification"
                        )
                        # Try to clean
                        invalid_mask = ~np.all(np.isfinite(roof_normals), axis=1)
                        if invalid_mask.sum() > len(roof_normals) * 0.5:
                            # Too many invalid, skip roof classification
                            features_valid = False
                        else:
                            # Replace invalid normals with vertical (0, 0, 1)
                            roof_normals[invalid_mask] = [0, 0, 1]

                if features_valid and roof_verticality is not None:
                    invalid_vert = ~np.isfinite(roof_verticality)
                    if invalid_vert.any():
                        logger.debug(
                            f"Building {building_id}: {invalid_vert.sum()} invalid verticality values"
                        )
                        roof_verticality[invalid_vert] = 0.0

                if not features_valid:
                    logger.warning(
                        f"Building {building_id}: Skipping roof classification due to invalid features"
                    )
                    stats["roof_classification_skipped"] = "invalid_features"
                    raise ValueError("Invalid features for roof classification")

                # Prepare features dict
                roof_features = {
                    "normals": roof_normals,
                    "planarity": None,  # Will compute if needed
                    "verticality": roof_verticality,
                    "curvature": roof_curvature,
                }

                # Only proceed if we have required features
                if (
                    roof_features["normals"] is not None
                    and roof_features["verticality"] is not None
                ):
                    # Classify roof
                    roof_result = self.roof_classifier.classify_roof(
                        points=building_points,
                        features=roof_features,
                        labels=labels_updated[building_mask],
                    )

                    # Apply roof classifications if successful
                    if roof_result.roof_type.value != "unknown":
                        # Map roof type to class
                        roof_class_map = {
                            "flat": 63,  # BUILDING_ROOF_FLAT
                            "gabled": 64,  # BUILDING_ROOF_GABLED
                            "hipped": 65,  # BUILDING_ROOF_HIPPED
                            "complex": 66,  # BUILDING_ROOF_COMPLEX
                        }
                        roof_class = roof_class_map.get(
                            roof_result.roof_type.value, 58
                        )  # Default to BUILDING_ROOF

                        # Get roof point indices in original array
                        building_indices = np.where(building_mask)[0]

                        # Apply main roof classification to all roof segments
                        for segment in roof_result.segments:
                            segment_indices = building_indices[segment.points]
                            labels_updated[segment_indices] = roof_class
                            all_classified_indices.update(segment_indices)

                        # Apply detailed classifications
                        if len(roof_result.ridge_lines) > 0:
                            ridge_indices = building_indices[roof_result.ridge_lines]
                            labels_updated[ridge_indices] = 67  # BUILDING_ROOF_RIDGE

                        if len(roof_result.edge_points) > 0:
                            edge_indices = building_indices[roof_result.edge_points]
                            labels_updated[edge_indices] = 68  # BUILDING_ROOF_EDGE

                        if len(roof_result.dormer_points) > 0:
                            dormer_indices = building_indices[roof_result.dormer_points]
                            labels_updated[dormer_indices] = 69  # BUILDING_DORMER

                        # Update statistics
                        stats["roof_type"] = roof_result.roof_type.value
                        stats["roof_confidence"] = float(roof_result.confidence)
                        stats["roof_segments"] = len(roof_result.segments)
                        stats["roof_points"] = roof_result.stats.get(
                            "total_roof_points", 0
                        )
                        stats["ridge_points"] = len(roof_result.ridge_lines)
                        stats["roof_edge_points"] = len(roof_result.edge_points)
                        stats["dormer_points"] = len(roof_result.dormer_points)

            except Exception as e:
                logger.warning(
                    f"Roof classification failed for building {building_id}: {e}"
                )
                stats["roof_classification_error"] = str(e)

        # 8.2. Enhanced LOD3 Classification (v3.4 - Phase 2.4)
        if self.enhanced_classifier is not None:
            try:
                # Prepare features for enhanced classification
                enhanced_features = {
                    "normals": normals[building_mask] if normals is not None else None,
                    "verticality": (
                        verticality[building_mask] if verticality is not None else None
                    ),
                    "curvature": (
                        curvature[building_mask] if curvature is not None else None
                    ),
                    "planarity": None,  # Will compute if needed
                }

                # Get ground elevation from building points
                building_z = building_points[:, 2]
                ground_elevation = float(np.min(building_z))

                # Classify building with enhanced LOD3 features
                enhanced_result = self.enhanced_classifier.classify_building(
                    points=building_points,
                    features=enhanced_features,
                    building_polygon=polygon,
                    ground_elevation=ground_elevation,
                )

                if enhanced_result.success:
                    # Get building point indices for mapping
                    building_indices = np.where(building_mask)[0]

                    # Apply enhanced classifications
                    # Priority: chimneys > balconies > roof > default
                    enhanced_labels = enhanced_result.point_labels

                    # Update labels for points with valid classifications
                    # Only update points not already classified as facades
                    for i, enhanced_label in enumerate(enhanced_labels):
                        if enhanced_label != 0:  # Skip unclassified
                            original_idx = building_indices[i]
                            # Don't override facade classifications (class 6)
                            if labels_updated[original_idx] != 6:
                                labels_updated[original_idx] = enhanced_label
                                all_classified_indices.add(original_idx)

                    # Update statistics
                    stats["enhanced_lod3_enabled"] = True
                    stats["roof_type_enhanced"] = (
                        enhanced_result.roof_result.roof_type.name
                        if enhanced_result.roof_result
                        else "N/A"
                    )
                    stats["num_chimneys"] = (
                        enhanced_result.chimney_result.num_chimneys
                        if enhanced_result.chimney_result
                        else 0
                    )
                    stats["num_balconies"] = (
                        enhanced_result.balcony_result.num_balconies
                        if enhanced_result.balcony_result
                        else 0
                    )

                    # Add architectural detail counts
                    if enhanced_result.chimney_result:
                        stats["chimney_points"] = len(
                            enhanced_result.chimney_result.all_chimney_points
                        )
                    if enhanced_result.balcony_result:
                        stats["balcony_points"] = len(
                            enhanced_result.balcony_result.all_balcony_points
                        )

                    logger.debug(
                        f"Building {building_id}: Enhanced LOD3 - "
                        f"Chimneys: {stats.get('num_chimneys', 0)}, "
                        f"Balconies: {stats.get('num_balconies', 0)}"
                    )

            except Exception as e:
                logger.warning(
                    f"Enhanced LOD3 classification failed for building "
                    f"{building_id}: {e}"
                )
                stats["enhanced_lod3_error"] = str(e)

        # 9. Statistiques finales
        stats["points_classified"] = len(all_classified_indices)
        if confidences:
            stats["avg_confidence"] = np.mean(confidences)

        # 🆕 Compute average rotation and scaling (v3.0.3)
        if stats["facades_rotated"] > 0:
            stats["avg_rotation_angle"] = np.degrees(
                stats["avg_rotation_angle"] / stats["facades_rotated"]
            )
        if stats["facades_scaled"] > 0:
            stats["avg_scale_factor"] = (
                stats["avg_scale_factor"] / stats["facades_scaled"]
            )

        return labels_updated, stats

    def _decompose_into_facades(
        self, building_id: int, polygon: "Polygon"
    ) -> List[FacadeSegment]:
        """
        Décomposer un polygone en 4 façades (N/S/E/W).

        Args:
            building_id: ID du bâtiment
            polygon: Polygone du bâtiment

        Returns:
            Liste de 4 FacadeSegment
        """
        # Obtenir bounding box
        minx, miny, maxx, maxy = polygon.bounds

        # Créer les 4 arêtes
        edges = {
            FacadeOrientation.NORTH: LineString([(minx, maxy), (maxx, maxy)]),
            FacadeOrientation.SOUTH: LineString([(minx, miny), (maxx, miny)]),
            FacadeOrientation.EAST: LineString([(maxx, miny), (maxx, maxy)]),
            FacadeOrientation.WEST: LineString([(minx, miny), (minx, maxy)]),
        }

        # Créer FacadeSegment pour chaque arête
        facades = []
        for orientation, edge_line in edges.items():
            facade = FacadeSegment(
                edge_line=edge_line,
                orientation=orientation,
                building_id=building_id,
                length=edge_line.length,
                centroid=np.array(edge_line.centroid.coords[0]),
                buffer_distance=self.initial_buffer,
                verticality_threshold=self.verticality_threshold,
            )
            facades.append(facade)

        return facades

    def _reconstruct_polygon_from_facades(
        self, facades: List[FacadeSegment]
    ) -> Optional["Polygon"]:
        """
        🆕 Reconstruire un polygone rectangulaire à partir des 4 façades adaptées.

        Cette méthode reconstruit un polygone en utilisant les arêtes ajustées
        de chaque façade, en calculant les intersections pour former un
        quadrilatère fermé.

        Args:
            facades: Liste des 4 façades (N, S, E, W)

        Returns:
            Polygon reconstruit ou None si impossible
        """
        if len(facades) != 4:
            logger.warning(f"Expected 4 facades, got {len(facades)}")
            return None

        # Organiser les façades par orientation
        facade_dict = {f.orientation: f for f in facades}

        # Utiliser arête ajustée si disponible, sinon arête originale
        north_line = (
            facade_dict[FacadeOrientation.NORTH].adjusted_edge_line
            if facade_dict[FacadeOrientation.NORTH].is_adapted
            else facade_dict[FacadeOrientation.NORTH].edge_line
        )
        south_line = (
            facade_dict[FacadeOrientation.SOUTH].adjusted_edge_line
            if facade_dict[FacadeOrientation.SOUTH].is_adapted
            else facade_dict[FacadeOrientation.SOUTH].edge_line
        )
        east_line = (
            facade_dict[FacadeOrientation.EAST].adjusted_edge_line
            if facade_dict[FacadeOrientation.EAST].is_adapted
            else facade_dict[FacadeOrientation.EAST].edge_line
        )
        west_line = (
            facade_dict[FacadeOrientation.WEST].adjusted_edge_line
            if facade_dict[FacadeOrientation.WEST].is_adapted
            else facade_dict[FacadeOrientation.WEST].edge_line
        )

        try:
            # Calculer les 4 coins en trouvant les intersections
            from shapely.geometry import Point, Polygon

            # Étendre les lignes pour assurer intersection
            def extend_line(line, factor=2.0):
                """Étendre une ligne des deux côtés."""
                coords = list(line.coords)
                if len(coords) < 2:
                    return line

                start = np.array(coords[0])
                end = np.array(coords[-1])
                direction = end - start
                length = np.linalg.norm(direction)

                if length > 0:
                    direction = direction / length
                    new_start = start - direction * length * factor
                    new_end = end + direction * length * factor

                    from shapely.geometry import LineString

                    return LineString([new_start, new_end])

                return line

            # Étendre toutes les lignes
            north_ext = extend_line(north_line)
            south_ext = extend_line(south_line)
            east_ext = extend_line(east_line)
            west_ext = extend_line(west_line)

            # Trouver les 4 coins (intersections)
            # Coin Nord-Ouest
            nw_intersection = north_ext.intersection(west_ext)
            if nw_intersection.is_empty:
                logger.warning("No intersection found for NW corner")
                return None
            nw_point = (
                Point(nw_intersection.coords[0])
                if hasattr(nw_intersection, "coords")
                else nw_intersection
            )

            # Coin Nord-Est
            ne_intersection = north_ext.intersection(east_ext)
            if ne_intersection.is_empty:
                logger.warning("No intersection found for NE corner")
                return None
            ne_point = (
                Point(ne_intersection.coords[0])
                if hasattr(ne_intersection, "coords")
                else ne_intersection
            )

            # Coin Sud-Est
            se_intersection = south_ext.intersection(east_ext)
            if se_intersection.is_empty:
                logger.warning("No intersection found for SE corner")
                return None
            se_point = (
                Point(se_intersection.coords[0])
                if hasattr(se_intersection, "coords")
                else se_intersection
            )

            # Coin Sud-Ouest
            sw_intersection = south_ext.intersection(west_ext)
            if sw_intersection.is_empty:
                logger.warning("No intersection found for SW corner")
                return None
            sw_point = (
                Point(sw_intersection.coords[0])
                if hasattr(sw_intersection, "coords")
                else sw_intersection
            )

            # Créer le polygone avec les 4 coins (sens horaire)
            corners = [
                nw_point.coords[0],
                ne_point.coords[0],
                se_point.coords[0],
                sw_point.coords[0],
            ]

            adapted_polygon = Polygon(corners)

            # Valider le polygone
            if not adapted_polygon.is_valid:
                logger.warning("Reconstructed polygon is invalid, attempting to fix")
                adapted_polygon = adapted_polygon.buffer(0)  # Tenter de réparer

            if adapted_polygon.is_valid:
                logger.info(
                    f"  Successfully reconstructed adapted polygon "
                    f"(area: {adapted_polygon.area:.1f}m²)"
                )
                return adapted_polygon
            else:
                logger.warning("Could not create valid adapted polygon")
                return None

        except Exception as e:
            logger.error(f"Error reconstructing polygon from facades: {e}")
            return None

    def _extract_building_points(
        self, polygon: "Polygon", points: np.ndarray, buffer: float = 10.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extraire les points proches d'un bâtiment avec bounding box optimisée.

        Améliorations:
        - Bounding box orientée (OBB) pour bâtiments obliques
        - Buffer adaptatif
        - Pré-filtrage spatial plus précis

        Args:
            polygon: Polygone du bâtiment
            points: Nuage de points [N, 3]
            buffer: Distance de buffer (m)

        Returns:
            (points_extracted, mask)
        """
        buffered = polygon.buffer(buffer)

        # 🔥 AMÉLIORATION 1: Bounding box orientée (OBB) pour pré-filtrage
        # Calculer l'orientation principale du bâtiment
        try:
            # Utiliser minimum rotated rectangle pour OBB
            mbr = polygon.minimum_rotated_rectangle
            mbr_coords = list(mbr.exterior.coords)

            if len(mbr_coords) >= 3:
                # Calculer l'angle du rectangle minimal
                dx = mbr_coords[1][0] - mbr_coords[0][0]
                dy = mbr_coords[1][1] - mbr_coords[0][1]
                angle = np.arctan2(dy, dx)

                # Centre du bâtiment
                center = np.array(polygon.centroid.coords[0])

                # Rotation inverse
                cos_a = np.cos(-angle)
                sin_a = np.sin(-angle)

                # Transformer points dans repère du bâtiment
                points_centered = points[:, :2] - center
                points_rotated = np.column_stack(
                    [
                        points_centered[:, 0] * cos_a - points_centered[:, 1] * sin_a,
                        points_centered[:, 0] * sin_a + points_centered[:, 1] * cos_a,
                    ]
                )

                # Limites OBB avec buffer
                mbr_rotated = np.array(
                    [[c[0] - center[0], c[1] - center[1]] for c in mbr_coords[:-1]]
                )
                mbr_rotated = np.column_stack(
                    [
                        mbr_rotated[:, 0] * cos_a - mbr_rotated[:, 1] * sin_a,
                        mbr_rotated[:, 0] * sin_a + mbr_rotated[:, 1] * cos_a,
                    ]
                )

                x_min, x_max = (
                    mbr_rotated[:, 0].min() - buffer,
                    mbr_rotated[:, 0].max() + buffer,
                )
                y_min, y_max = (
                    mbr_rotated[:, 1].min() - buffer,
                    mbr_rotated[:, 1].max() + buffer,
                )

                # Masque OBB
                obb_mask = (
                    (points_rotated[:, 0] >= x_min)
                    & (points_rotated[:, 0] <= x_max)
                    & (points_rotated[:, 1] >= y_min)
                    & (points_rotated[:, 1] <= y_max)
                )
            else:
                raise ValueError("MBR has insufficient coordinates")
        except:
            # Fallback: bounding box standard (AABB)
            bounds = buffered.bounds
            obb_mask = (
                (points[:, 0] >= bounds[0])
                & (points[:, 0] <= bounds[2])
                & (points[:, 1] >= bounds[1])
                & (points[:, 1] <= bounds[3])
            )

        # 🔥 AMÉLIORATION 2: Filtrage raffiné par polygone bufferisé
        if not HAS_SHAPELY:
            return points[obb_mask], obb_mask

        from shapely.vectorized import contains

        mask = obb_mask.copy()
        if np.any(obb_mask):
            mask[obb_mask] = contains(
                buffered, points[obb_mask, 0], points[obb_mask, 1]
            )

        return points[mask], mask
