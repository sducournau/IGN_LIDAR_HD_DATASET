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
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

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

    # Paramètres de traitement adaptatifs
    buffer_distance: float = 2.0  # Buffer pour capturer les points
    search_radius: float = 3.0  # Rayon de recherche de voisins
    verticality_threshold: float = 0.70  # Seuil pour mur (improved)

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
        """Identifier les points proches de cette façade."""
        # Créer zone de recherche (buffer autour de l'arête)
        search_zone = self.facade.edge_line.buffer(self.facade.buffer_distance)

        # Filtrage rapide par bounding box
        bounds = search_zone.bounds
        bbox_mask = (
            (self.points[:, 0] >= bounds[0])
            & (self.points[:, 0] <= bounds[2])
            & (self.points[:, 1] >= bounds[1])
            & (self.points[:, 1] <= bounds[3])
        )

        # Filtrage raffiné par polygone
        if not HAS_SHAPELY:
            self._candidate_mask = bbox_mask
        else:
            from shapely.vectorized import contains

            refined_mask = bbox_mask.copy()
            refined_mask[bbox_mask] = contains(
                search_zone, self.points[bbox_mask, 0], self.points[bbox_mask, 1]
            )
            self._candidate_mask = refined_mask

        # Stocker indices
        self.facade.point_indices = np.where(self._candidate_mask)[0]
        self.facade.n_points = len(self.facade.point_indices)

    def _classify_wall_points(self):
        """Classifier les points candidats (mur vs non-mur)."""
        if self.verticality is None:
            # Si pas de verticalité, tous les points sont considérés comme murs
            self._wall_mask = np.ones(self.facade.n_points, dtype=bool)
            self.facade.n_wall_points = self.facade.n_points
            return

        # Verticalité des points candidats
        candidate_verticality = self.verticality[self._candidate_mask]

        # Points de mur: haute verticalité
        self._wall_mask = candidate_verticality >= self.facade.verticality_threshold
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

    def adapt_facade_geometry(
        self,
        max_translation: float = 3.0,
        max_lateral_expansion: float = 2.0,
    ) -> "LineString":
        """
        🆕 Adapter la géométrie de la façade basée sur les points détectés.

        Cette méthode ajuste l'arête de la façade pour mieux correspondre
        à la distribution réelle des points du nuage.

        Args:
            max_translation: Translation maximale perpendiculaire (m)
            max_lateral_expansion: Extension latérale maximale (m)

        Returns:
            LineString ajustée
        """
        if not self.facade.is_processed:
            logger.warning("Facade must be processed before adaptation")
            return self.facade.edge_line

        adjusted_line = self.facade.edge_line

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
        initial_buffer: float = 2.0,
        verticality_threshold: float = 0.70,  # Improved from 0.60 to 0.70 for better facade detection
        min_point_density: float = 50.0,
        gap_detection_resolution: float = 0.5,
        adaptive_buffer_range: Tuple[float, float] = (0.5, 8.0),
        # 🆕 Paramètres d'adaptation géométrique
        enable_facade_adaptation: bool = True,
        max_translation: float = 3.0,
        max_lateral_expansion: float = 2.0,
        # Paramètres de classification
        building_class: int = 6,  # ASPRS building
        wall_subclass: Optional[int] = None,
        min_confidence: float = 0.35,  # ✅ ABAISSÉ de 0.50 à 0.35 pour capturer plus de façades
    ):
        """
        Initialiser le classificateur par façade.

        Args:
            initial_buffer: Buffer initial pour chaque façade
            verticality_threshold: Seuil de verticalité pour murs
                (0.70 = improved from 0.60 for better facade detection)
            min_point_density: Densité minimale attendue (pts/m²)
            gap_detection_resolution: Résolution pour détecter gaps
            adaptive_buffer_range: (min, max) pour buffers adaptatifs
            enable_facade_adaptation: Activer adaptation géométrique
            max_translation: Translation maximale perpendiculaire (m)
            max_lateral_expansion: Extension latérale maximale (m)
            building_class: Code de classification pour bâtiments
            wall_subclass: Code optionnel pour sous-classe "mur"
            min_confidence: Confiance minimale pour classifier
        """
        self.initial_buffer = initial_buffer
        self.verticality_threshold = verticality_threshold
        self.min_point_density = min_point_density
        self.gap_detection_resolution = gap_detection_resolution
        self.adaptive_buffer_range = adaptive_buffer_range

        self.enable_facade_adaptation = enable_facade_adaptation
        self.max_translation = max_translation
        self.max_lateral_expansion = max_lateral_expansion

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
        building_classes: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Classifier les points pour tous les bâtiments.

        Args:
            buildings_gdf: GeoDataFrame avec polygones BD TOPO
            points: Nuage de points [N, 3]
            heights: Hauteurs au-dessus du sol [N]
            labels: Labels actuels [N]
            normals: Normales [N, 3]
            verticality: Verticalité [N]
            building_classes: Classes existantes considérées comme bâtiments

        Returns:
            (labels_updated, statistics)
        """
        logger.info(
            f"Classifying {len(buildings_gdf)} buildings using facade-based approach"
        )

        labels_updated = labels.copy()
        building_classes = building_classes or [6]  # ASPRS building

        # Statistiques
        stats = {
            "buildings_processed": 0,
            "points_classified": 0,
            "facades_processed": 0,
            "avg_confidence": 0.0,
            "low_confidence_buildings": 0,
        }

        confidences = []

        # Traiter chaque bâtiment
        for idx, row in buildings_gdf.iterrows():
            polygon = row["geometry"]

            if not isinstance(polygon, Polygon):
                continue

            # Classifier ce bâtiment
            building_labels, building_stats = self.classify_single_building(
                building_id=idx,
                polygon=polygon,
                points=points,
                heights=heights,
                labels=labels_updated,
                normals=normals,
                verticality=verticality,
            )

            # Mettre à jour labels
            labels_updated = building_labels

            # Accumuler stats
            stats["buildings_processed"] += 1
            stats["points_classified"] += building_stats["points_classified"]
            stats["facades_processed"] += building_stats["facades_processed"]

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
    ) -> Tuple[np.ndarray, Dict]:
        """
        Classifier les points d'un bâtiment individuel par analyse de façades.

        Args:
            building_id: ID du bâtiment
            polygon: Polygone BD TOPO
            points: Nuage de points complet [N, 3]
            heights: Hauteurs au-dessus du sol [N]
            labels: Labels actuels [N]
            normals: Normales [N, 3]
            verticality: Verticalité [N]

        Returns:
            (labels_updated, statistics)
        """
        labels_updated = labels.copy()

        stats = {
            "points_classified": 0,
            "facades_processed": 0,
            "avg_confidence": 0.0,
        }

        # 1. Décomposer en 4 façades
        facades = self._decompose_into_facades(building_id, polygon)

        if not facades:
            return labels_updated, stats

        # 2. Estimer hauteur du bâtiment
        building_points, building_mask = self._extract_building_points(polygon, points)
        if len(building_points) == 0:
            return labels_updated, stats

        building_height = heights[building_mask].max()

        # 3. Traiter chaque façade
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

            # 🆕 Adapter géométrie de la façade si activé
            if self.enable_facade_adaptation:
                if (
                    processed_facade.needs_translation
                    or processed_facade.needs_lateral_expansion
                ):

                    adapted_line = processor.adapt_facade_geometry(
                        max_translation=self.max_translation,
                        max_lateral_expansion=self.max_lateral_expansion,
                    )

                    if processed_facade.is_adapted:
                        self.n_facades_adapted += 1
                        logger.debug(
                            f"  Adapted {processed_facade.orientation.value} facade: "
                            f"translation={processed_facade.translation_offset:.2f}m, "
                            f"expansion=({processed_facade.lateral_expansion[0]:.2f}m, "
                            f"{processed_facade.lateral_expansion[1]:.2f}m)"
                        )

            # Classifier les points si confiance suffisante
            if processed_facade.confidence_score >= self.min_confidence:
                if processed_facade.point_indices is not None:
                    # Classifier comme bâtiment
                    wall_indices = processed_facade.point_indices[processor._wall_mask]
                    labels_updated[wall_indices] = self.building_class

                    # Ajouter à l'ensemble des indices classifiés
                    all_classified_indices.update(wall_indices)

                    confidences.append(processed_facade.confidence_score)

            stats["facades_processed"] += 1

        # 🆕 4. Reconstruire polygone adapté (si au moins une façade adaptée)
        adapted_facades = [f for f in facades if f.is_adapted]
        if adapted_facades:
            adapted_polygon = self._reconstruct_polygon_from_facades(facades)
            stats["adapted_polygon"] = adapted_polygon

        # 5. Statistiques finales
        stats["points_classified"] = len(all_classified_indices)
        if confidences:
            stats["avg_confidence"] = np.mean(confidences)

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
        """Extraire les points proches d'un bâtiment."""
        buffered = polygon.buffer(buffer)

        # Bounding box filter
        bounds = buffered.bounds
        bbox_mask = (
            (points[:, 0] >= bounds[0])
            & (points[:, 0] <= bounds[2])
            & (points[:, 1] >= bounds[1])
            & (points[:, 1] <= bounds[3])
        )

        # Refined filter
        if not HAS_SHAPELY:
            return points[bbox_mask], bbox_mask

        from shapely.vectorized import contains

        mask = bbox_mask.copy()
        mask[bbox_mask] = contains(buffered, points[bbox_mask, 0], points[bbox_mask, 1])

        return points[mask], mask
