"""
🏢 Audit de Détection de Bâtiments - Analyse par Façade
========================================================

Script d'audit complet pour améliorer la détection individuelle de bâtiments
à partir de BD TOPO avec traitement détaillé de chaque façade de la bounding box.

Fonctionnalités:
1. ✅ Analyse individuelle de chaque bâtiment BD TOPO
2. ✅ Segmentation automatique des 4 façades (N, S, E, W) de la bounding box
3. ✅ Détection de points manquants par façade
4. ✅ Calcul de buffers adaptatifs par façade
5. ✅ Détection de gaps/occlusions sur chaque façade
6. ✅ Scoring de qualité par façade (couverture, verticalité, densité)
7. ✅ Génération de rapport détaillé avec recommandations
8. ✅ Visualisation des résultats par façade

Usage:
    python scripts/audit_building_facade_detection.py \
        --laz_file /path/to/tile.laz \
        --bd_topo_file /path/to/buildings.geojson \
        --output_dir /path/to/audit_results \
        --visualize

Author: Building Detection Audit v5.5
Date: October 2025
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import json

import numpy as np
import pandas as pd

try:
    from shapely.geometry import Polygon, LineString, Point, box
    from shapely.ops import unary_union
    import geopandas as gpd
    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MPLPolygon
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FacadeOrientation(str, Enum):
    """Orientation des façades."""
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    UNKNOWN = "unknown"


@dataclass
class FacadeAnalysis:
    """Résultats d'analyse pour une façade individuelle."""
    orientation: FacadeOrientation
    edge_line: LineString  # Ligne représentant l'arête de la façade
    
    # Statistiques de points
    n_points_expected: int = 0     # Points attendus (basé sur longueur × hauteur)
    n_points_detected: int = 0     # Points réellement détectés
    n_wall_points: int = 0         # Points verticaux (murs)
    coverage_ratio: float = 0.0    # Ratio de couverture (0-1)
    
    # Métriques géométriques
    length: float = 0.0            # Longueur de la façade (m)
    avg_verticality: float = 0.0   # Verticalité moyenne des points
    avg_planarity: float = 0.0     # Planéité moyenne
    point_density: float = 0.0     # Points par m² de façade
    
    # Détection de gaps
    has_gaps: bool = False
    gap_segments: List[Tuple[float, float]] = field(default_factory=list)  # [(start, end), ...]
    gap_total_length: float = 0.0  # Longueur totale des gaps (m)
    gap_ratio: float = 0.0         # Ratio de gaps (0-1)
    
    # Buffer adaptatif recommandé
    recommended_buffer: float = 1.0
    
    # Score de qualité (0-100)
    quality_score: float = 0.0
    
    # Diagnostic
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class BuildingAudit:
    """Résultats d'audit complet pour un bâtiment."""
    building_id: int
    polygon: Polygon
    
    # Statistiques globales
    total_points: int = 0
    building_height: float = 0.0
    perimeter: float = 0.0
    area: float = 0.0
    
    # Analyse par façade
    facades: Dict[FacadeOrientation, FacadeAnalysis] = field(default_factory=dict)
    
    # Scores globaux
    overall_coverage: float = 0.0
    overall_quality: float = 0.0
    
    # Buffers recommandés
    recommended_buffer_min: float = 1.0
    recommended_buffer_max: float = 5.0
    recommended_buffer_adaptive: Dict[FacadeOrientation, float] = field(default_factory=dict)
    
    # Diagnostic global
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class BuildingFacadeAuditor:
    """
    Auditeur de détection de bâtiments avec analyse détaillée par façade.
    
    Processus:
    1. Charger LAZ + BD TOPO
    2. Pour chaque bâtiment BD TOPO:
       a. Extraire bounding box et décomposer en 4 façades
       b. Pour chaque façade:
          - Identifier les points proches
          - Calculer métriques de couverture/qualité
          - Détecter gaps et occlusions
          - Recommander buffer adaptatif
       c. Calculer scores globaux
       d. Générer diagnostic et recommandations
    3. Produire rapport global et par bâtiment
    """
    
    def __init__(
        self,
        laz_file: Path,
        bd_topo_file: Path,
        output_dir: Path,
        # Paramètres d'analyse
        facade_search_distance: float = 3.0,  # Distance de recherche autour de chaque façade
        verticality_threshold: float = 0.55,   # Seuil pour considérer un point comme mur
        min_point_density: float = 50.0,       # Points/m² minimum attendu
        gap_detection_resolution: float = 0.5,  # Résolution pour détecter les gaps (m)
        # Paramètres de buffers
        buffer_min: float = 0.5,
        buffer_max: float = 8.0,
        buffer_step: float = 0.5,
    ):
        """
        Initialiser l'auditeur.
        
        Args:
            laz_file: Chemin vers le fichier LAZ
            bd_topo_file: Chemin vers les bâtiments BD TOPO (GeoJSON/SHP)
            output_dir: Répertoire de sortie pour les résultats
            facade_search_distance: Distance de recherche autour de chaque façade
            verticality_threshold: Seuil de verticalité pour murs
            min_point_density: Densité minimale attendue (pts/m²)
            gap_detection_resolution: Résolution pour détecter gaps
            buffer_min: Buffer minimum à tester
            buffer_max: Buffer maximum à tester
            buffer_step: Pas d'incrément pour tester les buffers
        """
        if not HAS_LASPY:
            raise ImportError("laspy is required. Install with: pip install laspy")
        if not HAS_SPATIAL:
            raise ImportError("shapely and geopandas required. Install with: pip install shapely geopandas")
        
        self.laz_file = Path(laz_file)
        self.bd_topo_file = Path(bd_topo_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Paramètres
        self.facade_search_distance = facade_search_distance
        self.verticality_threshold = verticality_threshold
        self.min_point_density = min_point_density
        self.gap_detection_resolution = gap_detection_resolution
        self.buffer_min = buffer_min
        self.buffer_max = buffer_max
        self.buffer_step = buffer_step
        
        # Data
        self.points = None
        self.heights = None
        self.normals = None
        self.verticality = None
        self.buildings_gdf = None
        self.audits: List[BuildingAudit] = []
        
        logger.info(f"BuildingFacadeAuditor initialized")
        logger.info(f"  LAZ file: {self.laz_file}")
        logger.info(f"  BD TOPO: {self.bd_topo_file}")
        logger.info(f"  Output: {self.output_dir}")
    
    def run(self, max_buildings: Optional[int] = None, visualize: bool = False):
        """
        Exécuter l'audit complet.
        
        Args:
            max_buildings: Limiter à N bâtiments (pour tests)
            visualize: Générer des visualisations
        """
        logger.info("=" * 80)
        logger.info("🏢 AUDIT DE DÉTECTION DE BÂTIMENTS PAR FAÇADE")
        logger.info("=" * 80)
        
        # 1. Charger les données
        logger.info("\n[1/5] Chargement des données...")
        self._load_laz()
        self._load_bd_topo()
        self._compute_features()
        
        # 2. Analyser chaque bâtiment
        logger.info(f"\n[2/5] Analyse de {len(self.buildings_gdf)} bâtiments...")
        if max_buildings:
            logger.info(f"  (limité à {max_buildings} pour ce test)")
            buildings_to_process = self.buildings_gdf.head(max_buildings)
        else:
            buildings_to_process = self.buildings_gdf
        
        for idx, row in buildings_to_process.iterrows():
            audit = self._audit_single_building(idx, row['geometry'])
            self.audits.append(audit)
            
            if (idx + 1) % 10 == 0:
                logger.info(f"  Processed {idx + 1}/{len(buildings_to_process)} buildings...")
        
        logger.info(f"✅ Analyzed {len(self.audits)} buildings")
        
        # 3. Générer statistiques globales
        logger.info("\n[3/5] Génération des statistiques globales...")
        self._compute_global_statistics()
        
        # 4. Générer rapports
        logger.info("\n[4/5] Génération des rapports...")
        self._generate_reports()
        
        # 5. Visualisations (optionnel)
        if visualize and HAS_MATPLOTLIB:
            logger.info("\n[5/5] Génération des visualisations...")
            self._generate_visualizations()
        else:
            logger.info("\n[5/5] Visualisations désactivées")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ AUDIT TERMINÉ")
        logger.info("=" * 80)
        logger.info(f"📁 Résultats dans: {self.output_dir}")
    
    def _load_laz(self):
        """Charger le fichier LAZ."""
        logger.info(f"  Loading LAZ: {self.laz_file.name}")
        
        with laspy.open(self.laz_file) as laz:
            las = laz.read()
            self.points = np.vstack([las.x, las.y, las.z]).T
        
        logger.info(f"  ✅ Loaded {len(self.points):,} points")
    
    def _load_bd_topo(self):
        """Charger les bâtiments BD TOPO."""
        logger.info(f"  Loading BD TOPO: {self.bd_topo_file.name}")
        
        self.buildings_gdf = gpd.read_file(self.bd_topo_file)
        
        # Filtrer uniquement les polygones
        self.buildings_gdf = self.buildings_gdf[
            self.buildings_gdf.geometry.type == 'Polygon'
        ]
        
        logger.info(f"  ✅ Loaded {len(self.buildings_gdf)} buildings")
    
    def _compute_features(self):
        """Calculer les features géométriques (normals, verticality)."""
        logger.info("  Computing geometric features...")
        
        from sklearn.neighbors import NearestNeighbors
        from sklearn.decomposition import PCA
        
        # Compute normals using PCA on k-nearest neighbors
        k = 20
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(self.points)
        distances, indices = nbrs.kneighbors(self.points)
        
        normals = np.zeros((len(self.points), 3))
        
        for i in range(len(self.points)):
            neighbors = self.points[indices[i]]
            neighbors_centered = neighbors - neighbors.mean(axis=0)
            
            # PCA to find normal
            pca = PCA(n_components=3)
            pca.fit(neighbors_centered)
            
            # Normal is the direction of smallest variance (last component)
            normal = pca.components_[-1]
            
            # Orient normal upward (positive Z component)
            if normal[2] < 0:
                normal = -normal
            
            normals[i] = normal
        
        self.normals = normals
        
        # Compute verticality (1 - |normal_z|)
        # High verticality (close to 1) = vertical surface (wall)
        # Low verticality (close to 0) = horizontal surface (roof/ground)
        self.verticality = 1.0 - np.abs(self.normals[:, 2])
        
        # Compute heights (Z relative to ground)
        z_ground = np.percentile(self.points[:, 2], 5)  # 5th percentile as ground
        self.heights = self.points[:, 2] - z_ground
        
        logger.info(f"  ✅ Features computed")
        logger.info(f"     Verticality range: {self.verticality.min():.2f} - {self.verticality.max():.2f}")
        logger.info(f"     Height range: {self.heights.min():.1f} - {self.heights.max():.1f} m")
    
    def _audit_single_building(self, building_id: int, polygon: Polygon) -> BuildingAudit:
        """
        Auditer un bâtiment individuel avec analyse détaillée par façade.
        
        Args:
            building_id: ID du bâtiment
            polygon: Polygone BD TOPO
            
        Returns:
            Audit complet du bâtiment
        """
        audit = BuildingAudit(
            building_id=building_id,
            polygon=polygon,
            perimeter=polygon.length,
            area=polygon.area
        )
        
        # 1. Extraire les points du bâtiment (buffer initial)
        building_points, building_mask = self._extract_building_points(polygon)
        
        if len(building_points) == 0:
            audit.critical_issues.append("No points found near building")
            return audit
        
        audit.total_points = len(building_points)
        audit.building_height = self.heights[building_mask].max()
        
        # 2. Décomposer le polygone en 4 façades (N, S, E, W)
        facades_edges = self._decompose_into_facades(polygon)
        
        # 3. Analyser chaque façade
        for orientation, edge_line in facades_edges.items():
            facade_analysis = self._analyze_facade(
                edge_line=edge_line,
                orientation=orientation,
                building_points=building_points,
                building_mask=building_mask,
                building_height=audit.building_height
            )
            audit.facades[orientation] = facade_analysis
        
        # 4. Calculer les scores globaux
        self._compute_building_scores(audit)
        
        # 5. Générer diagnostic et recommandations
        self._generate_building_recommendations(audit)
        
        return audit
    
    def _extract_building_points(
        self,
        polygon: Polygon,
        buffer: float = 5.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extraire les points proches d'un bâtiment.
        
        Args:
            polygon: Polygone du bâtiment
            buffer: Distance de recherche autour du polygone
            
        Returns:
            (points, mask) - Points extraits et masque sur self.points
        """
        buffered = polygon.buffer(buffer)
        
        # Bounding box filter (fast)
        bounds = buffered.bounds
        bbox_mask = (
            (self.points[:, 0] >= bounds[0]) &
            (self.points[:, 0] <= bounds[2]) &
            (self.points[:, 1] >= bounds[1]) &
            (self.points[:, 1] <= bounds[3])
        )
        
        # Refined polygon filter
        from shapely.vectorized import contains
        mask = bbox_mask.copy()
        mask[bbox_mask] = contains(buffered, self.points[bbox_mask, 0], self.points[bbox_mask, 1])
        
        return self.points[mask], mask
    
    def _decompose_into_facades(self, polygon: Polygon) -> Dict[FacadeOrientation, LineString]:
        """
        Décomposer un polygone en 4 façades basées sur l'orientation.
        
        Stratégie:
        1. Obtenir les 4 côtés de la bounding box (envelope)
        2. Assigner orientation Nord/Sud/Est/Ouest à chaque côté
        
        Args:
            polygon: Polygone du bâtiment
            
        Returns:
            Dict {orientation: LineString} pour chaque façade
        """
        # Get bounding box
        minx, miny, maxx, maxy = polygon.bounds
        
        # Create 4 edges
        facades = {
            FacadeOrientation.NORTH: LineString([(minx, maxy), (maxx, maxy)]),  # Top edge
            FacadeOrientation.SOUTH: LineString([(minx, miny), (maxx, miny)]),  # Bottom edge
            FacadeOrientation.EAST: LineString([(maxx, miny), (maxx, maxy)]),   # Right edge
            FacadeOrientation.WEST: LineString([(minx, miny), (minx, maxy)]),   # Left edge
        }
        
        return facades
    
    def _analyze_facade(
        self,
        edge_line: LineString,
        orientation: FacadeOrientation,
        building_points: np.ndarray,
        building_mask: np.ndarray,
        building_height: float
    ) -> FacadeAnalysis:
        """
        Analyser une façade individuelle.
        
        Args:
            edge_line: Ligne représentant l'arête de la façade
            orientation: Orientation de la façade
            building_points: Points du bâtiment
            building_mask: Masque sur self.points pour le bâtiment
            building_height: Hauteur du bâtiment
            
        Returns:
            Analyse complète de la façade
        """
        analysis = FacadeAnalysis(
            orientation=orientation,
            edge_line=edge_line,
            length=edge_line.length
        )
        
        # 1. Trouver les points proches de cette façade
        facade_buffer = edge_line.buffer(self.facade_search_distance)
        
        from shapely.vectorized import contains
        facade_mask = building_mask.copy()
        facade_points_bool = contains(
            facade_buffer,
            self.points[building_mask, 0],
            self.points[building_mask, 1]
        )
        facade_mask[building_mask] = facade_points_bool
        
        facade_points = self.points[facade_mask]
        n_facade_points = len(facade_points)
        
        if n_facade_points == 0:
            analysis.issues.append(f"No points detected near {orientation.value} facade")
            analysis.quality_score = 0.0
            return analysis
        
        analysis.n_points_detected = n_facade_points
        
        # 2. Calculer les points attendus (basé sur surface de la façade)
        facade_area = analysis.length * building_height
        analysis.n_points_expected = int(facade_area * self.min_point_density)
        
        # 3. Ratio de couverture
        if analysis.n_points_expected > 0:
            analysis.coverage_ratio = min(1.0, n_facade_points / analysis.n_points_expected)
        
        # 4. Analyser les points de mur (verticality)
        facade_verticality = self.verticality[facade_mask]
        wall_mask = facade_verticality >= self.verticality_threshold
        analysis.n_wall_points = wall_mask.sum()
        analysis.avg_verticality = facade_verticality.mean()
        
        # 5. Densité de points
        if facade_area > 0:
            analysis.point_density = n_facade_points / facade_area
        
        # 6. Détecter les gaps le long de la façade
        gaps = self._detect_facade_gaps(edge_line, facade_points)
        if gaps:
            analysis.has_gaps = True
            analysis.gap_segments = gaps
            analysis.gap_total_length = sum(end - start for start, end in gaps)
            analysis.gap_ratio = analysis.gap_total_length / analysis.length
        
        # 7. Recommander buffer adaptatif
        analysis.recommended_buffer = self._compute_adaptive_buffer_for_facade(analysis)
        
        # 8. Calculer score de qualité
        analysis.quality_score = self._compute_facade_quality_score(analysis)
        
        # 9. Générer issues et recommandations
        self._generate_facade_recommendations(analysis)
        
        return analysis
    
    def _detect_facade_gaps(
        self,
        edge_line: LineString,
        facade_points: np.ndarray
    ) -> List[Tuple[float, float]]:
        """
        Détecter les segments sans points le long d'une façade.
        
        Args:
            edge_line: Ligne de la façade
            facade_points: Points proches de la façade
            
        Returns:
            Liste de segments [(start_distance, end_distance), ...]
        """
        if len(facade_points) == 0:
            return [(0.0, edge_line.length)]
        
        # Projeter les points sur la ligne de la façade
        from shapely.geometry import Point
        
        projected_distances = []
        for point in facade_points:
            pt = Point(point[:2])
            dist = edge_line.project(pt)
            projected_distances.append(dist)
        
        projected_distances = np.array(sorted(projected_distances))
        
        # Détecter les gaps (segments sans points)
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
        if edge_line.length - projected_distances[-1] > gap_threshold:
            gaps.append((projected_distances[-1], edge_line.length))
        
        return gaps
    
    def _compute_adaptive_buffer_for_facade(self, analysis: FacadeAnalysis) -> float:
        """
        Calculer le buffer adaptatif recommandé pour une façade.
        
        Stratégie:
        - Faible couverture → buffer large
        - Gaps détectés → buffer large
        - Bonne couverture → buffer minimal
        
        Args:
            analysis: Analyse de la façade
            
        Returns:
            Buffer recommandé (m)
        """
        # Base buffer
        buffer = self.buffer_min
        
        # Ajuster selon la couverture
        if analysis.coverage_ratio < 0.3:
            # Très faible couverture → buffer maximum
            buffer = self.buffer_max
        elif analysis.coverage_ratio < 0.6:
            # Couverture moyenne → buffer moyen-élevé
            buffer = (self.buffer_min + self.buffer_max) * 0.7
        elif analysis.coverage_ratio < 0.8:
            # Bonne couverture → buffer moyen
            buffer = (self.buffer_min + self.buffer_max) * 0.5
        
        # Ajuster selon les gaps
        if analysis.has_gaps and analysis.gap_ratio > 0.2:
            # Gaps significatifs → augmenter buffer
            buffer = min(buffer * 1.5, self.buffer_max)
        
        # Ajuster selon la densité de points
        if analysis.point_density < self.min_point_density * 0.5:
            # Densité très faible → augmenter buffer
            buffer = min(buffer * 1.3, self.buffer_max)
        
        return round(buffer, 1)
    
    def _compute_facade_quality_score(self, analysis: FacadeAnalysis) -> float:
        """
        Calculer le score de qualité d'une façade (0-100).
        
        Critères:
        - Couverture: 40%
        - Verticalité moyenne: 20%
        - Absence de gaps: 20%
        - Densité de points: 20%
        
        Args:
            analysis: Analyse de la façade
            
        Returns:
            Score de qualité (0-100)
        """
        # Couverture (40 points)
        coverage_score = analysis.coverage_ratio * 40.0
        
        # Verticalité (20 points)
        verticality_score = analysis.avg_verticality * 20.0
        
        # Gaps (20 points)
        gap_score = (1.0 - analysis.gap_ratio) * 20.0
        
        # Densité (20 points)
        density_ratio = min(1.0, analysis.point_density / self.min_point_density)
        density_score = density_ratio * 20.0
        
        total_score = coverage_score + verticality_score + gap_score + density_score
        
        return round(total_score, 1)
    
    def _generate_facade_recommendations(self, analysis: FacadeAnalysis):
        """Générer issues et recommandations pour une façade."""
        # Issues
        if analysis.coverage_ratio < 0.3:
            analysis.issues.append(f"Very low coverage ({analysis.coverage_ratio:.1%})")
        elif analysis.coverage_ratio < 0.6:
            analysis.issues.append(f"Low coverage ({analysis.coverage_ratio:.1%})")
        
        if analysis.has_gaps and analysis.gap_ratio > 0.3:
            analysis.issues.append(f"Significant gaps detected ({analysis.gap_ratio:.1%} of facade)")
        
        if analysis.avg_verticality < 0.5:
            analysis.issues.append(f"Low verticality ({analysis.avg_verticality:.2f}), may not be a wall")
        
        if analysis.point_density < self.min_point_density * 0.5:
            analysis.issues.append(f"Very low point density ({analysis.point_density:.1f} pts/m²)")
        
        # Recommendations
        if analysis.coverage_ratio < 0.7:
            analysis.recommendations.append(
                f"Increase buffer to {analysis.recommended_buffer:.1f}m"
            )
        
        if analysis.has_gaps:
            analysis.recommendations.append(
                f"Apply adaptive buffering per segment to fill {len(analysis.gap_segments)} gaps"
            )
        
        if analysis.n_wall_points < analysis.n_points_detected * 0.3:
            analysis.recommendations.append(
                "Low wall point ratio - verify verticality threshold or check for occlusion"
            )
    
    def _compute_building_scores(self, audit: BuildingAudit):
        """Calculer les scores globaux du bâtiment."""
        if not audit.facades:
            audit.overall_quality = 0.0
            return
        
        # Moyenne des scores de qualité des façades
        facade_scores = [f.quality_score for f in audit.facades.values()]
        audit.overall_quality = np.mean(facade_scores)
        
        # Couverture globale (moyenne pondérée par longueur de façade)
        total_length = sum(f.length for f in audit.facades.values())
        if total_length > 0:
            weighted_coverage = sum(
                f.coverage_ratio * f.length for f in audit.facades.values()
            ) / total_length
            audit.overall_coverage = weighted_coverage
        
        # Buffers recommandés
        buffers = [f.recommended_buffer for f in audit.facades.values()]
        audit.recommended_buffer_min = min(buffers)
        audit.recommended_buffer_max = max(buffers)
        
        for orientation, facade in audit.facades.items():
            audit.recommended_buffer_adaptive[orientation] = facade.recommended_buffer
    
    def _generate_building_recommendations(self, audit: BuildingAudit):
        """Générer diagnostic et recommandations globales pour le bâtiment."""
        # Critical issues
        if audit.total_points == 0:
            audit.critical_issues.append("No points found near building - check BD TOPO alignment")
            return
        
        if audit.overall_quality < 30:
            audit.critical_issues.append(f"Very poor detection quality ({audit.overall_quality:.1f}/100)")
        
        # Count facades with issues
        facades_with_low_coverage = sum(1 for f in audit.facades.values() if f.coverage_ratio < 0.5)
        if facades_with_low_coverage >= 3:
            audit.critical_issues.append(f"{facades_with_low_coverage}/4 facades have low coverage")
        
        # Warnings
        if audit.overall_quality < 60:
            audit.warnings.append(f"Moderate detection quality ({audit.overall_quality:.1f}/100)")
        
        facades_with_gaps = sum(1 for f in audit.facades.values() if f.has_gaps)
        if facades_with_gaps >= 2:
            audit.warnings.append(f"{facades_with_gaps}/4 facades have gaps")
        
        # Recommendations
        if audit.overall_coverage < 0.7:
            audit.recommendations.append(
                f"Increase buffer range to [{audit.recommended_buffer_min:.1f}, "
                f"{audit.recommended_buffer_max:.1f}]m"
            )
        
        if len(audit.recommended_buffer_adaptive) > 0:
            audit.recommendations.append(
                "Apply facade-specific adaptive buffers for optimal coverage"
            )
        
        # Specific facade recommendations
        for orientation, facade in audit.facades.items():
            if facade.quality_score < 40:
                audit.recommendations.append(
                    f"{orientation.value.capitalize()} facade needs attention: "
                    f"score={facade.quality_score:.1f}, buffer={facade.recommended_buffer:.1f}m"
                )
    
    def _compute_global_statistics(self):
        """Calculer les statistiques globales sur tous les bâtiments."""
        if not self.audits:
            logger.warning("No audits to compute statistics")
            return
        
        # Global scores
        quality_scores = [a.overall_quality for a in self.audits]
        coverage_ratios = [a.overall_coverage for a in self.audits]
        
        logger.info(f"\n📊 STATISTIQUES GLOBALES")
        logger.info(f"{'=' * 80}")
        logger.info(f"Total buildings analyzed: {len(self.audits)}")
        logger.info(f"\nQuality scores:")
        logger.info(f"  Mean: {np.mean(quality_scores):.1f}/100")
        logger.info(f"  Median: {np.median(quality_scores):.1f}/100")
        logger.info(f"  Min: {np.min(quality_scores):.1f}/100")
        logger.info(f"  Max: {np.max(quality_scores):.1f}/100")
        
        logger.info(f"\nCoverage ratios:")
        logger.info(f"  Mean: {np.mean(coverage_ratios):.1%}")
        logger.info(f"  Median: {np.median(coverage_ratios):.1%}")
        
        # Count issues
        n_critical = sum(1 for a in self.audits if a.critical_issues)
        n_warnings = sum(1 for a in self.audits if a.warnings)
        
        logger.info(f"\nIssues:")
        logger.info(f"  Buildings with critical issues: {n_critical} ({100*n_critical/len(self.audits):.1f}%)")
        logger.info(f"  Buildings with warnings: {n_warnings} ({100*n_warnings/len(self.audits):.1f}%)")
        
        # Buffer recommendations
        all_buffers = [
            buf for audit in self.audits
            for buf in audit.recommended_buffer_adaptive.values()
        ]
        if all_buffers:
            logger.info(f"\nRecommended buffers:")
            logger.info(f"  Mean: {np.mean(all_buffers):.1f}m")
            logger.info(f"  Median: {np.median(all_buffers):.1f}m")
            logger.info(f"  Range: [{np.min(all_buffers):.1f}, {np.max(all_buffers):.1f}]m")
    
    def _generate_reports(self):
        """Générer les rapports JSON et CSV."""
        # 1. Rapport JSON détaillé
        json_report = {
            "metadata": {
                "laz_file": str(self.laz_file),
                "bd_topo_file": str(self.bd_topo_file),
                "n_buildings_analyzed": len(self.audits),
                "parameters": {
                    "facade_search_distance": self.facade_search_distance,
                    "verticality_threshold": self.verticality_threshold,
                    "min_point_density": self.min_point_density,
                    "gap_detection_resolution": self.gap_detection_resolution,
                    "buffer_range": [self.buffer_min, self.buffer_max],
                }
            },
            "buildings": []
        }
        
        for audit in self.audits:
            building_data = {
                "building_id": int(audit.building_id),
                "total_points": int(audit.total_points),
                "building_height": float(audit.building_height),
                "perimeter": float(audit.perimeter),
                "area": float(audit.area),
                "overall_quality": float(audit.overall_quality),
                "overall_coverage": float(audit.overall_coverage),
                "recommended_buffer_min": float(audit.recommended_buffer_min),
                "recommended_buffer_max": float(audit.recommended_buffer_max),
                "critical_issues": audit.critical_issues,
                "warnings": audit.warnings,
                "recommendations": audit.recommendations,
                "facades": {}
            }
            
            for orientation, facade in audit.facades.items():
                building_data["facades"][orientation.value] = {
                    "length": float(facade.length),
                    "n_points_detected": int(facade.n_points_detected),
                    "n_points_expected": int(facade.n_points_expected),
                    "coverage_ratio": float(facade.coverage_ratio),
                    "quality_score": float(facade.quality_score),
                    "recommended_buffer": float(facade.recommended_buffer),
                    "has_gaps": facade.has_gaps,
                    "gap_ratio": float(facade.gap_ratio),
                    "issues": facade.issues,
                    "recommendations": facade.recommendations,
                }
            
            json_report["buildings"].append(building_data)
        
        json_path = self.output_dir / "audit_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False)
        logger.info(f"  ✅ JSON report: {json_path}")
        
        # 2. Rapport CSV (une ligne par bâtiment)
        rows = []
        for audit in self.audits:
            row = {
                "building_id": audit.building_id,
                "total_points": audit.total_points,
                "height_m": audit.building_height,
                "perimeter_m": audit.perimeter,
                "area_m2": audit.area,
                "quality_score": audit.overall_quality,
                "coverage_ratio": audit.overall_coverage,
                "buffer_min_m": audit.recommended_buffer_min,
                "buffer_max_m": audit.recommended_buffer_max,
                "n_critical_issues": len(audit.critical_issues),
                "n_warnings": len(audit.warnings),
            }
            
            # Add per-facade metrics
            for orientation in [FacadeOrientation.NORTH, FacadeOrientation.SOUTH,
                               FacadeOrientation.EAST, FacadeOrientation.WEST]:
                if orientation in audit.facades:
                    facade = audit.facades[orientation]
                    prefix = orientation.value[:1].upper()  # N, S, E, W
                    row[f"{prefix}_coverage"] = facade.coverage_ratio
                    row[f"{prefix}_quality"] = facade.quality_score
                    row[f"{prefix}_buffer_m"] = facade.recommended_buffer
                    row[f"{prefix}_has_gaps"] = facade.has_gaps
                else:
                    row[f"{prefix}_coverage"] = None
                    row[f"{prefix}_quality"] = None
                    row[f"{prefix}_buffer_m"] = None
                    row[f"{prefix}_has_gaps"] = None
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_path = self.output_dir / "audit_summary.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"  ✅ CSV summary: {csv_path}")
        
        # 3. Rapport texte avec recommandations prioritaires
        txt_path = self.output_dir / "audit_recommendations.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("🏢 AUDIT DE DÉTECTION DE BÂTIMENTS - RECOMMANDATIONS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("📊 RÉSUMÉ GLOBAL\n")
            f.write("-" * 80 + "\n")
            f.write(f"Bâtiments analysés: {len(self.audits)}\n")
            
            quality_scores = [a.overall_quality for a in self.audits]
            f.write(f"Score de qualité moyen: {np.mean(quality_scores):.1f}/100\n")
            
            n_critical = sum(1 for a in self.audits if a.critical_issues)
            f.write(f"Bâtiments avec problèmes critiques: {n_critical}\n\n")
            
            # Top 10 worst buildings
            f.write("🔴 TOP 10 - BÂTIMENTS NÉCESSITANT ATTENTION\n")
            f.write("-" * 80 + "\n")
            
            sorted_audits = sorted(self.audits, key=lambda a: a.overall_quality)[:10]
            for i, audit in enumerate(sorted_audits, 1):
                f.write(f"\n{i}. Bâtiment {audit.building_id} (Score: {audit.overall_quality:.1f}/100)\n")
                f.write(f"   Couverture: {audit.overall_coverage:.1%}\n")
                f.write(f"   Buffer recommandé: [{audit.recommended_buffer_min:.1f}, {audit.recommended_buffer_max:.1f}]m\n")
                
                if audit.critical_issues:
                    f.write("   ❌ Problèmes critiques:\n")
                    for issue in audit.critical_issues:
                        f.write(f"      - {issue}\n")
                
                if audit.recommendations:
                    f.write("   💡 Recommandations:\n")
                    for rec in audit.recommendations[:3]:  # Top 3
                        f.write(f"      - {rec}\n")
            
            # Global recommendations
            f.write("\n\n📋 RECOMMANDATIONS GLOBALES\n")
            f.write("-" * 80 + "\n")
            
            all_buffers = [buf for audit in self.audits for buf in audit.recommended_buffer_adaptive.values()]
            if all_buffers:
                median_buffer = np.median(all_buffers)
                p75_buffer = np.percentile(all_buffers, 75)
                
                f.write(f"1. Buffer médian recommandé: {median_buffer:.1f}m\n")
                f.write(f"2. Buffer 75e percentile: {p75_buffer:.1f}m\n")
                f.write(f"3. Pour {n_critical} bâtiments, augmenter les buffers significativement\n")
                f.write(f"4. Activer le buffering adaptatif par façade pour meilleure couverture\n")
        
        logger.info(f"  ✅ Text recommendations: {txt_path}")
    
    def _generate_visualizations(self):
        """Générer des visualisations (top 5 worst buildings)."""
        logger.info("  Generating visualizations (top 5 worst buildings)...")
        
        # Select 5 worst buildings
        sorted_audits = sorted(self.audits, key=lambda a: a.overall_quality)[:5]
        
        for audit in sorted_audits:
            self._visualize_building(audit)
        
        logger.info(f"  ✅ Generated {len(sorted_audits)} visualizations")
    
    def _visualize_building(self, audit: BuildingAudit):
        """Visualiser un bâtiment avec ses façades et points."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(
            f"Building {audit.building_id} - Quality: {audit.overall_quality:.1f}/100",
            fontsize=16,
            fontweight='bold'
        )
        
        # Extract building points for visualization
        building_points, _ = self._extract_building_points(audit.polygon, buffer=10.0)
        
        # Plot 1: Overview avec polygon et points
        ax = axes[0, 0]
        ax.set_title("Overview - Polygon & Points")
        ax.set_aspect('equal')
        
        # Plot polygon
        x, y = audit.polygon.exterior.xy
        ax.plot(x, y, 'b-', linewidth=2, label='BD TOPO Polygon')
        
        # Plot points (colored by verticality)
        if len(building_points) > 0:
            building_mask = np.isin(self.points, building_points).all(axis=1)
            vert = self.verticality[building_mask]
            scatter = ax.scatter(
                building_points[:, 0],
                building_points[:, 1],
                c=vert,
                cmap='RdYlGn',
                s=1,
                alpha=0.6
            )
            plt.colorbar(scatter, ax=ax, label='Verticality')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Facades analysis
        ax = axes[0, 1]
        ax.set_title("Facades Coverage & Quality")
        
        orientations = [FacadeOrientation.NORTH, FacadeOrientation.SOUTH,
                       FacadeOrientation.EAST, FacadeOrientation.WEST]
        orientation_labels = ['North', 'South', 'East', 'West']
        
        coverages = [audit.facades[o].coverage_ratio * 100 if o in audit.facades else 0
                    for o in orientations]
        qualities = [audit.facades[o].quality_score if o in audit.facades else 0
                    for o in orientations]
        
        x_pos = np.arange(len(orientations))
        width = 0.35
        
        ax.bar(x_pos - width/2, coverages, width, label='Coverage (%)', color='steelblue')
        ax.bar(x_pos + width/2, qualities, width, label='Quality (0-100)', color='coral')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(orientation_labels)
        ax.set_ylabel('Score')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Buffer recommendations
        ax = axes[1, 0]
        ax.set_title("Recommended Buffers by Facade")
        
        buffers = [audit.facades[o].recommended_buffer if o in audit.facades else 0
                  for o in orientations]
        
        colors_buffer = ['red' if b > 5 else 'orange' if b > 3 else 'green' for b in buffers]
        ax.bar(x_pos, buffers, color=colors_buffer, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(orientation_labels)
        ax.set_ylabel('Buffer (m)')
        ax.axhline(y=self.buffer_min, color='g', linestyle='--', alpha=0.5, label='Min')
        ax.axhline(y=self.buffer_max, color='r', linestyle='--', alpha=0.5, label='Max')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Text summary
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = f"""
BUILDING SUMMARY
{'=' * 40}

Total Points: {audit.total_points:,}
Height: {audit.building_height:.1f} m
Perimeter: {audit.perimeter:.1f} m
Area: {audit.area:.1f} m²

Overall Quality: {audit.overall_quality:.1f}/100
Overall Coverage: {audit.overall_coverage:.1%}

Buffer Range: [{audit.recommended_buffer_min:.1f}, {audit.recommended_buffer_max:.1f}] m

CRITICAL ISSUES:
"""
        
        if audit.critical_issues:
            for issue in audit.critical_issues:
                summary_text += f"  ❌ {issue}\n"
        else:
            summary_text += "  ✅ None\n"
        
        summary_text += "\nRECOMMENDATIONS:\n"
        if audit.recommendations:
            for rec in audit.recommendations[:5]:
                summary_text += f"  💡 {rec}\n"
        else:
            summary_text += "  ✅ None needed\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Save figure
        fig_path = self.output_dir / f"building_{audit.building_id}_audit.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"  Saved visualization: {fig_path}")


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Audit de détection de bâtiments avec analyse par façade"
    )
    parser.add_argument(
        "--laz_file",
        type=str,
        required=True,
        help="Chemin vers le fichier LAZ"
    )
    parser.add_argument(
        "--bd_topo_file",
        type=str,
        required=True,
        help="Chemin vers les bâtiments BD TOPO (GeoJSON/SHP)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Répertoire de sortie pour les résultats"
    )
    parser.add_argument(
        "--max_buildings",
        type=int,
        default=None,
        help="Limiter à N bâtiments (pour tests)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Générer des visualisations"
    )
    parser.add_argument(
        "--facade_search_distance",
        type=float,
        default=3.0,
        help="Distance de recherche autour de chaque façade (m)"
    )
    parser.add_argument(
        "--verticality_threshold",
        type=float,
        default=0.55,
        help="Seuil de verticalité pour murs"
    )
    
    args = parser.parse_args()
    
    # Create auditor
    auditor = BuildingFacadeAuditor(
        laz_file=args.laz_file,
        bd_topo_file=args.bd_topo_file,
        output_dir=args.output_dir,
        facade_search_distance=args.facade_search_distance,
        verticality_threshold=args.verticality_threshold,
    )
    
    # Run audit
    auditor.run(
        max_buildings=args.max_buildings,
        visualize=args.visualize
    )
    
    logger.info("\n✅ Audit terminé avec succès!")


if __name__ == "__main__":
    main()
