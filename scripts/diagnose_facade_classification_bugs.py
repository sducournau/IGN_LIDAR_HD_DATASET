"""
🐛 Diagnostic des Bugs de Classification de Façades de Bâtiments
================================================================

Script pour diagnostiquer et identifier les artefacts et bugs dans
la classification des façades de bâtiments.

Problèmes à détecter:
1. Features avec artefacts (NaN, Inf, valeurs aberrantes)
2. Utilisation de features après modification
3. Points mal classifiés (sol → bâtiment, etc.)
4. Arêtes/bordures avec courbure artificielle
5. Incohérences dans les masques de filtrage

Usage:
    python scripts/diagnose_facade_classification_bugs.py \
        --laz_file /path/to/tile.laz \
        --bd_topo_file /path/to/buildings.geojson \
        --output_dir /path/to/diagnostic_results

Author: Building Classification Diagnostic v1.0
Date: October 26, 2025
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json

import numpy as np
import pandas as pd

try:
    from shapely.geometry import Polygon
    import geopandas as gpd

    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False

try:
    import laspy

    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class FeatureArtifactReport:
    """Rapport d'artefacts pour les features."""

    feature_name: str

    # Statistics
    n_nan: int = 0
    n_inf: int = 0
    n_outliers: int = 0  # Beyond 5 sigma

    # Value ranges
    min_value: float = 0.0
    max_value: float = 0.0
    mean_value: float = 0.0
    std_value: float = 0.0

    # Spatial artifacts
    n_border_artifacts: int = 0  # High values at borders
    n_isolated_spikes: int = 0  # Single-point anomalies

    # Status
    has_artifacts: bool = False
    severity: str = "OK"  # OK, WARNING, CRITICAL

    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ClassificationBugReport:
    """Rapport de bugs de classification."""

    building_id: int

    # Misclassification counts
    n_ground_as_building: int = 0
    n_roof_as_wall: int = 0
    n_wall_as_roof: int = 0
    n_edge_artifacts: int = 0

    # Feature issues
    feature_artifacts: Dict[str, FeatureArtifactReport] = field(default_factory=dict)

    # Mask inconsistencies
    mask_inconsistencies: List[str] = field(default_factory=list)

    # Overall assessment
    has_critical_bugs: bool = False
    bug_severity_score: float = 0.0  # 0-100

    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class FacadeClassificationDiagnostic:
    """
    Diagnostic des bugs de classification de façades.

    Vérifie:
    1. Qualité des features (NaN, Inf, outliers)
    2. Artefacts spatiaux (bordures, pics isolés)
    3. Cohérence des classifications
    4. Utilisation correcte des masques
    """

    def __init__(
        self,
        laz_file: Path,
        bd_topo_file: Path,
        output_dir: Path,
        # Thresholds
        outlier_sigma: float = 5.0,
        border_distance: float = 2.0,
        spike_threshold: float = 3.0,
    ):
        """Initialiser le diagnostic."""
        if not HAS_LASPY:
            raise ImportError("laspy required. Install: pip install laspy")
        if not HAS_SPATIAL:
            raise ImportError("shapely, geopandas required")

        self.laz_file = Path(laz_file)
        self.bd_topo_file = Path(bd_topo_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.outlier_sigma = outlier_sigma
        self.border_distance = border_distance
        self.spike_threshold = spike_threshold

        # Data
        self.points = None
        self.classification = None
        self.features = {}
        self.buildings_gdf = None
        self.reports: List[ClassificationBugReport] = []

        logger.info(f"FacadeClassificationDiagnostic initialized")
        logger.info(f"  LAZ: {self.laz_file}")
        logger.info(f"  BD TOPO: {self.bd_topo_file}")

    def run(self, max_buildings: Optional[int] = None):
        """Exécuter le diagnostic complet."""
        logger.info("=" * 80)
        logger.info("🐛 DIAGNOSTIC DES BUGS DE CLASSIFICATION DE FAÇADES")
        logger.info("=" * 80)

        # 1. Load data
        logger.info("\n[1/5] Chargement des données...")
        self._load_data()

        # 2. Compute and validate features
        logger.info("\n[2/5] Calcul et validation des features...")
        self._compute_features()
        self._validate_features()

        # 3. Analyze buildings
        logger.info(f"\n[3/5] Analyse de {len(self.buildings_gdf)} bâtiments...")
        buildings_to_check = (
            self.buildings_gdf.head(max_buildings)
            if max_buildings
            else self.buildings_gdf
        )

        for idx, row in buildings_to_check.iterrows():
            report = self._diagnose_building(idx, row["geometry"])
            self.reports.append(report)

            if (idx + 1) % 10 == 0:
                logger.info(
                    f"  Diagnosed {idx + 1}/{len(buildings_to_check)} buildings"
                )

        # 4. Generate reports
        logger.info("\n[4/5] Génération des rapports...")
        self._generate_reports()

        # 5. Summary
        logger.info("\n[5/5] Résumé...")
        self._print_summary()

        logger.info("\n" + "=" * 80)
        logger.info("✅ DIAGNOSTIC TERMINÉ")
        logger.info("=" * 80)
        logger.info(f"📁 Résultats: {self.output_dir}")

    def _load_data(self):
        """Charger LAZ et BD TOPO."""
        logger.info(f"  Loading LAZ: {self.laz_file.name}")
        with laspy.open(self.laz_file) as laz:
            las = laz.read()
            self.points = np.vstack([las.x, las.y, las.z]).T
            self.classification = np.array(las.classification)

        logger.info(f"  ✅ {len(self.points):,} points loaded")

        logger.info(f"  Loading BD TOPO: {self.bd_topo_file.name}")
        self.buildings_gdf = gpd.read_file(self.bd_topo_file)
        self.buildings_gdf = self.buildings_gdf[
            self.buildings_gdf.geometry.type == "Polygon"
        ]
        logger.info(f"  ✅ {len(self.buildings_gdf)} buildings loaded")

    def _compute_features(self):
        """Calculer les features géométriques."""
        logger.info("  Computing features...")

        from sklearn.neighbors import NearestNeighbors
        from sklearn.decomposition import PCA

        k = 20
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(self.points)
        distances, indices = nbrs.kneighbors(self.points)

        # Normals
        normals = np.zeros((len(self.points), 3))
        for i in range(len(self.points)):
            neighbors = self.points[indices[i]]
            neighbors_centered = neighbors - neighbors.mean(axis=0)

            pca = PCA(n_components=3)
            pca.fit(neighbors_centered)
            normal = pca.components_[-1]

            if normal[2] < 0:
                normal = -normal

            normals[i] = normal

        self.features["normals"] = normals

        # Verticality
        self.features["verticality"] = 1.0 - np.abs(normals[:, 2])

        # Curvature (simplified - based on normal variation)
        curvature = np.zeros(len(self.points))
        for i in range(len(self.points)):
            neighbor_normals = normals[indices[i]]
            # Curvature = variation in normals
            normal_diffs = neighbor_normals - normals[i]
            curvature[i] = np.linalg.norm(normal_diffs, axis=1).mean()

        self.features["curvature"] = curvature

        # Planarity (eigenvalue ratio)
        planarity = np.zeros(len(self.points))
        for i in range(len(self.points)):
            neighbors = self.points[indices[i]]
            neighbors_centered = neighbors - neighbors.mean(axis=0)

            pca = PCA(n_components=3)
            pca.fit(neighbors_centered)
            eigenvalues = pca.explained_variance_

            # Planarity = (λ2 - λ3) / λ1
            if eigenvalues[0] > 1e-10:
                planarity[i] = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]

        self.features["planarity"] = planarity

        # Heights
        z_ground = np.percentile(self.points[:, 2], 5)
        self.features["height_above_ground"] = self.points[:, 2] - z_ground

        # Is ground (simple heuristic)
        self.features["is_ground"] = (
            (self.features["height_above_ground"] < 0.5)
            & (self.features["verticality"] < 0.3)
        ).astype(int)

        logger.info("  ✅ Features computed")

    def _validate_features(self):
        """Valider la qualité des features."""
        logger.info("  Validating features...")

        global_artifacts = {}

        for feat_name, feat_values in self.features.items():
            if feat_name == "is_ground":
                continue  # Binary feature, skip

            report = self._check_feature_artifacts(feat_name, feat_values)
            global_artifacts[feat_name] = report

            if report.has_artifacts:
                logger.warning(
                    f"  ⚠️ {feat_name}: {report.severity} - {len(report.issues)} issues"
                )

        # Save global artifact report
        self.global_feature_artifacts = global_artifacts

        logger.info("  ✅ Feature validation complete")

    def _check_feature_artifacts(
        self, name: str, values: np.ndarray
    ) -> FeatureArtifactReport:
        """Vérifier les artefacts dans une feature."""
        report = FeatureArtifactReport(feature_name=name)

        # Check for NaN/Inf
        report.n_nan = np.isnan(values).sum()
        report.n_inf = np.isinf(values).sum()

        if report.n_nan > 0:
            report.issues.append(f"{report.n_nan} NaN values detected")
            report.has_artifacts = True

        if report.n_inf > 0:
            report.issues.append(f"{report.n_inf} Inf values detected")
            report.has_artifacts = True

        # Statistics on valid values
        valid_mask = np.isfinite(values)
        if valid_mask.sum() == 0:
            report.severity = "CRITICAL"
            report.issues.append("All values are NaN/Inf!")
            return report

        valid_values = values[valid_mask]
        report.min_value = float(valid_values.min())
        report.max_value = float(valid_values.max())
        report.mean_value = float(valid_values.mean())
        report.std_value = float(valid_values.std())

        # Check for outliers (>5 sigma)
        if report.std_value > 0:
            outlier_mask = (
                np.abs(valid_values - report.mean_value)
                > self.outlier_sigma * report.std_value
            )
            report.n_outliers = outlier_mask.sum()

            if report.n_outliers > len(valid_values) * 0.01:  # >1% outliers
                report.issues.append(f"{report.n_outliers} outliers (>5σ) detected")
                report.has_artifacts = True

        # Check for isolated spikes (single points with extreme values)
        # Compare each point to its neighbors
        if report.n_outliers > 0:
            from sklearn.neighbors import NearestNeighbors

            k = 5
            if len(self.points) >= k:
                nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(
                    self.points
                )
                _, indices = nbrs.kneighbors(self.points)

                spike_count = 0
                for i in range(len(values)):
                    if not valid_mask[i]:
                        continue

                    val = values[i]
                    neighbor_vals = values[indices[i]]
                    neighbor_mean = neighbor_vals[np.isfinite(neighbor_vals)].mean()

                    if (
                        np.abs(val - neighbor_mean)
                        > self.spike_threshold * report.std_value
                    ):
                        spike_count += 1

                report.n_isolated_spikes = spike_count

                if spike_count > 0:
                    report.issues.append(f"{spike_count} isolated spikes detected")
                    report.has_artifacts = True

        # Determine severity
        if report.n_nan > len(values) * 0.1 or report.n_inf > 0:
            report.severity = "CRITICAL"
        elif report.n_outliers > len(values) * 0.05 or report.n_isolated_spikes > 100:
            report.severity = "WARNING"
        elif report.has_artifacts:
            report.severity = "WARNING"
        else:
            report.severity = "OK"

        # Recommendations
        if report.n_nan > 0 or report.n_inf > 0:
            report.recommendations.append(
                "Filter out NaN/Inf values before using this feature"
            )

        if report.n_outliers > 0:
            report.recommendations.append(
                f"Clip values to [{report.mean_value - 3*report.std_value:.2f}, {report.mean_value + 3*report.std_value:.2f}]"
            )

        if report.n_isolated_spikes > 0:
            report.recommendations.append(
                "Apply median filter to remove isolated spikes"
            )

        return report

    def _diagnose_building(
        self, building_id: int, polygon: Polygon
    ) -> ClassificationBugReport:
        """Diagnostiquer un bâtiment individuel."""
        report = ClassificationBugReport(building_id=building_id)

        # Extract building points
        buffered = polygon.buffer(5.0)
        bounds = buffered.bounds
        bbox_mask = (
            (self.points[:, 0] >= bounds[0])
            & (self.points[:, 0] <= bounds[2])
            & (self.points[:, 1] >= bounds[1])
            & (self.points[:, 1] <= bounds[3])
        )

        if bbox_mask.sum() == 0:
            report.issues.append("No points found near building")
            return report

        # Check feature artifacts for this building
        for feat_name, feat_values in self.features.items():
            if feat_name == "is_ground":
                continue

            building_feat = feat_values[bbox_mask]
            feat_report = self._check_feature_artifacts(
                f"{feat_name}_local", building_feat
            )
            report.feature_artifacts[feat_name] = feat_report

            if feat_report.severity == "CRITICAL":
                report.has_critical_bugs = True
                report.issues.append(f"{feat_name} has critical artifacts")

        # Check for misclassifications
        building_class = self.classification[bbox_mask]
        is_ground = self.features["is_ground"][bbox_mask]
        verticality = self.features["verticality"][bbox_mask]
        height = self.features["height_above_ground"][bbox_mask]

        # Ground points classified as building (class 6)
        ground_as_building = (is_ground == 1) & (building_class == 6)
        report.n_ground_as_building = ground_as_building.sum()

        if report.n_ground_as_building > 0:
            report.issues.append(
                f"{report.n_ground_as_building} ground points classified as building"
            )

        # Roof points (low verticality, high elevation) classified as walls
        roof_mask = (verticality < 0.3) & (height > 5.0)
        wall_class_mask = (building_class >= 50) & (
            building_class < 60
        )  # LOD3 wall classes
        roof_as_wall = roof_mask & wall_class_mask
        report.n_roof_as_wall = roof_as_wall.sum()

        if report.n_roof_as_wall > 0:
            report.issues.append(
                f"{report.n_roof_as_wall} roof points classified as walls"
            )

        # Wall points (high verticality) classified as roofs
        wall_mask = verticality > 0.7
        roof_class_mask = (building_class >= 60) & (
            building_class < 70
        )  # LOD3 roof classes
        wall_as_roof = wall_mask & roof_class_mask
        report.n_wall_as_roof = wall_as_roof.sum()

        if report.n_wall_as_roof > 0:
            report.issues.append(
                f"{report.n_wall_as_roof} wall points classified as roofs"
            )

        # Calculate bug severity score
        total_building_points = bbox_mask.sum()
        if total_building_points > 0:
            error_rate = (
                report.n_ground_as_building
                + report.n_roof_as_wall
                + report.n_wall_as_roof
            ) / total_building_points

            report.bug_severity_score = min(100.0, error_rate * 100)

        # Recommendations
        if report.n_ground_as_building > 0:
            report.recommendations.append(
                "Apply ground filtering BEFORE classification"
            )

        if report.n_roof_as_wall > 0 or report.n_wall_as_roof > 0:
            report.recommendations.append(
                "Use verticality threshold to separate roofs from walls"
            )

        for feat_name, feat_report in report.feature_artifacts.items():
            if feat_report.has_artifacts:
                report.recommendations.extend(feat_report.recommendations)

        return report

    def _generate_reports(self):
        """Générer les rapports de diagnostic."""
        # JSON report
        json_report = {
            "metadata": {
                "laz_file": str(self.laz_file),
                "bd_topo_file": str(self.bd_topo_file),
                "n_buildings_analyzed": len(self.reports),
                "n_critical_bugs": sum(1 for r in self.reports if r.has_critical_bugs),
            },
            "global_feature_artifacts": {},
            "buildings": [],
        }

        # Global feature artifacts
        for feat_name, report in self.global_feature_artifacts.items():
            json_report["global_feature_artifacts"][feat_name] = {
                "n_nan": int(report.n_nan),
                "n_inf": int(report.n_inf),
                "n_outliers": int(report.n_outliers),
                "n_isolated_spikes": int(report.n_isolated_spikes),
                "severity": report.severity,
                "issues": report.issues,
                "recommendations": report.recommendations,
            }

        # Per-building reports
        for report in self.reports:
            building_data = {
                "building_id": int(report.building_id),
                "n_ground_as_building": int(report.n_ground_as_building),
                "n_roof_as_wall": int(report.n_roof_as_wall),
                "n_wall_as_roof": int(report.n_wall_as_roof),
                "bug_severity_score": float(report.bug_severity_score),
                "has_critical_bugs": report.has_critical_bugs,
                "issues": report.issues,
                "recommendations": report.recommendations,
            }
            json_report["buildings"].append(building_data)

        json_path = self.output_dir / "diagnostic_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False)

        logger.info(f"  ✅ JSON report: {json_path}")

        # CSV summary
        rows = []
        for report in self.reports:
            rows.append(
                {
                    "building_id": report.building_id,
                    "ground_as_building": report.n_ground_as_building,
                    "roof_as_wall": report.n_roof_as_wall,
                    "wall_as_roof": report.n_wall_as_roof,
                    "severity_score": report.bug_severity_score,
                    "has_critical_bugs": report.has_critical_bugs,
                    "n_issues": len(report.issues),
                }
            )

        df = pd.DataFrame(rows)
        csv_path = self.output_dir / "diagnostic_summary.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"  ✅ CSV summary: {csv_path}")

    def _print_summary(self):
        """Afficher un résumé des résultats."""
        logger.info("\n📊 RÉSUMÉ DU DIAGNOSTIC")
        logger.info("=" * 80)

        # Global feature artifacts
        logger.info("\n🔍 ARTEFACTS GLOBAUX DES FEATURES:")
        for feat_name, report in self.global_feature_artifacts.items():
            if report.has_artifacts:
                logger.warning(f"  {feat_name}: {report.severity}")
                for issue in report.issues:
                    logger.warning(f"    - {issue}")

        # Building-specific bugs
        n_critical = sum(1 for r in self.reports if r.has_critical_bugs)
        n_with_issues = sum(1 for r in self.reports if r.issues)

        logger.info(f"\n🏢 BUGS PAR BÂTIMENT:")
        logger.info(f"  Total analyzed: {len(self.reports)}")
        logger.info(
            f"  With critical bugs: {n_critical} ({100*n_critical/len(self.reports):.1f}%)"
        )
        logger.info(
            f"  With issues: {n_with_issues} ({100*n_with_issues/len(self.reports):.1f}%)"
        )

        # Top issues
        total_ground_as_building = sum(r.n_ground_as_building for r in self.reports)
        total_roof_as_wall = sum(r.n_roof_as_wall for r in self.reports)
        total_wall_as_roof = sum(r.n_wall_as_roof for r in self.reports)

        logger.info(f"\n🐛 ERREURS DE CLASSIFICATION:")
        logger.info(f"  Ground → Building: {total_ground_as_building:,} points")
        logger.info(f"  Roof → Wall: {total_roof_as_wall:,} points")
        logger.info(f"  Wall → Roof: {total_wall_as_roof:,} points")


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Diagnostic des bugs de classification de façades"
    )
    parser.add_argument("--laz_file", type=str, required=True, help="Fichier LAZ")
    parser.add_argument(
        "--bd_topo_file", type=str, required=True, help="Fichier BD TOPO"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Répertoire de sortie"
    )
    parser.add_argument(
        "--max_buildings", type=int, default=None, help="Limiter à N bâtiments"
    )

    args = parser.parse_args()

    diagnostic = FacadeClassificationDiagnostic(
        laz_file=args.laz_file,
        bd_topo_file=args.bd_topo_file,
        output_dir=args.output_dir,
    )

    diagnostic.run(max_buildings=args.max_buildings)

    logger.info("\n✅ Diagnostic terminé!")


if __name__ == "__main__":
    main()
