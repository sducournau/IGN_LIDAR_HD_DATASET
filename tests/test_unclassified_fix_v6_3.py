"""
Tests pour la classification relaxée des points non classifiés (v6.3)

Ce module teste les nouvelles fonctionnalités de classification avec
critères assouplis pour réduire le nombre de points non classifiés.
"""

import numpy as np
import pytest

from ign_lidar.core.classification.spectral_rules import SpectralRulesEngine


class TestRelaxedClassification:
    """Tests pour la méthode classify_unclassified_relaxed()"""

    @pytest.fixture
    def engine(self):
        """Créer un moteur de règles spectrales"""
        return SpectralRulesEngine()

    @pytest.fixture
    def sample_data(self):
        """Créer des données de test synthétiques"""
        n_points = 1000

        # Créer points avec différentes signatures
        rgb = np.random.rand(n_points, 3).astype(np.float32)
        nir = np.random.rand(n_points).astype(np.float32)
        labels = np.ones(n_points, dtype=np.int32)  # Tous non classifiés

        # Créer features géométriques
        verticality = np.random.rand(n_points).astype(np.float32)
        heights = np.random.rand(n_points) * 10.0  # 0-10m

        # Calculer NDVI
        red = rgb[:, 0]
        ndvi = (nir - red) / (nir + red + 1e-8)

        return {
            "rgb": rgb,
            "nir": nir,
            "labels": labels,
            "ndvi": ndvi,
            "verticality": verticality,
            "heights": heights,
        }

    def test_vegetation_relaxed_ndvi_moderate(self, engine, sample_data):
        """Tester détection végétation avec NDVI modéré (0.15-0.25)"""
        # Créer signature végétation NDVI modéré
        n_veg = 100
        sample_data["ndvi"][:n_veg] = np.random.uniform(0.15, 0.25, n_veg)
        sample_data["nir"][:n_veg] = 0.4

        # Appliquer classification relaxée
        labels, stats = engine.classify_unclassified_relaxed(
            rgb=sample_data["rgb"],
            nir=sample_data["nir"],
            current_labels=sample_data["labels"],
            ndvi=sample_data["ndvi"],
            verticality=sample_data["verticality"],
            heights=sample_data["heights"],
        )

        # Vérifier que végétation détectée
        n_veg_classified = stats.get("vegetation_relaxed", 0)
        assert n_veg_classified > 0, "Aucune végétation détectée avec NDVI modéré"
        assert (
            n_veg_classified >= n_veg * 0.7
        ), f"Seulement {n_veg_classified}/{n_veg} végétation détectée"

    def test_vegetation_relaxed_nir_ratio(self, engine, sample_data):
        """Tester détection végétation avec ratio NIR/Red élevé"""
        # Créer signature végétation avec NIR élevé
        n_veg = 100
        sample_data["nir"][:n_veg] = 0.5
        sample_data["rgb"][:n_veg, 0] = 0.2  # Red faible
        sample_data["ndvi"][:n_veg] = 0.1  # NDVI faible mais positif

        # Appliquer classification relaxée
        labels, stats = engine.classify_unclassified_relaxed(
            rgb=sample_data["rgb"],
            nir=sample_data["nir"],
            current_labels=sample_data["labels"],
            ndvi=sample_data["ndvi"],
            verticality=sample_data["verticality"],
            heights=sample_data["heights"],
        )

        # Vérifier que végétation détectée via ratio NIR/Red
        n_veg_classified = stats.get("vegetation_relaxed", 0)
        assert n_veg_classified > 0, "Aucune végétation détectée avec ratio NIR/Red"

    def test_vegetation_relaxed_green_signature(self, engine, sample_data):
        """Tester détection végétation avec signature verte forte"""
        # Créer signature végétation verte
        n_veg = 100
        sample_data["rgb"][:n_veg, 0] = 0.2  # Red
        sample_data["rgb"][:n_veg, 1] = 0.5  # Green dominant
        sample_data["rgb"][:n_veg, 2] = 0.2  # Blue
        sample_data["nir"][:n_veg] = 0.4
        sample_data["ndvi"][:n_veg] = 0.15

        # Appliquer classification relaxée
        labels, stats = engine.classify_unclassified_relaxed(
            rgb=sample_data["rgb"],
            nir=sample_data["nir"],
            current_labels=sample_data["labels"],
            ndvi=sample_data["ndvi"],
            verticality=sample_data["verticality"],
            heights=sample_data["heights"],
        )

        # Vérifier que végétation détectée via signature verte
        n_veg_classified = stats.get("vegetation_relaxed", 0)
        assert n_veg_classified > 0, "Aucune végétation détectée avec signature verte"

    def test_building_vertical_facades(self, engine, sample_data):
        """Tester détection façades verticales comme bâtiments"""
        # Créer signature façade verticale
        n_facades = 100
        sample_data["verticality"][:n_facades] = np.random.uniform(0.7, 0.95, n_facades)
        sample_data["heights"][:n_facades] = np.random.uniform(1.0, 8.0, n_facades)
        sample_data["ndvi"][:n_facades] = 0.1  # Pas de végétation

        # Appliquer classification relaxée
        labels, stats = engine.classify_unclassified_relaxed(
            rgb=sample_data["rgb"],
            nir=sample_data["nir"],
            current_labels=sample_data["labels"],
            ndvi=sample_data["ndvi"],
            verticality=sample_data["verticality"],
            heights=sample_data["heights"],
        )

        # Vérifier que façades détectées
        n_facades_classified = stats.get("building_vertical_relaxed", 0)
        assert n_facades_classified > 0, "Aucune façade verticale détectée"
        assert (
            n_facades_classified >= n_facades * 0.7
        ), f"Seulement {n_facades_classified}/{n_facades} façades détectées"

    def test_building_elevated_concrete(self, engine, sample_data):
        """Tester détection toitures béton élevées"""
        # Créer signature toiture béton
        n_roofs = 100
        sample_data["heights"][:n_roofs] = np.random.uniform(3.0, 10.0, n_roofs)
        sample_data["ndvi"][:n_roofs] = 0.1
        sample_data["nir"][:n_roofs] = 0.35
        brightness = np.mean(sample_data["rgb"][:n_roofs], axis=1)
        # Ajuster RGB pour brightness cible 0.5
        sample_data["rgb"][:n_roofs] *= (0.5 / (brightness + 1e-8))[:, np.newaxis]

        # Appliquer classification relaxée
        labels, stats = engine.classify_unclassified_relaxed(
            rgb=sample_data["rgb"],
            nir=sample_data["nir"],
            current_labels=sample_data["labels"],
            ndvi=sample_data["ndvi"],
            verticality=sample_data["verticality"],
            heights=sample_data["heights"],
        )

        # Vérifier que toitures détectées
        n_roofs_classified = stats.get("building_elevated_relaxed", 0)
        assert n_roofs_classified > 0, "Aucune toiture béton détectée"

    def test_building_elevated_metal(self, engine, sample_data):
        """Tester détection toitures métalliques (très réfléchissantes)"""
        # Créer signature toiture métal
        n_roofs = 100
        sample_data["heights"][:n_roofs] = np.random.uniform(3.0, 10.0, n_roofs)
        sample_data["ndvi"][:n_roofs] = 0.1
        sample_data["nir"][:n_roofs] = 0.5
        sample_data["rgb"][:n_roofs] = 0.7  # Très réfléchissant

        # Appliquer classification relaxée
        labels, stats = engine.classify_unclassified_relaxed(
            rgb=sample_data["rgb"],
            nir=sample_data["nir"],
            current_labels=sample_data["labels"],
            ndvi=sample_data["ndvi"],
            verticality=sample_data["verticality"],
            heights=sample_data["heights"],
        )

        # Vérifier que toitures métalliques détectées
        n_roofs_classified = stats.get("building_elevated_relaxed", 0)
        assert n_roofs_classified > 0, "Aucune toiture métal détectée"

    def test_reduces_unclassified_significantly(self, engine, sample_data):
        """Vérifier que classification relaxée réduit significativement les non classifiés"""
        # Créer mélange de signatures détectables
        n_total = len(sample_data["labels"])

        # 30% végétation NDVI modéré
        n_veg = int(n_total * 0.3)
        sample_data["ndvi"][:n_veg] = np.random.uniform(0.15, 0.3, n_veg)
        sample_data["nir"][:n_veg] = 0.4

        # 25% façades verticales
        n_facades = int(n_total * 0.25)
        idx_facades = slice(n_veg, n_veg + n_facades)
        sample_data["verticality"][idx_facades] = 0.8
        sample_data["heights"][idx_facades] = 4.0
        sample_data["ndvi"][idx_facades] = 0.1

        # 20% toitures élevées
        n_roofs = int(n_total * 0.2)
        idx_roofs = slice(n_veg + n_facades, n_veg + n_facades + n_roofs)
        sample_data["heights"][idx_roofs] = 5.0
        sample_data["ndvi"][idx_roofs] = 0.1
        sample_data["nir"][idx_roofs] = 0.35

        # Appliquer classification relaxée
        labels, stats = engine.classify_unclassified_relaxed(
            rgb=sample_data["rgb"],
            nir=sample_data["nir"],
            current_labels=sample_data["labels"],
            ndvi=sample_data["ndvi"],
            verticality=sample_data["verticality"],
            heights=sample_data["heights"],
        )

        # Vérifier réduction significative
        n_unclassified_after = np.sum(labels == 1)
        ratio_unclassified = n_unclassified_after / n_total

        assert (
            ratio_unclassified < 0.30
        ), f"Trop de points restent non classifiés: {ratio_unclassified*100:.1f}%"

        # Vérifier stats
        total_classified = stats.get("total_relaxed", 0)
        assert (
            total_classified > n_total * 0.5
        ), f"Classification relaxée devrait classifier >50% des points, got {total_classified/n_total*100:.1f}%"

    def test_no_false_positives_on_already_classified(self, engine, sample_data):
        """Vérifier que classification relaxée ne touche pas aux points déjà classifiés"""
        # Classifier une partie des points
        sample_data["labels"][:500] = 6  # Building

        # Appliquer classification relaxée
        labels, stats = engine.classify_unclassified_relaxed(
            rgb=sample_data["rgb"],
            nir=sample_data["nir"],
            current_labels=sample_data["labels"],
            ndvi=sample_data["ndvi"],
            verticality=sample_data["verticality"],
            heights=sample_data["heights"],
        )

        # Vérifier que points déjà classifiés ne sont pas modifiés
        assert np.all(labels[:500] == 6), "Points déjà classifiés ont été modifiés!"

    def test_statistics_completeness(self, engine, sample_data):
        """Vérifier que toutes les statistiques sont retournées"""
        labels, stats = engine.classify_unclassified_relaxed(
            rgb=sample_data["rgb"],
            nir=sample_data["nir"],
            current_labels=sample_data["labels"],
            ndvi=sample_data["ndvi"],
            verticality=sample_data["verticality"],
            heights=sample_data["heights"],
        )

        # Vérifier présence de toutes les stats
        required_stats = [
            "vegetation_relaxed",
            "building_vertical_relaxed",
            "building_elevated_relaxed",
            "total_relaxed",
        ]

        for stat_key in required_stats:
            assert stat_key in stats, f"Statistique manquante: {stat_key}"
            assert isinstance(
                stats[stat_key], (int, np.integer)
            ), f"Statistique {stat_key} n'est pas un entier"


class TestImprovedVegetationClassification:
    """Tests pour les améliorations de classify_by_spectral_signature()"""

    @pytest.fixture
    def engine(self):
        return SpectralRulesEngine()

    def test_vegetation_with_moderate_ndvi(self, engine):
        """Tester que végétation détectée avec NDVI 0.2-0.25 (au lieu de 0.3)"""
        n_points = 200

        # Créer signature végétation NDVI modéré
        rgb = np.random.rand(n_points, 3).astype(np.float32)
        nir = np.ones(n_points, dtype=np.float32) * 0.5
        rgb[:, 0] = 0.25  # Red pour NDVI ~0.23

        labels = np.ones(n_points, dtype=np.int32)
        ndvi = (nir - rgb[:, 0]) / (nir + rgb[:, 0] + 1e-8)

        # Vérifier NDVI dans range 0.2-0.25
        assert np.all((ndvi > 0.2) & (ndvi < 0.26))

        # Appliquer classification spectrale améliorée
        labels, stats = engine.classify_by_spectral_signature(
            rgb=rgb,
            nir=nir,
            current_labels=labels,
            ndvi=ndvi,
            apply_to_unclassified_only=True,
        )

        # Vérifier que végétation détectée
        n_veg = stats.get("vegetation_spectral", 0)
        assert n_veg > 0, "Végétation avec NDVI 0.2-0.25 non détectée"
        assert (
            n_veg >= n_points * 0.7
        ), f"Seulement {n_veg}/{n_points} végétation détectée avec NDVI modéré"

    def test_vegetation_alternatives_work(self, engine):
        """Tester que les 3 alternatives de végétation fonctionnent"""
        n_points = 300
        rgb = np.random.rand(n_points, 3).astype(np.float32)
        nir = np.random.rand(n_points).astype(np.float32)
        labels = np.ones(n_points, dtype=np.int32)

        # Option A: NDVI élevé + NIR élevé
        rgb[:100, 0] = 0.2
        nir[:100] = 0.5

        # Option B: NDVI modéré + ratio NIR/Red élevé
        rgb[100:200, 0] = 0.2
        nir[100:200] = 0.45

        # Option C: NDVI modéré + vert dominant
        rgb[200:300, 0] = 0.25
        rgb[200:300, 1] = 0.6  # Vert dominant
        rgb[200:300, 2] = 0.2
        nir[200:300] = 0.4

        # Calculer NDVI
        ndvi = (nir - rgb[:, 0]) / (nir + rgb[:, 0] + 1e-8)

        # Appliquer classification
        labels, stats = engine.classify_by_spectral_signature(
            rgb=rgb,
            nir=nir,
            current_labels=labels,
            ndvi=ndvi,
            apply_to_unclassified_only=True,
        )

        # Vérifier que les 3 groupes sont détectés
        n_veg = stats.get("vegetation_spectral", 0)
        assert (
            n_veg >= 200
        ), f"Au moins 2 des 3 alternatives devraient fonctionner, got {n_veg}/300"


class TestFacadeConfidenceThreshold:
    """Tests pour la réduction du seuil de confiance des façades"""

    def test_facade_classifier_default_confidence(self):
        """Vérifier que seuil par défaut est bien 0.35"""
        from ign_lidar.core.classification.building.facade_processor import (
            BuildingFacadeClassifier,
        )

        classifier = BuildingFacadeClassifier()
        assert (
            classifier.min_confidence == 0.35
        ), f"Seuil confiance devrait être 0.35, got {classifier.min_confidence}"

    def test_facade_classifier_accepts_custom_confidence(self):
        """Vérifier que seuil personnalisé peut être défini"""
        from ign_lidar.core.classification.building.facade_processor import (
            BuildingFacadeClassifier,
        )

        classifier = BuildingFacadeClassifier(min_confidence=0.25)
        assert classifier.min_confidence == 0.25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
