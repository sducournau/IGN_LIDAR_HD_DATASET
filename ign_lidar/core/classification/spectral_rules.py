"""
Spectral Rules Module for Advanced Material Classification

This module provides advanced spectral classification rules using RGB + NIR data
for material-specific classification beyond simple NDVI thresholds.

Features:
- Direct NIR-based vegetation detection
- Material classification (concrete, asphalt, water) using spectral signatures
- NIR/Red ratio for robust vegetation identification
- Brightness-based surface type discrimination

Author: Classification Optimization Team
Date: October 18, 2025
"""

import logging
from typing import Dict, Tuple, Optional
import numpy as np

from .constants import ASPRSClass

logger = logging.getLogger(__name__)


class SpectralRulesEngine:
    """
    Material classification using multi-band spectral analysis.

    Uses RGB + NIR for advanced classification beyond NDVI:
    - Vegetation: High NDVI + High NIR
    - Concrete: Moderate NIR + Moderate brightness + Low NDVI
    - Asphalt: Low NIR + Low brightness + Low NDVI
    - Water: Negative NDVI + Low NIR + Low brightness
    - Metal roofs: Moderate NIR + High brightness + Low NDVI
    """

    # Use canonical ASPRSClass from classification_schema via wrapper
    # Constants are intentionally NOT duplicated here. Use ASPRSClass.*

    def __init__(
        self,
        nir_vegetation_threshold: float = 0.35,  # 🌿 ABAISSÉ: 0.40 → 0.35 pour végétation moyenne
        nir_building_threshold: float = 0.3,
        brightness_concrete_min: float = 0.4,
        brightness_concrete_max: float = 0.7,
        ndvi_water_threshold: float = -0.1,
        nir_water_threshold: float = 0.2,
        brightness_asphalt_max: float = 0.3,
        nir_asphalt_threshold: float = 0.15,
        brightness_metal_min: float = 0.5,
        brightness_metal_max: float = 0.8,
        nir_red_ratio_veg_threshold: float = 1.8,  # 🌿 ABAISSÉ: 2.0 → 1.8 pour détecter végétation moins dense
    ):
        """
        Initialize spectral rules engine with improved vegetation detection thresholds.

        Args:
            nir_vegetation_threshold: Minimum NIR for vegetation classification (0.35 lowered)
            nir_building_threshold: Minimum NIR for building materials (concrete)
            brightness_concrete_min: Minimum brightness for concrete surfaces
            brightness_concrete_max: Maximum brightness for concrete surfaces
            ndvi_water_threshold: Maximum NDVI for water classification
            nir_water_threshold: Maximum NIR for water classification
            brightness_asphalt_max: Maximum brightness for asphalt surfaces
            nir_asphalt_threshold: Maximum NIR for asphalt surfaces
            brightness_metal_min: Minimum brightness for metal roofs
            brightness_metal_max: Maximum brightness for metal roofs
            nir_red_ratio_veg_threshold: Minimum NIR/Red ratio for vegetation (1.8 lowered)
        """
        self.nir_vegetation_threshold = nir_vegetation_threshold
        self.nir_building_threshold = nir_building_threshold
        self.brightness_concrete_min = brightness_concrete_min
        self.brightness_concrete_max = brightness_concrete_max
        self.ndvi_water_threshold = ndvi_water_threshold
        self.nir_water_threshold = nir_water_threshold
        self.brightness_asphalt_max = brightness_asphalt_max
        self.nir_asphalt_threshold = nir_asphalt_threshold
        self.brightness_metal_min = brightness_metal_min
        self.brightness_metal_max = brightness_metal_max
        self.nir_red_ratio_veg_threshold = nir_red_ratio_veg_threshold

        logger.info("🌈 Spectral Rules Engine initialized (IMPROVED vegetation detection)")
        logger.info(f"   NIR vegetation threshold: {nir_vegetation_threshold} (lowered)")
        logger.info(f"   NIR building threshold: {nir_building_threshold}")
        logger.info(
            f"   Brightness concrete range: [{brightness_concrete_min}, {brightness_concrete_max}]"
        )
        logger.info(
            f"   NIR/Red ratio vegetation threshold: {nir_red_ratio_veg_threshold} (lowered)"
        )

    def classify_by_spectral_signature(
        self,
        rgb: np.ndarray,
        nir: np.ndarray,
        current_labels: np.ndarray,
        ndvi: Optional[np.ndarray] = None,
        apply_to_unclassified_only: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Classify points based on spectral signatures with improved vegetation detection.

        Args:
            rgb: RGB values [N, 3] normalized to [0, 1]
            nir: NIR values [N] normalized to [0, 1]
            current_labels: Current classification labels [N]
            ndvi: Pre-computed NDVI values [N], if None will compute from NIR and Red
            apply_to_unclassified_only: Only reclassify unclassified points

        Returns:
            Tuple of (updated labels, statistics dict)
        """
        if len(rgb) != len(nir) or len(rgb) != len(current_labels):
            raise ValueError("RGB, NIR, and labels must have same length")

        labels = current_labels.copy()
        stats = {
            "vegetation_spectral": 0,
            "vegetation_sparse": 0,
            "water_spectral": 0,
            "building_concrete_spectral": 0,
            "building_metal_spectral": 0,
            "road_asphalt_spectral": 0,
            "total_reclassified": 0,
        }

        # Extract channels
        red = rgb[:, 0]
        green = rgb[:, 1]
        blue = rgb[:, 2]

        # Compute derived features
        if ndvi is None:
            ndvi = (nir - red) / (nir + red + 1e-8)

        brightness = np.mean(rgb, axis=1)
        nir_red_ratio = nir / (red + 1e-8)
        green_dominance = (green > red) & (green > blue)

        # Determine which points to classify
        if apply_to_unclassified_only:
            mask = labels == int(ASPRSClass.UNCLASSIFIED)
        else:
            mask = np.ones(len(labels), dtype=bool)

        initial_mask_count = np.sum(mask)

        # 🌿 AMÉLIORÉ - Règle 1: Végétation dense et moyenne (critères assouplis)
        # Au lieu de 3 conditions strictes, utiliser des alternatives plus permissives
        veg_mask = mask & (
            # Option A: NDVI élevé + NIR élevé (végétation dense)
            ((ndvi > 0.3) & (nir > self.nir_vegetation_threshold))
            |
            # Option B: NDVI modéré + ratio NIR/Red élevé (végétation moyenne)
            ((ndvi > 0.18) & (nir_red_ratio > self.nir_red_ratio_veg_threshold))
            |
            # Option C: NDVI modéré + NIR élevé + vert dominant (végétation claire)
            ((ndvi > 0.12) & (nir > 0.32) & green_dominance)
            |
            # Option D: NDVI faible + NIR très élevé + vert dominant (végétation jeune/claire)
            ((ndvi > 0.08) & (nir > 0.45) & green_dominance & (green > 0.4))
        )
        labels[veg_mask] = int(ASPRSClass.MEDIUM_VEGETATION)
        stats["vegetation_spectral"] = int(np.sum(veg_mask))
        mask[veg_mask] = False  # Remove classified points from mask

        # 🌾 NOUVEAU - Règle 1b: Végétation sparse/faible (herbe, pelouse)
        # NDVI positif mais faible + NIR modéré + brightness élevé
        sparse_veg_mask = mask & (
            (ndvi > 0.05) & 
            (ndvi <= 0.12) &
            (nir > 0.25) & 
            (nir < 0.45) &
            (brightness > 0.35) &
            green_dominance
        )
        labels[sparse_veg_mask] = int(ASPRSClass.LOW_VEGETATION)
        stats["vegetation_sparse"] = int(np.sum(sparse_veg_mask))
        mask[sparse_veg_mask] = False

        # Rule 2: Negative NDVI + Low NIR + Low Brightness = Water
        # Water absorbs NIR strongly and has low overall brightness
        water_mask = (
            mask
            & (ndvi < self.ndvi_water_threshold)
            & (nir < self.nir_water_threshold)
            & (brightness < 0.3)
        )
        labels[water_mask] = int(ASPRSClass.WATER)
        stats["water_spectral"] = int(np.sum(water_mask))
        mask[water_mask] = False

        # Rule 3: Moderate NIR + Moderate brightness + Low NDVI = Concrete buildings
        # Concrete reflects some NIR but not as much as vegetation
        concrete_mask = (
            mask
            & (nir >= self.nir_building_threshold)
            & (nir < self.nir_vegetation_threshold)
            & (brightness >= self.brightness_concrete_min)
            & (brightness <= self.brightness_concrete_max)
            & (ndvi < 0.15)
        )
        labels[concrete_mask] = int(ASPRSClass.BUILDING)
        stats["building_concrete_spectral"] = int(np.sum(concrete_mask))
        mask[concrete_mask] = False

        # Rule 4: Moderate NIR + High brightness + Low NDVI = Metal roofs
        # Metal roofs are highly reflective across all bands
        metal_mask = (
            mask
            & (nir >= 0.2)
            & (nir <= 0.5)
            & (brightness >= self.brightness_metal_min)
            & (brightness <= self.brightness_metal_max)
            & (ndvi < 0.2)
        )
        labels[metal_mask] = int(ASPRSClass.BUILDING)
        stats["building_metal_spectral"] = int(np.sum(metal_mask))
        mask[metal_mask] = False

        # Rule 5: Very low NIR + Low brightness + Low NDVI = Asphalt
        # Asphalt is dark and absorbs most wavelengths
        asphalt_mask = (
            mask
            & (nir < self.nir_asphalt_threshold)
            & (brightness < self.brightness_asphalt_max)
            & (ndvi < 0.1)
        )
        labels[asphalt_mask] = int(ASPRSClass.ROAD_SURFACE)
        stats["road_asphalt_spectral"] = int(np.sum(asphalt_mask))
        mask[asphalt_mask] = False

        # Calculate total reclassified
        stats["total_reclassified"] = initial_mask_count - np.sum(mask)

        # Log results
        if stats["total_reclassified"] > 0:
            logger.info(
                f"  🌈 Spectral rules classified {stats['total_reclassified']:,} points:"
            )
            if stats["vegetation_spectral"] > 0:
                logger.info(
                    f"     Vegetation/Medium (spectral): {stats['vegetation_spectral']:,}"
                )
            if stats["vegetation_sparse"] > 0:
                logger.info(
                    f"     Vegetation/Low-Sparse (spectral): {stats['vegetation_sparse']:,}"
                )
            if stats["water_spectral"] > 0:
                logger.info(f"     Water (spectral): {stats['water_spectral']:,}")
            if stats["building_concrete_spectral"] > 0:
                logger.info(
                    f"     Building/Concrete (spectral): {stats['building_concrete_spectral']:,}"
                )
            if stats["building_metal_spectral"] > 0:
                logger.info(
                    f"     Building/Metal (spectral): {stats['building_metal_spectral']:,}"
                )
            if stats["road_asphalt_spectral"] > 0:
                logger.info(
                    f"     Road/Asphalt (spectral): {stats['road_asphalt_spectral']:,}"
                )

        return labels, stats

    def classify_unclassified_relaxed(
        self,
        rgb: np.ndarray,
        nir: np.ndarray,
        current_labels: np.ndarray,
        ndvi: Optional[np.ndarray] = None,
        verticality: Optional[np.ndarray] = None,
        heights: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Classification finale avec critères assouplis pour points non classifiés.

        Cette méthode applique des règles plus permissives pour classifier les
        points restants non classifiés, en utilisant des combinaisons de features
        spectrales et géométriques.

        Args:
            rgb: RGB values [N, 3] normalized to [0, 1]
            nir: NIR values [N] normalized to [0, 1]
            current_labels: Current classification labels [N]
            ndvi: Pre-computed NDVI values [N]
            verticality: Verticality values [N] (0=horizontal, 1=vertical)
            heights: Heights above ground [N]

        Returns:
            Tuple of (updated labels, statistics dict)
        """
        labels = current_labels.copy()
        stats = {
            "vegetation_relaxed": 0,
            "building_vertical_relaxed": 0,
            "building_elevated_relaxed": 0,
            "total_relaxed": 0,
        }

        # Ne traiter que les points non classifiés
        unclassified_mask = labels == self.ASPRS_UNCLASSIFIED
        n_unclassified_initial = np.sum(unclassified_mask)

        if n_unclassified_initial == 0:
            return labels, stats

        logger.info(
            f"🔍 Applying relaxed classification rules to {n_unclassified_initial:,} unclassified points"
        )

        # Extract channels
        red = rgb[:, 0]
        green = rgb[:, 1]
        blue = rgb[:, 2]

        # Compute derived features
        if ndvi is None:
            ndvi = (nir - red) / (nir + red + 1e-8)

        brightness = np.mean(rgb, axis=1)
        nir_red_ratio = nir / (red + 1e-8)

        # Règle 1: Végétation avec critères assouplis
        # ✅ NDVI > 0.15 (au lieu de 0.3) OU NIR élevé + ratio NIR/Red élevé
        veg_mask_relaxed = unclassified_mask & (
            # Option A: NDVI modéré seul
            (ndvi > 0.15)
            |
            # Option B: NIR élevé + ratio favorable (sans exiger NDVI élevé)
            ((nir > 0.35) & (nir_red_ratio > 1.5) & (ndvi > 0.0))  # Juste positif
            |
            # Option C: Signature verte forte
            ((green > red) & (green > blue) & (ndvi > 0.1) & (nir > 0.3))
        )
        labels[veg_mask_relaxed] = self.ASPRS_MEDIUM_VEGETATION
        stats["vegetation_relaxed"] = np.sum(veg_mask_relaxed)
        unclassified_mask[veg_mask_relaxed] = False

        # Règle 2: Bâtiments avec critères géométriques (verticalité)
        # ✅ Points verticaux au-dessus du sol = façades/murs
        if verticality is not None and heights is not None:
            building_vertical_mask = (
                unclassified_mask
                & (verticality > 0.65)  # Assez vertical
                & (heights > 0.5)  # Au-dessus du sol
                & (ndvi < 0.25)  # Pas de végétation
            )
            labels[building_vertical_mask] = self.ASPRS_BUILDING
            stats["building_vertical_relaxed"] = np.sum(building_vertical_mask)
            unclassified_mask[building_vertical_mask] = False

        # Règle 3: Bâtiments élevés avec signature spectrale bâtiment
        # ✅ Points élevés + signature matériau construction
        if heights is not None:
            building_elevated_mask = (
                unclassified_mask
                & (heights > 2.0)  # Nettement au-dessus du sol
                & (ndvi < 0.2)  # Pas de végétation
                & (
                    # Signature béton/ciment
                    (
                        (brightness >= 0.35)
                        & (brightness <= 0.75)
                        & (nir > 0.25)
                        & (nir < 0.45)
                    )
                    |
                    # Signature tuile/ardoise (sombre)
                    ((brightness < 0.35) & (nir > 0.2) & (nir < 0.4))
                    |
                    # Signature métal (très réfléchissant)
                    ((brightness > 0.6) & (nir > 0.3))
                )
            )
            labels[building_elevated_mask] = self.ASPRS_BUILDING
            stats["building_elevated_relaxed"] = np.sum(building_elevated_mask)
            unclassified_mask[building_elevated_mask] = False

        # Calcul total
        stats["total_relaxed"] = n_unclassified_initial - np.sum(unclassified_mask)

        # Log results
        if stats["total_relaxed"] > 0:
            logger.info(
                f"  ✅ Relaxed rules classified {stats['total_relaxed']:,} additional points:"
            )
            if stats["vegetation_relaxed"] > 0:
                logger.info(
                    f"     Vegetation (relaxed NDVI): {stats['vegetation_relaxed']:,}"
                )
            if stats["building_vertical_relaxed"] > 0:
                logger.info(
                    f"     Building (vertical facades): {stats['building_vertical_relaxed']:,}"
                )
            if stats["building_elevated_relaxed"] > 0:
                logger.info(
                    f"     Building (elevated + material): {stats['building_elevated_relaxed']:,}"
                )

            remaining = np.sum(labels == self.ASPRS_UNCLASSIFIED)
            logger.info(
                f"     Remaining unclassified: {remaining:,} ({remaining/len(labels)*100:.1f}%)"
            )

        return labels, stats

    def get_spectral_features(
        self, rgb: np.ndarray, nir: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Extract spectral features for further analysis or classification.

        Args:
            rgb: RGB values [N, 3] normalized to [0, 1]
            nir: NIR values [N] normalized to [0, 1]

        Returns:
            Dictionary of spectral features
        """
        red = rgb[:, 0]
        green = rgb[:, 1]
        blue = rgb[:, 2]

        # Compute NDVI
        ndvi = (nir - red) / (nir + red + 1e-8)

        # Compute brightness (mean of RGB)
        brightness = np.mean(rgb, axis=1)

        # Compute NIR/Red ratio
        nir_red_ratio = nir / (red + 1e-8)

        # Compute NIR/Green ratio (useful for water detection)
        nir_green_ratio = nir / (green + 1e-8)

        # Compute RGB ratios
        red_green_ratio = red / (green + 1e-8)
        blue_green_ratio = blue / (green + 1e-8)

        # Compute saturation (useful for distinguishing natural vs. artificial)
        max_rgb = np.maximum(np.maximum(red, green), blue)
        min_rgb = np.minimum(np.minimum(red, green), blue)
        saturation = (max_rgb - min_rgb) / (max_rgb + 1e-8)

        return {
            "ndvi": ndvi,
            "brightness": brightness,
            "nir_red_ratio": nir_red_ratio,
            "nir_green_ratio": nir_green_ratio,
            "red_green_ratio": red_green_ratio,
            "blue_green_ratio": blue_green_ratio,
            "saturation": saturation,
            "red": red,
            "green": green,
            "blue": blue,
            "nir": nir,
        }

    def classify_with_confidence(
        self,
        rgb: np.ndarray,
        nir: np.ndarray,
        current_labels: np.ndarray,
        ndvi: Optional[np.ndarray] = None,
        confidence_threshold: float = 0.8,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """
        Classify with confidence scores, only applying high-confidence classifications.

        Args:
            rgb: RGB values [N, 3] normalized to [0, 1]
            nir: NIR values [N] normalized to [0, 1]
            current_labels: Current classification labels [N]
            ndvi: Pre-computed NDVI values [N]
            confidence_threshold: Minimum confidence (0-1) for classification

        Returns:
            Tuple of (updated labels, confidence scores, statistics dict)
        """
        # Get spectral features
        features = self.get_spectral_features(rgb, nir)
        if ndvi is None:
            ndvi = features["ndvi"]

        labels = current_labels.copy()
        confidence = np.zeros(len(labels))
        stats = {}

        # Compute confidence for each rule
        # Vegetation confidence: Based on how far NDVI and NIR exceed thresholds
        veg_confidence = np.clip(
            (ndvi - 0.3) / 0.3
            + (features["nir"] - self.nir_vegetation_threshold) / 0.3,
            0,
            1,
        )

        # Water confidence: Based on how negative NDVI is and how low NIR is
        water_confidence = np.clip(
            -(ndvi - self.ndvi_water_threshold) / 0.2
            + (self.nir_water_threshold - features["nir"]) / 0.2,
            0,
            1,
        )

        # Apply classifications with high confidence
        unclassified_mask = labels == self.ASPRS_UNCLASSIFIED

        high_conf_veg = unclassified_mask & (veg_confidence > confidence_threshold)
        labels[high_conf_veg] = self.ASPRS_MEDIUM_VEGETATION
        confidence[high_conf_veg] = veg_confidence[high_conf_veg]
        stats["high_confidence_vegetation"] = np.sum(high_conf_veg)

        high_conf_water = unclassified_mask & (water_confidence > confidence_threshold)
        labels[high_conf_water] = self.ASPRS_WATER
        confidence[high_conf_water] = water_confidence[high_conf_water]
        stats["high_confidence_water"] = np.sum(high_conf_water)

        return labels, confidence, stats


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    n_points = 10000

    # Simulate vegetation (high NDVI, high NIR)
    n_veg = 3000
    rgb_veg = np.random.rand(n_veg, 3) * 0.3 + np.array([0.2, 0.4, 0.2])
    nir_veg = np.random.rand(n_veg) * 0.3 + 0.5

    # Simulate concrete (moderate NIR, moderate brightness)
    n_concrete = 2000
    rgb_concrete = np.random.rand(n_concrete, 3) * 0.2 + 0.5
    nir_concrete = np.random.rand(n_concrete) * 0.1 + 0.25

    # Simulate asphalt (low NIR, low brightness)
    n_asphalt = 2000
    rgb_asphalt = np.random.rand(n_asphalt, 3) * 0.2 + 0.1
    nir_asphalt = np.random.rand(n_asphalt) * 0.1 + 0.05

    # Simulate water (very low NIR, low brightness)
    n_water = 1000
    rgb_water = np.random.rand(n_water, 3) * 0.15 + 0.05
    nir_water = np.random.rand(n_water) * 0.1 + 0.02

    # Combine
    rgb = np.vstack([rgb_veg, rgb_concrete, rgb_asphalt, rgb_water])
    rgb = np.clip(rgb, 0, 1)
    nir = np.concatenate([nir_veg, nir_concrete, nir_asphalt, nir_water])
    nir = np.clip(nir, 0, 1)

    # Shuffle
    indices = np.random.permutation(len(rgb))
    rgb = rgb[indices]
    nir = nir[indices]

    # Create unclassified labels
    labels = np.ones(len(rgb), dtype=np.int32)

    # Test spectral rules
    engine = SpectralRulesEngine()
    new_labels, stats = engine.classify_by_spectral_signature(
        rgb=rgb, nir=nir, current_labels=labels, apply_to_unclassified_only=True
    )

    print("\n=== Spectral Classification Results ===")
    print(f"Total points: {len(labels):,}")
    print(f"Classified: {stats['total_reclassified']:,}")
    print(f"Vegetation: {stats['vegetation_spectral']:,}")
    print(f"Water: {stats['water_spectral']:,}")
    print(f"Concrete: {stats['building_concrete_spectral']:,}")
    print(f"Asphalt: {stats['road_asphalt_spectral']:,}")
