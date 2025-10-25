"""
Spectral Rules Adapter

This module provides an adapter for the legacy SpectralRulesEngine to work
with the modern rules framework.

The adapter wraps SpectralRulesEngine and provides the standard BaseRule
interface, allowing spectral classification to be used in hierarchical
rule compositions.

Usage:
    from ign_lidar.core.classification.rules.spectral_adapter import SpectralRulesAdapter
    from ign_lidar.core.classification.rules.base import RuleConfig, RuleType, RulePriority

    # Create configuration
    config = RuleConfig(
        rule_id="spectral_vegetation",
        rule_type=RuleType.SPECTRAL,
        target_class=3,  # Low vegetation
        priority=RulePriority.MEDIUM
    )

    # Create adapter with engine parameters
    adapter = SpectralRulesAdapter(
        config=config,
        nir_vegetation_threshold=0.4
    )

    # Use in hierarchical engine
    mask, confidence = adapter.evaluate(points, features, context)

Author: Classification Enhancement Team
Date: October 23, 2025
"""

import logging
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

from ..spectral_rules import SpectralRulesEngine
from .adapters import LegacyEngineAdapter
from .base import RuleConfig, RuleType

logger = logging.getLogger(__name__)


class SpectralRulesAdapter(LegacyEngineAdapter):
    """
    Adapter for SpectralRulesEngine to work with rules framework.

    This adapter wraps the legacy SpectralRulesEngine and converts its
    multi-class classification results to the standard (mask, confidence)
    format expected by the rules framework.

    The adapter:
    - Wraps SpectralRulesEngine with all its configuration parameters
    - Converts multi-class results to single-class masks
    - Provides confidence scores based on spectral signature strength
    - Integrates with hierarchical rule execution

    Attributes:
        engine: SpectralRulesEngine instance

    Example:
        >>> config = RuleConfig(
        ...     rule_id="spectral_veg",
        ...     rule_type=RuleType.SPECTRAL,
        ...     target_class=3  # Low vegetation
        ... )
        >>> adapter = SpectralRulesAdapter(
        ...     config=config,
        ...     nir_vegetation_threshold=0.4
        ... )
        >>> mask, conf = adapter.evaluate(points, features, context)
    """

    def __init__(
        self,
        config: RuleConfig,
        nir_vegetation_threshold: float = 0.4,
        nir_building_threshold: float = 0.3,
        brightness_concrete_min: float = 0.4,
        brightness_concrete_max: float = 0.7,
        ndvi_water_threshold: float = -0.1,
        nir_water_threshold: float = 0.2,
        brightness_asphalt_max: float = 0.3,
        nir_asphalt_threshold: float = 0.15,
        brightness_metal_min: float = 0.5,
        brightness_metal_max: float = 0.8,
        nir_red_ratio_veg_threshold: float = 2.0,
    ):
        """
        Initialize spectral rules adapter.

        Args:
            config: Rule configuration (must have rule_type=SPECTRAL)
            nir_vegetation_threshold: Minimum NIR for vegetation
            nir_building_threshold: Minimum NIR for buildings
            brightness_concrete_min: Min brightness for concrete
            brightness_concrete_max: Max brightness for concrete
            ndvi_water_threshold: Max NDVI for water
            nir_water_threshold: Max NIR for water
            brightness_asphalt_max: Max brightness for asphalt
            nir_asphalt_threshold: Max NIR for asphalt
            brightness_metal_min: Min brightness for metal roofs
            brightness_metal_max: Max brightness for metal roofs
            nir_red_ratio_veg_threshold: Min NIR/Red ratio for vegetation
        """
        # Create SpectralRulesEngine with parameters
        engine = SpectralRulesEngine(
            nir_vegetation_threshold=nir_vegetation_threshold,
            nir_building_threshold=nir_building_threshold,
            brightness_concrete_min=brightness_concrete_min,
            brightness_concrete_max=brightness_concrete_max,
            ndvi_water_threshold=ndvi_water_threshold,
            nir_water_threshold=nir_water_threshold,
            brightness_asphalt_max=brightness_asphalt_max,
            nir_asphalt_threshold=nir_asphalt_threshold,
            brightness_metal_min=brightness_metal_min,
            brightness_metal_max=brightness_metal_max,
            nir_red_ratio_veg_threshold=nir_red_ratio_veg_threshold,
        )

        super().__init__(config, engine)

        logger.info(
            f"Created SpectralRulesAdapter for target class {config.target_class}"
        )

    def get_required_features(self) -> Set[str]:
        """Get required features for spectral classification"""
        return {"rgb", "nir"}

    def get_optional_features(self) -> Set[str]:
        """Get optional features that improve results"""
        return {"ndvi", "brightness"}

    def evaluate(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate spectral rule on points.

        This method calls the legacy SpectralRulesEngine and converts its
        multi-class results to a binary mask for the target class with
        confidence scores.

        Args:
            points: Point cloud array (N, 3+) with XYZ coordinates
            features: Dictionary with 'rgb', 'nir', optionally 'ndvi'
            context: Optional context (e.g., current_labels)

        Returns:
            Tuple of:
                - mask: Boolean array (N,) for target class matches
                - confidence: Float array (N,) with confidence [0, 1]

        Raises:
            ValueError: If required features missing
        """
        # Validate required features
        super().validate_features(features, len(points))

        if "rgb" not in features or "nir" not in features:
            raise ValueError("Spectral adapter requires 'rgb' and 'nir' features")

        # Get features
        rgb = features["rgb"]
        nir = features["nir"]
        ndvi = features.get("ndvi", None)

        # Get current labels from context or create unclassified
        if context and "current_labels" in context:
            current_labels = context["current_labels"].copy()
        else:
            current_labels = np.ones(len(points), dtype=np.int32)  # ASPRS unclassified

        # Call legacy engine
        try:
            updated_labels, stats = self.engine.classify_by_spectral_signature(
                rgb=rgb,
                nir=nir,
                current_labels=current_labels,
                ndvi=ndvi,
                apply_to_unclassified_only=True,
            )

            # ✅ NOUVEAU - Appliquer classification relaxée si beaucoup de points restent non classifiés
            n_unclassified = np.sum(updated_labels == 1)  # ASPRS_UNCLASSIFIED
            if n_unclassified > len(updated_labels) * 0.1:  # Si > 10% non classifié
                logger.debug(
                    f"Applying relaxed classification to {n_unclassified} remaining unclassified points"
                )

                # Obtenir features géométriques si disponibles
                verticality = features.get("verticality", None)
                heights = features.get("height_above_ground", None)

                updated_labels, relaxed_stats = (
                    self.engine.classify_unclassified_relaxed(
                        rgb=rgb,
                        nir=nir,
                        current_labels=updated_labels,
                        ndvi=ndvi,
                        verticality=verticality,
                        heights=heights,
                    )
                )
                stats.update(relaxed_stats)

        except Exception as e:
            logger.error(f"SpectralRulesEngine evaluation failed: {e}")
            # Return empty mask on error
            n_points = len(points)
            return np.zeros(n_points, dtype=bool), np.zeros(n_points, dtype=np.float32)

        # Convert to mask for target class
        mask = updated_labels == self.target_class

        # Compute confidence based on spectral signature strength
        confidence = self._compute_spectral_confidence(
            mask=mask, rgb=rgb, nir=nir, ndvi=ndvi
        )

        # Log statistics
        n_matched = np.sum(mask)
        if n_matched > 0:
            logger.debug(
                f"Spectral rule {self.rule_id} matched {n_matched} points "
                f"(mean confidence: {np.mean(confidence[mask]):.3f})"
            )

        return mask, confidence

    def _compute_spectral_confidence(
        self,
        mask: np.ndarray,
        rgb: np.ndarray,
        nir: np.ndarray,
        ndvi: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Compute confidence scores based on spectral signature strength.

        Confidence is higher when spectral values are strong/unambiguous.

        Args:
            mask: Boolean mask of matched points
            rgb: RGB values [N, 3]
            nir: NIR values [N]
            ndvi: Optional NDVI values [N]

        Returns:
            Confidence scores [0, 1] for all points
        """
        n_points = len(mask)
        confidence = np.zeros(n_points, dtype=np.float32)

        if not np.any(mask):
            return confidence

        # Extract matched points
        rgb_matched = rgb[mask]
        nir_matched = nir[mask]

        # Compute confidence based on target class
        if self.target_class in [3, 4, 5]:  # Vegetation
            # Higher NIR and NDVI = higher confidence
            if ndvi is not None:
                ndvi_matched = ndvi[mask]
                conf_values = np.clip(nir_matched * 0.5 + ndvi_matched * 0.5, 0.5, 0.95)
            else:
                conf_values = np.clip(nir_matched, 0.5, 0.95)

        elif self.target_class == 9:  # Water
            # Low NIR and negative/low NDVI = higher confidence
            if ndvi is not None:
                ndvi_matched = ndvi[mask]
                conf_values = np.clip(
                    (1.0 - nir_matched) * 0.5 + (1.0 - np.abs(ndvi_matched)) * 0.5,
                    0.5,
                    0.95,
                )
            else:
                conf_values = np.clip(1.0 - nir_matched, 0.5, 0.95)

        elif self.target_class == 6:  # Building
            # Moderate NIR with appropriate brightness = higher confidence
            brightness = np.mean(rgb_matched, axis=1)
            nir_score = 1.0 - np.abs(nir_matched - 0.4)  # Closer to 0.4 = better
            brightness_score = np.clip(brightness, 0.3, 0.8)
            conf_values = np.clip(nir_score * 0.4 + brightness_score * 0.6, 0.5, 0.95)

        elif self.target_class == 11:  # Road
            # Low NIR and low brightness = higher confidence
            brightness = np.mean(rgb_matched, axis=1)
            conf_values = np.clip(
                (1.0 - nir_matched) * 0.5 + (1.0 - brightness) * 0.5, 0.5, 0.95
            )

        else:
            # Default confidence for other classes
            conf_values = np.full(np.sum(mask), 0.75, dtype=np.float32)

        confidence[mask] = conf_values

        return confidence


def create_spectral_vegetation_rule(
    config: RuleConfig, nir_threshold: float = 0.4
) -> SpectralRulesAdapter:
    """
    Convenience factory for vegetation classification rule.

    Args:
        config: Rule configuration (should have target_class=3,4, or 5)
        nir_threshold: Minimum NIR value for vegetation

    Returns:
        Configured SpectralRulesAdapter for vegetation
    """
    return SpectralRulesAdapter(config=config, nir_vegetation_threshold=nir_threshold)


def create_spectral_water_rule(
    config: RuleConfig, ndvi_threshold: float = -0.1, nir_threshold: float = 0.2
) -> SpectralRulesAdapter:
    """
    Convenience factory for water classification rule.

    Args:
        config: Rule configuration (should have target_class=9)
        ndvi_threshold: Maximum NDVI for water
        nir_threshold: Maximum NIR for water

    Returns:
        Configured SpectralRulesAdapter for water
    """
    return SpectralRulesAdapter(
        config=config,
        ndvi_water_threshold=ndvi_threshold,
        nir_water_threshold=nir_threshold,
    )
