"""
Patch extraction and augmentation for machine learning datasets.

This module handles extracting fixed-size patches from processed point clouds
and applying data augmentation for training dataset diversity.

Extracted from LiDARProcessor as part of refactoring (v3.4.0).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from omegaconf import DictConfig

from .classification.patch_extractor import (
    AugmentationConfig,
    PatchConfig,
    extract_and_augment_patches,
    format_patch_for_architecture,
)

logger = logging.getLogger(__name__)


class PatchExtractor:
    """
    Extract and augment patches from point clouds for ML training.

    Handles:
    - Patch extraction with configurable size and overlap
    - Data augmentation (rotation, jitter, scaling)
    - Architecture-specific formatting (PointNet++, DGCNN, etc.)
    - Class-based patch filtering
    - Patch validation

    This class was extracted from LiDARProcessor to separate patch
    generation concerns from tile processing.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize patch extractor with configuration.

        Args:
            config: Configuration containing patch extraction parameters
        """
        self.config = config
        self.patch_size = config.processor.patch_size
        self.num_points = config.processor.num_points
        self.architecture = config.processor.architecture
        self.augment = config.processor.get("augment", False)
        self.num_augmentations = config.processor.get("num_augmentations", 3)
        self.patch_overlap = config.processor.get("patch_overlap", 0.1)

        # Create patch configuration
        self.patch_config = PatchConfig(
            patch_size=self.patch_size,
            num_points=self.num_points,
            overlap=self.patch_overlap,
        )

        # Create augmentation configuration if enabled
        self.aug_config = None
        if self.augment:
            self.aug_config = AugmentationConfig(
                rotation_range=config.processor.get("rotation_range", 360),
                jitter_sigma=config.processor.get("jitter_sigma", 0.01),
                jitter_clip=config.processor.get("jitter_clip", 0.05),
                scale_range=config.processor.get("scale_range", (0.8, 1.25)),
                mirror_probability=config.processor.get("mirror_prob", 0.5),
            )

        logger.debug(
            f"Initialized PatchExtractor: "
            f"patch_size={self.patch_size}m, "
            f"num_points={self.num_points}, "
            f"augment={self.augment}"
        )

    def extract_patches(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        classification: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Extract patches from processed point cloud.

        Args:
            points: Point cloud array [N, 3] with XYZ coordinates
            features: Dictionary of computed features
            classification: Classification codes [N]
            metadata: Optional tile metadata

        Returns:
            List of patch dictionaries, each containing:
            - 'points': Patch point cloud [M, 3]
            - 'features': Patch features dict
            - 'classification': Patch classifications [M]
            - 'patch_id': Unique patch identifier

        Note:
            Patches are extracted in a grid pattern with configurable overlap.
            Patches with insufficient points are discarded.
        """
        # Combine points, features, and classification
        combined_data = self._combine_data(points, features, classification)

        # Extract patches using grid-based approach
        raw_patches = extract_and_augment_patches(
            combined_data,
            self.patch_config,
            self.aug_config if self.augment else None,
            num_augmentations=self.num_augmentations if self.augment else 0,
        )

        # Validate and filter patches
        valid_patches = self._validate_patches(raw_patches)

        # Format patches for target architecture
        formatted_patches = self._format_patches(valid_patches)

        logger.info(
            f"Extracted {len(formatted_patches)} patches "
            f"(from {len(raw_patches)} raw patches)"
        )

        return formatted_patches

    def _combine_data(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        classification: np.ndarray,
    ) -> np.ndarray:
        """
        Combine points, features, and classification into single array.

        Args:
            points: XYZ coordinates [N, 3]
            features: Feature dictionary
            classification: Classification codes [N]

        Returns:
            Combined array [N, 3+F+1] where F is number of features
        """
        # Start with XYZ coordinates
        combined = [points]

        # Add features in consistent order
        feature_names = sorted(features.keys())
        for name in feature_names:
            feat = features[name]
            if feat.ndim == 1:
                feat = feat.reshape(-1, 1)
            combined.append(feat)

        # Add classification as last column
        combined.append(classification.reshape(-1, 1))

        return np.hstack(combined)

    def _validate_patches(
        self, patches: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Validate patches and filter out invalid ones.

        Args:
            patches: List of raw patch arrays

        Returns:
            List of valid patches

        Note:
            Patches are considered invalid if:
            - Too few points (< 50% of target)
            - All points have same classification
            - Contains NaN or Inf values
        """
        valid_patches = []
        min_points = int(self.num_points * 0.5)

        for i, patch in enumerate(patches):
            # Check point count
            if len(patch) < min_points:
                logger.debug(
                    f"Patch {i} discarded: too few points "
                    f"({len(patch)} < {min_points})"
                )
                continue

            # Check for NaN/Inf
            if not np.all(np.isfinite(patch)):
                logger.warning(f"Patch {i} discarded: contains NaN/Inf values")
                continue

            # Check classification diversity
            classification = patch[:, -1].astype(int)
            unique_classes = np.unique(classification)
            if len(unique_classes) < 2:
                logger.debug(
                    f"Patch {i} discarded: single class "
                    f"({unique_classes[0]})"
                )
                continue

            valid_patches.append(patch)

        return valid_patches

    def _format_patches(
        self, patches: List[np.ndarray]
    ) -> List[Dict[str, np.ndarray]]:
        """
        Format patches for target ML architecture.

        Args:
            patches: List of valid patch arrays

        Returns:
            List of formatted patch dictionaries

        Note:
            Different architectures require different input formats:
            - PointNet++: [N, 3+F] with separate classification
            - DGCNN: [N, 3+F] with edge features
            - PointCNN: [N, 3+F] with local ordering
        """
        formatted_patches = []

        for i, patch in enumerate(patches):
            # Separate coordinates, features, and classification
            coords = patch[:, :3]
            features = patch[:, 3:-1]
            classification = patch[:, -1].astype(int)

            # Format for architecture
            formatted = format_patch_for_architecture(
                coords=coords,
                features=features,
                classification=classification,
                architecture=self.architecture,
                num_points=self.num_points,
            )

            # Add patch metadata
            formatted["patch_id"] = i
            formatted["num_points_original"] = len(patch)

            formatted_patches.append(formatted)

        return formatted_patches

    def extract_patches_by_class(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        classification: np.ndarray,
        target_classes: List[int],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, List[Dict[str, np.ndarray]]]:
        """
        Extract patches grouped by classification.

        Useful for creating class-balanced datasets or analyzing
        specific object types.

        Args:
            points: Point cloud array [N, 3]
            features: Feature dictionary
            classification: Classification codes [N]
            target_classes: List of classes to extract
            metadata: Optional tile metadata

        Returns:
            Dictionary mapping class code to list of patches

        Example:
            >>> # Extract building and ground patches separately
            >>> patches_by_class = extractor.extract_patches_by_class(
            ...     points, features, classification,
            ...     target_classes=[2, 6]  # Ground=2, Building=6
            ... )
            >>> building_patches = patches_by_class[6]
            >>> ground_patches = patches_by_class[2]
        """
        patches_by_class = {cls: [] for cls in target_classes}

        # Extract all patches
        all_patches = self.extract_patches(
            points, features, classification, metadata
        )

        # Group by dominant class
        for patch in all_patches:
            patch_classification = patch["classification"]
            dominant_class = np.bincount(patch_classification).argmax()

            if dominant_class in target_classes:
                patches_by_class[dominant_class].append(patch)

        # Log statistics
        for cls, patches in patches_by_class.items():
            logger.info(f"Class {cls}: {len(patches)} patches extracted")

        return patches_by_class

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get patch extractor statistics.

        Returns:
            Dictionary with extraction configuration and stats
        """
        return {
            "patch_size": self.patch_size,
            "num_points": self.num_points,
            "architecture": self.architecture,
            "augmentation_enabled": self.augment,
            "num_augmentations": self.num_augmentations,
            "patch_overlap": self.patch_overlap,
        }
