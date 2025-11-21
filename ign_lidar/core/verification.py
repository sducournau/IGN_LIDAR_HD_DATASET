#!/usr/bin/env python3
"""
Feature Verification Module (Merged)

This module provides comprehensive functionality to verify geometric features, RGB, and NIR
values in enriched LAZ files. Combines artifact detection with detailed reporting.

Merged from verifier.py and verification.py for consolidated functionality.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try to import laspy
try:
    import laspy

    LASPY_AVAILABLE = True
except ImportError:
    LASPY_AVAILABLE = False
    logger.warning("laspy not available - feature verification disabled")


# CORE MODE FEATURES - Always computed (both core and full modes)
# These are the fundamental geometric features based on eigenvalue decomposition
CORE_FEATURES = [
    "linearity",  # (λ0-λ1)/Σλ - 1D structures (edges, cables)
    "planarity",  # (λ1-λ2)/Σλ - 2D structures (roofs, walls)
    "sphericity",  # λ2/Σλ - 3D structures (vegetation, noise)
    "anisotropy",  # (λ0-λ2)/λ0 - general directionality
    "roughness",  # λ2/Σλ - surface roughness
    "density",  # local point density
    "verticality",  # 1 - |nz| - how vertical the surface is
]

# FULL MODE ADDITIONAL FEATURES - Building-specific features
# Only computed in full mode (mode='full' with include_extra=True)
FULL_MODE_FEATURES = [
    # Building scores
    "wall_score",  # Probability of being a wall (verticality + height)
    "roof_score",  # Probability of being a roof (horizontality + height + curvature)
    # Eigenvalue features (LOD3_FULL)
    "eigenvalue_1",  # Largest eigenvalue (λ₀)
    "eigenvalue_2",  # Medium eigenvalue (λ₁)
    "eigenvalue_3",  # Smallest eigenvalue (λ₂)
    "sum_eigenvalues",  # Sum of all eigenvalues (Σλ)
    "eigenentropy",  # Shannon entropy of eigenvalues
    "omnivariance",  # Cubic root of eigenvalue product
    "change_curvature",  # Variance of eigenvalues (surface change rate)
    # Architectural features (LOD3_FULL)
    "edge_strength",  # Edge detection strength
    "corner_likelihood",  # Corner probability
    "overhang_indicator",  # Overhang/protrusion detection
    "surface_roughness",  # Fine-scale surface texture
    # Density features
    "num_points_2m",  # Number of points within 2m radius
    # Height features (when include_extra=True)
    "z_absolute",  # Absolute Z coordinate
    "z_normalized",  # Z normalized to [0,1]
    "z_from_ground",  # Height above ground
    "z_from_median",  # Height relative to median
    "distance_to_center",  # Distance to patch center (optional)
    # Local statistics
    "vertical_std",  # Vertical standard deviation in neighborhood
    "neighborhood_extent",  # Maximum distance to k-th neighbor
    "height_extent_ratio",  # Ratio of vertical std to spatial extent
    "local_roughness",  # Local surface roughness
]

# All expected geometric features (core + full mode)
EXPECTED_FEATURES = CORE_FEATURES + FULL_MODE_FEATURES

# Always present features (not in extra dims, but part of point cloud)
STANDARD_FEATURES = [
    "height_above_ground",  # Height relative to ground classification
    # Note: normals and curvature are computed but may not be stored
]

# Expected RGB features
RGB_FEATURES = ["red", "green", "blue"]

# Expected infrared features
INFRARED_FEATURES = ["infrared", "nir"]


@dataclass
class FeatureStats:
    """Statistics for a single feature dimension."""

    name: str
    present: bool
    count: int
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    has_artifacts: bool = False
    artifact_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "present": self.present,
            "count": self.count,
            "min": self.min_val,
            "max": self.max_val,
            "mean": self.mean,
            "std": self.std,
            "has_artifacts": self.has_artifacts,
            "artifact_reasons": self.artifact_reasons,
        }

    def __str__(self) -> str:
        """String representation."""
        if not self.present:
            return f"{self.name}: NOT PRESENT"

        status = "✓" if not self.has_artifacts else "⚠"
        result = (
            f"{status} {self.name}: "
            f"range=[{self.min_val:.4f}, {self.max_val:.4f}], "
            f"mean={self.mean:.4f}, std={self.std:.4f}"
        )

        if self.has_artifacts:
            result += f" - Issues: {', '.join(self.artifact_reasons)}"

        return result


class FeatureVerifier:
    """Verify features in LAZ files."""

    def __init__(
        self,
        expected_features: Optional[list[str]] = None,
        sample_size: int = 1000,
        check_rgb: bool = False,
        check_infrared: bool = False,
        mode: str = "core",
    ):
        """Initialize verifier.

        Args:
            expected_features: List of expected feature names (default: auto-detect based on mode)
            sample_size: Number of points to sample for analysis
            check_rgb: Also check for RGB features
            check_infrared: Also check for infrared features
            mode: Processing mode ('core' or 'full')
                  - core: Only check for CORE_FEATURES
                  - full: Check for CORE_FEATURES + FULL_MODE_FEATURES
        """
        # Auto-detect expected features based on mode if not specified
        if expected_features is None:
            if mode == "full":
                self.expected_features = CORE_FEATURES + FULL_MODE_FEATURES
            else:
                self.expected_features = CORE_FEATURES.copy()
        else:
            self.expected_features = expected_features

        self.sample_size = sample_size
        self.check_rgb = check_rgb
        self.check_infrared = check_infrared
        self.mode = mode

        if not LASPY_AVAILABLE:
            raise RuntimeError("laspy is required for feature verification")

    def verify_file(self, laz_path: Path) -> dict[str, FeatureStats]:
        """Verify features in a LAZ file.

        Args:
            laz_path: Path to LAZ file

        Returns:
            Dictionary mapping feature names to statistics
        """
        logger.debug(f"Verifying features in: {laz_path}")

        try:
            with laspy.open(str(laz_path)) as f:
                las = f.read()
        except Exception as e:
            logger.error(f"Failed to read LAZ file {laz_path}: {e}")
            raise

        results = {}

        # Verify expected features
        for feature_name in self.expected_features:
            stats = self._verify_feature(las, feature_name)
            results[feature_name] = stats

        # Optionally check RGB
        if self.check_rgb:
            for feature_name in RGB_FEATURES:
                stats = self._verify_feature(las, feature_name, is_rgb=True)
                results[feature_name] = stats

        # Optionally check infrared
        if self.check_infrared:
            for feature_name in INFRARED_FEATURES:
                stats = self._verify_feature(las, feature_name)
                results[feature_name] = stats

        return results

    def _verify_feature(
        self, las: "laspy.LasData", feature_name: str, is_rgb: bool = False
    ) -> FeatureStats:
        """Verify a single feature.

        Args:
            las: LasData object
            feature_name: Name of feature to verify
            is_rgb: Whether this is an RGB feature

        Returns:
            FeatureStats object with verification results
        """
        # Check if feature exists
        if feature_name not in las.point_format.dimension_names:
            return FeatureStats(name=feature_name, present=False, count=0)

        # Get feature data
        try:
            data = getattr(las, feature_name)
        except AttributeError:
            return FeatureStats(name=feature_name, present=False, count=0)

        # Sample for detailed analysis
        sample = self._sample_data(data)

        # Compute statistics
        stats = FeatureStats(
            name=feature_name,
            present=True,
            count=len(data),
            min_val=float(np.min(sample)),
            max_val=float(np.max(sample)),
            mean=float(np.mean(sample)),
            std=float(np.std(sample)),
        )

        # Check for artifacts
        stats.has_artifacts, stats.artifact_reasons = self._check_artifacts(
            sample, feature_name, is_rgb=is_rgb
        )

        return stats

    def _sample_data(self, data: np.ndarray) -> np.ndarray:
        """Sample data for analysis.

        Args:
            data: Full data array

        Returns:
            Sampled data array
        """
        if len(data) <= self.sample_size:
            return data

        indices = np.random.choice(len(data), size=self.sample_size, replace=False)
        return data[indices]

    def _check_artifacts(
        self, data: np.ndarray, feature_name: str, is_rgb: bool = False
    ) -> tuple[bool, list[str]]:
        """Check for artifacts in feature data.

        Args:
            data: Feature data array
            feature_name: Name of feature
            is_rgb: Whether this is an RGB feature

        Returns:
            Tuple of (has_artifacts, list_of_reasons)
        """
        reasons = []

        # Check for NaN/Inf
        if np.any(~np.isfinite(data)):
            nan_count = np.sum(np.isnan(data))
            inf_count = np.sum(np.isinf(data))
            reasons.append(f"Contains {nan_count} NaN and {inf_count} Inf values")

        # Check for constant values
        if np.std(data) < 1e-6:
            reasons.append("Nearly constant values (std < 1e-6)")

        # Check for unexpected range based on feature type
        if is_rgb:
            # RGB values should be 0-65535 (16-bit)
            if data.min() < 0 or data.max() > 65535:
                reasons.append(f"RGB values outside expected range [0, 65535]")
        else:
            # Define expected ranges for geometric features
            expected_ranges = {
                "linearity": (0, 1),
                "planarity": (0, 1),
                "sphericity": (0, 1),
                "anisotropy": (0, 1),
                "curvature": (0, 1),
                "omnivariance": (0, None),  # No upper bound
                "eigensum": (0, None),
                "roughness": (0, None),
            }

            if feature_name in expected_ranges:
                min_exp, max_exp = expected_ranges[feature_name]

                # Check lower bound
                if min_exp is not None and data.min() < min_exp - 0.1:
                    reasons.append(
                        f"Values below expected minimum {min_exp} "
                        f"(actual min: {data.min():.4f})"
                    )

                # Check upper bound
                if max_exp is not None and data.max() > max_exp + 0.1:
                    reasons.append(
                        f"Values above expected maximum {max_exp} "
                        f"(actual max: {data.max():.4f})"
                    )

        # Check for suspiciously peaked distribution
        unique_ratio = len(np.unique(data)) / len(data)
        if unique_ratio < 0.01:
            reasons.append(f"Low value diversity ({unique_ratio*100:.1f}% unique)")

        return len(reasons) > 0, reasons

    def generate_report(self, results: dict[str, FeatureStats], file_path: Path) -> str:
        """Generate a text report of verification results.

        Args:
            results: Dictionary of feature statistics
            file_path: Path to file that was verified

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 70,
            f"Feature Verification Report: {file_path.name}",
            "=" * 70,
            "",
        ]

        # Count features
        present_features = [s for s in results.values() if s.present]
        missing_features = [s for s in results.values() if not s.present]
        artifact_features = [s for s in present_features if s.has_artifacts]

        # Summary
        lines.append(f"Total features checked: {len(results)}")
        lines.append(f"Features present: {len(present_features)}")
        lines.append(f"Features missing: {len(missing_features)}")
        lines.append(f"Features with artifacts: {len(artifact_features)}")
        lines.append("")

        # Present features
        if present_features:
            lines.append("Present Features:")
            lines.append("-" * 70)
            for stats in present_features:
                lines.append(f"  {stats}")
            lines.append("")

        # Missing features
        if missing_features:
            lines.append("Missing Features:")
            lines.append("-" * 70)
            for stats in missing_features:
                lines.append(f"  ✗ {stats.name}")
            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

    def print_summary(self, all_results: List[Dict[str, "FeatureStats"]]):
        """Print summary of multiple file verifications.

        Args:
            all_results: List of verification result dictionaries
        """
        if not all_results or len(all_results) <= 1:
            return

        logger.info("\n" + "=" * 70)
        logger.info("VERIFICATION SUMMARY")
        logger.info("=" * 70)

        total_files = len(all_results)

        # Get all feature names from first result
        if all_results:
            feature_names = list(all_results[0].keys())
        else:
            return

        # Count feature presence - Vectorized implementation (P1 optimization)
        logger.info(f"Files verified: {total_files}\n")
        logger.info("Feature presence:")

        # Pre-compute boolean matrices for vectorized operations
        n_files = len(all_results)
        n_features = len(feature_names)
        
        # Create boolean matrices: presence[i, j] = feature j present in file i
        presence_matrix = np.zeros((n_files, n_features), dtype=bool)
        artifacts_matrix = np.zeros((n_files, n_features), dtype=bool)
        
        for i, result in enumerate(all_results):
            for j, feat_name in enumerate(feature_names):
                feat_stats = result[feat_name]
                presence_matrix[i, j] = feat_stats.present
                artifacts_matrix[i, j] = feat_stats.present and feat_stats.has_artifacts
        
        # Vectorized counting - much faster than nested loops
        present_counts = presence_matrix.sum(axis=0)  # Sum over files for each feature
        artifact_counts = artifacts_matrix.sum(axis=0)
        
        # Report per feature
        for j, feat_name in enumerate(feature_names):
            present_count = int(present_counts[j])
            artifact_count = int(artifact_counts[j])
            
            status = "✓" if present_count == total_files else "⚠️"
            line = f"  {status} {feat_name:15s}: {present_count}/{total_files} files"

            if artifact_count > 0:
                line += f" ({artifact_count} with artifacts)"

            logger.info(line)

        # Count total artifacts - vectorized
        total_artifacts = int(artifacts_matrix.sum())

        if total_artifacts > 0:
            # Files with at least one artifact
            files_with_artifacts = int((artifacts_matrix.sum(axis=1) > 0).sum())
            logger.info(f"\n⚠️  Total features with artifacts: {total_artifacts}")
            logger.info(f"  Files with artifacts: {files_with_artifacts}/{total_files}")
        else:
            logger.info(f"\n✓ No artifacts detected")

        logger.info("=" * 70)


def verify_laz_files(
    input_path: Optional[Path] = None,
    input_dir: Optional[Path] = None,
    max_files: Optional[int] = None,
    verbose: bool = True,
    show_samples: bool = False,
) -> List[Dict[str, FeatureStats]]:
    """
    Verify features in LAZ file(s).

    Compatibility function that maintains the original verifier.py interface
    while using the enhanced FeatureVerifier class.

    Args:
        input_path: Single LAZ file to verify
        input_dir: Directory containing LAZ files to verify
        max_files: Maximum number of files to verify
        verbose: If True, print detailed information
        show_samples: If True, display sample points (not implemented yet)

    Returns:
        List of verification result dictionaries
    """
    # Find LAZ files
    laz_files = []

    if input_path:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        laz_files = [input_path]

    elif input_dir:
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Search recursively for LAZ files
        laz_files = list(input_dir.rglob("*.laz")) + list(input_dir.rglob("*.LAZ"))

        if not laz_files:
            raise FileNotFoundError(f"No LAZ files found in {input_dir}")

        if verbose:
            logger.info(f"Found {len(laz_files)} LAZ files")

        # Limit to max_files if specified
        if max_files and len(laz_files) > max_files:
            if verbose:
                logger.info(f"Limiting to first {max_files} files")
            laz_files = sorted(laz_files)[:max_files]

    else:
        raise ValueError("Either input_path or input_dir must be specified")

    # Initialize verifier with all feature checks
    verifier = FeatureVerifier(
        expected_features=EXPECTED_FEATURES,
        sample_size=1000,
        check_rgb=True,
        check_infrared=True,
    )

    all_results = []

    for laz_file in laz_files:
        try:
            if verbose:
                logger.info(f"\n{'='*70}")
                logger.info(f"Analyzing: {laz_file.name}")
                logger.info("=" * 70)

            results = verifier.verify_file(laz_file)
            all_results.append(results)

            if verbose:
                # Print detailed stats for this file
                present_count = sum(1 for s in results.values() if s.present)
                artifact_count = sum(
                    1 for s in results.values() if s.present and s.has_artifacts
                )

                logger.info(f"\nSummary for {laz_file.name}:")
                logger.info(f"  Features present: {present_count}/{len(results)}")
                if artifact_count > 0:
                    logger.info(f"  ⚠️  Features with artifacts: {artifact_count}")
                else:
                    logger.info("  ✓ No artifacts detected")

                # Show detailed stats for each feature
                for feat_name, stats in results.items():
                    if stats.present:
                        status = "✓" if not stats.has_artifacts else "⚠"
                        logger.info(
                            f"  {status} {feat_name:15s}: "
                            f"range=[{stats.min_val:.4f}, {stats.max_val:.4f}], "
                            f"mean={stats.mean:.4f}, std={stats.std:.4f}"
                        )

                        if stats.has_artifacts:
                            for reason in stats.artifact_reasons:
                                logger.info(f"      Issue: {reason}")
                    else:
                        logger.info(f"  ✗ {feat_name:15s}: NOT PRESENT")

                logger.info("=" * 70)

        except Exception as e:
            logger.error(f"Error processing {laz_file.name}: {e}")
            if verbose:
                import traceback

                traceback.print_exc()

    # Print summary
    if verbose and len(all_results) > 1:
        verifier.print_summary(all_results)

    return all_results
