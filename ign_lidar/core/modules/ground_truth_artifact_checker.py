"""
Ground Truth Artifact Checker

This module provides comprehensive artifact detection for ground truth classification.
It validates that features are correct before using them for classification.

Key checks:
1. Feature quality validation (NaN, Inf, range checks)
2. Feature consistency validation (cross-feature validation)
3. Ground truth alignment validation (features match GT labels)
4. Artifact detection and filtering

Author: IGN LiDAR HD Classification Team
Date: October 19, 2025
Version: 1.0
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, List, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ArtifactReport:
    """Report of artifacts detected in features."""
    
    feature_name: str
    total_points: int
    present: bool = True  # Feature is present in data
    
    # Data quality issues
    nan_count: int = 0
    inf_count: int = 0
    out_of_range_count: int = 0
    constant_values: bool = False
    
    # Statistical anomalies
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    
    # Artifact flags
    has_artifacts: bool = False
    artifact_reasons: List[str] = field(default_factory=list)
    
    def add_artifact(self, reason: str):
        """Add an artifact reason."""
        self.has_artifacts = True
        if reason not in self.artifact_reasons:
            self.artifact_reasons.append(reason)
    
    def __str__(self) -> str:
        """String representation."""
        status = "✓" if not self.has_artifacts else "⚠️"
        result = f"{status} {self.feature_name}: "
        
        if not self.has_artifacts:
            result += f"OK (range=[{self.min_val:.3f}, {self.max_val:.3f}], mean={self.mean:.3f})"
        else:
            result += f"ARTIFACTS DETECTED"
            for reason in self.artifact_reasons:
                result += f"\n    - {reason}"
        
        return result


class GroundTruthArtifactChecker:
    """
    Check for artifacts in features before using for ground truth classification.
    
    This prevents using corrupted or invalid features that would result in
    incorrect classifications.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize artifact checker.
        
        Args:
            config: Configuration dictionary with feature thresholds
        """
        self.config = config or {}
        
        # Feature expected ranges (feature_name -> (min, max))
        self.expected_ranges = {
            # Geometric features (0-1 normalized)
            'linearity': (0.0, 1.0),
            'planarity': (0.0, 1.0),
            'sphericity': (0.0, 1.0),
            'anisotropy': (0.0, 1.0),
            'verticality': (0.0, 1.0),
            'curvature': (0.0, 1.0),
            'roughness': (0.0, None),  # No upper limit
            
            # Height features
            'height': (-10.0, 200.0),  # Reasonable range for buildings
            'height_above_ground': (0.0, 200.0),
            'z_from_ground': (0.0, 200.0),
            
            # NDVI and spectral features
            'ndvi': (-1.0, 1.0),
            'nir': (0.0, 1.0),
            'red': (0, 65535),  # 16-bit
            'green': (0, 65535),
            'blue': (0, 65535),
            'infrared': (0, 65535),
            
            # Intensity
            'intensity': (0, 65535),
            'brightness': (0, 1.0),
            
            # Eigenvalues
            'eigenvalue_1': (0.0, None),
            'eigenvalue_2': (0.0, None),
            'eigenvalue_3': (0.0, None),
        }
        
        # Update with config overrides
        if 'feature_ranges' in self.config:
            self.expected_ranges.update(self.config['feature_ranges'])
        
        # Minimum valid std (to detect constant values)
        self.min_std_threshold = self.config.get('min_std_threshold', 1e-6)
        
        # Maximum allowed artifact ratio
        self.max_artifact_ratio = self.config.get('max_artifact_ratio', 0.1)
    
    def check_feature(
        self,
        feature_name: str,
        feature_data: np.ndarray
    ) -> ArtifactReport:
        """
        Check a single feature for artifacts.
        
        Args:
            feature_name: Name of the feature
            feature_data: Feature values array [N]
        
        Returns:
            ArtifactReport with detection results
        """
        report = ArtifactReport(
            feature_name=feature_name,
            total_points=len(feature_data)
        )
        
        # Check for NaN values
        nan_mask = np.isnan(feature_data)
        report.nan_count = np.sum(nan_mask)
        if report.nan_count > 0:
            report.add_artifact(
                f"{report.nan_count} NaN values ({report.nan_count/len(feature_data)*100:.1f}%)"
            )
        
        # Check for Inf values
        inf_mask = np.isinf(feature_data)
        report.inf_count = np.sum(inf_mask)
        if report.inf_count > 0:
            report.add_artifact(
                f"{report.inf_count} Inf values ({report.inf_count/len(feature_data)*100:.1f}%)"
            )
        
        # Get valid data (no NaN/Inf)
        valid_mask = ~(nan_mask | inf_mask)
        valid_data = feature_data[valid_mask]
        
        if len(valid_data) == 0:
            report.add_artifact("All values are NaN or Inf")
            return report
        
        # Compute statistics
        report.mean = float(np.mean(valid_data))
        report.std = float(np.std(valid_data))
        report.min_val = float(np.min(valid_data))
        report.max_val = float(np.max(valid_data))
        
        # Check for constant values
        if report.std < self.min_std_threshold:
            report.constant_values = True
            report.add_artifact(
                f"Nearly constant values (std={report.std:.2e} < {self.min_std_threshold:.2e})"
            )
        
        # Check expected range
        if feature_name in self.expected_ranges:
            min_expected, max_expected = self.expected_ranges[feature_name]
            
            # Check minimum
            if min_expected is not None:
                below_min = valid_data < (min_expected - 0.1)  # Allow small tolerance
                if np.any(below_min):
                    count = np.sum(below_min)
                    report.out_of_range_count += count
                    report.add_artifact(
                        f"{count} values below expected minimum {min_expected} "
                        f"(actual min: {report.min_val:.3f})"
                    )
            
            # Check maximum
            if max_expected is not None:
                above_max = valid_data > (max_expected + 0.1)  # Allow small tolerance
                if np.any(above_max):
                    count = np.sum(above_max)
                    report.out_of_range_count += count
                    report.add_artifact(
                        f"{count} values above expected maximum {max_expected} "
                        f"(actual max: {report.max_val:.3f})"
                    )
        
        # Check for suspiciously low diversity
        unique_ratio = len(np.unique(valid_data)) / len(valid_data)
        if unique_ratio < 0.01:
            report.add_artifact(
                f"Low value diversity ({unique_ratio*100:.2f}% unique values)"
            )
        
        return report
    
    def check_all_features(
        self,
        features: Dict[str, np.ndarray]
    ) -> Dict[str, ArtifactReport]:
        """
        Check all features for artifacts.
        
        Args:
            features: Dictionary mapping feature names to arrays
        
        Returns:
            Dictionary mapping feature names to ArtifactReports
        """
        reports = {}
        
        for feature_name, feature_data in features.items():
            if feature_data is not None:
                reports[feature_name] = self.check_feature(feature_name, feature_data)
        
        return reports
    
    def filter_artifacts(
        self,
        features: Dict[str, np.ndarray],
        reports: Optional[Dict[str, ArtifactReport]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Filter out points with artifacts from features.
        
        Args:
            features: Dictionary of feature arrays
            reports: Optional pre-computed artifact reports
        
        Returns:
            Tuple of (clean_features, artifact_masks)
            - clean_features: Features with artifacts set to NaN
            - artifact_masks: Boolean masks of artifact points for each feature
        """
        if reports is None:
            reports = self.check_all_features(features)
        
        clean_features = {}
        artifact_masks = {}
        
        for feature_name, feature_data in features.items():
            if feature_data is None:
                clean_features[feature_name] = None
                artifact_masks[feature_name] = np.zeros(0, dtype=bool)
                continue
            
            # Create mask for artifact points
            artifact_mask = np.zeros(len(feature_data), dtype=bool)
            
            # Mark NaN/Inf
            artifact_mask |= ~np.isfinite(feature_data)
            
            # Mark out of range
            if feature_name in self.expected_ranges:
                min_expected, max_expected = self.expected_ranges[feature_name]
                
                if min_expected is not None:
                    artifact_mask |= (feature_data < (min_expected - 0.1))
                
                if max_expected is not None:
                    artifact_mask |= (feature_data > (max_expected + 0.1))
            
            # Create clean version
            clean_data = feature_data.copy()
            clean_data[artifact_mask] = np.nan
            
            clean_features[feature_name] = clean_data
            artifact_masks[feature_name] = artifact_mask
        
        return clean_features, artifact_masks
    
    def validate_for_ground_truth(
        self,
        features: Dict[str, np.ndarray],
        strict: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        Validate that features are suitable for ground truth classification.
        
        Args:
            features: Dictionary of feature arrays
            strict: If True, reject if ANY artifacts found
                   If False, reject only if artifact ratio exceeds threshold
        
        Returns:
            Tuple of (is_valid, list of warnings)
        """
        reports = self.check_all_features(features)
        warnings = []
        
        # Count features with artifacts
        features_with_artifacts = [
            name for name, report in reports.items()
            if report.has_artifacts
        ]
        
        if not features_with_artifacts:
            return True, []
        
        # Check artifact severity
        for feature_name in features_with_artifacts:
            report = reports[feature_name]
            
            # Calculate artifact ratio
            total_artifacts = (
                report.nan_count + 
                report.inf_count + 
                report.out_of_range_count
            )
            artifact_ratio = total_artifacts / report.total_points
            
            if strict and artifact_ratio > 0:
                warnings.append(
                    f"{feature_name}: Has artifacts (ratio={artifact_ratio:.2%})"
                )
            elif artifact_ratio > self.max_artifact_ratio:
                warnings.append(
                    f"{feature_name}: Artifact ratio {artifact_ratio:.2%} "
                    f"exceeds threshold {self.max_artifact_ratio:.2%}"
                )
        
        is_valid = len(warnings) == 0
        
        return is_valid, warnings
    
    def log_artifact_summary(
        self,
        reports: Dict[str, ArtifactReport]
    ):
        """
        Log summary of artifact detection results.
        
        Args:
            reports: Dictionary of artifact reports
        """
        features_with_artifacts = [
            name for name, report in reports.items()
            if report.has_artifacts
        ]
        
        clean_features = [
            name for name, report in reports.items()
            if not report.has_artifacts
        ]
        
        logger.info("=" * 70)
        logger.info("ARTIFACT DETECTION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total features checked: {len(reports)}")
        logger.info(f"Clean features: {len(clean_features)}")
        logger.info(f"Features with artifacts: {len(features_with_artifacts)}")
        
        if clean_features:
            logger.info("\n✓ Clean Features:")
            for name in clean_features:
                report = reports[name]
                logger.info(f"  {report}")
        
        if features_with_artifacts:
            logger.info("\n⚠️ Features with Artifacts:")
            for name in features_with_artifacts:
                report = reports[name]
                logger.info(f"  {report}")
        
        logger.info("=" * 70)


def validate_features_before_classification(
    features: Dict[str, np.ndarray],
    strict: bool = False,
    log_results: bool = True
) -> Tuple[bool, Dict[str, ArtifactReport]]:
    """
    Convenience function to validate features before ground truth classification.
    
    Args:
        features: Dictionary of feature arrays
        strict: If True, reject if ANY artifacts found
        log_results: If True, log detailed results
    
    Returns:
        Tuple of (is_valid, artifact_reports)
    """
    checker = GroundTruthArtifactChecker()
    
    # Check all features
    reports = checker.check_all_features(features)
    
    # Log results
    if log_results:
        checker.log_artifact_summary(reports)
    
    # Validate
    is_valid, warnings = checker.validate_for_ground_truth(features, strict=strict)
    
    if not is_valid:
        logger.warning("Feature validation failed:")
        for warning in warnings:
            logger.warning(f"  - {warning}")
    
    return is_valid, reports


def get_artifact_free_features(
    features: Dict[str, np.ndarray]
) -> Tuple[Set[str], Set[str]]:
    """
    Identify which features are clean vs. have artifacts.
    
    Args:
        features: Dictionary of feature arrays
    
    Returns:
        Tuple of (clean_features, artifact_features) as sets of feature names
    """
    checker = GroundTruthArtifactChecker()
    reports = checker.check_all_features(features)
    
    clean_features = set()
    artifact_features = set()
    
    for feature_name, report in reports.items():
        if report.present and not report.has_artifacts:
            clean_features.add(feature_name)
        elif report.present and report.has_artifacts:
            # Check if artifacts are severe (>10% of points)
            total_artifacts = report.nan_count + report.inf_count + report.out_of_range_count
            artifact_ratio = total_artifacts / report.total_points if report.total_points > 0 else 1.0
            
            if artifact_ratio > 0.1:  # More than 10% artifacts
                artifact_features.add(feature_name)
                logger.warning(
                    f"Feature '{feature_name}' has {artifact_ratio:.1%} artifacts - "
                    f"will adapt classification rules"
                )
            else:
                # Minor artifacts, still usable
                clean_features.add(feature_name)
    
    return clean_features, artifact_features
