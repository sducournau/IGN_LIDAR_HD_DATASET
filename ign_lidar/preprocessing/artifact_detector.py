"""
Artifact Detection Module

This module provides comprehensive artifact detection for LiDAR point clouds,
with a focus on identifying scan line artifacts (dash lines) and other spatial
patterns that indicate data quality issues.

Features:
- Dash line detection via spatial feature analysis
- Coefficient of variation (CV) metrics for artifact severity
- 2D heatmap visualization for artifact patterns
- Y-direction profile plotting (perpendicular to flight lines)
- Automatic field dropping based on artifact thresholds
- Batch processing capabilities
- Comprehensive reporting

Typical artifacts detected:
- Scan line striping (parallel dashes perpendicular to flight direction)
- Boundary effects from tile edges
- Noisy features from sparse neighborhoods
- Degenerate geometric features
"""

from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass, field

# Optional matplotlib import (only needed for visualization)
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    LinearSegmentedColormap = None

logger = logging.getLogger(__name__)


@dataclass
class ArtifactMetrics:
    """Container for artifact detection metrics."""
    feature_name: str
    cv_x: float  # Coefficient of variation in X direction
    cv_y: float  # Coefficient of variation in Y direction
    max_cv: float  # Maximum CV
    mean_value: float
    std_value: float
    spatial_variance_x: float  # Variance of bin means in X
    spatial_variance_y: float  # Variance of bin means in Y
    severity: str  # 'low', 'medium', 'high', 'severe'
    has_artifacts: bool
    recommended_action: str  # 'keep', 'review', 'drop'


@dataclass
class ArtifactDetectorConfig:
    """Configuration for artifact detection."""
    # CV thresholds for severity classification
    cv_low_threshold: float = 0.10  # Below = no artifacts
    cv_medium_threshold: float = 0.20  # Below = low severity
    cv_high_threshold: float = 0.35  # Below = medium severity
    # Above = severe artifacts
    
    # Grid parameters for spatial analysis
    n_bins_x: int = 25  # Number of bins in X direction
    n_bins_y: int = 50  # Number of bins in Y direction (perpendicular to flight lines)
    
    # Visualization parameters
    grid_size: int = 50  # Resolution for 2D heatmaps
    plot_dpi: int = 150
    show_dash_lines: bool = True  # Overlay detected dash lines
    
    # Default features to check
    default_features: List[str] = field(default_factory=lambda: [
        'planarity', 'roof_score', 'linearity', 'curvature',
        'verticality', 'surface_variation', 'omnivariance'
    ])
    
    # Action thresholds
    auto_drop_threshold: float = 0.40  # CV above = auto-drop recommended
    review_threshold: float = 0.25  # CV above = review recommended
    

class ArtifactDetector:
    """
    Detect and analyze artifacts in LiDAR point cloud features.
    
    Focuses on scan line artifacts that appear as parallel dash lines
    perpendicular to flight direction (typically Y-direction in LAMB93).
    """
    
    def __init__(self, config: Optional[ArtifactDetectorConfig] = None):
        """
        Initialize artifact detector.
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or ArtifactDetectorConfig()
        logger.info("Initialized ArtifactDetector")
    
    def detect_spatial_artifacts(
        self,
        coords: np.ndarray,
        feature_values: np.ndarray,
        feature_name: str
    ) -> ArtifactMetrics:
        """
        Detect spatial artifacts in a single feature.
        
        Analyzes spatial distribution of feature values to identify
        scan line artifacts and other spatial patterns.
        
        Args:
            coords: [N, 3] XYZ coordinates
            feature_values: [N] feature values
            feature_name: Name of the feature
            
        Returns:
            ArtifactMetrics object with detection results
        """
        x, y = coords[:, 0], coords[:, 1]
        
        # X-direction analysis (parallel to flight lines)
        x_bins = np.linspace(x.min(), x.max(), self.config.n_bins_x + 1)
        x_digitized = np.digitize(x, x_bins)
        
        x_means = []
        for i in range(1, self.config.n_bins_x + 1):
            mask = x_digitized == i
            if mask.sum() > 10:  # Require minimum points per bin
                x_means.append(feature_values[mask].mean())
        
        x_means = np.array(x_means)
        
        # Y-direction analysis (perpendicular to flight lines - where artifacts appear)
        y_bins = np.linspace(y.min(), y.max(), self.config.n_bins_y + 1)
        y_digitized = np.digitize(y, y_bins)
        
        y_means = []
        for i in range(1, self.config.n_bins_y + 1):
            mask = y_digitized == i
            if mask.sum() > 10:
                y_means.append(feature_values[mask].mean())
        
        y_means = np.array(y_means)
        
        # Compute coefficients of variation (CV)
        mean_val = feature_values.mean()
        std_val = feature_values.std()
        
        cv_x = np.std(x_means) / np.mean(x_means) if len(x_means) > 0 and np.mean(x_means) > 1e-10 else 0.0
        cv_y = np.std(y_means) / np.mean(y_means) if len(y_means) > 0 and np.mean(y_means) > 1e-10 else 0.0
        max_cv = max(cv_x, cv_y)
        
        # Spatial variances
        spatial_var_x = np.var(x_means) if len(x_means) > 0 else 0.0
        spatial_var_y = np.var(y_means) if len(y_means) > 0 else 0.0
        
        # Classify severity
        if max_cv < self.config.cv_low_threshold:
            severity = 'low'
            has_artifacts = False
            action = 'keep'
        elif max_cv < self.config.cv_medium_threshold:
            severity = 'medium'
            has_artifacts = True
            action = 'keep'
        elif max_cv < self.config.cv_high_threshold:
            severity = 'high'
            has_artifacts = True
            action = 'review' if max_cv > self.config.review_threshold else 'keep'
        else:
            severity = 'severe'
            has_artifacts = True
            action = 'drop' if max_cv > self.config.auto_drop_threshold else 'review'
        
        return ArtifactMetrics(
            feature_name=feature_name,
            cv_x=cv_x,
            cv_y=cv_y,
            max_cv=max_cv,
            mean_value=mean_val,
            std_value=std_val,
            spatial_variance_x=spatial_var_x,
            spatial_variance_y=spatial_var_y,
            severity=severity,
            has_artifacts=has_artifacts,
            recommended_action=action
        )
    
    def create_2d_heatmap(
        self,
        coords: np.ndarray,
        feature_values: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """
        Create 2D grid representation of feature values.
        
        Args:
            coords: [N, 3] XYZ coordinates
            feature_values: [N] feature values
            
        Returns:
            grid: 2D array of averaged feature values
            extent: (x_min, x_max, y_min, y_max) for plotting
        """
        x, y = coords[:, 0], coords[:, 1]
        
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        # Create bins
        grid_size = self.config.grid_size
        x_bins = np.linspace(x_min, x_max, grid_size + 1)
        y_bins = np.linspace(y_min, y_max, grid_size + 1)
        
        # Digitize points
        x_idx = np.digitize(x, x_bins) - 1
        y_idx = np.digitize(y, y_bins) - 1
        
        # Clip to valid range
        x_idx = np.clip(x_idx, 0, grid_size - 1)
        y_idx = np.clip(y_idx, 0, grid_size - 1)
        
        # Create grid with NaN for empty cells
        grid = np.full((grid_size, grid_size), np.nan)
        counts = np.zeros((grid_size, grid_size))
        
        # Accumulate values
        for i in range(len(feature_values)):
            if np.isfinite(feature_values[i]):
                grid[y_idx[i], x_idx[i]] = np.nansum([grid[y_idx[i], x_idx[i]], 
                                                       feature_values[i]])
                counts[y_idx[i], x_idx[i]] += 1
        
        # Average
        grid = np.where(counts > 0, grid / counts, np.nan)
        
        extent = (x_min, x_max, y_min, y_max)
        return grid, extent
    
    def detect_dash_lines(
        self,
        coords: np.ndarray,
        feature_values: np.ndarray,
        metrics: ArtifactMetrics
    ) -> List[float]:
        """
        Detect Y-coordinates of dash lines (high-variance bands).
        
        Args:
            coords: [N, 3] XYZ coordinates
            feature_values: [N] feature values
            metrics: ArtifactMetrics from spatial analysis
            
        Returns:
            List of Y-coordinates where dash lines are detected
        """
        if not metrics.has_artifacts or metrics.cv_y < self.config.cv_medium_threshold:
            return []
        
        y = coords[:, 1]
        n_bins = self.config.n_bins_y * 2  # Higher resolution for dash detection
        
        y_bins = np.linspace(y.min(), y.max(), n_bins + 1)
        y_digitized = np.digitize(y, y_bins)
        
        y_means = []
        y_centers = []
        
        for i in range(1, n_bins + 1):
            mask = y_digitized == i
            if mask.sum() > 5:
                y_means.append(feature_values[mask].mean())
                y_centers.append((y_bins[i-1] + y_bins[i]) / 2)
        
        if len(y_means) < 3:
            return []
        
        y_means = np.array(y_means)
        y_centers = np.array(y_centers)
        
        # Detect outliers (significantly different from neighbors)
        global_mean = y_means.mean()
        global_std = y_means.std()
        threshold = global_mean + 1.5 * global_std
        
        dash_locations = []
        for i, (val, y_pos) in enumerate(zip(y_means, y_centers)):
            if abs(val - global_mean) > global_std:
                # Check if it's part of a pattern (neighbors also deviate)
                neighbor_deviate = False
                if i > 0 and abs(y_means[i-1] - global_mean) > 0.5 * global_std:
                    neighbor_deviate = True
                if i < len(y_means) - 1 and abs(y_means[i+1] - global_mean) > 0.5 * global_std:
                    neighbor_deviate = True
                
                if neighbor_deviate or abs(val - global_mean) > 2 * global_std:
                    dash_locations.append(y_pos)
        
        return dash_locations
    
    def visualize_artifacts(
        self,
        coords: np.ndarray,
        feature_values: np.ndarray,
        feature_name: str,
        metrics: ArtifactMetrics,
        output_path: Optional[Path] = None,
        show: bool = False
    ) -> None:
        """
        Create comprehensive artifact visualization with dash lines.
        
        Args:
            coords: [N, 3] XYZ coordinates
            feature_values: [N] feature values
            feature_name: Name of the feature
            metrics: ArtifactMetrics from detection
            output_path: Path to save figure (optional)
            show: Whether to show the plot
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, skipping visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Color map
        cmap = plt.cm.viridis
        
        # Top-left: 2D Heatmap
        grid, extent = self.create_2d_heatmap(coords, feature_values)
        im = axes[0, 0].imshow(grid, cmap=cmap, origin='lower', 
                               extent=extent, aspect='equal')
        axes[0, 0].set_title(f'{feature_name} - 2D Heatmap\nSeverity: {metrics.severity.upper()}',
                            fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('X (m)')
        axes[0, 0].set_ylabel('Y (m)')
        plt.colorbar(im, ax=axes[0, 0], label=feature_name)
        
        # Detect and overlay dash lines if requested
        if self.config.show_dash_lines:
            dash_y = self.detect_dash_lines(coords, feature_values, metrics)
            for y_pos in dash_y:
                axes[0, 0].axhline(y=y_pos, color='red', linestyle='--', 
                                  alpha=0.5, linewidth=1)
            if dash_y:
                axes[0, 0].text(0.02, 0.98, f'{len(dash_y)} dash lines detected',
                              transform=axes[0, 0].transAxes,
                              verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='red', alpha=0.3),
                              fontsize=9, color='white', fontweight='bold')
        
        # Top-right: Y-direction Profile (perpendicular to flight lines)
        y = coords[:, 1]
        y_bins = np.linspace(y.min(), y.max(), self.config.n_bins_y + 1)
        y_digitized = np.digitize(y, y_bins)
        
        y_means = []
        y_stds = []
        y_centers = []
        
        for i in range(1, self.config.n_bins_y + 1):
            mask = y_digitized == i
            if mask.sum() > 0:
                y_means.append(feature_values[mask].mean())
                y_stds.append(feature_values[mask].std())
                y_centers.append((y_bins[i-1] + y_bins[i]) / 2)
        
        y_means = np.array(y_means)
        y_stds = np.array(y_stds)
        y_centers = np.array(y_centers)
        
        axes[0, 1].plot(y_centers, y_means, 'b-', linewidth=2, label='Mean')
        axes[0, 1].fill_between(y_centers, y_means - y_stds, y_means + y_stds,
                               alpha=0.3, label='± 1 std')
        axes[0, 1].axhline(y_means.mean(), color='r', linestyle='--', 
                          alpha=0.5, label='Global Mean')
        axes[0, 1].set_title(f'{feature_name} - Y-Profile (Dash Line Detection)\nCV_Y = {metrics.cv_y:.3f}',
                            fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Y Position (m)')
        axes[0, 1].set_ylabel(f'{feature_name} Value')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Add severity indicator
        severity_color = {'low': 'green', 'medium': 'yellow', 'high': 'orange', 'severe': 'red'}
        axes[0, 1].text(0.02, 0.98, f'Severity: {metrics.severity.upper()}',
                       transform=axes[0, 1].transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor=severity_color[metrics.severity], alpha=0.7),
                       fontsize=10, fontweight='bold', color='white')
        
        # Bottom-left: X-direction Profile (parallel to flight lines)
        x = coords[:, 0]
        x_bins = np.linspace(x.min(), x.max(), self.config.n_bins_x + 1)
        x_digitized = np.digitize(x, x_bins)
        
        x_means = []
        x_stds = []
        x_centers = []
        
        for i in range(1, self.config.n_bins_x + 1):
            mask = x_digitized == i
            if mask.sum() > 0:
                x_means.append(feature_values[mask].mean())
                x_stds.append(feature_values[mask].std())
                x_centers.append((x_bins[i-1] + x_bins[i]) / 2)
        
        x_means = np.array(x_means)
        x_stds = np.array(x_stds)
        x_centers = np.array(x_centers)
        
        axes[1, 0].plot(x_centers, x_means, 'g-', linewidth=2, label='Mean')
        axes[1, 0].fill_between(x_centers, x_means - x_stds, x_means + x_stds,
                               alpha=0.3, label='± 1 std')
        axes[1, 0].axhline(x_means.mean(), color='r', linestyle='--', 
                          alpha=0.5, label='Global Mean')
        axes[1, 0].set_title(f'{feature_name} - X-Profile (Reference)\nCV_X = {metrics.cv_x:.3f}',
                            fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('X Position (m)')
        axes[1, 0].set_ylabel(f'{feature_name} Value')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Bottom-right: Statistics and Recommendations
        axes[1, 1].axis('off')
        
        stats_text = f"""
ARTIFACT DETECTION SUMMARY
{'='*40}

Feature: {feature_name}

METRICS:
  Mean Value:           {metrics.mean_value:.4f}
  Std Deviation:        {metrics.std_value:.4f}
  CV X-direction:       {metrics.cv_x:.4f}
  CV Y-direction:       {metrics.cv_y:.4f}
  Max CV:               {metrics.max_cv:.4f}
  
SPATIAL VARIANCE:
  X-direction:          {metrics.spatial_variance_x:.6f}
  Y-direction:          {metrics.spatial_variance_y:.6f}

CLASSIFICATION:
  Severity:             {metrics.severity.upper()}
  Has Artifacts:        {'YES' if metrics.has_artifacts else 'NO'}
  
RECOMMENDATION:
  Action:               {metrics.recommended_action.upper()}
  
{'='*40}

CV Interpretation:
  < 0.10: Low/None   (Good quality)
  < 0.20: Medium     (Acceptable)
  < 0.35: High       (Review suggested)
  ≥ 0.35: Severe     (Drop recommended)

Note: Dash lines appear as parallel
stripes in Y-direction heatmap
        """
        
        axes[1, 1].text(0.05, 0.95, stats_text.strip(),
                       transform=axes[1, 1].transAxes,
                       verticalalignment='top',
                       fontfamily='monospace',
                       fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle(f'Artifact Analysis: {feature_name}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.config.plot_dpi, bbox_inches='tight')
            logger.info(f"Saved artifact visualization to {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def analyze_file(
        self,
        laz_file: Path,
        features_to_check: Optional[List[str]] = None,
        visualize: bool = True,
        output_dir: Optional[Path] = None
    ) -> Dict[str, ArtifactMetrics]:
        """
        Analyze a LAZ file for artifacts in all specified features.
        
        Args:
            laz_file: Path to LAZ file
            features_to_check: List of feature names (None = use defaults)
            visualize: Whether to create visualizations
            output_dir: Directory to save visualizations (optional)
            
        Returns:
            Dictionary mapping feature names to ArtifactMetrics
        """
        try:
            import laspy
        except ImportError:
            raise ImportError("laspy required for LAZ file loading. Install with: pip install laspy")
        
        logger.info(f"Analyzing artifacts in {laz_file.name}")
        
        # Load LAZ file
        las = laspy.read(str(laz_file))
        
        # Get coordinates
        x = las.X * las.header.scales[0] + las.header.offsets[0]
        y = las.Y * las.header.scales[1] + las.header.offsets[1]
        z = las.Z * las.header.scales[2] + las.header.offsets[2]
        coords = np.vstack([x, y, z]).T
        
        # Get available features
        available_features = set(las.point_format.dimension_names) - {
            'X', 'Y', 'Z', 'intensity', 'return_number', 'number_of_returns',
            'classification', 'red', 'green', 'blue', 'nir', 'scan_angle_rank',
            'user_data', 'point_source_id', 'gps_time'
        }
        
        if features_to_check is None:
            features_to_check = [f for f in self.config.default_features 
                               if f in available_features]
        else:
            features_to_check = [f for f in features_to_check if f in available_features]
        
        logger.info(f"Found {len(features_to_check)} features to analyze: {features_to_check}")
        
        results = {}
        
        for feat_name in features_to_check:
            feat_values = np.array(las[feat_name])
            
            # Skip if all NaN or constant
            if np.all(~np.isfinite(feat_values)) or np.std(feat_values) < 1e-10:
                logger.warning(f"Skipping {feat_name}: constant or invalid values")
                continue
            
            # Detect artifacts
            metrics = self.detect_spatial_artifacts(coords, feat_values, feat_name)
            results[feat_name] = metrics
            
            logger.info(f"  {feat_name}: CV_Y={metrics.cv_y:.3f}, "
                       f"severity={metrics.severity}, action={metrics.recommended_action}")
            
            # Visualize if requested
            if visualize and output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                vis_path = output_dir / f"{laz_file.stem}_artifact_{feat_name}.png"
                self.visualize_artifacts(coords, feat_values, feat_name, 
                                       metrics, vis_path, show=False)
        
        return results
    
    def batch_analyze(
        self,
        laz_files: List[Path],
        features_to_check: Optional[List[str]] = None,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Dict[str, ArtifactMetrics]]:
        """
        Batch analyze multiple LAZ files.
        
        Args:
            laz_files: List of LAZ file paths
            features_to_check: List of feature names (None = use defaults)
            output_dir: Directory to save results (optional)
            
        Returns:
            Dictionary mapping file paths to feature metrics
        """
        all_results = {}
        
        logger.info(f"Batch analyzing {len(laz_files)} files")
        
        for laz_file in laz_files:
            try:
                results = self.analyze_file(
                    laz_file,
                    features_to_check=features_to_check,
                    visualize=True if output_dir else False,
                    output_dir=output_dir
                )
                all_results[str(laz_file)] = results
            except Exception as e:
                logger.error(f"Failed to analyze {laz_file.name}: {e}")
                continue
        
        # Generate summary report
        if output_dir:
            self._generate_batch_report(all_results, output_dir)
        
        return all_results
    
    def _generate_batch_report(
        self,
        results: Dict[str, Dict[str, ArtifactMetrics]],
        output_dir: Path
    ) -> None:
        """Generate CSV report from batch analysis."""
        import csv
        
        report_path = output_dir / "artifact_analysis_report.csv"
        
        with open(report_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'File', 'Feature', 'CV_X', 'CV_Y', 'Max_CV',
                'Mean', 'Std', 'Severity', 'Has_Artifacts', 'Recommended_Action'
            ])
            
            for file_path, file_results in results.items():
                filename = Path(file_path).name
                for metrics in file_results.values():
                    writer.writerow([
                        filename,
                        metrics.feature_name,
                        f'{metrics.cv_x:.4f}',
                        f'{metrics.cv_y:.4f}',
                        f'{metrics.max_cv:.4f}',
                        f'{metrics.mean_value:.4f}',
                        f'{metrics.std_value:.4f}',
                        metrics.severity,
                        'Yes' if metrics.has_artifacts else 'No',
                        metrics.recommended_action
                    ])
        
        logger.info(f"Saved batch report to {report_path}")
    
    def get_fields_to_drop(
        self,
        results: Dict[str, ArtifactMetrics],
        threshold: Optional[float] = None
    ) -> List[str]:
        """
        Get list of feature names that should be dropped due to artifacts.
        
        Args:
            results: Dictionary mapping feature names to ArtifactMetrics
            threshold: CV threshold for dropping (None = use config default)
            
        Returns:
            List of feature names to drop
        """
        if threshold is None:
            threshold = self.config.auto_drop_threshold
        
        drop_list = []
        for feat_name, metrics in results.items():
            if metrics.max_cv >= threshold:
                drop_list.append(feat_name)
                logger.info(f"Marked {feat_name} for dropping (CV={metrics.max_cv:.3f} >= {threshold:.3f})")
        
        return drop_list
    
    def filter_clean_features(
        self,
        las: Any,
        results: Dict[str, ArtifactMetrics],
        threshold: Optional[float] = None
    ) -> Tuple[Any, List[str]]:
        """
        Remove artifact-contaminated features from LAZ object.
        
        Args:
            las: laspy.LasData object
            results: Dictionary mapping feature names to ArtifactMetrics
            threshold: CV threshold for dropping (None = use config default)
            
        Returns:
            filtered_las: LAZ object with artifact features removed
            dropped_features: List of dropped feature names
        """
        drop_list = self.get_fields_to_drop(results, threshold)
        
        if not drop_list:
            logger.info("No features need to be dropped")
            return las, []
        
        logger.info(f"Dropping {len(drop_list)} features with severe artifacts: {drop_list}")
        
        # Note: laspy doesn't support in-place field removal easily
        # This would require creating a new LAS file with selected dimensions
        # For now, return the list of fields to drop for manual handling
        
        return las, drop_list


def main():
    """Command-line interface for artifact detection."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Detect and visualize artifacts in LiDAR point cloud features'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input LAZ file or directory'
    )
    parser.add_argument(
        '--features',
        type=str,
        default=None,
        help='Comma-separated list of features to check (default: all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='artifact_analysis',
        help='Output directory for visualizations and reports'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.40,
        help='CV threshold for recommending field dropping (default: 0.40)'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all LAZ files in directory'
    )
    
    args = parser.parse_args()
    
    # Setup config
    config = ArtifactDetectorConfig()
    config.auto_drop_threshold = args.threshold
    
    detector = ArtifactDetector(config)
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    features_to_check = [f.strip() for f in args.features.split(',')] if args.features else None
    
    if args.batch or input_path.is_dir():
        # Batch mode
        laz_files = list(input_path.glob('*.laz')) + list(input_path.glob('*.LAZ'))
        results = detector.batch_analyze(laz_files, features_to_check, output_dir)
        
        print(f"\n{'='*60}")
        print(f"BATCH ANALYSIS COMPLETE: {len(results)} files processed")
        print(f"{'='*60}\n")
        
    else:
        # Single file mode
        results = detector.analyze_file(input_path, features_to_check, 
                                       visualize=True, output_dir=output_dir)
        
        print(f"\n{'='*60}")
        print(f"ARTIFACT ANALYSIS: {input_path.name}")
        print(f"{'='*60}\n")
        
        for feat_name, metrics in results.items():
            print(f"{feat_name:20s} | CV_Y: {metrics.cv_y:6.3f} | "
                  f"Severity: {metrics.severity:8s} | Action: {metrics.recommended_action.upper()}")
        
        drop_list = detector.get_fields_to_drop(results)
        if drop_list:
            print(f"\n⚠️  RECOMMENDED TO DROP: {', '.join(drop_list)}")
        else:
            print(f"\n✅ All features are acceptable (no dropping needed)")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
