#!/usr/bin/env python3
"""
Visualize spatial distribution of features to detect artifacts.

This script creates heatmaps showing feature values across the patch,
making dash line artifacts clearly visible.

Usage:
    python scripts/visualize_artifacts.py \
        --input /mnt/c/Users/Simon/ign/versailles/output/LHD_FXX_0635_6857_PTS_C_LAMB93_IGN69_hybrid_patch_0300.laz \
        --features planarity,roof_score,linearity \
        --output artifact_visualization.png
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_laz_file(filepath: str):
    """Load LAZ file and return coordinates and features."""
    try:
        import laspy
    except ImportError:
        logger.error("laspy not installed. Run: pip install laspy")
        return None, None, None
    
    las = laspy.read(filepath)
    
    # Get coordinates
    x = las.X * las.header.scales[0] + las.header.offsets[0]
    y = las.Y * las.header.scales[1] + las.header.offsets[1]
    z = las.Z * las.header.scales[2] + las.header.offsets[2]
    coords = np.vstack([x, y, z]).T
    
    # Get all available features
    features = {}
    for dim_name in las.point_format.dimension_names:
        if dim_name not in ['X', 'Y', 'Z', 'intensity', 'return_number',
                           'number_of_returns', 'classification', 'red', 
                           'green', 'blue', 'nir']:
            try:
                features[dim_name] = np.array(las[dim_name])
            except:
                pass
    
    return coords, features, las


def create_2d_grid(coords: np.ndarray, 
                   feature_values: np.ndarray,
                   grid_size: int = 50) -> tuple:
    """
    Create a 2D grid representation of feature values.
    
    Args:
        coords: [N, 3] XYZ coordinates
        feature_values: [N] feature values
        grid_size: Number of bins in each direction
    
    Returns:
        grid: 2D array of averaged feature values
        extent: (x_min, x_max, y_min, y_max) for plotting
    """
    x, y = coords[:, 0], coords[:, 1]
    
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Create bins
    x_bins = np.linspace(x_min, x_max, grid_size + 1)
    y_bins = np.linspace(y_min, y_max, grid_size + 1)
    
    # Digitize points
    x_idx = np.digitize(x, x_bins) - 1
    y_idx = np.digitize(y, y_bins) - 1
    
    # Clip to valid range
    x_idx = np.clip(x_idx, 0, grid_size - 1)
    y_idx = np.clip(y_idx, 0, grid_size - 1)
    
    # Create grid
    grid = np.full((grid_size, grid_size), np.nan)
    counts = np.zeros((grid_size, grid_size))
    
    # Accumulate values
    for i in range(len(feature_values)):
        grid[y_idx[i], x_idx[i]] = np.nansum([grid[y_idx[i], x_idx[i]], 
                                               feature_values[i]])
        counts[y_idx[i], x_idx[i]] += 1
    
    # Average
    grid = np.where(counts > 0, grid / counts, np.nan)
    
    extent = (x_min, x_max, y_min, y_max)
    
    return grid, extent


def visualize_feature_artifacts(filepath: str,
                                feature_names: list,
                                output_path: str = None,
                                grid_size: int = 50):
    """
    Create visualization showing spatial distribution of features.
    
    Args:
        filepath: Path to LAZ file
        feature_names: List of feature names to visualize
        output_path: Path to save figure
        grid_size: Grid resolution
    """
    logger.info(f"Loading {filepath}")
    coords, features, las = load_laz_file(filepath)
    
    if coords is None:
        return
    
    logger.info(f"Loaded {len(coords)} points")
    logger.info(f"Available features: {list(features.keys())}")
    
    # Filter to requested features
    target_features = [f for f in feature_names if f in features]
    
    if not target_features:
        logger.error(f"None of the requested features found: {feature_names}")
        return
    
    n_features = len(target_features)
    
    # Create figure
    fig, axes = plt.subplots(2, n_features, figsize=(6*n_features, 10))
    if n_features == 1:
        axes = axes.reshape(2, 1)
    
    # Color map for artifacts
    cmap = plt.cm.viridis
    
    for col, feat_name in enumerate(target_features):
        feat_values = features[feat_name]
        
        logger.info(f"Processing {feat_name}: min={feat_values.min():.3f}, "
                   f"max={feat_values.max():.3f}, mean={feat_values.mean():.3f}")
        
        # Top row: 2D grid heatmap
        grid, extent = create_2d_grid(coords, feat_values, grid_size)
        
        im1 = axes[0, col].imshow(grid, cmap=cmap, origin='lower', 
                                   extent=extent, aspect='equal')
        axes[0, col].set_title(f'{feat_name}\n2D Heatmap', fontsize=12, fontweight='bold')
        axes[0, col].set_xlabel('X (m)')
        axes[0, col].set_ylabel('Y (m)')
        plt.colorbar(im1, ax=axes[0, col], label=feat_name)
        
        # Add grid lines to show striping
        axes[0, col].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Bottom row: Y-direction profile (averaged across X)
        # This should clearly show the dash line artifacts
        n_y_bins = 50
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        y_bins = np.linspace(y_min, y_max, n_y_bins + 1)
        y_digitized = np.digitize(coords[:, 1], y_bins)
        
        y_means = []
        y_stds = []
        y_centers = []
        
        for i in range(1, n_y_bins + 1):
            mask = y_digitized == i
            if mask.sum() > 0:
                y_means.append(feat_values[mask].mean())
                y_stds.append(feat_values[mask].std())
                y_centers.append((y_bins[i-1] + y_bins[i]) / 2)
        
        y_means = np.array(y_means)
        y_stds = np.array(y_stds)
        y_centers = np.array(y_centers)
        
        # Plot Y-profile
        axes[1, col].plot(y_centers, y_means, 'b-', linewidth=2, label='Mean')
        axes[1, col].fill_between(y_centers, y_means - y_stds, y_means + y_stds,
                                   alpha=0.3, label='± 1 std')
        axes[1, col].set_title(f'{feat_name}\nY-direction Profile (Striping Detection)', 
                              fontsize=12, fontweight='bold')
        axes[1, col].set_xlabel('Y Position (m)')
        axes[1, col].set_ylabel(f'{feat_name} Value')
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].legend()
        
        # Calculate and display CV
        cv = y_stds.mean() / y_means.mean() if y_means.mean() > 0 else 0
        axes[1, col].text(0.02, 0.98, f'CV = {cv:.3f}',
                         transform=axes[1, col].transAxes,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                         fontsize=10, fontweight='bold')
        
        # Mark high variation regions
        if cv > 0.15:
            axes[1, col].axhline(y_means.mean(), color='r', linestyle='--', 
                                alpha=0.5, label='Mean')
    
    plt.suptitle(f'Artifact Detection: {filepath.split("/")[-1]}',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize spatial artifacts in LAZ files'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input LAZ file'
    )
    parser.add_argument(
        '--features',
        type=str,
        default='planarity,roof_score,linearity',
        help='Comma-separated list of features to visualize'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output PNG file (default: show plot)'
    )
    parser.add_argument(
        '--grid_size',
        type=int,
        default=50,
        help='Grid resolution for heatmap'
    )
    
    args = parser.parse_args()
    
    feature_list = [f.strip() for f in args.features.split(',')]
    
    visualize_feature_artifacts(
        args.input,
        feature_list,
        args.output,
        args.grid_size
    )


if __name__ == "__main__":
    main()
