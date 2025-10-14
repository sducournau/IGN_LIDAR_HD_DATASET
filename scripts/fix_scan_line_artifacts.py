#!/usr/bin/env python3
"""
Fix scan line artifacts in eigenvalue-based features using spatial smoothing.

This script addresses the dash-line patterns found in planarity, linearity,
and roof_score features caused by LiDAR scan line geometry.

Method: 2D spatial median filtering perpendicular to scan lines

Usage:
    python scripts/fix_scan_line_artifacts.py \
        --input /mnt/c/Users/Simon/ign/versailles/output/*.laz \
        --output /mnt/c/Users/Simon/ign/versailles/output_fixed/ \
        --window_size 5 \
        --features planarity linearity roof_score
"""

import numpy as np
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from scipy.ndimage import median_filter
from scipy.interpolate import griddata

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def detect_scan_direction(coords: np.ndarray) -> Tuple[float, str]:
    """
    Detect the dominant scan line direction from point distribution.
    
    Args:
        coords: [N, 3] XYZ coordinates
    
    Returns:
        angle: Scan line angle in radians
        direction: 'x' or 'y' (dominant direction)
    """
    # Sample points for efficiency
    n_sample = min(10000, len(coords))
    indices = np.random.choice(len(coords), n_sample, replace=False)
    sample_coords = coords[indices]
    
    # Compute spacing in X and Y directions
    x_sorted = np.sort(sample_coords[:, 0])
    y_sorted = np.sort(sample_coords[:, 1])
    
    # Estimate point spacing (median of differences)
    x_spacing = np.median(np.diff(x_sorted))
    y_spacing = np.median(np.diff(y_sorted))
    
    # Artifacts are typically perpendicular to flight direction
    # If variation is higher in Y, scan lines are along X (and vice versa)
    if y_spacing > x_spacing:
        return 0.0, 'y'  # Scan lines along X, artifacts in Y
    else:
        return np.pi/2, 'x'  # Scan lines along Y, artifacts in X


def spatial_median_filter_2d(
    feature_values: np.ndarray,
    coords: np.ndarray,
    window_size: int = 5,
    grid_resolution: float = 1.0,
    direction: str = 'auto'
) -> np.ndarray:
    """
    Apply 2D spatial median filter to remove stripe artifacts.
    
    Args:
        feature_values: [N] feature values
        coords: [N, 3] XYZ coordinates
        window_size: Size of median filter window (larger = more smoothing)
        grid_resolution: Grid cell size in meters
        direction: 'x', 'y', or 'auto' (detect scan direction)
    
    Returns:
        smoothed_values: [N] smoothed feature values
    """
    if direction == 'auto':
        _, direction = detect_scan_direction(coords)
        logger.info(f"Detected scan line artifacts in {direction.upper()} direction")
    
    # Create 2D grid
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Grid dimensions
    nx = int((x_max - x_min) / grid_resolution) + 1
    ny = int((y_max - y_min) / grid_resolution) + 1
    
    # Create regular grid
    x_grid = np.linspace(x_min, x_max, nx)
    y_grid = np.linspace(y_min, y_max, ny)
    xx, yy = np.meshgrid(x_grid, y_grid)
    
    # Interpolate feature values to grid
    grid_values = griddata(
        (x_coords, y_coords),
        feature_values,
        (xx, yy),
        method='linear',
        fill_value=np.nan
    )
    
    # Apply median filter
    # Stronger filtering in artifact direction, weaker in other direction
    if direction == 'y':
        # Artifacts in Y, preserve X variation
        footprint = np.ones((window_size * 2, window_size))  # Taller window
    else:
        # Artifacts in X, preserve Y variation
        footprint = np.ones((window_size, window_size * 2))  # Wider window
    
    # Handle NaN values
    valid_mask = ~np.isnan(grid_values)
    
    if valid_mask.sum() < 100:
        logger.warning("Too few valid grid points, skipping smoothing")
        return feature_values
    
    # Apply median filter only to valid regions
    # Use nanmedian for regions with NaN
    from scipy.ndimage import generic_filter
    
    def nanmedian_filter(values):
        valid = ~np.isnan(values)
        if valid.sum() > 0:
            return np.median(values[valid])
        return np.nan
    
    # For efficiency, use standard median_filter with NaN handling
    grid_filled = grid_values.copy()
    grid_filled[np.isnan(grid_filled)] = np.nanmean(grid_values)
    
    smoothed_grid = median_filter(grid_filled, footprint=footprint, mode='nearest')
    
    # Restore NaN where original was NaN
    smoothed_grid[~valid_mask] = np.nan
    
    # Interpolate back to original points
    smoothed_values = griddata(
        (xx.ravel(), yy.ravel()),
        smoothed_grid.ravel(),
        (x_coords, y_coords),
        method='linear',
        fill_value=np.nan
    )
    
    # Fill any remaining NaN with original values
    nan_mask = np.isnan(smoothed_values)
    if nan_mask.sum() > 0:
        smoothed_values[nan_mask] = feature_values[nan_mask]
        logger.info(f"Filled {nan_mask.sum()} NaN values with original")
    
    return smoothed_values


def fix_scan_line_artifacts(
    las_data,
    target_features: List[str] = ['planarity', 'linearity', 'roof_score'],
    window_size: int = 5,
    grid_resolution: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Fix scan line artifacts in specified features.
    
    Args:
        las_data: laspy LAS data object
        target_features: List of feature names to fix
        window_size: Median filter window size
        grid_resolution: Grid cell size for interpolation
    
    Returns:
        fixed_features: Dictionary of fixed feature arrays
    """
    # Get coordinates
    x = las_data.X * las_data.header.scales[0] + las_data.header.offsets[0]
    y = las_data.Y * las_data.header.scales[1] + las_data.header.offsets[1]
    z = las_data.Z * las_data.header.scales[2] + las_data.header.offsets[2]
    coords = np.column_stack([x, y, z])
    
    fixed_features = {}
    
    # Detect scan direction once
    _, scan_dir = detect_scan_direction(coords)
    logger.info(f"Scan line artifacts detected in {scan_dir.upper()} direction")
    
    for feat_name in target_features:
        if feat_name not in las_data.point_format.dimension_names:
            logger.warning(f"Feature '{feat_name}' not found in LAZ file, skipping")
            continue
        
        logger.info(f"Processing {feat_name}...")
        
        # Get original values
        original = las_data[feat_name]
        
        # Compute statistics before
        orig_mean = original.mean()
        orig_std = original.std()
        
        # Apply spatial median filter
        smoothed = spatial_median_filter_2d(
            original,
            coords,
            window_size=window_size,
            grid_resolution=grid_resolution,
            direction=scan_dir
        )
        
        # Compute statistics after
        smooth_mean = smoothed.mean()
        smooth_std = smoothed.std()
        
        # Compute artifact reduction
        # Measure Y-direction variation before/after
        n_bins = 20
        y_bins = np.linspace(y.min(), y.max(), n_bins + 1)
        y_digitized = np.digitize(y, y_bins)
        
        orig_cv_y = compute_cv_y(original, y_digitized, n_bins)
        smooth_cv_y = compute_cv_y(smoothed, y_digitized, n_bins)
        
        reduction = (orig_cv_y - smooth_cv_y) / orig_cv_y * 100 if orig_cv_y > 0 else 0
        
        logger.info(f"  Original: mean={orig_mean:.4f}, std={orig_std:.4f}, CV_y={orig_cv_y:.4f}")
        logger.info(f"  Smoothed: mean={smooth_mean:.4f}, std={smooth_std:.4f}, CV_y={smooth_cv_y:.4f}")
        logger.info(f"  Artifact reduction: {reduction:.1f}%")
        
        fixed_features[feat_name] = smoothed
    
    return fixed_features


def compute_cv_y(values: np.ndarray, y_digitized: np.ndarray, n_bins: int) -> float:
    """Compute coefficient of variation in Y direction."""
    y_means = []
    for i in range(1, n_bins + 1):
        mask = y_digitized == i
        if mask.sum() > 10:
            y_means.append(values[mask].mean())
    
    if len(y_means) == 0:
        return 0.0
    
    y_means = np.array(y_means)
    return np.std(y_means) / np.mean(y_means) if np.mean(y_means) > 0 else 0.0


def process_laz_file(
    input_path: str,
    output_path: str,
    target_features: List[str],
    window_size: int,
    grid_resolution: float
) -> Dict[str, float]:
    """
    Process a single LAZ file.
    
    Returns:
        stats: Processing statistics
    """
    try:
        import laspy
    except ImportError:
        logger.error("laspy not installed. Run: pip install laspy")
        return {}
    
    logger.info(f"\nProcessing: {os.path.basename(input_path)}")
    
    # Read LAZ file
    las = laspy.read(input_path)
    
    # Fix artifacts
    fixed_features = fix_scan_line_artifacts(
        las,
        target_features=target_features,
        window_size=window_size,
        grid_resolution=grid_resolution
    )
    
    if len(fixed_features) == 0:
        logger.warning("No features were fixed, copying original file")
        import shutil
        shutil.copy2(input_path, output_path)
        return {'status': 'skipped'}
    
    # Update LAS data with fixed features
    for feat_name, fixed_values in fixed_features.items():
        las[feat_name] = fixed_values
    
    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    las.write(output_path)
    
    logger.info(f"Saved to: {output_path}")
    
    return {
        'status': 'success',
        'features_fixed': len(fixed_features),
        'n_points': len(las.points)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Fix scan line artifacts in LAZ files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fix single file
    python scripts/fix_scan_line_artifacts.py \\
        --input data/output/patch_0300.laz \\
        --output data/output_fixed/patch_0300.laz
    
    # Fix all files in directory
    python scripts/fix_scan_line_artifacts.py \\
        --input /mnt/c/Users/Simon/ign/versailles/output/*.laz \\
        --output /mnt/c/Users/Simon/ign/versailles/output_fixed/ \\
        --window_size 7
        """
    )
    
    parser.add_argument('--input', required=True, nargs='+',
                       help='Input LAZ file(s) (supports wildcards)')
    parser.add_argument('--output', required=True,
                       help='Output directory or file path')
    parser.add_argument('--features', nargs='+',
                       default=['planarity', 'linearity', 'roof_score'],
                       help='Features to fix (default: planarity linearity roof_score)')
    parser.add_argument('--window_size', type=int, default=5,
                       help='Median filter window size (default: 5, larger = more smoothing)')
    parser.add_argument('--grid_resolution', type=float, default=1.0,
                       help='Grid cell size in meters (default: 1.0)')
    
    args = parser.parse_args()
    
    # Expand wildcards
    from glob import glob
    input_files = []
    for pattern in args.input:
        input_files.extend(glob(pattern))
    
    if len(input_files) == 0:
        logger.error("No input files found")
        return
    
    logger.info(f"Found {len(input_files)} files to process")
    
    # Determine output paths
    output_is_dir = os.path.isdir(args.output) or args.output.endswith('/')
    
    if output_is_dir:
        os.makedirs(args.output, exist_ok=True)
    
    # Process each file
    stats_list = []
    
    for input_path in input_files:
        if output_is_dir:
            output_path = os.path.join(args.output, os.path.basename(input_path))
        else:
            if len(input_files) > 1:
                logger.error("Multiple input files require output to be a directory")
                return
            output_path = args.output
        
        try:
            stats = process_laz_file(
                input_path,
                output_path,
                args.features,
                args.window_size,
                args.grid_resolution
            )
            stats['file'] = os.path.basename(input_path)
            stats_list.append(stats)
            
        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*80)
    
    success_count = sum(1 for s in stats_list if s.get('status') == 'success')
    logger.info(f"Successfully processed: {success_count}/{len(input_files)} files")
    
    if success_count > 0:
        logger.info(f"\nFixed features: {args.features}")
        logger.info(f"Window size: {args.window_size}")
        logger.info(f"Grid resolution: {args.grid_resolution}m")
        logger.info(f"\nOutput directory: {args.output}")
    
    logger.info("="*80)


if __name__ == "__main__":
    main()
