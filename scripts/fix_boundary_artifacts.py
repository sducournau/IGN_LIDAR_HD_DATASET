#!/usr/bin/env python3
"""
Post-process existing LAZ files to remove/flag boundary artifacts.

This script addresses the dash line artifacts found in patch 0300 and others
by identifying points near patch boundaries and either:
1. Flagging them for exclusion during training
2. Smoothing their feature values
3. Removing them entirely

Usage:
    python scripts/fix_boundary_artifacts.py \
        --input /mnt/c/Users/Simon/ign/versailles/output/*.laz \
        --output /mnt/c/Users/Simon/ign/versailles/output_fixed/ \
        --method flag \
        --boundary_width 2.5
"""

import numpy as np
import argparse
import os
from pathlib import Path
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def detect_patch_boundaries(coords: np.ndarray, 
                           boundary_width: float = 2.5) -> np.ndarray:
    """
    Detect points near patch boundaries.
    
    Args:
        coords: [N, 3] XYZ coordinates
        boundary_width: Distance from edge to flag as boundary (meters)
    
    Returns:
        boundary_mask: [N] boolean mask, True for boundary points
    """
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    
    # Points near any edge
    boundary_mask = (
        (coords[:, 0] < x_min + boundary_width) |
        (coords[:, 0] > x_max - boundary_width) |
        (coords[:, 1] < y_min + boundary_width) |
        (coords[:, 1] > y_max - boundary_width)
    )
    
    return boundary_mask


def smooth_features_at_boundaries(features: Dict[str, np.ndarray],
                                  coords: np.ndarray,
                                  boundary_mask: np.ndarray,
                                  sigma: float = 1.0) -> Dict[str, np.ndarray]:
    """
    Smooth feature values at boundary points using Gaussian kernel.
    
    Args:
        features: Dictionary of feature arrays
        coords: [N, 3] XYZ coordinates
        boundary_mask: [N] boolean mask for boundary points
        sigma: Gaussian kernel width
    
    Returns:
        smoothed_features: Dictionary with smoothed boundary features
    """
    from scipy.spatial import cKDTree
    from scipy.ndimage import gaussian_filter1d
    
    target_features = ['planarity', 'linearity', 'roof_score', 'wall_score']
    smoothed = features.copy()
    
    if boundary_mask.sum() == 0:
        return smoothed
    
    # Build KD-tree for interior points
    interior_mask = ~boundary_mask
    if interior_mask.sum() < 10:
        logger.warning("Too few interior points for smoothing")
        return smoothed
    
    tree = cKDTree(coords[interior_mask, :2])  # XY only
    
    # For each boundary point, interpolate from nearby interior points
    boundary_coords = coords[boundary_mask, :2]
    
    for feat_name in target_features:
        if feat_name not in features:
            continue
        
        feat = features[feat_name].copy()
        boundary_indices = np.where(boundary_mask)[0]
        
        # Query 10 nearest interior points
        distances, indices = tree.query(boundary_coords, k=10)
        
        # Weight by inverse distance
        weights = 1.0 / (distances + 1e-6)
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        # Get interior feature values
        interior_feat = feat[interior_mask]
        neighbor_values = interior_feat[indices]
        
        # Weighted average
        smoothed_values = (neighbor_values * weights).sum(axis=1)
        
        # Replace boundary values
        feat[boundary_indices] = smoothed_values
        smoothed[feat_name] = feat
        
        logger.info(f"Smoothed {feat_name} for {boundary_mask.sum()} boundary points")
    
    return smoothed


def process_laz_file(input_path: str,
                    output_path: str,
                    method: str = 'flag',
                    boundary_width: float = 2.5,
                    sigma: float = 1.0) -> Dict[str, any]:
    """
    Process a single LAZ file to handle boundary artifacts.
    
    Args:
        input_path: Path to input LAZ file
        output_path: Path to output LAZ file
        method: 'flag', 'smooth', or 'remove'
        boundary_width: Distance from edge for boundary detection
        sigma: Smoothing parameter
    
    Returns:
        stats: Dictionary with processing statistics
    """
    try:
        import laspy
    except ImportError:
        logger.error("laspy not installed. Run: pip install laspy")
        return {}
    
    logger.info(f"Processing {input_path}")
    
    # Read file
    las = laspy.read(input_path)
    n_total = len(las.points)
    
    # Get coordinates
    coords = np.vstack([las.x, las.y, las.z]).T
    
    # Detect boundary points
    boundary_mask = detect_patch_boundaries(coords, boundary_width)
    n_boundary = boundary_mask.sum()
    pct_boundary = 100 * n_boundary / n_total
    
    logger.info(f"Found {n_boundary} boundary points ({pct_boundary:.1f}%)")
    
    stats = {
        'file': os.path.basename(input_path),
        'total_points': n_total,
        'boundary_points': n_boundary,
        'boundary_percentage': pct_boundary,
        'method': method
    }
    
    if method == 'flag':
        # Add boundary flag to classification or user_data
        las.user_data = las.user_data.copy()
        las.user_data[boundary_mask] = 255  # Flag boundary points
        logger.info(f"Flagged {n_boundary} boundary points in user_data")
        
    elif method == 'smooth':
        # Smooth features at boundaries
        features = {}
        target_features = ['planarity', 'linearity', 'roof_score', 'wall_score']
        
        for feat_name in target_features:
            if feat_name in las.point_format.dimension_names:
                features[feat_name] = las[feat_name]
        
        if features:
            smoothed = smooth_features_at_boundaries(
                features, coords, boundary_mask, sigma
            )
            
            # Write back smoothed features
            for feat_name, feat_values in smoothed.items():
                setattr(las, feat_name, feat_values)
        else:
            logger.warning("No target features found for smoothing")
            
    elif method == 'remove':
        # Remove boundary points
        keep_mask = ~boundary_mask
        
        # Create new LAS with filtered points
        las = las[keep_mask]
        logger.info(f"Removed {n_boundary} boundary points, kept {keep_mask.sum()}")
        stats['points_kept'] = keep_mask.sum()
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    las.write(output_path)
    logger.info(f"Wrote output to {output_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Fix boundary artifacts in LAZ files'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input LAZ file(s) pattern (e.g., *.laz)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['flag', 'smooth', 'remove'],
        default='flag',
        help='Method to handle boundary points'
    )
    parser.add_argument(
        '--boundary_width',
        type=float,
        default=2.5,
        help='Distance from edge to consider as boundary (meters)'
    )
    parser.add_argument(
        '--sigma',
        type=float,
        default=1.0,
        help='Smoothing kernel width (for smooth method)'
    )
    
    args = parser.parse_args()
    
    # Find input files
    from glob import glob
    input_files = glob(args.input)
    
    if not input_files:
        logger.error(f"No files found matching: {args.input}")
        return
    
    logger.info(f"Found {len(input_files)} files to process")
    
    # Process each file
    all_stats = []
    for input_path in input_files:
        output_filename = os.path.basename(input_path)
        output_path = os.path.join(args.output, output_filename)
        
        try:
            stats = process_laz_file(
                input_path,
                output_path,
                args.method,
                args.boundary_width,
                args.sigma
            )
            all_stats.append(stats)
        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    
    if all_stats:
        total_points = sum(s['total_points'] for s in all_stats)
        total_boundary = sum(s['boundary_points'] for s in all_stats)
        avg_pct = 100 * total_boundary / total_points if total_points > 0 else 0
        
        print(f"\nProcessed {len(all_stats)} files")
        print(f"Total points: {total_points:,}")
        print(f"Boundary points: {total_boundary:,} ({avg_pct:.1f}%)")
        print(f"Method: {args.method}")
        print(f"Boundary width: {args.boundary_width}m")
        
        if args.method == 'remove':
            total_kept = sum(s.get('points_kept', 0) for s in all_stats)
            print(f"Points kept: {total_kept:,}")
    
    print(f"\nOutput directory: {args.output}")
    print("="*80)


if __name__ == "__main__":
    main()
