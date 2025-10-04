#!/usr/bin/env python3
"""
Test and compare different radius values for geometric features.

This script helps visualize the impact of radius parameter on geometric
feature computation, particularly for detecting/eliminating artifacts.

Usage:
    python test_radius_comparison.py <input_laz> --radii 0.5 1.0 1.5 2.0
    python test_radius_comparison.py <input_laz> --auto  # Compare auto vs fixed

Results are saved as separate LAZ files and CSV statistics.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import laspy
from typing import List, Tuple, Dict
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ign_lidar.features import (
    compute_normals,
    estimate_optimal_radius_for_features,
    extract_geometric_features
)


def compute_with_radius(points: np.ndarray, 
                       classification: np.ndarray,
                       radius: float = None) -> Dict[str, np.ndarray]:
    """Compute features with specified radius."""
    print(f"\n{'='*60}")
    if radius is None:
        estimated_radius = estimate_optimal_radius_for_features(points, 'geometric')
        print(f"AUTO-ESTIMATED RADIUS: {estimated_radius:.3f}m")
        radius = estimated_radius
    else:
        print(f"FIXED RADIUS: {radius:.3f}m")
    
    # Compute normals (using smaller radius for speed)
    print("Computing normals...")
    normals = compute_normals(points, k=20)
    
    # Compute geometric features with radius
    print(f"Computing geometric features (radius={radius:.3f}m)...")
    features = extract_geometric_features(points, normals, radius=radius)
    
    return features, radius


def analyze_features(features: Dict[str, np.ndarray], 
                     classification: np.ndarray,
                     name: str) -> Dict:
    """Analyze feature statistics."""
    stats = {}
    
    print(f"\n{name} - Feature Statistics:")
    print(f"{'Feature':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 60)
    
    for feat_name, values in features.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        print(f"{feat_name:<15} {mean_val:>9.4f} {std_val:>9.4f} "
              f"{min_val:>9.4f} {max_val:>9.4f}")
        
        stats[feat_name] = {
            'mean': float(mean_val),
            'std': float(std_val),
            'min': float(min_val),
            'max': float(max_val)
        }
    
    # Analyze by classification
    if np.any(classification == 6):  # Buildings
        building_mask = classification == 6
        print(f"\nBuilding points (class 6): {np.sum(building_mask):,} points")
        
        for feat_name, values in features.items():
            building_values = values[building_mask]
            if len(building_values) > 0:
                mean_val = np.mean(building_values)
                std_val = np.std(building_values)
                stats[f'{feat_name}_building'] = {
                    'mean': float(mean_val),
                    'std': float(std_val)
                }
    
    return stats


def save_enriched_laz(input_path: Path, 
                      output_path: Path,
                      features: Dict[str, np.ndarray],
                      normals: np.ndarray,
                      radius: float):
    """Save enriched LAZ file."""
    print(f"\nSaving to: {output_path}")
    
    # Read original
    las = laspy.read(input_path)
    
    # Create output LAZ
    las_out = laspy.LasData(las.header)
    las_out.points = las.points
    
    # Add normals
    for i, dim in enumerate(['normal_x', 'normal_y', 'normal_z']):
        las_out.add_extra_dim(laspy.ExtraBytesParams(name=dim, type=np.float32))
        setattr(las_out, dim, normals[:, i])
    
    # Add geometric features
    for key, values in features.items():
        las_out.add_extra_dim(laspy.ExtraBytesParams(name=key, type=np.float32))
        setattr(las_out, key, values)
    
    # Add radius as metadata
    las_out.add_extra_dim(laspy.ExtraBytesParams(name='search_radius', type=np.float32))
    las_out.search_radius = np.full(len(las_out.points), radius, dtype=np.float32)
    
    # Save
    las_out.write(output_path, do_compress=True)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Saved ({file_size_mb:.1f} MB)")


def compare_radii(input_path: Path, radii: List[float], output_dir: Path):
    """Compare different radius values."""
    print(f"\n{'='*70}")
    print(f"RADIUS COMPARISON TEST")
    print(f"{'='*70}")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Radii to test: {radii}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load point cloud
    print(f"\nLoading point cloud...")
    las = laspy.read(input_path)
    points = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
    classification = np.array(las.classification, dtype=np.uint8)
    
    print(f"✓ Loaded {len(points):,} points")
    
    # Compute normals once (reused for all radii)
    print(f"\nComputing normals (k=20)...")
    normals = compute_normals(points, k=20)
    print(f"✓ Normals computed")
    
    # Test each radius
    all_stats = {}
    
    for radius in radii:
        radius_label = f"r{radius:.1f}".replace('.', 'p')
        
        # Compute features
        features, actual_radius = compute_with_radius(points, classification, radius)
        
        # Analyze
        stats = analyze_features(features, classification, f"Radius={actual_radius:.3f}m")
        all_stats[radius_label] = {
            'radius': actual_radius,
            'stats': stats
        }
        
        # Save LAZ
        output_path = output_dir / f"{input_path.stem}_{radius_label}.laz"
        save_enriched_laz(input_path, output_path, features, normals, actual_radius)
    
    # Save comparison JSON
    json_path = output_dir / f"{input_path.stem}_comparison.json"
    with open(json_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n✓ Comparison statistics saved to: {json_path}")
    
    # Print comparison table
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n{'Feature':<15} ", end='')
    for radius in radii:
        radius_label = f"r{radius:.1f}".replace('.', 'p')
        print(f"{radius_label:>12}", end='')
    print()
    print("-" * 70)
    
    # Compare key features
    key_features = ['planarity', 'linearity', 'sphericity', 'density']
    for feat in key_features:
        if feat in list(all_stats.values())[0]['stats']:
            print(f"{feat:<15} ", end='')
            for radius in radii:
                radius_label = f"r{radius:.1f}".replace('.', 'p')
                mean_val = all_stats[radius_label]['stats'][feat]['mean']
                print(f"{mean_val:>12.4f}", end='')
            print()
    
    print(f"\n{'='*70}")
    print("Test complete! Compare LAZ files in CloudCompare or QGIS.")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Test different radius values for geometric features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare specific radius values
  python test_radius_comparison.py input.laz --radii 0.5 1.0 1.5 2.0
  
  # Compare auto-estimated vs fixed values
  python test_radius_comparison.py input.laz --auto
  
  # Save to specific directory
  python test_radius_comparison.py input.laz --radii 0.8 1.2 --output results/
        """
    )
    
    parser.add_argument('input', type=Path, help='Input LAZ file')
    parser.add_argument('--radii', type=float, nargs='+',
                       help='Radius values to test (e.g., 0.5 1.0 1.5 2.0)')
    parser.add_argument('--auto', action='store_true',
                       help='Compare auto-estimated vs fixed radii (0.5, 1.0, 1.5, 2.0)')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output directory (default: <input_dir>/radius_test)')
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    # Determine radii to test
    if args.auto:
        radii = [None, 0.5, 1.0, 1.5, 2.0]  # None = auto-estimate
    elif args.radii:
        radii = args.radii
    else:
        print("Error: Must specify --radii or --auto")
        parser.print_help()
        return 1
    
    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = args.input.parent / "radius_test"
    
    # Run comparison
    try:
        compare_radii(args.input, radii, output_dir)
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
