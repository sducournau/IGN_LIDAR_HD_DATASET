"""
Example: Multi-Mode Building Detection

This example demonstrates how to use the new building detection system
across ASPRS, LOD2, and LOD3 modes for different use cases.

Usage:
    python example_building_detection_modes.py --input tile.las --mode asprs
    python example_building_detection_modes.py --input tile.las --mode lod2
    python example_building_detection_modes.py --input tile.las --mode lod3
"""

import argparse
import numpy as np
import logging
from pathlib import Path

# Import building detection modules
from ign_lidar.core.modules.building_detection import (
    BuildingDetectionMode,
    BuildingDetectionConfig,
    BuildingDetector,
    detect_buildings_multi_mode
)

# Import feature computation
from ign_lidar.features import FeatureExtractor

# Import LiDAR I/O
from ign_lidar.io import read_las, write_las

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_asprs_mode(input_path: str, output_path: str):
    """
    Example: ASPRS mode for general building detection.
    
    Use case: Standard LiDAR classification workflow
    Output: Single building class (code 6)
    """
    logger.info("=" * 60)
    logger.info("ASPRS MODE: General Building Detection")
    logger.info("=" * 60)
    
    # Read point cloud
    logger.info(f"Reading {input_path}...")
    points, colors, classification = read_las(input_path)
    
    # Extract features
    logger.info("Extracting features...")
    extractor = FeatureExtractor(k_neighbors=20, mode='full')
    features = extractor.compute_features(points)
    
    # Detect buildings
    logger.info("Detecting buildings in ASPRS mode...")
    refined_labels, stats = detect_buildings_multi_mode(
        labels=classification,
        features=features,
        mode='asprs'
    )
    
    # Log results
    logger.info("\n📊 Detection Results:")
    logger.info(f"  Total buildings: {stats['total']:,} points")
    logger.info(f"  - Walls: {stats['walls']:,}")
    logger.info(f"  - Roofs: {stats['roofs']:,}")
    logger.info(f"  - Structured: {stats['structured']:,}")
    logger.info(f"  - Edges: {stats['edges']:,}")
    
    # Save result
    logger.info(f"\n💾 Saving to {output_path}...")
    write_las(output_path, points, colors, refined_labels)
    logger.info("✅ Done!")


def example_lod2_mode(input_path: str, output_path: str):
    """
    Example: LOD2 mode for building element detection.
    
    Use case: LOD2 building reconstruction training
    Output: Walls, flat roofs, sloped roofs
    """
    logger.info("=" * 60)
    logger.info("LOD2 MODE: Building Element Detection")
    logger.info("=" * 60)
    
    # Read point cloud
    logger.info(f"Reading {input_path}...")
    points, colors, classification = read_las(input_path)
    
    # Extract features (LOD2 mode)
    logger.info("Extracting LOD2 features...")
    extractor = FeatureExtractor(k_neighbors=20, mode='lod2')
    features = extractor.compute_features(points)
    
    # Create custom configuration
    config = BuildingDetectionConfig(mode=BuildingDetectionMode.LOD2)
    config.wall_verticality_min = 0.72  # Slightly stricter
    config.roof_planarity_min = 0.72
    
    # Detect buildings
    logger.info("Detecting building elements in LOD2 mode...")
    detector = BuildingDetector(config=config)
    refined_labels, stats = detector.detect_buildings(
        labels=classification,
        height=features['height'],
        planarity=features['planarity'],
        verticality=features['verticality'],
        normals=features['normals'],
        linearity=features.get('linearity'),
        anisotropy=features.get('anisotropy'),
        wall_score=features.get('wall_score'),
        roof_score=features.get('roof_score')
    )
    
    # Log results
    logger.info("\n📊 Detection Results:")
    logger.info(f"  Total building elements: {stats['total_building']:,} points")
    logger.info(f"  - Walls (class 0): {stats['walls']:,}")
    logger.info(f"  - Flat roofs (class 1): {stats['flat_roofs']:,}")
    logger.info(f"  - Sloped roofs (class 2/3): {stats['sloped_roofs']:,}")
    
    # Class distribution
    unique, counts = np.unique(refined_labels, return_counts=True)
    logger.info("\n📈 Class Distribution:")
    for cls, count in zip(unique, counts):
        logger.info(f"  Class {cls}: {count:,} points")
    
    # Save result
    logger.info(f"\n💾 Saving to {output_path}...")
    write_las(output_path, points, colors, refined_labels)
    logger.info("✅ Done!")


def example_lod3_mode(input_path: str, output_path: str):
    """
    Example: LOD3 mode for detailed architectural detection.
    
    Use case: LOD3 detailed building modeling
    Output: Walls, roofs, windows, doors, balconies, etc.
    """
    logger.info("=" * 60)
    logger.info("LOD3 MODE: Detailed Architectural Detection")
    logger.info("=" * 60)
    
    # Read point cloud
    logger.info(f"Reading {input_path}...")
    points, colors, classification = read_las(input_path)
    
    # Extract features (LOD3 mode - full features)
    logger.info("Extracting LOD3 features (this may take a while)...")
    extractor = FeatureExtractor(k_neighbors=20, mode='lod3')
    features = extractor.compute_features(points)
    
    # Add point coordinates to features
    features['points'] = points
    
    # Create custom configuration
    config = BuildingDetectionConfig(mode=BuildingDetectionMode.LOD3)
    config.wall_verticality_min = 0.75
    config.detect_windows = True
    config.detect_doors = True
    config.detect_balconies = True
    config.detect_chimneys = True
    config.opening_intensity_threshold = 0.25  # For glass detection
    
    # Detect buildings
    logger.info("Detecting architectural details in LOD3 mode...")
    detector = BuildingDetector(config=config)
    refined_labels, stats = detector.detect_buildings(
        labels=classification,
        height=features['height'],
        planarity=features['planarity'],
        verticality=features['verticality'],
        normals=features['normals'],
        linearity=features.get('linearity'),
        anisotropy=features.get('anisotropy'),
        curvature=features.get('curvature'),
        intensity=features.get('intensity'),
        wall_score=features.get('wall_score'),
        roof_score=features.get('roof_score'),
        points=features['points']
    )
    
    # Log detailed results
    logger.info("\n📊 Detection Results:")
    logger.info(f"  Total architectural elements: {stats['total_building']:,} points")
    logger.info("\n  Building Structure:")
    logger.info(f"    - Walls (class 0): {stats['walls']:,}")
    logger.info(f"    - Flat roofs (class 1): {stats['flat_roofs']:,}")
    logger.info(f"    - Sloped roofs (class 2/3): {stats['sloped_roofs']:,}")
    logger.info("\n  Architectural Details:")
    logger.info(f"    - Windows (class 13): {stats['windows']:,}")
    logger.info(f"    - Doors (class 14): {stats['doors']:,}")
    logger.info(f"    - Balconies (class 15): {stats['balconies']:,}")
    logger.info(f"    - Chimneys (class 18): {stats['chimneys']:,}")
    logger.info(f"    - Dormers (class 20): {stats['dormers']:,}")
    
    # Class distribution
    unique, counts = np.unique(refined_labels, return_counts=True)
    logger.info("\n📈 Complete Class Distribution:")
    class_names = {
        0: 'Wall', 1: 'Roof (Flat)', 2: 'Roof (Gable)', 3: 'Roof (Hip)',
        13: 'Window', 14: 'Door', 15: 'Balcony', 18: 'Chimney', 20: 'Dormer'
    }
    for cls, count in zip(unique, counts):
        name = class_names.get(cls, f'Other ({cls})')
        logger.info(f"  {name}: {count:,} points")
    
    # Save result
    logger.info(f"\n💾 Saving to {output_path}...")
    write_las(output_path, points, colors, refined_labels)
    logger.info("✅ Done!")


def example_comparison(input_path: str, output_dir: str):
    """
    Example: Compare all three modes on the same data.
    
    Demonstrates the differences between ASPRS, LOD2, and LOD3 modes.
    """
    logger.info("=" * 60)
    logger.info("MODE COMPARISON: ASPRS vs LOD2 vs LOD3")
    logger.info("=" * 60)
    
    # Read point cloud
    logger.info(f"Reading {input_path}...")
    points, colors, classification = read_las(input_path)
    
    # Extract full features (needed for all modes)
    logger.info("Extracting features...")
    extractor = FeatureExtractor(k_neighbors=20, mode='full')
    features = extractor.compute_features(points)
    features['points'] = points
    
    results = {}
    
    # Test each mode
    for mode in ['asprs', 'lod2', 'lod3']:
        logger.info(f"\n🔍 Testing {mode.upper()} mode...")
        
        labels, stats = detect_buildings_multi_mode(
            labels=classification.copy(),
            features=features,
            mode=mode
        )
        
        results[mode] = {
            'labels': labels,
            'stats': stats
        }
        
        # Save output
        output_path = Path(output_dir) / f"building_detection_{mode}.las"
        write_las(str(output_path), points, colors, labels)
        logger.info(f"  Saved to {output_path}")
    
    # Compare results
    logger.info("\n" + "=" * 60)
    logger.info("📊 COMPARISON RESULTS")
    logger.info("=" * 60)
    
    logger.info("\nASPRS Mode:")
    logger.info(f"  Total buildings: {results['asprs']['stats']['total']:,}")
    
    logger.info("\nLOD2 Mode:")
    logger.info(f"  Total building elements: {results['lod2']['stats']['total_building']:,}")
    logger.info(f"  - Walls: {results['lod2']['stats']['walls']:,}")
    logger.info(f"  - Roofs: {results['lod2']['stats']['flat_roofs'] + results['lod2']['stats']['sloped_roofs']:,}")
    
    logger.info("\nLOD3 Mode:")
    logger.info(f"  Total architectural elements: {results['lod3']['stats']['total_building']:,}")
    logger.info(f"  - Structure: {results['lod3']['stats']['walls'] + results['lod3']['stats']['flat_roofs'] + results['lod3']['stats']['sloped_roofs']:,}")
    logger.info(f"  - Details: {results['lod3']['stats']['windows'] + results['lod3']['stats']['doors'] + results['lod3']['stats']['balconies']:,}")
    
    logger.info("\n✅ Comparison complete!")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Multi-Mode Building Detection Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ASPRS mode (general building detection)
  python example_building_detection_modes.py --input tile.las --mode asprs --output buildings_asprs.las
  
  # LOD2 mode (building elements)
  python example_building_detection_modes.py --input tile.las --mode lod2 --output buildings_lod2.las
  
  # LOD3 mode (detailed architecture)
  python example_building_detection_modes.py --input tile.las --mode lod3 --output buildings_lod3.las
  
  # Compare all modes
  python example_building_detection_modes.py --input tile.las --mode compare --output output_dir/
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input LAS/LAZ file path'
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['asprs', 'lod2', 'lod3', 'compare'],
        default='lod2',
        help='Detection mode (default: lod2)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file path (or directory for compare mode)'
    )
    
    args = parser.parse_args()
    
    # Set default output if not provided
    if args.output is None:
        input_path = Path(args.input)
        if args.mode == 'compare':
            args.output = input_path.parent / f"{input_path.stem}_comparison"
        else:
            args.output = input_path.parent / f"{input_path.stem}_{args.mode}.las"
    
    # Run appropriate example
    if args.mode == 'asprs':
        example_asprs_mode(args.input, args.output)
    elif args.mode == 'lod2':
        example_lod2_mode(args.input, args.output)
    elif args.mode == 'lod3':
        example_lod3_mode(args.input, args.output)
    elif args.mode == 'compare':
        Path(args.output).mkdir(parents=True, exist_ok=True)
        example_comparison(args.input, args.output)


if __name__ == '__main__':
    main()
