#!/usr/bin/env python3
"""
Analyze features in enriched LAZ files to determine core vs full mode features.
"""

import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

try:
    import laspy
except ImportError:
    logger.error("laspy not available")
    sys.exit(1)


def analyze_laz_features(laz_path: Path):
    """Analyze features in a LAZ file."""
    logger.info(f"\nAnalyzing: {laz_path}")
    
    try:
        with laspy.open(laz_path) as f:
            las = f.read()
    except Exception as e:
        logger.error(f"Failed to read {laz_path}: {e}")
        return
    
    # Get all extra dimensions
    extra_dims = []
    if hasattr(las.point_format, 'extra_dimension_names'):
        extra_dims = las.point_format.extra_dimension_names
    elif hasattr(las, 'point_format'):
        # Try alternative method
        try:
            extra_dims = [dim.name for dim in las.point_format.extra_dims]
        except:
            pass
    
    if not extra_dims:
        logger.warning("No extra dimensions found")
        return
    
    logger.info(f"\nFound {len(extra_dims)} extra dimensions:")
    
    # Categorize features
    geometric_features = []
    building_features = []
    other_features = []
    
    known_geometric = [
        'linearity', 'planarity', 'sphericity', 'anisotropy',
        'curvature', 'omnivariance', 'eigensum', 'roughness',
        'density', 'verticality', 'horizontality'
    ]
    
    known_building = [
        'wall_score', 'roof_score', 'num_points_2m', 'num_points_1m',
        'vertical_std', 'neighborhood_extent', 'height_extent_ratio',
        'local_roughness'
    ]
    
    for dim in extra_dims:
        dim_lower = dim.lower()
        if dim_lower in known_geometric or 'normal' in dim_lower or dim_lower == 'height_above_ground':
            geometric_features.append(dim)
        elif dim_lower in known_building:
            building_features.append(dim)
        else:
            other_features.append(dim)
    
    if geometric_features:
        logger.info("\n  ✓ CORE/Geometric Features:")
        for feat in sorted(geometric_features):
            logger.info(f"    - {feat}")
    
    if building_features:
        logger.info("\n  ✓ FULL Mode Features (Building-specific):")
        for feat in sorted(building_features):
            logger.info(f"    - {feat}")
    
    if other_features:
        logger.info("\n  ℹ Other Features:")
        for feat in sorted(other_features):
            logger.info(f"    - {feat}")
    
    # Summary
    logger.info(f"\n  Summary:")
    logger.info(f"    Core features: {len(geometric_features)}")
    logger.info(f"    Full features: {len(building_features)}")
    logger.info(f"    Other: {len(other_features)}")
    logger.info(f"    Total: {len(extra_dims)}")
    
    return {
        'geometric': geometric_features,
        'building': building_features,
        'other': other_features
    }


def main():
    """Main function."""
    # Look for enriched LAZ files
    data_dir = Path(".")
    
    # Try to find enriched files
    enriched_files = list(data_dir.rglob("*_enriched*.laz"))
    
    if not enriched_files:
        logger.warning("No enriched LAZ files found in current directory")
        logger.info("Please provide a path to an enriched LAZ file:")
        logger.info("  python analyze_features.py /path/to/enriched.laz")
        
        if len(sys.argv) > 1:
            enriched_files = [Path(sys.argv[1])]
        else:
            return
    
    # Analyze all found files
    all_features = {
        'geometric': set(),
        'building': set(),
        'other': set()
    }
    
    for laz_file in enriched_files[:5]:  # Limit to 5 files
        result = analyze_laz_features(laz_file)
        if result:
            all_features['geometric'].update(result['geometric'])
            all_features['building'].update(result['building'])
            all_features['other'].update(result['other'])
    
    # Print comprehensive summary
    logger.info("\n" + "="*70)
    logger.info("COMPREHENSIVE FEATURE SUMMARY")
    logger.info("="*70)
    
    logger.info("\nCORE MODE FEATURES (Always Present):")
    for feat in sorted(all_features['geometric']):
        logger.info(f"  - {feat}")
    
    logger.info("\nFULL MODE FEATURES (Building-specific):")
    for feat in sorted(all_features['building']):
        logger.info(f"  - {feat}")
    
    logger.info("\nOTHER FEATURES:")
    for feat in sorted(all_features['other']):
        logger.info(f"  - {feat}")


if __name__ == '__main__':
    main()
