#!/usr/bin/env python3
"""
Analyze Unified Dataset - Multi-Scale Training Pipeline

This script analyzes the unified dataset structure and generates a comprehensive
report about available tiles, their quality, and suitability for multi-scale training.

Expected directory structure:
    unified_dataset/
    ├── asprs/          # ASPRS classification tiles
    │   ├── tile_001.laz
    │   ├── tile_002.laz
    │   └── ...
    ├── lod2/           # LOD2 building tiles
    │   ├── tile_001.laz
    │   └── ...
    └── lod3/           # LOD3 architectural tiles
        ├── tile_001.laz
        └── ...

Output: JSON report with:
- Total tile counts per classification level
- File sizes and point cloud statistics
- Quality metrics (point density, coverage, etc.)
- Recommendations for tile selection
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

try:
    import laspy
    import numpy as np
except ImportError:
    print("Error: Required packages not installed.")
    print("Please install: pip install laspy numpy")
    sys.exit(1)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_laz_file(laz_path: Path) -> Dict[str, Any]:
    """
    Analyze a single LAZ file and extract statistics.
    
    Args:
        laz_path: Path to the LAZ file
        
    Returns:
        Dictionary with file statistics
    """
    try:
        with laspy.open(laz_path) as laz:
            header = laz.header
            
            # Read a sample of points for analysis
            las = laz.read()
            
            # Calculate basic statistics
            point_count = len(las.points)
            
            # Calculate bounds
            bounds = {
                'x_min': float(header.x_min),
                'x_max': float(header.x_max),
                'y_min': float(header.y_min),
                'y_max': float(header.y_max),
                'z_min': float(header.z_min),
                'z_max': float(header.z_max),
            }
            
            # Calculate area and density
            area = (bounds['x_max'] - bounds['x_min']) * (bounds['y_max'] - bounds['y_min'])
            density = point_count / area if area > 0 else 0
            
            # Analyze classification distribution (if available)
            classification_dist = {}
            if hasattr(las, 'classification'):
                unique, counts = np.unique(las.classification, return_counts=True)
                classification_dist = {int(k): int(v) for k, v in zip(unique, counts)}
            
            return {
                'file_path': str(laz_path),
                'file_name': laz_path.name,
                'file_size_mb': laz_path.stat().st_size / (1024 * 1024),
                'point_count': point_count,
                'bounds': bounds,
                'area_m2': area,
                'density_pts_m2': density,
                'classification_distribution': classification_dist,
                'version': f"{header.version.major}.{header.version.minor}",
                'point_format': header.point_format.id,
                'success': True
            }
            
    except Exception as e:
        logger.error(f"Error analyzing {laz_path}: {e}")
        return {
            'file_path': str(laz_path),
            'file_name': laz_path.name,
            'error': str(e),
            'success': False
        }


def analyze_classification_level(level_path: Path, level_name: str) -> Dict[str, Any]:
    """
    Analyze all tiles in a classification level directory.
    
    Args:
        level_path: Path to the classification level directory
        level_name: Name of the classification level (asprs, lod2, lod3)
        
    Returns:
        Dictionary with level statistics
    """
    logger.info(f"Analyzing {level_name} tiles in {level_path}...")
    
    if not level_path.exists():
        logger.warning(f"Directory not found: {level_path}")
        return {
            'level': level_name,
            'exists': False,
            'tile_count': 0,
            'tiles': []
        }
    
    # Find all LAZ files
    laz_files = sorted(level_path.glob("*.laz"))
    logger.info(f"Found {len(laz_files)} LAZ files in {level_name}")
    
    # Analyze each file
    tiles = []
    total_points = 0
    total_size_mb = 0
    successful_tiles = 0
    
    for laz_file in laz_files:
        tile_info = analyze_laz_file(laz_file)
        tiles.append(tile_info)
        
        if tile_info.get('success', False):
            successful_tiles += 1
            total_points += tile_info.get('point_count', 0)
            total_size_mb += tile_info.get('file_size_mb', 0)
    
    # Calculate aggregate statistics
    densities = [t['density_pts_m2'] for t in tiles if t.get('success') and t.get('density_pts_m2', 0) > 0]
    avg_density = np.mean(densities) if densities else 0
    
    return {
        'level': level_name,
        'exists': True,
        'tile_count': len(laz_files),
        'successful_tiles': successful_tiles,
        'failed_tiles': len(laz_files) - successful_tiles,
        'total_points': total_points,
        'total_size_mb': total_size_mb,
        'avg_density_pts_m2': avg_density,
        'tiles': tiles
    }


def generate_recommendations(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate recommendations based on the analysis.
    
    Args:
        analysis: Complete analysis dictionary
        
    Returns:
        Dictionary with recommendations
    """
    recommendations = {
        'asprs': {},
        'lod2': {},
        'lod3': {},
        'warnings': []
    }
    
    for level in ['asprs', 'lod2', 'lod3']:
        level_data = analysis.get(level, {})
        
        if not level_data.get('exists', False):
            recommendations['warnings'].append(f"{level.upper()}: Directory not found")
            continue
        
        tile_count = level_data.get('successful_tiles', 0)
        avg_density = level_data.get('avg_density_pts_m2', 0)
        
        # Recommended tile counts for multi-scale training
        recommended_counts = {
            'asprs': 100,
            'lod2': 80,
            'lod3': 60
        }
        
        recommendations[level] = {
            'available_tiles': tile_count,
            'recommended_for_training': min(tile_count, recommended_counts[level]),
            'sufficient_for_training': tile_count >= recommended_counts[level] * 0.5,
            'quality_assessment': 'good' if avg_density > 1.0 else 'low_density'
        }
        
        if tile_count < recommended_counts[level] * 0.5:
            recommendations['warnings'].append(
                f"{level.upper()}: Only {tile_count} tiles available "
                f"(recommended: {recommended_counts[level]})"
            )
    
    return recommendations


def main():
    parser = argparse.ArgumentParser(
        description="Analyze unified dataset for multi-scale training"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to unified_dataset directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file for analysis report"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        sys.exit(1)
    
    logger.info(f"Analyzing unified dataset: {input_path}")
    
    # Analyze each classification level
    analysis = {}
    
    for level in ['asprs', 'lod2', 'lod3']:
        level_path = input_path / level
        analysis[level] = analyze_classification_level(level_path, level)
    
    # Generate recommendations
    recommendations = generate_recommendations(analysis)
    
    # Create complete report
    report = {
        'dataset_path': str(input_path),
        'analysis_timestamp': str(Path().cwd()),  # Placeholder
        'summary': {
            'asprs_tiles': analysis.get('asprs', {}).get('successful_tiles', 0),
            'lod2_tiles': analysis.get('lod2', {}).get('successful_tiles', 0),
            'lod3_tiles': analysis.get('lod3', {}).get('successful_tiles', 0),
            'total_tiles': sum([
                analysis.get(l, {}).get('successful_tiles', 0) 
                for l in ['asprs', 'lod2', 'lod3']
            ]),
        },
        'detailed_analysis': analysis,
        'recommendations': recommendations
    }
    
    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Analysis complete. Report saved to: {output_path}")
    logger.info(f"Summary: ASPRS={report['summary']['asprs_tiles']}, "
                f"LOD2={report['summary']['lod2_tiles']}, "
                f"LOD3={report['summary']['lod3_tiles']}")
    
    if recommendations['warnings']:
        logger.warning("Warnings found:")
        for warning in recommendations['warnings']:
            logger.warning(f"  - {warning}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
