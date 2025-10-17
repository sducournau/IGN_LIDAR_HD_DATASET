#!/usr/bin/env python3
"""
Select Optimal Tiles - Multi-Scale Training Pipeline

This script selects optimal tiles for multi-scale training based on the
analysis report. It implements selection strategies to ensure diverse,
high-quality training data across all classification levels.

Selection criteria:
- Point density (prefer tiles with good coverage)
- Spatial distribution (avoid clustering)
- File quality (no errors)
- Balanced representation across the dataset

Output: Text files listing selected tiles for each level
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import random

try:
    import numpy as np
except ImportError:
    print("Error: Required packages not installed.")
    print("Please install: pip install numpy")
    sys.exit(1)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def calculate_tile_score(tile: Dict[str, Any]) -> float:
    """
    Calculate a quality score for a tile.
    
    Args:
        tile: Tile information dictionary
        
    Returns:
        Score (higher is better)
    """
    if not tile.get('success', False):
        return 0.0
    
    score = 0.0
    
    # Point density contribution (0-50 points)
    density = tile.get('density_pts_m2', 0)
    if density > 0:
        # Normalize density (assume 10 pts/m2 is excellent)
        score += min(density / 10.0 * 50, 50)
    
    # Point count contribution (0-30 points)
    point_count = tile.get('point_count', 0)
    if point_count > 0:
        # Normalize point count (assume 10M points is excellent)
        score += min(point_count / 10_000_000 * 30, 30)
    
    # Classification diversity contribution (0-20 points)
    class_dist = tile.get('classification_distribution', {})
    if class_dist:
        # More diverse classification is better
        num_classes = len(class_dist)
        score += min(num_classes * 4, 20)
    
    return score


def select_tiles_by_strategy(
    tiles: List[Dict[str, Any]], 
    count: int,
    strategy: str = 'best'
) -> List[Dict[str, Any]]:
    """
    Select tiles based on a strategy.
    
    Args:
        tiles: List of tile information dictionaries
        count: Number of tiles to select
        strategy: Selection strategy ('best', 'random', 'diverse')
        
    Returns:
        List of selected tiles
    """
    # Filter out failed tiles
    valid_tiles = [t for t in tiles if t.get('success', False)]
    
    if not valid_tiles:
        logger.warning("No valid tiles available for selection")
        return []
    
    # Limit to available tiles
    count = min(count, len(valid_tiles))
    
    if strategy == 'best':
        # Select tiles with highest scores
        scored_tiles = [(t, calculate_tile_score(t)) for t in valid_tiles]
        scored_tiles.sort(key=lambda x: x[1], reverse=True)
        selected = [t for t, score in scored_tiles[:count]]
        
    elif strategy == 'random':
        # Random selection
        selected = random.sample(valid_tiles, count)
        
    elif strategy == 'diverse':
        # Try to select spatially diverse tiles
        # For now, use a simple approach: sort by score and take every Nth tile
        scored_tiles = [(t, calculate_tile_score(t)) for t in valid_tiles]
        scored_tiles.sort(key=lambda x: x[1], reverse=True)
        
        step = max(1, len(scored_tiles) // count)
        selected = [scored_tiles[i][0] for i in range(0, len(scored_tiles), step)][:count]
        
    else:
        # Default to best
        logger.warning(f"Unknown strategy '{strategy}', using 'best'")
        return select_tiles_by_strategy(tiles, count, 'best')
    
    logger.info(f"Selected {len(selected)} tiles using '{strategy}' strategy")
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Select optimal tiles for multi-scale training"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to unified_dataset directory"
    )
    parser.add_argument(
        "--analysis",
        type=str,
        required=True,
        help="Path to analysis report JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for tile lists"
    )
    parser.add_argument(
        "--asprs-count",
        type=int,
        default=100,
        help="Number of ASPRS tiles to select (default: 100)"
    )
    parser.add_argument(
        "--lod2-count",
        type=int,
        default=80,
        help="Number of LOD2 tiles to select (default: 80)"
    )
    parser.add_argument(
        "--lod3-count",
        type=int,
        default=60,
        help="Number of LOD3 tiles to select (default: 60)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=['best', 'random', 'diverse'],
        default='best',
        help="Tile selection strategy (default: best)"
    )
    
    args = parser.parse_args()
    
    # Load analysis report
    analysis_path = Path(args.analysis)
    if not analysis_path.exists():
        logger.error(f"Analysis report not found: {analysis_path}")
        sys.exit(1)
    
    with open(analysis_path, 'r') as f:
        report = json.load(f)
    
    logger.info(f"Loaded analysis report from: {analysis_path}")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Selection configuration
    selection_config = {
        'asprs': args.asprs_count,
        'lod2': args.lod2_count,
        'lod3': args.lod3_count
    }
    
    # Select tiles for each level (avoiding duplicates)
    selection_results = {}
    already_selected = set()  # Track tiles already selected
    
    # Selection order: ASPRS first (largest), then LOD2, then LOD3
    # This ensures we get the best tiles for each level without overlap
    selection_order = ['asprs', 'lod2', 'lod3']
    
    for level in selection_order:
        count = selection_config[level]
        logger.info(f"\nSelecting {count} tiles for {level.upper()}...")
        
        level_analysis = report.get('detailed_analysis', {}).get(level, {})
        tiles = level_analysis.get('tiles', [])
        
        if not tiles:
            logger.warning(f"No tiles found in analysis for {level}")
            selection_results[level] = []
            continue
        
        # Filter out already selected tiles to avoid duplicates
        available_tiles = [t for t in tiles if t['file_name'] not in already_selected]
        logger.info(f"  Available tiles (excluding duplicates): {len(available_tiles)} / {len(tiles)}")
        
        if not available_tiles:
            logger.warning(f"No available tiles for {level} after excluding duplicates!")
            selection_results[level] = []
            continue
        
        # Select from available tiles
        selected_tiles = select_tiles_by_strategy(available_tiles, count, args.strategy)
        selection_results[level] = selected_tiles
        
        # Mark these tiles as selected
        for tile in selected_tiles:
            already_selected.add(tile['file_name'])
        
        logger.info(f"  Selected {len(selected_tiles)} unique tiles for {level.upper()}")
        
        # Save tile list
        list_file = output_path / f"{level}_selected_tiles.txt"
        with open(list_file, 'w') as f:
            for tile in selected_tiles:
                f.write(f"{tile['file_name']}\n")
        
        logger.info(f"Saved {len(selected_tiles)} tile names to: {list_file}")
        
        # Also save detailed selection info
        detail_file = output_path / f"{level}_selection_details.json"
        with open(detail_file, 'w') as f:
            json.dump({
                'level': level,
                'requested_count': count,
                'selected_count': len(selected_tiles),
                'strategy': args.strategy,
                'tiles': selected_tiles
            }, f, indent=2)
        
        logger.info(f"Saved detailed selection info to: {detail_file}")
    
    # Create summary with duplicate detection
    all_selected_tiles = []
    for level in ['asprs', 'lod2', 'lod3']:
        all_selected_tiles.extend([t['file_name'] for t in selection_results[level]])
    
    # Check for duplicates (should be none if logic worked correctly)
    unique_tiles = set(all_selected_tiles)
    has_duplicates = len(all_selected_tiles) != len(unique_tiles)
    
    if has_duplicates:
        logger.warning(f"⚠️  WARNING: Found {len(all_selected_tiles) - len(unique_tiles)} duplicate tiles!")
        # Find which tiles are duplicated
        from collections import Counter
        tile_counts = Counter(all_selected_tiles)
        duplicates = {tile: count for tile, count in tile_counts.items() if count > 1}
        logger.warning(f"  Duplicated tiles: {duplicates}")
    else:
        logger.info(f"✓ No duplicate tiles found across all levels")
    
    summary = {
        'input_dataset': args.input,
        'analysis_report': args.analysis,
        'strategy': args.strategy,
        'duplicate_detection': {
            'has_duplicates': has_duplicates,
            'total_selections': len(all_selected_tiles),
            'unique_tiles': len(unique_tiles),
            'duplicates_count': len(all_selected_tiles) - len(unique_tiles) if has_duplicates else 0
        },
        'selection': {
            level: {
                'requested': selection_config[level],
                'selected': len(selection_results[level]),
                'tile_names': [t['file_name'] for t in selection_results[level]]
            }
            for level in ['asprs', 'lod2', 'lod3']
        }
    }
    
    summary_file = output_path / 'selection_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Selection Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info(f"\nTile counts:")
    logger.info(f"  ASPRS: {len(selection_results['asprs'])} tiles")
    logger.info(f"  LOD2:  {len(selection_results['lod2'])} tiles")
    logger.info(f"  LOD3:  {len(selection_results['lod3'])} tiles")
    logger.info(f"  Total: {sum(len(v) for v in selection_results.values())} tiles")
    logger.info(f"  Unique: {len(unique_tiles)} tiles")
    logger.info(f"\nDuplicate check: {'✓ PASS' if not has_duplicates else '✗ FAIL'}")
    logger.info(f"{'='*60}\n")
    
    return 0 if not has_duplicates else 1


if __name__ == "__main__":
    sys.exit(main())
