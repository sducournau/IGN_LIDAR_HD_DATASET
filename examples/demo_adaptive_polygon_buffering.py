"""
Demo: Adaptive Polygon Buffering for Building Classification

This demo shows how to use the adaptive polygon buffering system to address
the issue of unclassified building points (white bands in plan view).

The system:
1. Analyzes each building's point cloud individually
2. Detects actual building boundaries, edges, and endpoints
3. Computes optimal buffer distances per building
4. Applies variable buffers to capture all building points

This ensures complete building coverage while avoiding over-classification.

Usage:
    python demo_adaptive_polygon_buffering.py --tile path/to/tile.laz
    
Author: Simon Ducournau
Date: October 25, 2025
"""

import logging
import numpy as np
from pathlib import Path
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Demo adaptive polygon buffering')
    parser.add_argument('--tile', type=str, help='Path to LiDAR tile (.laz)')
    parser.add_argument('--output-dir', type=str, default='./output_adaptive_buffering',
                       help='Output directory')
    parser.add_argument('--bbox', type=float, nargs=4, 
                       help='Manual bounding box: xmin ymin xmax ymax')
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Adaptive Polygon Buffering Demo")
    logger.info("=" * 80)
    
    # Import required modules
    try:
        from ign_lidar.io.laz import LAZFile
        from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
        from ign_lidar.core.classification.building.adaptive_polygon_buffer import (
            AdaptivePolygonBuffer,
            AdaptiveBufferConfig
        )
        from ign_lidar.core.classification.building.adaptive_integration import (
            AdaptiveGroundTruthProcessor,
            integrate_adaptive_buffering_with_wfs
        )
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Make sure the ign_lidar package is installed")
        return 1
    
    # Step 1: Load point cloud
    if args.tile:
        logger.info(f"\nStep 1: Loading LiDAR tile from {args.tile}")
        tile_path = Path(args.tile)
        
        if not tile_path.exists():
            logger.error(f"Tile not found: {tile_path}")
            return 1
        
        try:
            laz_file = LAZFile(tile_path)
            points = laz_file.read_points()
            
            logger.info(f"  Loaded {len(points):,} points")
            
            # Get tile bounding box
            if args.bbox:
                bbox = tuple(args.bbox)
            else:
                bbox = (
                    np.min(points[:, 0]), np.min(points[:, 1]),
                    np.max(points[:, 0]), np.max(points[:, 1])
                )
            
            logger.info(f"  Bounding box: {bbox}")
            
        except Exception as e:
            logger.error(f"Failed to load tile: {e}")
            return 1
    else:
        # Use synthetic data for demo
        logger.info("\nStep 1: Generating synthetic point cloud")
        np.random.seed(42)
        
        # Create a rectangular building (20m x 30m)
        bbox = (650000, 6860000, 650100, 6860100)  # Versailles area
        
        building_center = [650050, 6860050, 10]
        building_size = [20, 30, 8]
        
        # Generate building points
        n_points = 10000
        building_points = np.random.rand(n_points, 3)
        building_points[:, 0] = building_center[0] + (building_points[:, 0] - 0.5) * building_size[0]
        building_points[:, 1] = building_center[1] + (building_points[:, 1] - 0.5) * building_size[1]
        building_points[:, 2] = building_center[2] + building_points[:, 2] * building_size[2]
        
        # Add some ground points
        n_ground = 5000
        ground_points = np.random.rand(n_ground, 3)
        ground_points[:, 0] = bbox[0] + (bbox[2] - bbox[0]) * ground_points[:, 0]
        ground_points[:, 1] = bbox[1] + (bbox[3] - bbox[1]) * ground_points[:, 1]
        ground_points[:, 2] = 0.5 * ground_points[:, 2]
        
        points = np.vstack([building_points, ground_points])
        
        logger.info(f"  Generated {len(points):,} synthetic points")
        logger.info(f"  Building: {n_points} points, Ground: {n_ground} points")
    
    # Step 2: Fetch ground truth polygons
    logger.info("\nStep 2: Fetching ground truth building polygons from BD TOPO")
    
    try:
        wfs_fetcher = IGNGroundTruthFetcher(cache_dir=Path('./cache_wfs'))
        building_polygons = wfs_fetcher.fetch_buildings(bbox=bbox)
        
        if building_polygons is None or len(building_polygons) == 0:
            logger.warning("No building polygons found in this area")
            logger.warning("Cannot demonstrate adaptive buffering without ground truth")
            return 1
        
        logger.info(f"  Fetched {len(building_polygons)} building polygons")
        
    except Exception as e:
        logger.error(f"Failed to fetch ground truth: {e}")
        return 1
    
    # Step 3: Configure adaptive buffering
    logger.info("\nStep 3: Configuring adaptive buffering system")
    
    config = AdaptiveBufferConfig(
        min_buffer_size=0.5,           # Minimum 0.5m buffer
        max_buffer_size=5.0,           # Maximum 5m buffer
        default_buffer_size=1.5,       # Default 1.5m when uncertain
        gap_detection_enabled=True,    # Detect gaps
        edge_detection_enabled=True,   # Detect edge points
        endpoint_detection_enabled=True,  # Detect corners/endpoints
        use_3d_bbox_fitting=True,      # Fit 3D bounding boxes
        use_directional_buffers=True,  # Variable buffers by direction
        min_points_per_building=20,    # Min points to analyze
        outlier_removal_enabled=True   # Remove outliers
    )
    
    logger.info("  Configuration:")
    logger.info(f"    Buffer range: {config.min_buffer_size}m - {config.max_buffer_size}m")
    logger.info(f"    Gap detection: {config.gap_detection_enabled}")
    logger.info(f"    Edge detection: {config.edge_detection_enabled}")
    logger.info(f"    3D bbox fitting: {config.use_3d_bbox_fitting}")
    
    # Step 4: Apply adaptive buffering
    logger.info("\nStep 4: Applying adaptive buffering to building polygons")
    
    try:
        buffer_system = AdaptivePolygonBuffer(config=config)
        
        buffered_polygons = buffer_system.apply_adaptive_buffers(
            building_polygons=building_polygons,
            points=points,
            per_building_analysis=True  # Analyze each building individually
        )
        
        logger.info(f"  Processed {len(buffered_polygons)} buildings")
        
        # Print statistics
        if 'adaptive_buffer' in buffered_polygons.columns:
            buffers = buffered_polygons['adaptive_buffer'].values
            logger.info(f"\n  Buffer Statistics:")
            logger.info(f"    Mean: {np.mean(buffers):.2f}m")
            logger.info(f"    Std:  {np.std(buffers):.2f}m")
            logger.info(f"    Min:  {np.min(buffers):.2f}m")
            logger.info(f"    Max:  {np.max(buffers):.2f}m")
        
        if 'point_coverage' in buffered_polygons.columns:
            coverage = buffered_polygons['point_coverage'].values
            logger.info(f"\n  Coverage Statistics:")
            logger.info(f"    Mean coverage: {np.mean(coverage)*100:.1f}%")
            logger.info(f"    Buildings with <80% coverage: {np.sum(coverage < 0.8)}")
        
        if 'has_gaps' in buffered_polygons.columns:
            n_gaps = np.sum(buffered_polygons['has_gaps'])
            logger.info(f"\n  Gap Detection:")
            logger.info(f"    Buildings with gaps: {n_gaps}/{len(buffered_polygons)}")
        
    except Exception as e:
        logger.error(f"Failed to apply adaptive buffering: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 5: Compare with fixed buffering
    logger.info("\nStep 5: Comparing with fixed buffering")
    
    fixed_buffer_sizes = [0.5, 1.0, 1.5, 2.0, 3.0]
    
    for fixed_buffer in fixed_buffer_sizes:
        fixed_buffered = building_polygons.copy()
        fixed_buffered['geometry'] = fixed_buffered.geometry.buffer(fixed_buffer)
        
        # Count points in buffered polygons
        from shapely.geometry import Point
        from shapely.strtree import STRtree
        
        tree = STRtree(fixed_buffered.geometry.tolist())
        n_covered = 0
        
        for pt_coords in points[:, :2]:
            pt = Point(pt_coords[0], pt_coords[1])
            if len(tree.query(pt)) > 0:
                for poly in tree.query(pt):
                    if poly.contains(pt):
                        n_covered += 1
                        break
        
        coverage_pct = (n_covered / len(points)) * 100
        logger.info(f"  Fixed buffer {fixed_buffer:.1f}m: {n_covered:,} points ({coverage_pct:.1f}%) covered")
    
    # Adaptive buffering coverage
    tree = STRtree(buffered_polygons.geometry.tolist())
    n_covered = 0
    
    for pt_coords in points[:, :2]:
        pt = Point(pt_coords[0], pt_coords[1])
        if len(tree.query(pt)) > 0:
            for poly in tree.query(pt):
                if poly.contains(pt):
                    n_covered += 1
                    break
    
    coverage_pct = (n_covered / len(points)) * 100
    logger.info(f"  Adaptive buffer: {n_covered:,} points ({coverage_pct:.1f}%) covered")
    
    # Step 6: Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nStep 6: Saving results to {output_dir}")
    
    try:
        # Save original polygons
        original_file = output_dir / 'original_polygons.geojson'
        building_polygons.to_file(original_file, driver='GeoJSON')
        logger.info(f"  Saved original polygons: {original_file}")
        
        # Save buffered polygons
        buffered_file = output_dir / 'adaptive_buffered_polygons.geojson'
        buffered_polygons.to_file(buffered_file, driver='GeoJSON')
        logger.info(f"  Saved buffered polygons: {buffered_file}")
        
        # Save statistics
        stats_file = output_dir / 'buffer_statistics.txt'
        with open(stats_file, 'w') as f:
            f.write("Adaptive Polygon Buffering Statistics\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Number of buildings: {len(buffered_polygons)}\n")
            
            if 'adaptive_buffer' in buffered_polygons.columns:
                buffers = buffered_polygons['adaptive_buffer'].values
                f.write(f"\nBuffer Statistics:\n")
                f.write(f"  Mean: {np.mean(buffers):.2f}m\n")
                f.write(f"  Std:  {np.std(buffers):.2f}m\n")
                f.write(f"  Min:  {np.min(buffers):.2f}m\n")
                f.write(f"  Max:  {np.max(buffers):.2f}m\n")
            
            if 'point_coverage' in buffered_polygons.columns:
                coverage = buffered_polygons['point_coverage'].values
                f.write(f"\nCoverage Statistics:\n")
                f.write(f"  Mean coverage: {np.mean(coverage)*100:.1f}%\n")
                f.write(f"  Buildings with <80% coverage: {np.sum(coverage < 0.8)}\n")
        
        logger.info(f"  Saved statistics: {stats_file}")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        return 1
    
    logger.info("\n" + "=" * 80)
    logger.info("Demo completed successfully!")
    logger.info("=" * 80)
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("\nVisualization tip:")
    logger.info("  Load both original_polygons.geojson and adaptive_buffered_polygons.geojson")
    logger.info("  in QGIS to visualize the adaptive buffering effect.")
    
    return 0


if __name__ == '__main__':
    exit(main())
