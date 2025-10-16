"""
Test script for geometric rules optimization in reclassification.

This script demonstrates the new geometric rules features:
1. Road-vegetation overlap detection and correction using height + NDVI
2. Building buffer zone classification for nearby unclassified points
3. NDVI-based refinement for all classification types

Usage:
    python examples/example_geometric_rules_reclassification.py

Author: Data Processing Team
Date: October 16, 2025
"""

import logging
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_geometric_rules_reclassification():
    """
    Example demonstrating geometric rules for intelligent reclassification.
    """
    from ign_lidar.core.modules.reclassifier import OptimizedReclassifier
    from ign_lidar.io.wfs_ground_truth import DataFetcher
    
    logger.info("=" * 80)
    logger.info("Geometric Rules Reclassification Example")
    logger.info("=" * 80)
    
    # Configuration
    input_dir = Path("data/test_integration/enriched_tiles")
    output_dir = Path("data/test_output/geometric_rules_reclassified")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Geometric rules parameters
    config = {
        'use_geometric_rules': True,
        'ndvi_vegetation_threshold': 0.3,  # NDVI >= 0.3 = vegetation
        'ndvi_road_threshold': 0.15,       # NDVI <= 0.15 = road/impervious
        'road_vegetation_height_threshold': 2.0,  # 2m height threshold
        'building_buffer_distance': 2.0,   # 2m buffer around buildings
        'max_building_height_difference': 3.0,  # 3m height tolerance
        'verticality_threshold': 0.7,      # 0.7 verticality score threshold
        'verticality_search_radius': 1.0,  # 1m search radius
        'min_vertical_neighbors': 5,       # Min 5 neighbors
    }
    
    logger.info("\n📋 Configuration:")
    logger.info(f"  Input directory: {input_dir}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"\n🔧 Geometric Rules Settings:")
    logger.info(f"  Use geometric rules: {config['use_geometric_rules']}")
    logger.info(f"  NDVI vegetation threshold: {config['ndvi_vegetation_threshold']}")
    logger.info(f"  NDVI road threshold: {config['ndvi_road_threshold']}")
    logger.info(f"  Road-vegetation height separation: {config['road_vegetation_height_threshold']}m")
    logger.info(f"  Building buffer distance: {config['building_buffer_distance']}m")
    logger.info(f"  Max building height difference: {config['max_building_height_difference']}m")
    logger.info(f"  Verticality threshold: {config['verticality_threshold']}")
    logger.info(f"  Verticality search radius: {config['verticality_search_radius']}m")
    logger.info(f"  Min vertical neighbors: {config['min_vertical_neighbors']}")
    
    # Find LAZ files
    laz_files = list(input_dir.glob("*.laz"))
    
    if not laz_files:
        logger.warning(f"No LAZ files found in {input_dir}")
        logger.info("\nTo run this example:")
        logger.info("1. Place enriched LAZ files in data/test_integration/enriched_tiles/")
        logger.info("2. Ensure files have NDVI data for best results")
        logger.info("3. Run this script again")
        return
    
    logger.info(f"\n📂 Found {len(laz_files)} LAZ file(s) to process")
    
    # Initialize reclassifier with geometric rules
    logger.info("\n🚀 Initializing optimized reclassifier with geometric rules...")
    reclassifier = OptimizedReclassifier(
        chunk_size=100000,
        show_progress=True,
        acceleration_mode='auto',
        use_geometric_rules=config['use_geometric_rules'],
        ndvi_vegetation_threshold=config['ndvi_vegetation_threshold'],
        ndvi_road_threshold=config['ndvi_road_threshold'],
        road_vegetation_height_threshold=config['road_vegetation_height_threshold'],
        building_buffer_distance=config['building_buffer_distance'],
        max_building_height_difference=config['max_building_height_difference'],
        verticality_threshold=config['verticality_threshold'],
        verticality_search_radius=config['verticality_search_radius'],
        min_vertical_neighbors=config['min_vertical_neighbors']
    )
    
    # Initialize data fetcher
    logger.info("\n📡 Initializing ground truth data fetcher...")
    data_fetcher = DataFetcher(
        cache_dir=Path("data/cache/ground_truth"),
        use_cache=True
    )
    
    # Process each file
    for i, input_laz in enumerate(laz_files, 1):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing file {i}/{len(laz_files)}: {input_laz.name}")
        logger.info(f"{'=' * 80}")
        
        output_laz = output_dir / input_laz.name
        
        # Load file to get bbox
        import laspy
        las = laspy.read(str(input_laz))
        points = np.vstack([las.x, las.y, las.z]).T
        
        # Calculate bbox
        bbox = (
            float(points[:, 0].min()),
            float(points[:, 1].min()),
            float(points[:, 0].max()),
            float(points[:, 1].max())
        )
        
        logger.info(f"\n🗺️  Tile bounds: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
        
        # Fetch ground truth features
        logger.info("\n📥 Fetching ground truth features...")
        ground_truth = data_fetcher.fetch_all_features(bbox=bbox)
        
        # Log feature counts
        logger.info("\n📊 Ground truth features available:")
        for feature_type, gdf in ground_truth.items():
            if gdf is not None and len(gdf) > 0:
                logger.info(f"  {feature_type}: {len(gdf)} features")
        
        # Reclassify
        logger.info("\n🎯 Starting reclassification with geometric rules...")
        stats = reclassifier.reclassify_file(
            input_laz=input_laz,
            output_laz=output_laz,
            ground_truth_features=ground_truth
        )
        
        # Display detailed statistics
        logger.info("\n📈 Detailed Statistics:")
        logger.info(f"  Total points changed: {stats.get('total_changed', 0):,}")
        
        if 'road_vegetation_fixed' in stats:
            logger.info(f"  Road-vegetation conflicts fixed: {stats['road_vegetation_fixed']:,}")
        
        if 'building_buffer_added' in stats:
            logger.info(f"  Points added via building buffer: {stats['building_buffer_added']:,}")
        
        if 'verticality_buildings_added' in stats:
            logger.info(f"  Points added via verticality: {stats['verticality_buildings_added']:,}")
        
        if 'ndvi_refined' in stats:
            logger.info(f"  Points refined with NDVI: {stats['ndvi_refined']:,}")
        
        logger.info(f"\n✅ Successfully processed: {input_laz.name}")
        logger.info(f"   Output saved to: {output_laz}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Geometric rules reclassification complete!")
    logger.info("=" * 80)
    logger.info(f"\n📁 Output files saved in: {output_dir}")
    logger.info("\nGeometric rules applied:")
    logger.info("  ✓ Road-vegetation overlap detection (height + NDVI)")
    logger.info("  ✓ Building buffer zone classification")
    logger.info("  ✓ Verticality-based building detection")
    logger.info("  ✓ NDVI-based refinement for all classes")


def example_compare_with_without_rules():
    """
    Compare reclassification results with and without geometric rules.
    """
    from ign_lidar.core.modules.reclassifier import OptimizedReclassifier
    from ign_lidar.io.wfs_ground_truth import DataFetcher
    
    logger.info("=" * 80)
    logger.info("Comparison: With vs Without Geometric Rules")
    logger.info("=" * 80)
    
    input_dir = Path("data/test_integration/enriched_tiles")
    output_dir_without = Path("data/test_output/comparison_without_rules")
    output_dir_with = Path("data/test_output/comparison_with_rules")
    
    output_dir_without.mkdir(parents=True, exist_ok=True)
    output_dir_with.mkdir(parents=True, exist_ok=True)
    
    laz_files = list(input_dir.glob("*.laz"))[:1]  # Process first file only
    
    if not laz_files:
        logger.warning(f"No LAZ files found in {input_dir}")
        return
    
    input_laz = laz_files[0]
    logger.info(f"\n📂 Processing: {input_laz.name}")
    
    # Load file to get bbox
    import laspy
    las = laspy.read(str(input_laz))
    points = np.vstack([las.x, las.y, las.z]).T
    
    bbox = (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max())
    )
    
    # Fetch ground truth once
    data_fetcher = DataFetcher(
        cache_dir=Path("data/cache/ground_truth"),
        use_cache=True
    )
    ground_truth = data_fetcher.fetch_all_features(bbox=bbox)
    
    # Test WITHOUT geometric rules
    logger.info("\n" + "=" * 80)
    logger.info("Test 1: WITHOUT Geometric Rules")
    logger.info("=" * 80)
    
    reclassifier_without = OptimizedReclassifier(
        chunk_size=100000,
        show_progress=True,
        acceleration_mode='cpu',
        use_geometric_rules=False
    )
    
    stats_without = reclassifier_without.reclassify_file(
        input_laz=input_laz,
        output_laz=output_dir_without / input_laz.name,
        ground_truth_features=ground_truth
    )
    
    # Test WITH geometric rules
    logger.info("\n" + "=" * 80)
    logger.info("Test 2: WITH Geometric Rules")
    logger.info("=" * 80)
    
    reclassifier_with = OptimizedReclassifier(
        chunk_size=100000,
        show_progress=True,
        acceleration_mode='cpu',
        use_geometric_rules=True,
        ndvi_vegetation_threshold=0.3,
        ndvi_road_threshold=0.15,
        road_vegetation_height_threshold=2.0,
        building_buffer_distance=2.0
    )
    
    stats_with = reclassifier_with.reclassify_file(
        input_laz=input_laz,
        output_laz=output_dir_with / input_laz.name,
        ground_truth_features=ground_truth
    )
    
    # Compare results
    logger.info("\n" + "=" * 80)
    logger.info("📊 COMPARISON RESULTS")
    logger.info("=" * 80)
    
    logger.info(f"\nWithout geometric rules:")
    logger.info(f"  Total points changed: {stats_without.get('total_changed', 0):,}")
    
    logger.info(f"\nWith geometric rules:")
    logger.info(f"  Total points changed: {stats_with.get('total_changed', 0):,}")
    logger.info(f"  Road-vegetation fixed: {stats_with.get('road_vegetation_fixed', 0):,}")
    logger.info(f"  Building buffer added: {stats_with.get('building_buffer_added', 0):,}")
    logger.info(f"  NDVI refined: {stats_with.get('ndvi_refined', 0):,}")
    
    improvement = stats_with.get('total_changed', 0) - stats_without.get('total_changed', 0)
    logger.info(f"\n✨ Additional points refined with rules: {improvement:,}")
    
    logger.info(f"\n📁 Output files:")
    logger.info(f"  Without rules: {output_dir_without / input_laz.name}")
    logger.info(f"  With rules: {output_dir_with / input_laz.name}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("IGN LIDAR HD - Geometric Rules Reclassification Examples")
    print("=" * 80)
    print("\nAvailable examples:")
    print("  1. Basic geometric rules reclassification")
    print("  2. Comparison: with vs without geometric rules")
    print("\nSelect an example to run (1-2), or press Enter to run all:")
    
    choice = input("> ").strip()
    
    if choice == "1" or choice == "":
        example_geometric_rules_reclassification()
    
    if choice == "2" or choice == "":
        print("\n")
        example_compare_with_without_rules()
    
    if choice not in ["", "1", "2"]:
        print(f"Invalid choice: {choice}")
        print("Please select 1 or 2")
