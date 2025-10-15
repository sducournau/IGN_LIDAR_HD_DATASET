"""
Example: Using Enhanced Transport Overlay System

This example demonstrates how to use the new transport enhancement features:
1. Adaptive buffering with curvature awareness
2. Fast spatial indexing for large point clouds
3. Quality metrics and confidence scoring

Author: Transport Enhancement Team
Date: October 15, 2025
"""

from pathlib import Path
import numpy as np
import laspy

# Import enhancement modules
from ign_lidar.core.modules.transport_enhancement import (
    AdaptiveTransportBuffer,
    AdaptiveBufferConfig,
    SpatialTransportClassifier,
    SpatialIndexConfig,
    TransportCoverageStats
)
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher


def example_1_adaptive_buffering():
    """Example 1: Adaptive buffering with curvature awareness."""
    
    print("=" * 70)
    print("Example 1: Adaptive Buffering")
    print("=" * 70)
    
    # Configure adaptive buffering
    config = AdaptiveBufferConfig(
        curvature_aware=True,
        curvature_factor=0.2,  # 20% width increase on curves
        min_curve_radius=50.0,  # Minimum radius for max adjustment
        type_specific_tolerance=True,
        intersection_enhancement=True
    )
    
    # Create buffer engine
    buffer_engine = AdaptiveTransportBuffer(config=config)
    
    # Fetch ground truth roads
    fetcher = IGNGroundTruthFetcher(cache_dir=Path("D:/ign/cache/ground_truth"))
    
    # Example bbox (Lambert 93 coordinates)
    bbox = (650000, 6860000, 651000, 6861000)  # 1km x 1km
    
    print(f"\nFetching roads from BD TOPO® for bbox {bbox}...")
    roads_gdf = fetcher.fetch_roads_with_polygons(bbox, default_width=4.0)
    
    if roads_gdf is not None and len(roads_gdf) > 0:
        print(f"Retrieved {len(roads_gdf)} roads")
        
        # Apply adaptive buffering
        print("\nApplying adaptive buffering...")
        enhanced_roads = buffer_engine.process_roads(roads_gdf)
        
        print(f"Enhanced {len(enhanced_roads)} roads with:")
        print(f"  - Curvature-aware variable width")
        print(f"  - Road-type specific tolerances")
        print(f"  - Intersection detection")
        
        # Show statistics
        if 'width_m' in enhanced_roads.columns:
            widths = enhanced_roads['width_m'].values
            print(f"\nRoad widths: {widths.min():.1f}m - {widths.max():.1f}m (avg: {widths.mean():.1f}m)")
        
        if 'tolerance_m' in enhanced_roads.columns:
            tolerances = enhanced_roads['tolerance_m'].values
            print(f"Tolerances:  {tolerances.min():.1f}m - {tolerances.max():.1f}m (avg: {tolerances.mean():.1f}m)")
    else:
        print("No roads found in this bbox")


def example_2_fast_spatial_classification():
    """Example 2: Fast classification using spatial indexing."""
    
    print("\n" + "=" * 70)
    print("Example 2: Fast Spatial Classification")
    print("=" * 70)
    
    # Load point cloud
    tile_path = Path("D:/ign/preprocessed/asprs/enriched_tiles/enriched/test_tile.laz")
    
    if not tile_path.exists():
        print(f"Tile not found: {tile_path}")
        return
    
    print(f"\nLoading point cloud from {tile_path.name}...")
    las = laspy.read(tile_path)
    
    points = np.vstack([las.x, las.y, las.z]).T
    labels = np.array(las.classification)
    
    print(f"Loaded {len(points):,} points")
    
    # Calculate bbox for ground truth fetch
    bbox = (
        float(points[:, 0].min() - 50),
        float(points[:, 1].min() - 50),
        float(points[:, 0].max() + 50),
        float(points[:, 1].max() + 50)
    )
    
    # Fetch ground truth
    print(f"\nFetching ground truth for bbox...")
    fetcher = IGNGroundTruthFetcher(cache_dir=Path("D:/ign/cache/ground_truth"))
    
    roads_gdf = fetcher.fetch_roads_with_polygons(bbox)
    railways_gdf = fetcher.fetch_railways_with_polygons(bbox)
    
    if roads_gdf is None or len(roads_gdf) == 0:
        print("No roads found")
        return
    
    print(f"Retrieved {len(roads_gdf)} roads, {len(railways_gdf) if railways_gdf else 0} railways")
    
    # Configure spatial indexing
    config = SpatialIndexConfig(
        enabled=True,
        index_type="rtree",
        cache_index=True
    )
    
    # Create spatial classifier
    print("\nBuilding spatial index...")
    classifier = SpatialTransportClassifier(config=config)
    
    # Index features
    classifier.index_roads(roads_gdf)
    if railways_gdf is not None and len(railways_gdf) > 0:
        classifier.index_railways(railways_gdf)
    
    # Fast classification
    print(f"\nClassifying {len(points):,} points...")
    labels_enhanced = classifier.classify_points_fast(
        points=points,
        labels=labels,
        asprs_code_road=11,
        asprs_code_rail=10
    )
    
    # Statistics
    n_road_points = (labels_enhanced == 11).sum()
    n_rail_points = (labels_enhanced == 10).sum()
    n_transport_total = n_road_points + n_rail_points
    
    print(f"\nClassification results:")
    print(f"  Road points:     {n_road_points:,}")
    print(f"  Railway points:  {n_rail_points:,}")
    print(f"  Total transport: {n_transport_total:,} ({100*n_transport_total/len(points):.1f}%)")
    
    # Save enhanced tile
    output_path = tile_path.parent / f"{tile_path.stem}_enhanced_fast.laz"
    las.classification = labels_enhanced
    las.write(output_path)
    print(f"\nSaved enhanced tile to: {output_path}")


def example_3_quality_metrics():
    """Example 3: Generate quality metrics and statistics."""
    
    print("\n" + "=" * 70)
    print("Example 3: Quality Metrics & Statistics")
    print("=" * 70)
    
    # Create sample statistics
    stats = TransportCoverageStats(
        n_roads_processed=89,
        n_road_points_classified=234567,
        avg_points_per_road=2636.4,
        road_width_range=(2.5, 18.0),
        road_types_detected={11: 200000, 32: 15000, 33: 19567},
        n_railways_processed=12,
        n_rail_points_classified=45678,
        avg_points_per_railway=3806.5,
        railway_track_counts=[1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1],
        avg_confidence=0.87,
        low_confidence_ratio=0.08,
        overlap_detections=234,
        centerline_coverage=0.94,
        buffer_utilization=0.76
    )
    
    # Generate text report
    print("\n" + stats.generate_report())
    
    # Export to JSON
    json_str = stats.to_json()
    print(f"\nJSON export ({len(json_str)} characters):")
    print(json_str[:200] + "...")


def example_4_complete_pipeline():
    """Example 4: Complete pipeline with all enhancements."""
    
    print("\n" + "=" * 70)
    print("Example 4: Complete Enhanced Pipeline")
    print("=" * 70)
    
    # Input/output paths
    input_dir = Path("D:/ign/raw")
    output_dir = Path("D:/ign/preprocessed/asprs_enhanced")
    
    print(f"\nInput:  {input_dir}")
    print(f"Output: {output_dir}")
    
    # Configuration
    print("\nConfiguration:")
    print("  ✅ Adaptive buffering (curvature-aware)")
    print("  ✅ Spatial indexing (R-tree)")
    print("  ✅ Quality metrics (confidence scoring)")
    print("  ✅ Road-type specific tolerances")
    print("  ✅ Intersection enhancement")
    
    # Workflow
    print("\nWorkflow:")
    print("  1. Load configuration")
    print("  2. Fetch ground truth from BD TOPO®")
    print("  3. Apply adaptive buffering")
    print("  4. Build spatial index")
    print("  5. Fast point classification")
    print("  6. Calculate confidence scores")
    print("  7. Generate quality reports")
    print("  8. Export enhanced LAZ + reports")
    
    print("\n" + "=" * 70)
    print("To run complete pipeline, use:")
    print("  python -m ign_lidar.cli.commands.process \\")
    print("    --config-file configs/multiscale/config_asprs_preprocessing.yaml \\")
    print("    transport_detection.adaptive_buffering.enabled=true \\")
    print("    transport_detection.spatial_indexing.enabled=true \\")
    print("    transport_detection.quality_metrics.enabled=true")
    print("=" * 70)


if __name__ == "__main__":
    print("\n🚂🛣️ TRANSPORT OVERLAY ENHANCEMENT EXAMPLES\n")
    
    # Run examples
    try:
        example_1_adaptive_buffering()
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    try:
        example_2_fast_spatial_classification()
    except Exception as e:
        print(f"Example 2 failed: {e}")
    
    try:
        example_3_quality_metrics()
    except Exception as e:
        print(f"Example 3 failed: {e}")
    
    try:
        example_4_complete_pipeline()
    except Exception as e:
        print(f"Example 4 failed: {e}")
    
    print("\n✅ Examples completed!")
