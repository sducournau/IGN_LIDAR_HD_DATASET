"""
Example: Enhanced Building Classification with Near-Vertical Wall Detection

This script demonstrates the improved building detection with:
1. Near-vertical wall detection (plans verticaux / murs)
2. Extended buffer to capture points up to wall boundaries
3. Multi-source building polygon fusion

Author: Building Detection Enhancement
Date: October 19, 2025
"""

import numpy as np
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_wall_detection_asprs():
    """
    Example 1: ASPRS mode with near-vertical wall detection
    
    Shows how to use the enhanced building clusterer to detect
    walls (murs) and extend building polygons to boundaries.
    """
    from ign_lidar.core.classification.building_clustering import BuildingClusterer
    from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
    from ign_lidar.features.compute.features import compute_normals
    
    # Load point cloud (example)
    points = load_point_cloud("tile.laz")  # [N, 3]
    labels = load_labels("tile.laz")       # [N]
    
    # Compute normals (required for wall detection)
    logger.info("Computing normals for wall detection...")
    normals = compute_normals(points, k_neighbors=20)
    
    # Compute verticality: 1 - |nz|
    verticality = 1.0 - np.abs(normals[:, 2])
    
    # Fetch building polygons
    bbox = compute_bbox(points)
    gt_fetcher = IGNGroundTruthFetcher()
    buildings_gdf = gt_fetcher.fetch_buildings(bbox)
    
    # Create enhanced clusterer with wall detection
    clusterer = BuildingClusterer(
        use_centroid_attraction=True,
        attraction_radius=5.0,
        min_points_per_building=10,
        polygon_buffer=0.5,          # Base buffer
        wall_buffer=0.3,             # Additional buffer for walls (ASPRS)
        detect_near_vertical_walls=True  # Enable wall detection
    )
    
    # Cluster with wall detection
    logger.info("Clustering buildings with wall detection...")
    building_ids, clusters = clusterer.cluster_points_by_buildings(
        points=points,
        buildings_gdf=buildings_gdf,
        labels=labels,
        building_classes=[6],  # ASPRS building code
        normals=normals,       # Required for wall detection
        verticality=verticality  # Optional (computed if None)
    )
    
    # Analyze results
    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS: {len(clusters)} buildings detected")
    logger.info(f"{'='*60}")
    
    for i, cluster in enumerate(clusters[:5]):  # Show first 5
        # Separate walls and roofs
        cluster_normals = normals[cluster.point_indices]
        
        # Walls: near-vertical (|nz| < 0.3)
        wall_mask = np.abs(cluster_normals[:, 2]) < 0.3
        n_walls = wall_mask.sum()
        
        # Roofs: near-horizontal (|nz| > 0.8)
        roof_mask = np.abs(cluster_normals[:, 2]) > 0.8
        n_roofs = roof_mask.sum()
        
        logger.info(f"\nBuilding {i+1} (ID: {cluster.building_id}):")
        logger.info(f"  Total points: {cluster.n_points:,}")
        logger.info(f"  Wall points: {n_walls:,} ({100*n_walls/cluster.n_points:.1f}%)")
        logger.info(f"  Roof points: {n_roofs:,} ({100*n_roofs/cluster.n_points:.1f}%)")
        logger.info(f"  Wall/Roof ratio: {n_walls/n_roofs:.2f}" if n_roofs > 0 else "  Wall/Roof ratio: N/A")
        logger.info(f"  Volume: {cluster.volume:.1f} m³")
        logger.info(f"  Height: {cluster.height_mean:.1f}m (max: {cluster.height_max:.1f}m)")
    
    return building_ids, clusters


def example_wall_detection_lod2():
    """
    Example 2: LOD2 mode for building reconstruction
    
    Shows how to use LOD2 mode with enhanced wall detection
    for detailed building element separation.
    """
    from ign_lidar.core.classification.building_detection import (
        BuildingDetector, BuildingDetectionConfig, BuildingDetectionMode
    )
    from ign_lidar.features.compute.features import compute_normals, compute_planarity
    
    # Load data
    points = load_point_cloud("tile.laz")
    labels = np.ones(len(points), dtype=np.uint8)  # Initialize
    
    # Compute features
    logger.info("Computing geometric features...")
    normals = compute_normals(points, k_neighbors=20)
    planarity = compute_planarity(points, k_neighbors=20)
    
    # Compute height and verticality
    height = compute_height_above_ground(points)
    verticality = 1.0 - np.abs(normals[:, 2])
    
    # Create LOD2 detector with enhanced wall detection
    config = BuildingDetectionConfig(mode=BuildingDetectionMode.LOD2)
    
    # Print configuration
    logger.info(f"\nLOD2 Configuration:")
    logger.info(f"  Wall verticality min: {config.wall_verticality_min}")
    logger.info(f"  Wall planarity min: {config.wall_planarity_min}")
    logger.info(f"  Wall buffer: {config.wall_buffer_distance}m")
    logger.info(f"  Roof horizontality min: {config.roof_horizontality_min}")
    
    # Detect buildings
    detector = BuildingDetector(config)
    labels_updated, stats = detector.detect_buildings(
        labels=labels,
        height=height,
        planarity=planarity,
        verticality=verticality,
        normals=normals
    )
    
    # Analyze LOD2 results
    LOD2_WALL = 0
    LOD2_ROOF_FLAT = 1
    LOD2_ROOF_GABLE = 2
    
    n_walls = (labels_updated == LOD2_WALL).sum()
    n_roofs_flat = (labels_updated == LOD2_ROOF_FLAT).sum()
    n_roofs_gable = (labels_updated == LOD2_ROOF_GABLE).sum()
    
    logger.info(f"\nLOD2 Detection Results:")
    logger.info(f"  Walls (0): {n_walls:,} points")
    logger.info(f"  Flat Roofs (1): {n_roofs_flat:,} points")
    logger.info(f"  Gable Roofs (2): {n_roofs_gable:,} points")
    logger.info(f"  Total building: {stats.get('total_building', 0):,} points")
    
    return labels_updated, stats


def example_plane_detection():
    """
    Example 3: Comprehensive plane detection (horizontal, vertical, inclined)
    
    Demonstrates detection of:
    - Horizontal planes (toits plats, terrasses)
    - Vertical planes (murs, façades) 
    - Inclined planes (toits en pente, versants)
    - Architectural elements (lucarnes, cheminées, balcons)
    """
    from ign_lidar.core.classification.plane_detection import (
        PlaneDetector, detect_architectural_elements
    )
    from ign_lidar.features.compute.features import compute_normals, compute_planarity
    
    logger.info("="*70)
    logger.info("PLANE DETECTION: Horizontal, Vertical, and Inclined Planes")
    logger.info("="*70)
    
    # Load data
    logger.info("\n[1/4] Loading point cloud...")
    points = load_point_cloud("tile.laz")
    
    # Compute features
    logger.info("\n[2/4] Computing features...")
    normals = compute_normals(points, k_neighbors=20)
    planarity = compute_planarity(points, k_neighbors=20)
    height = compute_height_above_ground(points)
    
    # Create plane detector
    detector = PlaneDetector(
        horizontal_angle_max=10.0,    # ±10° = horizontal
        vertical_angle_min=75.0,       # ≥75° = vertical
        inclined_angle_min=15.0,       # 15-70° = inclined
        inclined_angle_max=70.0,
        min_points_per_plane=50
    )
    
    # Detect all plane types
    logger.info("\n[3/4] Detecting planes...")
    planes = detector.detect_all_planes(points, normals, planarity, height)
    
    # Analyze results
    logger.info("\n" + "="*70)
    logger.info("PLANE DETECTION RESULTS")
    logger.info("="*70)
    
    # Horizontal planes (roofs)
    horizontal_planes = planes['horizontal']
    logger.info(f"\n🏢 HORIZONTAL PLANES (Toits plats, terrasses): {len(horizontal_planes)}")
    for i, plane in enumerate(horizontal_planes[:3]):
        logger.info(f"  Plane {i+1}:")
        logger.info(f"    Points: {plane.n_points:,}")
        logger.info(f"    Height: {plane.height_mean:.1f}m ±{plane.height_std:.2f}m")
        logger.info(f"    Area: {plane.area:.1f} m²")
        logger.info(f"    Planarity: {plane.planarity:.3f}")
        logger.info(f"    Angle: {plane.orientation_angle:.1f}° from horizontal")
    
    # Vertical planes (walls)
    vertical_planes = planes['vertical']
    logger.info(f"\n🧱 VERTICAL PLANES (Murs, façades): {len(vertical_planes)}")
    for i, plane in enumerate(vertical_planes[:3]):
        logger.info(f"  Plane {i+1}:")
        logger.info(f"    Points: {plane.n_points:,}")
        logger.info(f"    Height: {plane.height_mean:.1f}m ±{plane.height_std:.2f}m")
        logger.info(f"    Area: {plane.area:.1f} m²")
        logger.info(f"    Planarity: {plane.planarity:.3f}")
        logger.info(f"    Angle: {plane.orientation_angle:.1f}° from horizontal")
    
    # Inclined planes (pitched roofs)
    inclined_planes = planes['inclined']
    logger.info(f"\n🏠 INCLINED PLANES (Toits en pente): {len(inclined_planes)}")
    for i, plane in enumerate(inclined_planes[:3]):
        logger.info(f"  Plane {i+1}:")
        logger.info(f"    Points: {plane.n_points:,}")
        logger.info(f"    Height: {plane.height_mean:.1f}m ±{plane.height_std:.2f}m")
        logger.info(f"    Area: {plane.area:.1f} m²")
        logger.info(f"    Planarity: {plane.planarity:.3f}")
        logger.info(f"    Angle: {plane.orientation_angle:.1f}° from horizontal")
    
    # Classify roof types
    logger.info("\n" + "="*70)
    logger.info("ROOF TYPE CLASSIFICATION")
    logger.info("="*70)
    
    roof_types = detector.classify_roof_types(horizontal_planes, inclined_planes)
    
    for roof_type, roof_planes in roof_types.items():
        if roof_planes:
            total_points = sum(p.n_points for p in roof_planes)
            total_area = sum(p.area for p in roof_planes)
            logger.info(f"\n{roof_type.upper()} ROOF:")
            logger.info(f"  Planes: {len(roof_planes)}")
            logger.info(f"  Total points: {total_points:,}")
            logger.info(f"  Total area: {total_area:.1f} m²")
    
    # Detect architectural elements
    logger.info("\n[4/4] Detecting architectural elements...")
    elements = detect_architectural_elements(
        points, normals, planarity, height, planes
    )
    
    logger.info("\n" + "="*70)
    logger.info("ARCHITECTURAL ELEMENTS")
    logger.info("="*70)
    
    for elem_type, elem_list in elements.items():
        if elem_list:
            total_points = sum(len(indices) for indices in elem_list)
            logger.info(f"\n{elem_type.upper()}: {len(elem_list)} detected")
            logger.info(f"  Total points: {total_points:,}")
    
    logger.info("\n" + "="*70)
    logger.info("PLANE DETECTION COMPLETE!")
    logger.info("="*70)
    
    return planes, roof_types, elements


def example_complete_pipeline():
    """
    Example 4: Complete pipeline with all enhancements
    
    Demonstrates full workflow:
    1. Load point cloud
    2. Compute features (normals, planarity, NDVI)
    3. Fetch ground truth (buildings, cadastre)
    4. Detect buildings with wall detection
    5. Cluster by building polygons with extended buffers
    6. Analyze and save results
    """
    from ign_lidar.core.classification.building_clustering import cluster_buildings_multi_source
    from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
    from ign_lidar.features.compute.features import compute_normals
    
    logger.info("="*70)
    logger.info("COMPLETE PIPELINE: Building Detection with Near-Vertical Walls")
    logger.info("="*70)
    
    # Step 1: Load data
    logger.info("\n[1/6] Loading point cloud...")
    points = load_point_cloud("tile.laz")
    labels = load_labels("tile.laz")
    colors = load_colors("tile.laz")
    bbox = compute_bbox(points)
    
    logger.info(f"  Points: {len(points):,}")
    logger.info(f"  Bbox: {bbox}")
    
    # Step 2: Compute features
    logger.info("\n[2/6] Computing geometric features...")
    normals = compute_normals(points, k_neighbors=20)
    verticality = 1.0 - np.abs(normals[:, 2])
    
    # Statistics on verticality
    n_vertical = (verticality > 0.6).sum()
    logger.info(f"  Near-vertical points (walls): {n_vertical:,} ({100*n_vertical/len(points):.1f}%)")
    
    # Step 3: Fetch ground truth
    logger.info("\n[3/6] Fetching ground truth (BD TOPO + Cadastre)...")
    gt_fetcher = IGNGroundTruthFetcher()
    ground_truth = gt_fetcher.fetch_all_features(
        bbox,
        include_buildings=True,
        include_cadastre=True
    )
    
    buildings_gdf = ground_truth.get('buildings')
    cadastre_gdf = ground_truth.get('cadastre')
    
    logger.info(f"  Buildings (BD TOPO): {len(buildings_gdf) if buildings_gdf else 0}")
    logger.info(f"  Cadastre parcels: {len(cadastre_gdf) if cadastre_gdf else 0}")
    
    # Step 4: Multi-source clustering with wall detection
    logger.info("\n[4/6] Clustering buildings with wall detection...")
    building_ids, clusters = cluster_buildings_multi_source(
        points=points,
        ground_truth_features=ground_truth,
        labels=labels,
        building_classes=[6],  # ASPRS building
        normals=normals,
        verticality=verticality,
        use_centroid_attraction=True,
        attraction_radius=5.0,
        polygon_buffer=0.5,
        wall_buffer=0.3,  # Extended buffer for walls
        detect_near_vertical_walls=True
    )
    
    logger.info(f"  Clusters created: {len(clusters)}")
    logger.info(f"  Points assigned: {(building_ids >= 0).sum():,}")
    
    # Step 5: Analyze clusters
    logger.info("\n[5/6] Analyzing building clusters...")
    
    total_walls = 0
    total_roofs = 0
    
    for cluster in clusters:
        # Get normals for this cluster
        cluster_normals = normals[cluster.point_indices]
        
        # Count walls and roofs
        wall_mask = np.abs(cluster_normals[:, 2]) < 0.3
        roof_mask = np.abs(cluster_normals[:, 2]) > 0.8
        
        total_walls += wall_mask.sum()
        total_roofs += roof_mask.sum()
    
    logger.info(f"  Total wall points: {total_walls:,}")
    logger.info(f"  Total roof points: {total_roofs:,}")
    logger.info(f"  Average wall/roof ratio: {total_walls/total_roofs:.2f}" if total_roofs > 0 else "  N/A")
    
    # Step 6: Save results
    logger.info("\n[6/6] Saving results...")
    
    # Update labels with building IDs
    for cluster in clusters:
        labels[cluster.point_indices] = 6  # ASPRS building
    
    save_classified_las("output.laz", points, labels, colors)
    
    # Save cluster metadata
    import json
    metadata = {
        'n_clusters': len(clusters),
        'total_points': len(points),
        'assigned_points': (building_ids >= 0).sum(),
        'wall_points': total_walls,
        'roof_points': total_roofs,
        'clusters': [
            {
                'id': c.building_id,
                'n_points': c.n_points,
                'volume': float(c.volume),
                'height_mean': float(c.height_mean),
                'height_max': float(c.height_max)
            }
            for c in clusters
        ]
    }
    
    with open("building_clusters.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("  Saved: output.laz")
    logger.info("  Saved: building_clusters.json")
    
    logger.info("\n" + "="*70)
    logger.info("COMPLETE! Enhanced building detection with wall detection finished.")
    logger.info("="*70)
    
    return building_ids, clusters, metadata


# ============================================================================
# Helper functions (stubs - replace with actual implementations)
# ============================================================================

def load_point_cloud(filepath):
    """Load point cloud from LAS/LAZ file."""
    # TODO: Implement with laspy
    pass

def load_labels(filepath):
    """Load classification labels from LAS/LAZ file."""
    # TODO: Implement with laspy
    pass

def load_colors(filepath):
    """Load RGB colors from LAS/LAZ file."""
    # TODO: Implement with laspy
    pass

def compute_bbox(points):
    """Compute bounding box from points."""
    minx, miny, minz = points.min(axis=0)
    maxx, maxy, maxz = points.max(axis=0)
    return (minx, miny, maxx, maxy)

def compute_height_above_ground(points):
    """Compute height above ground."""
    # TODO: Implement with ground detection or RGE ALTI
    return points[:, 2] - points[:, 2].min()

def save_classified_las(filepath, points, labels, colors):
    """Save classified point cloud to LAS/LAZ."""
    # TODO: Implement with laspy
    pass


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Run examples
    print("\n" + "="*80)
    print("ENHANCED BUILDING CLASSIFICATION EXAMPLES")
    print("="*80)
    
    print("\n\nExample 1: ASPRS with Wall Detection")
    print("-" * 70)
    print("Detects near-vertical walls (murs) with extended polygon buffers")
    print("Buffer: 0.5m (base) + 0.3m (walls) = 0.8m total")
    # example_wall_detection_asprs()
    
    print("\n\nExample 2: LOD2 with Enhanced Walls")
    print("-" * 70)
    print("Separates walls, flat roofs, and sloped roofs for building reconstruction")
    # example_wall_detection_lod2()
    
    print("\n\nExample 3: Comprehensive Plane Detection")
    print("-" * 70)
    print("Detects horizontal, vertical, and inclined planes")
    print("Classifies roof types: flat, gable, hip, complex")
    print("Identifies architectural elements: balconies, chimneys, dormers, parapets")
    # example_plane_detection()
    
    print("\n\nExample 4: Complete Pipeline")
    print("-" * 70)
    print("Full workflow with all enhancements:")
    print("  - Load data")
    print("  - Compute features (normals, verticality)")
    print("  - Fetch ground truth (BD TOPO + Cadastre)")
    print("  - Cluster buildings with wall detection")
    print("  - Analyze and save results")
    # example_complete_pipeline()
    
    print("\n\n" + "="*80)
    print("Note: Uncomment examples to run. Requires point cloud data.")
    print("="*80)
    
    # Quick feature overview
    print("\n\n📋 FEATURE SUMMARY:")
    print("-" * 80)
    print("✅ Near-vertical wall detection (verticality > 0.6)")
    print("✅ Extended polygon buffers (0.3-0.5m for walls)")
    print("✅ Horizontal plane detection (toits plats, terrasses)")
    print("✅ Vertical plane detection (murs, façades)")
    print("✅ Inclined plane detection (toits en pente)")
    print("✅ Roof type classification (flat/gable/hip/complex)")
    print("✅ Architectural element detection (balconies, chimneys, dormers)")
    print("✅ Multi-source fusion (BD TOPO buildings + Cadastre)")
    print("✅ Centroid-based attraction for spatial coherence")
    print("-" * 80)
