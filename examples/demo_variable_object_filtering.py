#!/usr/bin/env python3
"""
Demo: Variable Object Filtering using DTM Heights

Demonstrates filtering of temporary/mobile objects:
- Vehicles on roads and parking
- Urban furniture (benches, poles, signs)
- Walls and fences (optional)

Uses RGE ALTI DTM to compute accurate height_above_ground.

Author: DTM Integration Enhancement
Date: October 19, 2025
Version: 5.2.1
"""

import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_vehicle_filtering():
    """Demonstrate vehicle filtering on a road."""
    logger.info("=" * 80)
    logger.info("DEMO 1: Vehicle Filtering on Road")
    logger.info("=" * 80)
    
    # Simulate a road with vehicles
    # Road surface at Z=100m, DTM at Z=99.5m → height_above_ground = 0.5m
    # Vehicles at Z=101.5-102.5m → height_above_ground = 2.0-3.0m
    
    n_road_points = 10000
    n_vehicle_points = 2000
    
    # Road points (close to ground)
    road_x = np.random.uniform(0, 50, n_road_points)
    road_y = np.random.uniform(0, 10, n_road_points)
    road_z = np.random.normal(100.0, 0.1, n_road_points)
    road_points = np.column_stack([road_x, road_y, road_z])
    road_classification = np.full(n_road_points, 11, dtype=np.uint8)  # Class 11 = Road
    road_height_dtm = np.random.uniform(0.3, 0.7, n_road_points)  # 0.3-0.7m above DTM
    
    # Vehicle points (elevated)
    vehicle_x = np.random.uniform(10, 40, n_vehicle_points)
    vehicle_y = np.random.uniform(2, 8, n_vehicle_points)
    vehicle_z = np.random.normal(101.8, 0.3, n_vehicle_points)  # Car height ~1.8m
    vehicle_points = np.column_stack([vehicle_x, vehicle_y, vehicle_z])
    vehicle_classification = np.full(n_vehicle_points, 11, dtype=np.uint8)  # Classified as road (wrong!)
    vehicle_height_dtm = np.random.uniform(1.4, 2.2, n_vehicle_points)  # 1.4-2.2m above DTM
    
    # Combine
    points = np.vstack([road_points, vehicle_points])
    classification = np.concatenate([road_classification, vehicle_classification])
    height_above_ground = np.concatenate([road_height_dtm, vehicle_height_dtm])
    
    logger.info(f"Initial state:")
    logger.info(f"  Total points: {len(points):,}")
    logger.info(f"  Road classification (11): {(classification == 11).sum():,}")
    logger.info(f"  Height range: {height_above_ground.min():.2f}m - {height_above_ground.max():.2f}m")
    
    # Apply filtering
    from ign_lidar.core.classification.variable_object_filter import VariableObjectFilter
    
    filter = VariableObjectFilter(
        filter_vehicles=True,
        vehicle_height_range=(0.8, 4.0),
        filter_urban_furniture=False,
        filter_walls=False,
        verbose=True
    )
    
    classification_filtered, stats = filter.filter_variable_objects(
        points=points,
        classification=classification,
        height_above_ground=height_above_ground
    )
    
    logger.info(f"\nAfter filtering:")
    logger.info(f"  Road classification (11): {(classification_filtered == 11).sum():,}")
    logger.info(f"  Unassigned (1): {(classification_filtered == 1).sum():,}")
    logger.info(f"  Vehicles filtered: {stats['vehicles_filtered']:,}")
    
    # Calculate accuracy
    expected_vehicles = n_vehicle_points
    detected_vehicles = stats['vehicles_filtered']
    accuracy = (detected_vehicles / expected_vehicles) * 100
    logger.info(f"\nAccuracy: {accuracy:.1f}% of vehicles detected")
    
    return classification_filtered, stats


def demo_urban_furniture_filtering():
    """Demonstrate urban furniture filtering."""
    logger.info("\n" + "=" * 80)
    logger.info("DEMO 2: Urban Furniture Filtering")
    logger.info("=" * 80)
    
    # Simulate parking lot with scattered furniture
    n_parking_points = 8000
    n_furniture_clusters = 5
    n_points_per_furniture = 30  # Small clusters
    
    # Parking surface
    parking_x = np.random.uniform(0, 40, n_parking_points)
    parking_y = np.random.uniform(0, 30, n_parking_points)
    parking_z = np.random.normal(100.0, 0.1, n_parking_points)
    parking_points = np.column_stack([parking_x, parking_y, parking_z])
    parking_classification = np.full(n_parking_points, 40, dtype=np.uint8)  # Class 40 = Parking
    parking_height_dtm = np.random.uniform(0.0, 0.5, n_parking_points)
    
    # Urban furniture (poles, signs, benches)
    furniture_points_list = []
    furniture_heights_list = []
    
    for i in range(n_furniture_clusters):
        # Random location
        center_x = np.random.uniform(5, 35)
        center_y = np.random.uniform(5, 25)
        
        # Small cluster around center
        fx = center_x + np.random.normal(0, 0.3, n_points_per_furniture)
        fy = center_y + np.random.normal(0, 0.3, n_points_per_furniture)
        fz = np.random.normal(101.5, 0.2, n_points_per_furniture)  # ~1.5m high object
        
        furniture_points_list.append(np.column_stack([fx, fy, fz]))
        furniture_heights_list.append(np.random.uniform(1.2, 1.8, n_points_per_furniture))
    
    furniture_points = np.vstack(furniture_points_list)
    furniture_classification = np.full(len(furniture_points), 40, dtype=np.uint8)
    furniture_height_dtm = np.concatenate(furniture_heights_list)
    
    # Combine
    points = np.vstack([parking_points, furniture_points])
    classification = np.concatenate([parking_classification, furniture_classification])
    height_above_ground = np.concatenate([parking_height_dtm, furniture_height_dtm])
    
    logger.info(f"Initial state:")
    logger.info(f"  Total points: {len(points):,}")
    logger.info(f"  Parking classification (40): {(classification == 40).sum():,}")
    logger.info(f"  Number of furniture objects: {n_furniture_clusters}")
    
    # Apply filtering
    from ign_lidar.core.classification.variable_object_filter import VariableObjectFilter
    
    filter = VariableObjectFilter(
        filter_vehicles=False,
        filter_urban_furniture=True,
        furniture_height_range=(0.5, 4.0),
        furniture_max_cluster_size=50,
        filter_walls=False,
        verbose=True
    )
    
    classification_filtered, stats = filter.filter_variable_objects(
        points=points,
        classification=classification,
        height_above_ground=height_above_ground
    )
    
    logger.info(f"\nAfter filtering:")
    logger.info(f"  Parking classification (40): {(classification_filtered == 40).sum():,}")
    logger.info(f"  Unassigned (1): {(classification_filtered == 1).sum():,}")
    logger.info(f"  Furniture filtered: {stats['furniture_filtered']:,}")
    
    return classification_filtered, stats


def demo_combined_filtering():
    """Demonstrate combined filtering scenario."""
    logger.info("\n" + "=" * 80)
    logger.info("DEMO 3: Combined Filtering (Vehicles + Furniture)")
    logger.info("=" * 80)
    
    # Realistic urban scene
    # - Road with cars
    # - Parking with cars and poles
    # - Sports field (clean, no filtering needed)
    
    # 1. Road with vehicles
    n_road = 5000
    n_cars_on_road = 800
    
    road_points = np.column_stack([
        np.random.uniform(0, 50, n_road),
        np.random.uniform(0, 8, n_road),
        np.random.normal(100.0, 0.1, n_road)
    ])
    road_class = np.full(n_road, 11, dtype=np.uint8)
    road_height = np.random.uniform(0.2, 0.6, n_road)
    
    cars_on_road = np.column_stack([
        np.random.uniform(10, 40, n_cars_on_road),
        np.random.uniform(2, 6, n_cars_on_road),
        np.random.normal(101.7, 0.3, n_cars_on_road)
    ])
    cars_on_road_class = np.full(n_cars_on_road, 11, dtype=np.uint8)
    cars_on_road_height = np.random.uniform(1.3, 2.1, n_cars_on_road)
    
    # 2. Parking with cars and poles
    n_parking = 6000
    n_cars_parked = 1200
    n_poles = 150
    
    parking_points = np.column_stack([
        np.random.uniform(0, 40, n_parking),
        np.random.uniform(10, 30, n_parking),
        np.random.normal(100.2, 0.1, n_parking)
    ])
    parking_class = np.full(n_parking, 40, dtype=np.uint8)
    parking_height = np.random.uniform(0.0, 0.5, n_parking)
    
    parked_cars = np.column_stack([
        np.random.uniform(5, 35, n_cars_parked),
        np.random.uniform(12, 28, n_cars_parked),
        np.random.normal(101.9, 0.3, n_cars_parked)
    ])
    parked_cars_class = np.full(n_cars_parked, 40, dtype=np.uint8)
    parked_cars_height = np.random.uniform(1.4, 2.2, n_cars_parked)
    
    poles = np.column_stack([
        np.random.uniform(5, 35, n_poles),
        np.random.uniform(12, 28, n_poles),
        np.random.normal(102.5, 0.2, n_poles)
    ])
    poles_class = np.full(n_poles, 40, dtype=np.uint8)
    poles_height = np.random.uniform(2.0, 3.0, n_poles)
    
    # 3. Sports field (clean reference)
    n_sports = 4000
    sports_points = np.column_stack([
        np.random.uniform(50, 90, n_sports),
        np.random.uniform(0, 30, n_sports),
        np.random.normal(99.8, 0.1, n_sports)
    ])
    sports_class = np.full(n_sports, 41, dtype=np.uint8)
    sports_height = np.random.uniform(0.0, 0.3, n_sports)
    
    # Combine all
    points = np.vstack([
        road_points, cars_on_road, parking_points, 
        parked_cars, poles, sports_points
    ])
    classification = np.concatenate([
        road_class, cars_on_road_class, parking_class,
        parked_cars_class, poles_class, sports_class
    ])
    height_above_ground = np.concatenate([
        road_height, cars_on_road_height, parking_height,
        parked_cars_height, poles_height, sports_height
    ])
    
    logger.info(f"Urban scene composition:")
    logger.info(f"  Total points: {len(points):,}")
    logger.info(f"  Roads (11): {(classification == 11).sum():,}")
    logger.info(f"  Parking (40): {(classification == 40).sum():,}")
    logger.info(f"  Sports (41): {(classification == 41).sum():,}")
    logger.info(f"  Expected to filter:")
    logger.info(f"    - Cars on road: {n_cars_on_road}")
    logger.info(f"    - Parked cars: {n_cars_parked}")
    logger.info(f"    - Poles: {n_poles}")
    logger.info(f"    - Total: {n_cars_on_road + n_cars_parked + n_poles}")
    
    # Apply combined filtering
    from ign_lidar.core.classification.variable_object_filter import VariableObjectFilter
    
    filter = VariableObjectFilter(
        filter_vehicles=True,
        vehicle_height_range=(0.8, 4.0),
        filter_urban_furniture=True,
        furniture_height_range=(0.5, 4.0),
        furniture_max_cluster_size=50,
        filter_walls=False,
        verbose=True
    )
    
    classification_filtered, stats = filter.filter_variable_objects(
        points=points,
        classification=classification,
        height_above_ground=height_above_ground
    )
    
    logger.info(f"\nAfter filtering:")
    logger.info(f"  Roads (11): {(classification_filtered == 11).sum():,}")
    logger.info(f"  Parking (40): {(classification_filtered == 40).sum():,}")
    logger.info(f"  Sports (41): {(classification_filtered == 41).sum():,} (unchanged)")
    logger.info(f"  Unassigned (1): {(classification_filtered == 1).sum():,}")
    logger.info(f"\nFiltering statistics:")
    logger.info(f"  Vehicles: {stats['vehicles_filtered']:,}")
    logger.info(f"  Furniture: {stats['furniture_filtered']:,}")
    logger.info(f"  Total: {stats['total_filtered']:,}")
    
    # Calculate detection rate
    expected_total = n_cars_on_road + n_cars_parked + n_poles
    detected_total = stats['total_filtered']
    detection_rate = (detected_total / expected_total) * 100
    logger.info(f"\nDetection rate: {detection_rate:.1f}%")
    
    return classification_filtered, stats


def main():
    """Run all demos."""
    logger.info("🚗 Variable Object Filtering Demo")
    logger.info("Uses DTM-based height_above_ground to filter temporary objects\n")
    
    try:
        # Demo 1: Vehicle filtering
        demo_vehicle_filtering()
        
        # Demo 2: Urban furniture
        demo_urban_furniture_filtering()
        
        # Demo 3: Combined filtering
        demo_combined_filtering()
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ All demos completed successfully!")
        logger.info("=" * 80)
        logger.info("\nTo use in production:")
        logger.info("1. Enable RGE ALTI: data_sources.rge_alti.enabled = true")
        logger.info("2. Enable filtering: variable_object_filtering.enabled = true")
        logger.info("3. Adjust thresholds as needed in config YAML")
        logger.info("\nSee: docs/DTM_VARIABLE_OBJECTS_FILTERING.md")
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Make sure the package is installed: pip install -e .")
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
