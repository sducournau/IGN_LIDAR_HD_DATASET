"""
Variable Object Filter - DTM-based filtering of temporary/mobile objects

This module filters variable objects from point clouds using height above DTM:
- Vehicles (cars, trucks, buses) on roads and parking
- Urban furniture (benches, poles, signs)
- Walls and fences (optional)
- Other temporary objects

Uses RGE ALTI DTM as ground reference to compute accurate heights.

Author: DTM Integration Enhancement
Date: October 19, 2025
Version: 5.2.1
"""

import numpy as np
from typing import Dict, Optional, Tuple, Set
from scipy.spatial import cKDTree
import logging

logger = logging.getLogger(__name__)


class VariableObjectFilter:
    """
    Filter temporary/variable objects using DTM-based height.
    
    Filters:
    - Vehicles on roads/parking
    - Urban furniture
    - Temporary objects
    - Walls/fences (optional)
    
    Configuration:
        variable_object_filtering:
          enabled: true
          filter_vehicles: true
          vehicle_height_range: [0.8, 4.0]
          filter_urban_furniture: true
          furniture_height_range: [0.5, 4.0]
          furniture_max_cluster_size: 50
          filter_walls: false
          wall_height_range: [0.5, 2.5]
          wall_min_verticality: 0.8
    """
    
    # ASPRS class codes for filtering context
    ROAD_CLASS = 11
    RAILWAY_CLASS = 10
    PARKING_CLASS = 40
    SPORTS_CLASS = 41
    CEMETERY_CLASS = 42
    
    def __init__(
        self,
        filter_vehicles: bool = True,
        filter_urban_furniture: bool = True,
        filter_walls: bool = False,
        vehicle_height_range: Tuple[float, float] = (0.8, 4.0),
        furniture_height_range: Tuple[float, float] = (0.5, 4.0),
        furniture_max_cluster_size: int = 50,
        wall_height_range: Tuple[float, float] = (0.5, 2.5),
        wall_min_verticality: float = 0.8,
        reclassify_to: int = 1,  # Unassigned
        create_vehicle_class: bool = False,
        vehicle_class_code: int = 18,
        verbose: bool = True
    ):
        """
        Initialize variable object filter.
        
        Args:
            filter_vehicles: Filter vehicles on roads/parking
            filter_urban_furniture: Filter urban furniture (small isolated objects)
            filter_walls: Filter walls and fences
            vehicle_height_range: Min/max height for vehicles (meters)
            furniture_height_range: Min/max height for furniture (meters)
            furniture_max_cluster_size: Max points in furniture cluster
            wall_height_range: Min/max height for walls (meters)
            wall_min_verticality: Min verticality for wall detection
            reclassify_to: Target class for filtered objects (default: 1=unassigned)
            create_vehicle_class: Create separate vehicle class instead of unassigned
            vehicle_class_code: Class code for vehicles if create_vehicle_class=True
            verbose: Log detailed statistics
        """
        self.filter_vehicles = filter_vehicles
        self.filter_urban_furniture = filter_urban_furniture
        self.filter_walls = filter_walls
        
        self.vehicle_height_min, self.vehicle_height_max = vehicle_height_range
        self.furniture_height_min, self.furniture_height_max = furniture_height_range
        self.furniture_max_cluster = furniture_max_cluster_size
        self.wall_height_min, self.wall_height_max = wall_height_range
        self.wall_min_verticality = wall_min_verticality
        
        self.reclassify_to = reclassify_to
        self.create_vehicle_class = create_vehicle_class
        self.vehicle_class_code = vehicle_class_code
        self.verbose = verbose
        
        logger.info("🔍 VariableObjectFilter initialized")
        if self.filter_vehicles:
            logger.info(f"  🚗 Vehicle filtering: {self.vehicle_height_min}-{self.vehicle_height_max}m")
        if self.filter_urban_furniture:
            logger.info(f"  🪑 Furniture filtering: {self.furniture_height_min}-{self.furniture_height_max}m, max {self.furniture_max_cluster} pts")
        if self.filter_walls:
            logger.info(f"  🧱 Wall filtering: {self.wall_height_min}-{self.wall_height_max}m, verticality>{self.wall_min_verticality}")
    
    def filter_variable_objects(
        self,
        points: np.ndarray,
        classification: np.ndarray,
        height_above_ground: np.ndarray,
        features: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Apply variable object filtering.
        
        Args:
            points: Point cloud [N, 3] (X, Y, Z)
            classification: ASPRS classifications [N]
            height_above_ground: Height above DTM [N]
            features: Geometric features (verticality, planarity, etc.)
            
        Returns:
            Tuple of:
            - Modified classification [N]
            - Statistics dict with counts per filter type
        """
        classification_filtered = classification.copy()
        stats = {
            'vehicles_filtered': 0,
            'furniture_filtered': 0,
            'walls_filtered': 0,
            'total_filtered': 0
        }
        
        logger.info("  🔍 Filtering variable objects using DTM heights...")
        
        # 1. Filter vehicles
        if self.filter_vehicles:
            n_veh = self._filter_vehicles(
                classification_filtered,
                height_above_ground
            )
            stats['vehicles_filtered'] = n_veh
            if n_veh > 0:
                logger.info(f"  🚗 Filtered {n_veh:,} vehicle points")
        
        # 2. Filter urban furniture
        if self.filter_urban_furniture:
            n_furn = self._filter_urban_furniture(
                points,
                classification_filtered,
                height_above_ground
            )
            stats['furniture_filtered'] = n_furn
            if n_furn > 0:
                logger.info(f"  🪑 Filtered {n_furn:,} urban furniture points")
        
        # 3. Filter walls/fences
        if self.filter_walls and features is not None:
            n_walls = self._filter_walls(
                classification_filtered,
                height_above_ground,
                features
            )
            stats['walls_filtered'] = n_walls
            if n_walls > 0:
                logger.info(f"  🧱 Filtered {n_walls:,} wall/fence points")
        
        stats['total_filtered'] = sum([
            stats['vehicles_filtered'],
            stats['furniture_filtered'],
            stats['walls_filtered']
        ])
        
        if stats['total_filtered'] > 0:
            pct = (stats['total_filtered'] / len(classification)) * 100
            logger.info(f"  ✅ Total variable objects filtered: {stats['total_filtered']:,} points ({pct:.2f}%)")
        else:
            logger.info("  ℹ️  No variable objects filtered")
        
        return classification_filtered, stats
    
    def _filter_vehicles(
        self,
        classification: np.ndarray,
        height: np.ndarray
    ) -> int:
        """
        Filter vehicles on roads, parking, railways.
        
        Strategy:
        - Roads (11): height 0.8-4.0m → likely vehicles
        - Parking (40): height 0.5-4.0m → likely parked vehicles
        - Railways (10): height 1.5-5.0m → likely trains/wagons
        
        Args:
            classification: Point classifications
            height: Height above DTM
            
        Returns:
            Number of points filtered
        """
        # Define transport surfaces where vehicles appear
        transport_classes = {
            self.ROAD_CLASS,      # Roads
            self.PARKING_CLASS,   # Parking
            self.RAILWAY_CLASS    # Railways (trains)
        }
        
        transport_mask = np.isin(classification, list(transport_classes))
        
        # Adjust height thresholds based on surface type
        vehicle_mask = np.zeros(len(classification), dtype=bool)
        
        # Roads: typical car height
        road_mask = classification == self.ROAD_CLASS
        vehicle_mask |= (
            road_mask &
            (height >= self.vehicle_height_min) &
            (height <= self.vehicle_height_max)
        )
        
        # Parking: slightly lower threshold (parked cars may be lower)
        parking_mask = classification == self.PARKING_CLASS
        vehicle_mask |= (
            parking_mask &
            (height >= max(0.5, self.vehicle_height_min - 0.3)) &
            (height <= self.vehicle_height_max)
        )
        
        # Railways: higher threshold (trains are taller)
        railway_mask = classification == self.RAILWAY_CLASS
        vehicle_mask |= (
            railway_mask &
            (height >= max(1.5, self.vehicle_height_min)) &
            (height <= max(5.0, self.vehicle_height_max + 1.0))
        )
        
        n_vehicles = vehicle_mask.sum()
        
        if n_vehicles > 0:
            # Reclassify to vehicle class or unassigned
            if self.create_vehicle_class:
                classification[vehicle_mask] = self.vehicle_class_code
                if self.verbose:
                    logger.debug(f"    Created vehicle class {self.vehicle_class_code}: {n_vehicles:,} points")
            else:
                classification[vehicle_mask] = self.reclassify_to
                if self.verbose:
                    logger.debug(f"    Reclassified to {self.reclassify_to}: {n_vehicles:,} points")
        
        return n_vehicles
    
    def _filter_urban_furniture(
        self,
        points: np.ndarray,
        classification: np.ndarray,
        height: np.ndarray
    ) -> int:
        """
        Filter urban furniture (small isolated elevated objects).
        
        Strategy:
        - Identify elevated points on artificial surfaces
        - Find small clusters (< max_cluster_size points)
        - Filter as furniture (benches, poles, signs, etc.)
        
        Args:
            points: Point cloud
            classification: Point classifications
            height: Height above DTM
            
        Returns:
            Number of points filtered
        """
        # Artificial surfaces where furniture appears
        artificial_classes = {
            self.ROAD_CLASS,
            self.PARKING_CLASS,
            self.SPORTS_CLASS,
            self.CEMETERY_CLASS
        }
        
        artificial_mask = np.isin(classification, list(artificial_classes))
        
        # Height range for furniture
        furniture_height_mask = (
            (height >= self.furniture_height_min) &
            (height <= self.furniture_height_max)
        )
        
        # Combine: elevated points on artificial surfaces
        candidate_mask = artificial_mask & furniture_height_mask
        candidate_indices = np.where(candidate_mask)[0]
        
        if len(candidate_indices) == 0:
            return 0
        
        # Build spatial index for cluster analysis
        candidate_points = points[candidate_indices]
        
        try:
            tree = cKDTree(candidate_points[:, :2])  # XY only
            
            # Count neighbors within 1m radius for each point
            neighbors_count = tree.query_ball_point(
                candidate_points[:, :2],
                r=1.0,
                return_length=True
            )
            
            # Small clusters = furniture (isolated objects)
            small_cluster_mask = neighbors_count < self.furniture_max_cluster
            furniture_indices = candidate_indices[small_cluster_mask]
            
            if len(furniture_indices) > 0:
                classification[furniture_indices] = self.reclassify_to
                if self.verbose:
                    logger.debug(f"    Small clusters (<{self.furniture_max_cluster} pts): {len(furniture_indices):,} points")
            
            return len(furniture_indices)
            
        except Exception as e:
            logger.warning(f"    Failed to build KDTree for furniture filtering: {e}")
            return 0
    
    def _filter_walls(
        self,
        classification: np.ndarray,
        height: np.ndarray,
        features: Dict[str, np.ndarray]
    ) -> int:
        """
        Filter walls and fences.
        
        Strategy:
        - Height: 0.5-2.5m (typical wall/fence height)
        - Verticality: > 0.8 (vertical structures)
        - Planarity: > 0.7 (smooth surfaces)
        
        Args:
            classification: Point classifications
            height: Height above DTM
            features: Geometric features (verticality, planarity)
            
        Returns:
            Number of points filtered
        """
        verticality = features.get('verticality')
        planarity = features.get('planarity')
        
        if verticality is None or planarity is None:
            logger.warning("    Verticality or planarity not available - skipping wall filtering")
            return 0
        
        # Wall criteria: height + geometry
        wall_mask = (
            (height >= self.wall_height_min) &
            (height <= self.wall_height_max) &
            (verticality >= self.wall_min_verticality) &
            (planarity >= 0.7)
        )
        
        n_walls = wall_mask.sum()
        
        if n_walls > 0:
            # Option 1: Create wall class (61)
            # Option 2: Reclassify to unassigned
            classification[wall_mask] = 61  # Wall class (or self.reclassify_to)
            if self.verbose:
                logger.debug(f"    Vertical structures: {n_walls:,} points")
        
        return n_walls


def apply_variable_object_filtering(
    points: np.ndarray,
    classification: np.ndarray,
    height_above_ground: np.ndarray,
    config: Dict,
    features: Optional[Dict[str, np.ndarray]] = None
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Apply variable object filtering using DTM heights.
    
    This is the main entry point for integration into the pipeline.
    
    Usage in processor.py:
        from ..classification.variable_object_filter import apply_variable_object_filtering
        
        labels_v, filter_stats = apply_variable_object_filtering(
            points=points_v,
            classification=labels_v,
            height_above_ground=height_above_ground,
            config=self.config,
            features=all_features
        )
    
    Args:
        points: Point cloud [N, 3]
        classification: ASPRS classifications [N]
        height_above_ground: Height above DTM [N]
        config: Full configuration dict
        features: Geometric features (verticality, planarity, etc.)
        
    Returns:
        Tuple of:
        - Modified classification [N]
        - Statistics dict
    """
    # Get filtering configuration
    filter_config = config.get('variable_object_filtering', {})
    
    if not filter_config.get('enabled', False):
        logger.debug("  Variable object filtering disabled in config")
        return classification, {'total_filtered': 0}
    
    # Check if height_above_ground is available
    if height_above_ground is None:
        logger.warning("  ⚠️  height_above_ground not available - skipping variable object filtering")
        logger.warning("      Enable DTM computation: features.compute_height_above_ground = true")
        return classification, {'total_filtered': 0}
    
    # Create filter instance
    filter = VariableObjectFilter(
        filter_vehicles=filter_config.get('filter_vehicles', True),
        filter_urban_furniture=filter_config.get('filter_urban_furniture', True),
        filter_walls=filter_config.get('filter_walls', False),
        vehicle_height_range=tuple(filter_config.get('vehicle_height_range', [0.8, 4.0])),
        furniture_height_range=tuple(filter_config.get('furniture_height_range', [0.5, 4.0])),
        furniture_max_cluster_size=filter_config.get('furniture_max_cluster_size', 50),
        wall_height_range=tuple(filter_config.get('wall_height_range', [0.5, 2.5])),
        wall_min_verticality=filter_config.get('wall_min_verticality', 0.8),
        reclassify_to=filter_config.get('reclassify_to', 1),
        create_vehicle_class=filter_config.get('create_vehicle_class', False),
        vehicle_class_code=filter_config.get('vehicle_class_code', 18),
        verbose=filter_config.get('verbose', True)
    )
    
    # Apply filtering
    classification_filtered, stats = filter.filter_variable_objects(
        points=points,
        classification=classification,
        height_above_ground=height_above_ground,
        features=features
    )
    
    return classification_filtered, stats


__all__ = [
    'VariableObjectFilter',
    'apply_variable_object_filtering'
]
