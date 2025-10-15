"""
3D Shape Grammar for Building Classification and Decomposition

This module implements a shape grammar system for hierarchical decomposition
and classification of buildings and their components. The grammar uses
production rules to recognize and refine architectural elements at multiple
levels of detail.

Shape Grammar Approach:
1. Building detection and envelope extraction
2. Decomposition into major components (walls, roofs, foundation)
3. Refinement of sub-elements (windows, doors, dormers, chimneys)
4. Validation and consistency checking

References:
- Stiny, G. (1980). "Introduction to shape and shape grammars"
- Müller et al. (2006). "Procedural modeling of buildings" (CGA Shape)
- Wonka et al. (2003). "Instant Architecture"

Author: IGN LiDAR HD Dataset Team
Date: October 15, 2025
"""

import logging
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Grammar Vocabulary and Symbols
# ============================================================================

class ArchitecturalSymbol(Enum):
    """Symbols representing architectural elements in the grammar."""
    
    # Top-level
    BUILDING = "Building"
    
    # Major components
    ENVELOPE = "Envelope"
    FOUNDATION = "Foundation"
    WALLS = "Walls"
    ROOF = "Roof"
    
    # Wall sub-elements
    WALL_SEGMENT = "WallSegment"
    FACADE = "Facade"
    WINDOW = "Window"
    DOOR = "Door"
    BALCONY = "Balcony"
    
    # Roof sub-elements
    ROOF_PLANE = "RoofPlane"
    ROOF_FLAT = "RoofFlat"
    ROOF_GABLE = "RoofGable"
    ROOF_HIP = "RoofHip"
    ROOF_MANSARD = "RoofMansard"
    DORMER = "Dormer"
    CHIMNEY = "Chimney"
    SKYLIGHT = "Skylight"
    ROOF_EDGE = "RoofEdge"
    
    # Special elements
    OVERHANG = "Overhang"
    PILLAR = "Pillar"
    CORNICE = "Cornice"
    BALUSTRADE = "Balustrade"
    
    # Primitives
    POINT_CLOUD = "PointCloud"
    PLANAR_SURFACE = "PlanarSurface"
    VERTICAL_SURFACE = "VerticalSurface"
    HORIZONTAL_SURFACE = "HorizontalSurface"
    EDGE = "Edge"
    CORNER = "Corner"


@dataclass
class Shape:
    """
    Represents a geometric shape with attributes in the grammar.
    
    A shape consists of:
    - Symbol (what it represents)
    - Geometry (point indices, parameters)
    - Attributes (dimensions, orientation, etc.)
    - Confidence score
    """
    
    symbol: ArchitecturalSymbol
    point_indices: np.ndarray  # Indices of points belonging to this shape
    
    # Geometric attributes
    centroid: Optional[np.ndarray] = None
    bbox_min: Optional[np.ndarray] = None
    bbox_max: Optional[np.ndarray] = None
    normal: Optional[np.ndarray] = None  # For planar shapes
    orientation: Optional[float] = None  # Rotation angle
    
    # Shape attributes
    area: Optional[float] = None
    height: Optional[float] = None
    width: Optional[float] = None
    depth: Optional[float] = None
    
    # Classification
    confidence: float = 0.0
    parent: Optional['Shape'] = None
    children: List['Shape'] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        n_points = len(self.point_indices) if self.point_indices is not None else 0
        return (f"Shape({self.symbol.value}, {n_points} points, "
                f"confidence={self.confidence:.2f})")


@dataclass
class ProductionRule:
    """
    A production rule in the shape grammar.
    
    Format: LHS → RHS [conditions]
    
    Example: Building → Foundation + Walls + Roof [height > 2.5m]
    """
    
    name: str
    left_hand_side: ArchitecturalSymbol  # Input symbol
    right_hand_side: List[ArchitecturalSymbol]  # Output symbols
    
    # Conditions for rule application
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Priority (higher = applied first)
    priority: int = 0
    
    def __repr__(self):
        rhs = " + ".join([s.value for s in self.right_hand_side])
        return f"{self.left_hand_side.value} → {rhs}"


# ============================================================================
# Grammar Rules Definition
# ============================================================================

class BuildingGrammar:
    """
    Defines the production rules for building decomposition.
    
    The grammar is hierarchical, decomposing buildings from coarse to fine:
    Level 0: Building detection
    Level 1: Major components (foundation, walls, roof)
    Level 2: Component refinement (wall segments, roof planes)
    Level 3: Detailed elements (windows, doors, dormers, chimneys)
    """
    
    def __init__(self):
        """Initialize grammar with production rules."""
        self.rules = []
        self._define_level0_rules()  # Building detection
        self._define_level1_rules()  # Major components
        self._define_level2_rules()  # Component refinement
        self._define_level3_rules()  # Detailed elements
    
    def _define_level0_rules(self):
        """Level 0: Building detection and envelope extraction."""
        
        # Rule 0.1: Detect building from point cloud
        self.rules.append(ProductionRule(
            name="detect_building",
            left_hand_side=ArchitecturalSymbol.POINT_CLOUD,
            right_hand_side=[ArchitecturalSymbol.BUILDING],
            conditions={
                'min_height': 2.5,  # Minimum building height (m)
                'min_points': 100,  # Minimum number of points
                'max_ground_distance': 0.5,  # Maximum distance from ground
                'planarity_threshold': 0.5,  # Some planar surfaces
            },
            priority=100
        ))
        
        # Rule 0.2: Extract building envelope
        self.rules.append(ProductionRule(
            name="extract_envelope",
            left_hand_side=ArchitecturalSymbol.BUILDING,
            right_hand_side=[ArchitecturalSymbol.ENVELOPE],
            conditions={
                'convex_hull': True,  # Use convex hull
                'buffer': 0.5,  # Buffer distance for envelope
            },
            priority=90
        ))
    
    def _define_level1_rules(self):
        """Level 1: Decompose building into major components."""
        
        # Rule 1.1: Building → Foundation + Walls + Roof
        self.rules.append(ProductionRule(
            name="decompose_building_full",
            left_hand_side=ArchitecturalSymbol.BUILDING,
            right_hand_side=[
                ArchitecturalSymbol.FOUNDATION,
                ArchitecturalSymbol.WALLS,
                ArchitecturalSymbol.ROOF
            ],
            conditions={
                'has_foundation': True,
                'foundation_height_max': 1.5,
                'wall_height_min': 2.0,
            },
            priority=80
        ))
        
        # Rule 1.2: Building → Walls + Roof (no visible foundation)
        self.rules.append(ProductionRule(
            name="decompose_building_simple",
            left_hand_side=ArchitecturalSymbol.BUILDING,
            right_hand_side=[
                ArchitecturalSymbol.WALLS,
                ArchitecturalSymbol.ROOF
            ],
            conditions={
                'has_foundation': False,
            },
            priority=75
        ))
    
    def _define_level2_rules(self):
        """Level 2: Refine major components."""
        
        # Rule 2.1: Walls → Multiple wall segments
        self.rules.append(ProductionRule(
            name="segment_walls",
            left_hand_side=ArchitecturalSymbol.WALLS,
            right_hand_side=[ArchitecturalSymbol.WALL_SEGMENT],  # Multiple instances
            conditions={
                'min_segment_length': 2.0,
                'verticality_threshold': 0.7,
            },
            priority=70
        ))
        
        # Rule 2.2: Roof → Roof planes by type
        self.rules.append(ProductionRule(
            name="classify_roof_flat",
            left_hand_side=ArchitecturalSymbol.ROOF,
            right_hand_side=[ArchitecturalSymbol.ROOF_FLAT],
            conditions={
                'horizontality': 0.9,  # Very horizontal
                'planarity': 0.8,
            },
            priority=65
        ))
        
        self.rules.append(ProductionRule(
            name="classify_roof_gable",
            left_hand_side=ArchitecturalSymbol.ROOF,
            right_hand_side=[ArchitecturalSymbol.ROOF_GABLE],
            conditions={
                'num_planes': 2,
                'planes_meet_at_ridge': True,
                'symmetry': 0.8,
            },
            priority=65
        ))
        
        self.rules.append(ProductionRule(
            name="classify_roof_hip",
            left_hand_side=ArchitecturalSymbol.ROOF,
            right_hand_side=[ArchitecturalSymbol.ROOF_HIP],
            conditions={
                'num_planes': [3, 4, 5, 6],  # Multiple planes
                'planes_meet_at_ridge': True,
            },
            priority=65
        ))
        
        self.rules.append(ProductionRule(
            name="classify_roof_mansard",
            left_hand_side=ArchitecturalSymbol.ROOF,
            right_hand_side=[ArchitecturalSymbol.ROOF_MANSARD],
            conditions={
                'has_two_slopes': True,
                'steep_lower_slope': True,
            },
            priority=65
        ))
    
    def _define_level3_rules(self):
        """Level 3: Extract detailed elements."""
        
        # Rule 3.1: Wall segment → Facade with openings
        self.rules.append(ProductionRule(
            name="detect_windows",
            left_hand_side=ArchitecturalSymbol.WALL_SEGMENT,
            right_hand_side=[
                ArchitecturalSymbol.FACADE,
                ArchitecturalSymbol.WINDOW
            ],
            conditions={
                'has_depth_variation': True,
                'opening_size_range': (0.5, 2.5),  # Window size range (m)
                'rectangular_shape': True,
            },
            priority=60
        ))
        
        self.rules.append(ProductionRule(
            name="detect_doors",
            left_hand_side=ArchitecturalSymbol.WALL_SEGMENT,
            right_hand_side=[
                ArchitecturalSymbol.FACADE,
                ArchitecturalSymbol.DOOR
            ],
            conditions={
                'ground_level': True,
                'height_range': (1.8, 2.5),
                'width_range': (0.8, 1.5),
            },
            priority=60
        ))
        
        self.rules.append(ProductionRule(
            name="detect_balcony",
            left_hand_side=ArchitecturalSymbol.WALL_SEGMENT,
            right_hand_side=[ArchitecturalSymbol.BALCONY],
            conditions={
                'protrudes_from_wall': True,
                'horizontal_surface': True,
                'has_railing': True,
            },
            priority=55
        ))
        
        # Rule 3.2: Roof → Roof elements
        self.rules.append(ProductionRule(
            name="detect_dormer",
            left_hand_side=ArchitecturalSymbol.ROOF,
            right_hand_side=[ArchitecturalSymbol.DORMER],
            conditions={
                'protrudes_from_roof': True,
                'has_window': True,
                'has_roof': True,
            },
            priority=55
        ))
        
        self.rules.append(ProductionRule(
            name="detect_chimney",
            left_hand_side=ArchitecturalSymbol.ROOF,
            right_hand_side=[ArchitecturalSymbol.CHIMNEY],
            conditions={
                'vertical_structure': True,
                'extends_above_roof': True,
                'small_footprint': True,
                'height_above_roof_min': 0.5,
            },
            priority=55
        ))
        
        self.rules.append(ProductionRule(
            name="detect_skylight",
            left_hand_side=ArchitecturalSymbol.ROOF,
            right_hand_side=[ArchitecturalSymbol.SKYLIGHT],
            conditions={
                'in_roof_plane': True,
                'different_reflectivity': True,
                'rectangular_shape': True,
            },
            priority=50
        ))
    
    def get_applicable_rules(
        self,
        shape: Shape,
        level: Optional[int] = None
    ) -> List[ProductionRule]:
        """
        Get production rules applicable to a given shape.
        
        Args:
            shape: Input shape to transform
            level: Optional grammar level to restrict to
        
        Returns:
            List of applicable rules, sorted by priority
        """
        applicable = []
        
        for rule in self.rules:
            # Check if left-hand side matches
            if rule.left_hand_side != shape.symbol:
                continue
            
            # Check level restriction
            if level is not None:
                rule_level = self._get_rule_level(rule)
                if rule_level != level:
                    continue
            
            # Check conditions
            if self._check_conditions(rule, shape):
                applicable.append(rule)
        
        # Sort by priority (descending)
        applicable.sort(key=lambda r: r.priority, reverse=True)
        
        return applicable
    
    def _get_rule_level(self, rule: ProductionRule) -> int:
        """Determine the grammar level of a rule."""
        if rule.priority >= 90:
            return 0
        elif rule.priority >= 70:
            return 1
        elif rule.priority >= 60:
            return 2
        else:
            return 3
    
    def _check_conditions(self, rule: ProductionRule, shape: Shape) -> bool:
        """Check if rule conditions are satisfied for a shape."""
        # This is a simplified check - full implementation would evaluate
        # geometric conditions based on shape attributes
        return True  # Placeholder


# ============================================================================
# Grammar Parser and Application
# ============================================================================

class GrammarParser:
    """
    Applies shape grammar rules to parse and decompose building point clouds.
    
    The parser:
    1. Starts with a building point cloud
    2. Iteratively applies production rules
    3. Builds a hierarchical shape tree
    4. Returns classified sub-elements
    """
    
    def __init__(
        self,
        grammar: Optional[BuildingGrammar] = None,
        max_iterations: int = 10,
        min_confidence: float = 0.5
    ):
        """
        Initialize grammar parser.
        
        Args:
            grammar: Grammar to use (default: BuildingGrammar)
            max_iterations: Maximum parsing iterations
            min_confidence: Minimum confidence to accept a derivation
        """
        self.grammar = grammar or BuildingGrammar()
        self.max_iterations = max_iterations
        self.min_confidence = min_confidence
        
        logger.info(f"Initialized GrammarParser with {len(self.grammar.rules)} rules")
    
    def parse(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        features: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Parse building point cloud using shape grammar.
        
        Args:
            points: Point coordinates [N, 3]
            labels: Initial classification labels [N]
            features: Dictionary of features (height, normals, planarity, etc.)
        
        Returns:
            Tuple of (refined_labels, derivation_tree)
        """
        logger.info(f"Parsing {len(points):,} points with shape grammar")
        
        # Step 1: Initialize with building detection
        building_mask = self._detect_buildings(points, labels, features)
        
        if not np.any(building_mask):
            logger.warning("No buildings detected")
            return labels, {}
        
        refined_labels = labels.copy()
        derivation_tree = {}
        
        # Process each building separately
        building_indices = self._segment_buildings(points, building_mask)
        
        for building_id, indices in enumerate(building_indices):
            logger.info(f"Processing building {building_id + 1}/{len(building_indices)} "
                       f"({len(indices):,} points)")
            
            # Create initial building shape
            building_shape = Shape(
                symbol=ArchitecturalSymbol.BUILDING,
                point_indices=indices,
                centroid=points[indices].mean(axis=0),
                confidence=1.0
            )
            
            # Parse building hierarchically
            building_tree = self._parse_shape(
                building_shape,
                points,
                features
            )
            
            # Update labels based on derivation
            refined_labels = self._apply_derivation(
                refined_labels,
                building_tree,
                points
            )
            
            derivation_tree[f"building_{building_id}"] = building_tree
        
        # Statistics
        n_refined = np.sum(refined_labels != labels)
        logger.info(f"Grammar parsing complete: {n_refined:,} points refined")
        
        return refined_labels, derivation_tree
    
    def _detect_buildings(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        features: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Detect building points from initial classification."""
        # Simple heuristic: points classified as building or high enough
        building_mask = (labels == 6) | (labels == 0)  # ASPRS building or LOD2 wall
        
        # Additional filtering by height
        if 'height' in features:
            height = features['height']
            building_mask &= (height > 2.5)
        
        return building_mask
    
    def _segment_buildings(
        self,
        points: np.ndarray,
        building_mask: np.ndarray
    ) -> List[np.ndarray]:
        """
        Segment building points into individual buildings.
        
        Uses connected component analysis in 2D.
        """
        try:
            from scipy.spatial import cKDTree
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import connected_components
        except ImportError:
            # Fallback: treat all as one building
            logger.warning("scipy not available, treating all points as one building")
            return [np.where(building_mask)[0]]
        
        building_points = points[building_mask]
        building_indices = np.where(building_mask)[0]
        
        if len(building_points) == 0:
            return []
        
        # Build adjacency graph (2D, horizontal plane)
        tree = cKDTree(building_points[:, :2])
        
        # Find neighbors within threshold
        threshold = 2.0  # meters
        neighbors = tree.query_ball_tree(tree, threshold)
        
        # Build sparse adjacency matrix
        n = len(building_points)
        rows = []
        cols = []
        for i, neighs in enumerate(neighbors):
            for j in neighs:
                if i != j:
                    rows.append(i)
                    cols.append(j)
        
        data = np.ones(len(rows))
        adj_matrix = csr_matrix((data, (rows, cols)), shape=(n, n))
        
        # Find connected components
        n_components, component_labels = connected_components(
            adj_matrix,
            directed=False
        )
        
        # Group indices by component
        segments = []
        for comp_id in range(n_components):
            comp_mask = component_labels == comp_id
            comp_indices = building_indices[comp_mask]
            
            # Filter out very small components
            if len(comp_indices) >= 50:
                segments.append(comp_indices)
        
        logger.info(f"Segmented into {len(segments)} buildings")
        
        return segments
    
    def _parse_shape(
        self,
        shape: Shape,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        level: int = 0
    ) -> Shape:
        """
        Recursively parse a shape by applying grammar rules.
        
        Args:
            shape: Shape to parse
            points: All point coordinates
            features: Feature dictionary
            level: Current grammar level
        
        Returns:
            Parsed shape with children
        """
        if level >= 4:  # Maximum depth
            return shape
        
        # Get applicable rules
        applicable_rules = self.grammar.get_applicable_rules(shape, level=level)
        
        if not applicable_rules:
            return shape
        
        # Apply best rule
        best_rule = applicable_rules[0]
        logger.debug(f"Applying rule: {best_rule.name} at level {level}")
        
        # Derive child shapes
        children = self._derive_shapes(
            shape,
            best_rule,
            points,
            features
        )
        
        # Recursively parse children
        for child in children:
            parsed_child = self._parse_shape(
                child,
                points,
                features,
                level=level + 1
            )
            shape.children.append(parsed_child)
        
        return shape
    
    def _derive_shapes(
        self,
        parent: Shape,
        rule: ProductionRule,
        points: np.ndarray,
        features: Dict[str, np.ndarray]
    ) -> List[Shape]:
        """
        Apply production rule to derive child shapes.
        
        This method implements the geometric operations for each rule type.
        """
        children = []
        parent_points = points[parent.point_indices]
        
        # Get features for parent points
        parent_features = {
            key: features[key][parent.point_indices]
            for key in features.keys()
        }
        
        # Dispatch based on rule name
        if rule.name == "decompose_building_full":
            children = self._decompose_building_full(
                parent, parent_points, parent_features
            )
        
        elif rule.name == "decompose_building_simple":
            children = self._decompose_building_simple(
                parent, parent_points, parent_features
            )
        
        elif rule.name == "segment_walls":
            children = self._segment_walls(
                parent, parent_points, parent_features
            )
        
        elif rule.name.startswith("classify_roof"):
            children = self._classify_roof(
                parent, parent_points, parent_features, rule
            )
        
        elif rule.name in ["detect_windows", "detect_doors", "detect_balcony"]:
            children = self._detect_wall_elements(
                parent, parent_points, parent_features, rule
            )
        
        elif rule.name in ["detect_dormer", "detect_chimney", "detect_skylight"]:
            children = self._detect_roof_elements(
                parent, parent_points, parent_features, rule
            )
        
        return children
    
    def _decompose_building_full(
        self,
        parent: Shape,
        points: np.ndarray,
        features: Dict[str, np.ndarray]
    ) -> List[Shape]:
        """Decompose building into foundation, walls, and roof."""
        children = []
        
        height = features.get('height')
        if height is None:
            return children
        
        normals = features.get('normals')
        
        # Foundation: low points
        foundation_mask = height < 1.5
        if np.any(foundation_mask):
            foundation_indices = parent.point_indices[foundation_mask]
            children.append(Shape(
                symbol=ArchitecturalSymbol.FOUNDATION,
                point_indices=foundation_indices,
                centroid=points[foundation_mask].mean(axis=0),
                confidence=0.8,
                parent=parent
            ))
        
        # Walls: vertical surfaces at medium height
        if normals is not None:
            verticality = 1.0 - np.abs(normals[:, 2])  # Low Z component = vertical
            walls_mask = (height >= 1.5) & (height < height.max() - 1.0) & (verticality > 0.7)
            
            if np.any(walls_mask):
                walls_indices = parent.point_indices[walls_mask]
                children.append(Shape(
                    symbol=ArchitecturalSymbol.WALLS,
                    point_indices=walls_indices,
                    centroid=points[walls_mask].mean(axis=0),
                    confidence=0.9,
                    parent=parent
                ))
        
        # Roof: horizontal surfaces at top
        if normals is not None:
            horizontality = np.abs(normals[:, 2])  # High Z component = horizontal
            roof_threshold = height.max() - 1.0
            roof_mask = (height >= roof_threshold) & (horizontality > 0.5)
            
            if np.any(roof_mask):
                roof_indices = parent.point_indices[roof_mask]
                children.append(Shape(
                    symbol=ArchitecturalSymbol.ROOF,
                    point_indices=roof_indices,
                    centroid=points[roof_mask].mean(axis=0),
                    confidence=0.9,
                    parent=parent
                ))
        
        return children
    
    def _decompose_building_simple(
        self,
        parent: Shape,
        points: np.ndarray,
        features: Dict[str, np.ndarray]
    ) -> List[Shape]:
        """Decompose building into walls and roof (no foundation)."""
        # Similar to full but without foundation
        children = []
        
        height = features.get('height')
        normals = features.get('normals')
        
        if height is None or normals is None:
            return children
        
        # Determine roof threshold
        roof_threshold = np.percentile(height, 75)
        
        # Walls: lower parts with vertical surfaces
        verticality = 1.0 - np.abs(normals[:, 2])
        walls_mask = (height < roof_threshold) & (verticality > 0.7)
        
        if np.any(walls_mask):
            walls_indices = parent.point_indices[walls_mask]
            children.append(Shape(
                symbol=ArchitecturalSymbol.WALLS,
                point_indices=walls_indices,
                confidence=0.85,
                parent=parent
            ))
        
        # Roof: upper parts
        roof_mask = height >= roof_threshold
        if np.any(roof_mask):
            roof_indices = parent.point_indices[roof_mask]
            children.append(Shape(
                symbol=ArchitecturalSymbol.ROOF,
                point_indices=roof_indices,
                confidence=0.85,
                parent=parent
            ))
        
        return children
    
    def _segment_walls(
        self,
        parent: Shape,
        points: np.ndarray,
        features: Dict[str, np.ndarray]
    ) -> List[Shape]:
        """Segment walls into individual wall segments."""
        # Placeholder: returns parent as single segment
        # Full implementation would use RANSAC or region growing
        return [Shape(
            symbol=ArchitecturalSymbol.WALL_SEGMENT,
            point_indices=parent.point_indices,
            confidence=0.7,
            parent=parent
        )]
    
    def _classify_roof(
        self,
        parent: Shape,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        rule: ProductionRule
    ) -> List[Shape]:
        """Classify roof type based on geometry."""
        # Determine roof symbol from rule
        roof_symbol = rule.right_hand_side[0]
        
        return [Shape(
            symbol=roof_symbol,
            point_indices=parent.point_indices,
            confidence=0.8,
            parent=parent
        )]
    
    def _detect_wall_elements(
        self,
        parent: Shape,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        rule: ProductionRule
    ) -> List[Shape]:
        """Detect windows, doors, balconies on wall segments."""
        # Placeholder: simplified detection
        children = []
        
        # This would use depth analysis, regularity detection, etc.
        # For now, return empty list
        
        return children
    
    def _detect_roof_elements(
        self,
        parent: Shape,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        rule: ProductionRule
    ) -> List[Shape]:
        """Detect dormers, chimneys, skylights on roofs."""
        # Placeholder: simplified detection
        children = []
        
        height = features.get('height')
        if height is None:
            return children
        
        # Detect chimneys: high vertical structures on roof
        if rule.name == "detect_chimney":
            # Find points significantly above roof plane
            roof_height = height.mean()
            chimney_mask = height > (roof_height + 0.5)
            
            if np.sum(chimney_mask) > 10:
                chimney_indices = parent.point_indices[chimney_mask]
                children.append(Shape(
                    symbol=ArchitecturalSymbol.CHIMNEY,
                    point_indices=chimney_indices,
                    confidence=0.7,
                    parent=parent
                ))
        
        return children
    
    def _apply_derivation(
        self,
        labels: np.ndarray,
        derivation_tree: Shape,
        points: np.ndarray
    ) -> np.ndarray:
        """
        Apply derivation tree to update point labels.
        
        Traverses the tree and assigns appropriate labels to points.
        """
        from ign_lidar.classes import LOD2_CLASSES, LOD3_CLASSES
        
        refined = labels.copy()
        
        # Map architectural symbols to class IDs (LOD2/LOD3)
        symbol_to_lod2 = {
            ArchitecturalSymbol.FOUNDATION: LOD2_CLASSES.get('foundation', 8),
            ArchitecturalSymbol.WALLS: LOD2_CLASSES.get('wall', 0),
            ArchitecturalSymbol.WALL_SEGMENT: LOD2_CLASSES.get('wall', 0),
            ArchitecturalSymbol.FACADE: LOD2_CLASSES.get('wall', 0),
            ArchitecturalSymbol.ROOF: LOD2_CLASSES.get('roof_flat', 1),
            ArchitecturalSymbol.ROOF_FLAT: LOD2_CLASSES.get('roof_flat', 1),
            ArchitecturalSymbol.ROOF_GABLE: LOD2_CLASSES.get('roof_gable', 2),
            ArchitecturalSymbol.ROOF_HIP: LOD2_CLASSES.get('roof_hip', 3),
            ArchitecturalSymbol.CHIMNEY: LOD2_CLASSES.get('chimney', 4),
            ArchitecturalSymbol.DORMER: LOD2_CLASSES.get('dormer', 5),
            ArchitecturalSymbol.BALCONY: LOD2_CLASSES.get('balcony', 6),
            ArchitecturalSymbol.OVERHANG: LOD2_CLASSES.get('overhang', 7),
        }
        
        # Recursive traversal
        def apply_recursive(shape: Shape):
            if shape.confidence >= self.min_confidence:
                class_id = symbol_to_lod2.get(shape.symbol)
                if class_id is not None:
                    refined[shape.point_indices] = class_id
            
            # Process children
            for child in shape.children:
                apply_recursive(child)
        
        apply_recursive(derivation_tree)
        
        return refined


# ============================================================================
# Convenience Functions
# ============================================================================

def classify_with_grammar(
    points: np.ndarray,
    labels: np.ndarray,
    features: Dict[str, np.ndarray],
    max_iterations: int = 10,
    min_confidence: float = 0.5
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience function to classify points using 3D shape grammar.
    
    Args:
        points: Point coordinates [N, 3]
        labels: Initial classification labels [N]
        features: Dictionary of features
        max_iterations: Maximum parsing iterations
        min_confidence: Minimum confidence threshold
    
    Returns:
        Tuple of (refined_labels, derivation_tree)
    """
    grammar = BuildingGrammar()
    parser = GrammarParser(
        grammar=grammar,
        max_iterations=max_iterations,
        min_confidence=min_confidence
    )
    
    refined_labels, derivation_tree = parser.parse(
        points=points,
        labels=labels,
        features=features
    )
    
    return refined_labels, derivation_tree


def visualize_derivation_tree(
    derivation_tree: Dict[str, Shape],
    output_file: Optional[str] = None
) -> str:
    """
    Generate a text representation of the derivation tree.
    
    Args:
        derivation_tree: Derivation tree from parser
        output_file: Optional file to write visualization
    
    Returns:
        String representation of tree
    """
    lines = []
    lines.append("=" * 80)
    lines.append("BUILDING DERIVATION TREE")
    lines.append("=" * 80)
    
    for building_name, root_shape in derivation_tree.items():
        lines.append(f"\n{building_name.upper()}:")
        lines.extend(_format_shape_tree(root_shape, indent=2))
    
    lines.append("\n" + "=" * 80)
    
    result = "\n".join(lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(result)
        logger.info(f"Derivation tree saved to: {output_file}")
    
    return result


def _format_shape_tree(shape: Shape, indent: int = 0) -> List[str]:
    """Format shape tree recursively."""
    lines = []
    
    prefix = " " * indent
    n_points = len(shape.point_indices) if shape.point_indices is not None else 0
    
    lines.append(
        f"{prefix}├─ {shape.symbol.value} "
        f"({n_points:,} points, confidence={shape.confidence:.2f})"
    )
    
    for child in shape.children:
        lines.extend(_format_shape_tree(child, indent + 3))
    
    return lines
