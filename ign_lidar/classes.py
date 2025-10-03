"""
Building LOD Classification Schemas
"""

# ============================================================================
# LOD2 Building-Focused Class Taxonomy (15 classes)
# ============================================================================
LOD2_CLASSES = {
    # Structural elements (walls)
    'wall': 0,
    
    # Roof types
    'roof_flat': 1,
    'roof_gable': 2,
    'roof_hip': 3,
    
    # Roof details
    'chimney': 4,
    'dormer': 5,
    
    # Facades
    'balcony': 6,
    'overhang': 7,
    
    # Foundation
    'foundation': 8,
    
    # Context (non-building)
    'ground': 9,
    'vegetation_low': 10,
    'vegetation_high': 11,
    'water': 12,
    'vehicle': 13,
    'other': 14,
}

# ============================================================================
# LOD3 Extended Building Taxonomy (30 classes)
# ============================================================================
LOD3_CLASSES = {
    # Structural elements (walls with openings)
    'wall_plain': 0,
    'wall_with_windows': 1,
    'wall_with_door': 2,
    
    # Roof types (detailed)
    'roof_flat': 3,
    'roof_gable': 4,
    'roof_hip': 5,
    'roof_mansard': 6,
    'roof_gambrel': 7,
    
    # Roof details
    'chimney': 8,
    'dormer_gable': 9,
    'dormer_shed': 10,
    'skylight': 11,
    'roof_edge': 12,
    
    # Windows and doors
    'window': 13,
    'door': 14,
    'garage_door': 15,
    
    # Facades
    'balcony': 16,
    'balustrade': 17,
    'overhang': 18,
    'pillar': 19,
    'cornice': 20,
    
    # Foundation
    'foundation': 21,
    'basement_window': 22,
    
    # Context (non-building)
    'ground': 23,
    'vegetation_low': 24,
    'vegetation_high': 25,
    'water': 26,
    'vehicle': 27,
    'street_furniture': 28,
    'other': 29,
}

# ============================================================================
# ASPRS to LOD Class Mapping
# ============================================================================
ASPRS_TO_LOD2 = {
    0: 14,   # Never classified → other
    1: 14,   # Unclassified → other
    2: 9,    # Ground → ground
    3: 10,   # Low Vegetation → vegetation_low
    4: 10,   # Medium Vegetation → vegetation_low
    5: 11,   # High Vegetation → vegetation_high
    6: 0,    # Building → wall (requires refinement)
    7: 10,   # Low Point (noise) → vegetation_low
    8: 14,   # Model Key-point → other
    9: 12,   # Water → water
    10: 14,  # Rail → other
    11: 14,  # Road Surface → other
    12: 14,  # Reserved → other
    13: 14,  # Wire - Guard (Shield) → other
    14: 14,  # Wire - Conductor → other
    15: 14,  # Transmission Tower → other
    16: 14,  # Wire-structure Connector → other
    17: 13,  # Bridge Deck → vehicle (temporary mapping)
    18: 11,  # High Noise → vegetation_high
    64: 14,  # Unknown → other
    65: 14,  # Unknown → other
    67: 14,  # Unknown → other
}

ASPRS_TO_LOD3 = {
    0: 29,   # Never classified → other
    1: 29,   # Unclassified → other
    2: 23,   # Ground → ground
    3: 24,   # Low Vegetation → vegetation_low
    4: 24,   # Medium Vegetation → vegetation_low
    5: 25,   # High Vegetation → vegetation_high
    6: 0,    # Building → wall_plain (requires refinement)
    7: 24,   # Low Point (noise) → vegetation_low
    8: 29,   # Model Key-point → other
    9: 26,   # Water → water
    10: 29,  # Rail → other
    11: 23,  # Road Surface → ground
    12: 29,  # Reserved → other
    13: 29,  # Wire - Guard (Shield) → other
    14: 29,  # Wire - Conductor → other
    15: 29,  # Transmission Tower → other
    16: 29,  # Wire-structure Connector → other
    17: 27,  # Bridge Deck → vehicle (temporary mapping)
    18: 25,  # High Noise → vegetation_high
    64: 29,  # Unknown → other
    65: 29,  # Unknown → other
    67: 29,  # Unknown → other
}