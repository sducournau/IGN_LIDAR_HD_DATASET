"""
Example: Using Architectural Style Functions for Tiles and Patches

This script demonstrates how to use the architectural style detection
functions for both full tiles and individual patches.
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.features import (
    get_tile_architectural_style,
    get_patch_architectural_style,
    compute_architectural_style_features,
    ARCHITECTURAL_STYLES,
    STYLE_NAME_TO_ID,
    infer_multi_styles_from_characteristics,
)


def example_1_tile_style_from_location():
    """
    Example 1: Get architectural style for a tile from location information.
    """
    print("=" * 70)
    print("Example 1: Get Tile Architectural Style from Location Info")
    print("=" * 70)
    
    # Versailles Palace location info
    versailles_info = {
        "location_name": "versailles_chateau",
        "category": "heritage_palace",
        "characteristics": ["chateau_royal", "architecture_classique", "toitures_complexes"]
    }
    
    # Get full style information
    style_info = get_tile_architectural_style(
        location_info=versailles_info,
        encoding="info"
    )
    
    print(f"\nLocation: {style_info['location_name']}")
    print(f"Category: {style_info['category']}")
    print(f"Characteristics: {style_info['characteristics']}")
    print(f"\nDominant Style:")
    print(f"  ID: {style_info['dominant_style']['style_id']}")
    print(f"  Name: {style_info['dominant_style']['style_name']}")
    print(f"  Weight: {style_info['dominant_style']['weight']}")
    print(f"\nConfidence: {style_info['confidence']:.2f}")
    
    # Get just the style ID
    style_id = get_tile_architectural_style(
        location_info=versailles_info,
        encoding="id"
    )
    print(f"\nSimple ID retrieval: {style_id}")
    
    # Get just the style name
    style_name = get_tile_architectural_style(
        location_info=versailles_info,
        encoding="name"
    )
    print(f"Simple name retrieval: {style_name}")


def example_2_multiple_styles():
    """
    Example 2: Detect multiple architectural styles in mixed areas.
    """
    print("\n" + "=" * 70)
    print("Example 2: Multiple Architectural Styles (Mixed Area)")
    print("=" * 70)
    
    # Paris Marais - mixed historical styles
    marais_info = {
        "location_name": "paris_marais",
        "category": "urban_dense",
        "characteristics": [
            "architecture_haussmannienne",
            "hotels_particuliers",
            "architecture_gothique"
        ]
    }
    
    style_info = get_tile_architectural_style(
        location_info=marais_info,
        encoding="info"
    )
    
    print(f"\nLocation: {style_info['location_name']}")
    print(f"\nAll Detected Styles:")
    for i, style in enumerate(style_info['all_styles'], 1):
        print(f"  {i}. {style['style_name']:15s} - ID: {style['style_id']:2d} - Weight: {style['weight']:.3f}")
    
    print(f"\nConfidence: {style_info['confidence']:.2f}")


def example_3_patch_style_inheritance():
    """
    Example 3: Patch inherits style from parent tile.
    """
    print("\n" + "=" * 70)
    print("Example 3: Patch Architectural Style (Inherit from Tile)")
    print("=" * 70)
    
    # Create synthetic point cloud patch
    num_points = 5000
    points = np.random.rand(num_points, 3) * 100  # 100m x 100m x 100m
    classification = np.random.choice([2, 6, 9], size=num_points)  # Ground, Building, Water
    
    # Get tile style first
    tile_info = {
        "location_name": "lyon_vieux_lyon",
        "category": "urban_dense",
        "characteristics": ["architecture_renaissance", "facades_ornees"]
    }
    
    tile_style = get_tile_architectural_style(location_info=tile_info)
    
    # Get patch style (inheriting from tile)
    patch_style = get_patch_architectural_style(
        points=points,
        classification=classification,
        tile_style_info=tile_style,
        encoding="info"
    )
    
    print(f"\nTile Style: {tile_style['dominant_style']['style_name']}")
    print(f"Patch Style: {patch_style['dominant_style']['style_name']}")
    print(f"Confidence: {patch_style['confidence']:.2f}")
    print(f"Number of points: {patch_style['num_points']}")
    
    # Get as constant feature array
    style_features = get_patch_architectural_style(
        points=points,
        classification=classification,
        tile_style_info=tile_style,
        encoding="constant"
    )
    
    print(f"\nConstant encoding shape: {style_features.shape}")
    print(f"Unique style IDs: {np.unique(style_features)}")
    print(f"First 10 values: {style_features[:10]}")


def example_4_patch_style_from_features():
    """
    Example 4: Infer patch style from building features.
    """
    print("\n" + "=" * 70)
    print("Example 4: Patch Architectural Style from Building Features")
    print("=" * 70)
    
    # Create synthetic point cloud
    num_points = 8000
    points = np.random.rand(num_points, 3) * 50
    
    # Building features that suggest Haussmannian architecture
    haussmann_features = {
        "roof_slope_mean": 38.0,  # Typical Haussmannian angle
        "wall_thickness_mean": 0.55,  # ~55cm walls
        "window_to_wall_ratio": 0.25,
        "geometric_regularity": 0.88,  # Very regular
        "building_height": 18.5,  # ~18m (6 stories)
        "footprint_area": 350.0
    }
    
    # Infer style from features
    patch_style = get_patch_architectural_style(
        points=points,
        building_features=haussmann_features,
        encoding="info"
    )
    
    print(f"\nBuilding Features:")
    print(f"  Roof slope: {haussmann_features['roof_slope_mean']}°")
    print(f"  Wall thickness: {haussmann_features['wall_thickness_mean']}m")
    print(f"  Height: {haussmann_features['building_height']}m")
    print(f"  Regularity: {haussmann_features['geometric_regularity']:.2f}")
    
    print(f"\nInferred Style: {patch_style['dominant_style']['style_name']}")
    print(f"Style ID: {patch_style['dominant_style']['style_id']}")
    print(f"Confidence: {patch_style['confidence']:.2f}")
    
    # Modern glass building features
    modern_features = {
        "roof_slope_mean": 2.0,  # Nearly flat
        "wall_thickness_mean": 0.25,
        "window_to_wall_ratio": 0.75,  # Mostly glass
        "geometric_regularity": 0.92,
        "building_height": 35.0,  # Tall building
        "footprint_area": 800.0
    }
    
    modern_style = get_patch_architectural_style(
        points=points,
        building_features=modern_features,
        encoding="info"
    )
    
    print(f"\n\nModern Building Features:")
    print(f"  Window ratio: {modern_features['window_to_wall_ratio']:.2f}")
    print(f"  Height: {modern_features['building_height']}m")
    
    print(f"\nInferred Style: {modern_style['dominant_style']['style_name']}")
    print(f"Confidence: {modern_style['confidence']:.2f}")


def example_5_ml_training_features():
    """
    Example 5: Generate ML training features with different encodings.
    """
    print("\n" + "=" * 70)
    print("Example 5: ML Training Features (Different Encodings)")
    print("=" * 70)
    
    # Create point cloud
    num_points = 3000
    points = np.random.rand(num_points, 3) * 100
    classification = np.full(num_points, 6)  # All building points
    
    # Tile style info
    tile_info = {
        "category": "urban_dense",
        "characteristics": ["architecture_haussmannienne"]
    }
    tile_style = get_tile_architectural_style(location_info=tile_info)
    
    # 1. Constant encoding (single value per point)
    constant_features = compute_architectural_style_features(
        points=points,
        classification=classification,
        tile_style_info=tile_style,
        encoding="constant"
    )
    
    print(f"\n1. Constant Encoding:")
    print(f"   Shape: {constant_features.shape}")
    print(f"   Dtype: {constant_features.dtype}")
    print(f"   Values: all {constant_features[0]} (style ID for Haussmannian)")
    
    # 2. One-hot encoding (13 dimensions)
    onehot_features = compute_architectural_style_features(
        points=points,
        classification=classification,
        tile_style_info=tile_style,
        encoding="onehot"
    )
    
    print(f"\n2. One-hot Encoding:")
    print(f"   Shape: {onehot_features.shape}")
    print(f"   Non-zero column: {np.where(onehot_features[0] == 1.0)[0][0]}")
    print(f"   Sample row: {onehot_features[0]}")
    
    # 3. Multi-hot encoding for mixed styles
    mixed_info = {
        "category": "urban_dense",
        "characteristics": ["architecture_haussmannienne", "architecture_moderne"]
    }
    mixed_tile_style = get_tile_architectural_style(location_info=mixed_info)
    
    multihot_features = compute_architectural_style_features(
        points=points,
        classification=classification,
        tile_style_info=mixed_tile_style,
        encoding="multihot"
    )
    
    print(f"\n3. Multi-hot Encoding (Mixed Styles):")
    print(f"   Shape: {multihot_features.shape}")
    print(f"   Non-zero values in first row: {multihot_features[0][multihot_features[0] > 0]}")
    print(f"   Styles detected:")
    for idx, val in enumerate(multihot_features[0]):
        if val > 0:
            print(f"     - {ARCHITECTURAL_STYLES[idx]}: {val:.3f}")


def example_6_inference_from_characteristics():
    """
    Example 6: Infer styles from various characteristics.
    """
    print("\n" + "=" * 70)
    print("Example 6: Style Inference from Characteristics")
    print("=" * 70)
    
    # Test various characteristic combinations
    test_cases = [
        {
            "name": "Gothic Cathedral",
            "characteristics": ["cathedrale_gothique", "architecture_medievale"]
        },
        {
            "name": "Industrial Zone",
            "characteristics": ["hangars", "entrepots", "architecture_industrielle"]
        },
        {
            "name": "Rural Village",
            "characteristics": ["architecture_rurale", "village_traditionnel"]
        },
        {
            "name": "Modern Office",
            "characteristics": ["tours_verre", "architecture_contemporaine"]
        }
    ]
    
    for test in test_cases:
        styles = infer_multi_styles_from_characteristics(test["characteristics"])
        
        print(f"\n{test['name']}:")
        print(f"  Characteristics: {test['characteristics']}")
        print(f"  Detected Styles:")
        for style in styles:
            print(f"    - {style['style_name']:15s} (ID: {style['style_id']:2d}, Weight: {style['weight']:.3f})")


def example_7_all_style_reference():
    """
    Example 7: Display all available architectural styles.
    """
    print("\n" + "=" * 70)
    print("Example 7: Complete Architectural Style Reference")
    print("=" * 70)
    
    print(f"\nTotal Styles Available: {len(ARCHITECTURAL_STYLES)}")
    print("\nStyle ID | Style Name         | Description")
    print("-" * 70)
    
    descriptions = {
        0: "Unknown or unclassified",
        1: "Classical/Traditional French architecture",
        2: "Gothic (medieval churches, cathedrals)",
        3: "Renaissance (châteaux, palaces)",
        4: "Baroque ornate style",
        5: "Haussmannian (Paris-style buildings)",
        6: "Modern/Contemporary (20th-21st century)",
        7: "Industrial buildings and warehouses",
        8: "Vernacular/Local traditional rural",
        9: "Art Deco style (1920s-1940s)",
        10: "Brutalist concrete architecture",
        11: "Modern glass and steel buildings",
        12: "Military fortifications and fortresses"
    }
    
    for style_id, style_name in ARCHITECTURAL_STYLES.items():
        desc = descriptions.get(style_id, "")
        print(f"{style_id:8d} | {style_name:18s} | {desc}")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  Architectural Style Detection - Complete Examples".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    try:
        example_1_tile_style_from_location()
        example_2_multiple_styles()
        example_3_patch_style_inheritance()
        example_4_patch_style_from_features()
        example_5_ml_training_features()
        example_6_inference_from_characteristics()
        example_7_all_style_reference()
        
        print("\n" + "=" * 70)
        print("All examples completed successfully! ✓")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
