"""
Architectural Style Classification System

Maps architectural characteristics to numerical class IDs for training.
Allows models to learn architectural style alongside geometry.
"""

from typing import List, Optional, Dict, Tuple, Union
import numpy as np

# ============================================================================
# ARCHITECTURAL STYLE CLASS DEFINITIONS
# ============================================================================

ARCHITECTURAL_STYLES = {
    0: "unknown",
    1: "classical",              # Classical/Traditional architecture
    2: "gothic",                 # Gothic (medieval churches, cathedrals)
    3: "renaissance",            # Renaissance (châteaux, palaces)
    4: "baroque",                # Baroque ornate style
    5: "haussmann",              # Haussmannian (Paris-style buildings)
    6: "modern",                 # Modern/Contemporary (20th-21st century)
    7: "industrial",             # Industrial buildings
    8: "vernacular",             # Local/traditional rural
    9: "art_deco",               # Art Deco style
    10: "brutalist",             # Brutalist concrete
    11: "glass_steel",           # Modern glass and steel
    12: "fortress",              # Military fortifications
}

# Reverse mapping
STYLE_NAME_TO_ID = {name: id for id, name in ARCHITECTURAL_STYLES.items()}

# ============================================================================
# CHARACTERISTIC TO STYLE MAPPING
# ============================================================================

CHARACTERISTIC_TO_STYLE = {
    # Classical/Traditional
    "architecture_classique": 1,
    "chateau_royal": 1,
    
    # Gothic
    "architecture_gothique": 2,
    "cathedrale_gothique": 2,
    "architecture_medievale": 2,
    "cathedrale": 2,
    
    # Renaissance
    "architecture_renaissance": 3,
    "chateau_renaissance": 3,
    "tourelles": 3,
    
    # Baroque
    "architecture_baroque": 4,
    "facades_ornees": 4,
    
    # Haussmannian
    "architecture_haussmannienne": 5,
    "haussmannien": 5,  # Short form
    "immeubles_haussmann": 5,
    "facades_continues": 5,
    "toitures_zinc": 5,
    "toits_zinc": 5,  # Variant spelling
    "boulevards_larges": 5,
    
    # Modern
    "architecture_moderne": 6,
    "gratte_ciel": 6,
    "buildings_verre": 6,
    "facades_verre": 6,
    
    # Industrial
    "architecture_industrielle": 7,
    "hangars": 7,
    "entrepots": 7,
    
    # Vernacular/Rural
    "architecture_rurale": 8,
    "village_traditionnel": 8,
    "batiments_agricoles": 8,
    
    # Art Deco
    "art_deco": 9,
    
    # Brutalist
    "beton_brut": 10,
    "architecture_brutale": 10,
    
    # Glass/Steel Modern
    "tours_verre": 11,
    "architecture_contemporaine": 11,
    
    # Fortress
    "cite_medievale": 12,
    "remparts": 12,
    "tours_defense": 12,
    "fortifications": 12,
}

# ============================================================================
# CATEGORY TO STYLE MAPPING (fallback)
# ============================================================================

CATEGORY_TO_STYLE = {
    "heritage_palace": 3,           # Renaissance
    "heritage_religious": 2,        # Gothic
    "heritage_fortress": 12,        # Fortress
    "urban_dense": 5,               # Haussmannian
    "urban_dense_historic": 5,      # Haussmannian
    "urban_modern": 6,              # Modern
    "coastal_urban": 6,             # Modern
    "coastal_residential": 8,       # Vernacular
    "suburban_residential": 6,      # Modern
    "rural_traditional": 8,         # Vernacular
    "mountain_resort": 6,           # Modern
    "infrastructure_airport": 7,    # Industrial
    "infrastructure_port": 7,       # Industrial
    "infrastructure_station": 7,    # Industrial
}


def get_architectural_style_id(
    characteristics: Optional[List[str]] = None,
    category: Optional[str] = None
) -> int:
    """
    Determine architectural style ID from characteristics or category.
    
    Args:
        characteristics: List of characteristic strings
        category: Location category string
        
    Returns:
        Style ID (0-12), where 0 = unknown
    """
    # Try to determine from characteristics first (most specific)
    if characteristics:
        for char in characteristics:
            if char in CHARACTERISTIC_TO_STYLE:
                return CHARACTERISTIC_TO_STYLE[char]
    
    # Fall back to category (less specific)
    if category and category in CATEGORY_TO_STYLE:
        return CATEGORY_TO_STYLE[category]
    
    # Default to unknown
    return 0


def get_style_name(style_id: int) -> str:
    """
    Get style name from ID.
    
    Args:
        style_id: Style ID
        
    Returns:
        Style name string
    """
    return ARCHITECTURAL_STYLES.get(style_id, "unknown")


def get_style_distribution(style_ids: np.ndarray) -> Dict[str, int]:
    """
    Get distribution of styles in a dataset.
    
    Args:
        style_ids: Array of style IDs
        
    Returns:
        Dictionary mapping style name to count
    """
    unique, counts = np.unique(style_ids, return_counts=True)
    
    distribution = {}
    for style_id, count in zip(unique, counts):
        style_name = get_style_name(int(style_id))
        distribution[style_name] = int(count)
    
    return distribution


def encode_style_as_feature(
    style_id: int,
    num_points: int,
    encoding: str = "constant"
) -> np.ndarray:
    """
    Encode architectural style as a point feature.
    
    Args:
        style_id: Architectural style ID
        num_points: Number of points in the patch
        encoding: Encoding method:
            - "constant": All points get same style ID
            - "onehot": One-hot encoding (13 dimensions)
            
    Returns:
        Feature array [N] or [N, 13]
    """
    if encoding == "constant":
        return np.full(num_points, style_id, dtype=np.int32)
    
    elif encoding == "onehot":
        onehot = np.zeros((num_points, len(ARCHITECTURAL_STYLES)), dtype=np.float32)
        onehot[:, style_id] = 1.0
        return onehot
    
    else:
        raise ValueError(f"Unknown encoding: {encoding}")


def encode_multi_style_feature(
    style_ids: List[int],
    weights: List[float],
    num_points: int,
    encoding: str = "multihot"
) -> np.ndarray:
    """
    Encode multiple architectural styles with weights as a point feature.
    
    Args:
        style_ids: List of architectural style IDs
        weights: List of weights for each style (should sum to ~1.0)
        num_points: Number of points in the patch
        encoding: Encoding method:
            - "multihot": Multi-hot encoding with weights [N, 13]
            - "constant": Use dominant style only [N]
            
    Returns:
        Feature array [N, 13] for multihot, [N] for constant
        
    Examples:
        >>> # Mixed Haussmannian (70%) and Gothic (30%)
        >>> feature = encode_multi_style_feature(
        ...     style_ids=[5, 2],
        ...     weights=[0.7, 0.3],
        ...     num_points=1000,
        ...     encoding="multihot"
        ... )
        >>> feature.shape
        (1000, 13)
        >>> feature[0, 5]  # Haussmannian
        0.7
        >>> feature[0, 2]  # Gothic
        0.3
    """
    if not style_ids or not weights:
        # No styles provided, return unknown
        return encode_style_as_feature(0, num_points, encoding)
    
    if len(style_ids) != len(weights):
        raise ValueError(f"style_ids and weights must have same length: {len(style_ids)} vs {len(weights)}")
    
    if encoding == "multihot":
        # Create multi-hot encoding
        feature = np.zeros((num_points, len(ARCHITECTURAL_STYLES)), dtype=np.float32)
        for style_id, weight in zip(style_ids, weights):
            if 0 <= style_id < len(ARCHITECTURAL_STYLES):
                feature[:, style_id] = weight
            else:
                raise ValueError(f"Invalid style_id: {style_id}")
        return feature
    
    elif encoding == "constant":
        # Use dominant style (highest weight)
        dominant_idx = np.argmax(weights)
        dominant_style = style_ids[dominant_idx]
        return np.full(num_points, dominant_style, dtype=np.int32)
    
    else:
        raise ValueError(f"Unknown encoding: {encoding}")


def infer_multi_styles_from_characteristics(
    characteristics: List[str],
    default_weights: Optional[Dict[int, float]] = None
) -> List[Dict]:
    """
    Infer multiple architectural styles from characteristics with estimated weights.
    
    Args:
        characteristics: List of characteristic strings
        default_weights: Optional custom weights for each style_id
        
    Returns:
        List of style dictionaries with style_id, style_name, and weight
        
    Examples:
        >>> chars = ["architecture_haussmannienne", "architecture_gothique"]
        >>> styles = infer_multi_styles_from_characteristics(chars)
        >>> styles
        [
            {"style_id": 5, "style_name": "haussmann", "weight": 0.6},
            {"style_id": 2, "style_name": "gothic", "weight": 0.4}
        ]
    """
    if not characteristics:
        return [{"style_id": 0, "style_name": "unknown", "weight": 1.0}]
    
    # Find all matching styles
    style_ids = []
    for char in characteristics:
        if char in CHARACTERISTIC_TO_STYLE:
            style_id = CHARACTERISTIC_TO_STYLE[char]
            if style_id not in style_ids:
                style_ids.append(style_id)
    
    if not style_ids:
        return [{"style_id": 0, "style_name": "unknown", "weight": 1.0}]
    
    # Assign weights
    if default_weights:
        # Use provided weights
        weights = [default_weights.get(sid, 1.0) for sid in style_ids]
    else:
        # Equal weights by default
        weights = [1.0] * len(style_ids)
    
    # Normalize weights to sum to 1.0
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    else:
        weights = [1.0 / len(style_ids)] * len(style_ids)
    
    # Sort by weight (descending)
    style_weight_pairs = sorted(
        zip(style_ids, weights),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Create result
    result = []
    for style_id, weight in style_weight_pairs:
        result.append({
            "style_id": style_id,
            "style_name": get_style_name(style_id),
            "weight": round(weight, 3)
        })
    
    return result


# ============================================================================
# TILE AND PATCH ARCHITECTURAL STYLE RETRIEVAL
# ============================================================================

def get_tile_architectural_style(
    tile_name: Optional[str] = None,
    tile_bbox: Optional[Tuple[float, float, float, float]] = None,
    location_info: Optional[Dict] = None,
    encoding: str = "info"
) -> Union[Dict, int, str]:
    """
    Get architectural style information for a tile.
    
    Args:
        tile_name: Tile filename (e.g., "HD_LIDARHD_FXX_0650_6860_PTS_C_LAMB93_IGN69")
        tile_bbox: Tile bounding box (xmin, ymin, xmax, ymax)
        location_info: Optional location info dict with keys:
            - location_name: str
            - category: str
            - characteristics: List[str]
        encoding: Return format:
            - "info": Full style information dict (default)
            - "id": Just the dominant style ID
            - "name": Just the dominant style name
            
    Returns:
        Dictionary with architectural style information:
        {
            "dominant_style": {"style_id": int, "style_name": str, "weight": float},
            "all_styles": List[Dict],  # All detected styles with weights
            "location_name": str,
            "category": str,
            "characteristics": List[str],
            "confidence": float  # 0-1 confidence score
        }
        
        If encoding="id", returns just the dominant style_id (int)
        If encoding="name", returns just the dominant style_name (str)
        
    Examples:
        >>> # From location info
        >>> info = {
        ...     "location_name": "versailles_chateau",
        ...     "category": "heritage_palace",
        ...     "characteristics": ["chateau_royal", "architecture_classique"]
        ... }
        >>> style = get_tile_architectural_style(location_info=info)
        >>> style["dominant_style"]["style_name"]
        'classical'
        
        >>> # Simple ID return
        >>> style_id = get_tile_architectural_style(location_info=info, encoding="id")
        >>> style_id
        1
    """
    # Default unknown style
    unknown_result = {
        "dominant_style": {"style_id": 0, "style_name": "unknown", "weight": 1.0},
        "all_styles": [{"style_id": 0, "style_name": "unknown", "weight": 1.0}],
        "location_name": None,
        "category": None,
        "characteristics": [],
        "confidence": 0.0
    }
    
    # Extract location info
    if location_info is None:
        location_info = {}
    
    characteristics = location_info.get("characteristics", [])
    category = location_info.get("category")
    location_name = location_info.get("location_name")
    
    # Infer styles from characteristics and category
    if characteristics:
        styles = infer_multi_styles_from_characteristics(characteristics)
        confidence = 0.9 if len(characteristics) >= 2 else 0.7
    elif category:
        # Fallback to category-based inference
        style_id = get_architectural_style_id(category=category)
        style_name = get_style_name(style_id)
        styles = [{"style_id": style_id, "style_name": style_name, "weight": 1.0}]
        confidence = 0.5
    else:
        # No information available
        if encoding == "id":
            return 0
        elif encoding == "name":
            return "unknown"
        return unknown_result
    
    # Build result
    result = {
        "dominant_style": styles[0] if styles else unknown_result["dominant_style"],
        "all_styles": styles,
        "location_name": location_name,
        "category": category,
        "characteristics": characteristics,
        "confidence": confidence
    }
    
    # Return according to encoding
    if encoding == "id":
        return result["dominant_style"]["style_id"]
    elif encoding == "name":
        return result["dominant_style"]["style_name"]
    else:
        return result


def get_patch_architectural_style(
    points: np.ndarray,
    classification: Optional[np.ndarray] = None,
    tile_style_info: Optional[Dict] = None,
    building_features: Optional[Dict] = None,
    encoding: str = "info"
) -> Union[Dict, int, str, np.ndarray]:
    """
    Get architectural style information for a point cloud patch.
    
    This function can:
    1. Inherit style from parent tile (tile_style_info)
    2. Refine style based on local building features
    3. Analyze point cloud geometry to infer style
    
    Args:
        points: Point cloud array [N, 3] (XYZ)
        classification: Point classification codes [N]
        tile_style_info: Style info from parent tile (from get_tile_architectural_style)
        building_features: Optional dict with extracted building features:
            - roof_slope_mean: Average roof slope (degrees)
            - wall_thickness_mean: Average wall thickness (meters)
            - window_to_wall_ratio: Ratio of window to wall area
            - geometric_regularity: 0-1 score
            - building_height: Height in meters
            - footprint_area: Building footprint area (m²)
        encoding: Return format ("info", "id", "name", "constant", "onehot")
            - "info": Full style information dict
            - "id": Dominant style ID
            - "name": Dominant style name
            - "constant": Array [N] with style ID for all points
            - "onehot": Array [N, 13] one-hot encoding
            
    Returns:
        Dictionary with architectural style information (if encoding="info")
        Or encoded features (if encoding="constant" or "onehot")
        
    Examples:
        >>> # Inherit from tile
        >>> patch_style = get_patch_architectural_style(
        ...     points=points,
        ...     tile_style_info=tile_style,
        ...     encoding="constant"
        ... )
        >>> patch_style.shape
        (10000,)
        
        >>> # Analyze building features
        >>> features = {
        ...     "roof_slope_mean": 42.0,
        ...     "wall_thickness_mean": 0.65,
        ...     "building_height": 18.0,
        ...     "geometric_regularity": 0.85
        ... }
        >>> style = get_patch_architectural_style(
        ...     points=points,
        ...     building_features=features,
        ...     encoding="info"
        ... )
        >>> style["dominant_style"]["style_name"]
        'haussmann'
    """
    num_points = len(points)
    
    # Start with tile style if available
    if tile_style_info is not None and isinstance(tile_style_info, dict):
        base_styles = tile_style_info.get("all_styles", [])
        confidence = tile_style_info.get("confidence", 0.5)
    else:
        base_styles = [{"style_id": 0, "style_name": "unknown", "weight": 1.0}]
        confidence = 0.3
    
    # Refine based on building features if available
    if building_features is not None and len(building_features) > 0:
        refined_style = _infer_style_from_building_features(building_features)
        if refined_style is not None:
            # Boost confidence if feature-based detection agrees with tile style
            if base_styles and refined_style["style_id"] == base_styles[0]["style_id"]:
                confidence = min(confidence + 0.2, 1.0)
                base_styles = [refined_style]
            else:
                # Feature-based detection differs, blend styles
                confidence = 0.6
                # Weighted blend: 60% features, 40% tile
                base_styles = [
                    {"style_id": refined_style["style_id"], "style_name": refined_style["style_name"], "weight": 0.6},
                    {"style_id": base_styles[0]["style_id"], "style_name": base_styles[0]["style_name"], "weight": 0.4}
                ]
    
    # Analyze point cloud geometry if classification is available
    elif classification is not None:
        building_mask = classification == 6  # Building points
        if np.sum(building_mask) > 100:  # Need enough building points
            # Extract simple geometric features
            building_points = points[building_mask]
            height_range = np.ptp(building_points[:, 2])
            
            # Simple heuristics
            if height_range > 20:
                # Likely modern high-rise
                base_styles = [{"style_id": 6, "style_name": "modern", "weight": 1.0}]
                confidence = 0.5
            elif height_range > 15:
                # Could be Haussmannian or modern
                base_styles = [
                    {"style_id": 5, "style_name": "haussmann", "weight": 0.6},
                    {"style_id": 6, "style_name": "modern", "weight": 0.4}
                ]
                confidence = 0.4
    
    # Build result
    result = {
        "dominant_style": base_styles[0] if base_styles else {"style_id": 0, "style_name": "unknown", "weight": 1.0},
        "all_styles": base_styles,
        "confidence": confidence,
        "num_points": num_points
    }
    
    # Return according to encoding
    if encoding == "id":
        return result["dominant_style"]["style_id"]
    elif encoding == "name":
        return result["dominant_style"]["style_name"]
    elif encoding == "constant":
        style_id = result["dominant_style"]["style_id"]
        return encode_style_as_feature(style_id, num_points, encoding="constant")
    elif encoding == "onehot":
        style_id = result["dominant_style"]["style_id"]
        return encode_style_as_feature(style_id, num_points, encoding="onehot")
    elif encoding == "multihot":
        if len(base_styles) > 1:
            style_ids = [s["style_id"] for s in base_styles]
            weights = [s["weight"] for s in base_styles]
            return encode_multi_style_feature(style_ids, weights, num_points, encoding="multihot")
        else:
            # Only one style, return regular onehot
            style_id = result["dominant_style"]["style_id"]
            return encode_style_as_feature(style_id, num_points, encoding="onehot")
    else:
        return result


def _infer_style_from_building_features(features: Dict) -> Optional[Dict]:
    """
    Infer architectural style from extracted building features.
    
    Args:
        features: Dictionary of building features
        
    Returns:
        Style dict or None if cannot infer
    """
    roof_slope = features.get("roof_slope_mean", 0)
    wall_thickness = features.get("wall_thickness_mean", 0)
    window_ratio = features.get("window_to_wall_ratio", 0)
    regularity = features.get("geometric_regularity", 0)
    height = features.get("building_height", 0)
    footprint_area = features.get("footprint_area", 0)
    
    # Traditional rural: Steep roofs, thick walls
    if roof_slope > 45 and wall_thickness > 0.6 and height < 12:
        return {"style_id": 8, "style_name": "vernacular", "weight": 1.0}
    
    # Haussmannian: Moderate slope, 15-25m height, regular geometry
    if 25 <= roof_slope <= 45 and 15 <= height <= 25 and regularity > 0.8:
        return {"style_id": 5, "style_name": "haussmann", "weight": 1.0}
    
    # Modern glass/steel: High window ratio, tall building
    if window_ratio > 0.6 and height > 20:
        return {"style_id": 11, "style_name": "glass_steel", "weight": 1.0}
    
    # Contemporary modern: Irregular geometry, moderate height
    if regularity < 0.5 and 10 <= height <= 30:
        return {"style_id": 6, "style_name": "modern", "weight": 1.0}
    
    # Industrial: Large footprint, low height, simple geometry
    if footprint_area > 1000 and height < 15 and regularity > 0.9:
        return {"style_id": 7, "style_name": "industrial", "weight": 1.0}
    
    # Gothic/Medieval: Very steep roofs, thick walls, tall
    if roof_slope > 55 and wall_thickness > 0.8 and height > 15:
        return {"style_id": 2, "style_name": "gothic", "weight": 1.0}
    
    # Classical/Renaissance: Regular, moderate proportions
    if regularity > 0.85 and 12 <= height <= 20 and 35 <= roof_slope <= 50:
        return {"style_id": 3, "style_name": "renaissance", "weight": 1.0}
    
    # Cannot confidently infer
    return None


def compute_architectural_style_features(
    points: np.ndarray,
    classification: Optional[np.ndarray] = None,
    tile_style_info: Optional[Dict] = None,
    building_features: Optional[Dict] = None,
    encoding: str = "constant"
) -> np.ndarray:
    """
    Compute architectural style features for a point cloud.
    
    This is a convenience wrapper around get_patch_architectural_style
    that always returns numpy arrays suitable for ML training.
    
    Args:
        points: Point cloud array [N, 3]
        classification: Point classification codes [N]
        tile_style_info: Style info from parent tile
        building_features: Optional building features dict
        encoding: Feature encoding ("constant", "onehot", "multihot")
        
    Returns:
        Feature array [N] for constant, [N, 13] for onehot/multihot
        
    Examples:
        >>> style_features = compute_architectural_style_features(
        ...     points=points,
        ...     tile_style_info=tile_style,
        ...     encoding="constant"
        ... )
        >>> style_features.shape
        (10000,)
        >>> style_features[:5]
        array([5, 5, 5, 5, 5], dtype=int32)  # Haussmannian
    """
    return get_patch_architectural_style(
        points=points,
        classification=classification,
        tile_style_info=tile_style_info,
        building_features=building_features,
        encoding=encoding
    )
