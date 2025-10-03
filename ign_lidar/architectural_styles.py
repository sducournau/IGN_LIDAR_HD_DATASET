"""
Architectural Style Classification System

Maps architectural characteristics to numerical class IDs for training.
Allows models to learn architectural style alongside geometry.
"""

from typing import List, Optional, Dict
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
    "immeubles_haussmann": 5,
    "facades_continues": 5,
    "toitures_zinc": 5,
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
