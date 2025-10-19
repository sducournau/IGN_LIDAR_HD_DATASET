"""
ASPRS LAS Specification 1.4 - Classification Codes

This module provides comprehensive classification codes according to the
ASPRS LAS 1.4 specification, along with mapping utilities and extended
classifications for French topographic data (IGN BD TOPO®).

⚠️  IGN-Specific Non-Standard Classes:
    - Class 67: Unknown/Invalid class found in some IGN LiDAR HD tiles
      → Automatically remapped to Class 1 (Unclassified) during preprocessing

Reference: ASPRS LAS Specification Version 1.4 - R15
https://www.asprs.org/wp-content/uploads/2019/07/LAS_1_4_r15.pdf

Features Required for Ground Truth Refinement & Classification:
==================================================================

Ground Truth Refinement (ign_lidar.core.modules.ground_truth_refinement):
--------------------------------------------------------------------------

1. Water Classification:
   - height: Z above ground (meters)
   - planarity: Flatness measure [0-1]
   - curvature: Surface curvature [0-∞]
   - normals: Surface normal vectors [N, 3] (nx, ny, nz)

2. Road Classification:
   - height: Z above ground (meters)
   - planarity: Flatness measure [0-1]
   - curvature: Surface curvature [0-∞]
   - normals: Surface normal vectors [N, 3]
   - ndvi: Normalized Difference Vegetation Index [-1, 1]

3. Vegetation Classification:
   - ndvi: Normalized Difference Vegetation Index [-1, 1]
   - height: Z above ground (meters)
   - curvature: Surface curvature [0-∞]
   - planarity: Flatness measure [0-1]
   - sphericity: Shape sphericity [0-1]
   - roughness: Surface roughness [0-∞]

4. Building Classification:
   - height: Z above ground (meters)
   - planarity: Flatness measure [0-1]
   - verticality: Wall-like measure [0-1]
   - ndvi: Normalized Difference Vegetation Index [-1, 1]

Feature Computation Pipeline:
------------------------------
Features are computed in ign_lidar.features module:
- height: Computed from Z - DTM elevation
- planarity: Eigenvalue-based local surface flatness
- curvature: Mean curvature from local surface fitting
- normals: PCA-based normal estimation
- sphericity: Eigenvalue-based shape measure (λ3 / λ1)
- roughness: Standard deviation of distances to fitted plane
- verticality: |normal_z| < threshold for vertical surfaces
- ndvi: (NIR - Red) / (NIR + Red) from RGB approximation

Core Feature Sets by Classification Task:
------------------------------------------
- Water: [height, planarity, curvature, normals]
- Roads: [height, planarity, curvature, normals, ndvi]
- Vegetation: [ndvi, height, curvature, planarity, sphericity, roughness]
- Buildings: [height, planarity, verticality, ndvi]
"""

from typing import Dict, List, Optional, Tuple, Any
from enum import IntEnum


# ============================================================================
# Feature Definitions for Classification & Ground Truth Refinement
# ============================================================================

# Required features for each classification type
WATER_FEATURES = ['height', 'planarity', 'curvature', 'normals']
ROAD_FEATURES = ['height', 'planarity', 'curvature', 'normals', 'ndvi']
VEGETATION_FEATURES = ['ndvi', 'height', 'curvature', 'planarity', 'sphericity', 'roughness']
BUILDING_FEATURES = ['height', 'planarity', 'verticality', 'ndvi']

# All unique features needed for ASPRS classification
ALL_CLASSIFICATION_FEATURES = [
    'height',        # Z above ground (meters)
    'planarity',     # Flatness measure [0-1]
    'curvature',     # Surface curvature [0-∞]
    'normals',       # Surface normal vectors [N, 3]
    'ndvi',          # Normalized Difference Vegetation Index [-1, 1]
    'sphericity',    # Shape sphericity [0-1]
    'roughness',     # Surface roughness [0-∞]
    'verticality',   # Wall-like measure [0-1]
]

# Feature descriptions
FEATURE_DESCRIPTIONS = {
    'height': 'Z coordinate above ground (meters). Computed as Z - DTM elevation.',
    'planarity': 'Local surface flatness [0-1]. High values = flat surfaces. Eigenvalue-based: (λ2 - λ3) / λ1.',
    'curvature': 'Surface curvature [0-∞]. Mean curvature from local surface fitting.',
    'normals': 'Surface normal vectors [N, 3]. PCA-based normal estimation from neighbors.',
    'ndvi': 'Normalized Difference Vegetation Index [-1, 1]. (NIR - Red) / (NIR + Red). High = vegetation.',
    'sphericity': 'Shape sphericity [0-1]. Eigenvalue-based: λ3 / λ1. High = spherical/organic shapes.',
    'roughness': 'Surface roughness [0-∞]. Std dev of distances to fitted plane. High = irregular surfaces.',
    'verticality': 'Wall-like measure [0-1]. |normal_z| for vertical surfaces. High = walls/facades.',
}

# Feature value ranges
FEATURE_RANGES = {
    'height': (-10.0, 100.0),       # meters
    'planarity': (0.0, 1.0),        # normalized
    'curvature': (0.0, 1.0),        # typically 0-0.1 for natural surfaces
    'normals': (-1.0, 1.0),         # normalized vectors
    'ndvi': (-1.0, 1.0),            # normalized index
    'sphericity': (0.0, 1.0),       # normalized
    'roughness': (0.0, 1.0),        # typically 0-0.5 for natural surfaces
    'verticality': (0.0, 1.0),      # normalized
}


# ============================================================================
# ASPRS LAS 1.4 Standard Classification Codes (0-255)
# ============================================================================

class ASPRSClass(IntEnum):
    """
    Standard ASPRS LAS 1.4 classification codes.
    
    These are the official classification codes defined in the LAS specification.
    Codes 0-31 are reserved by ASPRS. Codes 32-255 are available for user-defined
    classes or application-specific use.
    """
    # Standard Classifications (0-31) - Reserved by ASPRS
    CREATED_NEVER_CLASSIFIED = 0      # Created, never classified
    UNCLASSIFIED = 1                  # Unclassified
    GROUND = 2                        # Ground
    LOW_VEGETATION = 3                # Low Vegetation
    MEDIUM_VEGETATION = 4             # Medium Vegetation
    HIGH_VEGETATION = 5               # High Vegetation
    BUILDING = 6                      # Building
    LOW_POINT = 7                     # Low Point (noise)
    RESERVED_8 = 8                    # Reserved (formerly Model Key-point)
    WATER = 9                         # Water
    RAIL = 10                         # Rail
    ROAD_SURFACE = 11                 # Road Surface
    RESERVED_12 = 12                  # Reserved (formerly Wire - Guard)
    WIRE_GUARD = 13                   # Wire - Guard (Shield)
    WIRE_CONDUCTOR = 14               # Wire - Conductor (Phase)
    TRANSMISSION_TOWER = 15           # Transmission Tower
    WIRE_STRUCTURE_CONNECTOR = 16     # Wire-structure Connector (Insulator)
    BRIDGE_DECK = 17                  # Bridge Deck
    HIGH_NOISE = 18                   # High Noise
    OVERHEAD_STRUCTURE = 19           # Overhead Structure
    IGNORED_GROUND = 20               # Ignored Ground (Breakline Proximity)
    SNOW = 21                         # Snow
    TEMPORAL_EXCLUSION = 22           # Temporal Exclusion
    # 23-31 are reserved for future ASPRS use
    
    # Extended Classifications (32-255) - User Defined
    # These can be customized for specific applications
    # Below are IGN BD TOPO® specific classifications
    
    # Road types (32-49)
    ROAD_MOTORWAY = 32                # Autoroute
    ROAD_PRIMARY = 33                 # Route principale
    ROAD_SECONDARY = 34               # Route secondaire  
    ROAD_TERTIARY = 35                # Route tertiaire
    ROAD_RESIDENTIAL = 36             # Rue résidentielle
    ROAD_SERVICE = 37                 # Route de service
    ROAD_PEDESTRIAN = 38              # Zone piétonne
    ROAD_CYCLEWAY = 39                # Piste cyclable
    ROAD_PARKING = 40                 # Parking
    ROAD_BRIDGE = 41                  # Pont routier
    ROAD_TUNNEL = 42                  # Tunnel routier
    ROAD_ROUNDABOUT = 43              # Rond-point
    
    # Building types (50-69)
    BUILDING_RESIDENTIAL = 50         # Bâtiment résidentiel
    BUILDING_COMMERCIAL = 51          # Bâtiment commercial
    BUILDING_INDUSTRIAL = 52          # Bâtiment industriel
    BUILDING_RELIGIOUS = 53           # Bâtiment religieux
    BUILDING_PUBLIC = 54              # Bâtiment public
    BUILDING_AGRICULTURAL = 55        # Bâtiment agricole
    BUILDING_SPORTS = 56              # Bâtiment sportif
    BUILDING_HISTORIC = 57            # Bâtiment historique
    BUILDING_ROOF = 58                # Toit
    BUILDING_WALL = 59                # Mur
    BUILDING_FACADE = 60              # Façade
    BUILDING_CHIMNEY = 61             # Cheminée
    BUILDING_BALCONY = 62             # Balcon
    
    # Vegetation types (70-79)
    VEGETATION_TREE = 70              # Arbre
    VEGETATION_BUSH = 71              # Buisson
    VEGETATION_GRASS = 72             # Herbe
    VEGETATION_HEDGE = 73             # Haie
    VEGETATION_FOREST = 74            # Forêt
    VEGETATION_VINEYARD = 75          # Vignoble
    VEGETATION_ORCHARD = 76           # Verger
    
    # Water types (80-89)
    WATER_RIVER = 80                  # Rivière
    WATER_LAKE = 81                   # Lac
    WATER_POND = 82                   # Étang
    WATER_CANAL = 83                  # Canal
    WATER_FOUNTAIN = 84               # Fontaine
    WATER_SWIMMING_POOL = 85          # Piscine
    
    # Infrastructure (90-109)
    RAILWAY_TRACK = 90                # Voie ferrée
    RAILWAY_PLATFORM = 91             # Quai de gare
    RAILWAY_BRIDGE = 92               # Pont ferroviaire
    RAILWAY_TUNNEL = 93               # Tunnel ferroviaire
    POWER_LINE = 94                   # Ligne électrique
    POWER_PYLON = 95                  # Pylône électrique
    ANTENNA = 96                      # Antenne
    STREET_LIGHT = 97                 # Lampadaire
    TRAFFIC_SIGN = 98                 # Panneau de signalisation
    FENCE = 99                        # Clôture
    WALL_STANDALONE = 100             # Mur indépendant
    
    # Urban furniture (110-119)
    BENCH = 110                       # Banc
    BIN = 111                         # Poubelle
    SHELTER = 112                     # Abri
    BOLLARD = 113                     # Borne
    BARRIER = 114                     # Barrière
    
    # Terrain (120-129)
    TERRAIN_BARE = 120                # Sol nu
    TERRAIN_GRAVEL = 121              # Gravier
    TERRAIN_SAND = 122                # Sable
    TERRAIN_ROCK = 123                # Roche
    TERRAIN_CLIFF = 124               # Falaise
    TERRAIN_QUARRY = 125              # Carrière
    
    # Vehicles (130-139)
    VEHICLE_CAR = 130                 # Voiture
    VEHICLE_TRUCK = 131               # Camion
    VEHICLE_BUS = 132                 # Bus
    VEHICLE_TRAIN = 133               # Train
    VEHICLE_BOAT = 134                # Bateau
    VEHICLE_AIRCRAFT = 135            # Avion


# ============================================================================
# Classification Names and Descriptions
# ============================================================================

ASPRS_CLASS_NAMES: Dict[int, str] = {
    0: "Created, Never Classified",
    1: "Unclassified",
    2: "Ground",
    3: "Low Vegetation",
    4: "Medium Vegetation",
    5: "High Vegetation",
    6: "Building",
    7: "Low Point (Noise)",
    8: "Reserved",
    9: "Water",
    10: "Rail",
    11: "Road Surface",
    12: "Reserved",
    13: "Wire - Guard (Shield)",
    14: "Wire - Conductor (Phase)",
    15: "Transmission Tower",
    16: "Wire-Structure Connector",
    17: "Bridge Deck",
    18: "High Noise",
    19: "Overhead Structure",
    20: "Ignored Ground",
    21: "Snow",
    22: "Temporal Exclusion",
    # Extended classifications
    32: "Motorway",
    33: "Primary Road",
    34: "Secondary Road",
    35: "Tertiary Road",
    36: "Residential Road",
    37: "Service Road",
    38: "Pedestrian Zone",
    39: "Cycleway",
    40: "Parking",  # BD TOPO® - Parking areas (Aires de stationnement)
    41: "Sports Facility",  # BD TOPO® - Sports facilities (Équipements sportifs)
    42: "Cemetery",  # BD TOPO® - Cemeteries (Cimetières)
    43: "Power Line",  # BD TOPO® - Power lines (Lignes électriques)
    44: "Agriculture",  # RPG - Agricultural land (Terres agricoles)
    50: "Residential Building",
    51: "Commercial Building",
    52: "Industrial Building",
    53: "Religious Building",
    54: "Public Building",
    55: "Agricultural Building",
    56: "Sports Building",
    57: "Historic Building",
    58: "Roof",
    59: "Wall",
    60: "Facade",
    61: "Chimney",
    62: "Balcony",
    70: "Tree",
    71: "Bush",
    72: "Grass",
    73: "Hedge",
    74: "Forest",
    75: "Vineyard",
    76: "Orchard",
    80: "River",
    81: "Lake",
    82: "Pond",
    83: "Canal",
    84: "Fountain",
    85: "Swimming Pool",
    90: "Railway Track",
    91: "Railway Platform",
    92: "Railway Bridge",
    93: "Railway Tunnel",
    94: "Power Line",
    95: "Power Pylon",
    96: "Antenna",
    97: "Street Light",
    98: "Traffic Sign",
    99: "Fence",
    100: "Standalone Wall",
    110: "Bench",
    111: "Bin",
    112: "Shelter",
    113: "Bollard",
    114: "Barrier",
    120: "Bare Terrain",
    121: "Gravel",
    122: "Sand",
    123: "Rock",
    124: "Cliff",
    125: "Quarry",
    130: "Car",
    131: "Truck",
    132: "Bus",
    133: "Train",
    134: "Boat",
    135: "Aircraft",
}


# ============================================================================
# IGN BD TOPO® Nature Attribute to ASPRS Classification Mapping
# ============================================================================

# Building nature to ASPRS classification
BUILDING_NATURE_TO_ASPRS: Dict[str, int] = {
    # Residential
    "Indifférencié": ASPRSClass.BUILDING,
    "Résidentiel": ASPRSClass.BUILDING_RESIDENTIAL,
    "Immeuble": ASPRSClass.BUILDING_RESIDENTIAL,
    "Maison": ASPRSClass.BUILDING_RESIDENTIAL,
    
    # Commercial
    "Commercial et services": ASPRSClass.BUILDING_COMMERCIAL,
    "Commercial": ASPRSClass.BUILDING_COMMERCIAL,
    
    # Industrial
    "Industriel": ASPRSClass.BUILDING_INDUSTRIAL,
    "Industriel, agricole ou commercial": ASPRSClass.BUILDING_INDUSTRIAL,
    "Serre": ASPRSClass.BUILDING_AGRICULTURAL,
    
    # Religious
    "Religieux": ASPRSClass.BUILDING_RELIGIOUS,
    "Chapelle": ASPRSClass.BUILDING_RELIGIOUS,
    "Église": ASPRSClass.BUILDING_RELIGIOUS,
    "Cathédrale": ASPRSClass.BUILDING_RELIGIOUS,
    
    # Public and services
    "Sportif": ASPRSClass.BUILDING_SPORTS,
    "Enseignement": ASPRSClass.BUILDING_PUBLIC,
    "Santé": ASPRSClass.BUILDING_PUBLIC,
    "Science et recherche": ASPRSClass.BUILDING_PUBLIC,
    "Transport": ASPRSClass.BUILDING_PUBLIC,
    "Gare": ASPRSClass.BUILDING_PUBLIC,
    
    # Agricultural
    "Agricole": ASPRSClass.BUILDING_AGRICULTURAL,
    
    # Historic/monument
    "Monument": ASPRSClass.BUILDING_HISTORIC,
    "Château": ASPRSClass.BUILDING_HISTORIC,
    "Fort": ASPRSClass.BUILDING_HISTORIC,
    "Remarquable": ASPRSClass.BUILDING_HISTORIC,
}

# Road nature to ASPRS classification
ROAD_NATURE_TO_ASPRS: Dict[str, int] = {
    "Autoroute": ASPRSClass.ROAD_MOTORWAY,
    "Quasi-autoroute": ASPRSClass.ROAD_MOTORWAY,
    "Route à 2 chaussées": ASPRSClass.ROAD_PRIMARY,
    "Route à 1 chaussée": ASPRSClass.ROAD_SECONDARY,
    "Route empierrée": ASPRSClass.ROAD_TERTIARY,
    "Chemin": ASPRSClass.ROAD_SERVICE,
    "Bretelle": ASPRSClass.ROAD_SERVICE,
    "Rond-point": ASPRSClass.ROAD_ROUNDABOUT,
    "Place": ASPRSClass.ROAD_PEDESTRIAN,
    "Sentier": ASPRSClass.ROAD_PEDESTRIAN,
    "Escalier": ASPRSClass.ROAD_PEDESTRIAN,
    "Piste cyclable": ASPRSClass.ROAD_CYCLEWAY,
    "Parking": ASPRSClass.ROAD_PARKING,
}

# Vegetation nature to ASPRS classification
VEGETATION_NATURE_TO_ASPRS: Dict[str, int] = {
    "Arbre": ASPRSClass.VEGETATION_TREE,
    "Haie": ASPRSClass.VEGETATION_HEDGE,
    "Bois": ASPRSClass.VEGETATION_FOREST,
    "Forêt fermée de feuillus": ASPRSClass.VEGETATION_FOREST,
    "Forêt fermée de conifères": ASPRSClass.VEGETATION_FOREST,
    "Forêt fermée mixte": ASPRSClass.VEGETATION_FOREST,
    "Forêt ouverte": ASPRSClass.VEGETATION_FOREST,
    "Lande ligneuse": ASPRSClass.VEGETATION_BUSH,
    "Verger": ASPRSClass.VEGETATION_ORCHARD,
    "Vigne": ASPRSClass.VEGETATION_VINEYARD,
    "Peupleraie": ASPRSClass.VEGETATION_TREE,
}

# Water nature to ASPRS classification
WATER_NATURE_TO_ASPRS: Dict[str, int] = {
    "Cours d'eau": ASPRSClass.WATER_RIVER,
    "Plan d'eau": ASPRSClass.WATER_LAKE,
    "Étang": ASPRSClass.WATER_POND,
    "Lac": ASPRSClass.WATER_LAKE,
    "Canal": ASPRSClass.WATER_CANAL,
    "Bassin": ASPRSClass.WATER_POND,
    "Réservoir": ASPRSClass.WATER_POND,
}

# Railway nature to ASPRS classification
# BD TOPO® railways: "troncon_de_voie_ferree" layer
# All railways map to ASPRS code 10 (Rail)
RAILWAY_NATURE_TO_ASPRS: Dict[str, int] = {
    # Main railway lines
    "Principale": 10,  # ASPRS_RAIL - Main railway line
    "LGV": 10,  # ASPRS_RAIL - High-speed line (Ligne à Grande Vitesse)
    "Voie de service": 10,  # ASPRS_RAIL - Service track
    "Voie ferrée": 10,  # ASPRS_RAIL - Generic railway track
    
    # Tram and metro
    "Tramway": 10,  # ASPRS_RAIL - Tram line
    "Métro": 10,  # ASPRS_RAIL - Metro line
    
    # Default for any railway
    "default": 10,  # ASPRS_RAIL
}

# Sports facility nature to ASPRS classification
# BD TOPO® sports: "terrain_de_sport" layer
# All sports facilities map to ASPRS code 41 (Sports)
SPORTS_NATURE_TO_ASPRS: Dict[str, int] = {
    # Ball sports
    "Terrain de football": 41,  # ASPRS_SPORTS - Soccer/football field
    "Terrain de rugby": 41,  # ASPRS_SPORTS - Rugby field
    "Terrain de tennis": 41,  # ASPRS_SPORTS - Tennis court
    "Terrain de basketball": 41,  # ASPRS_SPORTS - Basketball court
    "Terrain de handball": 41,  # ASPRS_SPORTS - Handball court
    "Terrain de volleyball": 41,  # ASPRS_SPORTS - Volleyball court
    
    # Track and field
    "Piste d'athlétisme": 41,  # ASPRS_SPORTS - Athletics track
    "Stade": 41,  # ASPRS_SPORTS - Stadium
    
    # Other sports
    "Terrain multisports": 41,  # ASPRS_SPORTS - Multi-sport field
    "Terrain de golf": 41,  # ASPRS_SPORTS - Golf course
    "Piscine": 41,  # ASPRS_SPORTS - Swimming pool
    "Skatepark": 41,  # ASPRS_SPORTS - Skate park
    "Terrain de pétanque": 41,  # ASPRS_SPORTS - Pétanque court
    "Terrain de sport": 41,  # ASPRS_SPORTS - Generic sports field
    
    # Default for any sports facility
    "default": 41,  # ASPRS_SPORTS
}

# Cemetery nature to ASPRS classification
# BD TOPO® cemeteries: "cimetiere" layer
# All cemeteries map to ASPRS code 42 (Cemetery)
CEMETERY_NATURE_TO_ASPRS: Dict[str, int] = {
    "Cimetière": 42,  # ASPRS_CEMETERY - Cemetery
    "Cimetière militaire": 42,  # ASPRS_CEMETERY - Military cemetery
    "Cimetière communal": 42,  # ASPRS_CEMETERY - Municipal cemetery
    "Cimetière paroissial": 42,  # ASPRS_CEMETERY - Parish cemetery
    "Ossuaire": 42,  # ASPRS_CEMETERY - Ossuary
    "Columbarium": 42,  # ASPRS_CEMETERY - Columbarium
    
    # Default for any cemetery
    "default": 42,  # ASPRS_CEMETERY
}

# Power line nature to ASPRS classification
# BD TOPO® power lines: "ligne_electrique" layer
# All power lines map to ASPRS code 43 (Power Line)
POWER_LINE_NATURE_TO_ASPRS: Dict[str, int] = {
    # High voltage lines
    "Ligne électrique": 43,  # ASPRS_POWER_LINE - Power line
    "Ligne haute tension": 43,  # ASPRS_POWER_LINE - High voltage line
    "Ligne moyenne tension": 43,  # ASPRS_POWER_LINE - Medium voltage line
    "Ligne basse tension": 43,  # ASPRS_POWER_LINE - Low voltage line
    
    # Underground/overhead
    "Aérienne": 43,  # ASPRS_POWER_LINE - Overhead line
    "Souterraine": 43,  # ASPRS_POWER_LINE - Underground line (for corridor)
    
    # Default for any power line
    "default": 43,  # ASPRS_POWER_LINE
}

# Parking nature to ASPRS classification
# BD TOPO® parking: "parking" layer
# All parking areas map to ASPRS code 40 (Parking)
PARKING_NATURE_TO_ASPRS: Dict[str, int] = {
    "Parking": 40,  # ASPRS_PARKING - Parking area
    "Parking souterrain": 40,  # ASPRS_PARKING - Underground parking
    "Parking aérien": 40,  # ASPRS_PARKING - Surface parking
    "Parking couvert": 40,  # ASPRS_PARKING - Covered parking
    "Aire de stationnement": 40,  # ASPRS_PARKING - Parking area
    "Place de parking": 40,  # ASPRS_PARKING - Parking space
    "Parc de stationnement": 40,  # ASPRS_PARKING - Parking lot
    
    # Default for any parking
    "default": 40,  # ASPRS_PARKING
}

# Bridge nature to ASPRS classification
# BD TOPO® bridges: "pont" layer
# All bridges map to ASPRS code 17 (Bridge Deck)
BRIDGE_NATURE_TO_ASPRS: Dict[str, int] = {
    "Pont": 17,  # ASPRS_BRIDGE - Bridge
    "Viaduc": 17,  # ASPRS_BRIDGE - Viaduct
    "Passerelle": 17,  # ASPRS_BRIDGE - Footbridge
    "Pont-route": 17,  # ASPRS_BRIDGE - Road bridge
    "Pont ferroviaire": 17,  # ASPRS_BRIDGE - Railway bridge
    "Aqueduc": 17,  # ASPRS_BRIDGE - Aqueduct
    
    # Default for any bridge
    "default": 17,  # ASPRS_BRIDGE
}


# ============================================================================
# Classification Mode Configuration
# ============================================================================

class ClassificationMode:
    """
    Classification mode for LAS output.
    
    - ASPRS_STANDARD: Use standard ASPRS codes (0-31)
    - ASPRS_EXTENDED: Use ASPRS codes + extended codes (32-255)
    - LOD2: Use LOD2 building-focused classes (for training)
    - LOD3: Use LOD3 detailed building classes (for training)
    """
    ASPRS_STANDARD = "asprs_standard"
    ASPRS_EXTENDED = "asprs_extended"
    LOD2 = "lod2"
    LOD3 = "lod3"


def get_classification_for_building(
    nature: Optional[str] = None,
    mode: str = ClassificationMode.ASPRS_EXTENDED
) -> int:
    """
    Get ASPRS classification code for a building based on its nature.
    
    Args:
        nature: Building nature from BD TOPO®
        mode: Classification mode
        
    Returns:
        ASPRS classification code
    """
    if mode == ClassificationMode.ASPRS_STANDARD:
        return ASPRSClass.BUILDING
    
    if nature and nature in BUILDING_NATURE_TO_ASPRS:
        return BUILDING_NATURE_TO_ASPRS[nature]
    
    return ASPRSClass.BUILDING


def get_classification_for_road(
    nature: Optional[str] = None,
    mode: str = ClassificationMode.ASPRS_EXTENDED
) -> int:
    """
    Get ASPRS classification code for a road based on its nature.
    
    Args:
        nature: Road nature from BD TOPO®
        mode: Classification mode
        
    Returns:
        ASPRS classification code
    """
    if mode == ClassificationMode.ASPRS_STANDARD:
        return ASPRSClass.ROAD_SURFACE
    
    if nature and nature in ROAD_NATURE_TO_ASPRS:
        return ROAD_NATURE_TO_ASPRS[nature]
    
    return ASPRSClass.ROAD_SURFACE


def get_classification_for_vegetation(
    nature: Optional[str] = None,
    height: Optional[float] = None,
    mode: str = ClassificationMode.ASPRS_EXTENDED
) -> int:
    """
    Get ASPRS classification code for vegetation based on nature and height.
    
    Args:
        nature: Vegetation nature from BD TOPO®
        height: Vegetation height in meters
        mode: Classification mode
        
    Returns:
        ASPRS classification code
    """
    if mode == ClassificationMode.ASPRS_STANDARD:
        # Use height-based classification for standard mode
        if height is not None:
            if height < 0.5:
                return ASPRSClass.LOW_VEGETATION
            elif height < 2.0:
                return ASPRSClass.MEDIUM_VEGETATION
            else:
                return ASPRSClass.HIGH_VEGETATION
        return ASPRSClass.MEDIUM_VEGETATION
    
    # Extended mode: use nature if available
    if nature and nature in VEGETATION_NATURE_TO_ASPRS:
        return VEGETATION_NATURE_TO_ASPRS[nature]
    
    # Fallback to height-based
    if height is not None:
        if height < 0.5:
            return ASPRSClass.LOW_VEGETATION
        elif height < 2.0:
            return ASPRSClass.MEDIUM_VEGETATION
        else:
            return ASPRSClass.HIGH_VEGETATION
    
    return ASPRSClass.MEDIUM_VEGETATION


def get_classification_for_water(
    nature: Optional[str] = None,
    mode: str = ClassificationMode.ASPRS_EXTENDED
) -> int:
    """
    Get ASPRS classification code for water based on its nature.
    
    Args:
        nature: Water nature from BD TOPO®
        mode: Classification mode
        
    Returns:
        ASPRS classification code
    """
    if mode == ClassificationMode.ASPRS_STANDARD:
        return ASPRSClass.WATER
    
    if nature and nature in WATER_NATURE_TO_ASPRS:
        return WATER_NATURE_TO_ASPRS[nature]
    
    return ASPRSClass.WATER


def get_classification_for_railway(
    nature: Optional[str] = None,
    mode: str = ClassificationMode.ASPRS_STANDARD
) -> int:
    """
    Get ASPRS classification code for a railway based on its nature.
    
    All railways use ASPRS code 10 (Rail) regardless of type.
    
    Args:
        nature: Railway nature from BD TOPO®
        mode: Classification mode (not used, all return 10)
        
    Returns:
        ASPRS classification code (always 10 for railways)
    """
    # All railway types map to ASPRS code 10
    return 10


def get_classification_for_sports(
    nature: Optional[str] = None,
    mode: str = ClassificationMode.ASPRS_EXTENDED
) -> int:
    """
    Get ASPRS classification code for a sports facility based on its nature.
    
    All sports facilities use ASPRS code 41 (Sports).
    
    Args:
        nature: Sports facility nature from BD TOPO®
        mode: Classification mode
        
    Returns:
        ASPRS classification code (always 41 for sports)
    """
    if mode == ClassificationMode.ASPRS_STANDARD:
        # In standard mode, sports facilities are treated as unclassified
        return 1
    
    # Extended mode: all sports facilities use code 41
    return 41


def get_classification_for_cemetery(
    nature: Optional[str] = None,
    mode: str = ClassificationMode.ASPRS_EXTENDED
) -> int:
    """
    Get ASPRS classification code for a cemetery based on its nature.
    
    All cemeteries use ASPRS code 42 (Cemetery).
    
    Args:
        nature: Cemetery nature from BD TOPO®
        mode: Classification mode
        
    Returns:
        ASPRS classification code (always 42 for cemeteries)
    """
    if mode == ClassificationMode.ASPRS_STANDARD:
        # In standard mode, cemeteries are treated as unclassified
        return 1
    
    # Extended mode: all cemeteries use code 42
    return 42


def get_classification_for_power_line(
    nature: Optional[str] = None,
    mode: str = ClassificationMode.ASPRS_EXTENDED
) -> int:
    """
    Get ASPRS classification code for a power line based on its nature.
    
    All power lines use ASPRS code 43 (Power Line).
    
    Args:
        nature: Power line nature from BD TOPO®
        mode: Classification mode
        
    Returns:
        ASPRS classification code (always 43 for power lines)
    """
    if mode == ClassificationMode.ASPRS_STANDARD:
        # In standard mode, power lines are treated as unclassified
        return 1
    
    # Extended mode: all power lines use code 43
    return 43


def get_classification_for_parking(
    nature: Optional[str] = None,
    mode: str = ClassificationMode.ASPRS_EXTENDED
) -> int:
    """
    Get ASPRS classification code for a parking area based on its nature.
    
    All parking areas use ASPRS code 40 (Parking).
    
    Args:
        nature: Parking nature from BD TOPO®
        mode: Classification mode
        
    Returns:
        ASPRS classification code (always 40 for parking)
    """
    if mode == ClassificationMode.ASPRS_STANDARD:
        # In standard mode, parking is treated as unclassified
        return 1
    
    # Extended mode: all parking areas use code 40
    return 40


def get_classification_for_bridge(
    nature: Optional[str] = None,
    mode: str = ClassificationMode.ASPRS_STANDARD
) -> int:
    """
    Get ASPRS classification code for a bridge based on its nature.
    
    All bridges use ASPRS code 17 (Bridge Deck).
    
    Args:
        nature: Bridge nature from BD TOPO®
        mode: Classification mode
        
    Returns:
        ASPRS classification code (always 17 for bridges)
    """
    # All bridge types map to ASPRS code 17 (standard code)
    return 17


def get_class_name(code: int) -> str:
    """
    Get human-readable name for an ASPRS classification code.
    
    Args:
        code: ASPRS classification code
        
    Returns:
        Classification name
    """
    return ASPRS_CLASS_NAMES.get(code, f"User Defined ({code})")


def get_class_color(code: int) -> Tuple[int, int, int]:
    """
    Get RGB color for visualization of an ASPRS classification code.
    
    Args:
        code: ASPRS classification code
        
    Returns:
        RGB color tuple (0-255)
    """
    # Standard ASPRS colors
    color_map = {
        0: (128, 128, 128),  # Never classified - gray
        1: (200, 200, 200),  # Unclassified - light gray
        2: (165, 82, 42),    # Ground - brown
        3: (144, 238, 144),  # Low vegetation - light green
        4: (60, 179, 113),   # Medium vegetation - medium green
        5: (34, 139, 34),    # High vegetation - dark green
        6: (255, 0, 0),      # Building - red
        7: (255, 255, 0),    # Low point - yellow
        8: (128, 128, 128),  # Reserved - gray
        9: (0, 0, 255),      # Water - blue
        10: (128, 0, 128),   # Rail - purple
        11: (0, 0, 0),       # Road - black
        17: (139, 69, 19),   # Bridge - saddle brown
        18: (255, 165, 0),   # High noise - orange
    }
    
    # Extended colors for roads
    if 32 <= code <= 43:
        return (64, 64, 64)  # Dark gray for all roads
    
    # Extended colors for buildings
    if 50 <= code <= 62:
        return (220, 20, 60)  # Crimson for all buildings
    
    # Extended colors for vegetation
    if 70 <= code <= 76:
        return (50, 205, 50)  # Lime green for all vegetation
    
    # Extended colors for water
    if 80 <= code <= 85:
        return (30, 144, 255)  # Dodger blue for all water
    
    return color_map.get(code, (128, 128, 128))


# ============================================================================
# Feature Utility Functions
# ============================================================================

def get_required_features_for_class(asprs_class: int) -> List[str]:
    """
    Get list of required features for refining a specific ASPRS class.
    
    Args:
        asprs_class: ASPRS classification code
        
    Returns:
        List of required feature names
        
    Example:
        >>> get_required_features_for_class(9)  # Water
        ['height', 'planarity', 'curvature', 'normals']
        >>> get_required_features_for_class(3)  # Low vegetation
        ['ndvi', 'height', 'curvature', 'planarity', 'sphericity', 'roughness']
    """
    feature_map = {
        9: WATER_FEATURES,                    # Water
        11: ROAD_FEATURES,                    # Road
        3: VEGETATION_FEATURES,               # Low vegetation
        4: VEGETATION_FEATURES,               # Medium vegetation
        5: VEGETATION_FEATURES,               # High vegetation
        6: BUILDING_FEATURES,                 # Building
    }
    
    return feature_map.get(asprs_class, [])


def get_all_required_features() -> List[str]:
    """
    Get complete list of all features needed for ASPRS classification.
    
    Returns:
        List of all unique feature names required
    """
    return ALL_CLASSIFICATION_FEATURES.copy()


def get_feature_description(feature_name: str) -> str:
    """
    Get description of a feature.
    
    Args:
        feature_name: Name of the feature
        
    Returns:
        Human-readable description of the feature
    """
    return FEATURE_DESCRIPTIONS.get(feature_name, f"Unknown feature: {feature_name}")


def get_feature_range(feature_name: str) -> Tuple[float, float]:
    """
    Get expected value range for a feature.
    
    Args:
        feature_name: Name of the feature
        
    Returns:
        Tuple of (min_value, max_value)
    """
    return FEATURE_RANGES.get(feature_name, (float('-inf'), float('inf')))


def validate_features(features: Dict[str, Any], required_features: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that all required features are present.
    
    Args:
        features: Dictionary of available features
        required_features: List of required feature names
        
    Returns:
        Tuple of (is_valid, missing_features)
        
    Example:
        >>> features = {'height': np.array([...]), 'planarity': np.array([...])}
        >>> required = ['height', 'planarity', 'curvature']
        >>> is_valid, missing = validate_features(features, required)
        >>> print(f"Valid: {is_valid}, Missing: {missing}")
        Valid: False, Missing: ['curvature']
    """
    missing = [f for f in required_features if f not in features or features[f] is None]
    return len(missing) == 0, missing

