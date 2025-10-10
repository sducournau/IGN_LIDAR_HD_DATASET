---
sidebar_position: 4
title: Styles Architecturaux
description: Motifs de bâtiments régionaux et classification architecturale
keywords: [architecture, bâtiments, styles, régional, classification]
---

# Styles Architecturaux

Comprendre les motifs architecturaux régionaux est crucial pour une classification précise des bâtiments et l'extraction de caractéristiques à partir des données IGN LiDAR HD.

## Vue d'ensemble

Les styles architecturaux français varient considérablement selon la région, la période historique et le contexte urbain vs rural. Ce guide aide à configurer le pipeline de traitement pour des résultats optimaux à travers différents contextes architecturaux.

## Classifications Régionales

### Île-de-France (Région Parisienne)

**Caractéristiques :**

- Tissu urbain dense avec boulevards haussmanniens
- Hauteurs bâtiment uniformes (6-7 étages typiques)
- Toiture zinc avec angles pente caractéristiques
- Cours fermées et alignement rue cohérent

**Configuration Traitement :**

```python
ile_de_france_config = {
    "building_detection": {
        "min_height": 8.0,  # Hauteur étage typique
        "max_height": 25.0,  # Limite haussmannienne
        "roof_slope_range": [25, 45],  # Degrés
        "courtyard_detection": True
    },
    "architectural_features": {
        "mansard_roofs": True,
        "zinc_material": True,
        "balcony_detection": True
    }
}
```

### Provence-Alpes-Côte d'Azur

**Caractéristiques :**

- Toits plats méditerranéens et terrasses
- Construction pierre et crépi
- Hauteurs bâtiment plus basses
- Motifs urbains irréguliers

**Configuration Traitement :**

```python
provence_config = {
    "building_detection": {
        "min_height": 3.0,
        "max_height": 15.0,
        "flat_roof_threshold": 5,  # Pente max degrés
        "terrace_detection": True
    },
    "materials": {
        "stone_detection": True,
        "tile_roofing": True,
        "stucco_surfaces": True
    }
}
```

### Bretagne

**Caractéristiques :**

- Construction granit traditionnel
- Toiture ardoise avec pentes raides
- Habitats ruraux dispersés
- Influence maritime sur orientation bâtiment

**Configuration Traitement :**

```python
brittany_config = {
    "building_detection": {
        "min_height": 4.0,
        "max_height": 12.0,
        "roof_slope_range": [35, 55],
        "wind_orientation": True
    },
    "materials": {
        "granite_detection": True,
        "slate_roofing": True,
        "chimney_prominence": True
    }
}
```

## Périodes Historiques

### Architecture Médiévale (Avant 1500)

**Caractéristiques :**

- Emprises bâtiment irrégulières
- Murs épais (>50cm typique)
- Petites fenêtres et ouvertures
- Caractéristiques défensives

```python
medieval_features = {
    "wall_thickness": {"min": 0.5, "typical": 0.8},
    "window_ratio": {"max": 0.15},  # Ratio fenêtre/mur
    "footprint_regularity": {"threshold": 0.3},
    "defensive_elements": True
}
```

### Architecture Classique (1500-1800)

**Caractéristiques :**

- Régularité et symétrie géométriques
- Proportions standardisées
- Jardins et cours formels
- Construction pierre avec détails sculptés

```python
classical_features = {
    "symmetry_detection": True,
    "proportion_analysis": True,
    "courtyard_geometry": "formal",
    "material_refinement": "high"
}
```

### Architecture Industrielle (1800-1950)

**Caractéristiques :**

- Structures grande portée
- Construction brique et acier
- Systèmes travées répétitives
- Design fonctionnel sur décoratif

```python
industrial_features = {
    "span_detection": {"min": 10.0, "max": 50.0},
    "bay_repetition": True,
    "material_types": ["brick", "steel", "concrete"],
    "chimney_detection": True
}
```

### Architecture Contemporaine (1950+)

**Caractéristiques :**

- Matériaux et formes diverses
- Systèmes murs-rideaux
- Géométries irrégulières
- Développements usage mixte

```python
contemporary_features = {
    "material_diversity": True,
    "geometric_complexity": "high",
    "curtain_wall_detection": True,
    "mixed_use_analysis": True
}
```

## Typologies Bâtiment

### Bâtiments Résidentiels

#### Maisons Individuelles

```python
residential_single = {
    "footprint_area": {"min": 80, "max": 300},  # m²
    "height_range": {"min": 4, "max": 12},      # mètres
    "roof_types": ["gable", "hip", "mansard"],
    "garden_detection": True
}
```

#### Habitat Collectif

```python
residential_multi = {
    "footprint_area": {"min": 200, "max": 2000},
    "height_range": {"min": 8, "max": 30},
    "balcony_detection": True,
    "courtyard_likelihood": 0.7
}
```

### Bâtiments Commerciaux

#### Commerce/Magasins

```python
commercial_retail = {
    "ground_floor_height": {"min": 3.5, "max": 6.0},
    "large_windows": True,
    "signage_detection": True,
    "street_frontage": True
}
```

#### Immeubles Bureaux

```python
commercial_office = {
    "repetitive_floors": True,
    "curtain_walls": True,
    "regular_geometry": True,
    "parking_detection": True
}
```

### Bâtiments Industriels

#### Fabrication

```python
industrial_manufacturing = {
    "large_spans": True,
    "high_ceilings": {"min": 8, "max": 25},
    "loading_docks": True,
    "minimal_windows": True
}
```

## Exemples Configuration

### Traitement Contexte Urbain

```python
from ign_lidar import Processor, ArchitecturalAnalyzer

# Initialiser avec contexte architectural urbain
processor = Processor()
analyzer = ArchitecturalAnalyzer(
    region="ile_de_france",
    urban_context="dense_urban",
    historical_period="haussmanian"
)

# Traiter avec conscience architecturale
result = processor.process_tile(
    tile_path="paris_tile.las",
    architectural_context=analyzer,
    enable_style_classification=True
)
```

### Traitement Contexte Rural

```python
analyzer = ArchitecturalAnalyzer(
    region="brittany",
    urban_context="rural",
    building_density="scattered"
)

result = processor.process_tile(
    tile_path="rural_tile.las",
    architectural_context=analyzer,
    preserve_vernacular_features=True
)
```

## Pipeline Classification Style

### Détection Style Automatique

```python
def detect_architectural_style(building_features):
    """
    Détecte automatiquement style architectural depuis caractéristiques extraites
    """
    style_indicators = {
        "roof_slope": building_features["roof_slope_mean"],
        "wall_thickness": building_features["wall_thickness_mean"],
        "window_ratio": building_features["window_to_wall_ratio"],
        "regularity": building_features["geometric_regularity"],
        "height": building_features["building_height"]
    }

    # Logique classification style
    if style_indicators["roof_slope"] > 45 and style_indicators["wall_thickness"] > 0.6:
        return "traditional_rural"
    elif style_indicators["regularity"] > 0.8 and 15 < style_indicators["height"] < 25:
        return "haussmanian"
    elif style_indicators["window_ratio"] > 0.6:
        return "contemporary"
    else:
        return "mixed_urban"
```

### Configuration Style Manuel

```python
# Définir style architectural personnalisé
custom_style = {
    "name": "art_deco_paris",
    "period": "1920-1940",
    "characteristics": {
        "stepped_facades": True,
        "ornamental_details": True,
        "vertical_emphasis": True,
        "mixed_materials": True
    },
    "detection_parameters": {
        "facade_complexity": {"min": 0.6},
        "height_variation": {"tolerance": 0.15},
        "material_transitions": {"detect": True}
    }
}

analyzer.add_custom_style(custom_style)
```

## Considérations Performance

### Usage Mémoire par Complexité Style

- **Rural Simple** : ~200MB par km²
- **Résidentiel Urbain** : ~500MB par km²
- **Urbain Dense** : ~800MB par km²
- **Architecture Mixte** : ~1GB par km²

### Impact Temps Traitement

- Classification style ajoute ~15% au temps traitement
- Analyse architecturale détaillée : +30-40%
- Reconnaissance motifs historiques : +20%

## Validation et Précision

### Comparaison Vérité Terrain

- Validation enquête architecturale manuelle
- Corrélation données cadastrales historiques
- Évaluation expert architectural

### Métriques Précision par Style

| Type Style          | Précision Classification | Détection Caractéristiques |
| ------------------- | ------------------------ | -------------------------- |
| Rural Traditionnel  | 89%                      | 85%                        |
| Urbain Haussmannien | 94%                      | 91%                        |
| Contemporain        | 78%                      | 82%                        |
| Industriel          | 92%                      | 88%                        |

## Documentation Associée

- [Classification LOD3](./lod3-classification.md)
- [Extraction Caractéristiques Bâtiment](../api/features.md)
- [Guide Traitement Régional](../guides/regional-processing.md)
- [Outils Analyse Historique](../reference/historical-analysis.md)
