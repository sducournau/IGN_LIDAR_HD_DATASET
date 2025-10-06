---
sidebar_position: 4
title: Architectural Styles
description: Regional building patterns and architectural classification
keywords: [architecture, buildings, styles, regional, classification]
---

# Architectural Styles

Understanding regional architectural patterns is crucial for accurate building classification and feature extraction from IGN LiDAR HD data.

## Overview

French architectural styles vary significantly by region, historical period, and urban vs rural context. This guide helps configure the processing pipeline for optimal results across different architectural contexts.

## Regional Classifications

### Ile-de-France (Paris Region)

**Characteristics:**

- Dense urban fabric with Haussmanian boulevards
- Uniform building heights (6-7 stories typical)
- Zinc roofing with characteristic slope angles
- Enclosed courtyards and consistent street alignment

**Processing Configuration:**

```python
ile_de_france_config = {
    "building_detection": {
        "min_height": 8.0,  # Typical story height
        "max_height": 25.0,  # Haussmanian limit
        "roof_slope_range": [25, 45],  # Degrees
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

**Characteristics:**

- Mediterranean flat roofs and terraces
- Stone and stucco construction
- Lower building heights
- Irregular urban patterns

**Processing Configuration:**

```python
provence_config = {
    "building_detection": {
        "min_height": 3.0,
        "max_height": 15.0,
        "flat_roof_threshold": 5,  # Degrees max slope
        "terrace_detection": True
    },
    "materials": {
        "stone_detection": True,
        "tile_roofing": True,
        "stucco_surfaces": True
    }
}
```

### Brittany (Bretagne)

**Characteristics:**

- Traditional granite construction
- Slate roofing with steep slopes
- Scattered rural settlements
- Maritime influence on building orientation

**Processing Configuration:**

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

## Historical Periods

### Medieval Architecture (Pre-1500)

**Features:**

- Irregular building footprints
- Thick walls (>50cm typical)
- Small windows and openings
- Defensive characteristics

```python
medieval_features = {
    "wall_thickness": {"min": 0.5, "typical": 0.8},
    "window_ratio": {"max": 0.15},  # Window to wall ratio
    "footprint_regularity": {"threshold": 0.3},
    "defensive_elements": True
}
```

### Classical Architecture (1500-1800)

**Features:**

- Geometric regularity and symmetry
- Standardized proportions
- Formal gardens and courtyards
- Stone construction with carved details

```python
classical_features = {
    "symmetry_detection": True,
    "proportion_analysis": True,
    "courtyard_geometry": "formal",
    "material_refinement": "high"
}
```

### Industrial Architecture (1800-1950)

**Features:**

- Large span structures
- Brick and steel construction
- Repetitive bay systems
- Functional over decorative design

```python
industrial_features = {
    "span_detection": {"min": 10.0, "max": 50.0},
    "bay_repetition": True,
    "material_types": ["brick", "steel", "concrete"],
    "chimney_detection": True
}
```

### Contemporary Architecture (1950+)

**Features:**

- Diverse materials and forms
- Curtain wall systems
- Irregular geometries
- Mixed-use developments

```python
contemporary_features = {
    "material_diversity": True,
    "geometric_complexity": "high",
    "curtain_wall_detection": True,
    "mixed_use_analysis": True
}
```

## Building Typologies

### Residential Buildings

#### Single-Family Homes

```python
residential_single = {
    "footprint_area": {"min": 80, "max": 300},  # m²
    "height_range": {"min": 4, "max": 12},      # meters
    "roof_types": ["gable", "hip", "mansard"],
    "garden_detection": True
}
```

#### Multi-Family Housing

```python
residential_multi = {
    "footprint_area": {"min": 200, "max": 2000},
    "height_range": {"min": 8, "max": 30},
    "balcony_detection": True,
    "courtyard_likelihood": 0.7
}
```

### Commercial Buildings

#### Retail/Shops

```python
commercial_retail = {
    "ground_floor_height": {"min": 3.5, "max": 6.0},
    "large_windows": True,
    "signage_detection": True,
    "street_frontage": True
}
```

#### Office Buildings

```python
commercial_office = {
    "repetitive_floors": True,
    "curtain_walls": True,
    "regular_geometry": True,
    "parking_detection": True
}
```

### Industrial Buildings

#### Manufacturing

```python
industrial_manufacturing = {
    "large_spans": True,
    "high_ceilings": {"min": 8, "max": 25},
    "loading_docks": True,
    "minimal_windows": True
}
```

## Configuration Examples

### Urban Context Processing

```python
from ign_lidar import Processor, ArchitecturalAnalyzer

# Initialize with urban architectural context
processor = Processor()
analyzer = ArchitecturalAnalyzer(
    region="ile_de_france",
    urban_context="dense_urban",
    historical_period="haussmanian"
)

# Process with architectural awareness
result = processor.process_tile(
    tile_path="paris_tile.las",
    architectural_context=analyzer,
    enable_style_classification=True
)
```

### Rural Context Processing

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

## Style Classification Pipeline

### Automatic Style Detection

```python
def detect_architectural_style(building_features):
    """
    Automatically detect architectural style from extracted features
    """
    style_indicators = {
        "roof_slope": building_features["roof_slope_mean"],
        "wall_thickness": building_features["wall_thickness_mean"],
        "window_ratio": building_features["window_to_wall_ratio"],
        "regularity": building_features["geometric_regularity"],
        "height": building_features["building_height"]
    }

    # Style classification logic
    if style_indicators["roof_slope"] > 45 and style_indicators["wall_thickness"] > 0.6:
        return "traditional_rural"
    elif style_indicators["regularity"] > 0.8 and 15 < style_indicators["height"] < 25:
        return "haussmanian"
    elif style_indicators["window_ratio"] > 0.6:
        return "contemporary"
    else:
        return "mixed_urban"
```

### Manual Style Configuration

```python
# Define custom architectural style
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

## Performance Considerations

### Memory Usage by Style Complexity

- **Simple Rural**: ~200MB per km²
- **Urban Residential**: ~500MB per km²
- **Dense Urban**: ~800MB per km²
- **Mixed Architecture**: ~1GB per km²

### Processing Time Impact

- Style classification adds ~15% to processing time
- Detailed architectural analysis: +30-40%
- Historical pattern recognition: +20%

## Validation and Accuracy

### Ground Truth Comparison

- Manual architectural survey validation
- Historical cadastral data correlation
- Expert architectural assessment

### Accuracy Metrics by Style

| Style Type        | Classification Accuracy | Feature Detection |
| ----------------- | ----------------------- | ----------------- |
| Traditional Rural | 89%                     | 85%               |
| Haussmanian Urban | 94%                     | 91%               |
| Contemporary      | 78%                     | 82%               |
| Industrial        | 92%                     | 88%               |

## Related Documentation

- [LOD3 Classification](./lod3-classification)
- [Building Feature Extraction](../api/features)
- [Regional Processing Guide](../guides/regional-processing)
- [Historical Analysis Tools](../reference/historical-analysis)
