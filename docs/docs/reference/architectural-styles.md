---
sidebar_position: 3
title: Architectural Styles
description: French architectural style detection and classification
keywords: [architecture, buildings, styles, france, heritage]
---

# Architectural Styles

Classification and analysis of French architectural styles using LiDAR data.

## Overview

The IGN LiDAR HD dataset enables detailed analysis of architectural styles across French territories, supporting:

- **Heritage preservation** - Document historical buildings
- **Urban planning** - Understand architectural patterns
- **Cultural mapping** - Identify regional variations
- **Building classification** - Automated style recognition

## Style Classification

### Traditional French Styles

#### Haussmann (1853-1870)

Characteristic of central Paris, uniform height and façade organization.

```python
from ign_lidar.styles import HaussmannDetector

detector = HaussmannDetector()
buildings = detector.classify(lidar_data, criteria={
    'height_uniformity': 0.9,  # Very uniform heights
    'facade_alignment': 0.95,  # Aligned street facades
    'courtyard_ratio': 0.3,    # Inner courtyards
    'roof_type': 'mansard'     # Characteristic roofs
})
```

**Detection Features**:

- Consistent building heights (6-7 stories)
- Aligned street frontages
- Internal courtyards
- Mansard roofs with dormers
- Continuous urban fabric

#### Art Nouveau (1890-1910)

Organic forms, decorative elements, innovative materials.

```python
detector = ArtNouveauDetector()
buildings = detector.classify(lidar_data, criteria={
    'facade_complexity': 'high',
    'decorative_elements': True,
    'asymmetric_composition': True,
    'material_diversity': 'iron_glass_stone'
})
```

**Detection Features**:

- Irregular rooflines
- Curved architectural elements
- Glass and iron integration
- Asymmetrical compositions
- Ornamental details

#### Regional Vernacular

Traditional styles specific to French regions.

```python
# Alsatian timber-framed houses
alsatian_config = {
    'timber_framing': True,
    'steep_roofs': 45,  # degrees
    'gable_orientation': 'street_facing',
    'materials': ['timber', 'stone', 'render']
}

# Norman colombage
norman_config = {
    'half_timbered': True,
    'thatched_roofs': True,
    'irregular_plan': True,
    'courtyard_farms': True
}

# Provençal mas
provencal_config = {
    'stone_construction': True,
    'low_pitch_roofs': 30,  # degrees
    'tile_covering': 'terracotta',
    'thick_walls': True
}
```

### Modern Styles

#### Art Deco (1920-1939)

Geometric patterns, vertical emphasis, luxury materials.

```python
detector = ArtDecoDetector()
buildings = detector.classify(lidar_data, criteria={
    'geometric_patterns': True,
    'vertical_emphasis': 'strong',
    'setback_terraces': True,
    'decorative_cornices': True
})
```

#### Modernist/Bauhaus (1920-1960)

Functional design, flat roofs, geometric forms.

```python
detector = ModernistDetector()
buildings = detector.classify(lidar_data, criteria={
    'flat_roofs': 0.8,        # Mostly flat roofs
    'geometric_forms': True,
    'minimal_decoration': True,
    'large_windows': True,
    'white_facades': 0.6      # Often white/light colored
})
```

#### Brutalist (1950-1980)

Massive concrete structures, geometric forms.

```python
detector = BrutalistDetector()
buildings = detector.classify(lidar_data, criteria={
    'concrete_construction': True,
    'massive_forms': True,
    'minimal_windows': True,
    'geometric_complexity': 'high',
    'monumental_scale': True
})
```

## Implementation

### Style Detection Pipeline

```python
from ign_lidar import Processor, StyleClassifier

class ArchitecturalAnalyzer:
    def __init__(self):
        self.processor = Processor()
        self.classifier = StyleClassifier()

    def analyze_buildings(self, tile_path):
        """Analyze architectural styles in a LiDAR tile."""

        # Extract buildings
        buildings = self.processor.extract_buildings(tile_path)

        # Classify styles
        styles = {}
        for building_id, building in buildings.items():
            style = self.classifier.predict_style(building)
            styles[building_id] = style

        return styles

    def generate_style_map(self, tiles):
        """Generate architectural style map."""

        style_map = {}
        for tile in tiles:
            tile_styles = self.analyze_buildings(tile)
            style_map.update(tile_styles)

        return self.create_visualization(style_map)
```

### Feature Extraction

```python
def extract_architectural_features(building_points):
    """Extract features relevant to architectural style."""

    features = {
        # Geometric features
        'height': calculate_height(building_points),
        'footprint_area': calculate_area(building_points),
        'perimeter': calculate_perimeter(building_points),
        'compactness': calculate_compactness(building_points),

        # Roof features
        'roof_type': classify_roof_type(building_points),
        'roof_pitch': calculate_roof_pitch(building_points),
        'roof_complexity': measure_roof_complexity(building_points),

        # Facade features
        'facade_complexity': analyze_facade(building_points),
        'window_patterns': detect_openings(building_points),
        'setbacks': detect_setbacks(building_points),

        # Context features
        'urban_context': analyze_context(building_points),
        'alignment': check_street_alignment(building_points),
        'density': calculate_local_density(building_points)
    }

    return features
```

### Machine Learning Classification

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class StyleClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        self.feature_names = [
            'height', 'area', 'compactness', 'roof_pitch',
            'facade_complexity', 'window_regularity', 'setback_count'
        ]

    def train(self, training_data):
        """Train the style classifier."""

        features = []
        labels = []

        for building, style in training_data:
            feature_vector = self.extract_features(building)
            features.append(feature_vector)
            labels.append(style)

        X = np.array(features)
        y = np.array(labels)

        self.model.fit(X, y)

    def predict_style(self, building):
        """Predict architectural style."""

        features = self.extract_features(building)
        style = self.model.predict([features])[0]
        confidence = self.model.predict_proba([features]).max()

        return {
            'style': style,
            'confidence': confidence,
            'features': features
        }
```

## Regional Variations

### Northern France

```python
northern_styles = {
    'flemish_baroque': {
        'gabled_facades': True,
        'brick_construction': True,
        'ornate_details': True
    },
    'picard_traditional': {
        'timber_framing': True,
        'brick_infill': True,
        'steep_roofs': True
    }
}
```

### Central France

```python
central_styles = {
    'loire_renaissance': {
        'stone_construction': True,
        'decorative_turrets': True,
        'formal_gardens': True
    },
    'berry_vernacular': {
        'limestone_walls': True,
        'slate_roofs': True,
        'agricultural_buildings': True
    }
}
```

### Southern France

```python
southern_styles = {
    'mediterranean': {
        'flat_roofs': True,
        'tile_roofs': True,
        'stone_construction': True,
        'courtyards': True
    },
    'basque': {
        'timber_framing': True,
        'white_walls': True,
        'red_tile_roofs': True,
        'mountain_adaptation': True
    }
}
```

## Validation and Quality Control

### Ground Truth Comparison

```python
def validate_classification(predictions, ground_truth):
    """Validate architectural style predictions."""

    accuracy_metrics = {
        'overall_accuracy': calculate_accuracy(predictions, ground_truth),
        'style_precision': calculate_precision_by_style(predictions, ground_truth),
        'style_recall': calculate_recall_by_style(predictions, ground_truth),
        'confusion_matrix': create_confusion_matrix(predictions, ground_truth)
    }

    return accuracy_metrics
```

### Expert Validation

```python
def expert_review_pipeline(classified_buildings):
    """Submit classifications for expert review."""

    # Prioritize uncertain classifications
    uncertain = filter_by_confidence(classified_buildings, threshold=0.7)

    # Generate review materials
    review_package = {
        'building_images': generate_building_views(uncertain),
        'feature_summaries': create_feature_summaries(uncertain),
        'context_maps': create_context_visualizations(uncertain)
    }

    return review_package
```

## Applications

### Heritage Documentation

```python
def heritage_survey(region_tiles):
    """Survey heritage buildings in region."""

    analyzer = ArchitecturalAnalyzer()
    heritage_buildings = []

    for tile in region_tiles:
        styles = analyzer.analyze_buildings(tile)

        # Identify heritage styles
        heritage = {
            building_id: style
            for building_id, style in styles.items()
            if style['style'] in HERITAGE_STYLES and
               style['confidence'] > 0.8
        }

        heritage_buildings.extend(heritage)

    return create_heritage_database(heritage_buildings)
```

### Urban Evolution Analysis

```python
def analyze_urban_evolution(historical_tiles):
    """Analyze architectural evolution over time."""

    evolution_data = {}

    for year, tiles in historical_tiles.items():
        styles = analyze_architectural_styles(tiles)
        evolution_data[year] = aggregate_style_statistics(styles)

    trends = {
        'style_progression': track_style_changes(evolution_data),
        'densification_patterns': analyze_density_changes(evolution_data),
        'preservation_status': assess_heritage_preservation(evolution_data)
    }

    return trends
```

### Planning Support

```python
def architectural_planning_analysis(planning_zone):
    """Support urban planning with architectural analysis."""

    current_styles = analyze_existing_architecture(planning_zone)

    recommendations = {
        'compatible_styles': suggest_compatible_styles(current_styles),
        'height_guidelines': recommend_height_limits(current_styles),
        'density_limits': calculate_appropriate_density(current_styles),
        'preservation_priorities': identify_preservation_candidates(current_styles)
    }

    return recommendations
```

## Integration with Other Systems

### QGIS Integration

```python
def export_to_qgis(style_classifications):
    """Export style data to QGIS."""

    # Create styled layer
    layer = create_vector_layer(style_classifications)

    # Apply style-based symbology
    apply_architectural_styling(layer)

    # Add attribute data
    add_style_attributes(layer, style_classifications)

    return layer
```

### Database Integration

```python
def store_architectural_data(classifications, database_connection):
    """Store classifications in heritage database."""

    for building_id, classification in classifications.items():
        record = {
            'building_id': building_id,
            'architectural_style': classification['style'],
            'confidence': classification['confidence'],
            'period': classification.get('period'),
            'regional_variant': classification.get('variant'),
            'conservation_status': assess_conservation_status(classification)
        }

        database_connection.insert_architectural_record(record)
```

## Best Practices

### Classification Guidelines

1. **Multi-scale Analysis**: Consider building, block, and neighborhood scales
2. **Temporal Context**: Account for construction periods and modifications
3. **Regional Adaptation**: Adjust criteria for regional variations
4. **Expert Validation**: Incorporate architectural expert knowledge
5. **Continuous Learning**: Update models with new training data

### Quality Assurance

1. **Cross-validation**: Test on independent datasets
2. **Confidence Thresholds**: Set appropriate confidence levels
3. **Manual Review**: Review uncertain classifications
4. **Documentation**: Maintain detailed classification rationale
5. **Version Control**: Track model and criteria versions

## Related Documentation

- [Regional Processing](../guides/regional-processing)
- [Historical Analysis](./historical-analysis)
- [Building Features](../api/features)
- [Configuration Guide](../api/configuration)
