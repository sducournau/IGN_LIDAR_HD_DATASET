---
sidebar_position: 5
title: Axonometry Analysis
description: 3D geometric representation and axonometric projection methods for LiDAR data
keywords: [axonometry, 3d, geometry, projection, visualization, isometric]
---

Advanced 3D geometric representation and axonometric projection methods for architectural analysis of LiDAR-derived building models.

## Overview

Axonometry analysis provides tools for creating 3D geometric representations of buildings extracted from LiDAR data. These methods are particularly useful for:

- **Architectural Documentation**: Creating technical drawings and plans
- **Heritage Preservation**: Documenting historical buildings in 3D
- **Urban Planning**: Visualizing development projects
- **Engineering Analysis**: Structural assessment and modeling

## Axonometric Projection Types

### Isometric Projection

Equal scaling along all three axes, providing a balanced 3D view.

```python
from ign_lidar.axonometry import IsometricProjector

isometric = IsometricProjector(
    rotation_angles=(30, 30, 0),  # Standard isometric angles
    scale_factor=1.0,
    viewing_direction='northeast'
)

# Generate isometric projection of buildings
buildings_3d = isometric.project_buildings(
    building_points=building_point_cloud,
    include_wireframe=True,
    show_hidden_lines=False
)
```

### Dimetric Projection

Two axes scaled equally, one differently, for emphasizing specific dimensions.

```python
from ign_lidar.axonometry import DimetricProjector

dimetric = DimetricProjector(
    scale_ratios=(1.0, 1.0, 0.5),  # Z-axis compressed
    rotation_angles=(45, 20, 0),
    foreshortening=True
)

# Create dimetric view emphasizing horizontal dimensions
horizontal_view = dimetric.project_buildings(
    building_points=buildings,
    emphasis='horizontal',
    detail_level='high'
)
```

### Trimetric Projection

All three axes scaled differently for maximum flexibility.

```python
from ign_lidar.axonometry import TrimetricProjector

trimetric = TrimetricProjector(
    scale_ratios=(1.0, 0.8, 0.6),
    rotation_angles=(35, 25, 15),
    perspective_correction=False
)

# Generate custom trimetric projection
custom_view = trimetric.project_buildings(
    building_points=buildings,
    custom_angles=True,
    optimize_visibility=True
)
```

## 3D Building Reconstruction

### VolumetricReconstructor

Reconstruct 3D building volumes from LiDAR point clouds.

```python
from ign_lidar.reconstruction import VolumetricReconstructor

reconstructor = VolumetricReconstructor(
    voxel_size=0.5,  # 50cm voxel resolution
    method='alpha_shapes',  # 'convex_hull', 'alpha_shapes', 'poisson'
    smoothing=True,
    hole_filling=True
)

# Reconstruct building volumes
building_volumes = reconstructor.reconstruct_buildings(
    point_cloud=lidar_points,
    building_masks=building_classifications,
    detail_level='architectural'  # 'simple', 'detailed', 'architectural'
)
```

#### Level of Detail (LoD) Generation

```python
# Generate multiple LoD representations
lod_generator = reconstructor.create_lod_generator(
    source_points=building_points,
    target_lods=[1, 2, 3]  # LoD1: Block model, LoD2: Roof shapes, LoD3: Detailed
)

building_lods = {
    'lod1': lod_generator.generate_lod1(),  # Simple block model
    'lod2': lod_generator.generate_lod2(),  # With roof structures
    'lod3': lod_generator.generate_lod3()   # Detailed architectural features
}

# LoD1: Block model with flat roof
lod1_features = {
    'geometry': 'simple_box',
    'roof_type': 'flat',
    'details': 'minimal',
    'accuracy': '±2m'
}

# LoD2: Roof structures and major features
lod2_features = {
    'geometry': 'detailed_outline',
    'roof_type': 'complex',  # Gabled, hipped, etc.
    'details': 'moderate',
    'accuracy': '±0.5m'
}

# LoD3: Architectural details
lod3_features = {
    'geometry': 'full_architectural',
    'roof_type': 'complex_with_details',
    'details': 'high',  # Windows, doors, chimneys
    'accuracy': '±0.1m'
}
```

### RoofAnalyzer

Specialized analysis of roof structures for accurate 3D modeling.

```python
from ign_lidar.roofs import RoofAnalyzer

roof_analyzer = RoofAnalyzer(
    plane_detection_threshold=0.1,  # 10cm plane tolerance
    edge_detection_sensitivity=0.8,
    minimum_roof_area=20.0  # 20 m² minimum roof size
)

# Analyze roof structures
roof_analysis = roof_analyzer.analyze_roof_structures(
    building_points=building_point_cloud,
    ground_elevation=ground_level
)

# Roof classification results
{
    'roof_type': 'gabled',  # 'flat', 'gabled', 'hipped', 'mansard', 'complex'
    'roof_planes': [
        {
            'plane_id': 1,
            'normal_vector': [0.0, 0.7071, 0.7071],
            'area': 150.5,  # m²
            'slope': 45.0,  # degrees
            'aspect': 135.0  # degrees (compass direction)
        }
    ],
    'roof_edges': List[LineString],
    'roof_ridges': List[LineString],
    'gutters': List[LineString],
    'chimneys': List[Point],
    'dormers': List[Polygon]
}
```

## Technical Drawing Generation

### OrthographicProjector

Generate technical orthographic projections (plans, elevations, sections).

```python
from ign_lidar.technical_drawings import OrthographicProjector

ortho_projector = OrthographicProjector(
    drawing_scale='1:100',
    line_weights={
        'outline': 0.7,     # mm
        'details': 0.35,    # mm
        'hidden': 0.25      # mm
    },
    annotation_style='architectural'
)

# Generate building plans and elevations
technical_drawings = ortho_projector.generate_drawings(
    building_model=reconstructed_building,
    drawing_types=['plan', 'north_elevation', 'section_aa']
)

# Plan view generation
plan_view = ortho_projector.generate_plan(
    building_model=building,
    cut_height=1.5,  # Cut plane at 1.5m above ground
    show_dimensions=True,
    include_annotations=True
)

# Elevation views
elevations = {
    'north': ortho_projector.generate_elevation(building, direction='north'),
    'south': ortho_projector.generate_elevation(building, direction='south'),
    'east': ortho_projector.generate_elevation(building, direction='east'),
    'west': ortho_projector.generate_elevation(building, direction='west')
}

# Section views
sections = ortho_projector.generate_sections(
    building=building,
    section_planes=['aa', 'bb'],  # Predefined section lines
    section_depth=0.5  # Show 50cm depth
)
```

### DimensionAnnotator

Add measurements and annotations to technical drawings.

```python
from ign_lidar.annotations import DimensionAnnotator

annotator = DimensionAnnotator(
    unit_system='metric',
    precision=0.01,  # 1cm precision
    dimension_style='architectural',
    text_size=2.5  # mm
)

# Add dimensions to drawings
annotated_plan = annotator.add_dimensions(
    drawing=plan_view,
    dimension_types=[
        'overall_length',
        'overall_width',
        'room_dimensions',
        'wall_thicknesses',
        'opening_sizes'
    ]
)

# Add elevation annotations
annotated_elevation = annotator.add_elevation_dimensions(
    drawing=north_elevation,
    reference_levels=[
        {'name': 'Ground Level', 'elevation': 0.0},
        {'name': 'Floor Level', 'elevation': 0.15},
        {'name': 'Ceiling Level', 'elevation': 2.8},
        {'name': 'Ridge Level', 'elevation': 8.5}
    ]
)
```

## Geometric Analysis

### VolumeCalculator

Calculate building volumes and surface areas from 3D models.

```python
from ign_lidar.geometry import VolumeCalculator

volume_calc = VolumeCalculator(
    method='voxel_counting',  # 'mesh_volume', 'voxel_counting', 'convex_hull'
    accuracy='high'
)

# Calculate building metrics
building_metrics = volume_calc.calculate_building_metrics(
    building_mesh=reconstructed_building,
    ground_plane=ground_reference
)

# Results
{
    'gross_volume': 2450.8,      # m³ - Total building volume
    'net_volume': 2156.2,        # m³ - Usable interior volume
    'roof_volume': 294.6,        # m³ - Roof space volume
    'wall_surface_area': 856.4,  # m² - External wall area
    'roof_surface_area': 425.8,  # m² - Roof surface area
    'floor_area': 312.5,         # m² - Ground floor area
    'building_height': 8.5,      # m - Maximum height
    'footprint_area': 298.2,     # m² - Building footprint
    'compactness_ratio': 0.73,   # Volume/surface ratio
    'aspect_ratios': {           # Length/width ratios
        'plan': 1.8,
        'elevation': 0.4
    }
}
```

### FormAnalyzer

Analyze architectural forms and proportional relationships.

```python
from ign_lidar.analysis import FormAnalyzer

form_analyzer = FormAnalyzer(
    analysis_methods=['golden_ratio', 'modular_proportions', 'symmetry'],
    cultural_context='french_architecture'
)

# Analyze building proportions
form_analysis = form_analyzer.analyze_architectural_form(
    building_geometry=building_mesh,
    style_period='haussmann',  # Expected architectural style
    analysis_depth='comprehensive'
)

# Proportion analysis results
{
    'golden_ratio_compliance': 0.85,  # How well it follows golden ratio
    'modular_proportions': {
        'module_size': 3.2,  # meters
        'modular_grid_fit': 0.92
    },
    'symmetry_analysis': {
        'bilateral_symmetry': True,
        'radial_symmetry': False,
        'symmetry_axis': 'north-south'
    },
    'architectural_orders': {
        'detected_order': 'classical',
        'proportion_conformity': 0.78
    }
}
```

## Visualization and Export

### AxonometricRenderer

Render high-quality axonometric visualizations.

```python
from ign_lidar.visualization import AxonometricRenderer

renderer = AxonometricRenderer(
    render_quality='high',
    lighting_model='architectural',
    material_properties=True,
    shadow_casting=True
)

# Configure rendering parameters
render_config = {
    'projection_type': 'isometric',
    'viewing_angle': (35, 45, 0),
    'line_rendering': {
        'visible_lines': {'weight': 0.5, 'color': 'black'},
        'hidden_lines': {'weight': 0.25, 'color': 'gray', 'style': 'dashed'},
        'construction_lines': {'weight': 0.1, 'color': 'blue', 'style': 'dotted'}
    },
    'material_rendering': {
        'walls': {'color': '#E6E6E6', 'texture': 'stone'},
        'roof': {'color': '#8B4513', 'texture': 'tile'},
        'glass': {'color': '#87CEEB', 'transparency': 0.3}
    }
}

# Render axonometric view
axonometric_image = renderer.render_axonometric(
    building_models=[building_lod2, building_lod3],
    config=render_config,
    output_format='png',
    resolution=(2048, 1536)  # High resolution output
)
```

### ExplodedViewGenerator

Create exploded axonometric views for technical documentation.

```python
from ign_lidar.visualization import ExplodedViewGenerator

exploded_generator = ExplodedViewGenerator(
    explosion_factor=1.5,  # Distance multiplier for separation
    animation_capable=True
)

# Generate exploded view
exploded_view = exploded_generator.create_exploded_view(
    building_components={
        'foundation': foundation_mesh,
        'walls': wall_meshes,
        'floors': floor_meshes,
        'roof_structure': roof_frame_mesh,
        'roof_covering': roof_surface_mesh
    },
    explosion_direction='vertical',  # 'vertical', 'horizontal', 'radial'
    component_labels=True
)

# Create assembly animation
assembly_animation = exploded_generator.create_assembly_animation(
    exploded_view=exploded_view,
    animation_duration=10.0,  # seconds
    frame_rate=30,
    output_format='mp4'
)
```

## Integration with CAD Systems

### CADExporter

Export axonometric models to various CAD formats.

```python
from ign_lidar.export import CADExporter

cad_exporter = CADExporter(
    target_software='autocad',  # 'autocad', 'rhino', 'sketchup', 'revit'
    units='meters',
    coordinate_system='lambert_93'
)

# Export to DWG format
cad_exporter.export_to_dwg(
    building_models=[lod2_model, lod3_model],
    technical_drawings=orthographic_drawings,
    output_file='building_model.dwg',
    layer_organization='by_component'
)

# Export to IFC format for BIM
cad_exporter.export_to_ifc(
    building_model=detailed_building,
    building_metadata={
        'building_name': 'Haussmann Building #142',
        'construction_year': 1870,
        'architectural_style': 'Second Empire',
        'address': '15 Boulevard Saint-Germain, Paris'
    },
    output_file='building_model.ifc'
)
```

### BIMIntegration

Integration with Building Information Modeling systems.

```python
from ign_lidar.bim import BIMIntegration

bim_integration = BIMIntegration(
    bim_platform='revit',  # 'revit', 'archicad', 'vectorworks'
    accuracy_level='survey_grade'
)

# Convert LiDAR data to BIM model
bim_model = bim_integration.lidar_to_bim(
    lidar_points=building_points,
    architectural_style='haussmann',
    detail_level='lod3',
    include_interiors=False
)

# Generate BIM families
building_families = bim_integration.generate_families(
    bim_model=bim_model,
    family_types=['walls', 'windows', 'doors', 'roof_elements'],
    parametric=True  # Create parametric families
)
```

## Advanced Applications

### HeritageDocumentation

Specialized tools for documenting heritage buildings.

```python
from ign_lidar.heritage import HeritageDocumentation

heritage_doc = HeritageDocumentation(
    documentation_standard='icomos',  # International heritage standards
    accuracy_requirements='conservation_grade',
    metadata_schema='dublin_core'
)

# Create comprehensive heritage documentation
heritage_package = heritage_doc.create_documentation_package(
    building_lidar=historic_building_points,
    historical_context={
        'construction_period': '1860-1870',
        'architect': 'Georges-Eugène Haussmann',
        'architectural_significance': 'Outstanding example of Second Empire architecture',
        'protection_status': 'Monument Historique'
    },
    documentation_components=[
        'measured_drawings',
        'photogrammetric_models',
        '3d_axonometric_views',
        'condition_assessment',
        'materials_analysis'
    ]
)
```

### ConstructionAnalysis

Analyze construction techniques and structural systems.

```python
from ign_lidar.construction import ConstructionAnalysis

construction_analyzer = ConstructionAnalysis(
    structural_analysis=True,
    material_detection=True,
    construction_period_estimation=True
)

# Analyze construction characteristics
construction_analysis = construction_analyzer.analyze_construction(
    building_points=lidar_data,
    building_geometry=reconstructed_model,
    regional_context='paris_haussmann'
)

# Results
{
    'construction_system': {
        'primary_structure': 'masonry_load_bearing',
        'wall_thickness': 0.45,  # meters
        'floor_system': 'timber_beam_and_board',
        'roof_structure': 'timber_truss'
    },
    'materials_detected': {
        'walls': 'limestone_ashlar',
        'roof': 'slate_tiles',
        'windows': 'wrought_iron_casements'
    },
    'construction_period': {
        'estimated_period': '1865-1870',
        'confidence': 0.87,
        'style_indicators': [
            'mansard_roof',
            'stone_balconies',
            'wrought_iron_details'
        ]
    }
}
```

## Quality Control

### GeometricValidation

Validate the accuracy of axonometric reconstructions.

```python
from ign_lidar.validation import GeometricValidation

validator = GeometricValidation(
    tolerance_levels={
        'position': 0.05,    # 5cm position tolerance
        'dimension': 0.02,   # 2cm dimension tolerance
        'angle': 1.0         # 1 degree angle tolerance
    }
)

# Validate reconstruction accuracy
validation_report = validator.validate_reconstruction(
    source_points=original_lidar,
    reconstructed_model=axonometric_model,
    reference_measurements=survey_data
)

# Validation results
{
    'overall_accuracy': 0.94,
    'position_accuracy': 0.96,
    'dimensional_accuracy': 0.93,
    'angular_accuracy': 0.91,
    'completeness': 0.89,
    'error_statistics': {
        'mean_error': 0.023,     # meters
        'std_deviation': 0.015,  # meters
        'max_error': 0.087,      # meters
        'rms_error': 0.028       # meters
    },
    'quality_grade': 'A'  # A, B, C, D grading system
}
```

## Best Practices

### Workflow Optimization

```python
# Optimal axonometric analysis workflow
def optimized_axonometric_workflow(lidar_file, output_dir):
    """Optimized workflow for axonometric analysis."""

    # 1. Load and preprocess data
    points = PointCloud.from_file(lidar_file)
    buildings = extract_buildings(points, quality='high')

    # 2. Generate multiple LoD models
    lod_generator = VolumetricReconstructor()
    building_models = {
        'lod1': lod_generator.generate_lod1(buildings),
        'lod2': lod_generator.generate_lod2(buildings),
        'lod3': lod_generator.generate_lod3(buildings)
    }

    # 3. Create axonometric projections
    projector = IsometricProjector()
    axonometric_views = {}

    for lod, model in building_models.items():
        axonometric_views[lod] = projector.project_buildings(
            model,
            viewing_angle='optimal'
        )

    # 4. Generate technical drawings
    ortho_projector = OrthographicProjector()
    technical_drawings = ortho_projector.generate_complete_set(
        building_models['lod3']
    )

    # 5. Export results
    export_results(axonometric_views, technical_drawings, output_dir)

    return {
        'models': building_models,
        'projections': axonometric_views,
        'drawings': technical_drawings
    }
```

### Performance Guidelines

```python
# Performance optimization settings
optimization_config = {
    'mesh_simplification': True,   # Reduce polygon count
    'level_of_detail': 'adaptive', # Adjust detail based on viewing distance
    'culling': {
        'backface': True,          # Remove hidden faces
        'frustum': True,           # Remove objects outside view
        'occlusion': True          # Remove occluded objects
    },
    'caching': {
        'geometry': True,          # Cache geometric calculations
        'textures': True,          # Cache material textures
        'projections': True        # Cache projection matrices
    }
}
```

## Related Documentation

- [LOD Classification](./lod3-classification.md)
- [Architectural Styles](./architectural-styles.md)
- [Building Features](../api/features.md)
- [3D Visualization](../guides/visualization.md)
- [3D Visualization](../guides/visualization.md)
