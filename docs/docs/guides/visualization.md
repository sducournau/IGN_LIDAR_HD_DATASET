---
sidebar_position: 10
title: Visualization Guide
description: Visualization techniques and visual analysis of enriched LiDAR data
keywords: [visualization, analysis, rendering, 3D, mapping]
---

# Visualization Guide

Advanced visualization techniques and visual analysis to make the most of enriched LiDAR data.

## Overview

Visualization of enriched LiDAR data allows you to:

- **Analyze quality** of processing results
- **Identify errors** and problematic areas
- **Communicate results** to stakeholders
- **Validate classifications** obtained
- **Create professional renderings**

## Visualization Tools

### CloudCompare

Reference tool for professional LiDAR visualization.

```bash
# Install CloudCompare
sudo apt install cloudcompare

# Open with automatic classification
cloudcompare -O input_enriched.las -AUTO_SAVE OFF
```

**Recommended configuration:**

- Classification rendering
- Standard color palette
- Directional shading
- Adaptive point mode

### QGIS with LiDAR Plugin

```bash
# Install LiDAR plugin for QGIS
ign-lidar-hd qgis install-plugin

# Open in QGIS with automatic style
ign-lidar-hd qgis open input_enriched.las \
  --auto-style classification \
  --layer-name "LiDAR Enriched"
```

### Interactive Web Visualization

```bash
# Generate web viewer
ign-lidar-hd create-web-viewer \
  --input data_enriched.las \
  --output web_viewer/ \
  --features navigation,measurement,classification \
  --max-points 1000000
```

## Visualization Types

### Classification Visualization

```python
from ign_lidar.visualization import ClassificationViewer

viewer = ClassificationViewer()

# Configure colors by class
color_scheme = {
    'ground': '#8B4513',      # Brown
    'buildings': '#FF6B6B',   # Red
    'vegetation': '#4ECDC4',  # Green
    'water': '#45B7D1',       # Blue
    'infrastructure': '#96CEB4' # Gray-green
}

# Render with legend
viewer.render_classification(
    input_path="enriched.las",
    output_path="classification_view.png",
    colors=color_scheme,
    include_legend=True,
    resolution=(1920, 1080)
)
```

### Elevation Visualization

```python
# Elevation map with gradient
elevation_viewer = ElevationViewer(
    colormap='terrain',  # jet, viridis, terrain
    elevation_range='auto',
    hillshade=True
)

elevation_viewer.create_elevation_map(
    "input.las",
    "elevation_map.tif",
    resolution=0.5  # meters per pixel
)
```

### Intensity Visualization

```bash
# Render by LiDAR intensity
ign-lidar-hd visualize intensity input.las output_intensity.png \
  --colormap grayscale \
  --normalize true \
  --enhance-contrast true
```

### RGB Visualization

```python
# Display with RGB colors (if available)
rgb_viewer = RGBViewer()

if rgb_viewer.has_rgb_data("input.las"):
    rgb_viewer.render_rgb(
        "input.las",
        "rgb_view.png",
        enhance_colors=True,
        gamma_correction=1.2
    )
```

## Interactive 3D Visualization

### Basic Configuration

```python
from ign_lidar.visualization3d import Interactive3DViewer

viewer3d = Interactive3DViewer(
    backend='plotly',  # plotly, mayavi, open3d
    max_points=500000,  # Limitation for performance
    point_size=1.0,
    background='white'
)

# Load and display
viewer3d.load_data("enriched.las")
viewer3d.apply_classification_colors()
viewer3d.show()
```

### Navigation and Interaction

```python
# Configure controls
viewer3d.set_navigation_mode('orbit')  # orbit, fly, walk

# Add measurement tools
viewer3d.add_measurement_tools([
    'distance', 'area', 'volume', 'height'
])

# Cross-sections
viewer3d.enable_cross_sections()

# Annotations
viewer3d.enable_annotations()
```

### Advanced Rendering

```python
# High quality rendering configuration
viewer3d.set_render_quality('high')
viewer3d.enable_shadows(True)
viewer3d.set_lighting('natural')  # natural, studio, bright

# Export high-resolution images
viewer3d.export_image(
    "render_hq.png",
    resolution=(3840, 2160),  # 4K
    antialias=True,
    transparent_background=False
)
```

## Comparative Visual Analysis

### Before/After Processing

```python
from ign_lidar.visualization import ComparisonViewer

comparator = ComparisonViewer()

# Side-by-side comparison
comparator.side_by_side_comparison(
    before="raw_data.las",
    after="enriched_data.las",
    output="before_after.png",
    sync_viewports=True,
    difference_highlighting=True
)
```

### Temporal Evolution

```python
# Temporal evolution animation
timeline_viewer = TimelineViewer()

timeline_viewer.create_evolution_animation(
    data_series=[
        ("2020", "scan_2020.las"),
        ("2021", "scan_2021.las"),
        ("2022", "scan_2022.las"),
        ("2023", "scan_2023.las")
    ],
    output="evolution.gif",
    duration=10.0,  # seconds
    highlight_changes=True
)
```

### Difference Maps

```python
# Compute and visualize differences
diff_analyzer = DifferenceAnalyzer()

difference_map = diff_analyzer.compute_differences(
    reference="reference.las",
    comparison="current.las",
    method="height_difference"  # height, classification, intensity
)

diff_analyzer.visualize_differences(
    difference_map,
    output="difference_map.png",
    colorbar=True,
    scale_range=(-2.0, 2.0)  # meters
)
```

## Profiles and Cross-Sections

### Topographic Profiles

```python
from ign_lidar.profiles import ProfileExtractor

profiler = ProfileExtractor()

# Extract profile along a line
profile_line = [(x1, y1), (x2, y2)]  # Start/end coordinates

profile_data = profiler.extract_profile(
    "input.las",
    line_coordinates=profile_line,
    width=2.0,  # Band width in meters
    resolution=0.1  # Profile resolution
)

# Visualize profile
profiler.plot_profile(
    profile_data,
    output="topographic_profile.png",
    include_classification=True,
    vertical_exaggeration=2.0
)
```

### Vertical Cross-Sections

```python
# Vertical cross-section through a building
cross_section = profiler.vertical_cross_section(
    "building_scan.las",
    cutting_plane="vertical",  # vertical, horizontal, oblique
    plane_equation=(a, b, c, d),  # Plane equation
    thickness=0.5  # Section thickness
)

# Render cross-section
profiler.render_cross_section(
    cross_section,
    "building_cross_section.png",
    show_structure=True,
    color_by_material=True
)
```

## Visual Statistics

### Distribution Histograms

```python
from ign_lidar.statistics import StatisticalVisualizer

stat_viz = StatisticalVisualizer()

# Height histogram by class
height_stats = stat_viz.height_distribution_by_class(
    "classified.las",
    classes=['ground', 'buildings', 'vegetation'],
    bin_size=0.5  # meters
)

stat_viz.plot_distribution(
    height_stats,
    "height_distribution.png",
    title="Height Distribution by Class"
)
```

### Density Maps

```python
# Point density map
density_map = stat_viz.point_density_map(
    "input.las",
    grid_size=1.0,  # meters
    output="density_map.png",
    colormap='hot',
    include_contours=True
)
```

### Quality Metrics

```python
# Visualize quality metrics
quality_viz = QualityVisualizer()

quality_metrics = quality_viz.compute_quality_metrics(
    processed="enriched.las",
    reference="ground_truth.las"
)

quality_viz.plot_quality_dashboard(
    quality_metrics,
    "quality_dashboard.html",
    interactive=True
)
```

## Thematic Mapping

### Canopy Height Maps

```python
from ign_lidar.forestry import CanopyHeightMapper

canopy_mapper = CanopyHeightMapper()

# Canopy height model
canopy_height_model = canopy_mapper.create_chm(
    "forest_scan.las",
    resolution=0.5,
    smoothing=True
)

# Visualize with contours
canopy_mapper.visualize_chm(
    canopy_height_model,
    "canopy_height_map.png",
    contour_interval=5.0,  # meters
    color_scheme='forest_green'
)
```

### Land Cover Maps

```python
# Land cover classification
land_cover_mapper = LandCoverMapper()

land_cover = land_cover_mapper.classify_land_cover(
    "area_scan.las",
    classes=[
        'urban_dense', 'urban_sparse', 'agricultural',
        'forest_deciduous', 'forest_coniferous', 'water',
        'bare_soil', 'infrastructure'
    ]
)

land_cover_mapper.create_thematic_map(
    land_cover,
    "land_cover_map.png",
    include_legend=True,
    overlay_boundaries=True
)
```

## Export and Formats

### Image Formats

```python
# Export to different formats
exporter = ImageExporter()

# High quality PNG (default)
exporter.export_png(data, "output.png", dpi=300)

# Web-optimized JPEG
exporter.export_jpeg(data, "web_output.jpg", quality=85)

# Georeferenced TIFF
exporter.export_geotiff(
    data, "georeferenced.tif",
    crs="EPSG:2154",  # Lambert 93
    include_worldfile=True
)

# Vector SVG
exporter.export_svg(data, "vector_output.svg")
```

### 3D Formats

```bash
# Export to 3D formats
ign-lidar-hd export-3d input.las output.ply --format ply
ign-lidar-hd export-3d input.las output.obj --format obj --include-textures
ign-lidar-hd export-3d input.las output.x3d --format x3d --web-optimized
```

### Interactive Formats

```python
# Generate interactive viewers
interactive_exporter = InteractiveExporter()

# Plotly HTML
interactive_exporter.create_plotly_viewer(
    "input.las",
    "interactive_plotly.html",
    max_points=100000
)

# Three.js web viewer
interactive_exporter.create_threejs_viewer(
    "input.las",
    "web_viewer/",
    include_controls=True,
    mobile_optimized=True
)
```

## Visualization Workflow

### Automated Pipeline

```python
def create_visualization_pipeline(input_file, output_dir):
    """Complete visualization pipeline"""

    # 1. Data analysis
    analyzer = DataAnalyzer()
    data_info = analyzer.analyze(input_file)

    # 2. Automatic visualizations
    visualizations = [
        ('classification', ClassificationViewer()),
        ('elevation', ElevationViewer()),
        ('intensity', IntensityViewer()),
        ('quality', QualityViewer())
    ]

    for viz_type, viewer in visualizations:
        if viewer.is_applicable(data_info):
            output_path = f"{output_dir}/{viz_type}_view.png"
            viewer.render(input_file, output_path)

    # 3. Visualization report
    report_generator = VisualizationReport()
    report_generator.create_report(
        input_file,
        output_dir,
        f"{output_dir}/visualization_report.html"
    )
```

### Batch Processing

```bash
# Batch visualization
ign-lidar-hd batch-visualize \
  --input-directory processed_tiles/ \
  --output-directory visualization_outputs/ \
  --visualization-types classification,elevation,quality \
  --format png \
  --resolution high \
  --parallel-jobs 4
```

## Performance Optimization

### Memory Management

```python
# Configuration for large volumes
large_data_viewer = LargeDataViewer(
    streaming_mode=True,
    memory_limit="8GB",
    cache_strategy="lru",
    level_of_detail=True
)

# Progressive rendering
large_data_viewer.progressive_render(
    "huge_dataset.las",
    "progressive_view.png",
    target_fps=30,
    adaptive_quality=True
)
```

### GPU Optimization

```python
# GPU acceleration for rendering
gpu_renderer = GPURenderer(
    gpu_memory_limit="6GB",
    use_cuda=True,
    precision="half"  # half, single, double
)

# Accelerated rendering
gpu_renderer.fast_render(
    "large_dataset.las",
    "gpu_rendered.png",
    quality="high"
)
```

## Best Practices

### Data Preparation

1. **Pre-filtering** of outliers
2. **Format optimization** (LAZ compression)
3. **Spatial indexing** for fast access
4. **Complete metadata** for context

### Rendering Configuration

1. **Audience adaptation** (technical vs general public)
2. **Color consistency** between views
3. **Appropriate scales** according to use
4. **Explicit legends** and comprehensive

### Visual Validation

1. **Multi-scale verification** (overview and detail)
2. **Comparison with known references**
3. **Consistency checking** between adjacent areas
4. **Documentation of detected anomalies**

See also: [Performance Guide](./performance.md) | [Visualization API](../api/visualization.md) | [QGIS Integration](../reference/cli-qgis.md)
