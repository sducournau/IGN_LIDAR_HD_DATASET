---
sidebar_position: 10
title: Guide de Visualisation
description: Techniques de visualisation et d'analyse visuelle des données LiDAR enrichies
keywords: [visualisation, analyse, rendu, 3D, cartographie]
---

## Guide de Visualisation

Techniques avancées de visualisation et d'analyse visuelle pour exploiter au maximum les données LiDAR enrichies.

## Vue d'ensemble

La visualisation des données LiDAR enrichies permet de :

- **Analyser la qualité** des résultats de traitement
- **Identifier les erreurs** et zones problématiques
- **Communiquer les résultats** aux parties prenantes
- **Valider les classifications** obtenues
- **Créer des rendus** professionnels

## Outils de visualisation

### CloudCompare

Outil de référence pour la visualisation LiDAR professionelle.

```bash
# Installation CloudCompare
sudo apt install cloudcompare

# Ouverture avec classification automatique
cloudcompare -O input_enriched.las -AUTO_SAVE OFF
```

**Configuration recommandée :**

- Rendu par classification
- Palette de couleurs standard
- Ombrage directionnel
- Mode points adaptatif

### QGIS avec plugin LiDAR

```bash
# Installation du plugin LiDAR pour QGIS
ign-lidar-hd qgis install-plugin

# Ouverture dans QGIS avec style automatique
ign-lidar-hd qgis open input_enriched.las \
  --auto-style classification \
  --layer-name "LiDAR Enriched"
```

### Visualisation web interactive

```bash
# Génération d'un visualiseur web
ign-lidar-hd create-web-viewer \
  --input data_enriched.las \
  --output web_viewer/ \
  --features navigation,measurement,classification \
  --max-points 1000000
```

## Types de visualisation

### Visualisation par classification

```python
from ign_lidar.visualization import ClassificationViewer

viewer = ClassificationViewer()

# Configuration des couleurs par classe
color_scheme = {
    'ground': '#8B4513',      # Brun
    'buildings': '#FF6B6B',   # Rouge
    'vegetation': '#4ECDC4',  # Vert
    'water': '#45B7D1',       # Bleu
    'infrastructure': '#96CEB4' # Gris-vert
}

# Rendu avec légende
viewer.render_classification(
    input_path="enriched.las",
    output_path="classification_view.png",
    colors=color_scheme,
    include_legend=True,
    resolution=(1920, 1080)
)
```

### Visualisation par élévation

```python
# Carte d'élévation avec dégradé
elevation_viewer = ElevationViewer(
    colormap='terrain',  # jet, viridis, terrain
    elevation_range='auto',
    hillshade=True
)

elevation_viewer.create_elevation_map(
    "input.las",
    "elevation_map.tif",
    resolution=0.5  # mètres par pixel
)
```

### Visualisation par intensité

```bash
# Rendu par intensité LiDAR
ign-lidar-hd visualize intensity input.las output_intensity.png \
  --colormap grayscale \
  --normalize true \
  --enhance-contrast true
```

### Visualisation RGB

```python
# Affichage avec couleurs RGB (si disponibles)
rgb_viewer = RGBViewer()

if rgb_viewer.has_rgb_data("input.las"):
    rgb_viewer.render_rgb(
        "input.las",
        "rgb_view.png",
        enhance_colors=True,
        gamma_correction=1.2
    )
```

## Visualisation 3D interactive

### Configuration de base

```python
from ign_lidar.visualization3d import Interactive3DViewer

viewer3d = Interactive3DViewer(
    backend='plotly',  # plotly, mayavi, open3d
    max_points=500000,  # Limitation pour performance
    point_size=1.0,
    background='white'
)

# Chargement et affichage
viewer3d.load_data("enriched.las")
viewer3d.apply_classification_colors()
viewer3d.show()
```

### Navigation et interaction

```python
# Configuration des contrôles
viewer3d.set_navigation_mode('orbit')  # orbit, fly, walk

# Ajout d'outils de mesure
viewer3d.add_measurement_tools([
    'distance', 'area', 'volume', 'height'
])

# Coupes transversales
viewer3d.enable_cross_sections()

# Annotation
viewer3d.enable_annotations()
```

### Rendu avancé

```python
# Configuration rendu haute qualité
viewer3d.set_render_quality('high')
viewer3d.enable_shadows(True)
viewer3d.set_lighting('natural')  # natural, studio, bright

# Exportation d'images haute résolution
viewer3d.export_image(
    "render_hq.png",
    resolution=(3840, 2160),  # 4K
    antialias=True,
    transparent_background=False
)
```

## Analyse visuelle comparative

### Avant/Après traitement

```python
from ign_lidar.visualization import ComparisonViewer

comparator = ComparisonViewer()

# Comparaison côte à côte
comparator.side_by_side_comparison(
    before="raw_data.las",
    after="enriched_data.las",
    output="before_after.png",
    sync_viewports=True,
    difference_highlighting=True
)
```

### Évolution temporelle

```python
# Animation d'évolution temporelle
timeline_viewer = TimelineViewer()

timeline_viewer.create_evolution_animation(
    data_series=[
        ("2020", "scan_2020.las"),
        ("2021", "scan_2021.las"),
        ("2022", "scan_2022.las"),
        ("2023", "scan_2023.las")
    ],
    output="evolution.gif",
    duration=10.0,  # secondes
    highlight_changes=True
)
```

### Cartes de différences

```python
# Calcul et visualisation des différences
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
    scale_range=(-2.0, 2.0)  # mètres
)
```

## Profils et coupes

### Profils topographiques

```python
from ign_lidar.profiles import ProfileExtractor

profiler = ProfileExtractor()

# Extraction de profil le long d'une ligne
profile_line = [(x1, y1), (x2, y2)]  # Coordonnées début/fin

profile_data = profiler.extract_profile(
    "input.las",
    line_coordinates=profile_line,
    width=2.0,  # Largeur de bande en mètres
    resolution=0.1  # Résolution du profil
)

# Visualisation du profil
profiler.plot_profile(
    profile_data,
    output="topographic_profile.png",
    include_classification=True,
    vertical_exaggeration=2.0
)
```

### Coupes verticales

```python
# Coupe verticale à travers un bâtiment
cross_section = profiler.vertical_cross_section(
    "building_scan.las",
    cutting_plane="vertical",  # vertical, horizontal, oblique
    plane_equation=(a, b, c, d),  # Équation du plan
    thickness=0.5  # Épaisseur de coupe
)

# Rendu de la coupe
profiler.render_cross_section(
    cross_section,
    "building_cross_section.png",
    show_structure=True,
    color_by_material=True
)
```

## Statistiques visuelles

### Histogrammes de distribution

```python
from ign_lidar.statistics import StatisticalVisualizer

stat_viz = StatisticalVisualizer()

# Histogramme des hauteurs par classe
height_stats = stat_viz.height_distribution_by_class(
    "classified.las",
    classes=['ground', 'buildings', 'vegetation'],
    bin_size=0.5  # mètres
)

stat_viz.plot_distribution(
    height_stats,
    "height_distribution.png",
    title="Distribution des hauteurs par classe"
)
```

### Cartes de densité

```python
# Carte de densité de points
density_map = stat_viz.point_density_map(
    "input.las",
    grid_size=1.0,  # mètres
    output="density_map.png",
    colormap='hot',
    include_contours=True
)
```

### Métriques de qualité

```python
# Visualisation des métriques de qualité
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

## Cartographie thématique

### Cartes de hauteur de canopée

```python
from ign_lidar.forestry import CanopyHeightMapper

canopy_mapper = CanopyHeightMapper()

# Modèle numérique de canopée
canopy_height_model = canopy_mapper.create_chm(
    "forest_scan.las",
    resolution=0.5,
    smoothing=True
)

# Visualisation avec isolignes
canopy_mapper.visualize_chm(
    canopy_height_model,
    "canopy_height_map.png",
    contour_interval=5.0,  # mètres
    color_scheme='forest_green'
)
```

### Cartes d'occupation du sol

```python
# Classification d'occupation du sol
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

## Export et formats

### Formats d'image

```python
# Export en différents formats
exporter = ImageExporter()

# PNG haute qualité (défaut)
exporter.export_png(data, "output.png", dpi=300)

# JPEG optimisé web
exporter.export_jpeg(data, "web_output.jpg", quality=85)

# TIFF géoréférencé
exporter.export_geotiff(
    data, "georeferenced.tif",
    crs="EPSG:2154",  # Lambert 93
    include_worldfile=True
)

# SVG vectoriel
exporter.export_svg(data, "vector_output.svg")
```

### Formats 3D

```bash
# Export en formats 3D
ign-lidar-hd export-3d input.las output.ply --format ply
ign-lidar-hd export-3d input.las output.obj --format obj --include-textures
ign-lidar-hd export-3d input.las output.x3d --format x3d --web-optimized
```

### Formats interactifs

```python
# Génération de visualiseurs interactifs
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

## Workflow de visualisation

### Pipeline automatisé

```python
def create_visualization_pipeline(input_file, output_dir):
    """Pipeline complet de visualisation"""

    # 1. Analyse des données
    analyzer = DataAnalyzer()
    data_info = analyzer.analyze(input_file)

    # 2. Visualisations automatiques
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

    # 3. Rapport de visualisation
    report_generator = VisualizationReport()
    report_generator.create_report(
        input_file,
        output_dir,
        f"{output_dir}/visualization_report.html"
    )
```

### Batch processing

```bash
# Visualisation en lot
ign-lidar-hd batch-visualize \
  --input-directory processed_tiles/ \
  --output-directory visualization_outputs/ \
  --visualization-types classification,elevation,quality \
  --format png \
  --resolution high \
  --parallel-jobs 4
```

## Optimisation des performances

### Gestion mémoire

```python
# Configuration pour gros volumes
large_data_viewer = LargeDataViewer(
    streaming_mode=True,
    memory_limit="8GB",
    cache_strategy="lru",
    level_of_detail=True
)

# Rendu progressif
large_data_viewer.progressive_render(
    "huge_dataset.las",
    "progressive_view.png",
    target_fps=30,
    adaptive_quality=True
)
```

### Optimisation GPU

```python
# Accélération GPU pour rendu
gpu_renderer = GPURenderer(
    gpu_memory_limit="6GB",
    use_cuda=True,
    precision="half"  # half, single, double
)

# Rendu accéléré
gpu_renderer.fast_render(
    "large_dataset.las",
    "gpu_rendered.png",
    quality="high"
)
```

## Meilleures pratiques

### Préparation des données

1. **Filtrage préalable** des points aberrants
2. **Optimisation des formats** (LAZ compression)
3. **Indexation spatiale** pour accès rapide
4. **Métadonnées complètes** pour contexte

### Configuration de rendu

1. **Adaptation à l'audience** (technique vs grand public)
2. **Cohérence des couleurs** entre vues
3. **Échelles appropriées** selon l'usage
4. **Légendes explicites** et complètes

### Validation visuelle

1. **Vérification multi-échelle** (vue d'ensemble et détail)
2. **Comparaison avec références** connues
3. **Contrôle cohérence** entre zones adjacentes
4. **Documentation des anomalies** détectées

Voir aussi : [Guide Performance](./performance.md) | [API Visualisation](../api/visualization.md) | [Intégration QGIS](../reference/cli-qgis.md)
