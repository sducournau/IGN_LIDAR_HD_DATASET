---
sidebar_position: 4
---

# Dépannage QGIS

Problèmes courants et solutions lors du travail avec QGIS et la bibliothèque IGN LiDAR HD.

## Problèmes Courants

### Problèmes d'Installation

**Problème** : Le plugin QGIS ne se charge pas  
**Solution** : Vérifiez le chemin Python et les dépendances :

```bash
# Vérifier l'environnement Python de QGIS
qgis --version
```

**Problème** : Dépendances manquantes  
**Solution** : Installer les packages requis dans l'environnement Python de QGIS :

```bash
pip install laspy numpy
```

### Problèmes de Chargement de Données

**Problème** : Les fichiers LAS ne s'affichent pas dans QGIS  
**Solution** : Utilisez le plugin Point Cloud ou convertissez vers un format compatible :

```python
# Convertir LAS vers un format compatible
from ign_lidar import QGISConverter
converter = QGISConverter()
converter.las_to_shapefile("input.las", "output.shp")
```

**Problème** : Les gros fichiers causent des problèmes de mémoire  
**Solution** : Activez le traitement par chunks :

```python
config = Config(
    chunk_size=1000000,  # Traiter 1M points à la fois
    memory_limit=8.0     # Limiter à 8Go de RAM
)
```

### Problèmes de Performance

**Problème** : Traitement lent dans QGIS  
**Solutions** :

- Réduire la densité de points pour la visualisation
- Utiliser l'indexation spatiale
- Activer l'accélération GPU si disponible

### Problèmes de Projection

**Problème** : Mauvais alignement du système de coordonnées  
**Solution** : Vérifiez et définissez le bon CRS :

```python
# Définir le bon système de référence de coordonnées
converter.set_crs("EPSG:2154")  # RGF93 / Lambert-93
```

### Problèmes de Symbologie

**Problème** : Les couleurs ne s'affichent pas correctement  
**Solution** : Vérifiez la configuration de la symbologie :

```python
# Configurer la symbologie pour les attributs RGB
from ign_lidar.qgis_converter import configure_rgb_symbology
configure_rgb_symbology(layer, red_field='Red', green_field='Green', blue_field='Blue')
```

**Problème** : Classification des points non visible  
**Solution** : Appliquez un style de classification catégorisé :

```python
# Appliquer une classification par type de bâtiment
from ign_lidar.qgis_converter import apply_building_classification_style
apply_building_classification_style(layer, classification_field='building_class')
```

### Problèmes de Données

**Problème** : Attributs manquants après conversion  
**Solution** : Vérifiez la configuration des champs exportés :

```python
# Spécifier explicitement les champs à exporter
converter = QGISConverter()
converter.set_export_fields([
    'elevation', 'intensity', 'classification',
    'Red', 'Green', 'Blue',
    'building_class', 'facade_orientation'
])
```

**Problème** : Données corrompues ou incohérentes  
**Solution** : Activez la validation des données :

```python
# Activer la validation lors de l'export
converter.enable_data_validation(
    check_coordinates=True,
    check_attributes=True,
    fix_invalid_geometries=True
)
```

### Problèmes de Workflow

**Problème** : Pipeline de traitement lent  
**Solution** : Optimisez le workflow :

```python
# Workflow optimisé pour QGIS
from ign_lidar.processor import LiDARProcessor

processor = LiDARProcessor(
    use_gpu=True,           # Accélération GPU
    chunk_size=500000,      # Chunks plus petits pour QGIS
    qgis_compatible=True,   # Mode compatible QGIS
    simplify_geometry=True  # Simplifier pour la visualisation
)
```

**Problème** : Exports multiples échouent  
**Solution** : Utilisez le traitement par lot :

```python
# Traitement par lot pour multiple exports
from ign_lidar.batch_processor import QGISBatchProcessor

batch = QGISBatchProcessor()
batch.add_tiles(['tile1.laz', 'tile2.laz', 'tile3.laz'])
batch.set_output_format('shapefile')
batch.process_all()
```

## Solutions Avancées

### Optimisation de la Mémoire

```python
# Configuration mémoire pour gros datasets
config = {
    'memory_limit': 4.0,        # Limite RAM (Go)
    'chunk_size': 250000,       # Taille des chunks
    'use_disk_cache': True,     # Cache disque temporaire
    'cleanup_temp': True        # Nettoyer les fichiers temporaires
}

processor = LiDARProcessor(**config)
```

### Performance GPU pour QGIS

```python
# Configuration GPU optimisée pour QGIS
gpu_config = {
    'use_gpu': True,
    'gpu_memory_fraction': 0.7,  # Utiliser 70% de la VRAM
    'batch_size': 50000,         # Batch GPU plus petit
    'fallback_to_cpu': True      # Fallback si GPU plein
}

processor = LiDARProcessor(gpu_config=gpu_config)
```

### Intégration avec Processing Toolbox

```python
# Script pour Processing Toolbox QGIS
from qgis.core import QgsProcessingAlgorithm
from ign_lidar.qgis_integration import IGNLiDARAlgorithm

class EnrichTilesAlgorithm(QgsProcessingAlgorithm):
    def processAlgorithm(self, parameters, context, feedback):
        # Intégration dans QGIS Processing
        processor = IGNLiDARAlgorithm()
        return processor.enrich_tiles(parameters, feedback)
```

## Diagnostic et Débogage

### Logs Détaillés

```python
# Activer la journalisation détaillée
import logging
logging.basicConfig(level=logging.DEBUG)

# Logs spécifiques à QGIS
qgis_logger = logging.getLogger('ign_lidar.qgis')
qgis_logger.setLevel(logging.DEBUG)
```

### Validation des Données

```python
# Valider les données avant traitement
from ign_lidar.validation import QGISDataValidator

validator = QGISDataValidator()
validation_report = validator.validate_las_file('input.las')

if validation_report.has_errors:
    print("Erreurs détectées :")
    for error in validation_report.errors:
        print(f"- {error}")
```

### Test de Performance

```bash
# Test de performance QGIS
python -m ign_lidar.benchmarks.qgis_performance \
  --input test_tile.laz \
  --output benchmark_results.json \
  --test-formats shapefile,geopackage \
  --measure-memory
```

## Obtenir de l'Aide

Pour un support supplémentaire :

- Consultez la [documentation QGIS](https://qgis.org/documentation/)
- Visitez le [dépôt GitHub IGN LiDAR HD](https://github.com/sducournau/IGN_LIDAR_HD_DATASET)
- Rapportez les problèmes dans le tracker d'issues du projet
- Rejoignez les discussions communautaires sur les [GitHub Discussions](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/discussions)

### Support Communautaire

- **Forum QGIS France** : [Géorezo](https://georezo.net/)
- **Discord IGN LiDAR HD** : Discussions en temps réel
- **Stack Overflow** : Tag `ign-lidar-hd` pour les questions techniques

### Ressources Supplémentaires

- [Guide d'Intégration QGIS](/docs/guides/qgis-integration)
- [Exemples de Scripts QGIS](/docs/examples/qgis-scripts)
- [API QGIS Reference](/docs/api/qgis-api)
- [Tutoriels Vidéo](https://youtube.com/@ignlidarhd)
