---
sidebar_position: 4
---

# Dépannage QGIS

Problèmes courants et solutions lors de l'utilisation de QGIS avec la bibliothèque IGN LiDAR HD.

## Problèmes Courants

### Problèmes d'Installation

**Problème** : Le plugin QGIS ne se charge pas
**Solution** : Vérifiez le chemin Python et les dépendances :

```bash
# Vérifier l'environnement Python de QGIS
qgis --version
```

**Problème** : Dépendances manquantes
**Solution** : Installez les packages requis dans l'environnement Python de QGIS :

```bash
pip install laspy numpy
```

### Problèmes de Chargement de Données

**Problème** : Les fichiers LAS ne s'affichent pas dans QGIS
**Solution** : Utilisez le plugin Nuage de Points ou convertissez dans un format compatible :

```python
# Convertir LAS dans un format compatible
from ign_lidar import QGISConverter
converter = QGISConverter()
converter.las_to_shapefile("input.las", "output.shp")
```

**Problème** : Les fichiers volumineux causent des problèmes de mémoire
**Solution** : Activez le traitement par morceaux :

```python
config = Config(
    chunk_size=1000000,  # Traiter 1M de points à la fois
    memory_limit=8.0     # Limiter à 8 Go de RAM
)
```

### Problèmes de Performance

**Problème** : Traitement lent dans QGIS
**Solutions** :

- Réduire la densité de points pour la visualisation
- Utiliser l'indexation spatiale
- Activer l'accélération GPU si disponible

### Problèmes de Projection

**Problème** : Désalignement du système de coordonnées
**Solution** : Vérifiez et définissez le bon CRS :

```python
# Définir le système de référence de coordonnées correct
converter.set_crs("EPSG:2154")  # RGF93 / Lambert-93
```

## Obtenir de l'Aide

Pour un support supplémentaire :

- Consultez la [documentation QGIS](https://qgis.org/documentation/)
- Visitez le [dépôt GitHub IGN LiDAR HD](https://github.com/sducournau/IGN_LIDAR_HD_DATASET)
- Signalez les problèmes dans le système de suivi des problèmes du projet
