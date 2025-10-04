---
sidebar_position: 2
title: API d'Augmentation RGB
description: API pour intégrer les données couleur des orthophotos avec les nuages de points LiDAR
keywords: [api, rgb, couleur, orthophoto, augmentation]
---

## Référence API d'Augmentation RGB

L'API d'Augmentation RGB fournit des outils pour intégrer les données orthophoto IGN avec les nuages de points LiDAR pour créer des jeux de données enrichis en couleur.

## Classes principales

### RGBProcessor

Classe principale pour les opérations d'augmentation RGB.

```python
from ign_lidar import RGBProcessor

processor = RGBProcessor(
    interpolation_method='bilinear',
    quality_threshold=0.8,
    enable_caching=True
)
```

## Méthodes

### `augment_point_cloud(points, orthophoto_path)`

Ajoute des valeurs RGB au nuage de points à partir de l'orthophoto.

**Paramètres:**

- `points` (numpy.ndarray): Coordonnées des points (N×3)
- `orthophoto_path` (str): Chemin vers le fichier orthophoto

**Retourne:**

- `numpy.ndarray`: Valeurs RGB (N×3) en uint8

### `batch_augmentation(tile_list, ortho_dir)`

Traite plusieurs tuiles avec augmentation RGB.

**Paramètres:**

- `tile_list` (list): Liste des chemins de tuiles
- `ortho_dir` (str): Répertoire contenant les orthophotos

**Retourne:**

- `dict`: Résultats du traitement par lot

### `validate_orthophoto(orthophoto_path)`

Valide la qualité et la compatibilité d'une orthophoto.

**Paramètres:**

- `orthophoto_path` (str): Chemin vers l'orthophoto

**Retourne:**

- `bool`: True si l'orthophoto est valide

## Configuration

### Paramètres d'interpolation

```python
interpolation_options = {
    'nearest': 'Plus proche voisin',
    'bilinear': 'Interpolation bilinéaire',
    'bicubic': 'Interpolation bicubique'
}
```

### Seuils de qualité

- `quality_threshold`: Seuil minimum pour l'inclusion des pixels (0.0-1.0)
- `no_data_value`: Valeur pour les pixels sans données
- `alpha_threshold`: Seuil de transparence

## Gestion des erreurs

### Exceptions spécifiques

- `OrthophotoError`: Erreurs liées aux orthophotos
- `InterpolationError`: Erreurs d'interpolation
- `ColorMappingError`: Erreurs de mappage des couleurs

## Exemple d'utilisation

```python
from ign_lidar import RGBProcessor
import laspy

# Initialisation
rgb_processor = RGBProcessor(
    interpolation_method='bilinear',
    quality_threshold=0.9
)

# Lecture du fichier LAS
las_file = laspy.read("input.las")
points = np.vstack([las_file.x, las_file.y, las_file.z]).T

# Augmentation RGB
try:
    colors = rgb_processor.augment_point_cloud(
        points,
        "orthophoto.tif"
    )

    # Sauvegarde avec couleurs
    las_file.red = colors[:, 0] * 256
    las_file.green = colors[:, 1] * 256
    las_file.blue = colors[:, 2] * 256
    las_file.write("output_rgb.las")

except OrthophotoError as e:
    print(f"Erreur orthophoto: {e}")
except Exception as e:
    print(f"Erreur générale: {e}")
```

## Performance

### Optimisations disponibles

- **Cache orthophoto**: Mise en cache des tuiles fréquemment utilisées
- **Traitement GPU**: Accélération CUDA pour l'interpolation
- **Traitement par chunks**: Division automatique pour les gros volumes

### Recommandations

- Utiliser des orthophotos en format GeoTIFF avec pyramides
- Configurer le cache selon la RAM disponible
- Préférer l'interpolation bilinéaire pour un bon compromis vitesse/qualité

## Formats supportés

### Orthophotos

- GeoTIFF (.tif, .tiff)
- JPEG2000 (.jp2)
- ECW (.ecw)

### Nuages de points

- LAS (.las)
- LAZ (.laz)
- PLY (.ply)

Voir aussi: [Guide d'accélération GPU](../guides/gpu-acceleration.md)
