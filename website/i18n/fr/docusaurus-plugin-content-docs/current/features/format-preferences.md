---
sidebar_position: 2
title: Préférences de Format
description: Configurer les formats de sortie pour les fichiers LAZ et la compatibilité QGIS
keywords: [format, laz, qgis, configuration, préférences]
---

# Préférences de Format de Sortie

Configurez comment les fichiers LAZ enrichis sont sauvegardés pour équilibrer la complétude des caractéristiques avec la compatibilité logicielle.

## Vue d'Ensemble

La bibliothèque supporte deux stratégies de format de sortie :

1. **LAZ Augmenté (LAZ 1.4)** - Préservation complète des caractéristiques (par défaut)
2. **LAZ Compatible QGIS (LAZ 1.2)** - Compatibilité maximale (optionnel)

## Comportement par Défaut

Par défaut, la bibliothèque préserve toutes les caractéristiques géométriques en utilisant le format LAZ 1.4 :

```bash
# Par défaut : Crée un LAZ augmenté avec toutes les caractéristiques
ign-lidar-hd enrich \
  --input-dir tuiles_brutes/ \
  --output tuiles_enrichies/ \
  --mode building
```

**Sortie** : `tuiles_enrichies/tuile.laz` (LAZ 1.4 avec plus de 30 caractéristiques)

## Options de Configuration

### Préférer LAZ Augmenté (Par Défaut)

**Paramètre** : `PREFER_AUGMENTED_LAZ = True`

- **Format** : LAZ 1.4
- **Caractéristiques** : Tous les 30+ attributs géométriques préservés
- **Compatibilité** : Logiciels LiDAR modernes (CloudCompare, FME, etc.)
- **Taille de fichier** : Légèrement plus grande en raison des attributs étendus

**Avantages** :

- Ensemble complet de caractéristiques disponible
- Aucune perte de données pendant le traitement
- Format résistant au temps

### Mode Compatibilité QGIS

**Paramètre** : `PREFER_AUGMENTED_LAZ = False`

- **Format** : LAZ 1.2
- **Caractéristiques** : Sous-ensemble des attributs les plus importants
- **Compatibilité** : QGIS 3.10+, logiciels LiDAR plus anciens
- **Taille de fichier** : Plus compacte

**Configuration** :

```python
from ign_lidar.config import Config

# Activer le mode compatibilité QGIS
config = Config()
config.PREFER_AUGMENTED_LAZ = False
```

## Attributs Préservés par Mode

### Mode LAZ Augmenté (LAZ 1.4)

**Tous les attributs** sont préservés :

| Catégorie              | Attributs                                  | Count |
| ---------------------- | ------------------------------------------ | ----- |
| **Classification**     | building_component, lod2_class, lod3_class | 3     |
| **Géométrie Surface**  | planarity, sphericity, linearity           | 3     |
| **Géométrie Locale**   | curvature, verticality, normalX/Y/Z        | 4     |
| **Contexte Spatial**   | density, roughness, height_above_ground    | 3     |
| **Détection Contours** | edge_strength, corner_likelihood           | 2     |
| **Qualité**            | feature_confidence, point_quality          | 2     |
| **Métadonnées**        | processing_timestamp, version_info         | 2     |

**Total** : 30+ attributs géométriques complets

### Mode Compatibilité QGIS (LAZ 1.2)

**Attributs essentiels** seulement :

| Attribut              | Type    | Description                       |
| --------------------- | ------- | --------------------------------- |
| `building_component`  | uint8   | Classification principale (0-14)  |
| `planarity`           | float32 | Degré de planarité (0.0-1.0)      |
| `verticality`         | float32 | Degré de verticalité (0.0-1.0)    |
| `edge_strength`       | float32 | Force du contour (0.0-1.0)        |
| `height_above_ground` | float32 | Hauteur au-dessus du sol (mètres) |
| `point_quality`       | uint8   | Score de qualité (0-100)          |

**Total** : 6 attributs essentiels

## Configuration par Environnement

### Variable d'Environnement

```bash
# Forcer la compatibilité QGIS
export IGN_LIDAR_QGIS_MODE=true

# Ou forcer LAZ augmenté
export IGN_LIDAR_QGIS_MODE=false
```

### Configuration dans du Code

```python
# Configuration programmatique
from ign_lidar import configure_output_format

# Mode QGIS
configure_output_format(qgis_compatible=True)

# Mode augmenté
configure_output_format(qgis_compatible=False)
```

### Configuration par Fichier

**Fichier** : `~/.ign_lidar/config.yaml`

```yaml
output:
  prefer_augmented_laz: false # Mode QGIS
  preserve_all_features: false
  target_software: "qgis"
```

## Considérations sur les Performances

### Traitement

| Mode                | Vitesse     | Mémoire     | CPU         |
| ------------------- | ----------- | ----------- | ----------- |
| **LAZ Augmenté**    | Normale     | Plus élevée | Normale     |
| **QGIS Compatible** | Plus rapide | Plus faible | Plus faible |

### Stockage

| Mode                | Taille Fichier  | Compression      |
| ------------------- | --------------- | ---------------- |
| **LAZ Augmenté**    | ~15% plus grand | LAZ 1.4 optimisé |
| **QGIS Compatible** | Plus compact    | LAZ 1.2 standard |

## Recommandations d'Utilisation

### Utilisez LAZ Augmenté Quand

- **Recherche avancée** en géométrie 3D
- **Développement d'algorithmes** ML
- **Analyse complète** de bâtiments
- **Archivage long terme**

### Utilisez Mode QGIS Quand

- **Visualisation dans QGIS** principalement
- **Compatibilité** avec logiciels plus anciens
- **Contraintes de stockage** importantes
- **Traitement rapide** requis

## Migration Entre Formats

### Convertir LAZ Augmenté vers QGIS

```bash
ign-lidar-hd convert \
  --input enriched_augmented.laz \
  --output enriched_qgis.laz \
  --format qgis-compatible
```

### Ré-enrichir pour Format Complet

```bash
# Re-traitement complet pour récupérer tous les attributs
ign-lidar-hd enrich \
  --input raw_tile.laz \
  --output enriched_full.laz \
  --format augmented \
  --force
```

## Dépannage

### Erreurs de Compatibilité

**"Unknown attributes in LAZ file"**

- Utiliser le mode QGIS compatible
- Mettre à jour le logiciel LiDAR

**"QGIS cannot read extended attributes"**

- Vérifier la version QGIS (3.10+ requis)
- Utiliser le mode compatible si nécessaire

**"File size too large"**

- Basculer vers le mode QGIS compatible
- Considérer la compression supplémentaire

### Vérification de Format

```python
from ign_lidar.utils import check_laz_format

# Vérifier le format d'un fichier
info = check_laz_format("enriched.laz")
print(f"Version: {info.version}")
print(f"Attributs: {info.attributes}")
print(f"Compatible QGIS: {info.qgis_compatible}")
```

## Utilisation en Mémoire

### LAZ Augmenté

- **Augmentation mémoire** pendant le traitement
- **Attributs étendus** nécessitent plus de RAM

### QGIS Compatible

- **Mémoire réduite** (moins d'attributs)

## Voir Aussi

- [Guide d'Intégration QGIS](../guides/qgis-integration.md) - Utiliser les fichiers dans QGIS
- [Fonctionnalités Smart Skip](smart-skip.md) - Éviter le retraitement des fichiers
- [Commandes CLI](../guides/cli-commands.md) - Options de ligne de commande
