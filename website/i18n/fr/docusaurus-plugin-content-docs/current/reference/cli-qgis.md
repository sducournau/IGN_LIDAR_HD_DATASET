---
sidebar_position: 4
title: CLI QGIS
description: Interface en ligne de commande pour l'intégration QGIS
keywords: [cli, qgis, intégration, interface]
---

## Interface CLI QGIS

Guide d'utilisation de l'interface en ligne de commande pour l'intégration QGIS avec IGN LiDAR HD.

## Vue d'ensemble

L'interface CLI QGIS permet d'intégrer directement les fonctionnalités de traitement LiDAR dans les workflows QGIS.

## Commande principale

```bash
ign-lidar-hd qgis [OPTIONS] COMMAND [ARGS]...
```

## Sous-commandes

### convert

Convertit les fichiers LiDAR en formats compatibles QGIS.

```bash
ign-lidar-hd qgis convert INPUT OUTPUT [OPTIONS]
```

**Options :**

- `--format` : Format de sortie (las, laz, ply, xyz)
- `--crs` : Système de coordonnées de référence
- `--attributes` : Attributs à inclure dans la conversion

**Exemple :**

```bash
ign-lidar-hd qgis convert input.las output.ply --format ply --crs EPSG:2154
```

### project

Génère un projet QGIS avec les données traitées.

```bash
ign-lidar-hd qgis project INPUT_DIR OUTPUT_PROJECT [OPTIONS]
```

**Options :**

- `--template` : Template de projet à utiliser
- `--styles` : Fichiers de style à appliquer
- `--layers` : Configuration des couches

**Exemple :**

```bash
ign-lidar-hd qgis project data/ project.qgs --template building_analysis
```

### style

Applique des styles prédéfinis aux données LiDAR.

```bash
ign-lidar-hd qgis style INPUT STYLE_NAME [OPTIONS]
```

**Styles disponibles :**

- `elevation` : Coloration par altitude
- `classification` : Coloration par classe
- `intensity` : Coloration par intensité
- `rgb` : Couleurs RGB si disponibles

**Exemple :**

```bash
ign-lidar-hd qgis style processed.las elevation --output styled.qml
```

## Configuration QGIS

### Installation du plugin

```bash
# Installation automatique du plugin
ign-lidar-hd qgis install-plugin

# Vérification de l'installation
ign-lidar-hd qgis check-plugin
```

### Configuration des chemins

```bash
# Configuration des chemins QGIS
ign-lidar-hd qgis configure --qgis-path /usr/bin/qgis
```

## Intégration avec Processing

### Algorithmes disponibles

1. **Enrichissement LiDAR**

   - Classification automatique
   - Détection de bâtiments
   - Extraction de végétation

2. **Augmentation RGB**

   - Intégration orthophoto
   - Correction colorimétrique

3. **Analyse spatiale**
   - Calcul de statistiques
   - Génération de rapports

## Templates de projet

### Template analyse urbaine

```bash
ign-lidar-hd qgis project data/ urban_analysis.qgs \
    --template urban \
    --include-buildings \
    --include-vegetation \
    --add-statistics
```

### Template analyse forestière

```bash
ign-lidar-hd qgis project data/ forest_analysis.qgs \
    --template forest \
    --height-analysis \
    --canopy-metrics \
    --biomass-estimation
```

## Styles prédéfinis

### Style par classification

```bash
ign-lidar-hd qgis style input.las classification \
    --colors config/classification_colors.json \
    --output style.qml
```

### Style par élévation

```bash
ign-lidar-hd qgis style input.las elevation \
    --ramp viridis \
    --min-max auto \
    --output elevation_style.qml
```

## Exemples d'usage

### Workflow complet

```bash
# 1. Enrichissement des données
ign-lidar-hd enrich raw_data/ enriched_data/ \
    --features all \
    --output-format laz

# 2. Conversion pour QGIS
ign-lidar-hd qgis convert enriched_data/ qgis_data/ \
    --format ply \
    --attributes classification,rgb

# 3. Génération du projet
ign-lidar-hd qgis project qgis_data/ analysis_project.qgs \
    --template urban_analysis \
    --add-layers all \
    --auto-style

# 4. Ouverture dans QGIS
qgis analysis_project.qgs
```

Voir aussi : [CLI Enrich](./cli-enrich.md) | [Guide Performance](../guides/performance.md)
