---
sidebar_position: 2
title: Commandes CLI
description: Référence complète pour les commandes de l'interface en ligne de commande
keywords: [cli, commandes, référence, terminal]
---

# Référence des Commandes CLI

Référence complète pour toutes les commandes de l'interface en ligne de commande de la Bibliothèque de Traitement LiDAR HD IGN.

## Structure des Commandes

Toutes les commandes suivent cette structure :

```bash
python -m ign_lidar.cli COMMANDE [options]

# Ou en utilisant la commande installée (si dans PATH)
ign-lidar-process COMMANDE [options]
```

## Commandes Disponibles

- [`download`](#download) - Télécharger les tuiles LiDAR depuis les serveurs IGN
- [`enrich`](#enrich) - Ajouter des caractéristiques de bâtiment aux fichiers LAZ
- [`process`](#process) - Extraire des patches depuis les tuiles enrichies

## download

Télécharger les tuiles LiDAR pour une zone spécifiée.

### Syntaxe

```bash
python -m ign_lidar.cli download \
  --bbox MIN_LON,MIN_LAT,MAX_LON,MAX_LAT \
  --output REPERTOIRE_SORTIE \
  [--max-tiles MAX_TUILES] \
  [--force]
```

### Paramètres

| Paramètre     | Type                    | Requis | Description                                          |
| ------------- | ----------------------- | ------ | ---------------------------------------------------- |
| `--bbox`      | float,float,float,float | Oui    | Zone délimitée comme min_lon,min_lat,max_lon,max_lat |
| `--output`    | chaîne                  | Oui    | Répertoire de sortie pour les tuiles téléchargées    |
| `--max-tiles` | entier                  | Non    | Nombre maximum de tuiles à télécharger               |
| `--force`     | drapeau                 | Non    | Forcer le re-téléchargement des tuiles existantes    |

### Exemples

```bash
# Télécharger des tuiles pour Paris centre
python -m ign_lidar.cli download \
  --bbox 2.3,48.85,2.35,48.87 \
  --output ./data/raw/ \
  --max-tiles 10

# Forcer le re-téléchargement
python -m ign_lidar.cli download \
  --bbox 2.3,48.85,2.35,48.87 \
  --output ./data/raw/ \
  --force
```

## enrich

Ajouter des caractéristiques géométriques avancées aux fichiers LAZ.

### Syntaxe

```bash
python -m ign_lidar.cli enrich \
  --input FICHIER_LAZ \
  --output FICHIER_ENRICHI.laz \
  [--workers NOMBRE_WORKERS] \
  [--chunk-size TAILLE_CHUNK]
```

### Paramètres

| Paramètre      | Type   | Requis | Description                          |
| -------------- | ------ | ------ | ------------------------------------ |
| `--input`      | chaîne | Oui    | Fichier LAZ d'entrée à enrichir      |
| `--output`     | chaîne | Oui    | Fichier LAZ de sortie enrichi        |
| `--workers`    | entier | Non    | Nombre de processus parallèles       |
| `--chunk-size` | entier | Non    | Taille des chunks pour le traitement |

## process

Extraire des patches d'entraînement depuis les tuiles enrichies.

### Syntaxe

```bash
python -m ign_lidar.cli process \
  --input REPERTOIRE_TUILES \
  --output REPERTOIRE_PATCHES \
  [--patch-size TAILLE] \
  [--overlap CHEVAUCHEMENT] \
  [--min-points MIN_POINTS]
```

### Paramètres

| Paramètre      | Type   | Requis | Description                               |
| -------------- | ------ | ------ | ----------------------------------------- |
| `--input`      | chaîne | Oui    | Répertoire contenant les tuiles enrichies |
| `--output`     | chaîne | Oui    | Répertoire de sortie pour les patches     |
| `--patch-size` | entier | Non    | Taille des patches en mètres (défaut: 50) |
| `--overlap`    | float  | Non    | Chevauchement entre patches (défaut: 0.1) |
| `--min-points` | entier | Non    | Nombre minimum de points par patch        |

## Options Globales

### Gestion de la Mémoire

```bash
--memory-limit 4GB     # Limiter l'utilisation mémoire
--low-memory          # Mode mémoire réduite
```

### Journalisation

```bash
--verbose             # Journalisation détaillée
--quiet              # Mode silencieux
--log-file FILE      # Écrire les logs dans un fichier
```

### Parallélisation

```bash
--workers N          # Nombre de processus parallèles
--gpu               # Utiliser l'accélération GPU (si disponible)
```

## Exemples de Flux de Travail Complets

### Traitement Basique

```bash
# 1. Télécharger les données
python -m ign_lidar.cli download \
  --bbox 2.3,48.85,2.35,48.87 \
  --output ./data/raw/

# 2. Enrichir les tuiles
python -m ign_lidar.cli enrich \
  --input ./data/raw/*.laz \
  --output ./data/enriched/ \
  --workers 4

# 3. Extraire les patches
python -m ign_lidar.cli process \
  --input ./data/enriched/ \
  --output ./data/patches/ \
  --patch-size 50
```

### Traitement Avancé avec Optimisations

```bash
# Traitement optimisé avec contrôle mémoire
python -m ign_lidar.cli process \
  --input ./data/enriched/ \
  --output ./data/patches/ \
  --patch-size 100 \
  --overlap 0.2 \
  --workers 8 \
  --memory-limit 8GB \
  --gpu \
  --verbose
```

## Codes de Sortie

| Code | Description                    |
| ---- | ------------------------------ |
| 0    | Succès                         |
| 1    | Erreur générale                |
| 2    | Arguments invalides            |
| 3    | Fichier non trouvé             |
| 4    | Erreur de mémoire insuffisante |
| 5    | Interruption utilisateur       |

## Dépannage

### Erreurs Courantes

**"Module not found"**

```bash
pip install ign-lidar-hd
```

**"Memory error during processing"**

```bash
# Réduire la taille des chunks ou utiliser le mode low-memory
python -m ign_lidar.cli process --low-memory --chunk-size 1000
```

**"GPU acceleration not available"**

```bash
# Installer les dépendances GPU
pip install ign-lidar-hd[gpu]
```

## Voir Aussi

- [Guide d'Utilisation de Base](basic-usage.md) - Premiers pas
- [Fonctionnalités Smart Skip](../features/smart-skip.md) - Éviter le retraitement
- [Optimisation Mémoire](../reference/memory-optimization.md) - Gestion des ressources
