---
sidebar_position: 2
title: Commande CLI Patch
description: Générer patches d'entraînement depuis données LiDAR traitées
keywords: [cli, patches, apprentissage-automatique, entraînement, dataset]
---

# Référence Commande CLI Patch

La commande `ign-lidar patch` génère des patches d'entraînement pour l'apprentissage automatique à partir de nuages de points LiDAR traités.

## Syntaxe

```bash
ign-lidar patch [OPTIONS] CHEMIN_ENTREE CHEMIN_SORTIE
```

## Utilisation de Base

### Générer Patches Standard

```bash
ign-lidar patch enriched_data.las training_patches.h5
```

### Taille Patch Personnalisée

```bash
ign-lidar patch --patch-size 64 --overlap 0.3 input.las patches.h5
```

## Options de Commande

### Génération Patches

#### `--patch-size`

Taille patches carrés en points.
**Défaut :** `32`

#### `--overlap`

Chevauchement entre patches adjacents (0.0-1.0).
**Défaut :** `0.5`

#### `--min-points`

Points minimum requis par patch.
**Défaut :** `100`

### Options Sortie

#### `--output-format`

Format de sortie pour patches.
**Options :** `h5`, `npz`, `pkl`
**Défaut :** `h5`

#### `--augmentation`

Activer augmentation de données.

### Options Traitement

#### `--batch-size`

Taille batch traitement.
**Défaut :** `10000`

#### `--num-workers`

Nombre workers parallèles.

## Exemples

### Génération Patches de Base

```bash
ign-lidar patch --patch-size 48 --overlap 0.4 input.las patches.h5
```

### Données Entraînement Haute Qualité

```bash
ign-lidar patch \
  --patch-size 64 \
  --overlap 0.3 \
  --min-points 200 \
  --augmentation \
  input.las training_data.h5
```

## Commandes Associées

- [`ign-lidar enrich`](./cli-enrich.md) - Enrichir nuages de points
- [`ign-lidar download`](./cli-download.md) - Télécharger tuiles
