---
sidebar_position: 3
title: Commande CLI Download
description: Télécharger tuiles IGN LiDAR HD depuis serveurs officiels
keywords: [cli, téléchargement, tuiles, ign, lidar]
---

# Référence Commande CLI Download

La commande `ign-lidar download` récupère les tuiles LiDAR HD depuis les serveurs officiels IGN.

## Syntaxe

```bash
ign-lidar download [OPTIONS] IDS_TUILES REPERTOIRE_SORTIE
```

## Utilisation de Base

### Télécharger Tuile Unique

```bash
ign-lidar download C_3945-6730_2022 ./tiles/
```

### Télécharger Tuiles Multiples

```bash
ign-lidar download C_3945-6730_2022 C_3945-6735_2022 ./tiles/
```

### Télécharger depuis Liste

```bash
ign-lidar download --from-file tile_list.txt ./tiles/
```

## Options de Commande

### Options Entrée

#### `IDS_TUILES` (requis)

Un ou plusieurs identifiants de tuiles à télécharger.

#### `--from-file, -f`

Lire IDs tuiles depuis fichier texte (un par ligne).

#### `--bbox`

Télécharger toutes tuiles dans emprise.
Format : `xmin,ymin,xmax,ymax`

### Options Sortie

#### `REPERTOIRE_SORTIE` (requis)

Répertoire pour sauvegarder tuiles téléchargées.

#### `--format`

Format de téléchargement.
**Options :** `laz`, `las`
**Défaut :** `laz`

### Options Téléchargement

#### `--overwrite`

Écraser fichiers existants.

#### `--verify`

Vérifier fichiers téléchargés.

#### `--parallel, -p`

Nombre téléchargements parallèles.
**Défaut :** `4`

## Exemples

### Télécharger par Emprise

```bash
ign-lidar download --bbox 3945000,6730000,3950000,6735000 ./tiles/
```

### Téléchargements Parallèles avec Vérification

```bash
ign-lidar download --parallel 8 --verify --from-file tiles.txt ./data/
```

## Commandes Associées

- [`ign-lidar enrich`](./cli-enrich.md) - Enrichir tuiles téléchargées
- [`ign-lidar patch`](./cli-patch.md) - Générer patches entraînement
