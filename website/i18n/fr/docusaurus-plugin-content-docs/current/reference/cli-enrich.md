---
sidebar_position: 1
title: Commande CLI Enrich
description: Interface ligne de commande pour enrichir données LiDAR avec caractéristiques bâtiment
keywords: [cli, ligne-commande, enrich, caractéristiques, traitement]
---

# Référence Commande CLI Enrich

La commande `ign-lidar enrich` ajoute des caractéristiques de composants de bâtiment aux nuages de points LiDAR, les enrichissant avec des informations géométriques et architecturales.

## Syntaxe

```bash
ign-lidar enrich [OPTIONS] CHEMIN_ENTREE CHEMIN_SORTIE
```

## Utilisation de Base

### Enrichir un Fichier Unique

```bash
ign-lidar enrich input.las enriched_output.las
```

### Enrichir avec Données RGB

```bash
ign-lidar enrich --rgb-path orthophoto.tif input.las output.las
```

### Traitement par Lots

```bash
ign-lidar enrich --batch tiles/*.las --output-dir enriched/
```

## Options de Commande

### Options Entrée/Sortie

#### `CHEMIN_ENTREE` (requis)

Chemin vers fichier LAS/LAZ d'entrée ou répertoire pour traitement par lots.

```bash
# Fichier unique
ign-lidar enrich tile_001.las output.las

# Répertoire (mode batch)
ign-lidar enrich tiles/ --output-dir enriched/
```

#### `CHEMIN_SORTIE` (requis pour fichiers uniques)

Chemin sortie pour nuage points enrichi.

#### `--output-dir, -o` (mode batch)

Répertoire sortie pour traitement par lots.

```bash
ign-lidar enrich tiles/ --output-dir /chemin/vers/enriched/
```

#### `--output-format`

Format fichier de sortie.

**Options :** `las`, `laz`, `ply`, `h5`
**Défaut :** `las`

```bash
ign-lidar enrich input.las output.h5 --output-format h5
```

### Options de Traitement

#### `--features, -f`

Spécifier quelles caractéristiques extraire.

**Options :**

- `geometric` : Caractéristiques géométriques de base (planarité, linéarité, etc.)
- `architectural` : Caractéristiques spécifiques bâtiment (murs, toits, etc.)
- `all` : Toutes les caractéristiques disponibles
- `custom` : Ensemble caractéristiques défini utilisateur

```bash
# Extraire seulement caractéristiques géométriques
ign-lidar enrich --features geometric input.las output.las

# Extraire toutes caractéristiques
ign-lidar enrich --features all input.las output.las

# Ensemble caractéristiques personnalisé
ign-lidar enrich --features planarity,height,normal_z input.las output.las
```

#### `--neighborhood-size, -n`

Rayon voisinage pour calcul caractéristiques (mètres).

**Défaut :** `1.0`

```bash
ign-lidar enrich --neighborhood-size 2.0 input.las output.las
```

#### `--min-building-points`

Points minimum requis pour classifier comme bâtiment.

**Défaut :** `50`

```bash
ign-lidar enrich --min-building-points 100 input.las output.las
```

### Options Intégration RGB

#### `--rgb-path, -r`

Chemin vers orthophoto pour augmentation couleur.

```bash
ign-lidar enrich --rgb-path orthophoto.tif input.las output.las
```

#### `--rgb-interpolation`

Méthode interpolation pour attribution RGB.

**Options :** `nearest`, `bilinear`, `bicubic`
**Défaut :** `bilinear`

```bash
ign-lidar enrich --rgb-path ortho.tif --rgb-interpolation bicubic input.las output.las
```

#### `--rgb-bands`

Spécifier quelles bandes extraire de l'orthophoto.

**Défaut :** `1,2,3` (RGB)

```bash
# Inclure bande infrarouge
ign-lidar enrich --rgb-path ortho.tif --rgb-bands 1,2,3,4 input.las output.las
```

### Options Performance

#### `--gpu, -g`

Activer accélération GPU.

```bash
ign-lidar enrich --gpu input.las output.las
```

#### `--gpu-memory-fraction`

Fraction mémoire GPU à utiliser.

**Défaut :** `0.7`

```bash
ign-lidar enrich --gpu --gpu-memory-fraction 0.9 input.las output.las
```

#### `--batch-size, -b`

Taille batch traitement pour gestion mémoire.

**Défaut :** `100000`

```bash
ign-lidar enrich --batch-size 50000 input.las output.las
```

#### `--num-workers, -w`

Nombre workers parallèles pour traitement.

**Défaut :** Nombre cœurs CPU

```bash
ign-lidar enrich --num-workers 8 input.las output.las
```

### Options Contrôle Qualité

#### `--validate`

Effectuer vérifications validation sur sortie.

```bash
ign-lidar enrich --validate input.las output.las
```

#### `--quality-report`

Générer rapport évaluation qualité.

```bash
ign-lidar enrich --quality-report report.json input.las output.las
```

#### `--preserve-classification`

Conserver classifications points originales.

```bash
ign-lidar enrich --preserve-classification input.las output.las
```

### Options Analyse Architecturale

#### `--architectural-style`

Spécifier style architectural pour analyse renforcée.

**Options :** `haussmanian`, `traditional`, `contemporary`, `industrial`

```bash
ign-lidar enrich --architectural-style haussmanian input.las output.las
```

#### `--region`

Région géographique pour adaptation style.

**Options :** `ile_de_france`, `provence`, `brittany`, `alsace`

```bash
ign-lidar enrich --region ile_de_france input.las output.las
```

#### `--building-type`

Types bâtiment attendus dans jeu données.

**Options :** `residential`, `commercial`, `industrial`, `mixed`

```bash
ign-lidar enrich --building-type residential input.las output.las
```

## Fichier de Configuration

Utiliser fichier configuration YAML pour scénarios traitement complexes.

### Exemple Configuration

```yaml
# enrich_config.yaml
processing:
  features: ["geometric", "architectural"]
  neighborhood_size: 1.5
  min_building_points: 75

rgb:
  enabled: true
  interpolation: "bilinear"
  bands: [1, 2, 3]

performance:
  use_gpu: true
  gpu_memory_fraction: 0.8
  batch_size: 75000
  num_workers: 6

architectural:
  style: "haussmanian"
  region: "ile_de_france"
  building_type: "residential"

quality:
  validate: true
  generate_report: true
```

### Utilisation Fichier Configuration

```bash
ign-lidar enrich --config enrich_config.yaml input.las output.las
```

## Exemples Avancés

### Analyse Bâtiment Haute Qualité

```bash
ign-lidar enrich \
  --features all \
  --rgb-path orthophoto.tif \
  --architectural-style haussmanian \
  --region ile_de_france \
  --neighborhood-size 1.5 \
  --gpu \
  --validate \
  --quality-report quality.json \
  input.las output.las
```

### Traitement par Lots avec GPU

```bash
ign-lidar enrich \
  --batch tiles/*.las \
  --output-dir enriched/ \
  --features geometric,architectural \
  --gpu \
  --batch-size 200000 \
  --num-workers 4
```

### Traitement Optimisé Mémoire

```bash
ign-lidar enrich \
  --batch-size 25000 \
  --gpu-memory-fraction 0.5 \
  --num-workers 2 \
  input_large.las output.las
```

## Informations Sortie

### Champs Points Ajoutés

La commande enrich ajoute les champs suivants aux nuages de points :

| Nom Champ                          | Type  | Description                             |
| ---------------------------------- | ----- | --------------------------------------- |
| `planarity`                        | float | Planarité surface (0-1)                 |
| `linearity`                        | float | Force structure linéaire (0-1)          |
| `sphericity`                       | float | Compacité 3D (0-1)                      |
| `height_above_ground`              | float | Hauteur normalisée (mètres)             |
| `building_component`               | uint8 | Classe composant (0=sol, 1=mur, 2=toit) |
| `architectural_style`              | uint8 | Style architectural détecté             |
| `normal_x`, `normal_y`, `normal_z` | float | Vecteurs normaux surface                |

### Champs RGB (quand activé)

| Nom Champ              | Type   | Description                       |
| ---------------------- | ------ | --------------------------------- |
| `red`, `green`, `blue` | uint16 | Valeurs couleur RGB               |
| `infrared`             | uint16 | Proche infrarouge (si disponible) |
| `material_class`       | uint8  | Classification matériau           |

## Gestion d'Erreurs

### Messages Erreur Courants

#### "Mémoire GPU insuffisante"

**Solution :** Réduire taille batch ou fraction mémoire GPU

```bash
ign-lidar enrich --gpu-memory-fraction 0.5 --batch-size 50000 input.las output.las
```

#### "Fichier RGB introuvable"

**Solution :** Vérifier chemin orthophoto et permissions fichier

```bash
ls -la /chemin/vers/orthophoto.tif
```

#### "Points insuffisants pour extraction caractéristiques"

**Solution :** Réduire seuil minimum points bâtiment

```bash
ign-lidar enrich --min-building-points 25 input.las output.las
```

### Options Débogage

#### `--verbose, -v`

Activer sortie log détaillée.

```bash
ign-lidar enrich --verbose input.las output.las
```

#### `--debug`

Activer mode debug avec log étendu.

```bash
ign-lidar enrich --debug input.las output.las
```

#### `--log-file`

Sauvegarder logs dans fichier.

```bash
ign-lidar enrich --log-file enrich.log input.las output.las
```

## Références Performance

### Temps Traitement (approximatifs)

| Points | Caractéristiques | GPU  | CPU   | Accélération |
| ------ | ---------------- | ---- | ----- | ------------ |
| 1M     | Géométriques     | 30s  | 4min  | 8x           |
| 1M     | Toutes + RGB     | 45s  | 8min  | 11x          |
| 10M    | Géométriques     | 3min | 35min | 12x          |
| 10M    | Toutes + RGB     | 5min | 75min | 15x          |

### Utilisation Mémoire

- **Mode CPU** : ~6GB RAM par 10M points
- **Mode GPU** : ~3GB GPU + 2GB RAM par 10M points
- **Traitement Batch** : Empreinte mémoire configurable

## Commandes Associées

- [`ign-lidar download`](./cli-download.md) - Télécharger tuiles IGN LiDAR
- [`ign-lidar patch`](./cli-patch.md) - Extraire patches entraînement ML
- [`ign-lidar qgis`](./cli-qgis.md) - Outils intégration QGIS

## Voir Aussi

- [Guide Traitement](../guides/preprocessing.md)
- [API Caractéristiques](../api/features.md)
- [Optimisation Performance](../guides/performance.md)
- [Dépannage](../guides/troubleshooting.md)
