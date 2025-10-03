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
ign-lidar-hd COMMANDE [options]

# Ou en utilisant la commande installée (si dans PATH)
ign-lidar-hd COMMANDE [options]
```

## Commandes Disponibles

- [`download`](#download) - Télécharger les tuiles LiDAR depuis les serveurs IGN
- [`enrich`](#enrich) - Ajouter des caractéristiques de bâtiment aux fichiers LAZ
- [`patch`](#patch) - Extraire des patches depuis les tuiles enrichies (renommée depuis `process`)
- [`process`](#process-obsolète) - ⚠️ Alias obsolète pour `patch`

## download

Télécharger les tuiles LiDAR pour une zone spécifiée.

### Syntaxe

```bash
ign-lidar-hd download \
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
ign-lidar-hd download \
  --bbox 2.3,48.85,2.35,48.87 \
  --output ./data/raw/ \
  --max-tiles 10

# Forcer le re-téléchargement
ign-lidar-hd download \
  --bbox 2.3,48.85,2.35,48.87 \
  --output ./data/raw/ \
  --force
```

## enrich

Ajouter des caractéristiques géométriques avancées aux fichiers LAZ.

### Syntaxe

```bash
ign-lidar-hd enrich \
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

## patch

Extraire des patches d'entraînement depuis les tuiles enrichies avec augmentation RGB optionnelle.

### Syntaxe

```bash
ign-lidar-hd patch \
  --input REPERTOIRE_TUILES \
  --output REPERTOIRE_PATCHES \
  [--patch-size TAILLE] \
  [--overlap CHEVAUCHEMENT] \
  [--min-points MIN_POINTS] \
  [--include-rgb] \
  [--rgb-cache-dir REPERTOIRE_CACHE]
```

### Paramètres

| Paramètre         | Type    | Requis | Description                                         |
| ----------------- | ------- | ------ | --------------------------------------------------- |
| `--input`         | chaîne  | Oui    | Répertoire contenant les tuiles enrichies           |
| `--output`        | chaîne  | Oui    | Répertoire de sortie pour les patches               |
| `--patch-size`    | entier  | Non    | Taille des patches en mètres (défaut: 50)           |
| `--overlap`       | float   | Non    | Chevauchement entre patches (défaut: 0.1)           |
| `--min-points`    | entier  | Non    | Nombre minimum de points par patch                  |
| `--include-rgb`   | drapeau | Non    | Ajouter les couleurs RGB depuis les orthophotos IGN |
| `--rgb-cache-dir` | chaîne  | Non    | Répertoire de cache pour les orthophotos            |

### Exemples

```bash
# Créer des patches (géométrie uniquement)
ign-lidar-hd patch \
  --input ./data/enriched/ \
  --output ./data/patches/ \
  --patch-size 50

# Créer des patches avec augmentation RGB depuis les orthophotos IGN
ign-lidar-hd patch \
  --input ./data/enriched/ \
  --output ./data/patches/ \
  --include-rgb \
  --rgb-cache-dir ./cache/

# Traitement complet avec RGB
ign-lidar-hd patch \
  --input ./data/enriched/ \
  --output ./data/patches/ \
  --patch-size 100 \
  --overlap 0.2 \
  --include-rgb \
  --rgb-cache-dir ./cache/ \
  --workers 8
```

### Augmentation RGB

Lorsque `--include-rgb` est utilisé, la bibliothèque :

1. Récupère automatiquement les orthophotos depuis le service IGN BD ORTHO® (résolution 20cm)
2. Mappe chaque point 3D à son pixel 2D correspondant dans l'orthophoto
3. Extrait les couleurs RGB et les normalise dans la plage [0, 1]
4. Met en cache les orthophotos téléchargées pour améliorer les performances

**Avantages :**

- Apprentissage multi-modal (géométrie + photométrie)
- Meilleure précision des modèles ML
- Capacités de visualisation améliorées
- Automatique - aucun téléchargement manuel d'orthophotos nécessaire

**Prérequis :**

```bash
pip install requests Pillow
```

Voir le [Guide d'Augmentation RGB](../features/rgb-augmentation.md) pour plus d'informations.

## process (Obsolète)

:::warning Commande Obsolète
La commande `process` a été renommée en `patch` pour plus de clarté. Bien que `process` fonctionne toujours pour la compatibilité ascendante, elle sera supprimée dans une future version majeure. Veuillez utiliser `patch` à la place.
:::

### Migration

Remplacez simplement `process` par `patch` dans vos commandes :

```bash
# Ancien (obsolète)
ign-lidar-hd process --input tuiles/ --output patches/

# Nouveau (recommandé)
ign-lidar-hd patch --input tuiles/ --output patches/
```

Tous les paramètres et fonctionnalités restent identiques. Voir la [documentation de la commande `patch`](#patch) ci-dessus.

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
ign-lidar-hd download \
  --bbox 2.3,48.85,2.35,48.87 \
  --output ./data/raw/

# 2. Enrichir les tuiles
ign-lidar-hd enrich \
  --input ./data/raw/*.laz \
  --output ./data/enriched/ \
  --workers 4

# 3. Extraire les patches (commande renommée)
ign-lidar-hd patch \
  --input ./data/enriched/ \
  --output ./data/patches/ \
  --patch-size 50
```

### Traitement avec Augmentation RGB

```bash
# Pipeline complet avec couleurs RGB depuis les orthophotos IGN
# 1. Télécharger
ign-lidar-hd download \
  --bbox 2.3,48.85,2.35,48.87 \
  --output ./data/raw/

# 2. Enrichir
ign-lidar-hd enrich \
  --input ./data/raw/*.laz \
  --output ./data/enriched/ \
  --workers 4

# 3. Créer patches avec RGB
ign-lidar-hd patch \
  --input ./data/enriched/ \
  --output ./data/patches/ \
  --include-rgb \
  --rgb-cache-dir ./cache/ \
  --patch-size 50
```

### Traitement Avancé avec Optimisations

```bash
# Traitement optimisé avec contrôle mémoire et RGB
ign-lidar-hd patch \
  --input ./data/enriched/ \
  --output ./data/patches/ \
  --patch-size 100 \
  --overlap 0.2 \
  --workers 8 \
  --include-rgb \
  --rgb-cache-dir ./cache/ \
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
ign-lidar-hd process --low-memory --chunk-size 1000
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
