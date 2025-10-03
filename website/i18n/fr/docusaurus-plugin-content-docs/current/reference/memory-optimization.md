---
sidebar_position: 1
title: Optimisation mémoire
description: Guide de gestion de l'utilisation mémoire lors du traitement LiDAR
keywords: [mémoire, optimisation, performance, dépannage]
---

# Guide d'optimisation mémoire

Apprenez à optimiser l'utilisation de la mémoire et éviter les erreurs de mémoire insuffisante lors du traitement de gros jeux de données LiDAR.

## Comprendre les besoins mémoire

Le traitement LiDAR est intensif en mémoire, surtout pour l'analyse des composants de bâtiment. Voici ce qu'il faut savoir sur les modèles d'utilisation mémoire.

### Utilisation mémoire par mode de traitement

#### Caractéristiques du mode Core

- **Caractéristiques de base** : ~40 octets par point (normales, courbure, etc.)
- **KDTree** : ~24 octets par point
- **Total** : ~70 octets par point

#### Caractéristiques du mode Building

- **Caractéristiques de base** : ~40 octets par point
- **KDTree Building** : ~50 octets par point
- **Caractéristiques supplémentaires** : ~60 octets par point
- **Total** : ~150 octets par point

### Taille de fichier vs Besoins mémoire

| Taille fichier | Points      | RAM mode Core | RAM mode Building |
| -------------- | ----------- | ------------- | ----------------- |
| 100MB          | ~2M points  | ~140MB        | ~300MB            |
| 200MB          | ~4M points  | ~280MB        | ~600MB            |
| 300MB          | ~6M points  | ~420MB        | ~900MB            |
| 500MB          | ~10M points | ~700MB        | ~1.5GB            |
| 1GB            | ~20M points | ~1.4GB        | ~3GB              |

## Gestion automatique de la mémoire

La bibliothèque inclut des fonctionnalités de gestion mémoire intégrées :

### 1. Vérification mémoire avant traitement

Avant le début du traitement, le système :

- Vérifie la RAM disponible
- Détecte l'utilisation du swap
- Estime les besoins mémoire
- Ajuste automatiquement les paramètres

### 2. Traitement par chunks adaptatif

```python
from ign_lidar import LiDARProcessor

# Configuration automatique basée sur la RAM disponible
processor = LiDARProcessor(
    auto_memory_management=True,  # Activé par défaut
    max_memory_gb=8  # Limite maximale optionnelle
)
```

## Optimisations manuelles

### Configuration des chunks

```python
# Traitement par chunks pour gros fichiers
processor = LiDARProcessor(
    chunk_size=1000000,  # 1M points par chunk
    overlap_size=50000   # Chevauchement de 50k points
)
```

### Réduction de mémoire par mode

```python
# Mode minimal pour ressources limitées
processor = LiDARProcessor(
    mode="core",  # Au lieu de "building"
    enable_gpu=False,  # Désactiver GPU si RAM limitée
    cache_size=100  # Réduire la taille du cache
)
```

## Configuration système recommandée

### RAM minimale par mode

- **Mode Core** : 4GB RAM
- **Mode Building** : 8GB RAM
- **Traitement par lots** : 16GB RAM

### RAM recommandée

- **Développement** : 16GB RAM
- **Production** : 32GB RAM ou plus
- **GPU processing** : RAM GPU 8GB+

## Surveillance et dépannage

### Vérifier l'utilisation mémoire

```bash
# Surveiller l'utilisation mémoire pendant le traitement
python -m ign_lidar.cli process --input-dir data/ --output patches/ --verbose
```

### Messages d'erreur courants

#### "Out of Memory"

```python
# Solution : Réduire la taille des chunks
processor = LiDARProcessor(chunk_size=500000)
```

#### "Swap space full"

```bash
# Solution : Augmenter le swap ou utiliser moins de workers
python -m ign_lidar.cli process --num-workers 1
```

## Optimisations avancées

### Traitement GPU

```python
# Le GPU peut réduire l'utilisation RAM CPU
processor = LiDARProcessor(
    enable_gpu=True,
    gpu_memory_fraction=0.8  # Utiliser 80% de la RAM GPU
)
```

### Configuration de production

```python
# Configuration optimisée pour serveurs
processor = LiDARProcessor(
    mode="building",
    num_workers=8,
    chunk_size=2000000,
    max_memory_gb=24,
    enable_gpu=True,
    cache_size=1000
)
```

## Bonnes pratiques

### ✅ Recommandé

- **Surveiller la RAM** : Utiliser des outils de monitoring
- **Commencer petit** : Tester avec de petits fichiers
- **Ajuster progressivement** : Augmenter la taille des chunks graduellement
- **Utiliser le GPU** : Quand disponible pour réduire l'utilisation RAM CPU

### ❌ Éviter

- **Trop de workers** : Peut surcharger la mémoire
- **Chunks trop gros** : Risque de débordement mémoire
- **Ignorer les avertissements** : Les messages de mémoire sont importants
- **Traitement concurrent** : Éviter plusieurs traitements simultanés

## Exemples de configuration

### Configuration limitée (8GB RAM)

```python
processor = LiDARProcessor(
    mode="core",
    num_workers=2,
    chunk_size=500000,
    max_memory_gb=6
)
```

### Configuration standard (16GB RAM)

```python
processor = LiDARProcessor(
    mode="building",
    num_workers=4,
    chunk_size=1000000,
    max_memory_gb=12
)
```

### Configuration haute performance (32GB+ RAM)

```python
processor = LiDARProcessor(
    mode="building",
    num_workers=8,
    chunk_size=2000000,
    enable_gpu=True,
    max_memory_gb=24
)
```
