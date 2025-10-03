---
sidebar_position: 1
title: Détection intelligente de saut
description: Ignorer automatiquement les téléchargements, fichiers enrichis et patches existants
keywords: [saut, idempotent, reprendre, workflow]
---

La détection intelligente de saut évite les opérations redondantes en détectant et ignorant automatiquement les fichiers existants lors des workflows de téléchargement, enrichissement et traitement.

## Vue d'ensemble

Cette fonctionnalité ajoute une détection intelligente de saut à tous les workflows :

- **Saut de téléchargement** - Éviter de re-télécharger les tuiles existantes
- **Saut d'enrichissement** - Ignorer les fichiers déjà enrichis
- **Saut de traitement** - Ignorer les tuiles avec patches existants

## Avantages clés

### ⚡ Économies de temps

- **Téléchargements** : Ignorer le re-téléchargement de tuiles (~60 min économisées sur 50 tuiles)
- **Traitement** : Ignorer le retraitement de tuiles (~90 min économisées sur 50 tuiles)
- **Total** : ~150 minutes économisées sur un workflow typique

### 💾 Économies de ressources

- **Bande passante** : Éviter le téléchargement de gros fichiers en double (12+ GB sur 50 tuiles)
- **Espace disque** : Éviter la création de patches en double
- **CPU/Mémoire** : Éviter le calcul redondant de caractéristiques

### 🔄 Améliorations du workflow

- **Capacité de reprise** : Reprendre facilement après les interruptions
- **Constructions incrémentales** : Ajouter de nouvelles données aux jeux existants
- **Opérations idempotentes** : Sûr d'exécuter les commandes plusieurs fois

## Saut intelligent de téléchargement

Ignore automatiquement les tuiles existantes lors du téléchargement :

```bash
# Télécharge seulement les tuiles manquantes
python -m ign_lidar.cli download \
  --bbox 2.0,48.8,2.5,49.0 \
  --output tuiles/

# La sortie montre ce qui est ignoré vs téléchargé
⏭️  tuile_001.laz existe déjà (245 MB), ignore
Téléchargement tuile_002.laz...
✅ Téléchargé tuile_002.laz (238 MB)

📊 Résumé du téléchargement :
  Total de tuiles demandées : 10
  ✅ Téléchargées avec succès : 7
  ⏭️  Ignorées (déjà présentes) : 2
  ❌ Échec : 1
```

## Saut intelligent d'enrichissement

Ignore automatiquement les fichiers déjà enrichis :

```bash
# Enrichit seulement les nouveaux fichiers
python -m ign_lidar.cli enrich \
  --input-dir tuiles_brutes/ \
  --output tuiles_enrichies/

# Détection automatique des fichiers enrichis
⏭️  tuile_001_enrichie.laz existe déjà, ignore
Enrichissement tuile_002.laz...
✅ Fichier enrichi sauvegardé : tuile_002_enrichie.laz
```

## Saut intelligent de traitement

Ignore automatiquement les tuiles avec patches existants :

```bash
# Traite seulement les nouvelles tuiles
python -m ign_lidar.cli process \
  --input-dir tuiles_enrichies/ \
  --output patches/

# Détection automatique des patches existants
⏭️  tuile_001 a déjà 156 patches, ignore
Traitement tuile_002...
✅ Créé 142 patches depuis tuile_002
```

## Configuration

### Activer/Désactiver le saut

```python
from ign_lidar import LiDARProcessor

# Désactiver la détection de saut
processor = LiDARProcessor(skip_existing=False)

# Ou via CLI
python -m ign_lidar.cli process --no-skip
```

### Forcer le retraitement

```bash
# Forcer le re-téléchargement
python -m ign_lidar.cli download --force

# Forcer le re-enrichissement
python -m ign_lidar.cli enrich --force

# Forcer le retraitement
python -m ign_lidar.cli process --force
```

## Détection de fichiers

### Fichiers de téléchargement

Recherche les fichiers `.laz` correspondants :

```text
tuiles/
├── LIDARHD_FXX_0123_4567_LA93_IGN69_2020.laz ✅ Skip
├── LIDARHD_FXX_0124_4567_LA93_IGN69_2020.laz ✅ Skip
└── LIDARHD_FXX_0125_4567_LA93_IGN69_2020.laz ❌ Télécharger
```

### Fichiers enrichis

Recherche les fichiers avec suffixe `_enriched` :

```text
enrichies/
├── tuile_001_enriched.laz ✅ Skip
├── tuile_002_enriched.laz ✅ Skip
└── tuile_003.laz → tuile_003_enriched.laz ❌ Enrichir
```

### Patches de traitement

Recherche les répertoires de patches existants :

```text
patches/
├── tuile_001/ ✅ Skip (156 fichiers .npz)
├── tuile_002/ ✅ Skip (142 fichiers .npz)
└── tuile_003/ ❌ Traiter (répertoire manquant)
```

## Exemples pratiques

### Workflow de reprise

```bash
# Le traitement initial s'arrête après 3 tuiles
python -m ign_lidar.cli process --input-dir data/ --output patches/
# ❌ Erreur après tuile_003

# Reprendre automatiquement depuis la tuile_004
python -m ign_lidar.cli process --input-dir data/ --output patches/
# ⏭️  tuile_001 ignorée (patches existants)
# ⏭️  tuile_002 ignorée (patches existants)
# ⏭️  tuile_003 ignorée (patches existants)
# ✅ Traitement tuile_004...
```

### Construction incrémentale

```bash
# Traiter le lot initial
python -m ign_lidar.cli process --input-dir lot1/ --output patches/

# Ajouter plus de données plus tard
python -m ign_lidar.cli process --input-dir lot2/ --output patches/
# Traite seulement les nouvelles tuiles du lot2
```

## Meilleures pratiques

### ✅ Recommandé

- **Activer par défaut** - Laisser la détection de saut activée
- **Utiliser --force avec prudence** - Seulement quand nécessaire
- **Organiser par workflows** - Séparer téléchargement/enrichissement/traitement
- **Vérifier les logs** - S'assurer que les bons fichiers sont ignorés

### ❌ Éviter

- **Forcer sans raison** - Gaspille temps et ressources
- **Mélanger les versions** - Peut créer de la confusion
- **Ignorer les avertissements** - Les messages d'erreur sont importants

## Dépannage

### Fichiers non détectés

Si des fichiers existants ne sont pas détectés :

```bash
# Vérifier les noms de fichiers
ls -la tuiles/

# Vérifier les permissions
chmod 644 tuiles/*.laz

# Forcer la régénération si nécessaire
python -m ign_lidar.cli process --force
```

### Patches corrompus

Si des patches sont corrompus :

```bash
# Supprimer les patches corrompus
rm -rf patches/tuile_problematique/

# Retraiter cette tuile
python -m ign_lidar.cli process --input-dir data/ --output patches/
```
