# 🎯 Multi-Scale Training Setup - Quick Start Guide

## ✅ Ce qui a été créé

### 1. Plan Détaillé

📄 **`MULTISCALE_TRAINING_PLAN.md`** - Plan complet avec :

- Architecture des données
- 5 phases de traitement détaillées
- Estimations de temps et espace disque
- Métriques de succès
- Checklist de démarrage

### 2. Configurations de Prétraitement

Les configurations suivantes enrichissent les tuiles avec toutes les features :

- ✅ `configs/multiscale/config_unified_asprs_preprocessing.yaml`
- ✅ `configs/multiscale/config_unified_lod2_preprocessing.yaml`
- ✅ `configs/multiscale/config_unified_lod3_preprocessing.yaml`

**Features incluses** : RGB, NIR, NDVI, features géométriques complètes, ground truth IGN

### 3. Configurations de Patches Multi-Échelles

#### ASPRS (3 échelles)

- ✅ `configs/multiscale/asprs/config_asprs_patches_50m.yaml` (16k points)
- ✅ `configs/multiscale/asprs/config_asprs_patches_100m.yaml` (24k points)
- ✅ `configs/multiscale/asprs/config_asprs_patches_150m.yaml` (32k points)

#### LOD2 (3 échelles)

- ✅ `configs/multiscale/lod2/config_lod2_patches_50m.yaml`
- ✅ `configs/multiscale/lod2/config_lod2_patches_100m.yaml`
- ✅ `configs/multiscale/lod2/config_lod2_patches_150m.yaml`

#### LOD3 (3 échelles)

- ✅ `configs/multiscale/lod3/config_lod3_patches_50m.yaml` (24k points)
- ✅ `configs/multiscale/lod3/config_lod3_patches_100m.yaml` (32k points)
- ✅ `configs/multiscale/lod3/config_lod3_patches_150m.yaml` (40k points)

**Total : 12 configurations de patches** (3 LOD × 3 scales + 3 preprocessing)

### 4. Script d'Automatisation

✅ **`scripts/run_complete_pipeline.sh`** - Script bash complet qui :

- Orchestre toutes les 5 phases
- Peut être exécuté par phase ou complètement
- Supporte le mode parallèle pour les patches
- Génère des logs colorés avec progression
- Inclut dry-run pour tester

### 5. Documentation

✅ **`configs/multiscale/README.md`** - Guide de référence rapide avec :

- Commandes pour chaque phase
- Paramètres clés par échelle
- Personnalisation
- Troubleshooting

---

## 🚀 Pour Démarrer MAINTENANT

### Option 1 : Automatique (Recommandé)

```bash
cd /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET

# Lancer toute la pipeline
./scripts/run_complete_pipeline.sh \
    --unified-dataset /mnt/c/Users/Simon/ign/unified_dataset \
    --output-base /mnt/c/Users/Simon/ign \
    --phases all \
    --gpu
```

### Option 2 : Phase par Phase

```bash
# Phase 1 : Sélection des tuiles
./scripts/run_complete_pipeline.sh --phases 1

# Phase 2 : Prétraitement (enrichissement features)
./scripts/run_complete_pipeline.sh --phases 2

# Phase 3 : Génération patches multi-échelles
./scripts/run_complete_pipeline.sh --phases 3

# Phase 4 : Fusion des datasets
./scripts/run_complete_pipeline.sh --phases 4

# Phase 5 : Entraînement des modèles
./scripts/run_complete_pipeline.sh --phases 5
```

### Option 3 : Manuel (pour plus de contrôle)

#### Étape 1 : Prétraitement ASPRS

```bash
ign-lidar-hd process \
    --config configs/multiscale/config_unified_asprs_preprocessing.yaml
```

#### Étape 2 : Génération patches ASPRS 50m

```bash
ign-lidar-hd process \
    --config configs/multiscale/asprs/config_asprs_patches_50m.yaml
```

#### Étape 3 : Répéter pour autres échelles et LOD

---

## 📋 Checklist Avant de Commencer

### Prérequis

- [ ] Python 3.8+ installé
- [ ] Package `ign-lidar-hd` installé : `pip install -e .`
- [ ] GPU avec 16+ GB VRAM (recommandé)
- [ ] 32+ GB RAM système
- [ ] 3 TB espace disque disponible sur `C:\Users\Simon\ign\`

### Vérification Dataset

- [ ] Dataset unifié existe : `C:\Users\Simon\ign\unified_dataset`
- [ ] Tuiles au format LAZ
- [ ] Données RGB et NIR disponibles

### Configuration Système

- [ ] Variable d'environnement IGN_API_KEY configurée (si téléchargement nécessaire)
- [ ] Accès Internet pour BD TOPO® (ground truth)

---

## 📊 Structure des Données Générées

```
C:\Users\Simon\ign\
├── unified_dataset\              # Votre dataset actuel (input)
│
├── selected_tiles\               # Tuiles sélectionnées pour entraînement
│   ├── asprs\ (100 tuiles)
│   ├── lod2\ (80 tuiles)
│   └── lod3\ (60 tuiles)
│
├── preprocessed\                 # Tuiles enrichies avec features
│   ├── asprs\enriched_tiles\
│   ├── lod2\enriched_tiles\
│   └── lod3\enriched_tiles\
│
├── patches\                      # Patches d'entraînement
│   ├── asprs\{50m,100m,150m}\{train,val,test}\
│   ├── lod2\{50m,100m,150m}\{train,val,test}\
│   └── lod3\{50m,100m,150m}\{train,val,test}\
│
├── merged_datasets\              # Datasets fusionnés multi-échelles
│   ├── asprs_multiscale\
│   ├── lod2_multiscale\
│   └── lod3_multiscale\
│
└── models\                       # Modèles entraînés
    ├── asprs\{pointnet++,point_transformer,intelligent_index}\
    ├── lod2\{...}\
    └── lod3\{...}\
```

---

## 🎯 Caractéristiques Clés du Plan

### Features Complètes

- ✅ **RGB** : Couleurs Rouge, Vert, Bleu
- ✅ **NIR** : Infrarouge proche
- ✅ **NDVI** : Index de végétation = (NIR - R) / (NIR + R)
- ✅ **Features géométriques** : Normales, courbure, planéité, etc.
- ✅ **Ground truth** : Classification IGN BD TOPO®

### 3 Augmentations par Patch

1. **Rotation** : 0°, 90°, 180°, 270°
2. **Flip** : Horizontal et vertical
3. **Jitter** : Petites variations XYZ (±5%)
4. **Scale** (LOD2/3) : Variations d'échelle
5. **Color jitter** (LOD3) : Variations RGB légères

### Tile Stitching Intelligent

- Buffer zones aux bordures (8m, 12m, 15m selon échelle)
- Auto-détection des tuiles voisines
- Cache des tuiles pour performance
- Évite les artefacts aux limites

### Classification Progressive

**ASPRS** → **LOD2** → **LOD3**

Chaque niveau utilise le modèle précédent comme pré-entraînement :

- ASPRS : Formation de base (ground, vegetation, building, water)
- LOD2 : Spécialisation bâtiments (walls, roof types, details)
- LOD3 : Détails fins (windows, doors, balconies)

### 3 Architectures de Modèles

1. **PointNet++** : Rapide, robuste, baseline solide
2. **Point Transformer** : Attention globale, meilleure précision
3. **Intelligent Index** : Relations spatiales optimisées

---

## ⏱️ Temps Estimés

| Phase                 | Durée Estimée   | Parallélisable |
| --------------------- | --------------- | -------------- |
| 1. Sélection tuiles   | 2-4 heures      | Non            |
| 2. Prétraitement      | 3 jours         | Partiellement  |
| 3. Génération patches | 4 jours         | Oui            |
| 4. Fusion datasets    | 12 heures       | Non            |
| 5. Entraînement       | 8-16 jours      | Par modèle     |
| **TOTAL**             | **15-25 jours** | -              |

---

## 💾 Espace Disque Requis

| Composant                 | Taille      | Description                |
| ------------------------- | ----------- | -------------------------- |
| Tuiles sélectionnées      | 360 GB      | Liens vers unified_dataset |
| Enrichies (preprocessed)  | 600 GB      | Avec toutes les features   |
| Patches (toutes échelles) | 980 GB      | NPZ + LAZ                  |
| Datasets fusionnés        | 540 GB      | NPZ optimisés              |
| Modèles + checkpoints     | 37 GB       | Tous les modèles           |
| **TOTAL REQUIS**          | **~2.5 TB** | Recommandé : 3 TB          |

---

## 🔧 Personnalisation Facile

### Changer le Nombre de Tuiles

Dans `run_complete_pipeline.sh`, ligne ~200 :

```bash
--asprs-count 100  # Modifiez ici
--lod2-count 80
--lod3-count 60
```

### Ajuster les Tailles de Patches

Dans les fichiers config individuels :

```yaml
processor:
  patch_size: 75.0 # Au lieu de 50/100/150
  num_points: 20000 # Ajustez selon besoins
```

### Modifier les Augmentations

```yaml
processor:
  num_augmentations: 5 # Plus ou moins
  augmentation_types:
    - rotation
    - flip
    - jitter
    - scale
    - color_jitter
```

---

## 📞 Prochaines Étapes

### 1. Test Rapide (Dry-Run)

```bash
./scripts/run_complete_pipeline.sh --dry-run --phases 1,2
```

Affiche les commandes sans les exécuter.

### 2. Commencer Phase 1

```bash
./scripts/run_complete_pipeline.sh --phases 1
```

Sélectionne les meilleures tuiles du unified_dataset.

### 3. Lancer Prétraitement

```bash
./scripts/run_complete_pipeline.sh --phases 2
```

Enrichit les tuiles avec RGB, NIR, NDVI, features.

### 4. Générer Patches (peut être long)

```bash
./scripts/run_complete_pipeline.sh --phases 3 --parallel-patches
```

Mode parallèle recommandé si RAM suffisante.

### 5. Surveiller la Progression

```bash
# Logs en temps réel
tail -f /mnt/c/Users/Simon/ign/logs/*.log

# Vérifier les outputs
ls -lh /mnt/c/Users/Simon/ign/patches/asprs/50m/train/
```

---

## 🐛 Problèmes Courants

### Erreur : Out of Memory

**Solution** :

```bash
# Réduire les workers
num_workers: 2  # Au lieu de 4

# Ou traiter séquentiellement
./scripts/run_complete_pipeline.sh --phases 3  # Sans --parallel-patches
```

### Erreur : Ground Truth Non Trouvé

**Solution** :

```bash
# Vérifier variable d'environnement
echo $IGN_API_KEY

# Ou désactiver ground truth temporairement
ground_truth:
  enabled: false
```

### Traitement Trop Lent

**Solution** :

```bash
# Augmenter les workers (si RAM disponible)
num_workers: 6

# Augmenter GPU batch size
gpu_batch_size: 2000000
```

---

## 📚 Documentation Complète

- **Plan détaillé** : `MULTISCALE_TRAINING_PLAN.md`
- **Guide rapide** : `configs/multiscale/README.md`
- **Script principal** : `scripts/run_complete_pipeline.sh`
- **Configurations** : `configs/multiscale/`

---

## ✨ Résumé

Vous avez maintenant **un système complet** pour :

1. ✅ Sélectionner automatiquement les meilleures tuiles
2. ✅ Enrichir avec RGB, NIR, NDVI, features complètes
3. ✅ Générer patches à 3 échelles (50m, 100m, 150m)
4. ✅ Appliquer 3-5 augmentations par patch
5. ✅ Fusionner en datasets multi-échelles
6. ✅ Entraîner 3 architectures (PointNet++, Point Transformer, Intelligent Index)
7. ✅ Progression ASPRS → LOD2 → LOD3 avec transfer learning
8. ✅ Classifier des fichiers LAZ complets à différents niveaux

**Total : 15 fichiers créés** (1 plan, 12 configs, 1 script, 1 README)

---

**🎉 Vous êtes prêt à commencer l'entraînement multi-échelle !**

Lancez simplement :

```bash
cd /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET
./scripts/run_complete_pipeline.sh --phases 1
```
