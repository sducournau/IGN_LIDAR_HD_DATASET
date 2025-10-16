# 🚀 Plan d'Entraînement Multi-Échelle - ASPRS, LOD2, LOD3

## 📋 Vue d'Ensemble du Projet

Ce plan décrit une stratégie complète pour entraîner des modèles de classification de nuages de points à plusieurs échelles (50m, 100m, 150m) pour trois niveaux de détail progressifs :

1. **ASPRS** : Classification standard (sol, végétation, bâtiment, eau, etc.)
2. **LOD2** : Classification détaillée des bâtiments (15 classes - murs, types de toits, détails)
3. **LOD3** : Classification ultra-détaillée (30 classes - murs avec ouvertures, types de toits avancés)

### 🎯 Objectifs

- Utiliser les tuiles du dataset unifié : `C:\Users\Simon\ign\unified_dataset`
- Générer des patches à 3 échelles différentes (50m, 100m, 150m)
- Traiter avec toutes les features : RGB, NIR, NDVI, features géométriques complètes
- Appliquer 3 augmentations par patch
- Outputs : NPZ (entraînement) + LAZ (visualisation)
- Pré-classification avec ground truth IGN et NDVI
- Configuration tile stitcher pour gestion des bordures
- Entraînement sur architectures hybrides : PointNet++, Point Transformer, Intelligent Index

---

## 📂 Structure des Données

```
C:\Users\Simon\ign\
├── unified_dataset\              # Dataset de base (tuiles déjà téléchargées)
│   ├── tile_001.laz
│   ├── tile_002.laz
│   └── ...
│
├── selected_tiles\               # Sélection optimisée pour entraînement
│   ├── asprs_tiles.txt          # Liste des tuiles pour ASPRS
│   ├── lod2_tiles.txt           # Liste des tuiles pour LOD2
│   ├── lod3_tiles.txt           # Liste des tuiles pour LOD3
│   └── tiles\                   # Liens symboliques vers unified_dataset
│
├── preprocessed\                 # Tuiles enrichies avec features
│   ├── asprs\
│   │   ├── enriched_tiles\      # LAZ avec features + preclassification
│   │   └── metadata\
│   ├── lod2\
│   └── lod3\
│
├── patches\                      # Patches d'entraînement multi-échelles
│   ├── asprs\
│   │   ├── 50m\                 # Patches 50m x 50m
│   │   │   ├── train\
│   │   │   ├── val\
│   │   │   └── test\
│   │   ├── 100m\                # Patches 100m x 100m
│   │   └── 150m\                # Patches 150m x 150m
│   ├── lod2\
│   └── lod3\
│
├── merged_datasets\              # Datasets fusionnés multi-échelles
│   ├── asprs_multiscale\
│   ├── lod2_multiscale\
│   └── lod3_multiscale\
│
└── models\                       # Modèles entraînés
    ├── asprs\
    │   ├── pointnet++\
    │   ├── point_transformer\
    │   └── intelligent_index\
    ├── lod2\
    └── lod3\
```

---

## 🔄 Pipeline de Traitement (5 Phases)

### Phase 1 : Sélection des Tuiles Optimales 🎯

**Objectif** : Identifier et sélectionner les meilleures tuiles du `unified_dataset` pour chaque niveau de classification.

**Critères de Sélection** :

- **ASPRS** : Diversité des classes (bâtiments, végétation, eau, routes)
- **LOD2** : Zones avec bâtiments variés (résidentiel, commercial, industriel)
- **LOD3** : Bâtiments avec détails architecturaux (fenêtres, portes, balcons)
- Qualité des données RGB et NIR
- Absence de trous majeurs dans les données
- Couverture spatiale représentative

**Actions** :

```bash
# 1. Analyser le unified_dataset
cd /mnt/c/Users/Simon/ign
python scripts/analyze_unified_dataset.py \
    --input unified_dataset \
    --output analysis_report.json

# 2. Sélectionner les tuiles optimales
python scripts/select_optimal_tiles.py \
    --input unified_dataset \
    --analysis analysis_report.json \
    --output selected_tiles \
    --asprs-count 100 \
    --lod2-count 80 \
    --lod3-count 60

# 3. Créer des liens symboliques (Windows: mklink)
python scripts/create_tile_links.py \
    --source unified_dataset \
    --target selected_tiles/tiles \
    --lists selected_tiles/*.txt
```

---

### Phase 2 : Prétraitement et Enrichissement 🔧

**Objectif** : Enrichir les tuiles sélectionnées avec toutes les features nécessaires et effectuer une pré-classification.

#### 2.1 Configuration ASPRS

**Fichier** : `config_unified_asprs_preprocessing.yaml`

**Features Calculées** :

- RGB (Red, Green, Blue)
- NIR (Near-Infrared)
- NDVI = (NIR - Red) / (NIR + Red)
- Features géométriques : normales, courbure, planéité, etc.
- Ground truth IGN (BD TOPO®)

**Commande** :

```bash
ign-lidar-hd process \
    --config configs/multiscale/config_unified_asprs_preprocessing.yaml
```

#### 2.2 Configuration LOD2

**Fichier** : `config_unified_lod2_preprocessing.yaml`

**Spécificités LOD2** :

- Focus sur les bâtiments
- Classification préliminaire : murs, toits (plats, gables, hip)
- Features additionnelles : angles de toit, hauteur relative

**Commande** :

```bash
ign-lidar-hd process \
    --config configs/multiscale/config_unified_lod2_preprocessing.yaml
```

#### 2.3 Configuration LOD3

**Fichier** : `config_unified_lod3_preprocessing.yaml`

**Spécificités LOD3** :

- Détails architecturaux fins
- Classification préliminaire : 30 classes incluant fenêtres, portes, balcons
- Features haute résolution : k_neighbors=50, search_radius=0.5m

**Commande** :

```bash
ign-lidar-hd process \
    --config configs/multiscale/config_unified_lod3_preprocessing.yaml
```

---

### Phase 3 : Génération de Patches Multi-Échelles 📦

**Objectif** : Créer des patches d'entraînement à 3 échelles (50m, 100m, 150m) avec augmentations.

#### 3.1 ASPRS - Patches 50m

**Configuration** : `config_asprs_patches_50m.yaml`

**Paramètres** :

- Taille : 50m x 50m
- Overlap : 15% (7.5m)
- Points par patch : 16,384
- Augmentations : 3 (rotation, flip, jitter)
- Output : NPZ + LAZ

**Commande** :

```bash
ign-lidar-hd process \
    --config configs/multiscale/asprs/config_asprs_patches_50m.yaml
```

#### 3.2 ASPRS - Patches 100m

**Configuration** : `config_asprs_patches_100m.yaml`

**Paramètres** :

- Taille : 100m x 100m
- Overlap : 10% (10m)
- Points par patch : 24,576
- Augmentations : 3
- Output : NPZ + LAZ

**Commande** :

```bash
ign-lidar-hd process \
    --config configs/multiscale/asprs/config_asprs_patches_100m.yaml
```

#### 3.3 ASPRS - Patches 150m

**Configuration** : `config_asprs_patches_150m.yaml`

**Paramètres** :

- Taille : 150m x 150m
- Overlap : 10% (15m)
- Points par patch : 32,768
- Augmentations : 3
- Output : NPZ + LAZ

**Commande** :

```bash
ign-lidar-hd process \
    --config configs/multiscale/asprs/config_asprs_patches_150m.yaml
```

#### 3.4 LOD2 - Multi-Échelles (50m, 100m, 150m)

**Configurations** :

- `config_lod2_patches_50m.yaml`
- `config_lod2_patches_100m.yaml`
- `config_lod2_patches_150m.yaml`

**Spécificités LOD2** :

- Focus sur zones avec bâtiments
- k_neighbors augmenté pour détails de toit
- Buffer size adapté : 5m (50m), 8m (100m), 12m (150m)

**Commandes** :

```bash
# Génération séquentielle
for scale in 50 100 150; do
    ign-lidar-hd process \
        --config configs/multiscale/lod2/config_lod2_patches_${scale}m.yaml
done
```

#### 3.5 LOD3 - Multi-Échelles (50m, 100m, 150m)

**Configurations** :

- `config_lod3_patches_50m.yaml`
- `config_lod3_patches_100m.yaml`
- `config_lod3_patches_150m.yaml`

**Spécificités LOD3** :

- Haute résolution : k_neighbors=50
- Plus de points : 24k (50m), 32k (100m), 40k (150m)
- Augmentations plus agressives : 5 augmentations

**Commandes** :

```bash
# Génération séquentielle
for scale in 50 100 150; do
    ign-lidar-hd process \
        --config configs/multiscale/lod3/config_lod3_patches_${scale}m.yaml
done
```

---

### Phase 4 : Fusion des Datasets Multi-Échelles 🔀

**Objectif** : Combiner les patches de toutes les échelles en un seul dataset d'entraînement.

**Outil** : `merge_multiscale_dataset.py`

**Stratégies de Fusion** :

1. **Balanced** : Nombre égal de patches par échelle
2. **Weighted** : Plus de patches à l'échelle optimale par classe
3. **Adaptive** : Distribution basée sur la complexité des scènes

#### 4.1 Fusion ASPRS

```bash
python examples/merge_multiscale_dataset.py \
    --input-dirs \
        /mnt/c/Users/Simon/ign/patches/asprs/50m \
        /mnt/c/Users/Simon/ign/patches/asprs/100m \
        /mnt/c/Users/Simon/ign/patches/asprs/150m \
    --output /mnt/c/Users/Simon/ign/merged_datasets/asprs_multiscale \
    --strategy balanced \
    --split 0.7 0.15 0.15 \
    --balance-classes
```

#### 4.2 Fusion LOD2

```bash
python examples/merge_multiscale_dataset.py \
    --input-dirs \
        /mnt/c/Users/Simon/ign/patches/lod2/50m \
        /mnt/c/Users/Simon/ign/patches/lod2/100m \
        /mnt/c/Users/Simon/ign/patches/lod2/150m \
    --output /mnt/c/Users/Simon/ign/merged_datasets/lod2_multiscale \
    --strategy weighted \
    --weights 0.3 0.4 0.3 \
    --split 0.7 0.15 0.15 \
    --balance-classes
```

#### 4.3 Fusion LOD3

```bash
python examples/merge_multiscale_dataset.py \
    --input-dirs \
        /mnt/c/Users/Simon/ign/patches/lod3/50m \
        /mnt/c/Users/Simon/ign/patches/lod3/100m \
        /mnt/c/Users/Simon/ign/patches/lod3/150m \
    --output /mnt/c/Users/Simon/ign/merged_datasets/lod3_multiscale \
    --strategy adaptive \
    --split 0.7 0.15 0.15 \
    --balance-classes \
    --oversample-rare
```

---

### Phase 5 : Entraînement des Modèles 🤖

**Objectif** : Entraîner 3 architectures de modèles sur chaque niveau de classification.

#### Architectures Hybrides

1. **PointNet++** : Architecture de base, rapide et robuste
2. **Point Transformer** : Attention mechanism pour contexte global
3. **Intelligent Index** : Architecture spécialisée pour relations spatiales

#### 5.1 Entraînement ASPRS

##### PointNet++ - ASPRS

```bash
python -m ign_lidar.core.train \
    --config configs/training/asprs/pointnet++_asprs.yaml \
    --data /mnt/c/Users/Simon/ign/merged_datasets/asprs_multiscale \
    --output /mnt/c/Users/Simon/ign/models/asprs/pointnet++ \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001 \
    --patience 15
```

##### Point Transformer - ASPRS

```bash
python -m ign_lidar.core.train \
    --config configs/training/asprs/point_transformer_asprs.yaml \
    --data /mnt/c/Users/Simon/ign/merged_datasets/asprs_multiscale \
    --output /mnt/c/Users/Simon/ign/models/asprs/point_transformer \
    --epochs 150 \
    --batch-size 16 \
    --lr 0.0005 \
    --patience 20
```

##### Intelligent Index - ASPRS

```bash
python -m ign_lidar.core.train \
    --config configs/training/asprs/intelligent_index_asprs.yaml \
    --data /mnt/c/Users/Simon/ign/merged_datasets/asprs_multiscale \
    --output /mnt/c/Users/Simon/ign/models/asprs/intelligent_index \
    --epochs 120 \
    --batch-size 24 \
    --lr 0.0008 \
    --patience 18
```

#### 5.2 Entraînement LOD2

**Pré-requis** : Utiliser le meilleur modèle ASPRS comme pré-entraînement

##### PointNet++ - LOD2

```bash
python -m ign_lidar.core.train \
    --config configs/training/lod2/pointnet++_lod2.yaml \
    --data /mnt/c/Users/Simon/ign/merged_datasets/lod2_multiscale \
    --output /mnt/c/Users/Simon/ign/models/lod2/pointnet++ \
    --pretrained /mnt/c/Users/Simon/ign/models/asprs/pointnet++/best_model.pth \
    --epochs 150 \
    --batch-size 24 \
    --lr 0.0005 \
    --patience 20 \
    --freeze-backbone 10
```

##### Point Transformer - LOD2

```bash
python -m ign_lidar.core.train \
    --config configs/training/lod2/point_transformer_lod2.yaml \
    --data /mnt/c/Users/Simon/ign/merged_datasets/lod2_multiscale \
    --output /mnt/c/Users/Simon/ign/models/lod2/point_transformer \
    --pretrained /mnt/c/Users/Simon/ign/models/asprs/point_transformer/best_model.pth \
    --epochs 200 \
    --batch-size 12 \
    --lr 0.0003 \
    --patience 25 \
    --freeze-backbone 15
```

##### Intelligent Index - LOD2

```bash
python -m ign_lidar.core.train \
    --config configs/training/lod2/intelligent_index_lod2.yaml \
    --data /mnt/c/Users/Simon/ign/merged_datasets/lod2_multiscale \
    --output /mnt/c/Users/Simon/ign/models/lod2/intelligent_index \
    --pretrained /mnt/c/Users/Simon/ign/models/asprs/intelligent_index/best_model.pth \
    --epochs 180 \
    --batch-size 16 \
    --lr 0.0004 \
    --patience 22 \
    --freeze-backbone 12
```

#### 5.3 Entraînement LOD3

**Pré-requis** : Utiliser le meilleur modèle LOD2 comme pré-entraînement

##### PointNet++ - LOD3

```bash
python -m ign_lidar.core.train \
    --config configs/training/lod3/pointnet++_lod3.yaml \
    --data /mnt/c/Users/Simon/ign/merged_datasets/lod3_multiscale \
    --output /mnt/c/Users/Simon/ign/models/lod3/pointnet++ \
    --pretrained /mnt/c/Users/Simon/ign/models/lod2/pointnet++/best_model.pth \
    --epochs 200 \
    --batch-size 16 \
    --lr 0.0003 \
    --patience 25 \
    --freeze-backbone 15 \
    --class-weights auto
```

##### Point Transformer - LOD3

```bash
python -m ign_lidar.core.train \
    --config configs/training/lod3/point_transformer_lod3.yaml \
    --data /mnt/c/Users/Simon/ign/merged_datasets/lod3_multiscale \
    --output /mnt/c/Users/Simon/ign/models/lod3/point_transformer \
    --pretrained /mnt/c/Users/Simon/ign/models/lod2/point_transformer/best_model.pth \
    --epochs 250 \
    --batch-size 8 \
    --lr 0.0002 \
    --patience 30 \
    --freeze-backbone 20 \
    --class-weights auto \
    --focal-loss
```

##### Intelligent Index - LOD3

```bash
python -m ign_lidar.core.train \
    --config configs/training/lod3/intelligent_index_lod3.yaml \
    --data /mnt/c/Users/Simon/ign/merged_datasets/lod3_multiscale \
    --output /mnt/c/Users/Simon/ign/models/lod3/intelligent_index \
    --pretrained /mnt/c/Users/Simon/ign/models/lod2/intelligent_index/best_model.pth \
    --epochs 220 \
    --batch-size 12 \
    --lr 0.00025 \
    --patience 28 \
    --freeze-backbone 18 \
    --class-weights auto \
    --focal-loss
```

---

## 📊 Évaluation et Inférence

### Évaluation des Modèles

```bash
# Évaluer tous les modèles ASPRS
python -m ign_lidar.core.evaluate \
    --models \
        /mnt/c/Users/Simon/ign/models/asprs/pointnet++/best_model.pth \
        /mnt/c/Users/Simon/ign/models/asprs/point_transformer/best_model.pth \
        /mnt/c/Users/Simon/ign/models/asprs/intelligent_index/best_model.pth \
    --test-data /mnt/c/Users/Simon/ign/merged_datasets/asprs_multiscale/test \
    --output /mnt/c/Users/Simon/ign/evaluation/asprs_comparison.json \
    --metrics accuracy iou f1 precision recall \
    --visualize
```

### Classification de Fichiers LAZ Complets

#### Classification Progressive : ASPRS → LOD2 → LOD3

```bash
# 1. Classification ASPRS
python -m ign_lidar.core.classify \
    --input /mnt/c/Users/Simon/ign/unified_dataset/tile_example.laz \
    --model /mnt/c/Users/Simon/ign/models/asprs/point_transformer/best_model.pth \
    --output /mnt/c/Users/Simon/ign/classified/tile_example_asprs.laz \
    --level ASPRS \
    --patch-size 150 \
    --overlap 0.2 \
    --gpu

# 2. Classification LOD2 (sur les bâtiments détectés en ASPRS)
python -m ign_lidar.core.classify \
    --input /mnt/c/Users/Simon/ign/classified/tile_example_asprs.laz \
    --model /mnt/c/Users/Simon/ign/models/lod2/point_transformer/best_model.pth \
    --output /mnt/c/Users/Simon/ign/classified/tile_example_lod2.laz \
    --level LOD2 \
    --filter-class building \
    --patch-size 100 \
    --overlap 0.25 \
    --gpu

# 3. Classification LOD3 (détails architecturaux sur bâtiments LOD2)
python -m ign_lidar.core.classify \
    --input /mnt/c/Users/Simon/ign/classified/tile_example_lod2.laz \
    --model /mnt/c/Users/Simon/ign/models/lod3/point_transformer/best_model.pth \
    --output /mnt/c/Users/Simon/ign/classified/tile_example_lod3.laz \
    --level LOD3 \
    --filter-class wall roof \
    --patch-size 50 \
    --overlap 0.3 \
    --gpu \
    --refine-boundaries
```

---

## ⏱️ Estimation des Temps de Traitement

### Phase 1 : Sélection des Tuiles

- **Durée** : 2-4 heures
- **Dépend de** : Nombre de tuiles dans unified_dataset

### Phase 2 : Prétraitement

- **ASPRS** : ~10 minutes/tuile → 100 tuiles = 16-20 heures
- **LOD2** : ~15 minutes/tuile → 80 tuiles = 20-24 heures
- **LOD3** : ~20 minutes/tuile → 60 tuiles = 20-24 heures
- **Total Phase 2** : 56-68 heures (~3 jours)

### Phase 3 : Génération de Patches

- **Par échelle** : 8-12 heures
- **3 échelles × 3 niveaux** : 72-108 heures (~4 jours)

### Phase 4 : Fusion des Datasets

- **Par niveau** : 2-4 heures
- **Total Phase 4** : 6-12 heures

### Phase 5 : Entraînement

- **PointNet++** : 12-24 heures par niveau
- **Point Transformer** : 24-48 heures par niveau
- **Intelligent Index** : 18-36 heures par niveau
- **Total Phase 5** : ~200-400 heures (8-16 jours)

**TEMPS TOTAL ESTIMÉ** : 15-25 jours de calcul

---

## 💾 Espace Disque Requis

### Stockage par Phase

| Phase                    | ASPRS  | LOD2   | LOD3   | Total  |
| ------------------------ | ------ | ------ | ------ | ------ |
| Tuiles sélectionnées     | 150 GB | 120 GB | 90 GB  | 360 GB |
| Enrichies (preprocessed) | 250 GB | 200 GB | 150 GB | 600 GB |
| Patches 50m              | 80 GB  | 70 GB  | 60 GB  | 210 GB |
| Patches 100m             | 120 GB | 100 GB | 90 GB  | 310 GB |
| Patches 150m             | 180 GB | 150 GB | 130 GB | 460 GB |
| Datasets fusionnés       | 200 GB | 180 GB | 160 GB | 540 GB |
| Modèles + checkpoints    | 10 GB  | 12 GB  | 15 GB  | 37 GB  |

**TOTAL REQUIS** : ~2.5 TB (recommandé : 3 TB pour sécurité)

---

## 🛠️ Scripts d'Automatisation

### Script Principal : `run_complete_pipeline.sh`

Ce script orchestre toute la pipeline :

```bash
./scripts/run_complete_pipeline.sh \
    --unified-dataset /mnt/c/Users/Simon/ign/unified_dataset \
    --output-base /mnt/c/Users/Simon/ign \
    --phases all \
    --parallel-patches \
    --gpu
```

**Options** :

- `--phases` : Phases à exécuter (1,2,3,4,5 ou all)
- `--parallel-patches` : Générer patches en parallèle (requiert plus de RAM)
- `--skip-existing` : Sauter les fichiers déjà traités
- `--dry-run` : Afficher les commandes sans exécuter

### Scripts Auxiliaires

1. **`analyze_unified_dataset.py`** : Analyse qualité et diversité des tuiles
2. **`select_optimal_tiles.py`** : Sélectionne les meilleures tuiles
3. **`monitor_training.py`** : Surveillance en temps réel de l'entraînement
4. **`batch_classify.py`** : Classification batch de plusieurs fichiers LAZ

---

## 📈 Métriques de Succès

### ASPRS

- **Overall Accuracy** : > 92%
- **mIoU** : > 75%
- **F1-Score par classe** : > 0.80 (sol, végétation, bâtiment)

### LOD2

- **Overall Accuracy** : > 85%
- **mIoU** : > 65%
- **F1-Score murs/toits** : > 0.75

### LOD3

- **Overall Accuracy** : > 78%
- **mIoU** : > 55%
- **F1-Score détails** : > 0.60 (fenêtres, portes, balcons)

---

## 🚨 Points d'Attention

### Gestion de la Mémoire

- **GPU** : Recommandé 16+ GB VRAM pour LOD3
- **RAM** : 32+ GB pour traitement parallèle
- **Swap** : Configurer 64 GB de swap si RAM limitée

### Tile Stitching

- **Buffer zones** : Essentielles pour éviter artefacts aux bordures
- **Auto-download neighbors** : Activé pour continuité spatiale
- **Cache** : Garder les tuiles voisines en cache

### Augmentation de Données

- **Rotation** : 0°, 90°, 180°, 270°
- **Flip** : Horizontal, vertical
- **Jitter** : ±5% sur XYZ, ±10% sur RGB

### Class Balancing

- **ASPRS** : Équilibrage simple (classes principales)
- **LOD2** : Sur-échantillonnage toits rares (hip, gambrel)
- **LOD3** : Sur-échantillonnage fort classes rares (ouvertures)

---

## 📞 Support et Documentation

- **Documentation complète** : `docs/` directory
- **Exemples** : `examples/` directory
- **Tests** : `tests/` directory
- **Issues GitHub** : Pour questions techniques

---

## ✅ Checklist de Démarrage

- [ ] Vérifier espace disque disponible (3 TB)
- [ ] Confirmer GPU disponible (16+ GB VRAM)
- [ ] Installer toutes les dépendances
- [ ] Analyser le unified_dataset
- [ ] Créer les répertoires de sortie
- [ ] Configurer les chemins dans les fichiers YAML
- [ ] Lancer Phase 1 : Sélection des tuiles
- [ ] Lancer Phase 2 : Prétraitement
- [ ] Lancer Phase 3 : Génération patches
- [ ] Lancer Phase 4 : Fusion datasets
- [ ] Lancer Phase 5 : Entraînement ASPRS
- [ ] Évaluer modèles ASPRS
- [ ] Lancer Phase 5 : Entraînement LOD2
- [ ] Évaluer modèles LOD2
- [ ] Lancer Phase 5 : Entraînement LOD3
- [ ] Évaluer modèles LOD3
- [ ] Tests d'inférence sur données réelles

---

**Date de création** : 15 Octobre 2025  
**Version** : 1.0  
**Auteur** : Simon Ducournau  
**Projet** : IGN LiDAR HD Dataset - Multi-Scale Training
