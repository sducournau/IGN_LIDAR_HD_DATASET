# Plan d'Entraînement LOD2 Auto-Supervisé

## Modèle Hybride PointNet++ + Point Transformer

**Date:** 15 Octobre 2025  
**Objectif:** Classifier les bâtiments LiDAR HD IGN en LOD2 avec apprentissage auto-supervisé  
**Architecture:** Hybrid PointNet++ + Point Transformer avec index intelligent

---

## 📋 Table des Matières

- [Vue d'ensemble](#vue-densemble)
- [Architecture du Modèle](#architecture-du-modèle)
- [Stratégie Auto-Supervisée](#stratégie-auto-supervisée)
- [Estimation des Ressources](#estimation-des-ressources)
- [Pipeline de Données](#pipeline-de-données)
- [Plan d'Entraînement](#plan-dentraînement)
- [Implémentation](#implémentation)
- [Évaluation](#évaluation)

---

## 🎯 Vue d'ensemble

### Objectif Final

Développer un modèle hybride capable de :

1. **Pré-entraînement auto-supervisé** : Apprendre des représentations géométriques sans labels
2. **Classification LOD2** : Distinguer les niveaux de détail des bâtiments
3. **Généralisation** : Fonctionner sur différentes régions françaises

### Classes LOD2 (6 niveaux)

```
LOD2.0 : Blocs simples (pavés)                    [15-20% des bâtiments]
LOD2.1 : Toits simples (1-2 pans)                 [40-45% des bâtiments]
LOD2.2 : Toits complexes (3-4 pans)               [25-30% des bâtiments]
LOD2.3 : Toits très complexes (5+ pans)           [8-10% des bâtiments]
LOD2.4 : Structures spéciales (dômes, courbes)    [2-3% des bâtiments]
LOD2.5 : Structures industrielles complexes       [1-2% des bâtiments]
```

---

## 🏗️ Architecture du Modèle

### Modèle Hybride Proposé

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT POINT CLOUD                         │
│                 (24,576 points, 50m patches)                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
┌─────────────────┐         ┌──────────────────┐
│  PointNet++ SA  │         │ Point Transformer │
│  (Hierarchical) │         │   (Attention)     │
│                 │         │                   │
│  - SA Layer 1   │         │  - Self-Attention │
│  - SA Layer 2   │         │  - Cross-Attention│
│  - SA Layer 3   │         │  - Position Enc.  │
└────────┬────────┘         └─────────┬─────────┘
         │                            │
         └─────────┬──────────────────┘
                   │
                   ▼
         ┌─────────────────┐
         │ Fusion Module   │
         │ (Attention Mix) │
         └────────┬─────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Classification  │
         │   Head (LOD2)   │
         └────────┬─────────┘
                  │
                  ▼
            [6 classes LOD2]
```

### Composants Clés

1. **PointNet++ Branch** (Extraction hiérarchique)

   - Set Abstraction Layer 1: 2048 points → 512 points
   - Set Abstraction Layer 2: 512 points → 128 points
   - Set Abstraction Layer 3: 128 points → 1 global feature
   - Features: Géométrie locale, contexte multi-échelle

2. **Point Transformer Branch** (Attention globale)

   - Self-Attention sur tous les points
   - Positional Encoding 3D
   - Features: Relations long-range, structure globale

3. **Fusion Module** (Attention-based)

   - Cross-attention entre branches
   - Weighted combination
   - Output: Feature vector unifié (1024-dim)

4. **Index Intelligent** (Spatial Indexing)
   - Octree pour accélération
   - KNN graph pré-calculé
   - Ball Query optimisé GPU

---

## 🔬 Stratégie Auto-Supervisée

### Phase 1 : Pré-entraînement Auto-Supervisé (80% des données)

#### Tâches Pretext (sans labels)

**1. Reconstruction de Points**

```python
# Masquer 30% des points et les reconstruire
Task: Masked Point Modeling (MPM)
Loss: Chamfer Distance + Earth Mover's Distance
```

**2. Prédiction de Normales**

```python
# Prédire les normales des surfaces
Task: Normal Vector Prediction
Loss: Cosine Similarity Loss
```

**3. Apprentissage Contrastif**

```python
# Différencier augmentations du même patch vs patches différents
Task: Contrastive Learning (SimCLR style)
Loss: NT-Xent Loss (Temperature-scaled Cross-Entropy)
```

**4. Prédiction de Rotation**

```python
# Prédire l'angle de rotation appliqué au patch
Task: Rotation Prediction (0°, 90°, 180°, 270°)
Loss: Cross-Entropy (4 classes)
```

**5. Prédiction de Contexte Spatial**

```python
# Prédire la position relative de patches voisins
Task: Jigsaw Puzzle (9 patches)
Loss: Cross-Entropy (9! permutations → clustered to 100 classes)
```

#### Configuration Pré-entraînement

```yaml
pretraining:
  epochs: 150
  batch_size: 32
  learning_rate: 0.001
  optimizer: AdamW
  scheduler: CosineAnnealingWarmRestarts

  tasks:
    - name: masked_point_modeling
      weight: 0.3
      mask_ratio: 0.3

    - name: normal_prediction
      weight: 0.2

    - name: contrastive_learning
      weight: 0.3
      temperature: 0.07

    - name: rotation_prediction
      weight: 0.1
      rotations: [0, 90, 180, 270]

    - name: spatial_context
      weight: 0.1
      num_patches: 9
```

### Phase 2 : Fine-tuning Supervisé (20% des données avec labels)

#### Annotation des Labels LOD2

**Option A : Labels Automatiques (Recommandé pour démarrage)**

```python
# Heuristiques géométriques pour labeling initial
def auto_label_lod2(point_cloud):
    """
    Génère des pseudo-labels LOD2 basés sur:
    - Nombre de plans de toit détectés (RANSAC)
    - Complexité géométrique (courbure, angles)
    - Variance des normales
    - Volume englobant
    """
    roof_planes = detect_roof_planes(point_cloud)
    complexity = compute_geometric_complexity(point_cloud)

    if roof_planes == 0:
        return LOD2.0  # Bloc simple
    elif roof_planes <= 2:
        return LOD2.1  # Toit simple
    elif roof_planes <= 4:
        return LOD2.2  # Toit complexe
    # ... etc
```

**Option B : Annotation Semi-Automatique**

```bash
# Utiliser l'outil d'annotation fourni
python scripts/annotate_lod2.py \
  --input data/patches_train \
  --output data/patches_labeled \
  --auto-suggest \
  --confidence-threshold 0.8
```

**Option C : Crowd-sourcing / Expert**

- 5000 patches manuellement annotés par expert (~ 40 heures)
- Répartition équilibrée des classes

#### Configuration Fine-tuning

```yaml
finetuning:
  epochs: 50
  batch_size: 16
  learning_rate: 0.0001 # 10x plus faible que pré-entraînement
  optimizer: AdamW
  scheduler: ReduceLROnPlateau

  loss:
    type: weighted_cross_entropy # Pour gérer le déséquilibre des classes
    weights: [1.0, 1.0, 1.2, 1.5, 2.0, 2.5] # Plus de poids sur classes rares

  data_augmentation:
    - random_rotation: true
    - random_scale: [0.9, 1.1]
    - random_jitter: 0.01
    - point_dropout: 0.05
```

---

## 💾 Estimation des Ressources

### Nombre de Tuiles Nécessaires

#### Calcul pour 50m Patches

```
Hypothèses:
- 1 tuile IGN LiDAR HD = 1km × 1km
- Patch size = 50m × 50m
- Overlap = 15%

Calcul patches par tuile:
- Sans overlap: (1000/50)² = 400 patches/tuile
- Avec overlap 15%: ~400 × 1.35 = 540 patches/tuile
- Avec augmentation (5×): 540 × 6 = 3,240 patches/tuile

Pour dataset complet:
- Training (70%): 70,000 patches → ~22 tuiles
- Validation (15%): 15,000 patches → ~5 tuiles
- Test (15%): 15,000 patches → ~5 tuiles

TOTAL: ~32 tuiles minimum
RECOMMANDÉ: 50-100 tuiles pour diversité géographique
```

#### Distribution Géographique Recommandée

```yaml
dataset_tiles:
  # Phase 1: Pré-entraînement (40 tuiles)
  pretraining:
    urban_dense: 15 tuiles # Paris, Lyon, Marseille
    urban_medium: 10 tuiles # Villes moyennes
    suburban: 10 tuiles # Zones périurbaines
    rural: 5 tuiles # Villages

  # Phase 2: Fine-tuning (10 tuiles)
  finetuning:
    labeled_high_quality: 5 tuiles # Annotations expertes
    labeled_auto: 5 tuiles # Labels automatiques validés


  # Total: 50 tuiles
```

### Stockage et Mémoire

```
Stockage:
- 50m patches avec augmentation: ~4 GB par tuile
- 50 tuiles × 4 GB = 200 GB total
- Checkpoints modèle: ~500 MB par epoch
- Logs et metadata: ~10 GB

Total stockage: ~250 GB

Mémoire GPU (training):
- Batch size 32: ~12 GB VRAM
- Recommandé: RTX 3090 (24GB) ou A100 (40GB)

Mémoire GPU (inference):
- Batch size 1: ~2 GB VRAM
- OK pour RTX 3060 (12GB)
```

### Temps de Calcul Estimé

```
Configuration de référence: NVIDIA RTX 3090 (24GB)

Pré-entraînement:
- 150 epochs × 70,000 patches
- ~30s per epoch
- Total: ~1.25 heures

Fine-tuning:
- 50 epochs × 20,000 patches
- ~5s per epoch
- Total: ~4 minutes

TOTAL ENTRAÎNEMENT: ~1.5 heures

Génération des patches:
- 50 tuiles × 15 min/tuile
- Total: ~12.5 heures

TOTAL PIPELINE: ~14 heures
```

---

## 🔄 Pipeline de Données

### Étape 1 : Préparation des Données (12h)

```bash
# 1. Créer le dataset multi-échelle (50m prioritaire pour LOD2)
ign-lidar-hd process \
  experiment=dataset_50m \
  input_dir=/path/to/ign_tiles \
  output_dir=data/lod2_dataset \
  dataset.train_ratio=0.7 \
  dataset.val_ratio=0.15 \
  dataset.test_ratio=0.15 \
  processor.num_workers=4 \
  processor.use_gpu=true

# Résultat:
# data/lod2_dataset/
#   train/    (~70,000 patches)
#   val/      (~15,000 patches)
#   test/     (~15,000 patches)
#   dataset_metadata.json
```

### Étape 2 : Pré-entraînement Auto-Supervisé (1.25h)

```bash
# Script de pré-entraînement
python train_lod2_selfsupervised.py \
  --mode pretrain \
  --data_dir data/lod2_dataset/train \
  --output_dir models/lod2_pretrained \
  --epochs 150 \
  --batch_size 32 \
  --gpu 0
```

### Étape 3 : Génération des Pseudo-Labels (30min)

```bash
# Labels automatiques basés sur géométrie
python scripts/generate_lod2_labels.py \
  --input data/lod2_dataset/train \
  --output data/lod2_dataset_labeled \
  --method geometric_heuristics \
  --confidence_threshold 0.8
```

### Étape 4 : Fine-tuning Supervisé (4min)

```bash
# Fine-tuning avec pseudo-labels
python train_lod2_selfsupervised.py \
  --mode finetune \
  --pretrained_model models/lod2_pretrained/best_model.pth \
  --data_dir data/lod2_dataset_labeled \
  --output_dir models/lod2_finetuned \
  --epochs 50 \
  --batch_size 16 \
  --gpu 0
```

### Étape 5 : Évaluation (10min)

```bash
# Évaluation sur test set
python evaluate_lod2.py \
  --model models/lod2_finetuned/best_model.pth \
  --data_dir data/lod2_dataset/test \
  --output_dir results/lod2_evaluation
```

---

## 📊 Plan d'Entraînement Détaillé

### Semaine 1 : Préparation des Données

**Jour 1-2 : Collecte et prétraitement**

- [ ] Identifier 50 tuiles IGN représentatives
- [ ] Télécharger les tuiles via l'API IGN
- [ ] Vérifier la qualité des données (densité, couverture)

**Jour 3-4 : Génération des patches**

- [ ] Exécuter le pipeline de génération des patches (12h)
- [ ] Vérifier les statistiques du dataset
- [ ] Visualiser des échantillons de chaque split

**Jour 5 : Validation des données**

- [ ] Analyser la distribution spatiale
- [ ] Vérifier l'absence de leakage entre splits
- [ ] Créer des visualisations de contrôle qualité

### Semaine 2 : Pré-entraînement Auto-Supervisé

**Jour 1 : Configuration**

- [ ] Configurer l'environnement d'entraînement
- [ ] Tester le chargement des données
- [ ] Valider l'architecture du modèle

**Jour 2-3 : Pré-entraînement**

- [ ] Lancer le pré-entraînement (150 epochs)
- [ ] Monitorer les métriques de convergence
- [ ] Sauvegarder les checkpoints

**Jour 4 : Analyse du pré-entraînement**

- [ ] Évaluer la qualité des features apprises
- [ ] Visualiser les embeddings (t-SNE/UMAP)
- [ ] Sélectionner le meilleur checkpoint

**Jour 5 : Génération des pseudo-labels**

- [ ] Appliquer les heuristiques géométriques
- [ ] Valider manuellement un échantillon (100 patches)
- [ ] Analyser la distribution des classes

### Semaine 3 : Fine-tuning et Évaluation

**Jour 1-2 : Fine-tuning**

- [ ] Fine-tuner sur pseudo-labels (50 epochs)
- [ ] Monitorer l'overfitting
- [ ] Early stopping si nécessaire

**Jour 3 : Évaluation quantitative**

- [ ] Calculer les métriques sur test set
- [ ] Analyser la matrice de confusion
- [ ] Identifier les erreurs fréquentes

**Jour 4 : Évaluation qualitative**

- [ ] Visualiser les prédictions sur cas difficiles
- [ ] Tester sur tuiles hors-distribution
- [ ] Analyser les cas d'échec

**Jour 5 : Documentation et rapport**

- [ ] Documenter les résultats
- [ ] Créer des visualisations finales
- [ ] Préparer le rapport technique

### Semaine 4 : Optimisation et Déploiement

**Jour 1-2 : Amélioration du modèle**

- [ ] Annotation manuelle des erreurs majeures
- [ ] Re-fine-tuning avec corrections
- [ ] Optimisation des hyperparamètres

**Jour 3 : Export et optimisation**

- [ ] Exporter le modèle (ONNX, TorchScript)
- [ ] Optimisation pour l'inférence
- [ ] Tests de performance

**Jour 4-5 : Intégration**

- [ ] Intégrer le modèle dans le pipeline IGN
- [ ] Créer l'API d'inférence
- [ ] Tester sur production

---

## 🎯 Métriques de Succès

### Métriques de Pré-entraînement

```python
pretraining_metrics = {
    'masked_point_reconstruction': {
        'chamfer_distance': '< 0.05',  # mètres
        'earth_mover_distance': '< 0.08'
    },
    'normal_prediction': {
        'cosine_similarity': '> 0.85',
        'angle_error': '< 15°'
    },
    'contrastive_learning': {
        'contrastive_accuracy': '> 0.80',
        'embedding_separation': '> 0.70'
    },
    'rotation_prediction': {
        'accuracy': '> 0.90'
    }
}
```

### Métriques de Classification LOD2

```python
classification_metrics = {
    'global': {
        'overall_accuracy': '> 0.75',  # Target minimum
        'macro_f1_score': '> 0.70',
        'weighted_f1_score': '> 0.75'
    },
    'per_class': {
        'LOD2.0': {'precision': '> 0.80', 'recall': '> 0.75'},
        'LOD2.1': {'precision': '> 0.75', 'recall': '> 0.80'},
        'LOD2.2': {'precision': '> 0.70', 'recall': '> 0.70'},
        'LOD2.3': {'precision': '> 0.65', 'recall': '> 0.60'},
        'LOD2.4': {'precision': '> 0.60', 'recall': '> 0.50'},
        'LOD2.5': {'precision': '> 0.55', 'recall': '> 0.45'}
    }
}
```

---

## 🚀 Prochaines Étapes

### Immédiat (Cette Semaine)

1. **Exécuter le script de génération de configuration**

   ```bash
   cd /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET
   python scripts/create_lod2_training_pipeline.py
   ```

2. **Vérifier les ressources disponibles**

   - GPU disponible ? (minimum RTX 3060)
   - Espace disque ? (minimum 250 GB)
   - Tuiles IGN accessibles ?

3. **Commencer la génération des patches**
   ```bash
   ign-lidar-hd process experiment=lod2_selfsupervised \
     input_dir=/path/to/tiles \
     output_dir=data/lod2_training
   ```

### Court Terme (2 Semaines)

1. Implémenter les tâches pretext auto-supervisées
2. Tester le pré-entraînement sur petit subset
3. Valider le pipeline complet end-to-end

### Moyen Terme (1 Mois)

1. Entraînement complet sur 50 tuiles
2. Évaluation et optimisation
3. Documentation des résultats

---

## 📚 Références

- **PointNet++**: Qi et al. "PointNet++: Deep Hierarchical Feature Learning on Point Sets" (2017)
- **Point Transformer**: Zhao et al. "Point Transformer" (2021)
- **Self-Supervised Learning**: He et al. "Momentum Contrast for Unsupervised Visual Representation Learning" (2020)
- **Masked Autoencoders**: Pang et al. "Masked Autoencoders for Point Cloud Self-supervised Learning" (2022)

---

## 📞 Support

Pour toute question sur ce plan d'entraînement :

- Consulter la documentation complète : `docs/ML_DATASET_CREATION.md`
- Vérifier les exemples : `examples/MULTI_SCALE_TRAINING_STRATEGY.md`
- Ouvrir une issue sur GitHub
