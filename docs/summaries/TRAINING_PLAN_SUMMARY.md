# Résumé du Plan d'Entraînement LOD2 - Configuration Actuelle

**Date:** 15 Octobre 2025  
**Projet:** Classification LOD2 Auto-Supervisée  
**Architecture:** Hybrid PointNet++ + Point Transformer

---

## 📋 Ce Qui A Été Créé

### 1. Configuration d'Expérience

**Fichier:** `ign_lidar/configs/experiment/lod2_selfsupervised.yaml`

Configuration optimisée pour :

- Patches 50m (24,576 points)
- Splits train/val/test : 70/15/15
- Architecture hybride
- Augmentation 5×
- Features complètes (RGB, NIR, géométriques)

### 2. Script d'Entraînement

**Fichier:** `scripts/train_lod2_selfsupervised.py`

Implémente :

- **Phase 1** : Pré-entraînement auto-supervisé (150 epochs)
  - Masked Point Modeling
  - Rotation Prediction
  - Contrastive Learning
- **Phase 2** : Fine-tuning supervisé (50 epochs)
  - Classification 6 classes LOD2
  - Weighted Cross-Entropy

### 3. Documentation Complète

#### Plan Détaillé

**Fichier:** `docs/TRAINING_PLAN_LOD2_SELF_SUPERVISED.md`

- Architecture complète
- Stratégie auto-supervisée
- Estimation des ressources
- Planning 4 semaines

#### Guide Rapide

**Fichier:** `docs/LOD2_TRAINING_QUICK_START.md`

- Commandes TL;DR
- Configuration rapide
- Dépannage

---

## 🎯 Votre Configuration Actuelle

### Analyse de Votre Setup

D'après vos configurations existantes (`config_lod3_training_*.yaml`), vous avez :

```yaml
Configuration actuelle détectée:
  - LOD Level: LOD3 (peut être adapté à LOD2)
  - Architecture: hybrid ✅
  - Patch sizes: 50m, 100m, 150m ✅
  - GPU: Activé ✅
  - Features: Full (RGB, NIR, géométriques) ✅
  - Augmentation: 3-5× ✅
```

### Adaptations Recommandées pour LOD2

```yaml
# Changements pour LOD2:
processor:
  lod_level: LOD2 # ← Changé de LOD3 à LOD2
  patch_size: 50.0 # ✅ Optimal pour bâtiments individuels
  num_points: 24576 # ✅ Déjà bon
  num_augmentations: 5 # ✅ Déjà optimal

features:
  k_neighbors: 20 # ✅ Bon pour détails fins
  compute_curvature: true # ← Ajouter pour LOD2
  compute_eigenfeatures: true # ← Ajouter pour complexité toit
```

---

## 🚀 Plan d'Action Immédiat

### Option A : Test Rapide (Recommandé pour démarrer)

```bash
# 1. Tester sur 5 tuiles (2h de génération)
ign-lidar-hd process \
  experiment=lod2_selfsupervised \
  input_dir=/path/to/5_test_tiles \
  output_dir=data/lod2_test \
  processor.num_workers=4

# 2. Pré-entraînement court (30 epochs, ~15min)
python scripts/train_lod2_selfsupervised.py \
  --mode pretrain \
  --data_dir data/lod2_test/train \
  --output_dir models/lod2_test \
  --epochs 30 \
  --batch_size 32 \
  --gpu 0

# 3. Vérifier que tout fonctionne
ls -lh models/lod2_test/
```

**Résultat attendu :**

- Dataset : ~5,000 patches (5 tuiles × 540 patches/tuile × 6 avec aug)
- Temps total : ~2.5 heures
- Permet de valider le pipeline complet

### Option B : Production Complète

```bash
# 1. Génération dataset complet (50 tuiles, 12h)
ign-lidar-hd process \
  experiment=lod2_selfsupervised \
  input_dir=/path/to/ign_tiles_50 \
  output_dir=data/lod2_production

# 2. Pré-entraînement (150 epochs, 1.25h)
python scripts/train_lod2_selfsupervised.py \
  --mode pretrain \
  --data_dir data/lod2_production/train \
  --output_dir models/lod2_pretrained \
  --epochs 150 \
  --batch_size 32 \
  --gpu 0 \
  --use_wandb \
  --project_name lod2-production

# 3. Fine-tuning (50 epochs, 4min)
python scripts/train_lod2_selfsupervised.py \
  --mode finetune \
  --pretrained_model models/lod2_pretrained/best_model.pth \
  --data_dir data/lod2_production/train \
  --val_dir data/lod2_production/val \
  --output_dir models/lod2_finetuned \
  --epochs 50 \
  --batch_size 16 \
  --gpu 0 \
  --use_wandb
```

**Résultat attendu :**

- Dataset : ~100,000 patches
- Temps total : ~14 heures
- Modèle production-ready

---

## 📊 Nombre de Tuiles Recommandé

### Calcul pour Votre Cas

```python
# Paramètres
patch_size = 50  # mètres
tile_size = 1000  # mètres (1km×1km)
overlap = 0.15
augmentation = 5

# Calcul patches par tuile
patches_per_tile_base = (1000 / 50) ** 2  # 400
patches_with_overlap = patches_per_tile_base * 1.35  # ~540
patches_with_aug = patches_with_overlap * (1 + augmentation)  # ~3,240

# Pour dataset complet
target_patches = 100000
tiles_needed = target_patches / patches_with_aug
# ≈ 31 tuiles minimum

print(f"Tuiles nécessaires : {tiles_needed:.0f}")
print(f"Recommandé : 50-100 tuiles")
```

### Recommandations par Objectif

| Objectif                | Tuiles | Patches | Temps | Use Case          |
| ----------------------- | ------ | ------- | ----- | ----------------- |
| **Test Pipeline**       | 5      | ~16K    | 2.5h  | Validation rapide |
| **Proof of Concept**    | 15-20  | ~50K    | 6h    | Démo faisabilité  |
| **Production Minimum**  | 30-50  | ~100K   | 14h   | Modèle utilisable |
| **Production Optimale** | 80-100 | ~300K   | 36h   | Haute performance |
| **Recherche**           | 150+   | ~500K+  | 60h+  | SOTA performance  |

---

## 🎓 Classes LOD2 et Distribution

### Définition des 6 Classes

```python
LOD2_CLASSES = {
    0: {
        'name': 'LOD2.0',
        'description': 'Blocs simples (pavés)',
        'features': [
            'Aucun toit détectable',
            'Forme rectangulaire simple',
            'Faible variance des normales'
        ],
        'distribution': '15-20%'
    },
    1: {
        'name': 'LOD2.1',
        'description': 'Toits simples (1-2 pans)',
        'features': [
            '1-2 plans de toit',
            'Arête faîtière claire',
            'Géométrie simple'
        ],
        'distribution': '40-45%'  # Classe majoritaire
    },
    2: {
        'name': 'LOD2.2',
        'description': 'Toits complexes (3-4 pans)',
        'features': [
            '3-4 plans de toit',
            'Arêtes multiples',
            'Géométrie modérée'
        ],
        'distribution': '25-30%'
    },
    3: {
        'name': 'LOD2.3',
        'description': 'Toits très complexes (5+ pans)',
        'features': [
            '5+ plans de toit',
            'Intersections complexes',
            'Haute variance géométrique'
        ],
        'distribution': '8-10%'
    },
    4: {
        'name': 'LOD2.4',
        'description': 'Structures spéciales (dômes, courbes)',
        'features': [
            'Surfaces courbes',
            'Dômes, voûtes',
            'Haute courbure'
        ],
        'distribution': '2-3%'  # Classe rare
    },
    5: {
        'name': 'LOD2.5',
        'description': 'Structures industrielles complexes',
        'features': [
            'Géométrie irrégulière',
            'Structures métalliques',
            'Multi-niveaux'
        ],
        'distribution': '1-2%'  # Classe très rare
    }
}
```

### Gestion du Déséquilibre de Classes

Le script d'entraînement inclut déjà :

```python
# Poids pour cross-entropy
class_weights = torch.tensor([
    1.0,  # LOD2.0 (commun)
    1.0,  # LOD2.1 (très commun - classe majoritaire)
    1.2,  # LOD2.2 (moins commun)
    1.5,  # LOD2.3 (rare)
    2.0,  # LOD2.4 (très rare)
    2.5   # LOD2.5 (extrêmement rare)
])
```

---

## 💾 Ressources GPU Requises

### Configuration Minimale

```yaml
GPU: NVIDIA RTX 3060 (12GB)
Adjustments:
  batch_size: 16
  num_points: 16384
  num_workers: 2
Temps: +50% par rapport à RTX 3090
```

### Configuration Recommandée

```yaml
GPU: NVIDIA RTX 3090 (24GB) ou A100 (40GB)
Settings:
  batch_size: 32
  num_points: 24576
  num_workers: 4
Temps: Baseline (comme documenté)
```

### Configuration Optimale

```yaml
GPU: 2× NVIDIA A100 (80GB)
Settings:
  batch_size: 64
  num_points: 32768
  num_workers: 8
  distributed: true
Temps: -60% par rapport à RTX 3090
```

---

## 📈 Métriques de Succès Attendues

### Phase 1 : Pré-entraînement (après 150 epochs)

```yaml
Masked Point Reconstruction:
  chamfer_distance: < 0.05m
  earth_mover_distance: < 0.08m

Rotation Prediction:
  accuracy: > 90%
  confidence: High on all 4 rotations

Contrastive Learning:
  contrastive_accuracy: > 80%
  embedding_quality: Good separation in t-SNE
```

### Phase 2 : Fine-tuning (après 50 epochs)

```yaml
Global Metrics:
  overall_accuracy: > 75%
  macro_f1_score: > 70%
  weighted_f1_score: > 75%

Per-Class Performance:
  LOD2.0: {precision: >0.80, recall: >0.75}
  LOD2.1: {precision: >0.75, recall: >0.80}  # Classe majoritaire
  LOD2.2: {precision: >0.70, recall: >0.70}
  LOD2.3: {precision: >0.65, recall: >0.60}
  LOD2.4: {precision: >0.60, recall: >0.50}  # Classe rare
  LOD2.5: {precision: >0.55, recall: >0.45}  # Classe très rare
```

---

## 🔍 Checklist Avant de Démarrer

### Prérequis Système

- [ ] GPU NVIDIA avec CUDA 11.0+ installé
- [ ] Python 3.8+ avec PyTorch 1.12+
- [ ] Espace disque : 250 GB disponible
- [ ] RAM : 32 GB minimum
- [ ] Package IGN LiDAR HD installé : `pip install -e .`

### Prérequis Données

- [ ] Accès aux tuiles IGN LiDAR HD
- [ ] Tuiles téléchargées localement (ou accès API)
- [ ] Vérification qualité : densité > 10 points/m²
- [ ] Distribution géographique : urban + suburban + rural

### Prérequis Logiciels

```bash
# Vérifier les dépendances
pip install torch torchvision torchaudio  # PyTorch
pip install numpy scipy scikit-learn      # ML utilities
pip install tqdm wandb                    # Training utilities
pip install laspy pdal                    # Point cloud processing
```

---

## 🚦 Prochaines Étapes

### Aujourd'hui

1. **Vérifier** que tous les fichiers sont créés :

   ```bash
   ls -l ign_lidar/configs/experiment/lod2_selfsupervised.yaml
   ls -l scripts/train_lod2_selfsupervised.py
   ls -l docs/TRAINING_PLAN_LOD2_SELF_SUPERVISED.md
   ls -l docs/LOD2_TRAINING_QUICK_START.md
   ```

2. **Installer** les dépendances manquantes :

   ```bash
   pip install -e .
   pip install torch wandb tqdm
   ```

3. **Tester** sur 1 tuile :
   ```bash
   # Test unitaire
   ign-lidar-hd process \
     experiment=lod2_selfsupervised \
     input_dir=/path/to/single_tile \
     output_dir=data/test_single
   ```

### Cette Semaine

1. **Lancer** test sur 5 tuiles (Option A)
2. **Valider** que le pipeline fonctionne
3. **Analyser** les résultats du pré-entraînement

### Prochaines 2 Semaines

1. **Déployer** sur 50 tuiles (Option B)
2. **Entraîner** le modèle complet
3. **Évaluer** les performances

### Mois Prochain

1. **Optimiser** hyperparamètres
2. **Affiner** avec annotations expertes
3. **Déployer** en production

---

## 📞 Support et Documentation

### Documentation Créée

1. **Plan complet** : `docs/TRAINING_PLAN_LOD2_SELF_SUPERVISED.md`

   - Architecture détaillée
   - Stratégie auto-supervisée
   - Timeline 4 semaines

2. **Guide rapide** : `docs/LOD2_TRAINING_QUICK_START.md`

   - TL;DR commandes
   - Dépannage
   - Configuration GPU

3. **Configuration** : `ign_lidar/configs/experiment/lod2_selfsupervised.yaml`

   - Prêt à l'emploi
   - Paramètres optimisés

4. **Script entraînement** : `scripts/train_lod2_selfsupervised.py`
   - Pré-entraînement auto-supervisé
   - Fine-tuning supervisé
   - Checkpointing automatique

### Ressources Existantes

- Dataset ML : `docs/ML_DATASET_CREATION.md`
- Multi-échelle : `examples/MULTI_SCALE_TRAINING_STRATEGY.md`
- Configuration : `examples/README.md`

---

## 🎉 Résumé

Vous avez maintenant :

- ✅ Configuration complète pour LOD2 auto-supervisé
- ✅ Script d'entraînement hybride PointNet++ + Transformer
- ✅ Documentation détaillée (plan + guide rapide)
- ✅ Estimation précise des ressources (32-50 tuiles, ~14h)
- ✅ Plan d'action étape par étape

**Prêt à commencer !** 🚀

Lancez le test rapide (Option A) pour valider le pipeline :

```bash
ign-lidar-hd process \
  experiment=lod2_selfsupervised \
  input_dir=/path/to/5_test_tiles \
  output_dir=data/lod2_test
```

Bonne chance avec votre entraînement ! 💪
