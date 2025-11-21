# Exemples de Prompts Optimisés pour LiDAR Trainer Agent

> Collection de prompts testés et optimisés pour obtenir les meilleurs résultats avec l'agent

---

## 🎯 Catégories de Prompts

1. [Architecture & Design](#architecture--design)
2. [Feature Engineering](#feature-engineering)
3. [Entraînement & Optimisation](#entraînement--optimisation)
4. [Debugging & Troubleshooting](#debugging--troubleshooting)
5. [Évaluation & Métriques](#évaluation--métriques)
6. [Production & Déploiement](#production--déploiement)

---

## 1. Architecture & Design

### Prompt : Choix d'Architecture

```
@lidarTrainer Mon dataset IGN a [X]M points avec [N] classes
([liste des classes avec distribution]).

Je veux atteindre mIoU > [target] avec un temps d'entraînement
< [X] heures sur [GPU model].

Quelle architecture recommandes-tu entre PointNet++ SSG, MSG,
et Point Transformer ? Justifie ton choix avec des métriques concrètes.
```

**Exemple concret :**

```
@lidarTrainer Mon dataset IGN a 3.4M points avec 3 classes
(Ground 45%, Vegetation 35%, Buildings 20%).

Je veux atteindre mIoU > 0.85 avec un temps d'entraînement
< 3 heures sur RTX 3090.

Quelle architecture recommandes-tu entre PointNet++ SSG, MSG,
et Point Transformer ? Justifie ton choix avec des métriques concrètes.
```

---

### Prompt : Architecture Personnalisée

```
@lidarTrainer Conçois une architecture hybride qui combine :
- PointNet++ pour extraction features locales
- Attention mechanisms pour capturer dépendances long-range
- Multi-scale processing pour gérer densité variable LiDAR aérien

Target : Buildings segmentation avec F1 > 0.90
Dataset : IGN LiDAR HD, 10-15 pts/m², scènes urbaines
```

---

### Prompt : Comparaison Architectures

```
@lidarTrainer Génère un benchmark comparant :
1. PointNet++ SSG
2. PointNet++ MSG
3. Point Transformer
4. KPConv (si pertinent)

Sur mon dataset [path], avec métriques :
- mIoU, F1 per class
- Training time, GPU memory
- Inference speed

Format : Tableau markdown + recommandation argumentée
```

---

## 2. Feature Engineering

### Prompt : Analyse Features Existantes

```
@lidarTrainer Analyse mes features actuelles :
[liste des features : X, Y, Z, R, G, B, ...]

Dataset : [caractéristiques]
Classes cibles : [liste]

Identifie :
1. Features redondantes (forte corrélation)
2. Features peu discriminantes
3. Features manquantes recommandées

Propose une feature selection optimisée.
```

---

### Prompt : Génération Features Géométriques

```
@lidarTrainer Génère le code pour calculer ces features géométriques
sur mon nuage de points :

1. Verticality (priorité haute)
2. Planarity (priorité haute)
3. Omnivariance (priorité moyenne)
4. Normal change rate multi-échelle k=[10,30,50]
5. Height above ground (DTM-based)

Framework : Open3D + NumPy
Optimisation : GPU avec CuPy si possible
Format output : Compatible avec ign_lidar FeatureOrchestrator
```

---

### Prompt : Features pour Cas Spécifique

```
@lidarTrainer Mes prédictions confondent systématiquement :
- Végétation verticale (arbres) ↔ Façades de bâtiments
- Sol ↔ Toits plats

Quelles features géométriques discriminantes recommandes-tu ?
Génère le code de calcul + visualisation pour validation.
```

---

## 3. Entraînement & Optimisation

### Prompt : Setup Complet Entraînement

```
@lidarTrainer Configure un pipeline d'entraînement complet pour :

**Dataset**
- Train : [path], [X]M points
- Val : [path], [Y]M points (distribution différente !)
- Classes : [liste + distribution]

**Target**
- mIoU : > [target]
- F1 buildings : > [target]
- Temps : < [X] heures

**Hardware**
- GPU : [model], [X]GB RAM
- CPU : [cores], [X]GB RAM

Génère :
1. Configuration YAML complète
2. Script d'entraînement Python
3. Stratégie augmentation données
4. Monitoring (TensorBoard)
```

---

### Prompt : Hyperparamètres Tuning

```
@lidarTrainer Optimise les hyperparamètres de mon PointNet++ :

**Contexte**
- Baseline : mIoU=0.78, Gap train/val=15%
- Classes déséquilibrées (voir distribution ci-dessous)
- GPU RAM limitée : 8GB

**À optimiser**
- Learning rate + scheduler
- Batch size (actuellement 16)
- Dropout rate (actuellement 0.3)
- Weight decay
- Loss function (weighted ? focal ?)

Propose 3 configurations (conservative, balanced, aggressive)
avec justification et gains attendus.
```

---

### Prompt : Transfer Learning

```
@lidarTrainer Configure un fine-tuning à partir d'un modèle
pré-entraîné sur ShapeNet :

**Checkpoint**
- Architecture : PointNet++ MSG
- Pré-entraîné sur : ShapeNet Part (50 objets, 16 classes)
- Path : [checkpoint_path]

**Mon dataset**
- Domain : LiDAR aérien IGN
- Classes : Ground, Vegetation, Buildings

**Stratégie**
- Frozen encoder : combien d'epochs ?
- Learning rate : quelle valeur ?
- Unfreeze progressif : comment ?

Génère le code complet avec monitoring du fine-tuning.
```

---

### Prompt : Optimisation GPU

```
@lidarTrainer Mon dataset de [X]M points dépasse la RAM GPU ([Y]GB).

Implémente une stratégie de chunking avec :
- Batch processing intelligent
- Gradient accumulation
- Mixed precision (FP16) si bénéfique
- Optimal chunk size calculé dynamiquement

Maintenir : mIoU > [target]
Framework : PyTorch + RAPIDS cuML
```

---

## 4. Debugging & Troubleshooting

### Prompt : Diagnostic Overfitting

```
@lidarTrainer Debug mon overfitting :

**Symptômes**
- Train accuracy : [X]%
- Val accuracy : [Y]%
- Gap : [Z]%
- Val loss augmente après epoch [N]

**Configuration actuelle**
[coller config YAML]

Diagnostic complet avec :
1. Causes probables (classées par priorité)
2. Solutions concrètes avec code
3. Gains attendus par solution
4. Ordre d'implémentation recommandé
```

---

### Prompt : Analyse Convergence

```
@lidarTrainer Ma loss ne converge pas :

**Comportement observé**
- Loss oscille entre [X] et [Y]
- Pas d'amélioration après [N] epochs
- Gradient norm : [observations]

**Config**
- Learning rate : [lr]
- Optimizer : [optimizer]
- Batch size : [batch_size]

Analyse le problème et propose solutions (avec code).
```

---

### Prompt : Classes Problématiques

```
@lidarTrainer Une classe a des performances catastrophiques :

**Classe problématique : [nom]**
- F1-score : [X] (target : [Y])
- Recall : [X]%
- Precision : [X]%

**Confusion matrix**
[coller matrix ou décrire confusions principales]

Diagnostique :
1. Pourquoi cette classe pose problème ?
2. Features manquantes/insuffisantes ?
3. Stratégies de data augmentation ciblées ?
4. Architecture adaptation ?

Propose un plan d'action chiffré.
```

---

## 5. Évaluation & Métriques

### Prompt : Analyse Complète Résultats

```
@lidarTrainer Analyse mes résultats d'entraînement :

**Métriques finales**
[coller classification report ou tableau métriques]

**Confusion matrix**
[coller ou décrire]

**Objectifs vs Réalisé**
- mIoU target : [X], obtenu : [Y]
- F1 classes : [targets vs obtenus]

Fournis :
1. Analyse détaillée (forces/faiblesses)
2. Comparaison avec state-of-art
3. Recommandations d'amélioration prioritaires
4. Estimation gains attendus
```

---

### Prompt : Générer Rapport Évaluation

```
@lidarTrainer Génère un rapport d'évaluation complet pour :

**Modèle** : [architecture]
**Checkpoint** : [path]
**Datasets** : Train, Val, Test (+ validation externe)

**Contenu rapport**
1. Métriques globales et per-class
2. Confusion matrices
3. Visualisations prédictions (5 samples)
4. Analyse erreurs (cas d'échec typiques)
5. Comparaison avec baseline
6. Recommandations production

Format : Markdown + visualisations PNG
```

---

### Prompt : Benchmark Multi-Datasets

```
@lidarTrainer Évalue mon modèle sur plusieurs datasets pour
tester la généralisation :

**Modèle entraîné sur** : Louhans (IGN)

**Évaluer sur**
1. Manosque (IGN, urbain différent)
2. Paris-Lille-3D (urbain dense)
3. [custom dataset] (rural)

Génère :
- Tableau comparatif métriques
- Analyse des pertes de performance
- Caractérisation du domain gap
- Stratégies pour améliorer généralisation
```

---

## 6. Production & Déploiement

### Prompt : Optimisation Inference

```
@lidarTrainer Optimise mon modèle pour l'inférence production :

**Contraintes**
- Target latency : < [X]ms par tile
- Hardware : [CPU/GPU model]
- Batch inference : [tiles par batch]

**Optimisations à explorer**
1. TorchScript compilation
2. ONNX export + TensorRT
3. Quantization (INT8)
4. Pruning (si gain significatif)

Génère code + benchmark avant/après pour chaque optimisation.
```

---

### Prompt : Pipeline Inference Complet

```
@lidarTrainer Crée un pipeline d'inférence production pour :

**Input** : Tiles LiDAR .las (10-50M points)
**Output** : Tiles classifiées + métriques confidence

**Étapes**
1. Chargement + preprocessing
2. Chunking adaptatif (selon RAM disponible)
3. Inference batch
4. Post-processing (stitching, filtering)
5. Export résultats

**Requis**
- Logging complet
- Gestion erreurs robuste
- Monitoring performances
- API simple (CLI + Python)

Framework : ign_lidar compatible
```

---

### Prompt : Docker Containerization

```
@lidarTrainer Containerise mon modèle entraîné pour déploiement :

**Modèle**
- Architecture : [architecture]
- Checkpoint : [path]
- Dependencies : [requirements.txt]

**Container specs**
- Base image : nvidia/cuda:[version]
- Inference API : FastAPI
- GPU support : CUDA [version]

Génère :
1. Dockerfile optimisé
2. docker-compose.yml
3. API endpoint examples
4. Documentation déploiement
```

---

## 💡 Tips pour Prompts Efficaces

### ✅ Bonnes Pratiques

1. **Contexte précis**

   ```
   ❌ "Entraîne un modèle"
   ✅ "Entraîne PointNet++ MSG sur dataset IGN (3.4M pts, 3 classes)
       avec target mIoU > 0.85"
   ```

2. **Métriques chiffrées**

   ```
   ❌ "Améliore les performances"
   ✅ "Augmente F1 buildings de 0.78 → 0.90"
   ```

3. **Hardware explicite**

   ```
   ❌ "Optimise pour GPU"
   ✅ "Optimise pour RTX 3090 (24GB RAM), batch_size max ?"
   ```

4. **Distribution classes**

   ```
   ❌ "3 classes"
   ✅ "Ground 45%, Vegetation 35%, Buildings 20%"
   ```

5. **Contraintes claires**
   ```
   ❌ "Rapide"
   ✅ "< 2h entraînement, < 100ms inference par tile"
   ```

### ❌ Erreurs à Éviter

1. **Trop vague**

   ```
   ❌ "Aide-moi avec mon modèle"
   ```

2. **Sans contexte**

   ```
   ❌ "Implémente PointNet++"
   (Manque : dataset, classes, target, hardware)
   ```

3. **Multiples demandes non liées**

   ```
   ❌ "Entraîne un modèle ET optimise features ET débugge overfitting"
   (Séparer en 3 prompts)
   ```

4. **Jargon ambigu**
   ```
   ❌ "Rends-le meilleur"
   ✅ "Augmente mIoU de [X] → [Y]"
   ```

---

## 🎓 Templates Réutilisables

### Template : Nouveau Projet

```
@lidarTrainer Je démarre un nouveau projet de classification LiDAR :

**Dataset**
- Source : [IGN / custom / autre]
- Taille : [X]M points
- Format : [.las / .laz / .xyz]
- Classes : [liste avec distribution]

**Objectifs**
- mIoU : > [target]
- F1 per class : [targets]
- Contraintes : [temps / hardware / autre]

**Hardware**
- GPU : [model], [X]GB
- CPU : [cores], [X]GB RAM

Guide-moi pas-à-pas :
1. Feature engineering
2. Choix architecture
3. Configuration entraînement
4. Stratégie validation
```

### Template : Amélioration Existant

```
@lidarTrainer Améliore mon modèle existant :

**Baseline actuelle**
- Architecture : [architecture]
- Métriques : [coller classification report]
- Config : [path ou coller YAML]

**Problèmes identifiés**
1. [problème 1 avec métriques]
2. [problème 2 avec métriques]

**Nouvelles contraintes**
- [contrainte 1]
- [contrainte 2]

Propose un plan d'amélioration avec gains estimés.
```

---

**Dernière mise à jour** : Novembre 2025  
**Maintenu par** : LiDAR Trainer Agent  
**Contributeurs** : Simon Ducournau, communauté IGN LiDAR HD
