# LiDAR Trainer Agent - Guide d'Utilisation

> Agent IA spécialisé en Deep Learning pour nuages de points 3D LiDAR

## 🎯 Vue d'Ensemble

Le **LiDAR Trainer Agent** est un assistant IA expert conçu pour accompagner le développement, l'entraînement et l'optimisation de modèles de deep learning sur des nuages de points 3D, spécifiquement pour le projet **IGN LiDAR HD Dataset**.

### Expertise Couverte

- **PointNet++** (SSG, MSG, Set Abstraction, Feature Propagation)
- **Point Transformer** (Self-attention mechanisms)
- **Intelligent Indexing** (KD-Tree, Octree, Ball Query, FPS)
- **Attention Mechanisms** (Multi-head, Cross-attention)
- **Structures hiérarchiques** (Voxel CNN, Sparse Convolutions)

### Basé sur les Travaux de Florent Poux

Cet agent intègre les meilleures pratiques et enseignements des articles de référence de **Florent Poux, Ph.D.** :

- 3D Machine Learning Course (2022)
- 3D Python Workflows for LiDAR City Models (2023)
- Guide to Real-Time Visualization (2021)
- Voxel Modelling Automation (2021)

---

## 🚀 Démarrage Rapide

### Activer l'Agent

Dans GitHub Copilot ou Claude, mentionnez l'agent :

```
@lidarTrainer Je veux entraîner un modèle PointNet++ pour classifier
mes données LiDAR IGN en 3 classes (sol, végétation, bâtiments)
```

### Prérequis

1. **Environnement GPU configuré**

   ```bash
   # L'agent utilisera automatiquement :
   conda activate ign_gpu
   ```

2. **Dataset préparé**

   - Format : `.las`, `.laz`, `.xyz` avec features
   - Classes annotées (Classification ASPRS)
   - Features géométriques calculées (recommandé)

3. **Configuration projet**
   - `ign_lidar/` installé : `pip install -e .`
   - GPU disponible (CUDA 11.x/12.x)

---

## 📖 Exemples d'Utilisation

### Exemple 1 : Entraînement PointNet++ Basique

**Requête utilisateur :**

```
@lidarTrainer Entraîne un PointNet++ SSG pour classifier mon dataset
IGN Louhans (3.4M points, 3 classes). Je veux atteindre mIoU > 0.85.
```

**Réponse de l'agent :**

```
🔍 Analyse du contexte...

✅ Dataset détecté :
  - 3.4M points (Louhans)
  - Classes : Ground (45%), Vegetation (35%), Buildings (20%)
  - Features : XYZ, RGB, 12 geometric features

💡 Architecture recommandée : PointNet++ MSG
   (Multi-Scale Grouping meilleur pour densité variable LiDAR aérien)

📊 Plan d'entraînement :
   1. Prétraitement : Normalisation MinMax + augmentation
   2. Config : batch_size=16, lr=1e-3, epochs=200
   3. Loss : Weighted CrossEntropy (classes déséquilibrées)
   4. Validation : Split 70/30, early stopping patience=20

⏱️ Estimation : ~2h sur RTX 3090 (24GB)

Dois-je générer le code et la configuration ?
```

---

### Exemple 2 : Optimisation Features

**Requête utilisateur :**

```
@lidarTrainer Mes résultats sur bâtiments sont mauvais (F1=0.65).
Quelles features géométriques ajouter pour améliorer la détection des façades ?
```

**Réponse de l'agent :**

```
🔍 Analyse du problème...

⚠️ F1=0.65 sur buildings → Probable confusion avec végétation verticale

💡 Features recommandées pour façades :

1. **Verticality** (priorité haute)
   - Mesure alignement normal avec axe Z
   - Façades = verticality > 0.8

2. **Planarity** (priorité haute)
   - Façades planes vs végétation irrégulière
   - Seuil typique : planarity > 0.7

3. **Normal Change Rate** (priorité moyenne)
   - Détecte discontinuités (fenêtres, balcons)
   - Calcul multi-échelle k=[10, 30, 50]

4. **Height above ground** (priorité moyenne)
   - Discrimine rez-de-chaussée vs arbres

📊 Gains attendus : +10-15% F1 sur buildings

Dois-je générer le code pour calculer ces features ?
```

---

### Exemple 3 : Transfer Learning

**Requête utilisateur :**

```
@lidarTrainer Comment utiliser un modèle pré-entraîné sur ShapeNet
pour accélérer mon entraînement ?
```

**Réponse de l'agent :**

````
🔍 Stratégie de Transfer Learning...

📦 Modèles pré-entraînés disponibles :
   - PointNet++ sur ShapeNet Part (50 objets, 16 classes)
   - Point Transformer sur ModelNet40 (classification)

💡 Approche recommandée : Fine-tuning PointNet++

1. **Chargement weights pré-entraînés**
   ```python
   model = PointNet2SSG(num_classes=3)
   pretrained = torch.load('pointnet2_shapenet.pth')

   # Charger encodeur uniquement (Set Abstraction layers)
   model.load_state_dict(pretrained, strict=False)

   # Geler encodeur, entraîner seulement tête classification
   for param in model.encoder.parameters():
       param.requires_grad = False
````

2. **Configuration fine-tuning**
   - Learning rate : 1e-4 (10x plus faible)
   - Epochs : 50 (au lieu de 200)
   - Unfreeze progressif : epochs 20-50

⏱️ Gains :

- Temps : -60% (2h → 50min)
- Convergence : +15% plus rapide
- Performances : comparables voire meilleures

⚠️ Attention : Si domain gap trop important (objets indoor vs LiDAR aérien),
le fine-tuning peut plafonner. Tester d'abord sur petit subset.

Dois-je implémenter ce pipeline ?

```

---

### Exemple 4 : Debugging Overfitting

**Requête utilisateur :**
```

@lidarTrainer Mon modèle a 97% accuracy sur train mais seulement
68% sur validation. Comment corriger cet overfitting ?

```

**Réponse de l'agent :**
```

🚨 Overfitting détecté : gap 97% → 68%

🔍 Diagnostic :

- Gap > 20% → overfitting sévère
- Probable : modèle mémorise training data

💡 Solutions par ordre de priorité :

1. **Augmentation de données** (impact fort)

   ```python
   # Ajouter dans DataLoader
   - Rotation aléatoire : [-180°, 180°]
   - Jittering : bruit gaussien σ=0.01
   - Dropout points : 10% aléatoire
   - Scaling : [0.8, 1.2]
   ```

2. **Régularisation** (impact moyen)

   ```python
   # Dans architecture
   - Dropout : 0.3 → 0.5 dans MLPs
   - Weight decay : 1e-4 → 1e-3
   - Batch normalization si pas déjà présent
   ```

3. **Early Stopping** (impact faible, préventif)

   ```python
   # Callback
   patience = 20
   monitor = 'val_loss'
   mode = 'min'
   ```

4. **Réduire capacité modèle** (en dernier recours)

   ```python
   # Si solutions 1-3 insuffisantes
   - Diminuer hidden_dims : 256 → 128
   - Réduire num_layers : 4 → 3
   ```

5. **Plus de données** (idéal mais souvent impossible)
   - Utiliser dataset validation pour re-training
   - Synthétiser données (CutMix, MixUp adapté 3D)

📊 Attendu après solutions 1+2 :
Train accuracy : 92% (↓5%)
Val accuracy : 85% (↑17%)
Gap : 7% (acceptable)

Dois-je implémenter ces modifications dans ton code ?

```

---

## 🛠️ Commandes Avancées

### Benchmark de Modèles
```

@lidarTrainer Compare les performances de PointNet++ SSG, MSG et
Point Transformer sur mon dataset. Génère un tableau comparatif.

```

### Optimisation GPU
```

@lidarTrainer Mon dataset de 50M points ne tient pas en GPU RAM.
Implémente une stratégie de chunking pour l'entraînement.

```

### Analyse d'Erreurs
```

@lidarTrainer Génère une matrice de confusion et identifie les
classes les plus confondues. Propose des features pour les discriminer.

```

### Export Production
```

@lidarTrainer Optimise mon modèle entraîné pour l'inférence
(TorchScript, quantization). Target : <100ms par tile sur CPU.

```

---

## 📊 Métriques de Succès

L'agent vise toujours :

### Performances
- **mIoU ≥ 0.85** (métrique cible principale)
- **F1-score ≥ 0.90** par classe majoritaire
- **Recall ≥ 0.80** par classe minoritaire

### Généralisation
- **Gap Train/Val < 10%** (éviter overfitting)
- **Validation sur distribution différente** (ex: Louhans → Manosque)

### Efficacité
- **Temps entraînement** : communiqué clairement
- **GPU memory** : optimisé pour hardware disponible
- **Inference speed** : <1s par tile sur GPU

---

## 🎓 Base de Connaissances

Toutes les connaissances de l'agent sont documentées dans :
- **`lidarTrainer.agent.md`** : Définition complète de l'agent
- **`KNOWLEDGE_BASE.md`** : Synthèse techniques Florent Poux
- **`.github/articles/`** : Articles sources complets

### Concepts Clés Maîtrisés

**Architectures**
- PointNet, PointNet++, Point Transformer, KPConv
- Set Abstraction, Feature Propagation, Multi-Scale Grouping
- Attention mechanisms, Transformer blocks

**Feature Engineering**
- PCA local, eigenvalues décomposition
- Geometric descriptors (planarity, verticality, curvature)
- Multi-scale features

**Optimisation**
- GPU strategies (full, chunked, hybrid)
- Data augmentation 3D
- Transfer learning, fine-tuning
- Regularization techniques

**Évaluation**
- IoU, mIoU, F1-score, Precision, Recall
- Confusion matrix analysis
- Validation strategy (3 datasets)

---

## 🔄 Workflow Type

```

┌─────────────────────────────────────────────────┐
│ 1. ANALYSE CONTEXTE │
│ - Dataset properties │
│ - Hardware available │
│ - Performance targets │
└─────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────┐
│ 2. PROPOSITION ARCHITECTURE │
│ - Justification technique │
│ - Alternatives comparison │
│ - Estimation resources │
└─────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────┐
│ 3. IMPLÉMENTATION │
│ - Model code │
│ - Training script │
│ - Tests │
└─────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────┐
│ 4. ENTRAÎNEMENT │
│ - Hyperparameters config │
│ - Launch training (conda run -n ign_gpu) │
│ - Monitor metrics │
└─────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────┐
│ 5. ÉVALUATION & ITÉRATION │
│ - Validation metrics │
│ - Error analysis │
│ - Improvement recommendations │
└─────────────────────────────────────────────────┘

```

---

## 🤝 Communication

### Style de l'Agent
- **Technique mais pédagogique** : Explications claires des concepts complexes
- **Proactif** : Propose des améliorations même non demandées
- **Justifié** : Références scientifiques (articles Florent Poux)
- **Concret** : Code exécutable accompagnant chaque proposition

### Questions Systématiques
L'agent demandera toujours :
```

❓ Dataset : path, taille, features disponibles ?
❓ Classes cibles : nombre, distribution ?
❓ Métrique cible : mIoU, F1-score minimum ?
❓ Hardware : GPU disponible ? RAM ?
❓ Budget temps : entraînement acceptable ?

```

### Signalement Proactif
```

⚠️ Classes déséquilibrées détectées → Weighted loss recommandé
⚠️ Features non normalisées → MinMaxScaler nécessaire  
⚠️ GPU non utilisé → conda run -n ign_gpu recommandé
✅ Configuration optimale détectée
💡 Amélioration possible suggérée

```

---

## 🔧 Intégration Projet

### Structure de Code Générée
```

ign_lidar/
├── models/ # ← Agent crée ici
│ ├── **init**.py
│ ├── base_model.py
│ ├── pointnet2.py # PointNet++ implementation
│ └── point_transformer.py # Point Transformer
│
├── datasets/ # ← Agent crée ici
│ ├── **init**.py
│ └── lidar_dataset.py # PyTorch Dataset
│
└── training/ # ← Agent crée ici
├── **init**.py
├── trainer.py # Training loop
└── evaluator.py # Evaluation metrics

scripts/
├── train_pointnet2.py # ← Agent crée ici
└── evaluate_model.py # ← Agent crée ici

configs/
└── model_config.yaml # ← Agent crée ici

tests/
└── test_models/ # ← Agent crée ici
├── test_pointnet2.py
└── test_dataset.py

````

### Respect des Conventions Projet
- **PEP 8** compliance (88 chars, Black formatter)
- **Type hints** complets (Python 3.8+)
- **Google-style docstrings**
- **Tests unitaires** systématiques
- **Configuration Hydra** pour hyperparamètres

---

## 📚 Ressources Complémentaires

### Documentation
- [IGN LiDAR HD Docs](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/)
- [PointNet++ Paper](https://arxiv.org/abs/1706.02413)
- [Point Transformer Paper](https://arxiv.org/abs/2012.09164)
- [Florent Poux Tutorials](https://learngeodata.eu)

### Formations
- [3D Geodata Academy](https://learngeodata.eu)
- [Point Cloud Processing Course](https://learngeodata.eu)

### Outils
- [Open3D](http://www.open3d.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [RAPIDS cuML](https://docs.rapids.ai/api/cuml/stable/)

---

## 🐛 Troubleshooting

### L'agent ne démarre pas
```bash
# Vérifier que le fichier agent existe
ls .github/agents/lidarTrainer.agent.md

# Vérifier syntaxe TOON
head -20 .github/agents/lidarTrainer.agent.md
````

### GPU non détecté

```bash
# Vérifier environnement
conda activate ign_gpu
python -c "import torch; print(torch.cuda.is_available())"

# Si False :
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Erreurs d'import

```bash
# Réinstaller projet
cd /path/to/IGN_LIDAR_HD_DATASET
pip install -e .

# Vérifier installations
pip list | grep -E "torch|open3d|cupy"
```

---

## 📞 Support

Pour toute question ou amélioration :

1. Ouvrir une issue sur GitHub
2. Consulter la [documentation complète](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/)
3. Référencer `lidarTrainer.agent.md` et `KNOWLEDGE_BASE.md`

---

**Version** : 1.0  
**Dernière mise à jour** : Novembre 2025  
**Maintenu par** : Simon Ducournau  
**Basé sur les travaux de** : Florent Poux, Ph.D.
