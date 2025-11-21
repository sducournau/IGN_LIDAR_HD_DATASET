# Base de Connaissances - Deep Learning pour Nuages de Points 3D

> Synthèse des articles de Florent Poux pour le LiDAR Trainer Agent

## 📚 Sources

Cette base de connaissances compile les enseignements de **23 articles** de Florent Poux, Ph.D. :

### Articles Fondamentaux Deep Learning

1. **PointNet++ pour Segmentation Sémantique 3D** - Pipeline complet, modèles pré-entraînés vs personnalisés
2. **3D Machine Learning Course** - Segmentation sémantique de nuages de points avec Python
3. **3D Deep Learning Python Tutorial** - Préparation de données pour PointNet
4. **Towards 3D Deep Learning** - Réseaux de neurones artificiels avec Python

### Workflows & Processing

5. **3D Python Workflows for LiDAR City Models** - Guide pratique étape par étape
6. **How to Automate LiDAR Point Cloud Processing** - Sous-échantillonnage et optimisation
7. **How to Automate Voxel Modelling** - Modélisation par voxels 3D avec Python
8. **Guide de visualisation en temps réel** - Nuages de points massifs en Python

### Clustering & Segmentation

9. **3D Clustering with Graph Theory** - Segmentation euclidienne avec théorie des graphes
10. **Fundamentals to Clustering High-Dimensional Data** - Clustering non-supervisé pour nuages 3D
11. **3D Point Cloud Clustering Tutorial** - K-means et Python pour nuages de points

### Reconstruction & Modelling

12. **3D Reconstruction Tutorial** - Avec Python et Meshroom
13. **5-Step Guide to Generate 3D Meshes** - De nuages de points à maillages
14. **Transform Point Clouds into 3D Meshes** - Guide Python complet

### Advanced Techniques

15. **Segment Anything 3D (SAM 3D)** - Guide complet pour segmentation sémantique 3D
16. **Build 3D Scene Graphs for Spatial AI LLMs** - OpenUSD et intégration LLM
17. **Smart 3D Change Detection** - Détection de changements dans nuages de points
18. **How to Create 3D Models from Any Image** - Reconstruction 3D zéro-shot avec IA

### Multi-View & Rendering

19. **Multi-View 3D Renderer** - Python, Blender, 3D Gaussian Splatting automatisé
20. **How to Represent 3D Data** - Fondamentaux de représentation 3D

### Spatial Data & Integration

21. **3D Spatial Data Integration** - Intégration de données spatiales avec Python
22. **3D Scanning: Complete Sensor Guide** - Guide complet des capteurs
23. **Methods and Hardware Tools for 3D Scanning** - Outils matériels et méthodes

---

## 🏗️ Architecture PointNet++

### Principes Fondamentaux

#### Innovation Clé

PointNet++ introduit une **architecture hiérarchique** qui reproduit pour les nuages de points la philosophie des CNN pour les images.

#### Composants Principaux

1. **Set Abstraction Layers**

   ```
   Échantillonnage (FPS) → Regroupement (Ball Query) → Extraction features (mini-PointNet)
   ```

   - **Farthest Point Sampling (FPS)** : Sous-échantillonnage intelligent
   - **Ball Query** : Recherche de voisins dans un rayon R
   - **Mini-PointNet** : Extraction de features locales

2. **Feature Propagation Layers**

   ```
   Interpolation (3-NN) → Remontée à l'échelle originale
   ```

   - Interpolation pondérée (distance inverse)
   - Skip connections (style U-Net)
   - Concilie granularité locale et contexte global

3. **Variantes**
   - **SSG (Single-Scale Grouping)** : Un seul rayon de recherche
   - **MSG (Multi-Scale Grouping)** : Plusieurs rayons → meilleur pour densité variable

### Comparaison Modèles

| Critère          | PointNet | PointNet++        | Point Transformer |
| ---------------- | -------- | ----------------- | ----------------- |
| Structure locale | ❌       | ✅ (hiérarchique) | ✅✅ (attention)  |
| Multi-échelle    | ❌       | ✅ (MSG)          | ✅✅ (multi-head) |
| Complexité       | O(n)     | O(n log n)        | O(n²)             |
| Performances     | Bonnes   | Très bonnes       | Excellentes       |
| Vitesse GPU      | Rapide   | Moyen             | Lent              |

---

## 🔧 Pipeline Complet ML 3D

### Phase 1 : Prétraitement

#### 1.1 Nettoyage & Filtrage

```python
# Suppression outliers
- Radius outlier removal (PCL, Open3D)
- Statistical outlier removal (mean-K)
- Filtrage des artefacts capteurs

# Sous-échantillonnage
- Voxel grid (1 pt par voxel)
- Random sampling
- Farthest Point Sampling (FPS) ⭐ recommandé
```

#### 1.2 Normalisation

```python
# Recentrage et mise à l'échelle
points_centered = points - points.mean(axis=0)
points_normalized = points_centered / np.max(np.linalg.norm(points_centered, axis=1))

# MinMaxScaler (scikit-learn)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
```

#### 1.3 Feature Engineering Géométrique

**Features Essentielles (Florent Poux)**

```python
# Analyse PCA locale (k-NN ou rayon R)
eigenvalues = λ1 >= λ2 >= λ3  # Décomposition covariance locale

# Descripteurs invariants
- Normales : vecteur propre principal
- Courbure : λ3 / (λ1 + λ2 + λ3)
- Planarity : (λ2 - λ3) / λ1
- Linearity : (λ1 - λ2) / λ1
- Sphericity : λ3 / λ1
- Omnivariance : (λ1 * λ2 * λ3)^(1/3)
- Verticality : 1 - |normal_z|
- Normal change rate : variation normales voisines
```

**Implémentation Optimisée**

```python
import open3d as o3d

# Calcul normales
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
)

# Calcul courbature (custom)
def compute_curvature(pcd, k=30):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    curvatures = []
    for i in range(len(pcd.points)):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], k)
        # PCA local + λ3/(λ1+λ2+λ3)
        ...
    return np.array(curvatures)
```

#### 1.4 Augmentation de Données

**Stratégies Recommandées (Florent Poux)**

```python
# Rotations aléatoires (Euler angles)
rotation_angles = np.random.uniform(-np.pi, np.pi, 3)
R = rotation_matrix_from_euler(rotation_angles)
points_augmented = points @ R.T

# Translations aléatoires
translation = np.random.uniform(-1, 1, 3)
points_augmented = points + translation

# Scaling aléatoire
scale = np.random.uniform(0.8, 1.2)
points_augmented = points * scale

# Bruit gaussien
noise = np.random.normal(0, 0.01, points.shape)
points_augmented = points + noise

# Dropout de points
keep_ratio = 0.9
indices = np.random.choice(len(points), int(len(points)*keep_ratio))
points_augmented = points[indices]
```

---

### Phase 2 : Entraînement

#### 2.1 Configuration Hyperparamètres

**Recommandations Florent Poux**

```yaml
# Learning rate
initial_lr: 1e-3 # ou 1e-4 pour fine-tuning
scheduler:
  type: "StepLR"
  step_size: 20 # epochs
  gamma: 0.7

# Batch size
batch_size: 16 # ou 32 si GPU puissant
num_workers: 4 # DataLoader

# Optimizer
optimizer: "Adam" # ou SGD avec momentum 0.9
weight_decay: 1e-4

# Epochs
max_epochs: 200
early_stopping_patience: 20
```

#### 2.2 Loss Functions

**Pour Classification Déséquilibrée**

```python
import torch.nn as nn

# Weighted CrossEntropy
class_weights = compute_class_weights(train_labels)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Focal Loss (pour classes rares)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss
```

#### 2.3 Régularisation

```python
# Dropout (dans MLP)
dropout_rate = 0.3  # à 0.5

# Batch Normalization
use_batch_norm = True

# L2 regularization
weight_decay = 1e-4  # dans optimizer

# Early stopping
monitor = "val_miou"
patience = 20
```

---

### Phase 3 : Évaluation

#### 3.1 Métriques

**Métriques Essentielles (Florent Poux)**

```python
from sklearn.metrics import classification_report, confusion_matrix

# Accuracy globale
accuracy = (TP + TN) / (TP + TN + FP + FN)

# Par classe
precision = TP / (TP + FP)  # "Classifier is precise"
recall = TP / (TP + FN)     # "Finds all positives"
f1_score = 2 * (precision * recall) / (precision + recall)

# IoU (Intersection over Union)
iou = TP / (TP + FP + FN)

# Mean IoU (métrique vedette)
miou = np.mean([iou_class_i for i in classes])
```

**Implémentation**

```python
def compute_iou(y_true, y_pred, num_classes):
    ious = []
    for cls in range(num_classes):
        tp = ((y_true == cls) & (y_pred == cls)).sum()
        fp = ((y_true != cls) & (y_pred == cls)).sum()
        fn = ((y_true == cls) & (y_pred != cls)).sum()

        iou = tp / (tp + fp + fn + 1e-10)
        ious.append(iou)

    return np.array(ious), np.mean(ious)

# Classification report
print(classification_report(
    y_test,
    y_pred,
    target_names=['ground', 'vegetation', 'buildings']
))
```

#### 3.2 Validation Croisée

**Stratégie Florent Poux : 3 Datasets**

```python
# 1. Training Data (60-70%)
#    → Ajustement des poids

# 2. Test Data (20-30%)
#    → Évaluation et tuning hyperparamètres
#    ⚠️ Devient biaisé car utilisé pour tuning

# 3. Validation Data (dataset complètement différent)
#    → Évaluation finale, test de généralisation
#    ✅ Ex: Train sur Louhans, Val sur Manosque
```

---

## 🎯 Feature Modes IGN LiDAR HD

### Configuration Projet

```python
from ign_lidar.features import FeatureMode

# MINIMAL (~8 features)
# - Ultra-rapide, prototypage
# - XYZ + RGB + classification

# LOD2 (~12 features)  ⭐ Recommandé bâtiments
# - Essential geometric features
# - Normales, planarity, verticality, omnivariance

# LOD3 (~38 features)
# - Description géométrique complète
# - Tous descripteurs PCA locaux
# - Multi-scale features

# ASPRS_CLASSES (~25 features)
# - Classification standard ASPRS
# - Ground, vegetation, buildings, water, etc.

# FULL
# - Toutes features disponibles
# - Maximum de contexte, mais risque overfitting

# CUSTOM
# - Sélection manuelle
```

---

## 🚀 Optimisation GPU

### Stratégies (IGN LiDAR HD)

#### Strategy 1 : CPU (Baseline)

```python
# Standard scikit-learn
from sklearn.ensemble import RandomForestClassifier

# Utilisation : Datasets < 10M points
# Avantage : Stable, pas de dépendance GPU
# Inconvénient : Lent sur gros volumes
```

#### Strategy 2 : GPU Full

```python
# CuPy + RAPIDS cuML
import cupy as cp
from cuml.ensemble import RandomForestClassifier as cuRF

# Utilisation : Dataset entier tient en GPU RAM
# Avantage : x10-100 plus rapide
# Inconvénient : Limite RAM GPU

# ⚠️ TOUJOURS utiliser : conda run -n ign_gpu
```

#### Strategy 3 : GPU Chunked ⭐

```python
# Batch processing pour gros datasets
def train_gpu_chunked(X, y, chunk_size=1_000_000):
    model = cuRF()

    for i in range(0, len(X), chunk_size):
        X_chunk = cp.asarray(X[i:i+chunk_size])
        y_chunk = cp.asarray(y[i:i+chunk_size])

        if i == 0:
            model.fit(X_chunk, y_chunk)
        else:
            model.partial_fit(X_chunk, y_chunk)

    return model

# Utilisation : Datasets > GPU RAM
# Avantage : Traite volumes illimités
# Inconvénient : Plus lent que GPU full
```

### Sélection Automatique

```python
from ign_lidar.features.mode_selector import select_compute_mode

mode = select_compute_mode(
    num_points=len(points),
    gpu_available=True,
    gpu_memory_gb=16
)
# → Retourne "CPU", "GPU", ou "GPU_CHUNKED"
```

---

## 📊 Cas d'Usage Réels

### Exemple 1 : Classification Forestière

**Contexte** : Landes, Sologne, Vosges
**Classes** : Sol, Troncs, Bois mort, Canopée
**Features clés** : Verticality, Linearity, Height above ground

```python
# Pipeline type
features = ['Z', 'verticality', 'linearity', 'omnivariance',
            'normal_change_rate', 'height_above_ground']

# Modèle : Random Forest (interprétable)
# Accuracy : 92% (Sol), 87% (Canopée), 78% (Troncs)
```

### Exemple 2 : Cartographie Urbaine (IGN)

**Contexte** : Ville de Louhans + Manosque (validation)
**Classes** : Ground, Vegetation, Buildings
**Features clés** : RGB, Planarity, Verticality, Height

```python
# Pipeline type
features = ['Z', 'R', 'G', 'B', 'omnivariance_2',
            'normal_cr_2', 'NumberOfReturns', 'planarity_2',
            'omnivariance_1', 'verticality_1']

# Modèle : PointNet++ MSG
# Test accuracy : 97% (Louhans)
# Val accuracy : 85% (Manosque) → Bonne généralisation
```

### Exemple 3 : Détection Temps Réel (Voiture Autonome)

**Contexte** : Dataset KITTI
**Classes** : Road, Sidewalk, Vehicles, Pedestrians
**Contrainte** : > 10 FPS sur GPU

```python
# Modèle : PointNet++ léger (SSG)
# Optimisation :
- Voxel downsampling à 10 cm
- Feature extraction GPU (CUDA ops)
- Batch inference
- TensorRT optimization

# Performance : 15 FPS @ 1024x768, mIoU 82%
```

---

## 🛠️ Outils Python Essentiels

### Open3D (Visualisation + Processing)

```python
import open3d as o3d

# Chargement
pcd = o3d.io.read_point_cloud("data.pcd")

# Visualisation
o3d.visualization.draw_geometries([pcd])

# Normales
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
)

# Voxelisation
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
    pcd, voxel_size=0.5
)

# Export
o3d.io.write_point_cloud("output.ply", pcd)
```

### PPTK (Visualisation Massive)

```python
import pptk

# Visualisation 10-100M points
v = pptk.viewer(points)
v.attributes(colors/65535)

# Calcul features
normals = pptk.estimate_normals(points, k=30, r=np.inf)

# Sélection interactive
selection = v.get('selected')
selected_points = points[selection]
```

### LasPy (LiDAR I/O)

```python
import laspy as lp

# Lecture .las/.laz
point_cloud = lp.read("data.las")

# Accès attributs
x, y, z = point_cloud.x, point_cloud.y, point_cloud.z
classification = point_cloud.classification
rgb = np.vstack([point_cloud.red, point_cloud.green, point_cloud.blue]).T

# Écriture
header = lp.LasHeader(version="1.4", point_format=6)
las = lp.LasData(header)
las.x, las.y, las.z = x, y, z
las.classification = predicted_labels
las.write("output.las")
```

---

## 📈 Benchmarks & Performances

### Dataset IGN LiDAR HD (Louhans)

| Modèle         | Features   | Train Time | Val mIoU | Ground F1 | Vegetation F1 | Buildings F1 |
| -------------- | ---------- | ---------- | -------- | --------- | ------------- | ------------ |
| Random Forest  | XYZ+RGB    | 2 min      | 0.54     | 0.25      | 0.70          | 0.53         |
| Random Forest  | +Geometric | 5 min      | **0.85** | 0.85      | 0.92          | 0.73         |
| K-NN           | +Geometric | 15 min     | 0.91     | 0.91      | 0.90          | 0.92         |
| MLP            | +Geometric | 30 min     | 0.64     | 0.69      | 0.71          | 0.20         |
| PointNet++ SSG | +Geometric | 2h         | 0.96     | 0.98      | 0.97          | 0.93         |
| PointNet++ MSG | +Geometric | 3h         | **0.97** | 0.98      | 0.97          | 0.96         |

**Enseignements Clés (Florent Poux)**

1. Features géométriques : +31% mIoU sur Random Forest
2. PointNet++ : Meilleur pour bâtiments complexes
3. Transfer learning : -50% temps avec pré-entraînement ShapeNet
4. Validation différente distribution : -12% mIoU (importance généralisation)

---

## ⚠️ Pièges Courants

### 1. GPU + Multiprocessing ❌

```python
# NE PAS FAIRE
DataLoader(dataset, num_workers=8)  # Avec GPU
# → CUDA context conflicts

# FAIRE
DataLoader(dataset, num_workers=0)  # ou 1
```

### 2. Oublier la Normalisation

```python
# NE PAS FAIRE
model.fit(X_train, y_train)

# FAIRE
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model.fit(X_train_scaled, y_train)
```

### 3. Overfitting sur Test Set

```python
# PIÈGE : Tuner sur test → biais
for params in hyperparameters:
    model = Model(params)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)  # ❌

# SOLUTION : Dataset validation séparé
# Train (60%) → Test (20%) → Validation (20% autre distribution)
```

### 4. Features aux Frontières

```python
# Problème : Features calculées sur voisins
# → Artefacts aux bords de tiles

# Solution : Stitching ou overlap tiles
tile_overlap = 10  # mètres
# Ou post-process : TileStitcher dans ign_lidar
```

---

## 📖 Références Complètes

### Papers Fondamentaux

1. **Qi, C. R., et al. (2017)** - "PointNet++: Deep Hierarchical Feature Learning on Point Sets"
2. **Zhao, H., et al. (2021)** - "Point Transformer"
3. **Thomas, H., et al. (2019)** - "KPConv: Flexible and Deformable Convolution for Point Clouds"

### Tutorials Florent Poux

1. "3D Machine Learning 201 Guide: Point Cloud Semantic Segmentation" (2022)
2. "3D Python Workflows for LiDAR City Models: A Step-by-Step Guide" (2023)
3. "Guide to real-time visualization of massive 3D point clouds in Python" (2021)
4. "How to Automate Voxel Modelling of 3D Point Cloud with Python" (2021)

### Ressources Complémentaires

- **3D Geodata Academy** : learngeodata.eu
- **IGN LiDAR HD** : geoservices.ign.fr
- **Open3D Documentation** : open3d.org
- **PyTorch Geometric** : pytorch-geometric.readthedocs.io

---

**Dernière mise à jour** : Novembre 2025
**Maintenu par** : LiDAR Trainer Agent
**Basé sur les travaux de** : Florent Poux, Ph.D.
