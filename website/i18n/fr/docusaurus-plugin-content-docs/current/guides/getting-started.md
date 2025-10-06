---
sidebar_position: 1
title: Premiers Pas
description: Guide complet du débutant pour le traitement IGN LiDAR HD
keywords: [démarrage, débutant, tutoriel, premiers-pas, introduction]
---

# Premiers Pas avec IGN LiDAR HD

Bienvenue dans IGN LiDAR HD ! Ce guide complet vous aidera à démarrer avec le traitement des données LiDAR de l'Institut national de l'information géographique et forestière (IGN).

## Qu'est-ce qu'IGN LiDAR HD ?

IGN LiDAR HD est une bibliothèque Python conçue pour traiter les données LiDAR haute densité de l'IGN en jeux de données prêts pour l'apprentissage automatique. Elle fournit des outils pour :

- **Téléchargement de Données** : Téléchargement automatisé des dalles LiDAR IGN
- **Extraction de Caractéristiques** : Détection de bâtiments, classification de végétation, analyse du sol
- **Augmentation RGB** : Enrichissement en couleurs depuis orthophotos
- **Export de Données** : Multiples formats de sortie pour différentes applications
- **Accélération GPU** : Traitement haute performance pour gros jeux de données

## Prérequis

### Configuration Système

**Configuration Minimale :**

- Python 3.8 ou supérieur
- 8 GB RAM
- 10 GB d'espace disque libre
- Connexion Internet pour téléchargement de données

**Configuration Recommandée :**

- Python 3.11
- 16 GB+ RAM
- Stockage SSD avec 50 GB+ d'espace libre
- GPU NVIDIA avec 8 GB+ VRAM (optionnel)

### Environnement Python

Nous recommandons fortement l'utilisation d'un environnement virtuel :

```bash
# Créer environnement virtuel
python -m venv ign_lidar_env

# Activer l'environnement
# Linux/macOS:
source ign_lidar_env/bin/activate
# Windows:
ign_lidar_env\Scripts\activate
```

## Installation

### Installation Standard

```bash
# Installer depuis PyPI
pip install ign-lidar-hd

# Vérifier l'installation
ign-lidar-hd --version
```

### Installation Développement

```bash
# Cloner le dépôt
git clone https://github.com/sducournau/IGN_LIDAR_HD_DATASET.git
cd IGN_LIDAR_HD_DATASET

# Installer en mode développement
pip install -e .

# Installer avec dépendances optionnelles
pip install -e .[gpu,dev,docs]
```

### Support GPU (Optionnel)

Pour l'accélération GPU :

```bash
# Installer avec support GPU
pip install ign-lidar-hd[gpu]

# Vérifier la configuration GPU
python -c "import torch; print(f'CUDA Disponible: {torch.cuda.is_available()}')"
```

## Premiers Pas

### 1. Informations Système

Vérifier votre configuration système :

```bash
# Afficher les informations système
ign-lidar-hd system-info

# Sortie attendue :
# IGN LiDAR HD v1.7.5
# Python: 3.11.5
# Plateforme: Linux-6.2.0-39-generic
# Cœurs CPU: 16
# RAM Disponible: 31.3 GB
# GPU Disponible: True (NVIDIA RTX 4090)
```

### 2. Configuration

Créer votre premier fichier de configuration :

```bash
# Générer configuration par défaut
ign-lidar-hd config --template > ma_config.yaml
```

Éditer la configuration :

```yaml
# ma_config.yaml
processing:
  chunk_size: 1000000
  n_jobs: -1 # Utiliser tous les cœurs CPU
  use_gpu: false # Mettre à true si GPU disponible

output:
  format: "laz" # Format de sortie
  compression: 7

features:
  buildings: true
  vegetation: true
  ground: true

quality:
  validation: true
  generate_reports: true
```

### 3. Votre Premier Téléchargement

Télécharger votre première dalle LiDAR :

```bash
# Télécharger une dalle d'exemple (région parisienne)
ign-lidar-hd download --tiles 0631_6275 --output-dir ./data

# Vérifier les fichiers téléchargés
ls -la ./data/
# Attendu: 0631_6275.las (ou .laz)
```

### 4. Traitement Basique

Traiter la dalle téléchargée :

```bash
# Enrichissement basique
ign-lidar-hd enrich \
  --input ./data/0631_6275.las \
  --output ./data/enriched_0631_6275.laz \
  --features buildings vegetation

# Vérifier les résultats
ign-lidar-hd info ./data/enriched_0631_6275.laz
```

## Comprendre Vos Données

### Structure des Fichiers LiDAR

Les fichiers LiDAR IGN contiennent des données de nuage de points avec ces attributs :

```python
# Attributs basiques des points
attributs_points = {
    'X': 'Coordonnée Est (Lambert 93)',
    'Y': 'Coordonnée Nord (Lambert 93)',
    'Z': 'Élévation (NGF-IGN69)',
    'Intensity': 'Valeur d\'intensité du retour',
    'Return_Number': 'Séquence de retour (1er, 2ème, etc.)',
    'Number_of_Returns': 'Total de retours par impulsion',
    'Classification': 'Code de classification du point',
    'Scanner_Channel': 'ID du canal du scanner',
    'User_Data': 'Données utilisateur additionnelles',
    'Point_Source_ID': 'Identifiant de source',
    'GPS_Time': 'Horodatage GPS'
}

# Après enrichissement, attributs additionnels:
attributs_enrichis = {
    'Building_ID': 'Identifiant d\'instance de bâtiment',
    'Vegetation_Type': 'Classification de végétation',
    'Red': 'Couleur RGB - Canal Rouge',
    'Green': 'Couleur RGB - Canal Vert',
    'Blue': 'Couleur RGB - Canal Bleu',
    'NIR': 'Proche infrarouge',
    'Planarity': 'Planarité (0-1)',
    'Linearity': 'Linéarité (0-1)',
    'Curvature': 'Courbure',
    'Normal_X': 'Normale X',
    'Normal_Y': 'Normale Y',
    'Normal_Z': 'Normale Z'
}
```

### Classes LiDAR Standard

Les classes de points LiDAR IGN suivent la norme ASPRS :

| Code | Description        | Couleur       |
| ---- | ------------------ | ------------- |
| 0    | Non classifié      | Gris          |
| 1    | Non attribué       | Gris clair    |
| 2    | Sol                | Marron        |
| 3    | Végétation basse   | Vert clair    |
| 4    | Végétation moyenne | Vert          |
| 5    | Végétation haute   | Vert foncé    |
| 6    | Bâtiment           | Rouge         |
| 7    | Point bas          | Orange        |
| 9    | Eau                | Bleu          |
| 17   | Pont               | Violet        |

## Workflows Courants

### Workflow 1 : Traitement Basique

Pour un traitement simple avec extraction de caractéristiques :

```bash
# 1. Télécharger les données
ign-lidar-hd download --tiles 0631_6275 --output-dir ./data

# 2. Enrichir avec caractéristiques géométriques
ign-lidar-hd enrich \
  --input-dir ./data \
  --output ./enriched \
  --auto-params \
  --preprocess

# 3. Visualiser dans QGIS
ign-lidar-hd qgis-convert ./enriched/0631_6275.laz
```

### Workflow 2 : Traitement avec RGB

Ajouter de la couleur depuis orthophotos IGN :

```bash
# Enrichir avec couleurs RGB
ign-lidar-hd enrich \
  --input-dir ./data \
  --output ./enriched \
  --auto-params \
  --preprocess \
  --add-rgb \
  --cache-dir ./cache
```

### Workflow 3 : Traitement Multi-Modal Complet

Extraire toutes les caractéristiques (géométrie + RGB + NIR) :

```bash
# Traitement complet avec GPU
ign-lidar-hd enrich \
  --input-dir ./data \
  --output ./enriched \
  --auto-params \
  --preprocess \
  --add-rgb \
  --add-infrared \
  --use-gpu \
  --cache-dir ./cache
```

### Workflow 4 : Traitement par Lot

Traiter plusieurs dalles en parallèle :

```bash
# Télécharger plusieurs dalles
ign-lidar-hd download \
  --region "Île-de-France" \
  --output-dir ./data \
  --max-tiles 10

# Traiter en parallèle (4 workers)
ign-lidar-hd batch-process \
  --input-dir ./data \
  --output ./enriched \
  --n-jobs 4 \
  --auto-params \
  --preprocess \
  --add-rgb
```

## API Python

En plus du CLI, vous pouvez utiliser l'API Python directement :

### Exemple Basique

```python
from ign_lidar import Processor

# Initialiser le processeur
processor = Processor(
    verbose=True,
    use_gpu=False,
    auto_params=True
)

# Traiter un fichier
result = processor.process_tile(
    input_path="data/0631_6275.las",
    output_path="enriched/0631_6275.laz",
    add_rgb=True,
    preprocess=True
)

print(f"Traité {result['points_count']} points")
print(f"Classes détectées: {result['classes_found']}")
```

### Exemple Avancé

```python
from ign_lidar import Processor
from ign_lidar.config import ProcessingConfig

# Configuration personnalisée
config = ProcessingConfig(
    chunk_size=1000000,
    n_neighbors=50,
    search_radius=2.0,
    use_gpu=True,
    gpu_mode='full'  # 'hybrid' ou 'full'
)

# Initialiser avec configuration
processor = Processor(config=config)

# Traiter avec options avancées
result = processor.process_tile(
    input_path="data/large_tile.las",
    output_path="enriched/large_tile.laz",
    add_rgb=True,
    add_infrared=True,
    preprocess=True,
    # Options de prétraitement
    sor_k=20,
    sor_std=2.0,
    voxel_size=0.2
)

# Analyser les résultats
print(f"Statistiques:")
print(f"  Points: {result['points_count']}")
print(f"  Bâtiments: {result['building_count']}")
print(f"  Temps CPU: {result['cpu_time']:.2f}s")
print(f"  Temps GPU: {result['gpu_time']:.2f}s")
```

### Traitement par Lot avec Callbacks

```python
from ign_lidar import BatchProcessor

def progress_callback(tile_name, progress, status):
    print(f"{tile_name}: {progress:.1f}% - {status}")

def error_callback(tile_name, error):
    print(f"ERREUR {tile_name}: {error}")

# Traitement par lot
batch = BatchProcessor(
    n_jobs=4,
    verbose=True,
    on_progress=progress_callback,
    on_error=error_callback
)

# Traiter répertoire
results = batch.process_directory(
    input_dir="data/",
    output_dir="enriched/",
    pattern="*.las",
    add_rgb=True,
    preprocess=True
)

# Résumé
print(f"\nTraité {len(results['success'])} dalles avec succès")
print(f"Échecs: {len(results['failed'])}")
```

## Résolution de Problèmes

### Problèmes Courants

#### 1. Erreur Mémoire Insuffisante

```bash
# Symptôme: MemoryError ou OOMError
# Solution: Réduire chunk_size

ign-lidar-hd enrich --input data.las --output out.laz \
  --chunk-size 500000  # Réduire de 1M à 500k
```

#### 2. GPU Non Détecté

```bash
# Vérifier CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Si False, vérifier les drivers NVIDIA
nvidia-smi

# Réinstaller avec support CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Téléchargement Échoue

```bash
# Vérifier la connexion réseau
ping geoservices.ign.fr

# Utiliser l'option retry
ign-lidar-hd download --tiles 0631_6275 \
  --output-dir ./data \
  --retry 5 \
  --timeout 300
```

#### 4. Fichiers de Sortie Corrompus

```bash
# Valider le fichier de sortie
ign-lidar-hd validate ./enriched/output.laz

# Utiliser le mode sûr
ign-lidar-hd enrich --input data.las --output out.laz \
  --safe-mode  # Validations supplémentaires
```

### Obtenir de l'Aide

Si vous rencontrez des problèmes :

1. **Vérifier les logs** :
   ```bash
   ign-lidar-hd enrich ... --verbose --log-file debug.log
   ```

2. **Activer le mode débogage** :
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. **Signaler un problème** :
   - [Issues GitHub](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
   - Inclure : version, OS, logs, commande utilisée

## Prochaines Étapes

Maintenant que vous avez les bases, explorez :

- 📖 [Guide d'Utilisation Basique](/guides/basic-usage) - Workflows détaillés
- 🚀 [Guide d'Accélération GPU](/guides/gpu-acceleration) - Configuration GPU et optimisation
- 🎨 [Augmentation RGB](/features/rgb-augmentation) - Ajout de couleurs
- 🌿 [Augmentation Infrarouge](/features/infrared-augmentation) - NIR et NDVI
- 🔧 [Référence API](/api/cli) - Documentation complète des commandes

## Ressources Supplémentaires

- 📺 [Tutoriel Vidéo](https://www.youtube.com/watch?v=ksBWEhkVqQI)
- 📚 [Exemples de Code](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/tree/main/examples)
- 🎓 [Tutoriels Avancés](/tutorials/custom-features)
- 💬 [Discussions Communautaires](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/discussions)

---

**Félicitations /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET/website && python3 /tmp/update_fr_intro.py* Vous êtes maintenant prêt à commencer le traitement de données LiDAR avec IGN LiDAR HD. 🎉
