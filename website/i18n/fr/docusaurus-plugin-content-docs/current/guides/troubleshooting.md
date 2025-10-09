---
sidebar_position: 8
title: Guide de Dépannage
description: Solutions aux problèmes courants de traitement LiDAR
keywords: [depannage, erreurs, problemes, solutions, aide]
---

# Guide de Dépannage

Solutions aux problèmes courants rencontrés lors du traitement de données LiDAR avec IGN LiDAR HD.

## Problèmes d'Installation

### Échec d'Installation des Dépendances

**Problème** : L'installation du package Python échoue

```bash
ERROR: Failed building wheel for some-package
```

**Solutions** :

1. Mettre à jour pip

   ```bash
   pip install --upgrade pip setuptools wheel
   ```

2. Installer avec conda

   ```bash
   conda install -c conda-forge ign-lidar-hd
   ```

3. Installation manuelle des dépendances
   ```bash
   pip install numpy scipy laspy pdal
   pip install ign-lidar-hd
   ```

### Problèmes CUDA/GPU

**Problème** : CUDA non détecté

```bash
CUDA not available, falling back to CPU processing
```

**Diagnostic** :

```bash
# Vérifier CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**Solutions** :

1. Installer les pilotes NVIDIA
2. Installer CUDA Toolkit
3. Installer PyTorch avec support CUDA

## Problèmes de Traitement

### Erreurs de Mémoire

**Problème** : Mémoire insuffisante

```bash
MemoryError: Unable to allocate array
```

**Solutions** :

1. Réduire la taille des morceaux

   ```python
   processor = Processor(chunk_size=100000)
   ```

2. Traiter en petits lots

   ```python
   for batch in split_large_file(input_file, max_points=500000):
       process_batch(batch)
   ```

3. Utiliser la pagination
   ```python
   processor = Processor(use_pagination=True, page_size=50000)
   ```

### Mémoire GPU Insuffisante

**Problème** : VRAM insuffisante

```bash
CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions** :

1. Limiter la mémoire GPU

   ```python
   processor = Processor(gpu_memory_limit=0.5)
   ```

2. Vider le cache GPU

   ```python
   import torch
   torch.cuda.empty_cache()
   ```

3. Traitement hybride CPU/GPU
   ```python
   processor = Processor(
       use_gpu=True,
       fallback_to_cpu=True
   )
   ```

### Erreurs de Fichier

**Problème** : Fichier LAS corrompu

```bash
LASError: Invalid LAS file header
```

**Diagnostic** :

```bash
# Vérifier avec PDAL
pdal info input.las

# Valider avec laspy
python -c "import laspy; f=laspy.read('input.las'); print('OK')"
```

**Solutions** :

1. Réparer avec PDAL

   ```bash
   pdal translate input.las output_fixed.las --writers.las.forward=all
   ```

2. Valider et nettoyer
   ```python
   from ign_lidar.utils import validate_and_clean
   clean_file = validate_and_clean("input.las")
   ```

## Problèmes de Performance

### Traitement Lent

**Problème** : Très mauvaises performances

**Diagnostic** :

```python
from ign_lidar import Profiler

with Profiler() as prof:
    result = processor.process_tile("input.las")
prof.print_bottlenecks()
```

**Solutions** :

1. Optimiser les paramètres

   ```python
   processor = Processor(
       n_jobs=-1,  # Tous les cœurs
       chunk_size=1000000,
       use_gpu=True
   )
   ```

2. Vérifier la vitesse du disque

   ```bash
   # Tester la vitesse du disque
   dd if=/dev/zero of=test_file bs=1M count=1000
   ```

3. Surveiller les ressources
   ```bash
   htop  # CPU et RAM
   iotop  # E/S disque
   nvidia-smi  # GPU
   ```

### Goulots d'Étranglement E/S

**Problème** : Lecture/écriture lente

**Solutions** :

1. Optimiser les tampons

   ```python
   processor = Processor(
       read_buffer_size='128MB',
       write_buffer_size='128MB',
       io_threads=4
   )
   ```

2. Utiliser un stockage rapide

   - Préférer les SSD NVMe
   - Éviter les lecteurs réseau pour le traitement

3. Compression adaptative
   ```python
   # Équilibrer compression/vitesse
   processor = Processor(
       output_format='laz',
       compression_level=1
   )
   ```

## Problèmes de Configuration

### Configuration Invalide

**Problème** : Erreurs de paramètres

```bash
ConfigurationError: Invalid feature configuration
```

**Solutions** :

1. Valider la configuration

   ```python
   from ign_lidar import Config

   config = Config.from_file("config.yaml")
   config.validate()
   ```

2. Utiliser la configuration par défaut

   ```python
   config = Config.get_default()
   config.features = ['buildings', 'vegetation']
   ```

3. Générer des modèles de configuration
   ```bash
   # Générer un modèle
   ign-lidar-hd config --template > config.yaml
   ```

### Problèmes de Chemin

**Problème** : Fichiers introuvables

```bash
FileNotFoundError: No such file or directory
```

**Solutions** :

1. Vérifier les chemins

   ```python
   import os
   assert os.path.exists("input.las"), "Fichier introuvable"
   ```

2. Utiliser des chemins absolus

   ```python
   input_path = os.path.abspath("input.las")
   ```

3. Vérifier les permissions
   ```bash
   ls -la input.las
   chmod 644 input.las
   ```

## Problèmes Spécifiques

### Augmentation RGB

**Problème** : Échec de l'augmentation couleur

```bash
OrthophotoError: Cannot read orthophoto file
```

**Solutions** :

1. Vérifier le format

   ```bash
   gdalinfo orthophoto.tif
   ```

2. Convertir le format

   ```bash
   gdal_translate -of GTiff input.jp2 output.tif
   ```

3. Vérifier le géoréférencement
   ```python
   from ign_lidar.utils import check_crs_match
   match = check_crs_match("input.las", "orthophoto.tif")
   ```

### Détection de Bâtiments

**Problème** : Mauvaise détection des bâtiments

**Solutions** :

1. Ajuster les paramètres

   ```python
   config = Config(
       building_detection={
           'min_points': 100,
           'height_threshold': 2.0,
           'planarity_threshold': 0.1
       }
   )
   ```

2. Prétraitement adaptatif
   ```python
   processor = Processor(
       ground_classification=True,
       noise_removal=True
   )
   ```

## Journalisation et Débogage

### Activer la Journalisation Détaillée

```python
import logging

logging.basicConfig(level=logging.DEBUG)
processor = Processor(verbose=True)
```

### Enregistrer les Journaux

```python
import logging

logging.basicConfig(
    filename='ign_lidar.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

### Mode Débogage

```bash
# Exécution en mode débogage
IGN_LIDAR_DEBUG=1 python script.py

# Profilage détaillé
IGN_LIDAR_PROFILE=1 python script.py
```

## Support et Aide

### Documentation

- [Guide de Performance](./performance)
- [Guide GPU](./gpu-acceleration)
- [Référence API](../api/features)

### Outils de Diagnostic

```bash
# Informations système
ign-lidar-hd system-info

# Test de configuration
ign-lidar-hd config-test

# Validation des données
ign-lidar-hd validate input.las
```

### Signalement de Bugs

1. Collecter les informations

   ```bash
   ign-lidar-hd system-info > system_info.txt
   ```

2. Exemple minimal

   ```python
   # Code minimal reproduisant le problème
   from ign_lidar import Processor
   processor = Processor()
   # Erreur ici...
   ```

3. Fichiers de test
   - Fournir un petit fichier LAS de test si possible
   - Inclure la configuration utilisée

### Ressources Utiles

- **Dépôt GitHub** : Issues et discussions
- **Documentation** : Guides détaillés et API
- **Exemples** : Scripts d'exemple
- **Communauté** : Forums de discussion

## Solutions Rapides

### Liste de Vérification Générale

1. ✅ Python 3.8+ installé
2. ✅ Dépendances correctement installées
3. ✅ Fichiers d'entrée valides
4. ✅ Permissions de lecture/écriture
5. ✅ Espace disque suffisant
6. ✅ RAM disponible pour les morceaux
7. ✅ Pilotes GPU à jour (si utilisé)

### Commandes Utiles

```bash
# Diagnostics rapides
ign-lidar-hd doctor

# Vider le cache
ign-lidar-hd cache --clear

# Réinitialiser la configuration
ign-lidar-hd config --reset

# Test de performance
ign-lidar-hd benchmark --quick
```
