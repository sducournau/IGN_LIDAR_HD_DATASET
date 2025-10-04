---
sidebar_position: 8
title: Guide de dépannage
description: Solutions aux problèmes courants de traitement LiDAR
keywords: [dépannage, erreurs, problèmes, solutions, aide]
---

## Guide de dépannage

Solutions aux problèmes courants rencontrés lors du traitement des données LiDAR avec IGN LiDAR HD.

## Problèmes d'installation

### Erreur d'installation des dépendances

**Problème** : Échec d'installation de packages Python

```bash
ERROR: Failed building wheel for some-package
```

**Solutions** :

1. Mise à jour de pip

   ```bash
   pip install --upgrade pip setuptools wheel
   ```

2. Installation avec conda

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
# Vérification CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**Solutions** :

1. Installation des drivers NVIDIA
2. Installation de CUDA Toolkit
3. Installation de PyTorch avec support CUDA

## Problèmes de traitement

### Erreurs de mémoire

**Problème** : Mémoire insuffisante

```bash
MemoryError: Unable to allocate array
```

**Solutions** :

1. Réduction de la taille des chunks

   ```python
   processor = Processor(chunk_size=100000)
   ```

2. Traitement par petits lots

   ```python
   for batch in split_large_file(input_file, max_points=500000):
       process_batch(batch)
   ```

3. Utilisation de la pagination
   ```python
   processor = Processor(use_pagination=True, page_size=50000)
   ```

### GPU Out of Memory

**Problème** : VRAM insuffisante

```bash
CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions** :

1. Limitation mémoire GPU

   ```python
   processor = Processor(gpu_memory_limit=0.5)
   ```

2. Vidage du cache GPU

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

### Erreurs de fichiers

**Problème** : Fichier LAS corrompu

```bash
LASError: Invalid LAS file header
```

**Diagnostic** :

```bash
# Vérification avec PDAL
pdal info input.las

# Validation avec laspy
python -c "import laspy; f=laspy.read('input.las'); print('OK')"
```

**Solutions** :

1. Réparation avec PDAL

   ```bash
   pdal translate input.las output_fixed.las --writers.las.forward=all
   ```

2. Validation et nettoyage
   ```python
   from ign_lidar.utils import validate_and_clean
   clean_file = validate_and_clean("input.las")
   ```

## Problèmes de performance

### Traitement lent

**Problème** : Performance très faible

**Diagnostic** :

```python
from ign_lidar import Profiler

with Profiler() as prof:
    result = processor.process_tile("input.las")
prof.print_bottlenecks()
```

**Solutions** :

1. Optimisation des paramètres

   ```python
   processor = Processor(
       n_jobs=-1,  # Tous les cœurs
       chunk_size=1000000,
       use_gpu=True
   )
   ```

2. Vérification du stockage

   ```bash
   # Test vitesse disque
   dd if=/dev/zero of=test_file bs=1M count=1000
   ```

3. Surveillance des ressources
   ```bash
   htop  # CPU et RAM
   iotop  # I/O disque
   nvidia-smi  # GPU
   ```

### Goulots d'étranglement I/O

**Problème** : Lecture/écriture lente

**Solutions** :

1. Optimisation des buffers

   ```python
   processor = Processor(
       read_buffer_size='128MB',
       write_buffer_size='128MB',
       io_threads=4
   )
   ```

2. Utilisation de stockage rapide

   - Privilégier les SSD NVMe
   - Éviter les disques réseau pour le traitement

3. Compression adaptée
   ```python
   # Balance compression/vitesse
   processor = Processor(
       output_format='laz',
       compression_level=1
   )
   ```

## Problèmes de configuration

### Configuration invalide

**Problème** : Erreurs de paramètres

```bash
ConfigurationError: Invalid feature configuration
```

**Solutions** :

1. Validation de la configuration

   ```python
   from ign_lidar import Config

   config = Config.from_file("config.yaml")
   config.validate()
   ```

2. Configuration par défaut

   ```python
   config = Config.get_default()
   config.features = ['buildings', 'vegetation']
   ```

3. Templates de configuration
   ```bash
   # Génération d'un template
   ign-lidar-hd config --template > config.yaml
   ```

### Problèmes de chemins

**Problème** : Fichiers non trouvés

```bash
FileNotFoundError: No such file or directory
```

**Solutions** :

1. Vérification des chemins

   ```python
   import os
   assert os.path.exists("input.las"), "Fichier non trouvé"
   ```

2. Chemins absolus

   ```python
   input_path = os.path.abspath("input.las")
   ```

3. Vérification des permissions
   ```bash
   ls -la input.las
   chmod 644 input.las
   ```

## Problèmes spécifiques

### Augmentation RGB

**Problème** : Échec de l'augmentation couleur

```bash
OrthophotoError: Cannot read orthophoto file
```

**Solutions** :

1. Vérification du format

   ```bash
   gdalinfo orthophoto.tif
   ```

2. Conversion de format

   ```bash
   gdal_translate -of GTiff input.jp2 output.tif
   ```

3. Vérification de la géoréférence
   ```python
   from ign_lidar.utils import check_crs_match
   match = check_crs_match("input.las", "orthophoto.tif")
   ```

### Détection de bâtiments

**Problème** : Mauvaise détection des bâtiments

**Solutions** :

1. Ajustement des paramètres

   ```python
   config = Config(
       building_detection={
           'min_points': 100,
           'height_threshold': 2.0,
           'planarity_threshold': 0.1
       }
   )
   ```

2. Préprocessing adapté
   ```python
   processor = Processor(
       ground_classification=True,
       noise_removal=True
   )
   ```

## Logs et débogage

### Activation des logs détaillés

```python
import logging

logging.basicConfig(level=logging.DEBUG)
processor = Processor(verbose=True)
```

### Sauvegarde des logs

```python
import logging

logging.basicConfig(
    filename='ign_lidar.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

### Mode debug

```bash
# Exécution en mode debug
IGN_LIDAR_DEBUG=1 python script.py

# Profiling détaillé
IGN_LIDAR_PROFILE=1 python script.py
```

## Support et aide

### Documentation

- [Guide de performance](./performance.md)
- [Guide GPU](./gpu-acceleration.md)
- [Référence API](../api/features.md)

### Outils de diagnostic

```bash
# Information système
ign-lidar-hd system-info

# Test de configuration
ign-lidar-hd config-test

# Validation des données
ign-lidar-hd validate input.las
```

### Rapporter un bug

1. Collecte d'informations

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

### Ressources utiles

- **Repository GitHub** : Issues et discussions
- **Documentation** : Guides détaillés et API
- **Exemples** : Scripts d'exemple
- **Community** : Forum de discussions

## Solutions rapides

### Checklist générale

1. ✅ Python 3.8+ installé
2. ✅ Dépendances installées correctement
3. ✅ Fichiers d'entrée valides
4. ✅ Permissions de lecture/écriture
5. ✅ Espace disque suffisant
6. ✅ RAM disponible pour les chunks
7. ✅ GPU drivers à jour (si utilisé)

### Commandes utiles

```bash
# Diagnostic rapide
ign-lidar-hd doctor

# Nettoyage du cache
ign-lidar-hd cache --clear

# Reset configuration
ign-lidar-hd config --reset

# Test des performances
ign-lidar-hd benchmark --quick
```
