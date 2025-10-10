---
sidebar_position: 2
title: Configuration GPU
description: Guide complet pour configurer l'accélération GPU pour le traitement LiDAR
keywords: [gpu, cuda, installation, configuration, nvidia, accélération]
---

# Guide de Configuration GPU

Instructions complètes pour configurer l'accélération GPU afin d'améliorer considérablement les performances de traitement LiDAR.

## Configuration Système Requise

### Configuration Matérielle

**Configuration Minimale :**

- GPU NVIDIA avec CUDA Compute Capability 3.5+
- 4GB VRAM minimum (8GB+ recommandé)
- Slot PCIe 3.0 x16

**Configuration Recommandée :**

- NVIDIA RTX 3060/4060 ou supérieur
- 12GB+ VRAM pour les grands jeux de données
- PCIe 4.0 x16 pour une bande passante maximale

**GPU Supportés :**

```bash
# Vérifier la compatibilité du GPU
nvidia-smi
```

### Configuration Logicielle

- **CUDA Toolkit** : 11.8+ ou 12.x
- **Pilote NVIDIA** : 520.61.05+ (Linux) / 527.41+ (Windows)
- **Python** : 3.8-3.12
- **PyTorch** : 2.0+ avec support CUDA

## Étapes d'Installation

### Étape 1 : Installer les Pilotes NVIDIA

#### Linux (Ubuntu/Debian)

```bash
# Ajouter le dépôt NVIDIA
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Installer le dernier pilote
sudo apt install nvidia-driver-535
sudo reboot

# Vérifier l'installation
nvidia-smi
```

#### Windows

1. Télécharger les derniers pilotes depuis le [site NVIDIA](https://www.nvidia.com/drivers)
2. Exécuter l'installateur avec les paramètres par défaut
3. Redémarrer le système
4. Vérifier avec `nvidia-smi` dans l'Invite de commandes

### Étape 2 : Installer CUDA Toolkit

#### Option A : Installation Conda (Recommandée)

```bash
# Créer un nouvel environnement avec CUDA
conda create -n ign-gpu python=3.11
conda activate ign-gpu

# Installer CUDA toolkit
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

# Vérifier l'installation CUDA
python -c "import torch; print(f'CUDA Disponible: {torch.cuda.is_available()}')"
```

#### Option B : Installation CUDA Native

```bash
# Télécharger CUDA 12.1 depuis NVIDIA
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# Ajouter au PATH
echo 'export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc

# Vérifier l'installation
nvcc --version
```

### Étape 3 : Installer IGN LiDAR HD avec Support GPU

```bash
# Installer avec les dépendances GPU
pip install ign-lidar-hd[gpu]

# Ou installer la version de développement
pip install -e .[gpu]

# Vérifier le support GPU
ign-lidar-hd system-info --gpu
```

## Configuration

### Variables d'Environnement

```bash
# Définir l'allocation de mémoire GPU
export CUDA_VISIBLE_DEVICES=0  # Utiliser le premier GPU
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Optionnel : Limiter l'utilisation de la mémoire GPU
export IGN_GPU_MEMORY_FRACTION=0.8  # Utiliser 80% de la mémoire GPU
```

### Fichier de Configuration GPU

Créer `~/.ign-lidar/gpu-config.yaml`:

```yaml
gpu:
  enabled: true
  device: "cuda:0" # Périphérique GPU à utiliser
  memory_fraction: 0.8 # Fraction de mémoire GPU à utiliser
  batch_size: 10000 # Points par lot GPU

  # Réglage des performances
  pin_memory: true
  non_blocking: true
  compile_models: true # Compilation PyTorch 2.0+

  # Options de repli
  fallback_to_cpu: true
  auto_mixed_precision: true
```

### Configuration Multi-GPU

```yaml
gpu:
  enabled: true
  multi_gpu: true
  devices: ["cuda:0", "cuda:1"] # Plusieurs GPU
  data_parallel: true

  # Équilibrage de charge
  gpu_weights: [1.0, 0.8] # Poids de performance relatifs
```

## Vérification et Tests

### Test GPU de Base

```python
from ign_lidar import Processor
import torch

# Vérifier la disponibilité du GPU
print(f"CUDA Disponible: {torch.cuda.is_available()}")
print(f"Nombre de GPU: {torch.cuda.device_count()}")

# Tester le traitement GPU
processor = Processor(use_gpu=True)
print(f"GPU Activé: {processor.gpu_enabled}")
print(f"Périphérique GPU: {processor.device}")
```

### Test de Performance

```bash
# Exécuter un benchmark GPU
ign-lidar-hd benchmark --gpu --dataset-size large

# Comparer les performances CPU vs GPU
ign-lidar-hd benchmark --compare-devices
```

### Test de Mémoire

```python
from ign_lidar.gpu import GPUMemoryManager

# Vérifier la mémoire GPU
memory_manager = GPUMemoryManager()
memory_info = memory_manager.get_memory_info()

print(f"Mémoire GPU Totale: {memory_info['total']:.2f} GB")
print(f"Mémoire Disponible: {memory_info['available']:.2f} GB")
print(f"Taille de Lot Recommandée: {memory_info['recommended_batch_size']}")
```

## Optimisation des Performances

### Gestion de la Mémoire

```python
from ign_lidar import Config

# Optimiser pour la mémoire GPU disponible
config = Config(
    gpu_enabled=True,
    gpu_memory_fraction=0.8,

    # Configuration du traitement par lots
    gpu_batch_size="auto",  # Détection automatique de la taille optimale
    pin_memory=True,

    # Nettoyage de la mémoire
    clear_cache_every=100,  # Vider le cache GPU tous les N lots
)
```

### Optimisation du Traitement

```python
# Activer la précision mixte pour de meilleures performances
config = Config(
    gpu_enabled=True,
    mixed_precision=True,  # Utiliser FP16 quand possible
    compile_models=True,   # Compilation PyTorch 2.0

    # Optimiser le chargement des données
    num_workers=4,
    prefetch_factor=2,
)
```

## Dépannage

### Problèmes Courants

#### CUDA Non Disponible

```bash
# Vérifier l'installation CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Si False, réinstaller PyTorch avec CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Erreurs de Mémoire Insuffisante

```python
# Réduire la taille du lot
processor = Processor(
    use_gpu=True,
    gpu_batch_size=5000,  # Lot plus petit
    gpu_memory_fraction=0.6  # Utiliser moins de mémoire GPU
)

# Ou utiliser le gradient checkpointing
processor = Processor(
    use_gpu=True,
    gradient_checkpointing=True
)
```

#### Problèmes de Compatibilité des Pilotes

```bash
# Vérifier la version du pilote
nvidia-smi

# Vérifier la compatibilité CUDA
nvidia-smi --query-gpu=compute_cap --format=csv

# Mettre à jour les pilotes si nécessaire
sudo apt update && sudo apt upgrade nvidia-driver-535
```

### Problèmes de Performance

#### Traitement GPU Lent

```python
# Activer les optimisations
import torch
torch.backends.cudnn.benchmark = True  # Optimiser pour des tailles d'entrée cohérentes
torch.backends.cuda.matmul.allow_tf32 = True  # Utiliser TF32 sur les GPU Ampere

# Utiliser des modèles compilés (PyTorch 2.0+)
processor = Processor(
    use_gpu=True,
    compile_models=True
)
```

#### Goulots d'Étranglement des Transferts CPU-GPU

```python
# Optimiser le transfert de données
config = Config(
    gpu_enabled=True,
    pin_memory=True,      # Transferts CPU-GPU plus rapides
    non_blocking=True,    # Transferts asynchrones
    prefetch_to_gpu=True  # Pré-charger les données sur le GPU
)
```

## Surveillance et Profilage

### Surveillance de l'Utilisation GPU

```bash
# Surveiller l'utilisation GPU en temps réel
watch -n 1 nvidia-smi

# Enregistrer l'utilisation GPU dans un fichier
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 1 > gpu_usage.log
```

### Profilage des Performances

```python
from ign_lidar.profiling import GPUProfiler

# Profiler les performances GPU
with GPUProfiler() as profiler:
    result = processor.process_tile("input.las")

# Afficher les résultats du profilage
profiler.print_summary()
profiler.save_report("gpu_profile.html")
```

### Analyse des Goulots d'Étranglement

```python
# Identifier les goulots d'étranglement
from ign_lidar.diagnostics import analyze_gpu_performance

analysis = analyze_gpu_performance(
    las_file="sample.las",
    config=config
)

print(f"Utilisation GPU: {analysis['gpu_utilization']:.1f}%")
print(f"Efficacité Mémoire: {analysis['memory_efficiency']:.1f}%")
print(f"Goulot d'étranglement: {analysis['primary_bottleneck']}")
```

## Configuration Multi-GPU

### Traitement Parallèle des Données

```python
from ign_lidar import MultiGPUProcessor

# Configurer plusieurs GPU
processor = MultiGPUProcessor(
    devices=["cuda:0", "cuda:1"],
    strategy="data_parallel"
)

# Traiter avec plusieurs GPU
results = processor.process_batch(tile_list)
```

### Équilibrage de Charge

```yaml
# Configuration multi-GPU
multi_gpu:
  enabled: true
  devices: ["cuda:0", "cuda:1", "cuda:2"]

  # Équilibrage de charge basé sur les performances GPU
  device_weights:
    "cuda:0": 1.0 # RTX 4090
    "cuda:1": 0.8 # RTX 3080
    "cuda:2": 0.6 # RTX 2080

  # Paramètres de synchronisation
  sync_batch_norm: true
  find_unused_parameters: false
```

## Configuration GPU Cloud

### Google Colab

```python
# Vérifier la disponibilité du GPU dans Colab
import torch
print(f"GPU Disponible: {torch.cuda.is_available()}")
print(f"Nom du GPU: {torch.cuda.get_device_name(0)}")

# Installer IGN LiDAR HD
!pip install ign-lidar-hd[gpu]

# Monter Google Drive pour l'accès aux données
from google.colab import drive
drive.mount('/content/drive')
```

### Instances GPU AWS EC2

```bash
# Lancer une instance GPU (p3.2xlarge recommandé)
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type p3.2xlarge \
  --key-name your-key \
  --security-group-ids sg-your-security-group

# Installer CUDA et IGN LiDAR HD
sudo apt update
sudo apt install nvidia-driver-535
pip install ign-lidar-hd[gpu]
```

## Bonnes Pratiques

### Directives de Développement

1. **Toujours Vérifier la Disponibilité du GPU**

   ```python
   if not torch.cuda.is_available():
       print("GPU non disponible, repli sur CPU")
       use_gpu = False
   ```

2. **Surveiller l'Utilisation de la Mémoire**

   ```python
   # Vider le cache périodiquement
   if batch_count % 100 == 0:
       torch.cuda.empty_cache()
   ```

3. **Utiliser la Précision Mixte**

   ```python
   # Activer la précision mixte automatique
   from torch.cuda.amp import autocast

   with autocast():
       result = model(input_data)
   ```

### Déploiement en Production

1. **Allocation des Ressources**

   - Réserver de la mémoire GPU pour d'autres processus
   - Définir des tailles de lot appropriées
   - Surveiller la température et la consommation d'énergie

2. **Gestion des Erreurs**

   ```python
   try:
       result = gpu_processor.process(data)
   except torch.cuda.OutOfMemoryError:
       torch.cuda.empty_cache()
       result = cpu_processor.process(data)
   ```

3. **Surveillance**
   - Enregistrer l'utilisation du GPU
   - Suivre le débit de traitement
   - Surveiller la limitation thermique

## Documentation Connexe

- [Guide d'Accélération GPU](../guides/gpu-acceleration.md)
- [Optimisation des Performances](../guides/performance.md)
- [Référence API GPU](../api/gpu-api.md)
- [Guide de Dépannage](../guides/troubleshooting.md)
