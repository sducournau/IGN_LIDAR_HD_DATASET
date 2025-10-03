---
sidebar_position: 1
---

# Installation rapide

## Prérequis

- Python 3.8 ou supérieur
- Gestionnaire de paquets pip

## Installation depuis PyPI

```bash
pip install ign-lidar-hd
```

## Vérifier l'installation

```bash
ign-lidar-process --version
```

## Méthodes d'installation alternatives

### Depuis les sources

```bash
git clone https://github.com/sducournau/IGN_LIDAR_HD_DATASET.git
cd IGN_LIDAR_HD_DATASET
pip install -e .
```

### Avec les dépendances de développement

```bash
pip install -r requirements.txt
pip install -e .
```

## Optionnel : Support GPU

Pour le calcul de caractéristiques accéléré par GPU :

```bash
pip install ign-lidar-hd[gpu]
```

Ou installer les prérequis GPU manuellement :

```bash
pip install -r requirements_gpu.txt
```

**Prérequis GPU :**

- GPU NVIDIA avec support CUDA
- CUDA Toolkit 11.0 ou supérieur
- Paquet cupy-cuda11x

## Configuration de l'environnement

### Utilisation de conda (recommandé)

```bash
conda create -n ign-lidar python=3.9
conda activate ign-lidar
pip install ign-lidar-hd
```

### Utilisation de venv

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows
pip install ign-lidar-hd
```

## Tester l'installation

Tester que tout fonctionne :

```bash
# Vérifier l'accès CLI
python -m ign_lidar.cli --help

# Ou utiliser la commande installée
ign-lidar-process --help
```

Vous devriez voir les commandes disponibles :

- `download` - Télécharger les tuiles IGN LiDAR
- `enrich` - Ajouter des caractéristiques de bâtiment aux fichiers LAZ
- `process` - Extraire des patches depuis les tuiles enrichies

## Prochaines étapes

- Essayer le [Guide d'utilisation de base](../guides/basic-usage.md)
- Explorer les [Commandes CLI](../guides/cli-commands.md)
- En savoir plus sur les [Fonctionnalités de saut intelligent](../features/smart-skip.md)

## Dépannage

### Commande non trouvée

Si la commande `ign-lidar-process` n'est pas trouvée :

```bash
# Utiliser la syntaxe de module Python à la place
python -m ign_lidar.cli --help
```

### Erreurs d'import

Si vous obtenez des erreurs d'import :

```bash
# Réinstaller en mode développement
pip install -e .

# Ou vérifier votre chemin Python
python -c "import sys; print('\n'.join(sys.path))"
```

### Dépendances manquantes

Installer tous les paquets requis :

```bash
pip install -r requirements.txt
pip list  # Vérifier l'installation
```
