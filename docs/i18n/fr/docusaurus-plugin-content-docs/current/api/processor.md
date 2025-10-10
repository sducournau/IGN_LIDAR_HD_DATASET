---
sidebar_position: 1
---

# API Processor

Référence API complète pour le module de traitement IGN LiDAR HD.

## Classe Processor

La classe principale pour gérer les flux de traitement des données LiDAR.

### Constructeur

```python
from ign_lidar import Processor

processor = Processor(config=None, verbose=True)
```

**Paramètres:**

- `config` (Config, optionnel): Objet de configuration avec les paramètres de traitement
- `verbose` (bool): Active la sortie de journalisation détaillée

### Méthodes

#### process_tile()

Traite une seule tuile LAS/LAZ avec extraction de caractéristiques.

```python
def process_tile(self, input_path: str, output_path: str = None) -> dict:
```

**Paramètres:**

- `input_path` (str): Chemin vers le fichier LAS/LAZ d'entrée
- `output_path` (str, optionnel): Chemin pour le fichier de sortie

**Retourne:**

- `dict`: Résultats de traitement avec métriques et statut

**Exemple:**

```python
result = processor.process_tile("input.las", "output_enriched.las")
print(f"Traité {result['points_count']} points")
```

#### process_batch()

Traite plusieurs fichiers LiDAR en lot.

```python
def process_batch(self, input_dir: str, output_dir: str, pattern: str = "*.las") -> list:
```

**Paramètres:**

- `input_dir` (str): Répertoire contenant les fichiers d'entrée
- `output_dir` (str): Répertoire pour les fichiers traités
- `pattern` (str): Motif de fichier à traiter

**Retourne:**

- `list`: Liste des résultats de traitement pour chaque fichier

#### set_config()

Met à jour la configuration du processeur.

```python
def set_config(self, config: Config) -> None:
```

## Configuration

### Classe Config

```python
from ign_lidar import Config

config = Config(
    features=['buildings', 'vegetation', 'ground'],
    chunk_size=1000000,
    use_gpu=True
)
```

**Paramètres disponibles:**

- `features` (list): Liste des caractéristiques à extraire
- `chunk_size` (int): Taille des chunks pour le traitement
- `use_gpu` (bool): Utiliser l'accélération GPU si disponible
- `output_format` (str): Format de sortie ('las', 'laz', 'ply')

## Gestion d'erreurs

Le processeur lève plusieurs types d'exceptions:

- `ProcessingError`: Erreurs générales de traitement
- `ConfigurationError`: Erreurs de configuration
- `IOError`: Erreurs de lecture/écriture de fichiers

## Exemple complet

```python
from ign_lidar import Processor, Config

# Configuration
config = Config(
    features=['buildings', 'vegetation'],
    chunk_size=500000,
    use_gpu=True
)

# Initialisation
processor = Processor(config=config, verbose=True)

# Traitement d'un fichier unique
try:
    result = processor.process_tile("input.las", "output.las")
    print(f"Succès: {result['points_count']} points traités")
except Exception as e:
    print(f"Erreur: {e}")

# Traitement en lot
results = processor.process_batch("input/", "output/")
for result in results:
    print(f"Fichier: {result['filename']}, Points: {result['points_count']}")
```
