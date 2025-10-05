---
sidebar_position: 3
title: API CLI
description: Référence API de l'interface en ligne de commande et intégration
keywords: [cli, api, ligne-de-commande, intégration, automatisation]
---

# Référence API CLI

Documentation complète de l'API pour l'intégration avec l'interface en ligne de commande IGN LiDAR HD.

## Module CLI

### CommandLineInterface

Classe d'interface CLI principale pour l'exécution programmatique des commandes.

```python
from ign_lidar.cli import CommandLineInterface

cli = CommandLineInterface(
    verbose=True,
    log_file="ign_lidar.log",
    config_file="config.yaml"
)
```

#### Paramètres

- **verbose** (`bool`) : Activer la sortie détaillée
- **log_file** (`str`) : Chemin vers le fichier de log
- **config_file** (`str`) : Fichier de configuration par défaut
- **working_directory** (`str`) : Répertoire de travail pour les commandes

#### Méthodes

##### `run_command(command, args=None, **kwargs)`

Exécuter des commandes CLI de manière programmatique.

```python
# Commande de téléchargement
result = cli.run_command(
    command="download",
    args={
        "tiles": ["0631_6275", "0631_6276"],
        "output_dir": "/data/lidar",
        "format": "laz"
    }
)

# Vérifier le résultat de l'exécution
if result.success:
    print(f"Téléchargé {len(result.files)} fichiers")
    print(f"Taille totale : {result.total_size_mb:.1f} MB")
else:
    print(f"Erreur : {result.error_message}")
```

##### `enrich_batch(input_files, **kwargs)`

Traitement d'enrichissement par lots.

```python
# Enrichir plusieurs fichiers
results = cli.enrich_batch(
    input_files=["tile1.las", "tile2.las", "tile3.las"],
    output_dir="/data/enriched",
    features=["buildings", "vegetation"],
    rgb_source="/data/orthophotos",
    parallel=True,
    max_workers=4
)

# Traiter les résultats
for file_result in results:
    if file_result.success:
        print(f"✅ {file_result.input_file} -> {file_result.output_file}")
    else:
        print(f"❌ {file_result.input_file} : {file_result.error}")
```

### Classes de Commandes

#### DownloadCommand

```python
from ign_lidar.cli.commands import DownloadCommand

download_cmd = DownloadCommand()

# Configurer les paramètres de téléchargement
download_result = download_cmd.execute(
    tiles=["0631_6275"],
    output_directory="/data/raw",
    file_format="laz",
    overwrite=False,
    verify_checksum=True
)

# Accéder aux informations de téléchargement
print(f"Fichiers téléchargés : {download_result.file_count}")
print(f"Temps de téléchargement : {download_result.duration:.2f}s")
print(f"Vitesse moyenne : {download_result.speed_mbps:.1f} MB/s")
```

#### EnrichCommand

```python
from ign_lidar.cli.commands import EnrichCommand

enrich_cmd = EnrichCommand()

# Exécuter l'enrichissement
enrich_result = enrich_cmd.execute(
    input_file="input.las",
    output_file="output.laz",
    config_file="enrich_config.yaml",
    features=["buildings", "vegetation", "ground"],
    rgb_orthophoto="orthophoto.tif",
    overwrite=False
)

# Vérifier les résultats de l'enrichissement
print(f"Points traités : {enrich_result.points_processed:,}")
print(f"Fonctionnalités ajoutées : {enrich_result.features_added}")
print(f"Temps de traitement : {enrich_result.processing_time:.2f}s")
```

#### PatchCommand

```python
from ign_lidar.cli.commands import PatchCommand

patch_cmd = PatchCommand()

# Appliquer les correctifs de prétraitement
patch_result = patch_cmd.execute(
    input_file="raw.las",
    output_file="patched.las",
    patch_types=["noise_removal", "ground_classification"],
    patch_config="patch_config.yaml"
)
```

## API de Configuration

### CLIConfig

```python
from ign_lidar.cli.config import CLIConfig

# Charger la configuration CLI
config = CLIConfig.from_file("cli_config.yaml")

# Configuration programmatique
config = CLIConfig(
    default_output_format="laz",
    parallel_processing=True,
    max_workers=8,
    chunk_size=1000000,

    # Configuration de logging
    log_level="INFO",
    log_format="%(asctime)s - %(levelname)s - %(message)s",

    # Paramètres de performance
    gpu_acceleration=True,
    memory_limit="8GB"
)

# Appliquer la configuration au CLI
cli = CommandLineInterface(config=config)
```

### Configuration d'Environnement

```python
from ign_lidar.cli.config import setup_environment

# Configurer les variables d'environnement
env_config = setup_environment(
    data_directory="/data/lidar",
    cache_directory="/tmp/ign_cache",
    temp_directory="/tmp/ign_temp",
    gpu_device="cuda:0"
)

# Variables d'environnement définies :
# IGN_DATA_DIR=/data/lidar
# IGN_CACHE_DIR=/tmp/ign_cache
# IGN_TEMP_DIR=/tmp/ign_temp
# CUDA_VISIBLE_DEVICES=0
```

## Suivi de la Progression

### ProgressTracker

```python
from ign_lidar.cli.progress import ProgressTracker, ProgressCallback

class CustomProgressCallback(ProgressCallback):
    def on_start(self, total_items):
        print(f"Début du traitement de {total_items} éléments...")

    def on_progress(self, completed, total, current_item):
        percent = (completed / total) * 100
        print(f"Progression : {percent:.1f}% - {current_item}")

    def on_complete(self, total_items, elapsed_time):
        print(f"Terminé {total_items} éléments en {elapsed_time:.2f}s")

# Utiliser le suivi personnalisé de la progression
tracker = ProgressTracker(callback=CustomProgressCallback())

# Traiter avec suivi de la progression
cli = CommandLineInterface(progress_tracker=tracker)
result = cli.enrich_batch(input_files)
```

### Surveillance en Temps Réel

```python
from ign_lidar.cli.monitoring import ProcessMonitor

# Surveiller les processus CLI
monitor = ProcessMonitor()

# Démarrer la surveillance
monitor.start()

# Exécuter la commande CLI
result = cli.run_command("enrich", args={"input": "large_file.las"})

# Obtenir les données de surveillance
stats = monitor.get_stats()
print(f"Utilisation CPU : {stats.cpu_percent:.1f}%")
print(f"Utilisation Mémoire : {stats.memory_mb:.1f} MB")
print(f"E/S Disque : {stats.disk_io_mbps:.1f} MB/s")

monitor.stop()
```

## Gestion des Erreurs

### Classes d'Exception CLI

```python
from ign_lidar.cli.exceptions import (
    CLIError,
    CommandNotFoundError,
    InvalidArgumentError,
    ProcessingError,
    FileNotFoundError
)

try:
    result = cli.run_command("enrich", args={"invalid_arg": True})
except InvalidArgumentError as e:
    print(f"Argument invalide : {e.argument} - {e.message}")
except FileNotFoundError as e:
    print(f"Fichier non trouvé : {e.filepath}")
except ProcessingError as e:
    print(f"Échec du traitement : {e.details}")
```

### Nouvelle Tentative et Récupération

```python
from ign_lidar.cli.retry import RetryManager

# Configurer la logique de nouvelle tentative
retry_manager = RetryManager(
    max_attempts=3,
    backoff_factor=2.0,
    retry_on=[ProcessingError, IOError],
    exclude=[InvalidArgumentError]
)

# Exécuter avec nouvelle tentative
def process_with_retry():
    return cli.run_command("enrich", args=enrich_args)

result = retry_manager.execute(process_with_retry)
```

## Validation et Tests

### CommandValidator

```python
from ign_lidar.cli.validation import CommandValidator

validator = CommandValidator()

# Valider les arguments de commande
validation_result = validator.validate_command(
    command="enrich",
    args={
        "input": "input.las",
        "output": "output.laz",
        "features": ["buildings", "invalid_feature"]
    }
)

if not validation_result.is_valid:
    for error in validation_result.errors:
        print(f"Erreur de validation : {error}")
```

### Framework de Test CLI

```python
from ign_lidar.cli.testing import CLITestRunner

# Tester les commandes CLI
test_runner = CLITestRunner()

# Créer un cas de test
test_case = {
    "command": "enrich",
    "args": {
        "input": "test_data/sample.las",
        "output": "test_output/enriched.laz"
    },
    "expected_exit_code": 0,
    "expected_files": ["test_output/enriched.laz"],
    "timeout": 60
}

# Exécuter le test
test_result = test_runner.run_test(test_case)

if test_result.passed:
    print("✅ Test réussi")
else:
    print(f"❌ Test échoué : {test_result.error_message}")
```

## Modèles d'Intégration

### Intégration de Workflow

```python
from ign_lidar.cli.workflows import WorkflowRunner

# Définir le workflow de traitement
workflow = {
    "name": "Pipeline de Traitement Complet",
    "steps": [
        {
            "command": "download",
            "args": {"tiles": ["0631_6275"], "output_dir": "data/raw"}
        },
        {
            "command": "patch",
            "args": {"input": "data/raw/0631_6275.las", "output": "data/patched/0631_6275.las"}
        },
        {
            "command": "enrich",
            "args": {
                "input": "data/patched/0631_6275.las",
                "output": "data/enriched/0631_6275.laz",
                "features": ["buildings", "vegetation"]
            }
        }
    ]
}

# Exécuter le workflow
runner = WorkflowRunner(cli=cli)
workflow_result = runner.execute_workflow(workflow)

# Vérifier les résultats
for step_result in workflow_result.step_results:
    print(f"Étape {step_result.step_name} : {'✅' if step_result.success else '❌'}")
```

### Traitement par Lots

```python
from ign_lidar.cli.batch import BatchProcessor

# Configurer le traitement par lots
batch_processor = BatchProcessor(
    cli=cli,
    max_parallel=4,
    retry_failed=True,
    progress_tracking=True
)

# Définir le travail par lots
batch_job = {
    "command": "enrich",
    "input_pattern": "data/raw/*.las",
    "output_pattern": "data/enriched/{basename}.laz",
    "common_args": {
        "features": ["buildings", "vegetation"],
        "rgb_source": "data/orthophotos"
    }
}

# Exécuter le travail par lots
batch_result = batch_processor.execute_batch(batch_job)

# Résumé
print(f"Fichiers totaux : {batch_result.total_files}")
print(f"Réussis : {batch_result.successful_files}")
print(f"Échoués : {batch_result.failed_files}")
print(f"Temps total : {batch_result.total_time:.2f}s")
```

### Intégration de Systèmes Externes

#### Intégration de Base de Données

```python
from ign_lidar.cli.database import DatabaseIntegration

# Configurer la connexion à la base de données
db_integration = DatabaseIntegration(
    connection_string="postgresql://user:pass@localhost/lidar_db",
    table_name="processing_jobs"
)

# Enregistrer l'exécution CLI dans la base de données
def process_with_db_logging(command, args):
    job_id = db_integration.create_job(command, args)

    try:
        result = cli.run_command(command, args)
        db_integration.update_job_success(job_id, result)
        return result
    except Exception as e:
        db_integration.update_job_failure(job_id, str(e))
        raise

# Utilisation
result = process_with_db_logging("enrich", enrich_args)
```

#### Intégration de File d'Attente

```python
from ign_lidar.cli.queue import TaskQueue
import redis

# File d'attente de tâches basée sur Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)
task_queue = TaskQueue(redis_client)

# Producteur : Ajouter des tâches à la file d'attente
for las_file in las_files:
    task = {
        "command": "enrich",
        "args": {
            "input": las_file,
            "output": las_file.replace('.las', '_enriched.laz')
        }
    }
    task_queue.enqueue(task)

# Consommateur : Traiter les tâches de la file d'attente
def process_queue_tasks():
    while True:
        task = task_queue.dequeue(timeout=30)
        if task:
            try:
                result = cli.run_command(task["command"], task["args"])
                task_queue.mark_completed(task["id"], result)
            except Exception as e:
                task_queue.mark_failed(task["id"], str(e))
```

## Optimisation des Performances

### API de Mise en Cache

```python
from ign_lidar.cli.cache import CLICache

# Configurer la mise en cache
cache = CLICache(
    cache_dir="/tmp/ign_cli_cache",
    max_size_gb=10.0,
    ttl_hours=24
)

# Activer la mise en cache pour CLI
cli = CommandLineInterface(cache=cache)

# Exécution avec cache (les appels suivants avec les mêmes args retournent le résultat mis en cache)
result1 = cli.run_command("download", {"tiles": ["0631_6275"]})  # Télécharge
result2 = cli.run_command("download", {"tiles": ["0631_6275"]})  # Depuis le cache

print(f"Cache utilisé : {result2.from_cache}")
```

### Gestion des Ressources

```python
from ign_lidar.cli.resources import ResourceManager

# Configurer les limites de ressources
resource_manager = ResourceManager(
    max_cpu_percent=80.0,
    max_memory_gb=16.0,
    max_disk_io_mbps=500.0,
    gpu_memory_fraction=0.8
)

# Appliquer les limites de ressources au CLI
cli = CommandLineInterface(resource_manager=resource_manager)

# CLI se limitera automatiquement pour rester dans les limites
result = cli.run_command("enrich", large_file_args)
```

## Fonctionnalités Avancées

### Système de Plugins

```python
from ign_lidar.cli.plugins import CLIPlugin, register_plugin

class CustomPlugin(CLIPlugin):
    """Exemple de plugin CLI personnalisé."""

    def get_command_name(self):
        return "custom-process"

    def get_command_help(self):
        return "Commande de traitement personnalisée"

    def execute(self, args):
        # Logique de traitement personnalisée
        return {"status": "success", "message": "Traitement personnalisé terminé"}

# Enregistrer le plugin
register_plugin(CustomPlugin())

# Utiliser la commande personnalisée
result = cli.run_command("custom-process", {"input": "data.las"})
```

### Génération de Scripts

```python
from ign_lidar.cli.scripting import ScriptGenerator

# Générer un script bash à partir des commandes CLI
generator = ScriptGenerator(target="bash")

script_content = generator.generate_script([
    ("download", {"tiles": ["0631_6275"], "output_dir": "data"}),
    ("enrich", {"input": "data/0631_6275.las", "output": "data/enriched.laz"})
])

# Enregistrer le script
with open("process_lidar.sh", "w") as f:
    f.write(script_content)

# Le script généré peut être exécuté indépendamment :
# #!/bin/bash
# ign-lidar-hd download --tiles 0631_6275 --output-dir data
# ign-lidar-hd enrich --input data/0631_6275.las --output data/enriched.laz
```

### Gestion de Configuration

```python
from ign_lidar.cli.config_manager import ConfigManager

# Gérer plusieurs configurations
config_manager = ConfigManager(config_dir="~/.ign-lidar/configs")

# Créer des configurations nommées
config_manager.save_config("production", {
    "output_format": "laz",
    "parallel_processing": True,
    "gpu_acceleration": True,
    "quality_level": "high"
})

config_manager.save_config("development", {
    "output_format": "las",
    "parallel_processing": False,
    "gpu_acceleration": False,
    "quality_level": "medium"
})

# Utiliser une configuration spécifique
cli = CommandLineInterface()
cli.load_config("production")

# Lister les configurations disponibles
configs = config_manager.list_configs()
print(f"Configurations disponibles : {configs}")
```

## Bonnes Pratiques

### Modèles de Gestion des Erreurs

```python
from ign_lidar.cli.patterns import robust_execution

@robust_execution(
    retry_attempts=3,
    fallback_strategy="cpu",
    log_failures=True
)
def process_tile(input_file):
    return cli.run_command("enrich", {
        "input": input_file,
        "features": ["buildings", "vegetation"],
        "gpu": True
    })

# Gère automatiquement les nouvelles tentatives et les solutions de repli
result = process_tile("large_tile.las")
```

### Bonnes Pratiques de Logging

```python
import logging
from ign_lidar.cli.logging import setup_cli_logging

# Configurer le logging complet
logger = setup_cli_logging(
    log_file="ign_processing.log",
    console_level=logging.INFO,
    file_level=logging.DEBUG,
    include_performance=True
)

# Les opérations CLI sont automatiquement enregistrées
cli = CommandLineInterface(logger=logger)
result = cli.run_command("enrich", args)

# Les logs incluent :
# - Détails de l'exécution des commandes
# - Métriques de performance
# - Traces d'erreurs
# - Utilisation des ressources
```

## Documentation Connexe

- [Guide des Commandes CLI](../guides/cli-commands.md)
- [Référence de Configuration](./configuration.md)
- [API Processor](./processor.md)
- [Guide de Performance](../guides/performance.md)
