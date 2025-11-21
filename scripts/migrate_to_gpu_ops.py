"""
Script de Migration vers GPU Accelerated Ops

Ce script migre automatiquement les fichiers utilisant:
- np.linalg.eigh → gpu_ops.eigh
- np.linalg.eigvalsh → gpu_ops.eigvalsh
- scipy.spatial.cKDTree → gpu_ops.knn
- scipy.spatial.distance.cdist → gpu_ops.cdist

Usage:
    python scripts/migrate_to_gpu_ops.py --file path/to/file.py
    python scripts/migrate_to_gpu_ops.py --all  # Migrer tous les fichiers

Author: Development Team
Date: November 2025
"""

import argparse
import logging
import re
from pathlib import Path
from typing import List, Dict, Tuple
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Patterns de Migration
# ============================================================================

MIGRATION_PATTERNS = {
    "eigh": {
        "pattern": r"np\.linalg\.eigh\(",
        "replacement": "gpu_ops.eigh(",
        "import_add": "eigh"
    },
    "eigvalsh": {
        "pattern": r"np\.linalg\.eigvalsh\(",
        "replacement": "gpu_ops.eigvalsh(",
        "import_add": "eigvalsh"
    },
    "cKDTree": {
        "pattern": r"cKDTree\(",
        "replacement": "# TODO: Replace with gpu_ops.knn() - see migration guide",
        "import_add": "knn",
        "manual": True  # Nécessite intervention manuelle
    },
    "cdist": {
        "pattern": r"distance\.cdist\(",
        "replacement": "gpu_ops.cdist(",
        "import_add": "cdist"
    }
}


# ============================================================================
# Fichiers Prioritaires (depuis audit)
# ============================================================================

PRIORITY_FILES = [
    # Eigenvalue decomposition (10 fichiers)
    "ign_lidar/features/utils.py",
    "ign_lidar/features/gpu_processor.py",
    "ign_lidar/features/compute/normals.py",
    "ign_lidar/features/compute/curvature.py",
    "ign_lidar/features/compute/eigen_features.py",
    "ign_lidar/features/strategies/strategy_cpu.py",
    "ign_lidar/preprocessing/outliers.py",
    "ign_lidar/datasets/transforms.py",
    "ign_lidar/core/classification/building/plane_clustering.py",
    "ign_lidar/core/classification/building/facade_detector.py",
    # cKDTree (20+ fichiers - migration manuelle)
    # "ign_lidar/features/orchestrator.py",  # KNN partout
    # ... autres
]


def backup_file(file_path: Path) -> Path:
    """
    Backup fichier avant modification.

    Args:
        file_path: Chemin du fichier à backuper

    Returns:
        Chemin du backup créé
    """
    backup_path = file_path.with_suffix(file_path.suffix + ".bak")
    shutil.copy2(file_path, backup_path)
    logger.info(f"Backup créé: {backup_path}")
    return backup_path


def detect_imports(content: str) -> Dict[str, bool]:
    """
    Détecte les imports existants dans le fichier.

    Args:
        content: Contenu du fichier

    Returns:
        Dict avec status des imports
    """
    return {
        "numpy": bool(re.search(r"import numpy|from numpy", content)),
        "scipy": bool(re.search(r"import scipy|from scipy", content)),
        "gpu_ops": bool(
            re.search(r"from ign_lidar\.optimization\.gpu_accelerated_ops", content)
        ),
    }


def find_insertion_point(lines: List[str]) -> int:
    """
    Trouve le point d'insertion pour nouveaux imports.

    Args:
        lines: Lignes du fichier

    Returns:
        Index où insérer les imports
    """
    last_import_idx = 0

    for i, line in enumerate(lines):
        if line.strip().startswith(("import ", "from ")):
            last_import_idx = i

    return last_import_idx + 1


def migrate_file(
    file_path: Path, operations: List[str], dry_run: bool = False
) -> Tuple[bool, Dict[str, int]]:
    """
    Migre un fichier vers gpu_ops.

    Args:
        file_path: Chemin du fichier à migrer
        operations: Liste des opérations à migrer ['eigh', 'eigvalsh', etc.]
        dry_run: Si True, n'effectue pas les modifications

    Returns:
        Tuple (success, stats) avec stats = {'eigh': count, 'eigvalsh': count, ...}
    """
    if not file_path.exists():
        logger.error(f"Fichier non trouvé: {file_path}")
        return False, {}

    # Lire contenu
    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)

    # Détections
    imports = detect_imports(content)
    stats = {}

    # Backup
    if not dry_run:
        backup_file(file_path)

    # Appliquer migrations
    modified = False
    new_imports = []

    for op in operations:
        if op not in MIGRATION_PATTERNS:
            logger.warning(f"Opération inconnue: {op}")
            continue

        pattern = MIGRATION_PATTERNS[op]

        # Compter occurrences
        matches = re.findall(pattern["pattern"], content)
        count = len(matches)
        stats[op] = count

        if count == 0:
            continue

        logger.info(f"  {op}: {count} occurrences trouvées")

        # Migration manuelle requise ?
        if pattern.get("manual", False):
            logger.warning(
                f"  {op}: Migration manuelle requise - ajout de commentaires TODO"
            )

        # Remplacer
        content = re.sub(pattern["pattern"], pattern["replacement"], content)
        modified = True

        # Ajouter import si nécessaire
        if not imports["gpu_ops"]:
            new_imports.append(pattern["import_add"])

    if not modified:
        logger.info(f"  Aucune modification nécessaire")
        return False, stats

    # Insérer nouveaux imports
    if new_imports and not imports["gpu_ops"]:
        lines_list = content.splitlines(keepends=True)
        insertion_idx = find_insertion_point(lines_list)

        # Construire import consolidé
        consolidated_import = "from ign_lidar.optimization.gpu_accelerated_ops import ("
        import_names = []
        for imp in new_imports:
            if "import" in imp and "TODO" not in imp:
                # Extraire nom (eigh, eigvalsh, etc.)
                match = re.search(r"import (\w+)", imp)
                if match:
                    import_names.append(match.group(1))

        if import_names:
            consolidated_import += "\n    " + ",\n    ".join(import_names) + "\n)\n"
            lines_list.insert(insertion_idx, consolidated_import)
            content = "".join(lines_list)

    # Écrire modifications
    if not dry_run:
        file_path.write_text(content, encoding="utf-8")
        logger.info(f"✓ Fichier migré: {file_path}")
    else:
        logger.info(f"[DRY RUN] Modifications prêtes pour: {file_path}")

    return True, stats


def migrate_priority_files(dry_run: bool = False) -> Dict[str, Dict[str, int]]:
    """
    Migre tous les fichiers prioritaires.

    Args:
        dry_run: Si True, simulation sans modifications

    Returns:
        Dict avec résultats par fichier
    """
    repo_root = Path(__file__).parent.parent
    results = {}

    logger.info("=" * 80)
    logger.info("MIGRATION VERS GPU ACCELERATED OPS")
    logger.info("=" * 80)

    for file_rel in PRIORITY_FILES:
        file_path = repo_root / file_rel

        logger.info(f"\nMigration: {file_rel}")

        # Détecter opérations à migrer
        operations = []
        content = file_path.read_text(encoding="utf-8")

        if "np.linalg.eigh(" in content:
            operations.append("eigh")
        if "np.linalg.eigvalsh(" in content:
            operations.append("eigvalsh")
        if "cKDTree(" in content:
            operations.append("cKDTree")
        if "distance.cdist(" in content:
            operations.append("cdist")

        if not operations:
            logger.info(f"  Aucune opération à migrer")
            continue

        # Migrer
        success, stats = migrate_file(file_path, operations, dry_run=dry_run)

        if success:
            results[file_rel] = stats

    # Résumé
    logger.info("\n" + "=" * 80)
    logger.info("RÉSUMÉ DE MIGRATION")
    logger.info("=" * 80)

    total_stats = {"eigh": 0, "eigvalsh": 0, "cKDTree": 0, "cdist": 0}

    for file_rel, stats in results.items():
        logger.info(f"\n{file_rel}:")
        for op, count in stats.items():
            logger.info(f"  {op}: {count} occurrences")
            total_stats[op] += count

    logger.info("\nTOTAL:")
    for op, count in total_stats.items():
        logger.info(f"  {op}: {count} occurrences migrées")

    return results


def main():
    parser = argparse.ArgumentParser(description="Migrate files to GPU Accelerated Ops")
    parser.add_argument(
        "--file", type=str, help="Specific file to migrate (relative path)"
    )
    parser.add_argument(
        "--all", action="store_true", help="Migrate all priority files"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Simulate migration without changes"
    )
    parser.add_argument(
        "--operations",
        type=str,
        nargs="+",
        choices=["eigh", "eigvalsh", "cKDTree", "cdist"],
        help="Specific operations to migrate",
    )

    args = parser.parse_args()

    if args.all:
        # Migrer tous les fichiers prioritaires
        results = migrate_priority_files(dry_run=args.dry_run)

        if args.dry_run:
            logger.info("\n" + "=" * 80)
            logger.info("DRY RUN TERMINÉ - Aucune modification effectuée")
            logger.info("Relancer sans --dry-run pour appliquer les changements")
            logger.info("=" * 80)

    elif args.file:
        # Migrer fichier spécifique
        repo_root = Path(__file__).parent.parent
        file_path = repo_root / args.file

        operations = args.operations or ["eigh", "eigvalsh", "cdist"]

        success, stats = migrate_file(file_path, operations, dry_run=args.dry_run)

        if success:
            logger.info(f"\nMigration réussie: {stats}")
        else:
            logger.error("Migration échouée")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
