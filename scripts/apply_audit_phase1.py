#!/usr/bin/env python3
"""
Script d'automatisation pour la Phase 1 de l'audit du codebase IGN LiDAR HD

Ce script implémente les corrections prioritaires identifiées dans l'audit :
1. Fusion de GroundTruthOptimizer (optimization/ ← io/)
2. Création de GPUManager centralisé
3. Consolidation de compute_normals()

Usage:
    python scripts/apply_audit_phase1.py --task merge_ground_truth
    python scripts/apply_audit_phase1.py --task create_gpu_manager
    python scripts/apply_audit_phase1.py --task consolidate_normals
    python scripts/apply_audit_phase1.py --all  # Exécuter toutes les tâches

Author: LiDAR Trainer Agent
Date: November 21, 2025
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class AuditPhase1:
    """Classe principale pour appliquer les corrections de l'audit Phase 1."""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / ".audit_backups"
        self.backup_dir.mkdir(exist_ok=True)
    
    def backup_file(self, file_path: Path) -> Path:
        """Créer une sauvegarde d'un fichier avant modification."""
        backup_path = self.backup_dir / f"{file_path.name}.backup"
        shutil.copy2(file_path, backup_path)
        logger.info(f"✅ Backup créé : {backup_path}")
        return backup_path
    
    def merge_ground_truth_optimizer(self):
        """
        Tâche 1 : Fusionner GroundTruthOptimizer
        
        - Copier features V2 (cache) de io/ vers optimization/
        - Créer alias de dépréciation dans io/
        - Mettre à jour imports dans core/
        """
        logger.info("=" * 70)
        logger.info("🔧 TÂCHE 1 : Fusion de GroundTruthOptimizer")
        logger.info("=" * 70)
        
        src_file = self.project_root / "ign_lidar/optimization/ground_truth.py"
        src_v2_file = self.project_root / "ign_lidar/io/ground_truth_optimizer.py"
        
        # Backup
        self.backup_file(src_file)
        self.backup_file(src_v2_file)
        
        logger.info(f"📂 Source (publique) : {src_file} (553 lignes)")
        logger.info(f"📂 Source V2 (cache)  : {src_v2_file} (902 lignes)")
        
        logger.warning("⚠️  IMPLÉMENTATION MANUELLE REQUISE")
        logger.info("")
        logger.info("Étapes à suivre :")
        logger.info("1. Ouvrir optimization/ground_truth.py")
        logger.info("2. Ajouter les paramètres de cache de io/ground_truth_optimizer.py:")
        logger.info("   - enable_cache: bool")
        logger.info("   - cache_dir: Optional[Path]")
        logger.info("   - max_cache_size_mb: float")
        logger.info("   - max_cache_entries: int")
        logger.info("3. Copier les méthodes de cache (lignes 120-300 environ)")
        logger.info("4. Créer alias de dépréciation dans io/ground_truth_optimizer.py")
        logger.info("5. Mettre à jour les imports dans core/processor.py (ligne 2303)")
        logger.info("6. Mettre à jour les imports dans core/classification_applier.py (ligne 201)")
        logger.info("")
        logger.info("✅ Backups créés dans : .audit_backups/")
        logger.info("⏱️  Estimation : 3-4 heures")
        logger.info("📉 Impact : -350 lignes de code dupliqué")
        
        return True
    
    def create_gpu_manager(self):
        """
        Tâche 2 : Créer GPUManager centralisé
        
        - Créer core/gpu.py avec classe singleton GPUManager
        - Migrer 6+ détections GPU existantes
        - Créer alias backward compatible
        """
        logger.info("=" * 70)
        logger.info("🔧 TÂCHE 2 : Création de GPUManager centralisé")
        logger.info("=" * 70)
        
        target_file = self.project_root / "ign_lidar/core/gpu.py"
        
        if target_file.exists():
            logger.warning(f"⚠️  Le fichier {target_file} existe déjà")
            return False
        
        logger.info(f"📝 Création de : {target_file}")
        logger.info("")
        logger.info("Fonctionnalités à implémenter :")
        logger.info("- Classe GPUManager (singleton)")
        logger.info("- Propriétés : gpu_available, cuml_available, cuspatial_available, faiss_gpu_available")
        logger.info("- Lazy initialization avec cache")
        logger.info("- Alias backward compatible : GPU_AVAILABLE, HAS_CUPY")
        logger.info("")
        logger.info("Fichiers à migrer (6 locations) :")
        logger.info("  1. utils/normalization.py:21 → GPU_AVAILABLE")
        logger.info("  2. optimization/gpu_wrapper.py:39 → _GPU_AVAILABLE")
        logger.info("  3. optimization/gpu_wrapper.py:42 → check_gpu_available()")
        logger.info("  4. optimization/ground_truth.py:87 → _gpu_available")
        logger.info("  5. optimization/gpu_profiler.py:160 → gpu_available")
        logger.info("  6. features/gpu_processor.py:14 → GPU_AVAILABLE")
        logger.info("")
        
        # Template du fichier
        template = '''"""
Centralized GPU Detection and Management

Single source of truth for GPU availability across the entire codebase.

This module replaces 6+ scattered GPU detection implementations with
a unified singleton pattern.

Usage:
    from ign_lidar.core.gpu import GPUManager
    
    gpu = GPUManager()
    if gpu.gpu_available:
        # Use GPU
    
    # Legacy compatibility
    from ign_lidar.core.gpu import GPU_AVAILABLE

Author: LiDAR Trainer Agent (Audit Phase 1)
Date: November 21, 2025
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GPUManager:
    """Singleton for centralized GPU detection and management."""
    
    _instance: Optional['GPUManager'] = None
    _gpu_available: Optional[bool] = None
    _cuml_available: Optional[bool] = None
    _cuspatial_available: Optional[bool] = None
    _faiss_gpu_available: Optional[bool] = None
    
    def __new__(cls) -> 'GPUManager':
        """Ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def gpu_available(self) -> bool:
        """Check if basic GPU (CuPy) is available."""
        if self._gpu_available is None:
            self._gpu_available = self._check_cupy()
        return self._gpu_available
    
    @property
    def cuml_available(self) -> bool:
        """Check if cuML (GPU ML library) is available."""
        if self._cuml_available is None:
            self._cuml_available = self._check_cuml()
        return self._cuml_available
    
    @property
    def cuspatial_available(self) -> bool:
        """Check if cuSpatial (GPU spatial ops) is available."""
        if self._cuspatial_available is None:
            self._cuspatial_available = self._check_cuspatial()
        return self._cuspatial_available
    
    @property
    def faiss_gpu_available(self) -> bool:
        """Check if FAISS-GPU (GPU similarity search) is available."""
        if self._faiss_gpu_available is None:
            self._faiss_gpu_available = self._check_faiss()
        return self._faiss_gpu_available
    
    def _check_cupy(self) -> bool:
        """Check CuPy availability."""
        try:
            import cupy as cp
            _ = cp.array([1.0])
            return True
        except Exception:
            return False
    
    def _check_cuml(self) -> bool:
        """Check cuML availability."""
        if not self.gpu_available:
            return False
        try:
            from cuml.neighbors import NearestNeighbors
            import cupy as cp
            cp.cuda.Device(0).compute_capability
            return True
        except Exception:
            return False
    
    def _check_cuspatial(self) -> bool:
        """Check cuSpatial availability."""
        if not self.gpu_available:
            return False
        try:
            import cuspatial
            return True
        except ImportError:
            return False
    
    def _check_faiss(self) -> bool:
        """Check FAISS-GPU availability."""
        if not self.gpu_available:
            return False
        try:
            import faiss
            return hasattr(faiss, 'StandardGpuResources')
        except ImportError:
            return False
    
    def get_info(self) -> dict:
        """Get comprehensive GPU information."""
        return {
            'gpu_available': self.gpu_available,
            'cuml_available': self.cuml_available,
            'cuspatial_available': self.cuspatial_available,
            'faiss_gpu_available': self.faiss_gpu_available,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        info = self.get_info()
        status = "✅" if info['gpu_available'] else "❌"
        return (
            f"GPUManager({status} GPU, "
            f"cuML={info['cuml_available']}, "
            f"cuSpatial={info['cuspatial_available']}, "
            f"FAISS={info['faiss_gpu_available']})"
        )


# Convenience function
def get_gpu_manager() -> GPUManager:
    """Get the global GPUManager instance."""
    return GPUManager()


# Backward compatibility aliases
GPU_AVAILABLE = get_gpu_manager().gpu_available
HAS_CUPY = GPU_AVAILABLE


__all__ = [
    'GPUManager',
    'get_gpu_manager',
    'GPU_AVAILABLE',  # Backward compat
    'HAS_CUPY',       # Backward compat
]
'''
        
        logger.info("💾 Template prêt à être créé")
        logger.info("")
        logger.info("✅ Pour créer le fichier, exécutez :")
        logger.info(f"   cat > {target_file} << 'EOF'")
        logger.info("   [contenu du template]")
        logger.info("   EOF")
        logger.info("")
        logger.info("⏱️  Estimation : 4-6 heures")
        logger.info("📉 Impact : -150 lignes de code dupliqué + cohérence")
        
        return True
    
    def consolidate_normals(self):
        """
        Tâche 3 : Consolider compute_normals()
        
        - Identifier source de vérité (compute/normals.py)
        - Refactorer strategies pour utiliser source unique
        - Supprimer duplications
        """
        logger.info("=" * 70)
        logger.info("🔧 TÂCHE 3 : Consolidation de compute_normals()")
        logger.info("=" * 70)
        
        logger.info("")
        logger.info("📊 11 implémentations identifiées dans 6 fichiers :")
        logger.info("")
        
        implementations = [
            ("features/numba_accelerated.py", "compute_normals_from_eigenvectors_numba", 174),
            ("features/numba_accelerated.py", "compute_normals_from_eigenvectors_numpy", 212),
            ("features/numba_accelerated.py", "compute_normals_from_eigenvectors", 233),
            ("features/feature_computer.py", "compute_normals", 160),
            ("features/feature_computer.py", "compute_normals_with_boundary", 370),
            ("features/gpu_processor.py", "compute_normals", 359),
            ("features/compute/normals.py", "compute_normals", 28),
            ("features/compute/normals.py", "compute_normals_fast", 177),
            ("features/compute/normals.py", "compute_normals_accurate", 203),
            ("features/compute/features.py", "compute_normals", 237),
            ("optimization/gpu_kernels.py", "compute_normals_and_eigenvalues", 439),
        ]
        
        for i, (file, func, line) in enumerate(implementations, 1):
            logger.info(f"  {i:2d}. {file:45s} → {func:40s} (L{line})")
        
        logger.info("")
        logger.info("🎯 Source de vérité recommandée : features/compute/normals.py")
        logger.info("")
        logger.info("Stratégie de consolidation :")
        logger.info("  ✅ GARDER : compute/normals.py (source unique)")
        logger.info("  ✅ GARDER : numba_accelerated.py (optimisations Numba)")
        logger.info("  ✅ GARDER : gpu_kernels.py (CUDA kernels bas-niveau)")
        logger.info("  🔄 REFACTOR : strategy_cpu.py → utiliser compute/normals.py")
        logger.info("  🔄 REFACTOR : strategy_gpu.py → utiliser compute/normals.py")
        logger.info("  ❌ SUPPRIMER : Duplications dans feature_computer.py")
        logger.info("  ❌ SUPPRIMER : Duplications dans compute/features.py")
        logger.info("  🔄 ADAPTER : gpu_processor.py → déléguer à strategy_gpu.py")
        logger.info("")
        logger.info("⏱️  Estimation : 6-8 heures")
        logger.info("📉 Impact : -800 lignes de code dupliqué")
        
        return True
    
    def run_all(self):
        """Exécuter toutes les tâches de la Phase 1."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("🚀 AUDIT PHASE 1 : Corrections Critiques")
        logger.info("=" * 70)
        logger.info("")
        logger.info("📋 Tâches à exécuter :")
        logger.info("  1. Fusion de GroundTruthOptimizer (3-4h)")
        logger.info("  2. Création de GPUManager (4-6h)")
        logger.info("  3. Consolidation de compute_normals() (6-8h)")
        logger.info("")
        logger.info("⏱️  Temps total estimé : 13-18 heures")
        logger.info("📉 Impact total : -1,300 lignes de code")
        logger.info("")
        
        tasks = [
            ("merge_ground_truth", self.merge_ground_truth_optimizer),
            ("create_gpu_manager", self.create_gpu_manager),
            ("consolidate_normals", self.consolidate_normals),
        ]
        
        for task_name, task_func in tasks:
            logger.info("")
            task_func()
            logger.info("")
        
        logger.info("=" * 70)
        logger.info("✅ Analyse Phase 1 terminée")
        logger.info("=" * 70)
        logger.info("")
        logger.info("📁 Backups disponibles dans : .audit_backups/")
        logger.info("📊 Rapports complets :")
        logger.info("   - AUDIT_SUMMARY.md")
        logger.info("   - CODEBASE_AUDIT_FINAL_NOVEMBER_2025.md")
        logger.info("")
        logger.info("🔄 Prochaines étapes :")
        logger.info("   1. Valider ce plan avec l'équipe")
        logger.info("   2. Créer GitHub issues pour chaque tâche")
        logger.info("   3. Implémenter les corrections manuellement")
        logger.info("   4. Exécuter tests : pytest tests/ -v")
        logger.info("   5. Exécuter benchmarks GPU")
        logger.info("")
        
        return True


def main():
    """Point d'entrée principal du script."""
    parser = argparse.ArgumentParser(
        description="Appliquer les corrections de l'Audit Phase 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation :
  python scripts/apply_audit_phase1.py --all
  python scripts/apply_audit_phase1.py --task merge_ground_truth
  python scripts/apply_audit_phase1.py --task create_gpu_manager
  python scripts/apply_audit_phase1.py --task consolidate_normals

Pour plus d'informations, consultez :
  - AUDIT_SUMMARY.md
  - CODEBASE_AUDIT_FINAL_NOVEMBER_2025.md
        """
    )
    
    parser.add_argument(
        '--task',
        choices=['merge_ground_truth', 'create_gpu_manager', 'consolidate_normals'],
        help="Tâche spécifique à exécuter"
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help="Exécuter toutes les tâches de la Phase 1"
    )
    parser.add_argument(
        '--project-root',
        type=Path,
        default=Path.cwd(),
        help="Racine du projet (défaut: répertoire courant)"
    )
    
    args = parser.parse_args()
    
    # Vérifier que le projet root est valide
    if not (args.project_root / "ign_lidar").exists():
        logger.error(f"❌ Répertoire invalide : {args.project_root}")
        logger.error("   Le dossier 'ign_lidar' n'existe pas")
        return 1
    
    audit = AuditPhase1(args.project_root)
    
    if args.all:
        success = audit.run_all()
    elif args.task:
        task_map = {
            'merge_ground_truth': audit.merge_ground_truth_optimizer,
            'create_gpu_manager': audit.create_gpu_manager,
            'consolidate_normals': audit.consolidate_normals,
        }
        success = task_map[args.task]()
    else:
        parser.print_help()
        return 0
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
