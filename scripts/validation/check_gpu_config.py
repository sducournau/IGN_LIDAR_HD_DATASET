#!/usr/bin/env python3
"""
Script pour vérifier la configuration GPU et recommander les paramètres optimaux.

Usage:
    python scripts/check_gpu_config.py
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ign_lidar.config import DEFAULT_NUM_POINTS, DEFAULT_K_NEIGHBORS, DEFAULT_PATCH_SIZE


def check_gpu():
    """Vérifier la configuration GPU disponible."""
    
    print("=" * 80)
    print("🎮 VÉRIFICATION CONFIGURATION GPU")
    print("=" * 80)
    print()
    
    if not TORCH_AVAILABLE:
        print("⚠️  PyTorch n'est pas installé")
        print("   Pour vérifier le GPU, installer: pip install torch")
        print()
        return None
    
    if not torch.cuda.is_available():
        print("❌ Aucun GPU CUDA détecté")
        print()
        print("💡 Configuration recommandée (CPU):")
        print(f"   DEFAULT_NUM_POINTS: 4096-8192")
        print(f"   Batch size: 2-4")
        print()
        return None
    
    # Obtenir les infos GPU
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"✅ GPU détecté: {gpu_name}")
    print(f"   VRAM totale: {gpu_memory:.1f} GB")
    print()
    
    return gpu_memory


def recommend_config(gpu_memory_gb):
    """Recommander la configuration selon la VRAM."""
    
    print("=" * 80)
    print("💡 RECOMMANDATIONS")
    print("=" * 80)
    print()
    
    if gpu_memory_gb is None:
        print("⚠️  GPU non détecté - Configuration par défaut (CPU)")
        print()
        print("Configuration actuelle:")
        print(f"  DEFAULT_NUM_POINTS: {DEFAULT_NUM_POINTS}")
        print(f"  DEFAULT_K_NEIGHBORS: {DEFAULT_K_NEIGHBORS}")
        print(f"  DEFAULT_PATCH_SIZE: {DEFAULT_PATCH_SIZE}m")
        print()
        return
    
    # Recommandations selon VRAM
    if gpu_memory_gb < 8:
        recommended_points = 8192
        recommended_batch = 8
        verdict = "⚠️  GPU entrée de gamme"
        can_upgrade = False
    elif gpu_memory_gb < 12:
        recommended_points = 8192
        recommended_batch = 16
        verdict = "✅ GPU moyen"
        can_upgrade = "Possible 16384 avec batch=4-6"
    elif gpu_memory_gb < 16:
        recommended_points = 16384
        recommended_batch = 8
        verdict = "⭐ GPU optimal"
        can_upgrade = False
    else:
        recommended_points = 16384
        recommended_batch = 16
        verdict = "⭐⭐ GPU haut de gamme"
        can_upgrade = "Possible 32768 avec batch=4-8"
    
    print(f"Verdict: {verdict}")
    print()
    print("Configuration recommandée:")
    print(f"  ├─ DEFAULT_NUM_POINTS: {recommended_points}")
    print(f"  ├─ Batch size: {recommended_batch}")
    print(f"  ├─ K-neighbors: {DEFAULT_K_NEIGHBORS}")
    print(f"  └─ Patch size: {DEFAULT_PATCH_SIZE}m")
    print()
    
    if can_upgrade:
        print(f"💡 Upgrade possible: {can_upgrade}")
        print()
    
    # Comparer avec config actuelle
    if DEFAULT_NUM_POINTS != recommended_points:
        print("⚠️  Configuration actuelle différente:")
        print(f"   Actuel: {DEFAULT_NUM_POINTS}")
        print(f"   Recommandé: {recommended_points}")
        print()
        print("   Pour changer, éditer: ign_lidar/config.py")
        print(f"   DEFAULT_NUM_POINTS = {recommended_points}")
        print()
    else:
        print("✅ Configuration actuelle optimale!")
        print()
    
    # Estimation mémoire
    features_dim = 28  # Nombre de features
    memory_estimate = (recommended_points * features_dim * 4 * recommended_batch * 2.5) / 1e9
    
    print("📊 Estimation mémoire GPU:")
    print(f"   Points/patch: {recommended_points}")
    print(f"   Features: {features_dim}")
    print(f"   Batch size: {recommended_batch}")
    print(f"   Mémoire estimée: ~{memory_estimate:.1f} GB")
    print(f"   Marge de sécurité: {gpu_memory_gb - memory_estimate:.1f} GB")
    print()
    
    if memory_estimate > gpu_memory_gb * 0.9:
        print("⚠️  Attention: Utilisation élevée de la VRAM!")
        print("   Réduire batch_size si OOM (Out of Memory)")
        print()


def main():
    """Point d'entrée principal."""
    
    gpu_memory = check_gpu()
    recommend_config(gpu_memory)
    
    print("=" * 80)
    print("📚 DOCUMENTATION")
    print("=" * 80)
    print()
    print("Pour plus d'informations:")
    print("  - docs/NUM_POINTS_OPTIMIZATION.md")
    print("  - docs/K_NEIGHBORS_OPTIMIZATION.md")
    print()
    print("Tests de performance:")
    print("  python tests/benchmark_optimization.py")
    print()


if __name__ == '__main__':
    main()
