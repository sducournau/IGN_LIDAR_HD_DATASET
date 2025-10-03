#!/usr/bin/env python3
"""
Test rapide du système d'enrichissement GPU/CPU
Vérifie la configuration et affiche les performances attendues
"""

import sys
from pathlib import Path

print("=" * 70)
print("🔍 TEST CONFIGURATION - ENRICHISSEMENT LAZ OPTIMISÉ")
print("=" * 70)
print()

# Test imports de base
print("📦 Vérification imports de base...")
try:
    import numpy as np
    print("  ✅ NumPy:", np.__version__)
except ImportError:
    print("  ❌ NumPy manquant - pip install numpy")
    sys.exit(1)

try:
    import laspy
    print("  ✅ laspy:", laspy.__version__)
except ImportError:
    print("  ❌ laspy manquant - pip install laspy")
    sys.exit(1)

try:
    from sklearn.neighbors import KDTree
    from sklearn.decomposition import PCA
    print("  ✅ scikit-learn: OK")
except ImportError:
    print("  ❌ scikit-learn manquant - pip install scikit-learn")
    sys.exit(1)

print()

# Test GPU
print("🚀 Détection GPU...")
GPU_AVAILABLE = False
CUML_AVAILABLE = False

try:
    import cupy as cp
    GPU_AVAILABLE = True
    gpu_count = cp.cuda.runtime.getDeviceCount()
    print(f"  ✅ CuPy: {cp.__version__}")
    print(f"     GPU détectés: {gpu_count}")
    
    for i in range(gpu_count):
        device = cp.cuda.Device(i)
        compute_cap = device.compute_capability
        print(f"     GPU {i}: Compute Capability {compute_cap}")
        
except ImportError:
    print("  ⚠️  CuPy non installé")
    print("     Installation: pip install cupy-cuda11x (ou cuda12x)")
except Exception as e:
    print(f"  ⚠️  CuPy installé mais GPU inaccessible: {e}")

try:
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    from cuml.decomposition import PCA as cuPCA
    CUML_AVAILABLE = True
    print("  ✅ RAPIDS cuML: OK")
except ImportError:
    print("  ⚠️  RAPIDS cuML non installé")
    print("     Installation: conda install -c rapidsai -c conda-forge cuml")

print()

# Test modules locaux
print("📂 Vérification modules locaux...")
sys.path.insert(0, str(Path(__file__).parent))

try:
    from ign_lidar.features import compute_normals
    print("  ✅ ign_lidar.features (CPU)")
except ImportError as e:
    print(f"  ❌ ign_lidar.features manquant: {e}")
    sys.exit(1)

try:
    from ign_lidar.features_gpu import GPUFeatureComputer
    print("  ✅ ign_lidar.features_gpu (GPU)")
except ImportError as e:
    print(f"  ⚠️  ign_lidar.features_gpu non importable: {e}")

print()

# Résumé configuration
print("=" * 70)
print("📊 CONFIGURATION DÉTECTÉE")
print("=" * 70)
print()

if GPU_AVAILABLE and CUML_AVAILABLE:
    mode = "GPU ULTRA-RAPIDE (CuPy + RAPIDS cuML)"
    performance = "8-10x"
    time_per_tile = "~3 min"
    total_time = "~30 min"
    color = "🔥"
elif GPU_AVAILABLE:
    mode = "GPU RAPIDE (CuPy seul)"
    performance = "3-5x"
    time_per_tile = "~8 min"
    total_time = "~1h 20min"
    color = "⚡"
else:
    mode = "CPU OPTIMISÉ (multi-core)"
    performance = "1.7x"
    time_per_tile = "~15 min"
    total_time = "~2h 30min"
    color = "💻"

print(f"{color} Mode: {mode}")
print(f"   Accélération vs original: {performance}")
print(f"   Temps par tile (16M points): {time_per_tile}")
print(f"   Temps total (10 tiles): {total_time}")
print()

# Recommandations
print("💡 Recommandations:")
print()

if not GPU_AVAILABLE:
    print("  📌 Pour accélérer 3-10x avec GPU:")
    print("     1. Vérifier GPU NVIDIA: nvidia-smi")
    print("     2. Installer CuPy: pip install cupy-cuda11x")
    print("     3. (Optionnel) RAPIDS: conda install -c rapidsai cuml")
    print()
elif not CUML_AVAILABLE:
    print("  📌 Pour accélérer 2x supplémentaires:")
    print("     conda install -c rapidsai -c conda-forge cuml")
    print()

print("  🚀 Lancer enrichissement:")
print("     ./run_enrichment_optimized.sh")
print()
print("  📊 Comparer CPU vs GPU:")
print("     ./benchmark_gpu.sh")
print()

# Vérifier fichiers LAZ
print("=" * 70)
print("📁 FICHIERS LAZ À TRAITER")
print("=" * 70)
print()

dataset_dir = Path("urban_training_dataset")
raw_dir = dataset_dir / "raw_tiles"
enriched_dir = dataset_dir / "enriched_laz_computed"

if raw_dir.exists():
    laz_files = list(raw_dir.rglob("*.laz"))
    print(f"  LAZ originaux: {len(laz_files)}")
else:
    print("  ⚠️  Répertoire raw_tiles/ non trouvé")
    laz_files = []

if enriched_dir.exists():
    enriched_files = list(enriched_dir.rglob("*.laz"))
    print(f"  LAZ enrichis: {len(enriched_files)}")
else:
    print("  LAZ enrichis: 0")
    enriched_files = []

if laz_files:
    remaining = len(laz_files) - len(enriched_files)
    print(f"  Restants: {remaining}")
    
    if remaining > 0:
        print()
        print(f"  ⏱️  Temps estimé ({mode}):")
        
        if GPU_AVAILABLE and CUML_AVAILABLE:
            minutes = remaining * 3
        elif GPU_AVAILABLE:
            minutes = remaining * 8
        else:
            minutes = remaining * 15
        
        hours = minutes // 60
        mins = minutes % 60
        
        if hours > 0:
            print(f"     ~{hours}h {mins}min pour {remaining} fichiers")
        else:
            print(f"     ~{mins} minutes pour {remaining} fichiers")
    else:
        print()
        print("  ✅ Tous les fichiers sont déjà enrichis!")
        print()
        print("  🎯 Prochaine étape:")
        print("     python create_patches_from_enriched_laz.py")

print()
print("=" * 70)
print("✅ TEST TERMINÉ")
print("=" * 70)
