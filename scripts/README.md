# Scripts de Validation et Benchmark

Ce dossier contient les scripts de validation et benchmark pour les optimisations Phase 3.

## 📋 Scripts Disponibles

### 1. `validate_optimizations.py`

**Objectif :** Valider les optimisations GPU Phase 3

**Tests inclus :**

- Comptage précis des transfers CPU↔GPU
- Benchmark CPU vs GPU
- Test d'efficacité du cache GPU

**Usage :**

```bash
conda activate ign_gpu
python scripts/validate_optimizations.py
```

**Durée :** ~2-5 minutes  
**Sortie :** Rapport console + validation des objectifs

---

### 2. `benchmark_phase3_gains.py`

**Objectif :** Benchmark complet des gains Phase 3

**Tests inclus :**

1. **GPU Transfer Count** - Vérification ≤2 transfers par tile
2. **CPU vs GPU Performance** - Mesure du speedup (cible: 5-10x)
3. **Cache Effectiveness** - Speedup sur cache hits (cible: 2-3x)
4. **Multi-Scale Performance** - Tests sur 4 tailles de datasets

**Usage :**

```bash
conda activate ign_gpu
python scripts/benchmark_phase3_gains.py
```

**Durée :** ~5-10 minutes  
**Sortie :**

- Rapport console détaillé
- `benchmark_results.json` avec toutes les métriques

---

## 🎯 Objectifs Phase 3

| Métrique           | Avant  | Cible    | Validation                |
| ------------------ | ------ | -------- | ------------------------- |
| GPU Transfers/tile | 4-6    | ≤2       | validate_optimizations.py |
| Temps PCIe         | ~40%   | ~10%     | benchmark_phase3_gains.py |
| GPU Utilization    | 50-60% | 80-90%   | benchmark_phase3_gains.py |
| Speedup GPU        | 1.0x   | 5-10x    | benchmark_phase3_gains.py |
| Cache Speedup      | 1.0x   | 2-3x     | benchmark_phase3_gains.py |
| Gain Global        | 1.0x   | 1.2-1.3x | Les deux scripts          |

---

## 📊 Interprétation des Résultats

### GPU Transfers

```
✅ EXCELLENT: ≤2 transfers     # Objectif atteint
✓  GOOD: 3-4 transfers         # Acceptable
⚠️  NEEDS WORK: >4 transfers    # Optimisation nécessaire
```

### GPU Speedup

```
✅ EXCELLENT: ≥8x              # Excellente performance
✓  GOOD: 5-8x                  # Bonne performance
⚠️  ACCEPTABLE: 2-5x            # Performance moyenne
❌ POOR: <2x                   # Problème de performance
```

### Cache Speedup

```
✅ EXCELLENT: ≥5x              # Cache très efficace
✓  GOOD: 2-5x                  # Cache efficace
⚠️  ACCEPTABLE: 1-2x            # Cache peu efficace
```

---

## 🔍 Troubleshooting

### GPU Not Detected

**Symptôme :** `GPU not available - skipping`

**Solutions :**

```bash
# Vérifier CuPy
conda activate ign_gpu
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"

# Réinstaller si nécessaire
conda install -c conda-forge cupy cudatoolkit=11.8
```

### Out of Memory

**Symptôme :** `CUDA out of memory`

**Solutions :**

```bash
# Réduire la taille des tests
python scripts/validate_optimizations.py  # Utilise 1M points par défaut

# Augmenter mémoire GPU disponible
# Fermer autres applications GPU (Chrome, etc.)
```

### Slow Performance

**Symptôme :** Speedup GPU <2x

**Causes possibles :**

1. GPU partagé avec affichage
2. Drivers CUDA obsolètes
3. Thermal throttling
4. Dataset trop petit (overhead dominant)

**Solutions :**

```bash
# Vérifier GPU usage
nvidia-smi

# Tester avec dataset plus grand
# Modifier n_points dans le script
```

---

## 📈 Exemples de Résultats

### Résultat Optimal (RTX 4080)

```
GPU TRANSFER ANALYSIS
─────────────────────────────────────
Total transfers: 2
CPU→GPU: 1
GPU→CPU: 1
✅ EXCELLENT: 2 transfers (target: ≤2)

CPU vs GPU PERFORMANCE
─────────────────────────────────────
CPU Time:  12.45s
GPU Time:  1.23s
Speedup:   10.1x
✅ EXCELLENT: 10.1x speedup!

GPU CACHE EFFECTIVENESS
─────────────────────────────────────
Cold run (upload):  45.23ms
Warm run (cached):  2.15ms
Cache speedup:      21.0x
✅ EXCELLENT: 21.0x faster!
```

### Résultat Acceptable (RTX 3060)

```
GPU TRANSFER ANALYSIS
─────────────────────────────────────
Total transfers: 3
CPU→GPU: 2
GPU→CPU: 1
✓ GOOD: 3 transfers (target: ≤2)

CPU vs GPU PERFORMANCE
─────────────────────────────────────
CPU Time:  15.67s
GPU Time:  2.89s
Speedup:   5.4x
✓ GOOD: 5.4x speedup

GPU CACHE EFFECTIVENESS
─────────────────────────────────────
Cold run (upload):  67.45ms
Warm run (cached):  12.34ms
Cache speedup:      5.5x
✅ EXCELLENT: 5.5x faster!
```

---

## 🛠️ Customisation

### Modifier les paramètres de test

**validate_optimizations.py :**

```python
# Ligne 243 : Changer la taille du dataset
test_gpu_transfers(n_points=2_000_000)  # Plus grand

# Ligne 48 : Changer les features calculées
'features': {
    'mode': 'lod3',  # Plus de features
    'k_neighbors': 30,
}
```

**benchmark_phase3_gains.py :**

```python
# Ligne 336 : Changer les tailles testées
benchmark_multi_scale([500_000, 1_000_000, 5_000_000])

# Ligne 96 : Type de données
# Modifier generate_realistic_data() pour vos besoins
```

---

## 📝 Logs et Résultats

### Fichiers générés

```
scripts/
├── validate_optimizations.py
├── benchmark_phase3_gains.py
├── benchmark_results.json          # Résultats JSON détaillés
└── benchmark_output.log            # Log complet du benchmark
```

### Format JSON

```json
{
  "gpu_transfers": {
    "total_transfers": 2,
    "cpu_to_gpu": 1,
    "gpu_to_cpu": 1,
    "status": "excellent",
    "target_met": true
  },
  "cpu_vs_gpu": {
    "cpu_time": 12.45,
    "gpu_time": 1.23,
    "speedup": 10.1,
    "status": "excellent"
  },
  "cache": {
    "speedup": 21.0,
    "status": "excellent"
  }
}
```

---

## 🔗 Références

- **Documentation Phase 3:** `../PHASE_3_SUMMARY.md`
- **Rapport d'optimisation:** `../OPTIMIZATION_REPORT_2025-11-23.md`
- **Guide migration:** `../MIGRATION_GUIDE.md`

---

**Dernière mise à jour :** 23 Novembre 2025
