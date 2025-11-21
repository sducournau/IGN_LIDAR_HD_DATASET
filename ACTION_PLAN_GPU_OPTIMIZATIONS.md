# 🎯 Plan d'Action - Optimisations GPU IGN LIDAR HD

**Date:** 21 Novembre 2025  
**Status:** 🔴 READY TO IMPLEMENT  
**Documents Associés:**

- [PERFORMANCE_AUDIT_2025.md](./PERFORMANCE_AUDIT_2025.md) - Audit complet
- [GPU_OPTIMIZATION_IMPLEMENTATIONS.md](./GPU_OPTIMIZATION_IMPLEMENTATIONS.md) - Code détaillé

---

## 📅 Timeline Optimiste (4 Semaines)

```
Semaine 1: Quick Wins      [=====>              ] 25% complete
Semaine 2: Core Optims     [          =====>    ] 50% complete
Semaine 3: Tests           [               ====>] 75% complete
Semaine 4: Production      [====================] 100% DONE
```

---

## 🚀 Semaine 1: Quick Wins (5 jours)

### Jour 1: Setup & Audit (FAIT ✅)

- [x] Audit complet de la codebase
- [x] Identification goulots d'étranglement
- [x] Documentation détaillée
- [x] Code examples prêts

### Jour 2: P1.4 - Lower GPU Thresholds

**Temps:** 2 heures  
**Difficulté:** ⭐ Facile  
**Impact:** ⭐⭐⭐ Moyen

```bash
# Fichier: ign_lidar/optimization/ground_truth.py
# Ligne ~115

# Tâches:
- [ ] Modifier select_method()
- [ ] Changer seuil 10M → 1M pour gpu_chunked
- [ ] Ajouter seuil 100K pour gpu
- [ ] Test sur 3 datasets (small/medium/large)
- [ ] Commit: "feat: lower GPU thresholds for better utilization"
```

**Commande:**

```bash
# Edit
code ign_lidar/optimization/ground_truth.py +115

# Test
python -c "
from ign_lidar.optimization.ground_truth import GroundTruthOptimizer
import numpy as np

opt = GroundTruthOptimizer()
# Test avec 500K points (devrait être GPU maintenant)
method = opt.select_method(500_000, 100)
assert method == 'gpu', f'Expected gpu, got {method}'
print('✅ GPU threshold lowered successfully')
"

# Commit
git add ign_lidar/optimization/ground_truth.py
git commit -m "feat: lower GPU thresholds for automatic selection (1M for chunked, 100K for basic)"
```

---

### Jour 3: P1.3 - GPU KNN Façades

**Temps:** 3-4 heures  
**Difficulté:** ⭐⭐ Moyen  
**Impact:** ⭐⭐⭐ Moyen

```bash
# Fichier: ign_lidar/core/classification/building/facade_processor.py
# Ligne ~295

# Tâches:
- [ ] Remplacer scipy.cKDTree par gpu_accelerated_ops.knn()
- [ ] Ajouter try/except pour fallback CPU
- [ ] Test sur façade réelle
- [ ] Mesurer speedup (target: 5-10×)
- [ ] Commit
```

**Code à ajouter:**

```python
# AVANT (ligne ~295):
# from scipy.spatial import cKDTree
# tree = cKDTree(candidate_points[:, :2])
# distances, indices = tree.query(candidate_points[:, :2], k=50)

# APRÈS:
from ign_lidar.optimization.gpu_accelerated_ops import knn

try:
    # 🚀 GPU-accelerated KNN (15-20× speedup)
    distances, indices = knn(
        candidate_points[:, :2],
        k=min(50, len(candidate_points))
    )
except Exception as e:
    # Fallback to CPU if GPU fails
    logger.debug(f"GPU KNN failed, using CPU: {e}")
    from scipy.spatial import cKDTree
    tree = cKDTree(candidate_points[:, :2])
    distances, indices = tree.query(candidate_points[:, :2], k=50)
```

**Test:**

```bash
# Créer test simple
python -c "
import numpy as np
from ign_lidar.core.classification.building.facade_processor import FacadeProcessor

# Test KNN sur façade
points = np.random.rand(10000, 3) * 100
# ... (setup FacadeSegment) ...
# processor = FacadeProcessor(facade, points, heights)
# processor._classify_wall_points()  # Should use GPU KNN

print('✅ Façade GPU KNN working')
"

git add ign_lidar/core/classification/building/facade_processor.py
git commit -m "feat: GPU KNN for facade verticality checks (5-10× speedup)"
```

---

### Jours 4-5: P0.1 - GPU Road Classification

**Temps:** 12-14 heures  
**Difficulté:** ⭐⭐⭐ Difficile  
**Impact:** ⭐⭐⭐⭐⭐ CRITIQUE

```bash
# Fichier: ign_lidar/core/classification/reclassifier.py
# Nouvelle méthode: _classify_roads_with_nature_gpu()

# Tâches Jour 4:
- [ ] Copier méthode _classify_roads_with_nature_gpu() (voir GPU_OPTIMIZATION_IMPLEMENTATIONS.md)
- [ ] Ajouter imports cuSpatial
- [ ] Modifier reclassify() pour appeler GPU si disponible
- [ ] Test basique avec 10K points

# Tâches Jour 5:
- [ ] Test avec dataset réel (1M+ points)
- [ ] Validation CPU vs GPU (<1% diff)
- [ ] Mesurer speedup (target: >10×)
- [ ] Créer test unitaire tests/test_gpu_reclassifier.py
- [ ] Commit
```

**Implémentation (Voir GPU_OPTIMIZATION_IMPLEMENTATIONS.md pour code complet):**

**Test complet:**

```bash
# Test avec tile réel
python -c "
import numpy as np
import geopandas as gpd
import laspy
from ign_lidar.core.classification.reclassifier import Reclassifier

# Load real tile
las = laspy.read('data/tiles/example.laz')
points = np.vstack([las.x, las.y, las.z]).T

# Load roads
roads = gpd.read_file('data/bdtopo/example.gpkg', layer='roads')

# Test GPU
reclassifier = Reclassifier(acceleration_mode='gpu')
labels = np.zeros(len(points), dtype=np.int32)

import time
start = time.time()
n_classified = reclassifier._classify_roads_with_nature_gpu(
    points, labels, roads
)
elapsed = time.time() - start

print(f'✅ GPU road classification: {n_classified:,} points in {elapsed:.2f}s')
print(f'   Throughput: {n_classified/elapsed:,.0f} points/sec')
"

# Run full test suite
pytest tests/test_gpu_reclassifier.py -v

# Commit
git add ign_lidar/core/classification/reclassifier.py tests/test_gpu_reclassifier.py
git commit -m "feat: GPU road classification with cuSpatial (10-20× speedup)"
```

---

## ⚙️ Semaine 2: Core Optimizations (5 jours)

### Jours 6-10: P0.2 - GPU BBox Optimization

**Temps:** ~20 heures  
**Difficulté:** ⭐⭐⭐⭐ Très Difficile  
**Impact:** ⭐⭐⭐⭐⭐ CRITIQUE

```bash
# Fichier: ign_lidar/core/classification/building/building_clusterer.py
# Nouvelle méthode: optimize_bbox_for_building_gpu()

# Planning:
Jour 6:  Setup + imports + data preparation GPU
Jour 7:  Grid search vectorisé + broadcasting
Jour 8:  Scoring vectorisé + best bbox selection
Jour 9:  Tests unitaires + validation
Jour 10: Benchmarks + intégration + documentation
```

**Jour 6: Setup**

```bash
# Tâches:
- [ ] Créer méthode optimize_bbox_for_building_gpu()
- [ ] Transfer points/heights to GPU
- [ ] Generate shift grid (meshgrid)
- [ ] Test avec 1 bâtiment simple
```

**Jour 7: Vectorization**

```bash
# Tâches:
- [ ] Vectorized point-in-bbox test (broadcasting)
- [ ] Validate shapes correctes [n_shifts, n_points]
- [ ] Test avec 100 shifts, 10K points
```

**Jour 8: Scoring**

```bash
# Tâches:
- [ ] Vectorized building/ground counting
- [ ] Vectorized scoring
- [ ] argmax pour meilleur bbox
- [ ] Test avec différents paramètres
```

**Jour 9: Tests**

```bash
# Créer tests/test_gpu_bbox_optimization.py
- [ ] Test GPU vs CPU (résultats identiques)
- [ ] Test speedup (>50×)
- [ ] Test accuracy (>80% building capture)
- [ ] Test fallback CPU si erreur GPU
```

**Jour 10: Integration**

```bash
# Tâches:
- [ ] Ajouter use_gpu_bbox_optimization param dans BuildingClusterer.__init__()
- [ ] Modifier appels dans process_building_cluster()
- [ ] Benchmark sur 100 bâtiments réels
- [ ] Documentation
- [ ] Commit final
```

**Test final:**

```bash
# Benchmark complet
python -c "
from ign_lidar.core.classification.building import BuildingClusterer
import numpy as np
import time

clusterer = BuildingClusterer(use_gpu_bbox_optimization=True)

# Test 100 bâtiments
times_gpu = []
times_cpu = []

for i in range(100):
    # ... (generate building data) ...

    # GPU
    start = time.time()
    shift_gpu, bbox_gpu = clusterer.optimize_bbox_for_building_gpu(...)
    times_gpu.append(time.time() - start)

    # CPU
    clusterer.use_gpu_bbox_optimization = False
    start = time.time()
    shift_cpu, bbox_cpu = clusterer.optimize_bbox_for_building(...)
    times_cpu.append(time.time() - start)

print(f'GPU: {np.mean(times_gpu)*1000:.1f}ms avg')
print(f'CPU: {np.mean(times_cpu)*1000:.1f}ms avg')
print(f'Speedup: {np.mean(times_cpu)/np.mean(times_gpu):.1f}×')
"

pytest tests/test_gpu_bbox_optimization.py -v

git add ign_lidar/core/classification/building/building_clusterer.py \
        tests/test_gpu_bbox_optimization.py
git commit -m "feat: GPU bbox optimization with vectorized grid search (50-100× speedup)"
```

---

## 🧪 Semaine 3: Tests & Validation (5 jours)

### Jour 11-12: Tests GPU Complets

```bash
# Compléter suite de tests

# tests/test_gpu_reclassifier.py
- [ ] test_classify_roads_with_nature_gpu_vs_cpu
- [ ] test_gpu_fallback_on_error
- [ ] test_all_road_types_classified
- [ ] test_gpu_memory_cleanup

# tests/test_gpu_bbox_optimization.py
- [ ] test_optimize_bbox_gpu_vs_cpu
- [ ] test_bbox_optimization_accuracy
- [ ] test_grid_search_completeness
- [ ] test_gpu_memory_management

# tests/test_gpu_facades.py
- [ ] test_facade_knn_gpu_vs_cpu
- [ ] test_facade_processing_speedup
- [ ] test_gpu_fallback

# Run all
pytest tests/test_gpu_*.py -v --cov=ign_lidar --cov-report=html
```

### Jour 13: Validation Données Production

```bash
# Test sur tiles réels

# Petit tile (100K points)
python scripts/process_tile.py \
    --input data/tiles/small_tile.laz \
    --output results/small_gpu.laz \
    --ground-truth data/bdtopo/small.gpkg \
    --use-gpu

# Moyen tile (1M points)
python scripts/process_tile.py \
    --input data/tiles/medium_tile.laz \
    --output results/medium_gpu.laz \
    --ground-truth data/bdtopo/medium.gpkg \
    --use-gpu

# Grand tile (10M points)
python scripts/process_tile.py \
    --input data/tiles/large_tile.laz \
    --output results/large_gpu.laz \
    --ground-truth data/bdtopo/large.gpkg \
    --use-gpu

# Validation:
- [ ] Pas de crashes
- [ ] Classification cohérente avec CPU
- [ ] Speedup mesuré sur chaque taille
- [ ] GPU memory < 8GB
```

### Jour 14-15: Benchmarks & Profiling

```bash
# Benchmark complet

python scripts/benchmark_gpu_improvements.py \
    --tile data/tiles/benchmark_tile.laz \
    --ground-truth data/bdtopo/benchmark.gpkg \
    --output benchmarks/gpu_improvements.json

# Profiling détaillé
python -m cProfile -o profile_gpu.prof scripts/process_tile.py --use-gpu
snakeviz profile_gpu.prof

# Comparer CPU vs GPU
python -m cProfile -o profile_cpu.prof scripts/process_tile.py --use-cpu
diff_prof profile_cpu.prof profile_gpu.prof

# Générer rapport
python scripts/generate_performance_report.py \
    --benchmarks benchmarks/gpu_improvements.json \
    --output benchmarks/report.html
```

---

## 🚀 Semaine 4: Production Ready (5 jours)

### Jour 16-17: Documentation

```bash
# docs/docs/features/gpu-acceleration.md
- [ ] Ajouter section "GPU Road Classification"
- [ ] Ajouter section "GPU BBox Optimization"
- [ ] Exemples d'utilisation
- [ ] Troubleshooting GPU

# docs/docs/guides/performance-tuning.md
- [ ] Ajouter GPU best practices
- [ ] Thresholds recommandés
- [ ] Memory management

# README.md
- [ ] Mettre à jour section Performance
- [ ] Ajouter GPU requirements
- [ ] Exemples nouveaux speedups

# CHANGELOG.md
- [ ] Section v3.1.0
- [ ] Lister tous les GPU improvements
- [ ] Breaking changes (si applicable)
```

### Jour 18: CI/CD

```bash
# .github/workflows/gpu-tests.yml (si GPU disponible en CI)
- [ ] Créer workflow GPU tests
- [ ] Badge test coverage
- [ ] Automated benchmarks

# Configuration
- [ ] Ajouter pytest markers: @pytest.mark.gpu
- [ ] Setup pytest.ini pour GPU tests
- [ ] Documentation CI/CD
```

### Jour 19: Tutoriel & Examples

```bash
# examples/gpu_quickstart.py
- [ ] Créer exemple minimal GPU
- [ ] Road classification example
- [ ] BBox optimization example
- [ ] Benchmarking example

# examples/gpu_tuning.py
- [ ] GPU memory management
- [ ] Chunk size optimization
- [ ] Multi-GPU support (future)
```

### Jour 20: Release

```bash
# Checklist pré-release
- [ ] Tous tests passent (CPU + GPU)
- [ ] Documentation complète
- [ ] CHANGELOG.md à jour
- [ ] Examples fonctionnels
- [ ] Benchmarks validés
- [ ] Code review

# Release
git tag v3.1.0
git push origin v3.1.0

# PyPI (si applicable)
python -m build
python -m twine upload dist/*

# Annonce
- [ ] Release notes GitHub
- [ ] Tweet/LinkedIn
- [ ] Update documentation site
```

---

## 📊 Métriques de Succès

### Targets Phase 1 (Semaine 1)

- [x] Audit complet
- [ ] 2/3 quick wins implémentés (P1.3, P1.4)
- [ ] 1/2 P0 commencé (P0.1 road classification)
- [ ] Speedup mesuré: 5-10× sur roads

### Targets Phase 2 (Semaine 2)

- [ ] P0.1 & P0.2 terminés
- [ ] Tests unitaires créés
- [ ] Speedup mesuré: 50-100× sur bboxes
- [ ] Pas de régression CPU

### Targets Phase 3 (Semaine 3)

- [ ] Coverage tests GPU >80%
- [ ] Validation production OK
- [ ] Benchmarks complets
- [ ] Profiling identifie autres opportunités

### Targets Phase 4 (Semaine 4)

- [ ] Documentation complète
- [ ] Examples fonctionnels
- [ ] Release v3.1.0
- [ ] Users informés

---

## 🎯 KPIs Finaux

**Performance:**

- ✅ Reclassification tile 1km²: **<5 minutes** (vs 30+ actuellement)
- ✅ Building bbox optimization: **<30 seconds/tile** (vs 10-30 minutes)
- ✅ Speedup global: **10-15×**

**Qualité:**

- ✅ Test coverage GPU: **>80%**
- ✅ Accuracy vs CPU: **>99%**
- ✅ Pas de memory leaks

**Production:**

- ✅ Documentation complète
- ✅ Examples fonctionnels
- ✅ CI/CD intégré
- ✅ Users satisfaits

---

## 🚨 Risques & Mitigation

### Risque 1: GPU Memory OOM

**Probabilité:** Moyenne  
**Impact:** Élevé

**Mitigation:**

- Chunked processing automatique
- Fallback CPU si OOM
- Tests avec datasets variés
- Documentation memory requirements

### Risque 2: cuSpatial Bugs

**Probabilité:** Faible  
**Impact:** Élevé

**Mitigation:**

- Tests exhaustifs CPU vs GPU
- Fallback CPU toujours disponible
- Version pinning cuSpatial
- Tests de régression

### Risque 3: Breaking Changes

**Probabilité:** Faible  
**Impact:** Moyen

**Mitigation:**

- Backward compatibility maintenue
- GPU opt-in (pas opt-out)
- Deprecation warnings si needed
- Tests extensive

### Risque 4: Timeline Slip

**Probabilité:** Moyenne  
**Impact:** Faible

**Mitigation:**

- Prioriser P0 absolument
- P1 can slip si nécessaire
- Weekly checkpoint meetings
- Scope creep prevention

---

## 📞 Support & Questions

### Pendant Implémentation

**Questions Code:**

- Référence: GPU_OPTIMIZATION_IMPLEMENTATIONS.md
- Examples: examples/gpu\_\*.py
- Tests: tests/test*gpu*\*.py

**Questions Architecture:**

- Design patterns: docs/docs/architecture.md
- GPU best practices: docs/docs/features/gpu-acceleration.md

**Debugging:**

```python
# Enable verbose GPU logging
import logging
logging.getLogger('ign_lidar').setLevel(logging.DEBUG)

# Check GPU availability
from ign_lidar.optimization.gpu_accelerated_ops import HAS_CUPY, HAS_FAISS, HAS_CUML
print(f"CuPy: {HAS_CUPY}, FAISS: {HAS_FAISS}, cuML: {HAS_CUML}")

# Monitor GPU memory
import cupy as cp
mempool = cp.get_default_memory_pool()
print(f"GPU Memory: {mempool.used_bytes() / 1e9:.2f} GB")
```

---

## ✅ Daily Checklist Template

```markdown
## Jour X: [Task Name]

### Morning (9h-12h)

- [ ] Code implementation
- [ ] Unit tests
- [ ] Git commit

### Afternoon (14h-17h)

- [ ] Integration testing
- [ ] Documentation
- [ ] Git commit

### Evening

- [ ] Review day's work
- [ ] Update timeline
- [ ] Plan next day

### Blockers: None / [List if any]

### Notes: [Any observations]
```

---

## 🎉 Conclusion

### Ready to Start

✅ Audit complet  
✅ Code examples prêts  
✅ Tests définis  
✅ Timeline claire

### Success Criteria Clear

🎯 10-15× speedup global  
🎯 <5 minutes par tile  
🎯 Production ready v3.1.0

### Let's Ship! 🚀

**Next Action:** Jour 2 - P1.4 Lower GPU Thresholds (2h)

---

**Auteur:** AI Implementation Team  
**Date:** 21 Novembre 2025  
**Version:** 1.0  
**Status:** 🟢 READY TO EXECUTE
