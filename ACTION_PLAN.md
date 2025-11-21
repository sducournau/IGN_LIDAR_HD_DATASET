# Plan d'Action - Refactoring IGN LiDAR HD Dataset

**Date de création:** 21 Novembre 2025  
**Version:** 1.0  
**Statut:** En cours d'exécution

---

## 🎯 Objectifs

1. **Éliminer les duplications critiques** (compute_normals, etc.)
2. **Nettoyer les préfixes redondants** ("unified", "enhanced")
3. **Réduire la complexité** des classes oversized
4. **Optimiser la gestion GPU** (pooling, streaming)
5. **Améliorer la maintenabilité** du code

---

## 📋 Phase 1 - URGENT (Semaine 1-2)

### ✅ Tâche 1.1: Consolider compute_normals()

**Priorité:** 🔴 CRITIQUE  
**Effort:** 2-3 jours  
**Statut:** 🟢 EN COURS

**Problème:**

- 10 implémentations différentes de compute_normals
- Code dupliqué, maintenance difficile
- Risque d'incohérences

**Solution:**

1. **Désigner 2 implémentations canoniques:**

   - CPU: `ign_lidar/features/compute/normals.py::compute_normals()`
   - GPU: `ign_lidar/optimization/gpu_kernels.py::compute_normals_and_eigenvalues()`

2. **Refactoriser les autres pour déléguer:**

   - `FeatureComputer.compute_normals()` → délègue à compute/normals
   - `GPUProcessor.compute_normals()` → délègue à gpu_kernels
   - Supprimer duplications dans numba_accelerated.py

3. **Ajouter paramètres pour variantes:**
   ```python
   def compute_normals(
       points: np.ndarray,
       k_neighbors: int = 20,
       method: str = 'standard',  # 'fast' | 'accurate' | 'standard'
       with_boundary: bool = False,
       return_eigenvalues: bool = True,
       use_gpu: bool = False
   ):
   ```

**Actions:**

- [x] Audit des 10 implémentations
- [ ] Refactorer FeatureComputer.compute_normals() → déléguer
- [ ] Refactorer GPUProcessor.compute_normals() → déléguer
- [ ] Supprimer compute*normals_from_eigenvectors*\* duplications
- [ ] Tests unitaires pour chaque variante
- [ ] Documentation des choix

---

### ✅ Tâche 1.2: Nettoyer Préfixes "unified"/"enhanced"

**Priorité:** 🟠 MAJEUR  
**Effort:** 1-2 jours  
**Statut:** ⚪ PLANIFIÉ

**Fichiers prioritaires:**

1. **`ign_lidar/config/building_config.py`**

   - Renommer: `EnhancedBuildingConfig` → `BuildingConfig`
   - Impact: Configuration bâtiments

2. **`ign_lidar/core/processor.py`**

   - Nettoyer: "Phase 4.3: New unified orchestrator V5"
   - Remplacer: "unified orchestrator" → "orchestrator"

3. **`ign_lidar/core/classification/facade_processor.py`**

   - 30+ occurrences "enhanced"
   - Renommer: `enable_enhanced_lod3` → `enable_detailed_lod3`
   - Renommer: `enhanced_building_config` → `detailed_building_config`
   - Nettoyer commentaires

4. **`ign_lidar/cli/commands/migrate_config.py`**

   - Remplacer: "unified format" → "v3.2 format"

5. **Fichiers features/\*.py**
   - Nettoyer mentions "unified processor"
   - Remplacer par noms descriptifs

**Actions:**

- [ ] Recherche globale `(unified|enhanced|improved)` avec regex
- [ ] Renommer classes (EnhancedBuildingConfig, etc.)
- [ ] Renommer paramètres (enable_enhanced_lod3, etc.)
- [ ] Nettoyer commentaires et docstrings
- [ ] Mettre à jour exemples et documentation
- [ ] Tests de régression

---

## 📋 Phase 2 - IMPORTANT (Semaine 3-4)

### ✅ Tâche 2.1: Refactorer LiDARProcessor

**Priorité:** 🟠 MAJEUR  
**Effort:** 1 semaine  
**Statut:** ⚪ PLANIFIÉ

**Problème:**

- 3742 lignes (God Object anti-pattern)
- Trop de responsabilités
- Difficile à maintenir

**Solution - Décomposition:**

```python
# Architecture cible:
LiDARProcessor (API publique) - 400 lignes
├── TileOrchestrator - 300 lignes
│   ├── process_tile()
│   └── batch_process()
├── FeatureEngine - 250 lignes
│   ├── compute_features()
│   └── filter_features()
├── ClassificationEngine - 300 lignes
│   ├── classify_points()
│   └── refine_classification()
├── IOManager - 200 lignes
│   ├── load_tile()
│   └── save_results()
└── GroundTruthManager - 150 lignes
    ├── fetch_ground_truth()
    └── apply_ground_truth()
```

**Actions:**

- [ ] Extraire IOManager (load/save LAZ)
- [ ] Extraire GroundTruthManager (WFS operations)
- [ ] Extraire FeatureEngine (déléguer à orchestrator)
- [ ] Extraire ClassificationEngine (déléguer à classifier)
- [ ] Créer TileOrchestrator (coordination)
- [ ] Réduire LiDARProcessor à façade publique
- [ ] Tests d'intégration complets

---

### ✅ Tâche 2.2: Optimiser Gestion Mémoire GPU

**Priorité:** 🟡 MOYEN  
**Effort:** 3-5 jours  
**Statut:** ⚪ PLANIFIÉ

**Problèmes identifiés:**

1. **Imports répétés (50+ fois)**

   ```python
   # ❌ Actuel
   def func():
       import cupy as cp

   # ✅ Cible
   try:
       import cupy as cp
       HAS_CUPY = True
   except ImportError:
       cp = None
       HAS_CUPY = False
   ```

2. **Pas de Context Pooling**

   ```python
   # ✅ À implémenter
   class GPUContextPool:
       def __init__(self, max_contexts=4):
           self._pool = []
           self._max = max_contexts

       def acquire(self):
           if self._pool:
               return self._pool.pop()
           return GPUContext()

       def release(self, ctx):
           if len(self._pool) < self._max:
               self._pool.append(ctx)
   ```

3. **Transferts CPU↔GPU non optimisés**

   ```python
   # ✅ Batch processing
   def batch_process(data_chunks):
       # 1 transfert CPU→GPU
       gpu_data = cp.asarray(np.concatenate(data_chunks))
       # Processing
       results = process_gpu(gpu_data)
       # 1 transfert GPU→CPU
       return cp.asnumpy(results)
   ```

4. **Pas de streaming pour OOM**
   ```python
   # ✅ Chunked streaming
   def stream_process(large_data):
       for chunk in chunked(large_data, chunk_size):
           gpu_chunk = cp.asarray(chunk)
           yield process(gpu_chunk)
   ```

**Actions:**

- [ ] Créer `ign_lidar/core/gpu_pool.py` (Context pooling)
- [ ] Refactorer imports GPU (global avec fallback)
- [ ] Implémenter batch transferts CPU↔GPU
- [ ] Ajouter streaming pour grandes données
- [ ] Configurer CuPy memory pool limits
- [ ] Profiling GPU avec cupyx.profiler
- [ ] Benchmarks avant/après

---

## 📋 Phase 3 - SOUHAITABLE (Mois 2)

### ✅ Tâche 3.1: Réorganiser Architecture Processor

**Priorité:** 🟠 MAJEUR  
**Effort:** 2-3 semaines  
**Statut:** ⚪ PLANIFIÉ

**Classes actuelles (10):**

```
LiDARProcessor          - 3742 LOC
GPUProcessor            - 1668 LOC
ProcessorCore           - 737 LOC
TileProcessor           - 524 LOC
FacadeProcessor         - 1008 LOC
OptimizedProcessor      - 245 LOC
GeometricFeatureProcessor - 525 LOC
AsyncGPUProcessor       - 412 LOC
StreamingTileProcessor  - 398 LOC
ProcessorConfig         - Config
```

**Architecture cible (5 classes):**

```
LiDARProcessor          - 400 LOC (API publique)
TileOrchestrator        - 500 LOC (coordination tuiles)
FeatureComputer         - 600 LOC (features CPU+GPU)
ClassificationEngine    - 700 LOC (classification unifiée)
IOManager               - 300 LOC (I/O LAZ)
```

**Suppressions/Fusions:**

- ❌ ProcessorCore → fusionner dans LiDARProcessor
- ❌ OptimizedProcessor → fusionner dans LiDARProcessor
- ❌ GeometricFeatureProcessor → fusionner dans FeatureComputer
- ❌ AsyncGPUProcessor → intégrer dans GPUProcessor
- ❌ StreamingTileProcessor → intégrer dans TileOrchestrator
- ❌ FacadeProcessor → rester séparé (OK)
- ✅ TileProcessor → renommer TileOrchestrator
- ✅ GPUProcessor → garder mais réduire

**Actions:**

- [ ] Diagramme architecture cible (Mermaid)
- [ ] Fusion ProcessorCore → LiDARProcessor
- [ ] Fusion OptimizedProcessor
- [ ] Fusion GeometricFeatureProcessor → FeatureComputer
- [ ] Intégration AsyncGPU → GPUProcessor
- [ ] Renommage TileProcessor → TileOrchestrator
- [ ] Tests régression complets
- [ ] Documentation architecture

---

### ✅ Tâche 3.2: Améliorer Tests GPU

**Priorité:** 🟡 MOYEN  
**Effort:** 1 semaine  
**Statut:** ⚪ PLANIFIÉ

**Objectifs:**

- Couverture GPU: 70% → 90%+
- Ajouter mocks pour CI sans GPU
- Benchmarks performance GPU vs CPU

**Actions:**

- [ ] Tests compute_normals (CPU/GPU/chunked)
- [ ] Tests features (batch/streaming)
- [ ] Tests fallback GPU→CPU
- [ ] Tests OOM GPU (mémoire insuffisante)
- [ ] Mocks CuPy/cuML pour CI
- [ ] Benchmarks automated
- [ ] Documentation tests

---

## 📊 Métriques de Succès

| Métrique              | Avant | Cible | Après |
| --------------------- | ----- | ----- | ----- |
| compute_normals impl. | 10    | 2     | -     |
| Préfixes redondants   | 150+  | 0     | -     |
| LiDARProcessor LOC    | 3742  | <800  | -     |
| Classes Processor     | 10    | 5     | -     |
| Tests GPU coverage    | 70%   | 90%   | -     |
| GPU speedup           | 10x   | 20x   | -     |

---

## 🔄 Suivi Hebdomadaire

### Semaine 1 (25-29 Nov 2025)

- [ ] Tâche 1.1: Consolidation compute_normals (50%)
- [ ] Tâche 1.2: Nettoyage préfixes (0%)

### Semaine 2 (2-6 Dec 2025)

- [ ] Tâche 1.1: Consolidation compute_normals (100%)
- [ ] Tâche 1.2: Nettoyage préfixes (100%)

### Semaine 3-4 (9-20 Dec 2025)

- [ ] Tâche 2.1: Refactoring LiDARProcessor (50%)
- [ ] Tâche 2.2: Optimisation GPU (50%)

### Mois 2 (Jan 2026)

- [ ] Tâche 3.1: Réorganisation architecture
- [ ] Tâche 3.2: Tests GPU

---

## 🚀 Déploiement

### Prérequis avant merge:

- [ ] Tous les tests passent (CPU + GPU)
- [ ] Coverage >80%
- [ ] Documentation à jour
- [ ] CHANGELOG mis à jour
- [ ] Backward compatibility maintenue

### Migration utilisateurs:

- [ ] Guide migration v3.4 → v3.5
- [ ] Deprecation warnings (6 mois)
- [ ] Exemples mis à jour

---

## 📝 Notes

- **Backward compatibility:** Maintenir pendant 6 mois minimum
- **Tests:** Exécuter suite complète avant chaque merge
- **Documentation:** Mettre à jour en parallèle du code
- **Revue:** Code review systématique

---

**Dernière mise à jour:** 21 Novembre 2025  
**Prochaine revue:** 28 Novembre 2025
