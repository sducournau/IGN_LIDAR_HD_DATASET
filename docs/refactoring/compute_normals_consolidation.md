# Consolidation de compute_normals - Documentation

**Date:** 21 Novembre 2025  
**Version:** 3.5.0-dev  
**Statut:** En cours

---

## 🎯 Objectif

Réduire de **10 implémentations** à **2 implémentations canoniques** de `compute_normals()`.

---

## 📊 État Actuel (AVANT)

### Implémentations Existantes

| #   | Fichier                | Fonction                                          | Type       | Lignes  | Statut               |
| --- | ---------------------- | ------------------------------------------------- | ---------- | ------- | -------------------- |
| 1   | `compute/normals.py`   | `compute_normals()`                               | CPU        | 28-94   | ✅ **CANONICAL CPU** |
| 2   | `compute/normals.py`   | `compute_normals_fast()`                          | CPU        | 177-203 | ⚠️ Variante → param  |
| 3   | `compute/normals.py`   | `compute_normals_accurate()`                      | CPU        | 203-233 | ⚠️ Variante → param  |
| 4   | `feature_computer.py`  | `FeatureComputer.compute_normals()`               | Délégation | 160-216 | ✅ Délègue déjà      |
| 5   | `feature_computer.py`  | `FeatureComputer.compute_normals_with_boundary()` | CPU        | 371-420 | ✅ Cas spécial OK    |
| 6   | `gpu_processor.py`     | `GPUProcessor.compute_normals()`                  | GPU        | 359-381 | ✅ **CANONICAL GPU** |
| 7   | `numba_accelerated.py` | `compute_normals_from_eigenvectors_numba()`       | CPU/Numba  | 174-212 | ❌ À supprimer       |
| 8   | `numba_accelerated.py` | `compute_normals_from_eigenvectors_numpy()`       | CPU        | 212-233 | ❌ À supprimer       |
| 9   | `numba_accelerated.py` | `compute_normals_from_eigenvectors()`             | CPU        | 233-250 | ❌ À supprimer       |
| 10  | `gpu_kernels.py`       | `compute_normals_and_eigenvalues()`               | GPU        | 439-500 | ✅ GPU low-level OK  |

---

## ✅ Architecture Cible (APRÈS)

### 2 Implémentations Canoniques

#### 1. **CPU Canonical**: `ign_lidar/features/compute/normals.py::compute_normals()`

```python
def compute_normals(
    points: np.ndarray,
    k_neighbors: int = 20,
    search_radius: Optional[float] = None,
    method: str = 'standard',  # 'fast' | 'accurate' | 'standard'
    with_boundary: bool = False,
    boundary_points: Optional[np.ndarray] = None,
    return_eigenvalues: bool = True,
    use_gpu: bool = False,  # Dispatch to GPU if available
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    CANONICAL implementation for CPU normal computation.

    All other CPU implementations should call this function.

    Parameters
    ----------
    points : np.ndarray
        Point cloud [N, 3]
    k_neighbors : int
        Number of neighbors (default: 20)
    method : str
        'fast' (k=10), 'accurate' (k=50), 'standard' (use k_neighbors)
    with_boundary : bool
        Use boundary points for edge cases
    boundary_points : np.ndarray, optional
        Buffer points for boundary support
    return_eigenvalues : bool
        Return eigenvalues along with normals
    use_gpu : bool
        If True and GPU available, dispatch to GPU implementation

    Returns
    -------
    normals : np.ndarray
        Normal vectors [N, 3], unit length
    eigenvalues : np.ndarray or None
        Eigenvalues [N, 3] if return_eigenvalues=True
    """
    # GPU dispatch
    if use_gpu and GPU_AVAILABLE:
        from ..gpu_processor import GPUProcessor
        gpu_proc = GPUProcessor()
        return gpu_proc.compute_normals(points, k=k_neighbors)

    # Method selection
    if method == 'fast':
        k_neighbors = 10
    elif method == 'accurate':
        k_neighbors = 50

    # Boundary handling
    if with_boundary and boundary_points is not None:
        all_points = np.vstack([points, boundary_points])
        normals, eigenvalues = _compute_normals_cpu(all_points, k_neighbors)
        return normals[:len(points)], eigenvalues[:len(points)] if eigenvalues is not None else None

    # Standard computation
    return _compute_normals_cpu(points, k_neighbors, search_radius)
```

#### 2. **GPU Canonical**: `ign_lidar/optimization/gpu_kernels.py::compute_normals_and_eigenvalues()`

```python
def compute_normals_and_eigenvalues(
    points_gpu: cp.ndarray,
    k_neighbors: int = 20,
    return_normals: bool = True,
    return_eigenvalues: bool = True,
) -> Tuple[Optional[cp.ndarray], Optional[cp.ndarray]]:
    """
    CANONICAL implementation for GPU normal computation.

    Low-level GPU kernel using CuPy/FAISS.
    All GPU implementations should call this function.

    Parameters
    ----------
    points_gpu : cp.ndarray
        Point cloud on GPU [N, 3]
    k_neighbors : int
        Number of neighbors
    return_normals : bool
        Compute and return normals
    return_eigenvalues : bool
        Compute and return eigenvalues

    Returns
    -------
    normals_gpu : cp.ndarray or None
        Normal vectors on GPU [N, 3]
    eigenvalues_gpu : cp.ndarray or None
        Eigenvalues on GPU [N, 3]
    """
    # FAISS k-NN search on GPU
    # Compute covariance matrices
    # Eigendecomposition with cuSolver
    # Extract normals and eigenvalues
    ...
```

---

## 🔄 Plan de Migration

### Phase 1: Dépréciation (v3.5.0 - Décembre 2025)

1. **Ajouter warnings de dépréciation:**

   ```python
   # compute/normals.py
   def compute_normals_fast(points: np.ndarray) -> np.ndarray:
       """DEPRECATED: Use compute_normals(method='fast') instead."""
       warnings.warn(
           "compute_normals_fast() is deprecated, use compute_normals(method='fast')",
           DeprecationWarning,
           stacklevel=2
       )
       return compute_normals(points, method='fast', return_eigenvalues=False)[0]
   ```

2. **Marquer fonctions numba comme deprecated:**
   ```python
   # numba_accelerated.py
   @deprecated(version='3.5.0', reason='Use compute.normals.compute_normals()')
   def compute_normals_from_eigenvectors_numba(...):
       ...
   ```

### Phase 2: Refactoring (v3.5.0 - Janvier 2026)

1. **Mettre à jour `FeatureComputer.compute_normals()`:**

   - ✅ Délègue déjà correctement
   - Vérifier tous les appels

2. **Mettre à jour `GPUProcessor.compute_normals()`:**

   - Appeler `gpu_kernels.compute_normals_and_eigenvalues()`
   - Ajouter gestion mémoire

3. **Supprimer duplications numba:**
   - Garder seulement le wrapper si nécessaire
   - Rediriger vers compute/normals

### Phase 3: Suppression (v4.0.0 - Mars 2026)

1. **Supprimer fonctions dépréciées:**

   - `compute_normals_fast()`
   - `compute_normals_accurate()`
   - `compute_normals_from_eigenvectors_*()`

2. **Nettoyer imports:**
   - Mettre à jour tous les imports
   - Tests de régression

---

## 🧪 Tests Requis

### Tests Unitaires

```python
def test_compute_normals_cpu():
    """Test CPU canonical implementation."""
    points = np.random.rand(1000, 3)
    normals, eigenvalues = compute_normals(points, k_neighbors=20)
    assert normals.shape == (1000, 3)
    assert eigenvalues.shape == (1000, 3)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)

def test_compute_normals_method_fast():
    """Test fast method (k=10)."""
    points = np.random.rand(1000, 3)
    normals, _ = compute_normals(points, method='fast')
    # Should use k=10 internally
    assert normals.shape == (1000, 3)

def test_compute_normals_method_accurate():
    """Test accurate method (k=50)."""
    points = np.random.rand(1000, 3)
    normals, _ = compute_normals(points, method='accurate')
    # Should use k=50 internally
    assert normals.shape == (1000, 3)

def test_compute_normals_with_boundary():
    """Test boundary handling."""
    core_points = np.random.rand(1000, 3)
    buffer_points = np.random.rand(200, 3)
    normals, _ = compute_normals(
        core_points,
        with_boundary=True,
        boundary_points=buffer_points
    )
    assert normals.shape == (1000, 3)  # Only core points returned

@pytest.mark.gpu
def test_compute_normals_gpu():
    """Test GPU canonical implementation."""
    if not GPU_AVAILABLE:
        pytest.skip("GPU not available")

    points = np.random.rand(10000, 3)
    normals, eigenvalues = compute_normals(points, use_gpu=True)
    assert normals.shape == (10000, 3)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)

def test_compute_normals_gpu_cpu_consistency():
    """Test CPU and GPU produce similar results."""
    if not GPU_AVAILABLE:
        pytest.skip("GPU not available")

    points = np.random.rand(1000, 3)
    normals_cpu, _ = compute_normals(points, use_gpu=False)
    normals_gpu, _ = compute_normals(points, use_gpu=True)

    # Should be very similar (allow small numerical differences)
    assert np.allclose(normals_cpu, normals_gpu, atol=1e-3)
```

### Tests de Régression

```python
def test_backward_compatibility_fast():
    """Test deprecated compute_normals_fast() still works."""
    points = np.random.rand(1000, 3)
    with pytest.warns(DeprecationWarning):
        normals = compute_normals_fast(points)
    assert normals.shape == (1000, 3)

def test_feature_computer_normals():
    """Test FeatureComputer.compute_normals() delegation."""
    computer = FeatureComputer()
    points = np.random.rand(1000, 3)
    normals = computer.compute_normals(points, k=20)
    assert normals.shape == (1000, 3)
```

---

## 📝 Documentation à Mettre à Jour

### 1. Guides Utilisateurs

- [ ] `docs/features/normals.md` - Guide compute_normals
- [ ] `docs/migration/v3.5_changes.md` - Migration guide
- [ ] `README.md` - Exemples d'utilisation

### 2. API Reference

- [ ] Docstrings compute_normals()
- [ ] Docstrings GPUProcessor.compute_normals()
- [ ] Exemples dans docstrings

### 3. CHANGELOG

```markdown
## [3.5.0] - 2026-01-15

### Changed

- **BREAKING**: Consolidated compute_normals() implementations
  - Use `compute_normals(method='fast')` instead of `compute_normals_fast()`
  - Use `compute_normals(method='accurate')` instead of `compute_normals_accurate()`
  - Deprecated functions in numba_accelerated.py

### Deprecated

- `compute_normals_fast()` - Use `compute_normals(method='fast')`
- `compute_normals_accurate()` - Use `compute_normals(method='accurate')`
- `compute_normals_from_eigenvectors_*()` - Use canonical implementations

### Removed (planned for v4.0.0)

- All deprecated compute_normals variants will be removed in v4.0.0
```

---

## 🎯 Résultat Attendu

### Avant (10 implémentations)

```
compute_normals: 10 versions
Maintenance: Difficile
Cohérence: Risquée
Tests: Fragmentés
```

### Après (2 implémentations + wrappers)

```
compute_normals: 2 versions canoniques
Maintenance: Simple
Cohérence: Garantie
Tests: Centralisés
```

### Métriques de Succès

| Métrique          | Avant   | Après   | Amélioration |
| ----------------- | ------- | ------- | ------------ |
| Implémentations   | 10      | 2       | -80%         |
| Lignes de code    | ~800    | ~300    | -62%         |
| Tests requis      | 40+     | 12      | -70%         |
| Temps maintenance | 8h/mois | 2h/mois | -75%         |

---

## ⚠️ Risques et Mitigation

### Risque 1: Breaking Changes

**Mitigation:**

- Deprecation warnings pendant 6 mois
- Backward compatibility wrappers
- Guide migration détaillé

### Risque 2: Performance Regression

**Mitigation:**

- Benchmarks avant/après
- Tests performance automatisés
- Profiling CPU et GPU

### Risque 3: Différences CPU/GPU

**Mitigation:**

- Tests de cohérence CPU↔GPU
- Tolérance numérique appropriée
- Documentation des différences

---

**Prochaines Étapes:**

1. ✅ Audit complet (fait)
2. ⏳ Ajouter deprecation warnings
3. ⏳ Implémenter tests
4. ⏳ Mettre à jour documentation
5. ⏳ Review et merge

**Date de revue:** 28 Novembre 2025
