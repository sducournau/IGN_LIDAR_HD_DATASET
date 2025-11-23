# ADR 002: Unified KNN Engine

**Date:** 2025-11-22  
**Status:** Accepted  
**Context:** Phase 3 Refactoring - Architecture Cleanup

## Context

KNN operations were implemented in multiple places:

- `gpu_accelerated_ops.py`: Custom GPU KNN
- `hybrid_formatter.py`: FAISS-based KNN
- `multi_arch_formatter.py`: sklearn KNN
- `feature_computer.py`: Custom KNN wrapper

This led to:
- Inconsistent APIs (different function signatures)
- Performance variations (some optimized, some not)
- Difficult to maintain and optimize
- No fallback mechanism

## Decision

Create **unified KNNEngine** with:

1. **Auto-backend selection**: FAISS-GPU → cuML → sklearn
2. **Consistent API**: Same interface regardless of backend
3. **Lazy GPU transfers**: `return_gpu` parameter to avoid unnecessary transfers
4. **Graceful fallbacks**: CPU fallback if GPU unavailable

```python
# Usage
knn_engine = KNNEngine(backend='auto', use_gpu=True)
knn_engine.build_index(points)
distances, indices = knn_engine.search(points, k=30, return_gpu=False)
```

## Consequences

### Positive

✅ Single KNN implementation to maintain  
✅ Automatic optimization (best backend)  
✅ Consistent performance characteristics  
✅ Easy to benchmark and optimize  
✅ Reduced GPU transfers (Phase 2 optimization)

### Negative

⚠️ Additional abstraction layer  
⚠️ Requires careful backend testing

## Implementation Details

### Backend Priority

1. **FAISS-GPU** (fastest for large datasets)
2. **cuML** (good GPU integration, requires RAPIDS)
3. **sklearn** (CPU fallback, always available)

### GPU Optimization

```python
# Without return_gpu (old way)
distances, indices = knn_engine.search(points_gpu, k=30)
# → GPU→CPU transfer happens here

# With return_gpu (optimized)
distances, indices = knn_engine.search(points_gpu, k=30, return_gpu=True)
# → Results stay on GPU, no transfer
```

## Alternatives Considered

1. **Keep separate implementations**: Rejected due to maintenance burden
2. **Hard-code FAISS**: Rejected due to inflexibility
3. **Plugin system**: Too complex for current needs

## Related

- ADR 001: Strategy Pattern
- Phase 2: GPU Transfer Optimization

## References

- FAISS: https://github.com/facebookresearch/faiss
- cuML: https://docs.rapids.ai/api/cuml/stable/
