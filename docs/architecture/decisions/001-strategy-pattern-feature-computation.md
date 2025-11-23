# ADR 001: Strategy Pattern for Feature Computation

**Date:** 2025-11-22  
**Status:** Accepted  
**Context:** Phase 3 Refactoring - Architecture Cleanup

## Context

The codebase had multiple implementations of feature computation scattered across
different classes (GPUProcessor, FeatureComputer, ProcessorCore, etc.), leading to:

- Code duplication (~11.7% of codebase)
- Inconsistent APIs
- Difficult maintenance (bugs fixed in one place but not others)
- Unclear separation of concerns

## Decision

Adopt the **Strategy Pattern** for feature computation with:

1. **FeatureOrchestrator**: Main interface for feature computation
2. **Strategy classes**: `CPUStrategy`, `GPUStrategy`, `GPUChunkedStrategy`
3. **Mode selector**: Automatic strategy selection based on config and hardware

```python
# Architecture
┌─────────────────────────┐
│  FeatureOrchestrator    │  ← Main API
├─────────────────────────┤
│ - config                │
│ - strategy (auto-select)│
└───────────┬─────────────┘
            │
            ├─→ CPUStrategy
            ├─→ GPUStrategy
            └─→ GPUChunkedStrategy
```

## Consequences

### Positive

✅ Single entry point for all feature computation  
✅ Easy to add new strategies (e.g., DistributedStrategy)  
✅ Automatic optimization based on hardware/data size  
✅ Clear separation of concerns  
✅ Easy to test each strategy independently

### Negative

⚠️ More classes than direct implementation  
⚠️ Slight overhead from strategy dispatch (negligible)

## Alternatives Considered

1. **Single monolithic class**: Too complex, hard to test
2. **Function-based approach**: Lost state management benefits
3. **Inheritance hierarchy**: Too rigid, violated Liskov substitution

## Related

- ADR 002: Unified KNN Engine
- ADR 003: Configuration System

## References

- [Strategy Pattern - Gang of Four](https://en.wikipedia.org/wiki/Strategy_pattern)
- Phase 3 Refactoring Plan: docs/TODO_REFACTORING.md
