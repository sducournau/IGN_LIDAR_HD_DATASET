# ADR 003: Hydra-Based Configuration System

**Date:** 2025-11-22  
**Status:** Accepted  
**Context:** v3.0 Major Refactoring

## Context

Previous configuration system:

- Scattered parameters across multiple classes
- No validation
- Difficult to override for experiments
- No clear configuration hierarchy
- Hard-coded defaults

## Decision

Adopt **Hydra** configuration framework with:

1. **Hierarchical configs**: YAML-based, composable
2. **Schema validation**: Pydantic-based validation
3. **Override mechanism**: CLI overrides for experiments
4. **Typed configs**: Full type safety

```yaml
# config.yaml
processor:
  lod_level: LOD2
  use_gpu: true

features:
  mode: lod2
  k_neighbors: 30
  
data_sources:
  bd_topo:
    buildings: true
```

## Consequences

### Positive

✅ Clear configuration hierarchy  
✅ Easy to create experiment configs  
✅ Validation prevents errors  
✅ CLI overrides: `python main.py features.k_neighbors=50`  
✅ Config composition and inheritance

### Negative

⚠️ Learning curve for Hydra  
⚠️ More files (config YAMLs)

## Implementation

### Config Schema

```python
@dataclass
class ProcessorConfig:
    lod_level: str = "LOD2"
    use_gpu: bool = True
    patch_size: float = 150.0
```

### Validation

```python
class ConfigValidator:
    def validate(self, config: Dict) -> bool:
        # Validate structure, types, ranges
        pass
```

## Migration Path

1. v3.0-3.5: Both old and new config systems supported
2. v3.6+: Hydra config recommended, old system deprecated
3. v4.0: Old config system removed

## Related

- ADR 001: Strategy Pattern (uses config for mode selection)

## References

- Hydra: https://hydra.cc/
- Pydantic: https://pydantic-docs.helpmanual.io/
