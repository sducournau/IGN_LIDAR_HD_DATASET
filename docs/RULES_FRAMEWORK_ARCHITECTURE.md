# Rules Framework Architecture

> **Visual guide to the IGN LiDAR HD Rules Framework (v3.2.0)**  
> System architecture, data flow, and component interactions

---

## 📐 System Architecture

```mermaid
graph TB
    subgraph "User Code"
        A[Point Cloud Data]
        B[Classification Labels]
        C[Custom Rules]
    end

    subgraph "Rules Framework Core"
        D[RuleContext]
        E[RuleEngine]
        F[Rule Validation]
        G[Confidence Calculation]
        H[Conflict Resolution]
        I[ExecutionResult]
    end

    subgraph "Rule Implementations"
        J[GeometricRule]
        K[SpectralRule]
        L[ContextualRule]
        M[HybridRule]
    end

    A --> D
    B --> D
    C --> E
    D --> E
    E --> F
    F --> J
    F --> K
    F --> L
    F --> M
    J --> G
    K --> G
    L --> G
    M --> G
    G --> H
    H --> I
    I --> B

    style E fill:#4A90E2,stroke:#333,stroke-width:3px,color:#fff
    style I fill:#50C878,stroke:#333,stroke-width:2px,color:#fff
    style D fill:#FFA500,stroke:#333,stroke-width:2px,color:#fff
```

---

## 🔄 Data Flow: Sequential Execution

```mermaid
sequenceDiagram
    participant User
    participant Engine as RuleEngine
    participant Context as RuleContext
    participant Rule1
    participant Rule2
    participant Confidence
    participant Resolver as ConflictResolver

    User->>Engine: execute(points, labels)
    Engine->>Context: create(points, labels, features)

    Note over Engine: Sort rules by priority

    Engine->>Rule1: validate_features(context)
    Rule1-->>Engine: True
    Engine->>Rule1: evaluate(context)
    Rule1->>Confidence: calculate_confidence()
    Confidence-->>Rule1: confidence_scores
    Rule1-->>Engine: RuleResult

    Engine->>Rule2: validate_features(context)
    Rule2-->>Engine: True
    Engine->>Rule2: evaluate(context)
    Rule2->>Confidence: calculate_confidence()
    Confidence-->>Rule2: confidence_scores
    Rule2-->>Engine: RuleResult

    Note over Engine: Conflicts detected

    Engine->>Resolver: resolve_conflicts()
    Resolver-->>Engine: final_classifications

    Engine-->>User: ExecutionResult
```

---

## 🏗️ Hierarchical Execution Flow

```mermaid
graph LR
    subgraph "Level 1: Coarse Classification"
        A[Input Points] --> B[Ground Rule]
        A --> C[Non-Ground Rule]
        B --> D[Ground Points]
        C --> E[Non-Ground Points]
    end

    subgraph "Level 2: Fine Classification"
        D --> F[Low Vegetation]
        E --> G[Building Rule]
        E --> H[Vegetation Rule]
        E --> I[Power Line Rule]
        G --> J[Buildings]
        H --> K[Medium/High Veg]
        I --> L[Power Lines]
    end

    subgraph "Level 3: Refinement"
        J --> M[Roof Refinement]
        K --> N[Tree Detection]
        L --> O[Wire Geometry]
        M --> P[Final Buildings]
        N --> Q[Final Vegetation]
        O --> R[Final Power Lines]
    end

    style A fill:#FFA500,stroke:#333,stroke-width:3px
    style P fill:#50C878,stroke:#333,stroke-width:2px
    style Q fill:#50C878,stroke:#333,stroke-width:2px
    style R fill:#50C878,stroke:#333,stroke-width:2px
```

---

## 🎯 Component Relationships

```mermaid
classDiagram
    class BaseRule {
        <<abstract>>
        +name: str
        +rule_type: RuleType
        +priority: RulePriority
        +validate_features(context) bool
        +evaluate(context) RuleResult*
        +get_required_features() List
    }

    class RuleEngine {
        -rules: List[BaseRule]
        -strategy: ExecutionStrategy
        -conflict_resolution: ConflictResolution
        +add_rule(rule)
        +remove_rule(name)
        +execute(points, labels) ExecutionResult
        -_resolve_conflicts()
        -_apply_results()
    }

    class HierarchicalRuleEngine {
        -levels: List[RuleLevel]
        -level_strategy: str
        +add_level(level)
        +execute_hierarchical() ExecutionResult
        -_execute_level()
    }

    class RuleContext {
        +points: ndarray
        +labels: ndarray
        +additional_features: Dict
        +metadata: Dict
    }

    class RuleResult {
        +point_indices: ndarray
        +classifications: ndarray
        +confidence_scores: ndarray
        +metadata: Dict
    }

    class ExecutionResult {
        +updated_labels: ndarray
        +points_modified: int
        +rules_applied: int
        +execution_time: float
        +statistics: Dict
    }

    BaseRule <|-- GeometricRule
    BaseRule <|-- SpectralRule
    BaseRule <|-- ContextualRule
    RuleEngine o-- BaseRule
    HierarchicalRuleEngine --|> RuleEngine
    RuleEngine ..> RuleContext
    BaseRule ..> RuleContext
    BaseRule ..> RuleResult
    RuleEngine ..> ExecutionResult
```

---

## 🔢 Confidence Calculation Pipeline

```mermaid
graph TB
    A[Feature Values] --> B{Choose Method}

    B -->|Binary| C[Binary Threshold]
    B -->|Linear| D[Linear Ramp]
    B -->|Sigmoid| E[Sigmoid Curve]
    B -->|Gaussian| F[Gaussian Peak]
    B -->|Range| G[Threshold Range]
    B -->|Exponential| H[Exponential Decay]
    B -->|Composite| I[Multi-Feature]

    C --> J[Confidence Scores]
    D --> J
    E --> J
    F --> J
    G --> J
    H --> J
    I --> J

    subgraph "Combination Strategies"
        K[Multiple Rules] --> L{Strategy}
        L -->|Max| M[Highest Confidence]
        L -->|Average| N[Mean Confidence]
        L -->|Weighted| O[Weighted Mean]
        L -->|Min| P[Lowest Confidence]
        L -->|Product| Q[Product]
        L -->|Weighted Product| R[Weighted Product]
    end

    J --> K
    M --> S[Final Confidence]
    N --> S
    O --> S
    P --> S
    Q --> S
    R --> S

    style A fill:#FFA500,stroke:#333,stroke-width:2px
    style J fill:#4A90E2,stroke:#333,stroke-width:2px,color:#fff
    style S fill:#50C878,stroke:#333,stroke-width:3px,color:#fff
```

---

## 🔀 Conflict Resolution Strategies

```mermaid
graph TD
    A[Multiple Rules Classify Same Point] --> B{Conflict Resolution Strategy}

    B -->|HIGHEST_CONFIDENCE| C[Rule with Max Confidence Wins]
    B -->|FIRST_WINS| D[First Rule Applied Wins]
    B -->|LAST_WINS| E[Last Rule Applied Wins]
    B -->|MAJORITY_VOTE| F[Most Common Classification]
    B -->|WEIGHTED_VOTE| G[Weighted by Confidence]

    C --> H[Final Classification]
    D --> H
    E --> H
    F --> H
    G --> H

    subgraph "Example: Point classified by 3 rules"
        I["Rule 1: Building (0.8)"]
        J["Rule 2: Vegetation (0.6)"]
        K["Rule 3: Building (0.9)"]
    end

    I --> B
    J --> B
    K --> B

    style A fill:#FFA500,stroke:#333,stroke-width:2px
    style H fill:#50C878,stroke:#333,stroke-width:3px,color:#fff
```

---

## 📊 Rule Execution States

```mermaid
stateDiagram-v2
    [*] --> Initialization
    Initialization --> FeatureValidation: validate_features()

    FeatureValidation --> Skipped: Missing features
    FeatureValidation --> Evaluation: Features OK

    Evaluation --> ConfidenceCalc: evaluate()
    ConfidenceCalc --> ResultCreation: calculate_confidence()

    ResultCreation --> ResultValidation: RuleResult
    ResultValidation --> Failed: Invalid result
    ResultValidation --> Success: Valid result

    Success --> ConflictResolution: Multiple rules
    Success --> DirectApplication: Single rule

    ConflictResolution --> FinalUpdate
    DirectApplication --> FinalUpdate
    Failed --> [*]
    Skipped --> [*]
    FinalUpdate --> [*]

    note right of FeatureValidation
        Checks required features
        are present in context
    end note

    note right of ConflictResolution
        Resolves overlapping
        classifications
    end note
```

---

## 🏢 Module Organization

```
ign_lidar/core/classification/rules/
│
├── __init__.py              # Public API exports (40+ items)
│   ├── BaseRule
│   ├── RuleEngine
│   ├── HierarchicalRuleEngine
│   ├── RuleContext, RuleResult, ExecutionResult
│   ├── All enums and dataclasses
│   └── All functions from submodules
│
├── base.py                  # Core abstractions (513 lines)
│   ├── BaseRule (abstract)
│   ├── RuleEngine
│   ├── HierarchicalRuleEngine
│   ├── Dataclasses: RuleContext, RuleResult, ExecutionResult, RuleLevel
│   └── Enums: RuleType, RulePriority, ExecutionStrategy, etc.
│
├── validation.py            # Feature/result validation (339 lines)
│   ├── validate_rule_context()
│   ├── validate_rule_result()
│   ├── validate_feature_array()
│   ├── check_required_features()
│   └── FeatureRequirements dataclass
│
├── confidence.py            # Confidence calculations (347 lines)
│   ├── calculate_confidence() - 7 methods
│   ├── combine_confidences() - 6 strategies
│   └── ConfidenceMethod enum
│
└── hierarchy.py             # Multi-level execution (346 lines)
    ├── HierarchicalRuleEngine (enhanced)
    ├── RuleLevel dataclass
    └── Level execution strategies
```

---

## 🚀 Execution Performance Model

```mermaid
graph LR
    subgraph "Input Scale"
        A1[Small: <100K points]
        A2[Medium: 100K-1M]
        A3[Large: 1M-10M]
        A4[Huge: >10M]
    end

    subgraph "Optimization Strategy"
        B1[Sequential]
        B2[Vectorized]
        B3[Chunked]
        B4[GPU-accelerated]
    end

    subgraph "Expected Performance"
        C1[<1s]
        C2[1-10s]
        C3[10-60s]
        C4[1-10min]
    end

    A1 --> B1
    A2 --> B2
    A3 --> B3
    A4 --> B4

    B1 --> C1
    B2 --> C2
    B3 --> C3
    B4 --> C4

    style A4 fill:#FF6B6B,stroke:#333,stroke-width:2px,color:#fff
    style B4 fill:#4A90E2,stroke:#333,stroke-width:2px,color:#fff
    style C1 fill:#50C878,stroke:#333,stroke-width:2px,color:#fff
```

---

## 🔧 Typical Usage Patterns

### Pattern 1: Simple Sequential Classification

```
User creates rules → Engine sorts by priority → Sequential execution →
Confidence calculation → Conflict resolution → Updated labels
```

### Pattern 2: Hierarchical Classification

```
Level 1 (Coarse): Ground/Non-ground separation →
Level 2 (Medium): Building/Veg/Other detection →
Level 3 (Fine): Detailed sub-classification →
Final refinement → Updated labels
```

### Pattern 3: Multi-Feature Hybrid

```
Geometric features → Geometric rules →
Spectral features → Spectral rules →
Spatial context → Contextual rules →
Confidence combination → Conflict resolution → Updated labels
```

---

## 💡 Design Principles

```mermaid
mindmap
    root((Rules Framework))
        Extensibility
            Plugin architecture
            Custom rules easy
            No framework edits
        Type Safety
            Dataclasses
            Type hints
            Validation
        Performance
            NumPy vectorization
            Lazy evaluation
            Caching support
        Usability
            Clear API
            Comprehensive docs
            Working examples
        Reliability
            Input validation
            Error handling
            Statistics tracking
        Composability
            Rules combine
            Hierarchical levels
            Modular design
```

---

## 📈 Confidence Score Visualization

```
Binary Method:
          1.0 ┤     ████████████
              │     █
              │     █
          0.0 ┼█████
              ├─────┬─────┬─────
              0   threshold  max

Linear Method:
          1.0 ┤         ████████
              │       ██
              │     ██
          0.0 ┼█████
              ├─────┬─────┬─────
              0   threshold  max

Sigmoid Method:
          1.0 ┤        ███████
              │      ██
              │    ██
              │  ██
          0.0 ┼██
              ├─────┬─────┬─────
              0   threshold  max

Gaussian Method:
          1.0 ┤      ███
              │    ███████
              │  ███     ███
          0.0 ┼██         ██
              ├─────┬─────┬─────
              0   threshold  max
```

---

## 🎯 Integration Points

### With Existing Classification Pipeline

```mermaid
graph LR
    A[Point Cloud] --> B[Feature Computation]
    B --> C[Initial Classification]
    C --> D{Use Rules?}

    D -->|Yes| E[Rules Framework]
    D -->|No| F[Direct Output]

    E --> G[RuleEngine.execute]
    G --> H[Refined Classifications]

    F --> I[Final Output]
    H --> I

    style E fill:#4A90E2,stroke:#333,stroke-width:3px,color:#fff
    style G fill:#50C878,stroke:#333,stroke-width:2px,color:#fff
```

### With Ground Truth Data

```mermaid
graph TB
    A[Ground Truth Polygons] --> B[ContextualRule]
    C[Point Cloud] --> D[RuleContext]
    B --> E[Spatial Index<br/>STRtree]
    D --> B
    E --> F[Containment Tests]
    F --> G[High Confidence<br/>Classifications]

    style B fill:#4A90E2,stroke:#333,stroke-width:2px,color:#fff
    style E fill:#FFA500,stroke:#333,stroke-width:2px,color:#fff
```

---

## 🔍 Debugging & Monitoring

### Execution Statistics Flow

```mermaid
graph TD
    A[Rule Execution] --> B[Collect Metrics]
    B --> C{Statistics}

    C --> D[Points Modified]
    C --> E[Execution Time]
    C --> F[Rules Applied]
    C --> G[Conflicts Resolved]
    C --> H[Validation Errors]

    D --> I[ExecutionResult]
    E --> I
    F --> I
    G --> I
    H --> I

    I --> J[User Analysis]
    J --> K[Performance Tuning]
    J --> L[Rule Refinement]
    J --> M[Error Investigation]

    style I fill:#50C878,stroke:#333,stroke-width:2px,color:#fff
```

---

## 📚 Learning Path Flow

```mermaid
graph LR
    A[Start] --> B[Quick Reference]
    B --> C[Simple Example]
    C --> D[Run Demo Scripts]
    D --> E[Read Dev Guide]
    E --> F[Create Custom Rule]
    F --> G{Works?}

    G -->|No| H[Troubleshooting]
    H --> E

    G -->|Yes| I[Add Confidence]
    I --> J[Combine Rules]
    J --> K[Hierarchical Engine]
    K --> L[Production Use]

    style A fill:#FFA500,stroke:#333,stroke-width:2px
    style L fill:#50C878,stroke:#333,stroke-width:3px,color:#fff
    style H fill:#FF6B6B,stroke:#333,stroke-width:2px,color:#fff
```

---

## 🎓 Documentation Map

```
📚 Rules Framework Documentation

├── RULES_FRAMEWORK_QUICK_REFERENCE.md
│   └── One-page reference for rapid development
│
├── RULES_FRAMEWORK_DEVELOPER_GUIDE.md (this file)
│   ├── Complete API documentation
│   ├── Step-by-step tutorials
│   ├── Best practices
│   └── Troubleshooting
│
├── RULES_FRAMEWORK_ARCHITECTURE.md
│   ├── Visual diagrams (this file)
│   ├── Component relationships
│   └── Data flow charts
│
├── examples/README_RULES_EXAMPLES.md
│   ├── Usage examples
│   ├── Code walkthroughs
│   └── Real-world scenarios
│
└── examples/demo_*.py
    ├── demo_custom_geometric_rule.py
    ├── demo_hierarchical_rules.py
    └── demo_confidence_scoring.py
```

---

## 🔗 Related Documentation

- **Quick Reference**: `RULES_FRAMEWORK_QUICK_REFERENCE.md` - One-page API reference
- **Developer Guide**: `RULES_FRAMEWORK_DEVELOPER_GUIDE.md` - Comprehensive documentation
- **Examples Guide**: `examples/README_RULES_EXAMPLES.md` - Working code examples
- **Demo Scripts**: `examples/demo_*.py` - Executable demonstrations
- **Project Summary**: `PROJECT_CONSOLIDATION_SUMMARY.md` - Overall consolidation status

---

## 📞 Support & Resources

- **Documentation**: [sducournau.github.io/IGN_LIDAR_HD_DATASET](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/)
- **Issues**: [github.com/sducournau/IGN_LIDAR_HD_DATASET/issues](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
- **Source Code**: `ign_lidar/core/classification/rules/`

---

**Version**: 3.2.1 | **Date**: October 23, 2025 | **License**: MIT
