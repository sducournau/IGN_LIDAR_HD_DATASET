---
sidebar_position: 5
---

mermaid-reference.md

# Mermaid Diagrams Reference

This page demonstrates the various Mermaid diagrams used throughout the IGN LiDAR HD documentation to visualize workflows, architectures, and processes.

## 🔄 Workflow Diagrams

### Basique Processing Flow

```mermaid
flowchart TD
    A[Start] --> B{Input Available?}
    B -->|Yes| C[Process Data]
    B -->|No| D[Download Data]
    D --> C
    C --> E[Generate Output]
    E --> F[End]

    style A fill:#e8f5e8
    style F fill:#e8f5e8
    style C fill:#e3f2fd
```

### Complex Pipeline

```mermaid
graph TB
    subgraph "Input Layer"
        I1[Raw LiDAR]
        I2[Configuration]
    end

    subgraph "Processing Layer"
        P1[Download]
        P2[Enrich]
        P3[Process]
    end

    subgraph "Output Layer"
        O1[NPZ Files]
        O2[Metadata]
    end

    I1 --> P1
    I2 --> P1
    P1 --> P2
    P2 --> P3
    P3 --> O1
    P3 --> O2

    style I1 fill:#ffebee
    style O1 fill:#e8f5e8
```

## 📊 Performance Charts

### Traitement Speed Comparison

```mermaid
xychart-beta
    title "Processing Performance by Hardware"
    x-axis [CPU-4core, CPU-8core, CPU-16core, GPU-RTX3080, GPU-RTX4090]
    y-axis "Tiles per Hour" 0 --> 100
    bar "Small Tiles" [8, 15, 25, 60, 85]
    bar "Large Tiles" [3, 6, 10, 25, 40]
```

### Memory Usage Over Time

```mermaid
xychart-beta
    title "Memory Usage During Processing"
    x-axis [0min, 1min, 2min, 3min, 4min, 5min]
    y-axis "Memory GB" 0 --> 16
    line "RAM Usage" [2, 4, 8, 12, 8, 4]
    line "GPU Memory" [0, 2, 6, 8, 6, 2]
```

## 🔀 Sequence Diagrams

### CLI Command Flow

```mermaid
sequenceDiagram
    participant U as User
    participant C as CLI
    participant D as Downloader
    participant P as Processor
    participant F as FileSystem

    U->>C: ign-lidar download
    C->>D: Initialize downloader
    D->>D: Query IGN WFS
    D->>F: Save LAZ files
    D-->>C: Files downloaded
    C-->>U: Success message

    U->>C: ign-lidar enrich
    C->>P: Initialize processor
    P->>F: Read LAZ files
    P->>P: Compute features
    P->>F: Save enriched LAZ
    P-->>C: Files enriched
    C-->>U: Success message
```

### Error Handling Flow

```mermaid
sequenceDiagram
    participant U as User
    participant S as System
    participant L as Logger
    participant E as ErrorHandler

    U->>S: Execute command
    S->>S: Process data

    alt Success
        S-->>U: Return result
    else Error occurs
        S->>E: Handle error
        E->>L: Log error details
        E->>E: Generate user message
        E-->>U: Error message

        opt Retry possible
            E->>U: Suggest retry
        end
    end
```

## 📈 Gantt Charts

### Project Timeline

```mermaid
gantt
    title IGN LiDAR Processing Timeline
    dateFormat X
    axisFormat %s

    section Download
    Tile Discovery   :0, 30
    File Download    :20, 120
    Validation      :100, 140

    section Enrichment
    Feature Setup   :120, 150
    GPU Processing  :140, 300
    LAZ Generation  :280, 320

    section ML Preparation
    Patch Extract   :300, 400
    Augmentation    :380, 420
    Dataset Ready   :420, 450
```

### Resource Utilization

```mermaid
gantt
    title System Resource Usage
    dateFormat X
    axisFormat %s

    section CPU Usage
    Preprocessing :0, 60
    Feature Calc  :40, 200
    Postprocess   :180, 240

    section GPU Usage
    Memory Alloc  :60, 80
    Computation   :80, 180
    Memory Free   :180, 200

    section Disk I/O
    Read Input    :0, 40
    Write Temp    :40, 160
    Write Output  :160, 240
```

## 🌐 State Diagrams

### Traitement States

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Downloading : start_download()
    Downloading --> Downloaded : success
    Downloaded --> Enriching : start_enrich()
    Enriching --> Enriched : success
    Enriched --> Processing : start_process()
    Processing --> Complete : success
    Complete --> [*]

    Downloading --> Error : failure
    Enriching --> Error : failure
    Processing --> Error : failure
    Error --> Idle : reset()
```

### Configuration States

```mermaid
stateDiagram-v2
    [*] --> Default
    Default --> CPU_Mode : set_cpu()
    Default --> GPU_Mode : set_gpu()
    CPU_Mode --> Multi_Worker : add_workers()
    GPU_Mode --> GPU_Optimized : optimize()
    Multi_Worker --> Processing
    GPU_Optimized --> Processing
    Processing --> [*] : complete
```

## 🔧 Entity Relationship Diagrams

### Data Model

```mermaid
erDiagram
    TILE ||--o{ PATCH : contains
    TILE {
        string name
        float bbox_min_x
        float bbox_min_y
        float bbox_max_x
        float bbox_max_y
        datetime download_date
        string file_path
    }

    PATCH ||--o{ POINT : contains
    PATCH {
        string id
        string tile_id
        float center_x
        float center_y
        float size_m
        string lod_level
        int num_points
    }

    POINT {
        float x
        float y
        float z
        float intensity
        int classification
        float normal_x
        float normal_y
        float normal_z
        float curvature
    }

    FEATURE_SET ||--|| POINT : describes
    FEATURE_SET {
        float planarity
        float verticality
        float horizontality
        float density
        string arch_style
    }
```

## 💡 Usage Tips

### Diagram Selection Guide

```mermaid
flowchart TD
    Start([Choose Diagram Type]) --> Purpose{What to Show?}

    Purpose -->|Process Flow| Flowchart[Use Flowchart]
    Purpose -->|System Architecture| Graph[Use Graph]
    Purpose -->|Performance Data| Chart[Use XY Chart]
    Purpose -->|Time Sequence| Sequence[Use Sequence]
    Purpose -->|Project Schedule| Gantt_D[Use Gantt]
    Purpose -->|System States| State[Use State Diagram]
    Purpose -->|Data Structure| Entity[Use ER Diagram]

    Flowchart --> Simple{Simple or Complex?}
    Simple -->|Simple| Basic[Basic Flowchart]
    Simple -->|Complex| Subgraph[Use Subgraphs]

    style Start fill:#e8f5e8
    style Flowchart fill:#e3f2fd
    style Graph fill:#fff3e0
    style Chart fill:#f3e5f5
```

### Color Scheme Guidelines

- 🟢 **Success/Completion**: `fill:#e8f5e8`
- 🔵 **Processing/Active**: `fill:#e3f2fd`
- 🟡 **Warning/Attention**: `fill:#fff3e0`
- 🟣 **Configuration**: `fill:#f3e5f5`
- 🔴 **Error/Problem**: `fill:#ffebee`

### Best Practices

1. **Keep diagrams focused** - One concept per diagram
2. **Use consistent styling** - Same colors for similar elements
3. **Add meaningful labels** - Clear, descriptive text
4. **Optimize for readability** - Not too cluttered
5. **Update regularly** - Keep diagrams current with code changes

