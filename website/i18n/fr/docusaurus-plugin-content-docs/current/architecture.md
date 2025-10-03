---
sidebar_position: 3
---

# System Architecture

Understanding the library's architecture helps you make the most of its capabilities and customize it for your specific needs.

## 🏗️ Core Architecture

```mermaid
graph TB
    subgraph "User Interface Layer"
        CLI[Command Line Interface]
        API[Python API]
    end

    subgraph "Processing Core"
        PROC[LiDAR Processor]
        FEAT[Feature Engine]
        GPU[GPU Accelerator]
    end

    subgraph "Data Management"
        DOWN[IGN Downloader]
        TILE[Tile Manager]
        META[Metadata Store]
    end

    subgraph "Classification Layer"
        LOD2[LOD2 Schema<br/>15 Classes]
        LOD3[LOD3 Schema<br/>30+ Classes]
        ARCH[Architectural Styles]
    end

    subgraph "Output Formats"
        NPZ[NPZ Patches]
        LAZ[Enriched LAZ]
        QGIS[QGIS Compatible]
    end

    CLI --> PROC
    API --> PROC
    PROC --> FEAT
    PROC --> DOWN
    FEAT --> GPU
    FEAT --> LOD2
    FEAT --> LOD3
    FEAT --> ARCH
    DOWN --> TILE
    DOWN --> META
    PROC --> NPZ
    PROC --> LAZ
    PROC --> QGIS

    style CLI fill:#e3f2fd
    style API fill:#e3f2fd
    style PROC fill:#fff3e0
    style FEAT fill:#e8f5e8
    style GPU fill:#ffebee
```

## 🔄 Data Flow Architecture

```mermaid
sequenceDiagram
    participant U as User
    participant C as CLI/API
    participant D as Downloader
    participant F as Feature Engine
    participant P as Processor
    participant S as Storage

    U->>C: Request processing
    C->>D: Download tiles
    D->>D: Check existing files
    D->>S: Store raw LAZ
    D-->>C: Tiles available

    C->>F: Enrich with features
    F->>F: Compute normals
    F->>F: Extract curvature
    F->>F: Analyze geometry
    F->>S: Store enriched LAZ
    F-->>C: Features ready

    C->>P: Create patches
    P->>P: Extract patches
    P->>P: Apply augmentation
    P->>P: Assign LOD labels
    P->>S: Store NPZ patches
    P-->>C: Dataset ready
    C-->>U: Processing complete
```

## 🧩 Component Details

### Core Processor

The `LiDARProcessor` class orchestrates the entire pipeline:

- Manages workflow execution
- Handles parallel processing
- Coordinates smart skip detection
- Applies data augmentation

### Feature Engine

Advanced geometric analysis:

- Surface normal computation
- Principal curvature calculation
- Planarity and verticality measures
- Local density estimation
- Architectural style inference

### Smart Skip System

Intelligent workflow resumption:

- File existence checking
- Metadata validation
- Timestamp comparison
- Progress tracking

### GPU Acceleration (New in v1.5.0)

Optional CUDA acceleration for:

- K-nearest neighbor searches
- Matrix operations
- Feature computations
- **RGB color interpolation (24x faster)** 🆕
- **GPU memory caching for RGB tiles** 🆕
- Large dataset processing

:::tip Learn More
See [GPU Acceleration Guide](gpu/overview.md) for complete setup instructions and [GPU RGB Guide](gpu/rgb-augmentation.md) for RGB-specific details.
:::

#### GPU RGB Pipeline

```mermaid
flowchart LR
    A[Points] --> B[GPU Transfer]
    B --> C[Features GPU]
    C --> D[RGB Cache GPU]
    D --> E[Color Interpolation GPU]
    E --> F[Combined Results]
    F --> G[CPU Transfer]

    style B fill:#c8e6c9
    style C fill:#c8e6c9
    style D fill:#c8e6c9
    style E fill:#c8e6c9
    style F fill:#c8e6c9
```

**Performance:** 24x speedup for RGB augmentation (v1.5.0)

## 📊 Performance Characteristics

```mermaid
graph LR
    subgraph "Processing Speed"
        CPU[CPU Mode<br/>~1-2 tiles/min]
        GPU_ACC[GPU Mode<br/>~5-10 tiles/min]
    end

    subgraph "Memory Usage"
        SMALL[Small Tiles<br/>~512MB RAM]
        LARGE[Large Tiles<br/>~2-4GB RAM]
    end

    subgraph "Output Size"
        INPUT[Raw LAZ<br/>~50-200MB]
        OUTPUT[Enriched LAZ<br/>~80-300MB]
        PATCHES[NPZ Patches<br/>~10-50MB each]
    end

    style GPU_ACC fill:#e8f5e8
    style LARGE fill:#fff3e0
    style OUTPUT fill:#e3f2fd
```

## 🔧 Configuration System

The library uses a hierarchical configuration approach:

1. **Default Settings** - Built-in optimal defaults
2. **Configuration Files** - Project-specific settings
3. **Environment Variables** - Runtime overrides
4. **Command Arguments** - Immediate parameters

### Key Configuration Options

| Category    | Options                          | Impact                  |
| ----------- | -------------------------------- | ----------------------- |
| Performance | `num_workers`, `use_gpu`         | Processing speed        |
| Quality     | `k_neighbors`, `patch_size`      | Feature accuracy        |
| Output      | `lod_level`, `format_preference` | Dataset characteristics |
| Workflow    | `skip_existing`, `force`         | Resumability behavior   |

## 🚀 Extension Points

The architecture supports customization through:

- **Custom Feature Extractors** - Add domain-specific features
- **Classification Schemas** - Define new LOD levels
- **Output Formats** - Support additional file formats
- **Processing Hooks** - Insert custom processing steps
- **Validation Rules** - Add quality checks

This modular design ensures the library can adapt to various research and production requirements while maintaining performance and reliability.
