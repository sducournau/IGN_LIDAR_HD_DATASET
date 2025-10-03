---
sidebar_position: 4
---

# Workflow Guide

This guide demonstrates common processing workflows with visual representations to help you understand the data flow and decision points.

## 🚀 Basic Workflow

The most common workflow for processing LiDAR data into ML-ready datasets.

```mermaid
flowchart TD
    Start([Start Processing]) --> Check{Data Available?}
    Check -->|No| Download[Download LiDAR Tiles]
    Check -->|Yes| Skip1[Skip Download]

    Download --> Validate{Files Valid?}
    Skip1 --> Validate
    Validate -->|No| Error1[Report Error]
    Validate -->|Yes| Enrich[Enrich with Features]

    Enrich --> GPU{Use GPU?}
    GPU -->|Yes| GPU_Process[GPU Feature Computation]
    GPU -->|No| CPU_Process[CPU Feature Computation]

    GPU_Process --> Features[Geometric Features Ready]
    CPU_Process --> Features

    Features --> Process[Create Training Patches]
    Process --> Augment{Apply Augmentation?}

    Augment -->|Yes| Aug_Process[Apply Data Augmentation]
    Augment -->|No| NoAug[Skip Augmentation]

    Aug_Process --> Output[ML Dataset Ready]
    NoAug --> Output
    Output --> End([Process Complete])

    Error1 --> End

    style Start fill:#e8f5e8
    style End fill:#e8f5e8
    style Download fill:#e3f2fd
    style Enrich fill:#fff3e0
    style Process fill:#f3e5f5
    style Output fill:#e8f5e8
```

## 🔄 Smart Skip Workflow

Understanding how the smart skip system optimizes repeated runs.

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant SkipChecker as Smart Skip
    participant FileSystem as File System
    participant Processor

    User->>CLI: ign-lidar-process enrich
    CLI->>SkipChecker: Check existing files
    SkipChecker->>FileSystem: List output directory
    FileSystem-->>SkipChecker: File list

    loop For each input file
        SkipChecker->>SkipChecker: Check if output exists
        SkipChecker->>SkipChecker: Validate file integrity
        alt File exists and valid
            SkipChecker-->>CLI: Skip this file
        else File missing or invalid
            SkipChecker->>Processor: Process this file
            Processor-->>SkipChecker: File processed
        end
    end

    CLI-->>User: Processing complete

    Note over SkipChecker: Smart skip saves time<br/>on large datasets
```

## 🏗️ Parallel Processing Workflow

How the library handles multi-worker processing for optimal performance.

```mermaid
graph TB
    subgraph "Main Process"
        M[Main Controller]
        Q[Task Queue]
        R[Result Collector]
    end

    subgraph "Worker Pool"
        W1[Worker 1]
        W2[Worker 2]
        W3[Worker N]
    end

    subgraph "Processing Tasks"
        T1[Tile 1]
        T2[Tile 2]
        T3[Tile 3]
        TN[Tile N]
    end

    M --> Q
    Q --> W1
    Q --> W2
    Q --> W3

    W1 --> T1
    W2 --> T2
    W3 --> T3

    T1 --> R
    T2 --> R
    T3 --> R
    TN --> R

    R --> M

    style M fill:#e3f2fd
    style Q fill:#fff3e0
    style W1 fill:#e8f5e8
    style W2 fill:#e8f5e8
    style W3 fill:#e8f5e8
```

## 🎯 LOD Classification Workflow

Understanding how building components are classified into LOD levels.

```mermaid
flowchart LR
    Input[Point Cloud Input] --> Classify{Classification Type}

    Classify -->|LOD2| LOD2_Process[LOD2 Processing<br/>15 Classes]
    Classify -->|LOD3| LOD3_Process[LOD3 Processing<br/>30+ Classes]

    subgraph "LOD2 Classes"
        LOD2_1[Ground]
        LOD2_2[Building]
        LOD2_3[Vegetation]
        LOD2_4[Water]
        LOD2_5[Bridge]
        LOD2_More[...]
    end

    subgraph "LOD3 Classes"
        LOD3_1[Walls]
        LOD3_2[Roofs]
        LOD3_3[Windows]
        LOD3_4[Doors]
        LOD3_5[Balconies]
        LOD3_6[Chimneys]
        LOD3_More[...]
    end

    LOD2_Process --> LOD2_1
    LOD2_Process --> LOD2_2
    LOD2_Process --> LOD2_3
    LOD2_Process --> LOD2_4
    LOD2_Process --> LOD2_5

    LOD3_Process --> LOD3_1
    LOD3_Process --> LOD3_2
    LOD3_Process --> LOD3_3
    LOD3_Process --> LOD3_4
    LOD3_Process --> LOD3_5
    LOD3_Process --> LOD3_6

    style Input fill:#e3f2fd
    style LOD2_Process fill:#e8f5e8
    style LOD3_Process fill:#fff3e0
```

## 📊 Feature Extraction Pipeline

Detailed view of the geometric feature computation process.

```mermaid
graph TD
    Points[Raw Point Cloud] --> KNN[K-Nearest Neighbors]
    KNN --> Normals[Surface Normals]
    KNN --> Curvature[Principal Curvature]

    Normals --> Planarity[Planarity Measure]
    Normals --> Verticality[Verticality Measure]
    Normals --> Horizontality[Horizontality Measure]

    Points --> Density[Local Density]
    Points --> Height[Height Above Ground]
    Points --> Intensity[Normalized Intensity]

    Curvature --> GeometricFeatures[Geometric Features]
    Planarity --> GeometricFeatures
    Verticality --> GeometricFeatures
    Horizontality --> GeometricFeatures
    Density --> GeometricFeatures
    Height --> GeometricFeatures
    Intensity --> GeometricFeatures

    GeometricFeatures --> ArchStyle[Architectural Style Inference]
    GeometricFeatures --> EnrichedLAZ[Enriched LAZ Output]

    style Points fill:#e3f2fd
    style GeometricFeatures fill:#e8f5e8
    style EnrichedLAZ fill:#fff3e0
    style ArchStyle fill:#f3e5f5
```

## 🔧 Configuration Decision Tree

How to choose optimal settings for your use case.

```mermaid
flowchart TD
    Start([Choose Configuration]) --> DataSize{Dataset Size}

    DataSize -->|Small < 10 tiles| SmallConfig[Basic Configuration]
    DataSize -->|Medium 10-100 tiles| MediumConfig[Optimized Configuration]
    DataSize -->|Large > 100 tiles| LargeConfig[High-Performance Configuration]

    SmallConfig --> GPU1{GPU Available?}
    MediumConfig --> GPU2{GPU Available?}
    LargeConfig --> GPU3{GPU Available?}

    GPU1 -->|Yes| SmallGPU[use_gpu=True<br/>workers=2]
    GPU1 -->|No| SmallCPU[use_gpu=False<br/>workers=4]

    GPU2 -->|Yes| MediumGPU[use_gpu=True<br/>workers=4<br/>batch_size=large]
    GPU2 -->|No| MediumCPU[use_gpu=False<br/>workers=8<br/>batch_size=medium]

    GPU3 -->|Yes| LargeGPU[use_gpu=True<br/>workers=8<br/>distributed=True]
    GPU3 -->|No| LargeCPU[use_gpu=False<br/>workers=16<br/>chunk_processing=True]

    SmallGPU --> Quality{Quality Priority?}
    SmallCPU --> Quality
    MediumGPU --> Quality
    MediumCPU --> Quality
    LargeGPU --> Quality
    LargeCPU --> Quality

    Quality -->|High| HighQuality[k_neighbors=50<br/>patch_overlap=0.2<br/>augmentations=5]
    Quality -->|Balanced| Balanced[k_neighbors=20<br/>patch_overlap=0.1<br/>augmentations=3]
    Quality -->|Fast| Fast[k_neighbors=10<br/>patch_overlap=0.05<br/>augmentations=1]

    HighQuality --> Final[Final Configuration]
    Balanced --> Final
    Fast --> Final

    style Start fill:#e8f5e8
    style Final fill:#e8f5e8
    style SmallConfig fill:#e3f2fd
    style MediumConfig fill:#fff3e0
    style LargeConfig fill:#f3e5f5
```

## 💡 Best Practice Workflows

### Urban Area Processing

```bash
# Optimized for dense urban environments
ign-lidar-process download --bbox 2.0,48.8,2.1,48.9 --output urban_tiles/
ign-lidar-process enrich --input-dir urban_tiles/ --output urban_enriched/ --use-gpu --k-neighbors 30
ign-lidar-process process --input-dir urban_enriched/ --output urban_patches/ --lod-level LOD3 --num-augmentations 5
```

### Rural/Natural Area Processing

```bash
# Optimized for sparse rural environments
ign-lidar-process download --bbox -1.0,46.0,0.0,47.0 --output rural_tiles/
ign-lidar-process enrich --input-dir rural_tiles/ --output rural_enriched/ --k-neighbors 15
ign-lidar-process process --input-dir rural_enriched/ --output rural_patches/ --lod-level LOD2 --num-augmentations 2
```

### High-Performance Batch Processing

```bash
# Maximum throughput for large datasets
ign-lidar-process enrich --input-dir tiles/ --output enriched/ --use-gpu --num-workers 8 --batch-size large
ign-lidar-process process --input-dir enriched/ --output patches/ --num-workers 16 --skip-existing
```

