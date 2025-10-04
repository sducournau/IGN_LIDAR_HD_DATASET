---
sidebar_position: 2
title: Workflow Diagrams
description: Reusable mermaid diagrams for documentation
---

# Workflow Diagrams

This page contains reusable workflow diagrams that are referenced across multiple documentation pages.

## Basic Processing Pipeline

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
    GPU -->|Yes| GPU_Process[⚡ GPU Feature Computation]
    GPU -->|No| CPU_Process[CPU Feature Computation]

    GPU_Process --> RGB{Add RGB?}
    CPU_Process --> RGB

    RGB -->|Yes| FetchRGB[Fetch IGN Orthophotos]
    RGB -->|No| SkipRGB[LiDAR Only]

    FetchRGB --> Features[Enriched LAZ Ready]
    SkipRGB --> Features

    Features --> Process[Create Training Patches]
    Process --> Output[ML Dataset Ready]
    Output --> End([Process Complete])

    Error1 --> End

    style Start fill:#e8f5e8
    style End fill:#e8f5e8
    style Download fill:#e3f2fd
    style Enrich fill:#fff3e0
    style Process fill:#f3e5f5
    style Output fill:#e8f5e8
```

## Pipeline Architecture

```mermaid
graph LR
    A[YAML Config] --> B{Pipeline Command}
    B --> C[Download Stage]
    B --> D[Enrich Stage]
    B --> E[Patch Stage]
    C --> F[Raw LAZ Tiles]
    F --> D
    D --> G[Enriched LAZ<br/>+ Geometric Features<br/>+ RGB Data]
    G --> E
    E --> H[Training Patches<br/>NPZ Format]

    style A fill:#e1f5ff
    style B fill:#fff3cd
    style H fill:#d4edda
```

## Three-Stage Workflow

```mermaid
flowchart TD
    subgraph "Stage 1: Download"
        A[Define Area of Interest] --> B{Selection Method}
        B -->|Bounding Box| C[IGN WFS Query]
        B -->|Tile IDs| D[Direct Download]
        B -->|Strategic| E[Urban/Building Selection]
        C --> F[Download LAZ Tiles]
        D --> F
        E --> F
    end

    subgraph "Stage 2: Enrich"
        F --> G[Load Point Cloud]
        G --> H{Processing Mode}
        H -->|Building| I[Extract Buildings]
        H -->|Full| J[Full Classification]
        I --> K{Add RGB?}
        J --> K
        K -->|Yes| L[RGB Augmentation]
        K -->|No| M[Geometric Features]
        L --> M
        M --> N{GPU Available?}
        N -->|Yes| O[GPU Acceleration]
        N -->|No| P[CPU Processing]
        O --> Q[Export Enriched LAZ]
        P --> Q
    end

    subgraph "Stage 3: Patch Generation"
        Q --> R[Split into Patches<br/>150m × 150m]
        R --> U[Quality Filter]
        U --> V[LOD Classification<br/>LOD2 or LOD3]
        V --> W[ML-Ready Dataset]
    end

    style A fill:#e3f2fd
    style F fill:#fff9c4
    style Q fill:#fff9c4
    style W fill:#c8e6c9
    style O fill:#b2dfdb
    style P fill:#ffccbc
```
