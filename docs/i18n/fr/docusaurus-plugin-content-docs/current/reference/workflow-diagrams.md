---
sidebar_position: 2
title: Diagrammes de Workflows
description: Diagrammes mermaid réutilisables pour la documentation
---

# Workflow Diagrams

This page contains reusable workflow diagrams that are referenced across multiple documentation pages.

## Basic Traitementing Pipeline

```mermaid
flowchart TD
    Start([Start Traitementing]) --> Check{Data Available?}
    Check -->|No| Téléchargement[Téléchargement LiDAR Tiles]
    Check -->|Yes| Skip1[Skip Téléchargement]

    Téléchargement --> Validate{Files Valid?}
    Skip1 --> Validate
    Validate -->|No| Error1[Report Error]
    Validate -->|Yes| Enrichissement[Enrichissement with Features]

    Enrichissement --> GPU{Use GPU?}
    GPU -->|Yes| GPU_Traitement[⚡ GPU Feature Computation]
    GPU -->|No| CPU_Traitement[CPU Feature Computation]

    GPU_Traitement --> RGB{Add RGB?}
    CPU_Traitement --> RGB

    RGB -->|Yes| FetchRGB[Fetch IGN Orthophotos]
    RGB -->|No| SkipRGB[LiDAR Only]

    FetchRGB --> Features[Enrichissemented LAZ Ready]
    SkipRGB --> Features

    Features --> Traitement[Create Patches d'entraînement]
    Traitement --> Sortie[Jeu de données ML Ready]
    Sortie --> End([Traitement Complete])

    Error1 --> End

    style Start fill:#e8f5e8
    style End fill:#e8f5e8
    style Téléchargement fill:#e3f2fd
    style Enrichissement fill:#fff3e0
    style Traitement fill:#f3e5f5
    style Sortie fill:#e8f5e8
```

## Pipeline Architecture

```mermaid
graph LR
    A[YAML Config] --> B{Pipeline Command}
    B --> C[Téléchargement Stage]
    B --> D[Enrichissement Stage]
    B --> E[Patch Stage]
    C --> F[Raw LAZ Tiles]
    F --> D
    D --> G[Enrichissemented LAZ<br/>+ Caractéristiques géométriques<br/>+ RGB Data]
    G --> E
    E --> H[Patches d'entraînement<br/>NPZ Format]

    style A fill:#e1f5ff
    style B fill:#fff3cd
    style H fill:#d4edda
```

## Three-Stage Workflow

```mermaid
flowchart TD
    subgraph "Stage 1: Téléchargement"
        A[Define Area of Interest] --> B{Selection Method}
        B -->|Bounding Box| C[IGN WFS Query]
        B -->|Tile IDs| D[Direct Téléchargement]
        B -->|Strategic| E[Urban/Building Selection]
        C --> F[Téléchargement LAZ Tiles]
        D --> F
        E --> F
    end

    subgraph "Stage 2: Enrichissement"
        F --> G[Load Nuage de points]
        G --> H{Traitementing Mode}
        H -->|Building| I[Extract Buildings]
        H -->|Full| J[Full Classification]
        I --> K{Add RGB?}
        J --> K
        K -->|Yes| L[RGB Augmentation]
        K -->|No| M[Caractéristiques géométriques]
        L --> M
        M --> N{GPU Available?}
        N -->|Yes| O[GPU Acceleration]
        N -->|No| P[CPU Traitementing]
        O --> Q[Export Enrichissemented LAZ]
        P --> Q
    end

    subgraph "Stage 3: Patch Generation"
        Q --> R[Split into Patches<br/>150m × 150m]
        R --> U[Quality Filter]
        U --> V[LOD Classification<br/>LOD2 or LOD3]
        V --> W[Jeu de données ML]
    end

    style A fill:#e3f2fd
    style F fill:#fff9c4
    style Q fill:#fff9c4
    style W fill:#c8e6c9
    style O fill:#b2dfdb
    style P fill:#ffccbc
```
