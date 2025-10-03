---
sidebar_position: 5
---

# Référence des Diagrammes Mermaid

Cette page démontre les différents diagrammes Mermaid utilisés dans la documentation IGN LiDAR HD pour visualiser les workflows, architectures et processus.

## 🔄 Diagrammes de Workflow

### Flux de Traitement de Base

```mermaid
flowchart TD
    A[Début] --> B{Entrée Disponible?}
    B -->|Oui| C[Traiter Données]
    B -->|Non| D[Télécharger Données]
    D --> C
    C --> E[Générer Sortie]
    E --> F[Fin]

    style A fill:#e8f5e8
    style F fill:#e8f5e8
    style C fill:#e3f2fd
```

### Pipeline Complexe

```mermaid
graph TB
    subgraph "Couche d'Entrée"
        I1[LiDAR Brut]
        I2[Configuration]
    end

    subgraph "Couche de Traitement"
        P1[Téléchargement]
        P2[Enrichissement]
        P3[Traitement]
    end

    subgraph "Couche de Sortie"
        O1[Fichiers NPZ]
        O2[Métadonnées]
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

## 📊 Graphiques de Performance

### Comparaison de Vitesse de Traitement

```mermaid
xychart-beta
    title "Performance de Traitement par Matériel"
    x-axis [CPU-4cœurs, CPU-8cœurs, CPU-16cœurs, GPU-RTX3080, GPU-RTX4090]
    y-axis "Dalles par Heure" 0 --> 100
    bar "Petites Dalles" [8, 15, 25, 60, 85]
    bar "Grandes Dalles" [3, 6, 10, 25, 40]
```

### Utilisation Mémoire dans le Temps

```mermaid
xychart-beta
    title "Utilisation Mémoire Pendant le Traitement"
    x-axis [0min, 1min, 2min, 3min, 4min, 5min]
    y-axis "Mémoire GB" 0 --> 16
    line "Utilisation RAM" [2, 4, 8, 12, 8, 4]
    line "Mémoire GPU" [0, 2, 6, 8, 6, 2]
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

### Processing States

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

### Directives de Palette de Couleurs

- 🟢 **Succès/Complétion** : `fill:#e8f5e8`
- 🔵 **Traitement/Actif** : `fill:#e3f2fd`
- 🟡 **Avertissement/Attention** : `fill:#fff3e0`
- 🟣 **Configuration** : `fill:#f3e5f5`
- 🔴 **Erreur/Problème** : `fill:#ffebee`

### Bonnes Pratiques

1. **Gardez les diagrammes ciblés** - Un concept par diagramme
2. **Utilisez un style cohérent** - Mêmes couleurs pour éléments similaires
3. **Ajoutez des étiquettes significatives** - Texte clair et descriptif
4. **Optimisez pour la lisibilité** - Pas trop encombré
5. **Mettez à jour régulièrement** - Gardez les diagrammes à jour avec les changements de code
