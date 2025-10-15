---
sidebar_position: 4
title: Workflow Guide
description: Common LiDAR processing workflows with visual diagrams
keywords: [workflow, pipeline, processing, gpu, parallel, lod]
---

# Workflow Guide

This guide demonstrates common processing workflows with visual representations to help you understand the data flow and decision points.

:::tip Navigation

- [Basic Workflow](#-basic-workflow) - Standard processing pipeline
- [GPU-Accelerated Workflow](#-gpu-accelerated-workflow) - High-performance GPU processing
- [Smart Skip Workflow](#-smart-skip-workflow) - Resume interrupted jobs
- [Parallel Processing](#-parallel-processing-workflow) - Multi-worker processing
- [Best Practices](#-best-practice-workflows) - Optimized workflows for different scenarios
  :::

## 🚀 Basic Workflow

The most common workflow for processing LiDAR data into ML-ready datasets.

:::tip Diagram Reference
For the complete basic workflow diagram, see [Workflow Diagrams - Basic Processing Pipeline](reference/workflow-diagrams.md#basic-processing-pipeline).
:::

The workflow includes the following key stages:

1. **Data Availability Check**: Verify if LiDAR tiles are already downloaded
2. **Download**: Acquire tiles from IGN servers if needed
3. **Validation**: Ensure downloaded files are valid
4. **Enrichment**: Add geometric features and building component classification
5. **RGB Augmentation**: Optionally add color information from orthophotos
6. **Processing**: Create training patches for machine learning
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
   style Analyze fill:#c8e6c9
   style CleanData fill:#fff9c4
   style Enrich fill:#fff3e0
   style Process fill:#f3e5f5
   style Output fill:#e8f5e8

````

## ⚡ GPU-Accelerated Workflow

Workflow for processing large datasets with GPU acceleration (v1.3.0+).

```mermaid
flowchart TD
    Start([Start GPU Processing]) --> CheckGPU{GPU Available?}

    CheckGPU -->|Yes| GPUSetup[Initialize GPU<br/>CuPy + CUDA]
    CheckGPU -->|No| Fallback[⚠️ Fallback to CPU]

    GPUSetup --> LoadData[Load Point Cloud]
    Fallback --> LoadData

    LoadData --> TransferGPU[Transfer to GPU Memory]
    TransferGPU --> FeatureGPU[🚀 GPU Feature Computation<br/>5-10x faster]

    FeatureGPU --> RGBCheck{Include RGB?}
    RGBCheck -->|Yes v1.5.0+| RGBFetch[Fetch Orthophoto]
    RGBCheck -->|No| FeaturesDone[Features Complete]

    RGBFetch --> RGBCache{In GPU Cache?}
    RGBCache -->|Yes| RGBInterpolate[⚡ GPU Color Interpolation<br/>24x faster]
    RGBCache -->|No| RGBLoad[Load to GPU Cache]
    RGBLoad --> RGBInterpolate

    RGBInterpolate --> Combine[Combine Features + RGB]
    FeaturesDone --> Combine

    Combine --> TransferCPU[Transfer to CPU]
    TransferCPU --> SaveOutput[Save Enriched LAZ]

    SaveOutput --> MoreData{More Data?}
    MoreData -->|Yes| LoadData
    MoreData -->|No| End([Processing Complete])

    style Start fill:#e8f5e8
    style End fill:#e8f5e8
    style GPUSetup fill:#e3f2fd
    style FeatureGPU fill:#c8e6c9
    style RGBInterpolate fill:#c8e6c9
    style Fallback fill:#fff3e0
````

### GPU Performance Benefits

| Operation          | CPU Time | GPU Time | Speedup  |
| ------------------ | -------- | -------- | -------- |
| Feature Extraction | 45s      | 6s       | 8x       |
| RGB Interpolation  | 12s      | 0.5s     | 24x      |
| Normal Computation | 30s      | 5s       | 6x       |
| **Total (1M pts)** | ~87s     | ~11.5s   | **7.5x** |

:::tip GPU Optimization Tips

- **Cache RGB tiles** - Reuse orthophotos across patches
- **Batch processing** - Process multiple tiles in sequence
- **Monitor GPU memory** - Use `nvidia-smi` to check utilization
- **Use workers=1 with GPU** - GPU parallelizes internally

See [GPU Overview](gpu/overview.md) for detailed setup instructions.
:::

## 🔄 Smart Skip Workflow

Understanding how the smart skip system optimizes repeated runs.

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant SkipChecker as Smart Skip
    participant FileSystem as File System
    participant Processor

    User->>CLI: ign-lidar-hd enrich
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

:::tip Classification Schema
LOD2/LOD3 use a **building-focused classification** where buildings are Class 0 (`wall`), ground is Class 9, etc. This differs from ASPRS standards. See [Classification Taxonomy](./reference/classification-taxonomy.md) for details.
:::

```mermaid
flowchart LR
    Input[Point Cloud Input] --> Classify{Classification Type}

    Classify -->|LOD2| LOD2_Process[LOD2 Processing<br/>15 Classes]
    Classify -->|LOD3| LOD3_Process[LOD3 Processing<br/>30+ Classes]

    subgraph "LOD2 Classes"
        LOD2_1[0: Wall]
        LOD2_2[9: Ground]
        LOD2_3[10/11: Vegetation]
        LOD2_4[12: Water]
        LOD2_5[14: Other]
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

## 🎯 Enhanced Enrich Pipeline

Detailed view of the complete enrich workflow with auto-params and preprocessing.

:::info Feature Availability
Auto-params and advanced preprocessing features have been available since v1.7.1 and continue to work in v2.0+.
:::

```mermaid
flowchart TD
    Start[Load LAZ Tile] --> AutoCheck{Auto-Params<br/>Enabled?}

    AutoCheck -->|Yes| Sample[Sample 50K Points]
    AutoCheck -->|No| ManualParams[Use Manual Parameters]

    Sample --> CalcDensity[Calculate:<br/>• Point Density<br/>• NN Distance<br/>• Noise Level]
    CalcDensity --> Optimize[Determine Optimal:<br/>• Radius: 0.5-3.0m<br/>• SOR k: 8-15<br/>• SOR std: 1.5-2.5<br/>• ROR radius: 1.0-2.0m<br/>• ROR neighbors: 2-6]

    Optimize --> Report[📊 Display Analysis Report]
    ManualParams --> Report
    Report --> PreCheck{Preprocessing<br/>Enabled?}

    PreCheck -->|Yes| SOR[Statistical Outlier Removal<br/>Filter noise points]
    PreCheck -->|No| SkipPre[Skip to Features]

    SOR --> ROR[Radius Outlier Removal<br/>Remove scan artifacts]
    ROR --> VoxelCheck{Voxel<br/>Downsampling?}

    VoxelCheck -->|Yes| Voxel[Homogenize Point Density]
    VoxelCheck -->|No| CleanData[Clean Point Cloud]
    Voxel --> CleanData
    SkipPre --> Features

    CleanData --> Features[🔧 Feature Computation]
    Features --> SearchMode{Search Mode}

    SearchMode -->|Radius-based<br/>v1.6.5| RadiusSearch[Radius Search<br/>Artifact-free]
    SearchMode -->|k-NN<br/>legacy| KNNSearch[k-Nearest Neighbors]

    RadiusSearch --> BuildKDTree[Build KD-Tree]
    KNNSearch --> BuildKDTree

    BuildKDTree --> CoreFeatures[Compute Core Features:<br/>• Normals<br/>• Curvature<br/>• Planarity<br/>• Verticality]

    CoreFeatures --> ModeCheck{Feature Mode}
    ModeCheck -->|Core| SaveCore[Core Features Only]
    ModeCheck -->|Building| BuildFeatures[Building Features:<br/>• Wall probability<br/>• Roof probability<br/>• Edge detection<br/>• Corner detection]

    BuildFeatures --> RGBCheck{Add RGB?}
    SaveCore --> RGBCheck

    RGBCheck -->|Yes| FetchOrtho[Fetch IGN Orthophoto]
    RGBCheck -->|No| SaveLAZ[Save Enriched LAZ]

    FetchOrtho --> InterpolateRGB[Interpolate RGB Colors]
    InterpolateRGB --> SaveLAZ

    SaveLAZ --> End[✅ Enriched Tile Complete]

    style Start fill:#e3f2fd
    style Sample fill:#c8e6c9
    style Optimize fill:#c8e6c9
    style Report fill:#fff9c4
    style SOR fill:#fff9c4
    style ROR fill:#fff9c4
    style Features fill:#fff3e0
    style RadiusSearch fill:#c8e6c9
    style SaveLAZ fill:#e8f5e8
    style End fill:#e8f5e8
```

## 📊 Feature Extraction Pipeline

Detailed view of the geometric feature computation process.

```mermaid
graph TD
    Points[Raw Point Cloud] --> SearchMethod{Search Method}
    SearchMethod -->|Radius-based| Radius[Radius Search<br/>Eliminates Artifacts]
    SearchMethod -->|k-NN| KNN[K-Nearest Neighbors<br/>Legacy Mode]

    Radius --> Neighbors[Variable Neighbors<br/>Per Point]
    KNN --> FixedNeighbors[Fixed k Neighbors<br/>Per Point]

    Neighbors --> Normals[Surface Normals]
    FixedNeighbors --> Normals

    Normals --> Eigenvalues[Compute Eigenvalues<br/>Covariance Matrix]
    Eigenvalues --> Curvature[Principal Curvature]

    Curvature --> Planarity[Planarity Measure]
    Curvature --> Linearity[Linearity Measure]
    Curvature --> Anisotropy[Anisotropy Measure]

    Normals --> Verticality[Verticality Measure]
    Normals --> Horizontality[Horizontality Measure]

    Points --> Density[Local Density]
    Points --> Height[Height Above Ground]
    Points --> Intensity[Normalized Intensity]

    Planarity --> GeometricFeatures[Geometric Features]
    Linearity --> GeometricFeatures
    Anisotropy --> GeometricFeatures
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
ign-lidar-hd download --bbox 2.0,48.8,2.1,48.9 --output urban_tiles/
ign-lidar-hd enrich --input-dir urban_tiles/ --output urban_enriched/ --use-gpu --k-neighbors 30
ign-lidar-hd process --input-dir urban_enriched/ --output urban_patches/ --lod-level LOD3
```

### Rural/Natural Area Processing

```bash
# Optimized for sparse rural environments
ign-lidar-hd download --bbox -1.0,46.0,0.0,47.0 --output rural_tiles/
ign-lidar-hd enrich --input-dir rural_tiles/ --output rural_enriched/ --k-neighbors 15
ign-lidar-hd process --input-dir rural_enriched/ --output rural_patches/ --lod-level LOD2
```

### High-Performance Batch Processing

```bash
# Maximum throughput for large datasets
ign-lidar-hd enrich --input-dir tiles/ --output enriched/ --use-gpu --num-workers 8 --batch-size large
ign-lidar-hd process --input-dir enriched/ --output patches/ --num-workers 16 --skip-existing
```

### GPU-Accelerated with RGB (v1.5.0+)

```bash
# Fastest processing with GPU RGB augmentation
ign-lidar-hd enrich \
  --input-dir tiles/ \
  --output enriched/ \
  --use-gpu \
  --add-rgb \
  --rgb-cache-dir /data/rgb_cache/ \
  --num-workers 4

# Create patches with cached RGB
ign-lidar-hd process \
  --input-dir enriched/ \
  --output patches/ \
  --lod-level LOD3 \
  --num-workers 8
```

---

## 📚 Related Documentation

- **[GPU Acceleration Guide](gpu/overview.md)** - Detailed GPU setup and optimization
- **[RGB GPU Guide](gpu/rgb-augmentation.md)** - GPU-accelerated RGB augmentation (v1.5.0+)
- **[Architecture](architecture.md)** - System architecture and components
- **[CLI Commands](guides/cli-commands.md)** - Complete CLI reference
- **[Smart Skip](features/smart-skip.md)** - Smart skip system details
- **[LOD Classification](features/lod3-classification.md)** - LOD2/LOD3 classification

---

## 💡 Tips for Workflow Selection

### Choose Basic Workflow when:

- ✅ Learning the library
- ✅ Processing < 10 tiles
- ✅ No GPU available
- ✅ Prototyping and testing

### Choose GPU Workflow when:

- ✅ Processing > 50 tiles
- ✅ NVIDIA GPU available
- ✅ Production pipelines
- ✅ Time-sensitive projects

### Choose Smart Skip when:

- ✅ Resuming interrupted jobs
- ✅ Iterative processing
- ✅ Large datasets with failures
- ✅ Incremental updates

### Parallel Processing for:

- ✅ Multi-core systems
- ✅ Batch processing
- ✅ Production environments
- ✅ Maximizing throughput
