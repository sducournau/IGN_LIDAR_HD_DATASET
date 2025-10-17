# 🗺️ Multi-Scale Training Pipeline - Visual Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         UNIFIED DATASET (INPUT)                               │
│                   C:\Users\Simon\ign\unified_dataset                         │
│                        ~ 200+ LAZ tiles                                       │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 1: TILE SELECTION                                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │    ASPRS     │    │     LOD2     │    │     LOD3     │                   │
│  │  100 tiles   │    │   80 tiles   │    │   60 tiles   │                   │
│  │  (diverse)   │    │  (buildings) │    │  (detailed)  │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: PREPROCESSING & ENRICHMENT                        │
│  ┌────────────────────────────────────────────────────────────────────┐      │
│  │  Features Computed for ALL tiles:                                  │      │
│  │  • RGB Colors (Red, Green, Blue)                                   │      │
│  │  • NIR (Near-Infrared)                                             │      │
│  │  • NDVI = (NIR - R) / (NIR + R)                                   │      │
│  │  • Geometric: normals, curvature, planarity, etc.                  │      │
│  │  • Ground Truth: IGN BD TOPO® classification                       │      │
│  │  • Preclassification: NDVI-based vegetation detection              │      │
│  └────────────────────────────────────────────────────────────────────┘      │
│                                                                               │
│  Output: Enriched LAZ tiles with ALL features                                │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                  PHASE 3: MULTI-SCALE PATCH GENERATION                        │
│                                                                               │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │      50m SCALE      │  │     100m SCALE      │  │     150m SCALE      │  │
│  │  ╔═══════════════╗  │  │  ╔═══════════════╗  │  │  ╔═══════════════╗  │  │
│  │  ║     ASPRS     ║  │  │  ║     ASPRS     ║  │  │  ║     ASPRS     ║  │  │
│  │  ║ 16k points    ║  │  │  ║ 24k points    ║  │  │  ║ 32k points    ║  │  │
│  │  ║ 3 augment.    ║  │  │  ║ 3 augment.    ║  │  │  ║ 3 augment.    ║  │  │
│  │  ╚═══════════════╝  │  │  ╚═══════════════╝  │  │  ╚═══════════════╝  │  │
│  │  ╔═══════════════╗  │  │  ╔═══════════════╗  │  │  ╔═══════════════╗  │  │
│  │  ║     LOD2      ║  │  │  ║     LOD2      ║  │  │  ║     LOD2      ║  │  │
│  │  ║ 16k points    ║  │  │  ║ 24k points    ║  │  │  ║ 32k points    ║  │  │
│  │  ║ 3 augment.    ║  │  │  ║ 3 augment.    ║  │  │  ║ 3 augment.    ║  │  │
│  │  ╚═══════════════╝  │  │  ╚═══════════════╝  │  │  ╚═══════════════╝  │  │
│  │  ╔═══════════════╗  │  │  ╔═══════════════╗  │  │  ╔═══════════════╗  │  │
│  │  ║     LOD3      ║  │  │  ║     LOD3      ║  │  │  ║     LOD3      ║  │  │
│  │  ║ 24k points    ║  │  │  ║ 32k points    ║  │  │  ║ 40k points    ║  │  │
│  │  ║ 5 augment.    ║  │  │  ║ 5 augment.    ║  │  │  ║ 5 augment.    ║  │  │
│  │  ╚═══════════════╝  │  │  ╚═══════════════╝  │  │  ╚═══════════════╝  │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  │
│                                                                               │
│  Output: NPZ patches (training) + LAZ patches (visualization)                │
│  Split: 70% train / 15% val / 15% test                                       │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      PHASE 4: DATASET MERGING                                 │
│                                                                               │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐               │
│  │  50m patches   │   │ 100m patches   │   │ 150m patches   │               │
│  └───────┬────────┘   └───────┬────────┘   └───────┬────────┘               │
│          │                    │                     │                        │
│          └────────────────────┴─────────────────────┘                        │
│                               │                                              │
│                               ▼                                              │
│  ┌──────────────────────────────────────────────────────────────┐            │
│  │         MERGED MULTI-SCALE DATASET                           │            │
│  │  Strategy: Balanced (ASPRS) / Weighted (LOD2) / Adaptive     │            │
│  │  Class balancing + oversampling of rare classes              │            │
│  └──────────────────────────────────────────────────────────────┘            │
│                                                                               │
│  Output: Unified multi-scale training datasets (NPZ)                         │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                   PHASE 5: PROGRESSIVE TRAINING                               │
│                                                                               │
│  ╔══════════════════════════════════════════════════════════════════╗        │
│  ║                        LEVEL 1: ASPRS                            ║        │
│  ║  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐  ║        │
│  ║  │  PointNet++  │  │ Point Transformer│  │ Intelligent Index│  ║        │
│  ║  │  100 epochs  │  │   150 epochs     │  │   120 epochs     │  ║        │
│  ║  └──────────────┘  └──────────────────┘  └──────────────────┘  ║        │
│  ╚═══════════════════════════════╤══════════════════════════════════╝        │
│                                  │ Transfer Learning                         │
│                                  ▼                                           │
│  ╔══════════════════════════════════════════════════════════════════╗        │
│  ║                        LEVEL 2: LOD2                             ║        │
│  ║  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐  ║        │
│  ║  │  PointNet++  │  │ Point Transformer│  │ Intelligent Index│  ║        │
│  ║  │  150 epochs  │  │   200 epochs     │  │   180 epochs     │  ║        │
│  ║  │ + pretrained │  │  + pretrained    │  │  + pretrained    │  ║        │
│  ║  └──────────────┘  └──────────────────┘  └──────────────────┘  ║        │
│  ╚═══════════════════════════════╤══════════════════════════════════╝        │
│                                  │ Transfer Learning                         │
│                                  ▼                                           │
│  ╔══════════════════════════════════════════════════════════════════╗        │
│  ║                        LEVEL 3: LOD3                             ║        │
│  ║  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐  ║        │
│  ║  │  PointNet++  │  │ Point Transformer│  │ Intelligent Index│  ║        │
│  ║  │  200 epochs  │  │   250 epochs     │  │   220 epochs     │  ║        │
│  ║  │ + pretrained │  │  + pretrained    │  │  + pretrained    │  ║        │
│  ║  │ + focal loss │  │  + focal loss    │  │  + focal loss    │  ║        │
│  ║  └──────────────┘  └──────────────────┘  └──────────────────┘  ║        │
│  ╚══════════════════════════════════════════════════════════════════╝        │
│                                                                               │
│  Output: 9 trained models (3 architectures × 3 LOD levels)                   │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PIPELINE                                     │
│                                                                               │
│  Input: Full LAZ tile (any size)                                             │
│         │                                                                     │
│         ├──► ASPRS Model ──► Classified (ground, vegetation, building, etc.) │
│         │                     │                                               │
│         │                     └──► Filter buildings                           │
│         │                           │                                         │
│         └───────────────────────────┴──► LOD2 Model                          │
│                                           │                                   │
│                                           ├─► Walls, roofs, details           │
│                                           │                                   │
│                                           └──► Filter walls/roofs             │
│                                                 │                             │
│                                                 └──► LOD3 Model               │
│                                                      │                        │
│                                                      └─► Windows, doors, etc. │
│                                                                               │
│  Output: Fully classified LAZ with hierarchical labels                       │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 📊 Key Metrics by Level

### ASPRS Classification (9 classes)

```
Ground (1) ─────┐
Low Vegetation  │
Medium Vegetation├─► Base Classification
High Vegetation │
Building ───────┤
Water           │
Bridge          │
Vehicle         │
Other ──────────┘

Target Metrics:
• Overall Accuracy: > 92%
• mIoU: > 75%
• F1-Score: > 0.80 (main classes)
```

### LOD2 Classification (15 classes)

```
Wall ───────────┐
Flat Roof       │
Gable Roof      │
Hip Roof        ├─► Building-Focused
Chimney         │
Dormer          │
Balcony         │
Overhang        │
Foundation ─────┤
Ground          │
Low Vegetation  │
High Vegetation │
Water           │
Vehicle         │
Other ──────────┘

Target Metrics:
• Overall Accuracy: > 85%
• mIoU: > 65%
• F1-Score: > 0.75 (walls/roofs)
```

### LOD3 Classification (30 classes)

```
Plain Wall ──────────┐
Wall with Windows    │
Wall with Door       │
Flat Roof            │
Gable Roof           │
Hip Roof             │
Mansard Roof         │
Shed Roof            ├─► Detailed Architecture
Gambrel Roof         │
Butterfly Roof       │
Dome                 │
Other Roof           │
Window               │
Door                 │
Balcony Door         │
Chimney              │
Dormer               │
Balcony              │
Overhang             │
Gutter               │
Parapet              │
Foundation           │
Ground               │
Low Vegetation       │
Medium Vegetation    │
High Vegetation      │
Tree                 │
Water                │
Vehicle              │
Other ───────────────┘

Target Metrics:
• Overall Accuracy: > 78%
• mIoU: > 55%
• F1-Score: > 0.60 (openings)
```

## 🔄 Data Flow Summary

```
Unified Dataset (200+ tiles)
    │
    ├─► Select 100 ASPRS tiles ────────┐
    ├─► Select 80 LOD2 tiles ──────────┤
    └─► Select 60 LOD3 tiles ──────────┤
                                        │
    ┌───────────────────────────────────┘
    │
    ├─► Enrich with Features (RGB, NIR, NDVI, geometry)
    │
    ├─► Generate 50m patches ──────┐
    ├─► Generate 100m patches ─────┤─► Merge ──► Train ASPRS
    └─► Generate 150m patches ─────┘             │
                                                  ├─► Train LOD2
                                                  │
                                                  └─► Train LOD3
                                                       │
                                                       └─► Classify new tiles
```

## 💡 Scale Selection Guide

| Object Type      | Optimal Scale | Reason                      |
| ---------------- | ------------- | --------------------------- |
| Windows          | 50m           | Local detail capture        |
| Doors            | 50m           | Fine architectural features |
| Small buildings  | 50-100m       | Full building in context    |
| Medium buildings | 100m          | Balanced representation     |
| Large buildings  | 100-150m      | Complete structure          |
| Urban blocks     | 150m          | Neighborhood context        |
| Vegetation       | 100-150m      | Growth patterns             |
| Infrastructure   | 150m          | Large-scale features        |

## 🎯 Training Strategy

```
ASPRS (Baseline)
    ↓ freeze backbone 10-15 epochs
LOD2 (Building Specialization)
    ↓ freeze backbone 15-20 epochs
LOD3 (Fine Details)
    + Focal Loss (class imbalance)
    + Higher augmentation (5x)
    + Class weights (auto)
```

---

**Created**: October 15, 2025  
**Version**: 1.0  
**Project**: IGN LiDAR HD Multi-Scale Training System
