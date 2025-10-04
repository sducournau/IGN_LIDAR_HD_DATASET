# Architecture du Système de Réduction des Artefacts

## Vue d'Ensemble du Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      IGN LIDAR HD PROCESSING PIPELINE                    │
│                         (Artifact Mitigation)                            │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│  Raw LAZ     │
│  Tiles       │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         PHASE 1: PREPROCESSING                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │ Statistical  │ ─> │   Radius     │ ─> │   Voxel      │               │
│  │   Outlier    │    │   Outlier    │    │ Downsample   │               │
│  │   Removal    │    │   Removal    │    │  (optional)  │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│   k=12, std=2.0       r=1.0m, min=4       voxel=0.5m                     │
│                                                                           │
│  Removes:                                                                │
│  • Instrumental noise                                                    │
│  • Isolated points                                                       │
│  • Density heterogeneity                                                 │
└──────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       PHASE 2: TILE BORDER HANDLING                      │
│                                                                           │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐                       │
│  │  Tile N-W  │   │  Tile N    │   │  Tile N-E  │                       │
│  └─────┬──────┘   └─────┬──────┘   └─────┬──────┘                       │
│        │                │                │                               │
│        │   ┌────────────┴────────────┐   │                               │
│  ┌─────┴───┴─────┐   ┌──────────────┴───┴─────┐                         │
│  │  Tile W       │   │  MAIN TILE + BUFFER    │   ┌────────────┐        │
│  └─────┬───┬─────┘   │     (50m zone)         │   │  Tile E    │        │
│        │   └─────────┴──────────────┬───┬─────┘   └─────┬──────┘        │
│        │                │           │   │               │                │
│  ┌─────┴──────┐   ┌────┴──────┐   ┌────┴──────┐   ┌────┴──────┐        │
│  │  Tile S-W  │   │  Tile S   │   │  Tile S-E │   │           │        │
│  └────────────┘   └───────────┘   └───────────┘   └───────────┘        │
│                                                                           │
│  Ensures:                                                                │
│  • Continuous features at borders                                        │
│  • No edge discontinuities                                               │
│  • Smooth transitions between tiles                                      │
└──────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     PHASE 3: FEATURE COMPUTATION                         │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  Radius-Based Search (SUPERIOR to kNN)                         │     │
│  │  ┌──────────────────────────────────────────────────────────┐  │     │
│  │  │  Auto-Estimate Optimal Radius:                           │  │     │
│  │  │  • Sample 1000 points                                    │  │     │
│  │  │  • Compute average nearest neighbor distance            │  │     │
│  │  │  • Geometric features: radius = 20 × avg_nn_dist       │  │     │
│  │  │  • Clip to [0.5m, 2.0m] range                          │  │     │
│  │  └──────────────────────────────────────────────────────────┘  │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                           │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐     │
│  │  Normals   │   │ Planarity  │   │ Curvature  │   │  Density   │     │
│  │   (PCA)    │   │ Linearity  │   │   (MAD)    │   │  (local)   │     │
│  └────────────┘   │ Sphericity │   └────────────┘   └────────────┘     │
│                   │ Anisotropy │                                         │
│                   │ Roughness  │                                         │
│                   └────────────┘                                         │
│                                                                           │
│  Techniques:                                                             │
│  • Robust PCA (eigenvalue validation)                                    │
│  • MAD for outlier-resistant curvature                                   │
│  • Degenerate feature masking                                            │
└──────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       PHASE 4: QUALITY CONTROL                           │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────┐         │
│  │  Feature Quality Metrics (0-100 score)                      │         │
│  │  ┌────────────────────────────────────────────────────────┐ │         │
│  │  │ 1. Normal Coherence       (40%)                        │ │         │
│  │  │    → Low variance = good                               │ │         │
│  │  │                                                         │ │         │
│  │  │ 2. Degenerate Ratio       (30%)                        │ │         │
│  │  │    → <5% zeros = good                                  │ │         │
│  │  │                                                         │ │         │
│  │  │ 3. Curvature Outliers     (30%)                        │ │         │
│  │  │    → Few spikes = good                                 │ │         │
│  │  └────────────────────────────────────────────────────────┘ │         │
│  └─────────────────────────────────────────────────────────────┘         │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────┐         │
│  │  Artifact Detection (2D FFT)                                │         │
│  │  ┌────────────────────────────────────────────────────────┐ │         │
│  │  │ • Grid planarity values                                │ │         │
│  │  │ • Apply 2D Fourier Transform                           │ │         │
│  │  │ • Detect periodic patterns (scan lines)                │ │         │
│  │  │ • Flag if strong periodicities found                   │ │         │
│  │  └────────────────────────────────────────────────────────┘ │         │
│  └─────────────────────────────────────────────────────────────┘         │
│                                                                           │
│  Decision:                                                                │
│  • Score ≥ 70  → ✅ Accept                                                │
│  • Score < 50  → ⚠️  Warning + recommendation                             │
│  • Artifacts   → 🔴 Flag for review                                       │
└──────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│  Enriched    │
│  LAZ Tiles   │
│  (Artifact-  │
│   reduced)   │
└──────────────┘
```

---

## Module Architecture

```
ign_lidar/
├── preprocessing.py          [NEW - Phase 1]
│   ├── statistical_outlier_removal()
│   ├── radius_outlier_removal()
│   ├── voxel_downsample()
│   └── preprocess_point_cloud()
│
├── tile_borders.py           [NEW - Phase 2]
│   ├── extract_tile_with_buffer()
│   ├── find_neighbor_tiles()
│   └── get_tile_coordinates()
│
├── density_analysis.py       [NEW - Phase 3]
│   ├── estimate_multi_scale_density()
│   └── detect_environment_type()
│
├── quality_metrics.py        [NEW - Phase 4]
│   ├── compute_feature_quality_metrics()
│   └── detect_scan_line_artifacts()
│
├── features.py               [ENHANCED]
│   ├── estimate_optimal_radius_for_features()  [EXISTING ✅]
│   ├── compute_normals()                       [EXISTING ✅]
│   ├── extract_geometric_features()            [EXISTING ✅]
│   └── compute_all_features_optimized()        [EXISTING ✅]
│
├── processor.py              [MODIFIED]
│   └── process_tile()                          [ADD preprocessing call]
│
└── cli.py                    [MODIFIED]
    └── cmd_enrich()                            [ADD preprocessing args]
```

---

## Data Flow with Artifact Mitigation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         BEFORE (Current State)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Raw Points  ──────────────────────> Compute Features ─> Output        │
│  (with noise)                        (kNN or radius)     (artifacts!)  │
│                                                                         │
│  Issues:                                                                │
│  • Outliers contaminate neighborhoods                                   │
│  • Heterogeneous density → scan line artifacts                          │
│  • Border points missing neighbors → discontinuities                    │
│  • No quality control                                                   │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          AFTER (With Mitigation)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Raw Points                                                             │
│      │                                                                  │
│      ├─> [1] Load with Buffer (if border zone)                         │
│      │                                                                  │
│      ├─> [2] Preprocess                                                 │
│      │    ├─> Statistical Outlier Removal                              │
│      │    ├─> Radius Outlier Removal                                   │
│      │    └─> Voxel Downsample (optional)                              │
│      │                                                                  │
│      ├─> [3] Estimate Optimal Radius                                    │
│      │    └─> Density-adaptive parameter                               │
│      │                                                                  │
│      ├─> [4] Compute Features (radius-based)                            │
│      │    ├─> Normals (PCA with validation)                            │
│      │    ├─> Curvature (MAD robust)                                   │
│      │    └─> Geometric features                                       │
│      │                                                                  │
│      ├─> [5] Quality Control                                            │
│      │    ├─> Feature quality score                                    │
│      │    └─> Artifact detection                                       │
│      │                                                                  │
│      └─> Clean Output (artifact-reduced) ✅                             │
│                                                                         │
│  Benefits:                                                              │
│  • Outliers removed before feature computation                          │
│  • Homogeneous density → no scan artifacts                              │
│  • Border continuity preserved                                          │
│  • Quality guaranteed (score threshold)                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Configuration Priority                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [1] CLI Arguments (highest priority)                                   │
│      --preprocess --sor-k 12 --ror-radius 1.0 --buffer-distance 50     │
│                                                                         │
│  [2] YAML Configuration File                                            │
│      preprocessing:                                                     │
│        enable: true                                                     │
│        statistical_outlier:                                             │
│          k_neighbors: 12                                                │
│          std_multiplier: 2.0                                            │
│                                                                         │
│  [3] Python API                                                         │
│      processor = LiDARProcessor(                                        │
│          enable_preprocessing=True,                                     │
│          preprocessing_config={...}                                     │
│      )                                                                  │
│                                                                         │
│  [4] Default Values (lowest priority)                                   │
│      DEFAULTS = {                                                       │
│          'sor': {'k': 12, 'std': 2.0},                                 │
│          'ror': {'radius': 1.0, 'min': 4}                              │
│      }                                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Performance Impact

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Processing Time Breakdown                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  WITHOUT PREPROCESSING (baseline)                                       │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ Load LAZ        ████░░░░░░░░░░░░░░░░░░░░░░░  10%              │    │
│  │ Compute Features████████████████████████████  80%              │    │
│  │ Save LAZ        ████░░░░░░░░░░░░░░░░░░░░░░░  10%              │    │
│  └────────────────────────────────────────────────────────────────┘    │
│  Total: ~100 seconds per tile                                           │
│                                                                         │
│  WITH PREPROCESSING (+ artifacts mitigation)                            │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ Load + Buffer   ████░░░░░░░░░░░░░░░░░░░░░░░  12%              │    │
│  │ Preprocessing   ██░░░░░░░░░░░░░░░░░░░░░░░░░   8%              │    │
│  │ Compute Features████████████████████████████  70%              │    │
│  │ Quality Check   █░░░░░░░░░░░░░░░░░░░░░░░░░░   2%              │    │
│  │ Save LAZ        ████░░░░░░░░░░░░░░░░░░░░░░░   8%              │    │
│  └────────────────────────────────────────────────────────────────┘    │
│  Total: ~118 seconds per tile (+18% overhead)                           │
│                                                                         │
│  Trade-off:                                                             │
│  • +18% processing time                                                 │
│  • -80% artifacts (estimated)                                           │
│  • +quality guarantee                                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quality Score Calculation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  Feature Quality Score (0-100)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Component 1: Normal Coherence (40 points max)                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  score = 40 × (1 / (1 + variance(normals)))                      │  │
│  │                                                                   │  │
│  │  • High variance → low score (noisy normals)                     │  │
│  │  • Low variance  → high score (smooth normals)                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  Component 2: Degenerate Features (30 points max)                       │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  score = 30 × (1 - ratio_zeros)                                  │  │
│  │                                                                   │  │
│  │  • Many zeros (>15%) → low score                                 │  │
│  │  • Few zeros (<5%)   → high score                                │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  Component 3: Curvature Outliers (30 points max)                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  score = 30 × (1 - outlier_ratio)                                │  │
│  │                                                                   │  │
│  │  • Many outliers → low score (spikes/artifacts)                  │  │
│  │  • Few outliers  → high score (clean curvature)                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  TOTAL = Component1 + Component2 + Component3                           │
│                                                                         │
│  Interpretation:                                                        │
│  • 90-100: Excellent (no visible artifacts)                             │
│  • 70-90:  Good (minor artifacts, acceptable)                           │
│  • 50-70:  Fair (some artifacts, review recommended)                    │
│  • 0-50:   Poor (major artifacts, preprocessing needed)                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Artifact Types and Solutions Map

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  Artifact Type → Solution Mapping                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  SCAN LINE ARTIFACTS (dashed lines in features)                         │
│  ┌─────────────────┐                                                    │
│  │ Cause:          │ kNN on heterogeneous density                       │
│  │ Symptom:        │ Regular patterns, stripes in planarity             │
│  │ Solution:       │ ✅ Radius-based search (already implemented)       │
│  │                 │ ✅ Statistical Outlier Removal (Phase 1)           │
│  └─────────────────┘                                                    │
│                                                                         │
│  BORDER DISCONTINUITIES                                                 │
│  ┌─────────────────┐                                                    │
│  │ Cause:          │ Missing neighbors at tile edges                    │
│  │ Symptom:        │ Abrupt changes at tile junctions                   │
│  │ Solution:       │ 🔧 Buffer loading from neighbors (Phase 2)         │
│  └─────────────────┘                                                    │
│                                                                         │
│  NOISY NORMALS                                                          │
│  ┌─────────────────┐                                                    │
│  │ Cause:          │ Outliers contaminating PCA                         │
│  │ Symptom:        │ Erratic normal directions                          │
│  │ Solution:       │ 🔧 SOR before feature computation (Phase 1)        │
│  └─────────────────┘                                                    │
│                                                                         │
│  DEGENERATE FEATURES (all zeros)                                        │
│  ┌─────────────────┐                                                    │
│  │ Cause:          │ Colinear points, null eigenvalues                  │
│  │ Symptom:        │ >10% features = 0.0                                │
│  │ Solution:       │ ✅ Eigenvalue validation (already implemented)     │
│  │                 │ 🔧 ROR to remove sparse areas (Phase 1)            │
│  └─────────────────┘                                                    │
│                                                                         │
│  CURVATURE SPIKES                                                       │
│  ┌─────────────────┐                                                    │
│  │ Cause:          │ Outliers in neighborhood                           │
│  │ Symptom:        │ Extreme curvature values                           │
│  │ Solution:       │ ✅ MAD robust estimator (already implemented)      │
│  │                 │ 🔧 SOR preprocessing (Phase 1)                     │
│  └─────────────────┘                                                    │
│                                                                         │
│  Legend:                                                                │
│  ✅ Already implemented (keep)                                          │
│  🔧 To implement (new in plan)                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

**Author**: GitHub Copilot  
**Date**: October 4, 2025  
**Version**: 1.0
