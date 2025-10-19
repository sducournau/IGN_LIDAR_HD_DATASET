# Implementation Plan - Classification Optimization

## Parcel-Based Clustering with Multi-Feature Fusion

**Date:** October 19, 2025  
**Priority:** HIGH  
**Estimated Duration:** 4 weeks

---

## 🎉 PROGRESS UPDATE - PHASE 1 COMPLETE

**Status:** ✅ **Phase 1 Implementation Complete** (October 19, 2025)

### What's Done

✅ **Core Module:** `ign_lidar/core/modules/parcel_classifier.py` (806 lines)

- Parcel-based clustering using cadastral parcels
- 6-type classification (Forest, Agriculture, Building, Road, Water, Mixed)
- Multi-feature decision tree with confidence scores
- Point-level refinement with type-specific strategies
- Ground truth integration (Cadastre, BD Forêt, RPG)

✅ **Test Suite:** `tests/test_parcel_classifier.py` (19 tests, 100% passing)

- Configuration tests
- Feature computation tests
- Parcel classification tests
- Point refinement tests
- Ground truth integration tests
- Full pipeline integration test

✅ **Demo Application:** `examples/demo_parcel_classification.py`

- End-to-end demonstration
- CLI interface with configurable parameters
- Statistics export to CSV

✅ **Documentation:**

- PHASE1_PARCEL_CLASSIFICATION_COMPLETE.md (full technical documentation)
- PHASE1_QUICK_SUMMARY.md (executive summary)
- Comprehensive docstrings and code comments

### Next Immediate Steps

🔄 **Phase 1 Integration** (1-2 days):

1. Integrate ParcelClassifier into `advanced_classification.py` as Stage 0
2. Add configuration options to YAML files
3. Create benchmark comparing with/without parcel classification
4. Validate on test datasets (Versailles, ISPRS)

Then proceed to **Phase 2** (Week 2) as originally planned.

---

## Quick Reference

### Core Innovations

1. **Parcel-Based Clustering**: Group points by cadastral parcels for batch processing
2. **Multi-Level NDVI**: 6-level adaptive thresholds (0.6, 0.5, 0.4, 0.3, 0.2, 0.15)
3. **Multi-Feature Fusion**: Combine geometric + radiometric + ground truth
4. **Material Classification**: RGB + NIR spectral signatures
5. **GPU Acceleration**: 100-1000× speedup for verticality/spatial queries

### Expected Results

- **60-75% faster** processing (18M points: 15min → 3-5min)
- **+12-15%** accuracy improvement (78% → 92%)
- **+17%** vegetation detection (75% → 92%)
- **Spatially coherent** results within parcels

---

## Phase 1: Parcel-Based Clustering [Week 1]

### Priority: CRITICAL

### Deliverables

1. New module: `ign_lidar/core/modules/parcel_classifier.py`
2. Integration with `advanced_classification.py`
3. Configuration updates

### Implementation

```python
class ParcelClassifier:
    """Main parcel-based classification engine."""

    def classify_by_parcels(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        cadastre: gpd.GeoDataFrame,
        bd_foret: Optional[gpd.GeoDataFrame] = None,
        rpg: Optional[gpd.GeoDataFrame] = None
    ) -> np.ndarray:
        """
        Classify points using parcel-based clustering.

        Workflow:
        1. Group points by cadastral parcel
        2. Compute parcel-level features (mean NDVI, height, etc.)
        3. Classify parcel type (forest, agriculture, building, road, etc.)
        4. Refine point-level labels within each parcel
        """
        # 1. Group by parcel
        parcel_groups = self._group_by_parcels(points, cadastre)

        # 2. Classify each parcel
        labels = np.zeros(len(points), dtype=np.uint8)

        for parcel_id, point_indices in parcel_groups.items():
            parcel_points = points[point_indices]
            parcel_features = {k: v[point_indices] for k, v in features.items()}

            # Compute parcel aggregate features
            parcel_stats = self.compute_parcel_features(
                parcel_points,
                parcel_features
            )

            # Match with ground truth
            bd_foret_match = self._match_bd_foret(parcel_id, bd_foret)
            rpg_match = self._match_rpg(parcel_id, rpg)

            # Classify parcel type
            parcel_type, confidence = self.classify_parcel_type(
                parcel_stats,
                bd_foret_match,
                rpg_match
            )

            # Refine point-level labels
            parcel_labels = self.refine_parcel_points(
                parcel_type,
                parcel_points,
                parcel_features,
                bd_foret_match,
                rpg_match
            )

            labels[point_indices] = parcel_labels

        return labels
```

### Integration Points

**File:** `ign_lidar/core/modules/advanced_classification.py`

```python
class AdvancedClassifier:
    def __init__(self, ..., use_parcel_clustering: bool = True):
        self.use_parcel_clustering = use_parcel_clustering
        if use_parcel_clustering:
            from .parcel_classifier import ParcelClassifier
            self.parcel_classifier = ParcelClassifier()

    def classify_points(self, points, ground_truth_features, ...):
        # STAGE 0: Parcel-based clustering (NEW)
        if self.use_parcel_clustering and 'cadastre' in ground_truth_features:
            labels = self.parcel_classifier.classify_by_parcels(
                points=points,
                features={
                    'ndvi': ndvi,
                    'height': height,
                    'planarity': planarity,
                    'verticality': verticality,
                    'curvature': curvature,
                    'normals': normals
                },
                cadastre=ground_truth_features.get('cadastre'),
                bd_foret=ground_truth_features.get('bd_foret'),
                rpg=ground_truth_features.get('rpg')
            )

            coverage = np.sum(labels != ASPRS_UNCLASSIFIED) / len(labels) * 100
            logger.info(f"Parcel clustering: {coverage:.1f}% classified")
        else:
            labels = np.full(len(points), ASPRS_UNCLASSIFIED, dtype=np.uint8)

        # Continue with existing stages for unclassified points
        # ...
```

### Configuration

```yaml
# config_asprs_v5.1_parcel_optimized.yaml

parcel_classification:
  enabled: true
  use_cadastre: true
  use_bd_foret: true
  use_rpg: true
  min_parcel_points: 20
  parcel_confidence_threshold: 0.6

data_sources:
  cadastre:
    enabled: true
    cache_dir: "cache/cadastre"
  bd_foret:
    enabled: true
    cache_dir: "cache/bd_foret"
  rpg:
    enabled: true
    year: 2023
    cache_dir: "cache/rpg"
```

### Tests

```python
# tests/test_parcel_classifier.py

def test_parcel_grouping():
    """Test grouping points by parcel."""
    points = np.random.rand(1000, 3)
    cadastre = create_mock_cadastre_gdf(n_parcels=10)

    classifier = ParcelClassifier()
    labels = classifier.classify_by_parcels(points, {}, cadastre)

    assert len(labels) == len(points)
    assert np.sum(labels != 0) > 0

def test_parcel_feature_aggregation():
    """Test computing parcel-level features."""
    parcel_features = {
        'ndvi': np.array([0.5, 0.6, 0.55]),
        'height': np.array([5.0, 4.5, 5.5])
    }

    classifier = ParcelClassifier()
    stats = classifier.compute_parcel_features(None, parcel_features)

    assert 'mean_ndvi' in stats
    assert abs(stats['mean_ndvi'] - 0.55) < 0.01
```

---

## Phase 2: Multi-Level NDVI & Material Classification [Week 2]

### Priority: HIGH

### Deliverables

1. Multi-level NDVI classification in `advanced_classification.py`
2. New module: `ign_lidar/core/modules/material_classification.py`
3. Remove BD Topo vegetation from priority order

### Multi-Level NDVI Implementation

```python
# ign_lidar/core/modules/advanced_classification.py

NDVI_THRESHOLDS = {
    'dense_forest': 0.60,      # Dense forest
    'healthy_trees': 0.50,     # Healthy trees
    'moderate_veg': 0.40,      # Shrubs/bushes
    'grass': 0.30,             # Grass/low veg
    'sparse_veg': 0.20,        # Sparse vegetation
    'non_veg': 0.15,           # Non-vegetation
}

def _classify_vegetation_multi_level(
    self,
    ndvi: np.ndarray,
    height: np.ndarray,
    curvature: np.ndarray,
    planarity: np.ndarray,
    nir: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-level NDVI classification with feature validation.

    Returns:
        (labels, confidence_scores)
    """
    n_points = len(ndvi)
    labels = np.zeros(n_points, dtype=np.uint8)
    confidence = np.zeros(n_points, dtype=np.float32)

    # Level 1: Dense forest (NDVI >= 0.6)
    mask = (ndvi >= 0.60)
    valid = mask & (curvature > 0.25) & (planarity < 0.6)
    labels[valid] = ASPRS_HIGH_VEGETATION
    confidence[valid] = 0.95

    # Level 2: Healthy trees (0.5 <= NDVI < 0.6)
    mask = (ndvi >= 0.50) & (ndvi < 0.60) & (labels == 0)
    valid = mask & (curvature > 0.2) & (planarity < 0.65)
    high_mask = valid & (height > 2.0)
    med_mask = valid & (height <= 2.0)
    labels[high_mask] = ASPRS_HIGH_VEGETATION
    labels[med_mask] = ASPRS_MEDIUM_VEGETATION
    confidence[valid] = 0.85

    # Level 3-5: Similar pattern for lower NDVI ranges
    # ...

    # Optional: NIR/Red ratio validation
    if nir is not None:
        red = # extract from RGB if available
        nir_red_ratio = nir / (red + 1e-8)

        # High ratio (>3.0) strongly indicates vegetation
        strong_veg = (nir_red_ratio > 3.0) & (ndvi >= 0.3)
        labels[strong_veg & (height > 2.0)] = ASPRS_HIGH_VEGETATION
        confidence[strong_veg] = 0.95

    return labels, confidence
```

### Material Classification Module

```python
# ign_lidar/core/modules/material_classification.py

class MaterialClassifier:
    """Classify materials using RGB + NIR spectral signatures."""

    def classify_by_spectral_signature(
        self,
        rgb: np.ndarray,
        nir: np.ndarray,
        intensity: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Material-based classification.

        Spectral Signatures:
        - Vegetation: NIR > 0.4, NDVI > 0.4, NIR/Red > 2.5
        - Concrete: NIR 0.2-0.35, brightness 0.5-0.7
        - Asphalt: NIR < 0.2, brightness < 0.3
        - Water: NIR < 0.1, NDVI < -0.05
        - Metal: High intensity, NIR 0.3-0.5
        """
        n_points = len(rgb)
        labels = np.zeros(n_points, dtype=np.uint8)

        red, green, blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        brightness = np.mean(rgb, axis=1)
        ndvi = (nir - red) / (nir + red + 1e-8)
        nir_red_ratio = nir / (red + 1e-8)

        # Vegetation
        veg_mask = (nir > 0.4) & (ndvi > 0.4) & (nir_red_ratio > 2.5)
        labels[veg_mask] = ASPRS_MEDIUM_VEGETATION

        # Water
        water_mask = (nir < 0.1) & (ndvi < -0.05) & (brightness < 0.4)
        labels[water_mask] = ASPRS_WATER

        # Concrete
        concrete_mask = (
            (nir >= 0.2) & (nir < 0.35) &
            (brightness > 0.5) & (brightness < 0.7) &
            (ndvi < 0.15)
        )
        labels[concrete_mask] = ASPRS_BUILDING

        # Asphalt
        asphalt_mask = (nir < 0.2) & (brightness < 0.3) & (ndvi < 0.1)
        labels[asphalt_mask] = ASPRS_ROAD

        # Metal roofs
        if intensity is not None:
            intensity_norm = intensity / (np.max(intensity) + 1e-8)
            metal_mask = (intensity_norm > 0.7) & (ndvi < 0.2)
            labels[metal_mask] = ASPRS_BUILDING

        return labels, {
            'vegetation': np.sum(veg_mask),
            'water': np.sum(water_mask),
            'concrete': np.sum(concrete_mask),
            'asphalt': np.sum(asphalt_mask)
        }
```

### Remove BD Topo Vegetation

```python
# advanced_classification.py

def _classify_by_ground_truth(
    self,
    ...,
    use_bd_topo_vegetation: bool = False  # NEW: Default False
):
    """Ground truth with optional vegetation."""

    priority_order = []

    # REMOVED: vegetation from priority
    # if use_bd_topo_vegetation and 'vegetation' in ground_truth_features:
    #     priority_order.append(('vegetation', ASPRS_MEDIUM_VEGETATION))

    priority_order.extend([
        ('water', ASPRS_WATER),
        ('roads', ASPRS_ROAD),
        ('buildings', ASPRS_BUILDING),
        ...
    ])
```

### Configuration

```yaml
advanced_classification:
  # Remove BD Topo vegetation
  use_bd_topo_vegetation: false

  # Multi-level NDVI
  ndvi_thresholds:
    dense_forest: 0.60
    healthy_trees: 0.50
    moderate_veg: 0.40
    grass: 0.30
    sparse_veg: 0.20
    non_veg: 0.15

  # Feature validation
  feature_validation:
    enabled: true
    require_curvature: true
    require_low_planarity: true
    require_nir_ratio: true

  # Material classification
  material_classification:
    enabled: true
    use_nir_red_ratio: true
    use_intensity: true

data_sources:
  bd_topo:
    features:
      vegetation: false # Disabled for classification
```

---

## Phase 3: GPU Acceleration [Week 3]

### Priority: MEDIUM

### Deliverables

1. GPU verticality computation
2. GPU spatial queries (cuSpatial)
3. Clustering optimization (DBSCAN)

### GPU Verticality

```python
# ign_lidar/core/modules/geometric_rules.py

def compute_verticality_gpu_batch(
    points: np.ndarray,
    search_radius: float = 1.0,
    batch_size: int = 500000
) -> np.ndarray:
    """GPU-accelerated verticality with batch processing."""
    try:
        import cupy as cp
        from cuml.neighbors import NearestNeighbors

        n_points = len(points)
        verticality = np.zeros(n_points, dtype=np.float32)
        n_batches = (n_points + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_points)

            # GPU processing
            points_gpu = cp.asarray(points)
            batch_gpu = cp.asarray(points[start:end])

            nn = NearestNeighbors(n_neighbors=50)
            nn.fit(points_gpu)
            distances, indices = nn.radius_neighbors(batch_gpu, radius=search_radius)

            # Compute verticality on GPU
            batch_vert = cp.zeros(end - start)
            for i in range(len(batch_gpu)):
                neighbors = points_gpu[indices[i]]
                z_extent = cp.ptp(neighbors[:, 2])
                xy_extent = cp.maximum(cp.ptp(neighbors[:, 0]), cp.ptp(neighbors[:, 1]))
                if xy_extent > 0.01:
                    batch_vert[i] = cp.minimum(1.0, z_extent / xy_extent / 5.0)

            verticality[start:end] = batch_vert.get()

        return verticality

    except ImportError:
        logger.warning("RAPIDS not available, using CPU")
        return compute_verticality_cpu(points, search_radius)
```

### Configuration

```yaml
gpu_optimization:
  enable_gpu_verticality: true
  enable_gpu_spatial_queries: true
  gpu_batch_size: 500000
```

---

## Phase 4: Testing & Documentation [Week 4]

### Deliverables

1. Unit tests for all new modules
2. Integration tests
3. Performance benchmarks
4. User documentation
5. Migration guide

### Test Plan

```python
# Unit tests
tests/test_parcel_classifier.py
tests/test_multi_level_ndvi.py
tests/test_material_classification.py
tests/test_gpu_acceleration.py

# Integration tests
tests/test_classification_pipeline.py
tests/test_accuracy_benchmark.py
tests/test_performance_benchmark.py

# Validation
- ISPRS Vaihingen dataset
- Semantic3D dataset
- Real-world Versailles tile
```

### Documentation Updates

```
docs/guides/parcel-based-classification.md
docs/guides/multi-feature-fusion.md
docs/guides/gpu-optimization.md
docs/migration/v4-to-v5.1.md
```

---

## Performance Targets

### Processing Speed

| Dataset Size | Current | Target  | Improvement   |
| ------------ | ------- | ------- | ------------- |
| 5M points    | 5 min   | 1 min   | 80% faster    |
| 18M points   | 15 min  | 3-5 min | 67-80% faster |
| 50M points   | 45 min  | 10 min  | 78% faster    |

### Classification Accuracy

| Class      | Current | Target | Improvement |
| ---------- | ------- | ------ | ----------- |
| Overall    | 82%     | 94%    | +12%        |
| Vegetation | 75%     | 92%    | +17%        |
| Buildings  | 85%     | 94%    | +9%         |
| Roads      | 80%     | 90%    | +10%        |
| Ground     | 88%     | 95%    | +7%         |

---

## Risk Mitigation

### Technical Risks

1. **Cadastre data availability**

   - Mitigation: Fallback to non-parcel classification
   - Impact: Moderate (still usable without parcels)

2. **GPU memory constraints**

   - Mitigation: Batch processing with configurable size
   - Impact: Low (CPU fallback available)

3. **Performance regression**
   - Mitigation: Comprehensive benchmarking
   - Impact: Low (optional features can be disabled)

### Integration Risks

1. **Breaking changes**

   - Mitigation: Backward compatibility maintained
   - Impact: Low (parcel clustering is optional)

2. **Configuration complexity**
   - Mitigation: Sensible defaults, migration guide
   - Impact: Low (existing configs work unchanged)

---

## Success Criteria

### Must Have

- ✅ Parcel-based clustering working
- ✅ Multi-level NDVI implemented
- ✅ Classification accuracy > 90%
- ✅ Processing time reduced by 50%+
- ✅ All tests passing

### Should Have

- ✅ GPU acceleration working
- ✅ Material classification integrated
- ✅ BD Forêt and RPG integration
- ✅ Documentation complete

### Nice to Have

- ⭐ Real-time visualization of parcels
- ⭐ Interactive parcel statistics
- ⭐ Automatic parameter tuning
- ⭐ Export parcel-level reports

---

## Next Steps

### Immediate Actions (This Week)

1. Create `parcel_classifier.py` skeleton
2. Implement `group_by_parcels()` method
3. Create test dataset with mock cadastre
4. Run initial feasibility tests

### Communication

- Weekly progress updates
- Demo at end of each phase
- Benchmark results shared weekly
- Documentation updated incrementally

---

**Status:** Ready to Begin Phase 1  
**Start Date:** October 21, 2025  
**Expected Completion:** November 18, 2025

**For detailed technical specifications, see:**  
`COMPREHENSIVE_CLASSIFICATION_AUDIT_PARCEL_OPTIMIZATION.md`
