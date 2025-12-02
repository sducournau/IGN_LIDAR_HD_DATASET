# Version 4.0.0 Release Plan

**Target Release Date:** Q2 2025 (April-June)  
**Status:** Planning Phase  
**Type:** Major Release (Breaking Changes)  
**Current Version:** 3.6.3

---

## 🎯 Executive Summary

Version 4.0.0 represents a **major milestone** for the IGN LiDAR HD library, focusing on:

1. **Configuration System Finalization** - Complete the v4.0 config migration started in 3.6.x
2. **Breaking Changes & Cleanup** - Remove all deprecated code and APIs
3. **API Stabilization** - Establish stable public API for production use
4. **Performance Hardening** - Finalize GPU optimization suite
5. **Production Readiness** - Complete documentation, testing, and CI/CD

### Release Highlights

- ✅ **Unified Configuration System** - Single, clear configuration approach
- 🧹 **Code Cleanup** - Remove 1000+ lines of deprecated code
- 📚 **API Freeze** - Stable public API with semantic versioning commitment
- ⚡ **GPU Performance** - Finalized optimization suite (10-30× speedup)
- 🔒 **Production Grade** - Comprehensive testing, validation, CI/CD
- 📖 **Complete Documentation** - User guides, API reference, migration paths

---

## 📊 Current State Assessment

### Version History Context

| Version | Date       | Focus                           | Status      |
| ------- | ---------- | ------------------------------- | ----------- |
| 3.0.0   | 2024       | Initial restructure             | Released    |
| 3.5.x   | Nov 2024   | GPU optimizations phase 1-3     | Released    |
| 3.6.x   | Nov 2025   | Config v4.0 + GPU phase 4-5     | Released    |
| **4.0.0** | **Q2 2025** | **Breaking changes & cleanup** | **PLANNED** |

### Key Metrics

| Metric                   | v3.6.3 (Current) | v4.0.0 (Target) | Change    |
| ------------------------ | ---------------- | --------------- | --------- |
| Total lines of code      | ~35,000          | ~32,000         | -9%       |
| Deprecated functions     | 45+              | 0               | -100%     |
| Configuration systems    | 3 parallel       | 1 unified       | -67%      |
| Test coverage            | ~78%             | >85%            | +7pp      |
| Documentation pages      | ~50              | ~60             | +20%      |
| GPU speedup (1M points)  | 6.7×             | 8-10×           | +20-50%   |
| API stability score      | Beta             | Stable          | Production|

---

## 🚀 Major Changes & Breaking Changes

### 1. Configuration System (BREAKING) ⚠️

**Current State (v3.6.3):**
- ✅ Config v4.0 implemented but coexists with v3.x
- ⚠️ Multiple config loading paths still supported
- ⚠️ Deprecation warnings for old configs

**v4.0.0 Changes:**
- ❌ **REMOVE** v3.x configuration support (`schema.py`, `schema_simplified.py`)
- ❌ **REMOVE** backward compatibility layer in `config.py`
- ✅ **ENFORCE** v4.0 configuration structure only
- ✅ **SIMPLIFY** config loading to single path

**Breaking Changes:**
```python
# ❌ REMOVED in v4.0 (deprecated in v3.6)
from ign_lidar.config.schema import IGNLiDARConfig  # OLD v3.1 Hydra config
config = IGNLiDARConfig(...)

# ❌ REMOVED in v4.0 (deprecated in v3.2)
config = Config(processor={'lod_level': 'LOD2'})  # Nested processor config

# ✅ REQUIRED in v4.0 (introduced in v3.6)
from ign_lidar import Config
config = Config.preset('lod2_buildings')  # Unified flat config
config = Config(mode='lod2', use_gpu=True)  # Flat structure
```

**YAML Migration:**
```yaml
# ❌ REMOVED v3.x structure
processor:
  lod_level: LOD2
  use_gpu: true
features:
  feature_set: standard

# ✅ REQUIRED v4.0 structure
mode: lod2  # Flat, clear naming
use_gpu: true
features:
  mode: standard
```

**Migration Path:**
- Automatic migration tool: `ign-lidar migrate-config` (already available in v3.6.3)
- All v3.x configs MUST be migrated before upgrading to v4.0
- Tool will create backups and show diffs

---

### 2. Deprecated Feature APIs (BREAKING) ⚠️

**Removed Classes & Functions:**

#### FeatureComputer Class (deprecated in v3.7.0)
```python
# ❌ REMOVED in v4.0
from ign_lidar.features import FeatureComputer
computer = FeatureComputer(...)

# ✅ MIGRATION
from ign_lidar.features import FeatureOrchestrator
orchestrator = FeatureOrchestrator(...)
```

#### Normal Computation Functions (deprecated in v3.5.2)
```python
# ❌ REMOVED in v4.0
from ign_lidar.features.numba_accelerated import (
    compute_normals_from_eigenvectors,
    compute_normals_from_eigenvectors_numpy,
    compute_normals_from_eigenvectors_numba
)

# ✅ MIGRATION
from ign_lidar.features.compute.normals import compute_normals
normals = compute_normals(points, k_neighbors=30)
```

#### Legacy Import Paths (deprecated in v3.1.0)
```python
# ❌ REMOVED in v4.0
from ign_lidar.features.core import compute_normals  # Old path

# ✅ MIGRATION
from ign_lidar.features.compute import compute_normals  # Canonical path
```

**Files to be Deleted:**
- `ign_lidar/config/schema.py` (415 lines) - Old v3.1 Hydra config
- `ign_lidar/config/schema_simplified.py` (~300 lines) - Interim config
- `ign_lidar/features/numba_accelerated.py` - Deprecated normal functions (keep other functions)
- Legacy backward compatibility shims in `__init__.py` files

**Estimated Code Reduction:** ~1,000-1,500 lines

---

### 3. GPU API Consolidation (BREAKING) ⚠️

**Current State:**
- Multiple GPU availability check methods
- Scattered CuPy import patterns
- Redundant memory pool access

**v4.0.0 Changes:**

#### Centralized GPU Management
```python
# ❌ REMOVED in v4.0 (scattered patterns)
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# ❌ REMOVED in v4.0 (deprecated helper)
from ign_lidar.optimization.gpu_wrapper import check_gpu_available

# ✅ REQUIRED in v4.0 (centralized)
from ign_lidar.core.gpu import GPUManager
gpu = GPUManager()
if gpu.gpu_available:
    cp = gpu.get_cupy()
```

#### Memory Pool Access
```python
# ❌ REMOVED in v4.0 (direct access)
import cupy as cp
mempool = cp.get_default_memory_pool()

# ✅ REQUIRED in v4.0 (managed access)
from ign_lidar.core.gpu import GPUManager
gpu = GPUManager()
mempool = gpu.get_memory_pool()
```

**Files to Update:**
- Remove `check_gpu_available()` from `optimization/gpu_wrapper.py`
- Enforce `GPUManager` usage across all GPU code
- Remove scattered `try: import cupy` blocks

---

### 4. Classification Schema Stabilization

**Current State:**
- LOD2: 15 classes (stable)
- LOD3: 30+ classes (evolving)
- ASPRS: 25 classes (stable)

**v4.0.0 Changes:**
- ✅ **FREEZE** LOD2 schema (no more changes)
- ✅ **FREEZE** ASPRS schema (aligned with standard)
- ⚠️ **STABILIZE** LOD3 schema (minor adjustments allowed in v4.x)
- ✅ **DOCUMENT** class definitions and mapping rules

**Breaking Changes:**
- Remove experimental/unused classes
- Finalize class IDs (no more renumbering)
- Lock ASPRS mappings

---

## 🔄 Non-Breaking Improvements

### 1. Performance Enhancements

#### Finalize GPU Optimization Suite
- ✅ Complete Phase 5: Async I/O integration
- ✅ Kernel fusion optimization (Phase 7 continuation)
- ✅ Multi-GPU support (experimental)
- ✅ Benchmark suite integration

**Expected Performance Gains:**
- Target: **8-10× speedup** for 1M points (vs 6.7× in v3.6.3)
- Target: **12-15× speedup** for 5M+ points (vs 10× in v3.6.3)
- Memory efficiency: **<3GB VRAM** for 5M points

#### CPU Optimizations
- Vectorization improvements for NumPy operations
- Numba JIT compilation for hot paths
- Memory-mapped file I/O for large datasets

### 2. Testing & Quality Assurance

**Current Test Suite:**
- 887 valid tests (as of v3.6.3)
- ~78% code coverage
- GPU tests require hardware

**v4.0.0 Targets:**
- **>85% code coverage** (all production code)
- **100% public API coverage**
- **CI/CD regression detection** (<5% performance regression fails build)
- **Automated migration tests** (v3.x → v4.0 configs)

**New Test Categories:**
- Integration tests for full pipeline
- Performance regression tests
- Memory leak detection tests
- Multi-GPU tests (if available)

### 3. Documentation Overhaul

**Current State:**
- Mix of Docusaurus docs and markdown files
- Some outdated examples
- Configuration v4.0 guide added in v3.6.3

**v4.0.0 Documentation Plan:**

#### User Documentation
- ✅ **Quick Start Guide** (updated for v4.0)
- ✅ **Configuration Guide v4.0** (comprehensive reference)
- ✅ **Migration Guide** (v3.x → v4.0)
- 🆕 **Best Practices Guide** (production workflows)
- 🆕 **Performance Tuning Guide** (GPU/CPU optimization)
- 🆕 **Troubleshooting Guide** (common issues)

#### API Documentation
- ✅ Complete API reference (all public classes/functions)
- ✅ Type hints for all public APIs
- ✅ Comprehensive docstrings (Google style)
- 🆕 Interactive examples (Jupyter notebooks)

#### Developer Documentation
- 🆕 **Architecture Guide** (system design)
- 🆕 **Contributing Guide** (development workflow)
- 🆕 **Testing Guide** (test patterns)
- 🆕 **Release Process** (versioning, changelog)

### 4. CLI Enhancements

**Current CLI Commands:**
```bash
ign-lidar process          # Main processing
ign-lidar download         # Download IGN data
ign-lidar migrate-config   # Config migration
```

**v4.0.0 New Commands:**
```bash
ign-lidar validate-config  # Validate configuration file
ign-lidar benchmark       # Run performance benchmarks
ign-lidar info            # System information (GPU, versions)
ign-lidar check           # Environment health check
ign-lidar examples        # Generate example configs
```

### 5. Dependency Management

**Current Dependencies:**
- Python 3.8+ (minimum)
- NumPy, SciPy, scikit-learn (core)
- CuPy, cuML (optional GPU)
- Hydra, OmegaConf (configuration)

**v4.0.0 Updates:**
- ⬆️ **Bump minimum Python to 3.9** (EOL for 3.8 in Oct 2024)
- ⬆️ **Update NumPy minimum to 1.23.0** (type hints improvements)
- ⬆️ **Update CuPy for CUDA 12.x support**
- 🆕 **Add optional PyTorch integration** (dataset compatibility)
- 🔒 **Pin critical dependencies** (avoid breaking changes)

**Dependency Lock Files:**
```bash
requirements.txt           # Core dependencies (pinned versions)
requirements_gpu.txt       # GPU dependencies (pinned versions)
requirements_dev.txt       # Development dependencies (NEW)
requirements_docs.txt      # Documentation dependencies (NEW)
```

---

## 📅 Release Timeline

### Phase 1: Preparation (Month 1-2)

**Weeks 1-2: Audit & Planning**
- ✅ Complete deprecation audit
- ✅ Identify all breaking changes
- ✅ Create migration scripts
- ✅ Update documentation structure

**Weeks 3-4: Pre-release Cleanup**
- 🔄 Remove deprecated code (controlled deletion)
- 🔄 Update test suite for breaking changes
- 🔄 Create v3.7.0 transitional release (final v3.x)

**Deliverables:**
- Comprehensive breaking changes document
- Migration tooling tested on real projects
- v3.7.0 release (last v3.x with all warnings)

### Phase 2: Core Implementation (Month 3-4)

**Weeks 5-6: Configuration Finalization**
- Delete old config system files
- Enforce v4.0 config structure
- Update all examples and tests
- Validate migration paths

**Weeks 7-8: API Stabilization**
- Remove deprecated APIs
- Consolidate GPU management
- Finalize public API surface
- Add comprehensive type hints

**Deliverables:**
- v4.0.0-alpha.1 (internal testing)
- Updated test suite (100% pass rate)
- API reference documentation

### Phase 3: Testing & Polish (Month 5)

**Weeks 9-10: Comprehensive Testing**
- Integration testing with real datasets
- Performance regression testing
- Memory leak testing
- Multi-GPU testing (if available)
- Documentation review

**Weeks 11-12: Beta Release**
- v4.0.0-beta.1 release (public testing)
- Community feedback collection
- Bug fixes and refinements
- Migration assistance

**Deliverables:**
- v4.0.0-beta.1 public release
- Migration guide validated with users
- Performance benchmarks published

### Phase 4: Release (Month 6)

**Weeks 13-14: Release Preparation**
- Final bug fixes
- Documentation polish
- Release notes preparation
- PyPI release preparation

**Week 15: v4.0.0 Release**
- Official v4.0.0 release
- Blog post / announcement
- Social media campaign
- Conference talk (if applicable)

**Week 16: Post-Release Support**
- Monitor issue tracker
- Provide migration support
- Quick patch releases if needed (v4.0.1, v4.0.2)

---

## 🎯 Success Criteria

### Must-Have (Release Blockers)

- ✅ All deprecated code removed
- ✅ All v3.x configs migrate successfully
- ✅ Zero breaking changes without migration path
- ✅ Test coverage >85%
- ✅ All documentation updated
- ✅ Performance equal or better than v3.6.3
- ✅ No regressions in functionality

### Should-Have (High Priority)

- ✅ GPU optimization suite finalized
- ✅ CLI commands expanded
- ✅ Interactive examples (Jupyter)
- ✅ Performance benchmarks published
- ✅ Python 3.9+ support
- ✅ CI/CD fully automated

### Nice-to-Have (Future Enhancements)

- 🔮 Multi-GPU support (basic)
- 🔮 PyTorch dataset integration
- 🔮 Cloud deployment examples
- 🔮 Docker containers
- 🔮 Kubernetes helm charts

---

## 🚧 Migration Guide Preview

### For Users

**Step 1: Upgrade to v3.7.0 First**
```bash
# Install final v3.x release
pip install ign-lidar-hd==3.7.0

# Run with all warnings visible
ign-lidar process --config your_config.yaml
# Review all deprecation warnings
```

**Step 2: Migrate Configuration**
```bash
# Automatic migration
ign-lidar migrate-config your_config.yaml --output config_v4.yaml

# Preview changes
ign-lidar migrate-config your_config.yaml --dry-run --verbose
```

**Step 3: Update Code**
```python
# Update imports
# OLD (v3.x)
from ign_lidar.config.schema import IGNLiDARConfig
from ign_lidar.features import FeatureComputer

# NEW (v4.0)
from ign_lidar import Config
from ign_lidar.features import FeatureOrchestrator
```

**Step 4: Upgrade to v4.0**
```bash
pip install --upgrade ign-lidar-hd==4.0.0
```

**Step 5: Test & Validate**
```bash
# Validate new config
ign-lidar validate-config config_v4.yaml

# Test processing
ign-lidar process --config config_v4.yaml --dry-run

# Run full processing
ign-lidar process --config config_v4.yaml
```

### For Developers

**Update Development Environment**
```bash
# Clone repository
git clone https://github.com/sducournau/IGN_LIDAR_HD_DATASET.git
cd IGN_LIDAR_HD_DATASET
git checkout v4.0.0

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Build documentation
cd docs && npm install && npm run build
```

**Update Dependencies**
```python
# pyproject.toml
[project]
requires-python = ">=3.9"  # Updated from 3.8
dependencies = [
    "numpy>=1.23.0",  # Updated from 1.21.0
    # ... other updated dependencies
]
```

---

## 📊 Risk Assessment

### High Risk

| Risk                           | Impact | Probability | Mitigation                              |
| ------------------------------ | ------ | ----------- | --------------------------------------- |
| Breaking changes anger users   | High   | Medium      | Clear migration guide, v3.7 buffer      |
| Performance regression         | High   | Low         | Comprehensive benchmarks, CI/CD checks  |
| Complex migration fails        | High   | Low         | Automated migration tool, extensive testing |

### Medium Risk

| Risk                           | Impact | Probability | Mitigation                              |
| ------------------------------ | ------ | ----------- | --------------------------------------- |
| Timeline slippage              | Medium | Medium      | Phased release, clear milestones        |
| GPU optimization bugs          | Medium | Medium      | Extensive GPU testing, fallbacks        |
| Documentation gaps             | Medium | Low         | Doc review process, user feedback       |

### Low Risk

| Risk                           | Impact | Probability | Mitigation                              |
| ------------------------------ | ------ | ----------- | --------------------------------------- |
| Dependency conflicts           | Low    | Low         | Pin versions, test in clean env         |
| Community backlash             | Low    | Low         | Clear communication, benefits messaging |

---

## 🎉 Expected Benefits

### For Users

- **Simpler Configuration**: Single, clear config system (no confusion)
- **Better Performance**: 20-50% faster GPU processing
- **Cleaner API**: Stable, well-documented public API
- **Easier Migration**: Automatic tooling for upgrades
- **Better Docs**: Comprehensive guides and examples

### For Developers

- **Cleaner Codebase**: 1000+ lines of dead code removed
- **Easier Maintenance**: Single configuration path
- **Better Testing**: >85% coverage, regression detection
- **Clearer Architecture**: Well-documented design patterns
- **Faster Development**: Stable API reduces refactoring

### For the Project

- **Production Ready**: Stable v4.x series for long-term use
- **Better Adoption**: Clear documentation attracts users
- **Community Growth**: Easier contribution with stable API
- **Performance Leadership**: Best-in-class LiDAR processing
- **Long-term Viability**: Sustainable development model

---

## 📞 Communication Plan

### Pre-Release (Months 1-4)

- **Blog Post**: "Planning v4.0 - What to Expect"
- **GitHub Discussion**: Open forum for feedback
- **Email List**: Update subscribers on progress
- **Documentation**: Early access to migration guide

### Beta Release (Month 5)

- **Announcement**: v4.0.0-beta.1 available
- **Call for Testing**: Request community feedback
- **Live Demo**: Walkthrough of new features
- **Migration Workshop**: Help users upgrade

### Official Release (Month 6)

- **Blog Post**: "IGN LiDAR HD v4.0.0 Released"
- **Release Notes**: Comprehensive changelog
- **Video Tutorial**: "What's New in v4.0"
- **Social Media**: Twitter, LinkedIn announcements
- **Conference**: Present at relevant conferences

### Post-Release (Ongoing)

- **Weekly Updates**: Bug fixes and patches
- **Monthly Newsletter**: Tips, tricks, case studies
- **Quarterly Reviews**: Performance metrics, roadmap updates
- **Annual Survey**: User satisfaction and feature requests

---

## 🔮 Future Roadmap (v4.x Series)

### v4.1.0 (Q3 2025) - Enhancements
- Multi-GPU parallelization (basic)
- PyTorch dataset direct integration
- Enhanced visualization tools
- Performance profiling dashboard

### v4.2.0 (Q4 2025) - Enterprise Features
- Cloud deployment support (AWS, Azure, GCP)
- Kubernetes operators
- Distributed processing (Dask/Ray)
- REST API for processing

### v4.3.0 (Q1 2026) - Advanced ML
- Automatic feature selection
- Transfer learning support
- Model deployment tools
- Real-time processing mode

### v5.0.0 (Q3 2026) - Next Generation
- Complete architecture redesign
- Streaming processing
- Native multi-GPU
- Advanced ML pipelines

---

## ✅ Action Items

### Immediate (This Week)
- [ ] Review this release plan with team
- [ ] Create GitHub milestone for v4.0.0
- [ ] Create GitHub project board for tracking
- [ ] Set up v4.0.0 branch

### Short-term (This Month)
- [ ] Complete deprecation audit (all files)
- [ ] Test migration tool on real-world configs
- [ ] Create v3.7.0 release plan (transition release)
- [ ] Begin documentation restructuring

### Mid-term (Next 2 Months)
- [ ] Implement breaking changes in v4.0-dev branch
- [ ] Update test suite for new APIs
- [ ] Create alpha releases for internal testing
- [ ] Draft comprehensive migration guide

### Long-term (Next 3-6 Months)
- [ ] Public beta release and testing
- [ ] Community feedback integration
- [ ] Final polish and documentation
- [ ] Official v4.0.0 release

---

## 📚 References

- [Configuration v4.0 Implementation Summary](CONFIG_V4_IMPLEMENTATION_SUMMARY.md)
- [Configuration Harmonization Plan](docs/docs/configuration-harmonization-implementation-plan.md)
- [Configuration Guide v4.0](docs/docs/configuration-guide-v4.md)
- [Migration Guide v3→v4](docs/docs/migration-guide-v4.md)
- [CHANGELOG.md](CHANGELOG.md)
- [GitHub Issues](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)

---

## 👥 Contributors & Acknowledgments

**Core Team:**
- Lead Developer: imagodata
- Documentation: [Team members]
- Testing: [Team members]

**Community Contributors:**
- Migration testing and feedback
- Bug reports and feature requests
- Documentation improvements

**Special Thanks:**
- All v3.x users who provided feedback
- Beta testers for early v4.0 releases
- Open source community

---

**Document Version:** 1.0  
**Last Updated:** December 2, 2025  
**Next Review:** January 15, 2025  
**Status:** 📋 DRAFT - Ready for Review
