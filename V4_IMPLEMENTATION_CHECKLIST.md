# Version 4.0.0 Implementation Checklist

**Status:** 📋 Planning Phase  
**Target Release:** Q2 2025  
**Last Updated:** December 2, 2025

This document tracks the implementation progress for v4.0.0 release.

---

## 📊 Overall Progress

- **Phase 1 - Preparation:** ⬜ 0/24 tasks (0%)
- **Phase 2 - Core Implementation:** ⬜ 0/28 tasks (0%)
- **Phase 3 - Testing & Polish:** ⬜ 0/18 tasks (0%)
- **Phase 4 - Release:** ⬜ 0/12 tasks (0%)
- **Total:** ⬜ 0/82 tasks (0%)

---

## Phase 1: Preparation (Month 1-2)

### Week 1-2: Audit & Planning

#### Deprecation Audit
- [ ] Scan all Python files for `DeprecationWarning` usage
- [ ] Scan all Python files for `@deprecated` decorators
- [ ] List all deprecated functions in `features/` module
- [ ] List all deprecated functions in `config/` module
- [ ] List all deprecated functions in `optimization/` module
- [ ] List all deprecated imports in `__init__.py` files
- [ ] Create comprehensive deprecation inventory document

#### Breaking Changes Documentation
- [ ] Document all configuration changes (v3.x → v4.0)
- [ ] Document all API changes (removed functions)
- [ ] Document all import path changes
- [ ] Create side-by-side comparison examples
- [ ] Write "What's Breaking in v4.0" blog post

#### Migration Tooling
- [ ] Test `migrate-config` command on 10+ real configs
- [ ] Add validation to migration tool
- [ ] Create rollback mechanism for failed migrations
- [ ] Add dry-run mode improvements
- [ ] Create Python API migration examples
- [ ] Add automated tests for migration tool

### Week 3-4: Pre-release Cleanup

#### v3.7.0 Transitional Release
- [ ] Create v3.7.0 branch from main
- [ ] Add enhanced deprecation warnings (all code)
- [ ] Update all deprecation messages with v4.0 info
- [ ] Add migration guide links to warnings
- [ ] Test v3.7.0 with all deprecation warnings visible
- [ ] Create v3.7.0 changelog
- [ ] Release v3.7.0 to PyPI
- [ ] Announce v3.7.0 as final v3.x release

#### Test Suite Updates
- [ ] Update tests to work with v4.0 APIs
- [ ] Remove tests for deprecated functionality
- [ ] Add tests for migration tool
- [ ] Add tests for new CLI commands
- [ ] Ensure 100% test pass rate on v3.7.0

---

## Phase 2: Core Implementation (Month 3-4)

### Week 5-6: Configuration Finalization

#### Delete Old Config Files
- [ ] **DELETE** `ign_lidar/config/schema.py` (415 lines)
- [ ] **DELETE** `ign_lidar/config/schema_simplified.py` (~300 lines)
- [ ] Remove imports of deleted modules in `__init__.py`
- [ ] Remove imports of deleted modules in tests
- [ ] Update documentation removing old config references

#### Enforce v4.0 Config Structure
- [ ] Remove v3.x backward compatibility in `config.py`
- [ ] Remove `from_legacy_schema()` method
- [ ] Update `Config` class to reject v3.x structure
- [ ] Update preset loader for v4.0 only
- [ ] Add strict validation for v4.0 configs

#### Update Examples
- [ ] Migrate all YAML examples to v4.0 structure
- [ ] Update `examples/TEMPLATE_v3.2.yaml` → `TEMPLATE_v4.0.yaml`
- [ ] Update `examples/config_training_fast_50m_v3.2.yaml`
- [ ] Update `examples/config_asprs_production.yaml`
- [ ] Update `examples/config_multi_scale_adaptive.yaml`
- [ ] Add v4.0 example for each preset
- [ ] Test all example configs

#### Update Config Tests
- [ ] Update `tests/test_config*.py` for v4.0
- [ ] Remove v3.x compatibility tests
- [ ] Add v4.0 validation tests
- [ ] Test all presets load correctly
- [ ] Test config serialization/deserialization

### Week 7-8: API Stabilization

#### Remove FeatureComputer (deprecated v3.7.0)
- [ ] **DELETE** deprecation warnings from `features/feature_computer.py`
- [ ] **DELETE** entire `features/feature_computer.py` file
- [ ] Remove imports in `features/__init__.py`
- [ ] Update all internal code using FeatureComputer
- [ ] Update tests to use FeatureOrchestrator
- [ ] Update documentation/examples

#### Remove Deprecated Normal Functions
- [ ] **DELETE** `compute_normals_from_eigenvectors()` from `features/numba_accelerated.py`
- [ ] **DELETE** `compute_normals_from_eigenvectors_numpy()`
- [ ] **DELETE** `compute_normals_from_eigenvectors_numba()`
- [ ] **DELETE** `_compute_normals_cpu()` from `features/gpu_processor.py`
- [ ] Remove backward compatibility imports
- [ ] Update all call sites to use canonical API
- [ ] Update tests for canonical API

#### Remove Legacy Import Paths
- [ ] Remove `features.core` compatibility layer
- [ ] Remove deprecation warnings from `__init__.py` files
- [ ] Update all internal imports to canonical paths
- [ ] Test that old import paths fail cleanly
- [ ] Update import documentation

#### GPU API Consolidation
- [ ] **DELETE** `check_gpu_available()` from `optimization/gpu_wrapper.py`
- [ ] **DELETE** deprecation shims from `optimization/ground_truth.py`
- [ ] **DELETE** legacy GPU profiler shim from `optimization/gpu_profiler.py`
- [ ] Update all GPU code to use `GPUManager` exclusively
- [ ] Remove all scattered `try: import cupy` blocks (use GPUManager.get_cupy())
- [ ] Update all direct `cp.get_default_memory_pool()` calls
- [ ] Add tests for centralized GPU management

#### Type Hints & Documentation
- [ ] Add comprehensive type hints to all public APIs
- [ ] Add type hints to `Config` class
- [ ] Add type hints to `FeatureOrchestrator`
- [ ] Add type hints to `LiDARProcessor`
- [ ] Add type hints to classification modules
- [ ] Run mypy on entire codebase
- [ ] Fix all type checking errors

---

## Phase 3: Testing & Polish (Month 5)

### Week 9-10: Comprehensive Testing

#### Test Coverage Expansion
- [ ] Measure current test coverage (target baseline)
- [ ] Add tests to reach >85% coverage
- [ ] Add integration tests for full pipeline
- [ ] Add tests for all presets
- [ ] Add tests for CLI commands
- [ ] Add tests for error handling

#### Performance Testing
- [ ] Run benchmarks on v4.0 vs v3.6.3
- [ ] Ensure no performance regressions (±5%)
- [ ] Test GPU optimization suite
- [ ] Test memory efficiency (VRAM usage)
- [ ] Test CPU fallback paths
- [ ] Document performance metrics

#### Memory & Resource Testing
- [ ] Run memory leak detection tools
- [ ] Test with large datasets (10M+ points)
- [ ] Test GPU memory management
- [ ] Test multi-process scenarios
- [ ] Profile memory usage patterns

#### Real-world Dataset Testing
- [ ] Test with 10+ diverse IGN tiles
- [ ] Test LOD2 classification accuracy
- [ ] Test LOD3 classification accuracy
- [ ] Test ASPRS classification accuracy
- [ ] Verify output quality
- [ ] Compare results with v3.6.3

### Week 11-12: Beta Release

#### Beta Preparation
- [ ] Create v4.0.0-beta.1 tag
- [ ] Build distribution packages (wheel, source)
- [ ] Test installation in clean environments
- [ ] Test on Linux, macOS, Windows
- [ ] Test with Python 3.9, 3.10, 3.11, 3.12

#### Documentation Polish
- [ ] Complete API reference documentation
- [ ] Review all user guides
- [ ] Add interactive Jupyter examples
- [ ] Create video tutorials
- [ ] Update README.md for v4.0
- [ ] Update GitHub Pages documentation

#### Beta Release
- [ ] Upload beta to Test PyPI
- [ ] Test installation from Test PyPI
- [ ] Upload beta to production PyPI
- [ ] Announce beta release (blog, email, GitHub)
- [ ] Create beta testing feedback form
- [ ] Monitor feedback and issues

---

## Phase 4: Release (Month 6)

### Week 13-14: Release Preparation

#### Final Bug Fixes
- [ ] Address all critical beta feedback
- [ ] Fix any blocking bugs
- [ ] Review and close related GitHub issues
- [ ] Final code review
- [ ] Final security review

#### Documentation Finalization
- [ ] Complete all documentation sections
- [ ] Proofread all documentation
- [ ] Update all version references to 4.0.0
- [ ] Generate final API documentation
- [ ] Update CHANGELOG.md with v4.0.0 entry
- [ ] Create comprehensive release notes

#### Release Assets
- [ ] Update version in `pyproject.toml` to 4.0.0
- [ ] Update version in `ign_lidar/__init__.py` to 4.0.0
- [ ] Update version in `docs/docusaurus.config.ts`
- [ ] Build final distribution packages
- [ ] Sign release packages (if applicable)
- [ ] Create GitHub release draft

### Week 15: Official Release

#### Release Execution
- [ ] Create v4.0.0 git tag
- [ ] Push tag to GitHub
- [ ] Upload to PyPI
- [ ] Verify PyPI package metadata
- [ ] Create GitHub release from draft
- [ ] Update GitHub Pages documentation

#### Announcements
- [ ] Publish release blog post
- [ ] Send email announcement to mailing list
- [ ] Post on Twitter/LinkedIn
- [ ] Post in relevant forums/communities
- [ ] Update project homepage
- [ ] Add to "Show HN" (if appropriate)

#### Post-Release Monitoring
- [ ] Monitor GitHub issues for problems
- [ ] Monitor PyPI download stats
- [ ] Monitor social media feedback
- [ ] Respond to community questions
- [ ] Track migration success rate

### Week 16: Post-Release Support

#### Quick Patches (if needed)
- [ ] Prepare v4.0.1 for critical bugs (if any)
- [ ] Prepare v4.0.2 for minor issues (if any)
- [ ] Update documentation for patches

#### Community Support
- [ ] Answer questions on GitHub Discussions
- [ ] Provide migration assistance
- [ ] Collect feature requests for v4.1
- [ ] Create FAQ based on common questions

---

## Ongoing Tasks (Throughout All Phases)

### Documentation
- [ ] Keep migration guide up-to-date
- [ ] Update API documentation as code changes
- [ ] Add new examples as features evolve
- [ ] Review and improve existing docs

### Communication
- [ ] Weekly progress updates (internal)
- [ ] Monthly blog posts (public)
- [ ] GitHub Discussions engagement
- [ ] Email updates to subscribers

### Quality Assurance
- [ ] Run CI/CD on every commit
- [ ] Monitor test coverage trends
- [ ] Review code quality metrics
- [ ] Address static analysis warnings

---

## Success Metrics Tracking

### Code Quality
- [ ] Test coverage reaches >85%
- [ ] Mypy type checking passes with no errors
- [ ] Pylint/Flake8 score >9.0/10
- [ ] Zero critical bugs in issue tracker
- [ ] All deprecation code removed

### Performance
- [ ] No regression vs v3.6.3 (±5%)
- [ ] GPU speedup >8× for 1M points
- [ ] GPU speedup >12× for 5M points
- [ ] Memory usage <3GB VRAM for 5M points
- [ ] CI/CD benchmarks all passing

### Documentation
- [ ] 100% of public API documented
- [ ] All examples tested and working
- [ ] Migration guide validated by users
- [ ] >90% positive documentation feedback

### Community
- [ ] >80% successful migrations (no issues)
- [ ] <10 critical issues in first month
- [ ] >100 downloads in first week
- [ ] Positive feedback in issue tracker

---

## Risk Mitigation

### High-Priority Risks
- [ ] Create v3.7.0 buffer release (DONE/IN PROGRESS/BLOCKED)
- [ ] Comprehensive migration testing (DONE/IN PROGRESS/BLOCKED)
- [ ] Performance regression CI/CD (DONE/IN PROGRESS/BLOCKED)
- [ ] Rollback plan documented (DONE/IN PROGRESS/BLOCKED)

### Medium-Priority Risks
- [ ] Timeline buffer of 2 weeks added (DONE/IN PROGRESS/BLOCKED)
- [ ] GPU testing on multiple hardware (DONE/IN PROGRESS/BLOCKED)
- [ ] Documentation review process (DONE/IN PROGRESS/BLOCKED)

---

## Dependencies & Blockers

### External Dependencies
- [ ] Python 3.9 EOL timeline confirmed
- [ ] CuPy CUDA 12.x compatibility verified
- [ ] NumPy 2.0 compatibility verified (if releasing)
- [ ] PyPI release permissions confirmed

### Internal Blockers
- [ ] None currently identified

### Community Feedback
- [ ] Gather feedback on breaking changes
- [ ] Validate migration tooling with users
- [ ] Collect performance benchmarks from community

---

## Notes & Decisions

### Key Decisions Log
- **2025-12-02:** Release plan approved (this document created)
- **TBD:** Python minimum version confirmed (3.9)
- **TBD:** Configuration v3.x support dropped confirmed
- **TBD:** Beta release date confirmed

### Open Questions
- Should v4.0 support Python 3.8 or require 3.9+?
- Should multi-GPU support be in v4.0 or v4.1?
- Should PyTorch integration be in v4.0 or v4.1?
- Should we provide Docker images with v4.0?

### Lessons Learned
- (To be filled during implementation)

---

**Last Review:** December 2, 2025  
**Next Review:** December 16, 2025  
**Status:** 📋 Active Planning

**Maintainer:** imagodata  
**Contributors:** [To be added]
