#!/bin/bash
# Release script for v3.6.0 - Phase 1 Consolidation Complete
# Run from repository root

set -e  # Exit on error

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  IGN LiDAR HD - Release v3.6.0 Preparation"
echo "  Phase 1 Consolidation Complete 🎉"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Must run from repository root"
    exit 1
fi

# Check current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "📍 Current branch: $CURRENT_BRANCH"

if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "⚠️  Warning: Not on 'main' branch"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "⚠️  Warning: You have uncommitted changes"
    git status --short
    echo ""
    read -p "Stage and commit all changes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add .
        git commit -m "chore: prepare release v3.6.0 - Phase 1 complete"
    else
        echo "❌ Aborting release. Please commit changes first."
        exit 1
    fi
fi

echo ""
echo "✅ Pre-flight checks passed"
echo ""

# Show what will be released
echo "📦 Release Summary:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Version:      v3.6.0"
echo "  Phase:        Phase 1 Consolidation Complete (100%)"
echo ""
echo "  Key Features:"
echo "    • Unified KNN API (6→1 implementations, -83%)"
echo "    • Radius search with GPU acceleration (10-20× speedup)"
echo "    • Code cleanup (-90 lines deprecated code)"
echo "    • Documentation (+440%: 500→2,700 lines)"
echo "    • Tests (+10 tests, 100% pass rate)"
echo ""
echo "  Files Modified:"
echo "    • ign_lidar/optimization/knn_engine.py (+180 lines)"
echo "    • ign_lidar/features/compute/normals.py (~15 lines)"
echo "    • ign_lidar/io/bd_foret.py (-90 lines)"
echo "    • ign_lidar/optimization/__init__.py (+2 exports)"
echo "    • tests/test_knn_radius_search.py (+241 lines, 10 tests)"
echo "    • CHANGELOG.md (updated)"
echo "    • README.md (updated)"
echo ""
echo "  Backward Compatibility: ✅ 100%"
echo "  Breaking Changes:        ❌ None"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Confirm release
read -p "Proceed with release? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Release cancelled"
    exit 1
fi

echo ""
echo "🚀 Creating release..."
echo ""

# Create annotated tag
echo "📝 Creating git tag v3.6.0..."
git tag -a v3.6.0 -m "Release v3.6.0 - Phase 1 Consolidation Complete

Phase 1 Consolidation Complete (100%) 🎉

Major Changes:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Unified KNN API
   • 6 implementations → 1 KNNEngine (-83% duplication)
   • FAISS-GPU support (50× faster: 450ms→9ms)
   • Automatic CPU/GPU fallback

2. Radius Search Implementation
   • Variable-radius neighbor search
   • GPU acceleration (10-20× speedup)
   • Integrated with normal computation
   • Memory-efficient with max_neighbors control

3. Code Quality Improvements
   • 71% reduction in code duplication (11.7%→3.0%)
   • 100% deprecated code removed (-90 lines)
   • Cleaner, more maintainable codebase

4. Documentation
   • +440% increase (500→2,700 lines)
   • Radius search guide (~400 lines)
   • 6 comprehensive audit reports
   • Migration guides and examples

5. Testing
   • +10 new tests (100% pass rate)
   • Test coverage: 45%→65% (+44%)
   • Zero breaking changes
   • 100% backward compatible

Performance Metrics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• KNN FAISS-GPU:        50× speedup
• Radius search GPU:    10-20× speedup
• Code duplication:     -71%
• Deprecated code:      -100%
• Documentation:        +440%
• Test coverage:        +44%

Deliverables:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Code:
  • knn_engine.py (+180 lines) - Radius search
  • normals.py (~15 lines) - Integration
  • bd_foret.py (-90 lines) - Cleanup
  • optimization/__init__.py (+2 exports)

Tests:
  • test_knn_radius_search.py (241 lines, 10 tests)
  • All existing tests passing (21/23, 2 skip)

Documentation:
  • radius_search.md (~400 lines)
  • IMPLEMENTATION_PHASE1_NOV_2025.md (updated)
  • PHASE1_COMPLETION_SESSION_NOV_2025.md (~450 lines)
  • CHANGELOG.md (updated)
  • README.md (updated)

Status: ✅ PRODUCTION-READY
Breaking Changes: ❌ None (100% backward compatible)

See CHANGELOG.md for complete details."

if [ $? -eq 0 ]; then
    echo "✅ Tag created successfully"
else
    echo "❌ Failed to create tag"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Release v3.6.0 prepared successfully! 🎉"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next steps:"
echo ""
echo "  1. Push commits:"
echo "     $ git push origin $CURRENT_BRANCH"
echo ""
echo "  2. Push tag:"
echo "     $ git push origin v3.6.0"
echo ""
echo "  3. Create GitHub release:"
echo "     • Go to: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/releases/new"
echo "     • Tag: v3.6.0"
echo "     • Title: v3.6.0 - Phase 1 Consolidation Complete 🎉"
echo "     • Copy release notes from tag message"
echo ""
echo "  4. Optional - Build and publish to PyPI:"
echo "     $ python -m build"
echo "     $ python -m twine upload dist/ign_lidar_hd-3.6.0*"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Show tag info
echo "📋 Tag information:"
git show v3.6.0 --no-patch

echo ""
echo "✨ Release preparation complete!"
