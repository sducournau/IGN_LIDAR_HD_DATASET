#!/bin/bash
# Quick Start Guide for Package Consolidation
# Run this script to get started with Phase 1 consolidation

set -e  # Exit on error

echo "================================================================"
echo "IGN LiDAR HD Package Consolidation - Quick Start"
echo "================================================================"
echo ""

# Check we're in the right directory
if [ ! -d "ign_lidar" ]; then
    echo "❌ Error: Must run from project root directory"
    exit 1
fi

echo "✅ Running from project root"
echo ""

# Function to print section headers
section() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# Check Python version
section "1. Environment Check"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "❌ Python 3.8+ required"
    exit 1
fi
echo "✅ Python version OK"

# Check git status
if ! git diff-index --quiet HEAD -- 2>/dev/null; then
    echo "⚠️  Warning: You have uncommitted changes"
    echo "   Consider committing or stashing before proceeding"
    read -p "   Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run tests to establish baseline
section "2. Running Baseline Tests"
echo "Running test suite to establish baseline..."

if pytest tests/ -v --tb=short -q > test_baseline.log 2>&1; then
    echo "✅ All tests passing - baseline established"
    test_count=$(grep -c "PASSED" test_baseline.log || echo "0")
    echo "   Tests passed: $test_count"
else
    echo "⚠️  Some tests failing - check test_baseline.log"
    echo "   You may want to fix these before consolidating"
fi

# Analyze code structure
section "3. Code Analysis"

echo "📊 Analyzing codebase structure..."
echo ""

# Count files and lines
total_files=$(find ign_lidar -name "*.py" -type f | wc -l)
total_lines=$(find ign_lidar -name "*.py" -type f -exec wc -l {} + | tail -1 | awk '{print $1}')

echo "Total Python files: $total_files"
echo "Total lines of code: $total_lines"
echo ""

# Find largest files
echo "📏 Largest modules (Top 10):"
find ign_lidar -name "*.py" -type f -exec wc -l {} + | sort -rn | head -11 | tail -10 | \
while read lines file; do
    printf "   %6d LOC  %s\n" "$lines" "$(basename $file)"
done
echo ""

# Check for the specific duplicate function issue
echo "🔍 Checking for known issues..."
if grep -n "def compute_verticality" ign_lidar/features/features.py | head -2 | tail -1 | grep -q "877"; then
    echo "   ❌ Found: Duplicate compute_verticality at line 877"
    echo "      This is the CRITICAL issue from audit report"
else
    echo "   ✅ Duplicate compute_verticality already fixed"
fi

# Search for other potential duplicates
duplicate_count=$(find ign_lidar -name "*.py" -exec grep -l "def compute_normals" {} + | wc -l)
echo "   ℹ️  compute_normals found in $duplicate_count files"

duplicate_count=$(find ign_lidar -name "*.py" -exec grep -l "def compute_curvature" {} + | wc -l)
echo "   ℹ️  compute_curvature found in $duplicate_count files"

# Create working branch
section "4. Creating Working Branch"

branch_name="refactor/consolidation-phase1-$(date +%Y%m%d)"
echo "Creating branch: $branch_name"

if git show-ref --verify --quiet "refs/heads/$branch_name"; then
    echo "⚠️  Branch $branch_name already exists"
    read -p "   Delete and recreate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git branch -D "$branch_name"
        git checkout -b "$branch_name"
    fi
else
    git checkout -b "$branch_name"
    echo "✅ Branch created"
fi

# Prepare directory structure
section "5. Preparing Directory Structure"

echo "Creating new directory structure for Phase 1..."

# Create features/core directory if it doesn't exist
if [ ! -d "ign_lidar/features/core" ]; then
    mkdir -p ign_lidar/features/core
    touch ign_lidar/features/core/__init__.py
    echo "✅ Created ign_lidar/features/core/"
else
    echo "✅ ign_lidar/features/core/ already exists"
fi

# Generate initial report
section "6. Generating Initial Analysis Report"

echo "Running duplication analysis..."
if [ -f "scripts/analyze_duplication.py" ]; then
    python3 scripts/analyze_duplication.py --output consolidation_analysis_baseline.json --module features || true
    echo "✅ Analysis saved to consolidation_analysis_baseline.json"
else
    echo "⚠️  Analysis script not found - skipping"
fi

# Create task checklist
section "7. Phase 1 Task Checklist"

cat > CONSOLIDATION_CHECKLIST.md << 'EOF'
# Phase 1 Consolidation Checklist

## Week 1: Critical Fixes

### Task 1.1: Fix Duplicate Function (CRITICAL)
- [ ] Backup features.py
- [ ] Remove duplicate `compute_verticality` at line 877
- [ ] Run tests: `pytest tests/test_*.py -v`
- [ ] Verify imports: `python -c "from ign_lidar.features import compute_verticality"`
- [ ] Commit fix

### Task 1.2: Create Feature Core Module
- [ ] Create `ign_lidar/features/core/normals.py`
- [ ] Implement `compute_normals_cpu()`
- [ ] Implement `compute_normals_with_eigenvalues()`
- [ ] Add tests in `tests/test_features_core.py`
- [ ] Update `features.py` to use core functions

### Task 1.3: Extract Curvature Functions
- [ ] Create `ign_lidar/features/core/curvature.py`
- [ ] Extract curvature computation
- [ ] Update all callers
- [ ] Add tests

## Week 2: Memory Consolidation

### Task 1.4: Merge Memory Modules
- [ ] Review all 3 memory modules
- [ ] Create unified `core/memory.py`
- [ ] Migrate functions from `memory_utils.py`
- [ ] Migrate GPU functions from `modules/memory.py`
- [ ] Add backward compatibility shim
- [ ] Update imports across codebase
- [ ] Run full test suite

## Progress Tracking

Started: [DATE]
Current Task: Task 1.1
Completed Tasks: 0/7
Estimated Completion: [DATE]

## Notes

- Test baseline established on [DATE]
- [TOTAL] tests passing before consolidation
- Backup created: [BRANCH/TAG]
EOF

echo "✅ Checklist created: CONSOLIDATION_CHECKLIST.md"

# Final instructions
section "8. Next Steps"

cat << 'EOF'
You're now ready to start Phase 1 consolidation! 🚀

Quick Reference:

📋 Review the plan:
   - Read CONSOLIDATION_ROADMAP.md for full details
   - Check CONSOLIDATION_CHECKLIST.md for task list
   - Review PACKAGE_AUDIT_REPORT.md for context

🔧 Start with Task 1.1 (Critical Fix):
   1. Edit ign_lidar/features/features.py
   2. Remove duplicate compute_verticality at line ~877
   3. Run: pytest tests/ -v
   4. Commit: git commit -m "fix: remove duplicate compute_verticality"

📊 Monitor progress:
   - Run tests frequently: pytest tests/ -v
   - Check coverage: pytest tests/ --cov=ign_lidar
   - Run analysis: python scripts/analyze_duplication.py

🆘 Need help?
   - Refer to CONSOLIDATION_ROADMAP.md
   - Check test_baseline.log for pre-consolidation test results
   - Review consolidation_analysis_baseline.json for metrics

Good luck! 🎯
EOF

echo ""
echo "================================================================"
echo "Setup Complete! ✨"
echo "================================================================"
echo ""
echo "Current branch: $(git branch --show-current)"
echo "Files ready for editing"
echo ""
echo "Run this to get started:"
echo "  nano ign_lidar/features/features.py  # or your editor of choice"
echo ""
