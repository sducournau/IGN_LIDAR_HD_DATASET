#!/bin/bash
#
# Phase 1 Pre-Flight Check
#
# Run this script before starting Phase 1 implementation to verify:
# - Environment is correctly configured
# - All dependencies are installed
# - Tests are passing
# - Git state is clean
# - Analysis is complete

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
CHECKS_PASSED=0
CHECKS_FAILED=0
WARNINGS=0

echo "=========================================="
echo "🚀 Phase 1 Pre-Flight Check"
echo "=========================================="
echo ""

# Helper functions
check_pass() {
    echo -e "${GREEN}✅ PASS${NC}: $1"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
}

check_fail() {
    echo -e "${RED}❌ FAIL${NC}: $1"
    echo -e "   ${RED}→${NC} $2"
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
}

check_warn() {
    echo -e "${YELLOW}⚠️  WARN${NC}: $1"
    echo -e "   ${YELLOW}→${NC} $2"
    WARNINGS=$((WARNINGS + 1))
}

check_info() {
    echo -e "${BLUE}ℹ️  INFO${NC}: $1"
}

# Check 1: Python version
echo "📋 Checking Python environment..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
        check_pass "Python $PYTHON_VERSION (>= 3.8 required)"
    else
        check_fail "Python $PYTHON_VERSION" "Need Python 3.8+. Current: $PYTHON_VERSION"
    fi
else
    check_fail "Python not found" "Install Python 3.8+ first"
fi

# Check 2: Virtual environment
if [ -n "$VIRTUAL_ENV" ] || [ -n "$CONDA_DEFAULT_ENV" ]; then
    ENV_NAME=${CONDA_DEFAULT_ENV:-$(basename $VIRTUAL_ENV)}
    check_pass "Virtual environment active: $ENV_NAME"
else
    check_warn "No virtual environment detected" "Recommended: create venv or conda env"
fi

# Check 3: Required packages
echo ""
echo "📦 Checking Python packages..."

# Map package names to import names
declare -A PKG_IMPORT_MAP
PKG_IMPORT_MAP["numpy"]="numpy"
PKG_IMPORT_MAP["pytest"]="pytest"
PKG_IMPORT_MAP["laspy"]="laspy"
PKG_IMPORT_MAP["scikit-learn"]="sklearn"
PKG_IMPORT_MAP["scipy"]="scipy"
PKG_IMPORT_MAP["pyyaml"]="yaml"

for pkg in numpy pytest laspy scikit-learn scipy pyyaml; do
    import_name="${PKG_IMPORT_MAP[$pkg]}"
    if python3 -c "import $import_name" 2>/dev/null; then
        VERSION=$(python3 -c "import $import_name; print(getattr($import_name, '__version__', 'unknown'))" 2>/dev/null)
        check_pass "$pkg ($VERSION)"
    else
        check_fail "$pkg not installed" "Run: pip install $pkg"
    fi
done

# Check 4: Optional packages
OPTIONAL_PACKAGES=("cupy")
for pkg in "${OPTIONAL_PACKAGES[@]}"; do
    if python3 -c "import $pkg" 2>/dev/null; then
        VERSION=$(python3 -c "import $pkg; print($pkg.__version__)" 2>/dev/null)
        check_info "Optional: $pkg ($VERSION) available for GPU acceleration"
    else
        check_info "Optional: $pkg not installed (GPU features will be disabled)"
    fi
done

# Check 5: Git status
echo ""
echo "📁 Checking Git repository..."
if git rev-parse --git-dir > /dev/null 2>&1; then
    check_pass "Git repository detected"
    
    # Check for uncommitted changes
    if git diff-index --quiet HEAD -- 2>/dev/null; then
        check_pass "Working directory clean"
    else
        check_warn "Uncommitted changes detected" "Consider committing or stashing before Phase 1"
    fi
    
    # Check current branch
    CURRENT_BRANCH=$(git branch --show-current 2>/dev/null)
    if [ "$CURRENT_BRANCH" = "main" ] || [ "$CURRENT_BRANCH" = "master" ]; then
        check_warn "On main/master branch" "Recommended: create feature branch for Phase 1"
        echo -e "   ${YELLOW}→${NC} Run: git checkout -b refactor/phase1-consolidation-$(date +%Y%m%d)"
    else
        check_pass "On feature branch: $CURRENT_BRANCH"
    fi
else
    check_fail "Not a Git repository" "Initialize with: git init"
fi

# Check 6: Analysis report exists
echo ""
echo "📊 Checking analysis artifacts..."
if [ -f "duplication_report.json" ]; then
    REPORT_SIZE=$(wc -c < duplication_report.json)
    if [ $REPORT_SIZE -gt 100 ]; then
        check_pass "Analysis report exists ($(numfmt --to=iec-i --suffix=B $REPORT_SIZE))"
        
        # Parse report for key metrics
        if command -v jq &> /dev/null; then
            TOTAL_FILES=$(jq '.metrics.total_files' duplication_report.json)
            DUPLICATE_FUNCS=$(jq '.duplicate_functions | length' duplication_report.json)
            check_info "Report shows $TOTAL_FILES files analyzed, $DUPLICATE_FUNCS duplicate functions"
        fi
    else
        check_warn "Analysis report is empty or corrupt" "Re-run: python3 scripts/analyze_duplication.py --output duplication_report.json"
    fi
else
    check_warn "Analysis report not found" "Run: python3 scripts/analyze_duplication.py --output duplication_report.json"
fi

# Check 7: Test suite
echo ""
echo "🧪 Checking test suite..."
if [ -d "tests" ]; then
    TEST_COUNT=$(find tests -name "test_*.py" -o -name "*_test.py" | wc -l)
    if [ $TEST_COUNT -gt 0 ]; then
        check_pass "Test suite found ($TEST_COUNT test files)"
        
        # Try running tests (with timeout)
        check_info "Running test suite (this may take a minute)..."
        if timeout 60s pytest tests/ --tb=no -q --maxfail=1 > /tmp/pytest_check.log 2>&1; then
            check_pass "Baseline tests passing"
        else
            EXIT_CODE=$?
            if [ $EXIT_CODE -eq 124 ]; then
                check_warn "Test suite timeout" "Tests took >60s. They may still pass."
            else
                check_warn "Some tests failing" "Fix baseline issues before starting Phase 1"
                check_info "See /tmp/pytest_check.log for details"
            fi
        fi
    else
        check_warn "No test files found in tests/" "Consider adding tests"
    fi
else
    check_warn "tests/ directory not found" "Tests recommended for safe refactoring"
fi

# Check 8: Documentation files
echo ""
echo "📚 Checking consolidation documentation..."
DOCS=(
    "PHASE1_IMPLEMENTATION_GUIDE.md"
    "CONSOLIDATION_ROADMAP.md"
    "CONSOLIDATION_SUMMARY.md"
    "README_CONSOLIDATION.md"
)

for doc in "${DOCS[@]}"; do
    if [ -f "$doc" ]; then
        check_pass "$doc exists"
    else
        check_warn "$doc not found" "Generate with consolidation scripts"
    fi
done

# Check 9: Module structure
echo ""
echo "🏗️  Checking module structure..."
if [ -d "ign_lidar/features" ]; then
    check_pass "ign_lidar/features/ exists"
    
    # Check for key files
    if [ -f "ign_lidar/features/features.py" ]; then
        LINES=$(wc -l < ign_lidar/features/features.py)
        if [ $LINES -gt 2000 ]; then
            check_warn "features.py is large ($LINES lines)" "Target for consolidation"
        else
            check_info "features.py has $LINES lines"
        fi
    fi
    
    # Check if core/ already exists
    if [ -d "ign_lidar/features/core" ]; then
        check_warn "ign_lidar/features/core/ already exists" "Phase 1 may be partially complete"
    else
        check_info "ign_lidar/features/core/ will be created in Phase 1"
    fi
else
    check_fail "ign_lidar/features/ not found" "Are you in the correct directory?"
fi

# Check 10: Disk space
echo ""
echo "💾 Checking system resources..."
if command -v df &> /dev/null; then
    AVAILABLE_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    if [ $AVAILABLE_GB -gt 5 ]; then
        check_pass "Sufficient disk space (${AVAILABLE_GB}GB available)"
    else
        check_warn "Low disk space (${AVAILABLE_GB}GB available)" "Recommend 5GB+ for safe operation"
    fi
fi

# Summary
echo ""
echo "=========================================="
echo "📊 Pre-Flight Check Summary"
echo "=========================================="
echo -e "${GREEN}Passed:${NC}  $CHECKS_PASSED"
echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
echo -e "${RED}Failed:${NC}  $CHECKS_FAILED"
echo ""

if [ $CHECKS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ READY FOR PHASE 1${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Read PHASE1_IMPLEMENTATION_GUIDE.md (5 min)"
    echo "  2. Create feature branch: git checkout -b refactor/phase1-consolidation-$(date +%Y%m%d)"
    echo "  3. Start with Task 1.1: Fix duplicate compute_verticality (2 hours)"
    echo ""
    echo "Quick start command:"
    echo -e "  ${BLUE}git checkout -b refactor/phase1-consolidation-$(date +%Y%m%d) && code PHASE1_IMPLEMENTATION_GUIDE.md${NC}"
    exit 0
else
    echo -e "${RED}⚠️  NOT READY${NC}"
    echo ""
    echo "Please fix the failed checks above before starting Phase 1."
    echo "Re-run this script after making corrections."
    exit 1
fi
