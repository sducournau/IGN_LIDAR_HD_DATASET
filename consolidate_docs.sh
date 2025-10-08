#!/bin/bash
# Documentation Consolidation Script
# Safely archives old documentation files

set -e

echo "🗂️  IGN LIDAR HD - Documentation Consolidation"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Base directory
BASE_DIR="/mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET"
cd "$BASE_DIR"

echo "📋 Phase 1: Archiving Session Summaries"
echo "----------------------------------------"

# Archive session summaries (keep only SESSION_SUMMARY_OCT7.md)
SESSION_FILES=(
    "SESSION_COMPLETION_OCT7.md"
    "SESSION_CONTINUATION_OCT7.md"
    "DAILY_SUMMARY_OCT7_2025.md"
    "PHASE_1_3_COMPLETE.md"
    "SPRINT_2_INTEGRATION_COMPLETE.md"
)

for file in "${SESSION_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  Moving: $file"
        mv "$file" "docs/archive/sessions/"
    else
        echo "  ${YELLOW}Warning: $file not found${NC}"
    fi
done

echo ""
echo "📋 Phase 2: Archiving Audit Documents"
echo "---------------------------------------"

# Archive audit documents
AUDIT_FILES=(
    "AUDIT_SUMMARY.md"
    "AUDIT_RECOMMENDATIONS.md"
    "AUDIT_COMPLETE.txt"
    "AUDIT_ENHANCEMENTS_SUMMARY.txt"
    "COMPARISON_V1_V2.md"
)

for file in "${AUDIT_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  Moving: $file"
        mv "$file" "docs/archive/audit-oct7/"
    else
        echo "  ${YELLOW}Warning: $file not found${NC}"
    fi
done

echo ""
echo "📋 Phase 3: Archiving Sprint-Specific Documents"
echo "------------------------------------------------"

# Archive sprint-specific docs
SPRINT_FILES=(
    "INDEX_SPRINT2.md"
    "SPRINT_2_PROGRESS.md"
    "SPRINT2_TEST_INFRASTRUCTURE.md"
)

for file in "${SPRINT_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  Moving: $file"
        mv "$file" "docs/archive/sprints/"
    else
        echo "  ${YELLOW}Warning: $file not found${NC}"
    fi
done

echo ""
echo "📋 Phase 4: Deleting Obsolete Files"
echo "------------------------------------"

# Delete obsolete files
OBSOLETE_FILES=(
    "AUDIT_TREE.txt"
    "PROGRESS_SUMMARY.txt"
)

for file in "${OBSOLETE_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ${RED}Deleting: $file${NC}"
        rm "$file"
    else
        echo "  ${YELLOW}Warning: $file not found${NC}"
    fi
done

echo ""
echo "✅ Consolidation Complete!"
echo ""
echo "📊 Summary:"
echo "  - Session files archived: ${#SESSION_FILES[@]}"
echo "  - Audit files archived: ${#AUDIT_FILES[@]}"
echo "  - Sprint files archived: ${#SPRINT_FILES[@]}"
echo "  - Obsolete files deleted: ${#OBSOLETE_FILES[@]}"
echo ""
echo "📁 Archive location: docs/archive/"
echo ""
echo "Next steps:"
echo "  1. Review archived files in docs/archive/"
echo "  2. Run: git status"
echo "  3. Commit changes if satisfied"
echo ""
