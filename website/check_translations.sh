#!/bin/bash
# Quick check script to identify actual content differences in "outdated" French translations

echo "=================================================="
echo "CHECKING FOR ACTUAL CONTENT DIFFERENCES"
echo "=================================================="
echo ""

DOCS_EN="/mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET/website/docs"
DOCS_FR="/mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET/website/i18n/fr/docusaurus-plugin-content-docs/current"

# Critical files (1+ day old)
CRITICAL_FILES=(
    "features/architectural-styles.md"
    "guides/features/overview.md"
    "reference/cli-enrich.md"
    "reference/cli-patch.md"
    "release-notes/v1.5.0.md"
)

echo "🔴 CRITICAL FILES (1+ day old)"
echo "================================"
echo ""

for file in "${CRITICAL_FILES[@]}"; do
    echo "📄 Checking: $file"
    
    en_file="$DOCS_EN/$file"
    fr_file="$DOCS_FR/$file"
    
    if [ ! -f "$en_file" ]; then
        echo "   ❌ English file not found"
        continue
    fi
    
    if [ ! -f "$fr_file" ]; then
        echo "   ❌ French file not found"
        continue
    fi
    
    # Get file sizes
    en_size=$(stat -c%s "$en_file" 2>/dev/null || stat -f%z "$en_file")
    fr_size=$(stat -c%s "$fr_file" 2>/dev/null || stat -f%z "$fr_file")
    
    # Get modification times
    en_time=$(stat -c%y "$en_file" 2>/dev/null || stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$en_file")
    fr_time=$(stat -c%y "$fr_file" 2>/dev/null || stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$fr_file")
    
    # Get line counts
    en_lines=$(wc -l < "$en_file")
    fr_lines=$(wc -l < "$fr_file")
    
    # Calculate differences
    size_diff=$((en_size - fr_size))
    line_diff=$((en_lines - fr_lines))
    
    echo "   📊 Statistics:"
    echo "      EN: $en_lines lines, $en_size bytes, modified: $en_time"
    echo "      FR: $fr_lines lines, $fr_size bytes, modified: $fr_time"
    echo "      Diff: $line_diff lines, $size_diff bytes"
    
    # Check if sizes are very similar (within 10%)
    size_diff_abs=${size_diff#-}
    percentage=$((100 * size_diff_abs / en_size))
    
    if [ $percentage -lt 10 ]; then
        echo "   ✅ Sizes are similar (${percentage}% difference) - likely synchronized"
    else
        echo "   ⚠️  Significant size difference (${percentage}% difference)"
    fi
    
    # Check first 10 lines for frontmatter changes
    echo "   🔍 Checking frontmatter..."
    en_title=$(head -10 "$en_file" | grep "^title:" | cut -d':' -f2-)
    fr_title=$(head -10 "$fr_file" | grep "^title:" | cut -d':' -f2-)
    
    if [ -n "$en_title" ] && [ -n "$fr_title" ]; then
        echo "      EN title:$en_title"
        echo "      FR title:$fr_title"
    fi
    
    echo ""
done

echo ""
echo "=================================================="
echo "RECOMMENDATIONS"
echo "=================================================="
echo ""
echo "Based on the analysis:"
echo ""
echo "1. Files with <10% size difference are likely properly translated"
echo "2. Timestamp differences may be due to:"
echo "   - Build/formatting processes"
echo "   - Git operations"
echo "   - Minor whitespace changes"
echo ""
echo "3. Recommended actions:"
echo "   ✅ Spot-check 2-3 files manually for content alignment"
echo "   ✅ Focus on files with >10% size difference"
echo "   ✅ Set up automated monitoring for future changes"
echo ""

# Check for French-only files
echo "=================================================="
echo "FRENCH-ONLY FILES"
echo "=================================================="
echo ""

FR_ONLY=(
    "examples/index.md"
    "guides/visualization.md"
)

for file in "${FR_ONLY[@]}"; do
    fr_file="$DOCS_FR/$file"
    en_file="$DOCS_EN/$file"
    
    if [ -f "$fr_file" ] && [ ! -f "$en_file" ]; then
        fr_size=$(stat -c%s "$fr_file" 2>/dev/null || stat -f%z "$fr_file")
        fr_lines=$(wc -l < "$fr_file")
        
        echo "📝 $file"
        echo "   French only: $fr_lines lines, $fr_size bytes"
        echo "   ➡️  Recommend creating English version"
        echo ""
    fi
done

echo ""
echo "✅ Analysis complete!"
