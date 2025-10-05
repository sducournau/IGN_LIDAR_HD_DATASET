#!/bin/bash

EN_DIR="docs"
FR_DIR="i18n/fr/docusaurus-plugin-content-docs/current"

echo "=== Missing French Translations ==="
echo ""

# Find all English markdown files
find "$EN_DIR" -name "*.md" | while read en_file; do
    # Get relative path
    rel_path="${en_file#$EN_DIR/}"
    fr_file="$FR_DIR/$rel_path"
    
    if [ ! -f "$fr_file" ]; then
        echo "MISSING: $rel_path"
    fi
done

echo ""
echo "=== Summary ==="
EN_COUNT=$(find "$EN_DIR" -name "*.md" | wc -l)
FR_COUNT=$(find "$FR_DIR" -name "*.md" | wc -l)
echo "English files: $EN_COUNT"
echo "French files: $FR_COUNT"
echo "Missing: $((EN_COUNT - FR_COUNT))"
