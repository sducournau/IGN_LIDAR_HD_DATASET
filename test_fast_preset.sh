#!/bin/bash
# Quick test script for fast preset with k=10

echo "🚀 Testing Fast Preset (k=10) with Single Tile"
echo "================================================"

# Configuration
CONFIG="ign_lidar/configs/presets/asprs_rtx4080_fast.yaml"
INPUT_DIR="/mnt/d/ign/selected_tiles/asprs/tiles"
OUTPUT_DIR="/mnt/d/ign/test_fast_k10"

# Clean previous test
echo "🧹 Cleaning previous test output..."
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Run processing
echo "⚡ Starting processing with fast preset..."
echo "   Config: $CONFIG"
echo "   k_neighbors: 10 (50% faster than k=20)"
echo ""

ign-lidar-hd process \
  -c "$CONFIG" \
  input_dir="$INPUT_DIR" \
  output_dir="$OUTPUT_DIR" \
  processor.num_workers=1

# Check results
echo ""
echo "✅ Processing Complete!"
echo "================================================"
echo ""
echo "📊 Results Summary:"
echo ""

# Count enriched LAZ files
enriched_count=$(find "$OUTPUT_DIR" -name "*_enriched.laz" | wc -l)
echo "   Enriched LAZ tiles: $enriched_count"

# Count patch files
patch_count=$(find "$OUTPUT_DIR" -name "*.laz" -not -name "*_enriched.laz" | wc -l)
echo "   Patch files: $patch_count"

# Show sample files
echo ""
echo "📁 Sample Files:"
echo ""
ls -lh "$OUTPUT_DIR"/*_enriched.laz 2>/dev/null | head -3
echo ""
ls -lh "$OUTPUT_DIR"/*.laz 2>/dev/null | grep -v "_enriched" | head -3

echo ""
echo "✅ Test Complete!"
echo ""
echo "💡 Next Steps:"
echo "   1. Check timing in logs above"
echo "   2. Verify enriched LAZ files created"
echo "   3. Run full processing if satisfied"
echo ""
echo "📝 Full Processing Command:"
echo "   ign-lidar-hd process -c \"$CONFIG\" \\"
echo "     input_dir=\"/mnt/d/ign/selected_tiles/asprs/tiles\" \\"
echo "     output_dir=\"/mnt/d/ign/preprocessed_ground_truth\""
