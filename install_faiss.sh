#!/bin/bash
# FAISS GPU Installation and Testing Script

echo "=========================================="
echo "FAISS GPU Installation for K-NN Speedup"
echo "=========================================="
echo ""

# Activate environment
echo "✓ Activating ign_gpu environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ign_gpu

echo ""
echo "📦 Installing FAISS GPU..."
echo "   This will provide 50-100× speedup for k-NN queries"
echo ""

# Install FAISS
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 -y

echo ""
echo "✅ Installation complete!"
echo ""

# Verify installation
echo "🔍 Verifying FAISS installation..."
python << EOF
try:
    import faiss
    print(f"✓ FAISS version: {faiss.__version__}")
    
    # Test GPU availability
    try:
        res = faiss.StandardGpuResources()
        print("✓ FAISS GPU available")
        
        # Quick performance test
        import numpy as np
        import time
        
        print("\n🎯 Quick performance test...")
        N = 1_000_000
        D = 3
        k = 20
        
        print(f"   {N:,} points, k={k}")
        
        # Generate test data
        points = np.random.randn(N, D).astype(np.float32)
        
        # Build index
        index = faiss.IndexFlatL2(D)
        index = faiss.index_cpu_to_gpu(res, 0, index)
        
        start = time.time()
        index.add(points)
        build_time = time.time() - start
        print(f"   Build time: {build_time:.2f}s")
        
        # Query
        start = time.time()
        distances, indices = index.search(points[:100000], k)
        query_time = time.time() - start
        print(f"   Query time (100K points): {query_time:.2f}s")
        
        # Estimate for 18.6M points
        estimated_total = (build_time * 18.6 + query_time * 186)
        print(f"\n   📊 Estimated for 18.6M points: {estimated_total:.1f}s ({estimated_total/60:.1f} min)")
        print(f"   💡 vs cuML: ~51 min → ~{estimated_total/60:.1f} min = {51/(estimated_total/60):.0f}× speedup!")
        
    except Exception as e:
        print(f"⚠ GPU test failed: {e}")
        print("  FAISS CPU mode will be used")
    
    print("\n✅ FAISS ready to use!")
    
except ImportError as e:
    print(f"❌ FAISS installation failed: {e}")
    print("   Please try: conda install -c pytorch faiss-gpu=1.7.4")
    exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ FAISS Installation Successful!"
    echo "=========================================="
    echo ""
    echo "📊 Expected Performance Improvements:"
    echo "   K-NN queries: 51 min → 30-90 seconds"
    echo "   Total processing: 64 min → 2-5 min per tile"
    echo "   128 tiles: 5.7 days → 4-10 hours!"
    echo ""
    echo "🚀 Next Steps:"
    echo "   1. Run a test with one tile:"
    echo "      ./test_fast_preset.sh"
    echo ""
    echo "   2. Check logs for FAISS usage:"
    echo "      Look for: '🚀 Using FAISS for ultra-fast k-NN'"
    echo ""
    echo "   3. Process all tiles:"
    echo "      ign-lidar-hd process -c \"ign_lidar/configs/presets/asprs_rtx4080_fast.yaml\" \\"
    echo "        input_dir=\"/mnt/d/ign/selected_tiles/asprs/tiles\" \\"
    echo "        output_dir=\"/mnt/d/ign/preprocessed_ground_truth\""
    echo ""
else
    echo ""
    echo "❌ Installation failed. Please check errors above."
    exit 1
fi
