#!/usr/bin/env python3
"""
Performance Bottleneck Analysis Tool
Identifies GPU utilization issues and CPU fallbacks in the pipeline
"""

import sys
import time
import logging
import threading
import subprocess
from pathlib import Path
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def monitor_gpu_utilization(duration=60, interval=1):
    """Monitor GPU utilization over time."""
    gpu_data = []
    start_time = time.time()
    
    print(f"🔍 Monitoring GPU utilization for {duration}s...")
    
    while time.time() - start_time < duration:
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,utilization.memory',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                line = result.stdout.strip()
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    gpu_data.append({
                        'timestamp': time.time(),
                        'name': parts[0],
                        'memory_used_mb': int(parts[1]),
                        'memory_total_mb': int(parts[2]),
                        'gpu_util_pct': int(parts[3]),
                        'memory_util_pct': int(parts[4])
                    })
            
            time.sleep(interval)
        except Exception as e:
            print(f"GPU monitoring error: {e}")
            break
    
    return gpu_data

def analyze_gpu_usage_patterns(gpu_data):
    """Analyze GPU usage patterns to identify issues."""
    if not gpu_data:
        return "No GPU data collected"
    
    # Calculate stats
    gpu_utils = [d['gpu_util_pct'] for d in gpu_data]
    mem_utils = [d['memory_util_pct'] for d in gpu_data]
    mem_used = [d['memory_used_mb'] for d in gpu_data]
    
    avg_gpu_util = np.mean(gpu_utils)
    max_gpu_util = np.max(gpu_utils)
    avg_mem_util = np.mean(mem_utils)
    max_mem_used = np.max(mem_used)
    
    # Count periods of different utilization levels
    high_util_count = sum(1 for u in gpu_utils if u > 80)
    med_util_count = sum(1 for u in gpu_utils if 30 <= u <= 80)
    low_util_count = sum(1 for u in gpu_utils if u < 30)
    
    total_samples = len(gpu_utils)
    
    analysis = f"""
📊 GPU Usage Analysis:
==================
Total monitoring time: {total_samples} seconds
GPU Name: {gpu_data[0]['name']}

GPU Utilization:
- Average: {avg_gpu_util:.1f}%
- Maximum: {max_gpu_util:.1f}%
- High util (>80%): {high_util_count}/{total_samples} samples ({100*high_util_count/total_samples:.1f}%)
- Med util (30-80%): {med_util_count}/{total_samples} samples ({100*med_util_count/total_samples:.1f}%)
- Low util (<30%): {low_util_count}/{total_samples} samples ({100*low_util_count/total_samples:.1f}%)

Memory Usage:
- Average utilization: {avg_mem_util:.1f}%
- Peak memory used: {max_mem_used:.0f} MB ({100*max_mem_used/gpu_data[0]['memory_total_mb']:.1f}% of {gpu_data[0]['memory_total_mb']} MB)

🔍 Performance Issues Detected:
"""

    # Identify issues
    issues = []
    
    if avg_gpu_util < 30:
        issues.append("⚠️  LOW GPU UTILIZATION - Average GPU usage is very low, suggesting CPU fallback or inefficient GPU usage")
    
    if high_util_count < total_samples * 0.2:
        issues.append("⚠️  INSUFFICIENT GPU LOAD - GPU rarely reaches high utilization during processing")
    
    if max_mem_used < gpu_data[0]['memory_total_mb'] * 0.3:
        issues.append("⚠️  LOW MEMORY USAGE - GPU memory is underutilized, batch sizes could be increased")
    
    if low_util_count > total_samples * 0.7:
        issues.append("🚨 CRITICAL: Mostly CPU processing - GPU is barely used, check for fallback conditions")
    
    if not issues:
        issues.append("✅ GPU utilization appears normal")
    
    for issue in issues:
        analysis += f"\n{issue}"
    
    return analysis

def test_gpu_feature_computation():
    """Test GPU feature computation performance."""
    print("\n🧪 Testing GPU Feature Computation...")
    
    try:
        from ign_lidar.features.features_gpu import GPUFeatureComputer
        
        # Create test data
        n_points = 100_000
        points = np.random.randn(n_points, 3).astype(np.float32)
        
        # Test GPU computation
        computer = GPUFeatureComputer(use_gpu=True, batch_size=50_000)
        
        print(f"Testing with {n_points:,} points...")
        print(f"GPU enabled: {computer.use_gpu}")
        print(f"cuML available: {computer.use_cuml}")
        print(f"Batch size: {computer.batch_size:,}")
        
        # Time the computation
        start_time = time.time()
        try:
            normals = computer.compute_normals(points, k=10)
            elapsed = time.time() - start_time
            
            print(f"✅ GPU computation successful: {elapsed:.2f}s for {n_points:,} points")
            print(f"   Throughput: {n_points/elapsed:,.0f} points/second")
            
            return True, elapsed
        except Exception as e:
            print(f"❌ GPU computation failed: {e}")
            return False, None
            
    except Exception as e:
        print(f"❌ GPU feature computer initialization failed: {e}")
        return False, None

def test_ground_truth_processing():
    """Test ground truth processing performance."""
    print("\n🏗️  Testing Ground Truth Processing...")
    
    try:
        from ign_lidar.io.ground_truth_optimizer import GroundTruthOptimizer
        
        # Create test data
        n_points = 50_000
        points = np.random.rand(n_points, 3).astype(np.float32)
        points[:, :2] *= 1000  # Scale to reasonable coordinates
        
        # Test different methods
        methods_to_test = ['auto', 'strtree', 'gpu_chunked', 'gpu']
        results = {}
        
        for method in methods_to_test:
            try:
                if method == 'auto':
                    optimizer = GroundTruthOptimizer(force_method=None, verbose=False)
                else:
                    optimizer = GroundTruthOptimizer(force_method=method, verbose=False)
                
                # Create dummy ground truth features
                import geopandas as gpd
                from shapely.geometry import Polygon
                
                # Create a simple test polygon
                poly = Polygon([(0, 0), (500, 0), (500, 500), (0, 500)])
                gdf = gpd.GeoDataFrame([{'geometry': poly}])
                ground_truth_features = {'buildings': gdf}
                
                start_time = time.time()
                selected_method = optimizer.select_method(len(points), len(gdf))
                
                # Don't actually run the processing, just check method selection
                elapsed = time.time() - start_time
                
                results[method] = {
                    'selected_method': selected_method,
                    'time': elapsed,
                    'success': True
                }
                
                print(f"Method '{method}' -> Selected: '{selected_method}' ({elapsed:.4f}s)")
                
            except Exception as e:
                results[method] = {
                    'selected_method': None,
                    'time': None,
                    'success': False,
                    'error': str(e)
                }
                print(f"Method '{method}' -> ERROR: {e}")
        
        return results
        
    except Exception as e:
        print(f"❌ Ground truth optimizer test failed: {e}")
        return {}

def main():
    """Main performance analysis."""
    print("🔍 IGN LiDAR HD Dataset - Performance Bottleneck Analysis")
    print("=" * 60)
    
    # Test GPU availability first
    print("\n1️⃣  Testing GPU Libraries...")
    try:
        import cupy as cp
        print(f"✅ CuPy available - GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
        
        mem_info = cp.cuda.runtime.memGetInfo()
        print(f"   GPU Memory: {mem_info[1] / 1024**3:.1f} GB total, {mem_info[0] / 1024**3:.1f} GB free")
        
        gpu_available = True
    except Exception as e:
        print(f"❌ CuPy error: {e}")
        gpu_available = False
    
    try:
        from cuml.neighbors import NearestNeighbors
        print("✅ cuML available")
        cuml_available = True
    except Exception as e:
        print(f"❌ cuML error: {e}")
        cuml_available = False
    
    # Test GPU feature computation
    print("\n2️⃣  Testing GPU Feature Computation...")
    gpu_compute_success, gpu_compute_time = test_gpu_feature_computation()
    
    # Test ground truth processing
    print("\n3️⃣  Testing Ground Truth Processing...")
    ground_truth_results = test_ground_truth_processing()
    
    # Monitor GPU utilization during a short test
    print("\n4️⃣  Monitoring GPU Utilization...")
    
    def run_background_computation():
        """Run some GPU computation in background while monitoring."""
        if gpu_available:
            try:
                from ign_lidar.features.features_gpu import GPUFeatureComputer
                computer = GPUFeatureComputer(use_gpu=True)
                
                # Generate some work
                for i in range(3):
                    points = np.random.randn(500_000, 3).astype(np.float32)
                    _ = computer.compute_normals(points, k=10)
                    time.sleep(2)
            except Exception as e:
                print(f"Background computation error: {e}")
    
    # Start background computation
    if gpu_available:
        bg_thread = threading.Thread(target=run_background_computation, daemon=True)
        bg_thread.start()
        time.sleep(1)  # Let it start
    
    # Monitor GPU for 10 seconds
    gpu_data = monitor_gpu_utilization(duration=10, interval=0.5)
    gpu_analysis = analyze_gpu_usage_patterns(gpu_data)
    
    # Generate final report
    print("\n" + "=" * 60)
    print("📋 PERFORMANCE ANALYSIS REPORT")
    print("=" * 60)
    
    print(f"\n🔧 Hardware Status:")
    print(f"- GPU Available: {'✅' if gpu_available else '❌'}")
    print(f"- cuML Available: {'✅' if cuml_available else '❌'}")
    
    print(f"\n⚡ Performance Tests:")
    if gpu_compute_success and gpu_compute_time:
        print(f"- GPU Feature Computation: ✅ ({gpu_compute_time:.2f}s)")
    else:
        print(f"- GPU Feature Computation: ❌")
    
    print(f"\n🏗️  Ground Truth Method Selection:")
    for method, result in ground_truth_results.items():
        if result['success']:
            print(f"- {method}: {result['selected_method']}")
        else:
            print(f"- {method}: ❌ {result.get('error', 'Unknown error')}")
    
    print(gpu_analysis)
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if not gpu_available:
        print("🚨 CRITICAL: GPU libraries not available - all processing falls back to CPU")
        print("   → Install CuPy, cuML, and RAPIDS in the active environment")
    elif not cuml_available:
        print("⚠️  cuML not available - some GPU acceleration disabled")
        print("   → Install RAPIDS cuML for full GPU acceleration")
    else:
        print("✅ GPU libraries are properly installed")
    
    # Check for common performance issues
    if gpu_available and gpu_data:
        avg_gpu_util = np.mean([d['gpu_util_pct'] for d in gpu_data])
        if avg_gpu_util < 30:
            print("⚠️  Low GPU utilization detected:")
            print("   → Check if ground truth processing uses 'strtree' method (CPU only)")
            print("   → Increase batch sizes if GPU memory allows")
            print("   → Verify use_gpu=true in configuration")
    
    print(f"\n🔍 Check the following configuration settings:")
    print(f"   - processing.use_gpu: true")
    print(f"   - features.use_gpu: true") 
    print(f"   - ground_truth.optimization.force_method: 'gpu_chunked' or 'auto'")
    print(f"   - features.gpu_batch_size: increase if low GPU memory usage")

if __name__ == "__main__":
    main()