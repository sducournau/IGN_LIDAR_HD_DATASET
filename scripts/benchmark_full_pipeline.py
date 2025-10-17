#!/usr/bin/env python3
"""
Full Pipeline Benchmark Script

Benchmarks the complete IGN LiDAR HD processing pipeline to measure:
- Feature computation performance
- Ground truth classification performance
- GPU utilization
- Memory usage
- Overall processing time

Usage:
    # Benchmark single tile
    python scripts/benchmark_full_pipeline.py --tile path/to/tile.laz

    # Benchmark with config
    python scripts/benchmark_full_pipeline.py --config config.yaml --input-dir tiles/

    # Compare old vs new ground truth
    python scripts/benchmark_full_pipeline.py --tile tile.laz --compare-ground-truth
"""

import argparse
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor and record performance metrics."""
    
    def __init__(self):
        self.metrics = {
            'start_time': None,
            'end_time': None,
            'phases': {},
            'gpu_metrics': [],
            'memory_usage': []
        }
        self._phase_start = {}
    
    def start_phase(self, phase_name: str):
        """Start timing a phase."""
        self._phase_start[phase_name] = time.time()
        logger.info(f"▶️  Starting phase: {phase_name}")
    
    def end_phase(self, phase_name: str, metadata: Optional[Dict] = None):
        """End timing a phase and record results."""
        if phase_name not in self._phase_start:
            logger.warning(f"Phase {phase_name} was not started!")
            return
        
        elapsed = time.time() - self._phase_start[phase_name]
        self.metrics['phases'][phase_name] = {
            'duration_sec': elapsed,
            'metadata': metadata or {}
        }
        
        logger.info(f"⏱️  Phase {phase_name} completed in {elapsed:.2f}s")
        del self._phase_start[phase_name]
    
    def record_gpu_stats(self):
        """Record current GPU stats (requires pynvml)."""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            gpu_stats = []
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Get memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                gpu_stats.append({
                    'gpu_id': i,
                    'name': pynvml.nvmlDeviceGetName(handle).decode('utf-8'),
                    'utilization_percent': utilization.gpu,
                    'memory_used_mb': mem_info.used / 1024**2,
                    'memory_total_mb': mem_info.total / 1024**2,
                    'memory_percent': 100 * mem_info.used / mem_info.total,
                    'timestamp': time.time()
                })
            
            self.metrics['gpu_metrics'].append(gpu_stats)
            pynvml.nvmlShutdown()
            
        except ImportError:
            logger.debug("pynvml not available, skipping GPU monitoring")
        except Exception as e:
            logger.debug(f"GPU monitoring error: {e}")
    
    def get_summary(self) -> Dict:
        """Get performance summary."""
        total_time = sum(
            phase['duration_sec'] 
            for phase in self.metrics['phases'].values()
        )
        
        summary = {
            'total_duration_sec': total_time,
            'phases': self.metrics['phases'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Add GPU summary if available
        if self.metrics['gpu_metrics']:
            gpu_utilizations = []
            for snapshot in self.metrics['gpu_metrics']:
                for gpu in snapshot:
                    gpu_utilizations.append(gpu['utilization_percent'])
            
            summary['gpu_avg_utilization'] = sum(gpu_utilizations) / len(gpu_utilizations)
            summary['gpu_max_utilization'] = max(gpu_utilizations)
        
        return summary


def benchmark_tile(
    tile_path: Path,
    config_path: Optional[Path] = None,
    compare_ground_truth: bool = False
) -> Dict:
    """Benchmark processing of a single tile."""
    logger.info(f"\n{'='*80}")
    logger.info(f"BENCHMARKING TILE: {tile_path.name}")
    logger.info(f"{'='*80}\n")
    
    monitor = PerformanceMonitor()
    
    try:
        # Import here to avoid import errors if packages not installed
        import laspy
        import numpy as np
        from ign_lidar.io.laz_io import LAZReader
        from ign_lidar.features.orchestrator import FeatureOrchestrator
        from ign_lidar.io.ground_truth_optimizer import GroundTruthOptimizer
        
        # Phase 1: Load tile
        monitor.start_phase('load_tile')
        reader = LAZReader(str(tile_path))
        points, colors, classification, intensity = reader.read_laz()
        num_points = len(points)
        monitor.end_phase('load_tile', {'num_points': num_points})
        
        # Phase 2: Compute features
        monitor.start_phase('compute_features')
        orchestrator = FeatureOrchestrator(
            feature_mode='full',
            use_gpu=True,
            k_neighbors=20
        )
        features = orchestrator.compute_all_features(
            points=points,
            colors=colors,
            classification=classification,
            intensity=intensity
        )
        monitor.end_phase('compute_features', {
            'num_features': len(features),
            'feature_names': list(features.keys())
        })
        
        # Record GPU stats after feature computation
        monitor.record_gpu_stats()
        
        # Phase 3: Ground truth classification (if requested)
        if compare_ground_truth:
            # Create mock ground truth for testing
            from shapely.geometry import Polygon
            ground_truth = {
                'buildings': [Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])]
            }
            
            # Test old method (if available)
            try:
                from ign_lidar.core.modules.advanced_classification import AdvancedClassifier
                
                monitor.start_phase('ground_truth_old')
                classifier = AdvancedClassifier(use_ground_truth=True)
                labels_old = classifier._classify_by_ground_truth(
                    labels=np.zeros(num_points, dtype=np.int32),
                    points=points,
                    ground_truth_features=ground_truth
                )
                monitor.end_phase('ground_truth_old', {
                    'method': 'AdvancedClassifier',
                    'classified_points': int(np.sum(labels_old > 0))
                })
            except Exception as e:
                logger.warning(f"AdvancedClassifier benchmark skipped: {e}")
            
            # Test new method
            monitor.start_phase('ground_truth_new')
            optimizer = GroundTruthOptimizer(force_method='auto', verbose=True)
            labels_new = optimizer.label_points(
                points=points,
                ground_truth_features=ground_truth
            )
            monitor.end_phase('ground_truth_new', {
                'method': 'GroundTruthOptimizer',
                'classified_points': int(np.sum(labels_new > 0))
            })
            
            # Record GPU stats after classification
            monitor.record_gpu_stats()
        
        # Get summary
        summary = monitor.get_summary()
        
        # Print results
        print_benchmark_results(summary)
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}", exc_info=True)
        raise


def print_benchmark_results(summary: Dict):
    """Print formatted benchmark results."""
    logger.info(f"\n{'='*80}")
    logger.info("BENCHMARK RESULTS")
    logger.info(f"{'='*80}")
    
    # Total time
    logger.info(f"\n⏱️  Total Duration: {summary['total_duration_sec']:.2f}s")
    
    # Phase breakdown
    logger.info(f"\n📊 Phase Breakdown:")
    for phase_name, phase_data in summary['phases'].items():
        duration = phase_data['duration_sec']
        percent = 100 * duration / summary['total_duration_sec']
        logger.info(f"  {phase_name:.<30} {duration:>8.2f}s ({percent:>5.1f}%)")
        
        # Print metadata if available
        if phase_data.get('metadata'):
            for key, value in phase_data['metadata'].items():
                logger.info(f"    • {key}: {value}")
    
    # GPU stats
    if 'gpu_avg_utilization' in summary:
        logger.info(f"\n🎮 GPU Utilization:")
        logger.info(f"  Average: {summary['gpu_avg_utilization']:.1f}%")
        logger.info(f"  Maximum: {summary['gpu_max_utilization']:.1f}%")
    
    # Performance comparison (if ground truth was compared)
    if 'ground_truth_old' in summary['phases'] and 'ground_truth_new' in summary['phases']:
        old_time = summary['phases']['ground_truth_old']['duration_sec']
        new_time = summary['phases']['ground_truth_new']['duration_sec']
        speedup = old_time / new_time if new_time > 0 else 0
        
        logger.info(f"\n🚀 Ground Truth Performance:")
        logger.info(f"  Old method: {old_time:.2f}s")
        logger.info(f"  New method: {new_time:.2f}s")
        logger.info(f"  Speedup:    {speedup:.1f}× faster")
    
    logger.info(f"\n{'='*80}\n")


def save_benchmark_report(summary: Dict, output_path: Path):
    """Save benchmark results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"📝 Benchmark report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark IGN LiDAR HD processing pipeline"
    )
    parser.add_argument(
        "--tile",
        type=str,
        help="Path to single tile to benchmark"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing tiles (will benchmark first tile)"
    )
    parser.add_argument(
        "--compare-ground-truth",
        action="store_true",
        help="Compare old vs new ground truth classification"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_report.json",
        help="Output path for benchmark report (default: benchmark_report.json)"
    )
    
    args = parser.parse_args()
    
    # Determine tile to benchmark
    tile_path = None
    if args.tile:
        tile_path = Path(args.tile)
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        tiles = list(input_dir.glob("*.laz")) + list(input_dir.glob("*.las"))
        if tiles:
            tile_path = tiles[0]
            logger.info(f"Benchmarking first tile: {tile_path.name}")
        else:
            logger.error(f"❌ No tiles found in {input_dir}")
            exit(1)
    else:
        parser.print_help()
        logger.error("\n❌ Error: Please specify --tile or --input-dir")
        exit(1)
    
    if not tile_path.exists():
        logger.error(f"❌ Tile not found: {tile_path}")
        exit(1)
    
    # Run benchmark
    try:
        summary = benchmark_tile(
            tile_path=tile_path,
            config_path=Path(args.config) if args.config else None,
            compare_ground_truth=args.compare_ground_truth
        )
        
        # Save report
        output_path = Path(args.output)
        save_benchmark_report(summary, output_path)
        
        logger.info("✅ Benchmark completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
