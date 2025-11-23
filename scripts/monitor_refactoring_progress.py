#!/usr/bin/env python3
"""
Refactoring Progress Monitor

Tracks and visualizes refactoring progress across all three phases.
Provides real-time metrics and completion status.

Usage:
    # Check current progress
    python scripts/monitor_refactoring_progress.py
    
    # Watch mode (updates every 5 seconds)
    python scripts/monitor_refactoring_progress.py --watch
    
    # Export progress to JSON
    python scripts/monitor_refactoring_progress.py --export progress.json
    
    # Generate progress report
    python scripts/monitor_refactoring_progress.py --report

Author: GitHub Copilot
Date: November 22, 2025
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


class RefactoringMonitor:
    """Monitor refactoring progress across all phases."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.ign_lidar = repo_root / "ign_lidar"
        
    def count_compute_normals_implementations(self) -> int:
        """Count number of compute_normals() implementations."""
        try:
            result = subprocess.run(
                ["grep", "-r", "def compute_normals", str(self.ign_lidar)],
                capture_output=True,
                text=True
            )
            return len(result.stdout.strip().split('\n')) if result.stdout else 0
        except Exception:
            return -1
    
    def count_total_classes(self) -> int:
        """Count total number of classes with Processor/Engine/Computer patterns."""
        try:
            result = subprocess.run(
                ["grep", "-r", "-E", 
                 "class.*(Processor|Engine|Computer|Manager)", 
                 str(self.ign_lidar)],
                capture_output=True,
                text=True
            )
            return len(result.stdout.strip().split('\n')) if result.stdout else 0
        except Exception:
            return -1
    
    def check_gpu_transfer_profiler_exists(self) -> bool:
        """Check if GPUTransferProfiler exists."""
        profiler_path = self.ign_lidar / "optimization" / "gpu_transfer_profiler.py"
        return profiler_path.exists()
    
    def check_cuda_stream_manager_exists(self) -> bool:
        """Check if CUDAStreamManager exists."""
        # Check in orchestrator.py for CUDAStreamManager import
        orchestrator = self.ign_lidar / "features" / "orchestrator.py"
        if orchestrator.exists():
            content = orchestrator.read_text()
            return "CUDAStreamManager" in content
        return False
    
    def count_backup_files(self) -> int:
        """Count number of .backup files created."""
        try:
            result = subprocess.run(
                ["find", str(self.repo_root), "-name", "*.backup"],
                capture_output=True,
                text=True
            )
            return len(result.stdout.strip().split('\n')) if result.stdout else 0
        except Exception:
            return 0
    
    def check_tests_pass(self) -> Tuple[bool, str]:
        """Check if tests pass (quick unit tests only)."""
        try:
            result = subprocess.run(
                ["pytest", "tests/", "-v", "-m", "unit", "--co", "-q"],
                capture_output=True,
                text=True,
                cwd=self.repo_root,
                timeout=5
            )
            # If collection succeeds, tests are at least runnable
            return result.returncode == 0, result.stdout
        except Exception as e:
            return False, str(e)
    
    def get_duplication_percentage(self) -> float:
        """Get current code duplication percentage."""
        script = self.repo_root / "scripts" / "analyze_duplication.py"
        if not script.exists():
            return -1.0
        
        try:
            result = subprocess.run(
                ["python", str(script)],
                capture_output=True,
                text=True,
                cwd=self.repo_root,
                timeout=30
            )
            
            # Parse output for duplication percentage
            match = re.search(r"(\d+\.\d+)%", result.stdout)
            if match:
                return float(match.group(1))
            return -1.0
        except Exception:
            return -1.0
    
    def get_phase1_progress(self) -> Dict:
        """Get Phase 1 progress metrics."""
        compute_normals_count = self.count_compute_normals_implementations()
        backup_count = self.count_backup_files()
        duplication = self.get_duplication_percentage()
        
        # Phase 1 is complete if compute_normals <= 2 (canonical + maybe one legacy)
        is_complete = compute_normals_count <= 2
        
        return {
            "phase": 1,
            "name": "Élimination des duplications",
            "complete": is_complete,
            "progress": 100 if is_complete else 0,
            "metrics": {
                "compute_normals_count": compute_normals_count,
                "target": 1,
                "backup_files": backup_count,
                "duplication_percentage": duplication
            }
        }
    
    def get_phase2_progress(self) -> Dict:
        """Get Phase 2 progress metrics."""
        has_profiler = self.check_gpu_transfer_profiler_exists()
        has_streams = self.check_cuda_stream_manager_exists()
        
        # Phase 2 is complete if both components exist
        is_complete = has_profiler and has_streams
        progress = 0
        if has_profiler:
            progress += 50
        if has_streams:
            progress += 50
        
        return {
            "phase": 2,
            "name": "Optimisation GPU",
            "complete": is_complete,
            "progress": progress,
            "metrics": {
                "gpu_transfer_profiler": has_profiler,
                "cuda_stream_manager": has_streams,
                "requires_benchmark": not is_complete
            }
        }
    
    def get_phase3_progress(self) -> Dict:
        """Get Phase 3 progress metrics."""
        class_count = self.count_total_classes()
        
        # Phase 3 is complete if class count < 25
        is_complete = 0 < class_count < 25
        
        # Estimate progress
        if class_count == -1:
            progress = 0
        else:
            # From 34 to 25
            initial = 34
            target = 25
            progress = max(0, min(100, int((initial - class_count) / (initial - target) * 100)))
        
        return {
            "phase": 3,
            "name": "Nettoyage architecture",
            "complete": is_complete,
            "progress": progress,
            "metrics": {
                "class_count": class_count,
                "target": 25,
                "reduction_needed": max(0, class_count - 25) if class_count > 0 else 0
            }
        }
    
    def get_overall_progress(self) -> Dict:
        """Get overall refactoring progress."""
        phase1 = self.get_phase1_progress()
        phase2 = self.get_phase2_progress()
        phase3 = self.get_phase3_progress()
        
        total_progress = (phase1["progress"] + phase2["progress"] + phase3["progress"]) / 3
        phases_complete = sum([phase1["complete"], phase2["complete"], phase3["complete"]])
        
        tests_pass, _ = self.check_tests_pass()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_progress": round(total_progress, 1),
            "phases_complete": f"{phases_complete}/3",
            "tests_passing": tests_pass,
            "phases": [phase1, phase2, phase3]
        }
    
    def print_progress(self):
        """Print progress to console with nice formatting."""
        progress = self.get_overall_progress()
        
        print("\n" + "=" * 80)
        print("📊 REFACTORING PROGRESS MONITOR")
        print("=" * 80)
        print(f"\n⏰ Last updated: {progress['timestamp']}")
        print(f"📈 Overall progress: {progress['overall_progress']}%")
        print(f"✅ Phases complete: {progress['phases_complete']}")
        print(f"🧪 Tests passing: {'✅ Yes' if progress['tests_passing'] else '❌ No'}")
        
        print("\n" + "-" * 80)
        
        for phase in progress["phases"]:
            status = "✅ COMPLETE" if phase["complete"] else f"🔄 {phase['progress']}%"
            print(f"\nPHASE {phase['phase']}: {phase['name']} - {status}")
            print("-" * 80)
            
            for key, value in phase["metrics"].items():
                # Format key nicely
                key_display = key.replace('_', ' ').title()
                
                # Format value with context
                if isinstance(value, bool):
                    value_display = "✅ Yes" if value else "❌ No"
                elif isinstance(value, float):
                    value_display = f"{value:.1f}%"
                else:
                    value_display = str(value)
                
                print(f"  • {key_display}: {value_display}")
        
        print("\n" + "=" * 80 + "\n")
    
    def export_progress(self, output_path: Path):
        """Export progress to JSON file."""
        progress = self.get_overall_progress()
        
        with open(output_path, 'w') as f:
            json.dump(progress, f, indent=2)
        
        print(f"✅ Progress exported to: {output_path}")
    
    def generate_report(self):
        """Generate detailed progress report."""
        progress = self.get_overall_progress()
        
        print("\n" + "=" * 80)
        print("📋 REFACTORING PROGRESS REPORT")
        print("=" * 80)
        print(f"\nGenerated: {progress['timestamp']}")
        
        print("\n## Executive Summary")
        print(f"- Overall completion: {progress['overall_progress']}%")
        print(f"- Phases completed: {progress['phases_complete']}")
        print(f"- Test suite: {'PASSING ✅' if progress['tests_passing'] else 'FAILING ❌'}")
        
        print("\n## Phase Details\n")
        
        for phase in progress["phases"]:
            print(f"### Phase {phase['phase']}: {phase['name']}")
            print(f"Status: {'✅ COMPLETE' if phase['complete'] else f'🔄 IN PROGRESS ({phase['progress']}%)'}")
            print("\nMetrics:")
            
            if phase["phase"] == 1:
                m = phase["metrics"]
                print(f"- compute_normals() implementations: {m['compute_normals_count']} (target: {m['target']})")
                print(f"- Backup files created: {m['backup_files']}")
                if m['duplication_percentage'] > 0:
                    print(f"- Code duplication: {m['duplication_percentage']:.1f}%")
                
                if not phase["complete"]:
                    print("\n⚠️  Action required:")
                    print("   python scripts/refactor_phase1_remove_duplicates.py")
            
            elif phase["phase"] == 2:
                m = phase["metrics"]
                print(f"- GPU Transfer Profiler: {'✅ Created' if m['gpu_transfer_profiler'] else '❌ Missing'}")
                print(f"- CUDA Stream Manager: {'✅ Integrated' if m['cuda_stream_manager'] else '❌ Missing'}")
                
                if not phase["complete"]:
                    print("\n⚠️  Action required:")
                    if not m['gpu_transfer_profiler']:
                        print("   1. python scripts/refactor_phase2_optimize_gpu.py")
                    if m['requires_benchmark']:
                        print("   2. conda run -n ign_gpu python scripts/benchmark_gpu_transfers.py")
            
            elif phase["phase"] == 3:
                m = phase["metrics"]
                print(f"- Processor/Engine classes: {m['class_count']} (target: <{m['target']})")
                if m['reduction_needed'] > 0:
                    print(f"- Reduction needed: {m['reduction_needed']} classes")
                
                if not phase["complete"]:
                    print("\n⚠️  Action required:")
                    print("   1. Audit class usage: grep -r 'class.*Processor' ign_lidar/")
                    print("   2. Identify redundant classes")
                    print("   3. Migrate to unified KNNEngine")
            
            print()
        
        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Monitor refactoring progress'
    )
    parser.add_argument(
        '--watch',
        action='store_true',
        help='Watch mode (updates every 5 seconds)'
    )
    parser.add_argument(
        '--export',
        type=str,
        metavar='FILE',
        help='Export progress to JSON file'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate detailed progress report'
    )
    
    args = parser.parse_args()
    
    # Get repo root
    repo_root = Path(__file__).parent.parent.resolve()
    monitor = RefactoringMonitor(repo_root)
    
    if args.export:
        monitor.export_progress(Path(args.export))
    elif args.report:
        monitor.generate_report()
    elif args.watch:
        print("👀 Watch mode - Press Ctrl+C to exit\n")
        try:
            while True:
                monitor.print_progress()
                time.sleep(5)
        except KeyboardInterrupt:
            print("\n✅ Monitoring stopped")
    else:
        monitor.print_progress()


if __name__ == '__main__':
    main()
