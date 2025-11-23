#!/usr/bin/env python3
"""
Refactoring Impact Analysis

Analyzes the impact of refactoring changes by comparing before/after metrics:
- Code duplication changes
- Performance improvements
- Test coverage
- Code complexity

Usage:
    # Analyze current state
    python scripts/analyze_refactoring_impact.py
    
    # Compare with baseline
    python scripts/analyze_refactoring_impact.py --baseline baseline.json
    
    # Generate full report
    python scripts/analyze_refactoring_impact.py --report

Author: GitHub Copilot
Date: November 22, 2025
"""

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


class RefactoringImpactAnalyzer:
    """Analyze impact of refactoring changes."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.ign_lidar = repo_root / "ign_lidar"
        
    def count_files_and_lines(self) -> Dict:
        """Count Python files and lines of code."""
        py_files = list(self.ign_lidar.rglob("*.py"))
        py_files = [f for f in py_files if "__pycache__" not in str(f)]
        
        total_lines = 0
        code_lines = 0
        comment_lines = 0
        blank_lines = 0
        
        for py_file in py_files:
            try:
                content = py_file.read_text()
                lines = content.split('\n')
                
                total_lines += len(lines)
                
                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        blank_lines += 1
                    elif stripped.startswith('#'):
                        comment_lines += 1
                    else:
                        code_lines += 1
            except Exception as e:
                # Skip files with read errors
                pass
        
        return {
            "total_files": len(py_files),
            "total_lines": total_lines,
            "code_lines": code_lines,
            "comment_lines": comment_lines,
            "blank_lines": blank_lines,
            "avg_lines_per_file": total_lines / len(py_files) if py_files else 0
        }
    
    def analyze_duplication(self) -> Dict:
        """Analyze code duplication."""
        # Count specific duplicate patterns
        compute_normals_count = self._count_pattern(r"def compute_normals")
        validate_normals_count = self._count_pattern(r"def validate_normals")
        knn_implementations = self._count_pattern(r"def.*knn|def.*nearest_neighbor")
        
        return {
            "compute_normals_implementations": compute_normals_count,
            "validate_normals_implementations": validate_normals_count,
            "knn_implementations": knn_implementations,
        }
    
    def analyze_classes(self) -> Dict:
        """Analyze class structure."""
        processor_classes = self._count_pattern(r"class\s+\w*Processor")
        engine_classes = self._count_pattern(r"class\s+\w*Engine")
        computer_classes = self._count_pattern(r"class\s+\w*Computer")
        manager_classes = self._count_pattern(r"class\s+\w*Manager")
        
        total = processor_classes + engine_classes + computer_classes + manager_classes
        
        return {
            "processor_classes": processor_classes,
            "engine_classes": engine_classes,
            "computer_classes": computer_classes,
            "manager_classes": manager_classes,
            "total_related_classes": total
        }
    
    def check_phase_completion(self) -> Dict:
        """Check which phases are complete."""
        phase1_complete = self._count_pattern(r"def compute_normals") <= 2
        phase2_complete = (self.ign_lidar / "optimization" / "gpu_transfer_profiler.py").exists()
        phase3_complete = self.analyze_classes()["total_related_classes"] < 25
        
        return {
            "phase1_complete": phase1_complete,
            "phase2_complete": phase2_complete,
            "phase3_complete": phase3_complete,
            "overall_progress": sum([phase1_complete, phase2_complete, phase3_complete]) / 3 * 100
        }
    
    def analyze_test_coverage(self) -> Dict:
        """Analyze test coverage (basic)."""
        tests_dir = self.repo_root / "tests"
        
        if not tests_dir.exists():
            return {"error": "tests directory not found"}
        
        test_files = list(tests_dir.rglob("test_*.py"))
        
        total_tests = 0
        for test_file in test_files:
            try:
                content = test_file.read_text()
                # Count test functions
                total_tests += len(re.findall(r"def test_\w+", content))
            except Exception:
                pass
        
        return {
            "test_files": len(test_files),
            "total_tests": total_tests,
            "tests_per_file": total_tests / len(test_files) if test_files else 0
        }
    
    def analyze_imports(self) -> Dict:
        """Analyze import structure."""
        canonical_imports = self._count_pattern(r"from ign_lidar\.features\.compute import compute_normals")
        deprecated_imports = self._count_pattern(r"from ign_lidar\.features\.feature_computer import")
        knn_engine_imports = self._count_pattern(r"from ign_lidar\.optimization import KNNEngine")
        
        return {
            "canonical_compute_normals_imports": canonical_imports,
            "deprecated_feature_computer_imports": deprecated_imports,
            "knn_engine_imports": knn_engine_imports
        }
    
    def analyze_performance_markers(self) -> Dict:
        """Look for performance-related code markers."""
        gpu_checks = self._count_pattern(r"GPU_AVAILABLE|use_gpu")
        stream_usage = self._count_pattern(r"CUDAStreamManager|cuda.*stream")
        profiler_usage = self._count_pattern(r"GPUTransferProfiler|profile.*transfer")
        
        return {
            "gpu_conditional_checks": gpu_checks,
            "stream_usage_occurrences": stream_usage,
            "profiler_usage_occurrences": profiler_usage
        }
    
    def get_git_stats(self) -> Dict:
        """Get git statistics if available."""
        try:
            # Check if we're in a git repo
            subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                check=True,
                cwd=self.repo_root
            )
            
            # Get current branch
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                cwd=self.repo_root
            )
            current_branch = branch_result.stdout.strip()
            
            # Get uncommitted changes
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=self.repo_root
            )
            has_changes = bool(status_result.stdout.strip())
            
            # Count commits
            commit_result = subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.repo_root
            )
            total_commits = int(commit_result.stdout.strip()) if commit_result.stdout.strip() else 0
            
            return {
                "current_branch": current_branch,
                "has_uncommitted_changes": has_changes,
                "total_commits": total_commits,
                "git_available": True
            }
        except Exception:
            return {"git_available": False}
    
    def generate_full_analysis(self) -> Dict:
        """Generate complete analysis."""
        return {
            "timestamp": datetime.now().isoformat(),
            "repo_root": str(self.repo_root),
            "code_metrics": self.count_files_and_lines(),
            "duplication": self.analyze_duplication(),
            "classes": self.analyze_classes(),
            "phases": self.check_phase_completion(),
            "tests": self.analyze_test_coverage(),
            "imports": self.analyze_imports(),
            "performance": self.analyze_performance_markers(),
            "git": self.get_git_stats()
        }
    
    def _count_pattern(self, pattern: str) -> int:
        """Count occurrences of a regex pattern."""
        try:
            result = subprocess.run(
                ["grep", "-r", "-E", pattern, str(self.ign_lidar)],
                capture_output=True,
                text=True
            )
            return len(result.stdout.strip().split('\n')) if result.stdout else 0
        except Exception:
            return 0
    
    def print_analysis(self, analysis: Dict):
        """Print analysis in formatted output."""
        print("\n" + "=" * 80)
        print("📊 REFACTORING IMPACT ANALYSIS")
        print("=" * 80)
        print(f"\nTimestamp: {analysis['timestamp']}")
        
        # Code Metrics
        print("\n" + "-" * 80)
        print("📁 CODE METRICS")
        print("-" * 80)
        cm = analysis['code_metrics']
        print(f"  Files:        {cm['total_files']}")
        print(f"  Total lines:  {cm['total_lines']:,}")
        print(f"  Code lines:   {cm['code_lines']:,}")
        print(f"  Comments:     {cm['comment_lines']:,}")
        print(f"  Blank:        {cm['blank_lines']:,}")
        print(f"  Avg/file:     {cm['avg_lines_per_file']:.1f}")
        
        # Duplication
        print("\n" + "-" * 80)
        print("🔄 DUPLICATION ANALYSIS")
        print("-" * 80)
        dup = analysis['duplication']
        print(f"  compute_normals():   {dup['compute_normals_implementations']} (target: 1)")
        print(f"  validate_normals():  {dup['validate_normals_implementations']} (target: 1)")
        print(f"  KNN implementations: {dup['knn_implementations']} (target: 1)")
        
        # Classes
        print("\n" + "-" * 80)
        print("🏗️  CLASS STRUCTURE")
        print("-" * 80)
        cls = analysis['classes']
        print(f"  Processor classes: {cls['processor_classes']}")
        print(f"  Engine classes:    {cls['engine_classes']}")
        print(f"  Computer classes:  {cls['computer_classes']}")
        print(f"  Manager classes:   {cls['manager_classes']}")
        print(f"  Total:             {cls['total_related_classes']} (target: <25)")
        
        # Phases
        print("\n" + "-" * 80)
        print("✅ PHASE COMPLETION")
        print("-" * 80)
        phases = analysis['phases']
        p1_status = "✅ COMPLETE" if phases['phase1_complete'] else "❌ NOT COMPLETE"
        p2_status = "✅ COMPLETE" if phases['phase2_complete'] else "❌ NOT COMPLETE"
        p3_status = "✅ COMPLETE" if phases['phase3_complete'] else "❌ NOT COMPLETE"
        
        print(f"  Phase 1 (Duplications): {p1_status}")
        print(f"  Phase 2 (GPU):          {p2_status}")
        print(f"  Phase 3 (Architecture): {p3_status}")
        print(f"  Overall progress:       {phases['overall_progress']:.1f}%")
        
        # Tests
        print("\n" + "-" * 80)
        print("🧪 TEST COVERAGE")
        print("-" * 80)
        tests = analysis['tests']
        if 'error' not in tests:
            print(f"  Test files:     {tests['test_files']}")
            print(f"  Total tests:    {tests['total_tests']}")
            print(f"  Tests per file: {tests['tests_per_file']:.1f}")
        else:
            print(f"  Error: {tests['error']}")
        
        # Imports
        print("\n" + "-" * 80)
        print("📦 IMPORT ANALYSIS")
        print("-" * 80)
        imports = analysis['imports']
        print(f"  Canonical compute_normals: {imports['canonical_compute_normals_imports']}")
        print(f"  Deprecated imports:        {imports['deprecated_feature_computer_imports']}")
        print(f"  KNN engine imports:        {imports['knn_engine_imports']}")
        
        # Performance
        print("\n" + "-" * 80)
        print("⚡ PERFORMANCE MARKERS")
        print("-" * 80)
        perf = analysis['performance']
        print(f"  GPU checks:       {perf['gpu_conditional_checks']}")
        print(f"  Stream usage:     {perf['stream_usage_occurrences']}")
        print(f"  Profiler usage:   {perf['profiler_usage_occurrences']}")
        
        # Git
        print("\n" + "-" * 80)
        print("📝 GIT STATUS")
        print("-" * 80)
        git = analysis['git']
        if git['git_available']:
            print(f"  Current branch:     {git['current_branch']}")
            print(f"  Uncommitted changes: {'Yes' if git['has_uncommitted_changes'] else 'No'}")
            print(f"  Total commits:      {git['total_commits']}")
        else:
            print("  Git not available")
        
        print("\n" + "=" * 80 + "\n")
    
    def compare_with_baseline(self, baseline_path: Path):
        """Compare current state with baseline."""
        if not baseline_path.exists():
            print(f"❌ Baseline file not found: {baseline_path}")
            return
        
        baseline = json.loads(baseline_path.read_text())
        current = self.generate_full_analysis()
        
        print("\n" + "=" * 80)
        print("📊 REFACTORING IMPACT: BASELINE vs CURRENT")
        print("=" * 80)
        
        # Compare key metrics
        metrics = [
            ("Code lines", "code_metrics", "code_lines", "lower"),
            ("compute_normals() count", "duplication", "compute_normals_implementations", "lower"),
            ("Total classes", "classes", "total_related_classes", "lower"),
            ("Test count", "tests", "total_tests", "higher"),
            ("KNN engine imports", "imports", "knn_engine_imports", "higher"),
        ]
        
        for label, category, key, direction in metrics:
            baseline_val = baseline.get(category, {}).get(key, 0)
            current_val = current.get(category, {}).get(key, 0)
            
            if baseline_val == 0:
                change_pct = 0
            else:
                change_pct = ((current_val - baseline_val) / baseline_val) * 100
            
            if direction == "lower":
                symbol = "✅" if change_pct < 0 else "❌"
            else:
                symbol = "✅" if change_pct > 0 else "❌"
            
            print(f"\n{label}:")
            print(f"  Baseline: {baseline_val}")
            print(f"  Current:  {current_val}")
            print(f"  Change:   {symbol} {change_pct:+.1f}%")
        
        print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze refactoring impact'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        metavar='FILE',
        help='Compare with baseline JSON file'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate full report'
    )
    parser.add_argument(
        '--export',
        type=str,
        metavar='FILE',
        help='Export analysis to JSON file'
    )
    
    args = parser.parse_args()
    
    repo_root = Path(__file__).parent.parent.resolve()
    analyzer = RefactoringImpactAnalyzer(repo_root)
    
    if args.baseline:
        analyzer.compare_with_baseline(Path(args.baseline))
    else:
        analysis = analyzer.generate_full_analysis()
        
        if args.export:
            Path(args.export).write_text(json.dumps(analysis, indent=2))
            print(f"✅ Analysis exported to: {args.export}")
        
        if args.report or not args.export:
            analyzer.print_analysis(analysis)


if __name__ == '__main__':
    main()
