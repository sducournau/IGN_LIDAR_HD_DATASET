"""
Migration Helper Tools - Phase 6

Tools to help users migrate from old scattered modules to Phase 5 unified managers.

Features:
- Automated import analysis
- Migration suggestions
- Code pattern transformations
- Compatibility checking
"""

import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class MigrationHelper:
    """Helper for migrating code to Phase 5 managers."""

    # Old to new import mappings
    IMPORT_MAPPINGS = {
        # GPU Streams migrations
        ("ign_lidar.optimization.cuda_streams", "get_stream"): (
            "ign_lidar.core",
            "get_stream_manager",
            "manager = get_stream_manager(); stream = manager.get_stream()",
        ),
        ("ign_lidar.optimization.cuda_streams", "create_stream"): (
            "ign_lidar.core",
            "get_stream_manager",
            "manager = get_stream_manager()",
        ),
        ("ign_lidar.optimization.gpu_async", "async_transfer"): (
            "ign_lidar.core",
            "get_stream_manager",
            "manager = get_stream_manager(); manager.async_transfer(src, dst)",
        ),
        # Performance monitoring migrations
        ("ign_lidar.core.performance", "ProcessorPerformanceMonitor"): (
            "ign_lidar.core",
            "get_performance_manager",
            "manager = get_performance_manager()",
        ),
        ("ign_lidar.optimization.gpu_profiler", "GpuProfiler"): (
            "ign_lidar.core",
            "get_performance_manager",
            "Use PerformanceManager.record_metric() instead",
        ),
        ("ign_lidar.utils.performance_monitor", "PerformanceMonitor"): (
            "ign_lidar.core",
            "get_performance_manager",
            "manager = get_performance_manager()",
        ),
        # Config validation migrations
        ("ign_lidar.config.validator", "validate_lod"): (
            "ign_lidar.core",
            "get_config_validator",
            "validator = get_config_validator(); validator.add_lod_validator(); is_valid, _ = validator.validate(config)",
        ),
        ("ign_lidar.features.feature_validator", "FeatureValidator"): (
            "ign_lidar.core",
            "get_config_validator",
            "validator = get_config_validator(); validator.add_rule(...)",
        ),
    }

    # Pattern transformations
    PATTERN_TRANSFORMS = {
        # GPU stream patterns
        r"import.*cuda_streams": "from ign_lidar.core import get_stream_manager",
        r"import.*gpu_async": "from ign_lidar.core import get_stream_manager",
        r"cuda_streams\.get_stream\(\)": "get_stream_manager().get_stream()",
        r"cuda_streams\.create_stream\(\)": "get_stream_manager().get_stream()",
        r"gpu_async\.async_transfer": "get_stream_manager().async_transfer",
        # Performance monitoring patterns
        r"ProcessorPerformanceMonitor\(\)": "get_performance_manager()",
        r"GpuProfiler\(\)": "get_performance_manager()",
        r"PerformanceMonitor\(\)": "get_performance_manager()",
        r"\.start_timing\(\)": ".start_phase('operation')",
        r"\.end_timing\(\)": ".end_phase()",
        # Config validation patterns
        r"validate_lod\(": "get_config_validator().validate(",
        r"FeatureValidator\(\)": "get_config_validator()",
        r"\.validate_field\(": ".validate(",
    }

    def __init__(self):
        """Initialize migration helper."""
        self.issues: List[Dict] = []
        self.suggestions: List[Dict] = []

    def analyze_imports(self, file_path: str) -> Tuple[List[str], List[str]]:
        """
        Analyze imports in a Python file.

        Args:
            file_path: Path to Python file

        Returns:
            Tuple of (deprecated_imports, suggested_imports)
        """
        deprecated_imports = []
        suggested_imports = set()

        try:
            with open(file_path, "r") as f:
                content = f.read()

            # Check for deprecated imports
            for (old_module, old_name), (new_module, new_name, example) in self.IMPORT_MAPPINGS.items():
                pattern = rf"from {old_module} import.*{old_name}"
                if re.search(pattern, content):
                    deprecated_imports.append(f"{old_module}.{old_name}")
                    suggested_imports.add((new_module, new_name))

            return deprecated_imports, list(suggested_imports)

        except Exception as e:
            logger.error(f"Error analyzing imports in {file_path}: {e}")
            return [], []

    def suggest_replacements(self, file_path: str) -> List[Tuple[str, str, str]]:
        """
        Suggest code replacements.

        Args:
            file_path: Path to Python file

        Returns:
            List of (old_pattern, new_pattern, explanation)
        """
        replacements = []

        try:
            with open(file_path, "r") as f:
                content = f.read()

            # Check each pattern
            for old_pattern, new_pattern in self.PATTERN_TRANSFORMS.items():
                if re.search(old_pattern, content):
                    # Find matches
                    matches = re.finditer(old_pattern, content)
                    for match in matches:
                        start_line = content[:match.start()].count("\n") + 1
                        explanation = self._get_explanation(old_pattern, new_pattern)
                        replacements.append((
                            old_pattern,
                            new_pattern,
                            f"Line {start_line}: {explanation}",
                        ))

            return replacements

        except Exception as e:
            logger.error(f"Error suggesting replacements in {file_path}: {e}")
            return []

    @staticmethod
    def _get_explanation(old_pattern: str, new_pattern: str) -> str:
        """Get explanation for replacement."""
        if "cuda_streams" in old_pattern or "gpu_async" in old_pattern:
            return "Use Phase 5 GPUStreamManager instead"
        elif "Performance" in new_pattern or "performance" in old_pattern:
            return "Use Phase 5 PerformanceManager instead"
        elif "Validator" in new_pattern or "validator" in old_pattern:
            return "Use Phase 5 ConfigValidator instead"
        else:
            return "Phase 5 unified manager migration"

    def check_compatibility(self, file_path: str) -> Dict:
        """
        Check compatibility with Phase 5 managers.

        Args:
            file_path: Path to Python file

        Returns:
            Compatibility report
        """
        report = {
            "file": file_path,
            "compatible": True,
            "issues": [],
            "warnings": [],
            "notes": [],
        }

        try:
            deprecated, suggested = self.analyze_imports(file_path)

            if deprecated:
                report["compatible"] = False
                report["issues"].extend([
                    f"Deprecated import: {imp}"
                    for imp in deprecated
                ])
                report["notes"].extend([
                    f"Consider migrating to Phase 5 managers"
                ])

            return report

        except Exception as e:
            report["compatible"] = False
            report["issues"].append(f"Error checking compatibility: {e}")
            return report

    def generate_migration_report(self, file_paths: List[str]) -> str:
        """
        Generate migration report for multiple files.

        Args:
            file_paths: List of Python file paths

        Returns:
            Formatted migration report
        """
        report_lines = [
            "=" * 70,
            "PHASE 5 MIGRATION REPORT",
            "=" * 70,
            "",
        ]

        total_issues = 0
        total_files = len(file_paths)
        compatible_files = 0

        for file_path in file_paths:
            compat = self.check_compatibility(file_path)

            if compat["compatible"]:
                compatible_files += 1
            else:
                total_issues += len(compat["issues"])
                report_lines.extend([
                    f"FILE: {file_path}",
                    f"Status: REQUIRES MIGRATION",
                    "",
                ])

                if compat["issues"]:
                    report_lines.append("Issues:")
                    for issue in compat["issues"]:
                        report_lines.append(f"  • {issue}")
                    report_lines.append("")

                if compat["warnings"]:
                    report_lines.append("Warnings:")
                    for warning in compat["warnings"]:
                        report_lines.append(f"  ⚠ {warning}")
                    report_lines.append("")

                if compat["notes"]:
                    report_lines.append("Notes:")
                    for note in compat["notes"]:
                        report_lines.append(f"  → {note}")
                    report_lines.append("")

        # Summary
        report_lines.extend([
            "=" * 70,
            "SUMMARY",
            "=" * 70,
            f"Total files analyzed: {total_files}",
            f"Compatible: {compatible_files}",
            f"Requires migration: {total_files - compatible_files}",
            f"Total issues found: {total_issues}",
            "",
            "RECOMMENDATIONS",
            "-" * 70,
            "1. Review all deprecated imports",
            "2. Use suggested Phase 5 managers",
            "3. Refer to REFACTORING_PHASE_5_SUMMARY.md for migration guide",
            "4. Test thoroughly after migration",
            "",
        ])

        return "\n".join(report_lines)


class CodeTransformer:
    """Transform code to use Phase 5 managers."""

    @staticmethod
    def transform_gpu_streams(old_code: str) -> str:
        """Transform GPU stream code."""
        # Example transformations
        transforms = {
            r"cuda_streams\.get_stream\(\)": "get_stream_manager().get_stream()",
            r"cuda_streams\.wait_all\(\)": "get_stream_manager().wait_all()",
            r"gpu_async\.async_transfer\((\w+),\s*(\w+)\)": r"get_stream_manager().async_transfer(\1, \2)",
        }

        result = old_code
        for pattern, replacement in transforms.items():
            result = re.sub(pattern, replacement, result)

        return result

    @staticmethod
    def transform_performance(old_code: str) -> str:
        """Transform performance monitoring code."""
        transforms = {
            r"ProcessorPerformanceMonitor\(\)": "get_performance_manager()",
            r"GpuProfiler\(\)": "get_performance_manager()",
            r"PerformanceMonitor\(\)": "get_performance_manager()",
            r"\.start_timing\(\)": ".start_phase('operation')",
            r"\.end_timing\(\)": ".end_phase()",
            r"\.record\((\w+)\)": r".record_metric(\1)",
        }

        result = old_code
        for pattern, replacement in transforms.items():
            result = re.sub(pattern, replacement, result)

        return result

    @staticmethod
    def transform_config_validation(old_code: str) -> str:
        """Transform config validation code."""
        transforms = {
            r"FeatureValidator\(\)": "get_config_validator()",
            r"validate_lod\((\w+)\)": r"get_config_validator().validate(\1)",
            r"\.validate_field\(": ".validate(",
        }

        result = old_code
        for pattern, replacement in transforms.items():
            result = re.sub(pattern, replacement, result)

        return result

    @staticmethod
    def add_imports(old_code: str, imports_needed: set) -> str:
        """Add necessary imports to code."""
        import_lines = []

        if any("stream" in imp for imp in imports_needed):
            import_lines.append("from ign_lidar.core import get_stream_manager")

        if any("performance" in imp for imp in imports_needed):
            import_lines.append("from ign_lidar.core import get_performance_manager")

        if any("validator" in imp for imp in imports_needed):
            import_lines.append("from ign_lidar.core import get_config_validator")

        if not import_lines:
            return old_code

        # Find insertion point (after existing imports)
        lines = old_code.split("\n")
        insert_index = 0

        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                insert_index = i + 1
            elif insert_index > 0:
                break

        # Insert new imports
        result_lines = (
            lines[:insert_index]
            + import_lines
            + lines[insert_index:]
        )

        return "\n".join(result_lines)


def create_migration_helper() -> MigrationHelper:
    """Factory function to create migration helper."""
    return MigrationHelper()


if __name__ == "__main__":
    # Example usage
    helper = create_migration_helper()

    # Analyze a file
    deprecated, suggested = helper.analyze_imports("example.py")

    if deprecated:
        print("Deprecated imports found:")
        for imp in deprecated:
            print(f"  - {imp}")

        print("\nSuggested replacements:")
        for module, name in suggested:
            print(f"  - from {module} import {name}")
