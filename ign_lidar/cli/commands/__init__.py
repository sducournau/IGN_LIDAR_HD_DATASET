"""
CLI command modules for IGN LiDAR HD v3.2+

This package contains modular command implementations:
- process: Main processing command
- download: Dataset download command
- verify: Data verification command
- batch_convert: Batch conversion for QGIS compatibility
- info: Package and configuration information
- ground_truth: Ground truth fetching from IGN BD TOPO®
- update_classification: Update LAZ classification with ground truth and NDVI
- auto_config: Auto-configuration based on system capabilities
- presets: List and show configuration presets
- migrate_config: Migrate old configs to v3.2 format (NEW in v3.2)
"""

from .auto_config import auto_config
from .batch_convert import batch_convert_command
from .config_commands import (
    list_presets_command,
    list_profiles_command,
    show_config_command,
    validate_config_command,
)
from .download import download_command
from .ground_truth import ground_truth_command
from .info import info_command
from .migrate_config import migrate_config
from .presets import presets_command
from .process import process_command
from .update_classification import update_classification_command
from .verify import verify_command

__all__ = [
    "process_command",
    "download_command",
    "verify_command",
    "batch_convert_command",
    "info_command",
    "ground_truth_command",
    "update_classification_command",
    "auto_config",
    "presets_command",
    "migrate_config",
    "validate_config_command",
    "list_profiles_command",
    "list_presets_command",
    "show_config_command",
]
