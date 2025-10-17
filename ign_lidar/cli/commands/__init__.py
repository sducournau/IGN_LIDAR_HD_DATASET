"""
CLI command modules for IGN LiDAR HD v2.0.

This package contains modular command implementations:
- process: Main processing command
- download: Dataset download command  
- verify: Data verification command
- batch_convert: Batch conversion for QGIS compatibility
- info: Package and configuration information
- ground_truth: Ground truth fetching from IGN BD TOPO®
- update_classification: Update LAZ classification with ground truth and NDVI
- auto_config: Auto-configuration based on system capabilities
"""

from .process import process_command
from .download import download_command
from .verify import verify_command
from .batch_convert import batch_convert_command
from .info import info_command
from .ground_truth import ground_truth_command
from .update_classification import update_classification_command
from .auto_config import auto_config

__all__ = [
    'process_command',
    'download_command', 
    'verify_command',
    'batch_convert_command',
    'info_command',
    'ground_truth_command',
    'update_classification_command',
    'auto_config'
]
