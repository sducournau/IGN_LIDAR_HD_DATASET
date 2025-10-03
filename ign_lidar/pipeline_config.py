"""
Pipeline configuration loader for YAML-based workflows.

This module provides utilities to load and validate pipeline configurations
from YAML files, enabling complete workflow automation.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class PipelineConfig:
    """
    Load and validate pipeline configuration from YAML files.
    
    Supports complete workflows: download → enrich → patch
    """
    
    def __init__(self, config_path: Path):
        """
        Initialize pipeline configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            raise ValueError(f"Empty configuration file: {self.config_path}")
        
        return config
    
    def _validate_config(self):
        """Validate configuration structure."""
        # Check for at least one pipeline stage
        valid_stages = ['download', 'enrich', 'patch']
        has_stage = any(stage in self.config for stage in valid_stages)
        
        if not has_stage:
            raise ValueError(
                f"Configuration must include at least one stage: {valid_stages}"
            )
    
    @property
    def has_download(self) -> bool:
        """Check if download stage is configured."""
        return 'download' in self.config and self.config['download'] is not None
    
    @property
    def has_enrich(self) -> bool:
        """Check if enrich stage is configured."""
        return 'enrich' in self.config and self.config['enrich'] is not None
    
    @property
    def has_patch(self) -> bool:
        """Check if patch stage is configured."""
        return 'patch' in self.config and self.config['patch'] is not None
    
    def get_download_config(self) -> Optional[Dict[str, Any]]:
        """Get download stage configuration."""
        return self.config.get('download', None)
    
    def get_enrich_config(self) -> Optional[Dict[str, Any]]:
        """Get enrich stage configuration."""
        return self.config.get('enrich', None)
    
    def get_patch_config(self) -> Optional[Dict[str, Any]]:
        """Get patch stage configuration."""
        return self.config.get('patch', None)
    
    def get_global_config(self) -> Dict[str, Any]:
        """Get global configuration (applies to all stages)."""
        return self.config.get('global', {})
    
    def __repr__(self) -> str:
        stages = []
        if self.has_download:
            stages.append('download')
        if self.has_enrich:
            stages.append('enrich')
        if self.has_patch:
            stages.append('patch')
        
        return f"PipelineConfig(stages={stages})"


def create_example_config(output_path: Path, config_type: str = 'full'):
    """
    Create example configuration file.
    
    Args:
        output_path: Path to save example config
        config_type: Type of config ('full', 'enrich', 'patch')
    """
    if config_type == 'full':
        config = {
            'global': {
                'num_workers': 4,
                'output_dir': 'data/processed',
            },
            'download': {
                'bbox': '2.3, 48.8, 2.4, 48.9',  # Paris area (WGS84)
                'output': 'data/raw',
                'max_tiles': 10,
            },
            'enrich': {
                'input_dir': 'data/raw',
                'output': 'data/enriched',
                'mode': 'building',
                'k_neighbors': 10,
                'use_gpu': True,
                'add_rgb': True,
                'rgb_cache_dir': 'cache/orthophotos',
            },
            'patch': {
                'input_dir': 'data/enriched',
                'output': 'data/patches',
                'lod_level': 'LOD2',
                'patch_size': 150.0,
                'num_points': 16384,
                'augment': True,
                'num_augmentations': 3,
            },
        }
    elif config_type == 'enrich':
        config = {
            'global': {
                'num_workers': 4,
            },
            'enrich': {
                'input_dir': 'data/raw',
                'output': 'data/enriched',
                'mode': 'building',
                'k_neighbors': 10,
                'use_gpu': True,
                'add_rgb': True,
                'rgb_cache_dir': 'cache/orthophotos',
            },
        }
    elif config_type == 'patch':
        config = {
            'global': {
                'num_workers': 4,
            },
            'patch': {
                'input_dir': 'data/enriched',
                'output': 'data/patches',
                'lod_level': 'LOD2',
                'patch_size': 150.0,
                'num_points': 16384,
                'augment': True,
                'num_augmentations': 3,
            },
        }
    else:
        raise ValueError(f"Unknown config type: {config_type}")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Created example configuration: {output_path}")
    return output_path
