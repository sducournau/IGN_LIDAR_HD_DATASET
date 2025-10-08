#!/usr/bin/env python3
"""
Migration script for IGN LiDAR HD v1.7.7 → v2.0.0

Converts old command-line arguments and YAML configs to Hydra format.

Usage:
    # Convert command
    python scripts/migrate_to_v2.py command "ign-lidar-hd process --input-dir data/raw ..."
    
    # Convert YAML config
    python scripts/migrate_to_v2.py config pipeline.yaml -o new_config.yaml
"""

import argparse
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple


def parse_old_command(command: str) -> Dict[str, str]:
    """
    Parse old v1.7.7 command-line arguments.
    
    Args:
        command: Full command string
        
    Returns:
        Dictionary of parameter mappings
    """
    mappings = {}
    
    # Extract arguments
    patterns = {
        r'--input-dir\s+(\S+)': 'input_dir',
        r'--output\s+(\S+)': 'output_dir',
        r'--num-points\s+(\d+)': 'processor.num_points',
        r'--patch-size\s+([\d.]+)': 'processor.patch_size',
        r'--patch-overlap\s+([\d.]+)': 'processor.patch_overlap',
        r'--k-neighbors\s+(\d+)': 'features.k_neighbors',
        r'--lod-level\s+(\w+)': 'processor.lod_level',
        r'--num-workers\s+(\d+)': 'num_workers',
    }
    
    for pattern, hydra_key in patterns.items():
        match = re.search(pattern, command)
        if match:
            mappings[hydra_key] = match.group(1)
    
    # Boolean flags
    bool_flags = {
        '--use-gpu': ('processor.use_gpu', 'true'),
        '--augment': ('processor.augment', 'true'),
        '--add-rgb': ('features.use_rgb', 'true'),
        '--include-extra': ('features.include_extra', 'true'),
        '--preprocess': ('preprocess.enabled', 'true'),
    }
    
    for flag, (hydra_key, value) in bool_flags.items():
        if flag in command:
            mappings[hydra_key] = value
    
    return mappings


def generate_hydra_command(mappings: Dict[str, str]) -> str:
    """
    Generate Hydra command from parameter mappings.
    
    Args:
        mappings: Parameter mappings
        
    Returns:
        Hydra command string
    """
    # Check for common patterns to suggest presets
    preset = None
    
    if mappings.get('features.use_rgb') == 'true' and \
       mappings.get('features.include_extra') == 'true':
        if 'vegetation' in str(mappings):
            preset = 'experiment=vegetation_ndvi'
        elif mappings.get('processor.lod_level') == 'LOD2':
            preset = 'experiment=buildings_lod2'
        elif mappings.get('processor.lod_level') == 'LOD3':
            preset = 'experiment=buildings_lod3'
    
    # Build command
    parts = ['python -m ign_lidar.cli.hydra_main']
    
    if preset:
        parts.append(preset)
        # Filter out parameters covered by preset
        preset_params = {'processor.lod_level', 'features.use_rgb', 'features.include_extra'}
        mappings = {k: v for k, v in mappings.items() if k not in preset_params}
    else:
        # Suggest processor config
        if mappings.get('processor.use_gpu') == 'true':
            parts.append('processor=gpu')
            del mappings['processor.use_gpu']
    
    # Add remaining overrides
    for key, value in sorted(mappings.items()):
        parts.append(f'{key}={value}')
    
    return ' \\\n    '.join(parts)


def convert_yaml_config(input_file: Path, output_file: Path) -> None:
    """
    Convert old YAML config to Hydra format.
    
    Args:
        input_file: Path to old config file
        output_file: Path to new config file
    """
    import yaml
    
    # Load old config
    with open(input_file, 'r') as f:
        old_config = yaml.safe_load(f)
    
    # Convert to Hydra format
    hydra_config = {
        'defaults': [
            {'processor': 'default'},
            {'features': 'full'},
            {'preprocess': 'default'},
            '_self_'
        ]
    }
    
    # Map old structure to new
    if 'process' in old_config:
        process = old_config['process']
        hydra_config.update({
            'input_dir': process.get('input_dir', '???'),
            'output_dir': process.get('output_dir', '???'),
        })
        
        # Processor config
        processor_overrides = {}
        if 'num_points' in process:
            processor_overrides['num_points'] = process['num_points']
        if 'patch_size' in process:
            processor_overrides['patch_size'] = process['patch_size']
        if 'use_gpu' in process:
            processor_overrides['use_gpu'] = process['use_gpu']
        
        if processor_overrides:
            hydra_config['processor'] = processor_overrides
    
    # Save new config
    with open(output_file, 'w') as f:
        yaml.dump(hydra_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Converted config saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Migrate IGN LiDAR HD from v1.7.7 to v2.0.0'
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Migration mode')
    
    # Command mode
    cmd_parser = subparsers.add_parser('command', help='Convert command-line arguments')
    cmd_parser.add_argument('command', help='Old command string')
    
    # Config mode
    cfg_parser = subparsers.add_parser('config', help='Convert YAML config file')
    cfg_parser.add_argument('input', type=Path, help='Input config file')
    cfg_parser.add_argument('-o', '--output', type=Path, required=True, help='Output config file')
    
    args = parser.parse_args()
    
    if args.mode == 'command':
        print("="*70)
        print("Converting v1.7.7 command to v2.0.0 Hydra format")
        print("="*70)
        print(f"\nOld command:\n  {args.command}\n")
        
        mappings = parse_old_command(args.command)
        hydra_cmd = generate_hydra_command(mappings)
        
        print(f"New command:\n  {hydra_cmd}\n")
        print("="*70)
        
    elif args.mode == 'config':
        print("="*70)
        print("Converting v1.7.7 YAML config to v2.0.0 Hydra format")
        print("="*70)
        
        convert_yaml_config(args.input, args.output)
        
        print("="*70)
        
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
