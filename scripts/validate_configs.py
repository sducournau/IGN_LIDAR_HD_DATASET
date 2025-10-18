#!/usr/bin/env python3
"""
Configuration Validation Script
Validates all YAML configuration files for consistency and correctness.
Version: 5.1.0
Date: October 18, 2025
"""

import os
import sys
from pathlib import Path
import yaml
from typing import Dict, List, Tuple

# Expected version for all configs
EXPECTED_VERSION = "5.1.0"
PACKAGE_VERSION = "3.0.0"

# Colors for output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'


def load_yaml_safe(file_path: Path) -> Tuple[bool, Dict, str]:
    """Load YAML file safely."""
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        return True, config, ""
    except Exception as e:
        return False, {}, str(e)


def validate_config_version(config: Dict, file_path: Path) -> List[str]:
    """Validate configuration version."""
    issues = []
    
    if 'config_version' not in config:
        issues.append(f"Missing 'config_version' field")
    elif config['config_version'] != EXPECTED_VERSION:
        issues.append(
            f"Version mismatch: expected {EXPECTED_VERSION}, "
            f"got {config['config_version']}"
        )
    
    return issues


def validate_preset(config: Dict, file_path: Path) -> List[str]:
    """Validate preset configuration."""
    issues = []
    
    # Check for preset metadata
    if 'preset_name' not in config:
        issues.append("Missing 'preset_name' field")
    
    # Check for defaults inheritance
    if 'defaults' not in config:
        issues.append("Missing 'defaults' field (should inherit from base)")
    
    # Check for required sections
    required_sections = ['processor', 'features']
    for section in required_sections:
        if section not in config:
            issues.append(f"Missing required section: {section}")
    
    return issues


def validate_hardware(config: Dict, file_path: Path) -> List[str]:
    """Validate hardware configuration."""
    issues = []
    
    # Check for hardware metadata
    if 'hardware_target' not in config:
        issues.append("Missing 'hardware_target' field")
    
    if 'config_type' not in config or config['config_type'] != 'hardware_profile':
        issues.append("Should have config_type='hardware_profile'")
    
    # Check for processor section
    if 'processor' not in config:
        issues.append("Missing 'processor' section")
    else:
        processor = config['processor']
        if 'use_gpu' not in processor:
            issues.append("Missing 'processor.use_gpu' field")
    
    return issues


def validate_example(config: Dict, file_path: Path) -> List[str]:
    """Validate example configuration."""
    issues = []
    
    # Check for either defaults inheritance or config_type
    if 'defaults' not in config and 'config_type' not in config:
        issues.append("Should either inherit via 'defaults' or specify 'config_type'")
    
    return issues


def scan_configs(base_dir: Path) -> Dict[str, List[Path]]:
    """Scan for all configuration files."""
    configs = {
        'base': [],
        'presets': [],
        'hardware': [],
        'examples': [],
        'advanced': [],
        'other': []
    }
    
    # Scan ign_lidar/configs/
    config_dir = base_dir / 'ign_lidar' / 'configs'
    if config_dir.exists():
        # Base config
        base_yaml = config_dir / 'base.yaml'
        if base_yaml.exists():
            configs['base'].append(base_yaml)
        
        # Presets
        presets_dir = config_dir / 'presets'
        if presets_dir.exists():
            configs['presets'].extend(presets_dir.glob('*.yaml'))
        
        # Hardware
        hardware_dir = config_dir / 'hardware'
        if hardware_dir.exists():
            configs['hardware'].extend(hardware_dir.glob('*.yaml'))
        
        # Advanced
        advanced_dir = config_dir / 'advanced'
        if advanced_dir.exists():
            configs['advanced'].extend(advanced_dir.glob('*.yaml'))
    
    # Scan examples/
    examples_dir = base_dir / 'examples'
    if examples_dir.exists():
        configs['examples'].extend(examples_dir.glob('*.yaml'))
    
    return configs


def main():
    """Main validation function."""
    print(f"{BLUE}=== IGN LiDAR HD Configuration Validator V{EXPECTED_VERSION} ==={RESET}\n")
    
    # Get workspace directory
    workspace_dir = Path(__file__).parent
    
    # Scan for configs
    print(f"{BLUE}Scanning for configuration files...{RESET}")
    configs = scan_configs(workspace_dir)
    
    total_files = sum(len(files) for files in configs.values())
    print(f"Found {total_files} configuration files\n")
    
    # Validation results
    all_valid = True
    results = {}
    
    # Validate each category
    for category, files in configs.items():
        if not files:
            continue
        
        print(f"{BLUE}Validating {category} configs ({len(files)} files):{RESET}")
        
        for file_path in sorted(files):
            # Load config
            success, config, error = load_yaml_safe(file_path)
            
            if not success:
                print(f"  {RED}✗{RESET} {file_path.name}: YAML parse error - {error}")
                all_valid = False
                continue
            
            # Validate based on category
            issues = []
            
            # Version check (all configs)
            issues.extend(validate_config_version(config, file_path))
            
            # Category-specific validation
            if category == 'presets':
                issues.extend(validate_preset(config, file_path))
            elif category == 'hardware':
                issues.extend(validate_hardware(config, file_path))
            elif category == 'examples':
                issues.extend(validate_example(config, file_path))
            
            # Report results
            if issues:
                print(f"  {YELLOW}⚠{RESET} {file_path.name}:")
                for issue in issues:
                    print(f"      - {issue}")
                all_valid = False
            else:
                print(f"  {GREEN}✓{RESET} {file_path.name}")
            
            results[str(file_path)] = issues
        
        print()
    
    # Summary
    print(f"{BLUE}=== Validation Summary ==={RESET}")
    valid_count = sum(1 for issues in results.values() if not issues)
    total_count = len(results)
    issue_count = total_count - valid_count
    
    print(f"Total configs: {total_count}")
    print(f"{GREEN}Valid: {valid_count}{RESET}")
    if issue_count > 0:
        print(f"{YELLOW}With issues: {issue_count}{RESET}")
    
    if all_valid:
        print(f"\n{GREEN}✓ All configurations are valid!{RESET}")
        return 0
    else:
        print(f"\n{YELLOW}⚠ Some configurations need attention{RESET}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
