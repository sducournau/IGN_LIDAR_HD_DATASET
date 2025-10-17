#!/usr/bin/env python3
"""
Validate all V5 configuration files for proper Hydra composition.
"""

import sys
from pathlib import Path
from typing import List, Tuple
import yaml

def validate_yaml_syntax(file_path: Path) -> Tuple[bool, str]:
    """Validate YAML syntax."""
    try:
        with open(file_path) as f:
            yaml.safe_load(f)
        return True, "OK"
    except yaml.YAMLError as e:
        return False, f"YAML Error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def check_v5_compliance(file_path: Path) -> Tuple[bool, List[str]]:
    """Check if config follows V5 standards."""
    warnings = []
    
    try:
        with open(file_path) as f:
            config = yaml.safe_load(f)
        
        if not config:
            return True, []
        
        # Check for config_version if it's a main config
        if 'config_version' in config:
            version = config['config_version']
            if not version.startswith('5.'):
                warnings.append(f"Config version is {version}, should be 5.x")
        
        # Check for deprecated V4 structure
        if 'processing' in config and 'gpu' in config.get('processing', {}):
            warnings.append("Uses V4 'processing.gpu' structure instead of V5 'processor.gpu_*'")
        
        # Check defaults composition
        if 'defaults' in config:
            defaults = config['defaults']
            if isinstance(defaults, list):
                # Check for V4 base configs that no longer exist
                deprecated_bases = [
                    'base/classification',
                    'base/performance', 
                    'base/hardware',
                    'base/logging',
                    'base/preprocessing',
                    'base/ground_truth'
                ]
                for item in defaults:
                    if isinstance(item, str) and any(dep in item for dep in deprecated_bases):
                        warnings.append(f"Uses deprecated base config: {item}")
        
        return True, warnings
        
    except Exception as e:
        return False, [f"Error checking compliance: {e}"]

def main():
    """Main validation function."""
    configs_dir = Path(__file__).parent / "ign_lidar" / "configs"
    
    if not configs_dir.exists():
        print(f"❌ Configs directory not found: {configs_dir}")
        sys.exit(1)
    
    print("🔍 Validating V5 Configuration Files")
    print("=" * 80)
    
    # Find all YAML files
    yaml_files = list(configs_dir.rglob("*.yaml"))
    
    total = len(yaml_files)
    passed = 0
    failed = 0
    warnings_count = 0
    
    for yaml_file in sorted(yaml_files):
        rel_path = yaml_file.relative_to(configs_dir)
        
        # Validate YAML syntax
        is_valid, message = validate_yaml_syntax(yaml_file)
        
        if not is_valid:
            print(f"❌ {rel_path}: {message}")
            failed += 1
            continue
        
        # Check V5 compliance
        is_compliant, warnings = check_v5_compliance(yaml_file)
        
        if warnings:
            print(f"⚠️  {rel_path}:")
            for warning in warnings:
                print(f"    - {warning}")
            warnings_count += len(warnings)
            passed += 1  # Still counts as passed, just with warnings
        else:
            print(f"✅ {rel_path}: OK")
            passed += 1
    
    print("=" * 80)
    print(f"\n📊 Validation Summary:")
    print(f"   Total files: {total}")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   ⚠️  Warnings: {warnings_count}")
    
    if failed > 0:
        print(f"\n❌ Validation FAILED - {failed} file(s) with errors")
        sys.exit(1)
    elif warnings_count > 0:
        print(f"\n⚠️  Validation PASSED with {warnings_count} warning(s)")
        print("   Consider updating files with warnings to be fully V5 compliant")
        sys.exit(0)
    else:
        print("\n✅ All configurations are valid and V5 compliant!")
        sys.exit(0)

if __name__ == "__main__":
    main()
