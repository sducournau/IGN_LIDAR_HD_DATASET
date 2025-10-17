#!/usr/bin/env python3
"""Validate all preset configs have required fields."""

from pathlib import Path
from omegaconf import OmegaConf
import sys

REQUIRED_FIELDS = {
    'processor': ['use_gpu', 'num_workers', 'patch_overlap', 'num_points'],
    'features': ['include_extra', 'use_gpu_chunked', 'gpu_batch_size', 'use_nir'],
    'preprocess': ['enabled'],
    'stitching': ['enabled', 'buffer_size'],
    'output': ['format']
}

def validate_preset(preset_path):
    """Validate a preset has all required fields."""
    try:
        cfg = OmegaConf.load(preset_path)
    except Exception as e:
        print(f"\n❌ {preset_path.name}: FAILED TO LOAD")
        print(f"   Error: {e}")
        return False
    
    missing = []

    for section, fields in REQUIRED_FIELDS.items():
        if section not in cfg:
            missing.append(f'❌ Section missing: {section}')
        else:
            for field in fields:
                if field not in cfg[section]:
                    missing.append(f'❌ {section}.{field}')

    if missing:
        print(f"\n{preset_path.name}: FAILED")
        for m in missing:
            print(f"  {m}")
        return False
    else:
        print(f"✅ {preset_path.name}: PASSED")
        return True

if __name__ == '__main__':
    presets_dir = Path('ign_lidar/configs/presets')
    
    if not presets_dir.exists():
        print(f"❌ Presets directory not found: {presets_dir}")
        sys.exit(1)
    
    all_valid = True
    preset_files = list(presets_dir.glob('*.yaml'))
    
    if not preset_files:
        print(f"❌ No preset files found in {presets_dir}")
        sys.exit(1)
    
    print(f"Validating {len(preset_files)} preset configurations...\n")
    
    for preset in sorted(preset_files):
        if not validate_preset(preset):
            all_valid = False

    print("\n" + "="*50)
    if all_valid:
        print("🎉 All presets valid!")
        sys.exit(0)
    else:
        print("⚠️  Some presets need fixing")
        sys.exit(1)
