#!/usr/bin/env python3
"""
Configuration Harmonization Script

This script helps migrate and harmonize configuration files to the new structure.

Usage:
    python scripts/harmonize_configs.py --analyze    # Analyze current configs
    python scripts/harmonize_configs.py --migrate    # Perform migration (dry-run)
    python scripts/harmonize_configs.py --migrate --execute  # Execute migration
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
from collections import defaultdict


class ConfigAnalyzer:
    """Analyze configuration files for inconsistencies and duplicates."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.configs_dir = project_root / "configs"
        self.hydra_configs_dir = project_root / "ign_lidar" / "configs"
        
    def find_all_configs(self) -> Dict[str, List[Path]]:
        """Find all YAML configuration files."""
        locations = {
            "legacy_configs": list(self.configs_dir.glob("*.yaml")),
            "multiscale_configs": list((self.configs_dir / "multiscale").rglob("*.yaml")),
            "hydra_configs": list(self.hydra_configs_dir.rglob("*.yaml")),
            "example_configs": list((self.project_root / "examples").glob("config_*.yaml")),
        }
        return locations
    
    def check_hydra_composition(self, config_path: Path) -> Tuple[bool, List[str]]:
        """Check if config uses Hydra composition pattern."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
                if content and 'defaults' in content:
                    return True, content['defaults']
                return False, []
        except Exception as e:
            print(f"Error reading {config_path}: {e}")
            return False, []
    
    def detect_duplicates(self, configs: List[Path]) -> Dict[str, List[Path]]:
        """Detect duplicate configuration keys across files."""
        key_locations = defaultdict(list)
        
        for config_path in configs:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    content = yaml.safe_load(f)
                    if content:
                        self._extract_keys(content, config_path, key_locations, prefix="")
            except Exception as e:
                print(f"Error reading {config_path}: {e}")
        
        # Filter to only duplicates
        duplicates = {k: v for k, v in key_locations.items() if len(v) > 1}
        return duplicates
    
    def _extract_keys(self, data: dict, config_path: Path, key_locations: dict, prefix: str):
        """Recursively extract keys from nested config."""
        if not isinstance(data, dict):
            return
        
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            # Track location of this key
            if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                key_locations[full_key].append(config_path)
            
            # Recurse for nested dicts
            if isinstance(value, dict):
                self._extract_keys(value, config_path, key_locations, full_key)
    
    def analyze(self) -> Dict:
        """Run full analysis."""
        print("="*70)
        print("Configuration Analysis")
        print("="*70)
        
        configs = self.find_all_configs()
        
        # Summary
        print("\n📊 Configuration Files by Location:")
        total = 0
        for location, files in configs.items():
            count = len(files)
            total += count
            print(f"  {location:30s}: {count:3d} files")
        print(f"  {'TOTAL':30s}: {total:3d} files")
        
        # Check Hydra composition
        print("\n🔍 Hydra Composition Analysis:")
        all_configs = []
        for files in configs.values():
            all_configs.extend(files)
        
        hydra_count = 0
        non_hydra_count = 0
        for config_path in all_configs:
            uses_hydra, defaults = self.check_hydra_composition(config_path)
            if uses_hydra:
                hydra_count += 1
            else:
                non_hydra_count += 1
        
        print(f"  Configs using Hydra composition: {hydra_count}")
        print(f"  Configs NOT using Hydra:         {non_hydra_count}")
        
        # Duplicates
        print("\n🔎 Checking for Duplicate Keys...")
        duplicates = self.detect_duplicates(all_configs)
        
        # Show top duplicates
        sorted_duplicates = sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True)
        print(f"\n  Found {len(duplicates)} duplicate keys across files")
        print("\n  Top 10 Most Duplicated Keys:")
        for key, locations in sorted_duplicates[:10]:
            print(f"    {key:40s}: in {len(locations)} files")
        
        return {
            "configs": configs,
            "total": total,
            "hydra_count": hydra_count,
            "non_hydra_count": non_hydra_count,
            "duplicates": duplicates,
        }


class ConfigMigrator:
    """Migrate configurations to new harmonized structure."""
    
    def __init__(self, project_root: Path, dry_run: bool = True):
        self.project_root = project_root
        self.dry_run = dry_run
        self.migrations = []
        
    def plan_migration(self):
        """Plan configuration migration."""
        print("\n" + "="*70)
        print("Migration Plan")
        print("="*70)
        
        configs_dir = self.project_root / "configs"
        hydra_configs_dir = self.project_root / "ign_lidar" / "configs"
        
        # Step 1: Create new directories
        self._plan_mkdir(hydra_configs_dir / "data_sources")
        self._plan_mkdir(hydra_configs_dir / "experiment")
        self._plan_mkdir(configs_dir / "deprecated")
        
        # Step 2: Migrate multiscale configs to experiments
        multiscale_dir = configs_dir / "multiscale"
        if multiscale_dir.exists():
            for config_file in multiscale_dir.rglob("*.yaml"):
                if config_file.is_file():
                    # Determine new location
                    new_name = config_file.name.replace("config_", "")
                    new_path = hydra_configs_dir / "experiment" / new_name
                    self._plan_move(config_file, new_path)
        
        # Step 3: Deprecate legacy configs
        legacy_configs = [
            "classification_config.yaml",
            "processing_config.yaml",
            "enrichment_asprs_full.yaml",
            "example_enrichment_full.yaml",
        ]
        for config_name in legacy_configs:
            old_path = configs_dir / config_name
            if old_path.exists():
                new_path = configs_dir / "deprecated" / config_name
                self._plan_move(old_path, new_path)
        
        # Step 4: Migrate important configs
        migrations = {
            "reclassification_config.yaml": "experiment/reclassify_only.yaml",
            "processing_with_reclassification.yaml": "experiment/enriched_with_reclassification.yaml",
        }
        for old_name, new_path in migrations.items():
            old_path = configs_dir / old_name
            if old_path.exists():
                new_path = hydra_configs_dir / new_path
                self._plan_move(old_path, new_path)
        
        # Print plan
        print(f"\n📋 Planned Actions: {len(self.migrations)}")
        for i, (action, source, target) in enumerate(self.migrations, 1):
            print(f"\n  {i}. {action.upper()}")
            if source:
                print(f"     From: {source}")
            if target:
                print(f"     To:   {target}")
        
        return self.migrations
    
    def _plan_mkdir(self, path: Path):
        """Plan directory creation."""
        if not path.exists():
            self.migrations.append(("mkdir", None, path))
    
    def _plan_move(self, source: Path, target: Path):
        """Plan file move."""
        self.migrations.append(("move", source, target))
    
    def execute_migration(self):
        """Execute the planned migration."""
        if self.dry_run:
            print("\n⚠️  DRY RUN MODE - No changes will be made")
            print("    Run with --execute to perform actual migration")
            return
        
        print("\n" + "="*70)
        print("Executing Migration")
        print("="*70)
        
        for i, (action, source, target) in enumerate(self.migrations, 1):
            try:
                if action == "mkdir":
                    target.mkdir(parents=True, exist_ok=True)
                    print(f"✓ Created directory: {target}")
                
                elif action == "move":
                    # Ensure target directory exists
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source), str(target))
                    print(f"✓ Moved: {source.name} -> {target}")
                
            except Exception as e:
                print(f"✗ Error executing {action}: {e}")
        
        print(f"\n✅ Migration complete! Executed {len(self.migrations)} actions")


def create_data_source_configs(hydra_configs_dir: Path, dry_run: bool = True):
    """Create new modular data source configurations."""
    print("\n" + "="*70)
    print("Creating Data Source Configs")
    print("="*70)
    
    data_sources_dir = hydra_configs_dir / "data_sources"
    
    bd_topo_config = """# @package data_sources.bd_topo
# BD TOPO® V3 - Ground Truth Classification Configuration

enabled: true
cache_enabled: true

# Features to fetch and apply
features:
  buildings: true    # ASPRS 6
  roads: true        # ASPRS 11
  railways: true     # ASPRS 10
  water: true        # ASPRS 9
  vegetation: true   # ASPRS 5
  bridges: true      # ASPRS 17
  parking: true      # ASPRS 40
  cemeteries: false  # ASPRS 42
  power_lines: false # ASPRS 43
  sports: true       # ASPRS 41

# Geometry generation parameters
parameters:
  road_width_fallback: 4.0
  road_buffer_tolerance: 0.5
  railway_width_fallback: 3.5
  railway_buffer_tolerance: 0.6
  power_line_buffer: 2.0
  
  # Height filtering
  road_height_max: 1.5
  road_height_min: -0.3
  rail_height_max: 1.2
  rail_height_min: -0.2
  
  # Geometric filtering
  road_planarity_min: 0.6
  rail_planarity_min: 0.5
  
  # Intensity filtering
  enable_intensity_filter: true
  road_intensity_min: 0.15
  road_intensity_max: 0.7
  rail_intensity_min: 0.1
  rail_intensity_max: 0.8

cache_dir: null  # Uses default cache location
"""
    
    if not dry_run:
        data_sources_dir.mkdir(parents=True, exist_ok=True)
        (data_sources_dir / "bd_topo.yaml").write_text(bd_topo_config)
        print("✓ Created bd_topo.yaml")
    else:
        print("  Would create: bd_topo.yaml")
        print("  Content preview:")
        print("    " + "\n    ".join(bd_topo_config.split("\n")[:10]))


def main():
    parser = argparse.ArgumentParser(description="Harmonize configuration files")
    parser.add_argument("--analyze", action="store_true", help="Analyze current configs")
    parser.add_argument("--migrate", action="store_true", help="Migrate configs to new structure")
    parser.add_argument("--execute", action="store_true", help="Execute migration (not dry-run)")
    parser.add_argument("--create-modules", action="store_true", help="Create new modular configs")
    
    args = parser.parse_args()
    
    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print(f"\n📁 Project root: {project_root}")
    
    if args.analyze or (not args.migrate and not args.create_modules):
        analyzer = ConfigAnalyzer(project_root)
        results = analyzer.analyze()
    
    if args.migrate:
        dry_run = not args.execute
        migrator = ConfigMigrator(project_root, dry_run=dry_run)
        migrator.plan_migration()
        migrator.execute_migration()
    
    if args.create_modules:
        dry_run = not args.execute
        hydra_configs_dir = project_root / "ign_lidar" / "configs"
        create_data_source_configs(hydra_configs_dir, dry_run=dry_run)
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == "__main__":
    main()
