"""
Configuration Migration Utilities for IGN LiDAR HD v4.0

This module provides tools for migrating configurations from v3.x/v5.1 to v4.0.

Key Features:
- Automatic version detection
- Schema transformation (nested → flat)
- Parameter renaming (lod_level → mode, feature_set → mode)
- Validation and reporting
- Backup creation

Author: IGN LiDAR HD Team
Date: November 2025
Version: 4.0.0
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml
from datetime import datetime


class ConfigMigrator:
    """
    Migrates IGN LiDAR HD configurations from v3.x/v5.1 to v4.0.
    
    The migrator handles:
    - Version detection (v3.1, v3.2, v5.1)
    - Structural changes (nested → flat)
    - Parameter renaming
    - Value transformations
    - Validation
    
    Example:
        >>> migrator = ConfigMigrator()
        >>> result = migrator.migrate_file("old_config.yaml", "new_config.yaml")
        >>> print(f"Migration {'successful' if result.success else 'failed'}")
    """
    
    # Version detection patterns
    VERSION_PATTERNS = {
        "3.1": ["processor", "features.mode"],
        "3.2": ["mode", "features.feature_set"],
        "5.1": ["processor.lod_level", "config_version"],
    }
    
    # Parameter mappings for v4.0
    PARAMETER_MAPPINGS = {
        # Processor section → top-level
        "processor.lod_level": "mode",
        "processor.use_gpu": "use_gpu",
        "processor.num_workers": "num_workers",
        "processor.processing_mode": "processing_mode",
        "processor.patch_size": "patch_size",
        "processor.num_points": "num_points",
        "processor.patch_overlap": "patch_overlap",
        "processor.architecture": "architecture",
        
        # Features section
        "features.feature_set": "features.mode",
        "features.mode": "features.mode",  # v3.1 features.mode is different
        
        # GPU settings
        "processor.gpu_batch_size": "advanced.performance.gpu_batch_size",
        "processor.gpu_memory_target": "advanced.performance.gpu_memory_target",
        "processor.gpu_streams": "advanced.performance.gpu_streams",
    }
    
    # Value transformations
    VALUE_TRANSFORMS = {
        "mode": lambda x: x.lower() if isinstance(x, str) else x,  # LOD2 → lod2
        "features.mode": {
            # Map old feature modes to new
            "minimal": "minimal",
            "lod2": "standard",
            "lod3": "full",
            "asprs_classes": "full",
            "full": "full",
            "custom": "standard",
        },
    }
    
    def __init__(self, backup: bool = True):
        """
        Initialize ConfigMigrator.
        
        Args:
            backup: Whether to create backups of original files
        """
        self.backup = backup
        self.migration_log: List[str] = []
    
    def detect_version(self, config: Dict[str, Any]) -> Optional[str]:
        """
        Detect configuration version.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Version string ('3.1', '3.2', '5.1') or None if unknown
        """
        # Check for explicit version marker
        if "config_version" in config:
            version = config["config_version"]
            if version.startswith("5."):
                return "5.1"
            elif version.startswith("4."):
                return "4.0"  # Already v4.0
            elif version.startswith("3."):
                return "3.2"
        
        # Pattern-based detection
        if "processor" in config:
            if "lod_level" in config.get("processor", {}):
                return "5.1"  # v5.1 has processor.lod_level
            return "3.1"  # v3.1 has processor but different structure
        
        if "mode" in config and "features" in config:
            if isinstance(config.get("features"), dict):
                if "feature_set" in config["features"]:
                    return "3.2"  # v3.2 has features.feature_set
                elif "mode" in config["features"]:
                    return "3.1"  # v3.1 has features.mode
        
        return None  # Unknown version
    
    def migrate_dict(self, old_config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Migrate configuration dictionary to v4.0.
        
        Args:
            old_config: Old configuration dictionary
        
        Returns:
            Tuple of (new_config, warnings)
        """
        version = self.detect_version(old_config)
        warnings_list = []
        
        if version == "4.0":
            warnings_list.append("Configuration is already v4.0")
            return old_config, warnings_list
        
        if version is None:
            warnings_list.append("Unknown configuration version, attempting migration...")
        
        new_config = {
            "config_version": "4.0.0",
            "config_name": old_config.get("config_name", old_config.get("preset_name", "migrated")),
        }
        
        # Migrate based on detected version
        if version in ("5.1", "3.1"):
            new_config, version_warnings = self._migrate_from_v5_or_v3_1(old_config)
            warnings_list.extend(version_warnings)
        elif version == "3.2":
            new_config, version_warnings = self._migrate_from_v3_2(old_config)
            warnings_list.extend(version_warnings)
        else:
            # Generic migration
            new_config, version_warnings = self._migrate_generic(old_config)
            warnings_list.extend(version_warnings)
        
        return new_config, warnings_list
    
    def _migrate_from_v5_or_v3_1(self, old: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Migrate from v5.1 or v3.1 (nested processor structure)."""
        warnings_list = []
        new = {
            "config_version": "4.0.0",
            "config_name": old.get("config_name", old.get("preset_name", "migrated")),
        }
        
        # Required fields
        new["input_dir"] = old.get("input_dir")
        new["output_dir"] = old.get("output_dir")
        
        # Extract from processor section
        processor = old.get("processor", {})
        
        # Essential parameters (flattened)
        lod_level = processor.get("lod_level", "ASPRS")
        new["mode"] = lod_level.lower()  # LOD2 → lod2
        
        new["processing_mode"] = processor.get("processing_mode", "patches_only")
        new["use_gpu"] = processor.get("use_gpu", False)
        new["num_workers"] = processor.get("num_workers", 4)
        new["patch_size"] = processor.get("patch_size", 150.0)
        new["num_points"] = processor.get("num_points", 16384)
        new["patch_overlap"] = processor.get("patch_overlap", 0.1)
        new["architecture"] = processor.get("architecture", "pointnet++")
        
        # Features section
        old_features = old.get("features", {})
        if isinstance(old_features, dict):
            old_mode = old_features.get("mode", "full")
            # Map old feature mode to new
            mode_map = {
                "minimal": "minimal",
                "lod2": "standard",
                "lod3": "full",
                "asprs_classes": "full",
                "full": "full",
                "custom": "standard",
            }
            new_mode = mode_map.get(old_mode, "standard")
            
            new["features"] = {
                "mode": new_mode,
                "k_neighbors": old_features.get("k_neighbors", 30),
                "search_radius": old_features.get("search_radius"),
                "use_rgb": old_features.get("use_rgb", False),
                "use_nir": old_features.get("use_infrared", old_features.get("use_nir", False)),
                "compute_ndvi": old_features.get("compute_ndvi", False),
                "multi_scale": old_features.get("multi_scale_computation", old_features.get("multi_scale", False)),
                "scales": old_features.get("scales"),
            }
        
        # Optimizations (new in v4.0)
        new["optimizations"] = {
            "enabled": processor.get("enable_optimizations", True),
            "async_io": processor.get("enable_async_io", True),
            "async_workers": processor.get("async_workers", 2),
            "tile_cache_size": processor.get("tile_cache_size", 3),
            "batch_processing": processor.get("enable_batch_processing", True),
            "batch_size": processor.get("batch_size", 4),
            "gpu_pooling": processor.get("enable_gpu_pooling", True),
            "gpu_pool_max_size_gb": processor.get("gpu_pool_max_size_gb", 4.0),
            "print_stats": processor.get("print_optimization_stats", True),
        }
        
        # Advanced section
        advanced = {}
        
        # Preprocessing
        if "preprocessing" in old:
            advanced["preprocessing"] = old["preprocessing"]
        
        # Ground truth (from data_sources)
        if "data_sources" in old:
            advanced["ground_truth"] = old["data_sources"]
        elif "ground_truth" in old:
            advanced["ground_truth"] = old["ground_truth"]
        
        # Performance settings
        if any(k in processor for k in ["gpu_batch_size", "gpu_memory_target", "gpu_streams"]):
            advanced["performance"] = {
                "gpu_batch_size": processor.get("gpu_batch_size", 1_000_000),
                "gpu_memory_target": processor.get("gpu_memory_target", 0.85),
                "gpu_streams": processor.get("gpu_streams", 4),
                "show_progress": processor.get("show_progress", True),
                "log_level": processor.get("log_level", "INFO"),
            }
        
        if advanced:
            new["advanced"] = advanced
        
        # Optional sections
        if "data_sources" in old:
            new["data_sources"] = old["data_sources"]
        
        if "output" in old:
            new["output"] = old["output"]
        
        if "monitoring" in old:
            new["monitoring"] = old["monitoring"]
        
        if "bbox" in old:
            new["bbox"] = old["bbox"]
        
        if "cache_dir" in old:
            new["cache_dir"] = old["cache_dir"]
        
        # Copy defaults and hydra sections if present
        if "defaults" in old:
            # Update defaults to point to base_v4
            defaults = old["defaults"]
            new_defaults = []
            for item in defaults:
                if isinstance(item, str):
                    if "base" in item and "base_v4" not in item:
                        new_defaults.append("../base_v4")
                        warnings_list.append(f"Changed base reference to base_v4")
                    else:
                        new_defaults.append(item)
                else:
                    new_defaults.append(item)
            new["defaults"] = new_defaults
        
        if "hydra" in old:
            new["hydra"] = old["hydra"]
        
        return new, warnings_list
    
    def _migrate_from_v3_2(self, old: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Migrate from v3.2 (already mostly flat, just rename feature_set → mode)."""
        warnings_list = []
        new = old.copy()
        new["config_version"] = "4.0.0"
        
        # Rename features.feature_set → features.mode
        if "features" in new and isinstance(new["features"], dict):
            if "feature_set" in new["features"]:
                new["features"]["mode"] = new["features"].pop("feature_set")
                warnings_list.append("Renamed features.feature_set to features.mode")
        
        # Ensure mode is lowercase
        if "mode" in new and isinstance(new["mode"], str):
            new["mode"] = new["mode"].lower()
        
        # Update defaults
        if "defaults" in new:
            defaults = new["defaults"]
            new_defaults = []
            for item in defaults:
                if isinstance(item, str):
                    if "base" in item and "base_v4" not in item:
                        new_defaults.append("../base_v4")
                        warnings_list.append(f"Changed base reference to base_v4")
                    else:
                        new_defaults.append(item)
                else:
                    new_defaults.append(item)
            new["defaults"] = new_defaults
        
        return new, warnings_list
    
    def _migrate_generic(self, old: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Generic migration for unknown formats."""
        warnings_list = ["Using generic migration (version unknown)"]
        new = old.copy()
        new["config_version"] = "4.0.0"
        return new, warnings_list
    
    def migrate_file(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        overwrite: bool = False
    ) -> "MigrationResult":
        """
        Migrate configuration file from old version to v4.0.
        
        Args:
            input_path: Path to old configuration file
            output_path: Path for new configuration (default: input_path with _v4 suffix)
            overwrite: Whether to overwrite existing output file
        
        Returns:
            MigrationResult with success status and details
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            return MigrationResult(
                success=False,
                input_file=str(input_path),
                error=f"Input file not found: {input_path}"
            )
        
        # Default output path
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_v4{input_path.suffix}"
        else:
            output_path = Path(output_path)
        
        # Check if output exists
        if output_path.exists() and not overwrite:
            return MigrationResult(
                success=False,
                input_file=str(input_path),
                error=f"Output file already exists: {output_path}. Use overwrite=True to replace."
            )
        
        try:
            # Load old configuration
            with open(input_path, 'r', encoding='utf-8') as f:
                old_config = yaml.safe_load(f)
            
            # Detect version
            version = self.detect_version(old_config)
            
            # Check if already v4.0
            if version == "4.0":
                return MigrationResult(
                    success=True,
                    input_file=str(input_path),
                    output_file=str(input_path),
                    old_version="4.0.0",
                    new_version="4.0.0",
                    original_config=old_config,
                    migrated_config=old_config,
                    changes=[],
                    warnings=["Configuration is already v4.0 - no migration needed"],
                    migrated=False  # Explicitly no migration occurred
                )
            
            # Create backup if requested
            backup_path = None
            if self.backup:
                backup_path = input_path.parent / f"{input_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{input_path.suffix}"
                with open(backup_path, 'w', encoding='utf-8') as f:
                    yaml.dump(old_config, f, default_flow_style=False, allow_unicode=True)
            
            # Migrate
            new_config, warnings_list = self.migrate_dict(old_config)
            
            # Track changes
            changes = []
            if version in ["3.1", "3.2", "5.1"]:
                changes.append(f"Migrated from v{version} to v4.0.0")
                changes.append("Flattened configuration structure")
                changes.append("Renamed: lod_level → mode")
                changes.append("Renamed: features.feature_set → features.mode")
                if "optimizations" in new_config:
                    changes.append("Added optimizations section")
            
            # Save new configuration
            with open(output_path, 'w', encoding='utf-8') as f:
                # Add migration header
                f.write(f"# Migrated from v{version or 'unknown'} to v4.0 on {datetime.now().isoformat()}\n")
                f.write(f"# Original file: {input_path.name}\n")
                f.write("# Migration tool: IGN LiDAR HD ConfigMigrator v4.0\n\n")
                yaml.dump(new_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            return MigrationResult(
                success=True,
                input_file=str(input_path),
                output_file=str(output_path),
                old_version=version,
                new_version="4.0.0",
                original_config=old_config,
                migrated_config=new_config,
                changes=changes,
                warnings=warnings_list,
                backup_file=str(backup_path) if backup_path else None
            )
        
        except Exception as e:
            return MigrationResult(
                success=False,
                input_file=str(input_path),
                error=str(e)
            )


class MigrationResult:
    """Result of configuration migration."""
    
    def __init__(
        self,
        success: bool,
        input_file: str,
        output_file: Optional[str] = None,
        old_version: Optional[str] = None,
        new_version: str = "4.0.0",
        original_config: Optional[Dict[str, Any]] = None,
        migrated_config: Optional[Dict[str, Any]] = None,
        changes: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        backup_file: Optional[str] = None,
        error: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        migrated: Optional[bool] = None  # Explicit migrated flag
    ):
        self.success = success
        # migrated is True only if actual migration occurred (not if already v4.0)
        if migrated is None:
            # Infer: migrated=True if success and versions differ
            self.migrated = success and old_version != new_version and old_version != "4.0.0"
        else:
            self.migrated = migrated
        
        self.input_file = input_file
        self.output_file = output_file
        self.old_version = old_version
        self.new_version = new_version
        self.original_config = original_config or {}
        self.migrated_config = migrated_config or {}
        self.changes = changes or []
        self.warnings = warnings or []
        self.backup_file = backup_file
        self.error = error
        self.timestamp = timestamp or datetime.now()
    
    def __str__(self) -> str:
        if self.success:
            if self.migrated:
                msg = f"✓ Migration successful: {self.input_file} → {self.output_file}\n"
                msg += f"  Version: {self.old_version} → {self.new_version}\n"
                if self.backup_file:
                    msg += f"  Backup: {self.backup_file}\n"
                if self.changes:
                    msg += f"  Changes: {len(self.changes)} transformations\n"
            else:
                msg = f"✓ No migration needed: {self.input_file}\n"
                msg += f"  Already at version {self.new_version}\n"
            
            if self.warnings:
                msg += "  Warnings:\n"
                for w in self.warnings:
                    msg += f"    - {w}\n"
            return msg
        else:
            return f"✗ Migration failed: {self.input_file}\n  Error: {self.error}"
    
    def __repr__(self) -> str:
        return f"MigrationResult(success={self.success}, migrated={self.migrated}, {self.old_version}→{self.new_version}, changes={len(self.changes)})"

