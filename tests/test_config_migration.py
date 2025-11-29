"""
Test suite for configuration migration (v3.x/v5.1 → v4.0)

Tests:
- Version detection (v3.1, v3.2, v5.1, v4.0)
- Dictionary migration
- File migration
- CLI command
- Edge cases

Author: IGN LiDAR HD Team
Date: November 2025
Version: 4.0.0
"""

import pytest
from pathlib import Path
import yaml
import tempfile
from unittest.mock import MagicMock

from ign_lidar.config.migration import ConfigMigrator, MigrationResult


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_v5_1_config():
    """Sample v5.1 configuration (processor.lod_level structure)."""
    return {
        "input_dir": "/data/tiles",
        "output_dir": "/data/output",
        "processor": {
            "lod_level": "LOD2",
            "use_gpu": True,
            "num_workers": 0,
            "patch_size": 50.0,
            "num_points": 16384,
            "processing_mode": "patches_only",
        },
        "features": {
            "mode": "lod2",
            "k_neighbors": 30,
            "search_radius": 2.5,
            "use_rgb": True,
            "use_infrared": True,
        }
    }


@pytest.fixture
def sample_v3_2_config():
    """Sample v3.2 configuration (top-level mode, features.feature_set)."""
    return {
        "config_version": "3.2.0",
        "input_dir": "/data/tiles",
        "output_dir": "/data/output",
        "mode": "lod2",
        "use_gpu": True,
        "num_workers": 0,
        "patch_size": 50.0,
        "features": {
            "feature_set": "standard",
            "k_neighbors": 30,
            "search_radius": 2.5,
        }
    }


@pytest.fixture
def sample_v4_0_config():
    """Sample v4.0 configuration (already migrated)."""
    return {
        "config_version": "4.0.0",
        "config_name": "test_config",
        "input_dir": "/data/tiles",
        "output_dir": "/data/output",
        "mode": "lod2",
        "use_gpu": True,
        "features": {
            "mode": "standard",
            "k_neighbors": 30,
        }
    }


@pytest.fixture
def migrator():
    """ConfigMigrator instance."""
    return ConfigMigrator(backup=False)


# ============================================================================
# Test Version Detection
# ============================================================================

def test_detect_version_v5_1(migrator, sample_v5_1_config):
    """Test detection of v5.1 config (processor.lod_level)."""
    version = migrator.detect_version(sample_v5_1_config)
    assert version == "5.1", f"Expected v5.1, got {version}"


def test_detect_version_v3_2(migrator, sample_v3_2_config):
    """Test detection of v3.2 config (explicit version)."""
    version = migrator.detect_version(sample_v3_2_config)
    assert version == "3.2", f"Expected v3.2, got {version}"


def test_detect_version_v4_0(migrator, sample_v4_0_config):
    """Test detection of v4.0 config (already migrated)."""
    version = migrator.detect_version(sample_v4_0_config)
    assert version == "4.0", f"Expected v4.0, got {version}"


def test_detect_version_unknown(migrator):
    """Test detection of unknown version."""
    unknown_config = {"some_key": "some_value"}
    version = migrator.detect_version(unknown_config)
    assert version is None, "Expected None for unknown config"


# ============================================================================
# Test Dictionary Migration
# ============================================================================

def test_migrate_dict_v5_1_to_v4_0(migrator, sample_v5_1_config):
    """Test migrating v5.1 config dict to v4.0."""
    new_config, warnings = migrator.migrate_dict(sample_v5_1_config)
    
    # Check version
    assert new_config["config_version"] == "4.0.0"
    
    # Check flat structure
    assert "mode" in new_config
    assert new_config["mode"] == "lod2"
    
    # Check processor fields moved to top level
    assert new_config["use_gpu"] is True
    assert new_config["num_workers"] == 0
    assert new_config["patch_size"] == 50.0
    
    # Check features renamed
    assert "features" in new_config
    assert new_config["features"]["mode"] == "standard"  # lod2 → standard
    assert new_config["features"]["use_nir"] is True  # use_infrared → use_nir
    
    # Check no processor section
    assert "processor" not in new_config


def test_migrate_dict_v3_2_to_v4_0(migrator, sample_v3_2_config):
    """Test migrating v3.2 config dict to v4.0."""
    new_config, warnings = migrator.migrate_dict(sample_v3_2_config)
    
    # Check version
    assert new_config["config_version"] == "4.0.0"
    
    # Check features.feature_set → features.mode
    assert new_config["features"]["mode"] == "standard"


def test_migrate_dict_already_v4_0(migrator, sample_v4_0_config):
    """Test migrating already v4.0 config (no-op)."""
    new_config, warnings = migrator.migrate_dict(sample_v4_0_config)
    
    # Should return same config with warning
    assert new_config == sample_v4_0_config
    assert len(warnings) > 0
    assert "already v4.0" in warnings[0].lower()


def test_migrate_dict_preserves_input_output(migrator, sample_v5_1_config):
    """Test that input/output dirs are preserved."""
    new_config, warnings = migrator.migrate_dict(sample_v5_1_config)
    
    assert new_config["input_dir"] == "/data/tiles"
    assert new_config["output_dir"] == "/data/output"


def test_migrate_dict_lod_level_mapping(migrator):
    """Test lod_level → mode mapping."""
    config = {
        "processor": {
            "lod_level": "LOD3"
        }
    }
    
    new_config, _ = migrator.migrate_dict(config)
    assert new_config["mode"] == "lod3", "LOD3 should map to lod3"


# ============================================================================
# Test File Migration
# ============================================================================

def test_migrate_file_success(migrator, sample_v5_1_config):
    """Test successful file migration."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_v5_1_config, f)
        input_path = f.name
    
    try:
        output_path = input_path.replace('.yaml', '_v4.0.yaml')
        
        result = migrator.migrate_file(input_path, output_path=output_path, overwrite=True)
        
        assert result.success is True
        assert result.migrated is True
        assert result.old_version == "5.1"
        assert result.new_version == "4.0.0"
        assert len(result.changes) > 0
        
        # Check output file exists
        assert Path(output_path).exists()
        
        # Check output content
        with open(output_path) as f:
            migrated = yaml.safe_load(f)
        
        assert migrated["config_version"] == "4.0.0"
        assert "mode" in migrated
        
    finally:
        # Cleanup
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


def test_migrate_file_already_v4_0(migrator, sample_v4_0_config):
    """Test migration of already v4.0 file (should be no-op)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_v4_0_config, f)
        input_path = f.name
    
    try:
        result = migrator.migrate_file(input_path)
        
        assert result.success is True
        assert result.migrated is False  # No migration needed
        assert result.old_version == "4.0.0"
        assert result.new_version == "4.0.0"
        assert len(result.warnings) > 0
        
    finally:
        Path(input_path).unlink(missing_ok=True)


def test_migrate_file_not_found(migrator):
    """Test migration with non-existent input file."""
    result = migrator.migrate_file("/nonexistent/file.yaml")
    
    assert result.success is False
    assert "not found" in result.error.lower()


def test_migrate_file_output_exists_no_overwrite(migrator, sample_v5_1_config):
    """Test migration when output exists and overwrite=False."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_v5_1_config, f)
        input_path = f.name
    
    output_path = input_path.replace('.yaml', '_v4.0.yaml')
    
    try:
        # Create output file
        Path(output_path).touch()
        
        result = migrator.migrate_file(input_path, output_path=output_path, overwrite=False)
        
        assert result.success is False
        assert "already exists" in result.error.lower()
        
    finally:
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


# ============================================================================
# Test MigrationResult
# ============================================================================

def test_migration_result_success():
    """Test MigrationResult for successful migration."""
    result = MigrationResult(
        success=True,
        input_file="test.yaml",
        output_file="test_v4.0.yaml",
        old_version="5.1",
        new_version="4.0.0",
        changes=["Migrated from v5.1"],
        migrated=True
    )
    
    assert result.success is True
    assert result.migrated is True
    assert result.old_version == "5.1"
    assert result.new_version == "4.0.0"
    assert len(result.changes) == 1


def test_migration_result_already_v4():
    """Test MigrationResult for already v4.0 config."""
    result = MigrationResult(
        success=True,
        input_file="test.yaml",
        old_version="4.0.0",
        new_version="4.0.0",
        migrated=False
    )
    
    assert result.success is True
    assert result.migrated is False


def test_migration_result_failure():
    """Test MigrationResult for failed migration."""
    result = MigrationResult(
        success=False,
        input_file="test.yaml",
        error="File not found"
    )
    
    assert result.success is False
    assert result.error == "File not found"


def test_migration_result_repr():
    """Test MigrationResult string representations."""
    result = MigrationResult(
        success=True,
        input_file="test.yaml",
        old_version="5.1",
        new_version="4.0.0",
        changes=["change1", "change2"],
        migrated=True
    )
    
    str_repr = str(result)
    assert "5.1" in str_repr
    assert "4.0.0" in str_repr
    
    repr_str = repr(result)
    assert "MigrationResult" in repr_str
    assert "5.1" in repr_str


# ============================================================================
# Test Edge Cases
# ============================================================================

def test_migrate_empty_config(migrator):
    """Test migration with empty config."""
    empty_config = {}
    new_config, warnings = migrator.migrate_dict(empty_config)
    
    # Should still produce valid v4.0 config
    assert new_config["config_version"] == "4.0.0"
    assert len(warnings) > 0  # Should warn about unknown version


def test_migrate_partial_config(migrator):
    """Test migration with partial config (only some fields)."""
    partial_config = {
        "processor": {
            "lod_level": "LOD2"
        }
    }
    
    new_config, warnings = migrator.migrate_dict(partial_config)
    
    # Should have mode field
    assert new_config["mode"] == "lod2"
    assert new_config["config_version"] == "4.0.0"


def test_migrate_with_extra_fields(migrator, sample_v5_1_config):
    """Test migration preserves optimizations section."""
    sample_v5_1_config["custom_field"] = "custom_value"
    
    # Add processor.async_io (should become optimizations.async_io)
    sample_v5_1_config["processor"]["async_io"] = True
    sample_v5_1_config["processor"]["batch_processing"] = True
    
    new_config, _ = migrator.migrate_dict(sample_v5_1_config)
    
    # Should have optimizations section
    assert "optimizations" in new_config
    assert new_config["optimizations"]["async_io"] is True
    assert new_config["optimizations"]["batch_processing"] is True


# ============================================================================
# Test Parameter Mappings
# ============================================================================

def test_feature_mode_mapping(migrator):
    """Test features.mode mapping (lod2 → standard, etc.)."""
    test_cases = [
        ("lod2", "standard"),
        ("lod3", "full"),
        ("minimal", "minimal"),
        ("asprs_classes", "full"),
    ]
    
    for old_mode, expected_new_mode in test_cases:
        config = {
            "processor": {"lod_level": "LOD2"},
            "features": {"mode": old_mode}
        }
        
        new_config, _ = migrator.migrate_dict(config)
        
        assert new_config["features"]["mode"] == expected_new_mode, \
            f"Expected {old_mode} → {expected_new_mode}, got {new_config['features']['mode']}"


def test_infrared_to_nir_rename(migrator):
    """Test use_infrared → use_nir rename."""
    config = {
        "processor": {"lod_level": "LOD2"},
        "features": {
            "use_infrared": True,
            "compute_ndvi": True,
        }
    }
    
    new_config, _ = migrator.migrate_dict(config)
    
    assert "use_nir" in new_config["features"]
    assert new_config["features"]["use_nir"] is True
    assert "use_infrared" not in new_config["features"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
