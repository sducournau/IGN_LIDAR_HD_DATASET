"""Command-line tool to migrate configuration files to v4.0 format.

This tool helps users migrate from older configuration formats
(v3.1, v3.2, v5.1) to the new v4.0 unified flat format.

Usage:
    ign-lidar migrate-config old_config.yaml
    ign-lidar migrate-config old_config.yaml --output new_config.yaml
    ign-lidar migrate-config old_config.yaml --dry-run
    ign-lidar migrate-config configs/ --batch  # Migrate entire directory

Author: IGN LiDAR HD Team
Date: January 2025
Version: 4.0.0
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import yaml

from ign_lidar.config.migration import ConfigMigrator, MigrationResult


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file or directory (default: <input>_v4.0.yaml)",
)
@click.option(
    "--batch",
    is_flag=True,
    help="Batch migrate all YAML files in directory",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show changes without writing to file",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed migration information",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Overwrite existing output files",
)
def migrate_config(
    input_path: str, 
    output: Optional[str], 
    batch: bool,
    dry_run: bool, 
    verbose: bool,
    force: bool
) -> None:
    """
    Migrate configuration files to v4.0 format.

    This command converts configurations from v3.1, v3.2, or v5.1
    to the new v4.0 unified flat format with improved clarity.

    Supports single file or batch directory migration.

    Examples:
        \b
        # Preview migration
        ign-lidar migrate-config config.yaml --dry-run

        \b
        # Migrate single file
        ign-lidar migrate-config config.yaml

        \b
        # Migrate with custom output
        ign-lidar migrate-config config.yaml -o new_config.yaml

        \b
        # Batch migrate entire directory
        ign-lidar migrate-config configs/ --batch

        \b
        # Verbose output with details
        ign-lidar migrate-config config.yaml -v
    """
    input_path_obj = Path(input_path)
    
    # Initialize migrator
    migrator = ConfigMigrator()
    
    # Batch mode: migrate directory
    if batch:
        if not input_path_obj.is_dir():
            click.echo(f"❌ Error: --batch requires a directory, got: {input_path}", err=True)
            raise click.Abort()
        
        _batch_migrate(
            input_dir=input_path_obj,
            output_dir=Path(output) if output else None,
            migrator=migrator,
            dry_run=dry_run,
            verbose=verbose,
            force=force
        )
        return
    
    # Single file mode
    if not input_path_obj.is_file():
        click.echo(f"❌ Error: Input must be a file (use --batch for directories)", err=True)
        raise click.Abort()
    
    _migrate_single_file(
        input_file=input_path_obj,
        output_file=Path(output) if output else None,
        migrator=migrator,
        dry_run=dry_run,
        verbose=verbose,
        force=force
    )


def _migrate_single_file(
    input_file: Path,
    output_file: Optional[Path],
    migrator: ConfigMigrator,
    dry_run: bool,
    verbose: bool,
    force: bool
) -> None:
    """Migrate a single configuration file."""
    
    click.echo(f"📖 Loading configuration from: {input_file}")
    
    # Determine output path
    if output_file:
        output_path = output_file
    else:
        output_path = input_file.with_stem(input_file.stem + "_v4.0")
    
    # Migrate using ConfigMigrator (but don't write yet if dry-run)
    try:
        if dry_run:
            # Load and migrate in memory only
            with open(input_file, 'r') as f:
                old_config = yaml.safe_load(f)
            
            version = migrator.detect_version(old_config)
            
            if version == "4.0":
                click.echo(f"✅ Configuration is already in v4.0. No migration needed.")
                return
            
            click.echo(f"🔄 Migrating from v{version} → v4.0.0")
            new_config, warnings_list = migrator.migrate_dict(old_config)
            
            result = MigrationResult(
                success=True,
                input_file=str(input_file),
                output_file=str(output_path),
                old_version=version,
                new_version="4.0.0",
                original_config=old_config,
                migrated_config=new_config,
                changes=[f"Migrated from v{version} to v4.0.0"],
                warnings=warnings_list,
                migrated=True
            )
        else:
            result = migrator.migrate_file(input_file, output_path=output_path, overwrite=force)
    except Exception as e:
        click.echo(f"❌ Migration failed: {e}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        raise click.Abort()
    
    # Check if already v4.0
    if not result.migrated:
        click.echo(f"✅ Configuration is already in v{result.new_version}. No migration needed.")
        return
    
    # Display migration info (if not already displayed)
    if not dry_run:
        click.echo(f"🔄 Migrating from v{result.old_version} → v{result.new_version}")
    
    if result.warnings:
        click.echo("\n⚠️  Warnings:")
        for warning in result.warnings:
            click.echo(f"   • {warning}")
    
    # Statistics
    old_params = _count_params(result.original_config)
    new_params = _count_params(result.migrated_config)
    
    click.echo("\n" + "=" * 70)
    click.echo("📊 Migration Summary")
    click.echo("=" * 70)
    click.echo(f"  Version:           {result.old_version} → {result.new_version}")
    click.echo(f"  Parameters before: {old_params}")
    click.echo(f"  Parameters after:  {new_params}")
    click.echo(f"  Changes:           {len(result.changes)} transformations")
    click.echo("=" * 70)
    
    # Detailed changes
    if verbose and result.changes:
        click.echo("\n📋 Detailed Changes:")
        for change in result.changes:
            click.echo(f"   • {change}")
    
    # Show new config
    if dry_run or verbose:
        click.echo("\n📄 New Configuration:")
        click.echo("-" * 70)
        click.echo(yaml.dump(result.migrated_config, default_flow_style=False, sort_keys=False))
        click.echo("-" * 70)
    
    # Write to file
    if not dry_run:
        if output_file:
            output_path = output_file
        else:
            output_path = input_file.with_stem(input_file.stem + "_v4.0")
        
        # Check if output exists
        if output_path.exists() and not force:
            click.echo(f"\n⚠️  Output file exists: {output_path}")
            if not click.confirm("Overwrite?"):
                click.echo("❌ Aborted.")
                raise click.Abort()
        
        with open(output_path, "w") as f:
            f.write("# Configuration v4.0 - Migrated automatically\n")
            f.write(f"# Source: {input_file.name} (v{result.old_version})\n")
            f.write(f"# Migration date: {result.timestamp.isoformat()}\n\n")
            yaml.dump(result.migrated_config, f, default_flow_style=False, sort_keys=False)
        
        click.echo(f"\n✅ Migrated configuration written to: {output_path}")
        click.echo("\n💡 Next steps:")
        click.echo(f"   1. Review the new config: {output_path}")
        click.echo("   2. Test with: ign-lidar process --config {output_path.name}")
        click.echo("   3. Check documentation: docs/migration-guide-v4.md")
    else:
        click.echo("\n💡 Dry-run complete. Use without --dry-run to save changes.")


def _batch_migrate(
    input_dir: Path,
    output_dir: Optional[Path],
    migrator: ConfigMigrator,
    dry_run: bool,
    verbose: bool,
    force: bool
) -> None:
    """Batch migrate all YAML files in a directory."""
    
    # Find all YAML files
    yaml_files = list(input_dir.glob("*.yaml")) + list(input_dir.glob("*.yml"))
    
    if not yaml_files:
        click.echo(f"❌ No YAML files found in: {input_dir}", err=True)
        raise click.Abort()
    
    click.echo(f"📂 Found {len(yaml_files)} YAML file(s) in: {input_dir}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = input_dir / "migrated_v4.0"
    
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        click.echo(f"📁 Output directory: {output_dir}")
    
    # Track statistics
    successful = 0
    already_v4 = 0
    failed = 0
    
    click.echo("\n" + "=" * 70)
    click.echo("🔄 Batch Migration Progress")
    click.echo("=" * 70)
    
    # Migrate each file
    for i, yaml_file in enumerate(yaml_files, 1):
        click.echo(f"\n[{i}/{len(yaml_files)}] {yaml_file.name}")
        
        try:
            result = migrator.migrate_file(yaml_file)
            
            if not result.migrated:
                click.echo(f"  ✅ Already v{result.new_version}")
                already_v4 += 1
                continue
            
            click.echo(f"  🔄 v{result.old_version} → v{result.new_version}")
            
            if result.warnings and verbose:
                for warning in result.warnings[:3]:  # Limit warnings in batch mode
                    click.echo(f"     ⚠️  {warning}")
            
            # Write output
            if not dry_run:
                output_file = output_dir / yaml_file.name
                
                if output_file.exists() and not force:
                    click.echo(f"     ⚠️  Skipped (exists): {output_file.name}")
                    continue
                
                with open(output_file, "w") as f:
                    f.write("# Configuration v4.0 - Migrated automatically\n")
                    f.write(f"# Source: {yaml_file.name} (v{result.old_version})\n\n")
                    yaml.dump(result.migrated_config, f, default_flow_style=False, sort_keys=False)
                
                click.echo(f"  ✅ Saved: {output_file.name}")
            
            successful += 1
            
        except Exception as e:
            click.echo(f"  ❌ Failed: {e}", err=True)
            failed += 1
            if verbose:
                import traceback
                click.echo(traceback.format_exc(), err=True)
    
    # Summary
    click.echo("\n" + "=" * 70)
    click.echo("📊 Batch Migration Summary")
    click.echo("=" * 70)
    click.echo(f"  Total files:       {len(yaml_files)}")
    click.echo(f"  Successfully migrated: {successful}")
    click.echo(f"  Already v4.0:      {already_v4}")
    click.echo(f"  Failed:            {failed}")
    click.echo("=" * 70)
    
    if not dry_run and successful > 0:
        click.echo(f"\n✅ Migrated files saved to: {output_dir}")
    elif dry_run:
        click.echo("\n💡 Dry-run complete. Use without --dry-run to save changes.")


def _count_params(config: Dict[str, Any], prefix: str = "") -> int:
    """Count total number of parameters in config (recursive)."""
    count = 0
    for key, value in config.items():
        if isinstance(value, dict):
            count += _count_params(value, prefix=f"{prefix}{key}.")
        elif isinstance(value, list):
            count += 1  # Count list as single param
        else:
            count += 1
    return count


# Add to CLI
def register_command(cli_group):
    """Register migrate-config command with CLI."""
    cli_group.add_command(migrate_config)


if __name__ == "__main__":
    migrate_config()

