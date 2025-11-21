"""Command-line tool to migrate old configuration files to v3.2 format.

This tool helps users migrate from the old dual-schema configuration
(ProcessorConfig + FeaturesConfig) to the new v3.2 Config format.

Usage:
    ign-lidar migrate-config old_config.yaml
    ign-lidar migrate-config old_config.yaml --output new_config.yaml
    ign-lidar migrate-config old_config.yaml --dry-run

Author: IGN LiDAR HD Team
Date: October 25, 2025
Version: 3.2.0
"""

from pathlib import Path
from typing import Any, Dict

import click
import yaml


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file (default: input_v3.2.yaml)",
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
def migrate_config(input_file: str, output: str, dry_run: bool, verbose: bool) -> None:
    """
    Migrate old configuration format to v3.2 format.

    This command converts old v3.0-3.1 configurations (with separate
    processor and features sections) to the new v3.2 format.

    Examples:
        \b
        # Preview migration
        ign-lidar migrate-config config.yaml --dry-run

        \b
        # Migrate and save
        ign-lidar migrate-config config.yaml

        \b
        # Migrate with custom output name
        ign-lidar migrate-config config.yaml -o new_config.yaml
    """
    input_path = Path(input_file)

    # Load old config
    click.echo(f"📖 Loading configuration from: {input_path}")
    with open(input_path) as f:
        old_config = yaml.safe_load(f)

    # Check if already in new format
    if _is_new_format(old_config):
        click.echo("✅ Configuration is already in v3.2 format. No migration needed.")
        return

    # Convert to new format
    click.echo("🔄 Converting to v3.2 format...")
    new_config = _convert_config(old_config, verbose)

    # Calculate statistics
    old_params = _count_params(old_config)
    new_params = _count_params(new_config)
    simplification = (1 - new_params / old_params) * 100 if old_params > 0 else 0

    # Display summary
    click.echo("\n" + "=" * 60)
    click.echo("📊 Migration Summary")
    click.echo("=" * 60)
    click.echo(f"  Parameters before: {old_params}")
    click.echo(f"  Parameters after:  {new_params}")
    click.echo(f"  Simplified:        {simplification:.1f}%")
    click.echo("=" * 60)

    if verbose:
        click.echo("\n📋 Detailed Changes:")
        _show_detailed_changes(old_config, new_config)

    # Show new config
    if dry_run or verbose:
        click.echo("\n📄 New Configuration:")
        click.echo("-" * 60)
        click.echo(yaml.dump(new_config, default_flow_style=False, sort_keys=False))
        click.echo("-" * 60)

    # Write to file
    if not dry_run:
        if output:
            output_path = Path(output)
        else:
            output_path = input_path.with_stem(input_path.stem + "_v3.2")

        with open(output_path, "w") as f:
            yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)

        click.echo(f"\n✅ Migrated configuration written to: {output_path}")
        click.echo("\n💡 Next steps:")
        click.echo(f"   1. Review the new config: {output_path}")
        click.echo("   2. Test with: ign-lidar process <config>")
        click.echo("   3. Update your scripts to use the new format")
    else:
        click.echo("\n💡 Dry-run complete. Use without --dry-run to save changes.")


def _is_new_format(config: Dict[str, Any]) -> bool:
    """
    Check if config is already in new v3.2 format.

    New format has:
    - Top-level 'mode' instead of 'processor.lod_level'
    - 'features.feature_set' instead of 'features.mode'
    """
    has_top_level_mode = "mode" in config
    has_processor_section = "processor" in config
    has_old_features = (
        "features" in config
        and isinstance(config.get("features"), dict)
        and "mode" in config.get("features", {})
    )

    # New format has top-level mode and no processor section
    return has_top_level_mode and not has_processor_section and not has_old_features


def _convert_config(
    old_config: Dict[str, Any], verbose: bool = False
) -> Dict[str, Any]:
    """
    Convert old v3.1 config format to v3.2 format.

    Args:
        old_config: Old configuration dictionary
        verbose: Show detailed conversion info

    Returns:
        New configuration dictionary
    """
    new_config = {}

    # Top-level fields (copy directly)
    for key in ["input_dir", "output_dir"]:
        if key in old_config:
            new_config[key] = old_config[key]
            if verbose:
                click.echo(f"  ✓ {key}: {old_config[key]}")

    # Processor section → top-level
    if "processor" in old_config:
        proc = old_config["processor"]

        # Map lod_level to mode
        lod_level = proc.get("lod_level", "LOD2")
        new_config["mode"] = lod_level.lower()

        # Copy processor fields to top level
        field_map = {
            "use_gpu": "use_gpu",
            "num_workers": "num_workers",
            "patch_size": "patch_size",
            "num_points": "num_points",
            "patch_overlap": "patch_overlap",
            "architecture": "architecture",
            "processing_mode": "processing_mode",
        }

        for old_key, new_key in field_map.items():
            if old_key in proc:
                new_config[new_key] = proc[old_key]
                if verbose:
                    click.echo(f"  ✓ processor.{old_key} → {new_key}")

    # Features section → features
    if "features" in old_config and isinstance(old_config["features"], dict):
        old_features = old_config["features"]

        # Map old 'mode' to new 'feature_set'
        old_mode = old_features.get("mode", "full")
        feature_set_map = {
            "minimal": "minimal",
            "lod2": "standard",
            "lod3": "full",
            "asprs_classes": "standard",
            "full": "full",
            "custom": "standard",
        }
        feature_set = feature_set_map.get(old_mode, "standard")

        new_config["features"] = {
            "feature_set": feature_set,
        }

        # Copy feature parameters
        feature_field_map = {
            "k_neighbors": "k_neighbors",
            "search_radius": "search_radius",
            "use_rgb": "use_rgb",
            "use_infrared": "use_nir",  # Renamed!
            "compute_ndvi": "compute_ndvi",
            "multi_scale_computation": "multi_scale",  # Renamed!
        }

        for old_key, new_key in feature_field_map.items():
            if old_key in old_features:
                new_config["features"][new_key] = old_features[old_key]
                if verbose:
                    if old_key != new_key:
                        click.echo(
                            f"  ✓ features.{old_key} → features.{new_key} (renamed)"
                        )
                    else:
                        click.echo(f"  ✓ features.{old_key}")

        # Handle multi-scale scales
        if "scales" in old_features:
            new_config["features"]["scales"] = old_features["scales"]
            if verbose:
                click.echo(f"  ✓ features.scales (preserved)")

    # Advanced options → advanced
    advanced = {}

    if "preprocessing" in old_config:
        advanced["preprocessing"] = old_config["preprocessing"]
        if verbose:
            click.echo(f"  ✓ preprocessing → advanced.preprocessing")

    if "data_sources" in old_config:
        advanced["ground_truth"] = old_config["data_sources"]
        if verbose:
            click.echo(f"  ✓ data_sources → advanced.ground_truth")

    if "processor" in old_config:
        if "reclassification" in old_config["processor"]:
            advanced["reclassification"] = old_config["processor"]["reclassification"]
            if verbose:
                click.echo(
                    f"  ✓ processor.reclassification → advanced.reclassification"
                )

    if advanced:
        new_config["advanced"] = advanced

    return new_config


def _count_params(config: Dict[str, Any], prefix: str = "") -> int:
    """
    Count total number of parameters in config (recursive).
    """
    count = 0
    for key, value in config.items():
        if isinstance(value, dict):
            count += _count_params(value, prefix=f"{prefix}{key}.")
        else:
            count += 1
    return count


def _flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """
    Flatten nested dictionary for comparison.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _show_detailed_changes(
    old_config: Dict[str, Any], new_config: Dict[str, Any]
) -> None:
    """
    Show detailed parameter-by-parameter changes.
    """
    old_flat = _flatten_dict(old_config)
    new_flat = _flatten_dict(new_config)

    # Parameters removed (in old but not new)
    removed = set(old_flat.keys()) - set(new_flat.keys())
    if removed:
        click.echo("\n  ⚠️  Parameters removed (now defaults or moved):")
        for param in sorted(removed):
            click.echo(f"     - {param}")

    # Parameters added (in new but not old)
    added = set(new_flat.keys()) - set(old_flat.keys())
    if added:
        click.echo("\n  ✨ New parameters:")
        for param in sorted(added):
            click.echo(f"     + {param}: {new_flat[param]}")

    # Parameters renamed/moved
    click.echo("\n  🔀 Key renames:")
    renames = [
        ("processor.lod_level", "mode"),
        ("features.mode", "features.feature_set"),
        ("features.use_infrared", "features.use_nir"),
        ("features.multi_scale_computation", "features.multi_scale"),
    ]
    for old_name, new_name in renames:
        if old_name in old_flat and new_name in new_flat:
            click.echo(f"     {old_name} → {new_name}")


# Add to CLI
def register_command(cli_group):
    """Register migrate-config command with CLI."""
    cli_group.add_command(migrate_config)


if __name__ == "__main__":
    migrate_config()
