# Quick Reference: Tile Stitching Configurations

## TL;DR - Which Config Should I Use?

| Scenario                             | Config                    | Command                                                     |
| ------------------------------------ | ------------------------- | ----------------------------------------------------------- |
| **No boundary processing** (fastest) | `stitching=disabled`      | `stitching=disabled`                                        |
| **Local tiles only** (offline)       | `stitching=enhanced`      | `stitching=enhanced`                                        |
| **Auto-download neighbors**          | `stitching=auto_download` | `stitching=auto_download`                                   |
| **Custom settings**                  | Override as needed        | `stitching=enhanced stitching.auto_download_neighbors=true` |

## One-Line Commands

### Without Stitching (Fastest)

```bash
python -m ign_lidar.cli.process input_dir=/path output_dir=/path stitching=disabled
```

### With Local Neighbors Only

```bash
python -m ign_lidar.cli.process input_dir=/path output_dir=/path stitching=enhanced
```

### With Auto-Download

```bash
python -m ign_lidar.cli.process input_dir=/path output_dir=/path stitching=auto_download
```

### Enable Download on Any Config

```bash
python -m ign_lidar.cli.process input_dir=/path output_dir=/path \
  stitching=enhanced \
  stitching.auto_download_neighbors=true
```

## Key Settings

| Setting                   | Values     | Default | Description                |
| ------------------------- | ---------- | ------- | -------------------------- |
| `enabled`                 | true/false | false   | Enable tile stitching      |
| `buffer_size`             | 5.0-25.0   | 15.0    | Buffer zone width (meters) |
| `auto_detect_neighbors`   | true/false | true    | Auto-detect adjacent tiles |
| `auto_download_neighbors` | true/false | false   | Download missing neighbors |
| `validate_tiles`          | true/false | true    | Validate tile integrity    |

## Config Hierarchy

```
config.yaml (root)
└── defaults:
    └── stitching: enhanced  # Choose one:
        ├── disabled.yaml          # No stitching
        ├── enabled.yaml           # Basic stitching
        ├── enhanced.yaml          # Advanced (default)
        ├── advanced.yaml          # Research-grade
        └── auto_download.yaml     # With auto-download
```

## When to Use Auto-Download

### ✅ Use Auto-Download When:

- Processing isolated tiles
- Don't have neighbors pre-downloaded
- Want automatic recovery from corrupted tiles
- Exploring new areas
- Internet connection available

### ❌ Don't Use Auto-Download When:

- Processing large batches (pre-download instead)
- Offline processing required
- Network is unreliable
- Want fastest possible processing

## Performance Impact

| Configuration    | First Tile | Subsequent Tiles | Network  | Storage        |
| ---------------- | ---------- | ---------------- | -------- | -------------- |
| Disabled         | 30s        | 30s              | None     | ~150 MB/tile   |
| Enhanced (local) | 45s        | 45s              | None     | ~150 MB/tile   |
| Auto-download    | 15 min     | 1 min            | Required | ~1.5 GB/region |

## Troubleshooting

### Downloads not working?

```bash
# Check these settings:
stitching.enabled=true
stitching.auto_detect_neighbors=true
stitching.auto_download_neighbors=true
processor.use_stitching=true
```

### Tiles corrupted?

```bash
# System will auto-detect and re-download
# Or manually delete:
rm /path/to/tile.laz
# Then re-run
```

### Want more logs?

```bash
stitching.verbose_logging=true log_level=DEBUG
```

## Full Documentation

- **Feature Guide**: `AUTO_DOWNLOAD_NEIGHBORS.md`
- **Implementation**: `TILE_STITCHING_SUMMARY.md`
- **Config Guide**: `ign_lidar/configs/README.md`
