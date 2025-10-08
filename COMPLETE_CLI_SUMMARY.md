# IGN LiDAR HD CLI Commands Summary

## ✅ Complete Command Set Available

The `ign-lidar-hd` CLI now includes all essential commands for LiDAR data processing, downloading, verification, and conversion.

### 🚀 Available Commands

#### 1. **`process`** - Core LiDAR Processing

```bash
ign-lidar-hd process INPUT_DIR OUTPUT_DIR [OPTIONS]
```

**Purpose**: Process LiDAR data with feature extraction  
**Key Options**:

- `--patch-size FLOAT` - Patch size in meters
- `--num-points INT` - Points per patch
- `--use-gpu` - Enable GPU processing
- `--lod-level TEXT` - LOD level (LOD2/LOD3)
- `--experiment TEXT` - Use experiment configuration

#### 2. **`download`** - Download LiDAR Tiles

```bash
ign-lidar-hd download OUTPUT_DIR [OPTIONS]
```

**Purpose**: Download IGN LiDAR HD tiles by location, position, or bounding box  
**Key Options**:

- `-b, --bbox XMIN YMIN XMAX YMAX` - Bounding box (Lambert93)
- `-p, --position X Y` - Position coordinates (Lambert93)
- `-r, --radius FLOAT` - Radius around position (meters)
- `-l, --location TEXT` - Strategic location name
- `--list-locations` - List available strategic locations
- `-j, --max-concurrent INT` - Concurrent downloads
- `-f, --force` - Force download existing files

#### 3. **`verify`** - Data Quality Verification

```bash
ign-lidar-hd verify INPUT_PATH [OPTIONS]
```

**Purpose**: Verify LiDAR data quality and features  
**Key Options**:

- `-o, --output PATH` - Output verification report
- `-d, --detailed` - Show detailed statistics
- `-f, --features-only` - Verify features only

#### 4. **`batch-convert`** - Batch QGIS Conversion

```bash
ign-lidar-hd batch-convert INPUT_DIR [OPTIONS]
```

**Purpose**: Batch convert LAZ files for QGIS compatibility  
**Key Options**:

- `-o, --output PATH` - Output directory
- `-b, --batch-size INT` - Parallel processing batch size
- `-f, --force` - Overwrite existing files

#### 5. **`info`** - System Information

```bash
ign-lidar-hd info [OPTIONS]
```

**Purpose**: Show package information, version, and dependencies  
**Key Options**:

- `-v, --version` - Show version
- `-d, --dependencies` - Show dependency status
- `-c, --config` - Show configuration paths

#### 6. **`config-info`** - Configuration Information

```bash
ign-lidar-hd config-info
```

**Purpose**: Show available configuration options and experiments

### 📊 Strategic Locations (75+ Available)

The download command includes 75+ predefined strategic locations in France:

**Historic Sites**: Versailles, Carcassonne, Chambord, Fontainebleau...  
**City Centers**: Paris districts, Lyon, Marseille, Strasbourg...  
**Villages**: Provence (Gordes, Roussillon), Alsace (Riquewihr), Brittany...  
**Mountain Areas**: Chamonix, Megève, Cauterets...  
**Infrastructure**: Airports (CDG, Orly), Ports, Train stations...  
**Universities**: Saclay campus, Cité Universitaire...

### 🔧 Global Options

All commands support:

- `-v, --verbose` - Verbose output
- `--config-dir PATH` - Custom configuration directory
- `--help` - Command-specific help

### 📋 Usage Examples

#### Download by Location

```bash
ign-lidar-hd download /data/downloads --location versailles_chateau
```

#### Download by Position

```bash
ign-lidar-hd download /data/downloads --position 648545 6862130 --radius 2000
```

#### Download by Bounding Box

```bash
ign-lidar-hd download /data/downloads --bbox 648000 6862000 649000 6863000
```

#### Process Downloaded Data

```bash
ign-lidar-hd process /data/downloads /data/processed --patch-size 100 --num-points 2048
```

#### Verify Data Quality

```bash
ign-lidar-hd verify /data/processed --detailed --output verification_report.txt
```

#### Batch Convert for QGIS

```bash
ign-lidar-hd batch-convert /data/processed --output /data/qgis_ready
```

#### System Information

```bash
ign-lidar-hd info --version --dependencies
```

### ✅ Complete Workflow

```bash
# 1. Download data for Paris Defense area
ign-lidar-hd download /data/raw --location paris_defense

# 2. Process the downloaded data
ign-lidar-hd process /data/raw /data/processed --patch-size 50 --num-points 1000

# 3. Verify processing quality
ign-lidar-hd verify /data/processed --detailed

# 4. Convert for QGIS visualization
ign-lidar-hd batch-convert /data/processed --output /data/qgis

# 5. Check system status
ign-lidar-hd info --dependencies
```

## 🎯 All Essential Commands Added Successfully

The CLI now provides comprehensive functionality for:

- ✅ **Data Download**: Flexible tile downloading with 75+ locations
- ✅ **Data Processing**: Feature extraction with configurable options
- ✅ **Quality Verification**: Comprehensive data quality checks
- ✅ **Format Conversion**: QGIS compatibility (batch processing)
- ✅ **System Management**: Version, dependencies, configuration info
- ✅ **User Experience**: Detailed help, verbose output, progress tracking

**Total Commands**: 6 comprehensive commands covering the complete LiDAR processing workflow!
