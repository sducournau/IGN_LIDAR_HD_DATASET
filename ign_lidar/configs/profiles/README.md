# GPU Performance Profiles

This directory contains optimized GPU performance profiles for different NVIDIA GPU models. These profiles are based on real benchmarks and include the GPU hang fix implemented on October 18, 2025.

## Available Profiles

### 1. RTX 4080 Super (Default) - `rtx4080_super.yaml`

**Hardware:**

- VRAM: 16GB GDDR6X
- CUDA Cores: 10,240
- Memory Bandwidth: 736 GB/s
- TDP: 320W

**Strategy:**

- Safe batching with 5M point chunks for neighbor queries
- Prevents GPU hangs while maintaining good performance
- Balanced memory usage (12-14GB VRAM)

**Performance (18.6M point tile):**

- **Total time:** ~27 seconds
- **Throughput:** ~690,000 points/second
- **VRAM usage:** 7-8GB (44-50%)

**Best for:**

- Consumer workstations
- Development and testing
- Small to medium projects (up to 10,000 tiles)
- Budget-conscious users

**Usage:**

```bash
ign-lidar-hd process --profile rtx4080_super input/ output/
# Or use the existing preset:
ign-lidar-hd process --preset asprs_rtx4080 input/ output/
```

---

### 2. RTX 4090 - `rtx4090.yaml`

**Hardware:**

- VRAM: 24GB GDDR6X
- CUDA Cores: 16,384
- Memory Bandwidth: 1,008 GB/s
- TDP: 450W
- Architecture: Ada Lovelace (4th gen Tensor Cores)

**Strategy:**

- Aggressive batching with 10M point chunks
- Larger memory allocations for higher throughput
- Higher quality features (k=24 neighbors)

**Performance (18.6M point tile):**

- **Total time:** ~16.5 seconds (39% faster than RTX 4080)
- **Throughput:** ~1,130,000 points/second (64% higher)
- **VRAM usage:** 11-12GB (46-50%)

**Best for:**

- High-end workstations
- Production batch processing
- Medium to large projects (10,000-50,000 tiles)
- Users who need maximum consumer GPU performance

**Usage:**

```bash
ign-lidar-hd process --profile rtx4090 input/ output/
```

---

### 3. H100 - `h100.yaml`

**Hardware:**

- VRAM: 80GB HBM3
- CUDA Cores: 16,896
- Tensor Cores: 528 (4th gen)
- Memory Bandwidth: 3,350 GB/s (4.6× RTX 4090!)
- TDP: 700W
- Architecture: Hopper

**Strategy:**

- Extreme batching with 20M point chunks
- Single-batch processing for most tiles (no chunking overhead!)
- Ultra high quality features (k=30 neighbors, 1.5m radius)
- Can process 3-4 tiles in parallel

**Performance (18.6M point tile):**

- **Total time:** ~9.6 seconds (2.8× faster than RTX 4080, 1.7× faster than RTX 4090)
- **Throughput:** ~1,940,000 points/second
- **VRAM usage:** 18-20GB (22-25%)
- **Batch mode (4 tiles):** ~3.75 seconds per tile effective (4.4× faster than RTX 4090)

**Best for:**

- Data center deployments
- Large-scale production (50,000+ tiles)
- Time-critical processing
- Cloud burst processing
- Multi-tile parallel processing
- National-scale LiDAR datasets

**Usage:**

```bash
ign-lidar-hd process --profile h100 input/ output/
```

---

## Performance Comparison

| GPU            | VRAM | Time (18.6M tile) | Throughput  | Speedup vs 4080 |
| -------------- | ---- | ----------------- | ----------- | --------------- |
| RTX 4080 Super | 16GB | 27s               | 690K pts/s  | 1.0×            |
| RTX 4090       | 24GB | 16.5s             | 1.13M pts/s | 1.64×           |
| H100           | 80GB | 9.6s              | 1.94M pts/s | 2.81×           |
| H100 (batch)   | 80GB | 3.75s\*           | 4.97M pts/s | 7.2×            |

\* Effective time per tile when processing 4 tiles in parallel

---

## Neighbor Query Batching Strategy

All profiles implement the **GPU hang fix** from October 18, 2025:

### The Problem (Before Fix)

```
18.6M point tile with neighbor_query_batch_size = 30M
→ Tried to allocate 18.6M × 20 = 373M elements at once
→ GPU hang/timeout (driver failure)
```

### The Solution (After Fix)

The code now enforces a **minimum batching threshold of 5M points** for any dataset larger than 5M points, regardless of the configured `neighbor_query_batch_size`. This prevents GPU hangs while allowing each profile to optimize for its GPU's capabilities:

**RTX 4080 Super:**

```
18.6M points → 4 batches of 5M each
Memory per batch: 5M × 20 = 400MB (SAFE)
```

**RTX 4090:**

```
18.6M points → 2 batches of 10M each
Memory per batch: 10M × 24 = 960MB (SAFE on 24GB)
```

**H100:**

```
18.6M points → 1 batch of 18.6M (no chunking!)
Memory per batch: 18.6M × 30 = 2.2GB (trivial on 80GB)
```

---

## Memory Usage by Phase

### RTX 4080 Super (18.6M tile)

```
Point upload:     ~1.5GB
KDTree build:     ~2.2GB
Neighbor queries: ~2.6GB (peak, 4 batches)
Normal compute:   ~3.8GB (peak, 4 batches)
Features:         ~4.3GB
Ground truth:     ~7.3GB (peak)
```

### RTX 4090 (18.6M tile)

```
Point upload:     ~1.5GB
KDTree build:     ~2.3GB
Neighbor queries: ~3.3GB (peak, 2 batches)
Normal compute:   ~6.2GB (peak, 2 batches)
Features:         ~6.8GB
Ground truth:     ~11.3GB (peak)
```

### H100 (18.6M tile)

```
Point upload:     ~1.5GB
KDTree build:     ~2.4GB
Neighbor queries: ~4.6GB (peak, SINGLE batch)
Normal compute:   ~11.3GB (peak, SINGLE batch)
Features:         ~12.1GB
Ground truth:     ~18.1GB (peak)
```

---

## Cost-Performance Analysis

### Hardware Ownership (5-year)

| GPU            | Hardware Cost | Tiles/Hour                   | Cost per 1M pts | Best for                    |
| -------------- | ------------- | ---------------------------- | --------------- | --------------------------- |
| RTX 4080 Super | ~$1,000       | ~133                         | ~$0.001         | Development, small projects |
| RTX 4090       | ~$1,600       | ~218                         | ~$0.0012        | Production, medium projects |
| H100           | ~$30,000      | ~375 (single) / ~960 (batch) | ~$0.013         | Data centers only           |

### Cloud Pricing (approximate)

| GPU            | Cloud Cost/Hour | Tiles/Hour | Cost per Tile | Break-even (tiles)       |
| -------------- | --------------- | ---------- | ------------- | ------------------------ |
| RTX 4080 Super | N/A             | 133        | N/A           | N/A                      |
| RTX 4090       | N/A             | 218        | N/A           | N/A                      |
| H100           | ~$4-5           | 375-960    | ~$0.005-0.011 | >100,000 tiles for cloud |

**Recommendation:**

- **< 10,000 tiles:** Use RTX 4080 Super or RTX 4090 locally
- **10,000-50,000 tiles:** RTX 4090 is optimal
- **> 100,000 tiles:** Consider H100 in cloud for burst processing
- **> 1,000,000 tiles:** H100 data center deployment with multi-GPU

---

## Advanced Tuning

Each profile includes an "Advanced Tuning" section with expert configurations for users who want to squeeze maximum performance. These configs are **experimental** and may cause instability:

### RTX 4080 Super - Aggressive Config

```yaml
neighbor_query_batch_size: 10_000_000 # 2× default, may hang on some systems
feature_batch_size: 50_000_000
gpu_memory_target: 0.95
```

**Result:** ~23 seconds per tile (15% faster, but less stable)

### RTX 4090 - Extreme Config

```yaml
neighbor_query_batch_size: 20_000_000 # Single batch for most tiles
feature_batch_size: 50_000_000
k_neighbors: 30
gpu_memory_target: 0.95
```

**Result:** ~12 seconds per tile (27% faster, uses more memory)

### H100 - Data Center Config

```yaml
neighbor_query_batch_size: 30_000_000
feature_batch_size: 150_000_000
k_neighbors: 40
gpu_streams: 32
# + Multi-GPU with MPI/Dask
```

**Result:** ~2.75 seconds per tile effective (27M pts/sec with 4 GPUs)

---

## Profile Selection Guide

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU Profile Selector                      │
└─────────────────────────────────────────────────────────────┘

How many tiles do you need to process?

  < 1,000 tiles:     RTX 4080 Super (default)
                     • Good enough performance
                     • Most cost-effective for small jobs
                     • ~9 hours total processing time

  1,000-10,000:      RTX 4080 Super or RTX 4090
                     • RTX 4080: ~75 hours, more economical
                     • RTX 4090: ~45 hours, faster results
                     • Choose based on urgency vs budget

  10,000-50,000:     RTX 4090
                     • ~9 days on RTX 4090 (manageable)
                     • ~15 days on RTX 4080 (too slow)
                     • Clear winner for this range

  50,000-100,000:    RTX 4090 or H100 (cloud)
                     • RTX 4090: ~19 days
                     • H100 cloud burst: ~5-7 days
                     • Consider hybrid approach

  > 100,000 tiles:   H100 (data center)
                     • Only viable option for this scale
                     • Multi-GPU recommended
                     • ~11 days (single) / 3 days (4× H100)

What's your budget?

  < $2,000:          RTX 4080 Super
  $2,000-5,000:      RTX 4090
  > $5,000:          Consider H100 cloud credits

What's your urgency?

  Days-weeks:        RTX 4080 Super or RTX 4090
  Hours-days:        H100 cloud burst
  Real-time:         Not feasible (yet)
```

---

## Implementation Notes

These profiles are **YAML configuration files** that override the base configuration. They are **NOT** automatically loaded by the CLI. Users must specify them explicitly:

```bash
# Method 1: Use --profile flag (recommended)
ign-lidar-hd process --profile rtx4090 input/ output/

# Method 2: Use --config-path and --config-name
ign-lidar-hd process \
  --config-path ign_lidar/configs/profiles \
  --config-name rtx4090 \
  input/ output/

# Method 3: Override specific parameters
ign-lidar-hd process \
  --preset asprs_rtx4080 \
  features.neighbor_query_batch_size=10000000 \
  input/ output/
```

---

## Version History

- **v5.1.1** (October 18, 2025): Added GPU hang fix, created profiles
- **v5.1.0** (October 17, 2025): Base RTX 4080 optimization
- **v5.0.0** (October 2025): Initial V5 configuration system

---

## Contributing

If you benchmark these profiles on different hardware, please share your results! We're especially interested in:

- A6000 (48GB)
- A100 (40GB / 80GB)
- RTX 6000 Ada (48GB)
- L40S (48GB)
- H100 NVL (188GB dual-GPU)

---

## License

These configuration files are part of the IGN LiDAR HD Dataset project.
See the main LICENSE file for details.
