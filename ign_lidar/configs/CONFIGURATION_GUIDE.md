# Configuration System v5.5 - Quick Reference

> **⚠️ This guide has been consolidated into [CONFIG_GUIDE.md](CONFIG_GUIDE.md)**  
> **Please use CONFIG_GUIDE.md for the complete, up-to-date documentation.**  
> **🇫🇷 Pour la documentation en français, voir [README.md](README.md)**

---

**This file is maintained for backward compatibility only.**  
**Last Updated**: October 31, 2025

All content from this quick reference has been merged into the main [CONFIG_GUIDE.md](CONFIG_GUIDE.md)
for easier maintenance and navigation.

---

## 🚀 Quick Start (see CONFIG_GUIDE.md for full details)

### Zero-Config (Simplest)

```bash
ign-lidar-hd process input_dir=/data output_dir=/output
```

### Use a Preset (Recommended)

```bash
# ASPRS classification with GPU
ign-lidar-hd process --config-name presets/asprs_classification_gpu \
  input_dir=/data output_dir=/output

# Fast preview
ign-lidar-hd process --config-name presets/fast_preview \
  input_dir=/data output_dir=/output
```

### Select Hardware Profile

```bash
# RTX 4080 (16GB)
ign-lidar-hd process --config-name profiles/gpu_rtx4080 \
  input_dir=/data output_dir=/output
```

## 📂 Configuration Files

```
configs/
├── base_complete.yaml    # Complete defaults (430 lines)
├── profiles/             # Hardware-specific (6 files)
│   ├── gpu_rtx4090.yaml  # 24GB VRAM
│   ├── gpu_rtx4080.yaml  # 16GB VRAM
│   ├── gpu_rtx3080.yaml  # 12GB VRAM
│   ├── gpu_rtx3060.yaml  # 8GB VRAM
│   ├── cpu_high_end.yaml # 32+ cores
│   └── cpu_standard.yaml # 8-16 cores
└── presets/              # Task-specific (4 files)
    ├── asprs_classification_gpu.yaml
    ├── asprs_classification_cpu.yaml
    ├── fast_preview.yaml
    └── high_quality.yaml
```

## 🎯 Choose Your Config

| If you want...        | Use this...                | Example                                          |
| --------------------- | -------------------------- | ------------------------------------------------ |
| **Quick test**        | `fast_preview`             | `--config-name presets/fast_preview`             |
| **Standard workflow** | `asprs_classification_gpu` | `--config-name presets/asprs_classification_gpu` |
| **Best quality**      | `high_quality`             | `--config-name presets/high_quality`             |
| **No GPU**            | `asprs_classification_cpu` | `--config-name presets/asprs_classification_cpu` |
| **Custom hardware**   | Select profile             | `--config-name profiles/gpu_rtx4090`             |

## 🔧 Creating Custom Configs

**Old way (650 lines):**

```yaml
# All defaults repeated...
processor:
  lod_level: "ASPRS"
  use_gpu: true
  gpu_batch_size: 8_000_000
  # ... 600 more lines
```

**New way (20 lines):**

```yaml
# my_config.yaml
defaults:
  - /base_complete
  - /profiles/gpu_rtx4080

config_name: "my_custom"

# Only override what changes
processor:
  gpu_batch_size: 10_000_000
features:
  k_neighbors: 40
```

## 📊 Performance Comparison

| Profile   | Hardware  | Time/20M pts | Throughput   |
| --------- | --------- | ------------ | ------------ |
| RTX 4090  | 24GB VRAM | 6-10 min     | 120-160M/min |
| RTX 4080  | 16GB VRAM | 8-14 min     | 80-100M/min  |
| RTX 3080  | 12GB VRAM | 12-18 min    | 60-80M/min   |
| CPU (32c) | 64GB RAM  | 45-60 min    | 15-25M/min   |

## ✨ What's New in v5.5

- ✅ **97% smaller configs** (20 lines vs 650)
- ✅ **Zero-config mode** (works with just paths)
- ✅ **6 hardware profiles** (GPU + CPU)
- ✅ **4 task presets** (common workflows)
- ✅ **No more missing keys** (all required sections included)
- ✅ **Smart defaults** (works out-of-box for 80% of users)

## 🆘 Troubleshooting

**Out of memory?**

```bash
# Use smaller profile
--config-name profiles/gpu_rtx3060
# Or reduce batch size
processor.gpu_batch_size=4_000_000
```

**Config not found?**

```bash
# Check location
ls ign_lidar/configs/profiles/
ls ign_lidar/configs/presets/
```

**Want to see merged config?**

```bash
ign-lidar-hd process --config-name presets/asprs_classification_gpu \
  input_dir=/data output_dir=/output --cfg job
```

---

**Full documentation**: See [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md)
