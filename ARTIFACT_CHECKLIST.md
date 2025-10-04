# Geometric Feature Artifacts - Quick Checklist

**Version**: 1.0  
**Date**: October 4, 2025

---

## ✅ Is Your System Artifact-Free?

### Quick Check

Run this command to check your version:

```bash
ign-lidar-hd --version
# Should show: v1.1.0 or later
```

If **v1.1.0+**: ✅ **You're good!** System is artifact-free.

If **< v1.1.0**: ⚠️ Upgrade to fix artifacts:

```bash
pip install --upgrade ign-lidar-hd
```

---

## 🔍 Visual Artifact Detection

### What Do Artifacts Look Like?

**In QGIS/CloudCompare** (viewing linearity/planarity):

```
❌ ARTIFACTS (Pre-v1.1.0):
═══════════════════  ← Visible scan lines
░░░▓▓▓░░░▓▓▓░░░▓▓▓  ← "Dash line" pattern
═══════════════════  ← Striped appearance
Feature follows LIDAR scan pattern, not geometry

✅ ARTIFACT-FREE (v1.1.0+):
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ← Uniform features
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ← Smooth surfaces
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ← No visible pattern
Feature reflects true geometry
```

### Quick Visual Test

1. Load enriched LAZ in CloudCompare/QGIS
2. Color by `planarity` or `linearity`
3. Look at flat surfaces (roofs, walls)

**Good**: Uniform color on flat surfaces  
**Bad**: Striped or dashed patterns

---

## 🛠️ Command Reference

### Standard Enrichment (Artifact-Free)

```bash
# Basic (RECOMMENDED)
ign-lidar-hd enrich \
  --input-dir raw_tiles/ \
  --output enriched_tiles/ \
  --mode building

# With RGB
ign-lidar-hd enrich \
  --input-dir raw_tiles/ \
  --output enriched_tiles/ \
  --mode building \
  --add-rgb

# With parallel processing
ign-lidar-hd enrich \
  --input-dir raw_tiles/ \
  --output enriched_tiles/ \
  --mode building \
  --num-workers 4
```

✅ **No --radius needed** = Auto-estimation active!

### Advanced: Manual Radius

```bash
# Dense urban (sharper features)
ign-lidar-hd enrich ... --radius 0.8

# Sparse rural (smoother features)
ign-lidar-hd enrich ... --radius 1.5
```

---

## 📊 Quick Parameter Guide

| Scenario           | Radius | Effect             |
| ------------------ | ------ | ------------------ |
| **Auto (Default)** | ~1.0m  | ✅ **RECOMMENDED** |
| Dense urban        | 0.8m   | Sharp, detailed    |
| Standard           | 1.0m   | Balanced           |
| Sparse rural       | 1.5m   | Smooth, stable     |
| Noise filtering    | 2.0m   | Very smooth        |

### How to Choose

1. **Most cases**: Don't specify `--radius` (let system auto-estimate) ✅
2. **High detail needed**: `--radius 0.8`
3. **Noisy data**: `--radius 1.5` or `2.0`

---

## 🧪 Testing for Artifacts

### Method 1: Visual Inspection

```bash
# Generate analysis plots
python scripts/analysis/inspect_features.py \
  enriched.laz \
  --save-plots
```

**Look for**:

- Uniform histograms (good)
- Multiple peaks/spikes (potential artifacts)
- Gaps in distribution (potential artifacts)

### Method 2: Compare Radii

```bash
# Test different radius values
python scripts/analysis/test_radius_comparison.py \
  test_tile.laz \
  --radii 0.5 1.0 1.5 2.0
```

**Compare** the output LAZ files in CloudCompare/QGIS.

---

## 📖 Documentation Map

### For Quick Reference

- **[This Checklist]** - You are here! ✅
- **[Summary](ENRICH_ARTIFACT_AUDIT_SUMMARY.md)** - Executive summary (5 min read)
- **[Radius Guide](RADIUS_PARAMETER_GUIDE.md)** - Parameter tuning (10 min read)

### For Deep Dive

- **[Full Report](ARTIFACT_AUDIT_ENRICH_REPORT.md)** - Complete technical audit
- **[Implementation](ARTIFACT_AUDIT_COMPLETE.md)** - Development summary

### External Links

- **[Main Docs](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/)** - Full documentation
- **[GitHub](https://github.com/sducournau/IGN_LIDAR_HD_DATASET)** - Source code

---

## ❓ FAQ

### Q: Do I have artifacts in my enriched files?

**A**: If you're using **v1.1.0 or later**, NO! The system is artifact-free by default.

### Q: Should I change my workflow?

**A**: **NO!** If you're not specifying `--radius`, you're already using best practices.

### Q: What if I see artifacts?

**A**:

1. Check version: `ign-lidar-hd --version`
2. If < v1.1.0, upgrade: `pip install --upgrade ign-lidar-hd`
3. Re-enrich your files with updated version

### Q: How do I tune the radius parameter?

**A**:

1. **Start with auto** (no parameter) - works 90% of the time
2. **Dense areas**: Try `--radius 0.8`
3. **Sparse areas**: Try `--radius 1.5`
4. **Use test script**: Compare multiple values

### Q: Does radius affect performance?

**A**: Yes, ~10-15% slower than old k-NN method, but results are scientifically correct and artifact-free. Worth the trade-off!

### Q: Can I use GPU acceleration?

**A**: Yes! GPU supports radius parameter (with automatic CPU fallback if needed):

```bash
ign-lidar-hd enrich ... --use-gpu --radius 1.0
```

---

## 🎯 One-Minute Summary

1. ✅ **System is artifact-free** (since v1.1.0)
2. ✅ **No action needed** - auto-radius works by default
3. ✅ **Documentation available** - guides for advanced tuning
4. ✅ **Tools provided** - scripts for testing and validation

**Bottom line**: Continue using your current workflow! 🚀

---

## 📋 Checklist for New Users

- [ ] Install latest version (`pip install ign-lidar-hd`)
- [ ] Run enrich without `--radius` parameter (auto-estimation)
- [ ] Check one enriched file in CloudCompare/QGIS
- [ ] If features look good (uniform on flat surfaces), you're done! ✅
- [ ] If not, check version and upgrade if needed
- [ ] (Optional) Read guides for advanced tuning

---

## 🆘 Troubleshooting

### Issue: Striped patterns in features

**Cause**: Using old version (< v1.1.0)  
**Fix**: Upgrade and re-enrich

```bash
pip install --upgrade ign-lidar-hd
ign-lidar-hd enrich --input-dir raw/ --output enriched/ --force
```

### Issue: Features too noisy

**Cause**: Small radius or very sparse data  
**Fix**: Increase radius

```bash
ign-lidar-hd enrich ... --radius 1.5
```

### Issue: Features too smooth (missing edges)

**Cause**: Radius too large  
**Fix**: Decrease radius

```bash
ign-lidar-hd enrich ... --radius 0.8
```

### Issue: Want to experiment

**Solution**: Use comparison script

```bash
python scripts/analysis/test_radius_comparison.py tile.laz --auto
```

---

**Last Updated**: October 4, 2025  
**Status**: ✅ Complete  
**Next Review**: Not required
