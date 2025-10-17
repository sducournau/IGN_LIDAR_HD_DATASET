# Configuration Legacy Hydra - SUPPRIMÉE

## ⚠️ Configuration Obsolète

Les anciens fichiers de configuration Hydra ont été **supprimés** et archivés.

### 📦 Archive

Les fichiers ont été déplacés vers : `ign_lidar/configs_legacy_hydra_*`

### ✅ Migration Complète v4.0

**Utilisez maintenant la nouvelle structure :**

```bash
# ✅ Nouvelle méthode (v4.0)
./scripts/run_processing.sh --preset gpu_optimized --input /data/tiles

# ✅ Avec profil hardware
./scripts/run_processing.sh --preset asprs_classification --hardware rtx4080

# ✅ Configuration custom
./scripts/run_processing.sh --config configs/my_config.yaml
```

### 📚 Documentation

- **Nouvelle structure v4.0** : [`configs/README.md`](../../configs/README.md)
- **Guide de migration** : [`scripts/migrate_config_v4.py`](../../scripts/migrate_config_v4.py)
- **Presets disponibles** : [`configs/presets/`](../../configs/presets/)

### Archive Date

Configuration Hydra supprimée le 17 octobre 2025 - IGN LiDAR HD v4.0
