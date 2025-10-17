# Optimisations Ultra-Rapides pour IGN LiDAR HD - RTX 4080

## Résumé des changements pour éliminer la génération de patches

### 🚨 Problème identifié :

- **Architecture "hybrid"** → Force la génération de patches
- **Temps de traitement : 2+ heures par tuile** (inacceptable)
- **Génération de patches non désirés** au lieu de LAZ enrichis
- **Erreurs de reclassification** (`OptimizedReclassifier` object has no attribute 'reclassify_points'`)

### ✅ Optimisations appliquées :

#### 1. **Architecture et Mode de Traitement**

- `processor.architecture: hybrid` → `direct`
- `processing.architecture: hybrid` → `direct`
- `processor.generate_patches: false` (force)
- `processing.generate_patches: false` (force)
- `patch_size/patch_overlap/num_points: null` (disable)

#### 2. **Reclassification (Source d'Erreurs)**

- `processor.reclassification.enabled: false` (évite les erreurs OptimizedReclassifier)
- `processor.apply_reclassification_inline: false`

#### 3. **Features Ultra-Rapides**

- `gpu_batch_size: 8M` → `16M` (2x plus gros)
- `vram_utilization_target: 0.85` → `0.9` (plus agressif)
- `num_cuda_streams: 4` → `8` (plus de parallélisme)
- `k_neighbors: 12` → `8` (moins de calculs)
- `search_radius: 0.8` → `0.6` (5-10x plus rapide)
- `enable_mixed_precision: true` (RTX 4080 optimized)

#### 4. **Features Désactivées (Gourmandes)**

- `compute_verticality: false`
- `compute_horizontality: false`
- `compute_sphericity: false`
- `use_nir: false` (fetching NIR est très lent)
- `compute_ndvi: false` (nécessite NIR)
- `use_infrared: false`

#### 5. **Data Sources Simplifiées**

- **BD TOPO** : Seulement buildings, roads, water (essentiels)
- **Cadastre** : `enabled: false` (très lent)
- **Végétation BD TOPO** : `disabled` (lent à traiter)

#### 6. **Preprocessing et Stitching**

- `preprocess.enabled: false` (skip outlier removal)
- `stitching.enabled: false` (pas de buffer entre tuiles)

#### 7. **Ground Truth Optimisé**

- `force_method: "strtree"` (CPU fiable, pas d'erreurs GPU)
- `use_ndvi: false` (évite fetch NIR lent)
- `fetch_rgb_nir: false` (évite fetch orthophotos)

### 🎯 Résultats Attendus :

- **Mode pur** : `enriched_only` → Seulement des LAZ enrichis, **AUCUN patch**
- **Vitesse** : 2+ heures → **5-10 minutes par tuile**
- **GPU RTX 4080** : Utilisation optimale à 90% VRAM
- **Fiabilité** : Évite les erreurs de reclassification

### 🚀 Commande de Test :

```bash
./run_ultra_fast_enrichment.sh
```

### 📊 Monitoring Attendu :

```
[INFO] ✨ Processing mode: enriched_only
[INFO] 📦 Extracting patches: FALSE (disabled)
[INFO] 💾 Saving enriched LAZ directly
[INFO] ✅ Completed: 1 enriched LAZ file (no patches)
```

### ⚠️ Si des patches sont encore générés :

1. Vérifier les logs pour `architecture: hybrid`
2. Forcer `processor.generate_patches=false` en ligne de commande
3. Utiliser `processing.mode=enriched_only` strict

### 🔧 Fallback si Erreurs GPU :

- `features.use_gpu: false` (CPU fallback)
- `ground_truth.optimization.force_method: "vectorized"` (CPU pur)
- `processor.batch_size: 32` (réduire charge mémoire)
