# 🎮 GPU Batch Size - Correction Fallback CPU

## 🚨 **Problème Identifié :**

### **Symptôme :**

```
[INFO] 💾 GPU batch/chunk size: 16,000,000 points (config)
🚀 GPU mode enabled (batch_size=8,000,000) (réel)
```

- **Configuration** : 16M points
- **Réalité** : 8M points → Le système réduit automatiquement
- **Cause** : Batch trop grand → Risque de fallback CPU silencieux

### **Indicateurs de Fallback CPU :**

- GPU batch réduit de 16M → 8M automatiquement
- Processing très lent (pas d'accélération visible)
- GPU utilization faible dans nvidia-smi
- Pas de messages d'erreur explicites

## ✅ **Corrections Appliquées :**

### **1. Configuration Stable (`config_asprs_rtx4080.yaml`)**

```yaml
features:
  # GPU STABLE - Évite le fallback
  gpu_batch_size: 4_000_000 # 4M (au lieu de 16M)
  vram_utilization_target: 0.75 # 75% (au lieu de 90%)
  num_cuda_streams: 4 # 4 (au lieu de 8)

  # Paramètres de qualité ajustés
  k_neighbors: 12 # 12 (au lieu de 8)
  search_radius: 0.8 # 0.8 (au lieu de 0.6)

  # Optimisations désactivées pour stabilité
  gpu_optimization:
    enable_mixed_precision: false # OFF (peut causer fallback)
    enable_tensor_cores: true # ON (stable)
    adaptive_memory_management: true # ON (stable)
```

### **2. Script Test GPU Stable (`test_gpu_stable.sh`)**

- Test sur 1 fichier avec paramètres conservatifs
- Monitoring GPU en temps réel
- Validation que GPU est utilisé efficacement
- Paramètres : 4M batch, 75% VRAM, pas mixed precision

### **3. Pipeline GPU Conservative (`run_gpu_conservative.sh`)**

- Traitement complet avec paramètres ultra-conservatifs
- Fallback CPU activé si GPU échoue
- Monitoring GPU continu
- Paramètres : 2M batch, 70% VRAM, fonctionnalités minimales

## 🎯 **Progression des Tests :**

### **Étape 1 : Test GPU Stable**

```bash
./test_gpu_stable.sh
```

**Objectif :** Valider que GPU fonctionne sans fallback
**Paramètres :** 4M batch, 75% VRAM
**Durée :** 5-10 minutes

### **Étape 2 : Pipeline Conservative (si Étape 1 OK)**

```bash
./run_gpu_conservative.sh
```

**Objectif :** Traitement complet avec paramètres stables
**Paramètres :** 2M batch, 70% VRAM
**Durée :** 15-30 minutes/tuile

### **Étape 3 : Augmentation Progressive (si Étape 2 OK)**

- Augmenter batch_size à 6M puis 8M
- Augmenter vram_utilization à 0.8
- Tester performance vs stabilité

## 📊 **Validation GPU Effectiveness :**

### **Logs de Succès Attendus :**

```
🚀 GPU mode enabled (batch_size=4,000,000)  ✅
[INFO] 💾 GPU batch/chunk size: 4,000,000 points ✅
[INFO] ✓ Computed 7 geometric features in XXXs using gpu ✅
```

### **Métriques GPU (nvidia-smi) :**

```
GPU Utilization: >80%  ✅
Memory Usage: 8-12GB (out of 16GB)  ✅
Temperature: <80°C  ✅
```

### **Performance Attendue :**

- **Avec GPU stable** : 15-30 min/tuile
- **Sans GPU (fallback CPU)** : 2+ heures/tuile
- **Différence** : ~4-8x accélération avec GPU

## 🔧 **Paramètres par Niveau de Stabilité :**

### **Ultra-Conservative (Problèmes GPU persistants) :**

```yaml
gpu_batch_size: 1_000_000 # 1M points
vram_utilization_target: 0.6 # 60% VRAM
num_cuda_streams: 1 # 1 stream
enable_mixed_precision: false # OFF
enable_tensor_cores: false # OFF
memory_pool_enabled: false # OFF
```

### **Conservative (Recommandé) :**

```yaml
gpu_batch_size: 4_000_000 # 4M points
vram_utilization_target: 0.75 # 75% VRAM
num_cuda_streams: 4 # 4 streams
enable_mixed_precision: false # OFF
enable_tensor_cores: true # ON
memory_pool_enabled: true # ON
```

### **Aggressive (Si Conservative OK) :**

```yaml
gpu_batch_size: 8_000_000 # 8M points
vram_utilization_target: 0.85 # 85% VRAM
num_cuda_streams: 8 # 8 streams
enable_mixed_precision: true # ON
enable_tensor_cores: true # ON
memory_pool_enabled: true # ON
```

## 🚨 **Diagnostic Fallback CPU :**

### **Si GPU ne fonctionne toujours pas :**

```bash
# Vérifier GPU disponible
nvidia-smi

# Vérifier CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Vérifier CuPy
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"

# Mode CPU pur (fallback ultime)
features.use_gpu=false
```

### **Monitoring Continu :**

```bash
# Terminal 1: Lancer processing
./test_gpu_stable.sh

# Terminal 2: Monitor GPU
watch -n 1 nvidia-smi
```

La clé est de **commencer conservatif** et **augmenter progressivement** les paramètres GPU jusqu'à trouver le sweet spot stabilité/performance pour votre système RTX 4080.
