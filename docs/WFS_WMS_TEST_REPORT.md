# Rapport de Test des Services WFS/WMS IGN

## Date: 2025-11-28

## Résumé Exécutif

Tests complets effectués sur tous les endpoints de services IGN utilisés dans le projet IGN LIDAR HD Dataset.

**Taux de réussite global: 83.3% (5/6 services)**

---

## 1. Services WFS (Web Feature Service)

### ✅ Service Principal: BD TOPO V3

- **Endpoint**: `https://data.geopf.fr/wfs`
- **Status**: ✅ **Opérationnel**
- **Version**: 2.0.0
- **GetCapabilities**: 200 OK (4.8 MB)

#### Layers Testés

| Layer Name                          | Status | Features | Temps |
| ----------------------------------- | ------ | -------- | ----- |
| `BDTOPO_V3:batiment`                | ✅ OK  | 100      | 161ms |
| `BDTOPO_V3:troncon_de_route`        | ✅ OK  | 100      | 154ms |
| `BDTOPO_V3:troncon_de_voie_ferree`  | ✅ OK  | 11       | 103ms |
| `BDTOPO_V3:surface_hydrographique`  | ✅ OK  | 0        | 81ms  |
| `BDTOPO_V3:zone_de_vegetation`      | ✅ OK  | 100      | 131ms |
| `BDTOPO_V3:terrain_de_sport`        | ✅ OK  | 2        | 100ms |
| `BDTOPO_V3:cimetiere`               | ✅ OK  | 1        | 87ms  |
| `BDTOPO_V3:ligne_electrique`        | ✅ OK  | 0        | 85ms  |
| `BDTOPO_V3:construction_surfacique` | ✅ OK  | 1        | 94ms  |
| `BDTOPO_V3:reservoir`               | ✅ OK  | 0        | 89ms  |

**Temps de réponse moyen**: 108ms  
**Tous les layers BD TOPO V3 sont fonctionnels** ✅

### ❌ Service BD Forêt V2

- **Endpoint**: `https://data.geopf.fr/wfs`
- **Layer**: `BDFORET_V2:formation_vegetale`
- **Status**: ❌ **NON DISPONIBLE**
- **Erreur**: HTTP 400 - "Unknown namespace [BDFORET_V2]"

**Diagnostic**:

- Le namespace `BDFORET_V2` n'existe pas dans le service WFS actuel
- Possible que BD Forêt ait été déplacé vers un autre service ou supprimé
- Le code fait référence à un layer qui n'est plus disponible

**Recommandation**:

1. ⚠️ Désactiver ou supprimer les références à `BDFORET_V2` dans le code
2. Vérifier si BD Forêt est disponible sur un autre endpoint IGN
3. Mettre à jour la documentation pour refléter la non-disponibilité

---

## 2. Services WMS (Web Map Service)

### ✅ Service MNT: RGE ALTI / LiDAR HD MNT

- **Endpoint**: `https://data.geopf.fr/wms-r/wms`
- **Status**: ✅ **Opérationnel**
- **Version**: 1.3.0

#### Layers Disponibles

- ✅ `IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.SHADOW` (LiDAR HD MNT - 1m)
- ✅ `ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES` (RGE ALTI - 1-5m)

#### Test GetMap

- **Format**: GeoTIFF
- **Status**: ✅ OK (HTTP 200)
- **Taille**: 40,514 bytes (test 100x100)
- **Temps**: 404ms
- **Content-Type**: `image/geotiff`

**Le service MNT fonctionne correctement** ✅

### ✅ Service Orthophotos RGB

- **Endpoint**: `https://data.geopf.fr/wms-r`
- **Status**: ✅ **Opérationnel**
- **Layer**: ✅ `HR.ORTHOIMAGERY.ORTHOPHOTOS` disponible
- **Résolution**: 20cm

**Le service Orthophotos fonctionne correctement** ✅

---

## 3. Paramètres de Test

### Bbox de Test (Versailles)

```
Lambert 93 (EPSG:2154):
(650000, 6860000, 651000, 6861000)
```

### Configuration

- **Timeout**: 30 secondes
- **Format sortie**: `application/json` (WFS) / `image/geotiff` (WMS)
- **CRS**: EPSG:2154 (Lambert 93)
- **Max features**: 100 (pour tests)

---

## 4. Résultats Détaillés par Service

### WFS BD TOPO V3

```json
{
  "service_url": "https://data.geopf.fr/wfs",
  "version": "2.0.0",
  "tested_layers": 10,
  "success_rate": "100%",
  "average_response_time": "108ms",
  "status": "operational"
}
```

### WFS BD Forêt V2

```json
{
  "service_url": "https://data.geopf.fr/wfs",
  "layer": "BDFORET_V2:formation_vegetale",
  "error": "Unknown namespace [BDFORET_V2]",
  "http_code": 400,
  "status": "not_available"
}
```

### WMS MNT

```json
{
  "service_url": "https://data.geopf.fr/wms-r/wms",
  "version": "1.3.0",
  "layers_available": ["LiDAR HD MNT", "RGE ALTI"],
  "getmap_test": "success",
  "response_time": "404ms",
  "status": "operational"
}
```

### WMS Orthophotos

```json
{
  "service_url": "https://data.geopf.fr/wms-r",
  "layer": "HR.ORTHOIMAGERY.ORTHOPHOTOS",
  "resolution": "20cm",
  "status": "operational"
}
```

---

## 5. Actions Requises

### 🔴 Urgent

1. **Corriger ou supprimer les références à BD Forêt V2** dans:
   - `ign_lidar/io/bd_foret.py`
   - Toute autre référence à `BDFORET_V2:formation_vegetale`

### 🟡 Moyen Terme

2. **Documentation**:
   - Mettre à jour les docs pour indiquer que BD Forêt V2 n'est pas disponible
   - Documenter les layers WFS qui sont confirmés comme fonctionnels

### 🟢 Optionnel

3. **Amélioration**:
   - Ajouter des tests automatisés réguliers de ces endpoints
   - Implémenter une détection automatique des layers disponibles

---

## 6. Fichiers de Rapport Générés

1. **`wfs_test_report.json`**: Test détaillé des 10 layers BD TOPO V3
2. **`ign_services_test_report.json`**: Rapport complet de tous les services

---

## 7. Conclusion

L'infrastructure de services IGN Géoplateforme est **globalement fonctionnelle** avec une seule exception notable:

- ✅ **BD TOPO V3 WFS**: Tous les layers fonctionnent parfaitement
- ✅ **MNT WMS**: LiDAR HD et RGE ALTI disponibles
- ✅ **Orthophotos WMS**: Service opérationnel
- ❌ **BD Forêt V2 WFS**: Namespace inexistant, service non disponible

Le projet peut continuer à utiliser en toute confiance les services BD TOPO V3, MNT et Orthophotos. Seul le module BD Forêt nécessite une correction ou une suppression.

---

## Scripts de Test Disponibles

- **`scripts/test_wfs_endpoints.py`**: Test complet des layers WFS
- **`scripts/test_all_ign_services.py`**: Test de tous les services (WFS + WMS)

Usage:

```bash
python scripts/test_wfs_endpoints.py
python scripts/test_all_ign_services.py
```
