"""
Diagnostic script to check config parsing and data fetcher initialization.
"""

import logging
from pathlib import Path
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_path = Path('/mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET/configs/multiscale/config_asprs_preprocessing.yaml')

print("\n" + "="*80)
print("CONFIG PARSING DIAGNOSTIC")
print("="*80)

# Load config
logger.info(f"Loading config: {config_path.name}")
config = OmegaConf.load(config_path)

# Check data_sources section
print("\n" + "-"*80)
print("DATA_SOURCES CONFIGURATION")
print("-"*80)

if 'data_sources' in config:
    ds = config['data_sources']
    
    # BD TOPO
    if 'bd_topo' in ds:
        bd_topo = ds['bd_topo']
        print(f"\n📍 BD TOPO®:")
        print(f"   enabled: {bd_topo.get('enabled', 'NOT SET')}")
        
        if 'features' in bd_topo:
            features = bd_topo['features']
            print(f"   Features:")
            print(f"      buildings: {features.get('buildings', 'NOT SET')}")
            print(f"      roads: {features.get('roads', 'NOT SET')}")
            print(f"      railways: {features.get('railways', 'NOT SET')}")
            print(f"      water: {features.get('water', 'NOT SET')}")
            print(f"      vegetation: {features.get('vegetation', 'NOT SET')}")
            print(f"      bridges: {features.get('bridges', 'NOT SET')}")
            print(f"      parking: {features.get('parking', 'NOT SET')}")
            print(f"      cemeteries: {features.get('cemeteries', 'NOT SET')}")
            print(f"      power_lines: {features.get('power_lines', 'NOT SET')}")
            print(f"      sports: {features.get('sports', 'NOT SET')}")
        
        if 'parameters' in bd_topo:
            params = bd_topo['parameters']
            print(f"   Parameters:")
            print(f"      road_width_fallback: {params.get('road_width_fallback', 'NOT SET')}")
            print(f"      railway_width_fallback: {params.get('railway_width_fallback', 'NOT SET')}")
    
    # Check other sources
    print(f"\n🌲 BD Forêt® enabled: {ds.get('bd_foret', {}).get('enabled', 'NOT SET')}")
    print(f"🌾 RPG enabled: {ds.get('rpg', {}).get('enabled', 'NOT SET')}")
    print(f"🗺️  Cadastre enabled: {ds.get('cadastre', {}).get('enabled', 'NOT SET')}")
    
else:
    print("\n❌ NO data_sources section found in config!")

# Check if data fetcher would be initialized
print("\n" + "-"*80)
print("DATA FETCHER INITIALIZATION CHECK")
print("-"*80)

would_init = False
if 'data_sources' in config:
    ds = config['data_sources']
    bd_topo_enabled = ds.get('bd_topo', {}).get('enabled', False)
    bd_foret_enabled = ds.get('bd_foret', {}).get('enabled', False)
    rpg_enabled = ds.get('rpg', {}).get('enabled', False)
    cadastre_enabled = ds.get('cadastre', {}).get('enabled', False)
    
    would_init = bd_topo_enabled or bd_foret_enabled or rpg_enabled or cadastre_enabled
    
    print(f"BD TOPO enabled: {bd_topo_enabled}")
    print(f"BD Forêt enabled: {bd_foret_enabled}")
    print(f"RPG enabled: {rpg_enabled}")
    print(f"Cadastre enabled: {cadastre_enabled}")
    print(f"\n{'✅' if would_init else '❌'} Data fetcher WOULD be initialized: {would_init}")
    
    if would_init and bd_topo_enabled:
        features = ds.get('bd_topo', {}).get('features', {})
        print(f"\n📍 BD TOPO® features that would be fetched:")
        for feat, enabled in features.items():
            status = '✅' if enabled else '❌'
            print(f"   {status} {feat}: {enabled}")
else:
    print("❌ Cannot check - no data_sources section")

print("\n" + "="*80)
