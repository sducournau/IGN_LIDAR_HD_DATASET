#!/usr/bin/env python3
"""
Exemple simple : Télécharger 3 tuiles urbaines et créer patches LOD2
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent))

from ign_lidar.downloader import IGNLiDARDownloader
from ign_lidar.processor import LiDARProcessor
from ign_lidar.strategic_locations import STRATEGIC_LOCATIONS

def main():
    print("=" * 70)
    print("🚀 EXEMPLE : Téléchargement Tuiles Urbaines + Patches LOD2")
    print("=" * 70)
    print()
    
    # Configuration
    output_dir = Path("./example_urban_dataset")
    output_dir.mkdir(exist_ok=True)
    
    # Étape 1: Sélectionner Paris Haussmann
    print("📍 Localisation : Paris Haussmann")
    paris_location = STRATEGIC_LOCATIONS["paris_haussmann"]
    bbox_wgs84 = paris_location["bbox"]
    print(f"   BBox WGS84: {bbox_wgs84}")
    print()
    
    # Étape 2: Télécharger 2-3 tuiles
    print("📥 Téléchargement des tuiles...")
    tiles_dir = output_dir / "raw_tiles"
    tiles_dir.mkdir(exist_ok=True)
    
    downloader = IGNLiDARDownloader(tiles_dir)
    tiles_data = downloader.fetch_available_tiles(bbox_wgs84)
    
    features = tiles_data.get('features', [])[:3]
    print(f"   Trouvé {len(features)} tuiles")
    
    downloaded = []
    for idx, feature in enumerate(features, 1):
        props = feature.get('properties', {})
        tile_name = props.get('name', '')
        
        if tile_name:
            print(f"   [{idx}/{len(features)}] {tile_name}")
            if downloader.download_tile(tile_name):
                tile_path = tiles_dir / tile_name
                downloaded.append(tile_path)
    
    print(f"   ✓ {len(downloaded)} tuiles téléchargées")
    print()
    
    if not downloaded:
        print("❌ Aucune tuile téléchargée. Arrêt.")
        return
    
    # Étape 3: Créer patches LOD2 150m²
    print("🎯 Création des patches d'entraînement...")
    print("   Patch size: 150m × 150m")
    print("   LOD Level: LOD2")
    print("   Augmentation: Oui (3 variations)")
    print()
    
    patches_dir = output_dir / "patches_lod2" / "train"
    patches_dir.mkdir(parents=True, exist_ok=True)
    
    processor = LiDARProcessor(
        lod_level="LOD2",
        patch_size=150.0,
        augment=True,
        num_augmentations=3
    )
    
    total_patches = 0
    for idx, laz_file in enumerate(downloaded, 1):
        print(f"   [{idx}/{len(downloaded)}] {laz_file.name}")
        
        try:
            num_patches = processor.process_tile(laz_file, patches_dir)
            total_patches += num_patches
            print(f"      → {num_patches} patches créés")
        except Exception as e:
            print(f"      ❌ Erreur: {e}")
    
    print()
    print("=" * 70)
    print("✅ TERMINÉ!")
    print("=" * 70)
    print(f"📊 Statistiques:")
    print(f"   - Tuiles téléchargées: {len(downloaded)}")
    print(f"   - Patches créés: {total_patches:,}")
    print()
    print(f"📁 Dataset disponible dans: {output_dir}/")
    print(f"   - LAZ bruts: {tiles_dir}/")
    print(f"   - Patches: {patches_dir}/")
    print()
    print("💡 Utilisation des patches:")
    print("   import numpy as np")
    print(f"   data = np.load('{patches_dir}/[fichier].npz')")
    print("   points = data['points']")
    print("   labels = data['labels']")
    print("=" * 70)


if __name__ == "__main__":
    main()
