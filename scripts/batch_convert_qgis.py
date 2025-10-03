#!/usr/bin/env python3
"""
Script de conversion batch pour tous les fichiers LAZ enrichis.
Convertit tous les fichiers dans /mnt/c/Users/Simon/ign/pre_tiles
"""

import sys
from pathlib import Path
sys.path.insert(0, '/mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_downloader')

from ign_lidar.qgis_converter import simplify_for_qgis

# Répertoire contenant les fichiers enrichis
base_dir = Path("/mnt/c/Users/Simon/ign/pre_tiles")

# Trouver tous les fichiers LAZ (sauf ceux déjà convertis)
laz_files = []
for laz_file in base_dir.rglob("*.laz"):
    # Exclure les fichiers déjà convertis
    if "_qgis" not in laz_file.stem and ".copc" not in laz_file.stem:
        laz_files.append(laz_file)

print(f"\n{'='*70}")
print(f"CONVERSION BATCH POUR QGIS")
print(f"{'='*70}\n")
print(f"📂 Répertoire: {base_dir}")
print(f"📊 Fichiers à convertir: {len(laz_files)}\n")

# Lister les fichiers
for i, file_path in enumerate(laz_files, 1):
    relative_path = file_path.relative_to(base_dir)
    print(f"  [{i}] {relative_path}")

print(f"\n{'='*70}")
input("Appuyez sur ENTRÉE pour commencer la conversion (ou Ctrl+C pour annuler)...")
print(f"{'='*70}\n")

# Convertir chaque fichier
success_count = 0
error_count = 0
errors = []

for i, file_path in enumerate(laz_files, 1):
    relative_path = file_path.relative_to(base_dir)
    print(f"\n{'='*70}")
    print(f"[{i}/{len(laz_files)}] {relative_path}")
    print(f"{'='*70}")
    
    try:
        output_file = simplify_for_qgis(str(file_path), verbose=True)
        success_count += 1
        print(f"\n✅ Succès: {Path(output_file).name}")
    except Exception as e:
        error_count += 1
        error_msg = f"{relative_path}: {str(e)}"
        errors.append(error_msg)
        print(f"\n❌ Erreur: {e}")

# Résumé final
print(f"\n\n{'='*70}")
print(f"RÉSUMÉ DE LA CONVERSION")
print(f"{'='*70}\n")
print(f"✅ Fichiers convertis avec succès: {success_count}/{len(laz_files)}")
print(f"❌ Erreurs: {error_count}/{len(laz_files)}")

if errors:
    print(f"\n⚠️  Détails des erreurs:")
    for error in errors:
        print(f"  - {error}")

print(f"\n{'='*70}")
print(f"📁 Les fichiers *_qgis.laz sont prêts pour QGIS!")
print(f"{'='*70}\n")

print("🚀 Pour ouvrir dans QGIS:")
print("  1. Ouvrir QGIS")
print("  2. Menu: Couche > Ajouter une couche > Nuage de points")
print("  3. Sélectionner un fichier *_qgis.laz")
print("  4. Visualiser les attributs: height, planar, vertical\n")
