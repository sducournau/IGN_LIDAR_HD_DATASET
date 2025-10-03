#!/usr/bin/env python3
"""
Simplifier fichiers LAZ enrichis pour compatibilité maximale avec QGIS.

Ce script convertit un fichier LAZ 1.4 (format 6) avec extra dimensions
en fichier LAZ 1.2 (format 3) avec seulement 3 dimensions clés.

Usage:
    python simplify_for_qgis.py fichier_enriched.laz [output.laz]
"""

import laspy
import sys
from pathlib import Path


def simplify_for_qgis(input_file, output_file=None):
    """
    Simplifie un LAZ enrichi pour maximiser compatibilité QGIS.
    
    Changements:
    - Version 1.4 -> 1.2
    - Point format 6 -> 3
    - 15 extra dims -> 3 dims clés (height, planarity, verticality)
    """
    input_path = Path(input_file)
    
    if output_file is None:
        output_file = str(input_path.parent / (input_path.stem + "_qgis.laz"))
    
    print(f"\n{'='*70}")
    print(f"SIMPLIFICATION POUR QGIS")
    print(f"{'='*70}\n")
    
    print(f"📂 Input:  {input_file}")
    print(f"📂 Output: {output_file}")
    
    # Charger fichier
    print(f"\n1️⃣  Chargement du fichier...")
    las = laspy.read(input_file)
    print(f"   ✓ {len(las.points):,} points chargés")
    print(f"   ✓ Format actuel: LAS {las.header.version}, Point format {las.header.point_format.id}")
    
    # Compter extra dimensions
    standard_dims = {'X', 'Y', 'Z', 'intensity', 'return_number', 
                    'number_of_returns', 'classification', 'user_data',
                    'point_source_id', 'gps_time', 'red', 'green', 'blue',
                    'scan_direction_flag', 'edge_of_flight_line',
                    'synthetic', 'key_point', 'withheld', 'overlap',
                    'scan_angle_rank', 'scanner_channel', 'scan_angle'}
    
    all_dims = set(las.point_format.dimension_names)
    extra_dims = all_dims - standard_dims
    print(f"   ✓ {len(extra_dims)} dimensions enrichies détectées")
    
    # Créer nouveau header compatible
    print(f"\n2️⃣  Création format compatible QGIS...")
    from laspy import LasHeader
    
    # Format 3 = XYZ + RGB + GPS + classification (max compatible)
    header = LasHeader(version="1.2", point_format=3)
    header.scales = [0.01, 0.01, 0.01]
    header.offsets = [0.0, 0.0, 0.0]
    
    las_out = laspy.LasData(header)
    
    # Copier données de base
    print(f"   ✓ Copie XYZ, classification, intensité...")
    las_out.x = las.x
    las_out.y = las.y
    las_out.z = las.z
    
    if hasattr(las, 'classification'):
        # Point format 3 limite les classes à 0-31
        # Remapper les classes > 31 vers des valeurs compatibles
        import numpy as np
        classification = np.array(las.classification)
        classification = np.clip(classification, 0, 31)
        las_out.classification = classification
        print(f"   ✓ Classification remappée (max 31 pour format 3)")
    
    if hasattr(las, 'intensity'):
        las_out.intensity = las.intensity
    
    if hasattr(las, 'return_number'):
        las_out.return_number = las.return_number
    
    if hasattr(las, 'number_of_returns'):
        las_out.number_of_returns = las.number_of_returns
    
    if hasattr(las, 'gps_time'):
        las_out.gps_time = las.gps_time
    
    # Ajouter 3 dimensions clés SEULEMENT
    print(f"\n3️⃣  Ajout de 3 dimensions clés...")
    dims_added = 0
    
    if hasattr(las, 'height_above_ground'):
        las_out.add_extra_dim(laspy.ExtraBytesParams(
            name='height', 
            type='f4',
            description='Height above ground (m)'
        ))
        las_out.height = las.height_above_ground
        print(f"   ✓ height (hauteur au-dessus du sol)")
        dims_added += 1
    
    if hasattr(las, 'planarity'):
        las_out.add_extra_dim(laspy.ExtraBytesParams(
            name='planar', 
            type='f4',
            description='Planarity score [0-1]'
        ))
        las_out.planar = las.planarity
        print(f"   ✓ planar (score de planarité)")
        dims_added += 1
    
    if hasattr(las, 'verticality'):
        las_out.add_extra_dim(laspy.ExtraBytesParams(
            name='vertical', 
            type='f4',
            description='Verticality score [0-1]'
        ))
        las_out.vertical = las.verticality
        print(f"   ✓ vertical (score de verticalité)")
        dims_added += 1
    
    if dims_added == 0:
        print(f"   ⚠️  Aucune dimension enrichie trouvée!")
    
    # Écrire fichier
    print(f"\n4️⃣  Écriture fichier simplifié...")
    las_out.write(output_file, do_compress=True)
    
    # Vérifier taille
    output_path = Path(output_file)
    output_size = output_path.stat().st_size
    input_size = input_path.stat().st_size
    
    print(f"   ✓ Fichier écrit: {output_size:,} bytes ({output_size/1024/1024:.1f} MB)")
    print(f"   ✓ Réduction: {(1 - output_size/input_size)*100:.1f}%")
    
    print(f"\n{'='*70}")
    print(f"✅ SUCCÈS - Fichier compatible QGIS créé!")
    print(f"{'='*70}\n")
    
    print(f"📋 Résumé:")
    print(f"   - Format: LAS 1.2, Point format 3")
    print(f"   - Points: {len(las_out.points):,}")
    print(f"   - Dimensions: {dims_added} (height, planar, vertical)")
    print(f"   - Fichier: {output_file}")
    
    print(f"\n🚀 Prochaine étape:")
    print(f"   1. Ouvrir QGIS")
    print(f"   2. Menu: Couche > Ajouter une couche > Nuage de points")
    print(f"   3. Sélectionner: {output_file}")
    print(f"   4. Le fichier devrait s'ouvrir sans problème!")
    
    return output_file


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simplify_for_qgis.py fichier_enriched.laz [output.laz]")
        print("\nExemple:")
        print("  python simplify_for_qgis.py enriched.laz")
        print("  python simplify_for_qgis.py enriched.laz output_qgis.laz")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        simplify_for_qgis(input_file, output_file)
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
