#!/usr/bin/env python3
"""
Script de diagnostic pour fichiers LAZ enrichis et compatibilité QGIS.

Ce script vérifie TOUS les aspects qui peuvent empêcher QGIS de lire un fichier LAZ.
"""

import sys
from pathlib import Path
import struct


def check_laz_file(file_path: Path):
    """Vérification complète d'un fichier LAZ pour QGIS."""
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC COMPLET: {file_path.name}")
    print(f"{'='*70}\n")
    
    # 1. Vérification fichier existe et taille
    if not file_path.exists():
        print(f"❌ Fichier introuvable: {file_path}")
        return False
    
    file_size = file_path.stat().st_size
    print(f"✓ Fichier existe: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
    
    # 2. Vérification magic bytes LAZ
    with open(file_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'LASF':
            print(f"❌ Magic bytes incorrects: {magic} (attendu: LASF)")
            return False
        print(f"✓ Magic bytes LAZ valides: {magic}")
        
        # Lire version
        f.seek(24)
        version_major = struct.unpack('B', f.read(1))[0]
        version_minor = struct.unpack('B', f.read(1))[0]
        print(f"✓ Version LAS: {version_major}.{version_minor}")
        
        # Lire point format
        f.seek(104)
        point_format = struct.unpack('B', f.read(1))[0]
        print(f"✓ Point format: {point_format}")
    
    # 3. Vérification avec laspy
    try:
        import laspy
        print(f"\n📦 Test avec laspy {laspy.__version__}:")
        
        las = laspy.read(str(file_path))
        print(f"  ✓ Ouverture réussie")
        print(f"  ✓ Points: {len(las.points):,}")
        print(f"  ✓ Point format: {las.header.point_format.id}")
        
        # Extra dimensions
        standard_dims = {'X', 'Y', 'Z', 'intensity', 'return_number', 
                        'number_of_returns', 'classification', 'user_data',
                        'point_source_id', 'gps_time', 'red', 'green', 'blue',
                        'scan_direction_flag', 'edge_of_flight_line',
                        'synthetic', 'key_point', 'withheld', 'overlap',
                        'scan_angle_rank', 'scanner_channel', 'scan_angle'}
        
        all_dims = set(las.point_format.dimension_names)
        extra_dims = all_dims - standard_dims
        
        if extra_dims:
            print(f"  ✓ Dimensions enrichies: {len(extra_dims)}")
            for dim in sorted(extra_dims):
                print(f"      - {dim}")
        else:
            print(f"  ⚠️  Aucune dimension enrichie (fichier brut?)")
        
        # Vérifier COPC
        is_copc = any('copc' in vlr.description.lower() or 'copc' in vlr.user_id.lower() 
                     for vlr in las.header.vlrs)
        
        if is_copc:
            print(f"  ⚠️  COPC détecté - QGIS peut avoir des problèmes")
        else:
            print(f"  ✓ Format LAZ standard (non-COPC)")
        
        # Vérifier bounding box
        print(f"\n📊 Bounding box:")
        print(f"  X: [{las.header.x_min:.2f}, {las.header.x_max:.2f}]")
        print(f"  Y: [{las.header.y_min:.2f}, {las.header.y_max:.2f}]")
        print(f"  Z: [{las.header.z_min:.2f}, {las.header.z_max:.2f}]")
        
        # Vérifier échelle/offset
        print(f"\n📐 Échelle et offset:")
        print(f"  Scale: {las.header.scales}")
        print(f"  Offset: {las.header.offsets}")
        
        # Test de lecture de données
        print(f"\n🔍 Test données (premiers points):")
        for i in range(min(3, len(las.points))):
            x, y, z = las.x[i], las.y[i], las.z[i]
            print(f"  Point {i}: ({x:.2f}, {y:.2f}, {z:.2f})")
            
            # Vérifier valeurs valides
            if x < las.header.x_min or x > las.header.x_max:
                print(f"    ⚠️  X hors limites!")
            if y < las.header.y_min or y > las.header.y_max:
                print(f"    ⚠️  Y hors limites!")
            if z < las.header.z_min or z > las.header.z_max:
                print(f"    ⚠️  Z hors limites!")
        
        print(f"\n✅ Fichier techniquement valide pour QGIS")
        
    except Exception as e:
        print(f"\n❌ Erreur laspy: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Recommandations QGIS
    print(f"\n{'='*70}")
    print(f"RECOMMANDATIONS POUR QGIS")
    print(f"{'='*70}")
    
    print(f"\n1. Vérifier la version de QGIS:")
    print(f"   - QGIS 3.18+ recommandé pour support nuages de points")
    print(f"   - Menu: Aide > À propos")
    
    print(f"\n2. Chargement dans QGIS:")
    print(f"   a) Menu: Couche > Ajouter une couche > Ajouter une couche nuage de points")
    print(f"   b) OU glisser-déposer le fichier dans QGIS")
    print(f"   c) OU Data Source Manager > Point Cloud")
    
    print(f"\n3. Si le fichier n'apparaît pas:")
    print(f"   - Vérifier que l'extension Point Cloud est activée")
    print(f"   - Menu: Extensions > Gérer et installer les extensions")
    print(f"   - Chercher 'Point Cloud' dans les extensions installées")
    
    print(f"\n4. Si erreur 'Cannot open layer':")
    print(f"   - Vérifier le système de coordonnées (devrait être EPSG:2154 pour LAMB93)")
    print(f"   - Essayer de définir manuellement le CRS lors du chargement")
    
    print(f"\n5. Tester avec un autre logiciel:")
    print(f"   - CloudCompare: https://www.danielgm.net/cc/")
    print(f"   - Si CloudCompare peut ouvrir le fichier, le problème vient de QGIS")
    
    print(f"\n6. Alternative: Convertir en LAS non-compressé")
    print(f"   python -c \"import laspy; las = laspy.read('{file_path}'); las.write('{file_path.stem}.las')\"")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnostic_qgis.py <fichier.laz>")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    success = check_laz_file(file_path)
    
    sys.exit(0 if success else 1)
