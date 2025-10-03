#!/usr/bin/env python3
"""
Convertisseur LAZ pour compatibilité QGIS.

Ce module fournit des fonctions pour convertir des fichiers LAZ enrichis
(format 1.4, point format 6) en fichiers compatibles QGIS (format 1.2, point format 3).
"""

import laspy
import sys
import click
from pathlib import Path
import numpy as np


def simplify_for_qgis(input_file, output_file=None, verbose=True):
    """
    Simplifie un LAZ enrichi pour maximiser compatibilité QGIS.
    
    Changements:
    - Version 1.4 -> 1.2
    - Point format 6 -> 3
    - Multiple extra dims -> 3 dims clés (height, planarity, verticality)
    
    Args:
        input_file: Chemin vers le fichier LAZ enrichi
        output_file: Chemin de sortie (optionnel, auto-généré si None)
        verbose: Afficher les messages de progression
        
    Returns:
        str: Chemin du fichier de sortie créé
    """
    input_path = Path(input_file)
    
    if output_file is None:
        output_file = str(input_path.parent / (input_path.stem + "_qgis.laz"))
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"SIMPLIFICATION POUR QGIS")
        print(f"{'='*70}\n")
        print(f"📂 Input:  {input_file}")
        print(f"📂 Output: {output_file}")
    
    # Charger fichier
    if verbose:
        print(f"\n1️⃣  Chargement du fichier...")
    las = laspy.read(input_file)
    
    if verbose:
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
    
    if verbose:
        print(f"   ✓ {len(extra_dims)} dimensions enrichies détectées")
    
    # Créer nouveau header compatible
    if verbose:
        print(f"\n2️⃣  Création format compatible QGIS...")
    
    from laspy import LasHeader
    
    # Format 3 = XYZ + RGB + GPS + classification (max compatible)
    header = LasHeader(version="1.2", point_format=3)
    header.scales = las.header.scales
    header.offsets = las.header.offsets
    
    las_out = laspy.LasData(header)
    
    # Copier données de base
    if verbose:
        print(f"   ✓ Copie XYZ, classification, intensité...")
    
    las_out.x = las.x
    las_out.y = las.y
    las_out.z = las.z
    
    if hasattr(las, 'classification'):
        # Point format 3 limite les classes à 0-31
        classification = np.array(las.classification)
        classification = np.clip(classification, 0, 31)
        las_out.classification = classification
        if verbose:
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
    if verbose:
        print(f"\n3️⃣  Ajout de 3 dimensions clés...")
    
    dims_added = 0
    
    if hasattr(las, 'height_above_ground'):
        las_out.add_extra_dim(laspy.ExtraBytesParams(
            name='height', 
            type='f4',
            description='Height above ground (m)'
        ))
        las_out.height = las.height_above_ground
        if verbose:
            print(f"   ✓ height (hauteur au-dessus du sol)")
        dims_added += 1
    
    if hasattr(las, 'planarity'):
        las_out.add_extra_dim(laspy.ExtraBytesParams(
            name='planar', 
            type='f4',
            description='Planarity score [0-1]'
        ))
        las_out.planar = las.planarity
        if verbose:
            print(f"   ✓ planar (score de planarité)")
        dims_added += 1
    
    if hasattr(las, 'verticality'):
        las_out.add_extra_dim(laspy.ExtraBytesParams(
            name='vertical', 
            type='f4',
            description='Verticality score [0-1]'
        ))
        las_out.vertical = las.verticality
        if verbose:
            print(f"   ✓ vertical (score de verticalité)")
        dims_added += 1
    
    if dims_added == 0 and verbose:
        print(f"   ⚠️  Aucune dimension enrichie trouvée!")
    
    # Écrire fichier
    if verbose:
        print(f"\n4️⃣  Écriture fichier simplifié...")
    
    las_out.write(output_file, do_compress=True)
    
    # Vérifier taille
    output_path = Path(output_file)
    output_size = output_path.stat().st_size
    input_size = input_path.stat().st_size
    
    if verbose:
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
        
        print(f"\n🚀 Utilisation dans QGIS:")
        print(f"   1. Ouvrir QGIS")
        print(f"   2. Menu: Couche > Ajouter une couche > Nuage de points")
        print(f"   3. Sélectionner: {output_file}")
        print(f"   4. Visualiser les attributs: height, planar, vertical")
    
    return output_file


@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', required=False, type=click.Path())
@click.option('--quiet', '-q', is_flag=True, help='Mode silencieux (pas de messages)')
@click.option('--batch', '-b', multiple=True, type=click.Path(exists=True), 
              help='Convertir plusieurs fichiers (répéter pour chaque fichier)')
def main(input_file, output_file, quiet, batch):
    """
    Convertir fichier LAZ enrichi vers format compatible QGIS.
    
    Convertit LAZ 1.4 format 6 avec extra dimensions vers LAZ 1.2 format 3
    avec seulement 3 dimensions clés (height, planar, vertical).
    
    \b
    Exemples:
        # Conversion simple
        ign-lidar-qgis enriched.laz
        
        # Avec sortie personnalisée
        ign-lidar-qgis enriched.laz output.laz
        
        # Mode silencieux
        ign-lidar-qgis enriched.laz -q
        
        # Batch (plusieurs fichiers)
        ign-lidar-qgis -b file1.laz -b file2.laz -b file3.laz
    """
    verbose = not quiet
    
    try:
        if batch:
            # Mode batch
            if verbose:
                print(f"\n🔄 Mode batch: {len(batch)} fichiers à convertir\n")
            
            for i, file_path in enumerate(batch, 1):
                if verbose:
                    print(f"\n[{i}/{len(batch)}] Traitement: {file_path}")
                simplify_for_qgis(file_path, output_file=None, verbose=verbose)
                
            if verbose:
                print(f"\n✅ Tous les fichiers convertis avec succès!")
        else:
            # Mode simple
            simplify_for_qgis(input_file, output_file, verbose=verbose)
            
    except Exception as e:
        if verbose:
            print(f"\n❌ ERREUR: {e}")
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
