#!/usr/bin/env python3
"""
Exemple d'utilisation du traitement parallèle avec workers.

Ce script démontre comment utiliser efficacement les workers
pour accélérer le traitement de données LiDAR.
"""

import logging
from pathlib import Path
import multiprocessing as mp

from ign_lidar.downloader import IGNLiDARDownloader
from ign_lidar.processor import LiDARProcessor

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_parallel_download():
    """
    Exemple 1: Téléchargement parallèle de tuiles.
    
    Utilise ThreadPoolExecutor pour télécharger plusieurs tuiles
    simultanément (I/O bound).
    """
    logger.info("=" * 70)
    logger.info("EXEMPLE 1: Téléchargement parallèle")
    logger.info("=" * 70)
    
    # Créer le downloader
    output_dir = Path("data/raw_tiles")
    downloader = IGNLiDARDownloader(output_dir=output_dir)
    
    # Liste de tuiles à télécharger
    tile_list = [
        "HD_LIDARHD_FXX_0650_6860_PTS_C_LAMB93_IGN69.laz",
        "HD_LIDARHD_FXX_0651_6860_PTS_C_LAMB93_IGN69.laz",
        "HD_LIDARHD_FXX_0650_6861_PTS_C_LAMB93_IGN69.laz",
    ]
    
    # Télécharger avec 3 workers
    logger.info(f"Téléchargement de {len(tile_list)} tuiles avec 3 workers")
    
    results = downloader.batch_download(
        tile_list=tile_list,
        num_workers=3,  # 3 téléchargements simultanés
        max_retries=3
    )
    
    # Résultats
    success_count = sum(1 for v in results.values() if v)
    logger.info(f"✅ Téléchargement terminé: {success_count}/{len(tile_list)}")
    
    return results


def example_parallel_enrichment():
    """
    Exemple 2: Enrichissement parallèle de fichiers LAZ.
    
    Traite plusieurs fichiers LAZ en parallèle avec multiprocessing
    pour calculer les features géométriques (CPU bound).
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXEMPLE 2: Enrichissement parallèle")
    logger.info("=" * 70)
    
    input_dir = Path("data/raw_tiles")
    output_dir = Path("data/enriched_tiles")
    
    # Vérifier que les fichiers existent
    laz_files = list(input_dir.glob("*.laz"))
    if not laz_files:
        logger.warning("Aucun fichier LAZ trouvé dans data/raw_tiles")
        return
    
    logger.info(f"Trouvé {len(laz_files)} fichiers LAZ")
    
    # Déterminer le nombre optimal de workers
    num_workers = min(mp.cpu_count() - 1, len(laz_files))
    logger.info(f"Utilisation de {num_workers} workers (CPU: {mp.cpu_count()})")
    
    # Enrichir avec workers parallèles
    # Note: utiliser le CLI ign-lidar-process ou importer depuis le package
    from ign_lidar.processor import LiDARProcessor
    
    processor = LiDARProcessor()
    success_count = processor.process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        mode='building',
        num_workers=num_workers
    )
    
    logger.info(f"✅ Enrichissement terminé: {success_count} fichiers")


def example_parallel_processing():
    """
    Exemple 3: Traitement complet avec LiDARProcessor.
    
    Utilise le LiDARProcessor avec workers pour créer des patchs
    d'entraînement à partir de tuiles enrichies.
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXEMPLE 3: Traitement complet avec LiDARProcessor")
    logger.info("=" * 70)
    
    input_dir = Path("data/enriched_tiles")
    output_dir = Path("data/training_patches")
    
    # Vérifier les fichiers
    laz_files = list(input_dir.glob("*.laz"))
    if not laz_files:
        logger.warning("Aucun fichier LAZ enrichi trouvé")
        return
    
    logger.info(f"Trouvé {len(laz_files)} fichiers LAZ enrichis")
    
    # Créer le processeur
    processor = LiDARProcessor(
        lod_level='LOD2',
        augment=True,
        num_augmentations=2,
        patch_size=150.0,
        include_extra_features=True
    )
    
    # Déterminer le nombre de workers
    num_workers = min(mp.cpu_count() - 1, len(laz_files))
    logger.info(f"Traitement avec {num_workers} workers")
    
    # Traiter
    total_patches = processor.process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        num_workers=num_workers
    )
    
    logger.info(f"✅ Traitement terminé: {total_patches} patchs créés")


def example_adaptive_workers():
    """
    Exemple 4: Adaptation automatique du nombre de workers.
    
    Montre comment adapter le nombre de workers en fonction
    des ressources disponibles et de la charge de travail.
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXEMPLE 4: Adaptation automatique des workers")
    logger.info("=" * 70)
    
    import psutil
    
    # Analyser les ressources
    cpu_count = mp.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    logger.info("Ressources disponibles:")
    logger.info(f"  - CPUs: {cpu_count}")
    logger.info(f"  - CPU Usage: {cpu_percent}%")
    logger.info(f"  - RAM: {memory.available / (1024**3):.1f} GB disponible")
    logger.info(f"  - RAM: {memory.percent}% utilisée")
    
    # Adapter le nombre de workers
    # Règle: 500 MB par worker pour enrichissement
    memory_per_worker_gb = 0.5
    max_workers_memory = int(
        (memory.available / (1024**3)) / memory_per_worker_gb
    )
    
    # Règle: Laisser 1 CPU libre pour le système
    max_workers_cpu = max(1, cpu_count - 1)
    
    # Choisir le minimum
    optimal_workers = min(max_workers_cpu, max_workers_memory)
    
    logger.info(f"\nRecommandations:")
    logger.info(f"  - Max workers (CPU): {max_workers_cpu}")
    logger.info(f"  - Max workers (RAM): {max_workers_memory}")
    logger.info(f"  - Optimal: {optimal_workers}")
    
    # Si CPU très chargé, réduire
    if cpu_percent > 80:
        optimal_workers = max(1, optimal_workers // 2)
        logger.warning(
            f"  - CPU chargé ({cpu_percent}%), "
            f"réduction à {optimal_workers} workers"
        )
    
    return optimal_workers


def example_monitoring_progress():
    """
    Exemple 5: Monitoring de la progression avec tqdm.
    
    Montre comment monitorer la progression du traitement parallèle.
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXEMPLE 5: Monitoring de la progression")
    logger.info("=" * 70)
    
    from tqdm import tqdm
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import time
    
    def process_file(file_idx: int) -> dict:
        """Simule le traitement d'un fichier."""
        time.sleep(0.5)  # Simule le travail
        return {
            'file_idx': file_idx,
            'success': True,
            'points': 1_000_000 + file_idx * 10000
        }
    
    # Simuler 20 fichiers à traiter
    num_files = 20
    num_workers = 4
    
    logger.info(f"Traitement de {num_files} fichiers avec {num_workers} workers")
    
    results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Soumettre toutes les tâches
        futures = {
            executor.submit(process_file, i): i
            for i in range(num_files)
        }
        
        # Collecter les résultats avec barre de progression
        with tqdm(total=num_files, desc="Traitement") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)
    
    logger.info(f"✅ Terminé: {len(results)} fichiers traités")
    total_points = sum(r['points'] for r in results)
    logger.info(f"   Total points traités: {total_points:,}")


def main():
    """Point d'entrée principal."""
    logger.info("=" * 70)
    logger.info("DÉMONSTRATION DU TRAITEMENT PARALLÈLE")
    logger.info("=" * 70)
    logger.info("")
    
    # Exemple 1: Téléchargement parallèle
    # example_parallel_download()
    
    # Exemple 2: Enrichissement parallèle
    # example_parallel_enrichment()
    
    # Exemple 3: Traitement complet
    # example_parallel_processing()
    
    # Exemple 4: Adaptation automatique
    optimal_workers = example_adaptive_workers()
    
    # Exemple 5: Monitoring
    example_monitoring_progress()
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ Démonstration terminée!")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Pour utiliser dans votre code:")
    logger.info("")
    logger.info("1. Téléchargement parallèle:")
    logger.info("   downloader.batch_download(tiles, num_workers=3)")
    logger.info("")
    logger.info("2. Enrichissement parallèle:")
    logger.info("   python enrich_laz_building.py input/ output/ --workers 4")
    logger.info("")
    logger.info("3. Workflow complet:")
    logger.info("   python workflow_100_tiles_building.py --workers 4")
    logger.info("")
    logger.info(f"Workers recommandés pour votre système: {optimal_workers}")


if __name__ == "__main__":
    main()
