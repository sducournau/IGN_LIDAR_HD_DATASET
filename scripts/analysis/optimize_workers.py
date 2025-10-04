#!/usr/bin/env python3
"""
Script pour déterminer le nombre optimal de workers pour votre système.

Usage:
    python scripts/optimize_workers.py
    python scripts/optimize_workers.py --task enrichment
    python scripts/optimize_workers.py --task download
"""

import argparse
import sys
import multiprocessing as mp
from pathlib import Path

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️  psutil non installé, recommandations limitées")
    print("   Installez avec: pip install psutil")
    print()


def get_system_info():
    """Récupérer les informations système."""
    info = {
        'cpu_count': mp.cpu_count(),
        'cpu_physical': mp.cpu_count(),  # Approximation
    }
    
    if HAS_PSUTIL:
        info['cpu_physical'] = psutil.cpu_count(logical=False) or mp.cpu_count()
        info['cpu_percent'] = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        info['memory_total_gb'] = memory.total / (1024**3)
        info['memory_available_gb'] = memory.available / (1024**3)
        info['memory_percent'] = memory.percent
        
        # Swap
        swap = psutil.swap_memory()
        info['swap_total_gb'] = swap.total / (1024**3)
        info['swap_percent'] = swap.percent
    
    return info


def recommend_workers_enrichment(info):
    """
    Recommander le nombre de workers pour enrichissement.
    
    Contraintes:
    - CPU bound: utiliser les cores physiques
    - Mémoire: ~500 MB par worker
    - Laisser 1-2 cores libres pour le système
    """
    cpu_count = info['cpu_count']
    cpu_physical = info['cpu_physical']
    
    # Basé sur CPU (cores physiques - 1)
    workers_cpu = max(1, cpu_physical - 1)
    
    if HAS_PSUTIL:
        memory_available = info['memory_available_gb']
        
        # Basé sur RAM (500 MB par worker)
        workers_memory = int(memory_available / 0.5)
        
        # Si CPU chargé, réduire
        if info['cpu_percent'] > 70:
            workers_cpu = max(1, workers_cpu // 2)
        
        # Si mémoire limitée, prioriser
        optimal = min(workers_cpu, workers_memory)
        
        return {
            'optimal': optimal,
            'conservative': max(1, optimal // 2),
            'aggressive': min(cpu_count, optimal * 2),
            'workers_cpu': workers_cpu,
            'workers_memory': workers_memory,
        }
    else:
        return {
            'optimal': workers_cpu,
            'conservative': max(1, workers_cpu // 2),
            'aggressive': min(cpu_count, workers_cpu * 2),
        }


def recommend_workers_download(info):
    """
    Recommander le nombre de workers pour téléchargement.
    
    Contraintes:
    - I/O bound: pas limité par CPU
    - Limité par réseau et serveur IGN
    - 3-5 connexions simultanées recommandé
    """
    # Pour I/O, pas directement lié au CPU
    base_workers = 3
    
    if HAS_PSUTIL:
        # Si beaucoup de RAM et CPU libres, augmenter
        memory_available = info['memory_available_gb']
        cpu_percent = info['cpu_percent']
        
        if memory_available > 8 and cpu_percent < 50:
            max_workers = 5
        else:
            max_workers = 3
        
        return {
            'optimal': base_workers,
            'conservative': 2,
            'aggressive': max_workers,
        }
    else:
        return {
            'optimal': base_workers,
            'conservative': 2,
            'aggressive': 5,
        }


def recommend_workers_patches(info):
    """
    Recommander le nombre de workers pour création de patchs.
    
    Contraintes:
    - CPU et I/O mixte
    - Moins intensif que enrichissement
    - Peut utiliser plus de workers
    """
    cpu_physical = info['cpu_physical']
    
    # Peut utiliser tous les cores
    workers_cpu = cpu_physical
    
    if HAS_PSUTIL:
        memory_available = info['memory_available_gb']
        
        # Moins de mémoire par worker (~200 MB)
        workers_memory = int(memory_available / 0.2)
        
        optimal = min(workers_cpu, workers_memory)
        
        return {
            'optimal': optimal,
            'conservative': max(1, optimal // 2),
            'aggressive': min(info['cpu_count'], optimal * 2),
        }
    else:
        return {
            'optimal': workers_cpu,
            'conservative': max(1, workers_cpu // 2),
            'aggressive': info['cpu_count'],
        }


def display_recommendations(task, info, recommendations):
    """Afficher les recommandations."""
    print("=" * 70)
    print(f"RECOMMANDATIONS POUR: {task.upper()}")
    print("=" * 70)
    print()
    
    print("💻 Informations système:")
    print(f"   CPUs logiques:  {info['cpu_count']}")
    print(f"   CPUs physiques: {info['cpu_physical']}")
    
    if HAS_PSUTIL:
        print(f"   CPU usage:      {info['cpu_percent']:.1f}%")
        print(f"   RAM totale:     {info['memory_total_gb']:.1f} GB")
        print(f"   RAM disponible: {info['memory_available_gb']:.1f} GB "
              f"({100 - info['memory_percent']:.0f}%)")
        
        if info['swap_total_gb'] > 0:
            print(f"   Swap:           {info['swap_total_gb']:.1f} GB "
                  f"({info['swap_percent']:.0f}% utilisé)")
    print()
    
    print("🎯 Recommandations workers:")
    print(f"   Conservateur:   {recommendations['conservative']} workers")
    print(f"   Optimal:        {recommendations['optimal']} workers ✅")
    print(f"   Agressif:       {recommendations['aggressive']} workers")
    print()
    
    if HAS_PSUTIL and 'workers_cpu' in recommendations:
        print("📊 Détails:")
        print(f"   Limite CPU:     {recommendations.get('workers_cpu', 'N/A')}")
        print(f"   Limite RAM:     {recommendations.get('workers_memory', 'N/A')}")
        print()
    
    # Avertissements
    if HAS_PSUTIL:
        if info['cpu_percent'] > 70:
            print("⚠️  CPU actuellement chargé, recommandations réduites")
        
        if info['memory_percent'] > 80:
            print("⚠️  RAM limitée, risque de swap/OOM")
        
        if info['swap_percent'] > 50:
            print("⚠️  Swap utilisé, performance réduite")
        print()
    
    # Commandes
    print("📝 Commandes à utiliser:")
    
    if task == 'enrichment':
        print(f"   python enrich_laz_building.py input/ output/ \\ ")
        print(f"     --mode full --workers {recommendations['optimal']}")
    elif task == 'download':
        print(f"   python workflow_100_tiles_building.py \\ ")
        print(f"     --download-workers {recommendations['optimal']}")
    elif task == 'patches':
        print(f"   python workflow_100_tiles_building.py \\ ")
        print(f"     --workers {recommendations['optimal']}")
    elif task == 'workflow':
        enrich_workers = recommend_workers_enrichment(info)['optimal']
        download_workers = recommend_workers_download(info)['optimal']
        print(f"   python workflow_100_tiles_building.py \\ ")
        print(f"     --workers {enrich_workers} \\ ")
        print(f"     --download-workers {download_workers}")
    
    print()


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description='Optimiser le nombre de workers pour votre système',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python scripts/optimize_workers.py
  python scripts/optimize_workers.py --task enrichment
  python scripts/optimize_workers.py --task download
  python scripts/optimize_workers.py --task workflow
        """
    )
    
    parser.add_argument(
        '--task',
        choices=['enrichment', 'download', 'patches', 'workflow'],
        default='workflow',
        help='Type de tâche à optimiser (défaut: workflow)'
    )
    
    args = parser.parse_args()
    
    # Récupérer infos système
    print("Analyse de votre système...")
    if not HAS_PSUTIL:
        print("⚠️  Installez psutil pour des recommandations optimales:")
        print("   pip install psutil")
    print()
    
    info = get_system_info()
    
    # Recommandations selon la tâche
    if args.task == 'enrichment':
        recommendations = recommend_workers_enrichment(info)
        display_recommendations('enrichment', info, recommendations)
    
    elif args.task == 'download':
        recommendations = recommend_workers_download(info)
        display_recommendations('download', info, recommendations)
    
    elif args.task == 'patches':
        recommendations = recommend_workers_patches(info)
        display_recommendations('patches', info, recommendations)
    
    elif args.task == 'workflow':
        print("=" * 70)
        print("RECOMMANDATIONS POUR: WORKFLOW COMPLET")
        print("=" * 70)
        print()
        
        print("💻 Informations système:")
        print(f"   CPUs logiques:  {info['cpu_count']}")
        print(f"   CPUs physiques: {info['cpu_physical']}")
        
        if HAS_PSUTIL:
            print(f"   CPU usage:      {info['cpu_percent']:.1f}%")
            print(f"   RAM disponible: {info['memory_available_gb']:.1f} GB")
        print()
        
        # Recommandations par phase
        enrich = recommend_workers_enrichment(info)
        download = recommend_workers_download(info)
        patches = recommend_workers_patches(info)
        
        print("🎯 Recommandations par phase:")
        print(f"   Téléchargement: {download['optimal']} workers")
        print(f"   Enrichissement: {enrich['optimal']} workers")
        print(f"   Patchs:         {patches['optimal']} workers")
        print()
        
        print("📝 Commande complète:")
        print(f"   python workflow_100_tiles_building.py \\ ")
        print(f"     --download-workers {download['optimal']} \\ ")
        print(f"     --workers {enrich['optimal']}")
        print()
    
    print("=" * 70)
    print("✅ Analyse terminée!")
    print("=" * 70)


if __name__ == '__main__':
    main()
