# Example: Full Enrichment with All Data Sources
# 
# This script demonstrates how to enrich LiDAR tiles with:
# - BD TOPO® (buildings, roads, railways, water, etc.)
# - BD Forêt® (forest types and species)
# - RPG (agricultural parcels and crops)
# - BD PARCELLAIRE (cadastral parcels)
#
# Date: October 15, 2025

from pathlib import Path
import logging
from omegaconf import OmegaConf

from ign_lidar.core.processor import LiDARProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run full enrichment with all data sources."""
    
    # Load configuration
    config_path = Path("configs/enrichment_asprs_full.yaml")
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        logger.info("Please create the configuration file or use a different path")
        return
    
    logger.info(f"Loading configuration from: {config_path}")
    config = OmegaConf.load(config_path)
    
    # Override paths if needed (optional)
    # Uncomment and modify these lines to override config file paths:
    # config.input_dir = "D:/ign/raw"
    # config.output_dir = "D:/ign/preprocessed/asprs"
    # config.cache_dir = "D:/ign/cache"
    
    logger.info("="*70)
    logger.info("IGN LiDAR HD - Full Enrichment Pipeline")
    logger.info("="*70)
    logger.info(f"Input:  {config.input_dir}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Cache:  {config.cache_dir}")
    logger.info(f"Mode:   {config.mode}")
    logger.info("="*70)
    
    # Display enabled data sources
    logger.info("\nEnabled data sources:")
    if config.data_sources.bd_topo.enabled:
        logger.info("  ✓ BD TOPO® V3 (infrastructure)")
        features = config.data_sources.bd_topo.features
        enabled_features = [k for k, v in features.items() if v]
        logger.info(f"    Features: {', '.join(enabled_features)}")
    
    if config.data_sources.bd_foret.enabled:
        logger.info("  ✓ BD Forêt® V2 (forest types and species)")
    
    if config.data_sources.rpg.enabled:
        logger.info(f"  ✓ RPG {config.data_sources.rpg.year} (agriculture)")
    
    if config.data_sources.cadastre.enabled:
        logger.info("  ✓ BD PARCELLAIRE (cadastre)")
    
    logger.info("")
    
    # Create processor
    logger.info("Initializing LiDAR processor...")
    processor = LiDARProcessor(config=config)
    
    # Process tiles
    logger.info("Starting tile processing...")
    logger.info("="*70)
    
    try:
        total_tiles = processor.process_directory(
            input_dir=Path(config.input_dir),
            output_dir=Path(config.output_dir),
            num_workers=config.processor.num_workers,
            skip_existing=True  # Set to False to reprocess existing files
        )
        
        logger.info("="*70)
        logger.info("✅ Processing complete!")
        logger.info(f"  Total tiles processed: {total_tiles}")
        logger.info(f"  Enriched tiles: {config.output_dir}/enriched_tiles/enriched/")
        
        if config.output.save_statistics:
            logger.info(f"  Reports: {config.output_dir}/reports/")
        
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
