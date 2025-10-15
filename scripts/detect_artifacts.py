#!/usr/bin/env python3
"""
Detect and Visualize Artifacts in LAZ Files

This script provides an easy-to-use interface for detecting scan line artifacts
and other spatial patterns in processed LiDAR tiles.

Features:
- Dash line detection (scan line artifacts perpendicular to flight direction)
- Coefficient of variation (CV) metrics for artifact severity
- Comprehensive visualizations with 2D heatmaps and profiles
- Automatic recommendations for field dropping
- Batch processing support
- CSV reports for batch analysis

Usage Examples:

  # Analyze a single file
  python detect_artifacts.py --input data/tile.laz
  
  # Analyze specific features
  python detect_artifacts.py --input data/tile.laz --features planarity,roof_score
  
  # Batch analyze all files in directory
  python detect_artifacts.py --input data/ --batch --output artifacts_report/
  
  # Lower threshold for stricter artifact detection
  python detect_artifacts.py --input data/tile.laz --threshold 0.30
  
  # Skip visualization (faster for large batches)
  python detect_artifacts.py --input data/ --batch --no-visualize
"""

import sys
import argparse
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.preprocessing.artifact_detector import (
    ArtifactDetector,
    ArtifactDetectorConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for artifact detection CLI."""
    parser = argparse.ArgumentParser(
        description='Detect and visualize artifacts in LiDAR point cloud features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input/Output
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input LAZ file or directory (for batch mode)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='artifact_analysis',
        help='Output directory for visualizations and reports (default: artifact_analysis)'
    )
    
    # Features
    parser.add_argument(
        '--features',
        type=str,
        default=None,
        help='Comma-separated list of features to check (default: all available features)'
    )
    
    # Thresholds
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.40,
        help='CV threshold for auto-drop recommendation (default: 0.40)'
    )
    parser.add_argument(
        '--review-threshold',
        type=float,
        default=0.25,
        help='CV threshold for review warning (default: 0.25)'
    )
    
    # Processing mode
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch process all LAZ files in input directory'
    )
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Skip visualization generation (faster for large batches)'
    )
    
    # Grid parameters
    parser.add_argument(
        '--grid-size',
        type=int,
        default=50,
        help='Grid resolution for 2D heatmaps (default: 50)'
    )
    parser.add_argument(
        '--n-bins-y',
        type=int,
        default=50,
        help='Number of bins for Y-direction profile (default: 50)'
    )
    
    # Visualization
    parser.add_argument(
        '--show-dash-lines',
        action='store_true',
        default=True,
        help='Overlay detected dash lines on heatmaps (default: enabled)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='DPI for saved figures (default: 150)'
    )
    
    args = parser.parse_args()
    
    # Setup configuration
    config = ArtifactDetectorConfig()
    config.auto_drop_threshold = args.threshold
    config.review_threshold = args.review_threshold
    config.grid_size = args.grid_size
    config.n_bins_y = args.n_bins_y
    config.show_dash_lines = args.show_dash_lines
    config.plot_dpi = args.dpi
    
    # Initialize detector
    detector = ArtifactDetector(config)
    
    # Setup paths
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse features
    features_to_check = None
    if args.features:
        features_to_check = [f.strip() for f in args.features.split(',')]
        logger.info(f"Checking features: {features_to_check}")
    
    # Process
    if args.batch or input_path.is_dir():
        # ========== BATCH MODE ==========
        logger.info(f"🔍 Starting batch artifact detection in: {input_path}")
        
        # Find all LAZ files
        laz_files = list(input_path.glob('*.laz')) + list(input_path.glob('*.LAZ'))
        
        if not laz_files:
            logger.error(f"No LAZ files found in {input_path}")
            return 1
        
        logger.info(f"Found {len(laz_files)} LAZ files to process")
        
        # Batch analyze
        visualize_output = None if args.no_visualize else output_dir
        results = detector.batch_analyze(
            laz_files,
            features_to_check=features_to_check,
            output_dir=visualize_output
        )
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"BATCH ANALYSIS COMPLETE: {len(results)} files processed")
        print(f"{'='*80}\n")
        
        # Summary statistics
        total_features = 0
        features_to_drop = []
        features_to_review = []
        
        for file_path, file_results in results.items():
            filename = Path(file_path).name
            total_features += len(file_results)
            
            for feat_name, metrics in file_results.items():
                if metrics.recommended_action == 'drop':
                    features_to_drop.append(f"{filename}:{feat_name}")
                elif metrics.recommended_action == 'review':
                    features_to_review.append(f"{filename}:{feat_name}")
        
        print(f"Total features analyzed: {total_features}")
        print(f"Features flagged for DROP: {len(features_to_drop)}")
        print(f"Features flagged for REVIEW: {len(features_to_review)}")
        
        if features_to_drop:
            print(f"\n⚠️  FEATURES TO DROP (CV > {args.threshold}):")
            for item in features_to_drop[:10]:  # Show first 10
                print(f"  - {item}")
            if len(features_to_drop) > 10:
                print(f"  ... and {len(features_to_drop) - 10} more")
        
        if features_to_review:
            print(f"\n⚡ FEATURES TO REVIEW (CV > {args.review_threshold}):")
            for item in features_to_review[:10]:
                print(f"  - {item}")
            if len(features_to_review) > 10:
                print(f"  ... and {len(features_to_review) - 10} more")
        
        if not features_to_drop and not features_to_review:
            print(f"\n✅ All features are acceptable (no issues detected)")
        
        print(f"\nDetailed report saved to: {output_dir / 'artifact_analysis_report.csv'}")
        
        if not args.no_visualize:
            print(f"Visualizations saved to: {output_dir}")
        
    else:
        # ========== SINGLE FILE MODE ==========
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return 1
        
        logger.info(f"🔍 Analyzing artifacts in: {input_path.name}")
        
        # Analyze single file
        visualize_output = output_dir if not args.no_visualize else None
        results = detector.analyze_file(
            input_path,
            features_to_check=features_to_check,
            visualize=not args.no_visualize,
            output_dir=visualize_output
        )
        
        # Print results
        print(f"\n{'='*80}")
        print(f"ARTIFACT ANALYSIS: {input_path.name}")
        print(f"{'='*80}\n")
        
        if not results:
            print("⚠️  No features found to analyze")
            return 1
        
        # Table header
        print(f"{'Feature':<25} | {'CV_Y':>8} | {'Severity':<10} | {'Action':<10}")
        print(f"{'-'*25}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")
        
        # Sort by CV (worst first)
        sorted_results = sorted(results.items(), key=lambda x: x[1].cv_y, reverse=True)
        
        for feat_name, metrics in sorted_results:
            severity_color = {
                'low': '',
                'medium': '⚡',
                'high': '⚠️ ',
                'severe': '🔥'
            }
            action_symbol = {
                'keep': '✅',
                'review': '⚡',
                'drop': '❌'
            }
            
            print(f"{feat_name:<25} | {metrics.cv_y:8.4f} | "
                  f"{severity_color[metrics.severity]}{metrics.severity:<9} | "
                  f"{action_symbol[metrics.recommended_action]} {metrics.recommended_action.upper()}")
        
        # Summary
        drop_list = detector.get_fields_to_drop(results)
        review_list = [name for name, m in results.items() 
                      if m.recommended_action == 'review']
        
        print(f"\n{'='*80}")
        if drop_list:
            print(f"❌ RECOMMENDED TO DROP ({len(drop_list)}): {', '.join(drop_list)}")
        if review_list:
            print(f"⚡ REVIEW RECOMMENDED ({len(review_list)}): {', '.join(review_list)}")
        if not drop_list and not review_list:
            print(f"✅ All features are acceptable (no dropping needed)")
        print(f"{'='*80}\n")
        
        if not args.no_visualize:
            print(f"Visualizations saved to: {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
