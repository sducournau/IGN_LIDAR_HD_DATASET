"""Auto-Configuration System for IGN LiDAR HD
================================================

This module provides intelligent auto-configuration capabilities that analyze
system capabilities, data characteristics, and user requirements to automatically
generate optimal configuration settings.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
from dataclasses import dataclass

from .optimization_factory import optimization_factory, SystemCapabilities, DataCharacteristics, ProcessingStrategy

logger = logging.getLogger(__name__)


@dataclass
class AutoConfigRecommendation:
    """Container for auto-configuration recommendations."""
    
    strategy: ProcessingStrategy
    config_updates: Dict[str, Any]
    estimated_performance: str
    confidence_score: float  # 0-1, how confident we are in this recommendation
    reasoning: List[str]     # Human-readable explanations
    warnings: List[str]      # Potential issues or limitations
    alternative_configs: List[Dict[str, Any]]  # Alternative configurations


class AutoConfigurationEngine:
    """Auto-configuration engine that provides intelligent configuration
    recommendations based on comprehensive system and data analysis.
    """
    
    def __init__(self):
        self.system_caps = SystemCapabilities()
        logger.info(f"Auto-configuration engine initialized for system: "
                   f"{self.system_caps.cpu_cores}C/{self.system_caps.memory_gb:.0f}GB/"
                   f"GPU:{self.system_caps.gpu_category}")
    
    def analyze_input_data(self, input_path: Path) -> Dict[str, Any]:
        """
        Analyze input data to determine characteristics.
        
        Args:
            input_path: Path to input data directory
            
        Returns:
            Dictionary with data analysis results
        """
        data_info = {
            'total_files': 0,
            'total_size_gb': 0.0,
            'estimated_points': 0,
            'has_laz': False,
            'has_las': False,
            'has_rgb_data': False,
            'sample_point_density': 0,
            'file_sizes': []
        }
        
        try:
            # Count and analyze files
            laz_files = list(input_path.glob("**/*.laz"))
            las_files = list(input_path.glob("**/*.las"))
            
            data_info['has_laz'] = len(laz_files) > 0
            data_info['has_las'] = len(las_files) > 0
            data_info['total_files'] = len(laz_files) + len(las_files)
            
            # Analyze file sizes
            total_bytes = 0
            for file_path in laz_files + las_files:
                try:
                    size = file_path.stat().st_size
                    total_bytes += size
                    data_info['file_sizes'].append(size)
                except OSError:
                    continue
                    
            data_info['total_size_gb'] = total_bytes / (1024**3)
            
            # Estimate points based on file sizes (rough heuristic)
            if data_info['total_size_gb'] > 0:
                # Rough estimate: 1GB LAZ ≈ 25-50 million points
                points_per_gb = 35_000_000  # Conservative estimate
                data_info['estimated_points'] = int(data_info['total_size_gb'] * points_per_gb)
            
            # Look for RGB/orthophoto data
            rgb_patterns = ["*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.png"]
            for pattern in rgb_patterns:
                if list(input_path.glob(f"**/{pattern}")):
                    data_info['has_rgb_data'] = True
                    break
                    
            logger.info(f"Data analysis: {data_info['total_files']} files, "
                       f"{data_info['total_size_gb']:.1f}GB, "
                       f"~{data_info['estimated_points']:,} points")
                       
        except Exception as e:
            logger.warning(f"Could not fully analyze input data: {e}")
            
        return data_info
    
    def estimate_processing_requirements(self, data_info: Dict[str, Any], 
                                       config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate processing requirements based on data and configuration.
        
        Args:
            data_info: Data analysis results
            config: Configuration settings
            
        Returns:
            Dictionary with processing requirements
        """
        requirements = {
            'memory_gb_required': 0.0,
            'gpu_memory_gb_required': 0.0,
            'estimated_time_minutes': 0.0,
            'disk_space_gb_required': 0.0,
            'recommended_workers': 1
        }
        
        try:
            # Estimate memory requirements
            points = data_info.get('estimated_points', 1_000_000)
            
            # Base memory per point (32 bytes for coordinates + features)
            memory_per_point = 32
            
            # Adjust based on features
            feature_mode = config.get('features', {}).get('mode', 'lod2')
            if feature_mode == 'full':
                memory_per_point *= 2
            elif feature_mode == 'lod3':
                memory_per_point *= 1.5
                
            # Adjust for RGB data
            if data_info.get('has_rgb_data') or config.get('features', {}).get('use_rgb'):
                memory_per_point *= 1.3
                
            # Estimate total memory with safety factor
            base_memory_gb = (points * memory_per_point) / (1024**3)
            requirements['memory_gb_required'] = base_memory_gb * 2  # 2x safety factor
            
            # GPU memory requirements
            if config.get('processor', {}).get('use_gpu', False):
                gpu_batch_size = config.get('processor', {}).get('gpu_batch_size', 1_000_000)
                gpu_memory_per_point = memory_per_point * 1.5  # GPU overhead
                requirements['gpu_memory_gb_required'] = (gpu_batch_size * gpu_memory_per_point) / (1024**3)
            
            # Estimate processing time
            base_throughput = 50_000  # Conservative points/sec
            if config.get('processor', {}).get('use_gpu', False):
                base_throughput = 150_000  # GPU acceleration
                
            processing_mode = config.get('output', {}).get('processing_mode', 'patches_only')
            if processing_mode == 'enriched_only':
                base_throughput *= 2  # Faster for enriched only
                
            requirements['estimated_time_minutes'] = points / (base_throughput * 60)
            
            # Disk space requirements
            base_disk_space = data_info.get('total_size_gb', 1.0)
            if processing_mode in ['patches_only', 'both']:
                # Patches can be larger than source data
                requirements['disk_space_gb_required'] = base_disk_space * 3
            else:
                # Enriched LAZ similar to source
                requirements['disk_space_gb_required'] = base_disk_space * 1.5
                
            # Recommended workers
            if config.get('processor', {}).get('use_gpu', False):
                requirements['recommended_workers'] = min(4, self.system_caps.cpu_cores)
            else:
                requirements['recommended_workers'] = min(8, self.system_caps.cpu_cores)
                
        except Exception as e:
            logger.warning(f"Could not estimate processing requirements: {e}")
            
        return requirements
    
    def check_system_compatibility(self, requirements: Dict[str, Any]) -> List[str]:
        """
        Check if system can handle the estimated requirements.
        
        Args:
            requirements: Processing requirements
            
        Returns:
            List of compatibility issues (empty if all good)
        """
        issues = []
        
        # Memory check
        required_memory = requirements.get('memory_gb_required', 0)
        available_memory = self.system_caps.memory_gb * 0.8  # 80% usable
        
        if required_memory > available_memory:
            issues.append(
                f"Insufficient RAM: need {required_memory:.1f}GB, "
                f"have {available_memory:.1f}GB available"
            )
            
        # GPU memory check
        required_gpu_memory = requirements.get('gpu_memory_gb_required', 0)
        if required_gpu_memory > 0:
            if not self.system_caps.gpu_available:
                issues.append("GPU processing requested but no GPU available")
            elif required_gpu_memory > self.system_caps.gpu_memory_gb * 0.8:
                issues.append(
                    f"Insufficient GPU memory: need {required_gpu_memory:.1f}GB, "
                    f"have {self.system_caps.gpu_memory_gb * 0.8:.1f}GB available"
                )
                
        # Disk space check would require analyzing output directory
        # (skipped for now, could be added with output_dir parameter)
        
        return issues
    
    def generate_optimization_recommendations(self, 
                                            data_info: Dict[str, Any],
                                            base_config: Dict[str, Any],
                                            requirements: Dict[str, Any]) -> AutoConfigRecommendation:
        """
        Generate comprehensive optimization recommendations.
        
        Args:
            data_info: Data analysis results
            base_config: Base configuration
            requirements: Processing requirements
            
        Returns:
            AutoConfigRecommendation object
        """
        reasoning = []
        warnings = []
        config_updates = {}
        alternatives = []
        
        # Determine data characteristics
        data_chars = DataCharacteristics(
            points_per_tile=requirements.get('estimated_points', 1_000_000) // max(data_info.get('total_files', 1), 1),
            num_tiles=data_info.get('total_files', 1),
            has_rgb=data_info.get('has_rgb_data', False) or base_config.get('features', {}).get('use_rgb', False),
            feature_complexity="high" if base_config.get('features', {}).get('mode') == 'full' else "medium"
        )
        
        # Get strategy recommendation from optimization factory
        strategy = optimization_factory.select_strategy(data_chars)
        factory_config = optimization_factory.get_optimal_config(strategy, data_chars)
        
        # Merge factory recommendations
        config_updates.update(factory_config)
        reasoning.append(f"Selected {strategy.value} strategy based on data characteristics")
        
        # Memory optimization
        memory_required = requirements.get('memory_gb_required', 0)
        if memory_required > self.system_caps.memory_gb * 0.8:
            # Reduce batch sizes
            if 'gpu_batch_size' in config_updates:
                config_updates['gpu_batch_size'] = min(config_updates['gpu_batch_size'], 500_000)
            config_updates['chunk_size'] = 100_000
            reasoning.append("Reduced batch sizes due to memory constraints")
            warnings.append("Large dataset may require multiple processing runs")
            
        # GPU optimization
        if strategy == ProcessingStrategy.GPU_OPTIMIZED:
            if self.system_caps.gpu_memory_gb < 8:
                config_updates['gpu_batch_size'] = min(config_updates.get('gpu_batch_size', 1_000_000), 500_000)
                reasoning.append("Reduced GPU batch size for lower-memory GPU")
                
        # Worker optimization
        if data_info.get('total_files', 1) < self.system_caps.cpu_cores:
            # More workers than files is wasteful
            config_updates['num_workers'] = min(config_updates.get('num_workers', 4), data_info['total_files'])
            reasoning.append("Adjusted worker count to match number of input files")
            
        # Feature optimization
        if data_chars.has_rgb and not base_config.get('features', {}).get('use_rgb', False):
            alternatives.append({
                'name': 'RGB-Enabled Configuration',
                'updates': {'features': {'use_rgb': True}},
                'description': 'Enable RGB features for better classification accuracy'
            })
            
        # Performance estimation
        estimated_throughput = 50_000  # base
        if strategy == ProcessingStrategy.GPU_OPTIMIZED:
            estimated_throughput = 150_000
        elif strategy == ProcessingStrategy.CPU_PARALLEL:
            estimated_throughput = 80_000
            
        estimated_time = requirements.get('estimated_points', 1_000_000) / estimated_throughput
        
        if estimated_time < 300:  # 5 minutes
            performance_desc = f"Fast processing (~{estimated_time/60:.1f} minutes)"
        elif estimated_time < 3600:  # 1 hour
            performance_desc = f"Moderate processing (~{estimated_time/60:.0f} minutes)"
        else:
            performance_desc = f"Long processing (~{estimated_time/3600:.1f} hours)"
            
        # Confidence score based on data quality and system match
        confidence = 0.8  # base confidence
        
        if data_info.get('total_files', 0) == 0:
            confidence -= 0.3  # Low confidence with no data analysis
            
        if len(self.check_system_compatibility(requirements)) > 0:
            confidence -= 0.2  # Compatibility issues
            
        # Check for potential issues
        compatibility_issues = self.check_system_compatibility(requirements)
        warnings.extend(compatibility_issues)
        
        return AutoConfigRecommendation(
            strategy=strategy,
            config_updates=config_updates,
            estimated_performance=performance_desc,
            confidence_score=confidence,
            reasoning=reasoning,
            warnings=warnings,
            alternative_configs=alternatives
        )
    
    def generate_complete_config(self, 
                               input_dir: Path,
                               output_dir: Path,
                               base_config: Optional[Dict[str, Any]] = None,
                               user_preferences: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], AutoConfigRecommendation]:
        """
        Generate a complete optimized configuration.
        
        Args:
            input_dir: Input data directory
            output_dir: Output directory
            base_config: Base configuration to start from
            user_preferences: User preferences and constraints
            
        Returns:
            Tuple of (optimized_config, recommendation_details)
        """
        # Default base configuration
        if base_config is None:
            base_config = {
                'processor': {
                    'lod_level': 'ASPRS',
                    'processing_mode': 'patches_only',
                    'patch_size': 150.0,
                    'num_points': 16384,
                    'use_gpu': self.system_caps.gpu_available,
                    'num_workers': self.system_caps.parallel_capability
                },
                'features': {
                    'mode': 'lod2',
                    'use_rgb': False
                },
                'output': {
                    'format': 'npz',
                    'processing_mode': 'patches_only'
                }
            }
            
        # Apply user preferences
        if user_preferences:
            self._apply_user_preferences(base_config, user_preferences)
            
        # Analyze input data
        data_info = self.analyze_input_data(input_dir)
        
        # Estimate requirements
        requirements = self.estimate_processing_requirements(data_info, base_config)
        
        # Generate recommendations
        recommendation = self.generate_optimization_recommendations(data_info, base_config, requirements)
        
        # Apply recommended updates to base config
        optimized_config = self._merge_config_updates(base_config, recommendation.config_updates)
        
        # Add auto-configuration metadata
        optimized_config['_auto_config'] = {
            'generated_by': 'IGN LiDAR HD Auto-Configuration v4.0',
            'system_info': {
                'cpu_cores': self.system_caps.cpu_cores,
                'memory_gb': self.system_caps.memory_gb,
                'gpu_available': self.system_caps.gpu_available,
                'gpu_memory_gb': self.system_caps.gpu_memory_gb
            },
            'data_info': data_info,
            'requirements': requirements,
            'strategy': recommendation.strategy.value,
            'confidence': recommendation.confidence_score
        }
        
        # Set paths
        optimized_config['input_dir'] = str(input_dir)
        optimized_config['output_dir'] = str(output_dir)
        
        return optimized_config, recommendation
    
    def _apply_user_preferences(self, config: Dict[str, Any], preferences: Dict[str, Any]):
        """Apply user preferences to configuration."""
        if preferences.get('force_cpu', False):
            config['processor']['use_gpu'] = False
            
        if preferences.get('force_gpu', False):
            config['processor']['use_gpu'] = True
            
        if preferences.get('processing_mode'):
            config['output']['processing_mode'] = preferences['processing_mode']
            
        if preferences.get('enable_rgb', False):
            config['features']['use_rgb'] = True
            
        if preferences.get('num_workers'):
            config['processor']['num_workers'] = preferences['num_workers']
            
    def _merge_config_updates(self, base_config: Dict[str, Any], 
                            updates: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration updates into base configuration."""
        import copy
        result = copy.deepcopy(base_config)
        
        for key, value in updates.items():
            if key == 'architecture':
                if 'processing' not in result:
                    result['processing'] = {}
                result['processing']['architecture'] = value
            else:
                if 'processor' not in result:
                    result['processor'] = {}
                result['processor'][key] = value
                
        return result


# Global instance for easy access
auto_config_engine = AutoConfigurationEngine()


def generate_auto_config(input_dir: Path, 
                        output_dir: Path,
                        base_config: Optional[Dict[str, Any]] = None,
                        user_preferences: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], AutoConfigRecommendation]:
    """
    Convenience function to generate auto-configuration.
    
    Args:
        input_dir: Input data directory
        output_dir: Output directory  
        base_config: Base configuration to start from
        user_preferences: User preferences and constraints
        
    Returns:
        Tuple of (optimized_config, recommendation_details)
    """
    return auto_config_engine.generate_complete_config(
        input_dir=input_dir,
        output_dir=output_dir,
        base_config=base_config,
        user_preferences=user_preferences
    )


def save_auto_config(config: Dict[str, Any], 
                    output_path: Path,
                    recommendation: Optional[AutoConfigRecommendation] = None):
    """
    Save auto-generated configuration to file with documentation.
    
    Args:
        config: Configuration to save
        output_path: Output file path (.yaml or .json)
        recommendation: Recommendation details for documentation
    """
    from omegaconf import OmegaConf
    
    # Add header documentation
    header_comment = [
        "IGN LiDAR HD - Auto-Generated Configuration",
        "=" * 50,
        "This configuration was automatically generated based on:",
        f"- System capabilities: {auto_config_engine.system_caps.cpu_cores}C/{auto_config_engine.system_caps.memory_gb:.0f}GB",
        f"- GPU: {auto_config_engine.system_caps.gpu_category}",
        ""
    ]
    
    if recommendation:
        header_comment.extend([
            f"Strategy: {recommendation.strategy.value}",
            f"Confidence: {recommendation.confidence_score:.1%}",
            f"Expected performance: {recommendation.estimated_performance}",
            ""
        ])
        
        if recommendation.reasoning:
            header_comment.extend(["Reasoning:"] + [f"- {r}" for r in recommendation.reasoning] + [""])
            
        if recommendation.warnings:
            header_comment.extend(["Warnings:"] + [f"- {w}" for w in recommendation.warnings] + [""])
    
    # Convert to OmegaConf and save
    conf = OmegaConf.create(config)
    
    if output_path.suffix.lower() == '.yaml':
        with open(output_path, 'w') as f:
            # Write header as comments
            for line in header_comment:
                f.write(f"# {line}\n")
            f.write("\n")
            f.write(OmegaConf.to_yaml(conf))
    else:
        # JSON format
        with open(output_path, 'w') as f:
            json.dump(OmegaConf.to_object(conf), f, indent=2)
            
    logger.info(f"Auto-configuration saved to: {output_path}")


def print_auto_config_summary(recommendation: AutoConfigRecommendation):
    """Print a human-readable summary of auto-configuration recommendations."""
    print("\n" + "="*70)
    print("🧠 AUTO-CONFIGURATION RECOMMENDATIONS")
    print("="*70)
    
    print(f"📊 Strategy: {recommendation.strategy.value}")
    print(f"🎯 Confidence: {recommendation.confidence_score:.1%}")
    print(f"⚡ Expected Performance: {recommendation.estimated_performance}")
    
    if recommendation.reasoning:
        print("\n💡 Reasoning:")
        for reason in recommendation.reasoning:
            print(f"  • {reason}")
            
    if recommendation.config_updates:
        print("\n🔧 Key Optimizations:")
        for key, value in recommendation.config_updates.items():
            print(f"  • {key}: {value}")
            
    if recommendation.warnings:
        print("\n⚠️  Warnings:")
        for warning in recommendation.warnings:
            print(f"  • {warning}")
            
    if recommendation.alternative_configs:
        print("\n🔄 Alternative Configurations:")
        for alt in recommendation.alternative_configs:
            print(f"  • {alt['name']}: {alt['description']}")
            
    print("="*70)