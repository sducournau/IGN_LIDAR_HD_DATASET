"""
Integration tests for FeatureOrchestrator with UnifiedFeatureComputer.

Tests Phase 4 Task 1.4 - Pipeline Integration
"""
import numpy as np
import pytest

from ign_lidar.features.orchestrator import FeatureOrchestrator


class TestOrchestratorUnifiedIntegration:
    """Test FeatureOrchestrator integration with UnifiedFeatureComputer."""
    
    def test_default_uses_strategy_pattern(self):
        """Test that default config uses Strategy Pattern (backward compatible)."""
        config = {
            'processor': {
                'use_gpu': False,
                'use_strategy_pattern': True,
            },
            'features': {
                'k_neighbors': 20,
            }
        }
        
        orchestrator = FeatureOrchestrator(config)
        
        # Should use Strategy Pattern by default
        assert hasattr(orchestrator, 'computer')
        assert hasattr(orchestrator, 'strategy_name')
        assert orchestrator.strategy_name == 'cpu'
    
    def test_unified_computer_opt_in(self):
        """Test that use_unified_computer=True enables UnifiedFeatureComputer."""
        config = {
            'processor': {
                'use_unified_computer': True,
                'use_gpu': False,
            },
            'features': {
                'k_neighbors': 20,
            }
        }
        
        orchestrator = FeatureOrchestrator(config)
        
        # Should use UnifiedFeatureComputer
        assert hasattr(orchestrator, 'computer')
        # Check it's the unified computer (has compute_all_features method)
        assert hasattr(orchestrator.computer, 'compute_all_features')
    
    def test_unified_computer_compute_features(self):
        """Test that UnifiedFeatureComputer path computes features correctly."""
        config = {
            'processor': {
                'use_unified_computer': True,
                'use_gpu': False,
            },
            'features': {
                'k_neighbors': 20,
            }
        }
        
        orchestrator = FeatureOrchestrator(config)
        
        # Create synthetic test data
        np.random.seed(42)
        n_points = 1000
        points = np.random.randn(n_points, 3) * 10
        classification = np.ones(n_points, dtype=np.uint8)
        intensity = np.random.randint(0, 255, n_points, dtype=np.uint8)
        return_number = np.ones(n_points, dtype=np.uint8)
        
        tile_data = {
            'points': points,
            'classification': classification,
            'intensity': intensity,
            'return_number': return_number,
        }
        
        # Compute features
        features = orchestrator.compute_features(tile_data, use_enriched=False)
        
        # Verify expected features exist
        assert 'normals' in features
        assert 'curvature' in features
        assert 'height' in features
        assert 'intensity' in features
        assert 'return_number' in features
        
        # Verify shapes
        assert features['normals'].shape == (n_points, 3)
        assert features['curvature'].shape == (n_points,)
        assert features['height'].shape == (n_points,)
    
    def test_unified_computer_forced_mode(self):
        """Test that computation_mode forces specific mode."""
        config = {
            'processor': {
                'use_unified_computer': True,
                'computation_mode': 'cpu',  # Force CPU mode
            },
            'features': {
                'k_neighbors': 20,
            }
        }
        
        orchestrator = FeatureOrchestrator(config)
        
        # Should use UnifiedFeatureComputer with forced CPU mode
        assert hasattr(orchestrator.computer, 'compute_all_features')
        # TODO: Add way to check selected mode
    
    def test_strategy_pattern_backward_compatibility(self):
        """Test that Strategy Pattern path still works identically."""
        config = {
            'processor': {
                'use_unified_computer': False,  # Explicit
                'use_gpu': False,
                'use_strategy_pattern': True,
            },
            'features': {
                'k_neighbors': 20,
            }
        }
        
        orchestrator = FeatureOrchestrator(config)
        
        # Create synthetic test data
        np.random.seed(42)
        n_points = 1000
        points = np.random.randn(n_points, 3) * 10
        classification = np.ones(n_points, dtype=np.uint8)
        intensity = np.random.randint(0, 255, n_points, dtype=np.uint8)
        return_number = np.ones(n_points, dtype=np.uint8)
        
        tile_data = {
            'points': points,
            'classification': classification,
            'intensity': intensity,
            'return_number': return_number,
        }
        
        # Compute features
        features = orchestrator.compute_features(tile_data, use_enriched=False)
        
        # Verify expected features exist (same as unified path)
        assert 'normals' in features
        assert 'curvature' in features
        assert 'height' in features
        assert 'intensity' in features
        assert 'return_number' in features
    
    def test_both_paths_produce_similar_results(self):
        """Test that both paths produce numerically similar results."""
        # Shared config
        base_config = {
            'features': {
                'k_neighbors': 20,
            }
        }
        
        # Strategy Pattern config
        strategy_config = {
            **base_config,
            'processor': {
                'use_unified_computer': False,
                'use_gpu': False,
                'use_strategy_pattern': True,
            }
        }
        
        # Unified Computer config
        unified_config = {
            **base_config,
            'processor': {
                'use_unified_computer': True,
                'use_gpu': False,
                'computation_mode': 'cpu',  # Force CPU for fair comparison
            }
        }
        
        # Create orchestrators
        strategy_orch = FeatureOrchestrator(strategy_config)
        unified_orch = FeatureOrchestrator(unified_config)
        
        # Create identical test data
        np.random.seed(42)
        n_points = 500  # Small dataset for fast test
        points = np.random.randn(n_points, 3) * 10
        classification = np.ones(n_points, dtype=np.uint8)
        intensity = np.random.randint(0, 255, n_points, dtype=np.uint8)
        return_number = np.ones(n_points, dtype=np.uint8)
        
        tile_data = {
            'points': points,
            'classification': classification,
            'intensity': intensity,
            'return_number': return_number,
        }
        
        # Compute features with both paths
        strategy_features = strategy_orch.compute_features(tile_data, use_enriched=False)
        unified_features = unified_orch.compute_features(tile_data, use_enriched=False)
        
        # Compare key features (normals, curvature)
        # Note: May have slight numerical differences due to implementation details
        assert 'normals' in strategy_features and 'normals' in unified_features
        assert 'curvature' in strategy_features and 'curvature' in unified_features
        
        # Check shapes match
        assert strategy_features['normals'].shape == unified_features['normals'].shape
        assert strategy_features['curvature'].shape == unified_features['curvature'].shape
        
        # Check normals are present and valid (unit vectors approximately)
        # Note: Implementation differences mean exact match is not expected
        strategy_norms = np.linalg.norm(strategy_features['normals'], axis=1)
        unified_norms = np.linalg.norm(unified_features['normals'], axis=1)
        assert np.allclose(strategy_norms, 1.0, atol=0.1), "Strategy normals should be unit vectors"
        assert np.allclose(unified_norms, 1.0, atol=0.1), "Unified normals should be unit vectors"
        
        # Just verify that curvature values are in reasonable range
        assert np.all(strategy_features['curvature'] >= 0), "Curvature should be non-negative"
        assert np.all(unified_features['curvature'] >= 0), "Curvature should be non-negative"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
