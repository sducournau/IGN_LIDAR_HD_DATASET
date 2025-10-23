"""
Tests for rules framework base classes and data structures.

This module tests the foundational components of the rules framework:
- Enumerations (RuleType, RulePriority, etc.)
- Data classes (RuleResult, RuleStats, RuleConfig)
- Abstract base classes (BaseRule, RuleEngine)
- Utility functions (create_empty_result, merge_rule_results)
"""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

from ign_lidar.core.classification.rules.base import (
    # Enums
    RuleType,
    RulePriority,
    ExecutionStrategy,
    ConflictResolution,
    
    # Data classes
    RuleStats,
    RuleResult,
    RuleConfig,
    RuleEngineConfig,
    
    # Base classes
    BaseRule,
    RuleEngine,
    
    # Utilities
    create_empty_result,
    merge_rule_results,
)


# ================================
# Test Fixtures
# ================================

@pytest.fixture
def sample_points():
    """Sample point cloud for testing"""
    return np.random.rand(100, 3) * 10  # 100 points in 10x10x10 space


@pytest.fixture
def sample_features():
    """Sample feature dictionary"""
    n_points = 100
    return {
        'height': np.random.rand(n_points) * 10,
        'planarity': np.random.rand(n_points),
        'ndvi': np.random.rand(n_points) * 2 - 1,  # -1 to 1
        'normals': np.random.rand(n_points, 3),
    }


@pytest.fixture
def sample_labels():
    """Sample classification labels"""
    return np.random.randint(0, 10, size=100)


@pytest.fixture
def sample_confidence():
    """Sample confidence scores"""
    return np.random.rand(100)


@pytest.fixture
def sample_rule_stats():
    """Sample rule statistics"""
    return RuleStats(
        total_points=100,
        matched_points=75,
        unmatched_points=25,
        rules_applied=3,
        execution_time_ms=15.5,
        mean_confidence=0.85,
        min_confidence=0.5,
        max_confidence=1.0,
    )


@pytest.fixture
def sample_rule_result(sample_labels, sample_confidence, sample_rule_stats):
    """Sample rule result"""
    return RuleResult(
        labels=sample_labels,
        confidence=sample_confidence,
        rule_ids=['rule1', 'rule2'],
        stats=sample_rule_stats,
    )


# ================================
# Test Enumerations
# ================================

class TestEnumerations:
    """Test all rule framework enumerations"""
    
    def test_rule_type_enum(self):
        """Test RuleType enum values"""
        assert RuleType.GEOMETRIC == "geometric"
        assert RuleType.SPECTRAL == "spectral"
        assert RuleType.GRAMMAR == "grammar"
        assert RuleType.HYBRID == "hybrid"
        assert RuleType.CONTEXTUAL == "contextual"
        assert RuleType.TEMPORAL == "temporal"
        
        # Test all values are accessible
        assert len(RuleType) == 6
    
    def test_rule_priority_enum(self):
        """Test RulePriority enum values and ordering"""
        assert RulePriority.LOW.value == 1
        assert RulePriority.MEDIUM.value == 2
        assert RulePriority.HIGH.value == 3
        assert RulePriority.CRITICAL.value == 4
        
        # Test ordering via values
        assert RulePriority.LOW.value < RulePriority.MEDIUM.value
        assert RulePriority.HIGH.value < RulePriority.CRITICAL.value
    
    def test_execution_strategy_enum(self):
        """Test ExecutionStrategy enum values"""
        assert ExecutionStrategy.FIRST_MATCH == "first_match"
        assert ExecutionStrategy.ALL_MATCHES == "all_matches"
        assert ExecutionStrategy.PRIORITY == "priority"
        assert ExecutionStrategy.WEIGHTED == "weighted"
        assert ExecutionStrategy.HIERARCHICAL == "hierarchical"
        
        assert len(ExecutionStrategy) == 5
    
    def test_conflict_resolution_enum(self):
        """Test ConflictResolution enum values"""
        assert ConflictResolution.HIGHEST_PRIORITY == "highest_priority"
        assert ConflictResolution.HIGHEST_CONFIDENCE == "highest_confidence"
        assert ConflictResolution.WEIGHTED_VOTE == "weighted_vote"
        assert ConflictResolution.FIRST_WINS == "first_wins"
        assert ConflictResolution.LAST_WINS == "last_wins"
        
        assert len(ConflictResolution) == 5


# ================================
# Test Data Classes
# ================================

class TestRuleStats:
    """Test RuleStats dataclass"""
    
    def test_basic_creation(self):
        """Test creating RuleStats with required fields"""
        stats = RuleStats(
            total_points=100,
            matched_points=75,
            unmatched_points=25,
            rules_applied=3,
        )
        
        assert stats.total_points == 100
        assert stats.matched_points == 75
        assert stats.unmatched_points == 25
        assert stats.rules_applied == 3
        assert stats.execution_time_ms == 0.0  # Default value
    
    def test_coverage_calculation(self):
        """Test automatic coverage calculation"""
        stats = RuleStats(
            total_points=100,
            matched_points=75,
            unmatched_points=25,
            rules_applied=2,
        )
        
        assert stats.coverage == 0.75
    
    def test_zero_points_coverage(self):
        """Test coverage when no points"""
        stats = RuleStats(
            total_points=0,
            matched_points=0,
            unmatched_points=0,
            rules_applied=0,
        )
        
        assert stats.coverage == 0.0
    
    def test_per_rule_statistics(self):
        """Test per-rule statistics tracking"""
        stats = RuleStats(
            total_points=100,
            matched_points=75,
            unmatched_points=25,
            rules_applied=2,
            rule_match_counts={'rule1': 50, 'rule2': 25},
            rule_execution_times={'rule1': 10.5, 'rule2': 5.0},
        )
        
        assert stats.rule_match_counts['rule1'] == 50
        assert stats.rule_match_counts['rule2'] == 25
        assert stats.rule_execution_times['rule1'] == 10.5
        assert stats.rule_execution_times['rule2'] == 5.0


class TestRuleResult:
    """Test RuleResult dataclass"""
    
    def test_basic_creation(self, sample_labels, sample_confidence, sample_rule_stats):
        """Test creating RuleResult"""
        result = RuleResult(
            labels=sample_labels,
            confidence=sample_confidence,
            rule_ids=['rule1'],
            stats=sample_rule_stats,
        )
        
        assert len(result.labels) == 100
        assert len(result.confidence) == 100
        assert result.rule_ids == ['rule1']
        assert result.stats == sample_rule_stats
    
    def test_labels_confidence_length_mismatch(self, sample_rule_stats):
        """Test that mismatched lengths raise ValueError"""
        with pytest.raises(ValueError, match="must have same length"):
            RuleResult(
                labels=np.array([1, 2, 3]),
                confidence=np.array([0.9, 0.8]),  # Different length
                rule_ids=['rule1'],
                stats=sample_rule_stats,
            )
    
    def test_confidence_clipping(self, sample_rule_stats):
        """Test that confidence values are clipped to [0, 1]"""
        labels = np.array([1, 2, 3])
        confidence = np.array([-0.5, 0.5, 1.5])  # Out of range
        
        result = RuleResult(
            labels=labels,
            confidence=confidence,
            rule_ids=['rule1'],
            stats=sample_rule_stats,
        )
        
        assert np.all(result.confidence >= 0.0)
        assert np.all(result.confidence <= 1.0)
        assert result.confidence[0] == 0.0  # Clipped from -0.5
        assert result.confidence[1] == 0.5  # Unchanged
        assert result.confidence[2] == 1.0  # Clipped from 1.5
    
    def test_metadata_field(self, sample_labels, sample_confidence, sample_rule_stats):
        """Test metadata field"""
        metadata = {'source': 'test', 'version': '1.0'}
        
        result = RuleResult(
            labels=sample_labels,
            confidence=sample_confidence,
            rule_ids=['rule1'],
            stats=sample_rule_stats,
            metadata=metadata,
        )
        
        assert result.metadata == metadata
        assert result.metadata['source'] == 'test'


class TestRuleConfig:
    """Test RuleConfig dataclass"""
    
    def test_basic_creation(self):
        """Test creating RuleConfig"""
        config = RuleConfig(
            rule_id='test_rule',
            rule_type=RuleType.GEOMETRIC,
            priority=RulePriority.HIGH,
            target_class=6,
        )
        
        assert config.rule_id == 'test_rule'
        assert config.rule_type == RuleType.GEOMETRIC
        assert config.priority == RulePriority.HIGH
        assert config.target_class == 6
        assert config.enabled is True  # Default value
    
    def test_optional_fields(self):
        """Test optional configuration fields"""
        config = RuleConfig(
            rule_id='test_rule',
            rule_type=RuleType.SPECTRAL,
            priority=RulePriority.MEDIUM,
            target_class=3,
            description='Test rule for vegetation',
            enabled=False,
        )
        
        assert config.description == 'Test rule for vegetation'
        assert config.enabled is False


class TestRuleEngineConfig:
    """Test RuleEngineConfig dataclass"""
    
    def test_basic_creation(self):
        """Test creating RuleEngineConfig"""
        config = RuleEngineConfig(
            execution_strategy=ExecutionStrategy.PRIORITY,
            conflict_resolution=ConflictResolution.HIGHEST_CONFIDENCE,
        )
        
        assert config.execution_strategy == ExecutionStrategy.PRIORITY
        assert config.conflict_resolution == ConflictResolution.HIGHEST_CONFIDENCE


# ================================
# Test Abstract Base Classes
# ================================

class TestBaseRule:
    """Test BaseRule abstract class"""
    
    def test_cannot_instantiate(self):
        """Test that BaseRule cannot be instantiated directly"""
        config = RuleConfig(
            rule_id='test',
            rule_type=RuleType.GEOMETRIC,
            priority=RulePriority.MEDIUM,
            target_class=1,
        )
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseRule(config)
    
    def test_concrete_implementation(self, sample_points, sample_features):
        """Test creating a concrete rule implementation"""
        
        class ConcreteRule(BaseRule):
            """Concrete rule for testing"""
            
            def __init__(self, rule_id: str = 'test_rule'):
                config = RuleConfig(
                    rule_id=rule_id,
                    rule_type=RuleType.GEOMETRIC,
                    priority=RulePriority.MEDIUM,
                    target_class=1,
                )
                super().__init__(config)
            
            def evaluate(
                self,
                points: np.ndarray,
                features: Dict[str, np.ndarray],
                context: Optional[Dict] = None
            ) -> tuple:
                """Simple evaluation: match all points"""
                n_points = len(points)
                match_mask = np.ones(n_points, dtype=bool)
                confidence = np.ones(n_points)
                return match_mask, confidence
            
            def get_required_features(self) -> set:
                """No required features for this simple rule"""
                return set()
            
            def get_optional_features(self) -> set:
                """No optional features"""
                return set()
        
        # Create and use the concrete rule
        rule = ConcreteRule()
        match_mask, confidence = rule.evaluate(sample_points, sample_features)
        
        assert isinstance(match_mask, np.ndarray)
        assert isinstance(confidence, np.ndarray)
        assert len(match_mask) == len(sample_points)
        assert np.all(match_mask == True)


class TestRuleEngine:
    """Test RuleEngine abstract class"""
    
    def test_cannot_instantiate(self):
        """Test that RuleEngine cannot be instantiated directly"""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            RuleEngine(rules=[])
    
    def test_concrete_implementation(self, sample_points, sample_features):
        """Test creating a concrete engine implementation"""
        
        class ConcreteEngine(RuleEngine):
            """Concrete engine for testing"""
            
            def apply_rules(
                self,
                points: np.ndarray,
                features: Dict[str, np.ndarray],
                context: Optional[Dict] = None
            ) -> RuleResult:
                """Apply all rules"""
                n_points = len(points)
                stats = RuleStats(
                    total_points=n_points,
                    matched_points=n_points,
                    unmatched_points=0,
                    rules_applied=len(self.rules),
                )
                return RuleResult(
                    labels=np.zeros(n_points, dtype=int),
                    confidence=np.ones(n_points),
                    rule_ids=[r.rule_id for r in self.rules],
                    stats=stats,
                )
        
        # Create and use the concrete engine
        engine = ConcreteEngine(rules=[])
        result = engine.apply_rules(sample_points, sample_features)
        
        assert isinstance(result, RuleResult)
        assert len(result.labels) == len(sample_points)


# ================================
# Test Utility Functions
# ================================

class TestCreateEmptyResult:
    """Test create_empty_result utility"""
    
    def test_basic_creation(self):
        """Test creating empty result"""
        result = create_empty_result(n_points=100)
        
        assert len(result.labels) == 100
        assert len(result.confidence) == 100
        assert np.all(result.labels == 0)
        assert np.all(result.confidence == 0.0)
        assert result.rule_ids == []
        assert result.stats.total_points == 100
        assert result.stats.matched_points == 0
    
    def test_zero_points(self):
        """Test creating empty result with zero points"""
        result = create_empty_result(n_points=0)
        
        assert len(result.labels) == 0
        assert len(result.confidence) == 0
        assert result.stats.total_points == 0


class TestMergeRuleResults:
    """Test merge_rule_results utility"""
    
    def test_merge_two_results(self):
        """Test merging two rule results"""
        n_points = 100
        
        # First result: classify first 50 points as class 1
        result1 = RuleResult(
            labels=np.concatenate([np.ones(50), np.zeros(50)]),
            confidence=np.concatenate([np.ones(50) * 0.9, np.zeros(50)]),
            rule_ids=['rule1'],
            stats=RuleStats(
                total_points=n_points,
                matched_points=50,
                unmatched_points=50,
                rules_applied=1,
            ),
        )
        
        # Second result: classify last 50 points as class 2
        result2 = RuleResult(
            labels=np.concatenate([np.zeros(50), np.ones(50) * 2]),
            confidence=np.concatenate([np.zeros(50), np.ones(50) * 0.8]),
            rule_ids=['rule2'],
            stats=RuleStats(
                total_points=n_points,
                matched_points=50,
                unmatched_points=50,
                rules_applied=1,
            ),
        )
        
        # Merge results
        merged = merge_rule_results([result1, result2])
        
        assert len(merged.labels) == n_points
        assert len(merged.confidence) == n_points
        assert 'rule1' in merged.rule_ids
        assert 'rule2' in merged.rule_ids
        assert merged.stats.rules_applied == 2
    
    def test_merge_empty_list(self):
        """Test merging empty list raises appropriate error"""
        with pytest.raises(ValueError, match="Cannot merge empty list"):
            merge_rule_results([])
    
    def test_merge_single_result(self, sample_rule_result):
        """Test merging single result returns the same result"""
        merged = merge_rule_results([sample_rule_result])
        
        assert np.array_equal(merged.labels, sample_rule_result.labels)
        assert np.array_equal(merged.confidence, sample_rule_result.confidence)


# ================================
# Integration Tests
# ================================

class TestIntegration:
    """Integration tests for base module components"""
    
    def test_full_workflow(self, sample_points, sample_features):
        """Test complete workflow from rule creation to result"""
        
        # Create a simple rule
        class HeightRule(BaseRule):
            def __init__(self):
                config = RuleConfig(
                    rule_id='height_rule',
                    rule_type=RuleType.GEOMETRIC,
                    priority=RulePriority.HIGH,
                    target_class=6,
                )
                super().__init__(config)
            
            def evaluate(self, points, features, context=None):
                high_points = features['height'] > 5.0
                confidence = np.where(high_points, 0.9, 0.0)
                return high_points, confidence
            
            def get_required_features(self):
                return {'height'}
            
            def get_optional_features(self):
                return set()
        
        # Apply the rule
        rule = HeightRule()
        match_mask, confidence = rule.evaluate(sample_points, sample_features)
        
        # Verify result
        assert isinstance(match_mask, np.ndarray)
        assert isinstance(confidence, np.ndarray)
        assert len(match_mask) == len(sample_points)
        assert len(confidence) == len(sample_points)


# ================================
# Edge Cases and Error Handling
# ================================

class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_rule_stats_with_negative_values(self):
        """Test that negative values are handled appropriately"""
        # This should still create the object, but coverage might be weird
        stats = RuleStats(
            total_points=100,
            matched_points=150,  # More matches than total
            unmatched_points=-50,  # Negative
            rules_applied=1,
        )
        
        # Coverage will be > 1.0, which might be a bug indicator
        assert stats.coverage == 1.5
    
    def test_rule_result_with_empty_arrays(self):
        """Test RuleResult with empty arrays"""
        stats = RuleStats(
            total_points=0,
            matched_points=0,
            unmatched_points=0,
            rules_applied=0,
        )
        
        result = RuleResult(
            labels=np.array([]),
            confidence=np.array([]),
            rule_ids=[],
            stats=stats,
        )
        
        assert len(result.labels) == 0
        assert len(result.confidence) == 0
    
    def test_rule_config_with_invalid_class(self):
        """Test RuleConfig with unusual target class values"""
        # Should accept any integer value
        config = RuleConfig(
            rule_id='test',
            rule_type=RuleType.GEOMETRIC,
            priority=RulePriority.LOW,
            target_class=-1,  # Unusual but valid
        )
        
        assert config.target_class == -1
