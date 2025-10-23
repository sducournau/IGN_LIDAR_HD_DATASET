"""
Comprehensive tests for rules hierarchy module.

Tests hierarchical rule execution with multiple levels and strategies.
"""

import pytest
import numpy as np
from typing import Dict, Any, Tuple

from ign_lidar.core.classification.rules.hierarchy import (
    RuleLevel,
    HierarchicalRuleEngine
)
from ign_lidar.core.classification.rules.base import (
    BaseRule,
    RuleConfig,
    RuleEngineConfig,
    RuleType,
    RulePriority
)


# ============================================================================
# Test Helper Classes
# ============================================================================

class MockRule(BaseRule):
    """Mock rule for testing"""
    
    def __init__(
        self,
        rule_id: str,
        target_class: int,
        match_condition: callable,
        confidence_value: float = 1.0
    ):
        config = RuleConfig(
            rule_id=rule_id,
            rule_type=RuleType.GEOMETRIC,
            target_class=target_class,
            priority=RulePriority.MEDIUM,
            description=f"Mock rule {rule_id}"
        )
        super().__init__(config)
        self.match_condition = match_condition
        self.confidence_value = confidence_value
    
    def evaluate(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        context: Any = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate mock rule"""
        match_mask = self.match_condition(points, features)
        confidence = np.where(match_mask, self.confidence_value, 0.0)
        return match_mask, confidence
    
    def get_required_features(self) -> set:
        """Return required features"""
        return set()
    
    def get_optional_features(self) -> set:
        """Return optional features"""
        return set()


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_points():
    """Sample point cloud"""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0]
    ])


@pytest.fixture
def sample_features():
    """Sample features"""
    return {
        'height': np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        'intensity': np.array([100, 150, 200, 250, 300])
    }


@pytest.fixture
def simple_rules():
    """Create simple test rules"""
    # Rule 1: Low height -> ground (class 1)
    rule1 = MockRule(
        rule_id="ground_rule",
        target_class=1,
        match_condition=lambda p, f: f['height'] < 0.5,
        confidence_value=0.9
    )
    
    # Rule 2: Medium height -> vegetation (class 2)
    rule2 = MockRule(
        rule_id="veg_rule",
        target_class=2,
        match_condition=lambda p, f: (f['height'] >= 0.5) & (f['height'] < 2.5),
        confidence_value=0.8
    )
    
    # Rule 3: High height -> building (class 3)
    rule3 = MockRule(
        rule_id="building_rule",
        target_class=3,
        match_condition=lambda p, f: f['height'] >= 2.5,
        confidence_value=0.85
    )
    
    return [rule1, rule2, rule3]


# ============================================================================
# Test RuleLevel
# ============================================================================

class TestRuleLevel:
    """Test RuleLevel dataclass"""
    
    def test_basic_creation(self, simple_rules):
        """Test creating a basic rule level"""
        level = RuleLevel(
            level=0,
            rules=simple_rules[:2],
            strategy="first_match",
            description="Test level"
        )
        
        assert level.level == 0
        assert len(level.rules) == 2
        assert level.strategy == "first_match"
        assert level.description == "Test level"
    
    def test_default_strategy(self, simple_rules):
        """Test default strategy is first_match"""
        level = RuleLevel(level=0, rules=simple_rules)
        assert level.strategy == "first_match"
    
    def test_negative_level_raises(self, simple_rules):
        """Test that negative level raises error"""
        with pytest.raises(ValueError, match="Level must be >= 0"):
            RuleLevel(level=-1, rules=simple_rules)
    
    def test_invalid_strategy_raises(self, simple_rules):
        """Test that invalid strategy raises error"""
        with pytest.raises(ValueError, match="Invalid strategy"):
            RuleLevel(level=0, rules=simple_rules, strategy="invalid")
    
    def test_empty_rules_warning(self, caplog):
        """Test that empty rules list logs warning"""
        import logging
        caplog.set_level(logging.WARNING)
        
        level = RuleLevel(level=0, rules=[])
        assert "has no rules" in caplog.text
    
    def test_repr(self, simple_rules):
        """Test string representation"""
        level = RuleLevel(level=1, rules=simple_rules, strategy="all_matches")
        repr_str = repr(level)
        
        assert "RuleLevel" in repr_str
        assert "level=1" in repr_str
        assert "n_rules=3" in repr_str
        assert "strategy=all_matches" in repr_str
    
    def test_valid_strategies(self, simple_rules):
        """Test all valid strategies are accepted"""
        strategies = ["first_match", "all_matches", "weighted", "priority"]
        
        for strategy in strategies:
            level = RuleLevel(level=0, rules=simple_rules, strategy=strategy)
            assert level.strategy == strategy


# ============================================================================
# Test HierarchicalRuleEngine - Initialization
# ============================================================================

class TestHierarchicalRuleEngineInit:
    """Test HierarchicalRuleEngine initialization"""
    
    def test_basic_initialization(self, simple_rules):
        """Test basic initialization"""
        levels = [
            RuleLevel(level=0, rules=simple_rules[:1]),
            RuleLevel(level=1, rules=simple_rules[1:])
        ]
        
        engine = HierarchicalRuleEngine(levels)
        
        assert len(engine.levels) == 2
        assert len(engine.rules) == 3  # All rules from both levels
    
    def test_levels_sorted_by_number(self, simple_rules):
        """Test that levels are sorted by level number"""
        # Create levels in reverse order
        levels = [
            RuleLevel(level=2, rules=simple_rules[2:]),
            RuleLevel(level=0, rules=simple_rules[:1]),
            RuleLevel(level=1, rules=simple_rules[1:2])
        ]
        
        engine = HierarchicalRuleEngine(levels)
        
        # Should be sorted: 0, 1, 2
        assert engine.levels[0].level == 0
        assert engine.levels[1].level == 1
        assert engine.levels[2].level == 2
    
    def test_with_config(self, simple_rules):
        """Test initialization with custom config"""
        levels = [RuleLevel(level=0, rules=simple_rules)]
        config = RuleEngineConfig(
            min_confidence_threshold=0.5,
            parallel_execution=True
        )
        
        engine = HierarchicalRuleEngine(levels, config=config)
        
        assert engine.config.min_confidence_threshold == 0.5
        assert engine.config.parallel_execution is True
    
    def test_single_level(self, simple_rules):
        """Test engine with single level"""
        levels = [RuleLevel(level=0, rules=simple_rules)]
        engine = HierarchicalRuleEngine(levels)
        
        assert len(engine.levels) == 1
        assert len(engine.rules) == 3


# ============================================================================
# Test HierarchicalRuleEngine - First Match Strategy
# ============================================================================

class TestFirstMatchStrategy:
    """Test first_match strategy"""
    
    def test_first_match_basic(self, sample_points, sample_features, simple_rules):
        """Test basic first match execution"""
        # Create single level with first_match strategy
        levels = [
            RuleLevel(level=0, rules=simple_rules, strategy="first_match")
        ]
        engine = HierarchicalRuleEngine(levels)
        
        result = engine.apply_rules(sample_points, sample_features)
        
        # Check that all points are classified
        assert np.all(result.labels > 0)
        
        # Check specific classifications
        assert result.labels[0] == 1  # height=0.0 -> ground
        assert result.labels[1] == 2  # height=1.0 -> vegetation
        assert result.labels[2] == 2  # height=2.0 -> vegetation
        assert result.labels[3] == 3  # height=3.0 -> building
        assert result.labels[4] == 3  # height=4.0 -> building
    
    def test_first_match_stops_after_match(self, sample_points, sample_features):
        """Test that first_match stops after finding a match"""
        # Create rules that could overlap
        rule1 = MockRule(
            rule_id="rule1",
            target_class=1,
            match_condition=lambda p, f: f['height'] >= 0,  # Matches all
            confidence_value=0.5
        )
        rule2 = MockRule(
            rule_id="rule2",
            target_class=2,
            match_condition=lambda p, f: f['height'] >= 0,  # Would also match all
            confidence_value=0.9
        )
        
        levels = [RuleLevel(level=0, rules=[rule1, rule2], strategy="first_match")]
        engine = HierarchicalRuleEngine(levels)
        
        result = engine.apply_rules(sample_points, sample_features)
        
        # All points should be classified as class 1 (first rule)
        assert np.all(result.labels == 1)
        assert np.all(result.confidence == 0.5)


# ============================================================================
# Test HierarchicalRuleEngine - All Matches Strategy
# ============================================================================

class TestAllMatchesStrategy:
    """Test all_matches strategy"""
    
    def test_all_matches_combines_votes(self, sample_points, sample_features):
        """Test that all_matches combines votes from multiple rules"""
        # Create overlapping rules with different confidences
        rule1 = MockRule(
            rule_id="rule1",
            target_class=1,
            match_condition=lambda p, f: f['height'] < 3.0,
            confidence_value=0.6
        )
        rule2 = MockRule(
            rule_id="rule2",
            target_class=1,
            match_condition=lambda p, f: f['height'] < 2.0,
            confidence_value=0.9  # Higher confidence for subset
        )
        
        levels = [RuleLevel(level=0, rules=[rule1, rule2], strategy="all_matches")]
        engine = HierarchicalRuleEngine(levels)
        
        result = engine.apply_rules(sample_points, sample_features)
        
        # Points with height < 2.0 should have confidence 0.9 (max)
        assert result.confidence[0] == 0.9  # height=0.0
        assert result.confidence[1] == 0.9  # height=1.0
        
        # Points with 2.0 <= height < 3.0 should have confidence 0.6
        assert result.confidence[2] == 0.6  # height=2.0
    
    def test_all_matches_highest_confidence_wins(self, sample_points, sample_features):
        """Test that highest confidence wins for competing classes"""
        # Create competing rules
        rule1 = MockRule(
            rule_id="low_conf",
            target_class=1,
            match_condition=lambda p, f: f['height'] < 2.0,
            confidence_value=0.5
        )
        rule2 = MockRule(
            rule_id="high_conf",
            target_class=2,
            match_condition=lambda p, f: f['height'] < 2.0,
            confidence_value=0.9
        )
        
        levels = [RuleLevel(level=0, rules=[rule1, rule2], strategy="all_matches")]
        engine = HierarchicalRuleEngine(levels)
        
        result = engine.apply_rules(sample_points, sample_features)
        
        # Points should be classified as class 2 (higher confidence)
        assert result.labels[0] == 2
        assert result.labels[1] == 2
        assert result.confidence[0] == 0.9


# ============================================================================
# Test HierarchicalRuleEngine - Priority Strategy
# ============================================================================

class TestPriorityStrategy:
    """Test priority strategy"""
    
    def test_priority_strategy(self, sample_points, sample_features, simple_rules):
        """Test priority-based execution"""
        levels = [
            RuleLevel(level=0, rules=simple_rules, strategy="priority")
        ]
        engine = HierarchicalRuleEngine(levels)
        
        result = engine.apply_rules(sample_points, sample_features)
        
        # Should classify all points based on rule order
        assert np.all(result.labels > 0)


# ============================================================================
# Test HierarchicalRuleEngine - Multi-Level Execution
# ============================================================================

class TestMultiLevelExecution:
    """Test multi-level hierarchical execution"""
    
    def test_two_level_hierarchy(self, sample_points, sample_features):
        """Test two-level hierarchy"""
        # Level 0: Critical rules (high priority)
        rule_ground = MockRule(
            rule_id="ground",
            target_class=1,
            match_condition=lambda p, f: f['height'] < 0.5,
            confidence_value=0.95
        )
        
        # Level 1: Secondary rules
        rule_veg = MockRule(
            rule_id="vegetation",
            target_class=2,
            match_condition=lambda p, f: f['height'] >= 0.5,
            confidence_value=0.8
        )
        
        levels = [
            RuleLevel(level=0, rules=[rule_ground]),
            RuleLevel(level=1, rules=[rule_veg])
        ]
        
        engine = HierarchicalRuleEngine(levels)
        result = engine.apply_rules(sample_points, sample_features)
        
        # Point 0 should be classified by level 0
        assert result.labels[0] == 1
        assert result.confidence[0] == 0.95
        
        # Other points should be classified by level 1
        assert np.all(result.labels[1:] == 2)
        assert np.all(result.confidence[1:] == 0.8)
    
    def test_early_exit_when_all_classified(self, sample_points, sample_features):
        """Test early exit when all points classified"""
        # Level 0: Rule that classifies everything
        rule_all = MockRule(
            rule_id="classify_all",
            target_class=1,
            match_condition=lambda p, f: np.ones(len(p), dtype=bool),
            confidence_value=0.9
        )
        
        # Level 1: Should not be executed
        rule_never = MockRule(
            rule_id="never_executed",
            target_class=2,
            match_condition=lambda p, f: np.ones(len(p), dtype=bool),
            confidence_value=0.8
        )
        
        levels = [
            RuleLevel(level=0, rules=[rule_all]),
            RuleLevel(level=1, rules=[rule_never])
        ]
        
        engine = HierarchicalRuleEngine(levels)
        result = engine.apply_rules(sample_points, sample_features)
        
        # All should be classified by level 0
        assert np.all(result.labels == 1)
        assert "never_executed" not in result.rule_ids
    
    def test_unclassified_points_propagate(self, sample_points, sample_features):
        """Test that unclassified points propagate to next level"""
        # Level 0: Only classifies low height
        rule_low = MockRule(
            rule_id="low_only",
            target_class=1,
            match_condition=lambda p, f: f['height'] < 1.5,
            confidence_value=0.9
        )
        
        # Level 1: Classifies remaining
        rule_rest = MockRule(
            rule_id="rest",
            target_class=2,
            match_condition=lambda p, f: f['height'] >= 1.5,
            confidence_value=0.7
        )
        
        levels = [
            RuleLevel(level=0, rules=[rule_low]),
            RuleLevel(level=1, rules=[rule_rest])
        ]
        
        engine = HierarchicalRuleEngine(levels)
        result = engine.apply_rules(sample_points, sample_features)
        
        # Points 0, 1 classified by level 0
        assert result.labels[0] == 1
        assert result.labels[1] == 1
        
        # Points 2, 3, 4 classified by level 1
        assert result.labels[2] == 2
        assert result.labels[3] == 2
        assert result.labels[4] == 2


# ============================================================================
# Test Statistics and Metadata
# ============================================================================

class TestStatisticsAndMetadata:
    """Test statistics collection and metadata"""
    
    def test_statistics_collection(self, sample_points, sample_features, simple_rules):
        """Test that statistics are collected correctly"""
        levels = [RuleLevel(level=0, rules=simple_rules)]
        engine = HierarchicalRuleEngine(levels)
        
        result = engine.apply_rules(sample_points, sample_features)
        
        # Check statistics
        assert result.stats.total_points == 5
        assert result.stats.matched_points > 0
        assert result.stats.rules_applied > 0
        assert result.stats.execution_time_ms > 0
        assert 0 <= result.stats.coverage <= 1
    
    def test_coverage_calculation(self, sample_points, sample_features):
        """Test coverage calculation"""
        # Rule that only matches some points
        rule_partial = MockRule(
            rule_id="partial",
            target_class=1,
            match_condition=lambda p, f: f['height'] < 2.0,
            confidence_value=0.8
        )
        
        levels = [RuleLevel(level=0, rules=[rule_partial])]
        engine = HierarchicalRuleEngine(levels)
        
        result = engine.apply_rules(sample_points, sample_features)
        
        # Should have 40% coverage (2/5 points)
        assert result.stats.coverage == 0.4
    
    def test_metadata_includes_n_levels(self, sample_points, sample_features, simple_rules):
        """Test that metadata includes number of levels"""
        levels = [
            RuleLevel(level=0, rules=simple_rules[:1]),
            RuleLevel(level=1, rules=simple_rules[1:])
        ]
        engine = HierarchicalRuleEngine(levels)
        
        result = engine.apply_rules(sample_points, sample_features)
        
        assert 'n_levels' in result.metadata
        assert result.metadata['n_levels'] == 2
    
    def test_confidence_statistics(self, sample_points, sample_features):
        """Test confidence statistics calculation"""
        rule = MockRule(
            rule_id="test_rule",
            target_class=1,
            match_condition=lambda p, f: np.ones(len(p), dtype=bool),
            confidence_value=0.75
        )
        
        levels = [RuleLevel(level=0, rules=[rule])]
        engine = HierarchicalRuleEngine(levels)
        
        result = engine.apply_rules(sample_points, sample_features)
        
        assert result.stats.mean_confidence == 0.75
        assert result.stats.min_confidence == 0.75
        assert result.stats.max_confidence == 0.75


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_no_rules_match(self, sample_points, sample_features):
        """Test when no rules match any points"""
        rule_none = MockRule(
            rule_id="matches_none",
            target_class=1,
            match_condition=lambda p, f: np.zeros(len(p), dtype=bool),
            confidence_value=0.8
        )
        
        levels = [RuleLevel(level=0, rules=[rule_none])]
        engine = HierarchicalRuleEngine(levels)
        
        result = engine.apply_rules(sample_points, sample_features)
        
        # All labels should be 0 (unclassified)
        assert np.all(result.labels == 0)
        assert result.stats.matched_points == 0
        assert result.stats.coverage == 0.0
    
    def test_empty_level(self, sample_points, sample_features, simple_rules):
        """Test level with no rules"""
        levels = [
            RuleLevel(level=0, rules=[]),
            RuleLevel(level=1, rules=simple_rules)
        ]
        
        engine = HierarchicalRuleEngine(levels)
        result = engine.apply_rules(sample_points, sample_features)
        
        # Should still classify using level 1
        assert np.any(result.labels > 0)
    
    def test_single_point(self, sample_features):
        """Test with single point"""
        single_point = np.array([[1.0, 1.0, 1.0]])
        single_features = {
            'height': np.array([1.0]),
            'intensity': np.array([150])
        }
        
        rule = MockRule(
            rule_id="test",
            target_class=1,
            match_condition=lambda p, f: np.ones(len(p), dtype=bool),
            confidence_value=0.9
        )
        
        levels = [RuleLevel(level=0, rules=[rule])]
        engine = HierarchicalRuleEngine(levels)
        
        result = engine.apply_rules(single_point, single_features)
        
        assert len(result.labels) == 1
        assert result.labels[0] == 1


# ============================================================================
# Integration Tests
# ============================================================================

class TestHierarchyIntegration:
    """Test integration of hierarchical execution"""
    
    def test_complete_workflow(self, sample_points, sample_features):
        """Test complete hierarchical classification workflow"""
        # Create realistic hierarchy
        # Level 0: Ground detection (highest priority)
        ground_rule = MockRule(
            rule_id="ground_detector",
            target_class=1,
            match_condition=lambda p, f: f['height'] < 0.5,
            confidence_value=0.95
        )
        
        # Level 1: Building detection
        building_rule = MockRule(
            rule_id="building_detector",
            target_class=3,
            match_condition=lambda p, f: f['height'] > 2.5,
            confidence_value=0.85
        )
        
        # Level 2: Vegetation (default for middle heights)
        veg_rule = MockRule(
            rule_id="vegetation_detector",
            target_class=2,
            match_condition=lambda p, f: (f['height'] >= 0.5) & (f['height'] <= 2.5),
            confidence_value=0.7
        )
        
        levels = [
            RuleLevel(level=0, rules=[ground_rule], strategy="first_match"),
            RuleLevel(level=1, rules=[building_rule], strategy="first_match"),
            RuleLevel(level=2, rules=[veg_rule], strategy="first_match")
        ]
        
        engine = HierarchicalRuleEngine(levels)
        result = engine.apply_rules(sample_points, sample_features)
        
        # Verify complete classification
        assert np.all(result.labels > 0)
        assert result.stats.coverage == 1.0
        
        # Verify confidence levels match expectations
        assert result.confidence[0] == 0.95  # ground
        assert result.confidence[1] == 0.7   # vegetation
        assert result.confidence[2] == 0.7   # vegetation
        assert result.confidence[3] == 0.85  # building
        assert result.confidence[4] == 0.85  # building
    
    def test_mixed_strategies(self, sample_points, sample_features):
        """Test hierarchy with mixed strategies per level"""
        # Level 0: first_match for critical classes
        rule1 = MockRule(
            rule_id="critical",
            target_class=1,
            match_condition=lambda p, f: f['height'] < 1.0,
            confidence_value=0.9
        )
        
        # Level 1: all_matches for consensus
        rule2a = MockRule(
            rule_id="consensus_a",
            target_class=2,
            match_condition=lambda p, f: f['height'] >= 1.0,
            confidence_value=0.6
        )
        rule2b = MockRule(
            rule_id="consensus_b",
            target_class=2,
            match_condition=lambda p, f: f['height'] >= 1.0,
            confidence_value=0.8
        )
        
        levels = [
            RuleLevel(level=0, rules=[rule1], strategy="first_match"),
            RuleLevel(level=1, rules=[rule2a, rule2b], strategy="all_matches")
        ]
        
        engine = HierarchicalRuleEngine(levels)
        result = engine.apply_rules(sample_points, sample_features)
        
        # Point 0 classified by level 0
        assert result.labels[0] == 1
        
        # Other points classified by level 1 with max confidence
        assert np.all(result.labels[1:] == 2)
        assert np.all(result.confidence[1:] == 0.8)
