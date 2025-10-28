"""
Tests for the Rules Framework (v3.2.0)

Tests the base rule system, rule engine, and data structures
introduced in v3.2.0.
"""

import pytest
import numpy as np
from typing import Dict, Set, Optional, Any, Tuple

from ign_lidar.core.classification.rules.base import (
    RuleType,
    RulePriority,
    ExecutionStrategy,
    ConflictResolution,
    RuleStats,
    RuleResult,
    RuleConfig,
    RuleEngineConfig,
    BaseRule,
    RuleEngine,
    create_empty_result,
    merge_rule_results,
)


@pytest.mark.unit
class TestRuleEnums:
    """Test rule enumeration types."""

    def test_rule_type_values(self):
        """Test that all rule types are defined."""
        assert RuleType.GEOMETRIC.value == "geometric"
        assert RuleType.SPECTRAL.value == "spectral"
        assert RuleType.GRAMMAR.value == "grammar"
        assert RuleType.HYBRID.value == "hybrid"
        assert RuleType.CONTEXTUAL.value == "contextual"
        assert RuleType.TEMPORAL.value == "temporal"

    def test_rule_priority_ordering(self):
        """Test that priority values are correctly ordered."""
        assert RulePriority.CRITICAL.value > RulePriority.HIGH.value
        assert RulePriority.HIGH.value > RulePriority.MEDIUM.value
        assert RulePriority.MEDIUM.value > RulePriority.LOW.value

        # Specific values
        assert RulePriority.LOW.value == 1
        assert RulePriority.MEDIUM.value == 2
        assert RulePriority.HIGH.value == 3
        assert RulePriority.CRITICAL.value == 4

    def test_execution_strategy_values(self):
        """Test execution strategy enumeration."""
        assert ExecutionStrategy.FIRST_MATCH.value == "first_match"
        assert ExecutionStrategy.ALL_MATCHES.value == "all_matches"
        assert ExecutionStrategy.PRIORITY.value == "priority"
        assert ExecutionStrategy.WEIGHTED.value == "weighted"
        assert ExecutionStrategy.HIERARCHICAL.value == "hierarchical"

    def test_conflict_resolution_values(self):
        """Test conflict resolution strategies."""
        assert ConflictResolution.HIGHEST_PRIORITY.value == "highest_priority"
        assert ConflictResolution.HIGHEST_CONFIDENCE.value == "highest_confidence"
        assert ConflictResolution.WEIGHTED_VOTE.value == "weighted_vote"
        assert ConflictResolution.FIRST_WINS.value == "first_wins"
        assert ConflictResolution.LAST_WINS.value == "last_wins"


@pytest.mark.unit
class TestRuleDataClasses:
    """Test rule data classes."""

    def test_rule_stats_creation(self):
        """Test RuleStats dataclass creation."""
        stats = RuleStats(
            total_points=1000,
            matched_points=250,
            unmatched_points=750,
            rules_applied=3,
            execution_time_ms=45.2,
        )

        assert stats.total_points == 1000
        assert stats.matched_points == 250
        assert stats.unmatched_points == 750
        assert stats.rules_applied == 3
        assert stats.execution_time_ms == 45.2
        assert stats.coverage == 0.25  # 250/1000

    def test_rule_result_creation(self):
        """Test RuleResult dataclass creation."""
        result = RuleResult(
            labels=np.array([0, 6, 6, 6, 0]),
            confidence=np.array([0.0, 0.8, 0.9, 0.85, 0.0]),
            rule_ids=["rule1", "rule2"],
            stats=RuleStats(
                total_points=5,
                matched_points=3,
                unmatched_points=2,
                rules_applied=2,
                execution_time_ms=10.5,
            ),
        )

        assert len(result.labels) == 5
        assert len(result.confidence) == 5
        assert len(result.rule_ids) == 2
        assert result.stats.matched_points == 3
        assert np.sum(result.labels > 0) == 3

    def test_rule_config_defaults(self):
        """Test RuleConfig with default values."""
        config = RuleConfig(
            rule_id="test_rule",
            rule_type=RuleType.GEOMETRIC,
            priority=RulePriority.HIGH,
            target_class=6,
        )

        assert config.rule_id == "test_rule"
        assert config.enabled is True  # Default
        assert config.min_confidence == 0.0  # Default


@pytest.mark.unit
class TestBaseRule:
    """Test BaseRule abstract base class."""

    def test_base_rule_cannot_be_instantiated(self):
        """Test that BaseRule is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseRule(
                config=RuleConfig(
                    rule_id="test",
                    rule_type=RuleType.GEOMETRIC,
                    priority=RulePriority.MEDIUM,
                    target_class=6,
                )
            )

    def test_concrete_rule_implementation(self):
        """Test that concrete rule implementations work."""

        class ConcreteRule(BaseRule):
            """Simple concrete rule for testing."""

            def evaluate(
                self,
                points: np.ndarray,
                features: Dict[str, np.ndarray],
                context: Optional[Dict[str, Any]] = None,
            ) -> Tuple[np.ndarray, np.ndarray]:
                """Dummy evaluation - classify high points as buildings."""
                n_points = len(points)

                # Points above z=50 are buildings
                match_mask = points[:, 2] > 50
                confidence = np.where(match_mask, 0.9, 0.0)

                return match_mask, confidence

            def get_required_features(self) -> Set[str]:
                return set()  # No required features

            def get_optional_features(self) -> Set[str]:
                return set()  # No optional features

        config = RuleConfig(
            rule_id="concrete_test",
            rule_type=RuleType.GEOMETRIC,
            priority=RulePriority.HIGH,
            target_class=6,
        )

        rule = ConcreteRule(config)
        assert rule.rule_id == "concrete_test"
        assert rule.rule_type == RuleType.GEOMETRIC
        assert rule.priority == RulePriority.HIGH
        assert rule.target_class == 6

        # Test evaluation
        points = np.random.rand(100, 3) * 100  # Random points 0-100
        features = {}
        match_mask, confidence = rule.evaluate(points, features)

        assert len(match_mask) == 100
        assert len(confidence) == 100
        assert match_mask.dtype == bool
        assert np.all((confidence >= 0) & (confidence <= 1))


@pytest.mark.unit
class TestHelperFunctions:
    """Test helper functions."""

    def test_create_empty_result(self):
        """Test creating an empty rule result."""
        n_points = 1000
        result = create_empty_result(n_points)

        assert len(result.labels) == n_points
        assert len(result.confidence) == n_points
        assert len(result.rule_ids) == 0
        assert result.stats.total_points == n_points
        assert result.stats.matched_points == 0
        assert result.stats.unmatched_points == n_points
        assert np.all(result.labels == 0)
        assert np.all(result.confidence == 0)

    def test_merge_rule_results_highest_confidence(self):
        """Test merging results with HIGHEST_CONFIDENCE strategy."""
        n_points = 10

        # Result 1: Points 0-4 as buildings with confidence 0.8
        result1 = RuleResult(
            labels=np.array([6, 6, 6, 6, 6, 0, 0, 0, 0, 0]),
            confidence=np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0, 0, 0, 0, 0]),
            rule_ids=["rule1"],
            stats=RuleStats(n_points, 5, 5, 1),
        )

        # Result 2: Points 3-7 as vegetation with confidence 0.9
        result2 = RuleResult(
            labels=np.array([0, 0, 0, 4, 4, 4, 4, 4, 0, 0]),
            confidence=np.array([0, 0, 0, 0.9, 0.9, 0.9, 0.9, 0.9, 0, 0]),
            rule_ids=["rule2"],
            stats=RuleStats(n_points, 5, 5, 1),
        )

        merged = merge_rule_results(
            [result1, result2], ConflictResolution.HIGHEST_CONFIDENCE
        )

        # Points 0-2: building (only result1)
        # Points 3-4: vegetation (result2 has higher confidence 0.9 > 0.8)
        # Points 5-7: vegetation (only result2)
        # Points 8-9: unclassified

        assert len(merged.labels) == n_points
        assert merged.labels[0] == 6  # Building
        assert merged.labels[1] == 6  # Building
        assert merged.labels[2] == 6  # Building
        assert merged.labels[3] == 4  # Vegetation (higher confidence wins)
        assert merged.labels[4] == 4  # Vegetation (higher confidence wins)
        assert merged.stats.total_points == n_points


@pytest.mark.integration
class TestRulesIntegration:
    """Integration tests for rules framework."""

    def test_simple_rule_application(self):
        """Test applying a simple rule to point cloud data."""

        class HeightBasedRule(BaseRule):
            """Classify points by height."""

            def evaluate(
                self,
                points: np.ndarray,
                features: Dict[str, np.ndarray],
                context: Optional[Dict[str, Any]] = None,
            ) -> Tuple[np.ndarray, np.ndarray]:
                # Points above threshold are target class
                match_mask = points[:, 2] > 50
                confidence = np.where(match_mask, 0.95, 0.0)
                return match_mask, confidence

            def get_required_features(self) -> Set[str]:
                return set()

            def get_optional_features(self) -> Set[str]:
                return set()

        # Create test data
        n_points = 1000
        points = np.random.rand(n_points, 3) * 100
        features = {}

        # Create and apply rule
        rule = HeightBasedRule(
            config=RuleConfig(
                rule_id="height_classifier",
                rule_type=RuleType.GEOMETRIC,
                priority=RulePriority.HIGH,
                target_class=6,
            )
        )

        match_mask, confidence = rule.evaluate(points, features)

        # Verify results
        assert len(match_mask) == n_points
        assert len(confidence) == n_points
        assert np.sum(match_mask) > 0  # Some points should match
        assert np.all(confidence[match_mask] == 0.95)
        assert np.all(confidence[~match_mask] == 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
