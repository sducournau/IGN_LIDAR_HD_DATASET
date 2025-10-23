"""
Base classes and data structures for rule-based classification.

This module provides the foundation for all rule-based classification systems:
- Abstract base classes (BaseRule, RuleEngine)
- Standard enums (RuleType, RulePriority, ConfidenceMethod)
- Result containers (RuleResult, RuleStats)
- Configuration classes

All rule implementations (geometric, spectral, grammar) inherit from these base classes
to ensure consistent interfaces and behavior.

Usage:
    from ign_lidar.core.classification.rules import BaseRule, RuleEngine, RuleType
    
    class MyCustomRule(BaseRule):
        def evaluate(self, points, features, context):
            # Implementation
            pass
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ================================
# Enumerations
# ================================

class RuleType(str, Enum):
    """Type of classification rule"""
    GEOMETRIC = "geometric"
    SPECTRAL = "spectral"
    GRAMMAR = "grammar"
    HYBRID = "hybrid"
    CONTEXTUAL = "contextual"
    TEMPORAL = "temporal"


class RulePriority(Enum):
    """Priority level for rule application (higher value = higher priority)"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ExecutionStrategy(str, Enum):
    """Strategy for applying multiple rules"""
    FIRST_MATCH = "first_match"      # Apply first matching rule only
    ALL_MATCHES = "all_matches"      # Apply all matching rules
    PRIORITY = "priority"            # Apply rules in priority order
    WEIGHTED = "weighted"            # Weighted combination of all rules
    HIERARCHICAL = "hierarchical"    # Hierarchical application with levels


class ConflictResolution(str, Enum):
    """How to resolve conflicts when multiple rules match"""
    HIGHEST_PRIORITY = "highest_priority"
    HIGHEST_CONFIDENCE = "highest_confidence"
    WEIGHTED_VOTE = "weighted_vote"
    FIRST_WINS = "first_wins"
    LAST_WINS = "last_wins"


# ================================
# Result Data Classes
# ================================

@dataclass
class RuleStats:
    """Statistics from rule application"""
    total_points: int
    matched_points: int
    unmatched_points: int
    rules_applied: int
    execution_time_ms: float = 0.0
    
    # Per-rule statistics
    rule_match_counts: Dict[str, int] = field(default_factory=dict)
    rule_execution_times: Dict[str, float] = field(default_factory=dict)
    
    # Confidence statistics
    mean_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 1.0
    
    # Quality metrics
    coverage: float = 0.0  # Fraction of points matched
    ambiguity: float = 0.0  # Fraction with multiple rule matches
    
    def __post_init__(self):
        """Calculate derived statistics"""
        if self.total_points > 0:
            self.coverage = self.matched_points / self.total_points
            if self.unmatched_points > 0:
                self.ambiguity = 1.0 - (self.matched_points / self.total_points)


@dataclass
class RuleResult:
    """Standard result container for all rule engines
    
    Attributes:
        labels: Point classifications [N] (0 = unclassified)
        confidence: Confidence scores [N] in [0, 1]
        rule_ids: List of rule IDs that were applied
        stats: Detailed statistics about rule application
        metadata: Optional additional information
    """
    labels: np.ndarray
    confidence: np.ndarray
    rule_ids: List[str]
    stats: RuleStats
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate result structure"""
        if len(self.labels) != len(self.confidence):
            raise ValueError(
                f"Labels ({len(self.labels)}) and confidence ({len(self.confidence)}) "
                "must have same length"
            )
        
        # Ensure confidence is in valid range
        if np.any(self.confidence < 0) or np.any(self.confidence > 1):
            logger.warning("Confidence values outside [0, 1] range, clipping")
            self.confidence = np.clip(self.confidence, 0, 1)


# ================================
# Configuration Classes
# ================================

@dataclass
class RuleConfig:
    """Configuration for a single rule"""
    rule_id: str
    rule_type: RuleType
    priority: RulePriority
    target_class: int
    description: str = ""
    enabled: bool = True
    
    # Execution parameters
    min_confidence: float = 0.0
    require_all_features: bool = True
    allow_partial_match: bool = False
    
    # Performance tuning
    max_execution_time_ms: float = 1000.0
    cache_results: bool = False


@dataclass
class RuleEngineConfig:
    """Configuration for rule engine"""
    execution_strategy: ExecutionStrategy = ExecutionStrategy.PRIORITY
    conflict_resolution: ConflictResolution = ConflictResolution.HIGHEST_PRIORITY
    
    # Global thresholds
    min_confidence_threshold: float = 0.0
    min_points_per_class: int = 1
    
    # Performance settings
    parallel_execution: bool = False
    max_workers: int = 4
    cache_enabled: bool = True
    
    # Quality control
    validate_inputs: bool = True
    validate_outputs: bool = True
    strict_mode: bool = False


# ================================
# Abstract Base Classes
# ================================

class BaseRule(ABC):
    """Abstract base class for all classification rules
    
    All rule types (geometric, spectral, grammar) inherit from this class
    to ensure consistent interface and behavior.
    
    A rule evaluates point cloud data and returns a binary mask of matching points
    along with confidence scores.
    """
    
    def __init__(self, config: RuleConfig):
        """Initialize rule with configuration
        
        Args:
            config: Rule configuration
        """
        self.config = config
        self.rule_id = config.rule_id
        self.rule_type = config.rule_type
        self.priority = config.priority
        self.target_class = config.target_class
        self.description = config.description
        self.enabled = config.enabled
        
        # Statistics
        self.n_evaluations = 0
        self.total_execution_time = 0.0
        self.total_matches = 0
    
    @abstractmethod
    def evaluate(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate rule and return matching points with confidence
        
        Args:
            points: Point coordinates [N, 3]
            features: Dictionary of feature arrays {name: values[N]}
            context: Optional context information (e.g., neighboring points, metadata)
        
        Returns:
            Tuple of (match_mask, confidence_scores):
                - match_mask: Boolean array [N] indicating matches
                - confidence_scores: Float array [N] with confidence in [0, 1]
        """
        pass
    
    @abstractmethod
    def get_required_features(self) -> Set[str]:
        """Get list of required feature names
        
        Returns:
            Set of feature names that must be present in features dict
        """
        pass
    
    @abstractmethod
    def get_optional_features(self) -> Set[str]:
        """Get list of optional feature names
        
        Returns:
            Set of feature names that improve results if present
        """
        pass
    
    def validate_features(
        self,
        features: Dict[str, np.ndarray],
        n_points: int
    ) -> None:
        """Validate that required features are present and correctly shaped
        
        Args:
            features: Feature dictionary to validate
            n_points: Expected number of points
        
        Raises:
            ValueError: If required features are missing or incorrectly shaped
        """
        required = self.get_required_features()
        missing = required - set(features.keys())
        
        if missing:
            raise ValueError(
                f"Rule '{self.rule_id}' missing required features: {missing}"
            )
        
        # Validate shapes
        for name, values in features.items():
            if name in required or name in self.get_optional_features():
                if len(values) != n_points:
                    raise ValueError(
                        f"Feature '{name}' has {len(values)} values, "
                        f"expected {n_points}"
                    )
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id='{self.rule_id}', "
            f"type={self.rule_type.value}, "
            f"priority={self.priority.name}, "
            f"class={self.target_class})"
        )


class RuleEngine(ABC):
    """Abstract base class for rule execution engines
    
    A rule engine manages a collection of rules and applies them to point cloud data
    according to a specified strategy (priority, weighted, hierarchical, etc.).
    """
    
    def __init__(
        self,
        rules: List[BaseRule],
        config: Optional[RuleEngineConfig] = None
    ):
        """Initialize rule engine with rules and configuration
        
        Args:
            rules: List of rules to apply
            config: Engine configuration (uses defaults if None)
        """
        self.rules = rules
        self.config = config or RuleEngineConfig()
        
        # Sort rules by priority (highest first)
        self.rules = sorted(
            [r for r in rules if r.enabled],
            key=lambda r: r.priority.value,
            reverse=True
        )
        
        # Statistics
        self.n_executions = 0
        self.total_execution_time = 0.0
        
        logger.info(
            f"Initialized {self.__class__.__name__} with {len(self.rules)} rules"
        )
    
    @abstractmethod
    def apply_rules(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        context: Optional[Dict[str, Any]] = None
    ) -> RuleResult:
        """Apply all rules to point cloud data
        
        Args:
            points: Point coordinates [N, 3]
            features: Dictionary of feature arrays {name: values[N]}
            context: Optional context information
        
        Returns:
            RuleResult with labels, confidence, and statistics
        """
        pass
    
    def validate_all_features(
        self,
        features: Dict[str, np.ndarray],
        n_points: int
    ) -> None:
        """Validate features for all rules
        
        Args:
            features: Feature dictionary to validate
            n_points: Expected number of points
        
        Raises:
            ValueError: If any rule's required features are missing
        """
        if not self.config.validate_inputs:
            return
        
        for rule in self.rules:
            rule.validate_features(features, n_points)
    
    def get_all_required_features(self) -> Set[str]:
        """Get union of all required features across all rules
        
        Returns:
            Set of all feature names required by any rule
        """
        required = set()
        for rule in self.rules:
            required.update(rule.get_required_features())
        return required
    
    def get_all_optional_features(self) -> Set[str]:
        """Get union of all optional features across all rules
        
        Returns:
            Set of all feature names that are optional for any rule
        """
        optional = set()
        for rule in self.rules:
            optional.update(rule.get_optional_features())
        return optional
    
    def get_rules_by_type(self, rule_type: RuleType) -> List[BaseRule]:
        """Get all rules of a specific type
        
        Args:
            rule_type: Type of rules to retrieve
        
        Returns:
            List of rules with matching type
        """
        return [r for r in self.rules if r.rule_type == rule_type]
    
    def get_rules_by_priority(self, priority: RulePriority) -> List[BaseRule]:
        """Get all rules with a specific priority
        
        Args:
            priority: Priority level
        
        Returns:
            List of rules with matching priority
        """
        return [r for r in self.rules if r.priority == priority]
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_rules={len(self.rules)}, "
            f"strategy={self.config.execution_strategy.value})"
        )


# ================================
# Utility Functions
# ================================

def create_empty_result(n_points: int) -> RuleResult:
    """Create an empty RuleResult with no matches
    
    Args:
        n_points: Number of points
    
    Returns:
        RuleResult with zero labels and confidence
    """
    return RuleResult(
        labels=np.zeros(n_points, dtype=np.uint8),
        confidence=np.zeros(n_points, dtype=np.float32),
        rule_ids=[],
        stats=RuleStats(
            total_points=n_points,
            matched_points=0,
            unmatched_points=n_points,
            rules_applied=0
        )
    )


def merge_rule_results(
    results: List[RuleResult],
    strategy: ConflictResolution = ConflictResolution.HIGHEST_CONFIDENCE
) -> RuleResult:
    """Merge multiple RuleResults using specified strategy
    
    Args:
        results: List of RuleResults to merge
        strategy: How to resolve conflicts
    
    Returns:
        Merged RuleResult
    """
    if not results:
        raise ValueError("Cannot merge empty list of results")
    
    if len(results) == 1:
        return results[0]
    
    n_points = len(results[0].labels)
    merged_labels = np.zeros(n_points, dtype=np.uint8)
    merged_confidence = np.zeros(n_points, dtype=np.float32)
    all_rule_ids = []
    
    for result in results:
        all_rule_ids.extend(result.rule_ids)
    
    if strategy == ConflictResolution.HIGHEST_CONFIDENCE:
        # Keep label with highest confidence at each point
        for result in results:
            mask = result.confidence > merged_confidence
            merged_labels[mask] = result.labels[mask]
            merged_confidence[mask] = result.confidence[mask]
    
    elif strategy == ConflictResolution.FIRST_WINS:
        # First result takes precedence
        for result in results:
            mask = merged_labels == 0
            merged_labels[mask] = result.labels[mask]
            merged_confidence[mask] = result.confidence[mask]
    
    elif strategy == ConflictResolution.LAST_WINS:
        # Last result takes precedence
        for result in reversed(results):
            mask = result.labels > 0
            merged_labels[mask] = result.labels[mask]
            merged_confidence[mask] = result.confidence[mask]
    
    # Create merged statistics
    total_matched = np.sum(merged_labels > 0)
    merged_stats = RuleStats(
        total_points=n_points,
        matched_points=total_matched,
        unmatched_points=n_points - total_matched,
        rules_applied=len(all_rule_ids),
        mean_confidence=merged_confidence.mean() if total_matched > 0 else 0.0,
        min_confidence=merged_confidence[merged_labels > 0].min() if total_matched > 0 else 0.0,
        max_confidence=merged_confidence.max()
    )
    
    return RuleResult(
        labels=merged_labels,
        confidence=merged_confidence,
        rule_ids=all_rule_ids,
        stats=merged_stats
    )
