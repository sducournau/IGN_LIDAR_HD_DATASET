"""
Hierarchical rule execution utilities.

This module provides classes and functions for hierarchical rule application:
- Rule levels and hierarchies
- Multi-level execution strategies
- Priority-based execution
- Hierarchical conflict resolution

Usage:
    from ign_lidar.core.classification.rules.hierarchy import (
        RuleLevel,
        HierarchicalRuleEngine
    )
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import logging
import time

from .base import (
    BaseRule,
    RuleEngine,
    RuleResult,
    RuleStats,
    RuleEngineConfig,
    ExecutionStrategy,
    ConflictResolution
)

logger = logging.getLogger(__name__)


@dataclass
class RuleLevel:
    """Represents one level in a rule hierarchy
    
    Attributes:
        level: Integer level (0 = highest priority, 1 = next, etc.)
        rules: List of rules at this level
        strategy: How to apply rules within this level
        description: Human-readable description of this level
    """
    level: int
    rules: List[BaseRule]
    strategy: str = "first_match"  # or "all_matches", "weighted"
    description: str = ""
    
    def __post_init__(self):
        """Validate level configuration"""
        if self.level < 0:
            raise ValueError(f"Level must be >= 0, got {self.level}")
        
        if not self.rules:
            logger.warning(f"Level {self.level} has no rules")
        
        # Validate strategy
        valid_strategies = ["first_match", "all_matches", "weighted", "priority"]
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"Invalid strategy '{self.strategy}', "
                f"must be one of {valid_strategies}"
            )
    
    def __repr__(self) -> str:
        return (
            f"RuleLevel(level={self.level}, "
            f"n_rules={len(self.rules)}, "
            f"strategy={self.strategy})"
        )


class HierarchicalRuleEngine(RuleEngine):
    """Rule engine with hierarchical execution
    
    Rules are organized into levels, with higher levels (lower level number)
    taking precedence over lower levels. Within each level, a strategy
    determines how rules are applied (first match, all matches, etc.).
    
    Example:
        Level 0: Critical rules (ground, water) - first match
        Level 1: Building rules - all matches
        Level 2: Vegetation rules - weighted
        Level 3: Default rules - first match
    """
    
    def __init__(
        self,
        levels: List[RuleLevel],
        config: Optional[RuleEngineConfig] = None
    ):
        """Initialize hierarchical rule engine
        
        Args:
            levels: List of rule levels (will be sorted by level number)
            config: Engine configuration
        """
        # Extract all rules from levels
        all_rules = []
        for level in levels:
            all_rules.extend(level.rules)
        
        # Initialize base class
        super().__init__(all_rules, config)
        
        # Sort levels by level number (ascending)
        self.levels = sorted(levels, key=lambda l: l.level)
        
        logger.info(
            f"Initialized hierarchical engine with {len(self.levels)} levels, "
            f"{len(all_rules)} total rules"
        )
    
    def apply_rules(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        context: Optional[Dict[str, Any]] = None
    ) -> RuleResult:
        """Apply rules hierarchically
        
        Args:
            points: Point coordinates [N, 3]
            features: Dictionary of feature arrays
            context: Optional context information
        
        Returns:
            RuleResult with hierarchically applied labels and confidence
        """
        start_time = time.time()
        n_points = len(points)
        
        # Validate inputs if configured
        if self.config.validate_inputs:
            self.validate_all_features(features, n_points)
        
        # Initialize result arrays
        labels = np.zeros(n_points, dtype=np.uint8)
        confidence = np.zeros(n_points, dtype=np.float32)
        applied_rules = []
        rule_match_counts = {}
        rule_execution_times = {}
        
        # Track which points have been classified
        unclassified_mask = np.ones(n_points, dtype=bool)
        
        # Apply each level in order
        for level in self.levels:
            if not level.rules:
                continue
            
            logger.debug(
                f"Applying level {level.level} with {len(level.rules)} rules "
                f"({level.strategy} strategy)"
            )
            
            # Apply rules at this level
            level_result = self._apply_level(
                level=level,
                points=points,
                features=features,
                context=context,
                unclassified_mask=unclassified_mask
            )
            
            # Update global results
            # Only update points that were previously unclassified
            update_mask = unclassified_mask & (level_result.labels > 0)
            
            labels[update_mask] = level_result.labels[update_mask]
            confidence[update_mask] = level_result.confidence[update_mask]
            
            # Update unclassified mask
            unclassified_mask = labels == 0
            
            # Track statistics
            applied_rules.extend(level_result.rule_ids)
            rule_match_counts.update(level_result.stats.rule_match_counts)
            rule_execution_times.update(level_result.stats.rule_execution_times)
            
            # Early exit if all points classified
            if not np.any(unclassified_mask):
                logger.debug(f"All points classified at level {level.level}")
                break
        
        # Calculate final statistics
        execution_time = (time.time() - start_time) * 1000  # ms
        matched_points = np.sum(labels > 0)
        unmatched_points = n_points - matched_points
        
        stats = RuleStats(
            total_points=n_points,
            matched_points=matched_points,
            unmatched_points=unmatched_points,
            rules_applied=len(applied_rules),
            execution_time_ms=execution_time,
            rule_match_counts=rule_match_counts,
            rule_execution_times=rule_execution_times,
            mean_confidence=confidence[labels > 0].mean() if matched_points > 0 else 0.0,
            min_confidence=confidence[labels > 0].min() if matched_points > 0 else 0.0,
            max_confidence=confidence.max()
        )
        
        logger.info(
            f"Hierarchical execution complete: {matched_points}/{n_points} "
            f"points classified ({stats.coverage:.1%}) in {execution_time:.1f}ms"
        )
        
        return RuleResult(
            labels=labels,
            confidence=confidence,
            rule_ids=applied_rules,
            stats=stats,
            metadata={'n_levels': len(self.levels)}
        )
    
    def _apply_level(
        self,
        level: RuleLevel,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        context: Optional[Dict[str, Any]],
        unclassified_mask: np.ndarray
    ) -> RuleResult:
        """Apply all rules at a single level
        
        Args:
            level: Rule level to apply
            points: Point coordinates
            features: Feature dictionary
            context: Optional context
            unclassified_mask: Mask of points not yet classified
        
        Returns:
            RuleResult for this level
        """
        n_points = len(points)
        labels = np.zeros(n_points, dtype=np.uint8)
        confidence = np.zeros(n_points, dtype=np.float32)
        applied_rules = []
        rule_match_counts = {}
        rule_execution_times = {}
        
        if level.strategy == "first_match":
            # Apply rules until first match for each point
            for rule in level.rules:
                # Only evaluate on unclassified points
                eval_mask = unclassified_mask & (labels == 0)
                if not np.any(eval_mask):
                    break
                
                # Evaluate rule
                rule_start = time.time()
                match_mask, conf_scores = rule.evaluate(points, features, context)
                rule_time = (time.time() - rule_start) * 1000
                
                # Update labels for matched unclassified points
                update_mask = eval_mask & match_mask
                labels[update_mask] = rule.target_class
                confidence[update_mask] = conf_scores[update_mask]
                
                # Track statistics
                n_matches = np.sum(update_mask)
                if n_matches > 0:
                    applied_rules.append(rule.rule_id)
                    rule_match_counts[rule.rule_id] = n_matches
                    rule_execution_times[rule.rule_id] = rule_time
        
        elif level.strategy == "all_matches":
            # Apply all rules, combine with weighted voting
            rule_votes = {}  # {class_id: {point_idx: confidence}}
            
            for rule in level.rules:
                # Evaluate rule
                rule_start = time.time()
                match_mask, conf_scores = rule.evaluate(points, features, context)
                rule_time = (time.time() - rule_start) * 1000
                
                # Record votes
                matched_indices = np.where(match_mask)[0]
                if len(matched_indices) > 0:
                    if rule.target_class not in rule_votes:
                        rule_votes[rule.target_class] = {}
                    
                    for idx in matched_indices:
                        if idx in rule_votes[rule.target_class]:
                            # Combine confidences (max)
                            rule_votes[rule.target_class][idx] = max(
                                rule_votes[rule.target_class][idx],
                                conf_scores[idx]
                            )
                        else:
                            rule_votes[rule.target_class][idx] = conf_scores[idx]
                    
                    applied_rules.append(rule.rule_id)
                    rule_match_counts[rule.rule_id] = len(matched_indices)
                    rule_execution_times[rule.rule_id] = rule_time
            
            # Assign labels based on votes (highest confidence wins)
            for class_id, votes in rule_votes.items():
                for point_idx, conf in votes.items():
                    if conf > confidence[point_idx]:
                        labels[point_idx] = class_id
                        confidence[point_idx] = conf
        
        elif level.strategy == "priority":
            # Apply rules in priority order (already sorted)
            for rule in level.rules:
                # Only evaluate on unclassified points
                eval_mask = unclassified_mask & (labels == 0)
                if not np.any(eval_mask):
                    break
                
                rule_start = time.time()
                match_mask, conf_scores = rule.evaluate(points, features, context)
                rule_time = (time.time() - rule_start) * 1000
                
                update_mask = eval_mask & match_mask
                labels[update_mask] = rule.target_class
                confidence[update_mask] = conf_scores[update_mask]
                
                n_matches = np.sum(update_mask)
                if n_matches > 0:
                    applied_rules.append(rule.rule_id)
                    rule_match_counts[rule.rule_id] = n_matches
                    rule_execution_times[rule.rule_id] = rule_time
        
        # Create level result
        matched_points = np.sum(labels > 0)
        stats = RuleStats(
            total_points=n_points,
            matched_points=matched_points,
            unmatched_points=n_points - matched_points,
            rules_applied=len(applied_rules),
            rule_match_counts=rule_match_counts,
            rule_execution_times=rule_execution_times
        )
        
        return RuleResult(
            labels=labels,
            confidence=confidence,
            rule_ids=applied_rules,
            stats=stats
        )
