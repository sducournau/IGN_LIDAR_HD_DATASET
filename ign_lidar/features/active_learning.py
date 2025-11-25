"""
Active Learning Framework for IGN LiDAR HD Dataset Processing Library.

Implements active learning strategies to intelligently select data points
for annotation, reducing labeling effort while maintaining model accuracy
in building classification tasks.

Components:
    - UncertaintySampling: Select uncertain predictions
    - DiversitySampling: Select diverse samples
    - QueryByCommittee: Ensemble-based query strategy
    - HybridSampling: Combine multiple strategies
    - ActiveLearner: Complete active learning pipeline

Author: imagodata
Version: 1.0.0
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
import logging

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans

# Optional PyTorch support
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch not available. Some active learning features disabled.",
        UserWarning
    )

logger = logging.getLogger(__name__)


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning."""
    
    strategy: str = "uncertainty"
    """Query strategy: 'uncertainty', 'diversity', 'committee', 'hybrid'."""
    
    query_size: int = 10
    """Number of samples to query at each iteration."""
    
    batch_size: int = 100
    """Batch size for processing."""
    
    random_state: int = 42
    """Random seed for reproducibility."""
    
    diversity_metric: str = "euclidean"
    """Distance metric for diversity: 'euclidean', 'cosine', 'manhattan'."""
    
    uncertainty_type: str = "entropy"
    """Uncertainty measure: 'entropy', 'margin', 'confidence'."""
    
    clustering_n: Optional[int] = None
    """Number of clusters for diversity sampling."""
    
    committee_size: int = 5
    """Number of models in committee."""
    
    hybrid_weights: Optional[Dict[str, float]] = None
    """Weights for hybrid sampling strategies."""
    
    verbose: bool = False
    """Enable verbose logging."""


class QueryStrategy(ABC):
    """Base class for query strategies."""
    
    @abstractmethod
    def query(
        self,
        X_unlabeled: np.ndarray,
        model: Any,
        n_instances: int
    ) -> np.ndarray:
        """
        Query instances from unlabeled pool.
        
        Args:
            X_unlabeled: Unlabeled data [num_samples, features]
            model: Trained model
            n_instances: Number of instances to query
            
        Returns:
            Indices of selected instances
        """
        pass


class UncertaintySampling(QueryStrategy):
    """
    Uncertainty sampling strategy.
    
    Selects samples where the model is most uncertain about predictions.
    Implements entropy, margin-based, and confidence-based uncertainty measures.
    """
    
    def __init__(self, uncertainty_type: str = "entropy", device: str = "cuda"):
        """
        Initialize uncertainty sampler.
        
        Args:
            uncertainty_type: 'entropy', 'margin', 'confidence'
            device: Computation device
        """
        self.uncertainty_type = uncertainty_type
        self.device = device
    
    def _compute_entropy(self, proba: np.ndarray) -> np.ndarray:
        """
        Compute entropy of predictions.
        
        Args:
            proba: Probability predictions [num_samples, num_classes]
            
        Returns:
            Entropy scores [num_samples]
        """
        # Avoid log(0)
        proba = np.clip(proba, 1e-10, 1.0)
        entropy = -np.sum(proba * np.log(proba), axis=1)
        return entropy
    
    def _compute_margin(self, proba: np.ndarray) -> np.ndarray:
        """
        Compute margin between top two predictions.
        
        Args:
            proba: Probability predictions
            
        Returns:
            Margin scores [num_samples]
        """
        sorted_proba = np.sort(proba, axis=1)[:, ::-1]
        margin = sorted_proba[:, 0] - sorted_proba[:, 1]
        return 1.0 - margin  # Return uncertainty (lower margin = higher uncertainty)
    
    def _compute_confidence(self, proba: np.ndarray) -> np.ndarray:
        """
        Compute confidence as negative max probability.
        
        Args:
            proba: Probability predictions
            
        Returns:
            Confidence scores [num_samples]
        """
        return 1.0 - np.max(proba, axis=1)
    
    def query(
        self,
        X_unlabeled: np.ndarray,
        model: Any,
        n_instances: int
    ) -> np.ndarray:
        """
        Select uncertain samples.
        
        Args:
            X_unlabeled: Unlabeled data
            model: Trained model with predict_proba
            n_instances: Number to select
            
        Returns:
            Indices of most uncertain samples
        """
        if not hasattr(model, "predict_proba"):
            raise ValueError("Model must have predict_proba method")
        
        # Get predictions
        proba = model.predict_proba(X_unlabeled)
        
        # Compute uncertainty
        if self.uncertainty_type == "entropy":
            uncertainty = self._compute_entropy(proba)
        elif self.uncertainty_type == "margin":
            uncertainty = self._compute_margin(proba)
        elif self.uncertainty_type == "confidence":
            uncertainty = self._compute_confidence(proba)
        else:
            raise ValueError(f"Unknown uncertainty type: {self.uncertainty_type}")
        
        # Select top uncertain samples
        top_indices = np.argsort(uncertainty)[-n_instances:]
        
        return top_indices


class DiversitySampling(QueryStrategy):
    """
    Diversity sampling strategy.
    
    Selects samples that are diverse from already labeled set.
    Uses clustering or distance-based approaches.
    """
    
    def __init__(self, metric: str = "euclidean", n_clusters: Optional[int] = None):
        """
        Initialize diversity sampler.
        
        Args:
            metric: Distance metric ('euclidean', 'cosine', 'manhattan')
            n_clusters: Number of clusters for clustering-based diversity
        """
        self.metric = metric
        self.n_clusters = n_clusters
    
    def _distance_based_diversity(
        self,
        X_unlabeled: np.ndarray,
        X_labeled: np.ndarray,
        n_instances: int
    ) -> np.ndarray:
        """
        Select diverse samples using distance to labeled set.
        
        Args:
            X_unlabeled: Unlabeled data
            X_labeled: Labeled data
            n_instances: Number to select
            
        Returns:
            Indices of diverse samples
        """
        # Compute distances to nearest labeled point
        distances = euclidean_distances(X_unlabeled, X_labeled)
        min_distances = np.min(distances, axis=1)
        
        # Select farthest from labeled set
        top_indices = np.argsort(min_distances)[-n_instances:]
        
        return top_indices
    
    def _clustering_based_diversity(
        self,
        X_unlabeled: np.ndarray,
        n_instances: int
    ) -> np.ndarray:
        """
        Select diverse samples using clustering.
        
        Args:
            X_unlabeled: Unlabeled data
            n_instances: Number to select
            
        Returns:
            Indices of diverse samples
        """
        n_clusters = self.n_clusters or max(n_instances, 5)
        
        # Cluster unlabeled data
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_unlabeled)
        
        # Select closest to cluster centers
        selected = []
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            if np.any(cluster_mask):
                cluster_indices = np.where(cluster_mask)[0]
                center = kmeans.cluster_centers_[cluster_id]
                
                # Distance to center
                distances = euclidean_distances(
                    X_unlabeled[cluster_mask],
                    center.reshape(1, -1)
                ).flatten()
                
                selected_idx = cluster_indices[np.argmin(distances)]
                selected.append(selected_idx)
        
        # Return top diverse samples
        return np.array(selected[:n_instances])
    
    def query(
        self,
        X_unlabeled: np.ndarray,
        X_labeled: np.ndarray,
        n_instances: int
    ) -> np.ndarray:
        """
        Select diverse samples.
        
        Args:
            X_unlabeled: Unlabeled data
            X_labeled: Labeled data
            n_instances: Number to select
            
        Returns:
            Indices of diverse samples
        """
        if X_labeled is not None and len(X_labeled) > 0:
            return self._distance_based_diversity(
                X_unlabeled, X_labeled, n_instances
            )
        else:
            return self._clustering_based_diversity(
                X_unlabeled, n_instances
            )


class QueryByCommittee(QueryStrategy):
    """
    Query by committee strategy.
    
    Selects samples where the ensemble of models disagree most,
    indicating high uncertainty.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize QBC strategy.
        
        Args:
            device: Computation device
        """
        self.device = device
    
    def query(
        self,
        X_unlabeled: np.ndarray,
        ensemble: Any,
        n_instances: int
    ) -> np.ndarray:
        """
        Select disagreement-maximizing samples.
        
        Args:
            X_unlabeled: Unlabeled data
            ensemble: Ensemble of models
            n_instances: Number to select
            
        Returns:
            Indices of samples with highest disagreement
        """
        if not hasattr(ensemble, "models"):
            raise ValueError("Ensemble must have models attribute")
        
        predictions = []
        for model in ensemble.models:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_unlabeled)
                pred = np.argmax(proba, axis=1)
            else:
                pred = model.predict(X_unlabeled)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Compute disagreement (vote entropy)
        disagreement = np.zeros(len(X_unlabeled))
        for i in range(len(X_unlabeled)):
            # Count votes for each class
            votes = predictions[:, i]
            unique, counts = np.unique(votes, return_counts=True)
            
            # Entropy of vote distribution
            probs = counts / len(votes)
            disagreement[i] = -np.sum(probs * np.log(probs + 1e-10))
        
        # Select most disagreed upon samples
        top_indices = np.argsort(disagreement)[-n_instances:]
        
        return top_indices


class HybridSampling(QueryStrategy):
    """
    Hybrid sampling combining multiple strategies.
    
    Combines uncertainty, diversity, and committee strategies with
    configurable weights for balanced sample selection.
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        device: str = "cuda"
    ):
        """
        Initialize hybrid sampler.
        
        Args:
            weights: Dictionary of strategy weights
            device: Computation device
        """
        self.weights = weights or {
            "uncertainty": 0.5,
            "diversity": 0.3,
            "committee": 0.2
        }
        self.device = device
    
    def query(
        self,
        X_unlabeled: np.ndarray,
        model: Any,
        X_labeled: Optional[np.ndarray],
        ensemble: Optional[Any],
        n_instances: int
    ) -> np.ndarray:
        """
        Select samples using hybrid strategy.
        
        Args:
            X_unlabeled: Unlabeled data
            model: Primary model
            X_labeled: Labeled data
            ensemble: Ensemble for committee voting
            n_instances: Number to select
            
        Returns:
            Indices of selected samples
        """
        scores = np.zeros(len(X_unlabeled))
        norm_factor = sum(self.weights.values())
        
        # Uncertainty sampling
        if "uncertainty" in self.weights and self.weights["uncertainty"] > 0:
            uncertainty_sampler = UncertaintySampling(device=self.device)
            uncertainty_scores = np.zeros(len(X_unlabeled))
            
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_unlabeled)
                uncertainty_scores = -np.max(proba, axis=1)
            
            scores += uncertainty_scores * self.weights["uncertainty"] / norm_factor
        
        # Diversity sampling
        if "diversity" in self.weights and self.weights["diversity"] > 0:
            diversity_sampler = DiversitySampling()
            if X_labeled is not None and len(X_labeled) > 0:
                distances = euclidean_distances(X_unlabeled, X_labeled)
                diversity_scores = np.min(distances, axis=1)
                scores += diversity_scores * self.weights["diversity"] / norm_factor
        
        # Committee voting
        if "committee" in self.weights and self.weights["committee"] > 0:
            if ensemble is not None:
                committee_sampler = QueryByCommittee(device=self.device)
                # Get disagreement scores
                predictions = []
                for model_i in ensemble.models:
                    if hasattr(model_i, "predict_proba"):
                        proba = model_i.predict_proba(X_unlabeled)
                        pred = np.argmax(proba, axis=1)
                    else:
                        pred = model_i.predict(X_unlabeled)
                    predictions.append(pred)
                
                predictions = np.array(predictions)
                disagreement = np.zeros(len(X_unlabeled))
                
                for i in range(len(X_unlabeled)):
                    votes = predictions[:, i]
                    unique, counts = np.unique(votes, return_counts=True)
                    probs = counts / len(votes)
                    disagreement[i] = -np.sum(probs * np.log(probs + 1e-10))
                
                scores += disagreement * self.weights["committee"] / norm_factor
        
        # Select top scoring samples
        top_indices = np.argsort(scores)[-n_instances:]
        
        return top_indices


class ActiveLearner:
    """
    Active learning pipeline orchestrator.
    
    Manages the complete active learning workflow including model training,
    uncertainty estimation, sample selection, and iterative labeling.
    """
    
    def __init__(
        self,
        model: Any,
        config: Optional[ActiveLearningConfig] = None,
        device: str = "cuda"
    ):
        """
        Initialize active learner.
        
        Args:
            model: Base model for active learning
            config: Active learning configuration
            device: Computation device
        """
        self.model = model
        self.config = config or ActiveLearningConfig()
        self.device = device
        
        self.labeled_indices = np.array([], dtype=int)
        self.unlabeled_indices = None
        self.selected_history = []
        self.performance_history = []
    
    def initialize(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        n_initial: int = 10
    ) -> np.ndarray:
        """
        Initialize active learning with initial labeled set.
        
        Args:
            X: All available data
            y: Labels if pre-labeled
            n_initial: Number of initial labeled samples
            
        Returns:
            Indices of initial labeled samples
        """
        n_total = len(X)
        all_indices = np.arange(n_total)
        
        if y is not None:
            # Use stratified sampling
            unique_classes = np.unique(y)
            selected = []
            for class_id in unique_classes:
                class_indices = np.where(y == class_id)[0]
                n_class = max(1, n_initial // len(unique_classes))
                selected.extend(
                    np.random.choice(class_indices, size=n_class, replace=False)
                )
            self.labeled_indices = np.array(selected)
        else:
            # Random selection
            self.labeled_indices = np.random.choice(
                all_indices, size=n_initial, replace=False
            )
        
        self.unlabeled_indices = np.setdiff1d(all_indices, self.labeled_indices)
        
        return self.labeled_indices
    
    def query(
        self,
        X: np.ndarray,
        ensemble: Optional[Any] = None
    ) -> np.ndarray:
        """
        Query next batch of samples for labeling.
        
        Args:
            X: All available data
            ensemble: Optional ensemble for committee-based strategies
            
        Returns:
            Indices of selected samples
        """
        X_unlabeled = X[self.unlabeled_indices]
        
        if self.config.strategy == "uncertainty":
            sampler = UncertaintySampling(
                uncertainty_type=self.config.uncertainty_type,
                device=self.device
            )
            selected_rel = sampler.query(
                X_unlabeled, self.model, self.config.query_size
            )
        
        elif self.config.strategy == "diversity":
            X_labeled = X[self.labeled_indices]
            sampler = DiversitySampling(
                metric=self.config.diversity_metric,
                n_clusters=self.config.clustering_n
            )
            selected_rel = sampler.query(
                X_unlabeled, X_labeled, self.config.query_size
            )
        
        elif self.config.strategy == "committee":
            sampler = QueryByCommittee(device=self.device)
            selected_rel = sampler.query(
                X_unlabeled, ensemble, self.config.query_size
            )
        
        elif self.config.strategy == "hybrid":
            sampler = HybridSampling(
                weights=self.config.hybrid_weights,
                device=self.device
            )
            X_labeled = X[self.labeled_indices] if len(self.labeled_indices) > 0 else None
            selected_rel = sampler.query(
                X_unlabeled, self.model, X_labeled, ensemble, self.config.query_size
            )
        
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")
        
        # Convert to absolute indices
        selected_abs = self.unlabeled_indices[selected_rel]
        
        # Update indices
        self.labeled_indices = np.append(self.labeled_indices, selected_abs)
        self.unlabeled_indices = np.setdiff1d(
            np.arange(len(X)), self.labeled_indices
        )
        
        self.selected_history.append(selected_abs)
        
        return selected_abs
    
    def get_labeled_pool(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get labeled data pool."""
        return X[self.labeled_indices], y[self.labeled_indices]
    
    def get_unlabeled_pool(self, X: np.ndarray) -> np.ndarray:
        """Get unlabeled data pool."""
        return X[self.unlabeled_indices]


# Export public API
__all__ = [
    "ActiveLearningConfig",
    "QueryStrategy",
    "UncertaintySampling",
    "DiversitySampling",
    "QueryByCommittee",
    "HybridSampling",
    "ActiveLearner",
    "TORCH_AVAILABLE"
]
