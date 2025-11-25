"""
Model Ensemble System for IGN LiDAR HD Dataset Processing Library.

Provides ensemble learning techniques to combine multiple models for
improved prediction accuracy and robustness in building classification tasks.

Components:
    - EnsembleConfig: Configuration for ensemble models
    - VotingEnsemble: Majority voting across models
    - StackingEnsemble: Meta-learner based ensemble
    - BootstrappingEnsemble: Bootstrap aggregating (bagging)
    - EnsembleInference: Unified inference interface

Author: imagodata
Version: 1.0.0
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import warnings
import logging
from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels

# Optional PyTorch support
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch not available. Some ensemble features disabled. "
        "Install with: pip install torch",
        UserWarning
    )

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble models."""
    
    ensemble_type: str = "voting"
    """Type of ensemble: 'voting', 'stacking', 'bagging', 'boosting'."""
    
    num_models: int = 3
    """Number of models in ensemble."""
    
    voting_type: str = "hard"
    """Voting type for voting ensemble: 'hard', 'soft'."""
    
    bootstrap_size: float = 1.0
    """Bootstrap sample size as fraction of data."""
    
    bootstrap_replace: bool = True
    """Whether to sample with replacement."""
    
    meta_learner: Optional[str] = None
    """Meta-learner for stacking: 'logistic_regression', 'svm', 'random_forest'."""
    
    weights: Optional[List[float]] = None
    """Model weights for weighted voting/averaging."""
    
    diversity_metric: str = "disagreement"
    """Diversity metric: 'disagreement', 'q_statistic', 'correlation'."""
    
    max_workers: int = 4
    """Maximum workers for parallel prediction."""
    
    verbose: bool = False
    """Enable verbose logging."""


class VotingEnsemble:
    """
    Voting ensemble combining predictions from multiple models.
    
    Implements hard voting (majority) and soft voting (probability average)
    for combining predictions from diverse models.
    """
    
    def __init__(
        self,
        models: List[Union[BaseEstimator, "nn.Module"]],
        voting: str = "hard",
        weights: Optional[List[float]] = None,
        device: str = "cuda"
    ):
        """
        Initialize voting ensemble.
        
        Args:
            models: List of trained models
            voting: 'hard' for majority voting, 'soft' for probability average
            weights: Optional model weights
            device: Computation device for PyTorch models
            
        Raises:
            ValueError: If voting type invalid or weights shape mismatches
        """
        if voting not in ["hard", "soft"]:
            raise ValueError(f"Invalid voting type: {voting}")
        
        if weights is not None and len(weights) != len(models):
            raise ValueError("Number of weights must match number of models")
        
        self.models = models
        self.voting = voting
        self.weights = weights or [1.0] * len(models)
        self.device = device
        self.classes_ = None
    
    def _get_predictions(self, X: np.ndarray) -> List[np.ndarray]:
        """Get predictions from all models."""
        predictions = []
        
        for model in self.models:
            if isinstance(model, nn.Module):
                if TORCH_AVAILABLE:
                    with torch.no_grad():
                        X_tensor = torch.from_numpy(X).float().to(self.device)
                        outputs = model(X_tensor)
                        if outputs.dim() > 1:
                            pred = torch.argmax(outputs, dim=1).cpu().numpy()
                        else:
                            pred = outputs.cpu().numpy()
                    predictions.append(pred)
            else:
                pred = model.predict(X)
                predictions.append(pred)
        
        return predictions
    
    def _get_probabilities(self, X: np.ndarray) -> List[np.ndarray]:
        """Get probability predictions from all models."""
        probabilities = []
        
        for model in self.models:
            if isinstance(model, nn.Module):
                if TORCH_AVAILABLE:
                    with torch.no_grad():
                        X_tensor = torch.from_numpy(X).float().to(self.device)
                        outputs = model(X_tensor)
                        probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    probabilities.append(probs)
            else:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)
                    probabilities.append(probs)
        
        return probabilities
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input features [num_samples, num_features]
            
        Returns:
            Predicted labels [num_samples]
        """
        if self.voting == "hard":
            predictions = self._get_predictions(X)
            # Majority voting
            num_samples = len(X)
            votes = np.zeros(num_samples, dtype=int)
            
            for pred, weight in zip(predictions, self.weights):
                votes += np.array(pred) * weight
            
            return votes
        else:
            probabilities = self._get_probabilities(X)
            # Average probabilities
            avg_probs = np.zeros_like(probabilities[0])
            total_weight = sum(self.weights)
            
            for probs, weight in zip(probabilities, self.weights):
                avg_probs += probs * weight / total_weight
            
            return np.argmax(avg_probs, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities [num_samples, num_classes]
        """
        probabilities = self._get_probabilities(X)
        avg_probs = np.zeros_like(probabilities[0])
        total_weight = sum(self.weights)
        
        for probs, weight in zip(probabilities, self.weights):
            avg_probs += probs * weight / total_weight
        
        return avg_probs
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "VotingEnsemble":
        """Store class labels (required for sklearn compatibility)."""
        self.classes_ = unique_labels(y)
        return self


class StackingEnsemble:
    """
    Stacking ensemble using meta-learner.
    
    Trains a meta-learner on predictions from base models to learn
    optimal combination weights.
    """
    
    def __init__(
        self,
        base_models: List[Union[BaseEstimator, "nn.Module"]],
        meta_learner: BaseEstimator,
        device: str = "cuda"
    ):
        """
        Initialize stacking ensemble.
        
        Args:
            base_models: List of base models
            meta_learner: Meta-learner model
            device: Computation device
        """
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.device = device
        self.classes_ = None
    
    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Generate meta-features from base models."""
        meta_features = []
        
        for model in self.base_models:
            if isinstance(model, nn.Module):
                if TORCH_AVAILABLE:
                    with torch.no_grad():
                        X_tensor = torch.from_numpy(X).float().to(self.device)
                        outputs = model(X_tensor)
                        probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    meta_features.append(probs)
            else:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)
                else:
                    pred = model.predict(X)
                    probs = np.eye(len(self.classes_))[pred]
                meta_features.append(probs)
        
        return np.hstack(meta_features)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingEnsemble":
        """
        Fit stacking ensemble.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Self
        """
        self.classes_ = unique_labels(y)
        
        # Generate meta-features
        meta_features = self._get_meta_features(X)
        
        # Fit meta-learner
        self.meta_learner.fit(meta_features, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using meta-learner.
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels
        """
        meta_features = self._get_meta_features(X)
        return self.meta_learner.predict(meta_features)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using meta-learner.
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities
        """
        meta_features = self._get_meta_features(X)
        if hasattr(self.meta_learner, "predict_proba"):
            return self.meta_learner.predict_proba(meta_features)
        else:
            pred = self.meta_learner.predict(meta_features)
            probs = np.eye(len(self.classes_))[pred]
            return probs


class BootstrappingEnsemble:
    """
    Bootstrap aggregating (bagging) ensemble.
    
    Trains models on bootstrap samples and aggregates predictions
    for reduced variance and improved generalization.
    """
    
    def __init__(
        self,
        base_model_class: type,
        n_estimators: int = 10,
        bootstrap_size: float = 1.0,
        replace: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Initialize bagging ensemble.
        
        Args:
            base_model_class: Class for base models
            n_estimators: Number of base estimators
            bootstrap_size: Size of bootstrap samples
            replace: Whether to sample with replacement
            random_state: Random seed
        """
        self.base_model_class = base_model_class
        self.n_estimators = n_estimators
        self.bootstrap_size = bootstrap_size
        self.replace = replace
        self.random_state = random_state
        self.models = []
        self.classes_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BootstrappingEnsemble":
        """
        Fit bagging ensemble.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Self
        """
        self.classes_ = unique_labels(y)
        rng = np.random.RandomState(self.random_state)
        
        n_samples = int(len(X) * self.bootstrap_size)
        
        for i in range(self.n_estimators):
            # Bootstrap sample
            indices = rng.choice(
                len(X),
                size=n_samples,
                replace=self.replace
            )
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Train model
            model = self.base_model_class()
            model.fit(X_boot, y_boot)
            self.models.append(model)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict by majority voting.
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels
        """
        predictions = np.array([
            model.predict(X) for model in self.models
        ])
        
        # Majority voting
        return np.array([
            Counter(pred).most_common(1)[0][0]
            for pred in predictions.T
        ])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities by averaging.
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities
        """
        probabilities = np.array([
            model.predict_proba(X) if hasattr(model, "predict_proba")
            else np.eye(len(self.classes_))[model.predict(X)]
            for model in self.models
        ])
        
        return np.mean(probabilities, axis=0)


class EnsembleInference:
    """
    Unified inference interface for ensemble models.
    
    Provides consistent prediction and analysis interface across
    different ensemble types.
    """
    
    def __init__(
        self,
        ensemble: Union[VotingEnsemble, StackingEnsemble, BootstrappingEnsemble],
        config: Optional[EnsembleConfig] = None,
        device: str = "cuda"
    ):
        """
        Initialize ensemble inference.
        
        Args:
            ensemble: Trained ensemble model
            config: Ensemble configuration
            device: Computation device
        """
        self.ensemble = ensemble
        self.config = config or EnsembleConfig()
        self.device = device
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels
        """
        return self.ensemble.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities
        """
        if hasattr(self.ensemble, "predict_proba"):
            return self.ensemble.predict_proba(X)
        else:
            logger.warning("Ensemble does not support predict_proba")
            return None
    
    def predict_with_confidence(
        self,
        X: np.ndarray,
        return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict with confidence estimates.
        
        Args:
            X: Input features
            return_std: Whether to return standard deviation
            
        Returns:
            Predictions and optionally confidence/std
        """
        proba = self.predict_proba(X)
        predictions = np.argmax(proba, axis=1)
        confidence = np.max(proba, axis=1)
        
        if return_std:
            # Estimate std from probability distribution
            std = np.std(proba, axis=1)
            return predictions, confidence, std
        
        return predictions, confidence
    
    def get_disagreement(self, X: np.ndarray) -> np.ndarray:
        """
        Get disagreement between ensemble members.
        
        Args:
            X: Input features
            
        Returns:
            Disagreement scores [num_samples]
        """
        predictions = []
        
        if isinstance(self.ensemble, VotingEnsemble):
            for model in self.ensemble.models:
                if isinstance(model, nn.Module):
                    if TORCH_AVAILABLE:
                        with torch.no_grad():
                            X_tensor = torch.from_numpy(X).float().to(self.device)
                            outputs = model(X_tensor)
                            pred = torch.argmax(outputs, dim=1).cpu().numpy()
                        predictions.append(pred)
                else:
                    pred = model.predict(X)
                    predictions.append(pred)
            
            predictions = np.array(predictions)
            
            # Compute disagreement as ratio of incorrect predictions
            disagreement = np.zeros(len(X))
            for i in range(len(X)):
                ensemble_pred = np.argmax(np.bincount(predictions[:, i]))
                disagreement[i] = np.mean(predictions[:, i] != ensemble_pred)
            
            return disagreement
        
        return np.zeros(len(X))
    
    def get_model_contribution(self, X: np.ndarray) -> Dict[int, float]:
        """
        Estimate contribution of each model to predictions.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary mapping model index to contribution score
        """
        contribution = {}
        
        if isinstance(self.ensemble, VotingEnsemble):
            for i, model in enumerate(self.ensemble.models):
                if hasattr(model, "score"):
                    contribution[i] = model.score(X, np.zeros(len(X)))
                else:
                    contribution[i] = self.ensemble.weights[i]
        
        return contribution


# Export public API
__all__ = [
    "EnsembleConfig",
    "VotingEnsemble",
    "StackingEnsemble",
    "BootstrappingEnsemble",
    "EnsembleInference",
    "TORCH_AVAILABLE"
]
