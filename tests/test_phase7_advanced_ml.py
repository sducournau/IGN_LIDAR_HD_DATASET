"""
Comprehensive test suite for Phase 7: Advanced ML Features.

Tests transfer learning, model ensemble, and active learning components.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, Mock
import warnings

# Import test data fixtures
pytestmark = pytest.mark.unit

# Check for optional dependencies
torch_available = False
try:
    import torch
    import torch.nn as nn
    torch_available = True
except ImportError:
    pass

sklearn_available = False
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    sklearn_available = True
except ImportError:
    pass


# ============================================================================
# PHASE 7.1: TRANSFER LEARNING TESTS
# ============================================================================

class TestTransferConfig:
    """Test TransferConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        from ign_lidar.features.transfer_learning import TransferConfig
        
        config = TransferConfig()
        assert config.learning_rate == 1e-3
        assert config.freeze_backbone is True
        assert config.warmup_epochs == 5
        assert config.use_domain_adaptation is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        from ign_lidar.features.transfer_learning import TransferConfig
        
        config = TransferConfig(
            learning_rate=1e-4,
            freeze_backbone=False,
            num_epochs=100
        )
        assert config.learning_rate == 1e-4
        assert config.freeze_backbone is False
        assert config.num_epochs == 100


@pytest.mark.skipif(not torch_available, reason="PyTorch not available")
class TestFeatureExtractor:
    """Test FeatureExtractor component."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 64)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(64, 5)
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                return self.fc2(x)
        
        return SimpleModel()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        X = np.random.randn(32, 10).astype(np.float32)
        return X
    
    def test_feature_extractor_initialization(self, simple_model):
        """Test feature extractor initialization."""
        from ign_lidar.features.transfer_learning import FeatureExtractor
        
        extractor = FeatureExtractor(simple_model, "fc1", device="cpu")
        assert extractor.layer_name == "fc1"
        assert extractor.device == "cpu"
    
    def test_feature_extraction(self, simple_model, sample_data):
        """Test feature extraction."""
        from ign_lidar.features.transfer_learning import FeatureExtractor
        
        extractor = FeatureExtractor(simple_model, "relu", device="cpu")
        simple_model.eval()
        
        features = extractor.extract(sample_data)
        assert features.shape[0] == len(sample_data)
    
    def test_invalid_layer_raises_error(self, simple_model):
        """Test that invalid layer raises error."""
        from ign_lidar.features.transfer_learning import FeatureExtractor
        
        with pytest.raises(ValueError):
            FeatureExtractor(simple_model, "nonexistent_layer")


@pytest.mark.skipif(not torch_available, reason="PyTorch not available")
class TestDomainAdapter:
    """Test DomainAdapter component."""
    
    @pytest.fixture
    def domain_data(self):
        """Create domain adaptation data."""
        source = np.random.randn(100, 20).astype(np.float32)
        target = np.random.randn(80, 20).astype(np.float32)
        return source, target
    
    def test_mmd_loss_computation(self, domain_data):
        """Test MMD loss computation."""
        from ign_lidar.features.transfer_learning import DomainAdapter
        
        source, target = domain_data
        adapter = DomainAdapter(source, target, method="mmd", device="cpu")
        
        source_t = torch.from_numpy(source[:16]).float()
        target_t = torch.from_numpy(target[:16]).float()
        
        loss = adapter.compute_mmd_loss(source_t, target_t)
        assert loss.item() >= 0.0
    
    def test_coral_loss_computation(self, domain_data):
        """Test CORAL loss computation."""
        from ign_lidar.features.transfer_learning import DomainAdapter
        
        source, target = domain_data
        adapter = DomainAdapter(source, target, method="coral", device="cpu")
        
        source_t = torch.from_numpy(source[:16]).float()
        target_t = torch.from_numpy(target[:16]).float()
        
        loss = adapter.compute_coral_loss(source_t, target_t)
        assert loss.item() >= 0.0
    
    def test_invalid_method_raises_error(self, domain_data):
        """Test that invalid method raises error."""
        from ign_lidar.features.transfer_learning import DomainAdapter
        
        source, target = domain_data
        with pytest.raises(ValueError):
            DomainAdapter(source, target, method="invalid_method")


@pytest.mark.skipif(not torch_available, reason="PyTorch not available")
class TestProgressiveUnfreezing:
    """Test ProgressiveUnfreezing component."""
    
    @pytest.fixture
    def model_for_unfreezing(self):
        """Create model for unfreezing tests."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 32)
                self.layer2 = nn.Linear(32, 16)
                self.layer3 = nn.Linear(16, 5)
            
            def forward(self, x):
                x = torch.relu(self.layer1(x))
                x = torch.relu(self.layer2(x))
                return self.layer3(x)
        
        return TestModel()
    
    def test_progressive_unfreezing_initialization(self, model_for_unfreezing):
        """Test progressive unfreezing initialization."""
        from ign_lidar.features.transfer_learning import ProgressiveUnfreezing
        
        unfreezer = ProgressiveUnfreezing(model_for_unfreezing, num_stages=3)
        
        # Check all params frozen initially
        for param in model_for_unfreezing.parameters():
            assert param.requires_grad is False
    
    def test_unfreeze_stage(self, model_for_unfreezing):
        """Test unfreezing specific stage."""
        from ign_lidar.features.transfer_learning import ProgressiveUnfreezing
        
        unfreezer = ProgressiveUnfreezing(model_for_unfreezing, num_stages=2)
        
        # Unfreeze first stage
        unfreezer.unfreeze_stage(0)
        
        # Check some params are unfrozen
        unfrozen_count = sum(
            1 for param in model_for_unfreezing.parameters()
            if param.requires_grad
        )
        assert unfrozen_count > 0


# ============================================================================
# PHASE 7.2: MODEL ENSEMBLE TESTS
# ============================================================================

@pytest.mark.skipif(not sklearn_available, reason="scikit-learn not available")
class TestVotingEnsemble:
    """Test VotingEnsemble component."""
    
    @pytest.fixture
    def ensemble_models(self):
        """Create ensemble of models."""
        models = [
            RandomForestClassifier(n_estimators=5, random_state=i)
            for i in range(3)
        ]
        return models
    
    @pytest.fixture
    def classification_data(self):
        """Create classification data."""
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 3, 50)
        return X, y
    
    def test_voting_ensemble_initialization(self, ensemble_models):
        """Test voting ensemble initialization."""
        from ign_lidar.features.model_ensemble import VotingEnsemble
        
        ensemble = VotingEnsemble(ensemble_models, voting="hard")
        assert ensemble.voting == "hard"
        assert len(ensemble.models) == 3
    
    def test_voting_ensemble_with_weights(self, ensemble_models):
        """Test voting ensemble with custom weights."""
        from ign_lidar.features.model_ensemble import VotingEnsemble
        
        weights = [0.5, 0.3, 0.2]
        ensemble = VotingEnsemble(
            ensemble_models, voting="hard", weights=weights
        )
        assert ensemble.weights == weights
    
    def test_hard_voting_prediction(self, ensemble_models, classification_data):
        """Test hard voting prediction."""
        from ign_lidar.features.model_ensemble import VotingEnsemble
        
        X, y = classification_data
        
        # Train models
        for model in ensemble_models:
            model.fit(X[:40], y[:40])
        
        ensemble = VotingEnsemble(ensemble_models, voting="hard")
        ensemble.fit(X[:40], y[:40])
        
        predictions = ensemble.predict(X[40:])
        assert len(predictions) == len(X[40:])


@pytest.mark.skipif(not sklearn_available, reason="scikit-learn not available")
class TestStackingEnsemble:
    """Test StackingEnsemble component."""
    
    @pytest.fixture
    def stacking_setup(self):
        """Create stacking ensemble setup."""
        base_models = [
            RandomForestClassifier(n_estimators=5, random_state=i)
            for i in range(2)
        ]
        meta_learner = LogisticRegression()
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 3, 50)
        return base_models, meta_learner, X, y
    
    def test_stacking_ensemble_fit_predict(self, stacking_setup):
        """Test stacking ensemble fit and predict."""
        from ign_lidar.features.model_ensemble import StackingEnsemble
        
        base_models, meta_learner, X, y = stacking_setup
        
        ensemble = StackingEnsemble(base_models, meta_learner)
        ensemble.fit(X[:40], y[:40])
        
        predictions = ensemble.predict(X[40:])
        assert len(predictions) == len(X[40:])


@pytest.mark.skipif(not sklearn_available, reason="scikit-learn not available")
class TestBootstrappingEnsemble:
    """Test BootstrappingEnsemble component."""
    
    def test_bagging_ensemble_creation(self):
        """Test bagging ensemble creation."""
        from ign_lidar.features.model_ensemble import BootstrappingEnsemble
        
        ensemble = BootstrappingEnsemble(
            RandomForestClassifier,
            n_estimators=5,
            random_state=42
        )
        assert ensemble.n_estimators == 5
    
    def test_bagging_ensemble_fit_predict(self):
        """Test bagging ensemble fit and predict."""
        from ign_lidar.features.model_ensemble import BootstrappingEnsemble
        
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 3, 50)
        
        ensemble = BootstrappingEnsemble(
            RandomForestClassifier,
            n_estimators=3,
            random_state=42
        )
        ensemble.fit(X[:40], y[:40])
        
        predictions = ensemble.predict(X[40:])
        assert len(predictions) == len(X[40:])


# ============================================================================
# PHASE 7.3: ACTIVE LEARNING TESTS
# ============================================================================

class TestActiveLearningConfig:
    """Test ActiveLearningConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        from ign_lidar.features.active_learning import ActiveLearningConfig
        
        config = ActiveLearningConfig()
        assert config.strategy == "uncertainty"
        assert config.query_size == 10
    
    def test_custom_config(self):
        """Test custom configuration."""
        from ign_lidar.features.active_learning import ActiveLearningConfig
        
        config = ActiveLearningConfig(
            strategy="diversity",
            query_size=20
        )
        assert config.strategy == "diversity"
        assert config.query_size == 20


@pytest.mark.skipif(not sklearn_available, reason="scikit-learn not available")
class TestUncertaintySampling:
    """Test UncertaintySampling strategy."""
    
    @pytest.fixture
    def sampling_setup(self):
        """Create sampling setup."""
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X_train = np.random.randn(50, 10)
        y_train = np.random.randint(0, 3, 50)
        model.fit(X_train, y_train)
        
        X_unlabeled = np.random.randn(100, 10)
        return model, X_unlabeled
    
    def test_uncertainty_sampling_entropy(self, sampling_setup):
        """Test entropy-based uncertainty sampling."""
        from ign_lidar.features.active_learning import UncertaintySampling
        
        model, X_unlabeled = sampling_setup
        
        sampler = UncertaintySampling(uncertainty_type="entropy")
        selected = sampler.query(X_unlabeled, model, n_instances=10)
        
        assert len(selected) == 10
        assert np.all(selected < len(X_unlabeled))
    
    def test_uncertainty_sampling_margin(self, sampling_setup):
        """Test margin-based uncertainty sampling."""
        from ign_lidar.features.active_learning import UncertaintySampling
        
        model, X_unlabeled = sampling_setup
        
        sampler = UncertaintySampling(uncertainty_type="margin")
        selected = sampler.query(X_unlabeled, model, n_instances=10)
        
        assert len(selected) == 10


class TestDiversitySampling:
    """Test DiversitySampling strategy."""
    
    def test_diversity_sampling_basic(self):
        """Test basic diversity sampling."""
        from ign_lidar.features.active_learning import DiversitySampling
        
        X_unlabeled = np.random.randn(100, 10)
        X_labeled = np.random.randn(20, 10)
        
        sampler = DiversitySampling()
        selected = sampler.query(X_unlabeled, X_labeled, n_instances=10)
        
        assert len(selected) == 10
        assert np.all(selected < len(X_unlabeled))
    
    def test_clustering_based_diversity(self):
        """Test clustering-based diversity sampling."""
        from ign_lidar.features.active_learning import DiversitySampling
        
        X_unlabeled = np.random.randn(100, 10)
        
        sampler = DiversitySampling(n_clusters=5)
        selected = sampler._clustering_based_diversity(X_unlabeled, n_instances=10)
        
        assert len(selected) <= 10


@pytest.mark.skipif(not sklearn_available, reason="scikit-learn not available")
class TestQueryByCommittee:
    """Test QueryByCommittee strategy."""
    
    def test_qbc_basic(self):
        """Test basic query by committee."""
        from ign_lidar.features.active_learning import QueryByCommittee
        
        # Create mock ensemble
        ensemble = MagicMock()
        model1 = RandomForestClassifier(n_estimators=3, random_state=42)
        model2 = RandomForestClassifier(n_estimators=3, random_state=43)
        
        X_train = np.random.randn(50, 10)
        y_train = np.random.randint(0, 3, 50)
        
        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)
        
        ensemble.models = [model1, model2]
        
        X_unlabeled = np.random.randn(100, 10)
        
        sampler = QueryByCommittee()
        selected = sampler.query(X_unlabeled, ensemble, n_instances=10)
        
        assert len(selected) == 10


@pytest.mark.skipif(not sklearn_available, reason="scikit-learn not available")
class TestActiveLearner:
    """Test ActiveLearner orchestrator."""
    
    @pytest.fixture
    def active_learning_setup(self):
        """Create active learning setup."""
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.randn(200, 10)
        y = np.random.randint(0, 3, 200)
        return model, X, y
    
    def test_active_learner_initialization(self, active_learning_setup):
        """Test active learner initialization."""
        from ign_lidar.features.active_learning import ActiveLearner
        
        model, X, y = active_learning_setup
        
        learner = ActiveLearner(model)
        assert learner.model == model
        assert len(learner.selected_history) == 0
    
    def test_active_learner_initialize(self, active_learning_setup):
        """Test initializing labeled pool."""
        from ign_lidar.features.active_learning import ActiveLearner
        
        model, X, y = active_learning_setup
        
        learner = ActiveLearner(model)
        initial = learner.initialize(X, y, n_initial=10)
        
        assert len(initial) == 10
        assert len(learner.labeled_indices) == 10
        assert len(learner.unlabeled_indices) == 190
    
    def test_active_learner_query(self, active_learning_setup):
        """Test querying new samples."""
        from ign_lidar.features.active_learning import ActiveLearner
        
        model, X, y = active_learning_setup
        
        learner = ActiveLearner(model)
        learner.initialize(X, y, n_initial=10)
        
        # Train initial model
        X_labeled, y_labeled = learner.get_labeled_pool(X, y)
        model.fit(X_labeled, y_labeled)
        
        # Query new samples
        selected = learner.query(X)
        
        assert len(selected) == 10
        assert len(learner.labeled_indices) == 20


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.integration
@pytest.mark.skipif(not torch_available or not sklearn_available,
                    reason="PyTorch or scikit-learn not available")
class TestPhase7Integration:
    """Integration tests for Phase 7 components."""
    
    def test_transfer_learning_workflow(self):
        """Test complete transfer learning workflow."""
        from ign_lidar.features.transfer_learning import (
            TransferConfig, TransferLearningPipeline
        )
        
        # Create simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 32)
                self.fc2 = nn.Linear(32, 5)
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return self.fc2(x)
        
        model = SimpleModel()
        config = TransferConfig(num_epochs=5, device="cpu")
        pipeline = TransferLearningPipeline(model, config)
        
        # Create data
        X_train = np.random.randn(50, 10).astype(np.float32)
        y_train = np.random.randint(0, 5, 50)
        
        # Train
        history = pipeline.train(X_train, y_train)
        
        assert "train_loss" in history
        assert len(history["train_loss"]) > 0
    
    def test_ensemble_active_learning_workflow(self):
        """Test ensemble with active learning."""
        from ign_lidar.features.model_ensemble import VotingEnsemble
        from ign_lidar.features.active_learning import ActiveLearner
        
        # Create ensemble
        models = [
            RandomForestClassifier(n_estimators=3, random_state=i)
            for i in range(2)
        ]
        ensemble = VotingEnsemble(models)
        
        # Create active learner with ensemble model
        learner = ActiveLearner(ensemble)
        
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)
        
        learner.initialize(X, y, n_initial=10)
        
        # Train ensemble
        X_labeled, y_labeled = learner.get_labeled_pool(X, y)
        for model in models:
            model.fit(X_labeled, y_labeled)
        ensemble.fit(X_labeled, y_labeled)
        
        # Query
        selected = learner.query(X)
        assert len(selected) == 10


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

@pytest.mark.performance
@pytest.mark.skipif(not sklearn_available, reason="scikit-learn not available")
class TestPhase7Performance:
    """Performance tests for Phase 7 components."""
    
    def test_uncertainty_sampling_performance(self, benchmark):
        """Benchmark uncertainty sampling."""
        from ign_lidar.features.active_learning import UncertaintySampling
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = np.random.randn(100, 20)
        y_train = np.random.randint(0, 5, 100)
        model.fit(X_train, y_train)
        
        X_unlabeled = np.random.randn(1000, 20)
        sampler = UncertaintySampling()
        
        def run_sampling():
            return sampler.query(X_unlabeled, model, n_instances=50)
        
        result = benchmark(run_sampling)
        assert len(result) == 50
    
    def test_ensemble_prediction_performance(self, benchmark):
        """Benchmark ensemble prediction."""
        from ign_lidar.features.model_ensemble import VotingEnsemble
        
        models = [
            RandomForestClassifier(n_estimators=5, random_state=i)
            for i in range(5)
        ]
        
        X_train = np.random.randn(100, 20)
        y_train = np.random.randint(0, 3, 100)
        
        for model in models:
            model.fit(X_train, y_train)
        
        ensemble = VotingEnsemble(models)
        X_test = np.random.randn(100, 20)
        
        def run_prediction():
            return ensemble.predict(X_test)
        
        result = benchmark(run_prediction)
        assert len(result) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
