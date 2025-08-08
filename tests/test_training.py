"""
Comprehensive tests for training algorithms.
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.liquid_neural_network import LiquidNeuralNetwork
from algorithms.training import LiquidNetworkTrainer
from algorithms.optimization import AdaptiveOptimizer, ContinuousTimeOptimizer
from algorithms.continuous_learning import ContinuousLearner


class TestLiquidNetworkTrainer:
    """Test suite for LiquidNetworkTrainer."""
    
    @pytest.fixture
    def simple_model_and_data(self):
        """Create simple model and synthetic data for testing."""
        key = random.PRNGKey(42)
        
        model = LiquidNeuralNetwork(
            input_size=1,
            hidden_size=8,
            output_size=1,
            key=key
        )
        
        # Generate simple sine wave data
        t = jnp.linspace(0, 2*jnp.pi, 50)
        inputs = jnp.sin(t).reshape(-1, 1)
        targets = jnp.sin(t + 0.1).reshape(-1, 1)  # Phase shift
        
        return model, inputs, targets
    
    def test_trainer_initialization(self, simple_model_and_data):
        """Test trainer initialization with different configurations."""
        model, inputs, targets = simple_model_and_data
        
        # Test with different optimizers
        for optimizer in ['adam', 'adamw', 'sgd', 'rmsprop']:
            trainer = LiquidNetworkTrainer(
                model=model,
                learning_rate=1e-3,
                optimizer_name=optimizer
            )
            assert trainer.optimizer is not None
            assert trainer.opt_state is not None
    
    def test_loss_functions(self, simple_model_and_data):
        """Test different loss functions."""
        model, inputs, targets = simple_model_and_data
        
        loss_functions = ['mse', 'mae', 'huber', 'temporal_consistency']
        
        for loss_fn in loss_functions:
            trainer = LiquidNetworkTrainer(
                model=model,
                loss_fn=loss_fn
            )
            
            # Test single training step
            metrics = trainer.train_step(inputs, targets)
            
            assert 'loss' in metrics
            assert jnp.isfinite(metrics['loss'])
            assert metrics['loss'] >= 0
    
    def test_training_step(self, simple_model_and_data):
        """Test single training step."""
        model, inputs, targets = simple_model_and_data
        
        trainer = LiquidNetworkTrainer(model=model)
        
        # Get initial loss
        initial_metrics = trainer.validate(inputs, targets)
        initial_loss = initial_metrics['loss']
        
        # Training step
        train_metrics = trainer.train_step(inputs, targets)
        
        # Check metrics structure
        required_metrics = ['loss', 'gradient_norm']
        for metric in required_metrics:
            assert metric in train_metrics
            assert jnp.isfinite(train_metrics[metric])
        
        # Check loss is reasonable
        assert train_metrics['loss'] >= 0
        assert train_metrics['gradient_norm'] >= 0
    
    def test_validation(self, simple_model_and_data):
        """Test validation functionality."""
        model, inputs, targets = simple_model_and_data
        
        trainer = LiquidNetworkTrainer(model=model)
        
        val_metrics = trainer.validate(inputs, targets)
        
        assert 'loss' in val_metrics
        assert 'mse' in val_metrics
        assert 'mae' in val_metrics
        
        # All metrics should be finite and non-negative
        for metric_name, metric_value in val_metrics.items():
            assert jnp.isfinite(metric_value)
            if metric_name != 'l2_reg':  # L2 reg can be very small
                assert metric_value >= 0
    
    def test_fit_method(self, simple_model_and_data):
        """Test complete training with fit method."""
        model, inputs, targets = simple_model_and_data
        
        trainer = LiquidNetworkTrainer(
            model=model,
            learning_rate=1e-2  # Higher LR for faster convergence in test
        )
        
        # Split data for validation
        split_idx = len(inputs) // 2
        train_inputs, val_inputs = inputs[:split_idx], inputs[split_idx:]
        train_targets, val_targets = targets[:split_idx], targets[split_idx:]
        
        # Train for a few epochs
        history = trainer.fit(
            train_data=(train_inputs, train_targets),
            val_data=(val_inputs, val_targets),
            epochs=10,
            verbose=False
        )
        
        # Check history structure
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert 'gradient_norm' in history
        
        assert len(history['train_loss']) == 10
        assert len(history['val_loss']) == 10
        
        # Loss should generally decrease (with some tolerance for noise)
        initial_loss = history['train_loss'][0]
        final_loss = history['train_loss'][-1]
        
        # Allow for some training instability but expect general improvement
        assert final_loss < initial_loss * 2  # Very lenient check
    
    def test_gradient_clipping(self, simple_model_and_data):
        """Test gradient clipping functionality."""
        model, inputs, targets = simple_model_and_data
        
        # Trainer with gradient clipping
        trainer_clipped = LiquidNetworkTrainer(
            model=model,
            gradient_clip=1.0
        )
        
        # Trainer without gradient clipping
        trainer_unclipped = LiquidNetworkTrainer(
            model=model,
            gradient_clip=None
        )
        
        # Both should work without errors
        metrics_clipped = trainer_clipped.train_step(inputs, targets)
        metrics_unclipped = trainer_unclipped.train_step(inputs, targets)
        
        assert jnp.isfinite(metrics_clipped['loss'])
        assert jnp.isfinite(metrics_unclipped['loss'])
    
    def test_checkpoint_functionality(self, simple_model_and_data, tmp_path):
        """Test saving and loading checkpoints."""
        model, inputs, targets = simple_model_and_data
        
        trainer = LiquidNetworkTrainer(model=model)
        
        # Train a bit
        trainer.fit(
            train_data=(inputs, targets),
            epochs=5,
            verbose=False
        )
        
        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.npy"
        trainer.save_checkpoint(str(checkpoint_path))
        
        assert checkpoint_path.exists()
        
        # Create new trainer and load checkpoint
        new_model = LiquidNeuralNetwork(
            input_size=1, hidden_size=8, output_size=1,
            key=random.PRNGKey(999)
        )
        new_trainer = LiquidNetworkTrainer(model=new_model)
        
        new_trainer.load_checkpoint(str(checkpoint_path))
        
        # Check parameters match
        for key in trainer.model.params:
            assert jnp.allclose(
                trainer.model.params[key],
                new_trainer.model.params[key]
            )


class TestAdaptiveOptimizer:
    """Test suite for AdaptiveOptimizer."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = AdaptiveOptimizer(
            learning_rate=1e-3,
            adaptation_rate=1e-4
        )
        
        assert optimizer.learning_rate == 1e-3
        assert optimizer.adaptation_rate == 1e-4
    
    def test_state_initialization(self):
        """Test optimizer state initialization."""
        key = random.PRNGKey(42)
        model = LiquidNeuralNetwork(
            input_size=2, hidden_size=4, output_size=1, key=key
        )
        
        optimizer = AdaptiveOptimizer()
        state = optimizer.init_state(model.params)
        
        # Check state structure
        for param_name in model.params:
            assert param_name in state
            assert 'momentum' in state[param_name]
            assert 'second_moment' in state[param_name]
            assert 'step_count' in state[param_name]
    
    def test_time_constant_update(self):
        """Test specialized time constant updates."""
        optimizer = AdaptiveOptimizer()
        
        # Mock time constant gradients and current values
        tau_grads = jnp.array([0.1, -0.05, 0.2])
        tau_current = jnp.array([1.0, 0.5, 2.0])
        tau_state = {
            'momentum': jnp.zeros(3),
            'second_moment': jnp.zeros(3)
        }
        
        tau_new, new_state = optimizer.adaptive_time_constant_update(
            tau_grads, tau_current, tau_state, step=1
        )
        
        # Check time constants remain positive
        assert jnp.all(tau_new > 0)
        
        # Check state was updated
        assert not jnp.allclose(new_state['momentum'], tau_state['momentum'])
    
    def test_gradient_metrics(self):
        """Test gradient metrics computation."""
        key = random.PRNGKey(123)
        
        # Mock gradients
        grads = {
            'W_in': random.normal(key, (4, 2)),
            'W_out': random.normal(key, (1, 4)),
            'tau': random.normal(key, (4,))
        }
        
        optimizer = AdaptiveOptimizer()
        metrics = optimizer.compute_gradient_metrics(grads)
        
        # Check required metrics
        assert 'total_grad_norm' in metrics
        
        for param_name in grads:
            assert f'{param_name}_grad_norm' in metrics
            assert f'{param_name}_grad_mean' in metrics
            assert f'{param_name}_grad_std' in metrics
        
        # All metrics should be finite
        for metric_value in metrics.values():
            assert jnp.isfinite(metric_value)


class TestContinuousTimeOptimizer:
    """Test suite for ContinuousTimeOptimizer."""
    
    def test_initialization(self):
        """Test CT optimizer initialization."""
        optimizer = ContinuousTimeOptimizer(
            base_lr=1e-3,
            dynamics_lr_scale=0.1
        )
        
        assert optimizer.base_lr == 1e-3
        assert optimizer.dynamics_lr_scale == 0.1
    
    def test_optimizer_states(self):
        """Test optimizer state initialization."""
        key = random.PRNGKey(456)
        model = LiquidNeuralNetwork(
            input_size=2, hidden_size=4, output_size=1, key=key
        )
        
        optimizer = ContinuousTimeOptimizer()
        states = optimizer.init_optimizer_states(model.params)
        
        assert 'param_state' in states
        assert 'dynamics_state' in states
    
    def test_regularization(self):
        """Test regularization addition to gradients."""
        key = random.PRNGKey(789)
        
        params = {
            'W_in': random.normal(key, (4, 2)),
            'W_hh': random.normal(key, (4, 4)),
            'tau': jnp.array([1.0, 0.5, 2.0, 1.5])
        }
        
        grads = {
            'W_in': random.normal(key, (4, 2)),
            'W_hh': random.normal(key, (4, 4)),
            'tau': random.normal(key, (4,))
        }
        
        optimizer = ContinuousTimeOptimizer(regularization_strength=0.01)
        reg_grads = optimizer.add_regularization_to_grads(params, grads)
        
        # Regularized gradients should be different
        assert not jnp.allclose(reg_grads['W_in'], grads['W_in'])
        assert not jnp.allclose(reg_grads['tau'], grads['tau'])


class TestContinuousLearner:
    """Test suite for ContinuousLearner."""
    
    @pytest.fixture
    def basic_continuous_learner(self):
        """Create basic continuous learner setup."""
        key = random.PRNGKey(42)
        model = LiquidNeuralNetwork(
            input_size=2, hidden_size=6, output_size=1, key=key
        )
        
        learner = ContinuousLearner(
            model=model,
            memory_size=100,
            adaptation_rate=0.01
        )
        
        return learner, model
    
    def test_initialization(self, basic_continuous_learner):
        """Test continuous learner initialization."""
        learner, model = basic_continuous_learner
        
        assert learner.model is model
        assert learner.memory_size == 100
        assert len(learner.memory_buffer) == 0
        
        # Check importance weights initialized
        for param_name in model.params:
            assert param_name in learner.importance_weights
            assert param_name in learner.consolidated_params
            assert param_name in learner.plasticity
    
    def test_experience_addition(self, basic_continuous_learner):
        """Test adding experiences to memory buffer."""
        learner, model = basic_continuous_learner
        
        # Add some experiences
        for i in range(5):
            inputs = random.normal(random.PRNGKey(i), (10, 2))
            targets = random.normal(random.PRNGKey(i+100), (10, 1))
            learner.add_experience(inputs, targets, {'episode': i})
        
        assert len(learner.memory_buffer) == 5
        
        # Check experience structure
        exp = learner.memory_buffer[0]
        assert 'inputs' in exp
        assert 'targets' in exp
        assert 'timestamp' in exp
        assert 'metadata' in exp
    
    def test_experience_sampling(self, basic_continuous_learner):
        """Test sampling from experience buffer."""
        learner, model = basic_continuous_learner
        
        # Add experiences
        for i in range(20):
            inputs = random.normal(random.PRNGKey(i), (5, 2))
            targets = random.normal(random.PRNGKey(i+1000), (5, 1))
            learner.add_experience(inputs, targets)
        
        # Sample experiences
        sampled = learner.sample_experiences(10)
        
        assert len(sampled) == 10
        assert all('inputs' in exp and 'targets' in exp for exp in sampled)
    
    def test_fisher_information_computation(self, basic_continuous_learner):
        """Test Fisher Information Matrix computation."""
        learner, model = basic_continuous_learner
        
        # Add some experiences
        experiences = []
        for i in range(5):
            inputs = random.normal(random.PRNGKey(i), (8, 2))
            targets = random.normal(random.PRNGKey(i+500), (8, 1))
            exp = {
                'inputs': inputs,
                'targets': targets,
                'timestamp': i,
                'metadata': {}
            }
            experiences.append(exp)
        
        # Compute Fisher information
        fisher_info = learner.compute_fisher_information(experiences, n_samples=3)
        
        # Check structure
        for param_name in model.params:
            assert param_name in fisher_info
            assert fisher_info[param_name].shape == model.params[param_name].shape
            assert jnp.all(jnp.isfinite(fisher_info[param_name]))
            assert jnp.all(fisher_info[param_name] >= 0)  # Fisher info is non-negative
    
    def test_continual_learning_step(self, basic_continuous_learner):
        """Test single continual learning step."""
        learner, model = basic_continuous_learner
        
        # Generate new data
        new_inputs = random.normal(random.PRNGKey(999), (10, 2))
        new_targets = random.normal(random.PRNGKey(888), (10, 1))
        
        # Take learning step
        metrics = learner.continual_learning_step(
            new_inputs, new_targets,
            replay_batch_size=5,
            update_consolidation=False
        )
        
        # Check metrics structure
        required_metrics = [
            'prediction_loss', 'consolidation_loss', 'plasticity_loss',
            'total_loss', 'memory_size', 'avg_plasticity'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert jnp.isfinite(metrics[metric])
        
        # Check memory was updated
        assert len(learner.memory_buffer) == 1
    
    def test_knowledge_consolidation(self, basic_continuous_learner):
        """Test knowledge consolidation."""
        learner, model = basic_continuous_learner
        
        # Store original parameters
        original_params = {k: v.copy() for k, v in model.params.items()}
        
        # Modify model parameters slightly
        new_params = model.params.copy()
        new_params['W_in'] = new_params['W_in'] + 0.1
        model.update_params(new_params)
        
        # Consolidate
        learner.consolidate_knowledge()
        
        # Check consolidated parameters were updated
        for param_name in model.params:
            assert jnp.allclose(
                learner.consolidated_params[param_name],
                model.params[param_name]
            )
        
        # Check plasticity was reset
        for param_name in learner.plasticity:
            assert jnp.allclose(
                learner.plasticity[param_name],
                jnp.ones_like(learner.plasticity[param_name])
            )
    
    def test_learning_statistics(self, basic_continuous_learner):
        """Test learning statistics collection."""
        learner, model = basic_continuous_learner
        
        # Take several learning steps
        for i in range(5):
            inputs = random.normal(random.PRNGKey(i), (8, 2))
            targets = random.normal(random.PRNGKey(i+2000), (8, 1))
            learner.continual_learning_step(inputs, targets)
        
        # Get statistics
        stats = learner.get_learning_statistics()
        
        assert 'total_steps' in stats
        assert 'memory_utilization' in stats
        assert 'importance_weight_stats' in stats
        
        assert stats['total_steps'] == 5
        assert 0 <= stats['memory_utilization'] <= 1
        
        # Check importance weight stats structure
        for param_name in model.params:
            assert param_name in stats['importance_weight_stats']
            param_stats = stats['importance_weight_stats'][param_name]
            assert 'mean' in param_stats
            assert 'std' in param_stats
            assert 'max' in param_stats
            assert 'min' in param_stats


def run_integration_tests():
    """Run integration tests combining multiple components."""
    print("Running integration tests...")
    
    key = random.PRNGKey(42)
    
    # Create model
    model = LiquidNeuralNetwork(
        input_size=2, hidden_size=10, output_size=1, key=key
    )
    
    # Create trainer with adaptive optimizer
    trainer = LiquidNetworkTrainer(
        model=model,
        learning_rate=1e-3,
        optimizer_name='adam',
        loss_fn='mse'
    )
    
    # Generate training data
    t = jnp.linspace(0, 4*jnp.pi, 100)
    inputs = jnp.column_stack([jnp.sin(t), jnp.cos(t)])
    targets = jnp.sin(t + 0.5).reshape(-1, 1)
    
    # Train model
    history = trainer.fit(
        train_data=(inputs, targets),
        epochs=20,
        verbose=False
    )
    
    # Check training worked
    assert len(history['train_loss']) == 20
    assert jnp.isfinite(history['train_loss'][-1])
    
    # Test with continual learner
    continuous_learner = ContinuousLearner(model=model, memory_size=50)
    
    # Simulate continual learning
    for i in range(10):
        batch_inputs = inputs[i*10:(i+1)*10]
        batch_targets = targets[i*10:(i+1)*10]
        
        metrics = continuous_learner.continual_learning_step(
            batch_inputs, batch_targets
        )
        
        assert jnp.isfinite(metrics['total_loss'])
    
    print("Integration tests passed!")


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
    
    # Run integration tests
    run_integration_tests()