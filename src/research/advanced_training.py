"""
Advanced training algorithms for liquid neural networks.

This module implements state-of-the-art training techniques including
meta-learning, continual learning, and adaptive optimization strategies
specifically designed for liquid neural networks.
"""

try:
    import jax
    import jax.numpy as jnp
    from jax import random, grad, jit, vmap
    import optax
    import equinox as eqx
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    import numpy as jnp

from typing import Dict, Any, Tuple, Optional, List, Callable, Union
import time
import warnings
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Import model components
from ..models.liquid_neural_network import LiquidNeuralNetwork
from ..models.continuous_time_rnn import ContinuousTimeRNN


@dataclass
class TrainingConfig:
    """Configuration for advanced training procedures."""
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    gradient_clip_norm: Optional[float] = 1.0
    weight_decay: float = 1e-4
    
    # Meta-learning parameters
    meta_learning_rate: float = 1e-3
    inner_learning_rate: float = 1e-2
    num_inner_steps: int = 5
    
    # Continual learning parameters
    memory_strength: float = 1000.0
    memory_size: int = 1000
    rehearsal_batch_size: int = 16
    
    # Adaptive optimization
    warmup_steps: int = 1000
    decay_schedule: str = 'cosine'
    
    # Regularization
    liquid_regularization: float = 0.01
    temporal_consistency_weight: float = 0.1
    sparsity_weight: float = 0.01


class LiquidNetworkTrainer(ABC):
    """Abstract base class for liquid network trainers."""
    
    def __init__(
        self,
        model: Any,
        config: TrainingConfig,
        random_seed: int = 42
    ):
        self.model = model
        self.config = config
        self.random_seed = random_seed
        self.key = random.PRNGKey(random_seed)
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.training_history = []
        self.best_validation_loss = float('inf')
        
    @abstractmethod
    def compute_loss(
        self, 
        model: Any, 
        batch: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> Tuple[float, Dict[str, float]]:
        """Compute training loss and auxiliary metrics."""
        pass
    
    @abstractmethod
    def train_step(
        self, 
        model: Any, 
        optimizer_state: Any, 
        batch: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> Tuple[Any, Any, Dict[str, float]]:
        """Perform one training step."""
        pass


class StandardLiquidTrainer(LiquidNetworkTrainer):
    """Standard training procedure for liquid neural networks."""
    
    def __init__(self, model: Any, config: TrainingConfig, **kwargs):
        super().__init__(model, config, **kwargs)
        
        # Initialize optimizer
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.learning_rate,
            warmup_steps=config.warmup_steps,
            decay_steps=config.num_epochs * 100  # Approximate
        )
        
        optimizer_chain = [
            optax.clip_by_global_norm(config.gradient_clip_norm) if config.gradient_clip_norm else optax.identity(),
            optax.adam(schedule),
            optax.add_decayed_weights(config.weight_decay)
        ]
        
        self.optimizer = optax.chain(*optimizer_chain)
        self.optimizer_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))
    
    def liquid_regularization_loss(self, model: Any) -> float:
        """Compute liquid-specific regularization terms."""
        reg_loss = 0.0
        
        # Time constant regularization (encourage diversity)
        if hasattr(model, 'tau'):
            tau_diversity = -jnp.std(model.tau)  # Negative to encourage diversity
            reg_loss += self.config.liquid_regularization * tau_diversity
        
        # Sparsity regularization on recurrent weights
        if hasattr(model, 'W_rec'):
            sparsity_loss = jnp.mean(jnp.abs(model.W_rec))
            reg_loss += self.config.sparsity_weight * sparsity_loss
        
        return reg_loss
    
    def temporal_consistency_loss(
        self, 
        model: Any, 
        inputs: jnp.ndarray, 
        targets: jnp.ndarray
    ) -> float:
        """Encourage temporal consistency in liquid dynamics."""
        batch_size, seq_len, input_dim = inputs.shape
        
        # Initialize hidden state
        if hasattr(model, 'init_hidden_state'):
            hidden_state = model.init_hidden_state(batch_size)
        else:
            hidden_state = jnp.zeros((batch_size, getattr(model, 'hidden_size', 32)))
        
        consistency_loss = 0.0
        prev_hidden = hidden_state
        
        for t in range(min(seq_len, 10)):  # Limit to avoid memory issues
            if hasattr(model, '__call__'):
                output, hidden_state = model(inputs[:, t], hidden_state)
            else:
                output = model.forward(inputs[:, t], hidden_state)
            
            # Penalize large jumps in hidden state
            if t > 0:
                state_diff = jnp.mean((hidden_state - prev_hidden) ** 2)
                consistency_loss += state_diff
            
            prev_hidden = hidden_state
        
        return consistency_loss / seq_len
    
    def compute_loss(
        self, 
        model: Any, 
        batch: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> Tuple[float, Dict[str, float]]:
        """Compute total training loss."""
        inputs, targets = batch
        batch_size, seq_len, input_dim = inputs.shape
        
        # Initialize hidden state
        if hasattr(model, 'init_hidden_state'):
            hidden_state = model.init_hidden_state(batch_size)
        else:
            hidden_state = jnp.zeros((batch_size, getattr(model, 'hidden_size', 32)))
        
        predictions = []
        total_loss = 0.0
        
        # Forward pass through sequence
        for t in range(seq_len):
            if hasattr(model, '__call__'):
                output, hidden_state = model(inputs[:, t], hidden_state)
            else:
                output = model.forward(inputs[:, t], hidden_state)
            
            predictions.append(output)
            
            # Step-wise loss
            step_loss = jnp.mean((output - targets[:, t]) ** 2)
            total_loss += step_loss
        
        # Average over sequence
        prediction_loss = total_loss / seq_len
        
        # Regularization terms
        liquid_reg = self.liquid_regularization_loss(model)
        temporal_consistency = self.temporal_consistency_loss(model, inputs, targets)
        
        # Total loss
        total_loss = (
            prediction_loss + 
            liquid_reg +
            self.config.temporal_consistency_weight * temporal_consistency
        )
        
        # Auxiliary metrics
        predictions = jnp.array(predictions).transpose(1, 0, 2)
        mae = jnp.mean(jnp.abs(predictions - targets))
        
        metrics = {
            'prediction_loss': float(prediction_loss),
            'liquid_regularization': float(liquid_reg),
            'temporal_consistency': float(temporal_consistency),
            'mae': float(mae),
            'total_loss': float(total_loss)
        }
        
        return total_loss, metrics
    
    def train_step(
        self, 
        model: Any, 
        optimizer_state: Any, 
        batch: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> Tuple[Any, Any, Dict[str, float]]:
        """Perform one training step."""
        
        # Compute loss and gradients
        (loss, metrics), grads = jax.value_and_grad(
            lambda m: self.compute_loss(m, batch), has_aux=True
        )(model)
        
        # Apply optimizer
        updates, new_optimizer_state = self.optimizer.update(
            grads, optimizer_state, eqx.filter(model, eqx.is_array)
        )
        new_model = eqx.apply_updates(model, updates)
        
        # Update step counter
        self.step += 1
        
        # Add gradient norm to metrics
        grad_norm = optax.global_norm(grads)
        metrics['grad_norm'] = float(grad_norm)
        
        return new_model, new_optimizer_state, metrics


class MetaLearningTrainer(LiquidNetworkTrainer):
    """Meta-learning trainer using Model-Agnostic Meta-Learning (MAML)."""
    
    def __init__(self, model: Any, config: TrainingConfig, **kwargs):
        super().__init__(model, config, **kwargs)
        
        # Meta-optimizer
        self.meta_optimizer = optax.adam(config.meta_learning_rate)
        self.meta_optimizer_state = self.meta_optimizer.init(eqx.filter(self.model, eqx.is_array))
        
        # Inner optimizer for task adaptation
        self.inner_optimizer = optax.sgd(config.inner_learning_rate)
    
    def inner_loop_update(
        self, 
        model: Any, 
        support_batch: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> Any:
        """Perform inner loop adaptation for a specific task."""
        
        # Initialize inner optimizer state
        inner_state = self.inner_optimizer.init(eqx.filter(model, eqx.is_array))
        adapted_model = model
        
        # Inner loop updates
        for _ in range(self.config.num_inner_steps):
            # Compute loss and gradients on support set
            loss, grads = jax.value_and_grad(
                lambda m: self.compute_loss(m, support_batch)[0]
            )(adapted_model)
            
            # Apply inner update
            updates, inner_state = self.inner_optimizer.update(
                grads, inner_state, eqx.filter(adapted_model, eqx.is_array)
            )
            adapted_model = eqx.apply_updates(adapted_model, updates)
        
        return adapted_model
    
    def meta_loss(
        self, 
        model: Any, 
        task_batch: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    ) -> Tuple[float, Dict[str, float]]:
        """Compute meta-learning loss across multiple tasks."""
        
        support_batch = task_batch['support']
        query_batch = task_batch['query']
        
        # Inner loop adaptation
        adapted_model = self.inner_loop_update(model, support_batch)
        
        # Evaluate on query set
        query_loss, query_metrics = self.compute_loss(adapted_model, query_batch)
        
        return query_loss, query_metrics
    
    def compute_loss(
        self, 
        model: Any, 
        batch: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> Tuple[float, Dict[str, float]]:
        """Standard loss computation for meta-learning."""
        inputs, targets = batch
        batch_size, seq_len, input_dim = inputs.shape
        
        # Initialize hidden state
        if hasattr(model, 'init_hidden_state'):
            hidden_state = model.init_hidden_state(batch_size)
        else:
            hidden_state = jnp.zeros((batch_size, getattr(model, 'hidden_size', 32)))
        
        total_loss = 0.0
        
        # Forward pass
        for t in range(seq_len):
            if hasattr(model, '__call__'):
                output, hidden_state = model(inputs[:, t], hidden_state)
            else:
                output = model.forward(inputs[:, t], hidden_state)
            
            step_loss = jnp.mean((output - targets[:, t]) ** 2)
            total_loss += step_loss
        
        loss = total_loss / seq_len
        metrics = {'loss': float(loss)}
        
        return loss, metrics
    
    def train_step(
        self, 
        model: Any, 
        optimizer_state: Any, 
        task_batch: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    ) -> Tuple[Any, Any, Dict[str, float]]:
        """Meta-learning training step."""
        
        # Compute meta-gradients
        (meta_loss, metrics), meta_grads = jax.value_and_grad(
            lambda m: self.meta_loss(m, task_batch), has_aux=True
        )(model)
        
        # Meta-update
        updates, new_optimizer_state = self.meta_optimizer.update(
            meta_grads, optimizer_state, eqx.filter(model, eqx.is_array)
        )
        new_model = eqx.apply_updates(model, updates)
        
        self.step += 1
        metrics['meta_loss'] = float(meta_loss)
        
        return new_model, new_optimizer_state, metrics


class ContinualLearningTrainer(LiquidNetworkTrainer):
    """Continual learning trainer with elastic weight consolidation."""
    
    def __init__(self, model: Any, config: TrainingConfig, **kwargs):
        super().__init__(model, config, **kwargs)
        
        # Standard optimizer
        self.optimizer = optax.adam(config.learning_rate)
        self.optimizer_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))
        
        # Fisher information matrix and optimal parameters for EWC
        self.fisher_information = None
        self.optimal_params = None
        
        # Experience replay memory
        self.memory_inputs = []
        self.memory_targets = []
        self.memory_size = config.memory_size
        
    def estimate_fisher_information(
        self, 
        model: Any, 
        dataset: List[Tuple[jnp.ndarray, jnp.ndarray]]
    ) -> Dict[str, jnp.ndarray]:
        """Estimate Fisher Information Matrix for important parameters."""
        
        # Collect gradients from multiple samples
        all_grads = []
        
        for inputs, targets in dataset[:min(len(dataset), 100)]:  # Limit samples
            batch = (inputs.reshape(1, *inputs.shape), targets.reshape(1, *targets.shape))
            
            loss, grads = jax.value_and_grad(
                lambda m: self.compute_loss(m, batch)[0]
            )(model)
            
            # Flatten gradients
            flat_grads = jnp.concatenate([
                g.flatten() for g in jax.tree_util.tree_leaves(grads) 
                if g is not None
            ])
            all_grads.append(flat_grads)
        
        # Estimate Fisher Information as empirical variance of gradients
        stacked_grads = jnp.array(all_grads)
        fisher_info = jnp.var(stacked_grads, axis=0)
        
        # Reconstruct tree structure
        fisher_dict = {}
        start_idx = 0
        
        for key, param in jax.tree_util.tree_leaves_with_path(eqx.filter(model, eqx.is_array)):
            if param is not None:
                param_size = param.size
                fisher_dict[str(key)] = fisher_info[start_idx:start_idx + param_size].reshape(param.shape)
                start_idx += param_size
        
        return fisher_dict
    
    def ewc_loss(self, model: Any, previous_task_model: Any) -> float:
        """Compute Elastic Weight Consolidation regularization loss."""
        if self.fisher_information is None or previous_task_model is None:
            return 0.0
        
        ewc_loss = 0.0
        
        # Compare current parameters with previous task parameters
        for (key, current_param), (_, prev_param) in zip(
            jax.tree_util.tree_leaves_with_path(eqx.filter(model, eqx.is_array)),
            jax.tree_util.tree_leaves_with_path(eqx.filter(previous_task_model, eqx.is_array))
        ):
            if current_param is not None and prev_param is not None:
                key_str = str(key)
                if key_str in self.fisher_information:
                    fisher = self.fisher_information[key_str]
                    param_diff = current_param - prev_param
                    ewc_loss += 0.5 * jnp.sum(fisher * (param_diff ** 2))
        
        return self.config.memory_strength * ewc_loss
    
    def add_to_memory(self, inputs: jnp.ndarray, targets: jnp.ndarray) -> None:
        """Add examples to experience replay memory."""
        batch_size = inputs.shape[0]
        
        for i in range(batch_size):
            if len(self.memory_inputs) < self.memory_size:
                self.memory_inputs.append(inputs[i])
                self.memory_targets.append(targets[i])
            else:
                # Random replacement
                idx = random.randint(self.key, (), 0, len(self.memory_inputs))
                self.memory_inputs[idx] = inputs[i]
                self.memory_targets[idx] = targets[i]
    
    def sample_from_memory(self) -> Optional[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Sample batch from experience replay memory."""
        if len(self.memory_inputs) < self.config.rehearsal_batch_size:
            return None
        
        # Random sampling
        indices = random.choice(
            self.key, 
            len(self.memory_inputs), 
            (self.config.rehearsal_batch_size,), 
            replace=False
        )
        
        sampled_inputs = jnp.array([self.memory_inputs[i] for i in indices])
        sampled_targets = jnp.array([self.memory_targets[i] for i in indices])
        
        return sampled_inputs, sampled_targets
    
    def compute_loss(
        self, 
        model: Any, 
        batch: Tuple[jnp.ndarray, jnp.ndarray],
        previous_task_model: Optional[Any] = None
    ) -> Tuple[float, Dict[str, float]]:
        """Compute continual learning loss with EWC regularization."""
        
        # Standard task loss
        task_loss, metrics = super().compute_loss(model, batch)
        
        # EWC regularization loss
        ewc_reg = self.ewc_loss(model, previous_task_model)
        
        # Experience replay loss
        replay_loss = 0.0
        memory_batch = self.sample_from_memory()
        if memory_batch is not None:
            replay_task_loss, _ = super().compute_loss(model, memory_batch)
            replay_loss = replay_task_loss
        
        total_loss = task_loss + ewc_reg + replay_loss
        
        metrics.update({
            'task_loss': float(task_loss),
            'ewc_regularization': float(ewc_reg),
            'replay_loss': float(replay_loss),
            'total_loss': float(total_loss)
        })
        
        return total_loss, metrics
    
    def train_step(
        self, 
        model: Any, 
        optimizer_state: Any, 
        batch: Tuple[jnp.ndarray, jnp.ndarray],
        previous_task_model: Optional[Any] = None
    ) -> Tuple[Any, Any, Dict[str, float]]:
        """Continual learning training step."""
        
        # Add current batch to memory
        inputs, targets = batch
        self.add_to_memory(inputs, targets)
        
        # Compute gradients with EWC
        (loss, metrics), grads = jax.value_and_grad(
            lambda m: self.compute_loss(m, batch, previous_task_model), has_aux=True
        )(model)
        
        # Apply updates
        updates, new_optimizer_state = self.optimizer.update(
            grads, optimizer_state, eqx.filter(model, eqx.is_array)
        )
        new_model = eqx.apply_updates(model, updates)
        
        self.step += 1
        
        return new_model, new_optimizer_state, metrics
    
    def finish_task(self, model: Any, task_dataset: List[Tuple[jnp.ndarray, jnp.ndarray]]) -> None:
        """Called when finishing a task to compute Fisher Information."""
        
        # Estimate Fisher Information for current task
        self.fisher_information = self.estimate_fisher_information(model, task_dataset)
        
        # Store optimal parameters
        self.optimal_params = eqx.filter(model, eqx.is_array)


class AdaptiveOptimizationTrainer(LiquidNetworkTrainer):
    """Trainer with adaptive optimization strategies."""
    
    def __init__(self, model: Any, config: TrainingConfig, **kwargs):
        super().__init__(model, config, **kwargs)
        
        # Multiple optimizers with different strategies
        self.optimizers = {
            'adam': optax.adam(config.learning_rate),
            'adamw': optax.adamw(config.learning_rate, weight_decay=config.weight_decay),
            'sgd': optax.sgd(config.learning_rate, momentum=0.9),
            'rmsprop': optax.rmsprop(config.learning_rate)
        }
        
        self.optimizer_states = {
            name: opt.init(eqx.filter(self.model, eqx.is_array))
            for name, opt in self.optimizers.items()
        }
        
        # Adaptive optimizer selection
        self.current_optimizer = 'adam'
        self.optimizer_performance = {name: [] for name in self.optimizers.keys()}
        self.adaptation_window = 50
        
    def select_best_optimizer(self) -> str:
        """Select the best performing optimizer based on recent performance."""
        
        if self.step < self.adaptation_window:
            return self.current_optimizer
        
        # Compute average performance over recent window
        avg_performance = {}
        for name, performance_history in self.optimizer_performance.items():
            if len(performance_history) >= 10:
                recent_performance = performance_history[-10:]
                avg_performance[name] = jnp.mean(jnp.array(recent_performance))
        
        if avg_performance:
            best_optimizer = min(avg_performance.keys(), key=lambda k: avg_performance[k])
            return best_optimizer
        
        return self.current_optimizer
    
    def adaptive_learning_rate(self, loss_history: List[float]) -> float:
        """Compute adaptive learning rate based on loss history."""
        
        if len(loss_history) < 10:
            return self.config.learning_rate
        
        # Simple adaptive strategy: reduce LR if loss plateaus
        recent_losses = loss_history[-10:]
        loss_improvement = recent_losses[0] - recent_losses[-1]
        
        if loss_improvement < 0.001:  # Loss plateau
            return self.config.learning_rate * 0.9
        elif loss_improvement > 0.01:  # Good progress
            return min(self.config.learning_rate * 1.05, self.config.learning_rate * 2)
        
        return self.config.learning_rate
    
    def compute_loss(
        self, 
        model: Any, 
        batch: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> Tuple[float, Dict[str, float]]:
        """Standard loss computation with additional adaptive metrics."""
        
        inputs, targets = batch
        batch_size, seq_len, input_dim = inputs.shape
        
        # Initialize hidden state
        if hasattr(model, 'init_hidden_state'):
            hidden_state = model.init_hidden_state(batch_size)
        else:
            hidden_state = jnp.zeros((batch_size, getattr(model, 'hidden_size', 32)))
        
        predictions = []
        prediction_entropy = 0.0
        
        # Forward pass with entropy computation
        for t in range(seq_len):
            if hasattr(model, '__call__'):
                output, hidden_state = model(inputs[:, t], hidden_state)
            else:
                output = model.forward(inputs[:, t], hidden_state)
            
            predictions.append(output)
            
            # Compute prediction entropy (for adaptive strategies)
            output_probs = jax.nn.softmax(output, axis=-1)
            entropy = -jnp.sum(output_probs * jnp.log(output_probs + 1e-8))
            prediction_entropy += entropy
        
        predictions = jnp.array(predictions).transpose(1, 0, 2)
        
        # Loss computation
        mse_loss = jnp.mean((predictions - targets) ** 2)
        avg_entropy = prediction_entropy / seq_len
        
        metrics = {
            'mse_loss': float(mse_loss),
            'prediction_entropy': float(avg_entropy),
            'model_confidence': float(1.0 / (1.0 + avg_entropy))
        }
        
        return mse_loss, metrics
    
    def train_step(
        self, 
        model: Any, 
        optimizer_state: Any, 
        batch: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> Tuple[Any, Any, Dict[str, float]]:
        """Adaptive optimization training step."""
        
        # Compute loss and gradients
        (loss, metrics), grads = jax.value_and_grad(
            lambda m: self.compute_loss(m, batch), has_aux=True
        )(model)
        
        # Record performance for current optimizer
        self.optimizer_performance[self.current_optimizer].append(float(loss))
        
        # Select best optimizer
        if self.step % self.adaptation_window == 0:
            new_optimizer = self.select_best_optimizer()
            if new_optimizer != self.current_optimizer:
                print(f"Switching optimizer from {self.current_optimizer} to {new_optimizer}")
                self.current_optimizer = new_optimizer
        
        # Apply updates with current optimizer
        current_opt = self.optimizers[self.current_optimizer]
        current_state = self.optimizer_states[self.current_optimizer]
        
        updates, new_optimizer_state = current_opt.update(
            grads, current_state, eqx.filter(model, eqx.is_array)
        )
        new_model = eqx.apply_updates(model, updates)
        
        # Update optimizer state
        self.optimizer_states[self.current_optimizer] = new_optimizer_state
        
        self.step += 1
        
        metrics['current_optimizer'] = self.current_optimizer
        metrics['grad_norm'] = float(optax.global_norm(grads))
        
        return new_model, new_optimizer_state, metrics