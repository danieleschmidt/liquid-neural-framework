import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
import optax
import equinox as eqx
from typing import Dict, Tuple, Optional, Callable, Any
import numpy as np
from ..models.liquid_neural_network import LiquidNeuralNetwork
from ..models.continuous_time_rnn import ContinuousTimeRNN


class LiquidNetworkTrainer:
    """
    Training framework for liquid neural networks with continuous-time dynamics.
    
    Supports various loss functions, optimization algorithms, and training schedules
    specifically designed for liquid neural network architectures.
    """
    
    def __init__(
        self,
        model: LiquidNeuralNetwork,
        learning_rate: float = 1e-3,
        optimizer_name: str = 'adam',
        loss_fn: str = 'mse',
        gradient_clip: Optional[float] = 1.0
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.gradient_clip = gradient_clip
        
        # Initialize optimizer
        self.optimizer = self._get_optimizer(optimizer_name, learning_rate)
        # Use only differentiable parameters for optimizer state
        diff_params, _ = eqx.partition(model, eqx.is_array)
        self.opt_state = self.optimizer.init(diff_params)
        
        # Get loss function
        self.loss_fn = self._get_loss_function(loss_fn)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'gradient_norm': []
        }
    
    def _get_optimizer(self, optimizer_name: str, lr: float) -> optax.GradientTransformation:
        """Initialize optimizer."""
        optimizers = {
            'adam': optax.adam(lr),
            'adamw': optax.adamw(lr, weight_decay=1e-4),
            'sgd': optax.sgd(lr),
            'rmsprop': optax.rmsprop(lr),
            'adagrad': optax.adagrad(lr)
        }
        
        optimizer = optimizers.get(optimizer_name, optax.adam(lr))
        
        # Add gradient clipping if specified
        if self.gradient_clip is not None:
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.gradient_clip),
                optimizer
            )
        
        return optimizer
    
    def _get_loss_function(self, loss_name: str) -> Callable:
        """Get loss function by name."""
        losses = {
            'mse': self._mse_loss,
            'mae': self._mae_loss,
            'cross_entropy': self._cross_entropy_loss,
            'huber': self._huber_loss,
            'temporal_consistency': self._temporal_consistency_loss
        }
        return losses.get(loss_name, self._mse_loss)
    
    def _mse_loss(self, predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        """Mean squared error loss."""
        return jnp.mean((predictions - targets) ** 2)
    
    def _mae_loss(self, predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        """Mean absolute error loss."""
        return jnp.mean(jnp.abs(predictions - targets))
    
    def _cross_entropy_loss(self, predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        """Cross entropy loss for classification."""
        return -jnp.mean(targets * jax.nn.log_softmax(predictions))
    
    def _huber_loss(self, predictions: jnp.ndarray, targets: jnp.ndarray, delta: float = 1.0) -> jnp.ndarray:
        """Huber loss (less sensitive to outliers)."""
        residual = jnp.abs(predictions - targets)
        return jnp.where(
            residual < delta,
            0.5 * residual ** 2,
            delta * residual - 0.5 * delta ** 2
        ).mean()
    
    def _temporal_consistency_loss(self, predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        """Temporal consistency loss for sequence data."""
        # Standard prediction loss
        pred_loss = self._mse_loss(predictions, targets)
        
        # Temporal smoothness penalty
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            temporal_diff = jnp.diff(predictions, axis=1)
            smoothness_penalty = jnp.mean(temporal_diff ** 2) * 0.1
            return pred_loss + smoothness_penalty
        
        return pred_loss
    
    def _loss_and_metrics(
        self, 
        diff_params, 
        static_params,
        inputs: jnp.ndarray, 
        targets: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Compute loss and metrics using equinox partitioning approach."""
        # Reconstruct model from diff and static parts
        model = eqx.combine(diff_params, static_params)
        
        # Forward pass
        predictions = model(inputs)
        
        # Compute primary loss
        loss = self.loss_fn(predictions, targets)
        
        # Additional regularization
        # L2 regularization on differentiable parameters only
        array_params = jax.tree.leaves(diff_params)
        if array_params:
            l2_reg = sum(jnp.sum(p ** 2) for p in array_params)
        else:
            l2_reg = jnp.float32(0.0)
        loss += 1e-5 * l2_reg
        
        # Compute metrics
        metrics = {
            'loss': loss,
            'mse': self._mse_loss(predictions, targets),
            'mae': self._mae_loss(predictions, targets),
            'l2_reg': l2_reg
        }
        
        return loss, metrics
    
    def _update_step(
        self, 
        model: LiquidNeuralNetwork, 
        opt_state: Any, 
        inputs: jnp.ndarray, 
        targets: jnp.ndarray
    ) -> Tuple[LiquidNeuralNetwork, Any, Dict[str, jnp.ndarray]]:
        """Single training step using equinox partitioning."""
        # Partition model into differentiable and static parts
        diff_params, static_params = eqx.partition(model, eqx.is_array)
        
        # Compute loss and gradients w.r.t. differentiable parameters only
        def loss_fn(diff_p):
            return self._loss_and_metrics(diff_p, static_params, inputs, targets)
        
        (loss, metrics), grads = value_and_grad(loss_fn, has_aux=True)(diff_params)
        
        # Compute gradient norm for monitoring
        array_grads = jax.tree.leaves(grads)
        if array_grads:
            grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in array_grads))
        else:
            grad_norm = jnp.float32(0.0)
        metrics['gradient_norm'] = grad_norm
        
        # Update differentiable parameters
        updates, new_opt_state = self.optimizer.update(grads, opt_state, diff_params)
        new_diff_params = eqx.apply_updates(diff_params, updates)
        
        # Combine updated differentiable params with static params
        new_model = eqx.combine(new_diff_params, static_params)
        
        return new_model, new_opt_state, metrics
    
    def train_step(
        self, 
        inputs: jnp.ndarray, 
        targets: jnp.ndarray
    ) -> Dict[str, float]:
        """Single training step."""
        new_model, new_opt_state, metrics = self._update_step(
            self.model, self.opt_state, inputs, targets
        )
        
        # Update model and optimizer state
        self.model = new_model
        self.opt_state = new_opt_state
        
        # Convert to Python floats for logging
        return {k: float(v) for k, v in metrics.items()}
    
    def validate(
        self, 
        val_inputs: jnp.ndarray, 
        val_targets: jnp.ndarray
    ) -> Dict[str, float]:
        """Validation step."""
        diff_params, static_params = eqx.partition(self.model, eqx.is_array)
        loss, metrics = self._loss_and_metrics(
            diff_params, static_params, val_inputs, val_targets
        )
        return {k: float(v) for k, v in metrics.items()}
    
    def fit(
        self,
        train_data: Tuple[jnp.ndarray, jnp.ndarray],
        val_data: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        epochs: int = 100,
        verbose: bool = True,
        log_interval: int = 10
    ) -> Dict[str, list]:
        """Train the model."""
        train_inputs, train_targets = train_data
        
        for epoch in range(epochs):
            # Training step
            train_metrics = self.train_step(train_inputs, train_targets)
            
            # Validation step
            if val_data is not None:
                val_inputs, val_targets = val_data
                val_metrics = self.validate(val_inputs, val_targets)
            else:
                val_metrics = {'loss': float('nan')}
            
            # Log metrics
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['gradient_norm'].append(train_metrics.get('gradient_norm', 0.0))
            
            # Print progress
            if verbose and (epoch + 1) % log_interval == 0:
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"  Train Loss: {train_metrics['loss']:.6f}")
                if val_data is not None:
                    print(f"  Val Loss: {val_metrics['loss']:.6f}")
                print(f"  Grad Norm: {train_metrics.get('gradient_norm', 0.0):.6f}")
                print()
        
        return self.history
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        eqx.tree_serialise_leaves(filepath, self.model)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        self.model = eqx.tree_deserialise_leaves(filepath, self.model)


# JIT compile the training functions for performance
@jax.jit
def compiled_loss_fn(model, inputs, targets, loss_fn):
    """JIT-compiled loss function."""
    predictions = model(inputs)
    return loss_fn(predictions, targets)


@jax.jit  
def compiled_update_step(model, opt_state, optimizer, inputs, targets, loss_fn):
    """JIT-compiled training step."""
    def loss_fn_wrapper(model):
        predictions = model(inputs)
        return loss_fn(predictions, targets)
    
    loss, grads = value_and_grad(loss_fn_wrapper)(model)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_model = eqx.apply_updates(model, updates)
    
    return new_model, new_opt_state, loss


class AdvancedLiquidTrainer(LiquidNetworkTrainer):
    """
    Advanced trainer with additional features for liquid neural networks.
    
    Includes:
    - Learning rate scheduling
    - Early stopping
    - Liquid state analysis
    - Adaptive time constants
    """
    
    def __init__(
        self,
        model: LiquidNeuralNetwork,
        learning_rate: float = 1e-3,
        optimizer_name: str = 'adam',
        loss_fn: str = 'mse',
        gradient_clip: Optional[float] = 1.0,
        lr_schedule: str = 'constant',
        early_stopping_patience: int = 10
    ):
        super().__init__(model, learning_rate, optimizer_name, loss_fn, gradient_clip)
        
        self.lr_schedule = lr_schedule
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Liquid state tracking
        self.liquid_state_history = []
    
    def _get_learning_rate(self, epoch: int, base_lr: float) -> float:
        """Get learning rate based on schedule."""
        if self.lr_schedule == 'constant':
            return base_lr
        elif self.lr_schedule == 'exponential':
            return base_lr * (0.95 ** epoch)
        elif self.lr_schedule == 'cosine':
            return base_lr * 0.5 * (1 + jnp.cos(jnp.pi * epoch / 100))
        elif self.lr_schedule == 'step':
            if epoch < 50:
                return base_lr
            elif epoch < 80:
                return base_lr * 0.1
            else:
                return base_lr * 0.01
        else:
            return base_lr
    
    def analyze_liquid_states(self, inputs: jnp.ndarray) -> Dict[str, Any]:
        """Analyze liquid state dynamics."""
        if hasattr(self.model, 'get_liquid_states'):
            states = self.model.get_liquid_states(inputs)
            
            analysis = {}
            for i, state in enumerate(states):
                layer_analysis = {
                    'mean_activity': float(jnp.mean(jnp.abs(state))),
                    'activity_variance': float(jnp.var(state)),
                    'sparsity': float(jnp.mean(jnp.abs(state) < 0.1)),
                    'max_activity': float(jnp.max(jnp.abs(state)))
                }
                analysis[f'layer_{i}'] = layer_analysis
            
            return analysis
        else:
            return {}
    
    def fit_advanced(
        self,
        train_data: Tuple[jnp.ndarray, jnp.ndarray],
        val_data: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        epochs: int = 100,
        verbose: bool = True,
        log_interval: int = 10,
        analyze_states: bool = True
    ) -> Dict[str, list]:
        """Advanced training with additional features."""
        train_inputs, train_targets = train_data
        
        for epoch in range(epochs):
            # Update learning rate
            current_lr = self._get_learning_rate(epoch, self.learning_rate)
            
            # Training step
            train_metrics = self.train_step(train_inputs, train_targets)
            
            # Validation step
            if val_data is not None:
                val_inputs, val_targets = val_data
                val_metrics = self.validate(val_inputs, val_targets)
                
                # Early stopping check
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                val_metrics = {'loss': float('nan')}
            
            # Liquid state analysis
            if analyze_states and (epoch + 1) % (log_interval * 2) == 0:
                state_analysis = self.analyze_liquid_states(train_inputs[:1])  # Analyze single sample
                self.liquid_state_history.append(state_analysis)
            
            # Log metrics
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['learning_rate'].append(current_lr)
            self.history['gradient_norm'].append(train_metrics.get('gradient_norm', 0.0))
            
            # Print progress
            if verbose and (epoch + 1) % log_interval == 0:
                print(f"Epoch {epoch + 1}/{epochs} (LR: {current_lr:.2e})")
                print(f"  Train Loss: {train_metrics['loss']:.6f}")
                if val_data is not None:
                    print(f"  Val Loss: {val_metrics['loss']:.6f}")
                print(f"  Grad Norm: {train_metrics.get('gradient_norm', 0.0):.6f}")
                
                if analyze_states and self.liquid_state_history:
                    recent_analysis = self.liquid_state_history[-1]
                    if 'layer_0' in recent_analysis:
                        layer_0 = recent_analysis['layer_0']
                        print(f"  Liquid States - Activity: {layer_0['mean_activity']:.4f}, "
                              f"Sparsity: {layer_0['sparsity']:.3f}")
                print()
        
        return self.history