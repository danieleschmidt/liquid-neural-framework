import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
import optax
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
        self.opt_state = self.optimizer.init(model.params)
        
        # Get loss function
        self.loss_fn = self._get_loss_function(loss_fn)
        
        # JIT compile functions for speed
        self._compiled_loss = jit(self._loss_and_metrics)
        self._compiled_update = jit(self._update_step)
        
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
        if predictions.ndim > 1 and predictions.shape[0] > 1:
            temporal_diff = jnp.diff(predictions, axis=0)
            smoothness_penalty = jnp.mean(temporal_diff ** 2) * 0.1
            return pred_loss + smoothness_penalty
        
        return pred_loss
    
    def _loss_and_metrics(
        self, 
        params: Dict[str, jnp.ndarray], 
        inputs: jnp.ndarray, 
        targets: jnp.ndarray,
        dt: float = 0.1
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Compute loss and metrics."""
        # Update model params and forward pass
        self.model.update_params(params)
        predictions, states = self.model.forward(inputs, dt=dt)
        
        # Compute primary loss
        loss = self.loss_fn(predictions, targets)
        
        # Additional regularization
        # L2 regularization on parameters
        l2_reg = sum(jnp.sum(p ** 2) for p in params.values() if p.ndim > 0)
        loss += 1e-5 * l2_reg
        
        # Time constant regularization (encourage diversity)
        if 'tau' in params:
            tau_diversity = -jnp.var(params['tau'])  # Negative because we want to maximize variance
            loss += 1e-4 * tau_diversity
        
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
        params: Dict[str, jnp.ndarray], 
        opt_state: Any, 
        inputs: jnp.ndarray, 
        targets: jnp.ndarray,
        dt: float = 0.1
    ) -> Tuple[Dict[str, jnp.ndarray], Any, Dict[str, jnp.ndarray]]:
        """Single training step."""
        # Compute loss and gradients
        (loss, metrics), grads = value_and_grad(
            self._loss_and_metrics, has_aux=True
        )(params, inputs, targets, dt)
        
        # Compute gradient norm for monitoring
        grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in grads.values()))
        metrics['gradient_norm'] = grad_norm
        
        # Update parameters
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, metrics
    
    def train_step(
        self, 
        inputs: jnp.ndarray, 
        targets: jnp.ndarray,
        dt: float = 0.1
    ) -> Dict[str, float]:
        """Single training step."""
        new_params, new_opt_state, metrics = self._compiled_update(
            self.model.params, self.opt_state, inputs, targets, dt
        )
        
        # Update model and optimizer state
        self.model.update_params(new_params)
        self.opt_state = new_opt_state
        
        # Convert to Python floats for logging
        return {k: float(v) for k, v in metrics.items()}
    
    def validate(
        self, 
        val_inputs: jnp.ndarray, 
        val_targets: jnp.ndarray,
        dt: float = 0.1
    ) -> Dict[str, float]:
        """Validation step."""
        loss, metrics = self._compiled_loss(
            self.model.params, val_inputs, val_targets, dt
        )
        return {k: float(v) for k, v in metrics.items()}
    
    def fit(
        self,
        train_data: Tuple[jnp.ndarray, jnp.ndarray],
        val_data: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        epochs: int = 100,
        dt: float = 0.1,
        verbose: bool = True,
        log_interval: int = 10
    ) -> Dict[str, list]:
        """Train the model."""
        train_inputs, train_targets = train_data
        
        for epoch in range(epochs):
            # Training step
            train_metrics = self.train_step(train_inputs, train_targets, dt)
            
            # Validation step
            if val_data is not None:
                val_inputs, val_targets = val_data
                val_metrics = self.validate(val_inputs, val_targets, dt)
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
                
                # Print time constant info if available
                if 'tau' in self.model.params:
                    tau_stats = {
                        'mean': float(jnp.mean(self.model.params['tau'])),
                        'std': float(jnp.std(self.model.params['tau']))
                    }
                    print(f"  Time Constants - Mean: {tau_stats['mean']:.3f}, Std: {tau_stats['std']:.3f}")
                print()
        
        return self.history
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            'params': self.model.params,
            'opt_state': self.opt_state,
            'history': self.history
        }
        jnp.save(filepath, checkpoint)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = jnp.load(filepath, allow_pickle=True).item()
        self.model.update_params(checkpoint['params'])
        self.opt_state = checkpoint['opt_state']
        self.history = checkpoint['history']