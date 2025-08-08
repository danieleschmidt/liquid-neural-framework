import jax
import jax.numpy as jnp
from jax import grad, jit
import optax
from typing import Dict, Tuple, Optional, Callable, Any, List
import numpy as np


class AdaptiveOptimizer:
    """
    Adaptive optimization algorithms specifically designed for liquid neural networks.
    
    Implements custom optimization strategies that account for the continuous-time
    nature and adaptive parameters of liquid networks.
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-3,
        adaptation_rate: float = 1e-4,
        momentum: float = 0.9,
        eps: float = 1e-8
    ):
        self.learning_rate = learning_rate
        self.adaptation_rate = adaptation_rate
        self.momentum = momentum
        self.eps = eps
        
        # State tracking
        self.state = {}
    
    def init_state(self, params: Dict[str, jnp.ndarray]) -> Dict[str, Any]:
        """Initialize optimizer state."""
        state = {}
        for key, param in params.items():
            state[key] = {
                'momentum': jnp.zeros_like(param),
                'second_moment': jnp.zeros_like(param),
                'step_count': 0
            }
        return state
    
    def adaptive_time_constant_update(
        self,
        tau_grads: jnp.ndarray,
        tau_current: jnp.ndarray,
        tau_state: Dict[str, jnp.ndarray],
        step: int
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Specialized update for time constants with adaptive learning rates.
        
        Time constants require special handling because they should remain positive
        and their gradients often have different scales.
        """
        # Bias correction
        bias_correction1 = 1.0 - self.momentum ** step
        bias_correction2 = 1.0 - 0.999 ** step
        
        # Update momentum
        tau_state['momentum'] = (
            self.momentum * tau_state['momentum'] + 
            (1 - self.momentum) * tau_grads
        )
        
        # Update second moment
        tau_state['second_moment'] = (
            0.999 * tau_state['second_moment'] + 
            0.001 * tau_grads ** 2
        )
        
        # Bias-corrected estimates
        m_corrected = tau_state['momentum'] / bias_correction1
        v_corrected = tau_state['second_moment'] / bias_correction2
        
        # Adaptive learning rate for time constants
        adaptive_lr = self.adaptation_rate / (jnp.sqrt(v_corrected) + self.eps)
        
        # Ensure time constants remain positive using soft constraint
        tau_new = tau_current - adaptive_lr * m_corrected
        tau_new = jnp.maximum(tau_new, 0.01)  # Minimum time constant
        
        return tau_new, tau_state
    
    def liquid_adam_update(
        self,
        params: Dict[str, jnp.ndarray],
        grads: Dict[str, jnp.ndarray],
        state: Dict[str, Any]
    ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Any]]:
        """
        Liquid-specific Adam update with special handling for different parameter types.
        """
        new_params = {}
        new_state = {}
        
        for key in params:
            param = params[key]
            grad = grads[key]
            param_state = state.get(key, self.init_state({key: param})[key])
            
            param_state['step_count'] += 1
            step = param_state['step_count']
            
            # Special handling for time constants
            if key == 'tau':
                new_param, new_param_state = self.adaptive_time_constant_update(
                    grad, param, param_state, step
                )
            else:
                # Standard Adam update for other parameters
                new_param, new_param_state = self._standard_adam_update(
                    grad, param, param_state, step
                )
            
            new_params[key] = new_param
            new_state[key] = new_param_state
        
        return new_params, new_state
    
    def _standard_adam_update(
        self,
        grad: jnp.ndarray,
        param: jnp.ndarray,
        param_state: Dict[str, jnp.ndarray],
        step: int
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Standard Adam update for regular parameters."""
        # Bias correction
        bias_correction1 = 1.0 - self.momentum ** step
        bias_correction2 = 1.0 - 0.999 ** step
        
        # Update momentum
        param_state['momentum'] = (
            self.momentum * param_state['momentum'] + 
            (1 - self.momentum) * grad
        )
        
        # Update second moment
        param_state['second_moment'] = (
            0.999 * param_state['second_moment'] + 
            0.001 * grad ** 2
        )
        
        # Bias-corrected estimates
        m_corrected = param_state['momentum'] / bias_correction1
        v_corrected = param_state['second_moment'] / bias_correction2
        
        # Parameter update
        new_param = param - self.learning_rate * m_corrected / (jnp.sqrt(v_corrected) + self.eps)
        
        return new_param, param_state
    
    def compute_gradient_metrics(
        self, 
        grads: Dict[str, jnp.ndarray]
    ) -> Dict[str, float]:
        """Compute gradient statistics for monitoring."""
        total_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in grads.values()))
        
        metrics = {'total_grad_norm': float(total_norm)}
        
        for key, grad in grads.items():
            grad_norm = jnp.sqrt(jnp.sum(grad ** 2))
            metrics[f'{key}_grad_norm'] = float(grad_norm)
            metrics[f'{key}_grad_mean'] = float(jnp.mean(grad))
            metrics[f'{key}_grad_std'] = float(jnp.std(grad))
        
        return metrics


class ContinuousTimeOptimizer:
    """
    Optimization for continuous-time models with neural ODE integration.
    
    Handles the unique challenges of optimizing through continuous-time dynamics
    including proper gradient flow through ODE solvers.
    """
    
    def __init__(
        self,
        base_lr: float = 1e-3,
        dynamics_lr_scale: float = 0.1,
        regularization_strength: float = 1e-4
    ):
        self.base_lr = base_lr
        self.dynamics_lr_scale = dynamics_lr_scale
        self.regularization_strength = regularization_strength
        
        # Create different optimizers for different parameter groups
        self.param_optimizer = optax.adam(base_lr)
        self.dynamics_optimizer = optax.adam(base_lr * dynamics_lr_scale)
    
    def init_optimizer_states(
        self, 
        params: Dict[str, jnp.ndarray]
    ) -> Dict[str, Any]:
        """Initialize optimizer states for different parameter groups."""
        param_keys = ['W_in', 'W_out', 'b_h', 'b_out']
        dynamics_keys = ['W_hh', 'alpha', 'tau']
        
        param_subset = {k: v for k, v in params.items() if k in param_keys}
        dynamics_subset = {k: v for k, v in params.items() if k in dynamics_keys}
        
        return {
            'param_state': self.param_optimizer.init(param_subset),
            'dynamics_state': self.dynamics_optimizer.init(dynamics_subset)
        }
    
    def update(
        self,
        params: Dict[str, jnp.ndarray],
        grads: Dict[str, jnp.ndarray],
        opt_states: Dict[str, Any]
    ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Any]]:
        """Update parameters with specialized optimization for different groups."""
        param_keys = ['W_in', 'W_out', 'b_h', 'b_out']
        dynamics_keys = ['W_hh', 'alpha', 'tau']
        
        # Separate parameters and gradients
        param_subset = {k: params[k] for k in param_keys if k in params}
        param_grads = {k: grads[k] for k in param_keys if k in grads}
        
        dynamics_subset = {k: params[k] for k in dynamics_keys if k in params}
        dynamics_grads = {k: grads[k] for k in dynamics_keys if k in grads}
        
        # Update parameter subset
        if param_subset:
            param_updates, new_param_state = self.param_optimizer.update(
                param_grads, opt_states['param_state'], param_subset
            )
            new_param_subset = optax.apply_updates(param_subset, param_updates)
        else:
            new_param_subset = {}
            new_param_state = opt_states['param_state']
        
        # Update dynamics subset
        if dynamics_subset:
            dynamics_updates, new_dynamics_state = self.dynamics_optimizer.update(
                dynamics_grads, opt_states['dynamics_state'], dynamics_subset
            )
            new_dynamics_subset = optax.apply_updates(dynamics_subset, dynamics_updates)
            
            # Ensure positive time constants and alpha values
            if 'tau' in new_dynamics_subset:
                new_dynamics_subset['tau'] = jnp.maximum(new_dynamics_subset['tau'], 0.01)
            if 'alpha' in new_dynamics_subset:
                new_dynamics_subset['alpha'] = jnp.maximum(new_dynamics_subset['alpha'], 0.01)
        else:
            new_dynamics_subset = {}
            new_dynamics_state = opt_states['dynamics_state']
        
        # Combine updated parameters
        new_params = {**new_param_subset, **new_dynamics_subset}
        new_opt_states = {
            'param_state': new_param_state,
            'dynamics_state': new_dynamics_state
        }
        
        return new_params, new_opt_states
    
    def add_regularization_to_grads(
        self, 
        params: Dict[str, jnp.ndarray], 
        grads: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """Add regularization terms to gradients."""
        reg_grads = grads.copy()
        
        for key, param in params.items():
            if key in ['W_in', 'W_hh', 'W_out']:
                # L2 regularization for weight matrices
                reg_grads[key] += self.regularization_strength * param
            elif key == 'tau':
                # Encourage diversity in time constants
                tau_mean = jnp.mean(param)
                diversity_grad = -self.regularization_strength * (param - tau_mean)
                reg_grads[key] += diversity_grad
        
        return reg_grads


class MetaLearningOptimizer:
    """
    Meta-learning optimizer for liquid neural networks.
    
    Learns to adapt optimization parameters based on the current state
    of the network and training progress.
    """
    
    def __init__(self, base_lr: float = 1e-3):
        self.base_lr = base_lr
        self.meta_state = {}
        self.adaptation_history = []
    
    def compute_adaptive_lr(
        self,
        grad_norms: Dict[str, float],
        loss_history: List[float],
        current_epoch: int
    ) -> Dict[str, float]:
        """Compute adaptive learning rates based on training dynamics."""
        adaptive_lrs = {}
        
        # Base adaptation based on gradient norms
        for key, grad_norm in grad_norms.items():
            if grad_norm > 1.0:
                # Reduce learning rate for large gradients
                adaptive_lrs[key] = self.base_lr * 0.5
            elif grad_norm < 0.1:
                # Increase learning rate for small gradients
                adaptive_lrs[key] = self.base_lr * 2.0
            else:
                adaptive_lrs[key] = self.base_lr
        
        # Adaptation based on loss progression
        if len(loss_history) > 10:
            recent_improvement = loss_history[-10] - loss_history[-1]
            if recent_improvement < 1e-6:
                # Stagnation detected, increase learning rates
                for key in adaptive_lrs:
                    adaptive_lrs[key] *= 1.5
        
        return adaptive_lrs
    
    def meta_update(
        self,
        performance_metrics: Dict[str, float],
        gradient_metrics: Dict[str, float]
    ):
        """Update meta-learning parameters based on performance."""
        # Simple adaptation rule: track what works
        current_perf = performance_metrics.get('val_loss', float('inf'))
        
        if len(self.adaptation_history) > 0:
            prev_perf = self.adaptation_history[-1]['performance']
            if current_perf < prev_perf:
                # Performance improved, reinforce current strategy
                self.meta_state['success_count'] = self.meta_state.get('success_count', 0) + 1
            else:
                # Performance degraded, consider adaptation
                self.meta_state['failure_count'] = self.meta_state.get('failure_count', 0) + 1
        
        self.adaptation_history.append({
            'performance': current_perf,
            'gradients': gradient_metrics
        })