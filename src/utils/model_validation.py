"""
Comprehensive validation utilities for liquid neural network models.
Includes input validation, model state checking, and error recovery.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import functools


class ValidationError(Exception):
    """Custom validation error for model operations."""
    pass


class ModelValidator:
    """Comprehensive model validation and error checking."""
    
    @staticmethod
    def validate_input_tensor(x: jnp.ndarray, expected_shape: Tuple[int, ...], 
                            name: str = "input") -> jnp.ndarray:
        """Validate input tensor shape and values."""
        if not isinstance(x, jnp.ndarray):
            try:
                x = jnp.array(x)
            except Exception as e:
                raise ValidationError(f"{name} must be convertible to JAX array: {e}")
        
        if x.shape != expected_shape:
            raise ValidationError(
                f"{name} shape mismatch: expected {expected_shape}, got {x.shape}"
            )
        
        # Check for NaN/Inf values
        if jnp.any(jnp.isnan(x)):
            raise ValidationError(f"{name} contains NaN values")
        
        if jnp.any(jnp.isinf(x)):
            raise ValidationError(f"{name} contains infinite values")
        
        return x
    
    @staticmethod
    def validate_time_step(dt: float) -> float:
        """Validate time step parameter."""
        if not isinstance(dt, (int, float)):
            raise ValidationError(f"Time step must be numeric, got {type(dt)}")
        
        if dt <= 0:
            raise ValidationError(f"Time step must be positive, got {dt}")
        
        if dt > 1.0:
            warnings.warn(f"Large time step {dt} may cause instability")
        
        return float(dt)
    
    @staticmethod
    def validate_model_state(model: eqx.Module) -> bool:
        """Validate model parameters for NaN/Inf values."""
        def check_leaf(leaf):
            if isinstance(leaf, jnp.ndarray):
                if jnp.any(jnp.isnan(leaf)):
                    return False, "NaN detected in model parameters"
                if jnp.any(jnp.isinf(leaf)):
                    return False, "Inf detected in model parameters"
            return True, "OK"
        
        leaves = jax.tree_util.tree_leaves(model)
        for leaf in leaves:
            valid, msg = check_leaf(leaf)
            if not valid:
                raise ValidationError(f"Model state invalid: {msg}")
        
        return True
    
    @staticmethod
    def validate_hidden_states(hidden_states: Union[List, jnp.ndarray], 
                             expected_shapes: List[Tuple[int, ...]]) -> List[jnp.ndarray]:
        """Validate hidden state dimensions and values."""
        if isinstance(hidden_states, jnp.ndarray):
            hidden_states = [hidden_states]
        
        if len(hidden_states) != len(expected_shapes):
            raise ValidationError(
                f"Hidden state count mismatch: expected {len(expected_shapes)}, "
                f"got {len(hidden_states)}"
            )
        
        validated_states = []
        for i, (state, expected_shape) in enumerate(zip(hidden_states, expected_shapes)):
            validated_state = ModelValidator.validate_input_tensor(
                state, expected_shape, f"hidden_state_{i}"
            )
            validated_states.append(validated_state)
        
        return validated_states


def safe_forward_pass(func):
    """Decorator for safe model forward passes with error handling."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            
            # Validate outputs
            if isinstance(result, tuple):
                for i, output in enumerate(result):
                    if isinstance(output, jnp.ndarray):
                        if jnp.any(jnp.isnan(output)):
                            raise ValidationError(f"Output {i} contains NaN values")
                        if jnp.any(jnp.isinf(output)):
                            raise ValidationError(f"Output {i} contains infinite values")
            elif isinstance(result, jnp.ndarray):
                if jnp.any(jnp.isnan(result)):
                    raise ValidationError("Output contains NaN values")
                if jnp.any(jnp.isinf(result)):
                    raise ValidationError("Output contains infinite values")
            
            return result
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Forward pass failed: {str(e)}")
    
    return wrapper


class GradientClipper:
    """Utility for gradient clipping and normalization."""
    
    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm
    
    def clip_gradients(self, gradients: Any) -> Any:
        """Clip gradients by global norm."""
        def clip_leaf(grad):
            if isinstance(grad, jnp.ndarray):
                grad_norm = jnp.linalg.norm(grad)
                if grad_norm > self.max_norm:
                    return grad * (self.max_norm / grad_norm)
            return grad
        
        return jax.tree_util.tree_map(clip_leaf, gradients)


class ModelStateMonitor:
    """Monitor model health during training."""
    
    def __init__(self, history_length: int = 100):
        self.history_length = history_length
        self.loss_history = []
        self.gradient_norms = []
        self.parameter_norms = []
    
    def update(self, loss: float, gradients: Any, model: eqx.Module):
        """Update monitoring statistics."""
        # Track loss
        self.loss_history.append(loss)
        if len(self.loss_history) > self.history_length:
            self.loss_history.pop(0)
        
        # Track gradient norms
        grad_norm = jnp.sqrt(sum(
            jnp.sum(g**2) for g in jax.tree_util.tree_leaves(gradients)
            if isinstance(g, jnp.ndarray)
        ))
        self.gradient_norms.append(grad_norm)
        if len(self.gradient_norms) > self.history_length:
            self.gradient_norms.pop(0)
        
        # Track parameter norms
        param_norm = jnp.sqrt(sum(
            jnp.sum(p**2) for p in jax.tree_util.tree_leaves(model)
            if isinstance(p, jnp.ndarray)
        ))
        self.parameter_norms.append(param_norm)
        if len(self.parameter_norms) > self.history_length:
            self.parameter_norms.pop(0)
    
    def check_health(self) -> Dict[str, bool]:
        """Check model training health."""
        health = {
            "loss_stable": True,
            "gradients_healthy": True,
            "parameters_healthy": True
        }
        
        if len(self.loss_history) > 10:
            recent_losses = self.loss_history[-10:]
            if any(jnp.isnan(loss) or jnp.isinf(loss) for loss in recent_losses):
                health["loss_stable"] = False
        
        if len(self.gradient_norms) > 5:
            recent_grads = self.gradient_norms[-5:]
            if any(jnp.isnan(norm) or jnp.isinf(norm) or norm > 100.0 for norm in recent_grads):
                health["gradients_healthy"] = False
        
        if len(self.parameter_norms) > 5:
            recent_params = self.parameter_norms[-5:]
            if any(jnp.isnan(norm) or jnp.isinf(norm) for norm in recent_params):
                health["parameters_healthy"] = False
        
        return health
    
    def get_stats(self) -> Dict[str, float]:
        """Get monitoring statistics."""
        stats = {}
        
        if self.loss_history:
            stats["avg_loss"] = float(jnp.mean(jnp.array(self.loss_history)))
            stats["loss_std"] = float(jnp.std(jnp.array(self.loss_history)))
        
        if self.gradient_norms:
            stats["avg_grad_norm"] = float(jnp.mean(jnp.array(self.gradient_norms)))
            stats["max_grad_norm"] = float(jnp.max(jnp.array(self.gradient_norms)))
        
        if self.parameter_norms:
            stats["param_norm"] = float(self.parameter_norms[-1])
        
        return stats


class SafeModelWrapper:
    """Wrapper for models with automatic validation and error recovery."""
    
    def __init__(self, model: eqx.Module, validator: ModelValidator = None):
        self.model = model
        self.validator = validator or ModelValidator()
        self.monitor = ModelStateMonitor()
        self.clipper = GradientClipper()
        self.backup_model = model  # Keep a backup for recovery
    
    def safe_call(self, *args, **kwargs):
        """Safe model call with validation."""
        try:
            # Validate model state
            self.validator.validate_model_state(self.model)
            
            # Forward pass
            result = self.model(*args, **kwargs)
            
            return result
            
        except ValidationError as e:
            warnings.warn(f"Model validation failed: {e}")
            # Attempt recovery with backup model
            try:
                result = self.backup_model(*args, **kwargs)
                warnings.warn("Recovered using backup model")
                return result
            except Exception:
                raise e
    
    def update_backup(self):
        """Update backup model."""
        self.backup_model = self.model