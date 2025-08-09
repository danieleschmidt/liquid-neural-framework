"""
Input validation and sanitization utilities.

This module provides comprehensive validation functions for liquid neural networks
to ensure robustness and catch potential issues early.
"""

import jax.numpy as jnp
from jax import random
from typing import Union, Tuple, Optional, Dict, Any
import warnings


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_array_shape(
    array: jnp.ndarray, 
    expected_shape: Union[Tuple[int, ...], Tuple[Union[int, str], ...]],
    name: str = "array"
) -> None:
    """
    Validate array shape matches expected dimensions.
    
    Args:
        array: Input array to validate
        expected_shape: Expected shape, can include strings for flexible dimensions
        name: Name of the array for error messages
        
    Raises:
        ValidationError: If shape doesn't match
    """
    if not isinstance(array, jnp.ndarray):
        raise ValidationError(f"{name} must be a JAX array, got {type(array)}")
    
    actual_shape = array.shape
    
    if len(actual_shape) != len(expected_shape):
        raise ValidationError(
            f"{name} has {len(actual_shape)} dimensions but expected {len(expected_shape)}"
        )
    
    for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
        if isinstance(expected, int) and actual != expected:
            raise ValidationError(
                f"{name} dimension {i} has size {actual} but expected {expected}"
            )


def validate_positive_scalar(value: float, name: str = "value", min_val: float = 0.0) -> None:
    """
    Validate that a scalar value is positive.
    
    Args:
        value: Scalar value to validate
        name: Name of the value for error messages
        min_val: Minimum allowed value
        
    Raises:
        ValidationError: If value is not positive or is NaN/Inf
    """
    if not isinstance(value, (int, float, jnp.ndarray)):
        raise ValidationError(f"{name} must be a number, got {type(value)}")
    
    if jnp.isnan(value) or jnp.isinf(value):
        raise ValidationError(f"{name} cannot be NaN or Inf, got {value}")
    
    if value <= min_val:
        raise ValidationError(f"{name} must be > {min_val}, got {value}")


def validate_array_finite(array: jnp.ndarray, name: str = "array") -> None:
    """
    Validate that all array elements are finite (not NaN or Inf).
    
    Args:
        array: Array to validate
        name: Name of the array for error messages
        
    Raises:
        ValidationError: If array contains NaN or Inf values
    """
    if jnp.any(jnp.isnan(array)):
        raise ValidationError(f"{name} contains NaN values")
    
    if jnp.any(jnp.isinf(array)):
        raise ValidationError(f"{name} contains infinite values")


def validate_sequence_data(
    inputs: jnp.ndarray, 
    targets: Optional[jnp.ndarray] = None,
    min_seq_length: int = 1,
    max_seq_length: int = 10000
) -> None:
    """
    Validate sequence data for training/inference.
    
    Args:
        inputs: Input sequences [batch_size, seq_length, input_dim]
        targets: Target sequences [batch_size, seq_length, output_dim] (optional)
        min_seq_length: Minimum allowed sequence length
        max_seq_length: Maximum allowed sequence length
        
    Raises:
        ValidationError: If data doesn't meet requirements
    """
    if inputs.ndim < 2:
        raise ValidationError(f"Input sequences must have at least 2 dimensions, got {inputs.ndim}")
    
    if inputs.ndim == 2:
        # Single sequence: [seq_length, input_dim]
        seq_length, input_dim = inputs.shape
        batch_size = 1
    elif inputs.ndim == 3:
        # Batch of sequences: [batch_size, seq_length, input_dim]
        batch_size, seq_length, input_dim = inputs.shape
    else:
        raise ValidationError(f"Input sequences cannot have more than 3 dimensions, got {inputs.ndim}")
    
    # Validate sequence length
    if seq_length < min_seq_length:
        raise ValidationError(f"Sequence length {seq_length} is below minimum {min_seq_length}")
    
    if seq_length > max_seq_length:
        warnings.warn(f"Sequence length {seq_length} is very long, may cause memory issues")
    
    # Validate input dimension
    if input_dim < 1:
        raise ValidationError(f"Input dimension must be positive, got {input_dim}")
    
    # Check for finite values
    validate_array_finite(inputs, "inputs")
    
    # Validate targets if provided
    if targets is not None:
        if targets.shape[:-1] != inputs.shape[:-1]:
            raise ValidationError(
                f"Target batch/sequence dimensions {targets.shape[:-1]} "
                f"don't match input dimensions {inputs.shape[:-1]}"
            )
        validate_array_finite(targets, "targets")


def validate_model_parameters(
    input_size: int,
    hidden_size: int,
    output_size: int,
    tau_min: Optional[float] = None,
    tau_max: Optional[float] = None
) -> None:
    """
    Validate model architecture parameters.
    
    Args:
        input_size: Number of input features
        hidden_size: Number of hidden units
        output_size: Number of output features
        tau_min: Minimum time constant (optional)
        tau_max: Maximum time constant (optional)
        
    Raises:
        ValidationError: If parameters are invalid
    """
    # Validate sizes
    for size, name in [(input_size, "input_size"), (hidden_size, "hidden_size"), (output_size, "output_size")]:
        if not isinstance(size, int) or size < 1:
            raise ValidationError(f"{name} must be a positive integer, got {size}")
    
    # Check reasonable limits
    if hidden_size > 10000:
        warnings.warn(f"Hidden size {hidden_size} is very large, may cause memory/performance issues")
    
    # Validate time constants if provided
    if tau_min is not None:
        validate_positive_scalar(tau_min, "tau_min", min_val=0.001)
    
    if tau_max is not None:
        validate_positive_scalar(tau_max, "tau_max", min_val=0.001)
    
    if tau_min is not None and tau_max is not None and tau_min >= tau_max:
        raise ValidationError(f"tau_min ({tau_min}) must be less than tau_max ({tau_max})")


def validate_integration_parameters(dt: float, method: str = "euler") -> None:
    """
    Validate numerical integration parameters.
    
    Args:
        dt: Integration time step
        method: Integration method
        
    Raises:
        ValidationError: If parameters are invalid
    """
    # Validate time step
    validate_positive_scalar(dt, "dt", min_val=1e-6)
    
    if dt > 1.0:
        warnings.warn(f"Time step dt={dt} is very large, may cause numerical instability")
    
    # Validate integration method
    valid_methods = ["euler", "rk2", "rk4", "ode"]
    if method not in valid_methods:
        raise ValidationError(f"Integration method '{method}' not supported. Valid options: {valid_methods}")


def validate_training_parameters(
    learning_rate: float,
    batch_size: Optional[int] = None,
    epochs: Optional[int] = None,
    gradient_clip: Optional[float] = None
) -> None:
    """
    Validate training hyperparameters.
    
    Args:
        learning_rate: Learning rate for optimization
        batch_size: Batch size (optional)
        epochs: Number of training epochs (optional)
        gradient_clip: Gradient clipping threshold (optional)
        
    Raises:
        ValidationError: If parameters are invalid
    """
    # Validate learning rate
    validate_positive_scalar(learning_rate, "learning_rate", min_val=1e-10)
    
    if learning_rate > 1.0:
        warnings.warn(f"Learning rate {learning_rate} is very high, may cause training instability")
    
    # Validate batch size
    if batch_size is not None:
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValidationError(f"batch_size must be a positive integer, got {batch_size}")
    
    # Validate epochs
    if epochs is not None:
        if not isinstance(epochs, int) or epochs < 1:
            raise ValidationError(f"epochs must be a positive integer, got {epochs}")
    
    # Validate gradient clipping
    if gradient_clip is not None:
        validate_positive_scalar(gradient_clip, "gradient_clip", min_val=1e-6)


def validate_prng_key(key: Optional[random.PRNGKey]) -> random.PRNGKey:
    """
    Validate and potentially generate PRNG key.
    
    Args:
        key: JAX random key or None
        
    Returns:
        Valid PRNG key
        
    Raises:
        ValidationError: If key is invalid
    """
    if key is None:
        return random.PRNGKey(0)
    
    if not isinstance(key, jnp.ndarray) or key.shape != (2,) or key.dtype != jnp.uint32:
        raise ValidationError(
            f"PRNG key must be a JAX array with shape (2,) and dtype uint32, "
            f"got shape {key.shape if hasattr(key, 'shape') else 'N/A'} "
            f"and dtype {key.dtype if hasattr(key, 'dtype') else type(key)}"
        )
    
    return key


def sanitize_weights(weights: jnp.ndarray, max_norm: float = 10.0) -> jnp.ndarray:
    """
    Sanitize weight matrices by clipping extreme values.
    
    Args:
        weights: Weight matrix to sanitize
        max_norm: Maximum allowed weight magnitude
        
    Returns:
        Sanitized weights
    """
    # Clip extreme values
    weights = jnp.clip(weights, -max_norm, max_norm)
    
    # Replace any remaining NaN/Inf with small random values
    if jnp.any(jnp.isnan(weights)) or jnp.any(jnp.isinf(weights)):
        key = random.PRNGKey(42)
        replacement = random.normal(key, weights.shape) * 0.01
        weights = jnp.where(jnp.isfinite(weights), weights, replacement)
        warnings.warn("Replaced NaN/Inf values in weights with small random values")
    
    return weights


def check_numerical_stability(
    outputs: jnp.ndarray,
    hidden_states: jnp.ndarray,
    threshold: float = 1e6
) -> Dict[str, bool]:
    """
    Check for numerical stability issues in model outputs.
    
    Args:
        outputs: Model outputs
        hidden_states: Hidden state trajectories
        threshold: Threshold for detecting instability
        
    Returns:
        Dictionary of stability checks
    """
    checks = {}
    
    # Check for exploding values
    checks["outputs_stable"] = jnp.all(jnp.abs(outputs) < threshold)
    checks["states_stable"] = jnp.all(jnp.abs(hidden_states) < threshold)
    
    # Check for NaN/Inf
    checks["outputs_finite"] = jnp.all(jnp.isfinite(outputs))
    checks["states_finite"] = jnp.all(jnp.isfinite(hidden_states))
    
    # Check for vanishing gradients (very small values)
    checks["outputs_not_vanishing"] = jnp.any(jnp.abs(outputs) > 1e-8)
    checks["states_not_vanishing"] = jnp.any(jnp.abs(hidden_states) > 1e-8)
    
    # Overall stability
    checks["overall_stable"] = all(checks.values())
    
    return checks