"""
Robust validation and error handling for liquid neural networks.

This module provides comprehensive validation, error handling, and
input sanitization for all network components.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class ShapeValidationError(ValidationError):
    """Raised when tensor shapes are invalid."""
    pass


class ValueValidationError(ValidationError):
    """Raised when tensor values are invalid."""
    pass


class ParameterValidationError(ValidationError):
    """Raised when parameters are invalid."""
    pass


def validate_tensor_shape(tensor: np.ndarray, expected_shape: Tuple[int, ...], 
                         name: str = "tensor", allow_batch_dim: bool = True) -> np.ndarray:
    """
    Validate tensor shape with comprehensive error reporting.
    
    Args:
        tensor: Input tensor to validate
        expected_shape: Expected shape (can include None for flexible dimensions)
        name: Name of the tensor for error reporting
        allow_batch_dim: Whether to allow additional batch dimension
        
    Returns:
        Validated tensor
        
    Raises:
        ShapeValidationError: If shape is invalid
    """
    if not isinstance(tensor, np.ndarray):
        try:
            tensor = np.asarray(tensor)
        except Exception as e:
            raise ShapeValidationError(f"{name} cannot be converted to array: {e}")
    
    if tensor.ndim == 0:
        raise ShapeValidationError(f"{name} cannot be a scalar, got shape {tensor.shape}")
    
    # Handle batch dimension
    if allow_batch_dim and tensor.ndim == len(expected_shape) + 1:
        # Check all dimensions except the first (batch)
        for i, (actual, expected) in enumerate(zip(tensor.shape[1:], expected_shape)):
            if expected is not None and actual != expected:
                raise ShapeValidationError(
                    f"{name} shape mismatch at dimension {i+1}: expected {expected}, got {actual}. "
                    f"Full shape: {tensor.shape}, expected: (batch_size, {expected_shape})"
                )
    else:
        # Check exact shape match
        if len(tensor.shape) != len(expected_shape):
            raise ShapeValidationError(
                f"{name} has wrong number of dimensions: expected {len(expected_shape)}, "
                f"got {len(tensor.shape)}. Shape: {tensor.shape}, expected: {expected_shape}"
            )
        
        for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
            if expected is not None and actual != expected:
                raise ShapeValidationError(
                    f"{name} shape mismatch at dimension {i}: expected {expected}, got {actual}. "
                    f"Full shape: {tensor.shape}, expected: {expected_shape}"
                )
    
    return tensor


def validate_tensor_values(tensor: np.ndarray, name: str = "tensor",
                          min_val: Optional[float] = None, max_val: Optional[float] = None,
                          allow_nan: bool = False, allow_inf: bool = False) -> np.ndarray:
    """
    Validate tensor values for numerical stability.
    
    Args:
        tensor: Input tensor to validate
        name: Name of the tensor for error reporting
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        allow_nan: Whether to allow NaN values
        allow_inf: Whether to allow infinite values
        
    Returns:
        Validated tensor
        
    Raises:
        ValueValidationError: If values are invalid
    """
    if not allow_nan and np.any(np.isnan(tensor)):
        nan_count = np.sum(np.isnan(tensor))
        raise ValueValidationError(f"{name} contains {nan_count} NaN values")
    
    if not allow_inf and np.any(np.isinf(tensor)):
        inf_count = np.sum(np.isinf(tensor))
        raise ValueValidationError(f"{name} contains {inf_count} infinite values")
    
    if min_val is not None:
        if np.any(tensor < min_val):
            min_actual = np.min(tensor)
            violation_count = np.sum(tensor < min_val)
            raise ValueValidationError(
                f"{name} has {violation_count} values below minimum {min_val}. "
                f"Actual minimum: {min_actual}"
            )
    
    if max_val is not None:
        if np.any(tensor > max_val):
            max_actual = np.max(tensor)
            violation_count = np.sum(tensor > max_val)
            raise ValueValidationError(
                f"{name} has {violation_count} values above maximum {max_val}. "
                f"Actual maximum: {max_actual}"
            )
    
    return tensor


def validate_network_parameters(input_size: int, hidden_size: int, output_size: int,
                               num_layers: int = 1, tau_min: float = 0.01, 
                               tau_max: float = 100.0) -> Tuple[int, int, int, int, float, float]:
    """
    Validate network architecture parameters.
    
    Args:
        input_size: Input dimension
        hidden_size: Hidden dimension
        output_size: Output dimension
        num_layers: Number of layers
        tau_min: Minimum time constant
        tau_max: Maximum time constant
        
    Returns:
        Validated parameters
        
    Raises:
        ParameterValidationError: If parameters are invalid
    """
    if not isinstance(input_size, int) or input_size <= 0:
        raise ParameterValidationError(f"input_size must be positive integer, got {input_size}")
    
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ParameterValidationError(f"hidden_size must be positive integer, got {hidden_size}")
    
    if not isinstance(output_size, int) or output_size <= 0:
        raise ParameterValidationError(f"output_size must be positive integer, got {output_size}")
    
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise ParameterValidationError(f"num_layers must be positive integer, got {num_layers}")
    
    if not isinstance(tau_min, (int, float)) or tau_min <= 0:
        raise ParameterValidationError(f"tau_min must be positive number, got {tau_min}")
    
    if not isinstance(tau_max, (int, float)) or tau_max <= tau_min:
        raise ParameterValidationError(
            f"tau_max must be greater than tau_min ({tau_min}), got {tau_max}"
        )
    
    # Reasonable bounds checking
    if input_size > 10000:
        warnings.warn(f"Large input_size ({input_size}) may cause memory issues")
    
    if hidden_size > 10000:
        warnings.warn(f"Large hidden_size ({hidden_size}) may cause memory issues")
    
    if num_layers > 100:
        warnings.warn(f"Many layers ({num_layers}) may cause gradient issues")
    
    return input_size, hidden_size, output_size, num_layers, float(tau_min), float(tau_max)


def sanitize_inputs(inputs: np.ndarray, clip_range: Tuple[float, float] = (-10.0, 10.0)) -> np.ndarray:
    """
    Sanitize input data to prevent numerical instabilities.
    
    Args:
        inputs: Input tensor
        clip_range: Range to clip values to
        
    Returns:
        Sanitized inputs
    """
    # Handle NaN and inf
    inputs = np.nan_to_num(inputs, nan=0.0, posinf=clip_range[1], neginf=clip_range[0])
    
    # Clip to reasonable range
    inputs = np.clip(inputs, clip_range[0], clip_range[1])
    
    return inputs


def validate_learning_rate(lr: float, min_lr: float = 1e-8, max_lr: float = 1.0) -> float:
    """
    Validate learning rate parameter.
    
    Args:
        lr: Learning rate
        min_lr: Minimum allowed learning rate
        max_lr: Maximum allowed learning rate
        
    Returns:
        Validated learning rate
        
    Raises:
        ParameterValidationError: If learning rate is invalid
    """
    if not isinstance(lr, (int, float)):
        raise ParameterValidationError(f"Learning rate must be a number, got {type(lr)}")
    
    if lr <= 0:
        raise ParameterValidationError(f"Learning rate must be positive, got {lr}")
    
    if lr < min_lr:
        warnings.warn(f"Learning rate {lr} is very small (< {min_lr}), may cause slow convergence")
    
    if lr > max_lr:
        warnings.warn(f"Learning rate {lr} is very large (> {max_lr}), may cause instability")
        
    return float(lr)


def validate_batch_size(batch_size: int, max_batch_size: int = 10000) -> int:
    """
    Validate batch size parameter.
    
    Args:
        batch_size: Batch size
        max_batch_size: Maximum allowed batch size
        
    Returns:
        Validated batch size
        
    Raises:
        ParameterValidationError: If batch size is invalid
    """
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ParameterValidationError(f"batch_size must be positive integer, got {batch_size}")
    
    if batch_size > max_batch_size:
        warnings.warn(f"Large batch_size ({batch_size}) may cause memory issues")
    
    return batch_size


def check_gradient_health(gradients: Dict[str, np.ndarray], 
                         grad_clip_threshold: float = 10.0) -> Dict[str, Any]:
    """
    Check gradient health and detect common issues.
    
    Args:
        gradients: Dictionary of parameter gradients
        grad_clip_threshold: Threshold for gradient clipping warning
        
    Returns:
        Dictionary with gradient health metrics
    """
    health_report = {
        'healthy': True,
        'issues': [],
        'statistics': {}
    }
    
    for name, grad in gradients.items():
        grad_norm = np.linalg.norm(grad)
        grad_max = np.max(np.abs(grad))
        grad_mean = np.mean(np.abs(grad))
        
        health_report['statistics'][name] = {
            'norm': float(grad_norm),
            'max_abs': float(grad_max),
            'mean_abs': float(grad_mean)
        }
        
        # Check for gradient explosion
        if grad_norm > grad_clip_threshold:
            health_report['healthy'] = False
            health_report['issues'].append(f"Gradient explosion in {name}: norm = {grad_norm:.4f}")
        
        # Check for vanishing gradients
        if grad_norm < 1e-8:
            health_report['healthy'] = False
            health_report['issues'].append(f"Vanishing gradients in {name}: norm = {grad_norm:.2e}")
        
        # Check for NaN or inf gradients
        if np.any(np.isnan(grad)):
            health_report['healthy'] = False
            health_report['issues'].append(f"NaN gradients in {name}")
        
        if np.any(np.isinf(grad)):
            health_report['healthy'] = False
            health_report['issues'].append(f"Infinite gradients in {name}")
    
    return health_report


def validate_time_step(dt: float, min_dt: float = 1e-6, max_dt: float = 1.0) -> float:
    """
    Validate time step for numerical integration.
    
    Args:
        dt: Time step
        min_dt: Minimum allowed time step
        max_dt: Maximum allowed time step
        
    Returns:
        Validated time step
        
    Raises:
        ParameterValidationError: If time step is invalid
    """
    if not isinstance(dt, (int, float)):
        raise ParameterValidationError(f"Time step must be a number, got {type(dt)}")
    
    if dt <= 0:
        raise ParameterValidationError(f"Time step must be positive, got {dt}")
    
    if dt < min_dt:
        raise ParameterValidationError(f"Time step {dt} too small (< {min_dt}), may cause instability")
    
    if dt > max_dt:
        warnings.warn(f"Large time step {dt} (> {max_dt}) may reduce accuracy")
    
    return float(dt)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, 
               epsilon: float = 1e-8, name: str = "division") -> np.ndarray:
    """
    Perform safe division with numerical stability.
    
    Args:
        numerator: Numerator tensor
        denominator: Denominator tensor
        epsilon: Small value to add to denominator
        name: Operation name for error reporting
        
    Returns:
        Result of safe division
    """
    # Add small epsilon to prevent division by zero
    safe_denominator = denominator + epsilon * np.sign(denominator)
    
    # Handle case where denominator is exactly zero
    safe_denominator = np.where(np.abs(denominator) < epsilon, 
                               epsilon * np.ones_like(denominator), 
                               safe_denominator)
    
    result = numerator / safe_denominator
    
    # Check for any remaining numerical issues
    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
        logger.warning(f"Numerical instability detected in {name}")
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result


def validate_activation_function(activation_fn: callable, test_input: np.ndarray = None) -> bool:
    """
    Validate that activation function behaves correctly.
    
    Args:
        activation_fn: Activation function to test
        test_input: Test input (default: range from -5 to 5)
        
    Returns:
        True if activation function is valid
        
    Raises:
        ValidationError: If activation function is invalid
    """
    if test_input is None:
        test_input = np.linspace(-5, 5, 11)
    
    try:
        output = activation_fn(test_input)
    except Exception as e:
        raise ValidationError(f"Activation function failed on test input: {e}")
    
    # Check output shape
    if output.shape != test_input.shape:
        raise ValidationError(
            f"Activation function changed shape: {test_input.shape} -> {output.shape}"
        )
    
    # Check for numerical issues
    if np.any(np.isnan(output)):
        raise ValidationError("Activation function produced NaN values")
    
    if np.any(np.isinf(output)):
        raise ValidationError("Activation function produced infinite values")
    
    return True


class RobustNetworkWrapper:
    """
    Wrapper class that adds robust validation to any network model.
    """
    
    def __init__(self, network: Any, validate_inputs: bool = True, 
                 validate_outputs: bool = True, clip_gradients: bool = True):
        """
        Initialize robust wrapper.
        
        Args:
            network: Network to wrap
            validate_inputs: Whether to validate inputs
            validate_outputs: Whether to validate outputs
            clip_gradients: Whether to clip gradients
        """
        self.network = network
        self.validate_inputs = validate_inputs
        self.validate_outputs = validate_outputs
        self.clip_gradients = clip_gradients
        
        # Statistics tracking
        self.call_count = 0
        self.error_count = 0
        self.warning_count = 0
    
    def __call__(self, *args, **kwargs):
        """Forward pass with validation."""
        self.call_count += 1
        
        try:
            # Input validation
            if self.validate_inputs:
                validated_args = []
                for i, arg in enumerate(args):
                    if isinstance(arg, np.ndarray):
                        arg = validate_tensor_values(arg, f"input_{i}")
                        arg = sanitize_inputs(arg)
                    validated_args.append(arg)
                args = tuple(validated_args)
            
            # Forward pass
            result = self.network(*args, **kwargs)
            
            # Output validation
            if self.validate_outputs:
                if isinstance(result, tuple):
                    validated_result = []
                    for i, output in enumerate(result):
                        if isinstance(output, np.ndarray):
                            output = validate_tensor_values(output, f"output_{i}")
                        validated_result.append(output)
                    result = tuple(validated_result)
                elif isinstance(result, np.ndarray):
                    result = validate_tensor_values(result, "output")
            
            return result
            
        except ValidationError as e:
            self.error_count += 1
            logger.error(f"Validation error in network call {self.call_count}: {e}")
            raise
        except Exception as e:
            self.error_count += 1
            logger.error(f"Unexpected error in network call {self.call_count}: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, int]:
        """Get wrapper statistics."""
        return {
            'call_count': self.call_count,
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'success_rate': (self.call_count - self.error_count) / max(self.call_count, 1)
        }