"""
Comprehensive error handling and validation for liquid neural networks.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
import traceback
from functools import wraps
import time


class LiquidNetworkError(Exception):
    """Base exception for liquid neural network errors."""
    pass


class InvalidInputError(LiquidNetworkError):
    """Raised when input data is invalid."""
    pass


class ModelConfigurationError(LiquidNetworkError):
    """Raised when model configuration is invalid."""
    pass


class TrainingError(LiquidNetworkError):
    """Raised when training encounters issues."""
    pass


class NumericalInstabilityError(LiquidNetworkError):
    """Raised when numerical instability is detected."""
    pass


def validate_input_shapes(inputs: jnp.ndarray, targets: jnp.ndarray) -> None:
    """Validate input and target shapes."""
    if inputs.ndim < 2:
        raise InvalidInputError(f"Input must be at least 2D, got {inputs.ndim}D")
    
    if targets.ndim < 2:
        raise InvalidInputError(f"Targets must be at least 2D, got {targets.ndim}D")
    
    if inputs.shape[0] != targets.shape[0]:
        raise InvalidInputError(
            f"Batch size mismatch: inputs {inputs.shape[0]}, targets {targets.shape[0]}"
        )
    
    if inputs.ndim == 3 and targets.ndim == 3:
        if inputs.shape[1] != targets.shape[1]:
            raise InvalidInputError(
                f"Sequence length mismatch: inputs {inputs.shape[1]}, targets {targets.shape[1]}"
            )


def validate_model_config(config: Dict[str, Any]) -> None:
    """Validate model configuration."""
    required_keys = ['input_dim', 'hidden_dims', 'output_dim']
    
    for key in required_keys:
        if key not in config:
            raise ModelConfigurationError(f"Missing required config key: {key}")
    
    if config['input_dim'] <= 0:
        raise ModelConfigurationError(f"input_dim must be positive, got {config['input_dim']}")
    
    if config['output_dim'] <= 0:
        raise ModelConfigurationError(f"output_dim must be positive, got {config['output_dim']}")
    
    if not isinstance(config['hidden_dims'], (list, tuple)):
        raise ModelConfigurationError("hidden_dims must be a list or tuple")
    
    if len(config['hidden_dims']) == 0:
        raise ModelConfigurationError("hidden_dims cannot be empty")
    
    for i, dim in enumerate(config['hidden_dims']):
        if dim <= 0:
            raise ModelConfigurationError(f"hidden_dims[{i}] must be positive, got {dim}")


def check_numerical_stability(array: jnp.ndarray, name: str = "array") -> bool:
    """Check if array contains NaN or infinite values."""
    if jnp.any(jnp.isnan(array)):
        raise NumericalInstabilityError(f"{name} contains NaN values")
    
    if jnp.any(jnp.isinf(array)):
        raise NumericalInstabilityError(f"{name} contains infinite values")
    
    return True


def safe_forward_pass(model, inputs: jnp.ndarray) -> jnp.ndarray:
    """Perform forward pass with error checking."""
    try:
        check_numerical_stability(inputs, "inputs")
        outputs = model(inputs)
        check_numerical_stability(outputs, "outputs")
        return outputs
    except Exception as e:
        raise TrainingError(f"Forward pass failed: {str(e)}")


def robust_loss_computation(loss_fn, predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Compute loss with numerical stability checks."""
    try:
        check_numerical_stability(predictions, "predictions")
        check_numerical_stability(targets, "targets")
        
        loss = loss_fn(predictions, targets)
        check_numerical_stability(loss, "loss")
        
        # Check for reasonable loss values
        if loss > 1e6:
            logging.warning(f"Very large loss detected: {loss}")
        
        return loss
    except Exception as e:
        raise TrainingError(f"Loss computation failed: {str(e)}")


def gradient_clipping(grads, max_norm: float = 1.0):
    """Apply gradient clipping to prevent exploding gradients."""
    # Compute global gradient norm
    grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads)))
    
    # Clip if necessary
    if grad_norm > max_norm:
        scale = max_norm / grad_norm
        grads = jax.tree.map(lambda g: g * scale, grads)
        logging.info(f"Gradients clipped: norm {grad_norm:.4f} -> {max_norm}")
    
    return grads


def retry_on_failure(max_retries: int = 3, delay: float = 0.1):
    """Decorator to retry function calls on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logging.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                        time.sleep(delay)
                    else:
                        logging.error(f"All {max_retries} attempts failed")
            
            raise last_exception
        return wrapper
    return decorator


def monitor_training_progress(metrics: Dict[str, float], epoch: int) -> Dict[str, Any]:
    """Monitor training progress and detect potential issues."""
    warnings = []
    
    # Check for NaN/inf losses
    if np.isnan(metrics.get('loss', 0)) or np.isinf(metrics.get('loss', 0)):
        warnings.append("NaN or infinite loss detected")
    
    # Check for very large gradients
    grad_norm = metrics.get('gradient_norm', 0)
    if grad_norm > 10.0:
        warnings.append(f"Large gradient norm: {grad_norm:.4f}")
    
    # Check for very small gradients (potential vanishing gradients)
    if grad_norm < 1e-7:
        warnings.append(f"Very small gradient norm: {grad_norm:.2e}")
    
    # Check loss trends (requires history)
    loss = metrics.get('loss', float('inf'))
    if loss > 100:
        warnings.append(f"Large loss value: {loss:.4f}")
    
    return {
        'epoch': epoch,
        'warnings': warnings,
        'healthy': len(warnings) == 0,
        'metrics': metrics
    }


class TrainingMonitor:
    """Monitor training progress and detect issues."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.history = []
        
    def update(self, loss: float, epoch: int) -> Dict[str, Any]:
        """Update monitor with new loss value."""
        self.history.append(loss)
        
        improved = False
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.wait = 0
            improved = True
        else:
            self.wait += 1
        
        # Check for plateau
        plateau = self.wait >= self.patience
        
        # Check for divergence
        recent_losses = self.history[-5:] if len(self.history) >= 5 else self.history
        diverging = len(recent_losses) >= 3 and all(
            recent_losses[i] > recent_losses[i-1] for i in range(1, len(recent_losses))
        )
        
        return {
            'improved': improved,
            'plateau': plateau,
            'diverging': diverging,
            'best_loss': self.best_loss,
            'wait': self.wait,
            'current_loss': loss
        }


class InputValidator:
    """Validate inputs for different model components."""
    
    @staticmethod
    def validate_sequence_data(data: jnp.ndarray, min_length: int = 1) -> None:
        """Validate sequence data format."""
        if data.ndim != 3:
            raise InvalidInputError(f"Sequence data must be 3D (batch, time, features), got {data.ndim}D")
        
        batch_size, seq_len, features = data.shape
        
        if seq_len < min_length:
            raise InvalidInputError(f"Sequence length {seq_len} < minimum {min_length}")
        
        if features == 0:
            raise InvalidInputError("Feature dimension cannot be 0")
        
        if batch_size == 0:
            raise InvalidInputError("Batch size cannot be 0")
    
    @staticmethod
    def validate_time_constants(tau: jnp.ndarray, min_tau: float = 0.01, max_tau: float = 10.0) -> None:
        """Validate time constants for stability."""
        if jnp.any(tau <= 0):
            raise ModelConfigurationError("Time constants must be positive")
        
        if jnp.any(tau < min_tau):
            raise ModelConfigurationError(f"Time constants too small (< {min_tau})")
        
        if jnp.any(tau > max_tau):
            raise ModelConfigurationError(f"Time constants too large (> {max_tau})")
    
    @staticmethod
    def validate_learning_rate(lr: float) -> None:
        """Validate learning rate."""
        if lr <= 0:
            raise ModelConfigurationError(f"Learning rate must be positive, got {lr}")
        
        if lr > 1.0:
            logging.warning(f"Large learning rate detected: {lr}")
        
        if lr < 1e-6:
            logging.warning(f"Very small learning rate detected: {lr}")


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )


def safe_jit_compile(func, static_argnums=None):
    """Safely JIT compile functions with error handling."""
    try:
        if static_argnums is not None:
            return jax.jit(func, static_argnums=static_argnums)
        else:
            return jax.jit(func)
    except Exception as e:
        logging.warning(f"JIT compilation failed: {e}. Using non-compiled version.")
        return func


def memory_efficient_batch_processing(process_fn, data: jnp.ndarray, batch_size: int = 32):
    """Process large datasets in batches to manage memory."""
    if data.shape[0] <= batch_size:
        return process_fn(data)
    
    results = []
    for i in range(0, data.shape[0], batch_size):
        batch = data[i:i + batch_size]
        batch_result = process_fn(batch)
        results.append(batch_result)
    
    return jnp.concatenate(results, axis=0)


class PerformanceProfiler:
    """Profile performance of different components."""
    
    def __init__(self):
        self.timings = {}
        self.call_counts = {}
    
    def time_function(self, name: str):
        """Decorator to time function execution."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                duration = end_time - start_time
                if name not in self.timings:
                    self.timings[name] = []
                    self.call_counts[name] = 0
                
                self.timings[name].append(duration)
                self.call_counts[name] += 1
                
                return result
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics."""
        stats = {}
        for name, times in self.timings.items():
            stats[name] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'total_time': np.sum(times),
                'call_count': self.call_counts[name]
            }
        return stats
    
    def print_stats(self):
        """Print performance statistics."""
        stats = self.get_stats()
        print("\nPerformance Statistics:")
        print("-" * 60)
        for name, stat in stats.items():
            print(f"{name}:")
            print(f"  Calls: {stat['call_count']}")
            print(f"  Mean: {stat['mean_time']:.4f}s")
            print(f"  Total: {stat['total_time']:.4f}s")
            print(f"  Min/Max: {stat['min_time']:.4f}s / {stat['max_time']:.4f}s")
            print()


# Global profiler instance
profiler = PerformanceProfiler()