"""
Tests for validation utilities.
"""

import pytest
import jax.numpy as jnp
from jax import random

from src.utils.validation import (
    ValidationError, validate_array_shape, validate_positive_scalar,
    validate_array_finite, validate_sequence_data, validate_model_parameters,
    validate_integration_parameters, validate_training_parameters,
    validate_prng_key, sanitize_weights, check_numerical_stability
)


class TestValidation:
    """Test validation functions."""
    
    def test_validate_array_shape(self):
        """Test array shape validation."""
        arr = jnp.ones((3, 4))
        
        # Valid shape
        validate_array_shape(arr, (3, 4), "test_array")
        
        # Invalid shape
        with pytest.raises(ValidationError):
            validate_array_shape(arr, (3, 5), "test_array")
        
        # Wrong number of dimensions
        with pytest.raises(ValidationError):
            validate_array_shape(arr, (3,), "test_array")
    
    def test_validate_positive_scalar(self):
        """Test positive scalar validation."""
        # Valid values
        validate_positive_scalar(1.0)
        validate_positive_scalar(0.001)
        
        # Invalid values
        with pytest.raises(ValidationError):
            validate_positive_scalar(-1.0)
        
        with pytest.raises(ValidationError):
            validate_positive_scalar(0.0)
        
        with pytest.raises(ValidationError):
            validate_positive_scalar(float('nan'))
        
        with pytest.raises(ValidationError):
            validate_positive_scalar(float('inf'))
    
    def test_validate_array_finite(self):
        """Test finite array validation."""
        # Valid array
        arr = jnp.array([1.0, 2.0, 3.0])
        validate_array_finite(arr)
        
        # Array with NaN
        arr_nan = jnp.array([1.0, float('nan'), 3.0])
        with pytest.raises(ValidationError):
            validate_array_finite(arr_nan)
        
        # Array with Inf
        arr_inf = jnp.array([1.0, float('inf'), 3.0])
        with pytest.raises(ValidationError):
            validate_array_finite(arr_inf)
    
    def test_validate_sequence_data(self):
        """Test sequence data validation."""
        # Valid 2D sequence
        seq_2d = jnp.ones((50, 3))
        validate_sequence_data(seq_2d)
        
        # Valid 3D batch
        seq_3d = jnp.ones((10, 50, 3))
        validate_sequence_data(seq_3d)
        
        # Too few dimensions
        with pytest.raises(ValidationError):
            validate_sequence_data(jnp.array([1, 2, 3]))
        
        # Too many dimensions
        with pytest.raises(ValidationError):
            validate_sequence_data(jnp.ones((2, 3, 4, 5)))
        
        # Sequence too short
        with pytest.raises(ValidationError):
            validate_sequence_data(jnp.ones((0, 3)), min_seq_length=1)
    
    def test_validate_model_parameters(self):
        """Test model parameter validation."""
        # Valid parameters
        validate_model_parameters(10, 20, 5)
        validate_model_parameters(1, 100, 1, tau_min=0.1, tau_max=5.0)
        
        # Invalid sizes
        with pytest.raises(ValidationError):
            validate_model_parameters(0, 20, 5)
        
        with pytest.raises(ValidationError):
            validate_model_parameters(10, -1, 5)
        
        # Invalid tau range
        with pytest.raises(ValidationError):
            validate_model_parameters(10, 20, 5, tau_min=5.0, tau_max=1.0)
    
    def test_validate_integration_parameters(self):
        """Test integration parameter validation."""
        # Valid parameters
        validate_integration_parameters(0.01, "euler")
        validate_integration_parameters(0.001, "rk4")
        
        # Invalid dt
        with pytest.raises(ValidationError):
            validate_integration_parameters(-0.01)
        
        with pytest.raises(ValidationError):
            validate_integration_parameters(0.0)
        
        # Invalid method
        with pytest.raises(ValidationError):
            validate_integration_parameters(0.01, "invalid_method")
    
    def test_validate_training_parameters(self):
        """Test training parameter validation."""
        # Valid parameters
        validate_training_parameters(0.001)
        validate_training_parameters(0.01, batch_size=32, epochs=100)
        
        # Invalid learning rate
        with pytest.raises(ValidationError):
            validate_training_parameters(-0.001)
        
        with pytest.raises(ValidationError):
            validate_training_parameters(0.0)
        
        # Invalid batch size
        with pytest.raises(ValidationError):
            validate_training_parameters(0.001, batch_size=0)
        
        # Invalid epochs
        with pytest.raises(ValidationError):
            validate_training_parameters(0.001, epochs=-1)
    
    def test_validate_prng_key(self):
        """Test PRNG key validation."""
        # Valid key
        key = random.PRNGKey(42)
        validated_key = validate_prng_key(key)
        assert jnp.array_equal(key, validated_key)
        
        # None key (should return default)
        default_key = validate_prng_key(None)
        assert default_key.shape == (2,)
        assert default_key.dtype == jnp.uint32
        
        # Invalid key
        with pytest.raises(ValidationError):
            validate_prng_key(jnp.array([1, 2, 3]))
    
    def test_sanitize_weights(self):
        """Test weight sanitization."""
        # Normal weights
        weights = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        sanitized = sanitize_weights(weights, max_norm=5.0)
        assert jnp.allclose(weights, sanitized)
        
        # Weights with extreme values
        extreme_weights = jnp.array([[100.0, -200.0], [3.0, 4.0]])
        sanitized = sanitize_weights(extreme_weights, max_norm=5.0)
        assert jnp.all(jnp.abs(sanitized) <= 5.0)
        
        # Weights with NaN/Inf
        nan_weights = jnp.array([[1.0, float('nan')], [float('inf'), 4.0]])
        sanitized = sanitize_weights(nan_weights)
        assert jnp.all(jnp.isfinite(sanitized))
    
    def test_check_numerical_stability(self):
        """Test numerical stability checking."""
        # Stable outputs
        outputs = jnp.ones((10, 2))
        states = jnp.ones((10, 5))
        
        stability = check_numerical_stability(outputs, states)
        assert stability["overall_stable"]
        assert stability["outputs_stable"]
        assert stability["states_stable"]
        assert stability["outputs_finite"]
        assert stability["states_finite"]
        
        # Unstable outputs (large values)
        unstable_outputs = jnp.ones((10, 2)) * 1e7
        stability = check_numerical_stability(unstable_outputs, states)
        assert not stability["outputs_stable"]
        assert not stability["overall_stable"]
        
        # NaN outputs
        nan_outputs = jnp.array([[1.0, float('nan')]] * 10)
        stability = check_numerical_stability(nan_outputs, states)
        assert not stability["outputs_finite"]
        assert not stability["overall_stable"]