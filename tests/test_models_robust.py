"""
Robustness tests for liquid neural network models.
"""

import pytest
import jax.numpy as jnp
from jax import random
import warnings

from src.models.liquid_network import LiquidNeuralNetwork
from src.models.continuous_rnn import ContinuousTimeRNN
from src.models.adaptive_neuron import AdaptiveNeuron, AdaptiveNeuronLayer
from src.utils.validation import ValidationError


class TestLiquidNetworkRobustness:
    """Test robustness of LiquidNeuralNetwork."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.key = random.PRNGKey(42)
    
    def test_parameter_validation(self):
        """Test parameter validation in initialization."""
        # Valid parameters
        model = LiquidNeuralNetwork(1, 10, 1, key=self.key)
        assert model.input_size == 1
        assert model.hidden_size == 10
        assert model.output_size == 1
        
        # Invalid input size
        with pytest.raises(ValidationError):
            LiquidNeuralNetwork(0, 10, 1, key=self.key)
        
        # Invalid hidden size
        with pytest.raises(ValidationError):
            LiquidNeuralNetwork(1, -1, 1, key=self.key)
        
        # Invalid tau range
        with pytest.raises(ValidationError):
            LiquidNeuralNetwork(1, 10, 1, tau_min=5.0, tau_max=1.0, key=self.key)
    
    def test_input_validation(self):
        """Test input validation in forward pass."""
        model = LiquidNeuralNetwork(2, 5, 1, key=self.key)
        
        # Valid input
        valid_input = jnp.ones((10, 2))
        outputs, states = model(valid_input)
        assert outputs.shape == (10, 1)
        assert states.shape == (10, 5)
        
        # Wrong input dimension
        with pytest.raises(ValidationError):
            model(jnp.ones((10, 3)))  # Wrong input size
        
        # Wrong number of dimensions
        with pytest.raises(ValidationError):
            model(jnp.ones((10,)))  # 1D instead of 2D
        
        # Input with NaN
        with pytest.raises(ValidationError):
            nan_input = jnp.ones((10, 2))
            nan_input = nan_input.at[0, 0].set(float('nan'))
            model(nan_input)
        
        # Input with Inf
        with pytest.raises(ValidationError):
            inf_input = jnp.ones((10, 2))
            inf_input = inf_input.at[0, 0].set(float('inf'))
            model(inf_input)
    
    def test_numerical_stability_monitoring(self):
        """Test numerical stability monitoring."""
        # Create model that might be unstable
        model = LiquidNeuralNetwork(1, 5, 1, tau_min=0.001, tau_max=0.01, key=self.key)
        
        # Normal input should be stable
        normal_input = jnp.sin(jnp.linspace(0, 2*jnp.pi, 20)).reshape(-1, 1)
        
        with warnings.catch_warnings(record=True) as w:
            outputs, states = model(normal_input, dt=0.001)  # Small dt for stability
            # Should not trigger stability warnings for normal input
        
        # Large time step might cause instability
        with warnings.catch_warnings(record=True) as w:
            outputs, states = model(normal_input, dt=1.0)  # Large dt
            # Might trigger stability warnings
    
    def test_weight_sanitization(self):
        """Test that weights are properly sanitized."""
        model = LiquidNeuralNetwork(1, 10, 1, key=self.key)
        
        # Check that all weights are finite
        assert jnp.all(jnp.isfinite(model.W_in))
        assert jnp.all(jnp.isfinite(model.W_rec))
        assert jnp.all(jnp.isfinite(model.W_out))
        assert jnp.all(jnp.isfinite(model.bias))
        assert jnp.all(jnp.isfinite(model.tau))
        
        # Check that weights are within reasonable bounds
        assert jnp.all(jnp.abs(model.W_in) <= 10.0)  # Should be sanitized
        assert jnp.all(jnp.abs(model.W_rec) <= 10.0)
        assert jnp.all(jnp.abs(model.W_out) <= 10.0)


class TestContinuousRNNRobustness:
    """Test robustness of ContinuousTimeRNN."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.key = random.PRNGKey(123)
    
    def test_integration_method_validation(self):
        """Test validation of integration parameters."""
        model = ContinuousTimeRNN(1, 5, 1, key=self.key)
        test_input = jnp.ones((10, 1))
        
        # Valid dt
        outputs, states = model(test_input, dt=0.01)
        assert outputs.shape == (10, 1)
        
        # Invalid dt (too small)
        with pytest.raises(ValidationError):
            model(test_input, dt=0.0)
        
        # Invalid dt (negative)
        with pytest.raises(ValidationError):
            model(test_input, dt=-0.01)
    
    def test_fixed_point_analysis_robustness(self):
        """Test robustness of fixed point analysis."""
        model = ContinuousTimeRNN(2, 8, 1, key=self.key)
        
        # Normal input
        normal_input = jnp.array([0.5, 0.5])
        fixed_points = model.get_fixed_points(normal_input, num_inits=5)
        assert isinstance(fixed_points, jnp.ndarray)
        
        # Extreme input
        extreme_input = jnp.array([100.0, -100.0])
        fixed_points = model.get_fixed_points(extreme_input, num_inits=3)
        # Should handle extreme inputs gracefully
        assert isinstance(fixed_points, jnp.ndarray)


class TestAdaptiveNeuronRobustness:
    """Test robustness of AdaptiveNeuron."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.key = random.PRNGKey(456)
    
    def test_parameter_adaptation_stability(self):
        """Test that parameter adaptation remains stable."""
        neuron = AdaptiveNeuron(2, key=self.key)
        
        # Simulate long sequence
        state = 0.0
        n_steps = 1000
        
        for i in range(n_steps):
            inputs = jnp.array([jnp.sin(0.1*i), jnp.cos(0.1*i)])
            adaptation_signal = 0.01 * jnp.sin(0.01*i)
            
            state, neuron = neuron.forward(state, inputs, adaptation_signal=adaptation_signal)
            
            # Check that adapted parameters remain reasonable
            info = neuron.get_adaptation_info()
            assert 0.01 <= info['tau'] <= 100.0  # Time constant bounds
            assert -10.0 <= info['threshold'] <= 10.0  # Threshold bounds
            assert 0.01 <= info['sensitivity'] <= 10.0  # Sensitivity bounds
    
    def test_extreme_inputs(self):
        """Test behavior with extreme inputs."""
        neuron = AdaptiveNeuron(1, key=self.key)
        state = 0.0
        
        # Very large input
        large_input = jnp.array([1000.0])
        state, neuron = neuron.forward(state, large_input)
        assert jnp.isfinite(state)
        
        # Very small input
        small_input = jnp.array([1e-10])
        state, neuron = neuron.forward(state, small_input)
        assert jnp.isfinite(state)
        
        # Zero input
        zero_input = jnp.array([0.0])
        state, neuron = neuron.forward(state, zero_input)
        assert jnp.isfinite(state)


class TestAdaptiveLayerRobustness:
    """Test robustness of AdaptiveNeuronLayer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.key = random.PRNGKey(789)
    
    def test_layer_evolution_stability(self):
        """Test that layer evolution remains stable."""
        layer = AdaptiveNeuronLayer(1, num_neurons=5, key=self.key)
        states = jnp.zeros(5)
        
        # Evolve for many steps
        for i in range(200):
            inputs = jnp.array([jnp.sin(0.1*i)])
            adaptation_signals = 0.01 * jnp.ones(5)
            
            states, layer = layer(states, inputs, adaptation_signals=adaptation_signals)
            
            # Check that all states remain finite
            assert jnp.all(jnp.isfinite(states))
            
            # Check layer adaptation info
            info = layer.get_layer_info()
            for param, values in info.items():
                assert jnp.all(jnp.isfinite(values))
    
    def test_lateral_connections_stability(self):
        """Test stability of lateral connections."""
        layer = AdaptiveNeuronLayer(2, num_neurons=8, key=self.key)
        
        # Check that lateral weights are reasonable
        assert jnp.all(jnp.isfinite(layer.lateral_weights))
        assert jnp.all(layer.lateral_weights.diagonal() == 0.0)  # No self-connections
        
        # Test with varying inputs
        states = jnp.ones(8) * 0.1
        
        for amplitude in [0.1, 1.0, 5.0]:
            inputs = jnp.array([amplitude, -amplitude])
            states, layer = layer(states, inputs)
            
            # States should remain bounded
            assert jnp.all(jnp.abs(states) < 100.0)