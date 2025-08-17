"""
Comprehensive tests for model architectures.
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.liquid_neural_network import LiquidNeuralNetwork
from src.models.continuous_time_rnn import ContinuousTimeRNN
from src.models.adaptive_neuron import AdaptiveNeuron, NeuronNetwork


class TestLiquidNeuralNetwork:
    """Test suite for LiquidNeuralNetwork."""
    
    @pytest.fixture
    def basic_model(self):
        """Create a basic model for testing."""
        key = random.PRNGKey(42)
        return LiquidNeuralNetwork(
            input_size=2,
            hidden_size=8,
            output_size=1,
            key=key
        )
    
    def test_initialization(self, basic_model):
        """Test proper model initialization."""
        assert basic_model.input_size == 2
        assert basic_model.hidden_size == 8
        assert basic_model.output_size == 1
        
        # Check parameter shapes
        assert basic_model.params['W_in'].shape == (8, 2)
        assert basic_model.params['W_rec'].shape == (8, 8)
        assert basic_model.params['W_out'].shape == (1, 8)
        assert basic_model.params['tau'].shape == (8,)
        assert basic_model.params['b_rec'].shape == (8,)
        assert basic_model.params['b_out'].shape == (1,)
        
        # Check parameter finite values
        for param_name, param_value in basic_model.params.items():
            assert jnp.all(jnp.isfinite(param_value)), f"Parameter {param_name} contains non-finite values"
    
    def test_forward_pass_shapes(self, basic_model):
        """Test forward pass produces correct output shapes."""
        seq_len = 10
        inputs = random.normal(random.PRNGKey(123), (seq_len, 2))
        
        outputs, states = basic_model.forward(inputs)
        
        assert outputs.shape == (seq_len, 1)
        assert states.shape == (seq_len, 8)
        
        # Check outputs are finite
        assert jnp.all(jnp.isfinite(outputs))
        assert jnp.all(jnp.isfinite(states))
    
    def test_forward_pass_different_lengths(self, basic_model):
        """Test forward pass with different sequence lengths."""
        for seq_len in [1, 5, 20, 100]:
            inputs = random.normal(random.PRNGKey(seq_len), (seq_len, 2))
            outputs, states = basic_model.forward(inputs)
            
            assert outputs.shape == (seq_len, 1)
            assert states.shape == (seq_len, 8)
    
    def test_initial_state(self, basic_model):
        """Test forward pass with custom initial state."""
        seq_len = 5
        inputs = random.normal(random.PRNGKey(456), (seq_len, 2))
        initial_state = random.normal(random.PRNGKey(789), (8,))
        
        outputs, states = basic_model.forward(inputs, initial_state=initial_state)
        
        # First state should be influenced by initial state
        assert not jnp.allclose(states[0], jnp.zeros(8))
    
    def test_time_constant_properties(self, basic_model):
        """Test time constant properties."""
        tau = basic_model.get_time_constants()
        
        # Should be positive
        assert jnp.all(tau > 0)
        
        # Should be reasonable values
        assert jnp.all(tau < 100)  # Not too large
        assert jnp.all(tau > 0.001)  # Not too small
    
    def test_parameter_update(self, basic_model):
        """Test parameter update functionality."""
        original_params = basic_model.params.copy()
        
        # Modify parameters
        new_params = original_params.copy()
        new_params['W_in'] = new_params['W_in'] * 0.5
        
        basic_model.update_params(new_params)
        
        # Check parameters were updated
        assert not jnp.allclose(basic_model.params['W_in'], original_params['W_in'])
        assert jnp.allclose(basic_model.params['W_in'], new_params['W_in'])
    
    def test_different_configurations(self):
        """Test different model configurations."""
        key = random.PRNGKey(999)
        
        configs = [
            {'input_size': 1, 'hidden_size': 4, 'output_size': 1},
            {'input_size': 5, 'hidden_size': 16, 'output_size': 3},
            {'input_size': 10, 'hidden_size': 32, 'output_size': 10},
        ]
        
        for config in configs:
            model = LiquidNeuralNetwork(key=key, **config)
            
            # Test forward pass
            inputs = random.normal(key, (10, config['input_size']))
            outputs, states = model.forward(inputs)
            
            assert outputs.shape == (10, config['output_size'])
            assert states.shape == (10, config['hidden_size'])
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        key = random.PRNGKey(111)
        
        # Very small network
        model = LiquidNeuralNetwork(
            input_size=1, hidden_size=1, output_size=1, key=key
        )
        inputs = jnp.array([[1.0]])
        outputs, states = model.forward(inputs)
        
        assert outputs.shape == (1, 1)
        assert states.shape == (1, 1)
        
        # Empty input should raise appropriate error
        with pytest.raises((ValueError, TypeError)):
            empty_inputs = jnp.array([]).reshape(0, 1)
            model.forward(empty_inputs)


class TestContinuousTimeRNN:
    """Test suite for ContinuousTimeRNN."""
    
    @pytest.fixture
    def basic_ct_rnn(self):
        """Create a basic CT-RNN for testing."""
        key = random.PRNGKey(42)
        return ContinuousTimeRNN(
            input_size=3,
            hidden_size=6,
            output_size=2,
            key=key
        )
    
    def test_initialization(self, basic_ct_rnn):
        """Test proper CT-RNN initialization."""
        assert basic_ct_rnn.input_size == 3
        assert basic_ct_rnn.hidden_size == 6
        assert basic_ct_rnn.output_size == 2
        
        # Check parameter shapes
        assert basic_ct_rnn.params['W_in'].shape == (6, 3)
        assert basic_ct_rnn.params['W_hh'].shape == (6, 6)
        assert basic_ct_rnn.params['W_out'].shape == (2, 6)
        assert basic_ct_rnn.params['alpha'].shape == (6,)
        
        # Check parameters are finite
        for param_name, param_value in basic_ct_rnn.params.items():
            assert jnp.all(jnp.isfinite(param_value)), f"Parameter {param_name} contains non-finite values"
    
    def test_different_solvers(self):
        """Test different ODE solvers."""
        key = random.PRNGKey(222)
        
        for solver in ['euler', 'rk4']:
            model = ContinuousTimeRNN(
                input_size=2, hidden_size=4, output_size=1,
                solver=solver, key=key
            )
            
            inputs = random.normal(key, (5, 2))
            outputs, states = model.forward(inputs, dt=0.1)
            
            assert outputs.shape == (5, 1)
            assert states.shape == (5, 4)
            assert jnp.all(jnp.isfinite(outputs))
            assert jnp.all(jnp.isfinite(states))
    
    def test_different_activations(self):
        """Test different activation functions."""
        key = random.PRNGKey(333)
        
        activations = ['tanh', 'relu', 'sigmoid', 'swish', 'gelu']
        
        for activation in activations:
            model = ContinuousTimeRNN(
                input_size=2, hidden_size=4, output_size=1,
                activation=activation, key=key
            )
            
            inputs = random.normal(key, (3, 2))
            outputs, states = model.forward(inputs)
            
            assert outputs.shape == (3, 1)
            assert states.shape == (3, 4)
    
    def test_integration_steps(self, basic_ct_rnn):
        """Test different integration step configurations."""
        inputs = random.normal(random.PRNGKey(444), (5, 3))
        
        # Test different dt values
        for dt in [0.01, 0.1, 0.5]:
            outputs, states = basic_ct_rnn.forward(inputs, dt=dt)
            assert jnp.all(jnp.isfinite(outputs))
            assert jnp.all(jnp.isfinite(states))
        
        # Test different number of integration steps
        for n_steps in [1, 2, 5, 10]:
            outputs, states = basic_ct_rnn.forward(inputs, dt=0.1, n_steps=n_steps)
            assert jnp.all(jnp.isfinite(outputs))
            assert jnp.all(jnp.isfinite(states))
    
    def test_dynamics_info(self, basic_ct_rnn):
        """Test dynamics information extraction."""
        info = basic_ct_rnn.get_dynamics_info()
        
        assert 'alpha_mean' in info
        assert 'alpha_std' in info
        assert 'solver' in info
        assert 'activation' in info
        
        assert jnp.isfinite(info['alpha_mean'])
        assert jnp.isfinite(info['alpha_std'])
        assert info['solver'] in ['euler', 'rk4']


class TestAdaptiveNeuron:
    """Test suite for AdaptiveNeuron and AdaptiveNeuronLayer."""
    
    @pytest.fixture
    def basic_neuron(self):
        """Create a basic adaptive neuron."""
        key = random.PRNGKey(42)
        return AdaptiveNeuron(
            input_size=3,
            tau_init=1.0,
            threshold_init=0.5,
            key=key
        )
    
    def test_neuron_initialization(self, basic_neuron):
        """Test neuron initialization."""
        assert basic_neuron.input_size == 3
        
        # Check parameter shapes
        assert basic_neuron.params['w'].shape == (3,)
        assert basic_neuron.params['tau'].shape == ()
        assert basic_neuron.params['threshold'].shape == ()
        
        # Check initial state
        assert basic_neuron.params['membrane_potential'].shape == ()
        assert basic_neuron.params['adaptation'].shape == ()
    
    def test_neuron_step(self, basic_neuron):
        """Test single neuron time step."""
        inputs = jnp.array([0.1, -0.2, 0.3])
        
        output, new_params = basic_neuron.step(inputs, dt=0.01)
        
        # Check output properties
        assert output.shape == ()
        assert jnp.isfinite(output)
        
        # Check parameter updates
        assert 'membrane_potential' in new_params
        assert 'adaptation' in new_params
        assert jnp.isfinite(new_params['membrane_potential'])
        assert jnp.isfinite(new_params['adaptation'])
    
    def test_neuron_reset(self, basic_neuron):
        """Test neuron state reset."""
        # Run some steps to change state
        inputs = jnp.array([1.0, -0.5, 0.8])
        for _ in range(5):
            output, new_params = basic_neuron.step(inputs)
            basic_neuron.update_params(new_params)
        
        # Reset state
        basic_neuron.reset_state()
        
        # Check state is reset
        assert basic_neuron.params['membrane_potential'] == 0.0
        assert basic_neuron.params['adaptation'] == 0.0
    
    def test_adaptive_neuron_layer(self):
        """Test adaptive neuron layer."""
        key = random.PRNGKey(555)
        
        layer = AdaptiveNeuronLayer(
            input_size=4,
            n_neurons=8,
            tau_range=(0.5, 2.0),
            threshold_range=(0.2, 0.8),
            key=key
        )
        
        assert layer.input_size == 4
        assert layer.n_neurons == 8
        assert len(layer.neurons) == 8
        
        # Test forward pass
        inputs = random.normal(key, (4,))
        outputs = layer.forward(inputs, dt=0.01)
        
        assert outputs.shape == (8,)
        assert jnp.all(jnp.isfinite(outputs))
    
    def test_layer_population_stats(self):
        """Test population statistics."""
        key = random.PRNGKey(666)
        
        layer = AdaptiveNeuronLayer(
            input_size=2,
            n_neurons=10,
            key=key
        )
        
        stats = layer.get_population_stats()
        
        required_keys = ['tau_mean', 'tau_std', 'threshold_mean', 'threshold_std', 
                        'membrane_potential_mean', 'membrane_potential_std']
        
        for key in required_keys:
            assert key in stats
            assert jnp.isfinite(stats[key])
    
    def test_layer_reset(self):
        """Test layer state reset."""
        key = random.PRNGKey(777)
        
        layer = AdaptiveNeuronLayer(
            input_size=2,
            n_neurons=5,
            key=key
        )
        
        # Run some steps
        inputs = random.normal(key, (2,))
        for _ in range(10):
            layer.forward(inputs)
        
        # Reset all states
        layer.reset_all_states()
        
        # Check all neurons are reset
        for neuron in layer.neurons:
            assert neuron.params['membrane_potential'] == 0.0
            assert neuron.params['adaptation'] == 0.0


def run_stress_tests():
    """Run stress tests with large inputs and extreme parameters."""
    print("Running stress tests...")
    
    key = random.PRNGKey(42)
    
    # Test with large sequences
    model = LiquidNeuralNetwork(
        input_size=10, hidden_size=50, output_size=5, key=key
    )
    
    large_inputs = random.normal(key, (1000, 10))
    outputs, states = model.forward(large_inputs)
    
    assert outputs.shape == (1000, 5)
    assert jnp.all(jnp.isfinite(outputs))
    
    # Test with extreme parameter values
    extreme_model = LiquidNeuralNetwork(
        input_size=1, hidden_size=10, output_size=1,
        time_constant_init=0.001,  # Very fast dynamics
        leak_rate=0.9,  # High leak rate
        key=key
    )
    
    inputs = random.normal(key, (100, 1)) * 10  # Large input values
    outputs, states = extreme_model.forward(inputs, dt=0.001)
    
    # Should handle extreme cases gracefully
    assert jnp.all(jnp.isfinite(outputs))
    
    print("Stress tests passed!")


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
    
    # Run additional stress tests
    run_stress_tests()