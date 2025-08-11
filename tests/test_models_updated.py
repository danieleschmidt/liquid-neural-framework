"""
Updated comprehensive tests for model architectures.
"""

import pytest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    import numpy as jnp

from models.liquid_neural_network import LiquidNeuralNetwork
from models.continuous_time_rnn import ContinuousTimeRNN
from models.adaptive_neuron import AdaptiveNeuron, AdaptiveNeuronLayer

@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
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
        assert basic_model.W_in.shape == (8, 2)
        assert basic_model.W_rec.shape == (8, 8)
        assert basic_model.W_out.shape == (1, 8)
        assert basic_model.tau.shape == (8,)
        assert basic_model.b_rec.shape == (8,)
        assert basic_model.b_out.shape == (1,)
        
        # Check parameter finite values
        assert jnp.all(jnp.isfinite(basic_model.W_in))
        assert jnp.all(jnp.isfinite(basic_model.W_rec))
        assert jnp.all(jnp.isfinite(basic_model.W_out))
        assert jnp.all(basic_model.tau > 0)
    
    def test_forward_pass(self, basic_model):
        """Test forward pass functionality."""
        batch_size = 3
        inputs = jnp.ones((batch_size, 2))
        hidden_state = basic_model.init_hidden_state(batch_size)
        
        output, new_hidden = basic_model(inputs, hidden_state)
        
        # Check output shapes
        assert output.shape == (batch_size, 1)
        assert new_hidden.shape == (batch_size, 8)
        
        # Check output is finite
        assert jnp.all(jnp.isfinite(output))
        assert jnp.all(jnp.isfinite(new_hidden))
    
    def test_sequence_processing(self, basic_model):
        """Test sequence processing."""
        seq_len, batch_size = 5, 2
        input_sequence = jnp.ones((seq_len, batch_size, 2))
        
        outputs, hidden_states = basic_model.forward_sequence(input_sequence)
        
        assert outputs.shape == (seq_len, batch_size, 1)
        assert hidden_states.shape == (seq_len, batch_size, 8)
        assert jnp.all(jnp.isfinite(outputs))
        assert jnp.all(jnp.isfinite(hidden_states))
    
    def test_stability_measure(self, basic_model):
        """Test stability analysis."""
        stability = basic_model.stability_measure()
        assert isinstance(stability, float)
        assert stability >= 0
        assert jnp.isfinite(stability)
    
    def test_different_activations(self):
        """Test different activation functions."""
        key = random.PRNGKey(42)
        activations = ['tanh', 'relu', 'sigmoid', 'swish', 'gelu']
        
        for activation in activations:
            model = LiquidNeuralNetwork(
                input_size=2, hidden_size=4, output_size=1,
                activation=activation, key=key
            )
            
            inputs = jnp.ones((1, 2))
            hidden = model.init_hidden_state(1)
            output, _ = model(inputs, hidden)
            
            assert jnp.all(jnp.isfinite(output))
    
    def test_custom_tau_values(self):
        """Test custom time constants."""
        key = random.PRNGKey(42)
        custom_tau = jnp.array([0.5, 1.0, 2.0, 0.1])
        
        model = LiquidNeuralNetwork(
            input_size=2, hidden_size=4, output_size=1,
            tau_init=custom_tau, key=key
        )
        
        assert jnp.allclose(model.tau, custom_tau)
    
    def test_input_validation(self, basic_model):
        """Test input validation."""
        with pytest.raises(ValueError):
            # Wrong input dimensions
            bad_inputs = jnp.ones((3, 5))  # Wrong input size
            hidden = basic_model.init_hidden_state(3)
            basic_model(bad_inputs, hidden)
        
        with pytest.raises(ValueError):
            # Mismatched batch sizes
            inputs = jnp.ones((3, 2))
            hidden = basic_model.init_hidden_state(5)  # Different batch size
            basic_model(inputs, hidden)
        
        with pytest.raises(ValueError):
            # Negative dt
            inputs = jnp.ones((1, 2))
            hidden = basic_model.init_hidden_state(1)
            basic_model(inputs, hidden, dt=-0.1)


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestContinuousTimeRNN:
    """Test suite for ContinuousTimeRNN."""
    
    @pytest.fixture
    def basic_rnn(self):
        """Create a basic RNN for testing."""
        key = random.PRNGKey(42)
        return ContinuousTimeRNN(
            input_size=3,
            hidden_size=6,
            output_size=2,
            key=key
        )
    
    def test_initialization(self, basic_rnn):
        """Test RNN initialization."""
        assert basic_rnn.input_size == 3
        assert basic_rnn.hidden_size == 6
        assert basic_rnn.output_size == 2
        assert basic_rnn.tau > 0
    
    def test_ode_solvers(self):
        """Test different ODE solvers."""
        key = random.PRNGKey(42)
        solvers = ['euler', 'rk4']
        
        for solver in solvers:
            rnn = ContinuousTimeRNN(
                input_size=2, hidden_size=4, output_size=1,
                solver=solver, key=key
            )
            
            inputs = jnp.ones((1, 2))
            hidden = rnn.init_hidden_state(1)
            output, new_hidden = rnn(inputs, hidden)
            
            assert output.shape == (1, 1)
            assert new_hidden.shape == (1, 4)
            assert jnp.all(jnp.isfinite(output))
            assert jnp.all(jnp.isfinite(new_hidden))
    
    def test_dynamics_computation(self, basic_rnn):
        """Test dynamics computation."""
        h = jnp.ones((1, 6))
        x = jnp.ones((1, 3))
        
        dhdt = basic_rnn.dynamics(h, x)
        
        assert dhdt.shape == (1, 6)
        assert jnp.all(jnp.isfinite(dhdt))
    
    def test_integration_methods(self, basic_rnn):
        """Test different integration methods."""
        h = jnp.ones((1, 6))
        x = jnp.ones((1, 3))
        dt = 0.1
        
        h_euler = basic_rnn.euler_step(h, x, dt)
        h_rk4 = basic_rnn.rk4_step(h, x, dt)
        
        assert h_euler.shape == (1, 6)
        assert h_rk4.shape == (1, 6)
        assert jnp.all(jnp.isfinite(h_euler))
        assert jnp.all(jnp.isfinite(h_rk4))


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestAdaptiveNeuron:
    """Test suite for AdaptiveNeuron."""
    
    @pytest.fixture
    def basic_neuron(self):
        """Create a basic adaptive neuron."""
        key = random.PRNGKey(42)
        return AdaptiveNeuron(input_size=4, key=key)
    
    def test_initialization(self, basic_neuron):
        """Test neuron initialization."""
        assert basic_neuron.input_size == 4
        assert basic_neuron.v_rest < basic_neuron.v_thresh_init
        assert basic_neuron.tau_m > 0
        assert basic_neuron.tau_adapt > 0
    
    def test_state_initialization(self, basic_neuron):
        """Test state initialization."""
        state = basic_neuron.init_state()
        
        assert 'v_membrane' in state
        assert 'v_thresh' in state
        assert 'adaptation' in state
        assert 'spike_count' in state
        
        assert state['v_membrane'] == basic_neuron.v_rest
        assert state['v_thresh'] == basic_neuron.v_thresh_init
    
    def test_neuron_dynamics(self, basic_neuron):
        """Test neuron dynamics."""
        inputs = jnp.array([1.0, 0.5, -0.5, 2.0])
        state = basic_neuron.init_state()
        
        output, new_state = basic_neuron(inputs, state)
        
        assert isinstance(output, (float, int)) or output.shape == ()
        assert 'v_membrane' in new_state
        assert jnp.isfinite(output)
        assert jnp.isfinite(new_state['v_membrane'])
    
    def test_sequence_processing(self, basic_neuron):
        """Test processing sequences."""
        seq_len = 10
        input_sequence = jnp.ones((seq_len, 4))
        
        outputs, states = basic_neuron.process_sequence(input_sequence)
        
        assert len(outputs) == seq_len
        assert len(states) == seq_len
        assert jnp.all(jnp.isfinite(outputs))
    
    def test_adaptation_info(self, basic_neuron):
        """Test adaptation information retrieval."""
        state = basic_neuron.init_state()
        info = basic_neuron.get_adaptation_info(state)
        
        assert 'v_membrane' in info
        assert 'v_thresh' in info
        assert 'adaptation' in info
        assert 'distance_to_thresh' in info


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestAdaptiveNeuronLayer:
    """Test suite for AdaptiveNeuronLayer."""
    
    @pytest.fixture
    def neuron_layer(self):
        """Create a layer of adaptive neurons."""
        return AdaptiveNeuronLayer(input_size=3, layer_size=5)
    
    def test_layer_initialization(self, neuron_layer):
        """Test layer initialization."""
        assert neuron_layer.size == 5
        assert len(neuron_layer.neurons) == 5
    
    def test_layer_forward_pass(self, neuron_layer):
        """Test layer forward pass."""
        inputs = jnp.array([1.0, 0.5, -0.2])
        states = neuron_layer.init_states()
        
        outputs, new_states = neuron_layer(inputs, states)
        
        assert outputs.shape == (5,)
        assert len(new_states) == 5
        assert jnp.all(jnp.isfinite(outputs))


class TestModelIntegration:
    """Integration tests across models."""
    
    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_model_interoperability(self):
        """Test that models can work together."""
        key = random.PRNGKey(42)
        
        # Create models
        liquid_net = LiquidNeuralNetwork(
            input_size=2, hidden_size=4, output_size=2, key=key
        )
        
        cont_rnn = ContinuousTimeRNN(
            input_size=2, hidden_size=4, output_size=2, key=key
        )
        
        # Test with same inputs
        inputs = jnp.ones((1, 2))
        
        liquid_hidden = liquid_net.init_hidden_state(1)
        rnn_hidden = cont_rnn.init_hidden_state(1)
        
        liquid_out, _ = liquid_net(inputs, liquid_hidden)
        rnn_out, _ = cont_rnn(inputs, rnn_hidden)
        
        assert liquid_out.shape == rnn_out.shape
        assert jnp.all(jnp.isfinite(liquid_out))
        assert jnp.all(jnp.isfinite(rnn_out))
    
    def test_import_structure(self):
        """Test that imports work correctly."""
        try:
            from models import LiquidNeuralNetwork, ContinuousTimeRNN, AdaptiveNeuron
            # If we get here without exception, imports work
            assert True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")


# Performance benchmarks
@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")  
class TestPerformance:
    """Performance and benchmarking tests."""
    
    def test_forward_pass_performance(self):
        """Test forward pass performance."""
        key = random.PRNGKey(42)
        model = LiquidNeuralNetwork(
            input_size=10, hidden_size=50, output_size=5, key=key
        )
        
        # Large batch for performance testing
        batch_size = 100
        inputs = jnp.ones((batch_size, 10))
        hidden = model.init_hidden_state(batch_size)
        
        import time
        start_time = time.time()
        
        # Run multiple forward passes
        for _ in range(10):
            output, hidden = model(inputs, hidden)
            
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time (less than 1 second for 1000 forward passes)
        assert elapsed < 1.0
        
        # Check final outputs are reasonable
        assert jnp.all(jnp.isfinite(output))
        assert output.shape == (batch_size, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])