#!/usr/bin/env python3
"""
Test script for Generation 1 models implementation.
Direct testing of core model functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import jax
import jax.numpy as jnp
from models.liquid_neural_network import LiquidNeuralNetwork, LiquidLayer, AdaptiveLiquidNetwork
from models.continuous_time_rnn import ContinuousTimeRNN, GatedContinuousRNN, MultiScaleCTRNN
from models.adaptive_neuron import LiquidNeuron, ResonatorNeuron, AdaptiveNeuron, NeuronNetwork

def test_liquid_layer():
    """Test basic liquid layer functionality."""
    print("🧪 Testing LiquidLayer...")
    key = jax.random.PRNGKey(42)
    layer = LiquidLayer(input_size=10, hidden_size=20, key=key)
    
    # Test forward pass
    x = jax.random.normal(key, (10,))
    h = jnp.zeros(20)
    h_new = layer(x, h, dt=0.01)
    
    assert h_new.shape == (20,), f"Expected shape (20,), got {h_new.shape}"
    print("✅ LiquidLayer forward pass successful")
    return True

def test_liquid_neural_network():
    """Test complete liquid neural network."""
    print("🧪 Testing LiquidNeuralNetwork...")
    key = jax.random.PRNGKey(42)
    network = LiquidNeuralNetwork(
        input_size=10, hidden_size=20, output_size=5, num_layers=2, key=key
    )
    
    # Test forward pass
    x = jax.random.normal(key, (10,))
    hidden_states = network.init_hidden(batch_size=1)
    
    output, new_hidden = network(x, hidden_states, dt=0.01)
    
    assert output.shape == (5,), f"Expected output shape (5,), got {output.shape}"
    assert len(new_hidden) == 2, f"Expected 2 hidden states, got {len(new_hidden)}"
    print("✅ LiquidNeuralNetwork forward pass successful")
    return True

def test_continuous_time_rnn():
    """Test continuous-time RNN."""
    print("🧪 Testing ContinuousTimeRNN...")
    key = jax.random.PRNGKey(42)
    rnn = ContinuousTimeRNN(input_size=10, hidden_size=20, output_size=5, key=key)
    
    # Test forward pass
    x = jax.random.normal(key, (10,))
    h0 = jnp.zeros(20)
    
    output, h_final = rnn(x, h0, t_span=(0.0, 0.1))
    
    assert output.shape == (5,), f"Expected output shape (5,), got {output.shape}"
    assert h_final.shape == (20,), f"Expected hidden shape (20,), got {h_final.shape}"
    print("✅ ContinuousTimeRNN forward pass successful")
    return True

def test_adaptive_neuron():
    """Test adaptive neuron functionality."""
    print("🧪 Testing AdaptiveNeuron...")
    key = jax.random.PRNGKey(42)
    neuron = AdaptiveNeuron(input_size=10, key=key)
    
    # Test forward pass
    x = jax.random.normal(key, (10,))
    h = 0.0
    
    output, new_weights, new_history = neuron(x, h, dt=0.01)
    
    assert isinstance(output, (float, jnp.ndarray)), f"Expected scalar output, got {type(output)}"
    assert new_weights.shape == (10,), f"Expected weights shape (10,), got {new_weights.shape}"
    print("✅ AdaptiveNeuron forward pass successful")
    return True

def test_neuron_network():
    """Test neuron network functionality."""
    print("🧪 Testing NeuronNetwork...")
    key = jax.random.PRNGKey(42)
    network = NeuronNetwork(
        input_size=10, num_neurons=15, output_size=5, key=key
    )
    
    # Test forward pass
    x = jax.random.normal(key, (10,))
    states = network.init_states(batch_size=1)
    
    output, new_states = network(x, states[0], dt=0.01)
    
    assert output.shape == (5,), f"Expected output shape (5,), got {output.shape}"
    assert new_states.shape == (15,), f"Expected states shape (15,), got {new_states.shape}"
    print("✅ NeuronNetwork forward pass successful")
    return True

def test_all_models():
    """Run all model tests."""
    print("🚀 Starting Generation 1 Model Tests\n")
    
    tests = [
        test_liquid_layer,
        test_liquid_neural_network,
        test_continuous_time_rnn,
        test_adaptive_neuron,
        test_neuron_network
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
        print()
    
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL GENERATION 1 MODELS WORKING!")
        return True
    else:
        print("⚠️  Some tests failed")
        return False

if __name__ == "__main__":
    success = test_all_models()
    sys.exit(0 if success else 1)