#!/usr/bin/env python3
"""
Test script to verify Generation 1 basic functionality.
Tests core model instantiation and forward pass.
"""

import jax
import jax.numpy as jnp
import numpy as np
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import src.models.liquid_neural_network as lnn_module
    import src.models.adaptive_neuron as an_module  
    import src.algorithms.training as training_module
    
    LiquidNeuralNetwork = lnn_module.LiquidNeuralNetwork
    AdaptiveNeuron = lnn_module.AdaptiveNeuron
    ContinuousTimeRNN = lnn_module.ContinuousTimeRNN
    LiquidNetworkTrainer = training_module.LiquidNetworkTrainer
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def test_adaptive_neuron():
    """Test basic adaptive neuron functionality."""
    print("\n--- Testing AdaptiveNeuron ---")
    
    try:
        key = jax.random.PRNGKey(42)
        neuron = AdaptiveNeuron(input_dim=3, hidden_dim=5, key=key)
        print("✓ AdaptiveNeuron instantiated")
        
        # Test dynamics
        t = 0.0
        y = jnp.zeros(5)
        x = jnp.array([1.0, 0.5, -0.5])
        
        dydt = neuron(t, y, x)
        print(f"✓ Neuron dynamics computed: shape {dydt.shape}")
        
        return True
    except Exception as e:
        print(f"✗ AdaptiveNeuron test failed: {e}")
        return False

def test_continuous_time_rnn():
    """Test continuous-time RNN."""
    print("\n--- Testing ContinuousTimeRNN ---")
    
    try:
        key = jax.random.PRNGKey(42)
        rnn = ContinuousTimeRNN(input_dim=3, hidden_dim=5, output_dim=2, key=key)
        print("✓ ContinuousTimeRNN instantiated")
        
        # Test forward pass
        batch_size, seq_len, input_dim = 2, 4, 3
        x = jax.random.normal(key, (batch_size, seq_len, input_dim))
        
        outputs, hidden_states = rnn(x)
        print(f"✓ Forward pass successful: output shape {outputs.shape}, hidden shape {hidden_states.shape}")
        
        return True
    except Exception as e:
        print(f"✗ ContinuousTimeRNN test failed: {e}")
        return False

def test_liquid_neural_network():
    """Test liquid neural network."""
    print("\n--- Testing LiquidNeuralNetwork ---")
    
    try:
        key = jax.random.PRNGKey(42)
        model = LiquidNeuralNetwork(
            input_dim=3,
            hidden_dims=[8, 6],
            output_dim=2,
            key=key
        )
        print("✓ LiquidNeuralNetwork instantiated")
        
        # Test forward pass
        batch_size, seq_len, input_dim = 2, 5, 3
        x = jax.random.normal(key, (batch_size, seq_len, input_dim))
        
        outputs = model(x)
        print(f"✓ Forward pass successful: output shape {outputs.shape}")
        
        # Test liquid states extraction
        states = model.get_liquid_states(x)
        print(f"✓ Liquid states extracted: {len(states)} layers")
        
        return True
    except Exception as e:
        print(f"✗ LiquidNeuralNetwork test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training():
    """Test training framework."""
    print("\n--- Testing Training Framework ---")
    
    try:
        key = jax.random.PRNGKey(42)
        model = LiquidNeuralNetwork(
            input_dim=2,
            hidden_dims=[4],
            output_dim=1,
            key=key
        )
        
        trainer = LiquidNetworkTrainer(model, learning_rate=1e-3)
        print("✓ Trainer instantiated")
        
        # Create dummy data with explicit float32 dtype
        batch_size, seq_len = 4, 6
        x = jax.random.normal(key, (batch_size, seq_len, 2), dtype=jnp.float32)
        y = jax.random.normal(jax.random.split(key)[0], (batch_size, seq_len, 1), dtype=jnp.float32)
        
        # Single training step
        metrics = trainer.train_step(x, y)
        print(f"✓ Training step successful: loss = {metrics['loss']:.6f}")
        
        return True
    except Exception as e:
        print(f"✗ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Testing Generation 1 Basic Functionality")
    print("=" * 50)
    
    tests = [
        test_adaptive_neuron,
        test_continuous_time_rnn, 
        test_liquid_neural_network,
        test_training
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✓ All Generation 1 tests passed!")
        return True
    else:
        print("✗ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)