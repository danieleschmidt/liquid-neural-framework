#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import src.models.liquid_neural_network as lnn_module

# Simple test to debug the gradient issue
def test_simple_model():
    key = jax.random.PRNGKey(42)
    
    # Create very simple model
    model = lnn_module.LiquidNeuralNetwork(
        input_dim=2,
        hidden_dims=[4],
        output_dim=1,
        key=key
    )
    
    # Create simple data
    x = jnp.ones((1, 3, 2), dtype=jnp.float32)
    y = jnp.ones((1, 3, 1), dtype=jnp.float32)
    
    # Test forward pass
    print("Testing forward pass...")
    predictions = model(x)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions dtype: {predictions.dtype}")
    
    # Check model parameters
    print("\nChecking ALL model components...")
    import equinox as eqx
    
    # Check all components, not just arrays
    all_components = jax.tree.leaves(model)
    for i, component in enumerate(all_components):
        if hasattr(component, 'shape') and hasattr(component, 'dtype'):
            print(f"Component {i}: shape={component.shape}, dtype={component.dtype}")
        else:
            print(f"Component {i}: {type(component)} = {component}")
    
    print("\nChecking only array parameters...")
    params = eqx.filter(model, eqx.is_array)
    for i, param in enumerate(jax.tree.leaves(params)):
        print(f"Param {i}: shape={param.shape}, dtype={param.dtype}")
    
    # Simple loss
    def simple_loss(model, x, y):
        pred = model(x)
        return jnp.mean((pred - y) ** 2)
    
    print("\nTesting gradient computation with proper filtering...")
    try:
        # Use equinox's approach for gradients
        diff_model, static_model = eqx.partition(model, eqx.is_array)
        
        def loss_fn(diff_params):
            model_with_params = eqx.combine(diff_params, static_model)
            return simple_loss(model_with_params, x, y)
        
        loss, grads = jax.value_and_grad(loss_fn)(diff_model)
        print(f"✓ Loss computed: {loss}")
        
        print("\nChecking gradients...")
        for i, grad in enumerate(jax.tree.leaves(grads)):
            if hasattr(grad, 'shape'):
                print(f"Grad {i}: shape={grad.shape}, dtype={grad.dtype}")
            else:
                print(f"Grad {i}: {type(grad)} = {grad}")
        
        print("✓ Gradient computation successful!")
        
    except Exception as e:
        print(f"✗ Gradient computation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_model()